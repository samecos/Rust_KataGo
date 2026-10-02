//! Original-v1 transport/lifecycle CPU checks. No real model, GPU, or accuracy comparison.
//! An inert evaluator deliberately retains a nonempty private execution identity.
use super::*;
use crate::{EvalFailure, EvaluationReport, Metadata};
use prost::Message;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Condvar, Mutex};
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status, Streaming};

const WAIT: Duration = Duration::from_secs(10);
const INTERNAL_PROFILE: &str = "fixture-private-loaded-execution-profile";

fn metadata() -> Metadata {
    Metadata {
        model_sha256: "a".repeat(64),
        execution_profile_id: INTERNAL_PROFILE.into(),
        model_version: 15,
        engine_commit: "original-v1-cpu-fixture".into(),
        backend_info: "inert CPU evaluator; diagnostics only".into(),
        supports_shortterm_error: false,
        default_always_compute_pass_alive: false,
        default_exclude_territory_adjacent_to_atari: false,
    }
}

fn request(id: u64) -> EvalRequest {
    // All original-v1 fields are explicit: there is no execution-profile field.
    EvalRequest {
        task_id: id,
        generation: 7,
        session_id: "original-protocol-cpu".into(),
        input_hash: vec![1, 2, 3, id as u8],
        model_sha256: "a".repeat(64),
        position: None,
        parameters: None,
        lease_ms: 30_000,
    }
}

fn assert_identity(result: &EvalResult, input: &EvalRequest, model: &str) {
    assert_eq!(result.task_id, input.task_id);
    assert_eq!(result.generation, input.generation);
    assert_eq!(result.session_id, input.session_id);
    assert_eq!(result.input_hash, input.input_hash);
    assert_eq!(result.model_sha256, model);
}

struct InertEvaluator {
    metadata: Metadata,
    calls: AtomicU64,
    changed: Notify,
    released: Mutex<bool>,
    release: Condvar,
}

impl InertEvaluator {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            metadata: metadata(),
            calls: AtomicU64::new(0),
            changed: Notify::new(),
            released: Mutex::new(false),
            release: Condvar::new(),
        })
    }

    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.release.notify_all();
    }

    async fn started(&self, count: u64) {
        timeout(WAIT, async {
            loop {
                let changed = self.changed.notified();
                if self.calls.load(AtomicOrdering::SeqCst) >= count {
                    break;
                }
                changed.await;
            }
        })
        .await
        .expect("inert compute did not start");
    }

    fn assert_private_identity(&self) {
        assert_eq!(self.metadata.execution_profile_id, INTERNAL_PROFILE);
    }
}

impl Evaluator for InertEvaluator {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn stats(&self) -> (u64, u64) {
        (0, 0)
    }

    fn evaluate(&self, input: &EvalRequest) -> EvaluationReport {
        self.assert_private_identity();
        self.calls.fetch_add(1, AtomicOrdering::SeqCst);
        self.changed.notify_one();
        if input.task_id == 10 {
            let (released, _) = self
                .release
                .wait_timeout_while(self.released.lock().unwrap(), WAIT, |done| !*done)
                .unwrap();
            let was_released = *released;
            drop(released);
            assert!(
                was_released,
                "fixture must release its bounded physical call"
            );
        }
        assert_ne!(input.task_id, 66, "deliberate inert evaluator panic");
        EvaluationReport {
            result: if input.task_id == 99 {
                Err(EvalFailure::new("FIXTURE_ERROR", "deliberate inert error"))
            } else {
                Ok(wire::NnOutput::default())
            },
            context_us: None,
            evaluator_us: None,
        }
    }
}

struct Peer {
    incoming: Streaming<WorkerMessage>,
    outgoing: Option<mpsc::Sender<Result<wire::ServerMessage, Status>>>,
}

impl Peer {
    async fn send(&self, payload: ServerPayload) {
        timeout(
            WAIT,
            self.outgoing
                .as_ref()
                .unwrap()
                .send(Ok(wire::ServerMessage {
                    payload: Some(payload),
                })),
        )
        .await
        .unwrap()
        .unwrap();
    }

    async fn next(&mut self) -> WorkerPayload {
        timeout(WAIT, self.incoming.message())
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .payload
            .unwrap()
    }

    async fn handshake(&mut self) {
        let WorkerPayload::Hello(hello) = self.next().await else {
            panic!("first message must be the original Hello");
        };
        let decoded = wire::WorkerHello::decode(hello.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decoded.model_sha256, "a".repeat(64));
        assert_eq!(decoded.protocol_version, PROTOCOL_VERSION);
        assert_eq!(decoded.input_profile, INPUT_PROFILE);
        assert_eq!(decoded.worker_id, "original-protocol-cpu-fixture");
        assert_eq!(decoded.max_in_flight, 1);
        assert!(decoded.supports_friendly_pass_search);
        self.send(ServerPayload::Welcome(wire::Welcome {
            protocol_version: PROTOCOL_VERSION,
            connection_id: "unmodified-v1-peer".into(),
        }))
        .await;
    }

    async fn result(&mut self) -> EvalResult {
        timeout(WAIT, async {
            loop {
                match self.next().await {
                    WorkerPayload::Result(result) => {
                        return EvalResult::decode(result.encode_to_vec().as_slice()).unwrap();
                    }
                    WorkerPayload::Heartbeat(_) => {}
                    _ => panic!("unexpected second Hello"),
                }
            }
        })
        .await
        .unwrap()
    }

    async fn terminal_without_result(&mut self, normal: bool) {
        timeout(WAIT, async {
            loop {
                match self.incoming.message().await {
                    Ok(Some(message)) => {
                        assert!(
                            matches!(message.payload, Some(WorkerPayload::Heartbeat(_))),
                            "duplicate or late result must not retire an existing lease twice"
                        );
                    }
                    Ok(None) => break,
                    Err(error) if !normal => {
                        // A rejected stream may terminate with an HTTP/2 error.
                        // This is not relabeled as a normal half-close.
                        assert_ne!(error.code(), tonic::Code::Ok);
                        break;
                    }
                    Err(error) => panic!("normal drain stream failed: {error}"),
                }
            }
        })
        .await
        .expect("worker stream did not reach its bounded terminal state");
    }
}

#[derive(Clone)]
struct Service {
    connections: mpsc::Sender<Peer>,
}

#[tonic::async_trait]
impl wire::worker_service_server::WorkerService for Service {
    type ConnectStream = ReceiverStream<Result<wire::ServerMessage, Status>>;

    async fn connect(
        &self,
        input: Request<Streaming<WorkerMessage>>,
    ) -> Result<Response<Self::ConnectStream>, Status> {
        let (outgoing, receiver) = mpsc::channel(32);
        self.connections
            .send(Peer {
                incoming: input.into_inner(),
                outgoing: Some(outgoing),
            })
            .await
            .map_err(|_| Status::unavailable("fixture ended"))?;
        Ok(Response::new(ReceiverStream::new(receiver)))
    }
}

struct Harness {
    connections: mpsc::Receiver<Peer>,
    evaluator: Arc<InertEvaluator>,
    worker: tokio::task::JoinHandle<Result<()>>,
    server: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
    worker_shutdown: watch::Sender<bool>,
    server_shutdown: watch::Sender<bool>,
}

impl Harness {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, connections) = mpsc::channel(1);
        let (server_shutdown, mut server_stopped) = watch::channel(false);
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(wire::worker_service_server::WorkerServiceServer::new(
                    Service {
                        connections: sender,
                    },
                ))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
                    if !*server_stopped.borrow() {
                        let _ = server_stopped.changed().await;
                    }
                }),
        );
        let evaluator = InertEvaluator::new();
        let (worker_shutdown, worker_stopped) = watch::channel(false);
        let worker = tokio::spawn(run(
            evaluator.clone(),
            WorkerConfig {
                server: format!("http://{address}"),
                worker_id: "original-protocol-cpu-fixture".into(),
                capacity: 1,
                once: true,
                reconnect_delay: Duration::from_millis(10),
                heartbeat_interval: Duration::from_millis(20),
            },
            worker_stopped,
        ));
        Self {
            connections,
            evaluator,
            worker,
            server,
            worker_shutdown,
            server_shutdown,
        }
    }

    async fn peer(&mut self) -> Peer {
        timeout(WAIT, self.connections.recv())
            .await
            .unwrap()
            .unwrap()
    }

    async fn join_server(&mut self) {
        self.server_shutdown.send(true).unwrap();
        timeout(WAIT, &mut self.server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        self.evaluator.assert_private_identity();
    }

    async fn finish(&mut self, mut peer: Peer) {
        peer.send(ServerPayload::Drain(wire::Drain {
            reason: "CPU fixture completed".into(),
        }))
        .await;
        // Close only after enqueueing Drain. The ordered response stream still
        // delivers it; the real client can then complete its graceful drain.
        drop(peer.outgoing.take());
        peer.terminal_without_result(true).await;
        timeout(WAIT, &mut self.worker)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        drop(peer);
        self.join_server().await;
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.evaluator.release();
        let _ = self.worker_shutdown.send(true);
        let _ = self.server_shutdown.send(true);
        self.worker.abort();
        self.server.abort();
    }
}

#[test]
fn original_v1_roundtrips_preserve_task_and_model_without_wire_profile() {
    let meta = metadata();
    let input = request(42);
    let decoded = EvalRequest::decode(input.encode_to_vec().as_slice()).unwrap();
    assert_eq!(decoded, input);
    let welcome = wire::Welcome {
        protocol_version: PROTOCOL_VERSION,
        connection_id: "v1".into(),
    };
    let decoded = wire::Welcome::decode(welcome.encode_to_vec().as_slice()).unwrap();
    validate_welcome(&decoded, &meta).unwrap();
    let wrong_version = wire::Welcome {
        protocol_version: PROTOCOL_VERSION + 1,
        connection_id: "v1".into(),
    };
    assert!(validate_welcome(&wrong_version, &meta).is_err());
    assert!(identity_rejection(&input, &meta).is_none());
    let mut wrong = input.clone();
    wrong.model_sha256 = "b".repeat(64);
    assert_eq!(
        identity_rejection(&wrong, &meta).unwrap().0,
        "MODEL_MISMATCH"
    );
    for code in [
        "MODEL_MISMATCH",
        "CANCELLED",
        "LEASE_EXPIRED",
        "WORKER_INTERNAL_ERROR",
    ] {
        let mut result = result_identity(&wrong, &meta);
        set_error(&mut result, code, "fixture error");
        let decoded = EvalResult::decode(result.encode_to_vec().as_slice()).unwrap();
        assert_identity(&decoded, &wrong, &meta.model_sha256);
        assert_eq!(decoded.error_code, code);
        assert!(decoded.output.is_none());
    }
    assert_eq!(meta.execution_profile_id, INTERNAL_PROFILE);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn original_handshake_normal_error_and_panic_keep_private_identity() {
    let mut harness = Harness::start().await;
    let mut peer = harness.peer().await;
    peer.handshake().await;
    for (id, code) in [
        (1, ""),
        (99, "FIXTURE_ERROR"),
        (66, "WORKER_INTERNAL_ERROR"),
    ] {
        let input = request(id);
        peer.send(ServerPayload::Evaluate(input.clone())).await;
        let result = peer.result().await;
        assert_identity(&result, &input, &harness.evaluator.metadata.model_sha256);
        assert_eq!(result.error_code, code);
        assert_eq!(result.output.is_some(), code.is_empty());
        harness.evaluator.assert_private_identity();
    }
    assert_eq!(harness.evaluator.calls.load(AtomicOrdering::SeqCst), 3);
    harness.finish(peer).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wrong_model_and_invalid_lease_reject_before_compute() {
    let mut harness = Harness::start().await;
    let mut peer = harness.peer().await;
    peer.handshake().await;
    let mut wrong = request(1);
    wrong.model_sha256 = "b".repeat(64);
    let mut no_lease = request(2);
    no_lease.lease_ms = 0;
    let mut excessive_lease = request(3);
    excessive_lease.lease_ms = 3_600_001;
    for (input, code) in [
        (wrong, "MODEL_MISMATCH"),
        (no_lease, "INVALID_REQUEST"),
        (excessive_lease, "INVALID_REQUEST"),
    ] {
        peer.send(ServerPayload::Evaluate(input.clone())).await;
        let result = peer.result().await;
        assert_identity(&result, &input, &harness.evaluator.metadata.model_sha256);
        assert_eq!(result.error_code, code);
        assert!(result.output.is_none());
        assert_eq!(harness.evaluator.calls.load(AtomicOrdering::SeqCst), 0);
    }
    let mut valid = request(4);
    valid.model_sha256 = valid.model_sha256.to_ascii_uppercase();
    peer.send(ServerPayload::Evaluate(valid.clone())).await;
    let result = peer.result().await;
    assert_identity(&result, &valid, &harness.evaluator.metadata.model_sha256);
    assert!(result.error_code.is_empty());
    assert_eq!(harness.evaluator.calls.load(AtomicOrdering::SeqCst), 1);
    harness.finish(peer).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_retains_physical_credit_and_duplicate_has_one_completion() {
    let mut harness = Harness::start().await;
    let mut peer = harness.peer().await;
    peer.handshake().await;
    let blocked = request(10);
    peer.send(ServerPayload::Evaluate(blocked.clone())).await;
    harness.evaluator.started(1).await;
    peer.send(ServerPayload::Evaluate(blocked.clone())).await;
    peer.send(ServerPayload::Cancel(wire::Cancel {
        task_id: blocked.task_id,
        generation: blocked.generation,
        session_id: blocked.session_id.clone(),
    }))
    .await;
    let full = request(11);
    peer.send(ServerPayload::Evaluate(full.clone())).await;
    // Ordered capacity rejection proves Duplicate and Cancel were consumed,
    // yet the still-running physical call retains its sole admission slot.
    let rejected = peer.result().await;
    assert_identity(&rejected, &full, &harness.evaluator.metadata.model_sha256);
    assert_eq!(rejected.error_code, "CAPACITY_EXCEEDED");
    assert_eq!(harness.evaluator.calls.load(AtomicOrdering::SeqCst), 1);
    harness.evaluator.release();
    let cancelled = peer.result().await;
    assert_identity(
        &cancelled,
        &blocked,
        &harness.evaluator.metadata.model_sha256,
    );
    assert_eq!(cancelled.error_code, "CANCELLED");
    assert!(cancelled.output.is_none());
    let next = request(12);
    peer.send(ServerPayload::Evaluate(next.clone())).await;
    let result = peer.result().await;
    assert_identity(&result, &next, &harness.evaluator.metadata.model_sha256);
    assert!(result.error_code.is_empty());
    assert_eq!(harness.evaluator.calls.load(AtomicOrdering::SeqCst), 2);
    // Any extra duplicate completion causes finish's stream check to fail.
    harness.finish(peer).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn active_duplicate_with_wrong_model_ends_stream_without_second_completion() {
    let mut harness = Harness::start().await;
    let mut peer = harness.peer().await;
    peer.handshake().await;
    let input = request(10);
    peer.send(ServerPayload::Evaluate(input.clone())).await;
    harness.evaluator.started(1).await;
    let mut changed = input;
    changed.model_sha256 = "b".repeat(64);
    peer.send(ServerPayload::Evaluate(changed)).await;
    // Real client closes its outgoing stream before joining blocked work on an
    // admission failure; observing that terminal state avoids sleep/race guesses.
    peer.terminal_without_result(false).await;
    assert!(!harness.worker.is_finished());
    assert_eq!(harness.evaluator.calls.load(AtomicOrdering::SeqCst), 1);
    harness.evaluator.release();
    let error = timeout(WAIT, &mut harness.worker)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("active duplicate changed model"));
    drop(peer);
    harness.join_server().await;
}

#[test]
fn real_dummy_engine_accepts_original_request_through_prepare_and_compute() {
    use crate::evaluator::Engine;
    use kata_core::config::ConfigParser;
    let cfg = ConfigParser::from_str(
        "nnBackend=dummybackend\nnnMaxBatchSize=4\nnnCacheSizePowerOfTwo=4\nnnMutexPoolSizePowerOfTwo=2\n",
        false, false,
    ).unwrap();
    let engine = Engine::load("/dev/null", None, &cfg, 1, true).unwrap();
    let before = engine.metadata().clone();
    let mut input = request(1);
    input.model_sha256.clone_from(&before.model_sha256);
    input.position = Some(wire::Position {
        board_size: 19,
        komi: 7.5,
        rules: "chinese".into(),
        initial_player: 1,
        next_player: 1,
        ..Default::default()
    });
    input.parameters = Some(wire::EvalParameters {
        policy_temperature: 1.0,
        draw_equivalent_wins_for_white: 0.5,
        max_history: 1000,
        skip_cache: true,
        ..Default::default()
    });
    let input = EvalRequest::decode(input.encode_to_vec().as_slice()).unwrap();
    let report = engine.evaluate(&input);
    assert!(report.context_us.is_some());
    assert!(report.evaluator_us.is_some());
    assert!(report.result.is_ok()); // Execution/transport only; inspect no numerical outputs.
    let mut wrong = input;
    wrong.model_sha256 = "b".repeat(64);
    let rejected = engine.evaluate(&wrong);
    assert_eq!(rejected.result.unwrap_err().code, "INVALID_CONTEXT");
    assert!(rejected.evaluator_us.is_none());
    assert_eq!(engine.metadata().model_sha256, before.model_sha256);
    assert_eq!(
        engine.metadata().execution_profile_id,
        before.execution_profile_id
    );
}

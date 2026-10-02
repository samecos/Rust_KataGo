//! Identity/lifecycle-only CPU tests. The evaluator is inert; no model or GPU.
use super::*;
use crate::{EvalFailure, EvaluationReport, Metadata};
use prost::Message;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Condvar, Mutex};
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status, Streaming};

const WAIT: Duration = Duration::from_secs(3);
const PROFILE: &str = "fixture-loaded-profile-a";

fn metadata(profile: &str) -> Metadata {
    Metadata {
        model_sha256: "a".repeat(64),
        execution_profile_id: profile.into(),
        model_version: 15,
        engine_commit: "identity-fixture".into(),
        backend_info: "inert CPU boundary; not an identity source".into(),
        supports_shortterm_error: false,
        default_always_compute_pass_alive: false,
        default_exclude_territory_adjacent_to_atari: false,
    }
}

fn request(id: u64, profile: &str) -> EvalRequest {
    EvalRequest {
        task_id: id,
        generation: 1,
        session_id: "identity-test".into(),
        input_hash: vec![1, 2, 3],
        model_sha256: "a".repeat(64),
        execution_profile_id: profile.into(),
        lease_ms: 10_000,
        ..Default::default()
    }
}

struct InertEvaluator {
    metadata: Metadata,
    calls: AtomicU64,
    changed: Notify,
    released: Mutex<bool>,
    release: Condvar,
}
impl InertEvaluator {
    fn new(profile: &str) -> Arc<Self> {
        Arc::new(Self {
            metadata: metadata(profile),
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
        .expect("inert evaluation did not start");
    }
}
impl Evaluator for InertEvaluator {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }
    fn stats(&self) -> (u64, u64) {
        (0, 0)
    }
    fn evaluate(&self, request: &EvalRequest) -> EvaluationReport {
        self.calls.fetch_add(1, AtomicOrdering::SeqCst);
        self.changed.notify_one();
        if request.task_id == 10 {
            let (released, _) = self
                .release
                .wait_timeout_while(self.released.lock().unwrap(), WAIT, |released| !*released)
                .unwrap();
            assert!(*released, "test must release its inert blocking call");
        }
        assert_ne!(request.task_id, 66, "inert evaluator panic");
        EvaluationReport {
            result: if request.task_id == 99 {
                Err(EvalFailure::new("FIXTURE_ERROR", "inert evaluator error"))
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
    outgoing: mpsc::Sender<Result<wire::ServerMessage, Status>>,
}
impl Peer {
    async fn send(&self, payload: ServerPayload) {
        self.outgoing
            .send(Ok(wire::ServerMessage {
                payload: Some(payload),
            }))
            .await
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
    async fn hello(&mut self) -> wire::WorkerHello {
        let WorkerPayload::Hello(value) = self.next().await else {
            panic!("Hello required");
        };
        value
    }
    async fn welcome(&self, profile: &str) {
        self.send(ServerPayload::Welcome(wire::Welcome {
            protocol_version: PROTOCOL_VERSION,
            connection_id: "fixture-connection".into(),
            execution_profile_id: profile.into(),
        }))
        .await;
    }
    async fn result(&mut self) -> EvalResult {
        timeout(WAIT, async {
            loop {
                match self.next().await {
                    WorkerPayload::Result(value) => return value,
                    WorkerPayload::Heartbeat(_) => {}
                    _ => panic!("unexpected worker message"),
                }
            }
        })
        .await
        .unwrap()
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
        request: Request<Streaming<WorkerMessage>>,
    ) -> Result<Response<Self::ConnectStream>, Status> {
        let (outgoing, receiver) = mpsc::channel(32);
        self.connections
            .send(Peer {
                incoming: request.into_inner(),
                outgoing,
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
    shutdown: watch::Sender<bool>,
}
impl Harness {
    async fn start(profile: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (tx, connections) = mpsc::channel(1);
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(wire::worker_service_server::WorkerServiceServer::new(
                    Service { connections: tx },
                ))
                .serve_with_incoming(TcpListenerStream::new(listener)),
        );
        let evaluator = InertEvaluator::new(profile);
        let (shutdown, stopped) = watch::channel(false);
        let worker = tokio::spawn(run(
            evaluator.clone(),
            WorkerConfig {
                server: format!("http://{address}"),
                worker_id: "typed-CPU-fixture".into(),
                capacity: 1,
                once: true,
                reconnect_delay: Duration::from_millis(10),
                heartbeat_interval: Duration::from_millis(20),
            },
            stopped,
        ));
        Self {
            connections,
            evaluator,
            worker,
            server,
            shutdown,
        }
    }
    async fn peer(&mut self) -> Peer {
        timeout(WAIT, self.connections.recv())
            .await
            .unwrap()
            .unwrap()
    }
    async fn stop(&mut self) {
        self.evaluator.release();
        self.shutdown.send(true).unwrap();
        timeout(WAIT, &mut self.worker)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
}
impl Drop for Harness {
    fn drop(&mut self) {
        self.evaluator.release();
        let _ = self.shutdown.send(true);
        self.worker.abort();
        self.server.abort();
    }
}

#[test]
fn actual_getter_required_for_cuda_and_quant_never_downgrades() {
    use crate::evaluator::loaded_execution_profile as loaded;
    for backend in ["cudabackend", "cudaint8backend", "cudaquantbackend"] {
        assert!(loaded(false, backend, None, None).is_err());
        assert!(loaded(false, backend, None, Some("")).is_err());
    }
    assert!(loaded(false, "cudaquantbackend", Some(PROFILE), Some("other")).is_err());
    assert_eq!(
        loaded(false, "cudaquantbackend", Some(PROFILE), Some(PROFILE)).unwrap(),
        PROFILE
    );
    assert_eq!(
        loaded(false, "cudaint8backend", None, Some(PROFILE)).unwrap(),
        PROFILE
    );
    assert_eq!(loaded(true, "dummybackend", None, None).unwrap(), "");
    assert!(loaded(true, "cudaquantbackend", None, None).is_err());
    assert_eq!(loaded(false, "onnxbackend", None, None).unwrap(), "");
    assert!(loaded(false, "cudabackend", None, Some("bad profile")).is_err());
}

#[test]
fn typed_fields_roundtrip_and_error_helpers_use_actual_not_requested_identity() {
    let meta = metadata(PROFILE);
    let requested = request(1, "wrong-request-profile");
    let decoded = EvalRequest::decode(requested.encode_to_vec().as_slice()).unwrap();
    assert_eq!(decoded.execution_profile_id, "wrong-request-profile");
    let mut result = result_identity(&decoded, &meta);
    for code in [
        "EXECUTION_PROFILE_MISMATCH",
        "CANCELLED",
        "LEASE_EXPIRED",
        "WORKER_INTERNAL_ERROR",
    ] {
        set_error(&mut result, code, "fixture");
        let decoded = EvalResult::decode(result.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decoded.execution_profile_id, PROFILE);
        assert_eq!(decoded.model_sha256, meta.model_sha256);
        assert!(decoded.output.is_none());
    }
    let legacy = metadata("");
    assert!(identity_rejection(&request(1, ""), &legacy).is_none());
    assert_eq!(
        identity_rejection(&request(1, PROFILE), &legacy).unwrap().0,
        "EXECUTION_PROFILE_MISMATCH"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn welcome_requires_exact_loaded_profile_over_real_grpc() {
    for wrong in ["", "other-profile"] {
        let mut harness = Harness::start(PROFILE).await;
        let mut peer = harness.peer().await;
        assert_eq!(peer.hello().await.execution_profile_id, PROFILE);
        peer.welcome(wrong).await;
        let error = timeout(WAIT, &mut harness.worker)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert!(error.to_string().contains("profile ACK"));
        assert_eq!(harness.evaluator.calls.load(AtomicOrdering::SeqCst), 0);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn typed_request_rejection_precedes_compute_and_cancel_keeps_actual_identity() {
    let mut harness = Harness::start(PROFILE).await;
    let mut peer = harness.peer().await;
    assert_eq!(peer.hello().await.execution_profile_id, PROFILE);
    peer.welcome(PROFILE).await;
    for (id, profile) in [(1, ""), (2, "other-profile")] {
        peer.send(ServerPayload::Evaluate(request(id, profile)))
            .await;
        let rejected = peer.result().await;
        assert_eq!(rejected.error_code, "EXECUTION_PROFILE_MISMATCH");
        assert_eq!(rejected.execution_profile_id, PROFILE);
        assert!(rejected.output.is_none());
    }
    assert_eq!(harness.evaluator.calls.load(AtomicOrdering::SeqCst), 0);
    peer.send(ServerPayload::Evaluate(request(10, PROFILE)))
        .await;
    harness.evaluator.started(1).await;
    peer.send(ServerPayload::Cancel(wire::Cancel {
        task_id: 10,
        generation: 1,
        session_id: "identity-test".into(),
    }))
    .await;
    // Same stream ordering: this rejection proves the earlier Cancel was read.
    peer.send(ServerPayload::Evaluate(request(11, "other-profile")))
        .await;
    let rejected = peer.result().await;
    assert_eq!(rejected.task_id, 11);
    assert_eq!(rejected.error_code, "EXECUTION_PROFILE_MISMATCH");
    assert_eq!(rejected.execution_profile_id, PROFILE);
    harness.evaluator.release();
    let cancelled = peer.result().await;
    assert_eq!(cancelled.task_id, 10);
    assert_eq!(cancelled.error_code, "CANCELLED");
    assert_eq!(cancelled.execution_profile_id, PROFILE);
    assert!(cancelled.output.is_none());
    assert_eq!(harness.evaluator.calls.load(AtomicOrdering::SeqCst), 1);
    harness.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn normal_error_and_panic_results_report_loaded_profile() {
    let mut harness = Harness::start(PROFILE).await;
    let mut peer = harness.peer().await;
    assert_eq!(peer.hello().await.execution_profile_id, PROFILE);
    peer.welcome(PROFILE).await;
    for (id, expected_error) in [
        (1, ""),
        (99, "FIXTURE_ERROR"),
        (66, "WORKER_INTERNAL_ERROR"),
    ] {
        peer.send(ServerPayload::Evaluate(request(id, PROFILE)))
            .await;
        let result = peer.result().await;
        assert_eq!(result.task_id, id);
        assert_eq!(result.execution_profile_id, PROFILE);
        assert_eq!(result.error_code, expected_error);
        assert_eq!(result.output.is_some(), expected_error.is_empty());
    }
    assert_eq!(harness.evaluator.calls.load(AtomicOrdering::SeqCst), 3);
    harness.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explicit_empty_legacy_fixture_remains_compatible() {
    let mut harness = Harness::start("").await;
    let mut peer = harness.peer().await;
    assert_eq!(peer.hello().await.execution_profile_id, "");
    peer.welcome("").await;
    peer.send(ServerPayload::Evaluate(request(1, ""))).await;
    let result = peer.result().await;
    assert_eq!(result.execution_profile_id, "");
    assert_eq!(result.error_code, "");
    peer.send(ServerPayload::Evaluate(request(2, PROFILE)))
        .await;
    assert_eq!(peer.result().await.error_code, "EXECUTION_PROFILE_MISMATCH");
    assert_eq!(harness.evaluator.calls.load(AtomicOrdering::SeqCst), 1);
    harness.stop().await;
}

//! Exercise the real bidi gRPC boundary without loading a model or GPU.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use kata_worker::client::{WorkerConfig, run};
use kata_worker::wire::{self, server_message, worker_message};
use kata_worker::{EvalFailure, EvaluationReport, Evaluator, Metadata};
use tokio::net::TcpListener;
use tokio::sync::{Notify, mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{Request, Response, Status, Streaming};

const WAIT: Duration = Duration::from_secs(5);
const MODEL_HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

struct FakeEvaluator {
    metadata: Metadata,
    blocked: HashSet<u64>,
    panics: HashSet<u64>,
    released: Mutex<bool>,
    release_cv: Condvar,
    started: AtomicU64,
    finished: AtomicU64,
    changed: Notify,
}

impl FakeEvaluator {
    fn new(blocked: &[u64], panics: &[u64]) -> Arc<Self> {
        Arc::new(Self {
            metadata: Metadata {
                model_sha256: MODEL_HASH.into(),
                model_version: 15,
                engine_commit: "test-engine-provenance".into(),
                backend_info: "fake-transport-evaluator".into(),
                supports_shortterm_error: true,
                default_always_compute_pass_alive: false,
                default_exclude_territory_adjacent_to_atari: false,
            },
            blocked: blocked.iter().copied().collect(),
            panics: panics.iter().copied().collect(),
            released: Mutex::new(false),
            release_cv: Condvar::new(),
            started: AtomicU64::new(0),
            finished: AtomicU64::new(0),
            changed: Notify::new(),
        })
    }

    async fn wait_started(&self, count: u64) {
        timeout(WAIT, async {
            loop {
                let changed = self.changed.notified();
                if self.started.load(Ordering::SeqCst) >= count {
                    return;
                }
                changed.await;
            }
        })
        .await
        .expect("evaluation did not start");
    }

    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.release_cv.notify_all();
    }
}

impl Evaluator for FakeEvaluator {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn evaluate(&self, request: &wire::EvalRequest) -> EvaluationReport {
        self.started.fetch_add(1, Ordering::SeqCst);
        self.changed.notify_one();
        if self.blocked.contains(&request.task_id) {
            // A deadline also keeps a failed test from stranding a blocking thread.
            let (released, _) = self
                .release_cv
                .wait_timeout_while(self.released.lock().unwrap(), WAIT, |released| !*released)
                .unwrap();
            assert!(*released, "test did not release the fake evaluator");
        }
        assert!(
            !self.panics.contains(&request.task_id),
            "injected evaluator panic"
        );
        if request.task_id == 99 {
            return EvaluationReport {
                result: Err(EvalFailure::new(
                    "INVALID_CONTEXT",
                    "injected invalid history",
                )),
                context_us: Some(7),
                evaluator_us: None,
            };
        }
        self.finished.fetch_add(1, Ordering::SeqCst);
        EvaluationReport {
            result: Ok(wire::NnOutput {
                policy: vec![1.0 / 362.0; 362],
                white_win_prob: 0.6,
                white_loss_prob: 0.4,
                white_score_mean: 2.0,
                white_score_mean_sq: 9.0,
                white_lead: 1.5,
                has_shortterm_error: true,
                ..Default::default()
            }),
            context_us: Some(7),
            evaluator_us: Some(13),
        }
    }

    fn stats(&self) -> (u64, u64) {
        // Deliberately unlike request counts: heartbeat must use evaluator stats.
        let finished = self.finished.load(Ordering::SeqCst);
        (finished * 7, finished * 3)
    }
}

struct Peer {
    incoming: Streaming<wire::WorkerMessage>,
    outgoing: mpsc::Sender<Result<wire::ServerMessage, Status>>,
}

impl Peer {
    async fn send(&self, payload: server_message::Payload) {
        self.outgoing
            .send(Ok(wire::ServerMessage {
                payload: Some(payload),
            }))
            .await
            .unwrap();
    }

    async fn next(&mut self) -> worker_message::Payload {
        timeout(WAIT, self.incoming.message())
            .await
            .expect("worker message timed out")
            .expect("worker stream failed")
            .expect("worker stream closed unexpectedly")
            .payload
            .expect("empty worker payload")
    }

    async fn hello(&mut self) -> wire::WorkerHello {
        match self.next().await {
            worker_message::Payload::Hello(hello) => hello,
            _ => panic!("the first worker message must be Hello"),
        }
    }

    async fn welcome(&self) {
        self.send(server_message::Payload::Welcome(wire::Welcome {
            protocol_version: 1,
            connection_id: "test-connection".into(),
        }))
        .await;
    }

    async fn result(&mut self) -> wire::EvalResult {
        timeout(WAIT, async {
            loop {
                match self.next().await {
                    worker_message::Payload::Result(result) => return result,
                    worker_message::Payload::Heartbeat(_) => {}
                    worker_message::Payload::Hello(_) => panic!("second Hello on one connection"),
                }
            }
        })
        .await
        .expect("result timed out")
    }

    async fn heartbeat(&mut self, completed: u64) -> wire::WorkerHeartbeat {
        timeout(WAIT, async {
            loop {
                match self.next().await {
                    worker_message::Payload::Heartbeat(hb)
                        if hb.completed_requests >= completed =>
                    {
                        return hb;
                    }
                    worker_message::Payload::Heartbeat(_) => {}
                    _ => panic!("unexpected non-heartbeat message"),
                }
            }
        })
        .await
        .expect("heartbeat timed out")
    }
}

#[derive(Clone)]
struct TestService {
    connections: mpsc::Sender<Peer>,
}

#[tonic::async_trait]
impl wire::worker_service_server::WorkerService for TestService {
    type ConnectStream = ReceiverStream<Result<wire::ServerMessage, Status>>;

    async fn connect(
        &self,
        request: Request<Streaming<wire::WorkerMessage>>,
    ) -> Result<Response<Self::ConnectStream>, Status> {
        let (outgoing, responses) = mpsc::channel(32);
        self.connections
            .send(Peer {
                incoming: request.into_inner(),
                outgoing,
            })
            .await
            .map_err(|_| Status::unavailable("test finished"))?;
        Ok(Response::new(ReceiverStream::new(responses)))
    }
}

struct Harness {
    connections: mpsc::Receiver<Peer>,
    shutdown: watch::Sender<bool>,
    worker: JoinHandle<anyhow::Result<()>>,
    server: JoinHandle<Result<(), tonic::transport::Error>>,
    stop_server: Option<oneshot::Sender<()>>,
}

impl Harness {
    async fn start(evaluator: Arc<FakeEvaluator>, capacity: u32, once: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (connections_tx, connections) = mpsc::channel(8);
        let (stop_server, stopped) = oneshot::channel();
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(wire::worker_service_server::WorkerServiceServer::new(
                    TestService {
                        connections: connections_tx,
                    },
                ))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = stopped.await;
                }),
        );
        let (shutdown, shutdown_rx) = watch::channel(false);
        let worker = tokio::spawn(run(
            evaluator,
            WorkerConfig {
                server: format!("http://{address}"),
                worker_id: "rust-transport-test".into(),
                capacity,
                once,
                reconnect_delay: Duration::from_millis(10),
                heartbeat_interval: Duration::from_millis(20),
            },
            shutdown_rx,
        ));
        Self {
            connections,
            shutdown,
            worker,
            server,
            stop_server: Some(stop_server),
        }
    }

    async fn accept(&mut self) -> Peer {
        timeout(WAIT, self.connections.recv())
            .await
            .expect("worker did not connect")
            .unwrap()
    }

    async fn stop(mut self) {
        let _ = self.shutdown.send(true);
        timeout(WAIT, &mut self.worker)
            .await
            .expect("worker failed to stop")
            .expect("worker task panicked")
            .expect("worker shutdown failed");
        let _ = self.stop_server.take().unwrap().send(());
        timeout(WAIT, &mut self.server)
            .await
            .expect("server failed to stop")
            .unwrap()
            .unwrap();
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        let _ = self.shutdown.send(true);
        if let Some(stop) = self.stop_server.take() {
            let _ = stop.send(());
        }
        self.worker.abort();
        self.server.abort();
    }
}

fn request(task_id: u64) -> wire::EvalRequest {
    wire::EvalRequest {
        task_id,
        generation: 12,
        session_id: "session-α".into(),
        input_hash: vec![0, 255, 17, task_id as u8],
        model_sha256: MODEL_HASH.into(),
        lease_ms: 5_000,
        position: Some(wire::Position {
            board_size: 19,
            komi: 7.5,
            rules: "chinese".into(),
            initial_player: wire::Color::Black as i32,
            next_player: wire::Color::Black as i32,
            ..Default::default()
        }),
        parameters: Some(wire::EvalParameters {
            policy_temperature: 1.0,
            draw_equivalent_wins_for_white: 0.5,
            max_history: 1_000,
            ..Default::default()
        }),
    }
}

fn assert_identity(result: &wire::EvalResult, request: &wire::EvalRequest) {
    assert_eq!(result.task_id, request.task_id);
    assert_eq!(result.generation, request.generation);
    assert_eq!(result.session_id, request.session_id);
    assert_eq!(result.input_hash, request.input_hash);
    assert_eq!(result.model_sha256, MODEL_HASH);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hello_success_identity_and_real_evaluator_stats() {
    let fake = FakeEvaluator::new(&[], &[]);
    let mut harness = Harness::start(fake, 2, true).await;
    let mut peer = harness.accept().await;
    let hello = peer.hello().await;
    assert_eq!(hello.worker_id, "rust-transport-test");
    assert!(!hello.instance_id.is_empty());
    assert_eq!(hello.protocol_version, kata_worker::PROTOCOL_VERSION);
    assert_eq!(hello.input_profile, kata_worker::INPUT_PROFILE);
    assert_eq!(hello.model_sha256, MODEL_HASH);
    assert_eq!(hello.model_version, 15);
    assert_eq!(hello.engine_commit, "test-engine-provenance");
    assert_eq!(hello.backend_info, "fake-transport-evaluator");
    assert_eq!(hello.max_in_flight, 2);
    assert_eq!(hello.max_board_size, 19);
    assert!(hello.supports_friendly_pass_search);
    assert!(hello.supports_shortterm_error);
    peer.welcome().await;
    let req = request(1);
    peer.send(server_message::Payload::Evaluate(req.clone()))
        .await;
    let result = peer.result().await;
    assert_identity(&result, &req);
    assert!(result.error_code.is_empty());
    assert_eq!(result.context_us, Some(7));
    assert_eq!(result.evaluator_us, Some(13));
    assert!(result.queue_us.is_some());
    let output = result.output.unwrap();
    assert_eq!(output.policy.len(), 362);
    assert_eq!(output.white_win_prob, 0.6);
    let hb = peer.heartbeat(1).await;
    assert_eq!(
        (hb.in_flight, hb.completed_requests, hb.failed_requests),
        (0, 1, 0)
    );
    assert_eq!((hb.nn_rows, hb.nn_batches), (7, 3));
    harness.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn evaluation_before_welcome_is_never_executed() {
    let fake = FakeEvaluator::new(&[], &[]);
    let mut harness = Harness::start(fake.clone(), 1, true).await;
    let mut peer = harness.accept().await;
    peer.hello().await;
    peer.send(server_message::Payload::Evaluate(request(1)))
        .await;
    let outcome = timeout(WAIT, &mut harness.worker)
        .await
        .expect("bad handshake did not stop worker");
    assert!(outcome.expect("worker task panicked").is_err());
    assert_eq!(fake.started.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn duplicate_and_cancel_keep_capacity_until_physical_completion() {
    let fake = FakeEvaluator::new(&[1], &[]);
    let mut harness = Harness::start(fake.clone(), 1, true).await;
    let mut peer = harness.accept().await;
    peer.hello().await;
    peer.welcome().await;
    let req = request(1);
    peer.send(server_message::Payload::Evaluate(req.clone()))
        .await;
    fake.wait_started(1).await;
    peer.send(server_message::Payload::Evaluate(req.clone()))
        .await;
    peer.send(server_message::Payload::Cancel(wire::Cancel {
        task_id: req.task_id,
        generation: req.generation,
        session_id: req.session_id.clone(),
    }))
    .await;
    let overflow = request(2);
    peer.send(server_message::Payload::Evaluate(overflow.clone()))
        .await;
    let result = peer.result().await;
    assert_identity(&result, &overflow);
    assert_eq!(result.error_code, "CAPACITY_EXCEEDED");
    assert!(result.output.is_none());
    assert_eq!(fake.started.load(Ordering::SeqCst), 1);
    fake.release();
    let result = peer.result().await;
    assert_identity(&result, &req);
    assert_eq!(result.error_code, "CANCELLED");
    assert!(result.output.is_none());
    peer.send(server_message::Payload::Evaluate(request(3)))
        .await;
    let result = peer.result().await;
    assert_eq!(result.task_id, 3);
    assert!(result.error_code.is_empty());
    assert_eq!(fake.started.load(Ordering::SeqCst), 2);
    harness.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn expired_lease_discards_completed_output() {
    let fake = FakeEvaluator::new(&[1], &[]);
    let mut harness = Harness::start(fake.clone(), 1, true).await;
    let mut peer = harness.accept().await;
    peer.hello().await;
    peer.welcome().await;
    let mut req = request(1);
    req.lease_ms = 500;
    peer.send(server_message::Payload::Evaluate(req.clone()))
        .await;
    fake.wait_started(1).await;
    tokio::time::sleep(Duration::from_millis(550)).await;
    fake.release();
    let result = peer.result().await;
    assert_identity(&result, &req);
    assert_eq!(result.error_code, "LEASE_EXPIRED");
    assert!(result.output.is_none());
    assert!(result.elapsed_us >= 500_000);
    assert_eq!(fake.finished.load(Ordering::SeqCst), 1);
    harness.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reconnect_never_replays_old_stream_results() {
    let fake = FakeEvaluator::new(&[1], &[]);
    let mut harness = Harness::start(fake.clone(), 1, false).await;
    let mut first = harness.accept().await;
    let hello = first.hello().await;
    first.welcome().await;
    first
        .send(server_message::Payload::Evaluate(request(1)))
        .await;
    fake.wait_started(1).await;
    first
        .outgoing
        .send(Err(Status::unavailable("injected disconnect")))
        .await
        .unwrap();
    drop(first);
    fake.release();
    let mut second = harness.accept().await;
    let next_hello = second.hello().await;
    assert_eq!(next_hello.worker_id, hello.worker_id);
    assert_eq!(next_hello.instance_id, hello.instance_id);
    second.welcome().await;
    let mut new_request = request(1);
    new_request.input_hash = vec![42, 0, 99];
    second
        .send(server_message::Payload::Evaluate(new_request.clone()))
        .await;
    let result = second.result().await;
    assert_identity(&result, &new_request);
    assert!(result.error_code.is_empty());
    assert_eq!(fake.started.load(Ordering::SeqCst), 2);
    harness.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drain_completes_admitted_work_and_exits_without_reconnecting() {
    let fake = FakeEvaluator::new(&[1], &[]);
    let mut harness = Harness::start(fake.clone(), 1, false).await;
    let mut peer = harness.accept().await;
    peer.hello().await;
    peer.welcome().await;
    peer.send(server_message::Payload::Evaluate(request(1)))
        .await;
    fake.wait_started(1).await;
    peer.send(server_message::Payload::Drain(wire::Drain {
        reason: "rolling restart".into(),
    }))
    .await;
    peer.send(server_message::Payload::Evaluate(request(2)))
        .await;
    let result = peer.result().await;
    assert_eq!(result.task_id, 2);
    assert_eq!(result.error_code, "DRAINING");
    fake.release();
    let result = peer.result().await;
    assert_eq!(result.task_id, 1);
    assert!(result.error_code.is_empty());
    assert!(result.output.is_some());
    // The service closes its response stream once the drained worker's result
    // has arrived; the worker can then finish its graceful half-close.
    drop(peer);
    timeout(WAIT, &mut harness.worker)
        .await
        .expect("drain did not stop worker")
        .unwrap()
        .unwrap();
    assert!(harness.connections.try_recv().is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_waits_for_blocking_work_without_reconnecting() {
    let fake = FakeEvaluator::new(&[1], &[]);
    let mut harness = Harness::start(fake.clone(), 1, false).await;
    let mut peer = harness.accept().await;
    peer.hello().await;
    peer.welcome().await;
    peer.send(server_message::Payload::Evaluate(request(1)))
        .await;
    fake.wait_started(1).await;
    harness.shutdown.send(true).unwrap();
    assert!(
        timeout(Duration::from_millis(50), &mut harness.worker)
            .await
            .is_err()
    );
    fake.release();
    harness.stop().await;
    assert_eq!(fake.finished.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn panic_is_worker_failure_and_context_failure_keeps_its_classification() {
    let fake = FakeEvaluator::new(&[], &[1]);
    let mut harness = Harness::start(fake, 1, true).await;
    let mut peer = harness.accept().await;
    peer.hello().await;
    peer.welcome().await;
    for (task_id, expected) in [
        (1, "WORKER_INTERNAL_ERROR"),
        (99, "INVALID_CONTEXT"),
        (2, ""),
    ] {
        let req = request(task_id);
        peer.send(server_message::Payload::Evaluate(req.clone()))
            .await;
        let result = peer.result().await;
        assert_identity(&result, &req);
        assert_eq!(result.error_code, expected);
        assert_eq!(result.output.is_some(), expected.is_empty());
        if task_id == 99 {
            assert_eq!(result.context_us, Some(7));
            assert_eq!(result.evaluator_us, None);
        }
    }
    let hb = peer.heartbeat(1).await;
    assert_eq!((hb.completed_requests, hb.failed_requests), (1, 2));
    harness.stop().await;
}

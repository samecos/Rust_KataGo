//! One real gRPC evaluation through the unmodified original-v1 WorkerPool.
//! This executable never launches a Worker and never evaluates output accuracy.
//! Run only under a separately frozen plan that owns the external Worker process.
use clap::Parser;
use go_core::{Position, Search, SearchStep};
use go_protocol::{
    PROTOCOL_VERSION,
    v1::{
        self as wire,
        worker_service_server::{WorkerService, WorkerServiceServer},
    },
};
use go_server::{
    worker::{WorkerPool, WorkerView},
    worker_schedule::{Scheduler, SchedulingConfig},
};
use serde::Serialize;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc as sync_mpsc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{mpsc, oneshot, watch},
    task::JoinHandle,
    time::{Instant, sleep, timeout_at},
};
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};
use tonic::{Request, Response, Status, Streaming};

const SESSION: &str = "isolated-original-worker-probe";
const REQUEST_BUDGET: usize = 1;

#[derive(Parser, Debug)]
#[command(about = "One original-v1 WorkerPool request; no accuracy or performance verdict")]
struct Args {
    #[arg(long)]
    model_sha256: String,
    #[arg(long)]
    worker_id: String,
    #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(u32).range(1..=4096))]
    capacity: u32,
    /// Fresh endpoint JSON path. A separate <stem>.result.json is also reserved.
    #[arg(long)]
    output: PathBuf,
    /// Original overall bound, including Worker arrival and server shutdown.
    #[arg(long, value_parser = clap::value_parser!(u64).range(20..=600))]
    timeout_seconds: u64,
}

#[derive(Default, Serialize)]
struct Observations {
    connect_calls: usize,
    accepted_worker: Option<WorkerView>,
    remote_address: Option<String>,
    welcome_enqueued: usize,
    evaluations_enqueued: usize,
    original_cancels_enqueued: usize,
    drain_requested: bool,
    drain_enqueued: bool,
    original_response_terminal_status: Option<String>,
    original_response_stream_eof: bool,
    forward_loop_finished: bool,
    forward_task_joined: bool,
    requests_dispatched: usize,
    evaluation_request: Option<serde_json::Value>,
    outcome_token: Option<serde_json::Value>,
    outcomes_accepted: usize,
    evaluation_failure: Option<String>,
    completed_worker: Option<WorkerView>,
    pool_empty_after_drain: bool,
    server_joined: bool,
    server_aborted: bool,
    first_error: Option<String>,
    cleanup_errors: Vec<String>,
}

struct Shared {
    observations: Mutex<Observations>,
    connected: AtomicBool,
    forward_task: Mutex<Option<JoinHandle<()>>>,
}
impl Shared {
    fn fail(&self, error: impl Into<String>) {
        let mut facts = self.observations.lock().unwrap();
        if facts.first_error.is_none() {
            facts.first_error = Some(error.into());
        }
    }
    fn check(&self) -> Result<(), String> {
        self.observations
            .lock()
            .unwrap()
            .first_error
            .clone()
            .map_or(Ok(()), Err)
    }
    fn cleanup_fail(&self, error: impl Into<String>) {
        let error = error.into();
        self.fail(error.clone());
        self.observations.lock().unwrap().cleanup_errors.push(error);
    }
}

#[derive(Clone)]
struct ProbeService {
    pool: WorkerPool,
    expected: String,
    worker_id: String,
    capacity: u32,
    shared: Arc<Shared>,
    drain: watch::Receiver<bool>,
    deadline: Instant,
}

#[tonic::async_trait]
impl WorkerService for ProbeService {
    type ConnectStream = Pin<Box<dyn Stream<Item = Result<wire::ServerMessage, Status>> + Send>>;

    async fn connect(
        &self,
        request: Request<Streaming<wire::WorkerMessage>>,
    ) -> Result<Response<Self::ConnectStream>, Status> {
        self.shared.observations.lock().unwrap().connect_calls += 1;
        if self.shared.connected.swap(true, Ordering::SeqCst) {
            self.shared
                .fail("a second Connect RPC was attempted; retry is forbidden");
            return Err(Status::failed_precondition(
                "probe accepts exactly one Connect RPC",
            ));
        }
        let remote = request.remote_addr();
        self.shared.observations.lock().unwrap().remote_address = remote.map(|v| v.to_string());
        if !remote.is_some_and(|v| v.ip().is_loopback()) {
            self.shared
                .fail("probe peer is not an observed loopback address");
            return Err(Status::permission_denied("loopback peer required"));
        }
        // The original Pool consumes Hello and validates model/protocol. Incoming is
        // passed unchanged; this wrapper does not reimplement its receive loop.
        let response = match timeout_at(self.deadline, self.pool.connect(request)).await {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => {
                self.shared
                    .fail(format!("production pool rejected Connect: {error}"));
                return Err(error);
            }
            Err(_) => {
                self.shared.fail("deadline while admitting Worker Hello");
                return Err(Status::deadline_exceeded("probe admission deadline"));
            }
        };
        let views = self.pool.views();
        let accepted = views.len() == 1
            && views[0].id == self.worker_id
            && views[0]
                .model
                .eq_ignore_ascii_case(&self.expected)
            && views[0].capacity == self.capacity;
        if !accepted {
            self.shared
                .fail("accepted pool member differs from the exact planned Worker identity");
            return Err(Status::failed_precondition("unexpected Worker identity"));
        }
        let connection = views[0].connection_id.clone();
        self.shared.observations.lock().unwrap().accepted_worker = Some(views[0].clone());
        let (tx, rx) = mpsc::channel(8);
        let pool_stream = response.into_inner();
        let shared = self.shared.clone();
        let identity = self.expected.clone();
        let drain = self.drain.clone();
        let deadline = self.deadline;
        let task = tokio::spawn(async move {
            forward(
                pool_stream,
                tx,
                drain,
                shared,
                identity,
                connection,
                deadline,
            )
            .await;
        });
        *self.shared.forward_task.lock().unwrap() = Some(task);
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

async fn enqueue(
    tx: &mpsc::Sender<Result<wire::ServerMessage, Status>>,
    message: Result<wire::ServerMessage, Status>,
    deadline: Instant,
) -> Result<(), String> {
    timeout_at(deadline, tx.send(message))
        .await
        .map_err(|_| "deadline forwarding an outgoing gRPC message".to_owned())?
        .map_err(|_| "gRPC response receiver closed before forwarding completed".to_owned())
}

async fn forward(
    mut original: <WorkerPool as WorkerService>::ConnectStream,
    tx: mpsc::Sender<Result<wire::ServerMessage, Status>>,
    mut drain: watch::Receiver<bool>,
    shared: Arc<Shared>,
    expected: String,
    connection: String,
    deadline: Instant,
) {
    let result: Result<(), String> = async {
        let mut sent_drain = false;
        loop {
            if *drain.borrow() && !sent_drain {
                enqueue(&tx, Ok(wire::ServerMessage {
                    payload: Some(wire::server_message::Payload::Drain(wire::Drain {
                        reason: "isolated one-request probe finished; no reconnect".into(),
                    })),
                }), deadline).await?;
                sent_drain = true;
                shared.observations.lock().unwrap().drain_enqueued = true;
            }
            let item = tokio::select! {
                biased;
                _ = tokio::time::sleep_until(deadline) => return Err("original deadline reached while forwarding".into()),
                changed = drain.changed(), if !sent_drain => {
                    changed.map_err(|_| "probe Drain control closed".to_owned())?;
                    continue;
                }
                item = original.next() => item,
            };
            match item {
                Some(Ok(message)) => {
                    let kind = {
                        let facts = shared.observations.lock().unwrap();
                        match &message.payload {
                            Some(wire::server_message::Payload::Welcome(welcome)) => {
                                if welcome.protocol_version != PROTOCOL_VERSION
                                    || welcome.connection_id != connection
                                    || facts.welcome_enqueued != 0 {
                                    return Err("production Welcome identity/count mismatch".into());
                                }
                                0
                            }
                            Some(wire::server_message::Payload::Evaluate(request)) => {
                                if facts.welcome_enqueued != 1 || sent_drain
                                    || request.model_sha256 != expected
                                    || request.session_id != SESSION
                                    || facts.evaluations_enqueued >= REQUEST_BUDGET {
                                    return Err("production EvalRequest identity/order/budget mismatch".into());
                                }
                                1
                            }
                            Some(wire::server_message::Payload::Cancel(_)) => {
                                2
                            }
                            _ => return Err("unexpected original Pool message".into()),
                        }
                    };
                    let request_record = match &message.payload {
                        Some(wire::server_message::Payload::Evaluate(request)) => Some(serde_json::json!({
                            "task_id": request.task_id, "generation": request.generation,
                            "session_id": request.session_id, "input_hash": request.input_hash,
                            "model_sha256": request.model_sha256, "lease_ms": request.lease_ms,
                        })),
                        _ => None,
                    };
                    enqueue(&tx, Ok(message), deadline).await?;
                    let mut facts = shared.observations.lock().unwrap();
                    match kind {
                        0 => facts.welcome_enqueued += 1,
                        1 => {
                            facts.evaluations_enqueued += 1;
                            facts.evaluation_request = request_record;
                        },
                        _ => facts.original_cancels_enqueued += 1,
                    }
                }
                Some(Err(status)) => {
                    let allowed_teardown = sent_drain && status.code() == tonic::Code::Unavailable
                        && status.message() == "worker stream ended";
                    shared.observations.lock().unwrap().original_response_terminal_status = Some(status.to_string());
                    // Preserve even the original Pool terminal status. Pool uses
                    // this same status for incoming EOF and transport error;
                    // neither this status nor member removal proves which occurred.
                    enqueue(&tx, Err(status), deadline).await?;
                    if !allowed_teardown {
                        return Err("production response ended outside expected post-Drain teardown".into());
                    }
                }
                None => {
                    shared.observations.lock().unwrap().original_response_stream_eof = true;
                    if !sent_drain {
                        return Err("production response stream ended before Drain".into());
                    }
                    return Ok(());
                }
            }
        }
    }.await;
    if let Err(error) = result {
        shared.fail(error);
    }
    shared.observations.lock().unwrap().forward_loop_finished = true;
}

fn fresh_file(path: &Path) -> Result<File, String> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("reserve fresh output {}: {e}", path.display()))
}

fn write_json(file: &mut File, value: &impl Serialize) -> Result<(), String> {
    serde_json::to_writer_pretty(&mut *file, value).map_err(|e| e.to_string())?;
    file.write_all(b"\n")
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())
}

async fn protocol_work(
    pool: &WorkerPool,
    expected: &str,
    shared: &Shared,
    deadline: Instant,
) -> Result<(), String> {
    loop {
        shared.check()?;
        if shared.observations.lock().unwrap().welcome_enqueued == 1 {
            break;
        }
        if Instant::now() >= deadline {
            return Err("deadline waiting for the exact Worker Hello".into());
        }
        sleep(Duration::from_millis(5)).await;
    }
    if pool.model().as_deref() != Some(expected) {
        return Err("pool model changed before dispatch".into());
    }
    let mut search = Search::new(
        Position::new(19, 7.5).map_err(|e| e.to_string())?,
        Default::default(),
    )
    .map_err(|e| e.to_string())?;
    let SearchStep::Evaluate(request) = search.next_evaluation().map_err(|e| e.to_string())? else {
        return Err("empty 19x19 Search did not emit its first legal evaluation".into());
    };
    let token = request.token;
    let (tx, mailbox) = sync_mpsc::channel();
    if Instant::now() >= deadline {
        return Err("deadline before the sole dispatch".into());
    }
    if pool.dispatch(SESSION, &request, tx) {
        shared.observations.lock().unwrap().requests_dispatched += 1;
    } else {
        return Err("sole dispatch had no Worker credit; no retry".into());
    }
    let outcome = loop {
        shared.check()?;
        if Instant::now() >= deadline {
            return Err("deadline awaiting the sole evaluation result".into());
        }
        match mailbox.try_recv() {
            Ok(value) => break value,
            Err(sync_mpsc::TryRecvError::Empty) => sleep(Duration::from_millis(5)).await,
            Err(sync_mpsc::TryRecvError::Disconnected) => {
                return Err("result mailbox closed without Outcome".into());
            }
        }
    };
    let (actual_token, result) = outcome.into_parts();
    if actual_token.generation != token.generation || actual_token.id != token.id {
        return Err("Outcome token mismatch".into());
    }
    match result {
        Ok(output) => drop(output), // Do not inspect, compare, or grade any NN value.
        Err(error) => {
            let text = format!(
                "{}: {} (retryable={})",
                error.code, error.message, error.retryable
            );
            shared.observations.lock().unwrap().evaluation_failure = Some(text.clone());
            return Err(format!("Worker returned EvalFailure: {text}"));
        }
    }
    let views = pool.views();
    let accepted_connection = shared
        .observations
        .lock()
        .unwrap()
        .accepted_worker
        .as_ref()
        .map(|v| v.connection_id.clone())
        .ok_or("missing accepted Worker evidence")?;
    if views.len() != 1
        || views[0].connection_id != accepted_connection
        || views[0].in_flight != 0
        || views[0].retiring_in_flight != 0
        || views[0].assigned_requests != 1
        || views[0].completed != 1
        || views[0].failures != 0
        || views[0].retired_results != 0
        || views[0].metrics.result_messages_received != 1
        || views[0].metrics.request_bytes_enqueued == 0
        || views[0].metrics.request_bytes_streamed == 0
        || views[0].metrics.result_bytes_received == 0
    {
        return Err("one-request completion/capacity/protocol counters mismatch".into());
    }
    if mailbox.try_recv().is_ok() {
        return Err("more than one Outcome was delivered".into());
    }
    let mut facts = shared.observations.lock().unwrap();
    facts.completed_worker = Some(views[0].clone());
    facts.outcome_token = Some(serde_json::json!({"generation": actual_token.generation, "id": actual_token.id}));
    facts.outcomes_accepted = 1;
    Ok(())
}

async fn run(args: Args) -> Result<(), String> {
    let started = Instant::now();
    let deadline = started + Duration::from_secs(args.timeout_seconds);
    let work_deadline = deadline - Duration::from_secs(5);
    let started_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis();
    if !args.output.is_absolute() || args.worker_id.is_empty() || args.worker_id.len() > 128 {
        return Err("absolute fresh output and exact Worker ID (1..128 bytes) required".into());
    }
    if args.model_sha256.len() != 64 || !args.model_sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("model SHA256 must be exactly 64 hexadecimal characters".into());
    }
    let identity = args.model_sha256.to_ascii_lowercase();
    let pool = WorkerPool::with_scheduling(
        Some(identity.clone()),
        Duration::from_secs(args.timeout_seconds - 5),
        SchedulingConfig {
            scheduler: Scheduler::Legacy,
            target_inflight: Some(1),
            ..Default::default()
        },
    )
    .map_err(str::to_owned)?;
    let stem = args
        .output
        .file_stem()
        .and_then(|v| v.to_str())
        .ok_or("output needs a Unicode file stem")?;
    let result_path = args.output.with_file_name(format!("{stem}.result.json"));
    let pending_path = args
        .output
        .with_file_name(format!("{stem}.endpoint.pending"));
    let result_pending = args.output.with_file_name(format!("{stem}.result.pending"));
    if args.output.try_exists().map_err(|e| e.to_string())?
        || result_path.try_exists().map_err(|e| e.to_string())? {
        return Err("endpoint output already exists; no overwrite or retry".into());
    }
    // Reserve only the staging name. Publish a complete endpoint atomically
    // with a no-replace hard link; retain the staging file as evidence too.
    let mut endpoint_file = fresh_file(&pending_path)?;
    let mut result_file = fresh_file(&result_pending)?;
    let shared = Arc::new(Shared {
        observations: Mutex::new(Observations::default()),
        connected: AtomicBool::new(false),
        forward_task: Mutex::new(None),
    });
    let (drain_tx, drain_rx) = watch::channel(false);
    let mut endpoint: Option<String> = None;
    let mut server: Option<JoinHandle<Result<(), tonic::transport::Error>>> = None;
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let work = async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| e.to_string())?;
        let address = listener.local_addr().map_err(|e| e.to_string())?;
        endpoint = Some(format!("http://{address}"));
        let service = ProbeService {
            pool: pool.clone(),
            expected: identity.clone(),
            worker_id: args.worker_id.clone(),
            capacity: args.capacity,
            shared: shared.clone(),
            drain: drain_rx,
            deadline,
        };
        let grpc = WorkerServiceServer::new(service)
            .max_decoding_message_size(4 * 1024 * 1024)
            .max_encoding_message_size(4 * 1024 * 1024);
        server = Some(tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(grpc)
                .serve_with_incoming_shutdown(
                    tonic::transport::server::TcpIncoming::from(listener).with_nodelay(Some(true)),
                    async {
                        let _ = stop_rx.await;
                    },
                )
                .await
        }));
        write_json(
            &mut endpoint_file,
            &serde_json::json!({
                "schema": "rustgo-original-worker-grpc-probe-endpoint-v1",
                "endpoint": endpoint, "model_sha256": identity, "capacity": args.capacity,
                "worker_id": args.worker_id, "request_budget": REQUEST_BUDGET,
                "timeout_seconds": args.timeout_seconds, "started_unix_ms": started_unix_ms,
                "result_path": result_path, "status": "LISTENING_FOR_ONE_EXACT_WORKER",
            }),
        )?;
        if Instant::now() >= work_deadline {
            return Err("work deadline crossed before endpoint publication".into());
        }
        std::fs::hard_link(&pending_path, &args.output)
            .map_err(|e| format!("publish fresh complete endpoint: {e}"))?;
        protocol_work(&pool, &identity, &shared, work_deadline).await
    }
    .await;
    if let Err(error) = work {
        shared.fail(error);
    }
    if shared.check().is_err() {
        pool.cancel_session(SESSION);
    }
    shared.observations.lock().unwrap().drain_requested = true;
    let _ = drain_tx.send(true);
    // This is a bounded protocol teardown, not proof of OS process exit.
    loop {
        let facts = shared.observations.lock().unwrap();
        let finished = facts.forward_loop_finished;
        let rejected_before_admission =
            facts.first_error.is_some() && facts.accepted_worker.is_none();
        drop(facts);
        if !pool.available()
            && (finished || rejected_before_admission || !shared.connected.load(Ordering::SeqCst))
        {
            break;
        }
        if Instant::now() >= deadline {
            shared.cleanup_fail("deadline during Drain/response teardown");
            break;
        }
        sleep(Duration::from_millis(5)).await;
    }
    shared.observations.lock().unwrap().pool_empty_after_drain = !pool.available();
    let forward_task = shared.forward_task.lock().unwrap().take();
    if let Some(mut task) = forward_task {
        match timeout_at(deadline, &mut task).await {
            Ok(Ok(())) => shared.observations.lock().unwrap().forward_task_joined = true,
            Ok(Err(e)) => shared.cleanup_fail(format!("forward task failed: {e}")),
            Err(_) => {
                task.abort();
                let _ = task.await;
                shared.cleanup_fail("forward task required abort at original deadline");
            }
        }
    }
    let _ = stop_tx.send(());
    if let Some(mut task) = server {
        match timeout_at(deadline, &mut task).await {
            Ok(Ok(Ok(()))) => shared.observations.lock().unwrap().server_joined = true,
            Ok(result) => {
                shared.cleanup_fail(format!("gRPC server failed during shutdown: {result:?}"))
            }
            Err(_) => {
                task.abort();
                let _ = task.await;
                shared.observations.lock().unwrap().server_aborted = true;
                shared.cleanup_fail("gRPC server required abort at original deadline");
            }
        }
    }
    if Instant::now() >= deadline {
        shared.cleanup_fail("original probe deadline exceeded before final record");
    }
    if pool.model().as_deref() != Some(identity.as_str()) {
        shared.fail("pool model changed during protocol teardown");
    }
    {
        let facts = shared.observations.lock().unwrap();
        if facts.first_error.is_none()
            && !(facts.connect_calls == 1
                && facts.welcome_enqueued == 1
                && facts.evaluations_enqueued == REQUEST_BUDGET
                && facts.requests_dispatched == REQUEST_BUDGET
                && facts.outcomes_accepted == 1
                && facts.original_cancels_enqueued == 0
                && facts.drain_enqueued
                && facts.original_response_stream_eof
                && facts.forward_task_joined
                && facts.pool_empty_after_drain
                && facts.server_joined
                && !facts.server_aborted)
        {
            drop(facts);
            shared.fail("final protocol/teardown evidence is incomplete");
        }
    }
    let success = shared.check().is_ok();
    write_json(
        &mut result_file,
        &serde_json::json!({
            "schema": "rustgo-original-worker-grpc-probe-result-v1",
            "status": if success { "PROTOCOL_PROBE_COMPLETED" } else { "FAILED" },
            "endpoint": endpoint, "expected_model_sha256": identity, "worker_id": args.worker_id,
            "capacity": args.capacity, "pool_model_after_teardown": pool.model(),
            "request_budget": REQUEST_BUDGET, "timeout_seconds": args.timeout_seconds,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "observations": &*shared.observations.lock().unwrap(),
            "incoming_half_close_observed": null,
            "incoming_half_close_limit": "Pool maps incoming EOF and transport error to the same terminal status; unknown here",
            "drain_acknowledgement": "protocol has no explicit Drain ACK",
            "worker_exit_observed": false, "worker_launched_by_harness": false,
            "accuracy_evaluated": false, "performance_qualified": false,
            "requires_normal_probe_exit": true,
            "scope": "one legal Search request, unmodified original-v1 Pool model/task routing and gRPC completion; external owned Worker exit is separate",
        }),
    )?;
    let record_written_within_deadline = Instant::now() < deadline;
    drop(result_file);
    std::fs::hard_link(&result_pending, &result_path)
        .map_err(|e| format!("publish fresh complete result: {e}"))?;
    if !record_written_within_deadline || Instant::now() >= deadline {
        return Err("original deadline crossed during result publication; OS exit remains separate".into());
    }
    if success { Ok(()) } else { shared.check() }
}

#[tokio::main]
async fn main() {
    if let Err(error) = run(Args::parse()).await {
        eprintln!("original-worker-probe failed: {error}");
        std::process::exit(1);
    }
}

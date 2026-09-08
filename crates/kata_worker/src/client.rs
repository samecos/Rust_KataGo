//! Bounded, connection-scoped transport for the Go Server evaluator protocol.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;
use tokio::time::{MissedTickBehavior, timeout};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Endpoint;

use crate::wire::server_message::Payload as ServerPayload;
use crate::wire::worker_message::Payload as WorkerPayload;
use crate::wire::worker_service_client::WorkerServiceClient;
use crate::wire::{self, EvalRequest, EvalResult, WorkerHeartbeat, WorkerMessage};
use crate::{Evaluator, INPUT_PROFILE, PROTOCOL_VERSION};

const NETWORK_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_MESSAGE_BYTES: usize = 4 * 1024 * 1024;
const MAX_LEASE_MS: u64 = 3_600_000;

#[derive(Clone, Debug)]
pub struct WorkerConfig {
    pub server: String,
    pub worker_id: String,
    /// Accepted requests, including queued and cancelled-but-still-running work.
    /// This never changes the evaluator's GPU batch configuration.
    pub capacity: u32,
    pub once: bool,
    pub reconnect_delay: Duration,
    pub heartbeat_interval: Duration,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct TaskKey {
    session_id: String,
    generation: u64,
    task_id: u64,
}

impl From<&EvalRequest> for TaskKey {
    fn from(request: &EvalRequest) -> Self {
        Self {
            session_id: request.session_id.clone(),
            generation: request.generation,
            task_id: request.task_id,
        }
    }
}

struct Completed {
    key: TaskKey,
    result: EvalResult,
    cancelled: Arc<AtomicBool>,
    received: Instant,
    finished: Instant,
    lease: Duration,
}

#[derive(Default)]
struct Counters {
    completed: u64,
    failed: u64,
}

impl Counters {
    fn record(&mut self, result: &EvalResult) {
        if result.error_code.is_empty() {
            self.completed = self.completed.saturating_add(1);
        } else {
            self.failed = self.failed.saturating_add(1);
        }
    }
}

#[derive(Debug)]
enum SessionEnd {
    Shutdown,
    Drained,
}

/// Run until shutdown, server drain, or the first disconnection with `once`.
/// A new connection cannot start until all old physical evaluator calls return.
pub async fn run(
    evaluator: Arc<dyn Evaluator>,
    config: WorkerConfig,
    mut shutdown: watch::Receiver<bool>,
) -> Result<()> {
    if !(1..=4096).contains(&config.capacity) {
        bail!("worker capacity must be in 1..=4096");
    }
    if config.worker_id.is_empty() || config.worker_id.len() > 128 {
        bail!("worker_id must contain 1..=128 UTF-8 bytes");
    }
    if config.heartbeat_interval.is_zero() {
        bail!("heartbeat_interval must be positive");
    }
    let server = if config.server.contains("://") {
        config.server.clone()
    } else {
        format!("http://{}", config.server)
    };
    let endpoint = Endpoint::from_shared(server)
        .context("invalid worker server endpoint")?
        .connect_timeout(NETWORK_TIMEOUT)
        .tcp_keepalive(Some(Duration::from_secs(10)));
    let instance_id = uuid::Uuid::new_v4().to_string();
    let mut counters = Counters::default();

    loop {
        if shutdown_requested(&shutdown) {
            return Ok(());
        }
        let attempt = connect_and_serve(
            evaluator.clone(),
            &config,
            endpoint.clone(),
            &instance_id,
            &mut counters,
            &mut shutdown,
        )
        .await;
        match attempt {
            Ok(SessionEnd::Shutdown | SessionEnd::Drained) => return Ok(()),
            Err(_) if shutdown_requested(&shutdown) => return Ok(()),
            Err(error) if config.once => return Err(error),
            Err(error) => log::warn!("NN worker connection ended: {error:#}; reconnecting"),
        }
        tokio::select! {
            biased;
            () = wait_for_shutdown(&mut shutdown) => return Ok(()),
            () = tokio::time::sleep(config.reconnect_delay) => {},
        }
    }
}

async fn connect_and_serve(
    evaluator: Arc<dyn Evaluator>,
    config: &WorkerConfig,
    endpoint: Endpoint,
    instance_id: &str,
    counters: &mut Counters,
    shutdown: &mut watch::Receiver<bool>,
) -> Result<SessionEnd> {
    let channel = tokio::select! {
        biased;
        () = wait_for_shutdown(shutdown) => return Ok(SessionEnd::Shutdown),
        connection = timeout(NETWORK_TIMEOUT, endpoint.connect()) => {
            connection.context("worker connection timed out")?.context("worker connection failed")?
        }
    };
    // The RPC is itself named Connect, so the generated endpoint convenience
    // constructor is disabled. Build a channel explicitly and use `new`.
    let mut client = WorkerServiceClient::new(channel)
        .max_decoding_message_size(MAX_MESSAGE_BYTES)
        .max_encoding_message_size(MAX_MESSAGE_BYTES);
    let (outgoing, receiver) = mpsc::channel(config.capacity as usize * 2 + 8);
    let metadata = evaluator.metadata();
    enqueue(
        &outgoing,
        WorkerPayload::Hello(wire::WorkerHello {
            worker_id: config.worker_id.clone(),
            instance_id: instance_id.to_owned(),
            protocol_version: PROTOCOL_VERSION,
            model_sha256: metadata.model_sha256.clone(),
            model_version: metadata.model_version,
            input_profile: INPUT_PROFILE.to_owned(),
            max_in_flight: config.capacity,
            max_board_size: 19,
            supports_ownership: true,
            supports_shortterm_error: metadata.supports_shortterm_error,
            engine_commit: metadata.engine_commit.clone(),
            backend_info: metadata.backend_info.clone(),
            default_always_compute_pass_alive: metadata.default_always_compute_pass_alive,
            default_exclude_territory_adjacent_to_atari: metadata
                .default_exclude_territory_adjacent_to_atari,
            supports_friendly_pass_search: true,
        }),
    )?;
    let mut incoming = tokio::select! {
        biased;
        () = wait_for_shutdown(shutdown) => return Ok(SessionEnd::Shutdown),
        response = timeout(NETWORK_TIMEOUT, client.connect(ReceiverStream::new(receiver))) => {
            response.context("worker stream opening timed out")?
                .context("worker Connect RPC failed")?.into_inner()
        }
    };
    let welcome = tokio::select! {
        biased;
        () = wait_for_shutdown(shutdown) => return Ok(SessionEnd::Shutdown),
        message = timeout(NETWORK_TIMEOUT, incoming.message()) => {
            message.context("worker welcome timed out")?.context("worker welcome failed")?
                .ok_or_else(|| anyhow!("worker stream closed before Welcome"))?
        }
    };
    match welcome.payload {
        Some(ServerPayload::Welcome(welcome)) if welcome.protocol_version == PROTOCOL_VERSION => {
            log::info!("NN worker connected: {}", welcome.connection_id);
        }
        Some(ServerPayload::Welcome(welcome)) => {
            bail!(
                "unsupported worker protocol version {} (expected {PROTOCOL_VERSION})",
                welcome.protocol_version
            );
        }
        _ => bail!("first server message must be Welcome"),
    }

    let mut active: HashMap<TaskKey, Arc<AtomicBool>> = HashMap::new();
    let mut jobs = JoinSet::new();
    let mut heartbeat = tokio::time::interval(config.heartbeat_interval);
    heartbeat.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut draining = false;

    // Keep errors inside this future so every post-admission exit runs cleanup.
    let outcome: Result<SessionEnd> = async {
        loop {
            if draining && active.is_empty() {
                return Ok(SessionEnd::Drained);
            }
            tokio::select! {
                biased;
                () = wait_for_shutdown(shutdown) => return Ok(SessionEnd::Shutdown),
                _ = heartbeat.tick() => {
                    let (nn_rows, nn_batches) = evaluator.stats();
                    enqueue(&outgoing, WorkerPayload::Heartbeat(WorkerHeartbeat {
                        in_flight: active.len() as u32,
                        completed_requests: counters.completed,
                        failed_requests: counters.failed,
                        nn_rows,
                        nn_batches,
                    }))?;
                }
                completed = jobs.join_next(), if !jobs.is_empty() => {
                    let completed = completed.expect("nonempty job set")
                        .context("worker blocking task failed outside evaluator panic guard")?;
                    let completed = finalize(completed);
                    active.remove(&completed.key);
                    counters.record(&completed.result);
                    enqueue(&outgoing, WorkerPayload::Result(completed.result))?;
                }
                message = incoming.message() => {
                    let message = message.context("worker stream receive failed")?
                        .ok_or_else(|| anyhow!("worker server disconnected"))?;
                    match message.payload {
                        Some(ServerPayload::Evaluate(request)) => {
                            let received = Instant::now();
                            // Active exact duplicates share their original result.
                            // Never send a second completion that retires that lease.
                            let key = TaskKey::from(&request);
                            if active.contains_key(&key) {
                                continue;
                            }
                            let rejected = if draining {
                                Some(("DRAINING", "Worker is draining"))
                            } else if !request.model_sha256.eq_ignore_ascii_case(&evaluator.metadata().model_sha256) {
                                Some(("MODEL_MISMATCH", "Requested model hash differs from loaded model"))
                            } else if request.task_id == 0 || request.session_id.is_empty()
                                || request.input_hash.is_empty() || request.lease_ms == 0
                                || request.lease_ms > MAX_LEASE_MS {
                                Some(("INVALID_REQUEST", "Missing identity/input hash or lease outside 1..=3600000 ms"))
                            } else if active.len() >= config.capacity as usize {
                                Some(("CAPACITY_EXCEEDED", "Worker request capacity is full"))
                            } else {
                                None
                            };
                            if let Some((code, message)) = rejected {
                                let mut result = result_identity(&request, &evaluator.metadata().model_sha256);
                                set_error(&mut result, code, message);
                                result.elapsed_us = micros(received.elapsed());
                                counters.record(&result);
                                enqueue(&outgoing, WorkerPayload::Result(result))?;
                                continue;
                            }
                            let cancelled = Arc::new(AtomicBool::new(false));
                            active.insert(key.clone(), cancelled.clone());
                            let evaluator = evaluator.clone();
                            let enqueued = Instant::now();
                            jobs.spawn_blocking(move || compute(
                                evaluator, request, key, cancelled, received, enqueued,
                            ));
                        }
                        Some(ServerPayload::Cancel(cancel)) => {
                            let key = TaskKey {
                                session_id: cancel.session_id,
                                generation: cancel.generation,
                                task_id: cancel.task_id,
                            };
                            if let Some(cancelled) = active.get(&key) {
                                cancelled.store(true, Ordering::Release);
                            }
                        }
                        Some(ServerPayload::Drain(drain)) => {
                            log::info!("NN worker draining: {}", drain.reason);
                            draining = true;
                        }
                        Some(ServerPayload::Welcome(_)) => bail!("duplicate worker Welcome"),
                        None => bail!("server message has no payload"),
                    }
                }
            }
        }
    }.await;

    // Half-close only on graceful drain. Buffered results remain in ReceiverStream
    // and are consumed before its EOF; waiting for peer EOF lets them reach it.
    drop(outgoing);
    if matches!(&outcome, Ok(SessionEnd::Drained)) {
        tokio::select! {
            biased;
            () = wait_for_shutdown(shutdown) => {},
            _ = timeout(NETWORK_TIMEOUT, async {
                while incoming.message().await?.is_some() {}
                Ok::<(), tonic::Status>(())
            }) => {},
        }
    }
    drop(incoming);
    // A cancelled or disconnected request retains its physical slot until the
    // blocking task returns. Old results are counted but never enter a new stream.
    for cancelled in active.values() {
        cancelled.store(true, Ordering::Release);
    }
    while let Some(completed) = jobs.join_next().await {
        match completed {
            Ok(completed) => counters.record(&finalize(completed).result),
            Err(error) => {
                counters.failed = counters.failed.saturating_add(1);
                log::error!("NN worker task cleanup failed: {error}");
            }
        }
    }
    if draining {
        // Drain is a process-level request to stop accepting work, including
        // when the peer closes the old stream before all results can be sent.
        if let Err(error) = &outcome {
            log::warn!("NN worker stream ended during drain: {error:#}");
        }
        Ok(SessionEnd::Drained)
    } else {
        outcome
    }
}

fn compute(
    evaluator: Arc<dyn Evaluator>,
    request: EvalRequest,
    key: TaskKey,
    cancelled: Arc<AtomicBool>,
    received: Instant,
    enqueued: Instant,
) -> Completed {
    let started = Instant::now();
    let lease = Duration::from_millis(request.lease_ms);
    let mut result = result_identity(&request, &evaluator.metadata().model_sha256);
    result.queue_us = Some(micros(started.saturating_duration_since(enqueued)));
    if cancelled.load(Ordering::Acquire) {
        set_error(&mut result, "CANCELLED", "Evaluation cancelled");
    } else if started.saturating_duration_since(received) >= lease {
        set_error(
            &mut result,
            "LEASE_EXPIRED",
            "Evaluation lease expired before execution",
        );
    } else {
        match catch_unwind(AssertUnwindSafe(|| evaluator.evaluate(&request))) {
            Ok(report) => {
                result.context_us = report.context_us;
                result.evaluator_us = report.evaluator_us;
                match report.result {
                    Ok(output) => result.output = Some(output),
                    Err(error) => set_error(&mut result, &error.code, &error.message),
                }
            }
            Err(panic) => {
                let message = panic
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("non-string evaluator panic");
                // A panic can happen during any evaluator stage; do not invent a
                // context/evaluator timer when that stage was not reported.
                set_error(&mut result, "WORKER_INTERNAL_ERROR", message);
            }
        }
    }
    Completed {
        key,
        result,
        cancelled,
        received,
        finished: Instant::now(),
        lease,
    }
}

fn finalize(mut completed: Completed) -> Completed {
    if completed.cancelled.load(Ordering::Acquire) {
        set_error(&mut completed.result, "CANCELLED", "Evaluation cancelled");
    } else if completed
        .finished
        .saturating_duration_since(completed.received)
        >= completed.lease
    {
        set_error(
            &mut completed.result,
            "LEASE_EXPIRED",
            "Evaluation lease expired",
        );
    }
    // Output conversion finished on the blocking thread. Time spent waiting
    // for JoinSet polling belongs to result delivery, not evaluation latency.
    completed.result.elapsed_us = micros(
        completed
            .finished
            .saturating_duration_since(completed.received),
    );
    completed
}

fn result_identity(request: &EvalRequest, model_sha256: &str) -> EvalResult {
    EvalResult {
        task_id: request.task_id,
        generation: request.generation,
        session_id: request.session_id.clone(),
        input_hash: request.input_hash.clone(),
        model_sha256: model_sha256.to_owned(),
        ..Default::default()
    }
}

fn set_error(result: &mut EvalResult, code: &str, message: &str) {
    result.output = None;
    result.error_code = code.to_owned();
    result.error_message = message.to_owned();
}

fn micros(duration: Duration) -> u64 {
    duration.as_micros().min(u64::MAX as u128) as u64
}

fn enqueue(sender: &mpsc::Sender<WorkerMessage>, payload: WorkerPayload) -> Result<()> {
    // Never let a non-reading peer block cancellation or heartbeat processing.
    // Overflow tears down this connection; accepted computations still drain.
    sender
        .try_send(WorkerMessage {
            payload: Some(payload),
        })
        .map_err(|error| anyhow!("worker outgoing stream unavailable: {error}"))
}

fn shutdown_requested(shutdown: &watch::Receiver<bool>) -> bool {
    *shutdown.borrow() || shutdown.has_changed().is_err()
}

async fn wait_for_shutdown(shutdown: &mut watch::Receiver<bool>) {
    loop {
        if *shutdown.borrow_and_update() {
            return;
        }
        if shutdown.changed().await.is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elapsed_excludes_delay_between_compute_completion_and_finalization() {
        // Simulate a completed job that sat unpolled for almost ten seconds,
        // without sleeping or depending on scheduler timing.
        let received = Instant::now() - Duration::from_secs(10);
        let finished = received + Duration::from_micros(3_000);
        let completed = finalize(Completed {
            key: TaskKey {
                session_id: "timer-test".to_owned(),
                generation: 1,
                task_id: 1,
            },
            result: EvalResult::default(),
            cancelled: Arc::new(AtomicBool::new(false)),
            received,
            finished,
            lease: Duration::from_secs(1),
        });
        assert_eq!(completed.result.elapsed_us, 3_000);
        // The physical evaluation also completed inside its lease; later
        // delivery must not turn this into a lease-expired result.
        assert!(completed.result.error_code.is_empty());
    }
}

//! Pure neural-network worker for Go Server's bidirectional gRPC protocol.

use std::{sync::Arc, time::Duration};

use anyhow::Context;
use clap::Parser;
use kata_worker::{
    client::{self, WorkerConfig},
    evaluator::Engine,
};

use crate::cli::CommonArgs;

#[derive(Debug, Parser)]
#[command(about = "Serve Go Server evaluations; the server owns all search")]
struct WorkerArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Go Server gRPC address (HOST:PORT or http://HOST:PORT).
    #[arg(long, default_value = "127.0.0.1:50051")]
    server: String,
    /// Unique ID for this simultaneously running worker.
    #[arg(long, default_value = "rustgo-local")]
    worker_id: String,
    /// Maximum admitted NN requests; independent of nnMaxBatchSize.
    #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(u32).range(1..=4096))]
    capacity: u32,
    /// Expected SHA-256 of the exact local model file.
    #[arg(long)]
    model_sha256: Option<String>,
    /// Exit after one connection instead of reconnecting.
    #[arg(long)]
    once: bool,
    /// Permit synthetic dummy evaluations for isolated integration tests only.
    #[arg(long)]
    allow_dummy: bool,
}

pub fn nnworker(args: &[String]) -> i32 {
    let parsed = match WorkerArgs::try_parse_from(
        std::iter::once("nnworker").chain(args.iter().map(String::as_str)),
    ) {
        Ok(parsed) => parsed,
        Err(error) => {
            let code = error.exit_code();
            let _ = error.print();
            return code;
        }
    };
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();
    match run(parsed) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("nnworker: {error:#}");
            1
        }
    }
}

fn run(args: WorkerArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        !args.worker_id.is_empty() && args.worker_id.len() <= 128,
        "worker-id must contain 1..128 UTF-8 bytes"
    );
    let model = args
        .common
        .get_model_file()
        .map_err(|e| anyhow::anyhow!(e.message))?;
    let cfg = args
        .common
        .get_config("analysis_example.cfg")
        .map_err(|e| anyhow::anyhow!(e.message))?;
    let engine = Arc::new(Engine::load(
        &model,
        args.model_sha256.as_deref(),
        &cfg,
        args.capacity,
        args.allow_dummy,
    )?);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_stack_size(8 * 1024 * 1024)
        .enable_all()
        .build()
        .context("create worker runtime")?;
    runtime.block_on(async move {
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let signal = tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                let _ = shutdown_tx.send(true);
            }
        });
        let result = client::run(
            engine,
            WorkerConfig {
                server: args.server,
                worker_id: args.worker_id,
                capacity: args.capacity,
                once: args.once,
                reconnect_delay: Duration::from_secs(2),
                heartbeat_interval: Duration::from_secs(2),
            },
            shutdown_rx,
        )
        .await;
        signal.abort();
        result
    })
}

# Original v1 Worker probe

This separate Rust crate uses the unchanged `D:/Go/Server` crates through absolute path dependencies. It neither modifies Server sources nor starts a Worker. It is a bounded, one-request protocol integration program, not an accuracy test or a throughput benchmark. On 2026-10-02 the root controller compiled it and completed one original-v1 request with the new quantized Worker; both processes exited normally with code 0. Receipts are in `target/original-worker-probe-build-r1` and `target/worker-original-protocol-composition-r1`.

Build once into a fresh, independent target (root owns compilation):

```powershell
cargo build --offline --manifest-path D:/code/Rust_KataGo/tools/original_worker_probe/Cargo.toml --target-dir D:/code/Rust_KataGo/target/original-worker-probe-build-r1
```

This crate has its own `[workspace]` and checked-in `Cargo.lock`; use `--locked` for subsequent builds. The actual root build used `target/original-worker-probe-build-r1/cargo`, so its executable is in that directory's `debug` subdirectory. Do not substitute the old typed probe binary or attach this probe to an existing production Worker.

Example CLI, to be executed only by the external owned controller:

```powershell
& D:/code/Rust_KataGo/target/original-worker-probe-build-r1/debug/original-worker-probe.exe --model-sha256 MODEL_64_HEX --worker-id UNIQUE_PROBE_WORKER --capacity 32 --output D:/code/Rust_KataGo/target/FRESH/endpoint.json --timeout-seconds 300
```

The parent directory must already exist. Endpoint, endpoint staging, result, and result staging names must be fresh. `--capacity` defaults to 32; model SHA, Worker ID, output and timeout are required. Timeout is 20–600 seconds and starts at program entry; the final 5 seconds are reserved within it for Drain/server teardown, with no renewed deadline. The endpoint is always ephemeral `127.0.0.1:0`. The external controller waits for the complete endpoint JSON, then launches its owned original-v1 Worker with that endpoint, the exact Worker ID/model and `--once`. This program never launches, kills or claims an OS exit for that process.

Both JSON products are written as complete `create_new` staging files, newline + `sync_all`, then published by a no-replace hard link. For `endpoint.json`, products are `endpoint.endpoint.pending`, `endpoint.json`, `endpoint.result.pending`, and `endpoint.result.json`. Staging files are retained, including on errors. A final publication that crosses the original deadline yields a nonzero program exit even if its complete candidate record exists; consumers must require the external actual exit code 0.

Only original v1 APIs are used:

- `D:/Go/Server/crates/go-server/src/worker.rs:188`: `WorkerPool::with_scheduling(Some(model_sha256), lease, SchedulingConfig)` with `Scheduler::Legacy` and target in-flight 1. The original Pool validates Hello protocol/model and owns receive/result routing.
- `worker.rs:804`: the original `WorkerService::connect` consumes the unmodified incoming tonic stream. The wrapper checks the resulting one WorkerView for exact Worker ID, model and capacity, and forwards the original Welcome/Evaluate/Cancel/terminal status unchanged.
- `D:/Go/Server/crates/go-core/src/search.rs:906` and `:1068`: an empty 19×19 position and default real `Search` create exactly its first `SearchStep::Evaluate`. No synthetic EvalRequest replaces it.
- `worker.rs:379`: one call to `pool.dispatch(session, &request, sender)`. A false return is a permanent probe failure; there is no retry or second request.
- `worker.rs:36`: `Outcome::into_parts` supplies the actual token and result. The token is matched, successful output is immediately dropped, and no NN value is examined or applied to Search.

Success requires one connection, Welcome, dispatched/evaluated request and accepted Outcome; the actual completed WorkerView must have assigned 1, completed 1, failures 0, in-flight 0, retiring 0, retired-results 0 and result-messages 1. Actual request identifiers and Outcome token are recorded. The original Pool itself checks worker connection/task/generation/session/input hash/model when admitting the result. There is no wire execution profile, profile ACK, or Server recipe enforcement.

After the one result the wrapper enqueues original-v1 Drain, follows the original response stream through its terminal status/EOF, confirms Pool removal, and joins its forward/server tasks within the original bound. Original Pool maps incoming EOF and transport error to the same `Unavailable("worker stream ended")` terminal status, so incoming half-close stays `null`/unknown. Drain has no acknowledgement. External owned Worker wait/actual exit 0 remains a separate mandatory observation; this record cannot substitute for it.

The reference implementation was the preserved typed probe `target/unified-quant-server-profile-worktree-r1/crates/go-server/examples/execution_profile_probe.rs` (SHA256 `32fc782f4356bca04b6c5ed9b08ed9d41a522ec7b0bdad9587cd988ac6e7fd21`). Only a new crate was written. Original Server sources, old probe, measured executables and prior evidence are retained.

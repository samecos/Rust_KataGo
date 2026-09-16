//! Target-only B14/L2 forward diagnostic. The parent exposes the exact test name.
//! No production dispatch, kernel arithmetic, or installed file is changed here.

use super::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Barrier, Condvar, Mutex};
use std::time::Instant;

const BATCH: usize = 14;
const LANES: usize = 2;
const WARMUP: usize = 80;
const ITERATIONS: usize = 1000;
const EMPTY_SPATIAL_SHA: &str = "6a8a3ff7b49bbb0d286f3a610cf7429c1ad22e2d5c3ab6cce999fe78d2646c9f";
const EMPTY_GLOBAL_SHA: &str = "56d4b091feb300b528e541c469fef90bcef0ca1c6e7fbb3523440b1512e24ade";

// Thread creation must finish before any worker can enter the fixed-count
// barriers. A failed spawn cancels all waiting workers without entering them.
struct StartGate {
    decision: Mutex<Option<bool>>,
    changed: Condvar,
}

impl StartGate {
    fn new() -> Self {
        Self {
            decision: Mutex::new(None),
            changed: Condvar::new(),
        }
    }

    fn wait(&self) -> bool {
        let mut decision = self.decision.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(run) = *decision {
                return run;
            }
            decision = self
                .changed
                .wait(decision)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    fn resolve(&self, run: bool) {
        let mut decision = self.decision.lock().unwrap_or_else(|e| e.into_inner());
        if decision.is_none() {
            *decision = Some(run);
            self.changed.notify_all();
        }
    }
}

struct CancelPendingStarts<'a>(&'a StartGate);
impl Drop for CancelPendingStarts<'_> {
    fn drop(&mut self) {
        // No panic on poisoned synchronization state. First decision wins, so
        // normal Drop cannot cancel a successfully committed pair of workers.
        self.0.resolve(false);
    }
}

fn caught<T>(phase: &str, run: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(run)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic");
        Err(format!("{phase} panicked: {message}"))
    })
}

fn empty_inputs() -> (Vec<f32>, Vec<f32>, Value) {
    // Exact nnbench::batch_inputs(empty): one Black-to-play Tromp-Taylor board,
    // fill_row_v7 with NCHW, symmetry=0 and policy_optimism=0, then repeat bits.
    assert_eq!((SPATIAL, GLOBAL), (22 * 361, 19));
    let mut spatial = vec![0.0f32; BATCH * SPATIAL];
    let mut global = vec![0.0f32; BATCH * GLOBAL];
    let mut params = MiscNNInputParams::default();
    params.symmetry = 0;
    params.policy_optimism = 0.0;
    let board = Board::new(19, 19);
    let rules = Rules::get_tromp_taylorish();
    let hist = BoardHistory::new(board.clone(), P_BLACK, rules.clone(), 0);
    fill_row_v7(
        &board,
        &hist,
        P_BLACK,
        &params,
        19,
        19,
        false,
        &mut spatial[..SPATIAL],
        &mut global[..GLOBAL],
    );
    for row in 1..BATCH {
        spatial.copy_within(0..SPATIAL, row * SPATIAL);
        global.copy_within(0..GLOBAL, row * GLOBAL);
    }
    assert_eq!(input_hash(&spatial, &[]), EMPTY_SPATIAL_SHA);
    assert_eq!(input_hash(&[], &global), EMPTY_GLOBAL_SHA);
    let description = json!({
        "kind":"empty","board":[19,19],"inputs_version":7,
        "spatial":"B,22,19,19 NCHW f32","global":"B,19 f32",
        "input_hash_encoding":"SHA256 of contiguous little-endian f32 bytes, before device conversion",
        "position_generation":"one empty board repeated across the physical batch; Black to play",
        "rules":rules.to_json(),"encore_phase":0,"effective_symmetry":0,
        "policy_optimism":params.policy_optimism,
        "misc_params":{
            "draw_equivalent_wins_for_white":params.draw_equivalent_wins_for_white,
            "conservative_pass_and_is_root":params.conservative_pass_and_is_root,
            "enable_passing_hacks":params.enable_passing_hacks,
            "playout_doubling_advantage":params.playout_doubling_advantage,
            "nn_policy_temperature":params.nn_policy_temperature,
            "avoid_mytdagger_hack":params.avoid_mytdagger_hack,
            "max_history":params.max_history
        },
        "fork_comparison":"same semantic empty-board control as Fork benchmarknn; cross-implementation feature-byte equality is not asserted"
    });
    (spatial, global, description)
}

fn save_f32(
    directory: &Path,
    file: &str,
    values: &[f32],
    shape: &[usize],
) -> Result<Value, String> {
    if shape.iter().product::<usize>() != values.len() || !values.iter().all(|v| v.is_finite()) {
        return Err(format!("nonfinite or wrong-shaped raw output: {file}"));
    }
    let path = directory.join(file);
    if path.exists() {
        return Err(format!("refuse raw evidence overwrite: {}", path.display()));
    }
    let digest = write_f32(&path, values);
    Ok(json!({"file":file,"sha256":digest,"shape":shape,"dtype":"f32le","bytes":values.len()*4}))
}

fn input_readback(
    directory: &Path,
    lane: &Lane,
    lane_index: usize,
    phase: &str,
    spatial: &[f32],
    global: &[f32],
) -> Result<Value, String> {
    let actual_spatial = lane
        .stream
        .clone_dtoh(&lane.spatial)
        .map_err(|e| format!("input spatial DTOH: {e}"))?;
    let actual_global = lane
        .stream
        .clone_dtoh(&lane.global)
        .map_err(|e| format!("input global DTOH: {e}"))?;
    lane.stream
        .synchronize()
        .map_err(|e| format!("input DTOH sync: {e}"))?;
    let equal_bits = |a: &[f32], b: &[f32]| {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    };
    if !equal_bits(&actual_spatial, spatial) || !equal_bits(&actual_global, global) {
        return Err(format!("lane {lane_index} {phase} input bits changed"));
    }
    Ok(json!({
        "lane":lane_index,"phase":phase,"input_bits_equal":true,
        "spatial":save_f32(directory,&format!("lane{lane_index}/input-{phase}-spatial.f32le"),&actual_spatial,&[BATCH,22,19,19])?,
        "global":save_f32(directory,&format!("lane{lane_index}/input-{phase}-global.f32le"),&actual_global,&[BATCH,19])?
    }))
}

fn warmup(rt: &CudaRuntime, model: &CudaModel, lanes: &mut [Lane]) -> Result<(), String> {
    std::thread::scope(|scope| {
        let jobs: Vec<_> = lanes
            .iter_mut()
            .enumerate()
            .map(|(index, lane)| {
                scope.spawn(move || {
                    caught("lane warmup", || {
                        rt.device
                            .bind_to_thread()
                            .map_err(|e| format!("lane {index} context bind: {e}"))?;
                        for _ in 0..WARMUP {
                            model.apply(
                                rt,
                                &lane.stream,
                                &mut lane.workspace,
                                &lane.spatial,
                                &lane.global,
                            )?;
                        }
                        lane.stream
                            .synchronize()
                            .map_err(|e| format!("lane {index} warmup sync: {e}"))
                    })
                })
            })
            .collect();
        let mut errors = Vec::new();
        // Join every submitted lane even when an earlier lane failed.
        for (index, job) in jobs.into_iter().enumerate() {
            match job.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => errors.push(error),
                Err(_) => errors.push(format!("lane {index} warmup worker panicked")),
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    })
}

fn measured_forward(
    rt: &CudaRuntime,
    model: &CudaModel,
    lanes: &mut [Lane],
    report: &mut Value,
) -> Result<(), String> {
    warmup(rt, model, lanes)?;
    // Every allocation and warmup is finished before either timed barrier.
    let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let mut event_sets = Vec::with_capacity(LANES);
    for _ in 0..LANES {
        let mut events = Vec::with_capacity(ITERATIONS);
        for _ in 0..ITERATIONS {
            let start = rt
                .device
                .new_event(flags)
                .map_err(|e| format!("create start event: {e}"))?;
            let end = rt
                .device
                .new_event(flags)
                .map_err(|e| format!("create end event: {e}"))?;
            events.push((start, end));
        }
        event_sets.push(events);
    }
    let ready = Barrier::new(LANES + 1);
    let done = Barrier::new(LANES + 1);
    let setup_failed = AtomicBool::new(false);
    let start_gate = StartGate::new();
    let epoch = Instant::now();
    let intervals = std::thread::scope(|scope| -> Result<Vec<(Instant, Instant)>, String> {
        // This guard drops before scope performs implicit joins on unwind.
        let _cancel_pending = CancelPendingStarts(&start_gate);
        let mut jobs = Vec::with_capacity(LANES);
        for (index, (lane, events)) in lanes.iter_mut().zip(event_sets.iter()).enumerate() {
            let ready = &ready;
            let done = &done;
            let setup_failed = &setup_failed;
            let start_gate = &start_gate;
            let spawned = std::thread::Builder::new().spawn_scoped(scope, move || {
                if !start_gate.wait() {
                    return Err("timed start cancelled after thread creation failure".into());
                }
                let setup = caught("timed lane setup", || {
                    rt.device
                        .bind_to_thread()
                        .map_err(|e| format!("lane {index} context bind: {e}"))
                });
                if setup.is_err() {
                    setup_failed.store(true, Ordering::SeqCst);
                }
                ready.wait();
                let timed = caught("timed forward", || {
                    setup?;
                    if setup_failed.load(Ordering::SeqCst) {
                        return Err("a peer failed before the timed loop".into());
                    }
                    let started = Instant::now();
                    for (begin, end) in events {
                        begin
                            .record(&lane.stream)
                            .map_err(|e| format!("record start: {e}"))?;
                        model.apply(
                            rt,
                            &lane.stream,
                            &mut lane.workspace,
                            &lane.spatial,
                            &lane.global,
                        )?;
                        end.record(&lane.stream)
                            .map_err(|e| format!("record end: {e}"))?;
                    }
                    // Identical to nnbench: enqueue every event pair/apply first,
                    // then synchronize once, never once per iteration.
                    lane.stream
                        .synchronize()
                        .map_err(|e| format!("timed final sync: {e}"))?;
                    Ok((started, Instant::now()))
                });
                // Every failure/panic inside setup or timing still reaches done.
                done.wait();
                timed
            });
            match spawned {
                Ok(job) => jobs.push(job),
                Err(error) => {
                    // Release before joining; cancelled workers never enter
                    // ready/done, whose full participant set does not exist.
                    start_gate.resolve(false);
                    let mut errors = vec![format!("spawn timed lane {index}: {error}")];
                    for job in jobs {
                        match job.join() {
                            Ok(Ok(_)) => errors.push("cancelled lane unexpectedly ran".into()),
                            Ok(Err(error)) => errors.push(error),
                            Err(_) => errors.push("cancelled timed lane panicked".into()),
                        }
                    }
                    return Err(errors.join("; "));
                }
            }
        }
        // Commit only when BOTH threads exist. Between this point and the two
        // barriers there are no fallible operations or early returns.
        start_gate.resolve(true);
        ready.wait();
        done.wait();
        let mut intervals = Vec::with_capacity(LANES);
        let mut errors = Vec::new();
        for (index, job) in jobs.into_iter().enumerate() {
            match job.join() {
                Ok(Ok(interval)) => intervals.push(interval),
                Ok(Err(error)) => errors.push(format!("lane {index}: {error}")),
                Err(_) => errors.push(format!("lane {index} timed worker panicked")),
            }
        }
        if errors.is_empty() {
            Ok(intervals)
        } else {
            Err(errors.join("; "))
        }
    })?;
    // No event read or event destruction occurs before both lanes finish.
    let mut samples = Vec::with_capacity(LANES);
    for (index, (events, (started, ended))) in event_sets.iter().zip(intervals.iter()).enumerate() {
        let mut raw = Vec::with_capacity(ITERATIONS);
        for (begin, end) in events {
            let ms = begin
                .elapsed_ms(end)
                .map_err(|e| format!("read elapsed event: {e}"))?;
            if !ms.is_finite() || ms <= 0.0 {
                return Err(format!("invalid event time: {ms}"));
            }
            raw.push(ms);
        }
        // Preserve native f32 values in launch order, and use nnbench's f32 sort
        // followed by f64 conversion of the two middle elements.
        let mut sorted = raw.clone();
        sorted.sort_by(f32::total_cmp);
        let n = sorted.len();
        let median = (sorted[(n - 1) / 2] as f64 + sorted[n / 2] as f64) * 0.5;
        let elapsed = ended.duration_since(*started).as_secs_f64() * 1000.0;
        if !elapsed.is_finite() || elapsed <= 0.0 {
            return Err("invalid lane wall time".into());
        }
        let per_batch = elapsed / ITERATIONS as f64;
        samples.push(json!({
            "handle":index+1,"lane":index,"iterations":ITERATIONS,
            "elapsed_ms":elapsed,"per_batch_ms":per_batch,"nn_evals_per_s":BATCH as f64*1000.0/per_batch,
            "cuda_event_ms":raw,"cuda_event_median_ms":median,
            "cuda_event_median_nn_evals_per_s":BATCH as f64*1000.0/median,
            "started_offset_ns":started.duration_since(epoch).as_nanos() as u64,
            "ended_offset_ns":ended.duration_since(epoch).as_nanos() as u64
        }));
    }
    let started = intervals
        .iter()
        .map(|(s, _)| *s)
        .min()
        .ok_or("no timed lanes")?;
    let ended = intervals
        .iter()
        .map(|(_, e)| *e)
        .max()
        .ok_or("no timed lanes")?;
    let wall_ms = ended.duration_since(started).as_secs_f64() * 1000.0;
    if !wall_ms.is_finite() || wall_ms <= 0.0 {
        return Err("invalid full-forward wall time".into());
    }
    let rate_sum: f64 = samples
        .iter()
        .map(|s| s["cuda_event_median_nn_evals_per_s"].as_f64().unwrap())
        .sum();
    report["results"] = json!([{
        "batch":BATCH,"spatial_input_sha256":EMPTY_SPATIAL_SHA,"global_input_sha256":EMPTY_GLOBAL_SHA,
        "handles":samples,"wall_elapsed_ms":wall_ms,
        "wall_nn_evals_per_s":LANES as f64*BATCH as f64*ITERATIONS as f64/(wall_ms/1000.0),
        "sum_median_nn_evals_per_s":rate_sum
    }]);
    report["all_timed_lanes_joined"] = json!(true);
    report["raw_event_count"] = json!(LANES * ITERATIONS);
    Ok(())
}

fn execute(directory: &Path, enabled: bool, report: &mut Value) -> Result<(), String> {
    let selection = selected_tactics();
    require_current_tactics(&selection);
    let model_dir =
        std::env::var_os("KATAGO_TEST_MODEL_DIR").ok_or("KATAGO_TEST_MODEL_DIR is required")?;
    let model_path = PathBuf::from(model_dir).join(MODEL);
    let bytes = std::fs::read(&model_path).map_err(|e| format!("read model: {e}"))?;
    if hex::encode(Sha256::digest(&bytes)) != MODEL_SHA256 {
        return Err("compressed model SHA mismatch".into());
    }
    let desc = load_model_from_bytes(&bytes, true, true)
        .map_err(|e| format!("parse native model: {e}"))?;
    if desc.sha256 != MODEL_SHA256 {
        return Err("parser model identity differs".into());
    }
    let graph = lower_model(&desc).map_err(|e| format!("lower native model: {e}"))?;
    if (
        graph.board_size,
        graph.trunk_channels,
        graph.mid_channels,
        graph.num_blocks,
        graph.num_heads,
        graph.head_dim,
    ) != (19, 768, 384, 11, 12, 32)
    {
        return Err("native TF3 graph shape differs".into());
    }
    let rt = CudaRuntime::new()?;
    rt.validate_residual_algo_request()?;
    if rt.cublaslt_handle().is_none() {
        return Err("fixed L2 diagnostic requires cuBLASLt".into());
    }
    let build = backend_build_fingerprint();
    let device = device_fingerprint(&rt.device).map_err(|e| format!("device fingerprint: {e}"))?;
    if !build.capabilities.dual_ffn || build.fp16_encoding_revision != Some(1) {
        return Err("compiled DualFFN/RNE1 identity required".into());
    }
    let load_stream = rt
        .device
        .new_stream()
        .map_err(|e| format!("model load stream: {e}"))?;
    let model = CudaModel::load(&graph, &rt, &load_stream)?;
    rt.device
        .synchronize()
        .map_err(|e| format!("model upload sync: {e}"))?;
    let (spatial, global, input_description) = empty_inputs();
    report["model"] = json!(model_path);
    report["model_sha256"] = json!(MODEL_SHA256);
    report["build"] = serde_json::to_value(build).map_err(|e| e.to_string())?;
    report["device"] = json!({"gpu_name":device.gpu_name,"compute_capability":device.compute_capability,"sm_count":device.sm_count,"l2_cache_bytes":device.l2_cache_bytes});
    report["selected_tactics"] = json!(selection);
    report["tactics"] = json!(
        selection
            .iter()
            .map(|(k, v)| (k.clone(), v["resolved"].clone()))
            .collect::<BTreeMap<_, _>>()
    );
    report["input_description"] = input_description;
    report["input_sha256"] = json!(input_hash(&spatial, &global));
    report["installed_plan_id"] = json!(installed_plan_id());
    let mut lanes = Vec::with_capacity(LANES);
    let mut resources = Vec::with_capacity(LANES);
    for index in 0..LANES {
        let stream = rt
            .device
            .new_stream()
            .map_err(|e| format!("lane stream: {e}"))?;
        let workspace = CudaWorkspace::new(&stream, &model, BATCH)?;
        let mut d_spatial = stream
            .alloc_zeros::<f32>(spatial.len())
            .map_err(|e| format!("spatial allocation: {e}"))?;
        let mut d_global = stream
            .alloc_zeros::<f32>(global.len())
            .map_err(|e| format!("global allocation: {e}"))?;
        stream
            .memcpy_htod(&spatial, &mut d_spatial)
            .map_err(|e| format!("spatial upload: {e}"))?;
        stream
            .memcpy_htod(&global, &mut d_global)
            .map_err(|e| format!("global upload: {e}"))?;
        let (act384_ptr, act384_guard) = workspace.act384.device_ptr(&stream);
        let (spatial_ptr, spatial_guard) = d_spatial.device_ptr(&stream);
        let (global_ptr, global_guard) = d_global.device_ptr(&stream);
        let (lt_ptr, _) = rt.cublaslt_workspace_ptr(&stream);
        if act384_ptr == 0 || lt_ptr == 0 || workspace.act384.len() != BATCH * 361 * 384 {
            return Err("invalid lane workspace".into());
        }
        resources.push(json!({"lane":index,"handle":index+1,"stream_id":stream.cu_stream() as usize,
            "workspace_batch":workspace.batch(),"act384_device_ptr":act384_ptr,"act384_bytes":workspace.act384.len()*4,
            "spatial_device_ptr":spatial_ptr,"global_device_ptr":global_ptr,"cublaslt_workspace_device_ptr":lt_ptr,
            "cublaslt_workspace_bytes":rt.cublaslt_workspace_len()}));
        drop((act384_guard, spatial_guard, global_guard));
        std::fs::create_dir(directory.join(format!("lane{index}")))
            .map_err(|e| format!("create lane output directory: {e}"))?;
        lanes.push(Lane {
            stream,
            workspace,
            spatial: d_spatial,
            global: d_global,
        });
    }
    for key in [
        "stream_id",
        "act384_device_ptr",
        "spatial_device_ptr",
        "global_device_ptr",
        "cublaslt_workspace_device_ptr",
    ] {
        if resources[0][key] == resources[1][key] {
            return Err(format!("lane resources alias: {key}"));
        }
    }
    rt.device
        .synchronize()
        .map_err(|e| format!("all inputs ready: {e}"))?;
    let mut before = Vec::with_capacity(LANES);
    for (index, lane) in lanes.iter().enumerate() {
        before.push(input_readback(
            directory, lane, index, "before", &spatial, &global,
        )?);
    }
    report["lane_resources"] = json!(resources);
    report["input_before"] = json!(before);
    // CacheScope owns stream clones only. All lane allocations stay alive until
    // both scoped workers are joined, the device is drained and finish returns.
    let mut cache = l2_cache::CacheScope::begin(&rt, &lanes, enabled)?;
    let work = caught("forward experiment", || {
        report["l2_diagnostic"] = cache.metadata();
        report["status"] = json!("RUNNING_L2_INNER_FORWARD");
        measured_forward(&rt, &model, &mut lanes, report)?;
        let mut after = Vec::with_capacity(LANES);
        let mut outputs = Vec::with_capacity(LANES);
        for (index, lane) in lanes.iter().enumerate() {
            after.push(input_readback(
                directory, lane, index, "after", &spatial, &global,
            )?);
            let out = lane.workspace.to_host(&lane.stream)?;
            lane.stream
                .synchronize()
                .map_err(|e| format!("final output sync: {e}"))?;
            let mut files = serde_json::Map::new();
            for (name, values, width) in [
                ("policy", out.policy.as_slice(), 6 * 362),
                ("value", out.value.as_slice(), 3),
                ("misc", out.misc.as_slice(), 10),
                ("moremisc", out.moremisc.as_slice(), 8),
                ("ownership", out.ownership.as_slice(), 361),
            ] {
                files.insert(
                    name.into(),
                    save_f32(
                        directory,
                        &format!("lane{index}/final-{name}.f32le"),
                        values,
                        &[BATCH, width],
                    )?,
                );
            }
            outputs.push(json!({"lane":index,"files":files,"scope":"all physical rows and all five raw heads; policy includes six channels"}));
        }
        report["input_after"] = json!(after);
        report["inputs_unchanged"] = json!(true);
        report["final_outputs"] = json!(outputs);
        Ok(())
    });
    // Cleanup is attempted even after any warmup/timing/readback error or panic.
    let drained = caught("final device drain", || {
        rt.device
            .synchronize()
            .map_err(|e| format!("final device drain: {e}"))
    });
    let cleanup = caught("cache cleanup", || cache.finish());
    match &cleanup {
        Ok(metadata) => report["l2_diagnostic"] = metadata.clone(),
        Err(error) => report["l2_cleanup_error"] = json!(error),
    }
    let mut errors = Vec::new();
    if let Err(error) = work {
        errors.push(error);
    }
    if let Err(error) = drained {
        errors.push(error);
    }
    if let Err(error) = cleanup {
        errors.push(error);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

pub(super) fn fixed_l2_inner_forward() {
    match std::env::var("KATAGO_RUN_L2_INNER_DIAG") {
        Err(std::env::VarError::NotPresent) => {
            eprintln!("skipped: set KATAGO_RUN_L2_INNER_DIAG=1 in an exclusive GPU window");
            return;
        }
        Ok(value) if value == "1" => {}
        value => panic!("KATAGO_RUN_L2_INNER_DIAG must be 1 when set: {value:?}"),
    }
    let enabled = l2_enabled();
    integer_env("KATAGO_DUMP_BATCH", BATCH, &[BATCH]);
    integer_env("KATAGO_DUMP_LANES", LANES, &[LANES]);
    let directory =
        PathBuf::from(std::env::var_os("KATAGO_DUMP_DIR").expect("KATAGO_DUMP_DIR is required"));
    std::fs::create_dir_all(&directory).expect("create forward dump directory");
    assert!(
        std::fs::read_dir(&directory)
            .expect("read output directory")
            .next()
            .is_none(),
        "forward output directory must be new or empty"
    );
    let executable = std::env::current_exe().expect("test executable path");
    let mut report = json!({
        "schema":1,"status":"PREPARING_L2_INNER_FORWARD","pass":false,"production_certified":false,
        "command":"fixed_l2_inner_forward","mode":"kernel","kernel_only":true,"timing":"cuda-event",
        "input":"empty","batch":BATCH,"lanes":LANES,"requested_batches":[BATCH],"handles":[1,2],
        "warmup":WARMUP,"iterations":ITERATIONS,"l2_enabled":enabled,"graph_execution":false,
        "test_executable":executable,"test_executable_sha256":sha256_file(&executable).expect("hash test executable"),
        "forward_source_sha256":hex::encode(Sha256::digest(include_bytes!("l2_forward.rs"))),
        "dump_test_source_sha256":hex::encode(Sha256::digest(include_bytes!("../l2_inner_diag.rs"))),
        "l2_cache_source_sha256":hex::encode(Sha256::digest(include_bytes!("l2_cache.rs"))),
        "results":[],"timing_scope":{
            "loop":"enqueue all 1000 start/apply/end pairs, then one stream synchronize per lane",
            "cuda_event_storage":"native f32 milliseconds in launch order; all 2000 retained",
            "median":"sort f32; convert both middle values to f64; average in f64",
            "primary":"sum of B14*1000/lane_event_median_ms",
            "wall":"minimum lane started Instant to maximum lane ended Instant, including each lane final sync",
            "excludes":"allocation, input copies, warmup, event allocation/read/destruction, raw downloads, cache setup/cleanup",
            "not_a_worker_or_overlap_claim":true
        }
    });
    let outcome = caught("fixed L2 inner forward", || {
        execute(&directory, enabled, &mut report)
    });
    match &outcome {
        Ok(()) => {
            report["status"] = json!("COMPLETE_L2_INNER_FORWARD");
            report["pass"] = json!(true);
        }
        Err(error) => {
            report["status"] = json!("FAIL_L2_INNER_FORWARD");
            report["error"] = json!(error);
        }
    }
    let serialized =
        serde_json::to_vec_pretty(&report).expect("serialize retained forward evidence");
    std::fs::write(directory.join("report.json"), serialized).expect("write forward report");
    outcome.expect("fixed L2 inner forward failed; report retained");
    println!(
        "L2_INNER_FORWARD batch=14 lanes=2 l2={} path={}",
        u8::from(enabled),
        directory.display()
    );
}

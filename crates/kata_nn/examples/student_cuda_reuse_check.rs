//! Single-process raw-head validation across retained capacities and threads.
//! Arguments: MODEL SPATIAL-B8.f32 GLOBAL-B8.f32 OUTPUT.json.
//! The caller compares every case with independently prepared FP32 CPU goldens.

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use kata_nn::backends::student_cuda::{StudentCudaModel, StudentOutputs};
    use kata_nn::student_model::StudentModel;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::sync::{Arc, Barrier};

    const SPATIAL_ROW: usize = 22 * 361;
    const GLOBAL_ROW: usize = 19;
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err(
            "usage: student_cuda_reuse_check MODEL SPATIAL-B8.f32 GLOBAL-B8.f32 OUTPUT.json".into(),
        );
    }
    let read = |path: &str| -> Result<(Vec<f32>, String), Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        if bytes.len() % 4 != 0 {
            return Err("FP32 input byte count must be divisible by four".into());
        }
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        Ok((values, hex::encode(Sha256::digest(&bytes))))
    };
    let model_bytes = std::fs::read(&args[0])?;
    let model_sha = hex::encode(Sha256::digest(&model_bytes));
    let parsed = StudentModel::parse(&model_bytes)?;
    let model = Arc::new(StudentCudaModel::load(&parsed, 0)?);
    let (spatial8, spatial_sha) = read(&args[1])?;
    let (global8, global_sha) = read(&args[2])?;
    if spatial8.len() != 8 * SPATIAL_ROW || global8.len() != 8 * GLOBAL_ROW {
        return Err("reuse check requires exactly eight input rows".into());
    }
    let spatial32 = spatial8.repeat(4);
    let global32 = global8.repeat(4);
    let raw_case = |name: String, batch: usize, outputs: StudentOutputs| -> Value {
        json!({
            "name": name, "batch": batch,
            "shapes": {"policy_logits": [batch,362], "value_logits": [batch,3],
                "score": [batch], "ownership_logits": [batch,361]},
            "policy_logits": outputs.policy_logits, "value_logits": outputs.value_logits,
            "score": outputs.score, "ownership_logits": outputs.ownership_logits,
        })
    };
    let mut cases = Vec::new();
    for cycle in 0..2 {
        for batch in [32, 1, 8, 32] {
            let outputs = model.apply(
                &spatial32[..batch * SPATIAL_ROW],
                &global32[..batch * GLOBAL_ROW],
                batch,
            )?;
            cases.push(raw_case(
                format!("serial-cycle{cycle}-b{batch}-case{}", cases.len()),
                batch,
                outputs,
            ));
        }
    }
    let serial_stats = model.execution_reuse_stats()?;
    if serial_stats["execution_creations"] != 1
        || serial_stats["execution_reuses"] != 7
        || serial_stats["workspace_growths"] != 1
        || serial_stats["idle_executions"] != 1
        || serial_stats["discarded_executions"] != 0
    {
        return Err(format!("serial resources were not reused: {serial_stats}").into());
    }
    // Both workers begin their independent calls from the same barrier. The
    // pool checkout owns the workspace for the full kernel/copy completion.
    let barrier = Barrier::new(2);
    let parallel = std::thread::scope(|scope| -> Result<Vec<StudentOutputs>, String> {
        let mut threads = Vec::new();
        for _ in 0..2 {
            threads.push(scope.spawn(|| {
                barrier.wait();
                model.apply(&spatial32, &global32, 32)
            }));
        }
        threads
            .into_iter()
            .map(|thread| {
                thread
                    .join()
                    .map_err(|_| "student CUDA parallel validation thread panicked".to_owned())?
            })
            .collect()
    })?;
    for (index, outputs) in parallel.into_iter().enumerate() {
        cases.push(raw_case(format!("parallel-thread{index}-b32"), 32, outputs));
    }
    let final_stats = model.execution_reuse_stats()?;
    if final_stats["discarded_executions"] != 0 || final_stats["idle_limit"] != 8 {
        return Err(format!("unexpected resource completion state: {final_stats}").into());
    }
    let output = json!({
        "schema": "rust_go_student_cuda_reuse_check_v1", "status": "completed",
        "model_sha256": model_sha,
        "input_spatial_b8_sha256": spatial_sha, "input_global_b8_sha256": global_sha,
        "input_contract": "first B8 rows; B1 is first row; B32 repeats B8 four times; NCHW/NC",
        "execution_facts": model.execution_facts()?, "serial_stats": serial_stats,
        "final_stats": final_stats, "concurrent_threads": 2, "cases": cases,
        "scope": "raw four-head implementation and resource-reuse validation; no strength or speed claim",
    });
    // Create once: failures are preserved for the caller, never overwritten.
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args[3])?;
    file.write_all(&serde_json::to_vec(&output)?)?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("student_cuda_reuse_check requires the cuda feature");
    std::process::exit(1);
}

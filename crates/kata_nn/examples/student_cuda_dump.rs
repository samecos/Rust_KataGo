//! Dump raw CUDA heads for comparison with the FP32 PyTorch reference.
//! cargo run -p kata_nn --features cuda --example student_cuda_dump -- MODEL
//!   SPATIAL.f32 GLOBAL.f32 BATCH OUTPUT.json

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use kata_nn::backends::student_cuda::StudentCudaModel;
    use kata_nn::student_model::StudentModel;
    use std::path::Path;
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err(
            "usage: student_cuda_dump MODEL SPATIAL.f32 GLOBAL.f32 BATCH OUTPUT.json".into(),
        );
    }
    let read_f32 = |path: &Path| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        if bytes.len() % 4 != 0 {
            return Err("FP32 input byte count must be divisible by four".into());
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect())
    };
    let model = StudentModel::parse(&std::fs::read(&args[0])?)?;
    let cuda = StudentCudaModel::load(&model, 0)?;
    let batch = args[3].parse::<usize>()?;
    let outputs = cuda.apply(
        &read_f32(Path::new(&args[1]))?,
        &read_f32(Path::new(&args[2]))?,
        batch,
    )?;
    let json = serde_json::json!({
        "batch": batch,
        "shapes": {"policy_logits": [batch, 362], "value_logits": [batch, 3], "score": [batch], "ownership_logits": [batch, 361]},
        "execution_facts": cuda.execution_facts()?,
        "policy_logits": outputs.policy_logits, "value_logits": outputs.value_logits,
        "score": outputs.score, "ownership_logits": outputs.ownership_logits,
    });
    std::fs::write(&args[4], serde_json::to_vec(&json)?)?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("student_cuda_dump requires the cuda feature");
    std::process::exit(1);
}

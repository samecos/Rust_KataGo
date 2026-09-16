//! Real first-QKV operands and the public TN cuBLASLt half-output baseline.
//! Independent probe support only: no AOT ABI, timing, algorithm query or cache access.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use kata_nn::backends::cuda::{
    CublasLtWeightLayout, CudaRuntime, backend_build_fingerprint, device_fingerprint,
    f16_to_f32_bits, f32_to_f16_bits,
};
use kata_nn::model_parser::load_model_from_bytes;
use kata_nn::native_model::lower_model;
use kata_nn::onnx_parser::Layer;
use kata_nn::tactic_plan::{installed_plan_id, tactic_var};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub const BATCH: usize = 14;
pub const M: usize = BATCH * 361;
pub const N: usize = 1152;
pub const K: usize = 384;
const MODEL_SHA: &str = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6";
const INPUT_PAIR_SHA: &str = "f9caf09c4fbf32b48736dda5531a3292bdacccef07db6a9b974678c09ab22911";
const INPUT_SHA: &str = "080172756ec05b22e1a86494e98006dbcf2c754729ce8b94e20e48f72e5240e8";
const META_SHA: &str = "ded73f01e77679dbe8700e86afc4943d2d793ccb83edee48285f524fd2148aa3";
const SOURCE_SHA: &str = "75cd3b95c65f39fdcb189a4b709e6157c3eed147c914b023bfaab7198443cf5e";
const BINARY_SHA: &str = "b252f57021c2bb0864e99f3ba204b712887e627a74d143016a8768d52bd29937";
const RECEIPT_SHA: &str = "7f0b08ea5a3396dbb6935ad091d45dc1c72136d49e2afe4da90a1e56dd36f38d";
const WEIGHTS_FP32_SHA: &str = "486e1b6f8ab31722953102623e84620b1d37c9ac9888879bb29370077a8c6083";
const WEIGHTS_RNE1_SHA: &str = "3d765a0efd09d25445d8ab8de5c488b0b0943f1319d5f17fe4209232dcf92b5e";
const OPERAND_MANIFEST_SHA: &str =
    "a14ae0f0e53bee08ca3fee63e5a63e105d6a830e706e83505f57b839b8c31651";
const PREPARE_SOURCE_SHA: &str = "cbebb0fecabc57eb8a916cc399b60d7e9ddb96b4c2b75c5e0517dcf5460b1ab8";

pub struct HostOperands {
    pub input: Vec<u16>,
    pub weights: Vec<u16>,
    pub weights_fp32: Vec<f32>,
    pub provenance: Value,
}

pub struct DeviceOperands {
    pub input: CudaSlice<u16>,
    pub weights: CudaSlice<u16>,
}

fn require(ok: bool, message: &str) -> Result<(), String> {
    if ok { Ok(()) } else { Err(message.into()) }
}

fn sha(raw: &[u8]) -> String {
    hex::encode(Sha256::digest(raw))
}
fn half_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn float_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bound(path: &Path, expected: &str) -> Result<(Vec<u8>, Value), String> {
    let raw = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    require(
        sha(&raw) == expected,
        &format!("SHA mismatch: {}", path.display()),
    )?;
    let record = json!({"path":path,"sha256":expected,"bytes":raw.len()});
    Ok((raw, record))
}

/// CPU-only feature/weight preparation. Preserve half input bytes, pack weights
/// exactly as lower_model: out-first [Q rows | K rows | V rows, input channel].
pub fn load_operands(root: &Path, model_path: &Path) -> Result<HostOperands, String> {
    let evidence = root.join("target/fork-parity-20260908/g3-prefix-trace-wsl-r2");
    let archive = root.join("target/fork-parity-20260908/g3-prefix-trace-build-r2");
    let (meta_raw, meta_record) = bound(&evidence.join("meta.json"), META_SHA)?;
    let meta: Value = serde_json::from_slice(&meta_raw).map_err(|e| e.to_string())?;
    let (archived_meta, archived_meta_record) = bound(&archive.join("meta.json"), META_SHA)?;
    require(
        archived_meta == meta_raw,
        "archived prefix metadata differs",
    )?;
    let (_, source_record) = bound(
        &archive.join("trace_worker_input_prefix_cuda.rs"),
        SOURCE_SHA,
    )?;
    let (binary, binary_record) = bound(&archive.join("prefix-trace-wsl"), BINARY_SHA)?;
    require(binary.starts_with(b"\x7fELF"), "prefix artifact is not ELF")?;
    let (receipt_raw, receipt_record) = bound(&archive.join("receipt.json"), RECEIPT_SHA)?;
    let receipt: Value = serde_json::from_slice(&receipt_raw).map_err(|e| e.to_string())?;
    require(
        receipt["status"] == "PASS_FROZEN_PREFIX_R2_DIAGNOSTIC_ONLY",
        "prefix receipt incomplete",
    )?;
    require(
        receipt["source"]["sha256"] == SOURCE_SHA
            && receipt["binary"]["sha256"] == BINARY_SHA
            && receipt["meta"]["sha256"] == META_SHA,
        "receipt source/binary/meta identity",
    )?;
    require(
        meta["status"] == "PASS_PREFIX_TRACE_COMPLETE_NOT_CERTIFIED"
            && meta["diagnostic_revision"] == 2,
        "prefix trace must be completed revision2",
    )?;
    require(
        meta["test_source_sha256"] == SOURCE_SHA && meta["test_executable_sha256"] == BINARY_SHA,
        "prefix actual source/executable identity",
    )?;
    require(
        meta["model_sha256"] == MODEL_SHA && meta["actual_input_pair_sha256"] == INPUT_PAIR_SHA,
        "prefix model/real request input identity",
    )?;
    let prefixes = meta["prefixes"]
        .as_array()
        .ok_or("prefixes array missing")?;
    let selected: Vec<_> = prefixes
        .iter()
        .filter(|p| p["prefix_layers"] == 3 && p["physical_batch"] == BATCH)
        .collect();
    require(selected.len() == 1, "unique prefix3 B14 required")?;
    let selected = selected[0];
    require(
        selected["directory"] == "prefix3-b14"
            && selected["lane"] == 0
            && selected["m_tokens"] == M,
        "prefix3 B14 execution shape/lane",
    )?;
    let tensor = &selected["exported_tensors"]["normed"];
    require(
        tensor["file"] == "normed.f16le"
            && tensor["sha256"] == INPUT_SHA
            && tensor["shape"] == json!([BATCH, 361, K])
            && tensor["elements"] == M * K
            && tensor["bytes"] == M * K * 2
            && tensor["dtype"] == "float16"
            && tensor["byte_order"] == "little-endian"
            && tensor["raw_ieee_bits_preserved"] == true
            && tensor["all_finite"] == true,
        "normed raw half descriptor mismatch",
    )?;
    let (input_raw, input_record) = bound(&evidence.join("prefix3-b14/normed.f16le"), INPUT_SHA)?;
    require(input_raw.len() == M * K * 2, "normed input byte count")?;
    let input: Vec<u16> = input_raw
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
        .collect();
    require(
        input.iter().all(|v| f16_to_f32_bits(*v).is_finite()),
        "nonfinite raw input",
    )?;
    let (model_raw, model_record) = bound(model_path, MODEL_SHA)?;
    let desc = load_model_from_bytes(&model_raw, true, true).map_err(|e| e.to_string())?;
    require(desc.sha256 == MODEL_SHA, "parsed TF3 model SHA")?;
    let graph = lower_model(&desc)?;
    require(
        (
            graph.board_size,
            graph.trunk_channels,
            graph.mid_channels,
            graph.num_heads,
            graph.head_dim,
        ) == (19, 768, K, 12, 32),
        "TF3 graph architecture",
    )?;
    require(
        matches!(graph.layers.first(), Some(Layer::InitialConv(_)))
            && matches!(graph.layers.get(1), Some(Layer::Linear(_)))
            && matches!(graph.layers.get(2), Some(Layer::RmsNorm(_))),
        "first attention graph position",
    )?;
    let Some(Layer::Attention(attn)) = graph.layers.get(3) else {
        return Err("layer3 must be first Attention".into());
    };
    require(
        attn.qkv_weight.dims == [N as i64, K as i64]
            && attn.qkv_weight.numel() == N * K
            && (attn.num_heads, attn.head_dim, attn.seq_len) == (12, 32, 361),
        "first QKV matrix shape",
    )?;
    let weights_fp32 = attn.qkv_weight.f32_data().to_vec();
    require(
        weights_fp32.iter().all(|x| x.is_finite()),
        "nonfinite lowered QKV weights",
    )?;
    let weights: Vec<u16> = weights_fp32.iter().copied().map(f32_to_f16_bits).collect();
    require(
        weights.iter().all(|v| f16_to_f32_bits(*v).is_finite()),
        "nonfinite RNE1 QKV weights",
    )?;
    let operands_dir = root.join("target/fork-parity-20260908/qkv-fp32-aot-r1/operands");
    let (operand_manifest_raw, operand_manifest_record) = bound(
        &operands_dir.join("prepared-r1/manifest.json"),
        OPERAND_MANIFEST_SHA,
    )?;
    let operand_manifest: Value =
        serde_json::from_slice(&operand_manifest_raw).map_err(|e| e.to_string())?;
    require(
        operand_manifest["status"] == "CPU_PREPARED_QKV_OPERANDS_NOT_GPU_VERIFIED"
            && operand_manifest["model"]["sha256"] == MODEL_SHA
            && operand_manifest["input"]["sha256"] == INPUT_SHA
            && operand_manifest["prefix_meta"]["sha256"] == META_SHA
            && operand_manifest["first_attention_layer_index"] == 3
            && operand_manifest["weight_fp32"]["sha256"] == WEIGHTS_FP32_SHA
            && operand_manifest["weight_rne1"]["sha256"] == WEIGHTS_RNE1_SHA,
        "independent CPU operand manifest identity",
    )?;
    let (_, prepare_source_record) = bound(&operands_dir.join("prepare.py"), PREPARE_SOURCE_SHA)?;
    let (reference_fp32, reference_fp32_record) = bound(
        &operands_dir.join("prepared-r1/qkv-source.f32le"),
        WEIGHTS_FP32_SHA,
    )?;
    let (reference_half, reference_half_record) = bound(
        &operands_dir.join("prepared-r1/qkv-rne1.f16le"),
        WEIGHTS_RNE1_SHA,
    )?;
    require(
        reference_fp32.len() == N * K * 4 && reference_half.len() == N * K * 2,
        "independent weight byte counts",
    )?;
    require(
        float_bytes(&weights_fp32) == reference_fp32,
        "lowered QKV differs from independent FP32 parser bytes",
    )?;
    require(
        half_bytes(&weights) == reference_half,
        "RNE1 QKV differs from independent Python half bytes",
    )?;
    let provenance = json!({"scope":"public three-layer prefix raw input plus first Attention lowered QKV; no full-network capture claim",
        "prefix_meta":meta_record,"archived_prefix_meta":archived_meta_record,"source":source_record,
        "executed_binary":binary_record,"receipt":receipt_record,"input":input_record,
        "input_descriptor":tensor,"actual_input_pair_sha256":INPUT_PAIR_SHA,"model":model_record,
        "independent_cpu_oracle":{"manifest":operand_manifest_record,"source":prepare_source_record,
            "weights_fp32":reference_fp32_record,"weights_half":reference_half_record,
            "all_fp32_bytes_equal":true,"all_half_bytes_equal":true},
        "backend_build_fingerprint":meta["backend_build_fingerprint"],"device_fingerprint":meta["device_fingerprint"],
        "weights":{"layer_index":3,"shape":[N,K],"layout":"row-major [N,K], Q|K|V row groups",
            "source_fp32_sha256":sha(&float_bytes(&weights_fp32)),"rne1_half_sha256":sha(&half_bytes(&weights)),
            "source_fp32_bytes":N*K*4,"half_bytes":N*K*2,"fp16_encoding_revision":1,
            "qk_scale_applied":false,"transpose_applied":false,"padding_applied":false},
        "gemm":{"m":M,"n":N,"k":K,"input_layout":"row-major [M,K] half",
            "weights_layout":"TN [N,K] half","output_layout":"row-major [M,N] half Q|K|V",
            "alpha":1.0,"beta":0.0,"compute":"CUBLAS_COMPUTE_32F","scale":"CUDA_R_32F"}});
    Ok(HostOperands {
        input,
        weights,
        weights_fp32,
        provenance,
    })
}

fn baseline_guard(rt: &CudaRuntime, stream: &Arc<CudaStream>) -> Result<(), String> {
    require(
        installed_plan_id().is_none(),
        "standalone QKV baseline cannot use installed plan",
    )?;
    require(
        backend_build_fingerprint().fp16_encoding_revision == Some(1),
        "QKV baseline requires FP16 RNE revision1",
    )?;
    for (key, default, expected) in [
        ("KATAGO_CUDA_CUBLASLT", "1", "1"),
        ("KATAGO_CUDA_CUBLASLT_RANK", "heuristic", "heuristic"),
        ("KATAGO_CUDA_GEMM_LAYOUT", "tn", "tn"),
    ] {
        let actual = match tactic_var(key) {
            Ok(v) => v,
            Err(std::env::VarError::NotPresent) => default.into(),
            Err(e) => return Err(e.to_string()),
        };
        require(
            actual == expected,
            &format!("QKV baseline requires {key}={expected}"),
        )?;
    }
    require(
        !kata_nn::backends::cuda_exec::capturing(),
        "standalone QKV baseline cannot run in capture",
    )?;
    require(
        Arc::ptr_eq(stream.context(), &rt.device),
        "runtime/stream CUDA context mismatch",
    )?;
    require(
        rt.cublaslt_handle().is_some(),
        "cuBLASLt unavailable; no fallback permitted",
    )?;
    stream.context().bind_to_thread().map_err(|e| e.to_string())
}

pub fn upload_operands(
    rt: &CudaRuntime,
    stream: &Arc<CudaStream>,
    host: &HostOperands,
) -> Result<DeviceOperands, String> {
    baseline_guard(rt, stream)?;
    require(
        host.input.len() == M * K
            && host.weights.len() == N * K
            && host.weights_fp32.len() == N * K,
        "host operand dimensions",
    )?;
    require(
        sha(&half_bytes(&host.input)) == INPUT_SHA,
        "host raw input mutated",
    )?;
    require(
        host.provenance["weights"]["source_fp32_sha256"] == WEIGHTS_FP32_SHA
            && sha(&float_bytes(&host.weights_fp32)) == WEIGHTS_FP32_SHA
            && host.provenance["weights"]["rne1_half_sha256"] == WEIGHTS_RNE1_SHA
            && sha(&half_bytes(&host.weights)) == WEIGHTS_RNE1_SHA,
        "host weight identity mutated",
    )?;
    require(
        host.weights_fp32
            .iter()
            .copied()
            .map(f32_to_f16_bits)
            .eq(host.weights.iter().copied()),
        "host weights not exact RNE1",
    )?;
    require(
        serde_json::to_value(backend_build_fingerprint()).map_err(|e| e.to_string())?
            == host.provenance["backend_build_fingerprint"],
        "compiled backend differs from frozen prefix",
    )?;
    let device = device_fingerprint(&rt.device)?;
    require(
        json!({"gpu_name":device.gpu_name,"compute_capability":device.compute_capability,
        "sm_count":device.sm_count,"l2_cache_bytes":device.l2_cache_bytes})
            == host.provenance["device_fingerprint"],
        "actual GPU differs from frozen prefix",
    )?;
    let input = stream.clone_htod(&host.input).map_err(|e| e.to_string())?;
    let weights = stream
        .clone_htod(&host.weights)
        .map_err(|e| e.to_string())?;
    let read_input = stream.clone_dtoh(&input).map_err(|e| e.to_string())?;
    let read_weights = stream.clone_dtoh(&weights).map_err(|e| e.to_string())?;
    stream.synchronize().map_err(|e| e.to_string())?;
    require(
        half_bytes(&read_input) == half_bytes(&host.input),
        "uploaded input DTOH byte mismatch",
    )?;
    require(
        half_bytes(&read_weights) == half_bytes(&host.weights),
        "uploaded weights DTOH byte mismatch",
    )?;
    Ok(DeviceOperands { input, weights })
}

pub fn run_lt_baseline(
    rt: &CudaRuntime,
    stream: &Arc<CudaStream>,
    operands: &DeviceOperands,
    output: &mut CudaSlice<u16>,
) -> Result<(), String> {
    baseline_guard(rt, stream)?;
    require(
        // Public Lt descriptors write only M*N; an optional diagnostic tail
        // remains outside those descriptors for overwrite detection.
        operands.input.len() == M * K && operands.weights.len() == N * K && output.len() >= M * N,
        "device operand/output dimensions",
    )?;
    require(
        Arc::ptr_eq(operands.input.context(), &rt.device)
            && Arc::ptr_eq(operands.weights.context(), &rt.device)
            && Arc::ptr_eq(output.context(), &rt.device),
        "device buffer CUDA context mismatch",
    )?;
    let result = rt.cublaslt_gemm_f16out_with_layout(
        stream,
        &operands.input,
        &operands.weights,
        output,
        M,
        N,
        K,
        CublasLtWeightLayout::Tn,
    );
    // Even API failure is synchronized before the caller may release buffers.
    let completion = stream
        .synchronize()
        .map_err(|e| format!("QKV baseline completion: {e}"));
    let executed = result?;
    completion?;
    require(
        executed,
        "public cuBLASLt half-output path returned false; fallback forbidden",
    )
}

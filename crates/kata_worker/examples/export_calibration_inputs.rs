//! CPU-only export from pinned Worker PB requests to explicit GroupCost tensors.
//! --requests PB_MANIFEST --expected-sha256 SHA --output FRESH_TARGET_DIRECTORY
//! No Engine/NnEvaluator/model/CUDA instance is constructed. Packing is frozen
//! by the prepared manifest (B1/B3/B8, exact tails, at most 128 Inputs per shard).
use anyhow::{Context, Result, ensure};
use kata_worker::{
    evaluator::input_encoding::{
        EncodedWorkerRow, GLOBAL_FLOATS, SPATIAL_FLOATS, encode_request_v7_cpu, f32le,
        row_feature_sha256,
    },
    wire,
};
use prost::Message;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::Write,
    path::{Component, Path, PathBuf},
};

const TOTAL: usize = 2044;
const MAX_SHARD: usize = 128;
const CORPUS_SHA: &str = "c7b80205d26d5dd7c7cf38dceb8b88bcc8b3b452b764f7a83113d156fc88c842";
const REQUESTS_SHA: &str = "c3d3ee4b4f75fadc613e343beaf60c60d95879bb89e6d516cfa5b4278650238f";
const GAMES_SHA: &str = "c0c9de9d8ed7dc8e99f9dda57ebb974e99c49c7bb99601f93235be671acc0768";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    path: String,
    bytes: usize,
    sha256: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    role: String,
    original_path: String,
    file: Binding,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestEntry {
    source_line: usize,
    source_line_sha256: String,
    record: Value,
    task_id: u64,
    pb: Binding,
    input_hash_hex: String,
    wire_semantic_sha256: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Bundle {
    schema: String,
    mode: String,
    split: String,
    source_total: usize,
    selected_count: usize,
    packings: Vec<usize>,
    model_sha256: String,
    session_id: String,
    generation: u64,
    lease_ms: u64,
    ordered_pb_sha256: String,
    sources: Vec<Source>,
    requests: Vec<RequestEntry>,
    producer: Value,
}
#[derive(Clone, Debug, Serialize)]
struct Input {
    id: String,
    physical_batch: usize,
    spatial: Binding,
    global: Binding,
}
#[derive(Debug, Serialize)]
struct Manifest {
    schema: String,
    purpose: String,
    provenance: String,
    inputs: Vec<Input>,
}

fn hash(raw: &[u8]) -> String {
    hex::encode(Sha256::digest(raw))
}
fn valid_sha(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
fn relative_path(name: &str) -> Result<&Path> {
    let path = Path::new(name);
    ensure!(
        !name.is_empty() && path.components().all(|c| matches!(c, Component::Normal(_))),
        "nonlocal artifact path"
    );
    Ok(path)
}
fn read_bound(root: &Path, binding: &Binding, cap: usize) -> Result<Vec<u8>> {
    ensure!(
        valid_sha(&binding.sha256) && binding.bytes <= cap,
        "invalid/oversized binding"
    );
    let path = root.join(relative_path(&binding.path)?);
    // Resolve symlinks before a read so a relative path cannot escape the bundle.
    ensure!(
        path.canonicalize()?.starts_with(root.canonicalize()?),
        "artifact escapes bundle root"
    );
    ensure!(
        fs::metadata(&path)?.len() == binding.bytes as u64,
        "bound length changed"
    );
    let bytes = fs::read(path)?;
    ensure!(
        bytes.len() == binding.bytes && hash(&bytes) == binding.sha256,
        "bound SHA changed"
    );
    Ok(bytes)
}
fn write_new(root: &Path, name: &str, raw: &[u8]) -> Result<Binding> {
    let path = root.join(relative_path(name)?);
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    file.write_all(raw)?;
    file.sync_all()?;
    Ok(Binding {
        path: name.into(),
        bytes: raw.len(),
        sha256: hash(raw),
    })
}
fn write_json<T: Serialize>(root: &Path, name: &str, value: &T) -> Result<Binding> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_new(root, name, &bytes)
}

/// Publish only a fully written, fsynced completion marker, atomically without
/// replacing any existing name. Keep the same-directory pending file for audit.
/// No rename fallback: a filesystem without hard links fails before publication.
fn commit_new_with(
    root: &Path,
    name: &str,
    raw: &[u8],
    write: impl FnOnce(&mut fs::File, &[u8]) -> std::io::Result<()>,
    sync: impl FnOnce(&fs::File) -> std::io::Result<()>,
) -> Result<Binding> {
    let final_path = root.join(relative_path(name)?);
    let pending_path = root.join(relative_path(&format!("{name}.pending"))?);
    let binding = Binding {
        path: name.into(),
        bytes: raw.len(),
        sha256: hash(raw),
    };
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&pending_path)?;
    write(&mut file, raw)?;
    sync(&file)?;
    drop(file);
    fs::hard_link(&pending_path, &final_path)?;
    // No fallible work after the atomic no-replace publication.
    Ok(binding)
}

fn commit_json<T: Serialize>(root: &Path, name: &str, value: &T) -> Result<Binding> {
    let mut raw = serde_json::to_vec_pretty(value)?;
    raw.push(b'\n');
    commit_new_with(
        root,
        name,
        &raw,
        |file, raw| file.write_all(raw),
        fs::File::sync_all,
    )
}

/// Numeric JSON representations 0 and 0.0 express the same PB double here.
/// All integer-valued request fields are <=10000, so this does not blur u64 IDs.
fn same_semantics(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| same_semantics(a, b))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|b| same_semantics(v, b)))
        }
        _ => a == b,
    }
}

fn request_semantics(request: &wire::EvalRequest) -> Result<Value> {
    let p = request.position.as_ref().context("missing position")?;
    let q = request.parameters.as_ref().context("missing parameters")?;
    Ok(
        json!({"position": {"board_size":p.board_size,"komi":p.komi,"rules":p.rules,
        "initial_player":p.initial_player,"next_player":p.next_player,
        "initial_stones":p.initial_stones.iter().map(|s| json!({"color":s.color,"vertex":s.vertex})).collect::<Vec<_>>(),
        "moves":p.moves.iter().map(|m| json!({"color":m.color,"vertex":m.vertex})).collect::<Vec<_>>()},
        "parameters":{"symmetry":q.symmetry,"policy_temperature":q.policy_temperature,
        "policy_optimism":q.policy_optimism,"draw_equivalent_wins_for_white":q.draw_equivalent_wins_for_white,
        "playout_doubling_advantage":q.playout_doubling_advantage,"include_ownership":q.include_ownership,
        "max_history":q.max_history,"conservative_pass":q.conservative_pass,"enable_passing_hacks":q.enable_passing_hacks,
        "always_compute_pass_alive":q.always_compute_pass_alive,"exclude_territory_adjacent_to_atari":q.exclude_territory_adjacent_to_atari,
        "avoid_mytdagger_hack":q.avoid_mytdagger_hack,"skip_cache":q.skip_cache,
        "allow_terminal_search_history":q.allow_terminal_search_history,"force_non_terminal":q.force_non_terminal}}),
    )
}

fn source_sha(role: &str) -> Option<&'static str> {
    match role {
        "corpus_manifest" => Some(CORPUS_SHA),
        "calibration_requests" => Some(REQUESTS_SHA),
        "games_metadata" => Some(GAMES_SHA),
        "collector" => Some("840fb9fb15e60f400b080e453ba9e9912f87339944ae7e9b233328f0f02133c4"),
        "protocol_tools" => {
            Some("c01e02eb4a755074372ac7a8c0481502c5f5cc62a48f40ce91c0397a8d051b1a")
        }
        "validator" => Some("2dd88c396411e85d233c9723fafc9e8513beb376f42569bb5dbdd4fcd3de9f7f"),
        "proto" => Some("8a43d78331e64040f44e2c0fe49456d6d75c9ea88b11b111e03473fb48741e5e"),
        _ => None,
    }
}

fn validate_bundle(bundle: &Bundle) -> Result<()> {
    ensure!(
        bundle.schema == "rustgo-calibration-pb-input-v1"
            && bundle.split == "calibration"
            && bundle.source_total == TOTAL,
        "only the pinned complete calibration source is supported"
    );
    ensure!(
        (1..=TOTAL).contains(&bundle.selected_count)
            && bundle.requests.len() == bundle.selected_count,
        "invalid request count"
    );
    ensure!(
        matches!(
            (bundle.mode.as_str(), bundle.selected_count),
            ("complete-calibration", TOTAL)
        ) || (bundle.mode == "prefix-diagnostic" && bundle.selected_count < TOTAL),
        "full/prefix coverage identity mismatch"
    );
    ensure!(
        !bundle.packings.is_empty()
            && bundle.packings.len() <= 3
            && bundle.packings.iter().all(|b| [1, 3, 8].contains(b))
            && bundle.packings.windows(2).all(|p| p[0] < p[1]),
        "invalid frozen packing"
    );
    ensure!(
        valid_sha(&bundle.model_sha256)
            && valid_sha(&bundle.ordered_pb_sha256)
            && bundle.generation == 1
            && !bundle.session_id.is_empty()
            && bundle.session_id.len() <= 128
            && (1..=3600000).contains(&bundle.lease_ms),
        "invalid envelope binding"
    );
    ensure!(bundle.sources.len() == 8, "unexpected source inventory");
    let roles: BTreeSet<_> = bundle.sources.iter().map(|s| s.role.as_str()).collect();
    ensure!(
        roles.len() == 8 && roles.contains("preparer"),
        "duplicate/missing source role"
    );
    for s in &bundle.sources {
        ensure!(!s.original_path.is_empty(), "source origin missing");
        if s.role == "preparer" {
            ensure!(
                bundle.producer["script_sha256"].as_str() == Some(s.file.sha256.as_str()),
                "preparer source mismatch"
            );
        } else {
            ensure!(
                source_sha(&s.role) == Some(s.file.sha256.as_str()),
                "unrecognized/changed pinned source"
            );
        }
    }
    Ok(())
}

/// (first logical row, actual B); no padding, repetition, omission, or reordering.
fn batch_ranges(count: usize, target: usize) -> Result<Vec<(usize, usize)>> {
    ensure!(
        (1..=TOTAL).contains(&count) && [1, 3, 8].contains(&target),
        "invalid packing shape"
    );
    Ok((0..count)
        .step_by(target)
        .map(|i| (i, (count - i).min(target)))
        .collect())
}
fn tensor_label(batch: usize, spatial: &[u8], global: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"rustgo-encoded-group-cost-tensors-v1\0");
    digest.update((batch as u64).to_le_bytes());
    digest.update(spatial);
    digest.update(global);
    hex::encode(digest.finalize())
}
fn evidence(row: &EncodedWorkerRow) -> Result<Value> {
    let e = &row.evidence;
    let p = e.params;
    Ok(
        json!({"rules":serde_json::from_str::<Value>(&e.rules_json)?,"board_colors_yx":e.board_colors_yx,
        "next_player":e.next_player,"replayed_moves":e.replayed_moves,"is_game_finished":e.is_game_finished,
        "is_no_result":e.is_no_result,"encore_phase":e.encore_phase,"skip_cache":e.skip_cache,"include_ownership":e.include_ownership,
        "actual_misc":{"symmetry":p.symmetry,"policy_temperature_f32_bits":format!("{:08x}",p.nn_policy_temperature.to_bits()),
        "policy_temperature_f32":p.nn_policy_temperature,"policy_optimism":p.policy_optimism,
        "draw_equivalent_wins_for_white":p.draw_equivalent_wins_for_white,"playout_doubling_advantage":p.playout_doubling_advantage,
        "max_history":p.max_history,"conservative_pass_and_is_root":p.conservative_pass_and_is_root,
        "enable_passing_hacks":p.enable_passing_hacks,"avoid_mytdagger_hack":p.avoid_mytdagger_hack}}),
    )
}

fn compiled_sources() -> Value {
    // Includes exactly the bytes rustc consumed; no filesystem-time claim.
    json!({"export_calibration_inputs.rs":hash(include_bytes!("export_calibration_inputs.rs")),
        "evaluator.rs":hash(include_bytes!("../src/evaluator.rs")),
        "input_encoding.rs":hash(include_bytes!("../src/input_encoding.rs")),
        "worker.proto":hash(include_bytes!("../proto/worker.proto")),
        "inputs.rs":hash(include_bytes!("../../kata_nn/src/inputs.rs")),
        "eval.rs":hash(include_bytes!("../../kata_nn/src/eval.rs")),
        "version.rs":hash(include_bytes!("../../kata_nn/src/version.rs")),
        "cuda.rs":hash(include_bytes!("../../kata_nn/src/backends/cuda.rs")),
        "ffn_group_cost.rs":hash(include_bytes!("../../kata_nn/examples/ffn_group_cost.rs")),
        "board.rs":hash(include_bytes!("../../kata_game/src/board.rs")),
        "history.rs":hash(include_bytes!("../../kata_game/src/history.rs")),
        "rules.rs":hash(include_bytes!("../../kata_game/src/rules.rs")),
        "symmetry.rs":hash(include_bytes!("../../kata_game/src/symmetry.rs")),
        "Cargo.lock":hash(include_bytes!("../../../Cargo.lock"))})
}

fn export(request_path: &Path, expected_sha: &str, output: &Path) -> Result<()> {
    ensure!(
        valid_sha(expected_sha) && fs::metadata(request_path)?.len() <= 32 * 1024 * 1024,
        "invalid/oversized PB manifest"
    );
    let raw = fs::read(request_path)?;
    ensure!(hash(&raw) == expected_sha, "prepared manifest SHA mismatch");
    let bundle: Bundle = serde_json::from_slice(&raw)?;
    validate_bundle(&bundle)?;
    let root = request_path.parent().context("manifest parent")?;
    let mut request_lines = Vec::new();
    for source in &bundle.sources {
        let bytes = read_bound(root, &source.file, 32 * 1024 * 1024)?;
        if source.role == "calibration_requests" {
            request_lines = bytes
                .split(|&b| b == b'\n')
                .map(|line| line.strip_suffix(b"\r").unwrap_or(line).to_vec())
                .collect();
            if request_lines.last().is_some_and(Vec::is_empty) {
                request_lines.pop();
            }
        }
    }
    ensure!(
        request_lines.len() == TOTAL && request_lines.iter().all(|x| !x.is_empty()),
        "invalid original JSONL line inventory"
    );
    let mut encoded = Vec::with_capacity(bundle.selected_count);
    let mut original_pbs = Vec::with_capacity(bundle.selected_count);
    let mut names = BTreeSet::new();
    let mut ordered = Sha256::new();
    for (index, item) in bundle.requests.iter().enumerate() {
        ensure!(
            item.source_line == index + 1 && item.task_id == (index + 1) as u64,
            "request rows must be the exact ordered prefix"
        );
        let line = &request_lines[index];
        ensure!(
            hash(line) == item.source_line_sha256
                && same_semantics(&serde_json::from_slice::<Value>(line)?, &item.record),
            "request record differs from frozen source line"
        );
        ensure!(
            item.record["split"] == "calibration",
            "wrong request partition"
        );
        let name = item.record["name"]
            .as_str()
            .context("missing request name")?;
        ensure!(names.insert(name), "duplicate source request");
        let pb = read_bound(root, &item.pb, 512 * 1024)?;
        let request = wire::EvalRequest::decode(pb.as_slice())?;
        ensure!(
            request.encode_to_vec() == pb,
            "noncanonical/cross-ABI PB encoding"
        );
        ensure!(
            request.task_id == item.task_id
                && request.generation == bundle.generation
                && request.session_id == bundle.session_id
                && request.lease_ms == bundle.lease_ms
                && request.model_sha256 == bundle.model_sha256
                && request.input_hash.len() == 32
                && hex::encode(&request.input_hash) == item.input_hash_hex,
            "PB envelope/index mismatch"
        );
        let mut unsigned = request.clone();
        unsigned.input_hash.clear();
        ensure!(
            hash(&unsigned.encode_to_vec()) == item.input_hash_hex,
            "PB input_hash does not bind envelope"
        );
        let semantics = request_semantics(&request)?;
        ensure!(
            same_semantics(&semantics["position"], &item.record["position"])
                && same_semantics(&semantics["parameters"], &item.record["parameters"]),
            "PB semantics differ from original complete request"
        );
        let p = request.parameters.as_ref().unwrap();
        ensure!(
            p.skip_cache && p.include_ownership,
            "calibration requires skipped cache and ownership"
        );
        let mut semantic = Sha256::new();
        for data in [
            request.position.as_ref().unwrap().encode_to_vec(),
            p.encode_to_vec(),
        ] {
            semantic.update((data.len() as u64).to_le_bytes());
            semantic.update(data);
        }
        ensure!(
            hex::encode(semantic.finalize()) == item.wire_semantic_sha256,
            "wire semantic hash mismatch"
        );
        let row = encode_request_v7_cpu(&request, &bundle.model_sha256)
            .map_err(|e| anyhow::anyhow!("{}: {}", e.code, e.message))?;
        ordered.update((pb.len() as u64).to_le_bytes());
        ordered.update(&pb);
        original_pbs.push(pb);
        encoded.push(row);
    }
    ensure!(
        hex::encode(ordered.finalize()) == bundle.ordered_pb_sha256,
        "ordered PB inventory mismatch"
    );
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let parent = output
        .parent()
        .context("output needs existing parent")?
        .canonicalize()?;
    ensure!(
        parent.starts_with(workspace.join("target").canonicalize()?),
        "export destination must be inside workspace target"
    );
    fs::create_dir(output).context("refuse existing output")?;
    fs::create_dir(output.join("requests"))?;
    fs::create_dir(output.join("sources"))?;
    // Preserve the exact incoming binding and its source files inside the export.
    let copied_bundle = write_new(output, "prepared-requests.json", &raw)?;
    let mut copied_sources = Vec::new();
    for source in &bundle.sources {
        let data = read_bound(root, &source.file, 32 * 1024 * 1024)?;
        let name = source.file.path.clone();
        ensure!(
            relative_path(&name)?.starts_with("sources"),
            "source copy must stay in sources/"
        );
        fs::create_dir_all(output.join(&name).parent().context("source copy parent")?)?;
        copied_sources.push(json!({"role":source.role,"original_path":source.original_path,"file":write_new(output,&name,&data)?}));
    }
    let mut request_proof = Vec::new();
    for (index, ((item, row), pb)) in bundle
        .requests
        .iter()
        .zip(&encoded)
        .zip(&original_pbs)
        .enumerate()
    {
        let name = format!("requests/{:04}.pb", index + 1);
        request_proof.push(json!({"source_line":item.source_line,"source_line_sha256":item.source_line_sha256,
            "record":item.record,"task_id":item.task_id,"pb":write_new(output,&name,pb)?,
            "input_hash_hex":item.input_hash_hex,"wire_semantic_sha256":item.wire_semantic_sha256,
            "prepared":evidence(row)?,"pre_spatial_sha256":hash(&f32le(&row.pre_spatial)),
            "post_spatial_sha256":hash(&f32le(&row.spatial)),"global_sha256":hash(&f32le(&row.global)),
            "pre_feature_sha256":row_feature_sha256(&row.pre_spatial,&row.global).map_err(|e| anyhow::anyhow!(e.message))?,
            "post_feature_sha256":row_feature_sha256(&row.spatial,&row.global).map_err(|e| anyhow::anyhow!(e.message))?}));
    }
    let request_proof_binding = write_json(output, "request-provenance.json", &request_proof)?;
    let mut maps = Vec::new();
    let mut shards = Vec::new();
    for &target in &bundle.packings {
        let ranges = batch_ranges(encoded.len(), target)?;
        for (shard_index, chunk) in ranges.chunks(MAX_SHARD).enumerate() {
            let relative = format!("b{target}-shard-{shard_index:03}");
            let dir = output.join(&relative);
            fs::create_dir(&dir)?;
            let mut inputs = Vec::new();
            for &(first, batch) in chunk {
                let id = format!("calibration-b{target}-row-{first:04}");
                let mut spatial = Vec::with_capacity(batch * SPATIAL_FLOATS * 4);
                let mut global = Vec::with_capacity(batch * GLOBAL_FLOATS * 4);
                for row in &encoded[first..first + batch] {
                    spatial.extend(f32le(&row.spatial));
                    global.extend(f32le(&row.global));
                }
                let spatial_binding = write_new(&dir, &format!("{id}.spatial.f32le"), &spatial)?;
                let global_binding = write_new(&dir, &format!("{id}.global.f32le"), &global)?;
                // Read exact bytes back; row proof below binds slices of these actual files.
                ensure!(
                    read_bound(&dir, &spatial_binding, 8 * SPATIAL_FLOATS * 4)? == spatial
                        && read_bound(&dir, &global_binding, 8 * GLOBAL_FLOATS * 4)? == global,
                    "written tensor differs"
                );
                let label = tensor_label(batch, &spatial, &global);
                for row_index in 0..batch {
                    let source_index = first + row_index;
                    let s_off = row_index * SPATIAL_FLOATS * 4;
                    let g_off = row_index * GLOBAL_FLOATS * 4;
                    ensure!(
                        hash(&spatial[s_off..s_off + SPATIAL_FLOATS * 4])
                            == request_proof[source_index]["post_spatial_sha256"]
                                .as_str()
                                .unwrap()
                            && hash(&global[g_off..g_off + GLOBAL_FLOATS * 4])
                                == request_proof[source_index]["global_sha256"]
                                    .as_str()
                                    .unwrap(),
                        "packed row SHA mismatch"
                    );
                    maps.push(json!({"target_batch":target,"physical_batch":batch,"row":row_index,
                        "logical_row":source_index,"source_line":source_index+1,"task_id":bundle.requests[source_index].task_id,
                        "manifest":format!("{relative}/inputs.json"),"input_id":id,"input_tensor_sha256":label,
                        "spatial_file":format!("{relative}/{}",spatial_binding.path),"global_file":format!("{relative}/{}",global_binding.path),
                        "spatial_sha256":spatial_binding.sha256,"global_sha256":global_binding.sha256,
                        "spatial_offset_bytes":s_off,"spatial_bytes":SPATIAL_FLOATS*4,
                        "global_offset_bytes":g_off,"global_bytes":GLOBAL_FLOATS*4,
                        "row_feature_sha256":request_proof[source_index]["post_feature_sha256"],"padding":false}));
                }
                inputs.push(Input {
                    id,
                    physical_batch: batch,
                    spatial: spatial_binding,
                    global: global_binding,
                });
            }
            let manifest = Manifest {
                schema: "rustgo-encoded-group-cost-input-v1".into(),
                purpose: "calibration".into(),
                provenance: format!(
                    "CPU_WORKER_V7_INPUTS_ONLY; mode={}; prepared_sha256={expected_sha}; see external provenance.json and rows.jsonl; no model/accuracy/performance claim",
                    bundle.mode
                ),
                inputs,
            };
            let mut binding = write_json(&dir, "inputs.json", &manifest)?;
            binding.path = format!("{relative}/inputs.json");
            shards.push(binding);
        }
    }
    ensure!(
        maps.len() == bundle.selected_count * bundle.packings.len(),
        "row map coverage incomplete"
    );
    let mut row_bytes = Vec::new();
    for map in &maps {
        serde_json::to_writer(&mut row_bytes, map)?;
        row_bytes.push(b'\n');
    }
    let row_binding = write_new(output, "rows.jsonl", &row_bytes)?;
    // No completion artifact after any changed source or partial export.
    ensure!(
        fs::read(request_path)? == raw,
        "PB manifest changed during export"
    );
    for source in &bundle.sources {
        read_bound(root, &source.file, 32 * 1024 * 1024)?;
    }
    for item in &bundle.requests {
        read_bound(root, &item.pb, 512 * 1024)?;
    }
    let executable = std::env::current_exe()?;
    let provenance = json!({"schema":"rustgo-worker-calibration-input-provenance-v1",
        "status":"CPU_INPUT_ENCODING_ONLY_NOT_MODEL_VALIDATION", "source_split":"calibration",
        "coverage_mode":bundle.mode,"source_count":TOTAL,"encoded_count":bundle.selected_count,
        "packing":"source-order-exact-tail-no-padding", "packings":bundle.packings,
        "feature_contract":{"inputs_version":7,"spatial":[22,19,19],"global":[19],"layout":"NCHW",
        "dtype":"f32le","spatial_symmetry_applied":true,"global_transform":"identity",
        "model_contract_validation":"NOT_LOADED; caller must verify v7/22/19/no-SGF-metadata before inference"},
        "model_sha256":bundle.model_sha256,"prepared_manifest":copied_bundle,"prepared_manifest_sha256":expected_sha,
        "source_copies":copied_sources,"request_provenance":request_proof_binding,"rows":row_binding,"input_manifests":shards,
        "encoder_executable_sha256":hash(&fs::read(&executable)?),"compiled_sources":compiled_sources(),
        "model_loaded":false,"gpu_called":false,"outputs_read":false,
        "row_feature_domain":"rustgo-worker-v7-nchw19-row-f32le-v1\\0 + spatial-row-f32LE + global-row-f32LE",
        "group_cost_domain":"rustgo-encoded-group-cost-tensors-v1\\0 + u64LE(B) + all-spatial-f32LE + all-global-f32LE"});
    commit_json(output, "provenance.json", &provenance)?;
    // A closed diagnostic stdout cannot invalidate an already committed export.
    let _ = std::io::stdout()
        .lock()
        .write_all(b"CPU_INPUTS_EXPORTED; see provenance.json\n");
    Ok(())
}

fn main() -> Result<()> {
    // Match the CLI's stack allocation: Windows main has only 1 MiB, which
    // overflows during the shared board/Zobrist initialization in debug builds.
    std::thread::Builder::new()
        .name("calibration-input-encoder".into())
        .stack_size(256 * 1024 * 1024)
        .spawn(run_cli)?
        .join()
        .map_err(|_| anyhow::anyhow!("calibration input encoder thread panicked"))?
}

fn run_cli() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 6,
        "usage: --requests PB_MANIFEST --expected-sha256 SHA --output FRESH_TARGET_DIRECTORY"
    );
    let mut values = std::collections::BTreeMap::new();
    for pair in args.chunks_exact(2) {
        ensure!(
            ["--requests", "--expected-sha256", "--output"].contains(&pair[0].as_str())
                && values.insert(pair[0].as_str(), pair[1].as_str()).is_none(),
            "unknown/duplicate argument"
        );
    }
    let requests = PathBuf::from(values["--requests"]);
    let output = PathBuf::from(values["--output"]);
    export(&requests, values["--expected-sha256"], &output)
}

#[cfg(test)]
#[path = "export_calibration_inputs/tests.rs"]
mod tests;

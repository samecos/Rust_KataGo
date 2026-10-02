//! Read-only admission of the existing Worker calibration encoding format.
//! No model/GPU is constructed. Metadata may reside in memory; tensor payloads
//! are validated one physical input at a time and are never retained by Corpus.
use anyhow::{Context, Result, ensure};
use kata_worker::{
    evaluator::input_encoding::{self, EncodedWorkerRow, GLOBAL_FLOATS, SPATIAL_FLOATS},
    wire,
};
use prost::Message;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
};

const TOTAL: usize = 2044;
const MAX_SHARD: usize = 128;
const META_CAP: usize = 64 * 1024 * 1024;
const SPATIAL_BYTES: usize = SPATIAL_FLOATS * 4;
const GLOBAL_BYTES: usize = GLOBAL_FLOATS * 4;
const ROW_DOMAIN: &[u8] = b"rustgo-worker-v7-nchw19-row-f32le-v1\0";
const TENSOR_DOMAIN: &[u8] = b"rustgo-encoded-group-cost-tensors-v1\0";

#[derive(Clone, Debug, Serialize)]
pub struct RowDescriptor {
    pub logical_row: usize,
    pub source_line: usize,
    pub task_id: u64,
    pub name: String,
    pub game_id: String,
    pub phase: String,
    pub ply: usize,
    pub pb_sha256: String,
    pub row_feature_sha256: String,
    pub wire_semantic_sha256: String,
    pub input_hash_hex: String,
    pub source_line_sha256: String,
    pub semantic_position_sha256: String,
    pub board_state_sha256: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct InputDescriptor {
    pub id: String,
    pub manifest_path: PathBuf,
    pub manifest_sha256: String,
    pub target_batch: usize,
    pub physical_batch: usize,
    pub tensor_sha256: String,
    pub rows: Vec<RowDescriptor>,
}
#[derive(Debug)]
pub struct LoadedRow {
    pub descriptor: RowDescriptor,
    pub request_pb: Vec<u8>,
}
#[derive(Debug)]
pub struct LoadedInput {
    pub descriptor: InputDescriptor,
    pub spatial: Vec<f32>,
    pub global: Vec<f32>,
    pub tensor_sha256: String,
    pub rows: Vec<LoadedRow>,
}
/// Absolute file identity, useful for the collector's outer evidence inventory.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Source {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Binding {
    path: String,
    bytes: usize,
    sha256: String,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Origin {
    role: String,
    original_path: String,
    file: Binding,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    schema: String,
    status: String,
    source_split: String,
    coverage_mode: String,
    source_count: usize,
    encoded_count: usize,
    packing: String,
    packings: Vec<usize>,
    feature_contract: Value,
    model_sha256: String,
    prepared_manifest: Binding,
    prepared_manifest_sha256: String,
    source_copies: Vec<Origin>,
    request_provenance: Binding,
    rows: Binding,
    input_manifests: Vec<Binding>,
    encoder_executable_sha256: String,
    compiled_sources: BTreeMap<String, String>,
    model_loaded: bool,
    gpu_called: bool,
    outputs_read: bool,
    row_feature_domain: String,
    group_cost_domain: String,
}
#[derive(Deserialize)]
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
#[derive(Deserialize)]
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
    sources: Vec<Origin>,
    requests: Vec<RequestEntry>,
    producer: Value,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Proof {
    source_line: usize,
    source_line_sha256: String,
    record: Value,
    task_id: u64,
    pb: Binding,
    input_hash_hex: String,
    wire_semantic_sha256: String,
    prepared: Value,
    pre_spatial_sha256: String,
    post_spatial_sha256: String,
    global_sha256: String,
    pre_feature_sha256: String,
    post_feature_sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PackedInput {
    id: String,
    physical_batch: usize,
    spatial: Binding,
    global: Binding,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Shard {
    schema: String,
    purpose: String,
    provenance: String,
    inputs: Vec<PackedInput>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RowMap {
    target_batch: usize,
    physical_batch: usize,
    row: usize,
    logical_row: usize,
    source_line: usize,
    task_id: u64,
    manifest: String,
    input_id: String,
    input_tensor_sha256: String,
    spatial_file: String,
    global_file: String,
    spatial_sha256: String,
    global_sha256: String,
    spatial_offset_bytes: usize,
    spatial_bytes: usize,
    global_offset_bytes: usize,
    global_bytes: usize,
    row_feature_sha256: String,
    padding: bool,
}
struct RequestData {
    descriptor: RowDescriptor,
    pb: Source,
    spatial_sha256: String,
    global_sha256: String,
}
struct InputFiles {
    spatial: Source,
    global: Source,
    first: usize,
}
pub struct Corpus {
    root: PathBuf,
    provenance_sha256: String,
    model_sha256: String,
    coverage_mode: String,
    packings: Vec<usize>,
    requests: Vec<RequestData>,
    inputs: Vec<InputDescriptor>,
    files: Vec<InputFiles>,
    sources: Vec<Source>,
}

fn hash(raw: &[u8]) -> String {
    hex::encode(Sha256::digest(raw))
}
fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
fn local_path(name: &str) -> Result<&Path> {
    let path = Path::new(name);
    ensure!(
        !name.is_empty()
            && name.chars().all(|c| !matches!(c, ':' | '\\' | '\0'))
            && name
                .split('/')
                .all(|part| !part.is_empty() && part != "." && part != "..")
            && path.components().all(|c| matches!(c, Component::Normal(_))),
        "nonlocal artifact path: {name}"
    );
    Ok(path)
}
fn bounded_bytes(
    path: &Path,
    expected_bytes: u64,
    expected_sha: &str,
    cap: usize,
) -> Result<Vec<u8>> {
    ensure!(
        valid_sha(expected_sha) && expected_bytes <= cap as u64,
        "invalid/oversized file binding"
    );
    ensure!(
        fs::metadata(path)?.is_file() && fs::metadata(path)?.len() == expected_bytes,
        "bound size changed: {}",
        path.display()
    );
    let mut bytes = Vec::with_capacity(expected_bytes as usize);
    fs::File::open(path)?
        .take(expected_bytes + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 == expected_bytes && hash(&bytes) == expected_sha,
        "bound SHA changed: {}",
        path.display()
    );
    Ok(bytes)
}
fn verify_source(source: &Source) -> Result<()> {
    ensure!(
        source.path.is_absolute() && valid_sha(&source.sha256),
        "invalid source identity"
    );
    ensure!(
        fs::metadata(&source.path)?.is_file() && fs::metadata(&source.path)?.len() == source.bytes,
        "source size changed"
    );
    let mut file = fs::File::open(&source.path)?.take(source.bytes + 1);
    let mut digest = Sha256::new();
    let mut count = 0u64;
    let mut buffer = [0u8; 65536];
    loop {
        let size = file.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        count += size as u64;
        digest.update(&buffer[..size]);
    }
    ensure!(
        count == source.bytes && hex::encode(digest.finalize()) == source.sha256,
        "source SHA changed: {}",
        source.path.display()
    );
    Ok(())
}
fn register(map: &mut BTreeMap<PathBuf, Source>, source: Source) -> Result<Source> {
    if let Some(prior) = map.get(&source.path) {
        ensure!(prior == &source, "conflicting bindings for same file");
    } else {
        map.insert(source.path.clone(), source.clone());
    }
    Ok(source)
}
fn read_bound(
    root: &Path,
    binding: &Binding,
    cap: usize,
    inventory: &mut BTreeMap<PathBuf, Source>,
) -> Result<(Source, Vec<u8>)> {
    let path = root.join(local_path(&binding.path)?).canonicalize()?;
    ensure!(
        path.starts_with(root),
        "artifact escapes encoded corpus root"
    );
    let bytes = bounded_bytes(&path, binding.bytes as u64, &binding.sha256, cap)?;
    let source = register(
        inventory,
        Source {
            path,
            bytes: binding.bytes as u64,
            sha256: binding.sha256.clone(),
        },
    )?;
    Ok((source, bytes))
}
fn json<T: serde::de::DeserializeOwned>(raw: &[u8]) -> Result<T> {
    Ok(serde_json::from_slice(raw)?)
}
fn lines(raw: &[u8]) -> Result<Vec<&[u8]>> {
    ensure!(
        !raw.is_empty() && raw.ends_with(b"\n"),
        "JSONL requires final LF"
    );
    let lines: Vec<_> = raw[..raw.len() - 1]
        .split(|b| *b == b'\n')
        .map(|line| line.strip_suffix(b"\r").unwrap_or(line))
        .collect();
    ensure!(
        lines.iter().all(|line| !line.is_empty()),
        "empty JSONL line"
    );
    Ok(lines)
}
// Same restricted numeric equivalence as the frozen encoder, used only for
// semantic JSON (all request integers <=10000), never PB u64 envelope identity.
fn same_semantics(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => a.as_f64() == b.as_f64(),
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_semantics(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, a)| b.get(key).is_some_and(|b| same_semantics(a, b)))
        }
        _ => a == b,
    }
}
fn request_semantics(request: &wire::EvalRequest) -> Result<Value> {
    let p = request.position.as_ref().context("missing PB position")?;
    let q = request
        .parameters
        .as_ref()
        .context("missing PB parameters")?;
    Ok(
        json!({"position":{"board_size":p.board_size,"komi":p.komi,"rules":p.rules,
        "initial_player":p.initial_player,"next_player":p.next_player,
        "initial_stones":p.initial_stones.iter().map(|v|json!({"color":v.color,"vertex":v.vertex})).collect::<Vec<_>>(),
        "moves":p.moves.iter().map(|v|json!({"color":v.color,"vertex":v.vertex})).collect::<Vec<_>>()},
        "parameters":{"symmetry":q.symmetry,"policy_temperature":q.policy_temperature,"policy_optimism":q.policy_optimism,
        "draw_equivalent_wins_for_white":q.draw_equivalent_wins_for_white,"playout_doubling_advantage":q.playout_doubling_advantage,
        "include_ownership":q.include_ownership,"max_history":q.max_history,"conservative_pass":q.conservative_pass,
        "enable_passing_hacks":q.enable_passing_hacks,"always_compute_pass_alive":q.always_compute_pass_alive,
        "exclude_territory_adjacent_to_atari":q.exclude_territory_adjacent_to_atari,"avoid_mytdagger_hack":q.avoid_mytdagger_hack,
        "skip_cache":q.skip_cache,"allow_terminal_search_history":q.allow_terminal_search_history,"force_non_terminal":q.force_non_terminal}}),
    )
}
fn evidence(row: &EncodedWorkerRow) -> Result<Value> {
    let e = &row.evidence;
    let p = &e.params;
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
fn source_sha(role: &str) -> Option<&'static str> {
    Some(match role {
        "corpus_manifest" => "c7b80205d26d5dd7c7cf38dceb8b88bcc8b3b452b764f7a83113d156fc88c842",
        "calibration_requests" => {
            "c3d3ee4b4f75fadc613e343beaf60c60d95879bb89e6d516cfa5b4278650238f"
        }
        "games_metadata" => "c0c9de9d8ed7dc8e99f9dda57ebb974e99c49c7bb99601f93235be671acc0768",
        "collector" => "840fb9fb15e60f400b080e453ba9e9912f87339944ae7e9b233328f0f02133c4",
        "protocol_tools" => "c01e02eb4a755074372ac7a8c0481502c5f5cc62a48f40ce91c0397a8d051b1a",
        "validator" => "2dd88c396411e85d233c9723fafc9e8513beb376f42569bb5dbdd4fcd3de9f7f",
        "proto" => "8a43d78331e64040f44e2c0fe49456d6d75c9ea88b11b111e03473fb48741e5e",
        _ => return None,
    })
}
fn validate_mode(mode: &str, count: usize) -> Result<()> {
    ensure!(
        (1..=TOTAL).contains(&count)
            && ((mode == "complete-calibration" && count == TOTAL)
                || (mode == "prefix-diagnostic" && count < TOTAL)),
        "complete/prefix coverage mismatch"
    );
    Ok(())
}
fn ranges(count: usize, target: usize) -> Result<Vec<(usize, usize)>> {
    ensure!(
        (1..=TOTAL).contains(&count) && [1, 3, 8].contains(&target),
        "invalid packing"
    );
    Ok((0..count)
        .step_by(target)
        .map(|first| (first, (count - first).min(target)))
        .collect())
}
fn tensor_sha(batch: usize, spatial: &[u8], global: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(TENSOR_DOMAIN);
    h.update((batch as u64).to_le_bytes());
    h.update(spatial);
    h.update(global);
    hex::encode(h.finalize())
}
fn feature_sha(spatial: &[u8], global: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(ROW_DOMAIN);
    h.update(spatial);
    h.update(global);
    hex::encode(h.finalize())
}
fn floats(raw: &[u8], count: usize) -> Result<Vec<f32>> {
    ensure!(
        raw.len() == count.checked_mul(4).context("tensor byte size overflow")?,
        "tensor shape mismatch"
    );
    let values: Vec<_> = raw
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    ensure!(values.iter().all(|x| x.is_finite()), "nonfinite tensor");
    Ok(values)
}

fn validate_provenance(p: &Provenance, model: &str) -> Result<()> {
    ensure!(
        valid_sha(model) && p.model_sha256 == model,
        "provenance model mismatch"
    );
    ensure!(
        p.schema == "rustgo-worker-calibration-input-provenance-v1"
            && p.status == "CPU_INPUT_ENCODING_ONLY_NOT_MODEL_VALIDATION"
            && p.source_split == "calibration"
            && p.source_count == TOTAL
            && p.packing == "source-order-exact-tail-no-padding",
        "wrong calibration provenance scope"
    );
    validate_mode(&p.coverage_mode, p.encoded_count)?;
    ensure!(
        !p.packings.is_empty()
            && p.packings.len() <= 3
            && p.packings.iter().all(|b| [1, 3, 8].contains(b))
            && p.packings.windows(2).all(|v| v[0] < v[1]),
        "invalid/duplicate packings"
    );
    ensure!(
        !p.model_loaded && !p.gpu_called && !p.outputs_read,
        "unexpected encoder execution claim"
    );
    ensure!(
        p.feature_contract
            == json!({"inputs_version":7,"spatial":[22,19,19],"global":[19],"layout":"NCHW",
        "dtype":"f32le","spatial_symmetry_applied":true,"global_transform":"identity",
        "model_contract_validation":"NOT_LOADED; caller must verify v7/22/19/no-SGF-metadata before inference"}),
        "feature contract mismatch"
    );
    ensure!(
        p.row_feature_domain
            == "rustgo-worker-v7-nchw19-row-f32le-v1\\0 + spatial-row-f32LE + global-row-f32LE"
            && p.group_cost_domain
                == "rustgo-encoded-group-cost-tensors-v1\\0 + u64LE(B) + all-spatial-f32LE + all-global-f32LE",
        "wrong hash domain descriptions"
    );
    ensure!(
        p.prepared_manifest.path == "prepared-requests.json"
            && p.request_provenance.path == "request-provenance.json"
            && p.rows.path == "rows.jsonl"
            && p.prepared_manifest.sha256 == p.prepared_manifest_sha256,
        "metadata path/SHA mismatch"
    );
    ensure!(
        valid_sha(&p.encoder_executable_sha256)
            && !p.compiled_sources.is_empty()
            && p.compiled_sources.values().all(|x| valid_sha(x)),
        "invalid encoder provenance"
    );
    Ok(())
}
fn validate_bundle(b: &Bundle, p: &Provenance) -> Result<()> {
    ensure!(
        b.schema == "rustgo-calibration-pb-input-v1"
            && b.split == "calibration"
            && b.source_total == TOTAL,
        "wrong prepared scope"
    );
    ensure!(
        b.mode == p.coverage_mode
            && b.selected_count == p.encoded_count
            && b.requests.len() == b.selected_count
            && b.packings == p.packings
            && b.model_sha256 == p.model_sha256,
        "prepared/provenance coverage mismatch"
    );
    ensure!(
        b.generation == 1
            && !b.session_id.is_empty()
            && b.session_id.len() <= 128
            && (1..=3600000).contains(&b.lease_ms)
            && valid_sha(&b.ordered_pb_sha256),
        "invalid PB envelope inventory"
    );
    ensure!(
        b.sources == p.source_copies && b.sources.len() == 8,
        "source copy inventory differs"
    );
    let mut roles = BTreeSet::new();
    for s in &b.sources {
        ensure!(
            !s.original_path.is_empty()
                && roles.insert(s.role.as_str())
                && s.file.path.starts_with("sources/"),
            "duplicate/invalid source role"
        );
        if s.role == "preparer" {
            ensure!(
                b.producer["script_sha256"].as_str() == Some(s.file.sha256.as_str())
                    && valid_sha(&s.file.sha256),
                "preparer source mismatch"
            );
        } else {
            ensure!(
                source_sha(&s.role) == Some(s.file.sha256.as_str()),
                "changed/unknown pinned calibration source"
            );
        }
    }
    ensure!(roles.contains("preparer"), "missing preparer");
    Ok(())
}
fn text_field(record: &Value, key: &str) -> Result<String> {
    let value = record[key]
        .as_str()
        .with_context(|| format!("missing record {key}"))?;
    ensure!(!value.is_empty(), "empty record {key}");
    Ok(value.to_owned())
}
fn check_pb(raw: &[u8], entry: &RequestEntry, bundle: &Bundle) -> Result<wire::EvalRequest> {
    let request = wire::EvalRequest::decode(raw)?;
    ensure!(request.encode_to_vec() == raw, "noncanonical PB bytes");
    ensure!(
        request.task_id == entry.task_id
            && request.generation == bundle.generation
            && request.session_id == bundle.session_id
            && request.lease_ms == bundle.lease_ms
            && request.model_sha256 == bundle.model_sha256
            && request.input_hash.len() == 32
            && hex::encode(&request.input_hash) == entry.input_hash_hex,
        "PB envelope mismatch"
    );
    let mut unsigned = request.clone();
    unsigned.input_hash.clear();
    ensure!(
        hash(&unsigned.encode_to_vec()) == entry.input_hash_hex,
        "PB input_hash differs"
    );
    let semantic = request_semantics(&request)?;
    ensure!(
        same_semantics(&semantic["position"], &entry.record["position"])
            && same_semantics(&semantic["parameters"], &entry.record["parameters"]),
        "PB semantics differ from frozen JSON"
    );
    let p = request.parameters.as_ref().unwrap();
    ensure!(
        p.skip_cache && p.include_ownership,
        "calibration must skip cache and include ownership"
    );
    let mut digest = Sha256::new();
    for bytes in [
        request.position.as_ref().unwrap().encode_to_vec(),
        p.encode_to_vec(),
    ] {
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    ensure!(
        hex::encode(digest.finalize()) == entry.wire_semantic_sha256,
        "wire semantic SHA mismatch"
    );
    Ok(request)
}
fn check_encoding(encoded: &EncodedWorkerRow, proof: &Proof) -> Result<()> {
    ensure!(
        hash(&input_encoding::f32le(&encoded.pre_spatial)) == proof.pre_spatial_sha256
            && hash(&input_encoding::f32le(&encoded.spatial)) == proof.post_spatial_sha256
            && hash(&input_encoding::f32le(&encoded.global)) == proof.global_sha256,
        "actual Worker tensor hash mismatch"
    );
    ensure!(
        input_encoding::row_feature_sha256(&encoded.pre_spatial, &encoded.global)
            .map_err(|e| anyhow::anyhow!(e.message))?
            == proof.pre_feature_sha256
            && input_encoding::row_feature_sha256(&encoded.spatial, &encoded.global)
                .map_err(|e| anyhow::anyhow!(e.message))?
                == proof.post_feature_sha256,
        "actual Worker feature SHA mismatch"
    );
    ensure!(
        same_semantics(&evidence(encoded)?, &proof.prepared),
        "actual strict Worker history/params differ from evidence"
    );
    Ok(())
}

impl Corpus {
    pub fn load(
        provenance_path: &Path,
        expected_sha: &str,
        expected_model_sha: &str,
    ) -> Result<Self> {
        ensure!(valid_sha(expected_sha), "invalid expected provenance SHA");
        let path = provenance_path.canonicalize()?;
        let root = path
            .parent()
            .context("provenance has no parent")?
            .to_path_buf();
        let bytes = bounded_bytes(&path, fs::metadata(&path)?.len(), expected_sha, META_CAP)?;
        let p: Provenance = json(&bytes)?;
        validate_provenance(&p, expected_model_sha)?;
        let mut inventory = BTreeMap::new();
        register(
            &mut inventory,
            Source {
                path,
                bytes: bytes.len() as u64,
                sha256: expected_sha.into(),
            },
        )?;
        let (_, bytes) = read_bound(&root, &p.prepared_manifest, META_CAP, &mut inventory)?;
        let bundle: Bundle = json(&bytes)?;
        validate_bundle(&bundle, &p)?;
        let mut original_lines = Vec::new();
        for source in &bundle.sources {
            let (_, bytes) = read_bound(&root, &source.file, META_CAP, &mut inventory)?;
            if source.role == "calibration_requests" {
                let input_lines = lines(&bytes)?;
                ensure!(
                    input_lines.len() == TOTAL,
                    "original calibration must contain 2044 lines"
                );
                original_lines = input_lines
                    .into_iter()
                    .take(bundle.selected_count)
                    .map(<[u8]>::to_vec)
                    .collect();
            }
        }
        ensure!(
            original_lines.len() == bundle.selected_count,
            "missing original calibration source"
        );
        let (_, bytes) = read_bound(&root, &p.request_provenance, META_CAP, &mut inventory)?;
        let proofs: Vec<Proof> = json(&bytes)?;
        ensure!(
            proofs.len() == bundle.selected_count,
            "request provenance row count mismatch"
        );
        let mut requests = Vec::new();
        let mut ordered = Sha256::new();
        let mut names = BTreeSet::new();
        let mut semantics = BTreeSet::new();
        let mut boards = BTreeSet::new();
        let mut pb_shas = BTreeSet::new();
        for (index, ((entry, proof), line)) in bundle
            .requests
            .iter()
            .zip(&proofs)
            .zip(&original_lines)
            .enumerate()
        {
            ensure!(
                entry.source_line == index + 1
                    && entry.task_id == (index + 1) as u64
                    && proof.source_line == entry.source_line
                    && proof.task_id == entry.task_id,
                "request source/order/task mismatch"
            );
            let source_record: Value = json(line)?;
            ensure!(
                hash(line) == entry.source_line_sha256
                    && proof.source_line_sha256 == entry.source_line_sha256
                    && same_semantics(&source_record, &entry.record)
                    && same_semantics(&proof.record, &entry.record)
                    && entry.record["split"] == "calibration",
                "original JSON/PB proof record mismatch"
            );
            ensure!(
                entry.pb == proof.pb
                    && entry.pb.path == format!("requests/{:04}.pb", index + 1)
                    && entry.input_hash_hex == proof.input_hash_hex
                    && entry.wire_semantic_sha256 == proof.wire_semantic_sha256,
                "prepared/request proof PB binding mismatch"
            );
            let name = text_field(&entry.record, "name")?;
            let semantic = text_field(&entry.record, "semantic_position_sha256")?;
            let board = text_field(&entry.record, "board_state_sha256")?;
            let game = text_field(&entry.record, "game_id")?;
            ensure!(
                valid_sha(&semantic)
                    && valid_sha(&board)
                    && valid_sha(&game)
                    && names.insert(name.clone())
                    && semantics.insert(semantic.clone())
                    && boards.insert(board.clone())
                    && pb_shas.insert(entry.pb.sha256.clone()),
                "invalid/duplicate semantic request"
            );
            let (pb, raw) = read_bound(&root, &entry.pb, 512 * 1024, &mut inventory)?;
            let request = check_pb(&raw, entry, &bundle)?;
            let encoded = input_encoding::encode_request_v7_cpu(&request, expected_model_sha)
                .map_err(|e| anyhow::anyhow!("{}: {}", e.code, e.message))?;
            check_encoding(&encoded, proof)?;
            ordered.update((raw.len() as u64).to_le_bytes());
            ordered.update(&raw);
            requests.push(RequestData {
                descriptor: RowDescriptor {
                    logical_row: index,
                    source_line: entry.source_line,
                    task_id: entry.task_id,
                    name,
                    game_id: game,
                    phase: text_field(&entry.record, "phase")?,
                    ply: usize::try_from(entry.record["ply"].as_u64().context("invalid ply")?)?,
                    pb_sha256: entry.pb.sha256.clone(),
                    row_feature_sha256: proof.post_feature_sha256.clone(),
                    wire_semantic_sha256: entry.wire_semantic_sha256.clone(),
                    input_hash_hex: entry.input_hash_hex.clone(),
                    source_line_sha256: entry.source_line_sha256.clone(),
                    semantic_position_sha256: semantic,
                    board_state_sha256: board,
                },
                pb,
                spatial_sha256: proof.post_spatial_sha256.clone(),
                global_sha256: proof.global_sha256.clone(),
            });
        }
        ensure!(
            hex::encode(ordered.finalize()) == bundle.ordered_pb_sha256,
            "ordered PB inventory mismatch"
        );
        let (_, bytes) = read_bound(&root, &p.rows, META_CAP, &mut inventory)?;
        let maps: Vec<RowMap> = lines(&bytes)?
            .into_iter()
            .map(json)
            .collect::<Result<_>>()?;
        ensure!(
            maps.len() == p.encoded_count * p.packings.len(),
            "packed logical coverage mismatch"
        );
        let mut inputs = Vec::new();
        let mut files = Vec::new();
        let mut shard_position = 0;
        let mut map_position = 0;
        for &target in &p.packings {
            let batches = ranges(p.encoded_count, target)?;
            for (shard_index, chunk) in batches.chunks(MAX_SHARD).enumerate() {
                let directory = format!("b{target}-shard-{shard_index:03}");
                let manifest_name = format!("{directory}/inputs.json");
                let binding = p
                    .input_manifests
                    .get(shard_position)
                    .context("missing input shard")?;
                ensure!(
                    binding.path == manifest_name,
                    "shard order/path differs from exact packing"
                );
                shard_position += 1;
                let (manifest_source, bytes) =
                    read_bound(&root, binding, META_CAP, &mut inventory)?;
                let shard: Shard = json(&bytes)?;
                ensure!(
                    shard.schema == "rustgo-encoded-group-cost-input-v1"
                        && shard.purpose == "calibration"
                        && shard.inputs.len() == chunk.len(),
                    "invalid shard schema/count/scope"
                );
                ensure!(
                    shard.provenance
                        == format!(
                            "CPU_WORKER_V7_INPUTS_ONLY; mode={}; prepared_sha256={}; see external provenance.json and rows.jsonl; no model/accuracy/performance claim",
                            p.coverage_mode, p.prepared_manifest_sha256
                        ),
                    "shard provenance declaration differs"
                );
                for (input, &(first, batch)) in shard.inputs.iter().zip(chunk) {
                    let id = format!("calibration-b{target}-row-{first:04}");
                    ensure!(
                        input.id == id && input.physical_batch == batch,
                        "wrong input identity/physical tail batch"
                    );
                    ensure!(
                        input.spatial.path == format!("{id}.spatial.f32le")
                            && input.global.path == format!("{id}.global.f32le"),
                        "wrong tensor file identity"
                    );
                    let spatial_binding = Binding {
                        path: format!("{directory}/{}", input.spatial.path),
                        ..input.spatial.clone()
                    };
                    let global_binding = Binding {
                        path: format!("{directory}/{}", input.global.path),
                        ..input.global.clone()
                    };
                    ensure!(
                        spatial_binding.bytes == batch * SPATIAL_BYTES
                            && global_binding.bytes == batch * GLOBAL_BYTES,
                        "tensor declared shape mismatch"
                    );
                    let (spatial, spatial_bytes) =
                        read_bound(&root, &spatial_binding, 8 * SPATIAL_BYTES, &mut inventory)?;
                    let (global, global_bytes) =
                        read_bound(&root, &global_binding, 8 * GLOBAL_BYTES, &mut inventory)?;
                    let label = tensor_sha(batch, &spatial_bytes, &global_bytes);
                    let descriptor = InputDescriptor {
                        id,
                        manifest_path: manifest_source.path.clone(),
                        manifest_sha256: manifest_source.sha256.clone(),
                        target_batch: target,
                        physical_batch: batch,
                        tensor_sha256: label,
                        rows: requests[first..first + batch]
                            .iter()
                            .map(|r| r.descriptor.clone())
                            .collect(),
                    };
                    for row in 0..batch {
                        let map = maps.get(map_position).context("missing packed row")?;
                        map_position += 1;
                        check_map(
                            map,
                            &descriptor,
                            row,
                            &manifest_name,
                            &spatial_binding,
                            &global_binding,
                        )?;
                    }
                    check_tensor_rows(
                        &descriptor,
                        &spatial_bytes,
                        &global_bytes,
                        &requests[first..first + batch],
                    )?;
                    inputs.push(descriptor);
                    files.push(InputFiles {
                        spatial,
                        global,
                        first,
                    });
                }
            }
        }
        ensure!(
            shard_position == p.input_manifests.len() && map_position == maps.len(),
            "extra shard/row inventory"
        );
        let corpus = Self {
            root,
            provenance_sha256: expected_sha.into(),
            model_sha256: expected_model_sha.into(),
            coverage_mode: p.coverage_mode,
            packings: p.packings,
            requests,
            inputs,
            files,
            sources: inventory.into_values().collect(),
        };
        corpus.recheck()?;
        Ok(corpus)
    }
    pub fn inputs(&self) -> &[InputDescriptor] {
        &self.inputs
    }
    pub fn coverage_mode(&self) -> &str {
        &self.coverage_mode
    }
    pub fn selected_count(&self) -> usize {
        self.requests.len()
    }
    pub fn model_sha256(&self) -> &str {
        &self.model_sha256
    }
    pub fn provenance_sha256(&self) -> &str {
        &self.provenance_sha256
    }
    pub fn packings(&self) -> &[usize] {
        &self.packings
    }
    pub fn sources(&self) -> &[Source] {
        &self.sources
    }
    pub fn load_input(&self, index: usize) -> Result<LoadedInput> {
        let descriptor = self
            .inputs
            .get(index)
            .context("input index out of bounds")?;
        let files = &self.files[index];
        let spatial_bytes = bounded_bytes(
            &files.spatial.path,
            files.spatial.bytes,
            &files.spatial.sha256,
            8 * SPATIAL_BYTES,
        )?;
        let global_bytes = bounded_bytes(
            &files.global.path,
            files.global.bytes,
            &files.global.sha256,
            8 * GLOBAL_BYTES,
        )?;
        let requests = &self.requests[files.first..files.first + descriptor.physical_batch];
        check_tensor_rows(descriptor, &spatial_bytes, &global_bytes, requests)?;
        let rows = requests
            .iter()
            .map(|r| {
                Ok(LoadedRow {
                    descriptor: r.descriptor.clone(),
                    request_pb: bounded_bytes(&r.pb.path, r.pb.bytes, &r.pb.sha256, 512 * 1024)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(LoadedInput {
            descriptor: descriptor.clone(),
            spatial: floats(&spatial_bytes, descriptor.physical_batch * SPATIAL_FLOATS)?,
            global: floats(&global_bytes, descriptor.physical_batch * GLOBAL_FLOATS)?,
            tensor_sha256: descriptor.tensor_sha256.clone(),
            rows,
        })
    }
    pub fn recheck(&self) -> Result<()> {
        for source in &self.sources {
            ensure!(
                source.path.canonicalize()?.starts_with(&self.root),
                "artifact source escaped original root"
            );
            verify_source(source)?;
        }
        Ok(())
    }
}
fn check_map(
    map: &RowMap,
    input: &InputDescriptor,
    row: usize,
    manifest: &str,
    spatial: &Binding,
    global: &Binding,
) -> Result<()> {
    let expected = input.rows.get(row).context("row map out of bounds")?;
    ensure!(
        map.target_batch == input.target_batch
            && map.physical_batch == input.physical_batch
            && map.row == row
            && map.logical_row == expected.logical_row
            && map.source_line == expected.source_line
            && map.task_id == expected.task_id
            && map.manifest == manifest
            && map.input_id == input.id
            && !map.padding,
        "row packing/order/member mismatch"
    );
    ensure!(
        map.spatial_file == spatial.path
            && map.global_file == global.path
            && map.spatial_sha256 == spatial.sha256
            && map.global_sha256 == global.sha256
            && map.spatial_offset_bytes == row * SPATIAL_BYTES
            && map.spatial_bytes == SPATIAL_BYTES
            && map.global_offset_bytes == row * GLOBAL_BYTES
            && map.global_bytes == GLOBAL_BYTES,
        "row tensor file/offset/length mismatch"
    );
    ensure!(
        map.input_tensor_sha256 == input.tensor_sha256
            && map.row_feature_sha256 == expected.row_feature_sha256,
        "row tensor/feature SHA mismatch"
    );
    Ok(())
}
fn check_tensor_rows(
    input: &InputDescriptor,
    spatial: &[u8],
    global: &[u8],
    requests: &[RequestData],
) -> Result<()> {
    let batch = input.physical_batch;
    ensure!(
        requests.len() == batch && input.rows.len() == batch,
        "tensor row count mismatch"
    );
    floats(spatial, batch * SPATIAL_FLOATS)?;
    floats(global, batch * GLOBAL_FLOATS)?;
    ensure!(
        tensor_sha(batch, spatial, global) == input.tensor_sha256,
        "tensor domain SHA mismatch"
    );
    for (row, request) in requests.iter().enumerate() {
        let s = &spatial[row * SPATIAL_BYTES..(row + 1) * SPATIAL_BYTES];
        let g = &global[row * GLOBAL_BYTES..(row + 1) * GLOBAL_BYTES];
        ensure!(
            hash(s) == request.spatial_sha256
                && hash(g) == request.global_sha256
                && feature_sha(s, g) == request.descriptor.row_feature_sha256,
            "tensor slice differs from actual Worker PB features"
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "inputs_tests.rs"]
mod tests;

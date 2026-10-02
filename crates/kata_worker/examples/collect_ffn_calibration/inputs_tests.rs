//! Boundary tests use no model, CUDA, original corpus files or NN outputs.
use super::*;

fn temporary() -> PathBuf {
    static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let serial = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "rustgo-calibration-reader-{}-{nanos}-{serial}",
        std::process::id()
    ));
    fs::create_dir(&path).unwrap();
    path.canonicalize().unwrap()
}
fn source(path: &Path, raw: &[u8]) -> Source {
    fs::write(path, raw).unwrap();
    Source {
        path: path.canonicalize().unwrap(),
        bytes: raw.len() as u64,
        sha256: hash(raw),
    }
}
fn row(index: usize, spatial: &[u8], global: &[u8]) -> RequestData {
    let sha = "1".repeat(64);
    RequestData {
        descriptor: RowDescriptor {
            logical_row: index,
            source_line: index + 1,
            task_id: (index + 1) as u64,
            name: format!("row-{index}"),
            game_id: sha.clone(),
            phase: "early".into(),
            ply: index,
            pb_sha256: sha.clone(),
            row_feature_sha256: feature_sha(spatial, global),
            wire_semantic_sha256: sha.clone(),
            input_hash_hex: sha.clone(),
            source_line_sha256: sha.clone(),
            semantic_position_sha256: sha.clone(),
            board_state_sha256: sha.clone(),
        },
        pb: Source {
            path: PathBuf::from("unused"),
            bytes: 0,
            sha256: sha,
        },
        spatial_sha256: hash(spatial),
        global_sha256: hash(global),
    }
}
fn tensors(batch: usize) -> (InputDescriptor, Vec<RequestData>, Vec<u8>, Vec<u8>) {
    let mut spatial = Vec::new();
    let mut global = Vec::new();
    let mut requests = Vec::new();
    for index in 0..batch {
        let s = input_encoding::f32le(&vec![(index + 1) as f32 / 4.0; SPATIAL_FLOATS]);
        let g = input_encoding::f32le(&vec![(index + 1) as f32; GLOBAL_FLOATS]);
        requests.push(row(index, &s, &g));
        spatial.extend(s);
        global.extend(g);
    }
    (
        InputDescriptor {
            id: "input".into(),
            manifest_path: "unused".into(),
            manifest_sha256: "2".repeat(64),
            target_batch: batch,
            physical_batch: batch,
            tensor_sha256: tensor_sha(batch, &spatial, &global),
            rows: requests.iter().map(|r| r.descriptor.clone()).collect(),
        },
        requests,
        spatial,
        global,
    )
}

#[test]
fn exact_tail_packing_does_not_promote_b1_or_b4_to_target_batch() {
    let b1 = ranges(2044, 1).unwrap();
    let b3 = ranges(2044, 3).unwrap();
    let b8 = ranges(2044, 8).unwrap();
    assert_eq!((b1.len(), b3.len(), b8.len()), (2044, 682, 256));
    assert_eq!(b3.last(), Some(&(2043, 1)));
    assert_eq!(b8.last(), Some(&(2040, 4)));
    for batch in [b1, b3, b8] {
        assert_eq!(
            batch
                .iter()
                .flat_map(|(first, n)| *first..(*first + *n))
                .collect::<Vec<_>>(),
            (0..2044).collect::<Vec<_>>()
        );
    }
    assert_eq!(ranges(9, 8).unwrap(), vec![(0, 8), (8, 1)]);
    assert!(ranges(2044, 4).is_err());
    assert!(ranges(0, 1).is_err());
}

#[test]
fn prefix_and_complete_admission_are_distinct() {
    assert!(validate_mode("complete-calibration", 2044).is_ok());
    assert!(validate_mode("prefix-diagnostic", 9).is_ok());
    assert!(validate_mode("complete-calibration", 9).is_err());
    assert!(validate_mode("prefix-diagnostic", 2044).is_err());
    assert!(validate_mode("prefix-diagnostic", 0).is_err());
    assert!(validate_mode("selection", 2044).is_err());
}

#[test]
fn tensor_rows_reject_reordered_slices_even_with_rebound_batch_hash() {
    let (mut input, requests, mut spatial, global) = tensors(3);
    check_tensor_rows(&input, &spatial, &global, &requests).unwrap();
    for i in 0..SPATIAL_BYTES {
        spatial.swap(i, i + SPATIAL_BYTES);
    }
    input.tensor_sha256 = tensor_sha(3, &spatial, &global);
    assert!(check_tensor_rows(&input, &spatial, &global, &requests).is_err());
}

#[test]
fn tensor_rows_reject_nonfinite_and_wrong_physical_shape() {
    let (mut input, requests, mut spatial, global) = tensors(3);
    spatial[2 * SPATIAL_BYTES..2 * SPATIAL_BYTES + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    input.tensor_sha256 = tensor_sha(3, &spatial, &global);
    assert!(check_tensor_rows(&input, &spatial, &global, &requests).is_err());
    let (mut input, requests, spatial, global) = tensors(3);
    input.physical_batch = 1;
    assert!(check_tensor_rows(&input, &spatial, &global, &requests).is_err());
    assert!(floats(&[0, 0, 0], 1).is_err());
}

#[test]
fn signed_zero_and_domains_remain_distinct() {
    let plus = input_encoding::f32le(&[0.0]);
    let minus = input_encoding::f32le(&[-0.0]);
    assert_ne!(feature_sha(&plus, &plus), feature_sha(&minus, &plus));
    assert_ne!(tensor_sha(1, &plus, &plus), tensor_sha(3, &plus, &plus));
    assert_ne!(feature_sha(&plus, &plus), tensor_sha(1, &plus, &plus));
    assert_eq!(floats(&minus, 1).unwrap()[0].to_bits(), (-0.0f32).to_bits());
}

#[test]
fn row_offset_cannot_alias_another_physical_row_or_pad_tail() {
    let (input, _, spatial, global) = tensors(3);
    let s = Binding {
        path: "b3-shard-000/input.spatial.f32le".into(),
        bytes: spatial.len(),
        sha256: hash(&spatial),
    };
    let g = Binding {
        path: "b3-shard-000/input.global.f32le".into(),
        bytes: global.len(),
        sha256: hash(&global),
    };
    let mut map = RowMap {
        target_batch: 3,
        physical_batch: 3,
        row: 1,
        logical_row: 1,
        source_line: 2,
        task_id: 2,
        manifest: "b3-shard-000/inputs.json".into(),
        input_id: input.id.clone(),
        input_tensor_sha256: input.tensor_sha256.clone(),
        spatial_file: s.path.clone(),
        global_file: g.path.clone(),
        spatial_sha256: s.sha256.clone(),
        global_sha256: g.sha256.clone(),
        spatial_offset_bytes: SPATIAL_BYTES,
        spatial_bytes: SPATIAL_BYTES,
        global_offset_bytes: GLOBAL_BYTES,
        global_bytes: GLOBAL_BYTES,
        row_feature_sha256: input.rows[1].row_feature_sha256.clone(),
        padding: false,
    };
    check_map(&map, &input, 1, "b3-shard-000/inputs.json", &s, &g).unwrap();
    map.spatial_offset_bytes = 0;
    assert!(check_map(&map, &input, 1, "b3-shard-000/inputs.json", &s, &g).is_err());
    map.spatial_offset_bytes = SPATIAL_BYTES;
    map.padding = true;
    assert!(check_map(&map, &input, 1, "b3-shard-000/inputs.json", &s, &g).is_err());
    map.padding = false;
    map.task_id = 1;
    assert!(check_map(&map, &input, 1, "b3-shard-000/inputs.json", &s, &g).is_err());
}

fn request_fixture() -> (wire::EvalRequest, RequestEntry, Bundle) {
    let model = "a".repeat(64);
    let mut request = wire::EvalRequest {
        task_id: 1,
        generation: 1,
        session_id: "test".into(),
        model_sha256: model.clone(),
        lease_ms: 300000,
        position: Some(wire::Position {
            board_size: 19,
            komi: 7.5,
            rules: "chinese".into(),
            initial_player: 1,
            next_player: 1,
            moves: vec![],
            initial_stones: vec![],
        }),
        parameters: Some(wire::EvalParameters {
            symmetry: 5,
            policy_temperature: 0.75,
            policy_optimism: 0.25,
            draw_equivalent_wins_for_white: 0.5,
            include_ownership: true,
            max_history: 10000,
            skip_cache: true,
            ..Default::default()
        }),
        ..Default::default()
    };
    request.input_hash = Sha256::digest(request.encode_to_vec()).to_vec();
    let mut semantic = Sha256::new();
    for raw in [
        request.position.as_ref().unwrap().encode_to_vec(),
        request.parameters.as_ref().unwrap().encode_to_vec(),
    ] {
        semantic.update((raw.len() as u64).to_le_bytes());
        semantic.update(raw);
    }
    let raw = request.encode_to_vec();
    let entry = RequestEntry {
        source_line: 1,
        source_line_sha256: "b".repeat(64),
        record: request_semantics(&request).unwrap(),
        task_id: 1,
        pb: Binding {
            path: "requests/0001.pb".into(),
            bytes: raw.len(),
            sha256: hash(&raw),
        },
        input_hash_hex: hex::encode(&request.input_hash),
        wire_semantic_sha256: hex::encode(semantic.finalize()),
    };
    let bundle = Bundle {
        schema: "rustgo-calibration-pb-input-v1".into(),
        mode: "prefix-diagnostic".into(),
        split: "calibration".into(),
        source_total: TOTAL,
        selected_count: 1,
        packings: vec![1],
        model_sha256: model,
        session_id: "test".into(),
        generation: 1,
        lease_ms: 300000,
        ordered_pb_sha256: "c".repeat(64),
        sources: vec![],
        requests: vec![],
        producer: json!({}),
    };
    (request, entry, bundle)
}

#[test]
fn pb_semantics_bind_temperature_and_every_boolean_not_just_features() {
    let (request, mut entry, bundle) = request_fixture();
    let raw = request.encode_to_vec();
    check_pb(&raw, &entry, &bundle).unwrap();
    for key in [
        "skip_cache",
        "include_ownership",
        "conservative_pass",
        "enable_passing_hacks",
        "always_compute_pass_alive",
        "exclude_territory_adjacent_to_atari",
        "avoid_mytdagger_hack",
        "allow_terminal_search_history",
        "force_non_terminal",
    ] {
        let original = entry.record["parameters"][key].clone();
        entry.record["parameters"][key] = json!(!original.as_bool().unwrap());
        assert!(check_pb(&raw, &entry, &bundle).is_err(), "{key}");
        entry.record["parameters"][key] = original;
    }
    entry.record["parameters"]["policy_temperature"] = json!(1.0);
    assert!(check_pb(&raw, &entry, &bundle).is_err());
}

#[test]
fn pb_envelope_unknown_fields_and_input_hash_cannot_be_rebound() {
    let (mut request, entry, bundle) = request_fixture();
    let mut raw = request.encode_to_vec();
    raw.extend([0x98, 0x06, 0x01]);
    assert!(check_pb(&raw, &entry, &bundle).is_err());
    request.generation = 2;
    assert!(check_pb(&request.encode_to_vec(), &entry, &bundle).is_err());
    request.generation = 1;
    request.input_hash[0] ^= 1;
    assert!(check_pb(&request.encode_to_vec(), &entry, &bundle).is_err());
}

#[test]
fn real_worker_encoding_rejects_illegal_replay_and_changed_prepared_evidence() {
    let (mut request, _, bundle) = request_fixture();
    let encoded = input_encoding::encode_request_v7_cpu(&request, &bundle.model_sha256).unwrap();
    let mut proof = Proof {
        source_line: 1,
        source_line_sha256: "0".repeat(64),
        record: json!({}),
        task_id: 1,
        pb: Binding {
            path: "requests/0001.pb".into(),
            bytes: 0,
            sha256: "0".repeat(64),
        },
        input_hash_hex: "0".repeat(64),
        wire_semantic_sha256: "0".repeat(64),
        prepared: evidence(&encoded).unwrap(),
        pre_spatial_sha256: hash(&input_encoding::f32le(&encoded.pre_spatial)),
        post_spatial_sha256: hash(&input_encoding::f32le(&encoded.spatial)),
        global_sha256: hash(&input_encoding::f32le(&encoded.global)),
        pre_feature_sha256: input_encoding::row_feature_sha256(
            &encoded.pre_spatial,
            &encoded.global,
        )
        .unwrap(),
        post_feature_sha256: input_encoding::row_feature_sha256(&encoded.spatial, &encoded.global)
            .unwrap(),
    };
    check_encoding(&encoded, &proof).unwrap();
    proof.prepared["actual_misc"]["policy_temperature_f32_bits"] = json!("3f800000");
    assert!(check_encoding(&encoded, &proof).is_err());
    let position = request.position.as_mut().unwrap();
    position.moves = vec![
        wire::Move {
            color: 1,
            vertex: 21,
        },
        wire::Move {
            color: 2,
            vertex: 21,
        },
    ];
    assert!(input_encoding::encode_request_v7_cpu(&request, &bundle.model_sha256).is_err());
}

#[test]
fn sources_reject_changed_bytes_even_at_same_length_and_preserve_expected_sha() {
    let root = temporary();
    let file = root.join("data");
    let expected = source(&file, b"abcd");
    verify_source(&expected).unwrap();
    fs::write(&file, b"abce").unwrap();
    assert!(verify_source(&expected).is_err());
    assert!(bounded_bytes(&file, 4, &expected.sha256, 4).is_err());
    fs::write(&file, b"abcdx").unwrap();
    assert!(verify_source(&expected).is_err());
    let mut map = BTreeMap::new();
    register(&mut map, expected.clone()).unwrap();
    let mut changed = expected.clone();
    changed.sha256 = "1".repeat(64);
    assert!(register(&mut map, changed).is_err());
    assert_eq!(map[&expected.path], expected);
}

#[test]
fn relative_paths_and_jsonl_line_identity_are_strict() {
    for value in [
        "",
        "../data",
        "a/../b",
        "/absolute",
        "C:/absolute",
        "a\\b",
        "a:stream",
        "a/./b",
    ] {
        assert!(local_path(value).is_err(), "{value}");
    }
    assert!(local_path("b8-shard-001/inputs.json").is_ok());
    assert_eq!(
        lines(b"one\r\ntwo\n").unwrap(),
        vec![b"one".as_slice(), b"two".as_slice()]
    );
    assert!(lines(b"one\n\n").is_err());
    assert!(lines(b"one").is_err());
    assert!(same_semantics(&json!({"komi":0}), &json!({"komi":0.0})));
    assert!(!same_semantics(&json!({"komi":7.5}), &json!({"komi":6.5})));
}

#[test]
fn recheck_rejects_mutation_without_rebinding_and_index_is_bounded() {
    let root = temporary();
    let expected = source(&root.join("provenance.json"), b"original");
    let corpus = Corpus {
        root,
        provenance_sha256: expected.sha256.clone(),
        model_sha256: "a".repeat(64),
        coverage_mode: "prefix-diagnostic".into(),
        packings: vec![1],
        requests: vec![],
        inputs: vec![],
        files: vec![],
        sources: vec![expected.clone()],
    };
    corpus.recheck().unwrap();
    assert!(corpus.load_input(0).is_err());
    fs::write(&expected.path, b"modified").unwrap();
    assert!(corpus.recheck().is_err());
    assert_eq!(corpus.provenance_sha256(), expected.sha256);
}

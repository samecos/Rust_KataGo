use super::*;

fn marker_dir() -> PathBuf {
    let root = std::env::temp_dir().join(format!("rustgo-marker-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&root).unwrap();
    root
}
fn clean_marker_dir(root: &Path) {
    for name in ["provenance.json", "provenance.json.pending"] {
        let path = root.join(name);
        if path.exists() {
            fs::remove_file(path).unwrap();
        }
    }
    fs::remove_dir(root).unwrap();
}

#[test]
fn completion_write_failure_keeps_only_pending_bytes() {
    let root = marker_dir();
    let result = commit_new_with(
        &root,
        "provenance.json",
        b"complete",
        |file, raw| {
            file.write_all(&raw[..2])?;
            Err(std::io::Error::other("injected write failure"))
        },
        fs::File::sync_all,
    );
    assert!(result.is_err());
    assert!(!root.join("provenance.json").exists());
    assert_eq!(
        fs::read(root.join("provenance.json.pending")).unwrap(),
        b"co"
    );
    clean_marker_dir(&root);
}

#[test]
fn completion_sync_failure_cannot_publish_even_valid_complete_json() {
    let root = marker_dir();
    let raw = b"{\"ok\":true}";
    let result = commit_new_with(
        &root,
        "provenance.json",
        raw,
        |file, raw| file.write_all(raw),
        |_| Err(std::io::Error::other("injected sync failure")),
    );
    assert!(result.is_err());
    assert!(!root.join("provenance.json").exists());
    assert_eq!(fs::read(root.join("provenance.json.pending")).unwrap(), raw);
    clean_marker_dir(&root);
}

#[test]
fn completion_publication_is_no_replace_and_leaves_committed_bytes() {
    let root = marker_dir();
    let binding = commit_json(&root, "provenance.json", &json!({"ok":true})).unwrap();
    let raw = fs::read(root.join("provenance.json")).unwrap();
    assert_eq!(hash(&raw), binding.sha256);
    assert_eq!(raw, fs::read(root.join("provenance.json.pending")).unwrap());
    clean_marker_dir(&root);
    let root = marker_dir();
    fs::write(root.join("provenance.json"), b"existing").unwrap();
    assert!(commit_json(&root, "provenance.json", &json!({"ok":true})).is_err());
    assert_eq!(fs::read(root.join("provenance.json")).unwrap(), b"existing");
    assert!(root.join("provenance.json.pending").exists());
    clean_marker_dir(&root);
}

#[test]
fn full_calibration_has_explicit_shards_and_exact_tail_sizes() {
    let b1 = batch_ranges(2044, 1).unwrap();
    assert_eq!((b1.len(), b1.chunks(128).len()), (2044, 16));
    assert_eq!(b1.last(), Some(&(2043, 1)));
    let b3 = batch_ranges(2044, 3).unwrap();
    assert_eq!((b3.len(), b3.chunks(128).len()), (682, 6));
    assert_eq!(b3.last(), Some(&(2043, 1)));
    let b8 = batch_ranges(2044, 8).unwrap();
    assert_eq!((b8.len(), b8.chunks(128).len()), (256, 2));
    assert_eq!(b8.last(), Some(&(2040, 4)));
    for ranges in [&b1, &b3, &b8] {
        assert_eq!(ranges.iter().map(|(_, b)| b).sum::<usize>(), 2044);
        assert!(ranges.windows(2).all(|p| p[0].0 + p[0].1 == p[1].0));
    }
}

#[test]
fn small_prefix_does_not_replicate_or_drop_tail_rows() {
    assert_eq!(batch_ranges(9, 8).unwrap(), vec![(0, 8), (8, 1)]);
    assert_eq!(batch_ranges(2, 3).unwrap(), vec![(0, 2)]);
    for (count, batch) in [(0, 1), (2045, 1), (4, 2), (4, 64)] {
        assert!(batch_ranges(count, batch).is_err());
    }
}

#[test]
fn protobuf_semantics_preserve_all_explicit_flags() {
    let request = wire::EvalRequest {
        position: Some(wire::Position {
            board_size: 19,
            komi: 6.5,
            rules: "chinese".into(),
            initial_player: 2,
            next_player: 2,
            initial_stones: vec![wire::Stone {
                color: 1,
                vertex: 72,
            }],
            ..Default::default()
        }),
        parameters: Some(wire::EvalParameters {
            policy_temperature: 1.0,
            max_history: 10000,
            symmetry: 5,
            skip_cache: true,
            include_ownership: true,
            ..Default::default()
        }),
        ..Default::default()
    };
    let actual = request_semantics(&request).unwrap();
    assert_eq!(actual["parameters"].as_object().unwrap().len(), 15);
    assert_eq!(actual["position"]["komi"], 6.5);
    assert_eq!(actual["position"]["initial_stones"][0]["vertex"], 72);
    let mut changed = actual.clone();
    changed["parameters"]["symmetry"] = json!(0);
    assert!(!same_semantics(&actual, &changed));
    let mut omitted = actual.clone();
    omitted["parameters"]
        .as_object_mut()
        .unwrap()
        .remove("force_non_terminal");
    assert!(!same_semantics(&actual, &omitted));
    assert!(same_semantics(&json!({"x":0}), &json!({"x":0.0})));
}

#[test]
fn relative_bindings_reject_escape_or_rooted_paths() {
    for path in ["", "../request.pb", "sources/../request.pb", "/request.pb"] {
        assert!(relative_path(path).is_err(), "{path}");
    }
    assert!(relative_path("sources/requests/0001.pb").is_ok());
}

#[test]
fn binding_detects_same_length_tamper_and_writer_refuses_overwrite() {
    let root = std::env::temp_dir().join(format!("rustgo-encoding-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&root).unwrap();
    let binding = write_new(&root, "request.pb", b"original").unwrap();
    assert_eq!(read_bound(&root, &binding, 8).unwrap(), b"original");
    assert!(read_bound(&root, &binding, 7).is_err());
    assert!(write_new(&root, "request.pb", b"replaced").is_err());
    fs::write(root.join("request.pb"), b"modified").unwrap();
    assert!(read_bound(&root, &binding, 8).is_err());
    fs::remove_file(root.join("request.pb")).unwrap();
    fs::remove_dir(root).unwrap();
}

#[test]
fn group_cost_label_binds_batch_and_separate_complete_tensor_files() {
    let spatial = f32le(&[1.0, 2.0]);
    let global = f32le(&[3.0, 4.0]);
    let label = tensor_label(2, &spatial, &global);
    assert_ne!(label, tensor_label(1, &spatial, &global));
    assert_ne!(label, tensor_label(2, &global, &spatial));
    let mut interleaved = b"rustgo-encoded-group-cost-tensors-v1\0".to_vec();
    interleaved.extend(2u64.to_le_bytes());
    interleaved.extend(f32le(&[1.0, 3.0, 2.0, 4.0]));
    assert_ne!(label, hash(&interleaved));
    let mut reference = b"rustgo-encoded-group-cost-tensors-v1\0".to_vec();
    reference.extend(2u64.to_le_bytes());
    reference.extend(f32le(&[1.0, 2.0, 3.0, 4.0]));
    assert_eq!(label, hash(&reference));
}

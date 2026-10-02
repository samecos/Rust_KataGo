use super::*;

fn heads(batch: usize) -> [Vec<f32>; 5] {
    HEADS.map(|(_, width)| vec![0.25; batch * width])
}

#[test]
fn all_five_full_heads_reject_nan_and_wrong_sizes() {
    assert!(raw_bits(heads(3), 3).is_ok());
    for i in 0..5 {
        let mut short = heads(3);
        short[i].pop();
        assert!(raw_bits(short, 3).is_err());
        let mut nan = heads(1);
        nan[i][0] = f32::from_bits(0x7fc00000);
        assert!(raw_bits(nan, 1).is_err());
    }
}

#[test]
fn signed_zero_bits_are_preserved_by_the_comparison() {
    let mut a = heads(1);
    let mut b = heads(1);
    a[2][5] = 0.0;
    b[2][5] = -0.0;
    assert_ne!(raw_bits(a, 1).unwrap(), raw_bits(b, 1).unwrap());
}

#[test]
fn a_valid_self_hash_cannot_substitute_different_smoke_bytes() {
    let source = |bytes, sha: &str| Source {
        logical_name: "test".into(),
        path: PathBuf::from("unused"),
        bytes,
        sha256: sha.into(),
    };
    let mut row = Smoke {
        batch: 1,
        spatial: source(
            31768,
            "137f93ed0b5fe407fe20c5805429dddb478efb7478d4b85c42eabd50a4545ef1",
        ),
        global: source(
            76,
            "80643b217bb1565fc103796973ccd282b942a9863550ceeb1cec0687cc31762a",
        ),
    };
    assert!(require_frozen_smoke(&row).is_ok());
    row.global.sha256 = hash(&[0; 76]);
    assert!(require_frozen_smoke(&row).is_err());
    row.batch = 3;
    assert!(require_frozen_smoke(&row).is_err());
}

#[test]
fn finite_environment_cannot_enable_tuning_or_profile() {
    let environment = expected_environment();
    assert_eq!(environment.get("KATAGO_CUDA_INT8_GEMM_TUNE").unwrap(), "0");
    assert!(
        !environment
            .keys()
            .any(|k| k.contains("PROFILE") || k.contains("DUMP") || k.contains("DEBUG"))
    );
    assert_eq!(environment.len(), 9);
}

#[test]
fn result_commit_never_overwrites_prior_result() {
    let dir = std::env::temp_dir().join(format!(
        "model-probe-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&dir).unwrap();
    let path = dir.join("result.json");
    save(&path, b"prior").unwrap();
    assert!(commit_result(&path, &json!({"status":"PASS"})).is_err());
    assert_eq!(fs::read(&path).unwrap(), b"prior");
    fs::remove_file(path).unwrap();
    fs::remove_file(dir.join("result.json.pending")).unwrap();
    fs::remove_dir(dir).unwrap();
}

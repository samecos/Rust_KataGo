use super::*;

fn binding() -> Binding {
    let h = "1".repeat(64);
    Binding { model_sha256: h.clone(), graph_sha256: h.clone(), recipe_sha256: h.clone(),
        execution_policy_sha256: h.clone(), executable_sha256: h.clone(), host_source_sha256: h.clone(),
        kernel_build_sha256: h.clone(), device_fingerprint_sha256: h.clone(), gpu_uuid: "GPU-fixture".into(),
        compute_capability: [12, 0], sm_count: 70, driver_api_version: 13030,
        cublaslt_version: 130600, cublaslt_binary_sha256: h.clone(), cublas_binary_sha256: h.clone(),
        cuda_runtime_provenance_sha256: h.clone(), host_abi: "x86_64-windows-msvc".into(),
        pointer_width_bits: 64, little_endian: true, algo_size_bytes: 64, algo_alignment_bytes: 8,
        sources: vec![SourceIdentity { logical_name: "int8.rs".into(), sha256: h }] }
}
fn key() -> ProblemKey { ProblemKey::new(361, 768, 384, 384).unwrap() }
fn implementation(key: ProblemKey) -> Implementation {
    Implementation { descriptors: Descriptors::expected(key), opaque_algorithm_hex: "01".repeat(64),
        attributes: AlgorithmAttributes { algorithm_id: 17, tile_id: 2, stages_id: 3, split_k: 1,
            reduction_scheme: 0, cta_swizzling: 0, custom_option: 0,
            inner_shape_id: Queried::Value(0), cluster_shape_id: Queried::NotSupported,
            numerical_implementation_flags: 4 },
        minimum_alignment_bytes: [16; 4], required_workspace_bytes: 0,
        provided_workspace_bytes: 32 * 1024 * 1024, workspace_alignment_bytes: 256,
        workspace_ownership: "runtime-per-stream-exclusive".into() }
}
fn plan() -> Plan {
    Plan { schema: SCHEMA.into(), scope: SCOPE.into(), encoding: ENCODING.into(), binding: binding(),
        inventory: vec![key()], entries: vec![Entry { key: key(), implementation: implementation(key()) }] }
}

#[test]
fn exact_canonical_roundtrip_preserves_optional_attribute_state() {
    let p = plan(); let bytes = p.canonical_bytes().unwrap();
    let actual = Plan::parse_exact(&bytes, &sha256(&bytes), &binding(), &[key()]).unwrap();
    assert_eq!(actual, p);
    assert_eq!(actual.entries[0].implementation.attributes.cluster_shape_id, Queried::NotSupported);
}

#[test]
fn library_model_recipe_and_policy_bindings_are_independent() {
    let p = plan(); let bytes = p.canonical_bytes().unwrap();
    for mutation in 0..5 {
        let mut expected = binding();
        match mutation {
            0 => expected.cublaslt_version += 1,
            1 => expected.model_sha256 = "2".repeat(64),
            2 => expected.recipe_sha256 = "2".repeat(64),
            3 => expected.execution_policy_sha256 = "2".repeat(64),
            _ => expected.sources[0].sha256 = "2".repeat(64),
        }
        assert!(Plan::parse_exact(&bytes, &sha256(&bytes), &expected, &[key()]).is_err());
    }
}

#[test]
fn external_sha_rejects_modified_opaque_algorithm() {
    let p = plan(); let bytes = p.canonical_bytes().unwrap();
    let mut altered = p.clone(); altered.entries[0].implementation.opaque_algorithm_hex = "02".repeat(64);
    let altered_bytes = altered.canonical_bytes().unwrap();
    assert!(Plan::parse_exact(&altered_bytes, &sha256(&bytes), &binding(), &[key()]).is_err());
}

#[test]
fn whitespace_and_unknown_field_cannot_sneak_into_canonical_artifact() {
    let p = plan(); let bytes = serde_json::to_vec_pretty(&p).unwrap();
    assert!(Plan::parse_exact(&bytes, &sha256(&bytes), &binding(), &[key()]).is_err());
    let mut value = serde_json::to_value(&p).unwrap(); value["whole_network_certified"] = true.into();
    let bytes = serde_json::to_vec(&value).unwrap();
    assert!(Plan::parse_exact(&bytes, &sha256(&bytes), &binding(), &[key()]).is_err());
}

#[test]
fn physical_batch_and_padding_are_not_interchangeable() {
    assert!(ProblemKey::new(362, 768, 384, 384).is_err());
    assert!(ProblemKey::new(361, 768, 383, 400).is_err());
    assert!(ProblemKey::new(361, 769, 384, 384).is_err());
    assert!(ProblemKey::new(361, 768, 8193, 8208).is_err());
    let mut p = plan(); p.inventory[0].physical_batch = 2;
    assert!(p.validate().is_err());
}

#[test]
fn inventory_rejects_duplicates_order_and_unobserved_extra() {
    assert!(validate_inventory(&[key(), key()]).is_err());
    let b2 = ProblemKey::new(722, 768, 384, 384).unwrap();
    assert!(validate_inventory(&[b2, key()]).is_err());
    let mut p = plan(); p.inventory.push(b2);
    assert!(p.validate().is_err());
}

#[test]
fn shared_physical_cache_key_cannot_silently_change_algorithm() {
    let narrow = ProblemKey::new(361, 768, 383, 384).unwrap();
    let mut p = plan(); p.inventory.insert(0, narrow);
    p.entries.insert(0, Entry { key: narrow, implementation: implementation(narrow) });
    p.validate().unwrap();
    p.entries[1].implementation.opaque_algorithm_hex = "02".repeat(64);
    assert!(p.validate().is_err());
}

#[test]
fn layout_or_scalar_change_is_not_same_integer_problem() {
    for mutation in 0..5 {
        let mut p = plan(); let d = &mut p.entries[0].implementation.descriptors;
        match mutation { 0 => d.transpose_a = 0, 1 => d.beta = 1, 2 => d.layouts[1].leading_dimension += 16,
            3 => d.layouts[1].batch_count = 2, _ => d.auxiliary_pointers_null = false }
        assert!(p.validate().is_err());
    }
}

#[test]
fn actual_subview_alignment_checked_even_when_allocation_base_is_aligned() {
    let i = implementation(key()); let bytes = i.provided_workspace_bytes as usize;
    i.check_addresses([0x1000, 0x2000, 0x3000, 0x3000], 0x4000, bytes).unwrap();
    for index in 0..4 {
        let mut addresses = [0x1000, 0x2000, 0x3000, 0x3000]; addresses[index] += 1;
        assert!(i.check_addresses(addresses, 0x4000, bytes).is_err());
    }
}

#[test]
fn workspace_size_address_and_cd_alias_are_strict() {
    let i = implementation(key()); let bytes = i.provided_workspace_bytes as usize;
    assert!(i.check_addresses([0x1000, 0x2000, 0x3000, 0x3010], 0x4000, bytes).is_err());
    assert!(i.check_addresses([0x1000, 0x2000, 0x3000, 0x3000], 0x4010, bytes).is_err());
    assert!(i.check_addresses([0x1000, 0x2000, 0x3000, 0x3000], 0x4000, bytes - 256).is_err());
    assert!(i.check_addresses([0, 0x2000, 0x3000, 0x3000], 0x4000, bytes).is_err());
}

#[test]
fn algorithm_alignment_also_applies_to_leading_dimension_bytes() {
    let mut i = implementation(key()); i.minimum_alignment_bytes[0] = 256;
    assert!(i.validate(key()).is_err()); // 384-byte A column stride is not 256 aligned.
    i.minimum_alignment_bytes[0] = 3;
    assert!(i.validate(key()).is_err());
}

#[test]
fn record_requires_every_authoritative_shape_before_export() {
    let b2 = ProblemKey::new(722, 768, 384, 384).unwrap();
    let mut session = Session::record(binding(), vec![key(), b2]).unwrap();
    session.observe(key(), &implementation(key())).unwrap();
    assert!(session.export().is_err());
    session.observe(b2, &implementation(b2)).unwrap();
    assert_eq!(session.export().unwrap().entries.len(), 2);
}

#[test]
fn missing_restore_entry_fails_before_any_fallback_can_be_requested() {
    let session = Session::restore(plan()).unwrap();
    let b2 = ProblemKey::new(722, 768, 384, 384).unwrap();
    assert!(session.expected(b2).unwrap_err().contains("no heuristic fallback"));
    assert!(session.expected(key()).unwrap().is_some());
}

#[test]
fn same_session_algorithm_drift_and_failure_are_fail_closed() {
    let mut session = Session::record(binding(), vec![key()]).unwrap();
    session.observe(key(), &implementation(key())).unwrap();
    let mut changed = implementation(key()); changed.attributes.algorithm_id += 1;
    assert!(session.observe(key(), &changed).is_err());
    session.poison();
    assert!(session.expected(key()).is_err());
    assert!(session.observe(key(), &implementation(key())).is_err());
    assert!(session.export().is_err());
}

#[test]
fn restored_payload_is_compared_after_actual_inspection() {
    let mut session = Session::restore(plan()).unwrap();
    let mut changed = implementation(key()); changed.required_workspace_bytes = 4096;
    assert!(session.observe(key(), &changed).is_err());
    session.observe(key(), &implementation(key())).unwrap();
    assert_eq!(session.export().unwrap(), plan());
}

#[test]
fn malformed_abi_and_source_identity_are_rejected() {
    let mut b = binding(); b.algo_size_bytes = 32; assert!(b.validate().is_err());
    let mut b = binding(); b.sources.push(b.sources[0].clone()); assert!(b.validate().is_err());
    let mut b = binding(); b.sources[0].logical_name = "../int8.rs".into(); assert!(b.validate().is_err());
    let mut b = binding(); b.sources[0].sha256 = "A".repeat(64); assert!(b.validate().is_err());
}

#[test]
fn wrong_coverage_cannot_be_rebound_by_parse_caller() {
    let p = plan(); let bytes = p.canonical_bytes().unwrap();
    let b2 = ProblemKey::new(722, 768, 384, 384).unwrap();
    assert!(Plan::parse_exact(&bytes, &sha256(&bytes), &binding(), &[key(), b2]).is_err());
}

#[test]
fn typed_attribute_overflow_cannot_be_truncated_into_native_getter_types() {
    let mut value = serde_json::to_value(plan()).unwrap();
    value["entries"][0]["implementation"]["attributes"]["inner_shape_id"] =
        serde_json::json!({"state":"value","value":65536});
    let bytes = serde_json::to_vec(&value).unwrap();
    assert!(Plan::parse_exact(&bytes, &sha256(&bytes), &binding(), &[key()]).is_err());
    let mut value = serde_json::to_value(plan()).unwrap();
    value["entries"][0]["implementation"]["descriptors"]["layouts"][0]["batch_count"] =
        serde_json::json!(2147483648u64);
    let bytes = serde_json::to_vec(&value).unwrap();
    assert!(Plan::parse_exact(&bytes, &sha256(&bytes), &binding(), &[key()]).is_err());
}

#[test]
fn errors_before_or_after_gemm_poison_even_previously_complete_coverage() {
    // A previous successful forward has covered every GEMM. Each later public
    // operation/model-boundary error must prevent exporting that old table as
    // a successful current session even when stream synchronization succeeds.
    for stage in ["project-half-input-validation", "project-half-quantizer",
        "project-half-gemm", "project-half-dequant", "project-residual-dequant",
        "ffn-shape-validation", "ffn-rms-validation", "ffn-rms-kernel",
        "ffn-swiglu-kernel", "model-attention-kernel", "model-head-kernel",
        "group-cost-recorder-finish", "model-forward-prevalidation"] {
        let mut session = Session::record(binding(), vec![key()]).unwrap();
        session.observe(key(), &implementation(key())).unwrap();
        session.export().unwrap();
        let error = format!("injected {stage}");
        assert_eq!(session.complete_operation::<()>(Err(error.clone())), Err(error));
        assert!(session.ensure_healthy().is_err(), "{stage}");
        assert!(session.export().is_err(), "{stage}");
        assert!(session.complete_operation(Ok(())).is_err(), "{stage}");
    }
}

#[test]
fn successful_whole_operation_keeps_coverage_without_clearing_failure_state() {
    let mut session = Session::record(binding(), vec![key()]).unwrap();
    session.observe(key(), &implementation(key())).unwrap();
    assert_eq!(session.complete_operation(Ok(7)).unwrap(), 7);
    session.export().unwrap();
    session.complete_operation::<()>(Err("asynchronous sync failure".into())).unwrap_err();
    assert!(session.complete_operation(Ok(7)).is_err());
    assert!(session.export().is_err());
}

#[test]
fn equal_zero_default_stream_handles_do_not_prove_same_owner() {
    use std::sync::Arc;
    let owner = Arc::new(0usize);
    let foreign = Arc::new(0usize);
    let owner_context = Arc::new("context-a");
    let foreign_context = Arc::new("context-b");
    assert_eq!(*owner, *foreign);
    assert!(validate_owner_identity(&owner, &foreign, &owner_context, &foreign_context,
        [&owner; 3], [&owner_context; 3]).is_err());
    // Even a different wrapper for a same-context default stream is rejected;
    // this explicit API requires clones of the retained owning stream.
    assert!(validate_owner_identity(&owner, &foreign, &owner_context, &owner_context,
        [&owner; 3], [&owner_context; 3]).is_err());
}

#[test]
fn exact_owner_clone_and_its_scratch_are_admitted() {
    use std::sync::Arc;
    let owner = Arc::new(0usize); let incoming = Arc::clone(&owner);
    let context = Arc::new("context");
    let scratch = [Arc::clone(&owner), Arc::clone(&owner), Arc::clone(&owner)];
    validate_owner_identity(&owner, &incoming, &context, &context,
        [&scratch[0], &scratch[1], &scratch[2]], [&context; 3]).unwrap();
}

#[test]
fn each_replaced_scratch_owner_or_context_is_rejected() {
    use std::sync::Arc;
    let owner = Arc::new(0usize); let foreign = Arc::new(0usize);
    let context = Arc::new("context"); let foreign_context = Arc::new("other-context");
    for index in 0..3 {
        let mut owners = [&owner; 3]; owners[index] = &foreign;
        assert!(validate_owner_identity(&owner, &owner, &context, &context, owners, [&context; 3]).is_err());
        let mut contexts = [&context; 3]; contexts[index] = &foreign_context;
        assert!(validate_owner_identity(&owner, &owner, &context, &context, [&owner; 3], contexts).is_err());
    }
}

#[test]
fn external_weights_can_use_another_upload_stream_only_in_owner_context() {
    use std::sync::Arc;
    let owner_context = Arc::new("context");
    let upload_context = Arc::clone(&owner_context);
    require_same_context(&owner_context, &upload_context).unwrap();
    // Equal descriptive GPU/context labels do not establish object ownership.
    let same_label_different_context = Arc::new("context");
    assert!(require_same_context(&owner_context, &same_label_different_context).is_err());
}

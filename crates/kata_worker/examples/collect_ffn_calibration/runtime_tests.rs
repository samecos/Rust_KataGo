use super::*;

fn input(batch: usize) -> (Vec<f32>, Vec<f32>) {
    let mut spatial = vec![0.0; batch * 22 * 361];
    for row in spatial.chunks_exact_mut(22 * 361) {
        row[..361].fill(1.0);
    }
    (spatial, vec![0.0; batch * 19])
}
fn heads(batch: usize) -> Heads {
    Heads {
        policy: vec![0.0; batch * 2172],
        value: vec![0.0; batch * 3],
        misc: vec![0.0; batch * 10],
        moremisc: vec![0.0; batch * 8],
        ownership: vec![0.0; batch * 361],
    }
}

#[test]
fn exact_input_shape_finite_and_complete_board_required() {
    let (mut spatial, mut global) = input(3);
    assert!(input_label(&spatial, &global, 3).is_ok());
    assert!(input_label(&spatial[..spatial.len() - 1], &global, 3).is_err());
    assert!(input_label(&spatial, &global, 0).is_err());
    assert!(input_label(&spatial, &global, usize::MAX).is_err());
    global[0] = f32::INFINITY;
    assert!(input_label(&spatial, &global, 3).is_err());
    global[0] = 0.0;
    spatial[22 * 361] = 0.0;
    assert!(input_label(&spatial, &global, 3).is_err());
}

#[test]
fn tensor_hash_binds_order_signed_zero_and_physical_batch() {
    let (spatial, mut global) = input(1);
    let original = input_label(&spatial, &global, 1).unwrap();
    global[0] = -0.0;
    assert_ne!(input_label(&spatial, &global, 1).unwrap(), original);
    global[0] = 1.0;
    let first = input_label(&spatial, &global, 1).unwrap();
    global.swap(0, 1);
    assert_ne!(input_label(&spatial, &global, 1).unwrap(), first);
}

#[test]
fn all_five_full_heads_must_be_finite_and_exact_shape() {
    for index in 0..5 {
        let mut value = heads(3);
        let arrays = [
            &mut value.policy,
            &mut value.value,
            &mut value.misc,
            &mut value.moremisc,
            &mut value.ownership,
        ];
        *arrays.into_iter().nth(index).unwrap().last_mut().unwrap() = f32::NAN;
        assert!(value.validate(3).is_err());
    }
    let mut value = heads(1);
    assert!(value.validate(1).is_ok());
    value.policy.pop();
    assert!(value.validate(1).is_err());
}

#[test]
fn bit_gate_keeps_signed_zero_and_checks_all_raw_channels() {
    let a = heads(1);
    let mut b = heads(1);
    assert!(a.bitwise_eq(&b));
    b.policy[2171] = -0.0;
    assert!(!a.bitwise_eq(&b));
    assert_ne!(a.hashes(), b.hashes());
    b.policy[2171] = 0.0;
    b.moremisc[7] = f32::from_bits(1);
    assert!(!a.bitwise_eq(&b));
}

#[test]
fn fixed_policy_is_exact_probe_strategy_without_mutating_environment() {
    let env = expected_environment();
    assert_eq!(env.len(), 13);
    assert_eq!(env["KATAGO_CUDA_DUALFFN"], "0");
    assert_eq!(env["KATAGO_CUDA_INT8_GEMM_TUNE"], "0");
    assert_eq!(env["KATAGO_CUDA_CUBLASLT_RANK"], "heuristic");
    assert_eq!(env["KATAGO_CUDA_NOGRAPH"], "1");
}

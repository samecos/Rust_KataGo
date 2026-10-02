use super::*;

const MODEL: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
// Internal synthetic contract only. This is not a model-parser/source-binding test.
// Production cannot construct this private struct except from actual model bytes.
fn fixture_contract() -> NativeModelOutputContract {
    let graph = "1".repeat(64);
    let pp = ModelPostProcessParams::default();
    NativeModelOutputContract {
        model_sha256: MODEL.into(),
        graph_sha256: graph.clone(),
        model_version: 17,
        postprocess: pp,
        contract_sha256: contract_digest(MODEL, &graph, 17, pp),
    }
}
fn request() -> wire::EvalRequest {
    wire::EvalRequest {
        model_sha256: MODEL.into(),
        position: Some(wire::Position {
            board_size: 19,
            komi: 6.5,
            rules: "chinese".into(),
            initial_player: 1,
            next_player: 1,
            ..Default::default()
        }),
        parameters: Some(wire::EvalParameters {
            policy_temperature: 1.0,
            draw_equivalent_wins_for_white: 0.5,
            max_history: 10000,
            skip_cache: true,
            include_ownership: true,
            ..Default::default()
        }),
        ..Default::default()
    }
}
fn feature(request: &wire::EvalRequest) -> String {
    let row = super::super::input_encoding::encode_request_v7_cpu(request, MODEL).unwrap();
    super::super::input_encoding::row_feature_sha256(&row.spatial, &row.global).unwrap()
}
fn prepare(
    contract: &NativeModelOutputContract,
    request: &wire::EvalRequest,
    expected_feature: &str,
) -> Result<PreparedOutputRequest, EvalFailure> {
    let pb = request.encode_to_vec();
    contract.prepare_request_pb(&pb, &digest(&pb), expected_feature)
}
fn heads() -> [Vec<f32>; 5] {
    [
        vec![0.0; 6 * 362],
        vec![2.0, 0.0, -1.0],
        vec![0.0; 10],
        vec![0.0; 8],
        vec![0.25; 361],
    ]
}
fn raw(data: &[Vec<f32>; 5]) -> RawHeads<'_> {
    RawHeads {
        policy: &data[0],
        value: &data[1],
        misc: &data[2],
        moremisc: &data[3],
        ownership: &data[4],
    }
}

#[test]
fn actual_byte_constructor_rejects_mismatch_and_unparseable_model() {
    let bytes = b"this is not a native model";
    assert!(NativeModelOutputContract::from_native_model_bytes(bytes, MODEL, true, false).is_err());
    assert!(
        NativeModelOutputContract::from_native_model_bytes(bytes, &digest(bytes), true, false)
            .is_err()
    );
}

#[test]
fn public_pb_boundary_rejects_model_feature_and_illegal_history_substitution() {
    let contract = fixture_contract();
    let request = request();
    let expected = feature(&request);
    assert!(prepare(&contract, &request, &expected).is_ok());
    assert!(
        contract
            .prepare_request_pb(&request.encode_to_vec(), &"3".repeat(64), &expected)
            .is_err()
    );
    let mut wrong = request.clone();
    wrong.model_sha256 = "2".repeat(64);
    assert!(prepare(&contract, &wrong, &expected).is_err());
    let mut changed = request.clone();
    changed.position.as_mut().unwrap().komi = 7.5;
    assert!(prepare(&contract, &changed, &expected).is_err());
    let mut illegal = request.clone();
    illegal.position.as_mut().unwrap().moves = vec![wire::Move {
        color: 2,
        vertex: 0,
    }];
    assert!(prepare(&contract, &illegal, &expected).is_err());
    let mut unsupported = request;
    unsupported
        .parameters
        .as_mut()
        .unwrap()
        .always_compute_pass_alive = true;
    assert!(prepare(&contract, &unsupported, &expected).is_err());
}

#[test]
fn raw_pb_identity_is_distinct_even_when_feature_semantics_match() {
    let contract = fixture_contract();
    let mut a = request();
    a.task_id = 1;
    let mut b = a.clone();
    b.task_id = 2;
    let expected = feature(&a);
    let pa = prepare(&contract, &a, &expected).unwrap();
    let pb = prepare(&contract, &b, &expected).unwrap();
    assert_eq!(pa.row_feature_sha256(), pb.row_feature_sha256());
    assert_ne!(pa.request_pb_sha256(), pb.request_pb_sha256());
    assert_eq!(pa.request_pb_sha256(), digest(&a.encode_to_vec()));
}

#[test]
fn pb_output_preserves_white_perspective_and_ownership_request() {
    let contract = fixture_contract();
    let black = request();
    let mut white = black.clone();
    white.position.as_mut().unwrap().initial_player = 2;
    white.position.as_mut().unwrap().next_player = 2;
    let b = prepare(&contract, &black, &feature(&black)).unwrap();
    let w = prepare(&contract, &white, &feature(&white)).unwrap();
    let values = heads();
    let bout = contract
        .postprocess_raw_row(&b, raw(&values), 1, 0)
        .unwrap();
    let wout = contract
        .postprocess_raw_row(&w, raw(&values), 1, 0)
        .unwrap();
    assert_eq!(
        bout.white_win_prob.to_bits(),
        wout.white_loss_prob.to_bits()
    );
    assert_eq!(
        bout.white_no_result_prob.to_bits(),
        wout.white_no_result_prob.to_bits()
    );
    assert!(wout.white_no_result_prob > 0.0); // Chinese Simple does not suppress this class.
    assert_eq!(bout.ownership.len(), 361);
    assert_eq!(wout.ownership.len(), 361);
    assert_eq!(bout.ownership[0].to_bits(), (-wout.ownership[0]).to_bits());
    let mut no_owner = black;
    no_owner.parameters.as_mut().unwrap().include_ownership = false;
    let n = prepare(&contract, &no_owner, &feature(&no_owner)).unwrap();
    assert!(
        contract
            .postprocess_raw_row(&n, raw(&values), 1, 0)
            .unwrap()
            .ownership
            .is_empty()
    );
}

#[test]
fn prepared_request_cannot_cross_model_contract_or_accept_nonfinite_unused_head() {
    let contract = fixture_contract();
    let request = request();
    let prepared = prepare(&contract, &request, &feature(&request)).unwrap();
    let mut other = fixture_contract();
    other.postprocess.output_scale_multiplier = 2.0;
    other.contract_sha256 = contract_digest(
        &other.model_sha256,
        &other.graph_sha256,
        other.model_version,
        other.postprocess,
    );
    let mut data = heads();
    assert!(
        other
            .postprocess_raw_row(&prepared, raw(&data), 1, 0)
            .is_err()
    );
    data[3][7] = f32::NAN; // unused by PB mapping; still forbidden in full raw evidence.
    assert!(
        contract
            .postprocess_raw_row(&prepared, raw(&data), 1, 0)
            .is_err()
    );
}

#[test]
fn komi_and_setup_handicap_are_input_semantics_not_added_again_to_output_score() {
    let contract = fixture_contract();
    let mut plain = request();
    plain.position.as_mut().unwrap().initial_player = 2;
    plain.position.as_mut().unwrap().next_player = 2;
    let mut setup = plain.clone();
    setup.position.as_mut().unwrap().initial_stones = vec![
        wire::Stone {
            color: 1,
            vertex: 72,
        },
        wire::Stone {
            color: 1,
            vertex: 288,
        },
    ];
    let mut different_komi = setup.clone();
    different_komi.position.as_mut().unwrap().komi = 7.5;
    let requests = [plain, setup, different_komi];
    let features: Vec<_> = requests.iter().map(feature).collect();
    assert_ne!(features[0], features[1]);
    assert_ne!(features[1], features[2]);
    let mut data = heads();
    data[2][0] = 0.4;
    data[2][2] = 0.7;
    let outputs: Vec<_> = requests
        .iter()
        .zip(&features)
        .map(|(request, feature)| {
            let prepared = prepare(&contract, request, feature).unwrap();
            contract
                .postprocess_raw_row(&prepared, raw(&data), 1, 0)
                .unwrap()
        })
        .collect();
    for output in &outputs[1..] {
        assert_eq!(
            output.white_score_mean.to_bits(),
            outputs[0].white_score_mean.to_bits()
        );
        assert_eq!(
            output.white_score_mean_sq.to_bits(),
            outputs[0].white_score_mean_sq.to_bits()
        );
        assert_eq!(output.white_lead.to_bits(), outputs[0].white_lead.to_bits());
        assert_eq!(
            output.white_win_prob.to_bits(),
            outputs[0].white_win_prob.to_bits()
        );
    }
    // History remains relevant to legal masking even though no score offset is added.
    assert!(outputs[0].policy[72] >= 0.0);
    assert_eq!(outputs[1].policy[72], -1.0);
}

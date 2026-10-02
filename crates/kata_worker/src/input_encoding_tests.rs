use super::*;

const MODEL: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
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
fn encode(r: &wire::EvalRequest) -> EncodedWorkerRow {
    encode_request_v7_cpu(r, MODEL).unwrap()
}
fn played() -> wire::EvalRequest {
    let mut r = request();
    r.position.as_mut().unwrap().moves = vec![
        wire::Move {
            color: 1,
            vertex: 21,
        },
        wire::Move {
            color: 2,
            vertex: 300,
        },
    ];
    r
}

#[test]
fn nondefault_komi_and_player_perspective_are_encoded() {
    let black = encode(&request());
    assert_eq!(black.global[5].to_bits(), (-6.5f32 / 20.0).to_bits());
    assert!(black.spatial[..361].iter().all(|&x| x == 1.0));
    let mut r = request();
    let p = r.position.as_mut().unwrap();
    p.initial_player = 2;
    p.next_player = 2;
    let white = encode(&r);
    assert_eq!(white.global[5].to_bits(), (6.5f32 / 20.0).to_bits());
}

#[test]
fn simultaneous_setup_retains_explicit_chinese_handicap_bonus() {
    let mut r = request();
    let p = r.position.as_mut().unwrap();
    p.initial_player = 2;
    p.next_player = 2;
    p.initial_stones = vec![
        wire::Stone {
            color: 1,
            vertex: 72,
        },
        wire::Stone {
            color: 1,
            vertex: 288,
        },
    ];
    let first = encode(&r);
    assert_eq!(first.evidence.replayed_moves, 0);
    assert_eq!(first.pre_spatial[2 * 361 + 72], 1.0); // opponent of white
    // Chinese rules award N points for explicit initial handicap stones.
    // assume_multiple_starting_black_moves_are_handicap(false) only disables
    // inferring extra stones from opening moves; it does not erase setup.
    let rules: serde_json::Value = serde_json::from_str(&first.evidence.rules_json).unwrap();
    assert_eq!(rules["whiteHandicapBonus"], "N");
    assert_eq!(rules["komi"], 6.5);
    assert_eq!(first.global[5].to_bits(), (8.5f32 / 20.0).to_bits());
    r.position.as_mut().unwrap().initial_stones.reverse();
    assert_eq!(f32le(&first.pre_spatial), f32le(&encode(&r).pre_spatial));
}

#[test]
fn identical_final_board_does_not_erase_move_history() {
    let full = encode(&played());
    let mut setup = request();
    setup.position.as_mut().unwrap().initial_stones = vec![
        wire::Stone {
            color: 1,
            vertex: 21,
        },
        wire::Stone {
            color: 2,
            vertex: 300,
        },
    ];
    let no_moves = encode(&setup);
    assert_eq!(
        full.evidence.board_colors_yx,
        no_moves.evidence.board_colors_yx
    );
    assert_ne!(
        row_feature_sha256(&full.spatial, &full.global).unwrap(),
        row_feature_sha256(&no_moves.spatial, &no_moves.global).unwrap()
    );
    let mut limited = played();
    limited.parameters.as_mut().unwrap().max_history = 0;
    assert_ne!(
        f32le(&full.pre_spatial),
        f32le(&encode(&limited).pre_spatial)
    );
}

#[test]
fn horizontal_symmetry_is_one_transform_of_every_channel_and_no_global_transform() {
    let base = encode(&played());
    let mut r = played();
    r.parameters.as_mut().unwrap().symmetry = 2;
    let flipped = encode(&r);
    assert_eq!(f32le(&base.global), f32le(&flipped.global));
    for channel in 0..22 {
        for y in 0..19 {
            for x in 0..19 {
                assert_eq!(
                    flipped.spatial[channel * 361 + y * 19 + x].to_bits(),
                    base.pre_spatial[channel * 361 + y * 19 + (18 - x)].to_bits()
                );
            }
        }
    }
    assert_ne!(f32le(&base.spatial), f32le(&flipped.spatial));
}

#[test]
fn actual_worker_parameter_conversions_are_exposed() {
    let mut r = played();
    let p = r.parameters.as_mut().unwrap();
    p.policy_temperature = 1.00000007;
    p.policy_optimism = 0.4;
    p.draw_equivalent_wins_for_white = 0.25;
    p.playout_doubling_advantage = -1.5;
    p.conservative_pass = true;
    p.enable_passing_hacks = true;
    p.avoid_mytdagger_hack = true;
    p.max_history = 3;
    p.symmetry = 5;
    let e = encode(&r).evidence;
    assert_eq!(
        e.params.nn_policy_temperature.to_bits(),
        (1.00000007f64 as f32).to_bits()
    );
    assert_eq!(e.params.policy_optimism, 0.4);
    assert_eq!(e.params.draw_equivalent_wins_for_white, 0.25);
    assert_eq!(e.params.playout_doubling_advantage, -1.5);
    assert!(
        e.params.conservative_pass_and_is_root
            && e.params.enable_passing_hacks
            && e.params.avoid_mytdagger_hack
    );
    assert_eq!((e.params.max_history, e.params.symmetry), (3, 5));
}

#[test]
fn illegal_history_and_unsupported_modes_are_not_relaxed_by_export() {
    let mut r = played();
    r.position.as_mut().unwrap().moves[1].vertex = 21;
    assert_eq!(
        encode_request_v7_cpu(&r, MODEL).unwrap_err().code,
        "INVALID_CONTEXT"
    );
    let mut r = request();
    r.parameters.as_mut().unwrap().always_compute_pass_alive = true;
    assert!(encode_request_v7_cpu(&r, MODEL).is_err());
    let mut r = request();
    r.parameters.as_mut().unwrap().symmetry = -1;
    assert!(encode_request_v7_cpu(&r, MODEL).is_err());
    let mut r = request();
    r.position.as_mut().unwrap().rules = "japanese".into();
    assert!(encode_request_v7_cpu(&r, MODEL).is_err());
    assert!(encode_request_v7_cpu(&request(), &"f".repeat(64)).is_err());
}

#[test]
fn terminal_permissions_keep_the_workers_exact_friendly_pass_boundary() {
    let mut r = request();
    r.position.as_mut().unwrap().moves = vec![
        wire::Move {
            color: 1,
            vertex: -1,
        },
        wire::Move {
            color: 2,
            vertex: -1,
        },
    ];
    assert!(encode_request_v7_cpu(&r, MODEL).is_err());
    r.parameters.as_mut().unwrap().force_non_terminal = true;
    assert!(encode(&r).evidence.is_game_finished);
    let mut r = played();
    r.parameters.as_mut().unwrap().force_non_terminal = true;
    assert!(encode_request_v7_cpu(&r, MODEL).is_err());
}

#[test]
fn f32le_and_feature_identity_preserve_bits_and_reject_nonfinite() {
    assert_eq!(f32le(&[1.0, -0.0]), vec![0, 0, 128, 63, 0, 0, 0, 128]);
    let row = encode(&request());
    let mut changed = row.global.clone();
    changed[0] = -0.0;
    assert_ne!(
        row_feature_sha256(&row.spatial, &row.global).unwrap(),
        row_feature_sha256(&row.spatial, &changed).unwrap()
    );
    changed[0] = f32::NAN;
    assert!(row_feature_sha256(&row.spatial, &changed).is_err());
    assert!(row_feature_sha256(&row.spatial[..10], &row.global).is_err());
}

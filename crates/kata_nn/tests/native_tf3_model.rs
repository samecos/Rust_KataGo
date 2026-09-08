//! Structural and byte-identity gate for the exact model used by Go Server's
//! C++ worker. Set KATAGO_TEST_MODEL_DIR to the directory containing the model.

use kata_nn::desc::BlockDesc;
use kata_nn::model_parser::load_model_from_bytes;

const MODEL: &str = "kata1-tf3-b11c768-s11001M-d5973M.bin.gz";
const SHA256: &str = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6";

fn check_transformers(blocks: &[BlockDesc], counts: &mut (usize, usize)) {
    for block in blocks {
        match block {
            BlockDesc::NestedBottleneck(b) => check_transformers(&b.blocks, counts),
            BlockDesc::TransformerAttention(a) => {
                counts.0 += 1;
                assert_eq!(
                    (a.num_heads, a.num_kv_heads, a.q_head_dim, a.v_head_dim),
                    (12, 12, 32, 32)
                );
                assert_eq!(a.pre_ln.num_channels, 384);
                assert_eq!(a.pre_ln.epsilon, 1e-6);
                assert_eq!(a.pre_ln.weight.len(), 384);
                assert!(a.use_rope && a.learnable_rope);
                assert_eq!((a.rope_num_kv_heads, a.rope_num_pairs), (12, 16));
                assert_eq!(a.rope_freqs.len(), 12 * 16 * 2);
                for projection in [&a.q_proj, &a.k_proj, &a.v_proj, &a.out_proj] {
                    assert_eq!(
                        (projection.in_channels, projection.out_channels),
                        (384, 384)
                    );
                    assert_eq!(projection.weights.len(), 384 * 384);
                }
            }
            BlockDesc::TransformerFfn(f) => {
                counts.1 += 1;
                assert_eq!((f.num_channels, f.ffn_channels), (384, 1152));
                assert!(f.use_swi_glu);
                assert_eq!(f.pre_ln.epsilon, 1e-6);
                assert_eq!(f.linear1.weights.len(), 384 * 1152);
                assert_eq!(f.linear_gate.weights.len(), 384 * 1152);
                assert_eq!(f.linear2.weights.len(), 384 * 1152);
            }
            _ => panic!("TF3 fixture unexpectedly contains a convolution residual block"),
        }
    }
}

#[test]
fn exact_server_tf3_native_bytes_preserve_structure_and_identity() {
    let Some(directory) = std::env::var_os("KATAGO_TEST_MODEL_DIR") else {
        eprintln!("skipped: set KATAGO_TEST_MODEL_DIR for the native TF3 model gate");
        return;
    };
    let path = std::path::PathBuf::from(directory).join(MODEL);
    if !path.is_file() {
        eprintln!("skipped: {} is absent", path.display());
        return;
    }
    let bytes = std::fs::read(path).unwrap();
    let desc = load_model_from_bytes(&bytes, true, true).expect("native TF3 parsing");
    assert_eq!(desc.sha256, SHA256);
    assert_eq!(desc.name, "b11c768h12nbt3tflrs-fson-silu");
    assert_eq!(desc.model_version, 17);
    assert_eq!(
        (
            desc.num_input_channels,
            desc.num_input_global_channels,
            desc.num_input_meta_channels
        ),
        (22, 19, 0)
    );
    assert_eq!(
        (
            desc.num_policy_channels,
            desc.num_value_channels,
            desc.num_score_value_channels,
            desc.num_ownership_channels
        ),
        (2, 3, 6, 1)
    );
    assert!(!desc.prefer_pass_alive_under_suicide_rules);
    assert!(!desc.prefer_exclude_territory_adjacent_to_atari);
    assert_eq!(
        desc.post_process_params.shortterm_score_error_multiplier,
        150.0
    );
    assert_eq!(
        (
            desc.trunk.num_blocks,
            desc.trunk.trunk_num_channels,
            desc.trunk.mid_num_channels
        ),
        (11, 768, 384)
    );
    let mut counts = (0, 0);
    check_transformers(&desc.trunk.blocks, &mut counts);
    assert_eq!(counts, (33, 33));
    assert_eq!(desc.get_num_parameters(), 70_442_025);
    // Frozen independently from the native Y,X,IC,OC float block using the C++
    // descriptor's documented OC,IC,Y,X indexing. This catches silent reorder bugs.
    assert_eq!(
        desc.trunk.initial_conv.weights[..8]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        [
            3152067142, 1006837984, 3150926860, 3158047405, 1002249266, 3157690016, 3165158567,
            990746256
        ]
    );
    for norm in [
        &desc.policy_head.g1_bn,
        &desc.policy_head.p1_bn,
        &desc.value_head.v1_bn,
    ] {
        assert!(norm.merged_scale.iter().all(|scale| *scale == 1.0));
        assert!(norm.merged_bias.iter().any(|bias| *bias != 0.0));
    }
    assert_eq!(
        desc.trunk.trunk_tip_activation.activation,
        kata_nn::activations::ACTIVATION_SILU
    );
    eprintln!(
        "{} v{}: {}, policy={}, score={}, SHA256={}",
        desc.name,
        desc.model_version,
        desc.get_short_info_string(),
        desc.num_policy_channels,
        desc.num_score_value_channels,
        desc.sha256
    );
}

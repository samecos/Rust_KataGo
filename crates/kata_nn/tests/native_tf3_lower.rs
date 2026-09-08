//! Structural and weight-identity regression for the deployment's native TF3.
//! GPU numeric agreement is checked separately against the C++ Worker oracle.

use kata_nn::desc::BlockDesc;
use kata_nn::native_model::lower_model;
use kata_nn::onnx_parser::Layer;

const FILE: &str = "kata1-tf3-b11c768-s11001M-d5973M.bin.gz";
const SHA256: &str = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6";

#[test]
fn native_tf3_keeps_model_identity_topology_and_head_weight_semantics() {
    let Some(directory) = std::env::var_os("KATAGO_TEST_MODEL_DIR") else {
        eprintln!("skipping native TF3 lower test: KATAGO_TEST_MODEL_DIR is unset");
        return;
    };
    let path = std::path::PathBuf::from(directory).join(FILE);
    if !path.is_file() {
        eprintln!(
            "skipping native TF3 lower test: {} is absent",
            path.display()
        );
        return;
    }
    let bytes = std::fs::read(&path).expect("read native TF3 fixture");
    let desc = kata_nn::model_parser::load_model_from_bytes(&bytes, true, true)
        .expect("parse native TF3 fixture");
    assert_eq!(desc.sha256, SHA256);
    assert_eq!(desc.name, "b11c768h12nbt3tflrs-fson-silu");
    assert_eq!(desc.model_version, 17);
    assert_eq!(
        (desc.num_policy_channels, desc.num_score_value_channels),
        (2, 6)
    );
    let pp = desc.post_process_params;
    for scale in [
        pp.td_score_multiplier,
        pp.score_mean_multiplier,
        pp.score_stdev_multiplier,
        pp.lead_multiplier,
        pp.variance_time_multiplier,
        pp.shortterm_value_error_multiplier,
        pp.shortterm_score_error_multiplier,
    ] {
        assert!(scale.is_finite() && scale > 0.0);
    }
    assert!(pp.output_scale_multiplier.is_finite() && pp.output_scale_multiplier > 0.0);
    assert_eq!(pp.shortterm_score_error_multiplier, 150.0);
    assert_eq!(pp.shortterm_value_error_multiplier, 0.25);
    assert_eq!(pp.output_scale_multiplier, 1.0);
    assert!(!desc.prefer_pass_alive_under_suicide_rules);
    assert!(!desc.prefer_exclude_territory_adjacent_to_atari);
    let graph = lower_model(&desc).expect("lower supported native TF3");
    assert_eq!(
        desc.sha256, SHA256,
        "lowering must not substitute model identity"
    );
    assert_eq!(
        (graph.board_size, graph.trunk_channels, graph.mid_channels),
        (19, 768, 384)
    );
    assert_eq!(
        (graph.num_blocks, graph.num_heads, graph.head_dim),
        (11, 12, 32)
    );
    assert_eq!(graph.layers.len(), 179);
    assert_eq!(graph.num_attention_layers(), 33);
    assert_eq!(graph.num_ffn_layers(), 33);
    assert_eq!(graph.num_rmsnorm_layers(), 66);
    assert_eq!(graph.total_params, graph.layer_param_elts());

    let Layer::InitialConv(initial) = &graph.layers[0] else {
        panic!("initial convolution missing");
    };
    assert_eq!(initial.weight.dims, [768, 22, 3, 3]);
    assert_eq!(initial.weight.f32_data(), desc.trunk.initial_conv.weights);
    let BlockDesc::NestedBottleneck(first) = &desc.trunk.blocks[0] else {
        panic!("nested block missing");
    };
    assert_eq!(initial.gate_scale.f32_data(), first.pre_bn.merged_scale);
    assert_eq!(initial.gate_bias.f32_data(), first.pre_bn.merged_bias);
    let Layer::RmsNorm(rms) = &graph.layers[2] else {
        panic!("first RMSNorm missing");
    };
    let BlockDesc::TransformerAttention(attn) = &first.blocks[0] else {
        panic!("first attention missing");
    };
    assert_eq!(rms.eps, attn.pre_ln.epsilon);
    assert_eq!(rms.scale.f32_data(), attn.pre_ln.weight);
    let Layer::Ffn(ffn) = &graph.layers[5] else {
        panic!("first FFN missing");
    };
    let BlockDesc::TransformerFfn(native_ffn) = &first.blocks[1] else {
        panic!("native FFN missing");
    };
    // linear1 is the SiLU branch; linear_gate is the unactivated multiplier.
    for (input, output) in [(0, 0), (17, 42), (383, 1151)] {
        assert_eq!(
            ffn.gate_weight.f32_data()[output * 384 + input],
            native_ffn.linear1.weights[input * 1152 + output]
        );
        assert_eq!(
            ffn.up_weight.f32_data()[output * 384 + input],
            native_ffn.linear_gate.weights[input * 1152 + output]
        );
    }

    let Layer::PolicyHead(policy) = &graph.layers[177] else {
        panic!("policy head missing");
    };
    assert_eq!(
        &policy.conv2p_weight.f32_data()[..96],
        &desc.policy_head.p2_conv.weights[..96]
    );
    assert_eq!(
        &policy.conv2p_weight.f32_data()[5 * 96..],
        &desc.policy_head.p2_conv.weights[96..]
    );
    assert!(
        policy.conv2p_weight.f32_data()[96..5 * 96]
            .iter()
            .all(|&v| v == 0.0)
    );
    for input in 0..96 {
        assert_eq!(
            policy.pass_matmul2.f32_data()[input],
            desc.policy_head.gpool_to_pass_mul2.weights[input * 2]
        );
        assert_eq!(
            policy.pass_matmul2.f32_data()[5 * 96 + input],
            desc.policy_head.gpool_to_pass_mul2.weights[input * 2 + 1]
        );
    }
    let Layer::ValueHead(value) = &graph.layers[178] else {
        panic!("value head missing");
    };
    assert_eq!(
        &value.misc_bias.f32_data()[..4],
        &desc.value_head.sv3_bias.weights[..4]
    );
    assert_eq!(
        &value.moremisc_bias.f32_data()[..2],
        &desc.value_head.sv3_bias.weights[4..]
    );
    for input in [0, 87, 191] {
        for output in 0..4 {
            assert_eq!(
                value.misc_matmul.f32_data()[output * 192 + input],
                desc.value_head.sv3_mul.weights[input * 6 + output]
            );
        }
        for output in 0..2 {
            assert_eq!(
                value.moremisc_matmul.f32_data()[output * 192 + input],
                desc.value_head.sv3_mul.weights[input * 6 + output + 4]
            );
        }
    }
}

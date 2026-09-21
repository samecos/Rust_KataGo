//! Exact pruned fixture: topology, per-layer widths and weight layout.
use kata_nn::{desc::BlockDesc, native_model::lower_model, onnx_parser::Layer};

#[test]
fn pruned_native_preserves_variable_ffn_widths_and_rejects_invalid_shapes() {
    let Some(dir) = std::env::var_os("KATAGO_TEST_MODEL_DIR") else {
        return;
    };
    let path = std::path::PathBuf::from(dir).join("b15-ffn-pruned-a8.bin.gz");
    if !path.is_file() {
        eprintln!("skipped: {} is absent", path.display());
        return;
    }
    let mut desc =
        kata_nn::model_parser::load_model_from_bytes(&std::fs::read(path).unwrap(), true, true)
            .unwrap();
    assert_eq!(
        desc.sha256,
        "3f216ee88226ee49ca826eaa5f7e1e9982ff48a96625914bfc4e5ad20ee095a7"
    );
    let graph = lower_model(&desc).unwrap();
    assert_eq!(
        (
            graph.num_blocks,
            graph.trunk_channels,
            graph.mid_channels,
            graph.num_heads
        ),
        (15, 1024, 512, 16)
    );
    assert_eq!(graph.num_attention_layers(), 45);
    let widths: Vec<_> = graph
        .layers
        .iter()
        .filter_map(|l| match l {
            Layer::Ffn(f) => Some(f.hidden),
            _ => None,
        })
        .collect();
    assert_eq!(
        widths,
        [
            240, 48, 472, 72, 16, 16, 48, 48, 568, 16, 64, 72, 960, 368, 392, 48, 104, 88, 16, 24,
            40, 48, 160, 184, 304, 400, 656, 184, 256, 288, 1088, 992, 1160, 40, 96, 88, 112, 96,
            24, 264, 296, 304, 424, 760, 728
        ]
    );
    let native = desc
        .trunk
        .blocks
        .iter()
        .flat_map(|b| match b {
            BlockDesc::NestedBottleneck(b) => &b.blocks,
            _ => panic!(),
        })
        .filter_map(|b| match b {
            BlockDesc::TransformerFfn(f) => Some(f),
            _ => None,
        });
    let lowered = graph.layers.iter().filter_map(|l| match l {
        Layer::Ffn(f) => Some(f),
        _ => None,
    });
    for (a, b) in native.zip(lowered) {
        assert_eq!(b.down_weight.dims, [512, b.hidden as i64]);
        for (input, output) in [(0, 0), (511, b.hidden - 1), (17, b.hidden / 2)] {
            assert_eq!(
                b.gate_weight.f32_data()[output * 512 + input],
                a.linear1.weights[input * b.hidden + output]
            );
            assert_eq!(
                b.up_weight.f32_data()[output * 512 + input],
                a.linear_gate.weights[input * b.hidden + output]
            );
            assert_eq!(
                b.down_weight.f32_data()[input * b.hidden + output],
                a.linear2.weights[output * 512 + input]
            );
        }
    }
    for layer in &graph.layers {
        if let Layer::Attention(a) = layer {
            assert_eq!(a.qkv_weight.dims, [1536, 512]);
            assert_eq!(a.rope_cos.dims, [361, 256]);
        }
    }
    drop(graph);
    desc.trunk.mid_num_channels = 504;
    assert!(lower_model(&desc).is_err());
    desc.trunk.mid_num_channels = 512;
    if let BlockDesc::NestedBottleneck(b) = &mut desc.trunk.blocks[0] {
        b.num_blocks = 5;
    }
    assert!(
        lower_model(&desc)
            .err()
            .expect("must reject invalid shape")
            .contains("complete attention/FFN pairs")
    );
    if let BlockDesc::NestedBottleneck(b) = &mut desc.trunk.blocks[0] {
        b.num_blocks = 6;
        if let BlockDesc::TransformerFfn(f) = &mut b.blocks[1] {
            f.ffn_channels = 239;
        }
    }
    assert!(
        lower_model(&desc)
            .err()
            .expect("must reject invalid shape")
            .contains("8-aligned")
    );
}

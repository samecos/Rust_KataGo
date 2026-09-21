//! Inspect native model dimensions without creating a CUDA context.
use kata_nn::desc::BlockDesc;
use serde_json::json;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: native_model_info MODEL.bin.gz");
    let bytes = std::fs::read(path).expect("read model");
    let model = kata_nn::model_parser::load_model_from_bytes(&bytes, true, true)
        .expect("parse native model");
    let mut attention = Vec::new();
    let mut ffn = Vec::new();
    for block in &model.trunk.blocks {
        if let BlockDesc::NestedBottleneck(block) = block {
            for inner in &block.blocks {
                match inner {
                    BlockDesc::TransformerAttention(a) => attention.push(json!({"heads":a.num_heads,"kv_heads":a.num_kv_heads,"q_dim":a.q_head_dim,"v_dim":a.v_head_dim,"rope":a.use_rope,"learned_rope":a.learnable_rope})),
                    BlockDesc::TransformerFfn(f) => ffn.push(json!({"channels":f.num_channels,"hidden":f.ffn_channels,"swiglu":f.use_swi_glu})),
                    _ => (),
                }
            }
        }
    }
    let head_unit_scales = [
        &model.policy_head.p1_bn,
        &model.policy_head.g1_bn,
        &model.value_head.v1_bn,
    ]
    .map(|n| n.merged_scale.iter().all(|&s| s == 1.0));
    println!("{}", serde_json::to_string_pretty(&json!({"name":model.name,"sha256":model.sha256,
        "version":model.model_version,"blocks":model.trunk.num_blocks,"trunk":model.trunk.trunk_num_channels,
        "mid":model.trunk.mid_num_channels,"attention":attention,"ffn":ffn,
        "policy":{"p":model.policy_head.p1_conv.out_channels,"g":model.policy_head.g1_conv.out_channels,"pass":model.policy_head.gpool_to_pass_mul.out_channels},
        "value":{"v1":model.value_head.v1_conv.out_channels,"v2":model.value_head.v2_mul.out_channels},
        "head_unit_scales":head_unit_scales,
        "lowering":kata_nn::native_model::lower_model(&model).map(|g|json!({"layers":g.layers.len(),"params":g.total_params})),
    })).unwrap());
}

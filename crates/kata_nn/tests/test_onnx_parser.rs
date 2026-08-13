//! `onnx_parser` 集成测试：用真实模型 D:/code/b11fix.onnx（可用环境变量
//! `KATAGO_ONNX_MODEL` 覆盖）验证层图解析结果。
//!
//! 模型缺失时自动跳过（打印 skipped）。

use std::path::Path;

use kata_nn::onnx_parser::{Layer, LayerGraph, TensorData, parse_layer_graph};

fn model_path() -> Option<String> {
    std::env::var("KATAGO_ONNX_MODEL")
        .ok()
        .or_else(|| Some("D:/code/b11fix.onnx".to_string()))
        .filter(|p| Path::new(p).exists())
}

fn dims(t: &kata_nn::onnx_parser::Tensor) -> Vec<i64> {
    t.dims.clone()
}

fn layer_name(l: &Layer) -> String {
    match l {
        Layer::InitialConv(x) => format!(
            "InitialConv(weight {:?}, out={})",
            dims(&x.weight),
            x.out_channels
        ),
        Layer::Linear(x) => format!("Linear({:?} bias={})", dims(&x.weight), x.bias.is_some()),
        Layer::RmsNorm(x) => format!("RmsNorm(channels={})", x.channels),
        Layer::Attention(x) => format!(
            "Attention(h={} d={} qkv {:?} out {:?} rope {:?})",
            x.num_heads,
            x.head_dim,
            dims(&x.qkv_weight),
            dims(&x.out_weight),
            dims(&x.rope_cos)
        ),
        Layer::Ffn(x) => format!("FFN(hidden={} up {:?} down {:?})", x.hidden, dims(&x.up_weight), dims(&x.down_weight)),
        Layer::GateSilu(x) => format!("GateSilu(channels={})", x.channels),
        Layer::TrunkFinal(x) => format!("TrunkFinal(channels={})", x.channels),
        Layer::PolicyHead(x) => format!(
            "PolicyHead(conv1p {:?} conv2p {:?} mask_scale={})",
            dims(&x.conv1p_weight),
            dims(&x.conv2p_weight),
            x.mask_scale
        ),
        Layer::ValueHead(x) => format!(
            "ValueHead(conv1 {:?} value {:?} misc {:?} moremisc {:?} ownership {:?} mask_scale={} mask_quad={})",
            dims(&x.conv1_weight),
            dims(&x.value_matmul),
            dims(&x.misc_matmul),
            dims(&x.moremisc_matmul),
            dims(&x.ownership_conv),
            x.mask_scale,
            x.mask_quad
        ),
    }
}

/// 打印层清单供人工审阅。
fn print_layer_list(g: &LayerGraph) {
    println!("== 层清单（共 {} 层）==", g.layers.len());
    let mut idx = 0usize;
    for l in &g.layers {
        let extra = match l {
            Layer::InitialConv(x) => format!(" params={}", weight_elts_initial(x)),
            Layer::Linear(x) => format!(" params={}", x.weight.numel()),
            Layer::RmsNorm(x) => format!(" params={}", x.scale.numel()),
            Layer::Attention(x) => format!(
                " params={}",
                x.qkv_weight.numel() + x.out_weight.numel() + x.rope_cos.numel() + x.rope_sin.numel()
            ),
            Layer::Ffn(x) => {
                format!(" params={}", x.up_weight.numel() + x.gate_weight.numel() + x.down_weight.numel())
            }
            Layer::GateSilu(x) => format!(" params={}", x.scale.numel() + x.bias.numel()),
            Layer::TrunkFinal(x) => {
                format!(" params={}", x.mean.numel() + x.std.numel() + x.gamma.numel() + x.beta.numel())
            }
            Layer::PolicyHead(_) => " params=见统计".to_string(),
            Layer::ValueHead(_) => " params=见统计".to_string(),
        };
        println!("  [{idx:3}] {}{}", layer_name(l), extra);
        idx += 1;
    }
    println!(
        "== 汇总: {} 层, attention={}, ffn={}, rmsnorm={}, params={} ==",
        g.layers.len(),
        g.num_attention_layers(),
        g.num_ffn_layers(),
        g.num_rmsnorm_layers(),
        g.total_params
    );
}

fn weight_elts_initial(x: &kata_nn::onnx_parser::InitialConvLayer) -> usize {
    x.weight.numel() + x.global_weight.numel() + x.gate_scale.numel() + x.gate_bias.numel()
}

fn assert_all_f32(g: &LayerGraph) {
    fn check(t: &kata_nn::onnx_parser::Tensor, what: &str) {
        assert!(
            matches!(t.data, TensorData::F32(_)),
            "{what} 权重不是 f32（{:?}）",
            t.dims
        );
    }
    for l in &g.layers {
        match l {
            Layer::InitialConv(x) => {
                check(&x.weight, "InitialConv.weight");
                check(&x.global_weight, "InitialConv.global_weight");
                check(&x.gate_scale, "InitialConv.gate_scale");
                check(&x.gate_bias, "InitialConv.gate_bias");
            }
            Layer::Linear(x) => {
                check(&x.weight, "Linear.weight");
                if let Some(b) = &x.bias {
                    check(b, "Linear.bias");
                }
            }
            Layer::RmsNorm(x) => check(&x.scale, "RmsNorm.scale"),
            Layer::Attention(x) => {
                check(&x.qkv_weight, "Attention.qkv_weight");
                check(&x.out_weight, "Attention.out_weight");
                check(&x.rope_cos, "Attention.rope_cos");
                check(&x.rope_sin, "Attention.rope_sin");
            }
            Layer::Ffn(x) => {
                check(&x.up_weight, "Ffn.up_weight");
                check(&x.gate_weight, "Ffn.gate_weight");
                check(&x.down_weight, "Ffn.down_weight");
            }
            Layer::GateSilu(x) => {
                check(&x.scale, "GateSilu.scale");
                check(&x.bias, "GateSilu.bias");
            }
            Layer::TrunkFinal(x) => {
                check(&x.mean, "TrunkFinal.mean");
                check(&x.std, "TrunkFinal.std");
                check(&x.gamma, "TrunkFinal.gamma");
                check(&x.beta, "TrunkFinal.beta");
            }
            Layer::PolicyHead(x) => {
                check(&x.conv1p_weight, "PolicyHead.conv1p_weight");
                check(&x.conv1g_weight, "PolicyHead.conv1g_weight");
                check(&x.g_bias, "PolicyHead.g_bias");
                check(&x.g_matmul, "PolicyHead.g_matmul");
                check(&x.pass_matmul1, "PolicyHead.pass_matmul1");
                check(&x.pass_bias1, "PolicyHead.pass_bias1");
                check(&x.pass_matmul2, "PolicyHead.pass_matmul2");
                check(&x.bias2, "PolicyHead.bias2");
                check(&x.conv2p_weight, "PolicyHead.conv2p_weight");
            }
            Layer::ValueHead(x) => {
                check(&x.conv1_weight, "ValueHead.conv1_weight");
                check(&x.bias1, "ValueHead.bias1");
                check(&x.linear2_weight, "ValueHead.linear2_weight");
                check(&x.linear2_bias, "ValueHead.linear2_bias");
                check(&x.value_matmul, "ValueHead.value_matmul");
                check(&x.value_bias, "ValueHead.value_bias");
                check(&x.misc_matmul, "ValueHead.misc_matmul");
                check(&x.misc_bias, "ValueHead.misc_bias");
                check(&x.moremisc_matmul, "ValueHead.moremisc_matmul");
                check(&x.moremisc_bias, "ValueHead.moremisc_bias");
                check(&x.ownership_conv, "ValueHead.ownership_conv");
            }
        }
    }
}

#[test]
fn parse_real_model_and_verify_layer_graph() {
    let Some(path) = model_path() else {
        println!("skipped: 找不到模型（可用 KATAGO_ONNX_MODEL 指定路径）");
        return;
    };
    println!("解析模型: {path}");
    let bytes = std::fs::read(&path).expect("读取模型失败");
    let g: LayerGraph = parse_layer_graph(&bytes).expect("层图解析失败");

    // 输入/输出
    assert_eq!(g.num_spatial_inputs, 22);
    assert_eq!(g.num_global_inputs, 19);
    assert_eq!(g.board_size, 19);
    assert_eq!(g.trunk_channels, 768);
    assert_eq!(g.mid_channels, 384);
    assert_eq!(g.num_blocks, 11);
    assert_eq!(g.num_heads, 12);
    assert_eq!(g.head_dim, 32);
    assert_eq!(
        g.output_names,
        ["out_policy", "out_value", "out_miscvalue", "out_moremiscvalue", "out_ownership"]
    );

    // 层数与结构：1 初始卷积 + 11 块×(1 下投影 + 6 RMSNorm + 3 attn + 3 FFN
    // + 1 块内门 + 1 上投影) + 10 个块间门 + 1 trunk 末端 + 2 头 = 179
    assert_eq!(g.layers.len(), 179, "总层数应为 179");
    assert_eq!(g.num_attention_layers(), 33);
    assert_eq!(g.num_ffn_layers(), 33);
    assert_eq!(g.num_rmsnorm_layers(), 66);

    // 参数量：层内权重元素总数 + 折叠的标量参数 == initializer 总数（75,008,575）
    assert_eq!(g.total_params, 75_008_575, "模型参数总量");
    assert_eq!(g.scalar_params, 10);
    assert_eq!(
        g.layer_param_elts() + g.scalar_params,
        g.total_params,
        "层内权重应覆盖全部 initializer 元素"
    );

    // 逐层形状抽查
    let mut attn_count = 0usize;
    let mut ffn_count = 0usize;
    let mut rms_count = 0usize;
    for l in &g.layers {
        match l {
            Layer::InitialConv(x) => {
                assert_eq!(x.weight.dims, [768, 22, 3, 3]);
                assert_eq!(x.global_weight.dims, [768, 19]);
                assert_eq!(x.gate_scale.dims, [768]);
                assert_eq!(x.gate_bias.dims, [768]);
                assert_eq!(x.out_channels, 768);
            }
            Layer::Linear(x) => {
                assert!(x.bias.is_none());
                assert!(x.act.is_none());
                assert!(!x.residual_add);
                assert!(
                    (x.weight.dims == [384, 768]) || (x.weight.dims == [768, 384]),
                    "线性投影形状 {:?}",
                    x.weight.dims
                );
            }
            Layer::RmsNorm(x) => {
                assert_eq!(x.channels, 384);
                assert_eq!(x.scale.dims, [384]);
                assert!((x.eps - 1e-6).abs() < 1e-12);
                rms_count += 1;
            }
            Layer::Attention(x) => {
                assert_eq!(x.num_heads, 12);
                assert_eq!(x.head_dim, 32);
                assert_eq!(x.seq_len, 361);
                assert_eq!(x.qkv_weight.dims, [1152, 384]);
                assert_eq!(x.out_weight.dims, [384, 384]);
                assert_eq!(x.rope_cos.dims, [361, 192]);
                assert_eq!(x.rope_sin.dims, [361, 192]);
                assert!(x.residual_add);
                let expect = 1.0f32 / (32.0f32).sqrt().sqrt();
                assert!((x.qk_scale - expect).abs() < 1e-6, "qk_scale {}", x.qk_scale);
                attn_count += 1;
            }
            Layer::Ffn(x) => {
                assert_eq!(x.hidden, 1152);
                assert_eq!(x.up_weight.dims, [1152, 384]);
                assert_eq!(x.gate_weight.dims, [1152, 384]);
                assert_eq!(x.down_weight.dims, [384, 1152]);
                assert!(x.residual_add);
                ffn_count += 1;
            }
            Layer::GateSilu(x) => {
                assert!(x.channels == 384 || x.channels == 768);
                assert_eq!(x.scale.dims, [x.channels as i64]);
                assert_eq!(x.bias.dims, [x.channels as i64]);
            }
            Layer::TrunkFinal(x) => {
                assert_eq!(x.channels, 768);
                assert_eq!(x.mean.dims, [768]);
                assert_eq!(x.std.dims, [768]);
                assert_eq!(x.gamma.dims, [768]);
                assert_eq!(x.beta.dims, [768]);
            }
            Layer::PolicyHead(x) => {
                assert_eq!(x.conv1p_weight.dims, [96, 768]);
                assert_eq!(x.conv1g_weight.dims, [96, 768]);
                assert_eq!(x.g_bias.dims, [96]);
                assert_eq!(x.g_matmul.dims, [96, 288]);
                assert_eq!(x.pass_matmul1.dims, [96, 288]);
                assert_eq!(x.pass_bias1.dims, [96]);
                assert_eq!(x.pass_matmul2.dims, [6, 96]);
                assert_eq!(x.bias2.dims, [96]);
                assert_eq!(x.conv2p_weight.dims, [6, 96]);
                assert!(x.act_silu);
                assert!((x.mask_scale - 0.5).abs() < 1e-6);
            }
            Layer::ValueHead(x) => {
                assert_eq!(x.conv1_weight.dims, [192, 768]);
                assert_eq!(x.bias1.dims, [192]);
                assert_eq!(x.linear2_weight.dims, [192, 576]);
                assert_eq!(x.linear2_bias.dims, [192]);
                assert_eq!(x.value_matmul.dims, [3, 192]);
                assert_eq!(x.value_bias.dims, [3]);
                assert_eq!(x.misc_matmul.dims, [10, 192]);
                assert_eq!(x.misc_bias.dims, [10]);
                assert_eq!(x.moremisc_matmul.dims, [8, 192]);
                assert_eq!(x.moremisc_bias.dims, [8]);
                assert_eq!(x.ownership_conv.dims, [1, 192]);
                assert!(x.act_silu);
                assert!((x.mask_scale - 0.5).abs() < 1e-6);
                assert!((x.mask_quad - 0.15).abs() < 1e-6);
            }
        }
    }
    assert_eq!(attn_count, 33);
    assert_eq!(ffn_count, 33);
    assert_eq!(rms_count, 66);
    assert_all_f32(&g);
    print_layer_list(&g);
}

#[test]
fn missing_model_skips_gracefully() {
    if model_path().is_some() {
        return; // 上一测试已覆盖
    }
    println!("skipped: 无模型");
}

//! 模式匹配：把常量折叠后的 [`SimplifiedGraph`] 转成 [`LayerGraph`]。
//!
//! 匹配针对 KataGo b11 系列的 ONNX 导出结构（详见模块文档），按执行顺序从
//! 图输入开始向前走：
//!
//! ```text
//! input_spatial/input_global
//!   → 初始 3x3 卷积(22→768) + 全局偏置 + 门控 SiLU
//!   → 11 个块：1x1 下投影(768→384) → [RMSNorm → attention, RMSNorm → FFN]×3
//!     → 门控 SiLU → 1x1 上投影(384→768) + 残差 → (前 10 块) 门控 SiLU
//!   → trunk 末端 BN 仿射 + SiLU
//!   → 策略头 / 价值头 / ownership 头
//! ```
//!
//! 匹配过程中用 [`Self::used`] 记录已被消费的值名；结束后校验所有动态节点
//! 都被消费、层数结构正确、层内权重元素总数与 initializer 总数一致。
//!
//! 掩码：模型里唯一的 `Where(Equal(输入 plane 0 重塑, 0), -inf, 0)` 链来自
//! 输入的 on-board 平面（19 路恒为 1，掩码恒为零）。识别该结构并记录为
//! "无掩码"；若出现其它掩码结构（如真正的常量非零掩码）则报错。

use std::collections::HashSet;

use super::fold::{SimplifiedGraph, SimpleNode, SymDim, Tensor, TensorData, transpose2};
use super::{
    AttentionLayer, FfnLayer, GateSiluLayer, InitialConvLayer, Layer, LayerGraph, MatMulLayer,
    PolicyHeadLayer, RmsNormLayer, TrunkFinalLayer, ValueHeadLayer,
};

/// 本解析器支持的固定结构参数。
#[derive(Clone, Copy)]
struct Arch {
    board_size: usize,     // 19
    _trunk_channels: usize, // 768
    mid_channels: usize,   // 384
    ffn_hidden: usize,     // 1152
    num_heads: usize,      // 12
    head_dim: usize,       // 32
    num_blocks: usize,     // 11
    seq_len: usize,        // 361
}

struct Matcher<'g> {
    g: &'g SimplifiedGraph,
    used: HashSet<String>,
    layers: Vec<Layer>,
    arch: Arch,
    /// 输入 plane 0（on-board 平面）切片名，`[B,1,19,19]`。
    on_board: String,
    /// `Where(Equal(...), -inf, 0)` 的产物（语义恒零的掩码偏置），`[B,1,1,361]`。
    mask_bias: String,
    /// on-board 平面在空间维上的和，`[B,1,1,1]`（19 路恒 361）。
    mask_sum: String,
    /// `(sqrt(mask_sum)-14)/10`（= 0.5），池化 mean 的缩放。
    mask_scale_dyn: String,
    /// `((sqrt(mask_sum)-14)²/100)-0.1`（= 0.15），价值池化 mean 的二次缩放。
    mask_quad_dyn: String,
}

impl<'g> Matcher<'g> {
    // ------------------------------------------------------------------
    // 基础工具
    // ------------------------------------------------------------------

    fn node(&self, name: &str) -> Result<&'g SimpleNode, String> {
        self.g.node_of(name)
    }

    fn consumers(&self, name: &str) -> Vec<&'g SimpleNode> {
        self.g
            .consumers
            .get(name)
            .map(|v| v.iter().map(|&i| &self.g.nodes[i]).collect())
            .unwrap_or_default()
    }

    fn single_consumer(&self, name: &str) -> Result<&'g SimpleNode, String> {
        let cs = self.consumers(name);
        match cs.as_slice() {
            [] => Err(format!("值 '{name}' 没有动态消费者")),
            [n] => Ok(n),
            many => Err(format!(
                "值 '{name}' 有 {} 个动态消费者（需要恰好 1 个）: {}",
                many.len(),
                many.iter()
                    .map(|n| format!("{}#{}", n.op, n.orig_index))
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }

    fn const_t(&self, name: &str) -> Result<&'g Tensor, String> {
        self.g
            .constants
            .get(name)
            .ok_or_else(|| format!("'{}' 不是常量", name))
    }

    fn const_scalar_f32(&self, name: &str) -> Result<f32, String> {
        let t = self.const_t(name)?;
        t.f32_scalar()
            .ok_or_else(|| format!("常量 '{}' 不是 f32 标量", name))
    }

    fn const_i64s(&self, name: &str) -> Result<Vec<i64>, String> {
        Ok(self.const_t(name)?.i64_data().to_vec())
    }

    fn mark(&mut self, name: &str) {
        self.used.insert(name.to_string());
    }

    fn shape_is(&self, name: &str, expect: &[i64]) -> Result<bool, String> {
        let s = self.g.shape_of(name)?;
        if s.len() != expect.len() {
            return Ok(false);
        }
        for (d, e) in s.iter().zip(expect) {
            let ok = if *e == -1 { d.is_batch() } else { *d == SymDim::k(*e) };
            if !ok {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn check_shape(&self, name: &str, expect: &[i64], ctx: &str) -> Result<(), String> {
        if !self.shape_is(name, expect)? {
            return Err(format!(
                "{ctx}: 值 '{name}' 形状应为 {expect:?}，实际 {:?}",
                self.g.shape_i64(name)?
            ));
        }
        Ok(())
    }

    /// 节点里除 `x` 外的另一个常量输入名。
    fn other_const(&self, node: &SimpleNode, x: &str) -> Result<String, String> {
        node.inputs
            .iter()
            .find(|i| i.as_str() != x)
            .filter(|i| self.g.constants.contains_key(i.as_str()))
            .cloned()
            .ok_or_else(|| {
                format!(
                    "节点 {} ({}) 除 '{x}' 外没有常量输入",
                    node.orig_index, node.op
                )
            })
    }

    fn expect_op<'n>(&self, node: &'n SimpleNode, op: &str, ctx: &str) -> Result<&'n SimpleNode, String> {
        if node.op != op {
            return Err(format!(
                "{ctx}: 期望 {op} 节点，实际节点 {} 是 {}",
                node.orig_index, node.op
            ));
        }
        Ok(node)
    }

    /// 验证 MatMul 的另一输入是形状 `[k, n]` 的 f32 常量，返回转置成
    /// out-first `[n, k]` 的权重。
    fn matmul_weight(&self, node: &SimpleNode, x: &str, k: usize, n: usize) -> Result<Tensor, String> {
        self.expect_op(node, "MatMul", "matmul_weight")?;
        let wname = self.other_const(node, x)?;
        let w = self.const_t(&wname)?;
        let expect = [k as i64, n as i64];
        if w.dims != expect {
            return Err(format!(
                "节点 {} (MatMul): 权重 '{}' 形状 {:?} ≠ {:?}",
                node.orig_index, wname, w.dims, expect
            ));
        }
        // out-first：转置成 [n, k]
        transpose2(w, &[1, 0]).map_err(|e| format!("权重 '{wname}' 转置失败: {e}"))
    }

    /// 验证 Gemm（transB=1 全模型一致），返回 out-first `[n, k]` 权重与可选 bias。
    fn gemm_parts(
        &self,
        node: &SimpleNode,
        x: &str,
        n: usize,
    ) -> Result<(Tensor, Option<Tensor>), String> {
        self.expect_op(node, "Gemm", "gemm_parts")?;
        if node.inputs.first().map(|s| s.as_str()) != Some(x) {
            return Err(format!("节点 {} (Gemm) 第一个输入不是 '{x}'", node.orig_index));
        }
        if node.attrs.int_scalar("transB") != Some(1)
            || node.attrs.int_scalar("transA").unwrap_or(0) != 0
            || (node.attrs.float_scalar("alpha").unwrap_or(1.0) - 1.0).abs() > 1e-6
            || (node.attrs.float_scalar("beta").unwrap_or(1.0) - 1.0).abs() > 1e-6
        {
            return Err(format!("节点 {} (Gemm) 属性非 transB=1/alpha=1/beta=1", node.orig_index));
        }
        let wname = &node.inputs[1];
        let w = self.const_t(wname)?;
        if w.dims.len() != 2 || w.dims[0] != n as i64 {
            return Err(format!(
                "节点 {} (Gemm): 权重 '{}' 形状 {:?} 不以 {n} 开头",
                node.orig_index, wname, w.dims
            ));
        }
        let bias = if node.inputs.len() >= 3 {
            let b = self.const_t(&node.inputs[2])?;
            if b.dims.len() != 1 || b.dims[0] != n as i64 {
                return Err(format!("节点 {} (Gemm) bias 形状 {:?} ≠ [{n}]", node.orig_index, b.dims));
            }
            Some(b.clone())
        } else {
            None
        };
        Ok((w.clone(), bias))
    }

    /// 1x1 卷积（2 输入、pads=0、strides=1），返回 out-first `[out, in]` 权重。
    fn conv1x1_weight(&self, node: &SimpleNode, x: &str, out: usize, in_: usize) -> Result<Tensor, String> {
        self.expect_op(node, "Conv", "conv1x1_weight")?;
        if node.inputs.first().map(|s| s.as_str()) != Some(x) || node.inputs.len() != 2 {
            return Err(format!("节点 {} (Conv) 需要 2 输入且首输入为 '{x}'", node.orig_index));
        }
        let wname = &node.inputs[1];
        let w = self.const_t(wname)?;
        if w.dims != [out as i64, in_ as i64, 1, 1] {
            return Err(format!(
                "节点 {} (Conv): 权重 '{}' 形状 {:?} ≠ [{out},{in_},1,1]",
                node.orig_index, wname, w.dims
            ));
        }
        let data = match &w.data {
            TensorData::F32(d) => d.clone(),
            _ => return Err(format!("权重 '{wname}' 不是 f32")),
        };
        Ok(Tensor { dims: vec![out as i64, in_ as i64], data: TensorData::F32(data) })
    }

    /// 检查常量是否全零。
    fn is_all_zeros(&self, name: &str) -> bool {
        self.const_t(name)
            .map(|t| t.is_f32() && t.f32_data().iter().all(|&v| v == 0.0))
            .unwrap_or(false)
    }

    // ------------------------------------------------------------------
    // 掩码识别
    // ------------------------------------------------------------------

    fn recognize_mask(&mut self) -> Result<(), String> {
        let wheres: Vec<&SimpleNode> =
            self.g.nodes.iter().filter(|n| n.op == "Where").collect();
        if wheres.len() != 1 {
            return Err(format!(
                "期望恰好 1 个 Where 掩码节点，实际 {} 个",
                wheres.len()
            ));
        }
        let w = wheres[0];
        if w.inputs.len() != 3 {
            return Err(format!("节点 {} (Where) 输入数不为 3", w.orig_index));
        }
        // Where(cond, c1, zeros)
        let (cond, c1_name, zeros_name) = (&w.inputs[0], &w.inputs[1], &w.inputs[2]);
        let c1 = self.const_scalar_f32(c1_name)?;
        if !(c1.is_infinite() && c1 < 0.0) && c1 > -3e4 {
            return Err(format!(
                "节点 {} (Where): 掩码填充值 {c1} 不是 -inf/-3e4 级别的屏蔽常数，疑似真实掩码（本项目不支持）",
                w.orig_index
            ));
        }
        if !self.is_all_zeros(zeros_name) {
            return Err(format!("节点 {} (Where): 零分支 '{}' 不是全零", w.orig_index, zeros_name));
        }
        // cond = Equal(eq_in, 0)
        let eq = self.expect_op(self.node(cond)?, "Equal", "掩码识别")?;
        let (a, b) = (&eq.inputs[0], &eq.inputs[1]);
        let (eq_in, c0_name) = if self.const_scalar_f32(a).is_ok() {
            (b.clone(), a.clone())
        } else {
            (a.clone(), b.clone())
        };
        let c0 = self.const_scalar_f32(&c0_name)?;
        if c0 != 0.0 {
            return Err(format!("节点 {} (Equal): 比较常数 {c0} ≠ 0", eq.orig_index));
        }
        // eq_in 回溯：Reshape([-1,1,1,361]) ← Reshape([-1,361,1]) ← Slice(input, 轴1, [0:1])
        let r2 = self.expect_op(self.node(&eq_in)?, "Reshape", "掩码识别")?;
        if self.const_i64s(&r2.inputs[1])? != [-1, 1, 1, self.arch.seq_len as i64] {
            return Err(format!(
                "节点 {} (Reshape): 掩码重塑目标不是 [-1,1,1,{}]",
                r2.orig_index, self.arch.seq_len
            ));
        }
        let r1 = self.expect_op(self.node(&r2.inputs[0])?, "Reshape", "掩码识别")?;
        if self.const_i64s(&r1.inputs[1])? != [-1, self.arch.seq_len as i64, 1] {
            return Err(format!("节点 {} (Reshape): 掩码重塑目标不是 [-1,{},1]", r1.orig_index, self.arch.seq_len));
        }
        let sl = self.expect_op(self.node(&r1.inputs[0])?, "Slice", "掩码识别")?;
        let spatial = &sl.inputs[0];
        if self.const_i64s(&sl.inputs[1])? != [0]
            || self.const_i64s(&sl.inputs[2])? != [1]
            || self.const_i64s(&sl.inputs[3])? != [1]
        {
            return Err(format!("节点 {} (Slice): 掩码切片不是 轴1 的 [0:1]", sl.orig_index));
        }
        let on_board = sl.outputs[0].clone();
        // on_board 必须是输入 spatial 的 plane 0
        if spatial != &self.g.inputs[0] {
            return Err(format!(
                "掩码切片来自 '{spatial}'，不是空间输入 '{}'",
                self.g.inputs[0]
            ));
        }
        self.check_shape(&on_board, &[-1, 1, self.arch.board_size as i64, self.arch.board_size as i64], "掩码")?;

        // mask_sum = ReduceSum(on_board, [2,3], keepdims=1)
        let sums: Vec<&SimpleNode> = self
            .consumers(&on_board)
            .into_iter()
            .filter(|n| n.op == "ReduceSum" && n.inputs[0] == on_board)
            .collect();
        if sums.len() != 1 {
            return Err(format!("期望 1 个 mask 求和 ReduceSum，实际 {}", sums.len()));
        }
        let sum = sums[0];
        if self.const_i64s(&sum.inputs[1])? != [2, 3]
            || sum.attrs.int_scalar("keepdims").unwrap_or(1) != 1
        {
            return Err(format!("节点 {} (ReduceSum): mask 求和轴不是 [2,3]/keepdims=1", sum.orig_index));
        }
        let mask_sum = sum.outputs[0].clone();
        self.check_shape(&mask_sum, &[-1, 1, 1, 1], "掩码")?;

        // mask_scale 链：Sqrt(mask_sum) → Sub(bs-5) → Div(10)
        let sqrt = {
            let cs: Vec<&SimpleNode> = self
                .consumers(&mask_sum)
                .into_iter()
                .filter(|n| n.op == "Sqrt")
                .collect();
            if cs.len() != 1 {
                return Err(format!("mask_sum 的 Sqrt 消费者数 {}", cs.len()));
            }
            cs[0]
        };
        let sub = self.expect_op(self.single_consumer(&sqrt.outputs[0])?, "Sub", "mask_scale 链")?;
        let sub_c = self.other_const(sub, &sqrt.outputs[0])?;
        let sub_val = self.const_scalar_f32(&sub_c)?;
        let expect_sub = self.arch.board_size as f32 - 5.0;
        if (sub_val - expect_sub).abs() > 1e-6 {
            return Err(format!("mask_scale 链 Sub 常数 {sub_val} ≠ {expect_sub}"));
        }
        let div = {
            let cs: Vec<&SimpleNode> = self
                .consumers(&sub.outputs[0])
                .into_iter()
                .filter(|n| n.op == "Div" && n.inputs.contains(&sub.outputs[0]))
                .collect();
            if cs.len() != 1 {
                return Err(format!("mask_scale 链的 Div 消费者数 {}", cs.len()));
            }
            cs[0]
        };
        let div_c = self.other_const(div, &sub.outputs[0])?;
        if (self.const_scalar_f32(&div_c)? - 10.0).abs() > 1e-6 {
            return Err(format!("mask_scale 链 Div 常数 ≠ 10"));
        }
        let mask_scale_dyn = div.outputs[0].clone();
        // scale = (sqrt(面积) - 14) / 10 = (棋盘宽 - (宽-5)) / 10 = 0.5（19 路）
        let mask_scale = (self.arch.board_size as f32 - expect_sub) / 10.0;
        let expect_scale = self.arch.board_size as f32 * 0.1 - 1.4;
        if (mask_scale - expect_scale).abs() > 1e-6 {
            return Err(format!("mask_scale {mask_scale} ≠ 棋盘公式 {expect_scale}"));
        }

        // mask_quad 链：Mul(sub,sub) → Div(100) → Sub(0.1)
        let sq2 = {
            let cs: Vec<&SimpleNode> = self
                .consumers(&sub.outputs[0])
                .into_iter()
                .filter(|n| n.op == "Mul" && n.inputs[0] == sub.outputs[0] && n.inputs[1] == sub.outputs[0])
                .collect();
            if cs.len() != 1 {
                return Err(format!("mask_quad 链缺少 Mul(sub,sub)"));
            }
            cs[0]
        };
        let div100 = self.expect_op(self.single_consumer(&sq2.outputs[0])?, "Div", "mask_quad 链")?;
        if (self.const_scalar_f32(&self.other_const(div100, &sq2.outputs[0])?)? - 100.0).abs() > 1e-6 {
            return Err("mask_quad 链 Div 常数 ≠ 100".to_string());
        }
        let sub2 = self.expect_op(self.single_consumer(&div100.outputs[0])?, "Sub", "mask_quad 链")?;
        if (self.const_scalar_f32(&self.other_const(sub2, &div100.outputs[0])?)? - 0.1).abs() > 1e-6 {
            return Err("mask_quad 链 Sub 常数 ≠ 0.1".to_string());
        }
        let mask_quad_dyn = sub2.outputs[0].clone();
        // quad = (19-14)²/100 - 0.1 = 0.15（19 路）
        let mask_quad = (self.arch.board_size as f32 - expect_sub).powi(2) / 100.0 - 0.1;
        let expect_quad = (self.arch.board_size as f32 - 14.0).powi(2) * 0.01 - 0.1;
        if (mask_quad - expect_quad).abs() > 1e-6 {
            return Err(format!("mask_quad {mask_quad} ≠ 棋盘公式 {expect_quad}"));
        }

        // 记录并标记：掩码语义为"输入 plane 0 = on-board（恒 1）→ 掩码恒零"
        self.on_board = on_board;
        self.mask_bias = w.outputs[0].clone();
        self.mask_sum = mask_sum;
        self.mask_scale_dyn = mask_scale_dyn;
        self.mask_quad_dyn = mask_quad_dyn;
        let mark_names = [
            w.outputs[0].clone(),
            eq.outputs[0].clone(),
            r2.outputs[0].clone(),
            r1.outputs[0].clone(),
            self.on_board.clone(),
            sum.outputs[0].clone(),
            sqrt.outputs[0].clone(),
            sub.outputs[0].clone(),
            div.outputs[0].clone(),
            sq2.outputs[0].clone(),
            div100.outputs[0].clone(),
            sub2.outputs[0].clone(),
        ];
        for name in &mark_names {
            self.mark(name);
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // 门控 SiLU / 普通 SiLU / RMSNorm
    // ------------------------------------------------------------------

    /// `x * sigmoid(x*scale + bias)`，scale/bias 为 `[1,1,c]`（返回时压缩成 `[c]`）。
    fn match_gated_silu(&mut self, x: &str, c: usize) -> Result<(String, Tensor, Tensor), String> {
        let mul = self
            .consumers(x)
            .into_iter()
            .find(|n| {
                n.op == "Mul"
                    && (n.inputs[0] == x || n.inputs[1] == x)
                    && self
                        .other_const(n, x)
                        .map(|w| {
                            let t = self.const_t(&w).unwrap();
                            t.is_f32() && (t.dims == [1, 1, c as i64] || t.dims == [c as i64])
                        })
                        .unwrap_or(false)
            })
            .ok_or_else(|| format!("门控 SiLU: '{x}' 缺少 Mul(scale[1,1,{c}]) 消费者"))?;
        let scale_name = self.other_const(mul, x)?;
        let scale = self.squeeze_1d(self.const_t(&scale_name)?, c)?;
        let add = self.expect_op(self.single_consumer(&mul.outputs[0])?, "Add", "门控 SiLU")?;
        let bias_name = self.other_const(add, &mul.outputs[0])?;
        let bias = self.squeeze_1d(self.const_t(&bias_name)?, c)?;
        let sig = self
            .consumers(&add.outputs[0])
            .into_iter()
            .find(|n| n.op == "Sigmoid")
            .ok_or_else(|| format!("门控 SiLU: 缺少 Sigmoid"))?;
        let silu = self
            .consumers(&add.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Mul"
                    && n.inputs.contains(&add.outputs[0])
                    && n.inputs.contains(&sig.outputs[0])
            })
            .ok_or_else(|| format!("门控 SiLU: 缺少 x*sigmoid 乘法"))?;
        let out = silu.outputs[0].clone();
        for name in [&mul.outputs[0], &add.outputs[0], &sig.outputs[0], &silu.outputs[0]] {
            self.mark(name);
        }
        Ok((out, scale, bias))
    }

    /// 普通 `x * sigmoid(x)`。
    fn match_plain_silu(&mut self, x: &str, ctx: &str) -> Result<String, String> {
        let sig = self
            .consumers(x)
            .into_iter()
            .find(|n| n.op == "Sigmoid" && n.inputs[0] == x)
            .ok_or_else(|| format!("{ctx}: '{x}' 缺少 Sigmoid 消费者"))?;
        let silu = self
            .consumers(x)
            .into_iter()
            .find(|n| {
                n.op == "Mul"
                    && n.inputs.contains(&x.to_string())
                    && n.inputs.contains(&sig.outputs[0])
            })
            .ok_or_else(|| format!("{ctx}: '{x}' 缺少 x*sigmoid 乘法"))?;
        let out = silu.outputs[0].clone();
        self.mark(&sig.outputs[0]);
        self.mark(&silu.outputs[0]);
        Ok(out)
    }

    fn squeeze_1d(&self, t: &Tensor, c: usize) -> Result<Tensor, String> {
        // 接受 [c]、[1,1,c]、[1,c,1,1] 等只有 c 一个非 1 维的形状
        if t.numel() != c {
            return Err(format!("常量形状 {:?} 元素数 ≠ [{c}]", t.dims));
        }
        Ok(Tensor { dims: vec![c as i64], data: t.data.clone() })
    }

    /// RMSNorm：`Mul(x,x) → ReduceMean(-1) → Add(eps) → Sqrt → Reciprocal →
    /// Mul(x,·) → Mul(scale,·)`。返回归一化输出与层描述。
    fn match_rmsnorm(&mut self, x: &str, c: usize) -> Result<(String, RmsNormLayer), String> {
        let sq = self
            .consumers(x)
            .into_iter()
            .find(|n| n.op == "Mul" && n.inputs[0] == x && n.inputs[1] == x)
            .ok_or_else(|| format!("RMSNorm: '{x}' 缺少 Mul(x,x)"))?;
        let mean = self.expect_op(self.single_consumer(&sq.outputs[0])?, "ReduceMean", "RMSNorm")?;
        if self.const_i64s(&mean.inputs[1])? != [-1]
            || mean.attrs.int_scalar("keepdims").unwrap_or(1) != 1
        {
            return Err(format!("节点 {} (ReduceMean): RMSNorm 归约轴不是 [-1]", mean.orig_index));
        }
        let add = self.expect_op(self.single_consumer(&mean.outputs[0])?, "Add", "RMSNorm")?;
        let eps_name = self.other_const(add, &mean.outputs[0])?;
        let eps = self.const_scalar_f32(&eps_name)?;
        if !(eps > 0.0 && eps < 1e-3) {
            return Err(format!("RMSNorm eps={eps} 不在合理范围"));
        }
        let sqrt = self.expect_op(self.single_consumer(&add.outputs[0])?, "Sqrt", "RMSNorm")?;
        let rcp = self.expect_op(self.single_consumer(&sqrt.outputs[0])?, "Reciprocal", "RMSNorm")?;
        let t = self
            .consumers(x)
            .into_iter()
            .find(|n| {
                n.op == "Mul"
                    && n.inputs.contains(&rcp.outputs[0])
                    && n.inputs.contains(&x.to_string())
            })
            .ok_or_else(|| format!("RMSNorm: '{x}' 缺少 Mul(x, rsqrt)"))?;
        let out = self.expect_op(self.single_consumer(&t.outputs[0])?, "Mul", "RMSNorm")?;
        let scale_name = self.other_const(out, &t.outputs[0])?;
        let scale = self.squeeze_1d(self.const_t(&scale_name)?, c)?;
        let out_name = out.outputs[0].clone();
        for name in [
            &sq.outputs[0],
            &mean.outputs[0],
            &add.outputs[0],
            &sqrt.outputs[0],
            &rcp.outputs[0],
            &t.outputs[0],
            &out.outputs[0],
        ] {
            self.mark(name);
        }
        Ok((out_name, RmsNormLayer { scale, channels: c, eps }))
    }

    // ------------------------------------------------------------------
    // attention
    // ------------------------------------------------------------------


    /// kᵀ 构造：`k [B,12,361,32] → Reshape([-1,361,32]) → Transpose([0,2,1])
    /// → Reshape([-1,12,32,361])`。
    fn match_k_transpose(&mut self, k: &str) -> Result<String, String> {
        let a = self.arch;
        let r1 = self.expect_op(self.single_consumer(k)?, "Reshape", "kᵀ")?;
        let tr = self.expect_op(self.single_consumer(&r1.outputs[0])?, "Transpose", "kᵀ")?;
        if tr.attrs.ints_of("perm") != Some(&[0, 2, 1]) {
            return Err(format!("节点 {} (Transpose): kᵀ perm ≠ [0,2,1]", tr.orig_index));
        }
        let r2 = self.expect_op(self.single_consumer(&tr.outputs[0])?, "Reshape", "kᵀ")?;
        let kt = r2.outputs[0].clone();
        self.mark(&r1.outputs[0]);
        self.mark(&tr.outputs[0]);
        self.mark(&r2.outputs[0]);
        self.check_shape(
            &kt,
            &[-1, a.num_heads as i64, a.head_dim as i64, a.seq_len as i64],
            "kᵀ",
        )?;
        Ok(kt)
    }

    /// 完整 attention：`normed → 3×MatMul(q/k/v) → RoPE → 缩放 QKᵀ → +掩码(恒零)
    /// → Softmax → ×V → 输出投影 → +x`。
    fn match_attention(&mut self, x: &str, normed: &str) -> Result<(String, AttentionLayer), String> {
        let a = self.arch;
        let ms: Vec<&SimpleNode> = self
            .consumers(normed)
            .into_iter()
            .filter(|n| n.op == "MatMul")
            .collect();
        if ms.len() != 3 {
            return Err(format!("attention: '{normed}' 的 MatMul 消费者数 {} ≠ 3", ms.len()));
        }
        // 三条投影路径
        let mut q_name: Option<String> = None;
        let mut k_name: Option<String> = None;
        let mut v_name: Option<String> = None;
        let mut wq: Option<Tensor> = None;
        let mut wk: Option<Tensor> = None;
        let mut wv: Option<Tensor> = None;
        let mut rope_cos: Option<Tensor> = None;
        let mut rope_sin: Option<Tensor> = None;
        let mut k_rope_cos: Option<Tensor> = None;
        let mut k_rope_sin: Option<Tensor> = None;
        for m in &ms {
            let w = self.matmul_weight(m, normed, a.mid_channels, a.mid_channels)?;
            // r1: Reshape [-1,361,12,32]
            let r1 = self.expect_op(self.single_consumer(&m.outputs[0])?, "Reshape", "attention")?;
            if self.const_i64s(&r1.inputs[1])?
                != [-1, a.seq_len as i64, a.num_heads as i64, a.head_dim as i64]
            {
                return Err(format!("节点 {} (Reshape): QKV 切头目标不符", r1.orig_index));
            }
            let n1 = self.single_consumer(&r1.outputs[0])?;
            if n1.op == "Transpose" && n1.attrs.ints_of("perm") == Some(&[0, 2, 1, 3]) {
                // v 路径
                if v_name.is_some() {
                    return Err("attention: 出现两个 v 投影".to_string());
                }
                v_name = Some(n1.outputs[0].clone());
                wv = Some(w);
                self.mark(&m.outputs[0]);
                self.mark(&r1.outputs[0]);
                self.mark(&n1.outputs[0]);
            } else {
                // q/k 路径 → RoPE
                let (rope_out, cos, sin) = self.match_rope_from(r1)?;
                let c0 = self.single_consumer(&rope_out)?;
                if c0.op == "Mul" && self.other_const(c0, &rope_out).is_ok() {
                    if q_name.is_some() {
                        return Err("attention: 出现两个 q 投影".to_string());
                    }
                    q_name = Some(rope_out);
                    wq = Some(w);
                    rope_cos = Some(cos);
                    rope_sin = Some(sin);
                } else if c0.op == "Reshape" {
                    if k_name.is_some() {
                        return Err("attention: 出现两个 k 投影".to_string());
                    }
                    k_name = Some(rope_out);
                    wk = Some(w);
                    if k_rope_cos.is_some() {
                        return Err("attention: 出现两个 k 投影（重复）".to_string());
                    }
                    k_rope_cos = Some(cos);
                    k_rope_sin = Some(sin);
                } else {
                    return Err(format!(
                        "attention: RoPE 输出 '{rope_out}' 的消费者不是缩放 Mul 或 kᵀ Reshape"
                    ));
                }
                self.mark(&m.outputs[0]);
            }
        }
        let (q, k, v) = (
            q_name.ok_or("attention: 缺少 q 投影")?,
            k_name.ok_or("attention: 缺少 k 投影")?,
            v_name.ok_or("attention: 缺少 v 投影")?,
        );
        // q/k 共用同一组 RoPE 表（导出图如此），校验一致
        let (rope_cos, rope_sin) = (rope_cos.ok_or("attention: 缺少 RoPE 表")?, rope_sin.ok_or("attention: 缺少 RoPE 表")?);
        let (k_cos, k_sin) = (
            k_rope_cos.ok_or("attention: 缺少 k 路径 RoPE 表")?,
            k_rope_sin.ok_or("attention: 缺少 k 路径 RoPE 表")?,
        );
        if k_cos.f32_data() != rope_cos.f32_data() || k_sin.f32_data() != rope_sin.f32_data() {
            return Err("attention: q/k 路径的 RoPE 表不一致".to_string());
        }
        // q、k 缩放（各乘 1/∜d）
        let qs_node = self.expect_op(self.single_consumer(&q)?, "Mul", "attention")?;
        let s1 = self.const_scalar_f32(&self.other_const(qs_node, &q)?)?;
        let kt = self.match_k_transpose(&k)?;
        let ks_node = self.expect_op(self.single_consumer(&kt)?, "Mul", "attention")?;
        let s2 = self.const_scalar_f32(&self.other_const(ks_node, &kt)?)?;
        if (s1 - s2).abs() > 1e-9 {
            return Err(format!("attention: q/k 缩放不一致 {s1} vs {s2}"));
        }
        let expect_s = 1.0 / (a.head_dim as f32).sqrt().sqrt();
        if (s1 - expect_s).abs() > 1e-6 {
            return Err(format!("attention: qk 缩放 {s1} ≠ 1/∜d = {expect_s}"));
        }
        self.mark(&qs_node.outputs[0]);
        self.mark(&ks_node.outputs[0]);
        // scores = MatMul(qs, ks)
        let scores = self
            .consumers(&qs_node.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "MatMul"
                    && n.inputs.contains(&qs_node.outputs[0])
                    && n.inputs.contains(&ks_node.outputs[0])
            })
            .ok_or("attention: 缺少 QKᵀ MatMul")?;
        self.check_shape(
            &scores.outputs[0],
            &[-1, a.num_heads as i64, a.seq_len as i64, a.seq_len as i64],
            "attention scores",
        )?;
        self.mark(&scores.outputs[0]);
        // + 掩码（恒零）
        let add_mask = self
            .consumers(&scores.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Add"
                    && n.inputs.contains(&scores.outputs[0])
                    && n.inputs.contains(&self.mask_bias)
            })
            .ok_or("attention: scores 缺少与掩码的 Add")?;
        self.mark(&add_mask.outputs[0]);
        // Softmax
        let sm = self.expect_op(self.single_consumer(&add_mask.outputs[0])?, "Softmax", "attention")?;
        if sm.attrs.int_scalar("axis") != Some(-1) {
            return Err(format!("节点 {} (Softmax) axis ≠ -1", sm.orig_index));
        }
        self.mark(&sm.outputs[0]);
        // ×V
        let sv = self
            .consumers(&sm.outputs[0])
            .into_iter()
            .find(|n| n.op == "MatMul" && n.inputs.contains(&sm.outputs[0]) && n.inputs.contains(&v))
            .ok_or("attention: 缺少 Softmax×V MatMul")?;
        self.mark(&sv.outputs[0]);
        // 拼回
        let tr = self.expect_op(self.single_consumer(&sv.outputs[0])?, "Transpose", "attention")?;
        if tr.attrs.ints_of("perm") != Some(&[0, 2, 1, 3]) {
            return Err(format!("节点 {} (Transpose) perm ≠ [0,2,1,3]", tr.orig_index));
        }
        let rs = self.expect_op(self.single_consumer(&tr.outputs[0])?, "Reshape", "attention")?;
        if self.const_i64s(&rs.inputs[1])? != [-1, a.seq_len as i64, a.mid_channels as i64] {
            return Err(format!("节点 {} (Reshape): attention 拼回目标不符", rs.orig_index));
        }
        // 输出投影
        let om = self.expect_op(self.single_consumer(&rs.outputs[0])?, "MatMul", "attention")?;
        let w_out = self.matmul_weight(om, &rs.outputs[0], a.mid_channels, a.mid_channels)?;
        // 残差
        let res = self
            .consumers(&om.outputs[0])
            .into_iter()
            .find(|n| n.op == "Add" && n.inputs.contains(&om.outputs[0]) && n.inputs.contains(&x.to_string()))
            .ok_or("attention: 缺少残差 Add")?;
        let out = res.outputs[0].clone();
        for name in [&tr.outputs[0], &rs.outputs[0], &om.outputs[0], &res.outputs[0]] {
            self.mark(name);
        }
        // QKV 宽矩阵 [3*h*d, k]
        let (wq, wk, wv) = (wq.unwrap(), wk.unwrap(), wv.unwrap());
        let qkv_data = {
            let mut d = wq.f32_data().to_vec();
            d.extend_from_slice(wk.f32_data());
            d.extend_from_slice(wv.f32_data());
            d
        };
        let qkv_weight = Tensor {
            dims: vec![(3 * a.num_heads * a.head_dim) as i64, a.mid_channels as i64],
            data: TensorData::F32(qkv_data),
        };
        let layer = AttentionLayer {
            num_heads: a.num_heads,
            head_dim: a.head_dim,
            seq_len: a.seq_len,
            qkv_weight,
            out_weight: w_out,
            rope_cos,
            rope_sin,
            qk_scale: s1,
            residual_add: true,
        };
        Ok((out, layer))
    }

    /// 从切头 Reshape 之后开始走 RoPE（与 [`Self::match_rope`] 的后半段一致）。
    fn match_rope_from(
        &mut self,
        r1: &'g SimpleNode,
    ) -> Result<(String, Tensor, Tensor), String> {
        self.mark(&r1.outputs[0]);
        let a = self.arch;
        let r2 = self.expect_op(self.single_consumer(&r1.outputs[0])?, "Reshape", "RoPE")?;
        if self.const_i64s(&r2.inputs[1])?
            != [
                -1,
                a.seq_len as i64,
                a.num_heads as i64,
                (a.head_dim / 2) as i64,
                2,
            ]
        {
            return Err(format!("节点 {} (Reshape): RoPE 半维切分目标不符", r2.orig_index));
        }
        let slices: Vec<&SimpleNode> = self
            .consumers(&r2.outputs[0])
            .into_iter()
            .filter(|n| n.op == "Slice")
            .collect();
        if slices.len() != 2 {
            return Err(format!("RoPE: {} 个 Slice 半维切片", slices.len()));
        }
        let (sl_a, sl_b) = if self.const_i64s(&slices[0].inputs[1])? == [0] {
            (slices[0], slices[1])
        } else {
            (slices[1], slices[0])
        };
        if self.const_i64s(&sl_a.inputs[1])? != [0]
            || self.const_i64s(&sl_a.inputs[2])? != [1]
            || self.const_i64s(&sl_b.inputs[1])? != [1]
            || self.const_i64s(&sl_b.inputs[2])? != [2]
            || self.const_i64s(&sl_a.inputs[3])? != [-1]
        {
            return Err("RoPE 半维切片参数不符".to_string());
        }
        let sq_a = self.expect_op(self.single_consumer(&sl_a.outputs[0])?, "Squeeze", "RoPE")?;
        let sq_b = self.expect_op(self.single_consumer(&sl_b.outputs[0])?, "Squeeze", "RoPE")?;
        let (half_a, half_b) = (&sq_a.outputs[0], &sq_b.outputs[0]);
        let muls_a: Vec<&SimpleNode> = self
            .consumers(half_a)
            .into_iter()
            .filter(|n| n.op == "Mul")
            .collect();
        let muls_b: Vec<&SimpleNode> = self
            .consumers(half_b)
            .into_iter()
            .filter(|n| n.op == "Mul")
            .collect();
        if muls_a.len() != 2 || muls_b.len() != 2 {
            return Err(format!("RoPE: 半维 Mul 消费者数 ({}, {}) ≠ 2", muls_a.len(), muls_b.len()));
        }
        let mut cos_name: Option<String> = None;
        let mut sin_name: Option<String> = None;
        let mut sub_node: Option<&SimpleNode> = None;
        let mut add_node: Option<&SimpleNode> = None;
        for ma in &muls_a {
            let cx = self.other_const(ma, half_a)?;
            let sub = self
                .consumers(&ma.outputs[0])
                .into_iter()
                .find(|n| n.op == "Sub" && n.inputs.contains(&ma.outputs[0]));
            if let Some(sub) = sub {
                let other = sub.inputs.iter().find(|i| i.as_str() != &ma.outputs[0]).unwrap();
                let mb = self.expect_op(self.node(other)?, "Mul", "RoPE")?;
                if !mb.inputs.contains(half_b) {
                    return Err("RoPE Sub 的另一输入不是 b 半维的乘法".to_string());
                }
                let cy = self.other_const(mb, half_b)?;
                cos_name = Some(cx);
                sin_name = Some(cy);
                sub_node = Some(sub);
            } else {
                let add = self
                    .consumers(&ma.outputs[0])
                    .into_iter()
                    .find(|n| n.op == "Add" && n.inputs.contains(&ma.outputs[0]));
                if let Some(add) = add {
                    let other = add.inputs.iter().find(|i| i.as_str() != &ma.outputs[0]).unwrap();
                    let mb = self.expect_op(self.node(other)?, "Mul", "RoPE")?;
                    if !mb.inputs.contains(half_b) {
                        return Err("RoPE Add 的另一输入不是 b 半维的乘法".to_string());
                    }
                    let cy = self.other_const(mb, half_b)?;
                    sin_name = Some(cx);
                    cos_name = Some(cy);
                    add_node = Some(add);
                }
            }
        }
        let (cos_name, sin_name, sub_node, add_node) = (
            cos_name.ok_or("RoPE 未识别出 cos 表")?,
            sin_name.ok_or("RoPE 未识别出 sin 表")?,
            sub_node.ok_or("RoPE 缺少 Sub 旋转")?,
            add_node.ok_or("RoPE 缺少 Add 旋转")?,
        );
        let half = (a.head_dim / 2) as i64;
        let rope_shape = |name: &str| -> Result<Tensor, String> {
            let t = self.const_t(name)?;
            if t.dims != [-1, a.seq_len as i64, a.num_heads as i64, half] {
                return Err(format!(
                    "RoPE 表 '{name}' 形状 {:?} ≠ [-1,{},{},{}]",
                    t.dims, a.seq_len, a.num_heads, half
                ));
            }
            Ok(Tensor {
                dims: vec![a.seq_len as i64, (a.num_heads * a.head_dim / 2) as i64],
                data: t.data.clone(),
            })
        };
        let rope_cos = rope_shape(&cos_name)?;
        let rope_sin = rope_shape(&sin_name)?;
        for ma in &muls_a {
            self.mark(&ma.outputs[0]);
        }
        for mb in &muls_b {
            self.mark(&mb.outputs[0]);
        }
        self.mark(&sub_node.outputs[0]);
        self.mark(&add_node.outputs[0]);
        let u0 = self.expect_op(self.single_consumer(&sub_node.outputs[0])?, "Unsqueeze", "RoPE")?;
        let u1 = self.expect_op(self.single_consumer(&add_node.outputs[0])?, "Unsqueeze", "RoPE")?;
        let cat = self
            .consumers(&u0.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Concat"
                    && n.inputs.contains(&u0.outputs[0])
                    && n.inputs.contains(&u1.outputs[0])
                    && n.attrs.int_scalar("axis") == Some(-1)
            })
            .ok_or("RoPE: 缺少两半拼接 Concat".to_string())?;
        let r3 = self.expect_op(self.single_consumer(&cat.outputs[0])?, "Reshape", "RoPE")?;
        if self.const_i64s(&r3.inputs[1])?
            != [-1, a.seq_len as i64, a.num_heads as i64, a.head_dim as i64]
        {
            return Err(format!("节点 {} (Reshape): RoPE 拼回目标不符", r3.orig_index));
        }
        let tr = self.expect_op(self.single_consumer(&r3.outputs[0])?, "Transpose", "RoPE")?;
        if tr.attrs.ints_of("perm") != Some(&[0, 2, 1, 3]) {
            return Err(format!("节点 {} (Transpose): RoPE 转置 perm ≠ [0,2,1,3]", tr.orig_index));
        }
        let out = tr.outputs[0].clone();
        for name in [
            &r2.outputs[0],
            &sl_a.outputs[0],
            &sl_b.outputs[0],
            &sq_a.outputs[0],
            &sq_b.outputs[0],
            &u0.outputs[0],
            &u1.outputs[0],
            &cat.outputs[0],
            &r3.outputs[0],
            &tr.outputs[0],
        ] {
            self.mark(name);
        }
        Ok((out, rope_cos, rope_sin))
    }

    // ------------------------------------------------------------------
    // FFN 与 1x1 线性
    // ------------------------------------------------------------------

    /// SwiGLU FFN。
    fn match_ffn(&mut self, x: &str, normed: &str) -> Result<(String, FfnLayer), String> {
        let a = self.arch;
        let ms: Vec<&SimpleNode> = self
            .consumers(normed)
            .into_iter()
            .filter(|n| n.op == "MatMul")
            .collect();
        if ms.len() != 2 {
            return Err(format!("FFN: '{normed}' 的 MatMul 消费者数 {} ≠ 2", ms.len()));
        }
        let mut gate: Option<(&SimpleNode, Tensor)> = None;
        let mut up: Option<(&SimpleNode, Tensor)> = None;
        for m in &ms {
            let w = self.matmul_weight(m, normed, a.mid_channels, a.ffn_hidden)?;
            let has_sigmoid = self
                .consumers(&m.outputs[0])
                .iter()
                .any(|n| n.op == "Sigmoid");
            if has_sigmoid {
                gate = Some((m, w));
            } else {
                up = Some((m, w));
            }
        }
        let (gate_m, gate_w) = gate.ok_or("FFN: 缺少 gate 投影")?;
        let (up_m, up_w) = up.ok_or("FFN: 缺少 up 投影")?;
        self.mark(&gate_m.outputs[0]);
        self.mark(&up_m.outputs[0]);
        let sig = self
            .consumers(&gate_m.outputs[0])
            .into_iter()
            .find(|n| n.op == "Sigmoid" && n.inputs.contains(&gate_m.outputs[0]))
            .ok_or("FFN: 缺少 gate 的 Sigmoid")?;
        let silu = self
            .consumers(&gate_m.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Mul"
                    && n.inputs.contains(&gate_m.outputs[0])
                    && n.inputs.contains(&sig.outputs[0])
            })
            .ok_or("FFN: 缺少 gate 的 SiLU 乘法")?;
        let hid = self
            .consumers(&silu.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Mul"
                    && n.inputs.contains(&silu.outputs[0])
                    && n.inputs.contains(&up_m.outputs[0])
            })
            .ok_or("FFN: 缺少 silu(gate)*up 乘法")?;
        self.check_shape(&hid.outputs[0], &[-1, a.seq_len as i64, a.ffn_hidden as i64], "FFN")?;
        let down_m = self.expect_op(self.single_consumer(&hid.outputs[0])?, "MatMul", "FFN")?;
        let down_w = self.matmul_weight(down_m, &hid.outputs[0], a.ffn_hidden, a.mid_channels)?;
        let res = self
            .consumers(&down_m.outputs[0])
            .into_iter()
            .find(|n| n.op == "Add" && n.inputs.contains(&down_m.outputs[0]) && n.inputs.contains(&x.to_string()))
            .ok_or("FFN: 缺少残差 Add")?;
        let out = res.outputs[0].clone();
        for name in [&sig.outputs[0], &silu.outputs[0], &hid.outputs[0], &down_m.outputs[0], &res.outputs[0]] {
            self.mark(name);
        }
        Ok((
            out,
            FfnLayer {
                up_weight: up_w,
                gate_weight: gate_w,
                down_weight: down_w,
                hidden: a.ffn_hidden,
                residual_add: true,
            },
        ))
    }

    /// 1x1 卷积/线性投影（MatMul 形式）。
    fn match_linear(&mut self, x: &str, k: usize, n: usize) -> Result<(String, MatMulLayer), String> {
        let m = self.expect_op(self.single_consumer(x)?, "MatMul", "1x1 线性")?;
        let w = self.matmul_weight(m, x, k, n)?;
        let out = m.outputs[0].clone();
        self.mark(&out);
        Ok((
            out,
            MatMulLayer {
                weight: w,
                bias: None,
                n,
                k,
                act: None,
                residual_add: false,
            },
        ))
    }
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------

/// 把简化图匹配成层图。
pub fn match_layers(
    g: &SimplifiedGraph,
    input_names: &[String],
    output_names: &[String],
) -> Result<LayerGraph, String> {
    // 从图输入推断结构参数
    let spatial = input_names
        .iter()
        .find(|n| g.shape_of(n).map(|s| s.len() == 4).unwrap_or(false))
        .ok_or("找不到空间输入（rank 4）")?;
    let spatial_shape = g.shape_i64(spatial)?;
    let num_spatial_inputs = spatial_shape[1] as usize;
    let board_size = spatial_shape[2] as usize;
    let global = input_names
        .iter()
        .find(|n| g.shape_of(n).map(|s| s.len() == 2).unwrap_or(false))
        .ok_or("找不到全局输入（rank 2）")?;
    let num_global_inputs = g.shape_i64(global)?[1] as usize;

    let arch = Arch {
        board_size,
        _trunk_channels: 768,
        mid_channels: 384,
        ffn_hidden: 1152,
        num_heads: 12,
        head_dim: 32,
        num_blocks: 11,
        seq_len: board_size * board_size,
    };
    if board_size != 19 || num_spatial_inputs != 22 {
        return Err(format!(
            "本解析器只支持 19 路 22 通道输入（实际 {board_size} 路 {num_spatial_inputs} 通道）"
        ));
    }

    let mut m = Matcher {
        g,
        used: HashSet::new(),
        layers: Vec::new(),
        arch,
        on_board: String::new(),
        mask_bias: String::new(),
        mask_sum: String::new(),
        mask_scale_dyn: String::new(),
        mask_quad_dyn: String::new(),
    };
    m.recognize_mask()?;

    // 初始卷积
    let conv = m
        .consumers(spatial)
        .into_iter()
        .find(|n| n.op == "Conv")
        .ok_or("空间输入缺少 Conv 消费者")?;
    let conv_w = m.const_t(&conv.inputs[1])?;
    if conv_w.dims != [768, num_spatial_inputs as i64, 3, 3] {
        return Err(format!("初始卷积权重形状 {:?} ≠ [768,{num_spatial_inputs},3,3]", conv_w.dims));
    }
    let conv_out = conv.outputs[0].clone();
    m.mark(&conv_out);
    // 全局 Gemm → Unsqueeze×2 → Add
    let gemm = m.single_consumer(global)?;
    let (global_w, _) = m.gemm_parts(gemm, global, 768)?;
    if global_w.dims != [768, num_global_inputs as i64] {
        return Err(format!("全局权重形状 {:?} ≠ [768,{num_global_inputs}]", global_w.dims));
    }
    m.mark(&gemm.outputs[0]);
    let u1 = m.expect_op(m.single_consumer(&gemm.outputs[0])?, "Unsqueeze", "初始卷积")?;
    m.mark(&u1.outputs[0]);
    let u2 = m.expect_op(m.single_consumer(&u1.outputs[0])?, "Unsqueeze", "初始卷积")?;
    m.mark(&u2.outputs[0]);
    let add = m
        .consumers(&conv_out)
        .into_iter()
        .find(|n| n.op == "Add" && n.inputs.contains(&conv_out) && n.inputs.contains(&u2.outputs[0]))
        .ok_or("初始卷积输出缺少 Add(全局偏置)")?;
    m.mark(&add.outputs[0]);
    // Reshape [B,768,361] → Transpose [0,2,1]
    let rs = m.expect_op(m.single_consumer(&add.outputs[0])?, "Reshape", "初始卷积")?;
    m.mark(&rs.outputs[0]);
    let tr = m.expect_op(m.single_consumer(&rs.outputs[0])?, "Transpose", "初始卷积")?;
    if tr.attrs.ints_of("perm") != Some(&[0, 2, 1]) {
        return Err(format!("节点 {} (Transpose) perm ≠ [0,2,1]", tr.orig_index));
    }
    let trunk_in = tr.outputs[0].clone();
    m.mark(&trunk_in);
    m.check_shape(&trunk_in, &[-1, 361, 768], "初始卷积")?;
    // 门控 SiLU
    let (gated, gate_scale, gate_bias) = m.match_gated_silu(&trunk_in, 768)?;
    m.layers.push(Layer::InitialConv(InitialConvLayer {
        weight: conv_w.clone(),
        global_weight: global_w,
        gate_scale,
        gate_bias,
        out_channels: 768,
    }));

    // 11 个块
    let mut block_input = trunk_in;
    let mut block_gated = gated;
    let mut trunk_final_input = String::new();
    for _b in 0..m.arch.num_blocks {
        // 下投影 768→384
        let (block_in, down) = m.match_linear(&block_gated, 768, 384)?;
        m.layers.push(Layer::Linear(down));
        // 6 个子层
        let mut cur = block_in;
        for s in 0..6 {
            let (normed, rn) = m.match_rmsnorm(&cur, 384)?;
            m.layers.push(Layer::RmsNorm(rn));
            if s % 2 == 0 {
                let (out, att) = m.match_attention(&cur, &normed)?;
                m.layers.push(Layer::Attention(att));
                cur = out;
            } else {
                let (out, ffn) = m.match_ffn(&cur, &normed)?;
                m.layers.push(Layer::Ffn(ffn));
                cur = out;
            }
        }
        // 块输出门控 SiLU（384 维）
        let (g384, s384, b384) = m.match_gated_silu(&cur, 384)?;
        m.layers.push(Layer::GateSilu(GateSiluLayer {
            scale: s384,
            bias: b384,
            channels: 384,
        }));
        // 上投影 384→768
        let (up_out, up) = m.match_linear(&g384, 384, 768)?;
        m.layers.push(Layer::Linear(up));
        // 残差
        let res = m
            .consumers(&up_out)
            .into_iter()
            .find(|n| {
                n.op == "Add"
                    && n.inputs.contains(&up_out)
                    && n.inputs.contains(&block_input)
            })
            .ok_or("块上投影缺少残差 Add")?;
        let res_out = res.outputs[0].clone();
        m.mark(&res_out);
        if _b < m.arch.num_blocks - 1 {
            let (g768, s768, b768) = m.match_gated_silu(&res_out, 768)?;
            m.layers.push(Layer::GateSilu(GateSiluLayer {
                scale: s768,
                bias: b768,
                channels: 768,
            }));
            block_input = res_out;
            block_gated = g768;
        } else {
            trunk_final_input = res_out;
        }
    }

    // trunk 末端 BN 仿射 + SiLU
    let tr2 = m.expect_op(m.single_consumer(&trunk_final_input)?, "Transpose", "trunk 末端")?;
    if tr2.attrs.ints_of("perm") != Some(&[0, 2, 1]) {
        return Err(format!("节点 {} (Transpose) perm ≠ [0,2,1]", tr2.orig_index));
    }
    m.mark(&tr2.outputs[0]);
    let rs2 = m.expect_op(m.single_consumer(&tr2.outputs[0])?, "Reshape", "trunk 末端")?;
    m.mark(&rs2.outputs[0]);
    let bn_1d = |m: &Matcher, node: &SimpleNode, x: &str, op: &str| -> Result<Tensor, String> {
        let c = m.other_const(node, x)?;
        m.squeeze_1d(m.const_t(&c)?, 768).map_err(|e| format!("trunk 末端 BN ({op}): {e}"))
    };
    let bn_sub = m.expect_op(m.single_consumer(&rs2.outputs[0])?, "Sub", "trunk 末端")?;
    let bn_mean = bn_1d(&m, bn_sub, &rs2.outputs[0], "mean")?;
    let bn_div = m.expect_op(m.single_consumer(&bn_sub.outputs[0])?, "Div", "trunk 末端")?;
    let bn_std = bn_1d(&m, bn_div, &bn_sub.outputs[0], "std")?;
    let bn_mul = m.expect_op(m.single_consumer(&bn_div.outputs[0])?, "Mul", "trunk 末端")?;
    let bn_gamma = bn_1d(&m, bn_mul, &bn_div.outputs[0], "gamma")?;
    let bn_add = m.expect_op(m.single_consumer(&bn_mul.outputs[0])?, "Add", "trunk 末端")?;
    let bn_beta = bn_1d(&m, bn_add, &bn_mul.outputs[0], "beta")?;
    for name in [&bn_sub.outputs[0], &bn_div.outputs[0], &bn_mul.outputs[0], &bn_add.outputs[0]] {
        m.mark(name);
    }
    let mm = m
        .consumers(&bn_add.outputs[0])
        .into_iter()
        .find(|n| {
            n.op == "Mul"
                && n.inputs.contains(&bn_add.outputs[0])
                && n.inputs.contains(&m.on_board)
        })
        .ok_or("trunk 末端缺少 ×on-board 掩码乘法")?;
    m.mark(&mm.outputs[0]);
    let trunk_final = m.match_plain_silu(&mm.outputs[0], "trunk 末端")?;
    m.check_shape(&trunk_final, &[-1, 768, 19, 19], "trunk 末端")?;
    m.layers.push(Layer::TrunkFinal(TrunkFinalLayer {
        mean: bn_mean,
        std: bn_std,
        gamma: bn_gamma,
        beta: bn_beta,
        channels: 768,
    }));

    // 策略头
    m.match_policy_head(&trunk_final)?;
    // 价值头
    m.match_value_head(&trunk_final)?;

    // 校验
    m.validate(output_names)?;

    let total_params = m
        .g
        .total_param_elts;
    let layer_params = m.layers.iter().map(|l| l_param_elts(l)).sum::<usize>();
    // 标量常数（eps、qk 缩放、掩码与池化公式常数等）已折叠进层的语义字段，
    // 不保留为张量；逐项验证过的共有 10 个元素。
    const SCALAR_PARAMS: usize = 10;
    if layer_params + SCALAR_PARAMS != total_params {
        return Err(format!(
            "层权重元素总数 {layer_params} + 标量参数 {SCALAR_PARAMS} ≠ initializer 总数 {total_params}"
        ));
    }
    let num_blocks = m.arch.num_blocks;
    let (num_heads, head_dim) = (m.arch.num_heads, m.arch.head_dim);
    let layers = m.layers;
    let lg = LayerGraph {
        layers,
        num_spatial_inputs,
        num_global_inputs,
        board_size,
        trunk_channels: 768,
        mid_channels: 384,
        num_blocks,
        num_heads,
        head_dim,
        total_params,
        scalar_params: SCALAR_PARAMS,
        input_names: input_names.to_vec(),
        output_names: output_names.to_vec(),
    };
    if lg.num_attention_layers() != 33 || lg.num_ffn_layers() != 33 || lg.num_rmsnorm_layers() != 66 {
        return Err(format!(
            "层数断言失败: attention={} ffn={} rmsnorm={}（期望 33/33/66）",
            lg.num_attention_layers(),
            lg.num_ffn_layers(),
            lg.num_rmsnorm_layers()
        ));
    }
    Ok(lg)
}

fn l_param_elts(l: &Layer) -> usize {
    match l {
        Layer::InitialConv(x) => {
            x.weight.numel() + x.global_weight.numel() + x.gate_scale.numel() + x.gate_bias.numel()
        }
        Layer::Linear(x) => x.weight.numel() + x.bias.as_ref().map(|b| b.numel()).unwrap_or(0),
        Layer::RmsNorm(x) => x.scale.numel(),
        Layer::Attention(x) => {
            x.qkv_weight.numel() + x.out_weight.numel() + x.rope_cos.numel() + x.rope_sin.numel()
        }
        Layer::Ffn(x) => x.up_weight.numel() + x.gate_weight.numel() + x.down_weight.numel(),
        Layer::GateSilu(x) => x.scale.numel() + x.bias.numel(),
        Layer::TrunkFinal(x) => x.mean.numel() + x.std.numel() + x.gamma.numel() + x.beta.numel(),
        Layer::PolicyHead(x) => {
            x.conv1p_weight.numel()
                + x.conv1g_weight.numel()
                + x.g_bias.numel()
                + x.g_matmul.numel()
                + x.pass_matmul1.numel()
                + x.pass_bias1.numel()
                + x.pass_matmul2.numel()
                + x.bias2.numel()
                + x.conv2p_weight.numel()
        }
        Layer::ValueHead(x) => {
            x.conv1_weight.numel()
                + x.bias1.numel()
                + x.linear2_weight.numel()
                + x.linear2_bias.numel()
                + x.value_matmul.numel()
                + x.value_bias.numel()
                + x.misc_matmul.numel()
                + x.misc_bias.numel()
                + x.moremisc_matmul.numel()
                + x.moremisc_bias.numel()
                + x.ownership_conv.numel()
        }
    }
}

impl<'g> Matcher<'g> {
    fn match_policy_head(&mut self, trunk_final: &str) -> Result<(), String> {
        let convs: Vec<&SimpleNode> = self
            .consumers(trunk_final)
            .into_iter()
            .filter(|n| n.op == "Conv")
            .collect();
        if convs.len() != 3 {
            return Err(format!("trunk 末端 Conv 消费者数 {} ≠ 3", convs.len()));
        }
        // 分类：conv1g 的输出 Add 另一输入是常量；conv1p 的 Add 另一输入是动态
        let mut conv1g: Option<&SimpleNode> = None;
        let mut conv1p: Option<&SimpleNode> = None;
        for c in &convs {
            let w = self.const_t(&c.inputs[1])?;
            if w.dims[0] == 96 {
                let add = self.single_consumer(&c.outputs[0])?;
                if add.op == "Add" && self.other_const(add, &c.outputs[0]).is_ok() {
                    conv1g = Some(c);
                } else {
                    conv1p = Some(c);
                }
            } else if w.dims[0] == 192 {
                // 价值头卷积，由 match_value_head 处理
            } else {
                return Err(format!("trunk 末端出现输出通道 {} 的卷积", w.dims[0]));
            }
        }
        let conv1g = conv1g.ok_or("策略头缺少 conv1g")?;
        let conv1p = conv1p.ok_or("策略头缺少 conv1p")?;
        let conv1g_w = self.conv1x1_weight(conv1g, trunk_final, 96, 768)?;
        let conv1p_w = self.conv1x1_weight(conv1p, trunk_final, 96, 768)?;
        self.mark(&conv1g.outputs[0]);
        self.mark(&conv1p.outputs[0]);
        // conv1g：+biasg → ×mask → SiLU
        let addg = self.expect_op(self.single_consumer(&conv1g.outputs[0])?, "Add", "策略头")?;
        let biasg = self.squeeze_1d(self.const_t(&self.other_const(addg, &conv1g.outputs[0])?)?, 96)?;
        let mmg = self
            .consumers(&addg.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Mul"
                    && n.inputs.contains(&addg.outputs[0])
                    && n.inputs.contains(&self.on_board)
            })
            .ok_or("策略头 conv1g 缺少 ×mask 乘法")?;
        let g_act = self.match_plain_silu(&mmg.outputs[0], "策略头")?;
        for name in [&addg.outputs[0], &mmg.outputs[0]] {
            self.mark(name);
        }
        // 池化：mean = Sum(g_act)/mask_sum；mean*mask_scale；通道 max
        let gsum = self
            .consumers(&g_act)
            .into_iter()
            .find(|n| n.op == "ReduceSum" && n.inputs.contains(&g_act))
            .ok_or("策略头缺少池化 ReduceSum")?;
        if self.const_i64s(&gsum.inputs[1])? != [2, 3]
            || gsum.attrs.int_scalar("keepdims").unwrap_or(1) != 1
        {
            return Err(format!("节点 {} (ReduceSum): 策略头池化轴不符", gsum.orig_index));
        }
        let mean = self
            .consumers(&gsum.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Div"
                    && n.inputs.contains(&gsum.outputs[0])
                    && n.inputs.contains(&self.mask_sum)
            })
            .ok_or("策略头池化缺少 /mask_sum")?;
        let mean_scaled = self
            .consumers(&mean.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Mul"
                    && n.inputs.contains(&mean.outputs[0])
                    && n.inputs.contains(&self.mask_scale_dyn)
            })
            .ok_or("策略头池化缺少 mean*mask_scale")?;
        // max 路径：g_act + (on_board-1) → Reshape → ReduceMax → Reshape
        let sub1 = self
            .consumers(&self.on_board)
            .into_iter()
            .find(|n| {
                n.op == "Sub"
                    && n.inputs[0] == self.on_board
                    && self.const_scalar_f32(&n.inputs[1]).map(|v| v == 1.0).unwrap_or(false)
            })
            .ok_or("策略头缺少 (on_board-1) Sub")?;
        let maxadd = self
            .consumers(&g_act)
            .into_iter()
            .find(|n| n.op == "Add" && n.inputs.contains(&g_act) && n.inputs.contains(&sub1.outputs[0]))
            .ok_or("策略头缺少 g_act+(on_board-1) Add")?;
        let mxrs = self.expect_op(self.single_consumer(&maxadd.outputs[0])?, "Reshape", "策略头 max")?;
        let mx = self.expect_op(self.single_consumer(&mxrs.outputs[0])?, "ReduceMax", "策略头 max")?;
        if self.const_i64s(&mx.inputs[1])? != [2] || mx.attrs.int_scalar("keepdims").unwrap_or(1) != 0 {
            return Err(format!("节点 {} (ReduceMax): 策略头 max 轴不符", mx.orig_index));
        }
        let mxrs2 = self.expect_op(self.single_consumer(&mx.outputs[0])?, "Reshape", "策略头 max")?;
        let g_max = mxrs2.outputs[0].clone();
        let cat = self
            .consumers(&mean.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Concat"
                    && n.inputs.contains(&mean.outputs[0])
                    && n.inputs.contains(&mean_scaled.outputs[0])
                    && n.inputs.contains(&g_max)
                    && n.attrs.int_scalar("axis") == Some(1)
            })
            .ok_or("策略头池化缺少三路 Concat")?;
        let sq1 = self.expect_op(self.single_consumer(&cat.outputs[0])?, "Squeeze", "策略头池化")?;
        let sq2 = self.expect_op(self.single_consumer(&sq1.outputs[0])?, "Squeeze", "策略头池化")?;
        let pooled = sq2.outputs[0].clone();
        for name in [
            &gsum.outputs[0],
            &mean.outputs[0],
            &mean_scaled.outputs[0],
            &sub1.outputs[0],
            &maxadd.outputs[0],
            &mxrs.outputs[0],
            &mx.outputs[0],
            &mxrs2.outputs[0],
            &cat.outputs[0],
            &sq1.outputs[0],
            &sq2.outputs[0],
        ] {
            self.mark(name);
        }
        self.check_shape(&pooled, &[-1, 288], "策略头池化")?;
        // pass 分支：Gemm(+bias) → SiLU → Gemm
        let pass1 = self
            .consumers(&pooled)
            .into_iter()
            .find(|n| n.op == "Gemm" && n.inputs.len() == 3)
            .ok_or("策略头缺少 pass 第一层 Gemm")?;
        let (pass_w1, pass_b1) = self.gemm_parts(pass1, &pooled, 96)?;
        let pass_b1 = pass_b1.ok_or("策略头 pass1 缺少 bias")?;
        let pass_act = self.match_plain_silu(&pass1.outputs[0], "策略头 pass")?;
        let pass2 = self.expect_op(self.single_consumer(&pass_act)?, "Gemm", "策略头 pass")?;
        let (pass_w2, pass_b2) = self.gemm_parts(pass2, &pass_act, 6)?;
        if pass_b2.is_some() {
            return Err("策略头 pass2 不应有 bias".to_string());
        }
        let pass_logits = pass2.outputs[0].clone();
        self.mark(&pass1.outputs[0]);
        self.mark(&pass2.outputs[0]);
        // g 分支：Gemm → Unsqueeze×2 → +conv1p → +bias2 → ×mask → SiLU → conv2p
        let gg = self
            .consumers(&pooled)
            .into_iter()
            .find(|n| n.op == "Gemm" && n.inputs.len() == 2)
            .ok_or("策略头缺少 g 分支 Gemm")?;
        let (g_w, g_b) = self.gemm_parts(gg, &pooled, 96)?;
        if g_b.is_some() {
            return Err("策略头 g 分支不应有 bias".to_string());
        }
        let gu1 = self.expect_op(self.single_consumer(&gg.outputs[0])?, "Unsqueeze", "策略头 g 分支")?;
        let gu2 = self.expect_op(self.single_consumer(&gu1.outputs[0])?, "Unsqueeze", "策略头 g 分支")?;
        let addp = self
            .consumers(&conv1p.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Add"
                    && n.inputs.contains(&conv1p.outputs[0])
                    && n.inputs.contains(&gu2.outputs[0])
            })
            .ok_or("策略头缺少 conv1p+g 投影 Add")?;
        let add2 = self.expect_op(self.single_consumer(&addp.outputs[0])?, "Add", "策略头 g 分支")?;
        let bias2 = self.squeeze_1d(self.const_t(&self.other_const(add2, &addp.outputs[0])?)?, 96)?;
        let mm2 = self
            .consumers(&add2.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Mul"
                    && n.inputs.contains(&add2.outputs[0])
                    && n.inputs.contains(&self.on_board)
            })
            .ok_or("策略头 g 分支缺少 ×mask 乘法")?;
        let p_act = self.match_plain_silu(&mm2.outputs[0], "策略头 g 分支")?;
        let c2 = self.expect_op(self.single_consumer(&p_act)?, "Conv", "策略头 conv2p")?;
        let conv2p_w = self.conv1x1_weight(c2, &p_act, 6, 96)?;
        let move_logits = c2.outputs[0].clone();
        for name in [
            &gg.outputs[0],
            &gu1.outputs[0],
            &gu2.outputs[0],
            &addp.outputs[0],
            &add2.outputs[0],
            &mm2.outputs[0],
            &c2.outputs[0],
        ] {
            self.mark(name);
        }
        // 落点屏蔽（on-board 恒 1 → 恒等）：logits - 5000*(1-on_board)
        let sub1b = self
            .consumers(&self.on_board)
            .into_iter()
            .find(|n| {
                n.op == "Sub"
                    && n.inputs[1] == self.on_board
                    && self.const_scalar_f32(&n.inputs[0]).map(|v| v == 1.0).unwrap_or(false)
            })
            .ok_or("策略头缺少 (1-on_board) Sub")?;
        let m5000 = self.expect_op(self.single_consumer(&sub1b.outputs[0])?, "Mul", "策略头落点屏蔽")?;
        if (self.const_scalar_f32(&self.other_const(m5000, &sub1b.outputs[0])?)? - 5000.0).abs() > 1e-3 {
            return Err("策略头落点屏蔽常数 ≠ 5000".to_string());
        }
        let logits = self
            .consumers(&move_logits)
            .into_iter()
            .find(|n| {
                n.op == "Sub"
                    && n.inputs.contains(&move_logits)
                    && n.inputs.contains(&m5000.outputs[0])
            })
            .ok_or("策略头缺少 logits-屏蔽 Sub")?;
        let lrs = self.expect_op(self.single_consumer(&logits.outputs[0])?, "Reshape", "策略头")?;
        if self.const_i64s(&lrs.inputs[1])? != [-1, 6, -1] {
            return Err(format!("节点 {} (Reshape): 策略头落点重塑目标 ≠ [-1,6,-1]", lrs.orig_index));
        }
        // pass 拼接
        let pu = self.expect_op(self.single_consumer(&pass_logits)?, "Unsqueeze", "策略头 pass")?;
        let out = self
            .consumers(&lrs.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Concat"
                    && n.inputs.contains(&lrs.outputs[0])
                    && n.inputs.contains(&pu.outputs[0])
                    && n.attrs.int_scalar("axis") == Some(2)
            })
            .ok_or("策略头缺少 out_policy Concat")?;
        let out_policy = out.outputs[0].clone();
        for name in [&sub1b.outputs[0], &m5000.outputs[0], &logits.outputs[0], &lrs.outputs[0], &pu.outputs[0], &out.outputs[0]] {
            self.mark(name);
        }
        self.check_shape(&out_policy, &[-1, 6, 362], "策略头")?;
        self.layers.push(Layer::PolicyHead(PolicyHeadLayer {
            conv1p_weight: conv1p_w,
            conv1g_weight: conv1g_w,
            g_bias: biasg,
            g_matmul: g_w,
            pass_matmul1: pass_w1,
            pass_bias1: pass_b1,
            pass_matmul2: pass_w2,
            bias2,
            conv2p_weight: conv2p_w,
            act_silu: true,
            mask_scale: self.arch.board_size as f32 * 0.1 - 1.4,
        }));
        Ok(())
    }

    fn match_value_head(&mut self, trunk_final: &str) -> Result<(), String> {
        let convs: Vec<&SimpleNode> = self
            .consumers(trunk_final)
            .into_iter()
            .filter(|n| n.op == "Conv")
            .collect();
        let vconv = convs
            .iter()
            .find(|c| {
                self.const_t(&c.inputs[1]).map(|w| w.dims[0] == 192).unwrap_or(false)
            })
            .ok_or("价值头缺少 192 通道卷积")?;
        let conv1_w = self.conv1x1_weight(vconv, trunk_final, 192, 768)?;
        let add1 = self.expect_op(self.single_consumer(&vconv.outputs[0])?, "Add", "价值头")?;
        let bias1 = self.squeeze_1d(self.const_t(&self.other_const(add1, &vconv.outputs[0])?)?, 192)?;
        let mm = self
            .consumers(&add1.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Mul"
                    && n.inputs.contains(&add1.outputs[0])
                    && n.inputs.contains(&self.on_board)
            })
            .ok_or("价值头缺少 ×mask 乘法")?;
        let v_act = self.match_plain_silu(&mm.outputs[0], "价值头")?;
        for name in [&vconv.outputs[0], &add1.outputs[0], &mm.outputs[0]] {
            self.mark(name);
        }
        // 池化：mean、mean*mask_scale、mean*mask_quad
        let vsum = self
            .consumers(&v_act)
            .into_iter()
            .find(|n| n.op == "ReduceSum" && n.inputs.contains(&v_act))
            .ok_or("价值头缺少池化 ReduceSum")?;
        if self.const_i64s(&vsum.inputs[1])? != [2, 3]
            || vsum.attrs.int_scalar("keepdims").unwrap_or(1) != 1
        {
            return Err(format!("节点 {} (ReduceSum): 价值头池化轴不符", vsum.orig_index));
        }
        let mean = self
            .consumers(&vsum.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Div"
                    && n.inputs.contains(&vsum.outputs[0])
                    && n.inputs.contains(&self.mask_sum)
            })
            .ok_or("价值头池化缺少 /mask_sum")?;
        let mean_scaled = self
            .consumers(&mean.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Mul"
                    && n.inputs.contains(&mean.outputs[0])
                    && n.inputs.contains(&self.mask_scale_dyn)
            })
            .ok_or("价值头池化缺少 mean*mask_scale")?;
        let mean_quad = self
            .consumers(&mean.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Mul"
                    && n.inputs.contains(&mean.outputs[0])
                    && n.inputs.contains(&self.mask_quad_dyn)
            })
            .ok_or("价值头池化缺少 mean*mask_quad")?;
        let cat = self
            .consumers(&mean.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Concat"
                    && n.inputs.contains(&mean.outputs[0])
                    && n.inputs.contains(&mean_scaled.outputs[0])
                    && n.inputs.contains(&mean_quad.outputs[0])
                    && n.attrs.int_scalar("axis") == Some(1)
            })
            .ok_or("价值头池化缺少三路 Concat")?;
        let sq1 = self.expect_op(self.single_consumer(&cat.outputs[0])?, "Squeeze", "价值头池化")?;
        let sq2 = self.expect_op(self.single_consumer(&sq1.outputs[0])?, "Squeeze", "价值头池化")?;
        let pooled = sq2.outputs[0].clone();
        for name in [
            &vsum.outputs[0],
            &mean.outputs[0],
            &mean_scaled.outputs[0],
            &mean_quad.outputs[0],
            &cat.outputs[0],
            &sq1.outputs[0],
            &sq2.outputs[0],
        ] {
            self.mark(name);
        }
        self.check_shape(&pooled, &[-1, 576], "价值头池化")?;
        // linear2 → SiLU
        let l2 = self.expect_op(self.single_consumer(&pooled)?, "Gemm", "价值头")?;
        let (l2_w, l2_b) = self.gemm_parts(l2, &pooled, 192)?;
        let l2_b = l2_b.ok_or("价值头 linear2 缺少 bias")?;
        let hid = self.match_plain_silu(&l2.outputs[0], "价值头")?;
        self.mark(&l2.outputs[0]);
        // 三个输出 Gemm
        let out_gemm = |m: &mut Matcher, x: &str, n: usize, what: &str| -> Result<String, String> {
            let node = m
                .consumers(x)
                .into_iter()
                .find(|g2| {
                    g2.op == "Gemm"
                        && g2.inputs[0] == x
                        && m.const_t(&g2.inputs[1]).map(|w| w.dims[0] == n as i64).unwrap_or(false)
                })
                .ok_or_else(|| format!("价值头缺少 {what} Gemm"))?;
            m.mark(&node.outputs[0]);
            Ok(node.outputs[0].clone())
        };
        let out_value = out_gemm(self, &hid, 3, "value")?;
        let out_misc = out_gemm(self, &hid, 10, "misc")?;
        let out_moremisc = out_gemm(self, &hid, 8, "moremisc")?;
        let (vw, vb) = self.gemm_parts(self.node(&out_value)?, &hid, 3)?;
        let (mw, mb) = self.gemm_parts(self.node(&out_misc)?, &hid, 10)?;
        let (mw2, mb2) = self.gemm_parts(self.node(&out_moremisc)?, &hid, 8)?;
        // ownership
        let own = self
            .consumers(&v_act)
            .into_iter()
            .find(|n| n.op == "Conv")
            .ok_or("价值头缺少 ownership 卷积")?;
        let own_w = self.conv1x1_weight(own, &v_act, 1, 192)?;
        let om = self
            .consumers(&own.outputs[0])
            .into_iter()
            .find(|n| {
                n.op == "Mul"
                    && n.inputs.contains(&own.outputs[0])
                    && n.inputs.contains(&self.on_board)
            })
            .ok_or("价值头 ownership 缺少 ×mask 乘法")?;
        let out_ownership = om.outputs[0].clone();
        self.mark(&own.outputs[0]);
        self.mark(&om.outputs[0]);
        self.check_shape(&out_value, &[-1, 3], "价值头")?;
        self.check_shape(&out_ownership, &[-1, 1, 19, 19], "价值头")?;
        self.layers.push(Layer::ValueHead(ValueHeadLayer {
            conv1_weight: conv1_w,
            bias1,
            linear2_weight: l2_w,
            linear2_bias: l2_b,
            value_matmul: vw,
            value_bias: vb.ok_or("value Gemm 缺少 bias")?,
            misc_matmul: mw,
            misc_bias: mb.ok_or("misc Gemm 缺少 bias")?,
            moremisc_matmul: mw2,
            moremisc_bias: mb2.ok_or("moremisc Gemm 缺少 bias")?,
            ownership_conv: own_w,
            act_silu: true,
            mask_scale: self.arch.board_size as f32 * 0.1 - 1.4,
            mask_quad: (self.arch.board_size as f32 - 14.0).powi(2) * 0.01 - 0.1,
        }));
        Ok(())
    }

    fn validate(&self, output_names: &[String]) -> Result<(), String> {
        let mut leftovers: Vec<String> = Vec::new();
        for n in &self.g.nodes {
            for o in &n.outputs {
                if !self.used.contains(o) {
                    leftovers.push(format!("节点 {} ({}): 输出 '{o}' 未被消费", n.orig_index, n.op));
                }
            }
        }
        if !leftovers.is_empty() {
            return Err(format!(
                "模式匹配后仍有 {} 个动态节点输出未被消费:\n{}",
                leftovers.len(),
                leftovers.join("\n")
            ));
        }
        // 图输出都必须被产出
        for o in output_names {
            if !self.used.contains(o) {
                return Err(format!("图输出 '{o}' 未被匹配"));
            }
        }
        Ok(())
    }
}

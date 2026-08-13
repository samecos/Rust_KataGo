//! 常量折叠与形状传播。
//!
//! 把 ONNX `GraphProto` 里的节点按拓扑序扫描一遍：
//!
//! 1. 所有输入都是常量的节点在加载期求值（Reshape/Transpose/Slice/Concat/
//!    Shape/Expand/Equal/Where/Add/Sub/Mul/Div/Reduce*/Sqrt/Reciprocal/Sigmoid/
//!    Sin/Cos 等），输出并入常量表；RoPE 的 `Add(表a,表b) → Sin/Cos → Unsqueeze`
//!    链因此折叠成预计算的 cos/sin 常数表。
//! 2. 无法折叠的节点（至少一个动态输入）保留为 `SimpleNode`，其输出形状用
//!    符号形状（batch 用 `-1` 表示）向前传播。
//!
//! 折叠产物是 [`SimplifiedGraph`]：剩下的节点只有 MatMul/Gemm/Conv/Softmax/
//! ReduceMean/ReduceSum/ReduceMax/Sqrt/Reciprocal/Sigmoid/Mul/Add/Sub/Div 等
//! 数值节点以及少量动态布局节点（Reshape/Transpose/Slice/…，由模式匹配消化）。

use std::collections::HashMap;

use crate::onnx_proto::{GraphProto, NodeProto, TensorProto, ValueInfoProto};

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

/// 张量数据。仅支持本模型用到的 FLOAT 与 INT64。
#[derive(Clone, Debug, PartialEq)]
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

/// 行主序张量。
///
/// FLOAT 张量的 `dims` 里 `-1` 表示动态 batch 维（始终为第 0 维）；
/// 数据本身按"单 batch"存放（`data.len()` = 其余维之积）。INT64 张量
/// （形状/轴向量）没有 batch 语义，`-1` 只是普通数值。
#[derive(Clone, Debug)]
pub struct Tensor {
    pub dims: Vec<i64>,
    pub data: TensorData,
}

impl Tensor {
    pub fn is_f32(&self) -> bool {
        matches!(self.data, TensorData::F32(_))
    }

    pub fn numel(&self) -> usize {
        match &self.data {
            TensorData::F32(d) => d.len(),
            TensorData::I64(d) => d.len(),
        }
    }

    /// 把 batch 维（-1）当 1 的具体元素数。
    pub fn concrete_numel(&self) -> usize {
        self.dims.iter().map(|&d| if d == -1 { 1 } else { d as usize }).product()
    }

    pub fn f32_data(&self) -> &[f32] {
        match &self.data {
            TensorData::F32(d) => d,
            TensorData::I64(_) => panic!("tensor is not F32"),
        }
    }

    pub fn i64_data(&self) -> &[i64] {
        match &self.data {
            TensorData::I64(d) => d,
            TensorData::F32(_) => panic!("tensor is not I64"),
        }
    }

    /// 标量（0 维或 1 元素）的 f32 值。
    pub fn f32_scalar(&self) -> Option<f32> {
        match &self.data {
            TensorData::F32(d) if d.len() == 1 => Some(d[0]),
            _ => None,
        }
    }

    /// 1 维 INT64 向量内容。
    pub fn i64_vec(&self) -> Option<&[i64]> {
        match &self.data {
            TensorData::I64(d) if self.dims.len() <= 1 => Some(d),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// 符号形状（batch 用 SymDim{b:1,c:0} 表示）
// ---------------------------------------------------------------------------

/// 符号维：`b * batch + c`。本模型所有维都形如 `k*B` 或常数。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SymDim {
    pub b: i64,
    pub c: i64,
}

impl SymDim {
    pub fn k(c: i64) -> Self {
        SymDim { b: 0, c }
    }
    pub fn batch() -> Self {
        SymDim { b: 1, c: 0 }
    }
    pub fn is_batch(&self) -> bool {
        self.b == 1 && self.c == 0
    }

    /// 序列化为 i64（用于 Shape 输出 / reshape 目标里的 -1 约定）。
    pub fn to_i64(self) -> Result<i64, String> {
        if self.is_batch() {
            Ok(-1)
        } else if self.b == 0 {
            Ok(self.c)
        } else {
            Err(format!("dim {:?} 无法序列化为 ONNX 常数", self))
        }
    }
}

/// 形状乘积 = `C * batch^e`（要求 e ≤ 1，且不允许 b、c 同时非零的混合维）。
pub fn linear_product(dims: &[SymDim]) -> Result<(i64, i64), String> {
    let mut e = 0i64;
    let mut c = 1i64;
    for d in dims {
        if d.b != 0 {
            if d.c != 0 {
                return Err(format!("shape {:?} 含混合批维（b、c 同时非零）", dims));
            }
            e += 1;
            if e > 1 {
                return Err(format!("shape {:?} 含 batch 高次项", dims));
            }
            c *= d.b;
        } else if d.c == 0 {
            c = 0;
        } else {
            c *= d.c;
        }
    }
    Ok((e, c))
}

fn tensor_dims_to_sym(dims: &[i64]) -> Vec<SymDim> {
    dims.iter().map(|&d| if d == -1 { SymDim::batch() } else { SymDim::k(d) }).collect()
}

// ---------------------------------------------------------------------------
// 简化图
// ---------------------------------------------------------------------------

/// 折叠后保留下来的节点属性（常用子集）。
#[derive(Clone, Debug, Default)]
pub struct NodeAttrs {
    pub ints: HashMap<String, Vec<i64>>,
    pub floats: HashMap<String, Vec<f32>>,
    pub ints_scalar: HashMap<String, i64>,
    pub floats_scalar: HashMap<String, f32>,
}

impl NodeAttrs {
    pub fn int_scalar(&self, name: &str) -> Option<i64> {
        self.ints_scalar.get(name).copied()
    }
    pub fn float_scalar(&self, name: &str) -> Option<f32> {
        self.floats_scalar.get(name).copied()
    }
    pub fn ints_of(&self, name: &str) -> Option<&[i64]> {
        self.ints.get(name).map(|v| v.as_slice())
    }
}

/// 折叠后仍需要运行时执行的节点。
#[derive(Clone, Debug)]
pub struct SimpleNode {
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: NodeAttrs,
    /// 原始 ONNX 图里的节点序号（调试用）。
    pub orig_index: usize,
}

/// 常量折叠后的简化图。
pub struct SimplifiedGraph {
    /// 所有常量（initializer + 折叠产物），名字 → 张量。
    pub constants: HashMap<String, Tensor>,
    /// 剩余动态节点（拓扑序）。
    pub nodes: Vec<SimpleNode>,
    /// 每个值的形状（-1 = batch）。
    pub shapes: HashMap<String, Vec<SymDim>>,
    /// 值名 → 产出它的动态节点下标（图输入无此条目）。
    pub producers: HashMap<String, usize>,
    /// 值名 → 消费它的动态节点下标列表。
    pub consumers: HashMap<String, Vec<usize>>,
    /// 图输入/输出名。
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// 全部 FLOAT initializer 的元素总数（参数量）。
    pub total_param_elts: usize,
}

impl SimplifiedGraph {
    /// 值产出节点（无则说明是图输入）。
    pub fn producer_of(&self, name: &str) -> Option<&SimpleNode> {
        self.producers.get(name).map(|&i| &self.nodes[i])
    }

    pub fn node_of(&self, name: &str) -> Result<&SimpleNode, String> {
        self.producer_of(name)
            .ok_or_else(|| format!("值 '{name}' 是图输入，不是节点输出"))
    }

    pub fn shape_of(&self, name: &str) -> Result<&[SymDim], String> {
        self.shapes
            .get(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| format!("缺少值 '{name}' 的形状信息"))
    }

    /// 符号形状转便于断言的形式：batch 维 → -1。
    pub fn shape_i64(&self, name: &str) -> Result<Vec<i64>, String> {
        let s = self.shape_of(name)?;
        s.iter().map(|d| d.to_i64()).collect()
    }
}

// ---------------------------------------------------------------------------
// 属性解析
// ---------------------------------------------------------------------------

fn parse_attrs(node: &NodeProto) -> NodeAttrs {
    let mut attrs = NodeAttrs::default();
    for a in &node.attribute {
        let name = a.name.clone().unwrap_or_default();
        if let Some(v) = a.f {
            attrs.floats_scalar.insert(name, v);
        } else if let Some(v) = a.i {
            attrs.ints_scalar.insert(name, v);
        } else if !a.ints.is_empty() {
            attrs.ints.insert(name, a.ints.clone());
        } else if !a.floats.is_empty() {
            attrs.floats.insert(name, a.floats.clone());
        }
        // s / t / g 等属性本模型不使用
    }
    attrs
}

// ---------------------------------------------------------------------------
// TensorProto 解码
// ---------------------------------------------------------------------------

/// 解码一个 TensorProto（FLOAT / INT64，raw_data 与类型化字段两路都支持）。
pub fn decode_tensor_proto(t: &TensorProto) -> Result<Tensor, String> {
    let name = t.name.clone().unwrap_or_default();
    let dt = t.data_type.unwrap_or(0);
    let dims: Vec<i64> = t.dims.clone();
    let err = |m: &str| format!("initializer '{name}': {m}");
    match dt {
        1 => {
            // FLOAT
            let data = if let Some(raw) = &t.raw_data {
                if raw.len() % 4 != 0 {
                    return Err(err("FLOAT raw_data 长度不是 4 的倍数"));
                }
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            } else {
                t.float_data.clone()
            };
            Ok(Tensor { dims, data: TensorData::F32(data) })
        }
        7 => {
            // INT64
            let data = if let Some(raw) = &t.raw_data {
                if raw.len() % 8 != 0 {
                    return Err(err("INT64 raw_data 长度不是 8 的倍数"));
                }
                raw.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            } else {
                t.int64_data.clone()
            };
            Ok(Tensor { dims, data: TensorData::I64(data) })
        }
        other => Err(err(&format!("不支持的数据类型 data_type={other}（只支持 FLOAT=1/INT64=7）"))),
    }
}

// ---------------------------------------------------------------------------
// 形状传播
// ---------------------------------------------------------------------------

fn normalize_axis(axis: i64, rank: usize) -> Result<usize, String> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a >= r {
        return Err(format!("axis={axis} 超出 rank={rank}"));
    }
    Ok(a as usize)
}

fn broadcast_shape(a: &[SymDim], b: &[SymDim]) -> Result<Vec<SymDim>, String> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let da = if i < n - a.len() { SymDim::k(1) } else { a[i - (n - a.len())] };
        let db = if i < n - b.len() { SymDim::k(1) } else { b[i - (n - b.len())] };
        if da == db {
            out.push(da);
        } else if da == SymDim::k(1) {
            out.push(db);
        } else if db == SymDim::k(1) {
            out.push(da);
        } else {
            return Err(format!("形状不可广播: {:?} vs {:?}", a, b));
        }
    }
    Ok(out)
}

/// 从节点输入名 + 常量表取形状。
fn input_shapes(
    names: &[String],
    shapes: &HashMap<String, Vec<SymDim>>,
) -> Result<Vec<Vec<SymDim>>, String> {
    names
        .iter()
        .map(|n| {
            shapes
                .get(n)
                .cloned()
                .ok_or_else(|| format!("缺少输入 '{n}' 的形状（图未按拓扑序？）"))
        })
        .collect()
}

/// 需要读取 axes 输入（INT64 常量）的归约类算子的输出形状。
/// `input_axes` 优先（opset 18 的 axes 输入），否则看 `axes` 属性。
fn reduce_shape(
    input: &[SymDim],
    keepdims: bool,
    attrs: &NodeAttrs,
    input_axes: Option<&[i64]>,
) -> Result<Vec<SymDim>, String> {
    let rank = input.len();
    let axes: Vec<usize> = if let Some(ia) = input_axes {
        ia.iter().map(|&a| normalize_axis(a, rank)).collect::<Result<_, _>>()?
    } else if let Some(av) = attrs.ints_of("axes") {
        av.iter().map(|&a| normalize_axis(a, rank)).collect::<Result<_, _>>()?
    } else {
        (0..rank).collect()
    };
    let mut out = Vec::with_capacity(rank);
    for (i, d) in input.iter().enumerate() {
        if axes.contains(&i) {
            if keepdims {
                out.push(SymDim::k(1));
            }
        } else {
            out.push(*d);
        }
    }
    Ok(out)
}

fn conv_out_spatial(
    input_spatial: i64,
    kernel: i64,
    pads: &[i64],
    strides: &[i64],
    dilations: &[i64],
) -> Result<i64, String> {
    if input_spatial == -1 {
        return Err("卷积的空间维不能是 batch 符号".to_string());
    }
    let p = pads.iter().copied().sum::<i64>();
    let s = strides.iter().copied().product::<i64>();
    let d = dilations.iter().copied().product::<i64>();
    if s <= 0 {
        return Err("conv stride 必须为正".to_string());
    }
    let num = input_spatial + p - d * (kernel - 1) - 1;
    if num < 0 {
        return Err("卷积输出尺寸为负".to_string());
    }
    Ok(num / s + 1)
}

/// 一个动态节点的输出形状。
pub fn shape_fn(
    op: &str,
    attrs: &NodeAttrs,
    input_names: &[String],
    shapes: &[Vec<SymDim>],
    constants: &HashMap<String, Tensor>,
) -> Result<Vec<Vec<SymDim>>, String> {
    let one = |s: Vec<SymDim>| Ok(vec![s]);
    match op {
        "MatMul" => {
            if shapes.len() != 2 {
                return Err("MatMul 需要 2 个输入".to_string());
            }
            let (a, b) = (&shapes[0], &shapes[1]);
            if a.len() < 2 || b.len() < 2 {
                return Err("MatMul 输入 rank < 2".to_string());
            }
            let mut out = a[..a.len() - 1].to_vec();
            out.push(b[b.len() - 1]);
            one(out)
        }
        "Gemm" => {
            if shapes.len() < 2 || shapes.len() > 3 {
                return Err("Gemm 需要 2-3 个输入".to_string());
            }
            let (a, b) = (&shapes[0], &shapes[1]);
            if a.len() != 2 || b.len() != 2 {
                return Err("Gemm 输入必须为 2 维".to_string());
            }
            let ta = attrs.int_scalar("transA").unwrap_or(0);
            let tb = attrs.int_scalar("transB").unwrap_or(0);
            let m = if ta == 0 { a[0] } else { a[1] };
            let n = if tb == 0 { b[1] } else { b[0] };
            one(vec![m, n])
        }
        "Conv" => {
            if shapes.len() < 2 || shapes.len() > 3 {
                return Err("Conv 需要 2-3 个输入".to_string());
            }
            let (x, w) = (&shapes[0], &shapes[1]);
            if x.len() != 4 || w.len() != 4 {
                return Err("Conv 仅支持 NCHW".to_string());
            }
            let pads = attrs.ints_of("pads").unwrap_or(&[]);
            let strides = attrs.ints_of("strides").unwrap_or(&[1, 1]);
            let dils = attrs.ints_of("dilations").unwrap_or(&[1, 1]);
            let (ph, pw) = (pads[0], pads[2]);
            let (sh_, sw) = (strides[0], strides[1]);
            let (dh, dw) = (dils[0], dils[1]);
            let out_h = conv_out_spatial(x[2].c, w[2].c, &[ph, ph], &[sh_], &[dh])?;
            let out_w = conv_out_spatial(x[3].c, w[3].c, &[pw, pw], &[sw], &[dw])?;
            one(vec![x[0], w[0], SymDim::k(out_h), SymDim::k(out_w)])
        }
        "Softmax" | "Sigmoid" | "Sqrt" | "Reciprocal" | "Sin" | "Cos" | "Neg" | "Abs" | "Relu"
        | "Identity" => {
            if shapes.len() != 1 {
                return Err(format!("{op} 需要 1 个输入"));
            }
            one(shapes[0].to_vec())
        }
        "Add" | "Sub" | "Mul" | "Div" | "Equal" => {
            if shapes.len() != 2 {
                return Err(format!("{op} 需要 2 个输入"));
            }
            one(broadcast_shape(&shapes[0], &shapes[1])?)
        }
        "Where" => {
            if shapes.len() != 3 {
                return Err("Where 需要 3 个输入".to_string());
            }
            let t = broadcast_shape(&shapes[1], &shapes[2])?;
            one(broadcast_shape(&shapes[0], &t)?)
        }
        "ReduceMean" | "ReduceSum" | "ReduceMax" => {
            if shapes.len() < 1 || shapes.len() > 2 {
                return Err(format!("{op} 需要 1-2 个输入"));
            }
            let keepdims = attrs.int_scalar("keepdims").unwrap_or(1) == 1;
            let axes_input: Option<Vec<i64>> = if shapes.len() == 2 {
                let t = constants.get(&input_names[1]).ok_or_else(|| {
                    format!("{op} 的 axes 输入必须是常量")
                })?;
                Some(t.i64_data().to_vec())
            } else {
                None
            };
            one(reduce_shape(&shapes[0], keepdims, attrs, axes_input.as_deref())?)
        }
        "Reshape" => {
            if shapes.len() != 2 {
                return Err("Reshape 需要 2 个输入".to_string());
            }
            let shape_t = constants
                .get(&input_names[1])
                .ok_or_else(|| format!("Reshape 的 shape 输入 '{}' 必须是常量", input_names[1]))?;
            let target: Vec<i64> = shape_t.i64_data().to_vec();
            let total = linear_product(&shapes[0]).map_err(|e| {
                format!(
                    "Reshape(数据输入 '{}' 形状 {:?}, 目标 {target:?}): {e}",
                    input_names[0], shapes[0]
                )
            })?;
            let n_infer = target.iter().filter(|&&d| d == -1).count();
            if n_infer == 0 {
                let dims: Vec<SymDim> = target.iter().map(|&d| SymDim::k(d)).collect();
                let (ke, kc) = linear_product(&dims)?;
                if ke != total.0 || kc != total.1 {
                    return Err(format!(
                        "Reshape 元素数不匹配: {:?} -> {:?}",
                        shapes[0], target
                    ));
                }
                return one(dims);
            }
            // 单个 -1：按 ONNX 语义推断（导出图里 batch 符号也写作 -1，
            // 推断结果恰好就是 batch）。
            if n_infer == 1 {
                let known: Vec<SymDim> = target
                    .iter()
                    .filter(|&&d| d != -1)
                    .map(|&d| SymDim::k(d))
                    .collect();
                let (ke, kc) = linear_product(&known)?;
                if kc == 0 || total.1 % kc != 0 || total.0 < ke {
                    return Err(format!(
                        "Reshape 无法推断 -1 维: {:?} -> {:?}",
                        shapes[0], target
                    ));
                }
                let q = total.1 / kc;
                let infer = if total.0 - ke == 1 {
                    SymDim { b: q, c: 0 }
                } else {
                    SymDim::k(q)
                };
                let mut out = Vec::with_capacity(target.len());
                for &d in &target {
                    if d == -1 {
                        out.push(infer);
                    } else {
                        out.push(SymDim::k(d));
                    }
                }
                return one(out);
            }
            // 两个 -1：导出图的约定是第 1 个为 batch 符号、第 2 个为推断
            // （例如 [-1, 96, -1] → [B, 96, 361]）。
            if n_infer == 2 && target.first() == Some(&-1) {
                if total.0 != 1 {
                    return Err(format!(
                        "Reshape 双 -1 目标要求输入含 batch 维: {:?} -> {:?}",
                        shapes[0], target
                    ));
                }
                let known: i64 = target.iter().filter(|&&d| d != -1).product();
                if known == 0 || total.1 % known != 0 {
                    return Err(format!(
                        "Reshape 无法推断 -1 维: {:?} -> {:?}",
                        shapes[0], target
                    ));
                }
                let infer = SymDim::k(total.1 / known);
                let mut out = Vec::with_capacity(target.len());
                let mut seen_batch = false;
                for &d in &target {
                    if d == -1 {
                        if !seen_batch {
                            out.push(SymDim::batch());
                            seen_batch = true;
                        } else {
                            out.push(infer);
                        }
                    } else {
                        out.push(SymDim::k(d));
                    }
                }
                return one(out);
            }
            Err(format!("Reshape 目标里 -1 过多: {target:?}"))
        }
        "Transpose" => {
            let perm = attrs.ints_of("perm").ok_or("Transpose 缺少 perm")?;
            let rank = shapes[0].len();
            if perm.len() != rank {
                return Err("Transpose perm 长度与 rank 不符".to_string());
            }
            let mut out = Vec::with_capacity(rank);
            for &p in perm {
                let a = normalize_axis(p, rank)?;
                out.push(shapes[0][a]);
            }
            one(out)
        }
        "Slice" => {
            let rank = shapes[0].len();
            let get = |i: usize| -> Result<&[i64], String> {
                let t = constants
                    .get(&input_names[i])
                    .ok_or_else(|| format!("Slice 第 {i} 个输入必须是常量"))?;
                let d = t.i64_data();
                if d.len() != 1 {
                    return Err("本解析器只支持单轴的 Slice".to_string());
                }
                Ok(d)
            };
            let starts = get(1)?.to_vec();
            let ends = get(2)?.to_vec();
            let axes: Vec<i64> = if input_names.len() >= 4 { get(3)?.to_vec() } else { (0..rank as i64).collect() };
            let steps: Vec<i64> = if input_names.len() >= 5 { get(4)?.to_vec() } else { vec![1] };
            if steps.len() != starts.len() || axes.len() != starts.len() {
                return Err("Slice starts/ends/axes/steps 长度不一致".to_string());
            }
            let mut out = shapes[0].to_vec();
            for i in 0..starts.len() {
                let axis = normalize_axis(axes[i], rank)?;
                let d = shapes[0][axis];
                if d.b != 0 {
                    return Err("不支持对 batch 维 Slice（符号形状）".to_string());
                }
                let dim = d.c;
                let step = steps[i];
                if step == 0 {
                    return Err("Slice step 不能为 0".to_string());
                }
                let mut s = starts[i];
                let mut e = ends[i];
                if step > 0 {
                    if s == i64::MIN {
                        s = 0;
                    }
                    if e == i64::MAX {
                        e = dim;
                    }
                    if s < 0 {
                        s += dim;
                    }
                    if e < 0 {
                        e += dim;
                    }
                    s = s.clamp(0, dim);
                    e = e.clamp(0, dim);
                    let n = if e > s { (e - s + step - 1) / step } else { 0 };
                    out[axis] = SymDim::k(n);
                } else {
                    return Err("不支持负 step 的 Slice".to_string());
                }
            }
            one(out)
        }
        "Squeeze" => {
            let rank = shapes[0].len();
            let axes: Vec<i64> = if input_names.len() >= 2 {
                constants
                    .get(&input_names[1])
                    .ok_or("Squeeze 的 axes 输入必须是常量")?
                    .i64_data()
                    .to_vec()
            } else if let Some(av) = attrs.ints_of("axes") {
                av.to_vec()
            } else {
                return Err("Squeeze 缺少 axes".to_string());
            };
            let axes: Vec<usize> = axes
                .iter()
                .map(|&a| normalize_axis(a, rank))
                .collect::<Result<_, _>>()?;
            let mut out = Vec::new();
            for (i, d) in shapes[0].iter().enumerate() {
                if !axes.contains(&i) {
                    out.push(*d);
                }
            }
            one(out)
        }
        "Unsqueeze" => {
            let rank = shapes[0].len();
            let axes: Vec<i64> = if input_names.len() >= 2 {
                constants
                    .get(&input_names[1])
                    .ok_or("Unsqueeze 的 axes 输入必须是常量")?
                    .i64_data()
                    .to_vec()
            } else if let Some(av) = attrs.ints_of("axes") {
                av.to_vec()
            } else {
                return Err("Unsqueeze 缺少 axes".to_string());
            };
            let out_rank = rank + axes.len();
            let axes: Vec<usize> = axes
                .iter()
                .map(|&a| normalize_axis(a, out_rank))
                .collect::<Result<_, _>>()?;
            let mut out = Vec::with_capacity(out_rank);
            let mut src = 0usize;
            for i in 0..out_rank {
                if axes.contains(&i) {
                    out.push(SymDim::k(1));
                } else {
                    out.push(shapes[0][src]);
                    src += 1;
                }
            }
            one(out)
        }
        "Concat" => {
            let axis = attrs.int_scalar("axis").ok_or("Concat 缺少 axis")?;
            let rank = shapes[0].len();
            let a = normalize_axis(axis, rank)?;
            let mut out = shapes[0].to_vec();
            let mut sum = shapes[0][a];
            for s in &shapes[1..] {
                if s.len() != rank {
                    return Err("Concat 输入 rank 不一致".to_string());
                }
                if s[a].b != 0 || sum.b != 0 {
                    return Err("不支持沿 batch 维 Concat（符号形状）".to_string());
                }
                sum = SymDim::k(sum.c + s[a].c);
            }
            out[a] = sum;
            one(out)
        }
        "Expand" => {
            if shapes.len() != 2 {
                return Err("Expand 需要 2 个输入".to_string());
            }
            let target = constants
                .get(&input_names[1])
                .ok_or("Expand 的 shape 输入必须是常量")?
                .i64_data();
            let mut out = Vec::with_capacity(target.len());
            for (i, &t) in target.iter().enumerate() {
                if t == -1 {
                    if i == 0 && target.len() > shapes[0].len() {
                        // 标量 broadcast 到 batch 维：沿用 -1
                        out.push(SymDim::batch());
                    } else if i >= target.len() - shapes[0].len() {
                        out.push(shapes[0][i - (target.len() - shapes[0].len())]);
                    } else {
                        out.push(SymDim::batch());
                    }
                } else {
                    out.push(SymDim::k(t));
                }
            }
            one(out)
        }
        "Shape" => {
            // Shape 一定可折叠，动态路径不应到达这里
            Err("Shape 节点应已在折叠阶段消除".to_string())
        }
        other => Err(format!("形状传播不支持算子 {other}")),
    }
}

// ---------------------------------------------------------------------------
// 常量求值
// ---------------------------------------------------------------------------

fn err_op(op: &str, m: &str) -> String {
    format!("折叠 {op}: {m}")
}

/// 具体化形状：-1 → 1，用于按"单 batch"数据做索引运算。
fn concrete_dims(dims: &[i64]) -> Vec<usize> {
    dims.iter().map(|&d| if d == -1 { 1 } else { d.max(0) as usize }).collect()
}

fn strides_of(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// 广播二元 f32 运算。dims 里 -1 视为 1（batch 为外层广播维）。
fn broadcast_binary(
    a: &Tensor,
    b: &Tensor,
    f: impl Fn(f32, f32) -> f32,
) -> Result<Tensor, String> {
    let (da, db) = (a.f32_data(), b.f32_data());
    let (ca, cb) = (concrete_dims(&a.dims), concrete_dims(&b.dims));
    let out_dims: Vec<i64> = broadcast_shape(
        &tensor_dims_to_sym(&a.dims),
        &tensor_dims_to_sym(&b.dims),
    )?
    .iter()
    .map(|d| d.to_i64())
    .collect::<Result<_, _>>()?;
    let oc = concrete_dims(&out_dims);
    let total: usize = oc.iter().product();
    let rank = oc.len();
    // 标量（0 维或单元素）输入的全部 stride 为 0
    let sa: Vec<usize> = if ca.is_empty() || da.len() == 1 {
        vec![0; rank]
    } else {
        strides_of(&ca)
    };
    let sb: Vec<usize> = if cb.is_empty() || db.len() == 1 {
        vec![0; rank]
    } else {
        strides_of(&cb)
    };
    let mut out = vec![0f32; total];
    let mut idx = vec![0usize; rank];
    for o in 0..total {
        let mut ia = 0usize;
        let mut ib = 0usize;
        for k in 0..rank {
            let kk = k.saturating_sub(rank - ca.len());
            if k >= rank - ca.len() {
                let n = idx[k] % ca[kk];
                ia += n * sa[kk];
            }
            if k >= rank - cb.len() {
                let n = idx[k] % cb[k - (rank - cb.len())];
                ib += n * sb[k - (rank - cb.len())];
            }
        }
        out[o] = f(da[ia], db[ib]);
        // 进位
        for k in (0..rank).rev() {
            idx[k] += 1;
            if idx[k] < oc[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    Ok(Tensor { dims: out_dims, data: TensorData::F32(out) })
}

fn elemwise(t: &Tensor, f: impl Fn(f32) -> f32) -> Tensor {
    let data = t.f32_data().iter().map(|&v| f(v)).collect();
    Tensor { dims: t.dims.clone(), data: TensorData::F32(data) }
}

/// 一般秩转置（支持 FLOAT 常量；batch 维 `-1` 必须留在第 0 维）。
pub fn transpose2(t: &Tensor, perm: &[i64]) -> Result<Tensor, String> {
    let dims = t.dims.clone();
    let concrete = concrete_dims(&dims);
    let rank = concrete.len();
    let perm: Vec<usize> = perm
        .iter()
        .map(|&p| normalize_axis(p, rank))
        .collect::<Result<_, _>>()?;
    // batch（-1）必须留在第 0 维
    if dims[0] == -1 && perm[0] != 0 {
        return Err("不支持把 batch 维转置到非 0 位置".to_string());
    }
    let in_strides = strides_of(&concrete);
    let out_dims: Vec<i64> = perm.iter().map(|&p| dims[p]).collect();
    let out_concrete: Vec<usize> = perm.iter().map(|&p| concrete[p]).collect();
    let total: usize = out_concrete.iter().product();
    let mut out = vec![0f32; total];
    let mut idx = vec![0usize; rank];
    for o in 0..total {
        let mut src = 0usize;
        for k in 0..rank {
            src += idx[perm[k]] * in_strides[k];
        }
        out[o] = t.f32_data()[src];
        for k in (0..rank).rev() {
            idx[k] += 1;
            if idx[k] < out_concrete[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    Ok(Tensor { dims: out_dims, data: TensorData::F32(out) })
}

/// 常量求值。成功返回各输出张量；失败（含"无法折叠"）返回 Err。
/// `dynamic_ok` 为 true 时，非常量输入导致的失败返回 `Ok(None)`（视为动态节点）。
#[allow(clippy::too_many_arguments)]
pub fn fold_op(
    op: &str,
    attrs: &NodeAttrs,
    input_names: &[String],
    inputs: &[&Tensor],
    input_shapes: &[Vec<SymDim>],
    dynamic_ok: bool,
) -> Result<Option<Vec<Tensor>>, String> {
    let none_dyn = |e: String| {
        if dynamic_ok { Ok(None) } else { Err(e) }
    };
    let get_axes_input = |i: usize| -> Result<Vec<i64>, String> {
        let t = inputs[i];
        match &t.data {
            TensorData::I64(d) => Ok(d.clone()),
            _ => Err(err_op(op, "axes 输入必须是 INT64 常量")),
        }
    };
    match op {
        "Add" | "Sub" | "Mul" | "Div" => {
            let f: fn(f32, f32) -> f32 = match op {
                "Add" => |a, b| a + b,
                "Sub" => |a, b| a - b,
                "Mul" => |a, b| a * b,
                _ => |a, b| a / b,
            };
            let t = broadcast_binary(inputs[0], inputs[1], f).map_err(|e| err_op(op, &e))?;
            Ok(Some(vec![t]))
        }
        "Sqrt" => Ok(Some(vec![elemwise(inputs[0], f32::sqrt)])),
        "Reciprocal" => Ok(Some(vec![elemwise(inputs[0], |v| 1.0 / v)])),
        "Sigmoid" => Ok(Some(vec![elemwise(inputs[0], |v| {
            let e = (-v).exp();
            1.0 / (1.0 + e)
        })])),
        "Sin" => Ok(Some(vec![elemwise(inputs[0], f32::sin)])),
        "Cos" => Ok(Some(vec![elemwise(inputs[0], f32::cos)])),
        "Neg" => Ok(Some(vec![elemwise(inputs[0], |v| -v)])),
        "Identity" => Ok(Some(vec![inputs[0].clone()])),
        "Reshape" => {
            let target = get_axes_input(1)?;
            let t = inputs[0];
            if !matches!(t.data, TensorData::F32(_)) {
                return none_dyn(err_op(op, "只支持 FLOAT 常量 Reshape"));
            }
            let per_batch = t.f32_data().len();
            let has_batch = t.dims.first() == Some(&-1);
            let known: i64 = target.iter().filter(|&&d| d != -1).product();
            if has_batch {
                // -1 是 batch 维（保持第 0 维）
                if target.first() != Some(&-1) || target.iter().filter(|&&d| d == -1).count() > 1 {
                    return Err(err_op(op, "含 batch 的常量 reshape 目标必须以 -1 打头且只出现一次"));
                }
                if known != per_batch as i64 {
                    return Err(err_op(op, "reshape 常量元素数不匹配"));
                }
                Ok(Some(vec![Tensor { dims: target, data: t.data.clone() }]))
            } else {
                let has_infer = target.contains(&-1);
                let out_dims: Vec<i64> = if has_infer {
                    if known == 0 || per_batch as i64 % known != 0 {
                        return Err(err_op(op, "reshape 常量时 -1 无法整除"));
                    }
                    let infer = per_batch as i64 / known;
                    target.iter().map(|&d| if d == -1 { infer } else { d }).collect()
                } else {
                    if known != per_batch as i64 {
                        return Err(err_op(op, "reshape 常量元素数不匹配"));
                    }
                    target
                };
                Ok(Some(vec![Tensor { dims: out_dims, data: t.data.clone() }]))
            }
        }
        "Transpose" => {
            let perm = attrs.ints_of("perm").ok_or(err_op(op, "缺少 perm"))?.to_vec();
            let t = transpose2(inputs[0], &perm).map_err(|e| err_op(op, &e))?;
            Ok(Some(vec![t]))
        }
        "Slice" => {
            // 常量 Slice：本模型只有对 INT64 形状向量的一维切片
            let t = inputs[0];
            if !matches!(t.data, TensorData::I64(_)) {
                return none_dyn(err_op(op, "只支持对 INT64 常量切片"));
            }
            let dims = concrete_dims(&t.dims);
            if dims.len() != 1 {
                return none_dyn(err_op(op, "只支持 1 维 INT64 常量切片"));
            }
            let rank = 1usize;
            let starts = get_axes_input(1)?;
            let ends = get_axes_input(2)?;
            let axes: Vec<i64> = if input_names.len() >= 4 {
                get_axes_input(3)?
            } else {
                vec![0]
            };
            let steps: Vec<i64> = if input_names.len() >= 5 { get_axes_input(4)? } else { vec![1] };
            if starts.len() != 1 || ends.len() != 1 || axes.len() != 1 || steps.len() != 1 {
                return Err(err_op(op, "只支持单轴切片"));
            }
            let axis = normalize_axis(axes[0], rank)?;
            let dim = dims[axis] as i64;
            let step = steps[0];
            if step <= 0 {
                return Err(err_op(op, "不支持负 step"));
            }
            let mut s = starts[0];
            let mut e = ends[0];
            if s == i64::MIN {
                s = 0;
            }
            if e == i64::MAX {
                e = dim;
            }
            if s < 0 {
                s += dim;
            }
            if e < 0 {
                e += dim;
            }
            s = s.clamp(0, dim);
            e = e.clamp(0, dim);
            let data: Vec<i64> = if e > s {
                t.i64_data()[(s as usize)..(e as usize)]
                    .iter()
                    .step_by(step as usize)
                    .copied()
                    .collect()
            } else {
                Vec::new()
            };
            let mut out_dims = t.dims.clone();
            out_dims[axis] = data.len() as i64;
            Ok(Some(vec![Tensor { dims: out_dims, data: TensorData::I64(data) }]))
        }
        "Squeeze" | "Unsqueeze" => {
            let t = inputs[0];
            let rank = t.dims.len();
            let axes: Vec<i64> = if input_names.len() >= 2 {
                get_axes_input(1)?
            } else if let Some(av) = attrs.ints_of("axes") {
                av.to_vec()
            } else {
                return Err(err_op(op, "缺少 axes"));
            };
            if op == "Squeeze" {
                let axes: Vec<usize> = axes
                    .iter()
                    .map(|&a| normalize_axis(a, rank))
                    .collect::<Result<_, _>>()?;
                let mut out_dims = Vec::new();
                for (i, &d) in t.dims.iter().enumerate() {
                    if !axes.contains(&i) {
                        out_dims.push(d);
                    }
                }
                Ok(Some(vec![Tensor { dims: out_dims, data: t.data.clone() }]))
            } else {
                let out_rank = rank + axes.len();
                let axes: Vec<usize> = axes
                    .iter()
                    .map(|&a| normalize_axis(a, out_rank))
                    .collect::<Result<_, _>>()?;
                let mut out_dims = Vec::with_capacity(out_rank);
                let mut src = 0usize;
                for i in 0..out_rank {
                    if axes.contains(&i) {
                        out_dims.push(if i == 0 { -1 } else { 1 });
                    } else {
                        out_dims.push(t.dims[src]);
                        src += 1;
                    }
                }
                Ok(Some(vec![Tensor { dims: out_dims, data: t.data.clone() }]))
            }
        }
        "Concat" => {
            let axis = attrs.int_scalar("axis").ok_or(err_op(op, "缺少 axis"))?;
            let rank = inputs[0].dims.len();
            let a = normalize_axis(axis, rank)?;
            if a == 0 && inputs.iter().any(|t| t.dims[0] == -1) {
                return none_dyn(err_op(op, "不支持沿 batch 维拼接常量"));
            }
            match &inputs[0].data {
                TensorData::F32(first) => {
                    let mut data = first.clone();
                    let mut dims = inputs[0].dims.clone();
                    for t in &inputs[1..] {
                        data.extend_from_slice(t.f32_data());
                        dims[a] += t.dims[a];
                    }
                    Ok(Some(vec![Tensor { dims, data: TensorData::F32(data) }]))
                }
                TensorData::I64(first) => {
                    let mut data = first.clone();
                    let mut dims = inputs[0].dims.clone();
                    for t in &inputs[1..] {
                        data.extend_from_slice(t.i64_data());
                        dims[a] += t.dims[a];
                    }
                    Ok(Some(vec![Tensor { dims, data: TensorData::I64(data) }]))
                }
            }
        }
        "Shape" => {
            let start = attrs.int_scalar("start").unwrap_or(0);
            let end = attrs.int_scalar("end").unwrap_or(i64::MAX);
            let s = &input_shapes[0];
            let n = s.len() as i64;
            let start = if start < 0 { start + n } else { start }.max(0);
            let end = if end < 0 { end + n } else { end }.min(n);
            let mut data = Vec::new();
            if end > start {
                for d in &s[start as usize..end as usize] {
                    data.push(d.to_i64().map_err(|e| err_op(op, &e))?);
                }
            }
            Ok(Some(vec![Tensor {
                dims: vec![data.len() as i64],
                data: TensorData::I64(data),
            }]))
        }
        "Expand" => {
            let t = inputs[0];
            let target = get_axes_input(1)?;
            let out_dims: Vec<i64> = target;
            // 本模型 Expand 只用于标量/单元素广播
            if t.f32_data().len() != 1 {
                return none_dyn(err_op(op, "只支持单元素 Expand"));
            }
            Ok(Some(vec![Tensor { dims: out_dims, data: t.data.clone() }]))
        }
        "Equal" => {
            let t = broadcast_binary(inputs[0], inputs[1], |a, b| if a == b { 1.0 } else { 0.0 })?;
            let data: Vec<i64> = t.f32_data().iter().map(|&v| v as i64).collect();
            Ok(Some(vec![Tensor { dims: t.dims, data: TensorData::I64(data) }]))
        }
        "Where" => {
            let (cond, x, y) = (inputs[0], inputs[1], inputs[2]);
            let cond: &[i64] = cond.i64_data();
            let mut out = broadcast_binary(x, y, |_, _| 0.0).map_err(|e| err_op(op, &e))?;
            // 逐元素选择（cond 与 out 同形，或为单元素）
            let out_data = out.f32_data().to_vec();
            let xd = x.f32_data();
            let yd = y.f32_data();
            let n = out_data.len();
            let mut data = vec![0f32; n];
            let xscalar = xd.len() == 1;
            let yscalar = yd.len() == 1;
            let cscalar = cond.len() == 1;
            for i in 0..n {
                let c = if cscalar { cond[0] } else { cond[i] };
                let xv = if xscalar { xd[0] } else { xd[i] };
                let yv = if yscalar { yd[0] } else { yd[i] };
                data[i] = if c != 0 { xv } else { yv };
            }
            out.data = TensorData::F32(data);
            Ok(Some(vec![out]))
        }
        "ReduceMean" | "ReduceSum" | "ReduceMax" => {
            let t = inputs[0];
            let rank = t.dims.len();
            let axes: Vec<i64> = if input_names.len() >= 2 {
                get_axes_input(1)?
            } else if let Some(av) = attrs.ints_of("axes") {
                av.to_vec()
            } else {
                (0..rank as i64).collect()
            };
            let axes: Vec<usize> = axes
                .iter()
                .map(|&a| normalize_axis(a, rank))
                .collect::<Result<_, _>>()?;
            let keepdims = attrs.int_scalar("keepdims").unwrap_or(1) == 1;
            let concrete = concrete_dims(&t.dims);
            let strides = strides_of(&concrete);
            let mut out_dims: Vec<i64> = Vec::with_capacity(rank);
            for (i, d) in t.dims.iter().enumerate() {
                if axes.contains(&i) {
                    if keepdims {
                        out_dims.push(1);
                    }
                } else {
                    out_dims.push(*d);
                }
            }
            let out_concrete = concrete_dims(&out_dims);
            let out_strides = strides_of(&out_concrete);
            let total: usize = out_concrete.iter().product();
            let mut out = match op {
                "ReduceMax" => vec![f32::NEG_INFINITY; total],
                _ => vec![0f32; total],
            };
            let n_reduce: usize = axes.iter().map(|&a| concrete[a]).product();
            let mut idx = vec![0usize; rank];
            for _ in 0..concrete.iter().product::<usize>() {
                let mut src = 0usize;
                for k in 0..rank {
                    src += idx[k] * strides[k];
                }
                // 输出线性位置：只由未被归约的轴贡献；keepdims 时归约轴为 1。
                let mut o = 0usize;
                let mut out_k = 0usize;
                for k in 0..rank {
                    let reduced = axes.contains(&k);
                    if !reduced {
                        o += idx[k] * out_strides[out_k];
                    }
                    if !reduced || keepdims {
                        out_k += 1;
                    }
                }
                let v = t.f32_data()[src];
                match op {
                    "ReduceMax" => out[o] = out[o].max(v),
                    _ => out[o] += v,
                }
                for k in (0..rank).rev() {
                    idx[k] += 1;
                    if idx[k] < concrete[k] {
                        break;
                    }
                    idx[k] = 0;
                }
            }
            if op == "ReduceMean" {
                for v in out.iter_mut() {
                    *v /= n_reduce as f32;
                }
            }
            Ok(Some(vec![Tensor { dims: out_dims, data: TensorData::F32(out) }]))
        }
        other => none_dyn(err_op(other, "常量折叠未实现该算子")),
    }
}

// ---------------------------------------------------------------------------
// 折叠驱动器
// ---------------------------------------------------------------------------

/// 从 ValueInfoProto 读取形状；batch（dim_value=0 或符号维）→ `SymDim::batch()`。
fn parse_value_info_dims(vi: &ValueInfoProto) -> Vec<SymDim> {
    let mut dims = Vec::new();
    if let Some(tp) = &vi.r#type {
        if let Some(crate::onnx_proto::type_proto::Value::TensorType(tensor)) = &tp.value {
            if let Some(shape) = &tensor.shape {
                for d in &shape.dim {
                    match &d.value {
                        Some(crate::onnx_proto::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                            dims.push(if *v <= 0 { SymDim::batch() } else { SymDim::k(*v) });
                        }
                        _ => dims.push(SymDim::batch()),
                    }
                }
            }
        }
    }
    dims
}

/// 对整张图做常量折叠与形状传播，产出 [`SimplifiedGraph`]。
pub fn fold_graph(graph: &GraphProto) -> Result<SimplifiedGraph, String> {
    // 1. initializer → 常量表
    let mut constants: HashMap<String, Tensor> = HashMap::new();
    let mut total_param_elts = 0usize;
    for t in &graph.initializer {
        let name = t.name.clone().unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        let tensor = decode_tensor_proto(t)?;
        if tensor.is_f32() {
            total_param_elts += tensor.numel();
        }
        constants.insert(name, tensor);
    }
    if !graph.sparse_initializer.is_empty() {
        return Err("不支持 sparse_initializer".to_string());
    }

    // 2. 图输入形状 + 常量形状
    let mut shapes: HashMap<String, Vec<SymDim>> = HashMap::new();
    for (name, t) in &constants {
        shapes.insert(name.clone(), tensor_dims_to_sym(&t.dims));
    }
    let mut inputs = Vec::new();
    for vi in &graph.input {
        let name = vi.name.clone().unwrap_or_default();
        let dims = parse_value_info_dims(vi);
        shapes.insert(name.clone(), dims);
        inputs.push(name);
    }
    let outputs: Vec<String> = graph
        .output
        .iter()
        .filter_map(|vi| vi.name.clone())
        .collect();

    // 3. 逐节点折叠 / 保留
    let mut nodes: Vec<SimpleNode> = Vec::new();
    let mut producers: HashMap<String, usize> = HashMap::new();
    for (idx, node) in graph.node.iter().enumerate() {
        let op = node.op_type.clone().unwrap_or_default();
        let attrs = parse_attrs(node);
        let in_shapes = input_shapes(&node.input, &shapes)?;

        if op == "Shape" {
            // Shape 永远可折叠（形状已知，batch 记为 -1）
            if node.input.len() != 1 || node.output.len() != 1 {
                return Err(format!("节点 {idx} (Shape) 输入/输出数量异常"));
            }
            let out = fold_op("Shape", &attrs, &node.input, &[], &in_shapes, false)?
                .ok_or_else(|| format!("节点 {idx} (Shape) 折叠失败"))?;
            let t = out.into_iter().next().unwrap();
            let sd = tensor_dims_to_sym(&t.dims);
            shapes.insert(node.output[0].clone(), sd);
            constants.insert(node.output[0].clone(), t);
            continue;
        }

        let all_const = !node.input.is_empty() && node.input.iter().all(|n| constants.contains_key(n));
        let mut folded = false;
        if all_const {
            let tensors: Vec<&Tensor> = node.input.iter().map(|n| &constants[n]).collect();
            if let Ok(Some(out)) =
                fold_op(&op, &attrs, &node.input, &tensors, &in_shapes, true)
            {
                if out.len() != node.output.len() {
                    return Err(format!("节点 {idx} ({op}) 折叠输出数量不匹配"));
                }
                for (o, t) in node.output.iter().zip(out) {
                    let sd = tensor_dims_to_sym(&t.dims);
                    shapes.insert(o.clone(), sd);
                    constants.insert(o.clone(), t);
                }
                folded = true;
            }
        }
        if folded {
            continue;
        }

        // 动态节点
        let out_shapes = shape_fn(&op, &attrs, &node.input, &in_shapes, &constants)
            .map_err(|e| format!("节点 {idx} ({op}): {e}"))?;
        if node.output.len() != out_shapes.len() {
            return Err(format!("节点 {idx} ({op}) 输出数量与形状数量不符"));
        }
        let new_idx = nodes.len();
        nodes.push(SimpleNode {
            op: op.clone(),
            inputs: node.input.clone(),
            outputs: node.output.clone(),
            attrs,
            orig_index: idx,
        });
        for (o, s) in node.output.iter().zip(out_shapes) {
            shapes.insert(o.clone(), s);
            producers.insert(o.clone(), new_idx);
        }
    }

    // 4. consumers 表（同一节点可能重复消费同一输入，去重）
    let mut consumers: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        for inp in &n.inputs {
            let v = consumers.entry(inp.clone()).or_default();
            if !v.contains(&i) {
                v.push(i);
            }
        }
    }

    // 5. 回收未被动态节点引用的常量（如 RoPE 原始表、折叠中间量）
    let mut needed: Vec<String> = Vec::new();
    for n in &nodes {
        for i in &n.inputs {
            if constants.contains_key(i) {
                needed.push(i.clone());
            }
        }
    }
    for o in &outputs {
        if constants.contains_key(o) {
            needed.push(o.clone());
        }
    }
    constants.retain(|k, _| needed.contains(k));

    Ok(SimplifiedGraph {
        constants,
        nodes,
        shapes,
        producers,
        consumers,
        inputs,
        outputs,
        total_param_elts,
    })
}

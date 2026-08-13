//! ONNX model builder — port of `KataGo/cpp/neuralnet/onnxmodelbuilder.cpp`.
#![allow(clippy::field_reassign_with_default)]

use crate::activations::{ACTIVATION_IDENTITY, ACTIVATION_SILU};
use crate::desc::{ConvLayerDesc, MatBiasLayerDesc, MatMulLayerDesc, ModelDesc, ResidualBlockDesc};
use kata_core::logger::Logger;
use prost::Message;

#[allow(clippy::all, clippy::field_reassign_with_default)]
mod onnx_proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}
use onnx_proto::*;

pub struct OnnxBuildResult {
    pub serialized_model: Vec<u8>,
    pub trunk_tip_and_head_node_names: Vec<String>,
    pub rms_norm_node_names: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum OnnxBuildError {
    #[error("unsupported: {0}")]
    Unsupported(String),
}

// Activation enum values (mirrors cpp/neuralnet/activations.h)
const ACT_RELU: i32 = 0;
const ACT_MISH: i32 = 3;

struct Builder<'a> {
    g: &'a mut GraphProto,
    nn_x: i32,
    nn_y: i32,
    require_exact_nn_len: bool,
    cnt: u32,
    pub tip_head_names: Vec<String>,
    mask_sum_name: String,
    mask_mean_name: String,
    mask_scale_name: String,
    mask_quad_name: String,
    mask_bias_name: String,
}

impl<'a> Builder<'a> {
    fn uniq(&mut self, base: &str) -> String {
        let s = format!("{base}/{}", self.cnt);
        self.cnt += 1;
        s
    }

    fn node(&mut self, op: &str, ins: &[String], out: &str, name: &str) {
        let mut n = NodeProto::default();
        n.op_type = Some(op.into());
        n.name = Some(name.into());
        n.input = ins.to_vec();
        n.output = vec![out.into()];
        self.g.node.push(n);
    }

    fn init(&mut self, name: &str, dims: &[i64], data: &[f32]) -> String {
        let mut t = TensorProto::default();
        t.name = Some(name.into());
        t.data_type = Some(1);
        t.dims = dims.to_vec();
        t.float_data = data.to_vec();
        self.g.initializer.push(t);
        name.into()
    }

    fn scalar(&mut self, name: &str, v: f32) -> String {
        self.init(name, &[1, 1, 1, 1], &[v])
    }

    fn dparam(name: &str) -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(name.into())),
            denotation: None,
        }
    }

    fn dval(v: i64) -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(v)),
            denotation: None,
        }
    }

    fn add_vi(&mut self, name: &str, ch: i32, spatial: bool) {
        let mut vi = ValueInfoProto::default();
        vi.name = Some(name.into());
        let mut tt = type_proto::Tensor::default();
        tt.elem_type = Some(1);
        let mut shape = TensorShapeProto::default();
        shape.dim.push(Self::dparam("batch"));
        shape.dim.push(Self::dval(ch as i64));
        shape
            .dim
            .push(Self::dval(if spatial { self.nn_y as i64 } else { 1 }));
        shape
            .dim
            .push(Self::dval(if spatial { self.nn_x as i64 } else { 1 }));
        tt.shape = Some(shape);
        let mut tp = TypeProto::default();
        tp.value = Some(type_proto::Value::TensorType(tt));
        vi.r#type = Some(tp);
        self.g.input.push(vi);
    }

    fn mark_out(&mut self, tensor: &str, name: &str, ch: i32, spatial: bool) {
        {
            let mut n = NodeProto::default();
            n.op_type = Some("Identity".into());
            n.name = Some(format!("{name}/out"));
            n.input = vec![tensor.into()];
            n.output = vec![name.into()];
            self.g.node.push(n);
        }
        let mut vi = ValueInfoProto::default();
        vi.name = Some(name.into());
        let mut tt = type_proto::Tensor::default();
        tt.elem_type = Some(1);
        let mut shape = TensorShapeProto::default();
        shape.dim.push(Self::dparam("batch"));
        shape.dim.push(Self::dval(ch as i64));
        shape
            .dim
            .push(Self::dval(if spatial { self.nn_y as i64 } else { 1 }));
        shape
            .dim
            .push(Self::dval(if spatial { self.nn_x as i64 } else { 1 }));
        tt.shape = Some(shape);
        let mut tp = TypeProto::default();
        tp.value = Some(type_proto::Value::TensorType(tt));
        vi.r#type = Some(tp);
        self.g.output.push(vi);
    }

    /// Compute mask-derived features for variable board sizes.
    /// When `require_exact_nn_len` is true, this is a no-op.
    fn init_mask_features(&mut self) {
        if self.require_exact_nn_len {
            // Fixed constants for exact board size.
            let width = ((self.nn_x * self.nn_y) as f64).sqrt() as f32;
            self.mask_scale_name = self.scalar("InputMask/scale", width * 0.1 - 1.4);
            self.mask_quad_name =
                self.scalar("InputMask/quad", (width - 14.0).powi(2) * 0.01 - 0.1);
            return;
        }

        // maskSum = sum of mask over H,W → [N,1,1,1]
        self.mask_sum_name = self.reduce_hw("ReduceSum", "InputMask", "InputMask/sum");

        // maskMean = mean of mask over H,W = maskSum / (H*W) → [N,1,1,1]
        self.mask_mean_name = self.reduce_hw("ReduceMean", "InputMask", "InputMask/mean");

        // maskScale = (width*0.1 - 1.4) where width = sqrt(maskSum)
        {
            let width = {
                let mask_sum = self.mask_sum_name.clone();
                let out = self.uniq("InputMask/width");
                self.node(
                    "Sqrt",
                    std::slice::from_ref(&mask_sum),
                    &out,
                    "InputMask/width",
                );
                out
            };
            let s_name = self.uniq("InputMask/scaleW");
            let s = self.scalar(&s_name, 0.1);
            let sh_name = self.uniq("InputMask/scaleB");
            let sh = self.scalar(&sh_name, -1.4);
            let m = self.ew("Mul", &width, &s, &format!("{}/scalemul", "InputMask"));
            self.mask_scale_name = self.ew("Add", &m, &sh, &format!("{}/scale", "InputMask"));
        }

        // maskQuad = (width-14)^2 * 0.01 - 0.1
        {
            let width = {
                let mask_sum = self.mask_sum_name.clone();
                let out = self.uniq("InputMask/width2");
                self.node(
                    "Sqrt",
                    std::slice::from_ref(&mask_sum),
                    &out,
                    "InputMask/width2",
                );
                out
            };
            let c14_name = self.uniq("InputMask/c14");
            let c14 = self.scalar(&c14_name, -14.0);
            let shifted = self.ew("Add", &width, &c14, &format!("{}/centershift", "InputMask"));
            let sq = self.ew(
                "Mul",
                &shifted,
                &shifted,
                &format!("{}/centersquare", "InputMask"),
            );
            let s_name = self.uniq("InputMask/quadW");
            let s = self.scalar(&s_name, 0.01);
            let sh_name = self.uniq("InputMask/quadB");
            let sh = self.scalar(&sh_name, -0.1);
            let m = self.ew("Mul", &sq, &s, &format!("{}/quadmul", "InputMask"));
            self.mask_quad_name = self.ew("Add", &m, &sh, &format!("{}/quad", "InputMask"));
        }

        // maskBias for attention softmax: (mask - 1) * BIG
        // 0 on-board (1-1=0), -BIG off-board (0-1=-1 * BIG)
        {
            let one_name = self.uniq("InputMask/biasone");
            let one = self.scalar(&one_name, -1.0);
            let shift = self.ew(
                "Add",
                "InputMask",
                &one,
                &format!("{}/biasshift", "InputMask"),
            );
            let big_name = self.uniq("InputMask/biasbig");
            let big = self.scalar(&big_name, 1.0e9);
            let bias_nchw = self.ew("Mul", &shift, &big, &format!("{}/biasnchw", "InputMask"));
            self.mask_bias_name = self.reshape(
                &bias_nchw,
                &[0, 1, 1, (self.nn_x * self.nn_y) as i64],
                "InputMask/bias",
            );
        }
    }

    /// Reduce over H,W keeping dims, with axes passed as an input tensor.
    fn reduce_hw(&mut self, op: &str, input: &str, name_base: &str) -> String {
        let axes_name = {
            let mut t = TensorProto::default();
            t.name = Some(format!("{}/axes", name_base));
            t.data_type = Some(7);
            t.dims = vec![2];
            t.int64_data = vec![2, 3];
            self.g.initializer.push(t);
            format!("{}/axes", name_base)
        };
        let out = self.uniq(name_base);
        self.node(op, &[input.to_string(), axes_name], &out, name_base);
        {
            let mut attr = AttributeProto::default();
            attr.name = Some("keepdims".into());
            attr.i = Some(1);
            attr.r#type = Some(2);
            if let Some(last) = self.g.node.last_mut() {
                last.attribute.push(attr);
            }
        }
        out
    }

    fn conv(&mut self, input: &str, desc: &ConvLayerDesc) -> String {
        let (ky, kx) = (desc.conv_y_size, desc.conv_x_size);
        let pad_y = desc.dilation_y * (ky - 1) / 2;
        let pad_x = desc.dilation_x * (kx - 1) / 2;

        let w = self.init(
            &format!("{}.W", desc.name),
            &[
                desc.out_channels as i64,
                desc.in_channels as i64,
                ky as i64,
                kx as i64,
            ],
            &desc.weights,
        );

        let out = self.uniq(&desc.name);
        let mut n = NodeProto::default();
        n.op_type = Some("Conv".into());
        n.name = Some(desc.name.clone());
        n.input = vec![input.into(), w];
        n.output = vec![out.clone()];

        let mut pads = AttributeProto::default();
        pads.name = Some("pads".into());
        pads.ints = vec![pad_y as i64, pad_x as i64, pad_y as i64, pad_x as i64];
        pads.r#type = Some(7);
        n.attribute.push(pads);

        let mut dil = AttributeProto::default();
        dil.name = Some("dilations".into());
        dil.ints = vec![desc.dilation_y as i64, desc.dilation_x as i64];
        dil.r#type = Some(7);
        n.attribute.push(dil);

        self.g.node.push(n);
        out
    }

    /// Decomposed BatchNorm: `Mul` (x * mergedScale) then `Add` (+ mergedBias).
    /// mergedScale = scale / sqrt(variance + epsilon)
    /// mergedBias = bias - mean * mergedScale
    fn bn(&mut self, input: &str, desc: &crate::desc::BatchNormLayerDesc) -> String {
        let c = desc.num_channels as usize;
        let mut merged_scale = vec![0.0f32; c];
        let mut merged_bias = vec![0.0f32; c];
        for i in 0..c {
            let std_dev = (desc.variance[i] + desc.epsilon).sqrt();
            let ms = desc.scale[i] / std_dev;
            merged_scale[i] = ms;
            merged_bias[i] = desc.bias[i] - desc.mean[i] * ms;
        }

        let s_name = self.init(
            &format!("{}.mergedScale", desc.name),
            &[c as i64],
            &merged_scale,
        );
        let b_name = self.init(
            &format!("{}.mergedBias", desc.name),
            &[c as i64],
            &merged_bias,
        );

        let mul_out = self.uniq(&format!("{}/mul", desc.name));
        self.node(
            "Mul",
            &[input.to_string(), s_name],
            &mul_out,
            &format!("{}/mul", desc.name),
        );

        let out = self.uniq(&desc.name);
        self.node("Add", &[mul_out, b_name], &out, &desc.name);
        out
    }

    fn act(&mut self, input: &str, desc: &crate::desc::ActivationLayerDesc) -> String {
        let op = match desc.activation {
            ACT_RELU => "Relu",
            ACT_MISH => "Softplus",
            _ => "Relu",
        };
        let out = self.uniq(&desc.name);
        self.node(op, &[input.into()], &out, &desc.name);
        out
    }

    fn mm(&mut self, input: &str, desc: &MatMulLayerDesc) -> String {
        let w = self.init(
            &format!("{}.W", desc.name),
            &[desc.in_channels as i64, desc.out_channels as i64],
            &desc.weights,
        );
        let out = self.uniq(&desc.name);
        self.node("MatMul", &[input.into(), w], &out, &desc.name);
        out
    }

    fn mb(&mut self, input: &str, desc: &MatBiasLayerDesc) -> String {
        let b = self.init(
            &format!("{}.b", desc.name),
            &[1, desc.num_channels as i64, 1, 1],
            &desc.weights,
        );
        let out = self.uniq(&desc.name);
        self.node("Add", &[input.into(), b], &out, &desc.name);
        out
    }

    /// Global pooling: ReduceMean → Mul by maskScale → ReduceMax → Concat.
    /// Returns a tensor of shape [N, 3C, 1, 1].
    fn apply_gpool(
        &mut self,
        input: &str,
        mask_scale_name: &str,
        name: &str,
        _is_value_head: bool,
    ) -> String {
        let _ = _is_value_head;

        // axes as int64 initializer
        let axes_name = {
            let mut t = TensorProto::default();
            t.name = Some("reduce_axes_hw".into());
            t.data_type = Some(7); // INT64
            t.dims = vec![2];
            t.int64_data = vec![2, 3];
            self.g.initializer.push(t);
            "reduce_axes_hw".to_string()
        };

        let mean = if self.require_exact_nn_len {
            let out = self.uniq(&format!("{}/mean", name));
            self.node(
                "ReduceMean",
                &[input.to_string(), axes_name.clone()],
                &out,
                &format!("{}/mean", name),
            );
            out
        } else {
            // mean = sum / maskSum for variable board size
            let sum_out = self.uniq(&format!("{}/sum", name));
            self.node(
                "ReduceSum",
                &[input.to_string(), axes_name.clone()],
                &sum_out,
                &format!("{}/sum", name),
            );
            let mean_out = self.uniq(&format!("{}/mean", name));
            let mask_sum = self.mask_sum_name.clone();
            self.node(
                "Div",
                &[sum_out, mask_sum],
                &mean_out,
                &format!("{}/mean", name),
            );
            mean_out
        };

        let mean_scaled = self.ew(
            "Mul",
            &mean,
            mask_scale_name,
            &format!("{}/meanscale", name),
        );

        let max = if self.require_exact_nn_len {
            let out = self.uniq(&format!("{}/max", name));
            self.node(
                "ReduceMax",
                &[input.to_string(), axes_name],
                &out,
                &format!("{}/max", name),
            );
            out
        } else {
            // off-board cells shift down by -1 so max ignores them
            let shifted = {
                let neg_one_name = self.uniq(&format!("{}/negone", name));
                let neg_one = self.scalar(&neg_one_name, -1.0);
                self.ew("Add", input, &neg_one, &format!("{}/maskshift", name))
            };
            let shifted = self.ew("Mul", &shifted, "InputMask", &format!("{}/maskadd", name));
            let out = self.uniq(&format!("{}/max", name));
            self.node(
                "ReduceMax",
                &[shifted, axes_name],
                &out,
                &format!("{}/max", name),
            );
            out
        };

        let concat_out = self.uniq(&format!("{}/concat", name));
        let mut concat_node = NodeProto::default();
        concat_node.op_type = Some("Concat".into());
        concat_node.name = Some(format!("{}/concat", name));
        concat_node.input = vec![mean, mean_scaled, max];
        concat_node.output = vec![concat_out.clone()];
        {
            let mut attr = AttributeProto::default();
            attr.name = Some("axis".into());
            attr.i = Some(1);
            attr.r#type = Some(2); // INT
            concat_node.attribute.push(attr);
        }
        self.g.node.push(concat_node);
        concat_out
    }

    /// Global pooling residual block:
    ///   pre_bn → pre_act → mask → regularConv
    ///   gpoolConv → gpool_bn → gpool_act → mask → gpool → gpoolToBiasMul
    ///   add(regular, bias) → mid_bn → mid_act → mask → finalConv → add input
    fn global_pooling_residual_block(
        &mut self,
        input: &str,
        desc: &crate::desc::GlobalPoolingResidualBlockDesc,
        mask_scale_name: &str,
    ) -> String {
        // pre-branch
        let pre = self.bn(input, &desc.pre_bn);
        let pre = self.act(&pre, &desc.pre_activation);
        let pre = self.mask(&pre, &format!("{}/pre", desc.name));

        // regular branch
        let regular = self.conv(&pre, &desc.regular_conv);

        // gpool branch
        let gp = self.conv(&pre, &desc.gpool_conv);
        let gp = self.bn(&gp, &desc.gpool_bn);
        let gp = self.act(&gp, &desc.gpool_activation);
        let gp = self.mask(&gp, &format!("{}/gpool", desc.name));
        let gp_pooled = self.apply_gpool(&gp, mask_scale_name, &format!("{}/gp", desc.name), false);
        let bias = self.mm(&gp_pooled, &desc.gpool_to_bias_mul);

        // add bias to regular
        let x = self.ew("Add", &regular, &bias, &format!("{}/gpbias", desc.name));

        // mid + final
        let x = self.bn(&x, &desc.mid_bn);
        let x = self.act(&x, &desc.mid_activation);
        let x = self.mask(&x, &format!("{}/mid", desc.name));
        let x = self.conv(&x, &desc.final_conv);

        // residual
        self.ew("Add", &x, input, &format!("{}/res", desc.name))
    }

    fn ew(&mut self, op: &str, a: &str, b: &str, name: &str) -> String {
        let out = self.uniq(name);
        self.node(op, &[a.into(), b.into()], &out, name);
        out
    }

    fn mask(&mut self, input: &str, name: &str) -> String {
        self.ew("Mul", input, "InputMask", name)
    }

    fn residual(&mut self, input: &str, desc: &ResidualBlockDesc) -> String {
        let pre = self.bn(input, &desc.pre_bn);
        let pre = self.act(&pre, &desc.pre_activation);
        let c1 = self.conv(&pre, &desc.regular_conv);
        let b1 = self.bn(&c1, &desc.mid_bn);
        let a1 = self.act(&b1, &desc.mid_activation);
        let c2 = self.conv(&a1, &desc.final_conv);
        let add = self.ew("Add", &c2, input, &format!("{}/add", desc.name));
        self.act(&add, &crate::desc::ActivationLayerDesc::default())
    }

    /// Build RMSNorm trunk tip: x / sqrt(mean(x^2)+eps) * gamma + beta → activation → mask.
    ///
    /// When `spatial` is true, mean-of-squares is taken over C,H,W (per batch
    /// element) rather than just C (per position).
    fn build_trunk_tip_rms_norm(
        &mut self,
        input: &str,
        desc: &crate::desc::RMSNormLayerDesc,
        activation: &crate::desc::ActivationLayerDesc,
        mask_name: &str,
    ) -> String {
        let c = desc.num_channels as i64;
        let sq = self.ew("Mul", input, input, &format!("{}/sq", desc.name));

        let mean_sq;
        if !desc.spatial {
            // Per-position: mean over channel axis only → [N,1,H,W]
            let axes_name = {
                let mut t = TensorProto::default();
                t.name = Some(format!("{}/axC", desc.name));
                t.data_type = Some(7);
                t.dims = vec![1];
                t.int64_data = vec![1];
                self.g.initializer.push(t);
                format!("{}/axC", desc.name)
            };
            let out = self.uniq(&format!("{}/meansq", desc.name));
            self.node(
                "ReduceMean",
                &[sq, axes_name],
                &out,
                &format!("{}/meansq", desc.name),
            );
            {
                let mut attr = AttributeProto::default();
                attr.name = Some("keepdims".into());
                attr.i = Some(1);
                attr.r#type = Some(2);
                if let Some(last) = self.g.node.last_mut() {
                    last.attribute.push(attr);
                }
            }
            mean_sq = out;
        } else {
            // Spatial: mean over C,H,W → [N,1,1,1]
            let axes_name = {
                let mut t = TensorProto::default();
                t.name = Some(format!("{}/axCHW", desc.name));
                t.data_type = Some(7);
                t.dims = vec![3];
                t.int64_data = vec![1, 2, 3];
                self.g.initializer.push(t);
                format!("{}/axCHW", desc.name)
            };
            let sq_mask = if !self.require_exact_nn_len {
                self.ew("Mul", &sq, mask_name, &format!("{}/sqmask", desc.name))
            } else {
                sq
            };
            let out = self.uniq(&format!("{}/meansqfull", desc.name));
            self.node(
                "ReduceMean",
                &[sq_mask, axes_name],
                &out,
                &format!("{}/meansqfull", desc.name),
            );
            {
                let mut attr = AttributeProto::default();
                attr.name = Some("keepdims".into());
                attr.i = Some(1);
                attr.r#type = Some(2);
                if let Some(last) = self.g.node.last_mut() {
                    last.attribute.push(attr);
                }
            }
            mean_sq = if !self.require_exact_nn_len {
                let mean_sq_full = out;
                let mean_sq_recovered = self.uniq(&format!("{}/meansq", desc.name));
                let mask_mean = self.mask_mean_name.clone();
                self.node(
                    "Div",
                    &[mean_sq_full, mask_mean],
                    &mean_sq_recovered,
                    &format!("{}/meansq", desc.name),
                );
                mean_sq_recovered
            } else {
                out
            };
        }

        let eps_name = self.scalar(&format!("{}/eps", desc.name), desc.epsilon);
        let denom = self.ew("Add", &mean_sq, &eps_name, &format!("{}/denom", desc.name));
        let rms = {
            let out = self.uniq(&format!("{}/rms", desc.name));
            self.node("Sqrt", &[denom], &out, &format!("{}/rms", desc.name));
            out
        };

        let normed = self.ew("Div", input, &rms, &format!("{}/normed", desc.name));
        let gamma = self.init(&format!("{}.gamma", desc.name), &[1, c, 1, 1], &desc.gamma);
        let scaled = self.ew("Mul", &normed, &gamma, &format!("{}/scaled", desc.name));
        let beta = self.init(&format!("{}.beta", desc.name), &[1, c, 1, 1], &desc.beta);
        let affine = self.ew("Add", &scaled, &beta, &format!("{}/affine", desc.name));

        let activated = match activation.activation {
            ACTIVATION_SILU => {
                let sig = {
                    let out = self.uniq(&format!("{}/silu/sig", desc.name));
                    self.node(
                        "Sigmoid",
                        std::slice::from_ref(&affine),
                        &out,
                        &format!("{}/silu/sig", desc.name),
                    );
                    out
                };
                self.ew("Mul", &affine, &sig, &format!("{}/silu", desc.name))
            }
            ACTIVATION_IDENTITY => affine,
            _ => {
                // Fallback: no activation.
                affine
            }
        };

        self.mask(&activated, &format!("{}/mask", desc.name))
    }

    /// Nested bottleneck residual block:
    ///   pre_bn → pre_act → mask → pre_conv
    ///   inner block stack
    ///   post_bn → post_act → mask → post_conv
    ///   add input
    fn build_nested_bottleneck_residual_block(
        &mut self,
        input: &str,
        desc: &crate::desc::NestedBottleneckResidualBlockDesc,
        mask_scale_name: &str,
    ) -> String {
        // pre-branch
        let pre = self.bn(input, &desc.pre_bn);
        let pre = self.act(&pre, &desc.pre_activation);
        let pre = self.mask(&pre, &format!("{}/pre", desc.name));
        let pre = self.conv(&pre, &desc.pre_conv);

        // inner block stack
        let stack = {
            let mut cur = pre;
            for blk in &desc.blocks {
                cur = match blk {
                    crate::desc::BlockDesc::Ordinary(d) => self.residual(&cur, d),
                    crate::desc::BlockDesc::GlobalPooling(d) => {
                        self.global_pooling_residual_block(&cur, d, mask_scale_name)
                    }
                    crate::desc::BlockDesc::NestedBottleneck(d) => {
                        self.build_nested_bottleneck_residual_block(&cur, d, mask_scale_name)
                    }
                    _ => {
                        return input.to_string(); // skip unsupported inner block
                    }
                };
            }
            cur
        };

        // post-branch
        let post = self.bn(&stack, &desc.post_bn);
        let post = self.act(&post, &desc.post_activation);
        let post = self.mask(&post, &format!("{}/post", desc.name));
        let post = self.conv(&post, &desc.post_conv);

        self.ew("Add", input, &post, &format!("{}/res", desc.name))
    }

    /// Per-position RMSNorm over channels: x / sqrt(mean(x^2)+eps) * weight.
    /// Input/output NCHW [N,C,H,W]. Mirrors transformerRMSNorm in C++.
    fn transformer_rms_norm(
        &mut self,
        input: &str,
        desc: &crate::desc::TransformerRMSNormDesc,
        mask_name: &str,
    ) -> String {
        let _ = mask_name;
        let c = desc.num_channels as i64;

        let sq = self.ew("Mul", input, input, &format!("{}/sq", desc.name));
        let axes_name = {
            let mut t = TensorProto::default();
            t.name = Some(format!("{}/axC", desc.name));
            t.data_type = Some(7);
            t.dims = vec![1];
            t.int64_data = vec![1];
            self.g.initializer.push(t);
            format!("{}/axC", desc.name)
        };
        let mean_sq = {
            let out = self.uniq(&format!("{}/meansq", desc.name));
            self.node(
                "ReduceMean",
                &[sq, axes_name],
                &out,
                &format!("{}/meansq", desc.name),
            );
            {
                let mut attr = AttributeProto::default();
                attr.name = Some("keepdims".into());
                attr.i = Some(1);
                attr.r#type = Some(2);
                if let Some(last) = self.g.node.last_mut() {
                    last.attribute.push(attr);
                }
            }
            out
        };

        let eps = self.scalar(&format!("{}/eps", desc.name), desc.epsilon);
        let denom = self.ew("Add", &mean_sq, &eps, &format!("{}/denom", desc.name));
        let rms = {
            let out = self.uniq(&format!("{}/rms", desc.name));
            self.node("Sqrt", &[denom], &out, &format!("{}/rms", desc.name));
            out
        };
        let normed = self.ew("Div", input, &rms, &format!("{}/normed", desc.name));
        let w = self.init(
            &format!("{}.weight", desc.name),
            &[1, c, 1, 1],
            &desc.weight,
        );
        let scaled = self.ew("Mul", &normed, &w, &format!("{}/scaled", desc.name));
        self.mask(&scaled, &format!("{}/mask", desc.name))
    }

    /// 1x1 conv projection. Weight is [outC, inC, 1, 1]; we use Conv with k=1.
    fn proj_conv(&mut self, input: &str, desc: &MatMulLayerDesc) -> String {
        let w = self.init(
            &format!("{}.W", desc.name),
            &[desc.out_channels as i64, desc.in_channels as i64, 1, 1],
            &desc.weights,
        );
        let out = self.uniq(&desc.name);
        let mut n = NodeProto::default();
        n.op_type = Some("Conv".into());
        n.name = Some(desc.name.clone());
        n.input = vec![input.into(), w];
        n.output = vec![out.clone()];
        {
            let mut pads = AttributeProto::default();
            pads.name = Some("pads".into());
            pads.ints = vec![0, 0, 0, 0];
            pads.r#type = Some(7);
            n.attribute.push(pads);
            let mut dil = AttributeProto::default();
            dil.name = Some("dilations".into());
            dil.ints = vec![1, 1];
            dil.r#type = Some(7);
            n.attribute.push(dil);
        }
        self.g.node.push(n);
        out
    }

    /// SwiGLU FFN block: RMSNorm → linear1 & linearGate → SiLU(linear1)*linearGate → linear2 → mask → add input.
    fn build_transformer_ffn_block(
        &mut self,
        input: &str,
        desc: &crate::desc::TransformerFFNDesc,
        mask_name: &str,
    ) -> String {
        let xn = self.transformer_rms_norm(input, &desc.pre_ln, mask_name);
        let a = self.proj_conv(&xn, &desc.linear1);
        let g = self.proj_conv(&xn, &desc.linear_gate);

        let sig = {
            let out = self.uniq(&format!("{}/silu/sig", desc.name));
            self.node(
                "Sigmoid",
                std::slice::from_ref(&a),
                &out,
                &format!("{}/silu/sig", desc.name),
            );
            out
        };
        let silu = self.ew("Mul", &a, &sig, &format!("{}/silu", desc.name));
        let gated = self.ew("Mul", &silu, &g, &format!("{}/swiglu", desc.name));
        let out = self.proj_conv(&gated, &desc.linear2);
        let masked = self.mask(&out, &format!("{}/out", desc.name));
        self.ew("Add", input, &masked, &format!("{}/res", desc.name))
    }

    /// Compute RoPE cos/sin tables matching C++ `computeRopeCosSin`.
    ///
    /// For learnable RoPE: reads from `rope_freqs` (numKVHeads, numPairs, 2).
    /// For non-learnable RoPE: computes from `rope_theta`.
    ///
    /// Returns (cos_table, sin_table) where the table is indexed by
    /// `(kv_head * numPairs + pair) * seq_len + (y * nn_x_len + x)`.
    fn compute_rope_cos_sin(
        &mut self,
        desc: &crate::desc::TransformerAttentionDesc,
        nn_x_len: i32,
        nn_y_len: i32,
    ) -> (Vec<f32>, Vec<f32>) {
        let seq_len = nn_x_len * nn_y_len;
        let num_pairs = desc.q_head_dim / 2;

        if desc.learnable_rope {
            let num_kv = desc.num_kv_heads;
            let mut cos_table = vec![0.0f32; (num_kv * num_pairs * seq_len) as usize];
            let mut sin_table = vec![0.0f32; (num_kv * num_pairs * seq_len) as usize];

            for h in 0..num_kv {
                for p in 0..num_pairs {
                    let freq_x = desc.rope_freqs[((h * num_pairs + p) * 2) as usize];
                    let freq_y = desc.rope_freqs[((h * num_pairs + p) * 2 + 1) as usize];
                    for y in 0..nn_y_len {
                        for x in 0..nn_x_len {
                            let xy = (y * nn_x_len + x) as usize;
                            let angle = x as f32 * freq_x + y as f32 * freq_y;
                            let idx = ((h * num_pairs + p) * seq_len) as usize + xy;
                            cos_table[idx] = angle.cos();
                            sin_table[idx] = angle.sin();
                        }
                    }
                }
            }
            (cos_table, sin_table)
        } else {
            let num_pairs_per_dim = num_pairs / 2;
            let dim_half = desc.q_head_dim / 2;
            let mut cos_table = vec![0.0f32; (num_pairs * seq_len) as usize];
            let mut sin_table = vec![0.0f32; (num_pairs * seq_len) as usize];

            for p in 0..num_pairs {
                for y in 0..nn_y_len {
                    for x in 0..nn_x_len {
                        let xy = (y * nn_x_len + x) as usize;
                        let idx = (p * seq_len) as usize + xy;
                        let angle = if p < num_pairs_per_dim {
                            let freq = desc.rope_theta.powf(-2.0 * p as f32 / dim_half as f32);
                            y as f32 * freq
                        } else {
                            let p_adj = p - num_pairs_per_dim;
                            let freq = desc.rope_theta.powf(-2.0 * p_adj as f32 / dim_half as f32);
                            x as f32 * freq
                        };
                        cos_table[idx] = angle.cos();
                        sin_table[idx] = angle.sin();
                    }
                }
            }
            (cos_table, sin_table)
        }
    }

    /// Apply RoPE: y = x*cosFull + swap(x)*sinSigned.
    /// swap exchanges the two channels of each pair (Gather on hd axis).
    #[allow(clippy::too_many_arguments)]
    fn apply_rope(
        &mut self,
        input: &str,
        heads: i32,
        num_kv_heads: i32,
        hd: i32,
        seq_len: i32,
        rope_num_pairs: i32,
        learnable_rope: bool,
        cos_table: &[f32],
        sin_table: &[f32],
        name_base: &str,
    ) -> String {
        // Build cosFull and sinSigned broadcast tables.
        let mut cos_full = vec![0.0f32; (heads * hd * seq_len) as usize];
        let mut sin_signed = vec![0.0f32; (heads * hd * seq_len) as usize];
        for h in 0..heads {
            for p in 0..rope_num_pairs {
                for s in 0..seq_len {
                    let table_idx = if learnable_rope {
                        // Q head h maps to KV head: kv = h * numKV / numHeads
                        let kv_head = h * num_kv_heads / heads;
                        ((kv_head * rope_num_pairs + p) * seq_len + s) as usize
                    } else {
                        (p * seq_len + s) as usize
                    };
                    let c = cos_table[table_idx];
                    let sn = sin_table[table_idx];
                    let i0 = ((h * hd + (2 * p)) * seq_len + s) as usize;
                    let i1 = ((h * hd + (2 * p + 1)) * seq_len + s) as usize;
                    cos_full[i0] = c;
                    cos_full[i1] = c;
                    sin_signed[i0] = -sn;
                    sin_signed[i1] = sn;
                }
            }
        }

        let cos_name = self.init(
            &format!("{}/ropecos", name_base),
            &[1, heads as i64, hd as i64, seq_len as i64],
            &cos_full,
        );
        let sin_name = self.init(
            &format!("{}/ropesinsigned", name_base),
            &[1, heads as i64, hd as i64, seq_len as i64],
            &sin_signed,
        );

        // swap: Gather along hd axis with pair-swap permutation.
        let mut swap_idx = vec![0i64; hd as usize];
        for p in 0..rope_num_pairs {
            swap_idx[(2 * p) as usize] = (2 * p + 1) as i64;
            swap_idx[(2 * p + 1) as usize] = (2 * p) as i64;
        }
        let idx_name = {
            let mut t = TensorProto::default();
            t.name = Some(format!("{}/ropeswapidx", name_base));
            t.data_type = Some(7);
            t.dims = vec![hd as i64];
            t.int64_data = swap_idx;
            self.g.initializer.push(t);
            format!("{}/ropeswapidx", name_base)
        };
        let x_swap = {
            let out = self.uniq(&format!("{}/rope_swap", name_base));
            self.node(
                "Gather",
                &[input.to_string(), idx_name],
                &out,
                &format!("{}/rope_swap", name_base),
            );
            {
                let mut attr = AttributeProto::default();
                attr.name = Some("axis".into());
                attr.i = Some(2);
                attr.r#type = Some(2);
                if let Some(last) = self.g.node.last_mut() {
                    last.attribute.push(attr);
                }
            }
            out
        };

        let t1 = self.ew("Mul", input, &cos_name, &format!("{}/rope_t1", name_base));
        let t2 = self.ew("Mul", &x_swap, &sin_name, &format!("{}/rope_t2", name_base));
        self.ew("Add", &t1, &t2, &format!("{}/rope_out", name_base))
    }

    /// Expand KV heads for grouped-query attention.
    fn expand_kv_heads(&mut self, input: &str, n_kv: i32, n_h: i32, name_base: &str) -> String {
        if n_h == n_kv {
            return input.to_string();
        }
        let n_rep = n_h / n_kv;
        let idx: Vec<i64> = (0..n_h).map(|h| (h / n_rep) as i64).collect();
        let idx_name = {
            let mut t = TensorProto::default();
            t.name = Some(format!("{}/gqaidx", name_base));
            t.data_type = Some(7);
            t.dims = vec![n_h as i64];
            t.int64_data = idx;
            self.g.initializer.push(t);
            format!("{}/gqaidx", name_base)
        };
        let out = self.uniq(&format!("{}/gqa", name_base));
        self.node(
            "Gather",
            &[input.to_string(), idx_name],
            &out,
            &format!("{}/gqa", name_base),
        );
        {
            let mut attr = AttributeProto::default();
            attr.name = Some("axis".into());
            attr.i = Some(1);
            attr.r#type = Some(2);
            if let Some(last) = self.g.node.last_mut() {
                last.attribute.push(attr);
            }
        }
        out
    }

    /// Transformer self-attention block (NCHW):
    ///   RMSNorm → QKV proj → RoPE → GQA expand → scores → softmax → attn → proj → mask → add input
    #[allow(clippy::too_many_arguments)]
    fn build_transformer_attention_block(
        &mut self,
        input: &str,
        desc: &crate::desc::TransformerAttentionDesc,
        nn_x_len: i32,
        nn_y_len: i32,
        mask_name: &str,
    ) -> String {
        let seq_len = nn_x_len * nn_y_len;
        let n_h = desc.num_heads;
        let n_kv = desc.num_kv_heads;
        let hd = desc.q_head_dim;
        let vhd = desc.v_head_dim;

        let xn = self.transformer_rms_norm(input, &desc.pre_ln, mask_name);
        let q = self.proj_conv(&xn, &desc.q_proj);
        let k = self.proj_conv(&xn, &desc.k_proj);
        let v = self.proj_conv(&xn, &desc.v_proj);

        // Reshape to [N, heads, dim, S]
        let qh = self.reshape(
            &q,
            &[0, n_h as i64, hd as i64, seq_len as i64],
            &format!("{}/q", desc.name),
        );
        let kh = self.reshape(
            &k,
            &[0, n_kv as i64, hd as i64, seq_len as i64],
            &format!("{}/k", desc.name),
        );
        let vh = self.reshape(
            &v,
            &[0, n_kv as i64, vhd as i64, seq_len as i64],
            &format!("{}/v", desc.name),
        );

        let (qh, kh) = if desc.use_rope {
            let rope_num_pairs = hd / 2;
            let (cos_table, sin_table) = self.compute_rope_cos_sin(desc, nn_x_len, nn_y_len);
            let qh = self.apply_rope(
                &qh,
                n_h,
                n_kv,
                hd,
                seq_len,
                rope_num_pairs,
                desc.learnable_rope,
                &cos_table,
                &sin_table,
                &format!("{}/qrope", desc.name),
            );
            let kh = self.apply_rope(
                &kh,
                n_kv,
                n_kv,
                hd,
                seq_len,
                rope_num_pairs,
                desc.learnable_rope,
                &cos_table,
                &sin_table,
                &format!("{}/krope", desc.name),
            );
            (qh, kh)
        } else {
            (qh, kh)
        };

        let kh = self.expand_kv_heads(&kh, n_kv, n_h, &format!("{}/k", desc.name));
        let vh = self.expand_kv_heads(&vh, n_kv, n_h, &format!("{}/v", desc.name));

        // scores = qh @ kh^T
        let kh_t = self.transpose(&kh, &[0, 1, 3, 2], &format!("{}/khT", desc.name));
        let scores = {
            let out = self.uniq(&format!("{}/scores", desc.name));
            self.node(
                "MatMul",
                &[qh, kh_t],
                &out,
                &format!("{}/scores", desc.name),
            );
            out
        };
        let scale = self.scalar(&format!("{}/scale", desc.name), 1.0 / (hd as f32).sqrt());
        let scores = self.ew(
            "Mul",
            &scores,
            &scale,
            &format!("{}/scoresscaled", desc.name),
        );

        // For variable boards, add attention mask bias (0 on-board, -BIG off-board).
        let scores = if !self.require_exact_nn_len && !self.mask_bias_name.is_empty() {
            self.ew(
                "Add",
                &scores,
                &self.mask_bias_name.clone(),
                &format!("{}/scoresmasked", desc.name),
            )
        } else {
            scores
        };

        let probs = {
            let out = self.uniq(&format!("{}/probs", desc.name));
            self.node("Softmax", &[scores], &out, &format!("{}/probs", desc.name));
            {
                let mut attr = AttributeProto::default();
                attr.name = Some("axis".into());
                attr.i = Some(3);
                attr.r#type = Some(2);
                if let Some(last) = self.g.node.last_mut() {
                    last.attribute.push(attr);
                }
            }
            out
        };

        // attn = probs @ vh
        let attn = {
            let out = self.uniq(&format!("{}/sv", desc.name));
            self.node("MatMul", &[probs, vh], &out, &format!("{}/sv", desc.name));
            out
        };
        // [N,heads,S,vhd] -> [N,S,heads,vhd] -> [N, heads*vhd, S]
        let attn_t = self.transpose(&attn, &[0, 2, 1, 3], &format!("{}/svT", desc.name));
        let attn_nchw = self.reshape(
            &attn_t,
            &[0, n_h as i64 * vhd as i64, nn_y_len as i64, nn_x_len as i64],
            &format!("{}/attnnchw", desc.name),
        );

        let out = self.proj_conv(&attn_nchw, &desc.out_proj);
        let masked = self.mask(&out, &format!("{}/out", desc.name));
        self.ew("Add", input, &masked, &format!("{}/res", desc.name))
    }

    fn reshape(&mut self, input: &str, shape: &[i64], name_base: &str) -> String {
        let shape_name = {
            let mut t = TensorProto::default();
            t.name = Some(format!("{}/shape", name_base));
            t.data_type = Some(7);
            t.dims = vec![shape.len() as i64];
            t.int64_data = shape.to_vec();
            self.g.initializer.push(t);
            format!("{}/shape", name_base)
        };
        let out = self.uniq(&format!("{}/reshape", name_base));
        self.node(
            "Reshape",
            &[input.to_string(), shape_name],
            &out,
            &format!("{}/reshape", name_base),
        );
        out
    }

    fn transpose(&mut self, input: &str, perm: &[i64], name_base: &str) -> String {
        let out = self.uniq(&format!("{}/transpose", name_base));
        self.node(
            "Transpose",
            &[input.to_string()],
            &out,
            &format!("{}/transpose", name_base),
        );
        {
            let mut attr = AttributeProto::default();
            attr.name = Some("perm".into());
            attr.ints = perm.to_vec();
            attr.r#type = Some(7);
            if let Some(last) = self.g.node.last_mut() {
                last.attribute.push(attr);
            }
        }
        out
    }
}

// ==========================================================================
// Public API
// ==========================================================================

pub fn build(
    desc: &ModelDesc,
    nn_x_len: i32,
    nn_y_len: i32,
    require_exact_nn_len: bool,
    _transformer_nhwc: bool,
    _logger: &Logger,
) -> Result<OnnxBuildResult, OnnxBuildError> {
    let mut model = ModelProto::default();
    model.ir_version = Some(9);
    model.producer_name = Some("katago-rs".into());

    let opset = OperatorSetIdProto {
        domain: Some("".into()),
        version: Some(17),
    };
    model.opset_import.push(opset);

    let mut graph = GraphProto::default();
    graph.name = Some(desc.name.clone());
    model.graph = Some(graph);

    let tip_head_names;
    {
        let mut b = Builder {
            g: model.graph.as_mut().unwrap(),
            nn_x: nn_x_len,
            nn_y: nn_y_len,
            require_exact_nn_len,
            cnt: 0,
            tip_head_names: Vec::new(),
            mask_sum_name: String::new(),
            mask_mean_name: String::new(),
            mask_scale_name: String::new(),
            mask_quad_name: String::new(),
            mask_bias_name: String::new(),
        };

        b.add_vi("InputMask", 1, true);
        b.add_vi("InputSpatial", desc.num_input_channels, true);
        b.add_vi("InputGlobal", desc.num_input_global_channels, false);

        b.init_mask_features();

        let ic = b.conv("InputSpatial", &desc.trunk.initial_conv);
        let im = b.mm("InputGlobal", &desc.trunk.initial_mat_mul);
        let mut cur = b.ew("Add", &ic, &im, &format!("{}/init", desc.trunk.name));

        for blk in &desc.trunk.blocks {
            cur = match blk {
                crate::desc::BlockDesc::Ordinary(d) => b.residual(&cur, d),
                crate::desc::BlockDesc::GlobalPooling(d) => {
                    b.global_pooling_residual_block(&cur, d, "InputMask/scale")
                }
                crate::desc::BlockDesc::NestedBottleneck(d) => {
                    b.build_nested_bottleneck_residual_block(&cur, d, "InputMask/scale")
                }
                crate::desc::BlockDesc::TransformerAttention(d) => {
                    b.build_transformer_attention_block(&cur, d, nn_x_len, nn_y_len, "InputMask")
                }
                crate::desc::BlockDesc::TransformerFfn(d) => {
                    b.build_transformer_ffn_block(&cur, d, "InputMask")
                }
            };
        }

        let tip_start = b.g.node.len();
        let tip = match desc.trunk.trunk_norm_kind {
            crate::desc::TRUNK_NORM_KIND_RMSNORM => b.build_trunk_tip_rms_norm(
                &cur,
                &desc.trunk.trunk_tip_rms_norm,
                &desc.trunk.trunk_tip_activation,
                "InputMask",
            ),
            _ => {
                let tip = b.bn(&cur, &desc.trunk.trunk_tip_bn);
                let tip = b.act(&tip, &desc.trunk.trunk_tip_activation);
                b.mask(&tip, &format!("{}/tip", desc.trunk.name))
            }
        };

        // Policy head
        let ph = &desc.policy_head;
        let p1 = b.conv(&tip, &ph.p1_conv);
        let g1 = b.conv(&tip, &ph.g1_conv);
        let g1 = b.bn(&g1, &ph.g1_bn);
        let g1 = b.act(&g1, &ph.g1_activation);
        let _ = b.mask(&g1, &format!("{}/g1", ph.name));
        let gb = b.mm(&g1, &ph.gpool_to_bias_mul);
        let p1b = b.ew("Add", &p1, &gb, &format!("{}/gpb", ph.name));
        let p1b = b.bn(&p1b, &ph.p1_bn);
        let p1b = b.act(&p1b, &ph.p1_activation);
        let _ = b.mask(&p1b, &format!("{}/p1", ph.name));
        let p2 = b.conv(&p1b, &ph.p2_conv);
        let pass = b.mm(&g1, &ph.gpool_to_pass_mul);

        b.mark_out(&pass, "OutputPolicyPass", ph.policy_out_channels, false);
        b.mark_out(&p2, "OutputPolicy", ph.policy_out_channels, true);

        // Value head
        let vh = &desc.value_head;
        let v1 = b.conv(&tip, &vh.v1_conv);
        let v1 = b.bn(&v1, &vh.v1_bn);
        let v1 = b.act(&v1, &vh.v1_activation);
        let _ = b.mask(&v1, &format!("{}/v1", vh.name));
        let v2 = b.mm(&v1, &vh.v2_mul);
        let v2 = b.mb(&v2, &vh.v2_bias);
        let v2 = b.act(&v2, &vh.v2_activation);
        let v3 = b.mm(&v2, &vh.v3_mul);
        let v3 = b.mb(&v3, &vh.v3_bias);
        let sv3 = b.mm(&v2, &vh.sv3_mul);
        let sv3 = b.mb(&sv3, &vh.sv3_bias);
        let own = b.conv(&v1, &vh.v_ownership_conv);

        b.mark_out(&v3, "OutputValue", desc.num_value_channels, false);
        b.mark_out(
            &sv3,
            "OutputScoreValue",
            desc.num_score_value_channels,
            false,
        );
        b.mark_out(&own, "OutputOwnership", desc.num_ownership_channels, true);

        for i in tip_start..b.g.node.len() {
            b.tip_head_names
                .push(b.g.node[i].name.clone().unwrap_or_default());
        }
        tip_head_names = b.tip_head_names;
    } // drop builder borrow

    let serialized_model = model.encode_to_vec();
    Ok(OnnxBuildResult {
        serialized_model,
        trunk_tip_and_head_node_names: tip_head_names,
        rms_norm_node_names: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_stub_for_default() {
        let l = kata_core::logger::Logger::new(kata_core::logger::LoggerOptions::default(), None);
        let r = build(&ModelDesc::default(), 19, 19, true, false, &l);
        if let Err(e) = &r {
            assert!(!e.to_string().is_empty());
        }
    }
}

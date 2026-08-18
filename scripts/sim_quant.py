#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""混合精度端到端精度模拟（不碰 CUDA，纯 numpy 语义级）：

对 b11fix.onnx 的 231 个 trunk MatMul（动态激活 × 静态权重）做
量化-反量化（QD）模拟，比较三种处理后的网络输出与 FP32 基线：
  - fp8w    : 仅权重 E4M3（per-output-channel scale）
  - fp8full : 权重 E4M3 per-channel + 激活 E4M3 per-token（动态 amax）
  - int8full: 权重 int8 per-channel + 激活 int8 per-token（动态 amax）
指标：policy 通道0 top-1 一致率、policy KL、value/score 最大偏差。
输入用 dump_nn_io_cuda 的 16 个确定性局面（与 CUDA 对拍同数据）。

用法：.venv/Scripts/python.exe scripts/sim_quant.py [dump_dir]
"""
import sys
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

DUMP = sys.argv[1] if len(sys.argv) > 1 else "crates/kata_nn/target/nn_io_dump_cuda"
MODEL = "D:/code/b11fix.onnx"
NPOS = 16


# ---------- E4M3 ----------
def e4m3_round(x):
    """numpy 向量化 E4M3FN 量化-反量化（round-to-nearest-even 近似用 round）。"""
    x = np.asarray(x, dtype=np.float32)
    sign = np.sign(x)
    ax = np.abs(x)
    ax = np.minimum(ax, 448.0)
    out = np.zeros_like(ax)
    # 规格化区间 [2^-6, 448]
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.maximum(ax, 1e-30)))
    e = np.clip(e, -6, 8)
    m = ax / np.power(2.0, e)  # [1,2)
    mant = np.round((m - 1.0) * 8.0)
    mant = np.clip(mant, 0, 7)
    # mant 进位（m≈2 → 2.0 表示为下一指数）
    carry = mant > 7.5
    e2 = np.where(carry, e + 1, e)
    mant2 = np.where(carry, 0.0, mant)
    normal = (1.0 + mant2 / 8.0) * np.power(2.0, e2)
    # subnormal: ax < 2^-6 → 步长 2^-9
    sub = np.round(ax / 2**-9) * 2**-9
    sub = np.minimum(sub, 2**-6)
    out = np.where(ax >= 2**-6, normal, sub)
    # 顶部钳到 448
    out = np.minimum(out, 448.0)
    out[ax == 0] = 0.0
    return sign * out


def qdq_rows(x, levels, maxval):
    """按行（token）动态 amax 量化-反量化。levels=量化级数(448 或 127)。"""
    amax = np.abs(x).max(axis=-1, keepdims=True)
    scale = np.maximum(amax, 1e-12) / maxval
    q = np.round(x / scale)
    q = np.clip(q, -maxval, maxval)
    return q * scale


class E4M3Act(OpRun):
    op_domain = "custom"

    def _run(self, x):
        amax = np.abs(x).max(axis=-1, keepdims=True)
        scale = np.maximum(amax, 1e-12) / 448.0
        return (e4m3_round(x / scale) * scale,)


class INT8Act(OpRun):
    op_domain = "custom"

    def _run(self, x):
        return (qdq_rows(x, 127, 127).astype(np.float32),)


# 静态 per-tensor scale 校准/应用（官方 ff27077 activation 语义：max 校准、
# 整张量一个 scale、推理期固定）。每个插入点一个独立 op_type + 类实例。
_CALIB_AMAX = {}


def make_amax_probe(idx):
    op_type = f"AmaxProbe{idx}"

    class Probe(OpRun):
        op_domain = "custom"

        def _run(self, x):
            m = float(np.abs(x).max())
            if m > _CALIB_AMAX.get(idx, 0.0):
                _CALIB_AMAX[idx] = m
            return (x,)

    Probe.__name__ = op_type
    return op_type, Probe


def make_static_e4m3(idx, amax_value):
    """官方静态 per-tensor scale:amax_value 在构建期烧入类（非运行期查表,
    便于多 scope 校准共存）。"""
    op_type = f"StaticE4M3{idx}"
    scale = max(float(amax_value), 1e-12) / 448.0

    class Q(OpRun):
        op_domain = "custom"

        def _run(self, x):
            return (e4m3_round(x / scale) * scale,)

    Q.__name__ = op_type
    return op_type, Q


def quantize_weights_perchannel(w, kind):
    """w: [N, K] 按输出通道（行）量化-反量化。"""
    amax = np.abs(w).max(axis=-1, keepdims=True)
    if kind == "e4m3":
        scale = np.maximum(amax, 1e-12) / 448.0
        return e4m3_round(w / scale) * scale
    else:
        scale = np.maximum(amax, 1e-12) / 127.0
        q = np.clip(np.round(w / scale), -127, 127)
        return (q * scale).astype(np.float32)


def quantize_weights_peroutchannel(w, kind):
    """w: [K, N]（ONNX MatMul 标准布局）按输出通道（N，即列/axis=0 归约）
    量化-反量化——官方 ff27077 的 per-output-channel 语义。

    注意:b11fix 实测 initializer 即 [K,N]（(384,1152)=gate/up、(1152,384)=
    ffn down），旧 quantize_weights_perchannel 的按行量化实为
    per-input-channel，轴是错的；保留旧函数仅为复现历史数据。"""
    amax = np.abs(w).max(axis=0, keepdims=True)
    if kind == "e4m3":
        scale = np.maximum(amax, 1e-12) / 448.0
        return e4m3_round(w / scale) * scale
    else:
        scale = np.maximum(amax, 1e-12) / 127.0
        q = np.clip(np.round(w / scale), -127, 127)
        return (q * scale).astype(np.float32)


def build_variant(model, kind, static_scales=None):
    """kind: calib / calib_ffn / fp8w / fp8full / int8full / fp8w_off /
    fp8_official / fp8_ffn_official / fp8w_ffn_perK。
    - calib*      : 仅插 amax 探针（FP32 网络），跑校准 pass 填 _CALIB_AMAX。
    - *_ffn_*     : scope 仅 FFN 两类（(384,1152) gate/up、(1152,384) down），
                    attention (384,384) 保持 FP32（隔离实验:注意力类对量化
                    轴敏感且误差大 6.7x,FFN 类几乎免费）。
    - *_off       : 权重 per-output-channel（[K,N] 列向,官方语义）。
    - *official   : 另加激活静态 per-tensor E4M3（官方 ff27077 全语义,
                    static_scales = 校准快照,idx 与同 scope 校准 pass 对齐）。
    其余为历史变体（量化轴 per-input,保留复现）。"""
    import copy

    m = copy.deepcopy(model)
    g = m.graph
    init = {i.name for i in g.initializer}
    wname2init = {i.name: i for i in g.initializer}

    def wdims(n):
        for inp in n.input[1:]:
            if inp in init:
                return tuple(wname2init[inp].dims)
        return None

    # 目标：动态激活 × 静态权重的 MatMul。scope:ffn 变体仅 FFN 两类
    # （隔离实验 scripts/e1_axis_isolate.py 的结论），否则 trunk 全三类
    # （= 官方 --scope transformer 的 q/k/v/out_proj/ffn_linear1/gate/linear2）。
    if "ffn" in kind:
        trunk_shapes = {(384, 1152), (1152, 384)}
    else:
        trunk_shapes = {(384, 384), (384, 1152), (1152, 384)}
    cnt = 0
    new_nodes = []
    custom_ops = []
    for n in g.node:
        if (
            n.op_type == "MatMul"
            and n.input[0] not in init
            and n.input[1] in init
            and wdims(n) in trunk_shapes
        ):
            cnt += 1
            if kind.startswith("calib"):
                op_type, cls = make_amax_probe(cnt)
                pname = n.input[0] + f"__probe_{cnt}"
                new_nodes.append(
                    onnx.helper.make_node(op_type, [n.input[0]], [pname], domain="custom")
                )
                n.input[0] = pname
                custom_ops.append(cls)
            else:
                # 权重量化
                wi = wname2init[n.input[1]]
                w = onnx.numpy_helper.to_array(wi)
                if kind.endswith("_off") or "official" in kind:
                    wq = quantize_weights_peroutchannel(w, "e4m3" if kind.startswith("fp8") else "int8")
                else:
                    wq = quantize_weights_perchannel(w, "e4m3" if kind.startswith("fp8") else "int8")
                wi.CopyFrom(onnx.numpy_helper.from_array(wq.astype(np.float32), wi.name))
                # 激活量化
                if kind == "fp8full" or kind == "int8full":
                    op = "E4M3Act" if kind.startswith("fp8") else "INT8Act"
                    qname = n.input[0] + f"__{op}_{cnt}"
                    qn = onnx.helper.make_node(op, [n.input[0]], [qname], domain="custom")
                    n.input[0] = qname
                    new_nodes.append(qn)
                elif "official" in kind:
                    assert static_scales is not None and cnt in static_scales, \
                        f"{kind}: 缺 idx={cnt} 的校准 scale(校准 pass scope 须与变体一致)"
                    op_type, cls = make_static_e4m3(cnt, static_scales[cnt])
                    qname = n.input[0] + f"__sq_{cnt}"
                    new_nodes.append(
                        onnx.helper.make_node(op_type, [n.input[0]], [qname], domain="custom")
                    )
                    n.input[0] = qname
                    custom_ops.append(cls)
        new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    # 声明 custom domain（ReferenceEvaluator 要求）
    if custom_ops and not any(o.domain == "custom" for o in m.opset_import):
        m.opset_import.append(onnx.helper.make_opsetid("custom", 1))
    print(f"  {kind}: {cnt} 个 trunk MatMul 已改写", file=sys.stderr)
    return m, custom_ops


def run_model(sess, sp, gl):
    return sess.run(
        None, {"input_spatial": sp[None], "input_global": gl[None]}
    )


def main():
    print(f"加载模型 {MODEL} ...", file=sys.stderr)
    model = onnx.load(MODEL)
    print("加载 dump 局面 ...", file=sys.stderr)
    pos = []
    for i in range(NPOS):
        sp = np.fromfile(f"{DUMP}/pos{i}_spatial.bin", dtype="<f4").reshape(22, 19, 19)
        gl = np.fromfile(f"{DUMP}/pos{i}_global.bin", dtype="<f4").reshape(19)
        pos.append((sp.astype(np.float32), gl.astype(np.float32)))

    print("跑 FP32 基线（ReferenceEvaluator，较慢）...", file=sys.stderr)
    base = ReferenceEvaluator(model)
    base_out = [run_model(base, sp, gl) for sp, gl in pos]

    def calibrate(calib_kind):
        """跑校准 pass 并返回快照（官方 max 校准;对 16 个 dump 局面取 max,
        与评估集同源 = 乐观界）。"""
        print(f"跑 amax 校准 pass({calib_kind}) ...", file=sys.stderr)
        _CALIB_AMAX.clear()
        cm, cops = build_variant(model, calib_kind)
        sess = ReferenceEvaluator(cm, new_ops=cops)
        for sp, gl in pos:
            run_model(sess, sp, gl)
        snap = dict(_CALIB_AMAX)
        print(f"  {len(snap)} 个激活张量, median={np.median(list(snap.values())):.4f}",
              file=sys.stderr)
        return snap

    # kinds 可由 argv[2] 覆盖(逗号分隔)
    kinds = (sys.argv[2].split(",") if len(sys.argv) > 2
             else ["fp8_official", "fp8_ffn_official", "fp8w_ffn_perK"])
    scales_by_kind = {}
    for kind in kinds:
        if "official" in kind:
            scales_by_kind[kind] = calibrate("calib_ffn" if "ffn" in kind else "calib")
    variants = {}
    for kind in kinds:
        vm, custom_ops = build_variant(model, kind, scales_by_kind.get(kind))
        sess = ReferenceEvaluator(vm, new_ops=custom_ops)
        print(f"跑 {kind} ...", file=sys.stderr)
        variants[kind] = [run_model(sess, sp, gl) for sp, gl in pos]

    # 指标对比
    for kind, outs in variants.items():
        top1 = 0
        kl_sum = 0.0
        vdiff = 0.0
        sdiff = 0.0
        details = []
        for i in range(NPOS):
            pb = base_out[i][0][0, 0]  # [362] logits ch0
            pq = outs[i][0][0, 0]
            same = pb.argmax() == pq.argmax()
            if same:
                top1 += 1
            # KL(softmax base || softmax quant)
            def logsoftmax(x):
                x = x - x.max()
                return x - np.log(np.exp(x).sum())
            lb, lq = logsoftmax(pb), logsoftmax(pq)
            kl = float((np.exp(lb) * (lb - lq)).sum())
            kl_sum += kl
            vb = base_out[i][1][0]  # [3]
            vq = outs[i][1][0]
            vd = float(np.abs(vb - vq).max())
            vdiff = max(vdiff, vd)
            # win/loss softmax 空间误差（logits → prob）
            pb_prob = np.exp(vb - vb.max()); pb_prob /= pb_prob.sum()
            pq_prob = np.exp(vq - vq.max()); pq_prob /= pq_prob.sum()
            wd = float(np.abs(pb_prob - pq_prob).max())
            sb = base_out[i][2][0, 0]  # score mean
            sq = outs[i][2][0, 0]
            sd = float(abs(sb - sq))
            sdiff = max(sdiff, sd)
            margin = float(np.sort(pb)[-1] - np.sort(pb)[-2])
            details.append(
                f"  pos{i:2d} top1={'=' if same else 'X'} margin={margin:6.3f} "
                f"KL={kl:.2e} winprob_d={wd:.4f} score_d={sd:.4f}"
            )
        print(
            f"{kind:9s} top1={top1}/{NPOS}  policyKL_mean={kl_sum/NPOS:.4e}  "
            f"value_maxabs={vdiff:.4e}  score_maxabs={sdiff:.4e}"
        )
        for d in details:
            print(d)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E1 轴差异隔离实验:对三类 trunk MatMul 权重分别按 per-K(旧)/per-N(官方)
量化,逐形状类对比端到端误差,定位 fp8w_off 与 fp8w 的 5x KL 差异来源。

用法:.venv/Scripts/python.exe scripts/e1_axis_isolate.py [dump_dir]
"""
import sys
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, "scripts")
from sim_quant import (  # noqa: E402
    DUMP, MODEL, NPOS, E4M3Act,
    quantize_weights_perchannel, quantize_weights_peroutchannel,
)

SHAPE_CLASSES = ["384x384", "384x1152", "1152x384"]
CLASS_OF = {(384, 384): "384x384", (384, 1152): "384x1152", (1152, 384): "1152x384"}


def build(model, shape_class, axis):
    import copy
    m = copy.deepcopy(model)
    g = m.graph
    init = {i.name for i in g.initializer}
    w2i = {i.name: i for i in g.initializer}

    def wdims(n):
        for inp in n.input[1:]:
            if inp in init:
                return tuple(w2i[inp].dims)
        return None

    cnt = 0
    for n in g.node:
        if (n.op_type == "MatMul" and n.input[0] not in init and n.input[1] in init
                and CLASS_OF.get(wdims(n)) == shape_class):
            wi = w2i[n.input[1]]
            w = onnx.numpy_helper.to_array(wi)
            f = quantize_weights_perchannel if axis == "perK" else quantize_weights_peroutchannel
            wq = f(w, "e4m3")
            wi.CopyFrom(onnx.numpy_helper.from_array(wq.astype(np.float32), wi.name))
            cnt += 1
    return m, cnt


def metrics(base_out, outs):
    top1 = 0
    kl_sum = 0.0
    vmax = 0.0
    smax = 0.0
    for i in range(NPOS):
        pb = base_out[i][0][0, 0]
        pq = outs[i][0][0, 0]
        top1 += pb.argmax() == pq.argmax()

        def logsoftmax(x):
            x = x - x.max()
            return x - np.log(np.exp(x).sum())
        lb, lq = logsoftmax(pb), logsoftmax(pq)
        kl_sum += float((np.exp(lb) * (lb - lq)).sum())
        vmax = max(vmax, float(np.abs(base_out[i][1][0] - outs[i][1][0]).max()))
        smax = max(smax, float(abs(base_out[i][2][0, 0] - outs[i][2][0, 0])))
    return top1, kl_sum / NPOS, vmax, smax


def main():
    model = onnx.load(MODEL)
    pos = []
    for i in range(NPOS):
        sp = np.fromfile(f"{DUMP}/pos{i}_spatial.bin", dtype="<f4").reshape(22, 19, 19)
        gl = np.fromfile(f"{DUMP}/pos{i}_global.bin", dtype="<f4").reshape(19)
        pos.append((sp.astype(np.float32), gl.astype(np.float32)))
    base = ReferenceEvaluator(model)
    base_out = [base.run(None, {"input_spatial": sp[None], "input_global": gl[None]})
                for sp, gl in pos]
    print(f"{'variant':24s} {'n':>4s}  {'top1':>6s} {'KL_mean':>9s} {'vmax':>9s} {'smax':>8s}")
    for cls in SHAPE_CLASSES:
        for axis in ["perK", "perN"]:
            vm, cnt = build(model, cls, axis)
            sess = ReferenceEvaluator(vm)
            outs = [sess.run(None, {"input_spatial": sp[None], "input_global": gl[None]})
                    for sp, gl in pos]
            t, kl, v, s = metrics(base_out, outs)
            print(f"{cls + '/' + axis:24s} {cnt:4d}  {t:4d}/16 {kl:9.3e} {v:9.4f} {s:8.4f}",
                  file=sys.stderr)
            print(f"{cls + '/' + axis:24s} {cnt:4d}  {t:4d}/16 {kl:9.3e} {v:9.4f} {s:8.4f}")


if __name__ == "__main__":
    main()

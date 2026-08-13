#!/usr/bin/env python
"""Compare Rust KataGo TRT backend outputs against ONNX Runtime FP32 reference.

Usage: .venv/Scripts/python.exe scripts/compare_nn_output.py [dump_dir]
Exit code 0 = all gates passed.
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort


def load(path):
    return np.fromfile(path, dtype=np.float32)


def compare(name, rust, ref, gate):
    err = np.abs(rust - ref)
    max_abs = float(err.max())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ok = max_abs <= gate
    print(f"  {name:28s} max_abs={max_abs:.3e} rmse={rmse:.3e} gate={gate:.1e} {'OK' if ok else 'FAIL'}")
    return ok


def main():
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "target/nn_io_dump"
    meta = json.load(open(os.path.join(dump_dir, "meta.json"), encoding="utf-8"))
    model = meta["model"]
    n = meta["n"]

    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]

    all_ok = True
    policy_top1_ok = 0
    detailed_printed = False

    for i in range(n):
        spatial = load(f"{dump_dir}/pos{i}_spatial.bin").reshape(1, 22, 19, 19)
        global_ = load(f"{dump_dir}/pos{i}_global.bin").reshape(1, 19)
        feeds = {in_names[0]: spatial, in_names[1]: global_}
        outs = sess.run(out_names, feeds)
        out_map = dict(zip(out_names, outs))
        out_policy = out_map["out_policy"]
        out_value = out_map["out_value"]
        out_misc = out_map["out_miscvalue"]
        out_moremisc = out_map["out_moremiscvalue"]
        out_ownership = out_map["out_ownership"]

        rust_policy = load(f"{dump_dir}/pos{i}_policy.bin")     # 362 logits (optimism 0)
        rust_value = load(f"{dump_dir}/pos{i}_value.bin")       # 3
        rust_misc = load(f"{dump_dir}/pos{i}_misc.bin")         # 6
        rust_own = load(f"{dump_dir}/pos{i}_ownership.bin")     # 361

        # policy top-1 (excluding pass position 361? KataGo compares full incl. pass)
        policy_top1_ok += int(np.argmax(rust_policy) == np.argmax(out_policy[0, 0]))

        if not detailed_printed:
            detailed_printed = True
            print(f"position {i} detail:")
            all_ok &= compare("policy logits", rust_policy, out_policy[0, 0], 0.05)
            all_ok &= compare("value logits", rust_value, out_value[0], 0.025)  # FP16 后端口径:TRT 实测 1.77e-2、CUDA 1.4e-2(2026-08-14)
            all_ok &= compare("score mean/sq/lead/vt", rust_misc[:4], out_misc[0, :4], 0.01)
            all_ok &= compare("shortterm errs", rust_misc[4:6], out_moremisc[0, :2], 0.01)
            all_ok &= compare("ownership", rust_own, out_ownership[0, 0].reshape(-1), 0.005)

    top1 = policy_top1_ok / n
    print(f"policy top-1 agreement: {policy_top1_ok}/{n} ({top1:.1%})")
    all_ok &= top1 == 1.0
    print("RESULT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

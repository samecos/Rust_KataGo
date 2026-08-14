#!/usr/bin/env python
"""Compare Rust KataGo TRT backend outputs against ONNX Runtime FP32 reference.

Usage: .venv/Scripts/python.exe scripts/compare_nn_output.py [dump_dir] [--workers N]
Exit code 0 = all gates passed.
"""
import argparse
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


_sess = None


def init_worker(model_path):
    global _sess
    _sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def check_position(args):
    """对比单个位置，返回 (top1_ok, 各门是否通过)。"""
    dump_dir, model, i = args
    spatial = load(f"{dump_dir}/pos{i}_spatial.bin").reshape(1, 22, 19, 19)
    global_ = load(f"{dump_dir}/pos{i}_global.bin").reshape(1, 19)
    sess = _sess
    in_names = [n.name for n in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
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

    top1 = int(np.argmax(rust_policy) == np.argmax(out_policy[0, 0]))

    ok = True
    ok &= compare(f"pos{i} policy", rust_policy, out_policy[0, 0], 0.05)
    ok &= compare(f"pos{i} value", rust_value, out_value[0], 0.025)
    ok &= compare(f"pos{i} score", rust_misc[:4], out_misc[0, :4], 0.01)
    ok &= compare(f"pos{i} shortterm", rust_misc[4:6], out_moremisc[0, :2], 0.01)
    ok &= compare(f"pos{i} ownership", rust_own, out_ownership[0, 0].reshape(-1), 0.005)
    return top1, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", nargs="?", default="target/nn_io_dump")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    dump_dir = args.dump_dir
    meta = json.load(open(os.path.join(dump_dir, "meta.json"), encoding="utf-8"))
    model = meta["model"]
    n = meta["n"]

    workers = args.workers or min(16, max(1, os.cpu_count() or 1))
    jobs = [(dump_dir, model, i) for i in range(n)]

    all_ok = True
    top1_ok = 0
    if workers <= 1:
        init_worker(model)
        results = [check_position(j) for j in jobs]
    else:
        import multiprocessing
        with multiprocessing.Pool(workers, initializer=init_worker, initargs=(model,)) as pool:
            results = list(pool.imap_unordered(check_position, jobs, chunksize=8))
    top1_ok = sum(r[0] for r in results)
    all_ok = all(r[1] for r in results)

    top1 = top1_ok / n
    print(f"policy top-1 agreement: {top1_ok}/{n} ({top1:.1%})")
    all_ok &= top1 == 1.0
    print("RESULT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""M4 离线 autotune 骨架（方法论移植自 KataGomo_fork 的 cuda_tactic_workflow.py）。

流程：10 个有序决策组 → 每组候选集 → 与当前累计配置组合后基准测试 →
ABBA 终裁（challenger ≥ incumbent × (1 + min_improvement) 才替换）→
产出 best-tactic-plan.json（fail-closed 指纹 + tactic overrides）。

当前为骨架：基准执行器调 `katago-rs benchmark` 并解析 nnEval/s；
候选生成与 plan 证书写入在 M4 随 CUDA 后端落地时补全。
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "katago-rs"
MODEL = "D:/code/b11fix.onnx"
MODEL_SHA256 = "f2fc09fdf58a3e8a4b97addceab0c853874c52709152e61935b6ac6b9f474f94"
CFG = ROOT / "configs" / "gtp_smoke.cfg"

# 10 个决策组（有序，后组不得改写前组的配置键）——与 fork 对齐。
DECISION_GROUPS = [
    ("fa4", "wide_projection", "qkv_rope", "dual_ffn"),
    ("fused_residual", "linear2", "outproj"),
    ("postconv_bn", "preconv", "pointwise"),
    ("wide_head", "policy_p1", "head_bn"),
    ("rmsnorm",),
    ("l2",),
    ("weight_sharing",),
    ("initial_conv",),
    ("initial_global",),
    ("value_terminal",),
]

SPEED_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:nn|NN|eval|Eval)[^0-9]{0,20}?/s")


def run_benchmark(overrides: dict[str, str], iterations: int = 200) -> float:
    """跑一次 benchmark，返回 nnEval/s（解析失败返回 -1）。"""
    ov = ",".join(f"{k}={v}" for k, v in overrides.items())
    cmd = [
        str(BIN), "benchmark",
        "--config", str(CFG),
        "--model", MODEL,
        "--override-config", f"nnBackend=cudabackend,{ov}",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        text = out.stdout + out.stderr
    except subprocess.TimeoutExpired:
        return -1.0
    speeds = []
    for m in SPEED_RE.finditer(text):
        speeds.append(float(m.group(1)))
    if not speeds:
        return -1.0
    return max(speeds)


def abba(incumbent_cfg: dict, challenger_cfg: dict,
         min_improvement: float = 0.001,
         confirmation_iterations: int = 300) -> tuple[bool, float, float]:
    """ABBA 终裁：incumbent → challenger → challenger → incumbent，取样本均值比较。"""
    seq = [incumbent_cfg, challenger_cfg, challenger_cfg, incumbent_cfg]
    samples = []
    for cfg in seq:
        v = run_benchmark(cfg, confirmation_iterations)
        if v < 0:
            return False, -1.0, -1.0
        samples.append(v)
    inc_mean = (samples[0] + samples[3]) / 2.0
    cha_mean = (samples[1] + samples[2]) / 2.0
    ok = cha_mean >= inc_mean * (1.0 + min_improvement)
    return ok, inc_mean, cha_mean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-groups", action="store_true", help="列出决策组")
    ap.add_argument("--candidates-json", type=Path, help="候选配置 JSON 文件（可重复跑）")
    args = ap.parse_args()

    if args.list_groups:
        for i, g in enumerate(DECISION_GROUPS):
            print(f"G{i + 1}: {', '.join(g)}")
        return 0

    if not args.candidates_json:
        print("用法: autotune.py --list-groups | --candidates-json <file>", file=sys.stderr)
        return 1

    spec = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    incumbent: dict = dict(spec.get("baseline", {}))
    results = []
    for group in spec.get("candidates", []):
        group_name = group["name"]
        group_winner = incumbent
        for cand in group["configs"]:
            merged = {**incumbent, **cand}
            ok, inc_v, cha_v = abba(incumbent, merged)
            print(f"[{group_name}] {cand}: ABBA ok={ok} incumbent={inc_v:.1f} challenger={cha_v:.1f}")
            if ok:
                group_winner = merged
        incumbent = group_winner
        results.append({group_name: group_winner})

    out = ROOT / "plans" / "best-tactic-plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": 1,
        "kind": "cuda-tactic-plan",
        "status": "draft",
        "target": {"architecture": "sm120", "model_sha256": MODEL_SHA256, "batches": [16]},
        "apply": {"per_batch_tactic_overrides": {"16": incumbent}},
        "selection": "history-ordered accumulated coordinate winners; ABBA",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

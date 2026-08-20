#!/usr/bin/env python
"""M4 离线 autotune（方法论移植自 KataGomo_fork 的 cuda_tactic_workflow.py）。

流程：有序决策组 → 每组候选（真实 tactic 开关映射）→ 与当前累计配置组合
→ benchmark → ABBA 终裁（challenger ≥ incumbent × (1 + min_improvement) 才
替换）→ 产出 best-tactic-plan.json（fail-closed 指纹 + tactic_overrides，
由 cudabackend 的 cudaTacticPlan 配置键加载，见 kata_nn/src/tactic_plan.rs）。

用法：
  python scripts/autotune.py --threads 8            # 全流程
  python scripts/autotune.py --groups gemm graph    # 只跑指定组
  python scripts/autotune.py --list-groups          # 列出组与候选
  python scripts/autotune.py --quick                # 短迭代快速模式

前置：target/release/katago-rs.exe（--features cuda，release 构建）。
"""
import argparse
import datetime
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "katago-rs.exe"
MODEL = "D:/code/b11fix.onnx"
CFG = ROOT / "configs" / "gtp_smoke.cfg"
PLAN_OUT = ROOT / "plans" / "best-tactic-plan.json"
HISTORY_OUT = ROOT / "plans" / "autotune-history.json"

# ---------------------------------------------------------------------------
# 决策组定义（有序；后组不得改写前组已选定的键）。
# 每个候选 = 环境变量 tactic 开关的显式赋值。键域必须与
# kata_nn/src/tactic_plan.rs 的 ALLOWED_TACTIC_KEYS 一致。
# ---------------------------------------------------------------------------

DECISION_GROUPS = [
    {
        # GEMM 引擎族：手写 tile vs cuBLASLt 启发式（默认 cuBLASLt）。
        # 历史结论：cuBLASLt t=1 +42%。证伪的 tile 变体保留为候选。
        "name": "gemm",
        "candidates": [
            {},  # 默认 = cuBLASLt（挑战基线时即 incumbent 默认值）
            {"KATAGO_CUDA_CUBLASLT": "0"},  # 纯手写 kernel
            {"KATAGO_CUDA_CUBLASLT": "0", "KATAGO_CUDA_T32": "1"},
            {"KATAGO_CUDA_CUBLASLT": "0", "KATAGO_CUDA_T64N32": "1"},
        ],
    },
    {
        # kernel 融合族（up/down 投影 + FFN gatesilu epilogue）。
        # 历史结论（手写 GEMM 时代）：融合 +5%；cuBLASLt 接管后（2026-08-15
        # autotune 复测）默认已改 none——融合 epilogue 绕过 cuBLASLt 反而
        # 更慢。all/up/down 保留为 CUBLASLT=0 组合下的候选。
        "name": "fusion",
        "candidates": [
            {},  # 默认 = none（cuBLASLt 时代最优）
            {"KATAGO_CUDA_FUSION": "all"},
            {"KATAGO_CUDA_FUSION": "up"},
            {"KATAGO_CUDA_FUSION": "down"},
        ],
    },
    {
        # attention 路径：FA2 全张量核（默认）vs 旧 warp-per-row v3。
        "name": "attention",
        "candidates": [
            {},
            # q64 is the certified production incumbent; explicitly measure
            # q128 as the rollback challenger on every future autotune run.
            {"KATAGO_CUDA_ATTN_TILE": "q128"},
            {"KATAGO_CUDA_ATTN_TILE": "q64"},
            {"KATAGO_CUDA_ATTN": "v3"},
        ],
    },
    {
        # RMSNorm 路径：warp4-vec8（默认）vs 旧 v1。
        "name": "rmsnorm",
        "candidates": [
            {},
            {"KATAGO_CUDA_RMS": "v1"},
        ],
    },
    {
        # split-K GEMM（已证伪两次，保留候选供 batch 更大时复查）。
        "name": "splitk",
        "candidates": [
            {},
            {"KATAGO_CUDA_SPLITK": "1"},
        ],
    },
    {
        # 执行拓扑：CUDA Graph + 事件门控流水线（默认全开）。
        # 历史结论：graph 与 pipeline 均实测为正收益。
        "name": "graph_pipeline",
        "candidates": [
            {},
            {"KATAGO_CUDA_NOGRAPH": "1"},
            {"KATAGO_CUDA_NOPIPELINE": "1"},
        ],
    },
    {
        # 凑批策略：精确尺寸（默认，B16 实验证伪 padding）vs PAD。
        "name": "batching",
        "candidates": [
            {},
            {"KATAGO_CUDA_PADBATCH": "1"},
        ],
    },
    {
        # 凑批等待窗口（GPU 忙时，微秒）：默认 8ms。
        "name": "batch_window",
        "candidates": [
            {},
            {"KATAGO_NN_BATCH_WINDOW_US": "3000"},
            {"KATAGO_NN_BATCH_WINDOW_US": "12000"},
        ],
    },
]

# 基准命令的线程数候选（每组配置在多个线程档下取几何均值，选整体最优；
# 避免只在单档最优而其他档回退）。CLI 参数为字符串，这里做转换。
THREAD_PROFILES = {
    "8": [8],
    "16": [16],
    "both": [4, 16],
}

SPEED_RE = re.compile(r"nnEvals/s = ([0-9]+(?:\.[0-9]+)?)")


def run_benchmark(env_overrides: dict[str, str], threads: int,
                  iterations: int = 200) -> float:
    """跑一次 benchmark（每线程档），返回 nnEvals/s（失败返回 -1）。"""
    cmd = [
        str(BIN), "benchmark",
        "--config", str(CFG),
        "--model", MODEL,
        "--override-config", "nnBackend=cudabackend",
        "--override-config", f"numSearchThreads={threads}",
        "-v", "1500", "-n", "1", "-t", str(threads),
    ]
    if iterations != 200:
        cmd += ["--override-config", f"nnMaxVisits={iterations}"]
    env = dict(**{k: v for k, v in __import__('os').environ.items()
                  if not k.startswith("KATAGO_")},
               **env_overrides)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=1800, env=env)
        text = out.stdout + out.stderr
    except subprocess.TimeoutExpired:
        return -1.0
    speeds = [float(m) for m in SPEED_RE.findall(text)]
    return max(speeds) if speeds else -1.0


def score_config(cfg: dict[str, str], thread_list: list[int],
                 iterations: int) -> float:
    """多线程档几何均值；任一档失败返回 -1。"""
    vals = []
    for t in thread_list:
        v = run_benchmark(cfg, t, iterations)
        if v <= 0:
            return -1.0
        vals.append(v)
    geo = 1.0
    for v in vals:
        geo *= v
    return geo ** (1.0 / len(vals))


def abba(incumbent: dict, challenger: dict, thread_list: list[int],
         min_improvement: float, iterations: int):
    """ABBA 终裁：I→C→C→I，几何均值比较。返回 (win, inc_geo, cha_geo)。"""
    seq = [(incumbent, "I"), (challenger, "C"), (challenger, "C"), (incumbent, "I")]
    samples = {"I": [], "C": []}
    for cfg, tag in seq:
        v = score_config(cfg, thread_list, iterations)
        if v <= 0:
            return False, -1.0, -1.0
        samples[tag].append(v)
        print(f"    [{tag}] geo={v:.1f}", flush=True)
    inc_geo = (samples["I"][0] * samples["I"][1]) ** 0.5
    cha_geo = (samples["C"][0] * samples["C"][1]) ** 0.5
    return cha_geo >= inc_geo * (1.0 + min_improvement), inc_geo, cha_geo


def get_fingerprint() -> dict:
    """调 cuda-fingerprint 子命令取设备指纹 + 模型 SHA-256。"""
    out = subprocess.run(
        [str(BIN), "cuda-fingerprint", "--model", MODEL],
        capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        sys.exit(f"cuda-fingerprint failed: {out.stdout}{out.stderr}")
    fp = json.loads(out.stdout)
    if "backend_build" not in fp:
        sys.exit("cuda-fingerprint output lacks backend_build; rebuild katago-rs with the CUDA feature")
    return fp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-groups", action="store_true")
    ap.add_argument("--groups", default="", help="逗号分隔，只跑这些组")
    ap.add_argument("--threads", default="8", help="8|16|both（ABBA 线程档）")
    ap.add_argument("--min-improvement", type=float, default=0.01,
                    help="challenger 需超出的最小相对改善（默认 1%%）")
    ap.add_argument("--iterations", type=int, default=200,
                    help="每次 benchmark 的 visits（--quick 时 60）")
    ap.add_argument("--quick", action="store_true", help="短迭代快速模式")
    ap.add_argument("--plan-id", default="", help="plan id（默认自动生成）")
    args = ap.parse_args()

    if args.list_groups:
        for g in DECISION_GROUPS:
            print(f"{g['name']}:")
            for c in g["candidates"]:
                desc = " ".join(f"{k}={v}" for k, v in c.items()) or "(defaults)"
                print(f"  - {desc}")
        return 0

    if not BIN.exists():
        sys.exit(f"binary not found: {BIN}（先 cargo build --release -p katago --features cuda）")

    thread_list = THREAD_PROFILES[args.threads]
    iterations = 60 if args.quick else args.iterations
    only = {g.strip() for g in args.groups.split(",") if g.strip()}

    print(f"== autotune: threads={thread_list} iters={iterations} "
          f"min_improvement={args.min_improvement:.3f} ==", flush=True)
    fp = get_fingerprint()
    print(f"device: {fp['gpu_name']} cc={fp['compute_capability']} "
          f"sm={fp['sm_count']} l2={fp['l2_cache_bytes']}", flush=True)
    print(f"model sha256: {fp['model_sha256']}", flush=True)

    # Start from the authenticated production plan.  This prevents a partial
    # --groups run (or a future script edit) from silently dropping DualFFN or
    # reverting q64 to an unmeasured empty configuration.
    incumbent: dict[str, str] = {
        "KATAGO_CUDA_DUALFFN": "1",
        "KATAGO_CUDA_ATTN_TILE": "q64",
    }
    history = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "threads": thread_list,
        "iterations": iterations,
        "min_improvement": args.min_improvement,
        "device": fp,
        "groups": [],
    }

    for g in DECISION_GROUPS:
        if only and g["name"] not in only:
            continue
        print(f"\n-- group {g['name']} --", flush=True)
        winner = incumbent
        group_rec = {"name": g["name"], "candidates": []}
        # 候选 0 = 组默认（空增量）；若与 incumbent 相同则跳过测量。
        for cand in g["candidates"]:
            merged = {**incumbent, **cand}
            if merged == winner:
                group_rec["candidates"].append(
                    {"config": cand, "skipped": "identical to current winner"})
                continue
            ok, inc_v, cha_v = abba(winner, merged, thread_list,
                                    args.min_improvement, iterations)
            rec = {"config": cand, "incumbent_geo": inc_v, "challenger_geo": cha_v,
                   "win": ok}
            group_rec["candidates"].append(rec)
            desc = " ".join(f"{k}={v}" for k, v in cand.items()) or "(defaults)"
            print(f"  [{g['name']}] {desc}: {'WIN' if ok else 'lose'} "
                  f"inc={inc_v:.1f} cha={cha_v:.1f}", flush=True)
            if ok:
                winner = merged
        if winner is incumbent:
            winner = dict(incumbent)  # 组内无变化也要落成新引用
        incumbent = winner
        group_rec["winner"] = dict(incumbent)
        history["groups"].append(group_rec)

    # ---- 证书写出 ----
    plan_id = args.plan_id or (
        f"sm{fp['compute_capability'].replace('.', '')}-"
        f"{fp['gpu_name'].lower().replace(' ', '-')[:24]}-autotune-"
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    PLAN_OUT.parent.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": 2,
        "kind": "cuda-tactic-plan",
        "plan_id": plan_id,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "backend_build": fp["backend_build"],
        "target": {
            "architecture": fp.get("architecture", ""),
            "gpu_name": fp["gpu_name"],
            "compute_capability": fp["compute_capability"],
            "sm_count": fp["sm_count"],
            "l2_cache_bytes": fp["l2_cache_bytes"],
            "model_sha256": fp["model_sha256"],
        },
        "apply": {"tactic_overrides": incumbent},
        "selection": {
            "method": "ordered decision groups, ABBA (I-C-C-I) geometric mean",
            "min_improvement": args.min_improvement,
            "threads": thread_list,
            "iterations": iterations,
            "host": platform.node(),
        },
        "history_file": str(HISTORY_OUT.relative_to(ROOT)),
    }
    PLAN_OUT.write_text(json.dumps(plan, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    HISTORY_OUT.write_text(json.dumps(history, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nplan:    {PLAN_OUT}")
    print(f"history: {HISTORY_OUT}")
    print(f"tactic_overrides: {json.dumps(incumbent)}")
    print("接入：--override-config nnBackend=cudabackend,"
          f"cudaTacticPlan={PLAN_OUT.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

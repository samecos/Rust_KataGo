#!/usr/bin/env python
"""M4 离线 autotune（方法论移植自 KataGomo_fork 的 cuda_tactic_workflow.py）。

流程：有序决策组 → 每组候选（真实 tactic 开关映射）→ 与当前累计配置组合
→ benchmark → ABBA 终裁（challenger ≥ incumbent × (1 + min_improvement) 才
替换）→ 产出 best-tactic-plan.json（fail-closed 指纹 + tactic_overrides，
由 cudabackend 的 cudaTacticPlan 配置键加载，见 kata_nn/src/tactic_plan.rs）。

决策组跑完后，用获胜 tactic 组合对 --threads-sweep 各档做 benchmark，
按 nnEvals/s 选出 numSearchThreads，生成可直接 `gtp --config` 使用的
CFG（引用刚产出的 plan），并用该 CFG 跑一次 GTP genmove 冒烟验证。

用法：
  python scripts/autotune.py --threads 8            # 全流程
  python scripts/autotune.py --groups gemm graph    # 只跑指定组
  python scripts/autotune.py --list-groups          # 列出组与候选
  python scripts/autotune.py --quick                # 短迭代快速模式
  python scripts/autotune.py --model D:/path/to/model.onnx \
      --out-plan plans/my-plan.json --out-cfg configs/gtp_my.cfg

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
# argparse 默认值；main() 解析参数后按 --model/--base-cfg/--out-plan 改写下方的
# 模块级目标（benchmark/指纹/plan 函数直接引用这些全局名）。
DEFAULT_MODEL = "D:/code/b11fix.onnx"
DEFAULT_CFG = ROOT / "configs" / "gtp_smoke.cfg"
DEFAULT_PLAN_OUT = ROOT / "plans" / "best-tactic-plan.json"
DEFAULT_OUT_CFG = ROOT / "configs" / "gtp_autotuned.cfg"
MODEL = DEFAULT_MODEL
CFG = DEFAULT_CFG
PLAN_OUT = DEFAULT_PLAN_OUT
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
        # Fork-inspired K384 weight layout; FP32 compute and half boundaries
        # remain unchanged. Full-model numerical acceptance is a prerequisite.
        "name": "gemm_layout",
        "candidates": [
            {},
            {"KATAGO_CUDA_GEMM_LAYOUT": "tn"},
            {"KATAGO_CUDA_GEMM_LAYOUT": "nn_k384"},
            {"KATAGO_CUDA_GEMM_LAYOUT": "nn_k384_b8"},
            {"KATAGO_CUDA_GEMM_LAYOUT": "nn_k384_b16"},
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


def sweep_threads(env_overrides: dict[str, str], sweep: list[int],
                  iterations: int) -> list[tuple[int, float]]:
    """对每档线程跑一次 benchmark，返回 [(threads, nnEvals/s)]。"""
    results = []
    for t in sweep:
        v = run_benchmark(env_overrides, t, iterations)
        results.append((t, v))
        print(f"  [sweep] numSearchThreads={t}: nnEvals/s={v:.1f}", flush=True)
    return results


def write_gtp_config(path: Path, *, rules: str, komi: str, threads: int,
                     max_visits: int, plan_path: Path, fp: dict,
                     plan_id: str, sweep_results: list[tuple[int, float]]) -> None:
    """写出可直接 `gtp --config` 使用的 CFG（引用认证 plan，fail-closed）。"""
    sweep_desc = ", ".join(
        f"{t}:{v:.0f}/s" if v > 0 else f"{t}:FAIL" for t, v in sweep_results)
    contents = f"""# Auto-generated by scripts/autotune.py — 请勿手改，重跑脚本即覆盖
# generated_utc = {datetime.datetime.now(datetime.timezone.utc).isoformat()}
# device = {fp['gpu_name']} (cc={fp['compute_capability']}, sm={fp['sm_count']})
# model_sha256 = {fp['model_sha256']}
# tactic_plan = {plan_id}（fail-closed：绑定 GPU 指纹 + 模型 SHA-256，换机/换模型需重新 autotune）
# numSearchThreads 由 nnEvals/s 扫描选出：{sweep_desc}

rules = {rules}
komi = {komi}

# ---- 搜索 ----
numSearchThreads = {threads}
# genmove 每手上限；流式分析（kata-analyze/lz-analyze）不受此行限制。
maxVisits = {max_visits}

# ---- NN 后端 ----
nnBackend = cudabackend
cudaTacticPlan = {plan_path.as_posix()}
# batch 上限默认由后端按线程数推导；如需固定可取消注释：
# nnMaxBatchSize = 16
nnCacheSizePowerOfTwo = 20
nnMutexPoolSizePowerOfTwo = 16
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def validate_cfg(cfg_path: Path, model: str) -> bool:
    """用生成的 CFG 跑一次 GTP genmove 冒烟（加载模型 + plan + 真实搜索）。"""
    cmds = ("protocol_version\nname\nboardsize 19\nclear_board\n"
            "komi 7.5\ngenmove B\nquit\n")
    try:
        out = subprocess.run(
            [str(BIN), "gtp", "--config", str(cfg_path), "--model", model],
            input=cmds, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("  [validate] TIMEOUT (600s)", flush=True)
        return False
    replies = [ln for ln in out.stdout.splitlines()
               if ln.startswith("=") or ln.startswith("?")]
    ok = (out.returncode == 0 and len(replies) >= 6
          and not any(ln.startswith("?") for ln in replies))
    print(f"  [validate] gtp genmove smoke: {'PASS' if ok else 'FAIL'} "
          f"(replies={len(replies)}, rc={out.returncode})", flush=True)
    if not ok:
        print(f"  [validate] stdout tail: {out.stdout[-500:]}", flush=True)
        print(f"  [validate] stderr tail: {out.stderr[-500:]}", flush=True)
    return ok


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
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="模型路径（默认 %(default)s）；plan 的 SHA-256 绑定该文件")
    ap.add_argument("--base-cfg", default=str(DEFAULT_CFG),
                    help="benchmark 使用的基础 cfg（默认 %(default)s）")
    ap.add_argument("--out-plan", default=str(DEFAULT_PLAN_OUT),
                    help="tactic plan JSON 输出路径（默认 %(default)s）")
    ap.add_argument("--out-cfg", default=str(DEFAULT_OUT_CFG),
                    help="生成的可运行 GTP 配置路径；传空串只产出 plan")
    ap.add_argument("--threads-sweep", default="4,8,12,16,24",
                    help="为 CFG 选择 numSearchThreads 的候选档（逗号分隔）；"
                         "空串跳过扫描并沿用 --threads 首档")
    ap.add_argument("--rules", default="chinese", help="写入 CFG 的规则（默认 %(default)s）")
    ap.add_argument("--komi", default="7.5", help="写入 CFG 的贴目（默认 %(default)s）")
    ap.add_argument("--max-visits", type=int, default=800,
                    help="写入 CFG 的 maxVisits（默认 %(default)s）")
    ap.add_argument("--skip-validate", action="store_true",
                    help="跳过生成 CFG 后的 GTP genmove 冒烟验证")
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

    # 按 CLI 参数改写给 benchmark/指纹/plan 函数使用的模块级目标。
    global MODEL, CFG, PLAN_OUT, HISTORY_OUT
    MODEL = args.model
    CFG = Path(args.base_cfg)
    PLAN_OUT = Path(args.out_plan)
    HISTORY_OUT = PLAN_OUT.parent / "autotune-history.json"

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
        "history_file": (str(HISTORY_OUT.relative_to(ROOT))
                         if HISTORY_OUT.is_absolute()
                         and HISTORY_OUT.is_relative_to(ROOT)
                         else str(HISTORY_OUT)),
    }
    PLAN_OUT.write_text(json.dumps(plan, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    HISTORY_OUT.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_OUT.write_text(json.dumps(history, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nplan:    {PLAN_OUT}")
    print(f"history: {HISTORY_OUT}")
    print(f"tactic_overrides: {json.dumps(incumbent)}")

    if not args.out_cfg:
        print("接入：--override-config nnBackend=cudabackend,"
              f"cudaTacticPlan={PLAN_OUT.as_posix()}")
        return 0

    # ---- 生成可运行 CFG：用获胜 tactic 扫描线程档，按 nnEvals/s 选 numSearchThreads ----
    sweep = [int(x) for x in args.threads_sweep.split(",") if x.strip()]
    if sweep:
        print(f"\n-- threads sweep for CFG: {sweep} --", flush=True)
        sweep_results = sweep_threads(incumbent, sweep, iterations)
        best = max(((t, v) for t, v in sweep_results if v > 0),
                   key=lambda x: x[1], default=None)
        if best is None:
            print("threads sweep 全部失败；plan 已写出，CFG 未生成")
            return 1
        best_threads = best[0]
    else:
        best_threads = thread_list[0]
        sweep_results = []
        print(f"\nthreads sweep 已跳过，沿用 --threads 首档：{best_threads}", flush=True)

    out_cfg = Path(args.out_cfg)
    write_gtp_config(out_cfg, rules=args.rules, komi=args.komi,
                     threads=best_threads, max_visits=args.max_visits,
                     plan_path=PLAN_OUT.resolve(), fp=fp, plan_id=plan_id,
                     sweep_results=sweep_results)
    print(f"cfg:     {out_cfg} (numSearchThreads={best_threads})")

    if not args.skip_validate:
        if not validate_cfg(out_cfg, MODEL):
            print("GTP 冒烟验证失败；plan/cfg 已写出，请检查上方日志")
            return 1
    print(f"接入：katago-rs gtp --config {out_cfg.as_posix()} --model {MODEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

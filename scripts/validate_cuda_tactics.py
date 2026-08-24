"""Validate CUDA tactic selection, real launch paths, and fail-closed plans.

The script intentionally drives the production ``nnbench --mode eval`` stack.
It records every command and log under a timestamped output directory and
asserts both expected and forbidden structured ``[cuda-tactic]`` markers.

With ``--with-numeric``, each positive case additionally runs the
``dump_nn_io_cuda`` integration test under the same tactic environment and
compares the dumps against the ONNX Runtime FP32 golden reference
(``scripts/compare_nn_output.py``) — mirroring upstream KataGo's
``runcudaopttests.sh``, which pairs path-confirmation markers with numerical
error thresholds so a knob that silently fails to switch paths (the M2
DualFFN breakage precedent) cannot pass unnoticed.

Example:
  python scripts/validate_cuda_tactics.py \
    --binary target/cudarocmopt-fp32/release/katago-rs.exe \
    --model D:/code/b11fix.onnx
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path("D:/code/b11fix.onnx")
DEFAULT_PLAN = ROOT / "plans" / "best-tactic-plan.json"


@dataclass
class Case:
    name: str
    env: dict[str, str] = field(default_factory=dict)
    plan: Path | None = None
    expect: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()
    exit_code: int = 0
    # 数值门（--with-numeric）：跑 dump+compare 的组合开关；None = 跳过。
    numeric: bool = False
    # nnbench 固定物理 batch（默认 1；padbatch 等路径需 n>1 才触发）。
    batch: int = 1
    # nnbench worker/迭代数（凑批路径需足够并发才能合并出 n>1 的批）。
    workers: int = 2
    iterations: int = 2


def clean_env(overrides: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("KATAGO_CUDA_")}
    env.update(overrides)
    return env


def fingerprint(binary: Path, model: Path) -> dict:
    proc = subprocess.run(
        [str(binary), "cuda-fingerprint", "--model", str(model)],
        capture_output=True,
        text=True,
        timeout=180,
        env=clean_env({}),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cuda-fingerprint failed:\n{proc.stdout}{proc.stderr}")
    value = json.loads(proc.stdout)
    if "backend_build" not in value:
        raise RuntimeError("cuda-fingerprint lacks backend_build; rebuild with --features cuda")
    return value


def schema2_plan(fp: dict, plan_id: str, overrides: dict[str, str]) -> dict:
    return {
        "schema": 2,
        "kind": "cuda-tactic-plan",
        "plan_id": plan_id,
        "backend_build": fp["backend_build"],
        "target": {
            "architecture": fp["architecture"],
            "gpu_name": fp["gpu_name"],
            "compute_capability": fp["compute_capability"],
            "sm_count": fp["sm_count"],
            "l2_cache_bytes": fp["l2_cache_bytes"],
            "model_sha256": fp["model_sha256"],
            "max_batch_size": 1,
        },
        "apply": {"tactic_overrides": overrides},
    }


def write_plan(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def run_case(binary: Path, model: Path, out_dir: Path, case: Case) -> list[str]:
    cmd = [
        str(binary),
        "nnbench",
        "--model",
        str(model),
        "--override-config",
        "nnBackend=cudabackend",
    ]
    if case.plan is not None:
        cmd.extend(
            [
                "--override-config",
                f"cudaTacticPlan={case.plan.as_posix()}",
            ]
        )
    cmd.extend(
        [
            "--mode",
            "eval",
            "--batch",
            str(case.batch),
            "--workers",
            str(case.workers),
            "--warmup",
            "2",
            "--iterations",
            str(case.iterations),
        ]
    )

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        env=clean_env(case.env),
    )
    log = proc.stdout + proc.stderr
    (out_dir / f"{case.name}.log").write_text(log, encoding="utf-8")
    (out_dir / f"{case.name}.json").write_text(
        json.dumps(
            {
                "command": cmd,
                "env": case.env,
                "expected_exit": case.exit_code,
                "actual_exit": proc.returncode,
                "expect": case.expect,
                "forbid": case.forbid,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    errors: list[str] = []
    if proc.returncode != case.exit_code:
        errors.append(f"exit {proc.returncode}, expected {case.exit_code}")
    for marker in case.expect:
        if marker not in log:
            errors.append(f"missing marker: {marker}")
    for marker in case.forbid:
        if marker in log:
            errors.append(f"forbidden marker present: {marker}")
    return errors


def run_numeric_gate(
    binary: Path, model: Path, out_dir: Path, case: Case, compare_script: Path, python: Path
) -> list[str]:
    """数值门：同 tactic 环境下 dump 4 个局面前向，对 ORT FP32 黄金参考比对。

    等价于官方 runcudaopttests.sh 的 testgpuerror 数值阈值部分；我们用
    dump_nn_io_cuda + compare_nn_output.py（gates: policy 5e-2 / value 2.5e-2 /
    misc 1e-2 / ownership 5e-3 / top-1 100%）。
    """
    dump_dir = out_dir / f"{case.name}-numeric"
    env = clean_env(case.env)
    env["KATAGO_DUMP_DIR"] = str(dump_dir)
    env["KATAGO_DUMP_POSITIONS"] = "4"
    env["KATAGO_ONNX_MODEL"] = str(model)
    proc = subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "kata_nn",
            "--test",
            "dump_nn_io_cuda",
            "--features",
            "cuda",
            "--release",
            "--",
            "--nocapture",
            "dump_nn_io_cuda",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
        cwd=ROOT,
    )
    (out_dir / f"{case.name}-numeric-dump.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0 or not (dump_dir / "meta.json").exists():
        return [f"numeric dump failed (exit {proc.returncode})"]

    cmp_proc = subprocess.run(
        [str(python), str(compare_script), str(dump_dir), "--model", str(model)],
        capture_output=True,
        text=True,
        timeout=600,
        env=clean_env(case.env),
        cwd=ROOT,
    )
    log = cmp_proc.stdout + cmp_proc.stderr
    (out_dir / f"{case.name}-numeric-compare.log").write_text(log, encoding="utf-8")
    if "RESULT: PASS" not in log:
        return [f"numeric gate FAIL: see {case.name}-numeric-compare.log"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=Path,
        default=ROOT / "target" / "release" / "katago-rs.exe",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--with-numeric",
        action="store_true",
        help="run the dump+compare numeric gate for positive cases (slow)",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=ROOT / ".venv" / "Scripts" / "python.exe",
        help="python interpreter with onnxruntime for the numeric gate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "target"
        / "cudarocmopt-validation"
        / f"tactic-matrix-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    args = parser.parse_args()

    binary = args.binary.resolve()
    model = args.model.resolve()
    plan = args.plan.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not binary.exists() or not model.exists() or not plan.exists():
        parser.error(f"missing binary/model/plan: {binary}, {model}, {plan}")

    fp = fingerprint(binary, model)
    (out_dir / "fingerprint.json").write_text(
        json.dumps(fp, indent=2), encoding="utf-8"
    )

    valid = schema2_plan(fp, "v1-schema2-positive", {"KATAGO_CUDA_DUALFFN": "1"})
    valid_path = write_plan(out_dir / "valid-schema2.json", valid)

    bad_value = schema2_plan(fp, "v1-bad-value", {"KATAGO_CUDA_DUALFFN": "bogus"})
    bad_value_path = write_plan(out_dir / "bad-value.json", bad_value)

    unknown = schema2_plan(fp, "v1-unknown-key", {"KATAGO_CUDA_UNKNOWN": "1"})
    unknown_path = write_plan(out_dir / "unknown-key.json", unknown)

    model_mismatch = schema2_plan(fp, "v1-model-mismatch", {})
    model_mismatch["target"]["model_sha256"] = "00" * 32
    model_mismatch_path = write_plan(out_dir / "model-mismatch.json", model_mismatch)

    build_mismatch = schema2_plan(fp, "v1-build-mismatch", {})
    build_mismatch["backend_build"]["kernel_build_id"] = "sha256:" + "00" * 32
    build_mismatch_path = write_plan(out_dir / "build-mismatch.json", build_mismatch)

    bool_zero = {
        "KATAGO_CUDA_CUBLASLT": "0",
        "KATAGO_CUDA_DUALFFN": "0",
        "KATAGO_CUDA_ATTN_TILE": "q128",
        "KATAGO_CUDA_NOGRAPH": "0",
        "KATAGO_CUDA_NOPIPELINE": "0",
        "KATAGO_CUDA_PADBATCH": "0",
        "KATAGO_CUDA_SPLITK": "0",
        "KATAGO_CUDA_T32": "0",
        "KATAGO_CUDA_T64N32": "0",
    }
    cases = [
        Case(
            "default-r1-plan-priority",
            env={"KATAGO_CUDA_DUALFFN": "0"},
            plan=plan,
            expect=(
                "name=dual_ffn requested=1 compiled=1 probe=pass",
                "name=dual_ffn launch=fused",
                "name=attention requested=fa2 launch=fa2",
                "name=graph requested=graph launch=graph",
            ),
            forbid=("name=dual_ffn launch=unfused", "launch=direct"),
            numeric=True,
        ),
        Case(
            "schema2-positive",
            plan=valid_path,
            expect=("name=dual_ffn launch=fused", "launch=graph"),
            forbid=("launch=unfused", "launch=direct"),
            numeric=True,
        ),
        Case(
            "dual-off",
            env={"KATAGO_CUDA_DUALFFN": "0"},
            expect=("name=dual_ffn launch=unfused",),
            forbid=("name=dual_ffn launch=fused", "probe=pass"),
            numeric=True,
        ),
        Case(
            "dual-on",
            env={"KATAGO_CUDA_DUALFFN": "1"},
            expect=("probe=pass", "name=dual_ffn launch=fused"),
            forbid=("name=dual_ffn launch=unfused",),
            numeric=True,
        ),
        Case(
            "graph-off",
            env={"KATAGO_CUDA_DUALFFN": "0", "KATAGO_CUDA_NOGRAPH": "1"},
            expect=("name=graph requested=direct launch=direct effective=direct",),
            forbid=("launch=graph",),
            numeric=True,
        ),
        Case(
            "graph-on",
            env={"KATAGO_CUDA_DUALFFN": "0", "KATAGO_CUDA_NOGRAPH": "0"},
            expect=("name=graph requested=graph launch=graph effective=graph",),
            forbid=("launch=direct",),
        ),
        Case(
            "attention-v3",
            env={"KATAGO_CUDA_DUALFFN": "0", "KATAGO_CUDA_ATTN": "v3"},
            expect=("name=attention requested=v3 launch=v3 effective=v3",),
            forbid=("name=attention requested=fa2",),
            numeric=True,
        ),
        Case(
            "attention-q64",
            env={"KATAGO_CUDA_DUALFFN": "0", "KATAGO_CUDA_ATTN_TILE": "q64"},
            expect=("name=attention requested=fa2 launch=fa2 effective=fa2 tile=q64",),
            forbid=("tile=q128",),
            numeric=True,
        ),
        Case(
            "attention-q128",
            env={"KATAGO_CUDA_DUALFFN": "0", "KATAGO_CUDA_ATTN_TILE": "q128"},
            expect=("name=attention requested=fa2 launch=fa2 effective=fa2 tile=q128",),
            forbid=("tile=q64",),
            numeric=True,
        ),
        Case(
            "gemm-handwritten",
            env={"KATAGO_CUDA_DUALFFN": "0", "KATAGO_CUDA_CUBLASLT": "0"},
            expect=(
                "name=gemm kind=plain engine=handwritten",
                "name=gemm kind=f16out engine=handwritten",
                "name=gemm kind=residual engine=handwritten",
            ),
            forbid=("engine=cublaslt",),
            numeric=True,
        ),
        Case(
            "gemm-cublaslt",
            env={"KATAGO_CUDA_DUALFFN": "0"},
            expect=(
                "name=gemm kind=f16out engine=cublaslt",
                "name=gemm kind=residual engine=cublaslt",
            ),
            # 注：plain kind 在小 M 形状（如 m=361 的 head GEMM）仍会按
            # tile 启发式选手写 kernel，这里只断言 cublaslt 引擎真的接管
            # f16out/residual（qkv/dual/outproj 主热点）。
            forbid=("kind=f16out engine=handwritten", "kind=residual engine=handwritten"),
            numeric=True,
        ),
        Case(
            "gemm-cublaslt-rank-time",
            env={"KATAGO_CUDA_DUALFFN": "0", "KATAGO_CUDA_CUBLASLT_RANK": "time"},
            expect=("name=cublaslt_rank engine=time",),
            numeric=True,
        ),
        Case(
            "splitk-on",
            env={"KATAGO_CUDA_DUALFFN": "0", "KATAGO_CUDA_SPLITK": "1"},
            expect=("name=splitk launch=on",),
            numeric=True,
        ),
        Case(
            "fusion-all",
            env={"KATAGO_CUDA_DUALFFN": "0", "KATAGO_CUDA_FUSION": "all"},
            expect=("name=fusion mode=all",),
            numeric=True,
        ),
        Case(
            "rms-v1",
            env={"KATAGO_CUDA_DUALFFN": "0", "KATAGO_CUDA_RMS": "v1"},
            expect=("name=rms launch=v1",),
            forbid=("name=rms launch=w4",),
            numeric=True,
        ),
        Case(
            "rms-w4",
            env={"KATAGO_CUDA_DUALFFN": "0"},
            expect=("name=rms launch=w4",),
            forbid=("name=rms launch=v1",),
            numeric=True,
        ),
        Case(
            "padbatch-on",
            env={"KATAGO_CUDA_DUALFFN": "0", "KATAGO_CUDA_PADBATCH": "1"},
            # pad 只在 n>1 且未满 batch 时触发，需足够并发把行合并成批
            # （workers=2×iter=2 时 avgBatch=1，永远单行）。pad×graph 组合
            # 会打出 graph 重放 INVALID_VALUE WARNING——已知证伪组合
            # （ABBA 259），此处仅验证 pad 路径确实接管 + 进程不崩。
            expect=("name=padbatch launch=on",),
            batch=2,
            workers=8,
            iterations=32,
        ),
        Case(
            "explicit-zero",
            env=bool_zero,
            expect=("launch=unfused", "launch=fa2", "launch=graph"),
            forbid=("launch=fused", "launch=direct"),
        ),
        Case("bad-value", plan=bad_value_path, expect=("invalid value",), exit_code=1),
        Case("unknown-key", plan=unknown_path, expect=("unknown tactic key",), exit_code=1),
        Case(
            "model-mismatch",
            plan=model_mismatch_path,
            expect=("model sha256 mismatch",),
            exit_code=1,
        ),
        Case(
            "build-mismatch",
            plan=build_mismatch_path,
            expect=("kernel_build_id mismatch",),
            exit_code=1,
        ),
        Case(
            "forced-probe-failure",
            env={"KATAGO_CUDA_TEST_FORCE_DUALFFN_PROBE_FAIL": "1"},
            plan=plan,
            expect=("certified plan requests DualFFN", "forced to fail"),
            forbid=("launch=fused", "nnEvals/s"),
            exit_code=1,
        ),
        Case(
            "forced-warmup-failure",
            env={"KATAGO_CUDA_TEST_FORCE_WARMUP_FAIL": "1"},
            plan=plan,
            expect=("CUDA handle warmup forced to fail",),
            forbid=("nnEvals/s",),
            exit_code=1,
        ),
    ]

    failures = 0
    for case in cases:
        errors = run_case(binary, model, out_dir, case)
        if not errors and args.with_numeric and case.numeric:
            errors += run_numeric_gate(
                binary, model, out_dir, case, ROOT / "scripts" / "compare_nn_output.py", args.python
            )
        status = "PASS" if not errors else "FAIL"
        print(f"{status:4} {case.name}")
        for error in errors:
            print(f"     {error}")
        failures += bool(errors)

    summary = {
        "binary": str(binary),
        "model": str(model),
        "plan": str(plan),
        "cases": len(cases),
        "failures": failures,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"logs: {out_dir}")
    print("RESULT: PASS" if failures == 0 else f"RESULT: FAIL ({failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Sequential Fork/Rust ABBA on the same TF3 model and physical B14/S2.

CUDA event median-rate sums match the Fork certification metric. Wall timings
are also retained but have different boundaries and are never divided to claim
an end-to-end speed ratio. Default inputs are semantically matching empty boards;
cross-implementation feature-byte equality is not asserted. Neither production
plan is modified. Run only in the root-controlled exclusive GPU window.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORK_ROOT = "/root/katagomo-fullflow"
FORK_RUN = FORK_ROOT + "/.final-migration-env/results/sm120-top3-from-4-32-s2-gpu0"
FORK_MODEL = FORK_ROOT + "/.final-migration-env/assets/b11c768h12nbt3tflrs-fson-silu.bin.gz"
MODEL = Path("D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz")
MODEL_SHA = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6"
FORK_BINARY = FORK_RUN + "/build/katago"
FORK_PLAN = FORK_RUN + "/best-tactic-plan.json"
FORK_BINARY_SHA = "5df63305086e42f67e3171874976bd8c7d441c36e8603e899d051419c6c00868"
FORK_PLAN_SHA = "b5b4701b58f94b5878e38f40062435d22dad60de07ae374d1dee810829e3162b"
FORK_MARKERS = (
    "initial-conv frontend eng47/k2=2/k6=1/k13=1/k14=0/k22=2 active",
    "one-warp C384 RMSNorm active",
    "wide QKV AOT active, tactic=wide_qkv-m128-n128-k64-s2-cute-atom4x2-packed",
    "FA4 AOT active, tactic=fa4-b14-s361-h12-d32-tm128-tn96-s1-qk16",
    "out-projection residual AOT active, tactic=outproj-m128-n128-k32-s3-t128-mb3-tilelang-49k",
    "linear2 residual AOT active, tactic=linear2-m128-n128-k32-s3-cutlass",
    "partial C288 no-split g1+v1 head active",
)
EMPTY_RULES = dict(ko="POSITIONAL", scoring="AREA", tax="NONE", suicide=True,
                   hasButton=False, whiteHandicapBonus="0", friendlyPassOk=False, komi=7.5)
EMPTY_MISC = dict(draw_equivalent_wins_for_white=0.5, conservative_pass_and_is_root=False,
                  enable_passing_hacks=False, playout_doubling_advantage=0.0,
                  nn_policy_temperature=1.0, avoid_mytdagger_hack=False, max_history=1000)


def wsl(command):
    return ["wsl", "-d", "Ubuntu-24.04", "--", "bash", "-lc", command]


def capture(command):
    return subprocess.check_output(command, text=True, encoding="utf-8").strip()


def require(condition, message):
    # Validation must remain active under python -O.
    if not condition:
        raise ValueError(message)


def wsl_sha(path):
    return capture(wsl(shlex.join(["sha256sum", path]))).split()[0]


def positive(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0, f"invalid {label}: {value!r}")
    return float(value)


def matching_rate(reported, expected, label):
    positive(reported, label)
    require(math.isclose(reported, expected, rel_tol=1e-7, abs_tol=1e-6),
            f"{label} mismatch: reported={reported}, recomputed={expected}")


def same_path(actual, expected):
    # Normalize slash spelling only; do not permit another model with the same name.
    return isinstance(actual, str) and actual.replace("\\", "/") == str(expected).replace("\\", "/")


def parse_report(stdout):
    # Logs precede JSON in both engines. Try line starts, rather than accepting
    # an arbitrary brace from a diagnostic as a result object.
    decoder = json.JSONDecoder()
    reports = []
    for match in re.finditer(r"(?m)^[ \t]*\{", stdout):
        try:
            result, _ = decoder.raw_decode(stdout[match.start():].lstrip())
            if not isinstance(result, dict):
                continue
            rust = result.get("command") == "nnbench" and isinstance(result.get("results"), list)
            fork = all(key in result for key in ("modelFile", "batchSize", "numServerThreads", "combinedNNEvalsPerSec"))
            if rust or fork:
                reports.append(result)
        except ValueError:
            pass
    require(len(reports) == 1, f"expected one benchmark report in stdout, found {len(reports)}")
    return reports[0]


def fork_event_rate(data, log, iterations):
    require(same_path(data.get("modelFile"), FORK_MODEL), "Fork model path mismatch")
    require(data.get("batchSize") == 14 and data.get("numServerThreads") == 2, "Fork B14/S2 mismatch")
    require(data.get("numIterations") == iterations and data.get("phaseOffsetUs") == 0,
            "Fork iteration count/barrier mismatch")
    require(data.get("gpuIdxs") == [0], "Fork device assignment mismatch")
    devices = data.get("cudaDevices", [])
    require(len(devices) == 1 and devices[0].get("name") == "NVIDIA GeForce RTX 5070 Ti"
            and devices[0].get("multiProcessorCount") == 70
            and devices[0].get("computeCapabilityMajor") == 12
            and devices[0].get("computeCapabilityMinor") == 0, "Fork device fingerprint mismatch")
    require(f"from {FORK_PLAN} fileSha256={FORK_PLAN_SHA} B14 streamsPerDevice=2 evaluatorThreads=2" in log,
            "Fork did not load the hash-bound certified plan")
    for lane in (0, 1):
        require(f"Cuda backend thread {lane}: Model version 17 useFP16 = true useNHWC = true" in log,
                f"Fork lane {lane} model/precision marker missing")
    for marker in FORK_MARKERS:
        require(log.count("SM120 backend: " + marker) >= 2,
                f"Fork selected tactic not observed twice: {marker}")
    medians, rates = data.get("perServerMedianMs", []), data.get("perServerNNEvalsPerSec", [])
    require(len(medians) == 2 and len(rates) == 2, "Fork per-lane sample summaries missing")
    recomputed = []
    for lane, (median, rate) in enumerate(zip(medians, rates)):
        expected = 14000.0 / positive(median, f"Fork lane {lane} median")
        matching_rate(rate, expected, f"Fork lane {lane} rate")
        recomputed.append(expected)
    matching_rate(data.get("combinedNNEvalsPerSec"), sum(recomputed), "Fork median-rate sum")
    return sum(recomputed)


def rust_event_rate(data, log, *, iterations, warmup, input_kind, model, tactics):
    require(data.get("command") == "nnbench" and data.get("mode") == "kernel"
            and data.get("timing") == "cuda-event" and data.get("kernel_only") is True,
            "Rust benchmark mode/timing mismatch")
    require(same_path(data.get("model"), model) and data.get("model_sha256") == MODEL_SHA,
            "Rust model path/hash mismatch")
    require(data.get("input") == input_kind, "Rust input control mismatch")
    require(data.get("warmup") == warmup and data.get("iterations") == iterations,
            "Rust warmup/iteration count mismatch")
    require(data.get("requested_batches") == [14] and data.get("handles") == [1, 2],
            "Rust batch/handle request mismatch")
    require(data.get("device_target") == "sm_120", "Rust device target mismatch")
    require(isinstance(data.get("build", {}).get("cuda_compiler"), str)
            and data["build"]["cuda_compiler"], "Rust CUDA compiler provenance missing")
    actual_tactics = data.get("tactics", {})
    require(all(actual_tactics.get(key) == value for key, value in tactics.items()),
            "Rust effective tactics differ from requested tactics")
    hidden = {key: value for key, value in actual_tactics.items()
              if key not in tactics and value is not None}
    require(not hidden, f"unexpected effective Rust tactics: {hidden}")
    description = data.get("input_description", {})
    require(description.get("kind") == input_kind and description.get("inputs_version") == 7
            and description.get("board") == [19, 19], "Rust input description mismatch")
    if input_kind == "empty":
        require(description.get("rules") == EMPTY_RULES and description.get("encore_phase") == 0
                and description.get("effective_symmetry") == 0
                and description.get("policy_optimism") == 0.0
                and description.get("misc_params") == EMPTY_MISC, "Rust empty-board rules/params mismatch")
    rows = data.get("results", [])
    require(len(rows) == 1 and rows[0].get("batch") == 14, "Rust result batch mismatch")
    row = rows[0]
    for key in ("spatial_input_sha256", "global_input_sha256"):
        require(isinstance(row.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", row[key]),
                f"Rust {key} missing/invalid")
    handles = row.get("handles", [])
    require([handle.get("handle") for handle in handles] == [1, 2], "Rust result lane IDs mismatch")
    expected_rates = []
    for handle in handles:
        values = handle.get("cuda_event_ms", [])
        require(handle.get("iterations") == iterations and len(values) == iterations,
                "Rust lane event sample count mismatch")
        median = statistics.median([positive(value, "Rust event time") for value in values])
        matching_rate(handle.get("cuda_event_median_ms"), median, "Rust lane median")
        rate = 14000.0 / median
        matching_rate(handle.get("cuda_event_median_nn_evals_per_s"), rate, "Rust lane median-rate")
        expected_rates.append(rate)
    matching_rate(row.get("sum_median_nn_evals_per_s"), sum(expected_rates), "Rust median-rate sum")
    layout = tactics["KATAGO_CUDA_GEMM_LAYOUT"]
    launch = "tn" if layout in ("tn", "nn_k384_b16") else "nn_k384"
    require(re.search(rf"name=gemm_layout launch={launch}(?:\s|$)", log), "Rust layout did not execute")
    require(re.search(r"name=dual_ffn launch=fused(?:\s|$)", log), "Rust DualFFN did not execute")
    require(re.search(r"tile=q64-serial(?:\s|$)", log), "Rust serial attention did not execute")
    return sum(expected_rates)


def summarize(runs, maximum_relative_spread):
    summary = {}
    for flavor in sorted({run["flavor"] for run in runs}):
        values = [positive(run["event_rate_sum"], "run rate") for run in runs if run["flavor"] == flavor]
        summary[flavor] = {"samples": values, "geomean": math.exp(sum(map(math.log, values)) / len(values)),
                           "relative_spread": (max(values) - min(values)) / min(values)}
    stable = set(summary) == {"fork", "rust"} and all(
        len(row["samples"]) >= 2 and row["relative_spread"] <= maximum_relative_spread
        for row in summary.values())
    if "fork" in summary and "rust" in summary:
        key = "rust_over_fork_event_ratio" if stable else "diagnostic_only_rust_over_fork_event_ratio"
        summary[key] = summary["rust"]["geomean"] / summary["fork"]["geomean"]
    return stable, summary


def fork_command(iterations, warmup):
    overrides = {
        "cudaTacticPlanFile": FORK_RUN + "/best-tactic-plan.json",
        "cudaTacticPlanBatch": "14", "nnMaxBatchSize": "14",
        "numNNServerThreadsPerModel": "2", "nnBatchAwareDispatch": "true",
        "cudaWarmupOnlyMaxBatchSize": "true", "cudaAsyncInferPipeline": "true",
        "cudaEventPipelineUseGraph": "false", "requireMaxBoardSize": "true",
        "cudaDeviceToUseThread0": "0", "cudaDeviceToUseThread1": "0",
    }
    argv = [FORK_RUN + "/build/katago", "benchmarknn", "-model", FORK_MODEL,
            "-config", FORK_ROOT + "/cpp/configs/gtp_example.cfg", "-override-config",
            ",".join(f"{key}={value}" for key, value in overrides.items()),
            "-iterations", str(iterations), "-warmup", str(warmup),
            "-batch-size", "14", "-boardsize", "19", "-phase-offset-us", "0", "-json"]
    return wsl("cd " + shlex.quote(FORK_ROOT) + " && " + shlex.join(argv))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-platform", choices=["windows", "wsl"], default="wsl")
    parser.add_argument("--rust-binary", default=None)
    parser.add_argument("--layout", choices=["tn", "nn_k384", "nn_k384_b8", "nn_k384_b16"], default="tn")
    parser.add_argument("--input", choices=["positions", "empty"], default="empty")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--maximum-relative-spread", type=float, default=0.1)
    parser.add_argument("--order", default="fork,rust,rust,fork")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    order = args.order.split(",")
    if args.iterations < 100 or args.warmup < 1 or any(x not in ("fork", "rust") for x in order):
        parser.error("require iterations>=100, warmup>=1, and fork/rust order")
    if not math.isfinite(args.maximum_relative_spread) or not 0 <= args.maximum_relative_spread <= 0.1:
        parser.error("maximum-relative-spread must be finite and between 0 and 0.1")
    args.output.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA:
        parser.error("local TF3 model identity mismatch")
    fork_sha = wsl_sha(FORK_MODEL)
    if fork_sha != MODEL_SHA:
        parser.error("Fork TF3 model identity mismatch")
    require(wsl_sha(FORK_BINARY) == FORK_BINARY_SHA, "Fork binary differs from the certified executable")
    require(wsl_sha(FORK_PLAN) == FORK_PLAN_SHA, "Fork plan differs from the certified B14/S2 plan")
    plan = json.loads(capture(wsl(shlex.join(["cat", FORK_PLAN]))))
    require(plan["target"]["model_sha256"] == MODEL_SHA and plan["target"]["streams"] == 2
            and plan["batches"] == [14] and plan["final_joint"]["14"]["binary_sha256"] == FORK_BINARY_SHA,
            "Fork plan model/batch/topology/binary binding mismatch")
    fork_version = capture(wsl(shlex.join([FORK_BINARY, "version"])))

    tactics = {"KATAGO_CUDA_DUALFFN": "1", "KATAGO_CUDA_ATTN_TILE": "q64-serial",
               "KATAGO_CUDA_NOGRAPH": "1", "KATAGO_CUDA_GEMM_LAYOUT": args.layout}
    rust_binary = args.rust_binary or (
        "/mnt/d/code/Rust_KataGo/target/cudarocmopt-wsl/release/katago-rs"
        if args.rust_platform == "wsl" else str(ROOT / "target/release/katago-rs.exe"))
    rust_model = "/mnt/d/Go/Server/models/" + MODEL.name if args.rust_platform == "wsl" else str(MODEL)
    rust_args = [rust_binary, "nnbench", "--model", rust_model, "--mode", "kernel",
                 "--handles", "1,2", "--batch", "14", "--iterations", str(args.iterations),
                 "--warmup", str(args.warmup), "--timing", "cuda-event", "--input", args.input, "--json"]
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("KATAGO_CUDA_") and key != "KATAGO_NN_BATCH_WINDOW_US"}
    if args.rust_platform == "wsl":
        # env -u prevents the launch shell from supplying hidden tactic values.
        unset = [item for key in capture(wsl("compgen -A variable KATAGO_CUDA_ || true")).splitlines()
                 for item in ("-u", key)]
        unset += ["-u", "KATAGO_NN_BATCH_WINDOW_US"]
        rust_args = wsl(shlex.join(["env", *unset,
            "LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:/usr/lib/wsl/lib",
            *[f"{key}={value}" for key, value in tactics.items()], *rust_args]))
        rust_sha = wsl_sha(rust_binary)
        require(wsl_sha(rust_model) == MODEL_SHA, "actual WSL Rust model path/hash mismatch")
    else:
        environment.update(tactics)
        rust_sha = hashlib.sha256(Path(rust_binary).read_bytes()).hexdigest()
    report = {
        "schema": 2, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_sha256": MODEL_SHA, "physical_batch": 14, "streams": 2,
        "warmup": args.warmup, "iterations": args.iterations, "order": order,
        "rust_platform": args.rust_platform, "rust_tactics": tactics, "input": args.input,
        "maximum_relative_spread": args.maximum_relative_spread,
        "rust_binary_sha256": rust_sha,
        "fork_binary_sha256": FORK_BINARY_SHA, "fork_plan_sha256": FORK_PLAN_SHA,
        "fork_version": fork_version,
        "toolchain_caveat": "Fork is the pinned CUDA 13.0.88/cuDNN 9.24 build; Rust records its own CUDA compiler per run. Toolchains and runtime libraries are not asserted equal.",
        "precision": {"fork": "certified qk16; some residual GEMMs also accumulate in FP16",
                      "rust": "FP16 operands, FP32 accumulation, precise expf"},
        "comparison_metric": "sum of per-stream batch / median(CUDA event forward milliseconds)",
        "output_workload_caveat": "Both compute all semantic policy, value/score and ownership heads. Rust also computes zero-padded compatibility channels (policy 6 vs native 2; terminal 21 vs native 9); the operation graphs are not identical.",
        "input_caveat": ("Both use the same semantic empty 19x19 Black-to-play TrompTaylorish 7.5-komi control; cross-implementation feature-byte equality is not asserted."
                         if args.input == "empty" else "Fork repeats one empty TrompTaylorish board; Rust uses different deterministic legal histories."),
        "sample_caveat": "Rust raw per-lane CUDA event samples are verified. Fork JSON exposes two medians/rates and a per-lane iteration count, not raw samples; its bound source checks vector lengths. One-shot tactic markers lack lane IDs, so counts and both lane model markers are checked separately.",
        "wall_caveat": "Fork wall includes warmup/setup/teardown; Rust wall excludes them. No wall ratio computed.",
        "runs": [],
    }
    output = args.output / "report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    input_identity = None
    for index, flavor in enumerate(order):
        command = fork_command(args.iterations, args.warmup) if flavor == "fork" else rust_args
        if flavor == "fork":
            require(wsl_sha(FORK_BINARY) == FORK_BINARY_SHA and wsl_sha(FORK_PLAN) == FORK_PLAN_SHA,
                    "Fork executable/plan changed during the experiment")
        else:
            current_sha = wsl_sha(rust_binary) if args.rust_platform == "wsl" else hashlib.sha256(Path(rust_binary).read_bytes()).hexdigest()
            require(current_sha == rust_sha, "Rust executable changed during the experiment")
        print(f"starting {index + 1}/{len(order)} {flavor}", flush=True)
        result = subprocess.run(command, text=True, capture_output=True, encoding="utf-8",
                                env=environment, timeout=600)
        stem = args.output / f"{index + 1}-{flavor}"
        stem.with_suffix(".stdout.log").write_text(result.stdout, encoding="utf-8")
        stem.with_suffix(".stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{flavor} failed ({result.returncode}); see {stem}.stderr.log")
        data = parse_report(result.stdout)
        if flavor == "fork":
            metric = fork_event_rate(data, result.stdout + result.stderr, args.iterations)
        else:
            metric = rust_event_rate(data, result.stderr, iterations=args.iterations, warmup=args.warmup,
                                     input_kind=args.input, model=rust_model, tactics=tactics)
            row = data["results"][0]
            identity = (row["spatial_input_sha256"], row["global_input_sha256"], data["input_description"])
            require(input_identity is None or identity == input_identity, "Rust input changed between repetitions")
            input_identity = identity
        report["runs"].append(dict(flavor=flavor, command=command, event_rate_sum=metric, result=data))
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{flavor}: event median-rate sum {metric:.1f}", flush=True)
    report["stable"], summary = summarize(report["runs"], args.maximum_relative_spread)
    report["status"] = "STABLE" if report["stable"] else "UNSTABLE_DIAGNOSTIC_ONLY"
    report["summary"] = summary
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

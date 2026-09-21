#!/usr/bin/env python3
"""Generate a NEW CUDA tactic plan on the target machine for native TF3 v17.

Each candidate: real nnworker outputs vs portable C++ FP32 gold -> executed-path
checks -> sequential ABBA. Never rebind an existing certificate. No compilation
or production Server is required when a compatible CUDA binary/reference exists.
"""
import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import autotune_reference as reference_io
import tune_runtime as runtime
from validate_cuda_tactics import has_marker

ROOT = runtime.ROOT
PREFIX = "KATAGO_CUDA_"
GROUPS = ("batch", "lanes_graph", "gemm", "dual_ffn", "attention", "layout", "rank",
          "fusion", "rms", "splitk", "batching", "specialized", "reschedule", "threads")
# Pin every boolean/enum default except RMS: the engine only accepts the opt-in
# 'v1' value; absence selects w4. The binary hash and clean launchers bind w4.
DEFAULTS = {PREFIX + key: value for key, value in {
    "ATTN": "fa2", "ATTN_TILE": "q128", "CUBLASLT": "1", "CUBLASLT_RANK": "heuristic",
    "RESIDUAL_ALGO": "heuristic", "GEMM_LAYOUT": "tn", "DUALFFN": "0", "FUSION": "none",
    "NOGRAPH": "0", "NOPIPELINE": "0", "PADBATCH": "0", "SPLITK": "0", "T32": "0", "T64N32": "0",
    "STEM_MAP3D": "0", "GATE_ROWQUAD_R1": "0", "QKV_IMMUTABLE_R1": "0",
    "QKV_CLASSIC_N128_R1": "0", "OUTPROJ_CLASSIC_N128_R1": "0", "FFN_COMPACT_R1": "0",
}.items()}
DEFAULTS["KATAGO_NN_BATCH_WINDOW_US"] = "8000"


def state(tactics=None, batch=8, lanes=1, threads=8):
    return dict(tactics=dict(DEFAULTS if tactics is None else tactics), batch=batch, lanes=lanes, threads=threads)


def state_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def changed(value, overrides=None, **settings):
    result = copy.deepcopy(value)
    result.update(settings)
    for key, item in (overrides or {}).items():
        full_key = key if key.startswith("KATAGO_") else PREFIX + key
        if item is None:
            result["tactics"].pop(full_key, None)
        else:
            result["tactics"][full_key] = item
    return result


def specialized_states(value, fingerprint):
    """Reuse *candidate definitions* only on their exact supported identity.

    These still undergo new numeric gates and ABBA; the old selection history is
    never copied into a new certificate. Other GPUs use the portable groups.
    """
    found, _ = runtime.discover_profiles(fingerprint)
    return [state({**DEFAULTS, **json.loads(Path(p["plan"]).read_text(encoding="utf-8"))["apply"]["tactic_overrides"]},
                  p["batch"], p["lanes"], value["threads"]) for p in found if p["plan"]]


def candidates(group, current, fingerprint, args):
    caps = fingerprint["backend_build"]["capabilities"]
    architecture = fingerprint.get("model_architecture", {})
    if group in ("dual_ffn", "fusion", "splitk") and architecture.get("supports_fixed_ffn_tactics") is False:
        return []
    if group == "layout" and architecture.get("supports_k384_layout") is False:
        return []
    tweaks = []
    if group == "batch":
        return [changed(current, batch=b) for b in args.batches]
    if group in ("lanes_graph", "reschedule"):
        batches = args.batches if group == "reschedule" else [current["batch"]]
        for batch in batches:
            for lanes in (1, 2):
                for graph_off in ("0", "1"):
                    # Strict B14 AOT has a direct-launch-only contract.
                    if current["tactics"][PREFIX + "ATTN"] == "fa4-strict-b14-r1" and (batch != 14 or graph_off != "1"):
                        continue
                    tweaks.append(changed(current, {"NOGRAPH": graph_off}, batch=batch, lanes=lanes))
        return tweaks
    if group == "gemm":
        return [changed(current, {"CUBLASLT": lt, "T32": t32, "T64N32": t64,
                                  "DUALFFN": "0", "SPLITK": "0", "FUSION": "none"})
                for lt, t32, t64 in (("1", "0", "0"), ("0", "0", "0"), ("0", "1", "0"), ("0", "0", "1"))]
    if group == "dual_ffn":
        return [changed(current, {"DUALFFN": v}) for v in ("0", "1") if v == "0" or caps.get("dual_ffn")]
    if group == "attention":
        tiles = ["q128"]
        if caps.get("attention_q64"):
            tiles.append("q64")
        if caps.get("attention_q64_serial"):
            tiles.append("q64-serial")
        return ([changed(current, {"ATTN": "fa2", "ATTN_TILE": tile}) for tile in tiles]
                + [changed(current, {"ATTN": "v3", "ATTN_TILE": "q128"})])
    if group == "layout":
        if current["tactics"][PREFIX + "CUBLASLT"] != "1":
            return []
        return [changed(current, {"GEMM_LAYOUT": v}) for v in ("tn", "nn_k384", "nn_k384_b8", "nn_k384_b16")
                if v != "nn_k384_b16" or current["batch"] >= 16]
    if group == "rank":
        return ([changed(current, {"CUBLASLT_RANK": v}) for v in ("heuristic", "time")]
                if current["tactics"][PREFIX + "CUBLASLT"] == "1" else [])
    if group == "fusion":
        return [changed(current, {"FUSION": v, "DUALFFN": "0"}) for v in ("none", "up", "down", "all")]
    if group == "rms":
        return [changed(current, {"RMS": v}) for v in (None, "v1")]
    if group == "splitk":
        return [changed(current, {"SPLITK": v}) for v in ("0", "1")]
    if group == "batching":
        return ([changed(current, {"PADBATCH": v}) for v in ("0", "1")]
                + [changed(current, {"NOPIPELINE": v}) for v in ("0", "1")]
                + [changed(current, {"KATAGO_NN_BATCH_WINDOW_US": v}) for v in ("3000", "8000", "12000")])
    if group == "specialized":
        return specialized_states(current, fingerprint)
    if group == "threads":
        return [changed(current, threads=t) for t in args.threads] if args.mode == "local" else []
    raise ValueError(f"unknown group: {group}")


def expected_markers(value):
    t = value["tactics"]
    # Compact FFN dispatches through its own DualGemm executor and emits
    # ffn_compact stage=execute below, not dual_ffn launch=fused. Require its
    # capability probe AND execution marker instead of rejecting a live path.
    dual_marker = ("name=dual_ffn requested=1 compiled=1 probe=pass"
                   if t[PREFIX + "FFN_COMPACT_R1"] == "1" else
                   "name=dual_ffn launch=" + ("fused" if t[PREFIX + "DUALFFN"] == "1" else "unfused"))
    required = [dual_marker,
                "name=graph requested=" + ("direct launch=direct" if t[PREFIX + "NOGRAPH"] == "1" else "graph launch=graph"),
                "name=rms launch=" + ("v1" if t.get(PREFIX + "RMS") == "v1" else "w4"),
                "name=fusion mode=" + t[PREFIX + "FUSION"]]
    attention = t[PREFIX + "ATTN"]
    if attention == "fa4-strict-b14-r1":
        required.append("name=attention requested=fa4-strict-b14-r1 launch=strict-aot")
    elif attention == "v3":
        required.append("name=attention requested=v3 launch=v3 effective=v3")
    else:
        required.append("name=attention requested=fa2 launch=fa2 effective=fa2 tile=" + t[PREFIX + "ATTN_TILE"])
    if t[PREFIX + "FUSION"] == "none" and t[PREFIX + "ATTN"] != "fa4-strict-b14-r1":
        required.append("name=gemm kind=f16out engine=" + ("cublaslt" if t[PREFIX + "CUBLASLT"] == "1" else "handwritten"))
    for key, marker in (
        ("SPLITK", "name=splitk launch=on"), ("PADBATCH", "name=padbatch launch=on"),
        ("T32", "tile=hgemm_t32"), ("T64N32", "tile=hgemm_t64n32"),
        ("STEM_MAP3D", "name=stem_global_map requested=1 launch=map3d effective=1"),
        ("GATE_ROWQUAD_R1", "name=gate_rowquad requested=1 effective=1"),
        ("QKV_IMMUTABLE_R1", "name=qkv_immutable requested=1 effective=1"),
        ("QKV_CLASSIC_N128_R1", "name=qkv_classic_n128 requested=1 effective=1"),
        ("OUTPROJ_CLASSIC_N128_R1", "name=outproj_classic_n128 requested=1 effective=1"),
        ("FFN_COMPACT_R1", "name=ffn_compact stage=execute effective=1"),
    ):
        if t[PREFIX + key] == "1":
            required.append(marker)
    if t[PREFIX + "CUBLASLT_RANK"] == "time":
        required.append("name=cublaslt_rank engine=time")
    layout = t[PREFIX + "GEMM_LAYOUT"]
    if layout != "tn":
        required.append("name=gemm_layout launch=nn_k384 effective=nn_k384 k=384 requested=" + layout)
    return required


def verify_paths(log, value):
    missing = [marker for marker in expected_markers(value) if not has_marker(log, marker)]
    if missing:
        raise ValueError("requested paths not observed: " + "; ".join(missing))
    if "probe=fail" in log or "compiled=0 effective=q128" in log:
        raise ValueError("CUDA tactic silently fell back")


def new_plan(fingerprint, value, identifier, selection):
    target = {key: fingerprint[key] for key in (*runtime.TARGET_KEYS, "architecture")}
    target["max_batch_size"] = value["batch"]
    return dict(schema=2, kind="cuda-tactic-plan", plan_id=identifier,
                generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(), target=target,
                backend_build=copy.deepcopy(fingerprint["backend_build"]),
                apply=dict(tactic_overrides=copy.deepcopy(value["tactics"])), selection=selection)


class FullTuner(runtime.Tuner):
    def __init__(self, args, output):
        super().__init__(args, output)
        self.numeric_cache = {}
        self.report.update(tool="full_autotune", scope="native TF3 v17; fixed 128-case C++ FP32 corpus",
                           search_scope="FULL_CATALOG" if args.groups == list(GROUPS) else "CUSTOM_GROUPS",
                           groups=args.groups, numerical=[], groups_completed=[], skipped=[],
                           numerical_scope="Every measured configuration must pass C++ FP32 outputs and actual kernel-path checks.")

    def prepare_reference(self):
        from compare_worker_outputs import build_requests
        self.prepare_worker()  # Temporary protocol + separate uncached performance request set.
        fixture = json.loads((ROOT / "scripts/fixtures/worker_positions.json").read_text(encoding="utf-8"))
        self.numeric_requests = build_requests(self.protocol.pb, fixture, self.fingerprint["model_sha256"], 0)
        self.numeric_fingerprint = reference_io.fingerprints(self.numeric_requests, self.fingerprint["model_sha256"])
        path = self.args.reference or reference_io.load_bundled(self.fingerprint["model_sha256"])
        if path:
            self.reference = reference_io.validate_reference(reference_io.read_reference(path), self.numeric_fingerprint,
                                                              self.numeric_requests, self.protocol)
            self.report["reference_source"] = str(path.resolve())
        elif self.args.cpp_worker:
            if self.args.dry_run:
                self.report["reference_source"] = "WILL_COLLECT_CPP_FP32_ON_REAL_RUN"
                return
            self.reference = reference_io.collect_reference(
                self.args.model, self.args.cpp_worker, self.args.cpp_config, self.output / "reference-collection",
                self.protocol, self.numeric_requests, self.numeric_fingerprint, self.args.timeout)
            self.report["reference_source"] = "fresh C++ CUDA FP32 collection"
        else:
            raise ValueError("No FP32 reference for this model. Supply --reference <exported.json.gz> or "
                             "--cpp-worker <C++ nnworker binary> --cpp-config <FP32.cfg>. No plan will be certified without it.")
        destination = self.output / "reference.json.gz"
        reference_io.write_reference(destination, self.reference)
        self.report["reference_sha256"] = runtime.file_hash(destination)
        self.report["numeric_request_fingerprint"] = self.numeric_fingerprint
        self.report["reference_binary_sha256"] = self.reference["binary_sha256"]

    def materialize(self, value):
        identifier = state_id(value)
        directory = self.output / "candidates" / identifier
        directory.mkdir(parents=True, exist_ok=True)
        plan_path = directory / "candidate-plan.json"
        config_path = directory / "candidate.cfg"
        plan = new_plan(self.fingerprint, value, "full-autotune-candidate-" + identifier,
                        dict(status="UNVALIDATED_CANDIDATE", runtime=value, note="Not a final deployment certificate"))
        if not plan_path.exists():
            runtime.save_json(plan_path, plan)
        profile = dict(name=identifier, batch=value["batch"], lanes=value["lanes"], plan=str(plan_path))
        config_path.write_text(runtime.config_text(profile, value["threads"], self.args.max_visits), encoding="utf-8")
        return dict(name=identifier, config=str(config_path), threads=value["threads"], profile=profile, state=value)

    def numeric(self, candidate, force=False):
        from compare_worker_outputs import collect_worker, compare_outputs
        key = candidate["name"]
        if key in self.numeric_cache and not force:
            return self.numeric_cache[key]
        value = candidate["state"]
        # Exercise B1 and both the requested deployment concurrency and a high
        # enough burst to actually reach the requested physical batch/lane path.
        windows = sorted({1, self.args.capacity, min(128, max(32, 2 * value["batch"] * value["lanes"]))})
        record = dict(candidate=key, windows=windows, status="RUNNING", comparisons=[])
        self.report["numerical"].append(record)
        try:
            combined_log = ""
            for window in windows:
                self.counter += 1
                directory = self.output / f"{self.counter:03d}-numeric-{key}-w{window}"
                directory.mkdir()
                print(f"  numeric {key}: 128 cases, W{window}", flush=True)
                args = SimpleNamespace(output=directory, request_window=window, model=self.args.model,
                                       startup_timeout=self.args.timeout, task_timeout=self.args.timeout)
                with (directory / "collection.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                    actual = collect_worker("rust", self.args.binary, Path(candidate["config"]), args, self.protocol,
                                            self.numeric_requests, self.numeric_fingerprint, self.environment)
                if actual["hello"]["model_version"] != self.reference["hello"]["model_version"]:
                    raise ValueError("candidate and reference model versions differ")
                comparison = compare_outputs(self.reference, actual)
                runtime.save_json(directory / "comparison.json", comparison)
                record["comparisons"].append(dict(window=window, path=str(directory / "comparison.json"),
                                                   result=comparison["result"], top1=comparison["policy_top1_matches"]))
                if comparison["result"] != "PASS":
                    raise ValueError(f"FP32 numeric gate failed at W{window}; see {directory / 'comparison.json'}")
                combined_log += (directory / "rust/worker.log").read_text(encoding="utf-8", errors="replace") + "\n"
            verify_paths(combined_log, value)
            record.update(status="PASS", observed_markers=expected_markers(value))
            self.numeric_cache[key] = True
            print(f"  numeric {key}: PASS", flush=True)
            return True
        except Exception as error:
            record.update(status="REJECTED", error=str(error))
            self.numeric_cache[key] = False
            print(f"  numeric {key}: REJECTED: {error}", flush=True)
            return False
        finally:
            self.save()

    def measure(self, candidate):
        if not self.numeric_cache.get(candidate["name"]):
            raise RuntimeError("refusing performance measurement before the numeric/path gate")
        return super().measure(candidate)

    def compare_candidate(self, incumbent, candidate, stage):
        # Candidate failures keep the incumbent; baseline failures abort the run.
        first = self.measure(incumbent)
        try:
            middle = [self.measure(candidate), self.measure(candidate)]
        except Exception as error:
            self.report["comparisons"].append(dict(stage=stage, baseline=incumbent["name"], candidate=candidate["name"],
                                                    accepted=False, status="CANDIDATE_FAILED", error=str(error)))
            self.save()
            return False
        last = self.measure(incumbent)
        decision = runtime.decide([first, last], middle, self.args.min_improvement, self.args.max_spread)
        self.report["comparisons"].append(dict(stage=stage, baseline=incumbent["name"], candidate=candidate["name"],
                                                samples=[first, *middle, last], **decision))
        self.save()
        print(f"  {stage}: {'ACCEPT' if decision['accepted'] else 'KEEP'} "
              f"{decision['improvement']:+.2%}, stable={decision['stable']}", flush=True)
        return decision["accepted"]

    def execute(self):
        args = self.args
        self.report["binary_sha256"] = runtime.file_hash(args.binary)
        self.report["tool_sources"] = {name: runtime.file_hash(ROOT / "scripts" / name) for name in (
            "full_autotune.py", "autotune_reference.py", "tune_runtime.py", "compare_worker_outputs.py", "benchmark_workers.py")}
        self.fingerprint = json.loads(self.run_command([str(args.binary), "cuda-fingerprint", "--model", str(args.model)], "fingerprint"))
        if self.fingerprint["model_sha256"] != runtime.file_hash(args.model):
            raise ValueError("model file/fingerprint mismatch")
        if self.fingerprint["backend_build"].get("fp16_encoding_revision") != 1:
            raise ValueError("CUDA binary lacks the required FP16 encoding contract")
        self.report["fingerprint"] = self.fingerprint
        print(f"GPU: {self.fingerprint['gpu_name']}; model: {self.fingerprint['model_sha256']}", flush=True)
        baseline = state(batch=args.batches[0], threads=args.threads[0] if args.mode == "local" else 1)
        self.report["baseline"] = baseline
        self.report["limits"] = ["Native TF3 v17 only; no ONNX or arbitrary model-architecture support",
                                 "Ordered coordinate search, not the Cartesian product of all kernels",
                                 "FP32 postprocessed Worker corpus, not independent raw-tensor certification",
                                 "Plan is scoped to the tested runtime, corpus and workload; no Elo claim"]
        self.prepare_reference()
        if args.dry_run:
            self.report["catalog_at_baseline"] = {g: candidates(g, baseline, self.fingerprint, args) for g in args.groups}
            self.report["status"] = "DRY_RUN"
            return
        original = winner = self.materialize(baseline)
        if not self.numeric(original):
            raise ValueError("baseline failed the C++ FP32/path gate; do not tune on an incorrect baseline")
        for group in args.groups:
            print(f"\n== group {group} ==", flush=True)
            options = candidates(group, winner["state"], self.fingerprint, args)
            if not options:
                self.report["skipped"].append(dict(group=group, reason="unsupported capability, identity, or mode/dependency"))
            seen = set()
            for value in options:
                key = state_id(value)
                if key == winner["name"] or key in seen:
                    continue
                seen.add(key)
                candidate = self.materialize(value)
                if self.numeric(candidate) and self.compare_candidate(winner, candidate, group):
                    winner = candidate
            self.report["groups_completed"].append(group)
            self.report["current_winner"] = winner
            self.save()
        if winner["name"] != original["name"]:
            if not self.compare_candidate(original, winner, "final-confirmation"):
                winner = original
        else:
            # A successful run may find no improvement. Measure twice without
            # relabeling an unstable baseline as an optimized speed result.
            rates = [self.measure(original), self.measure(original)]
            self.report["baseline_rates"] = rates
            self.report["baseline_stable"] = max(rates) / min(rates) - 1 <= args.max_spread
        if not self.numeric(winner, force=True):
            raise ValueError("final numerical revalidation failed; no deployment plan published")
        if runtime.file_hash(args.binary) != self.report["binary_sha256"] or runtime.file_hash(args.model) != self.fingerprint["model_sha256"]:
            raise ValueError("binary/model changed during tuning")
        current_fp = json.loads(self.run_command([str(args.binary), "cuda-fingerprint", "--model", str(args.model)], "final-fingerprint"))
        if current_fp != self.fingerprint:
            raise ValueError("CUDA fingerprint changed during tuning")
        self.publish(winner, original)

    def publish(self, winner, original):
        # A READY marker publishes the exact final plan/config only after they
        # have been tested together through the engine.
        value = winner["state"]
        staging = self.output / "staging"
        staging.mkdir()
        final_dir = self.output / "result"
        # Config uses its final absolute plan path. Temporarily create the final
        # directory for validation; a READY marker is the only publication gate.
        final_dir.mkdir()
        plan_path, config_path = final_dir / "plan.json", final_dir / "rustgo.cfg"
        self.report["selection_status"] = "IMPROVED" if winner["name"] != original["name"] else "BASELINE_RETAINED"
        selection = dict(status="NUMERICALLY_VALIDATED", search_scope=self.report["search_scope"],
                         selection_status=self.report["selection_status"], runtime=value,
                         reference_sha256=self.report["reference_sha256"], numeric_fingerprint=self.numeric_fingerprint,
                         numerical_windows=sorted({1, self.args.capacity, min(128, max(32, 2 * value["batch"] * value["lanes"]))}),
                         mode=self.args.mode, capacity=self.args.capacity, metric=self.report["metric"],
                         min_improvement=self.args.min_improvement, max_spread=self.args.max_spread,
                         binary_sha256=self.report["binary_sha256"], report="../report.json")
        try:
            runtime.save_json(plan_path, new_plan(self.fingerprint, value, "full-autotune-" + self.output.name, selection))
            profile = dict(name=winner["name"], batch=value["batch"], lanes=value["lanes"], plan=str(plan_path))
            config_path.write_text(runtime.config_text(profile, value["threads"], self.args.max_visits), encoding="utf-8")
            final = dict(winner, config=str(config_path), profile=profile)
            if self.args.mode == "local":
                self.smoke_local(final)
            else:
                self.measure(final)
                self.report["smoke"] = "Final plan through real uncached Worker + NN counters + Drain PASS"
            for mode in (("gtp", "analysis") if self.args.mode == "local" else ("nnworker",)):
                (final_dir / f"run-{mode}.ps1").write_text(runtime.launch_script(
                    self.args.binary, self.args.model, config_path, mode, self.args.capacity, self.fingerprint["model_sha256"]), encoding="utf-8-sig")
            self.report.update(status="PASS", winner=final, result_directory=str(final_dir),
                               plan_sha256=runtime.file_hash(plan_path), config_sha256=runtime.file_hash(config_path))
            self.save()
            (final_dir / "READY").write_text("PASS: see ../report.json\n", encoding="utf-8")
        except BaseException:
            # Keep failed final artifacts for diagnosis, but outside result/.
            for path in list(final_dir.iterdir()):
                path.replace(staging / path.name)
            final_dir.rmdir()
            raise
        print(f"\nREADY: {final_dir}\nPLAN: {plan_path}\nCFG: {config_path}", flush=True)


def positive_list(value):
    values = runtime.thread_list(value)
    if any(v > 16 for v in values):
        raise argparse.ArgumentTypeError("physical batch candidates must be 1..16")
    return values


def group_list(value):
    values = value.split(",")
    if len(set(values)) != len(values) or any(v not in GROUPS for v in values):
        raise argparse.ArgumentTypeError("groups must be distinct values from: " + ",".join(GROUPS))
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release" / ("katago-rs.exe" if os.name == "nt" else "katago-rs"))
    parser.add_argument("--mode", choices=("worker", "local"), default="worker")
    parser.add_argument("--reference", type=Path, help="Portable C++ FP32 .json.gz; known model has a bundled reference")
    parser.add_argument("--cpp-worker", type=Path, help="Optional reference generator if no portable reference exists")
    parser.add_argument("--cpp-config", type=Path, default=ROOT / "configs/worker_cpp_fp32.cfg")
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--batches", type=positive_list, default=positive_list("8,4,12,14,16"))
    parser.add_argument("--threads", type=runtime.thread_list, default=runtime.thread_list("8,4,12,16"))
    parser.add_argument("--groups", type=group_list, default=list(GROUPS), help="Diagnostic subset; output explicitly records CUSTOM_GROUPS")
    parser.add_argument("--requests", type=int, default=1024)
    parser.add_argument("--ownership", action="store_true")
    parser.add_argument("--visits", type=int, default=800)
    parser.add_argument("--positions", type=int, default=3)
    parser.add_argument("--max-visits", type=int, default=800)
    parser.add_argument("--min-improvement", type=float, default=.01)
    parser.add_argument("--max-spread", type=float, default=.05)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate fingerprint/reference and list candidates; no CUDA inference")
    args = parser.parse_args(argv)
    if args.model.suffix.lower() == ".onnx":
        parser.error("This full tuner targets native TF3 v17 (.bin.gz), not ONNX; an ONNX file cannot use native FP32 reference outputs")
    if not (1 <= args.capacity <= 128 and args.requests >= 1024 and args.visits >= 400 and 1 <= args.positions <= 100 and args.max_visits > 0):
        parser.error("capacity 1..128, requests >=1024, visits >=400, positions 1..100 and positive max-visits required")
    if not (0.01 <= args.min_improvement <= 1 and 0 < args.max_spread <= .05 and math.isfinite(args.timeout) and args.timeout > 0):
        parser.error("min-improvement >=1%, max-spread <=5%, finite positive timeout required")
    for key in ("model", "binary", "reference", "cpp_worker", "cpp_config"):
        value = getattr(args, key)
        if value is not None:
            value = value.resolve()
            if not value.is_file():
                parser.error(f"{key}: file not found: {value}")
            setattr(args, key, value)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = (args.output or ROOT / "autotune-output" / f"full-{stamp}-{args.mode}").resolve()
    if output.exists():
        parser.error("output directory exists; use a new directory (old evidence is never overwritten)")
    output.mkdir(parents=True)
    tuner = FullTuner(args, output)
    # Hold a per-checkout lock to prevent this tool from racing another tuner.
    lock_path = ROOT / "autotune-output/full-autotune.lock"
    lock_path.parent.mkdir(exist_ok=True)
    lock_fd = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, f"pid={os.getpid()}\noutput={output}\n".encode())
        tuner.execute()
    except (Exception, KeyboardInterrupt) as error:
        tuner.report.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAIL", error=str(error))
        print(f"FAILED: {error}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        tuner.save()
        if tuner.protocol:
            tuner.protocol.close()
        if lock_fd is not None:
            os.close(lock_fd)
            lock_path.unlink()
        print(f"Report: {output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

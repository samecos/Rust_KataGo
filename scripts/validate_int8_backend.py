#!/usr/bin/env python3
"""Evaluate INT8 accuracy separately from the unchanged FP32 certification gate.

Runs real local gRPC workers sequentially on 128 fixed semantic requests. The
FP16 control must pass the original C++ FP32 gates. Quantized profiles retain
their original-gate PASS/FAIL and expose errors; this tool never certifies Elo
or writes a production tactic plan. Run int8_kernels first for the independent
CPU integer implementation oracle.
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import autotune_reference as reference_io
from compare_worker_outputs import build_requests, collect_worker, compare_outputs, file_hash, write_json
from tune_runtime import ROOT, clean_environment
from worker_protocol_tools import Protocol


def quantization_metrics(reference, candidate):
    kls, win, score, disagreement = [], [], [], []
    for a, b in zip(reference["results"], candidate["results"], strict=True):
        ref, got = a["result"]["output"], b["result"]["output"]
        p, q = ref["policy"], got["policy"]
        if any((x < 0) != (y < 0) for x, y in zip(p, q, strict=True)):
            raise ValueError("quantized policy changed legality")
        kl = sum(x * math.log(x / max(y, 1e-30)) for x, y in zip(p, q, strict=True) if x > 0)
        kls.append(max(0.0, kl))
        win.append(abs(ref["white_win_prob"] - got["white_win_prob"]))
        score.append(abs(ref["white_score_mean"] - got["white_score_mean"]))
        top = max(range(len(p)), key=p.__getitem__)
        other = max(range(len(q)), key=q.__getitem__)
        if top != other:
            disagreement.append(dict(case=a["name"], reference_top=top, candidate_top=other,
                                     reference_probability_gap=p[top] - p[other]))
    return dict(cases=len(kls), policy_kl_mean=sum(kls) / len(kls), policy_kl_max=max(kls),
                win_probability_max_abs=max(win), win_probability_mean_abs=sum(win) / len(win),
                score_mean_max_abs=max(score), score_mean_mean_abs=sum(score) / len(score),
                top1_matches=len(kls) - len(disagreement), disagreements=disagreement)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--windows", default="1,32")
    ap.add_argument("--scopes", default="ffn,transformer")
    ap.add_argument("--min-ffn-width", type=int, default=0)
    args = ap.parse_args()
    scopes = args.scopes.split(",")
    windows = [int(v) for v in args.windows.split(",")]
    if (not scopes or len(set(scopes)) != len(scopes) or any(s not in ("ffn", "transformer") for s in scopes)
            or not windows or len(set(windows)) != len(windows) or min(windows) < 1 or max(windows) > 128
            or not 1 <= args.batch <= 64 or not 0 <= args.min_ffn_width <= 8192 or args.min_ffn_width % 8):
        ap.error("invalid scope, window or batch")
    args.output.mkdir(parents=True, exist_ok=False)
    sha = file_hash(args.model)
    reference_path = args.reference or reference_io.load_bundled(sha)
    if reference_path is None:
        ap.error("no matching FP32 reference; provide --reference")
    protocol = Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
    report = dict(model=str(args.model.resolve()), model_sha256=sha, binary_sha256=file_hash(args.binary),
                  quantization="w8a8-row-out-rne-v1", min_ffn_width=args.min_ffn_width, batch=args.batch, results=[],
                  status="RUNNING", playing_strength="NOT_MEASURED", production_certified=False)
    try:
        fixture = json.loads((ROOT / "scripts/fixtures/worker_positions.json").read_text(encoding="utf-8"))
        requests = build_requests(protocol.pb, fixture, sha, 0)
        fingerprints = reference_io.fingerprints(requests, sha)
        reference = reference_io.validate_reference(reference_io.read_reference(reference_path), fingerprints, requests, protocol)
        environment = clean_environment()
        base = (f"rules=chinese\nkomi=7.5\nnnMaxBatchSize={args.batch}\n"
                "numNNServerThreadsPerModel=1\nnnCacheSizePowerOfTwo=10\nnnMutexPoolSizePowerOfTwo=8\n")
        for window in windows:
            options = SimpleNamespace(output=args.output, model=args.model, request_window=window,
                                      startup_timeout=180, task_timeout=180)
            for scope in ["fp16", *scopes]:
                label = f"{scope}-w{window}"
                cfg = args.output / f"{label}.cfg"
                cfg.write_text(base + ("nnBackend=cudabackend\n" if scope == "fp16" else
                               f"nnBackend=cudaint8backend\ncudaInt8Scope={scope}\ncudaInt8MinFfnWidth={args.min_ffn_width}\n"), encoding="utf-8")
                outputs = collect_worker(label, args.binary, cfg, options, protocol, requests, fingerprints, environment)
                original = compare_outputs(reference, outputs)
                if scope == "fp16" and original["result"] != "PASS":
                    raise AssertionError("FP16 control failed original C++ FP32 gates")
                if scope != "fp16":
                    log = (args.output / label / "worker.log").read_text(encoding="utf-8", errors="replace")
                    if "launch=cublaslt-int8" not in log or f"int8-scope={scope}" not in outputs["hello"]["backend_info"]:
                        raise AssertionError("missing actual INT8 execution/identity")
                    if args.min_ffn_width and f"int8-min-ffn-width={args.min_ffn_width};" not in outputs["hello"]["backend_info"]:
                        raise AssertionError("missing INT8 layer selection identity")
                result = dict(profile=label, original_fp32_gate=original, accuracy=quantization_metrics(reference, outputs))
                report["results"].append(result)
                write_json(args.output / "report.json", report)
                print(json.dumps(dict(profile=label, original_fp32_gate=original["result"], accuracy=result["accuracy"]), ensure_ascii=False), flush=True)
        report["status"] = "COLLECTED"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        protocol.close()
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()

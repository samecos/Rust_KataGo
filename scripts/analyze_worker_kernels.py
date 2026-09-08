"""Read-only Nsight SQLite analysis for TF3 B16 direct C++/Rust traces.

Run with Python; no CUDA or external dependencies. Kernel durations are summed
GPU activity, not CUDA API time or host wall time. Last-two-second normalization
uses observed attention calls / 33, never nominal benchmark iteration counts.
Stage attribution is verified against each full forward's launch sequence.

Only the native TF3 11 x 3 attention/FFN architecture, C++ CUDA FP16 execution
sequence, and Rust CUDA q128 + DualFFN execution sequence are supported here.
Different models/tactics/kernel fusion fail sequence assertions rather than
silently producing misleading stage attribution. Inputs are opened read-only.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

def read(path):
    path = path.resolve()
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    kernels = [dict(row) for row in connection.execute("""
        select k.start,k.end,k.streamId,s.value short_name,d.value name,
               k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
               k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory,k.localMemoryPerThread
        from CUPTI_ACTIVITY_KIND_KERNEL k
        join StringIds s on s.id=k.shortName
        join StringIds d on d.id=k.demangledName order by k.start
    """)]
    copies = [dict(row) for row in connection.execute(
        "select start,end,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start")]
    connection.close()
    return path, kernels, copies


def annotate_forward(rows, flavor):
    """Match local launch order; shared CUTLASS names alone cannot identify stages."""
    expected = 378 if flavor == "cpp" else 302
    assert len(rows) == expected, (flavor, len(rows), expected)
    attention_name = "flashAttentionMmaKernel" if flavor == "cpp" else "attention_fa2_kernel"
    attention = [i for i, row in enumerate(rows) if row["short_name"] == attention_name]
    assert len(attention) == 33, (flavor, len(attention))
    expected_launch = (6, 12, 16, 128) if flavor == "cpp" else (3, 192, 1, 256)
    for i in attention:
        row = rows[i]
        assert (row["gridX"], row["gridY"], row["gridZ"], row["blockX"]) == expected_launch, (
            "only the recorded B16 Q64 C++ / Q128 Rust launch shape is supported", flavor, row)
    # Each attention/FFN pair is preceded by two RMS calls. The C++ attention
    # has a separate RoPE kernel and FFN has a separate SwiGLU kernel; Rust
    # fuses each of those into attention and DualGemm, respectively.
    if flavor == "cpp":
        pattern = [(-3, "rms", "rmsNorm"), (-2, "attention.qkv_gemm", "nvjet"),
                   (-1, "attention.rope", "applyRoPE"), (0, "attention.core", "flashAttention"),
                   (1, "attention.out_gemm", "gemm"), (2, "rms", "rmsNorm"),
                   (3, "ffn.up_gate_gemm", "gemm"), (4, "ffn.swiglu", "swiGLU"),
                   (5, "ffn.down_gemm", "gemm")]
        first_down = attention[0] - 4
        head_start = attention[-1] + 9
    else:
        pattern = [(-2, "rms", "rms_norm"), (-1, "attention.qkv_gemm", "gemm"),
                   (0, "attention.core_with_rope", "attention_fa2"),
                   (1, "attention.out_gemm", "gemm"), (2, "rms", "rms_norm"),
                   (3, "ffn.up_gate_gemm_with_swiglu", "DualGemm"),
                   (4, "ffn.down_gemm", "gemm")]
        first_down = attention[0] - 3
        head_start = attention[-1] + 9
    tags = {}
    for i in attention:
        for offset, tag, token in pattern:
            index = i + offset
            assert token.lower() in rows[index]["name"].lower(), (flavor, index, tag, rows[index]["name"])
            assert index not in tags
            tags[index] = tag
    for i, row in enumerate(rows):
        if i in tags:
            row["stage"] = tags[i]
        elif i < first_down:
            row["stage"] = "stem"
        elif i >= head_start:
            row["stage"] = "heads"
        elif row["short_name"] == "f32_to_half_kernel":
            row["stage"] = "layout_conversion"
        elif "silu" in row["name"].lower():
            row["stage"] = "trunk_gates"
        else:
            assert "gemm" in row["name"].lower() or "nvjet" in row["name"], row
            # The two bottleneck matrices have unique selected implementations
            # after attention and FFN launches have been removed.
            down_token = "64x256" if flavor == "cpp" else "128x128"
            row["stage"] = "bottleneck.down_gemm" if down_token in row["name"] else "bottleneck.up_gemm"
    counts = Counter(row["stage"] for row in rows)
    assert counts["rms"] == 66 and counts["bottleneck.down_gemm"] == counts["bottleneck.up_gemm"] == 11, counts
    assert counts["trunk_gates"] == (22 if flavor == "cpp" else 22), counts
    return counts


def aggregate(rows, divisor):
    stages = defaultdict(lambda: dict(count=0, gpu_ns=0))
    kernels = defaultdict(lambda: dict(count=0, gpu_ns=0, shapes=Counter()))
    for row in rows:
        duration = row["end"] - row["start"]
        item = stages[row["stage"]]
        item["count"] += 1
        item["gpu_ns"] += duration
        item = kernels[(row["stage"], row["name"])]
        item["count"] += 1
        item["gpu_ns"] += duration
        item["shapes"][str((row["gridX"], row["gridY"], row["gridZ"],
                           row["blockX"], row["blockY"], row["blockZ"], row["registersPerThread"],
                           row["staticSharedMemory"], row["dynamicSharedMemory"], row["localMemoryPerThread"]))] += 1
    for item in stages.values():
        item["calls_per_forward"] = item["count"] / divisor
        item["gpu_ms_per_forward"] = item["gpu_ns"] / divisor / 1e6
    detail = []
    for (stage, name), item in kernels.items():
        detail.append(dict(stage=stage, name=name, count=item["count"],
                           gpu_ms_per_forward=item["gpu_ns"] / divisor / 1e6,
                           mean_kernel_us=item["gpu_ns"] / item["count"] / 1000,
                           launch_shapes=dict(item["shapes"])))
    return dict(stages=dict(sorted(stages.items())),
                kernel_count=len(rows), forward_equivalents=divisor,
                sum_gpu_ms_per_forward=sum(row["end"] - row["start"] for row in rows) / divisor / 1e6,
                kernels=sorted(detail, key=lambda item: item["gpu_ms_per_forward"], reverse=True))


def analyze(flavor, path):
    path, rows, copies = read(path)
    last = max(row["end"] for row in rows)
    first = last - 2_000_000_000
    marker = "sumChannelsNCHWKernel" if flavor == "cpp" else "im2col_f16_kernel"
    # C++ converts both inputs and extracts/reconverts its mask immediately
    # before sumChannels. Include those four launches in the same forward.
    offset = 4 if flavor == "cpp" else 0
    starts = [i - offset for i, row in enumerate(rows) if row["short_name"] == marker]
    complete = []
    boundaries = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(rows)
        segment = rows[start:end]
        if segment[-1]["end"] < first:
            continue
        counts = annotate_forward(segment, flavor)
        if segment[0]["start"] >= first:
            complete.extend(segment)
            boundaries.append((segment[0]["start"], segment[-1]["end"]))
    window = [row for row in rows if row["start"] >= first and row["end"] <= last]
    assert all("stage" in row for row in window)
    attention_name = "flashAttentionMmaKernel" if flavor == "cpp" else "attention_fa2_kernel"
    calls = sum(row["short_name"] == attention_name for row in window)
    normalized = aggregate(window, calls / 33)
    exact = aggregate(complete, len(boundaries))
    exact["per_forward_counts"] = dict(counts)
    exact["wall_span_ms_per_forward"] = (boundaries[-1][1] - boundaries[0][0]) / len(boundaries) / 1e6
    spans = [(a[0] - b[0]) / 1e6 for b, a in zip(boundaries, boundaries[1:])]
    exact["mean_start_to_start_ms"] = sum(spans) / len(spans)
    normalized["window_wall_ms_per_forward_equivalent"] = 2000 / normalized["forward_equivalents"]
    normalized["streams"] = dict(Counter(row["streamId"] for row in window))
    selected_copies = [row for row in copies if row["start"] >= first and row["end"] <= last]
    copy_data = {}
    for kind in sorted({row["copyKind"] for row in selected_copies}):
        selected = [row for row in selected_copies if row["copyKind"] == kind]
        copy_data[{1: "HtoD", 2: "DtoH", 8: "DtoD"}.get(kind, str(kind))] = dict(
            count=len(selected), bytes=sum(row["bytes"] for row in selected),
            gpu_ms_per_forward=sum(row["end"] - row["start"] for row in selected) / normalized["forward_equivalents"] / 1e6)
    return dict(source=str(path), window_ns=[first, last],
                all_trace_attention_calls=sum(row["short_name"] == attention_name for row in rows),
                all_trace_forward_equivalents=sum(row["short_name"] == attention_name for row in rows) / 33,
                last_two_seconds=normalized, complete_forwards_within_window=exact, memcpy=copy_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-sqlite", required=True, type=Path, help="C++ CUDA FP16 B16 Nsight SQLite export")
    parser.add_argument("--rust-sqlite", required=True, type=Path, help="Rust CUDA q128 + DualFFN B16 Nsight SQLite export")
    parser.add_argument("--output", required=True, type=Path, help="Analysis JSON path; created after successful attribution")
    args = parser.parse_args()
    report = dict(method="Last 2 seconds ending at final GPU kernel end. Only wholly contained kernels/copies. "
                        "Window forward-equivalent count = actual attention calls / 33. Independent complete-forward "
                        "subset validates stage mapping/counts. Single GPU stream, durations are GPU execution sums. "
                        "Memcpy listed separately; API/wait/startup not included. Shape/layout/precision do not by "
                        "themselves establish why a selected kernel is slower. Kernel launch_shapes tuple fields: "
                        "gridX,Y,Z, blockX,Y,Z, registers/thread, staticSharedBytes, dynamicSharedBytes, localBytes/thread.",
                  architecture="native TF3 B16, 11 nested blocks x 3 attention/FFN pairs; captured kernel sequence only",
                  cpp=analyze("cpp", args.cpp_sqlite), rust=analyze("rust", args.rust_sqlite))
    cpp = report["cpp"]["last_two_seconds"]
    rust = report["rust"]["last_two_seconds"]
    report["stage_deltas_ms_per_forward"] = {
        stage: rust["stages"].get(stage, {}).get("gpu_ms_per_forward", 0)
               - cpp["stages"].get(stage, {}).get("gpu_ms_per_forward", 0)
        for stage in sorted(set(cpp["stages"]) | set(rust["stages"]))}
    report["categories_ms_per_forward"] = {}
    for category in ("attention", "ffn", "bottleneck", "heads", "layout_conversion", "rms", "stem", "trunk_gates"):
        totals = {flavor: sum(item["gpu_ms_per_forward"] for stage, item in table["stages"].items()
                             if stage.split(".")[0] == category)
                  for flavor, table in (("cpp", cpp), ("rust", rust))}
        totals["rust_minus_cpp"] = totals["rust"] - totals["cpp"]
        report["categories_ms_per_forward"][category] = totals
    report["source_evidence_and_limits"] = [
        "native_model.rs:42-59,78-97 verifies 11 nested blocks with 3 attention/FFN pairs each, thus 33 attention calls/forward.",
        f"C++ nneval.cpp:580 loops batch sizes 1..maxBatchSize during handle warmup. Observed all-trace forward equivalents: C++ {report['cpp']['all_trace_forward_equivalents']}, Rust {report['rust']['all_trace_forward_equivalents']}. Nominal iteration counts are not used as divisors.",
        "C++ flash attention launch uses Q64/K64, 128 threads, 111 registers/thread, 21008 static shared bytes; Rust Q128/K64, 256 threads, 127 registers/thread, 40960 bytes. These observed launch-resource differences do not measure actual occupancy or its isolated performance effect.",
        "C++ cudaflashmma.cuh:68-72 uses ex2.approx.ftz.f32; Rust attention_fa2.cu:319-328 uses expf. Both use FP32 tensor-core accumulators, but exponential implementation and RoPE placement differ. Tail timings cannot isolate precision, tiling, and fusion effects.",
        "C++ cudaandrocmbackend.inc:1872,1955 invokes cublasHgemm/StridedBatched with half output/residual state; Rust cuda.rs:495 selects CUBLAS_COMPUTE_32F and cuda_exec.rs:656,807,872 retains FP32 residual state. Kernel names also show NN vs TN layouts and hhh vs hss/f32 output variants. A same-precision layout/algorithm ABBA is required before attributing all GEMM time to precision.",
        "Attention.core_with_rope must be compared with C++ attention.core plus attention.rope. FFN.up_gate_gemm_with_swiglu must be compared with C++ up_gate_gemm plus swiglu. Comparing fused kernels to only one unfused kernel overstates the gap.",
        "Copy sums are only about 0.02 ms/forward. GPU sum and launch-to-launch wall span differ by ~1 ms/forward; that residual includes idle gaps, copies, host synchronization and launch work, not a separately measured Python or worker queue cost. These traces use direct NN inference.",
    ]
    output = args.output.resolve()
    if output in (args.cpp_sqlite.resolve(), args.rust_sqlite.resolve()):
        raise ValueError("output must not overwrite an input trace")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("stage                                  cpp ms   rust ms   delta ms")
    for stage, delta in report["stage_deltas_ms_per_forward"].items():
        a = cpp["stages"].get(stage, {}).get("gpu_ms_per_forward", 0)
        b = rust["stages"].get(stage, {}).get("gpu_ms_per_forward", 0)
        print(f"{stage:38} {a:8.4f} {b:9.4f} {delta:+10.4f}")
    for flavor in ("cpp", "rust"):
        entry = report[flavor]
        tail = entry["last_two_seconds"]
        exact = entry["complete_forwards_within_window"]
        print(flavor, "attention-equivalent forwards", tail["forward_equivalents"],
              "GPU ms/forward", tail["sum_gpu_ms_per_forward"],
              "exact forwards", exact["forward_equivalents"],
              "exact GPU ms/forward", exact["sum_gpu_ms_per_forward"],
              "start-to-start ms", exact["mean_start_to_start_ms"])
    print("combined categories:", json.dumps(report["categories_ms_per_forward"], indent=2))
    print(output)


if __name__ == "__main__":
    main()

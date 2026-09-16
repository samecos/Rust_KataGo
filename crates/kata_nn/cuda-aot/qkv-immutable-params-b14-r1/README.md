# B14 immutable-parameter QKV/RoPE AOT

Windows x64, RTX 5070 Ti SM120; TF3 rows=14×361, N=1152, K=384.
The throughput plan enables `KATAGO_CUDA_QKV_IMMUTABLE_R1=1` with
`qkv_immutable_artifact`, strict Attention and TN weights explicitly bound.
Other batches keep the existing QKV path.

CUTLASS 3.9.2: M128/N64/K32, warp64×32, stage3, swizzle1;
128 threads, grid40×18, 36KiB shared, 146 registers and zero local memory.
FP16 inputs/weights, FP32 GEMM accumulation. The normal epilogue first rounds
to half; the iterator applies the original exact FP32 RoPE and rounds to half
again before its vector store. V is unchanged. No fast math.

The host fills all six addresses in a single 400-byte, 8-byte-aligned Params
image. The device parameter is const and grid-constant. Pointer offsets are
input64, weight112, source208, cos288, sin296, output304. `abi.json` and
`params-template.bin` come from the exact C++ type exporter. The Rust loader
checks artifact hashes, ABI, device resources and buffer boundaries.

`source/` preserves exact measured source bytes, including the independent
operator harness. `source-manifest.json` records provenance. From the project:

```powershell
.venv/Scripts/python.exe crates/kata_nn/cuda-aot/qkv-immutable-params-b14-r1/rebuild.py --output target/qkv-immutable-rebuild
```

The output directory must be new. This compiles with the pinned local CUDA
compiler, verifies the full disassembly and parameter image, and uses no GPU.
Different source/debug paths can change CUBIN bytes. Rebuilding does not replace
the certified binary or transfer its plan fingerprint.

Same-CLI Windows ABBA: operator +26.393169% event; complete forward +2.986361%;
uncached B14/S2/C64 Worker 1233.865301→1274.991675 RPC/s (+3.333133%).
These are separate measurements. The historical Fork/WSL ratio is unchanged.
All numerical gates, native raw-bit checks, Worker protobuf gates and six-plan
migration evidence are recorded in
`target/fork-parity-20260908/qkv-immutable-params-integration-r1/promotion-r1/acceptance.json`.

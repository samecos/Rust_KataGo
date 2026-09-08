# Residual GEMM probes

`residual_bench.cu` is the unchanged historical ONNX probe. Its FFN down
shape has K=2304; it does not measure the current TF3 FFN down K=1152.

`tf3_residual_bench.cu` is an independent opt-in diagnostic, outside Cargo and
all production tactic plans. Default shapes are B14/B16 (`M=5054/5776`),
N=384, K=1152 (FFN down) and K=384 (attention output projection).
`--suite legacy` selects the five historical shapes using the new methodology.

Build on Windows from the existing CUDA/MSVC developer environment:

```powershell
nvcc -O2 -std=c++17 --expt-relaxed-constexpr -gencode arch=compute_120,code=sm_120 -I D:/code/cutlass/include -Xcompiler "/EHsc /bigobj /std:c++17 /Zc:preprocessor /Zc:__cplusplus" scripts/b4_residual_probe/tf3_residual_bench.cu -o target/fork-parity-20260908/tf3_residual_bench.exe -lcublasLt
./target/fork-parity-20260908/tf3_residual_bench.exe --suite tf3 --warmup 10 --iterations 200 --rounds 3 --output target/fork-parity-20260908/tf3-residual-report.json
```

Use the same CUDA/CUTLASS versions as the baseline under comparison. Do not
add `--use_fast_math`. Build/run only while the GPU measurement coordinator
has reserved the device; this probe runs GPU work. Source updates are prepared
without building or running GPU work; the root agent coordinates those operations.

The three CUTLASS configurations are 128x128x32, 128x64x32 and 128x256x32,
all Sm80 TensorOp, instruction 16x8x16, three stages. Warp shapes are
64x64x32, 64x32x32 and 64x64x32 respectively. All use half A/B, FP32
accumulation and epilogue, FP32 C=D, alpha=beta=1, and a four-float epilogue
vector. Weight layout matches Rust's existing TN weights `[N,K]`, consumed
by CUTLASS as column-major `[K,N]`; no weight conversion is timed or required.

The production baseline exactly reproduces Rust's **TN** cuBLASLt descriptor
and heuristic selection: COMPUTE_32F, FP32 scale/output, transA=T/transB=N,
32 MiB workspace, request eight heuristics, retain workspace-compatible
entries in order, choose the first. Descriptors and CUTLASS Params are
persistent during timing. CUTLASS launches its cached Params directly, with
dynamic shared-memory setup outside timing. The probe therefore isolates
kernel/tile alternatives; it does not reproduce per-call production host
descriptor overhead. It does not cover the optional NN-layout C32 profile.

Every CUTLASS candidate must first match **all output elements** of the
production Lt baseline and 256 spread-out FP64 CPU dot products, including
the first/last output. Inputs are deterministic random half A/B and nonzero
FP32 residuals. Both checks use `abs(error) <= 5e-5 + 5e-5*abs(reference)` and
reject nonfinite results. Numerical failures are recorded and not timed;
any such CUTLASS failure gives exit code 2. CUDA/Lt/setup failures abort with
exit code 1. This is a standalone operation gate, not an entire-model gate.

After its numerical gate, each tile runs all requested ABBA rounds against
the production heuristic. Every sample is retained. Residual C is reset to
the same initial data outside each timed sample. The timed loop repeatedly
adds the same finite matrix product; it contains no copies or conversions.
The event metric is the entire iteration span divided by iteration count,
**not** the per-forward event median used by G0. Wall time is recorded
separately. Repeated phase spread is `max/min-1`; spread over 10% labels the
pair diagnostic. Stable local improvement still requires subsequent full
model numerical checks and Worker/G0 ABBA before any production adoption.

Top-eight Lt timings appear only in `top8_diagnostic`, after candidate ABBA.
Each timed algorithm also passes the numerical checks. Its fastest sample
never replaces the production heuristic in the paired comparisons.

The initial `target/fork-parity-20260908/tf3-residual-report.json` is preserved
as historical CUTLASS/diagnostic evidence (SHA256
`83859747256f74e2e910805ab723e6820b18e5d7183337a80bcce7b2a37b1ca4`).
The next smaller candidate is a fixed cuBLASLt algorithm, with this list
registered from that report before its new ABBA measurement:

| M | N | K | Candidate filtered index | Baseline filtered index |
|---:|---:|---:|---:|---:|
| 5054 (B14) | 384 | 1152 | 1 | 0 |
| 5776 (B16) | 384 | 1152 | 3 | 0 |
| 5776 (B16) | 384 | 384 | 2 | 0 |

After rebuilding the same standalone executable, run:

```powershell
./target/fork-parity-20260908/tf3_residual_bench.exe --suite tf3 --candidates lt --warmup 10 --iterations 200 --rounds 3 --output target/fork-parity-20260908/tf3-residual-lt-abba-report.json
```

`--candidates lt` measures only these three Lt pairs, without CUTLASS launches
or another adaptive top-eight scan. Missing or invalid pre-registered indices
abort; numeric failures are recorded without timing and return exit code 2.
The complete ABBA code and numerical gates are shared with the CUTLASS mode.
`--candidates cutlass` remains the default; `--candidates all` runs both.
Fixed Lt indices are defined only for `--suite tf3`.

Schema 2 records `lt_candidates`, the source report/hash, and current baseline
and candidate opaque algorithm bytes for diagnosis. Indices are positions
after the same workspace-only filter used by production, not permanent
algorithm IDs or portable serialized plans. Keep device, CUDA/Lt version,
layout, precision, and workspace unchanged. This experiment compares fixed
algorithms under beta=1; it does not validate production `rank=time`, whose
existing scratch ranking uses beta=0. A new shape-scoped production policy
still needs its own full-model and Worker/G0 validation. Numeric pass fields
for an unmeasured candidate family are `null`, rather than claiming a pass.

References: the shape/tile in KataGomo_fork
`cpp/neuralnet/sm120_aot/linear2_residual_cutlass.cu` (local Fork HEAD
`025f314d86be8fdefec181b824304dcad5678d44`) and the historical probe here.
The Fork template uses FP16 accumulator/epilogue/residual; changing only
the GEMM accumulator would not reproduce Rust's FP32 residual boundary.
This standalone source uses its own Rust-compatible FP32 template and no
Fork AOT binary. CUTLASS remains an external dependency under its own license.

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

The semantic-audit extension keeps the fixed candidate list, full-output
numerical gates and ABBA timing functions unchanged. Each shape now records
`heuristic_metadata` for every entry in the same workspace-filtered pool,
including the baseline. Normal runs query it after that shape's measurements.
Opaque algorithm bytes remain intact; no private word or bit is interpreted
or masked. The public API queries use the CUDA 13.3 header types:

| AlgoConfig attribute | API type | Bytes |
|---|---|---:|
| ID (0), SPLITK_NUM (2) | int32_t | 4 |
| TILE_ID (1), REDUCTION_SCHEME (3), CTA_SWIZZLING (4), CUSTOM_OPTION (5), STAGES_ID (6) | uint32_t | 4 |
| INNER_SHAPE_ID (7), CLUSTER_SHAPE_ID (8) | uint16_t | 2 |

Capabilities additionally include `NUMERICAL_IMPL_FLAGS` as uint64_t (8
bytes), and `MIN_ALIGNMENT_A/B/C/D_BYTES` as uint32_t (4 bytes). Each query
records its API status, requested/written sizes and `complete`; unsupported
or incomplete results have a null value. `algo_check_for_stream` records
`cublasLtMatmulAlgoCheckForStream` status, result state, workspace and waves
for the actual descriptors and stream. A successful check alone does not
guarantee launch validity or numerical correctness, including buffer alignment.
Metadata API failures remain visible in JSON and do not change candidate
selection or the existing numerical exit codes. Require successful, complete
metadata when using these fields as evidence.

`--metadata-only` skips all GEMM launches, numerical comparisons and timings;
the numeric pass fields are null and `abba_rounds` is zero. It still creates
CUDA resources and uploads inputs, so the GPU coordinator must schedule it.
Exit code zero in this mode means metadata collection finished, not that any
algorithm passed a numerical or performance gate.

The top-level report distinguishes OS/architecture macros, compile-time
`cublas_header_version` (including build), `cublasLtGetVersion()`, and runtime
`cublasLtGetProperty()` major/minor/patch results. GetProperty has no build
property: runtime build is null, not inferred from the header. Keep the
resolved library path, package version and file hash alongside the report.
Public attributes and an equal version number do not establish portable
opaque identity across Windows and WSL, or authorize relaxing a production
identity check.

The exact pre-extension source is preserved under
`target/fork-parity-20260908/fp16-rne-r1/lt-semantic-audit/tf3_residual_bench-before-semantic-868e1e72972478a607c2afc9b8615569cf521588d178d42304da7a2d550eac33.cu`.
Its SHA256 is `868e1e72972478a607c2afc9b8615569cf521588d178d42304da7a2d550eac33`;
an identical earlier snapshot remains under
`target/fp16-rne-r1/lt-semantic-audit/`. These snapshots preserve the source
identity of the earlier reports; enhanced runs need fresh filenames and
their own source/binary hashes.

The coordinator can build and query the same source inside WSL as follows
(the output filenames must be unused):

```bash
cd /mnt/d/code/Rust_KataGo
audit=/mnt/d/code/Rust_KataGo/target/fork-parity-20260908/fp16-rne-r1/lt-semantic-audit
/usr/local/cuda-13.3/bin/nvcc -O2 -std=c++17 --expt-relaxed-constexpr -gencode arch=compute_120,code=sm_120 -I /mnt/d/code/cutlass/include scripts/b4_residual_probe/tf3_residual_bench.cu -o "$audit/tf3_residual_bench-wsl" -lcublasLt
LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:/usr/lib/wsl/lib ldd "$audit/tf3_residual_bench-wsl" > "$audit/ldd-wsl-r1.txt"
readlink -f /usr/local/cuda-13.3/lib64/libcublasLt.so > "$audit/library-path-wsl-r1.txt"
dpkg-query -W libcublas-13-3 > "$audit/library-package-wsl-r1.txt"
sha256sum scripts/b4_residual_probe/tf3_residual_bench.cu "$audit/tf3_residual_bench-wsl" /usr/local/cuda-13.3/lib64/libcublasLt.so > "$audit/sha256-wsl-r1.txt"
LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:/usr/lib/wsl/lib "$audit/tf3_residual_bench-wsl" --suite tf3 --candidates lt --metadata-only --output "$audit/metadata-wsl-r1.json"
```

Before interpreting the report, check that `ldd` resolves cuBLASLt to the
hashed library. After metadata review, the separate full numerical and ABBA
run is:

```bash
LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:/usr/lib/wsl/lib "$audit/tf3_residual_bench-wsl" --suite tf3 --candidates lt --warmup 10 --iterations 200 --rounds 3 --output "$audit/lt-abba-wsl-r1.json"
```

On Windows, reuse the earlier nvcc command with an unused executable path in
the same audit directory, then use the same `--metadata-only` arguments and
a distinct `metadata-win-r1.json`. Record the actual loaded DLL and its hash
separately. Windows and WSL builds and metadata queries have now completed.
Evidence is under
`target/fork-parity-20260908/fp16-rne-r1/lt-semantic-audit/`: `build-win.log`,
`build-wsl.log`, `metadata-win-r1.json`, `metadata-win-r2.json`, and
`metadata-wsl-r2.json` through `metadata-wsl-r4.json`. Library provenance is
recorded in `library-win-r2.json`, `ldd-wsl-r2.txt`, `library-path-wsl-r2.txt`,
`library-package-wsl-r2.txt`, and `sha256-wsl-r2.txt`. The preserved
`source-semantic-extension-r1.json` describes the earlier source-review stage.
The original numerical gates and ABBA were not rerun for this metadata extension;
these metadata-only reports retain null numerical results and zero ABBA rounds.

All three fixed candidates report the same nine public config attributes and
five capabilities on both platforms, with complete queries and successful
CheckForStream results. The two Windows queries match the registered opaque
identities. In the three WSL processes, both B16 candidates instead have
opaque `data[4]` values `0x0001fe2896ab0001`, `0x0001fe2883530001`, and
`0x0001fe2881080001`; their public properties remain equal. The query buffers
were zero-initialized before the library calls. The library-internal cause
is unknown: equal public attributes do not prove this difference harmless
and do not authorize masking or otherwise bypassing the full identity guard.
WSL B16 candidate inference remains unrun despite completed metadata queries;
the preset remains uncertified on WSL.

## Independent public-initialization candidate (2026-09-08)

`tf3_initialized_lt_bench.cu` is a separate executable. It includes the shared
probe with `KATAGO_RESIDUAL_PROBE_LIBRARY` to reuse its unchanged inputs,
full-output/FP64 gates, and ABBA timing; hash **both** source files for a build.
Build with the same commands and compiler flags above, replacing the source
and executable names. Use `--suite tf3 --candidates lt` for all invocations.
`--metadata-only` keeps all numerical claims null; a normal run checks the
candidate before timing it. Existing report paths are refused.

This candidate starts from a zeroed object and the public `AlgoInit` API,
ID 21 and `COMPUTE_32F`, FP32 scale/C/D, FP16 A/B. It explicitly sets and
reads back tile 15, split K 1, reduction/swizzle/custom 0, stages 12, and
inner/cluster shape 0. Those last values mean undefined/automatic, so the
experiment does not claim to eliminate all internal automatic selection.
It requires complete numerical capability 66050 (HMMA, FP32 accumulator,
FP16 input), all four alignment capabilities of 16 bytes, actual pointer
and stride alignment, 256-byte workspace alignment, and a successful check
on the actual stream. It runs the original initialized object; the check
result's algorithm field is not initialized by the API and is never used.

Evidence is under
`target/fork-parity-20260908/fp16-rne-r1/lt-initialized-r1/standalone-summary.json`.
Both platform builds, repeated metadata processes, and three-round ABBA
completed. All three shapes have full output bits equal to the heuristic
baseline and pass the independent 256-point FP64 gate. The final WSL run
measured +11.24% / +6.12% / +16.52%, and Windows +11.21% / +5.69% / +16.61%
for B14 FFN down / B16 FFN down / B16 output projection, respectively.
The earlier complete WSL run is retained separately. All spreads pass the
existing 10% standalone gate. These are operation-level results; they do
not certify an entire model, Worker profile, or a new production plan.

Initialized objects kept identical complete bytes across these runs, before
and after read-only checks and launches. The heuristic objects were neither
altered nor masked. The candidate has its own identity and validation; it
does not inherit the old preset's certification or establish that arbitrary
opaque differences are harmless.

The subsequent full-model experiment rejected this candidate (2026-09-09).
Windows and WSL fixed B14/B16 with two lanes passed all native raw gates,
including exact output-bit agreement with each same-batch baseline. WSL
forward ABBA then measured B14 **1127.95 to 1127.14** event rows/s (-0.072%)
and B16 **1130.96 to 1106.29** (-2.182%), both within the registered spread
gate. The production opt-in implementation was rolled back exactly to the
three pre-experiment source hashes; staged
plans are not certified or installed. The frozen experimental binaries,
source and reports remain in `lt-initialized-r1/` for reproduction.
`benchmark_rust_forward_abba.py` retains the explicit initialized-candidate
choice for those frozen binaries, not as a supported production preset.

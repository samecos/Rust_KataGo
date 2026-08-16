# Fork SM120 CUDA Optimization — Technical Reference Notes

Source repo: `D:/code/KataGomo_fork` (KataGo fork with a dedicated SM120 backend).
Purpose: extract the exact kernel/tactic/scheduling decisions for Rust_KataGo's M4
CUDA optimization phase (decision groups G1-G10 in `docs/cuda-optimization-plan.md`).
All identifiers and shapes below are verbatim from the fork. Line numbers refer to
the fork files as of 2026-08-13.

Hardware note: the certified production plan was scanned on an RTX 5080 (SM120,
84 SMs, 64 MB L2) with physical batch B16 and 2 streams. Our target RTX 5070 Ti is
the same SM120 arch (70 SMs per cuda-fingerprint, 48 MB L2) — kernels transfer, only the persisting-L2
budget shrinks.

## Certified production plan (B16, S2) — one-glance summary

File: `final-migration/plans/sm120/rtx5080-b16-s2/best-tactic-plan.json`
(plan_id `sm120-rtx5080-96c8d332dc452f3d`, `batches:[16]`, `streams:2`,
`precision:"FP16/NHWC"`, `model_sha256` bound to the b11c768h12nbt3tflrs-fson-silu model).

| Family | Tactic (key=value) |
|---|---|
| FA4 attention | `cudaFlashAttentionAotTacticSm120=fa4-b16-s361-h12-d32-tm128-tn64-s1-both16`, `cudaFlashAttentionSm120Accum=both16` |
| Wide QKV | `cudaUseWideQKV=true`, `cudaUseQKVGemmAot=true`, `cudaWideQKVAotTacticSm120=wide_qkv-m128-n128-k64-s2-cute-atom4x2-packed` |
| Q/K RoPE | `cudaUseFusedQKRoPE=true`, `cudaUseBatchSharedRoPE=true`, `cudaUseBatchSharedRoPEUnrolledSm120=true` (separate kernel, NOT GEMM epilogue) |
| Fused FFN | `cudaUseFusedFFN=true`, `cudaFusedFFNAotTacticSm120=dual_ffn-m128-n64-k32-s2-mb3-tanh-half2` (SwiGLU in epilogue) |
| Residual GEMMs | `cudaUseFusedResidualGemmSm120=true`, `cudaUseLinear2ResidualAot=true`, `cudaUseOutProjectionResidualAot=true` (both `*-m128-n128-k32-s3-cutlass`) |
| Outer projection | `cudaOuterProjectionDownTacticSm120=warp64x32`, `cudaOuterProjectionUpTacticSm120=warp64x32` |
| RMSNorm | `cudaRMSNormTacticSm120=warp4-vec8` |
| Affine+SiLU | `cudaAffineSiluTacticSm120=half2` |
| Head | `cudaWideHeadProjectionTacticSm120=full-c384`, `cudaUseFusedPolicyP1=true`, `cudaUseHeadBNHalfToFloat=true` |
| L2 | `cudaUsePersistingL2Trunk=true`, `cudaUsePersistingL2Inner=true`, `cudaPersistingL2StreamsSm120=2`, `cudaPersistingL2HitRatioSm120=1` |
| Initial conv | `cudaInitialConvFrontendPlanSm120=eng45-tile0-stages2` (cuDNN frontend, NOT im2col) |
| Disabled in plan | `cudaUseSwiGLU1152Sm120=false`, `cudaUsePostConvBNSiluSm120=false`, `cudaUseFusedValueTerminalSm120=false`, `cudaUseWideFFNSingleGemm=false`, `cudaUseQKVStridedSm120=false`, `cudaQKVRopeAotTacticSm120=disabled` |

---

## 1. G1 — Wide QKV single GEMM + fused Q/K RoPE

### Wide QKV (`cudabackend_sm120.cpp:1412-1578`, weight build `1473-1497`)

- Weights: one buffer `[384, 1152]` half, row-major `[inC, outC]`. Built once per
  model by three `cudaMemcpy2DAsync` chunks of `[384,384]`: Q at cols `0..384`,
  K at `384..768`, V at `768..1152`.
- GEMM: `M = batchSize*361` (tokens), `N = 1152`, `K = 384`.
- Selected AOT: `wide_qkv-m128-n128-k64-s2-cute-atom4x2-packed` → tile M128 × N128 ×
  K64, 2 stages, CUTE atom 4x2, **packed output**: row stride 1152 halves per token,
  `[Q384 | K384 | V384]` (`cudabackend_sm120_kernels.h:161-169`).
- Fallback (unused in plan): `cublasHgemmStridedBatched` with 3 batches, planar
  output `[Q|K|V]` each `[384, tokens]` (`sm120.cpp:1565-1570`).
- Dispatch gate: `useWideQKV && useQKVGemmAot`, exact-batch AOT table
  `wideQKVAotByBatch` (`sm120.cpp:861-865`); packed path additionally requires
  FA4 enabled (`packedAttentionReady`, `sm120.cpp:1451-1458`).

### Q/K RoPE — separate kernel, NOT fused into the GEMM epilogue

SM120 fork decision: the plan uses a standalone batch-shared RoPE kernel after the
wide QKV GEMM. (The SM89 fork *does* fuse RoPE into the QKV GEMM output iterator —
`cpp/neuralnet/FLASH_ATTENTION_SM89.md:23-29`; SM120 moved it out.)

- Selected: `batchSharedPackedFusedQKRoPEUnrolledHalf2Kernel<Batch>`
  (`cudabackend_sm120_kernels.cu:536-568`): grid `361×1`, 192 threads
  (12 heads × 16 pairs); each thread rotates one `half2` (cos/sin recomputed via
  `__sincosf` from the 1.5 KiB frequency vector — no table); loop `for n in 0..Batch`
  with `#pragma unroll` so ptxas software-pipelines the global loads; dispatch is a
  compile-time switch on B=1..32 (`581-627`).
- Rotation: `q0*cos - q1*sin`, `q0*sin + q1*cos` on the packed buffer; `kBuf`
  points at offset `384` inside the packed row (kernels.h comment, `qBuf` at 0).
- Alternative variant with precomputed table: `precomputeQKVRopeTable19Half`
  (`391-417`): `table[xy*192 + hp] = half2(cos, sin)`, grid `361×192`, table
  361×192×2 halves ≈ 271 KB, cached per `ropeFreqs` pointer. Only used by the
  `qkvRopeAot` route, which is **disabled** in the certified plan.
- Gates (`sm120.cpp:1736-1743`): `numHeads==12, numKVHeads==12, qHeadDim==32,
  numPairs==16, seqLen==361, ropeXLen==19`.

Rust port notes:
- Our hand-written `hgemm_m16n8k16` already matches the CUTLASS instruction shape
  (16×8×16); the wide GEMM only needs N=1152 (or keep N=384 × 3 writes into a
  strided packed row) — the packed output layout (`[Q|K|V]` per token) is what
  unlocks FA4.
- Port RoPE as the fork's B-unrolled batch-shared kernel (a `const B` generic, 361
  blocks × 192 threads, half2 loads); do not fuse it into the GEMM epilogue first —
  SM120 found the separate kernel wins.
- K=384 is already a clean k-tile multiple (no padding needed for the QKV GEMM).

## 2. G1 — FA4 attention (FlashAttention-4 AOT)

Generator: `cpp/neuralnet/fa4_aot/build_aot.py` (flash-attn 4.0.0b25 +
`flash_attn.cute.flash_fwd_sm120.FlashAttentionForwardSm120`, CUTE DSL AOT compile
→ C header → bridge `fa4_aot/fa4_cuda_bridge.cpp` with `_cuda*` runtime stubs).

- Fixed shape: `B(runtime) × S=361 × H=12 × D=32`, FP16, `is_causal=False`,
  `is_local=False`, `mask_mod=None`, `pack_gqa=False`, `Q_in_regs=False`,
  `num_threads=128` (`build_aot.py:143-197`).
- Tile (plan): `tm128 × tn64`, `num_stages=1`.
- **both16 mechanism** (`build_aot.py:150-180`): `qk_acc_dtype=Float16` AND
  `pv_acc_dtype=Float16` (env `FA4_QK_ACC`/`FA4_PV_ACC`, default `fp16`). The QK
  MMA and PV MMA accumulators are both FP16 (not FP32). Because flash-attn 4.0.0b25's
  online-softmax `Softmax.rescale_O` stores an FP32 product without converting back,
  the fork patches `rescale_O` to `acc_o_mn[row].store((loaded * row_scale).to(acc element_type))`
  — i.e. every rescale **rounds back into the FP16 PV accumulator**. This is the
  exact semantics to replicate: two FP16 halves per element, rescale keeps FP16.
- Launch contract (`sm120.cpp:1086-1110`): `launch(qBuf,kBuf,vBuf,attnOutBuf,
  batchSize,seqLen,numHeads,qHeadDim, scale=1/sqrtf(32), packedQKV, stream)`;
  packed stride `rowStride = (packedQKV ? 3 : 1)*heads*dim` (`build_aot.py:246-250`).
- Gate (`sm120.cpp:1075-1084`): FP16, `maskBuf==NULL` (noncausal full-board),
  `numHeads==numKVHeads` (MHA), `qHeadDim==vHeadDim==32`, `seqLen==361`;
  exact-batch tactic lookup, missing entry or launch failure with explicit
  `fa4AotTactic` → throw (fail-closed, no silent fallback).
- Softmax normalization is handled inside the AOT kernel (online softmax, no mask);
  `scale` is the only host-side parameter.

Rust port notes:
- FA4 is a full online-softmax flash-attention: 128 threads, tile M128×N64, single
  stage (no K-loop pipelining — S=361 is short). Our existing
  `attention_row_kernel` can be reworked toward this tile; the critical accuracy
  detail is the **both16 rescale that must round back to FP16** (test against the
  FP32-accumulator variant; the fork certifies FP16 accumulators are accurate
  enough).
- Batch is runtime but the kernel is compiled per exact batch (fat registry).

## 3. G2 — GEMM beta=1 in-place residual (C==D)

Two mechanisms, both used by the plan:

- **CUTLASS residual GEMMs** (`sm120_aot/linear2_residual_cutlass.cu:88-95`,
  `outproj_residual_cutlass.cu`, registered in
  `cudabackend_sm120_aot_registry.cu:34-68` for every B4-B32):
  `Gemm::Arguments(GemmCoord(rows, 384, K), {input,K}, {weights,384},
  {output,384}, {output,384}, {alpha=1, beta=1})` — **C and D are the same
  pointer** (the trunk/mid buffer). Tile `GemmShape<128,128,32>`, warp
  `<64,64,32>`, instr `<16,8,16>`, 3 stages, swizzle 1. Shapes: linear2
  `[M,384,1152]` (K=1152), outproj `[M,384,384]`. CUTLASS `Params` cached
  thread-locally per `(weights, rows)` (`linear2_residual_cutlass.cu:54-58`).
- **cuBLAS fallback** (`sm120.cpp:1656-1663`): `cublasHgemm(OP_N,OP_N, 384, tokens,
  inC, alpha=1, W, 384, input, inC, beta=1, trunkBuf, 384)` — `C==D==trunkBuf`,
  in-place add. Constraints (`1580-1601`): `maskBuf==NULL`, `outC==384`,
  `inC∈{384,1152}`, `matBatchSize % 361 == 0`.
- The outer-projection-up kernel also uses beta=1 in-place (`outer_projection.cu:
  160-175, 440`).
- In-place safety: valid because the epilogue reads the C tile into registers and
  then writes D for the same tile — no cross-tile aliasing.

Rust port notes:
- Our hgemm epilogue must read C (residual) before writing D; both FP16 halves stay
  in the same accumulator layout — zero extra cost vs a separate add kernel
  (~1 full 4.4-8.9 MB buffer pass saved per block).
- `rows` (tokens) is the only runtime dimension; cache compiled kernel params per
  `(weights_ptr, tokens)` like the fork's thread-local handle map.

## 4. G3 — RMSNorm / affine+SiLU / SwiGLU cross-boundary fusion

Dispatch: `sm120.cpp:1673-1719` (RMSNorm), `1823-1871` (affineSilu), `1795-1821`
(SwiGLU). All require `maskBuf==NULL`, FP16, 19×19.

### RMSNorm C384 (`cudabackend_sm120_kernels.cu`)

| Tactic | Kernel | Layout | Notes |
|---|---|---|---|
| `warp4-vec8` (plan) | `rmsNorm384Vec8Kernel` `177-255` | 128 thr/block, 4 rows/block (1 warp/row); lane loads `uint4` (8 halves) + `uint2` (4 halves) = 12 halves; 32 lanes × 12 = 384 | **no shared memory at all**: single `__shfl_xor_sync` chain (offsets 16..1) over the per-lane 12-square sum; writes back `uint4`+`uint2` |
| `one-warp-exact` | `rmsNorm384Half2Kernel` `48-100` | 128 thr, 4 rows/block; 6×32-pair groups | 2-level XOR reduction (6 group sums, then one more chain) |
| `ordered-ept3` | `rmsNorm384OrderedEpt3Kernel` `116-162` | 1 row/block, 128 thr = 4 warps × 32 lanes × 3 channels | shared `float warpSums[4]`, two-stage `shfl_down` |
| `two-warp` | `rmsNorm384TwoWarpHalf2Kernel` `271-339` | 64 thr, 1 row/block | shared `groupSums[6]` + `scales[32]` |

- All: `scale = rsqrtf(sumSquares/384 + epsilon)`, output `x*scale*gamma + beta`
  computed in FP32, stored back to half. Epsilon passed per-call.

### Affine+SiLU (`786-921`)

- `half2` (plan): template `<384>/<768>`, one row per block, threads = channels/2;
  `__hfma2(input, scale, bias)` then per-half silu in float.
- `half2x3`: 512 threads, 3 pairs per thread, flat indexing.
- `flat-vec8-c768`: `uint4` = 8 halves; `channelVector = idx % 96`; C768 only.

### SwiGLU

- `swiGLU1152Half8Kernel` (`742-769`): `uint4` per thread (8 halves), `a` = first
  1152 of the wide row, `b` = second 1152; silu(a)*b.
- In the certified plan this kernel is **off** — SwiGLU is folded into the fused
  FFN epilogue (`dual_ffn` AOT, `LeftSiLUAndMul` in CUTLASS
  `sm120_aot/dual_ffn_shared_a.cu:27-41`, or TileLang `tanh_half2` variant:
  sigmoid-via-tanh, half2).

Rust port notes:
- `warp4-vec8` is pure registers+shuffle — no smem allocation, no barriers; one
  block per 4 rows. Our elementwise kernels should adopt the `uint4`+`uint2` load
  pattern for C384.
- half2 `__hfma2` semantics: affine is fused, silu is not (no half2 sigmoid in
  hardware) — compute sigmoid in float per half.

## 5. G4 — wide head projection / policy P1 / head BN

- **Wide head** (`sm120.cpp:798-846`, launch `1948-1989`): merges the three head
  convs (all 1×1, 768→{96,96,192}) into one CUTLASS GEMM. Weights `[768, outC]`
  built by transposing each conv's weights into offsets: full-c384 → P1@0, G1@96,
  V1@192; partial-c288 → G1@0, V1@96. `katago_create_head_projection_sm120` reuses
  the outer-projection CUTLASS handle (tile 128×128×32, warp64x32, 3 stages,
  swizzle 1, beta=0, `outer_projection.cu:405-415`). Output row stride = 384 (or
  288); the official forward reads sub-slices via `wideHeadP1Offset` etc.
  (`cudabackend.cpp:3927-3945`).
- **Fused policy P1** (`kernels.cu:923-964`): `block(96,5)`, `grid(73,B)`; half
  input → **float output**; `value + globalBias[batch*96+ch]`, `*scale + bias`
  (BN-fold), silu; called from `cudabackend.cpp:3406-3423` with
  `p1BN.mergedScaleBuf/mergedBiasBuf` and the g-pool bias as `globalBias`.
- **Head BN half→float** (`kernels.cu:966-1025`): `__hfma(input, scale, bias)` +
  silu; variant `<96,5,false>` (G1 head: float-only output, feeds g-pool directly,
  `cudabackend.cpp:3355-3376`) and `<192,2,true>` (V1 head: **writes both half and
  float** outputs — half for the ownership conv/v2 matmul, float for the value
  g-pool). Both use BN merged scale/bias.

Rust port notes:
- The head GEMM is just another `[M,384(288),768]` hgemm with beta=0 + column
  offsets; the win is replacing three small convs + three BN kernels with one GEMM
  + fused elementwise tails.
- "FP32 直出" is cheap: P1/G1/V1 epilogues write float from the half accumulators
  directly, skipping the official half→float copy kernel.

## 6. G6 — persisting-L2 windows

- Window math (`sm120.cpp:883-927`):
  `trunkWindow = maxBatchSize*361*768*sizeof(half)` = 8,871,936 B (≈8.5 MiB);
  `innerWindow = maxBatchSize*361*384*sizeof(half)` ≈ 4.2 MiB;
  `total = persistingL2Streams * (trunk+inner)`; requested =
  `min(cudaDevAttrMaxPersistingL2CacheSize, total)`; then
  `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, requested)`; effective
  `hitRatio = min(configHitRatio, actualGranted/total)`.
- Per-stream attribute (`sm120.cpp:28-50`):
  `cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
  {base_ptr, num_bytes, hitRatio, hitProp=cudaAccessPropertyPersisting,
  missProp=cudaAccessPropertyStreaming})`; clear sets both props to Normal.
- Placement (`cudabackend.cpp:3118-3124, 3205-3211`): trunk window is set on
  `trunkScratch.buf` (size `getBufSizeXY(768)`) before the initial conv and cleared
  after the trunk block loop (before the heads); the inner (C384 mid) window is
  applied by a second hook on the mid buffer.
- B16/S2 on RTX 5080: 2×13.3 MiB = 26.6 MiB ≪ 48 MiB persisting budget → hitRatio
  1.0. On RTX 5070 Ti (48 MB L2, 36 MB max persisting) the same windows still fit.

Rust port notes:
- Pure host-side API usage — port as-is; keep the `hitProp/missProp` pair and the
  limit clamp; verify `cudaDevAttrMaxAccessPolicyWindowSize` per device.

## 7. G8 — initial 3×3 conv

**The fork does NOT use im2col+GEMM (no K=198→208 anywhere in this repo).** The
SM120 initial conv is a **cuDNN frontend graph engine** (`cudabackend.cpp:868-955`,
execution `1040-1072`):

- Conditions: SM120, FP16/NHWC, `inChannels==22`, `outChannels==768`, 3×3, stride 1,
  pad 1, dilation 1, `cudnnGetVersion()>=92400`, exact 19×19.
- One graph per exact batch: `{B,22,19,19}` NHWC strides; `conv_fprop` with
  `set_padding({1,1})`; intermediate/compute data type FLOAT.
- `eng45-tile0-stages2` (plan): `create_execution_plan(45, {{TILE_SIZE,0},{STAGES,2}})`.
- `eng47-k2-2-k6-1-k13-1-k14-0-k22-2`: heuristic plans (`HeurMode_t::A`), selects
  the plan whose tag is `eng47_k2=2_k6=1_k13=1_k14=0_k22=2`.
- Workspace/plan index cached per batch; `execute_plan_at_index` on the apply path.

Rust port notes:
- If we stay pure-CUDA (no cuDNN), the equivalent is the classic 3×3 im2col GEMM:
  K = 22×9 = 198 per output pixel → pad K to 208 (= 13×16) for clean tensor-core
  k-tiles; M = tokens, N = 768. That is exactly the "K=198→pad 208" idea from the
  plan doc — the fork simply chose cuDNN over implementing it.

## 8. G10 — value terminal split

- `splitValueTerminalKernel` (`kernels.cu:1027-1046`): 1 block per batch,
  `combinedChannels = valueChannels + scoreValueChannels` threads; reads the
  combined float row, adds per-channel bias, writes `value[]` and `scoreValue[]`
  slices. Guards: `combinedChannels<=1024`.
- **Disabled in the certified plan** (`cudaUseFusedValueTerminalSm120=false`) — the
  official path does the split. Trivial to port either way.

## 9. Scheduling layer (`cpp/neuralnet/nneval.cpp` + `cudabackend.cpp`)

### Batch-aware dispatch (`nnBatchAwareDispatch`, `nneval.cpp:125-203`)

- `NNBatchingDispatcher::waitForBatch`: fixed physical batch
  `desiredBatchSize = min(maxBatchSize, currentBatchSize)`; a batch is popped only
  when the queue has ≥ desired rows **or the GPU is idle**
  (`serverThreadHasActiveBatch` per GPU, `125-203`); `completeBatch` clears the
  flag. This maximizes padding efficiency: wait for a full B16 unless nothing else
  is in flight.
- Tail padding (`nneval.cpp:720-731`): `inferenceBatchSize = batchAwareDispatch ?
  maxBatchSize : requestBatchSize`; the request vector is resized by duplicating
  the **last request pointer**; padded rows' outputs are discarded (only
  `requestBatchSize` results finalized, `753-779`).

### Event-gated async pipeline (`cudaAsyncInferPipeline`, `nneval.cpp:583-959` +
`cudabackend.cpp:4438-4471, 5024-5186`)

- Per slot (one per NN server thread): `computeStream` (external) +
  `uploadStream` + `downloadStream` (both `cudaStreamNonBlocking`) + 5 timing-free
  events: `inputReadyEvent, inputConsumedEvent, applyCompleteEvent,
  outputConsumedEvent, outputReadyEvent`.
- Handshake per inference: upload waits `inputConsumedEvent` → 3× `cudaMemcpyAsync`
  H2D (input/global/meta) → record `inputReadyEvent`; compute waits
  `inputReadyEvent` → full `apply()` → record `applyCompleteEvent`; download waits
  `applyCompleteEvent` → 5× D2H (policyPass/policy/value/scoreValue/ownership) →
  record `outputConsumedEvent` + `outputReadyEvent`
  (`cudabackend.cpp:5047-5180`).
- GPU buffers are **single-slot**: `ComputeHandle::apply` waits
  `outputConsumedEvent` before starting and records `inputConsumedEvent` after the
  whole model finishes (`cudabackend.cpp:4496-4507`) — no double-buffering, pure
  event-gated reuse.
- Host staging: everything pinned (`cudaHostAlloc` Portable, `4823-4870`) and
  inputs pre-converted float→half on the host into a second pinned half buffer
  (`enablePinnedHalfInputs`, `4872-4897`; `prepareEventPipelineInput`,
  `5024-5045`); the host input slot is reusable only when `inputReadyEvent` fired
  (`eventPipelineInputHostReusable`, `4967-4971`).
- Host-side structure (`nneval.cpp:34-71, 583-959`): each slot holds up to 3 batch
  states (`front/next/submitting`); a per-slot **SubmitWorker thread** makes
  inference submission nonblocking (launch returns before compute finishes);
  `slotCanAccept` requires the host input reusable; output finalized via
  `eventPipelineOutputReady` polling; optional whole-graph capture
  (`cudaEventPipelineUseGraph`, `5107-5134`).
- Dual-stream topology: 2 NN server threads, 2 slots, both on GPU 0
  (plan `topology`: `cudaDeviceToUseThread0=0, cudaDeviceToUseThread1=0,
  numNNServerThreadsPerModel=2`); plan validation forces
  `evaluatorThreads % streams == 0` (`cudatacticplan.cpp:235-241`).
- Warmup: `cudaWarmupOnlyMaxBatchSize=true` — warmup runs only at the max (physical)
  batch.

Rust port notes:
- Port the event handshake exactly (5 events, single-slot GPU buffers, pinned
  staging); it is simpler than multi-buffer round-robin and is what the certified
  config relies on.
- Tail padding by duplicating the last input row is what makes "fixed physical
  batch" safe for uneven request streams.

## 10. Plan JSON mechanism (`cudatacticplan.cpp/h`)

- Plan file: `final-migration/plans/sm120/rtx5080-b16-s2/best-tactic-plan.json`
  (~110 KB): root fields `schema:1, kind:"cuda-tactic-plan", plan_id, plan_sha256,
  status:"complete_long_stable", production_ready:true, ready_for_scan_bypass:true,
  positive_history_closure:{complete:true}, target:{architecture:"sm120",
  gpu_class:"rtx5080", streams:2, precision:"FP16/NHWC", fixed_board:[19,19],
  model_sha256:"1881600c...", cuda_device_capabilities_at_scan:[...]},
  batches:[16], apply:{topology:{...}, per_batch_tactic_overrides:{"16":
  "<comma-separated cuda* key=value list>"}}, final_joint:{...},
  families:{...} (per-family candidate + activation markers + binary_sha256),
  selection:{method, metric, maximum_relative_spread:0.1, minimum_iterations:1000,
  minimum_samples:2}, coverage/missing/identity_missing, generated_utc,
  reproducibility, source_results`.
- Fail-closed validation (`cudatacticplan.cpp:178-293`): schema/kind/status;
  `final_joint[B]` must be `long_stable` with `correctness.status=="passed"`;
  overrides must all start with `cuda` and must not contain device ordinals;
  config must satisfy `nnMaxBatchSize==batch`, `requireMaxBoardSize=true`,
  `nnBatchAwareDispatch=true`, `cudaWarmupOnlyMaxBatchSize=true`, FP16/NHWC not
  false, `requireExactNNLen=true`; the plan then **overrides** config keys and
  forces `cudaSm120Backend=true`. File SHA-256 + plan/model SHA-256 recorded.
- `validateDevices` (`295-331`): runtime GPU name, compute capability, SM count,
  regs/SM, shared-mem, L2 size, bus width, async engines, concurrent kernels,
  global mem must all match the scanned device exactly; per-device stream counts
  must equal `streamsPerDevice`.
- Runtime enforcement (`cudabackend_sm120.cpp`): every explicit tactic is looked up
  by exact `(batchSize, id)`; a missing entry, unmet packed-path precondition, or
  AOT launch failure **throws** (e.g. `sm120.cpp:1064-1070, 1086-1110, 1466-1472,
  1622-1627, 1649-1654`); AOT registry `cudabackend_sm120_aot_registry.cu` resolves
  fat (generated, per-batch, explicit-ID only) then single-slot search symbols.

Rust port notes:
- Adopt the same strict contract: tactic tables keyed by exact batch + ID, runtime
  throw on mismatch — never silent fallback — plus device fingerprint validation.

## 11. B16 shape conventions (NHWC, FP16)

| Tensor | Shape (B=16) | Layout/notes |
|---|---|---|
| Spatial input | `16×361×22` half | NHWC (22 = v7 spatial feature set, `nninputs.h:97`; 19 binary-ish planes + 3 aux) |
| Global input | `16×19` half | v7: 19 global features |
| Meta input | `16×2` half | v15 meta channels (verify against model JSON) |
| Packed QKV | `16×361×1152` half | row = `[Q384\|K384\|V384]`, token stride 1152 |
| Trunk buffer | `16×361×768` half | ≈ 8.9 MB; persisting-L2 trunk window |
| Mid buffer | `16×361×384` half | ≈ 4.4 MB; persisting-L2 inner window |
| FFN wide scratch | `16×361×2304` half | only for the disabled single-GEMM fallback |
| Head projection | `16×361×384` half | P1@0..96, G1@96..192, V1@192..384 (full-c384) |
| Policy P1 out | `16×361×96` float | FP32, direct from fused kernel |
| Policy pass/2D | `16×2` / `16×361×2` float | v15 → policyOutChannels = 2 (`desc.cpp:2067-2071`: v12-15 → 2, v16 → 4, v17 → model-declared 2/4) |
| Value / score / ownership | `16×2` / `16×1` / `16×361×2` float | v15 (verify counts from model JSON) |

GEMM token dimension M = `B*361` = 5776 for every trunk/head GEMM at B16; all
shapes stay multiples of the M128 tile.

---

## Appendix — file map

- `cpp/neuralnet/cudabackend_sm120.{h,cpp}` — Sm120Model, option parsing, AOT dispatch, L2 windows, weight interleaving (2039 lines)
- `cpp/neuralnet/cudabackend_sm120_kernels.{h,cu}` — hand-written elementwise/RoPE/RMSNorm/policy kernels (1067 lines)
- `cpp/neuralnet/cudabackend_sm120_aot_registry.cu` — tactic tables + fat lookup
- `cpp/neuralnet/cudatacticplan.{h,cpp}` — plan load/validation (fail-closed)
- `cpp/neuralnet/sm120_aot/*.cu` — CUTLASS wrappers (outer projection, linear2/outproj residual, dual FFN, postConv)
- `cpp/neuralnet/fa4_aot/` — FA4 generator (`build_aot.py`) + runtime bridge
- `cpp/neuralnet/nneval.cpp` — batch dispatcher + event pipeline scheduler
- `cpp/neuralnet/cudabackend.cpp` — official forward + hooks, cuDNN initial conv, 3-stream pipeline, pinned buffers
- `cpp/neuralnet/FLASH_ATTENTION_SM89.md`, `docs/cuda-sm120-rebuild.md`, `docs/sm120-tilelang-fat-scan.md` — reference docs
- `final-migration/plans/sm120/rtx5080-b16-s2/best-tactic-plan.json` — certified production plan (B16, S2)

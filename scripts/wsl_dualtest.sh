#!/bin/bash
# 方案 A(双流拓扑)复测——DUALFFN 修复后 kernel 提速 ~25%,按 parity plan §4
# 的"证伪结论需重审"条款重测;fork 可比配置 = serve=2 + NOGRAPH(fork 不用
# CUDA Graph)。WSL(Ubuntu-24.04)下运行,配对方差 ±0.05%。
BIN=/root/rk-target/release/katago-rs
MODEL=/mnt/d/code/b11fix.onnx
PLAN=/mnt/d/code/Rust_KataGo/plans/best-tactic-plan.json
OV="nnBackend=cudabackend,cudaTacticPlan=$PLAN"

r() { # $1=serve $2=workers
  "$BIN" nnbench --model "$MODEL" --override-config "$OV,numNNServerThreadsPerModel=$1" \
    --mode eval --batch 16 --workers "$2" --iterations 200 2>/dev/null | grep -E "^ +16 \|"
}
rs() { # $1=serve
  "$BIN" benchmark --config /mnt/d/code/Rust_KataGo/configs/gtp_smoke.cfg --model "$MODEL" \
    --override-config "$OV,numNNServerThreadsPerModel=$1" --override-config nnMaxBatchSize=16 \
    -n 30 -t 32 2>/dev/null | grep -oE "nnEvals/s = [0-9.]+, nnBatches/s = [0-9.]+, avgBatchSize = [0-9.]+"
}

echo "=== A) serve=2 NOGRAPH W32 ==="; KATAGO_CUDA_NOGRAPH=1 r 2 32
echo "=== B) serve=2 NOGRAPH W64 ==="; KATAGO_CUDA_NOGRAPH=1 r 2 64
echo "=== C) serve=2 NOGRAPH PADBATCH W64(fork 全拓扑)==="; KATAGO_CUDA_NOGRAPH=1 KATAGO_CUDA_PADBATCH=1 r 2 64
echo "=== D) serve=1 NOGRAPH W32(单流无图参照)==="; KATAGO_CUDA_NOGRAPH=1 r 1 32
echo "=== E) 搜索 t=32 serve=2 NOGRAPH ==="; KATAGO_CUDA_NOGRAPH=1 rs 2
echo "=== F) 搜索 t=32 serve=1 NOGRAPH(参照)==="; KATAGO_CUDA_NOGRAPH=1 rs 1
echo "=== done ==="

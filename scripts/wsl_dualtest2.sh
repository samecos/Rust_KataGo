#!/bin/bash
# 双流复测 round 2:稳定性复跑 + 扩展探顶(W96/128,serve=3)+ 归因对照。
BIN=/mnt/d/code/Rust_KataGo/target/cudarocmopt-wsl/release/katago-rs
MODEL=/mnt/d/code/b11fix.onnx
PLAN=/mnt/d/code/Rust_KataGo/plans/best-tactic-plan-sm120-q64-wsl.json
OV="nnBackend=cudabackend,cudaTacticPlan=$PLAN"

r() { # $1=serve $2=workers
  "$BIN" nnbench --model "$MODEL" --override-config "$OV,numNNServerThreadsPerModel=$1" \
    --mode eval --batch 16 --workers "$2" --iterations 200 2>/dev/null | grep -E "^ +16 \|"
}

echo "=== B 复跑1) serve=2 NOGRAPH W64 ==="; KATAGO_CUDA_NOGRAPH=1 r 2 64
echo "=== B 复跑2) serve=2 NOGRAPH W64 ==="; KATAGO_CUDA_NOGRAPH=1 r 2 64
echo "=== G) serve=2 NOGRAPH W96 ==="; KATAGO_CUDA_NOGRAPH=1 r 2 96
echo "=== H) serve=2 NOGRAPH W128 ==="; KATAGO_CUDA_NOGRAPH=1 r 2 128
echo "=== I) serve=1 NOGRAPH W64(供数归因)==="; KATAGO_CUDA_NOGRAPH=1 r 1 64
echo "=== J) serve=2 graph 尝试 W64(capture 互斥开销)==="; r 2 64
echo "=== K) serve=3 NOGRAPH W96 ==="; KATAGO_CUDA_NOGRAPH=1 r 3 96
echo "=== done ==="

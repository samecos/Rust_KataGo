#!/bin/bash
# RELAXED capture 模式验证:serve=2 能否带 graph(双流+图叠加)。
BIN=/root/rk-target/release/katago-rs
MODEL=/mnt/d/code/b11fix.onnx
PLAN=/mnt/d/code/Rust_KataGo/plans/best-tactic-plan.json
OV="nnBackend=cudabackend,cudaTacticPlan=$PLAN"

r() { # $1=serve $2=workers
  "$BIN" nnbench --model "$MODEL" --override-config "$OV,numNNServerThreadsPerModel=$1" \
    --mode eval --batch 16 --workers "$2" --iterations 200 2>&1 | grep -E "^ +16 \||WARNING" | head -3
}
rs() { # $1=serve $2=threads
  "$BIN" benchmark --config /mnt/d/code/Rust_KataGo/configs/gtp_smoke.cfg --model "$MODEL" \
    --override-config "$OV,numNNServerThreadsPerModel=$1" --override-config nnMaxBatchSize=16 \
    -n 30 -t "$2" 2>/dev/null | grep -oE "nnEvals/s = [0-9.]+, nnBatches/s = [0-9.]+, avgBatchSize = [0-9.]+"
}

echo "=== L) serve=2 graph(RELAXED) W64 ==="; r 2 64
echo "=== L 复跑) serve=2 graph(RELAXED) W64 ==="; r 2 64
echo "=== M) serve=1 graph(RELAXED) 回归 W32 ==="; r 1 32
echo "=== N) 搜索 t=32 serve=1 graph(RELAXED) 回归 ==="; rs 1 32
echo "=== O) 搜索 t=48 serve=1 graph(RELAXED) ==="; rs 1 48
echo "=== done ==="

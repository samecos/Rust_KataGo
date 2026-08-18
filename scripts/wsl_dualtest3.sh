#!/bin/bash
# 搜索语境高线程下双流能否翻正:t=48 serve=1 vs serve=2(NOGRAPH)。
BIN=/root/rk-target/release/katago-rs
MODEL=/mnt/d/code/b11fix.onnx
PLAN=/mnt/d/code/Rust_KataGo/plans/best-tactic-plan.json
OV="nnBackend=cudabackend,cudaTacticPlan=$PLAN"

rs() { # $1=serve $2=threads
  "$BIN" benchmark --config /mnt/d/code/Rust_KataGo/configs/gtp_smoke.cfg --model "$MODEL" \
    --override-config "$OV,numNNServerThreadsPerModel=$1" --override-config nnMaxBatchSize=16 \
    -n 30 -t "$2" 2>/dev/null | grep -oE "nnEvals/s = [0-9.]+, nnBatches/s = [0-9.]+, avgBatchSize = [0-9.]+"
}

echo "=== t=48 serve=1 graph(基线)==="; rs 1 48
echo "=== t=48 serve=2 NOGRAPH ==="; KATAGO_CUDA_NOGRAPH=1 rs 2 48
echo "=== t=48 serve=1 NOGRAPH ==="; KATAGO_CUDA_NOGRAPH=1 rs 1 48
echo "=== done ==="

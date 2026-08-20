#!/usr/bin/env bash
set -eu

export CUDA_HOME=/usr/local/cuda-13.3
export PATH=/root/.cargo/bin:/usr/local/cuda-13.3/bin:/usr/bin:/bin:/usr/sbin:/sbin
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:/usr/lib/wsl/lib

ROOT=/mnt/d/code/Rust_KataGo
BIN=$ROOT/target/cudarocmopt-wsl/release/katago-rs
MODEL=/mnt/d/code/b11fix.onnx
PLAN=$ROOT/plans/best-tactic-plan-sm120-q64-wsl.json

cd "$ROOT"
for disabled in false true; do
  for batch in 1 16; do
    for repeat in 1 2; do
      override="nnBackend=cudabackend,cudaTacticPlan=$PLAN"
      if [ "$disabled" = true ]; then
        override="$override,cudaDisableWarmup=true"
      fi
      echo "=== disabled=$disabled batch=$batch repeat=$repeat ==="
      /usr/bin/time -f 'wall_ms=%e' "$BIN" nnbench \
        --model "$MODEL" --override-config "$override" --mode eval \
        --batch "$batch" --workers $((batch * 2)) --warmup 0 --iterations 1 \
        2>&1 | grep -E 'nnbench|^ +[0-9]+ \\||warmup|wall_ms|cuda-tactic'
    done
  done
done

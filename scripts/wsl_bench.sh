#!/bin/bash
# WSL 下的构建 + fork 基线一致性基准(消除 Windows WDDM 桌面噪声)。
# 用法(Windows 侧): wsl -d Ubuntu-24.04 -- bash -c "tr -d '\r' < /mnt/d/code/Rust_KataGo/scripts/wsl_bench.sh > /tmp/wb.sh && bash /tmp/wb.sh"
set -e
export CUDA_HOME=/usr/local/cuda-13.3
export PATH="/root/.cargo/bin:/usr/local/cuda-13.3/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export LD_LIBRARY_PATH="/usr/local/cuda-13.3/lib64:/usr/lib/wsl/lib"
export KATAGO_CUTLASS_ROOT=/mnt/d/code/cutlass
export CARGO_TARGET_DIR=/mnt/d/code/Rust_KataGo/target/cudarocmopt-wsl
REPO=/mnt/d/code/Rust_KataGo
BIN=$CARGO_TARGET_DIR/release/katago-rs
MODEL=/mnt/d/code/b11fix.onnx

echo "=== env ==="
cargo --version
nvcc --version | tail -1

echo "=== build ==="
cd "$REPO"
cargo build -p katago --features cuda --release 2>&1 | grep -E "^error|warning: kata_nn@|Finished" | tail -5 || true
test -x "$BIN" || { echo "FATAL: binary missing"; exit 1; }
echo "cfg dualffn: $(grep -h 'rustc-cfg=katago_dualffn' "$CARGO_TARGET_DIR"/release/build/kata_nn-*/output | wc -l) (必须 ≥1)"

run_eval() { # $1 = extra override (plan path or empty)
  if [ -n "$1" ]; then
    "$BIN" nnbench --model "$MODEL" --override-config "nnBackend=cudabackend,cudaTacticPlan=$1" --mode eval --batch 16 --iterations 200 2>/dev/null | grep -E "^ +16 \|"
  else
    "$BIN" nnbench --model "$MODEL" --override-config nnBackend=cudabackend --mode eval --batch 16 --iterations 200 2>/dev/null | grep -E "^ +16 \|"
  fi
}

run_search() { # $1 = extra override
  if [ -n "$1" ]; then
    "$BIN" benchmark --config "$REPO/configs/gtp_smoke.cfg" --model "$MODEL" \
      --override-config "nnBackend=cudabackend,cudaTacticPlan=$1" --override-config nnMaxBatchSize=16 \
      -n 30 -t 32 2>/dev/null | grep -oE "nnEvals/s = [0-9.]+, nnBatches/s = [0-9.]+, avgBatchSize = [0-9.]+"
  else
    "$BIN" benchmark --config "$REPO/configs/gtp_smoke.cfg" --model "$MODEL" \
      --override-config nnBackend=cudabackend --override-config nnMaxBatchSize=16 \
      -n 30 -t 32 2>/dev/null | grep -oE "nnEvals/s = [0-9.]+, nnBatches/s = [0-9.]+, avgBatchSize = [0-9.]+"
  fi
}

PLAN="$REPO/plans/best-tactic-plan-sm120-q64-wsl.json"

echo "=== eval B16 ABAB(q64 schema2 plan vs no-plan)==="
for round in 1 2; do
  echo "--- round $r ---"
  echo -n "A no-plan : "; run_eval ""
  echo -n "B q64 plan : "; run_eval "$PLAN"
done

echo "=== search t=32 cap16 ABAB ==="
for round in 1 2; do
  echo "--- round $r ---"
  echo -n "A no-plan : "; run_search ""
  echo -n "B q64 plan : "; run_search "$PLAN"
done

echo "=== direct T(B)(注意:direct 模式不安装 plan——此曲线为默认 tactic)==="
"$BIN" nnbench --model "$MODEL" --override-config "nnBackend=cudabackend,cudaTacticPlan=$PLAN" \
  --mode direct --batch 1,4,8,12,16,24,32 --iterations 200 2>/dev/null | grep -E "^ +[0-9]+ \|"
echo "=== done ==="

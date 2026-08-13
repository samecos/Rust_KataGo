#!/usr/bin/env bash
# GTP 冒烟测试：喂给 katago-rs 一个最小 GTP 会话并检查基本应答。
# 用法：scripts/gtp_smoke.sh
# 环境变量：KATAGO_BIN（默认 target/debug/katago-rs）
#           KATAGO_CFG（默认 configs/gtp_smoke.cfg）
#           KATAGO_MODEL（默认 D:/code/b11fix.onnx）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${KATAGO_BIN:-$ROOT/target/debug/katago-rs}"
CFG="${KATAGO_CFG:-$ROOT/configs/gtp_smoke.cfg}"
MODEL="${KATAGO_MODEL:-D:/code/b11fix.onnx}"

if [ ! -f "$BIN" ]; then
  echo "error: binary not found at $BIN (先 cargo build --workspace)" >&2
  exit 1
fi

"$BIN" gtp --config "$CFG" --model "$MODEL" <<'EOF'
protocol_version
name
version
list_commands
boardsize 19
clear_board
komi 7.5
play B D4
showboard
genmove W
undo
final_score
quit
EOF

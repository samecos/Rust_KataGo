#!/usr/bin/env python
"""两个 GTP 引擎对局（M5 棋力对拍）。

用法:
  .venv/Scripts/python.exe scripts/play_match.py \
      --engine-a "D:/code/Rust_KataGo/target/debug/katago-rs gtp --config configs/gtp_trt.cfg --model D:/code/b11fix.onnx" \
      --engine-b "D:/code/KataGo/build_eigen/Release/katago.exe gtp -config <cfg> -model D:/code/Rust_KataGo/models/b11c768h12nbt3tflrs-fson-silu.bin.gz" \
      --games 100 --visits 800

默认：双方各下一半黑棋；对局用固定 visit 数（公平对拍）。
"""
import argparse
import random
import shlex
import subprocess
import sys


def start_engine(cmdline: str):
    return subprocess.Popen(
        shlex.split(cmdline), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )


def cmd(proc, line: str) -> str:
    proc.stdin.write(line + "\n")
    proc.stdin.flush()
    while True:
        out = proc.stdout.readline()
        if out == "":
            raise RuntimeError(f"engine died (exit {proc.poll()})")
        out = out.rstrip("\n")
        if out.startswith("= "):
            return out[2:]
        if out.startswith("? "):
            raise RuntimeError(f"engine error: {out[2:]}")


def play_game(cmd_a, cmd_b, a_black: bool, visits: int) -> str:
    """返回胜者 'B'/'W'，或异常时抛错。"""
    black, white = (cmd_a, cmd_b) if a_black else (cmd_b, cmd_a)
    for engine in (black, white):
        cmd(engine, "boardsize 19")
        cmd(engine, "clear_board")
        cmd(engine, "komi 7.5")
        cmd(engine, f"kata-set-rules chinese")  # 若引擎不支持会报错——见主流程 fallback
        cmd(engine, f"maxVisits {visits}" if False else f"kata-set-param maxVisits {visits}")

    # 某些引擎不支持 kata-set-param maxVisits；退而求其次用配置（调用方保证）。
    moves = []
    for turn in range(1000):
        eng = black if turn % 2 == 0 else white
        color = "B" if turn % 2 == 0 else "W"
        mv = cmd(eng, f"genmove {color}").strip()
        print(f"[dbg] turn {turn} {color} -> {mv}", flush=True)
        moves.append((color, mv))
        if mv.lower() == "resign":
            return "W" if color == "B" else "B"
        # genmove 方已自行落子（GTP 规范），只需同步给对方引擎
        for e in (black, white):
            if e is eng:
                continue
            cmd(e, f"play {color} {mv}")
        if mv.lower() == "pass" and len(moves) >= 2 and moves[-2][1].lower() == "pass":
            # 双 pass 终局
            score = cmd(black, "final_score").strip()
            return "B" if score.startswith("B+") else "W"
    # 走满手上限仍未双 pass：直接计分兜底（等强引擎偶发互不虚着时会走到这里）
    score = cmd(black, "final_score").strip()
    return "B" if score.startswith("B+") else "W"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-a", required=True)
    ap.add_argument("--engine-b", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--visits", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    wins_a = 0
    for i in range(args.games):
        a = start_engine(args.engine_a)
        b = start_engine(args.engine_b)
        try:
            a_black = (i % 2 == 0)
            winner = play_game(a, b, a_black, args.visits)
            a_won = (winner == "B") == a_black
            wins_a += int(a_won)
            print(f"game {i + 1}/{args.games}: winner={winner} a_black={a_black} a_won={a_won}")
        finally:
            try:
                a.stdin.write("quit\n"); a.stdin.flush()
            except Exception:
                pass
            try:
                b.stdin.write("quit\n"); b.stdin.flush()
            except Exception:
                pass
            a.terminate(); b.terminate()

    print(f"A wins: {wins_a}/{args.games} ({wins_a / args.games:.1%})")


if __name__ == "__main__":
    main()

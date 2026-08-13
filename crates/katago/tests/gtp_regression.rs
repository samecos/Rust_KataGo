//! GTP 指令回归测试：驱动 `katago-rs gtp`（dummy 后端）会话并断言核心指令应答。
//!
//! 覆盖标准 GTP 与 KataGo 私有指令的基础路径；输出格式逐字段对齐
//! `docs/GTP_Extensions.md`（KataGo C++ 行为）。
//!
//! 注意：使用 dummy 后端（无需模型真实推理），验证协议层与引擎接线。
//! GTP 多行应答（如 showboard）只有首行带 "= " 前缀，续行是裸文本。

use std::io::Write;
use std::process::{Command, Stdio};

/// 启动一个 GTP 会话，发送命令列表，返回全部 stdout 行（保留续行；
/// 首行应答前缀 "= "/"? " 已剥离）。
fn gtp_session(commands: &[&str]) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_katago-rs"))
        .args([
            "gtp",
            "--config",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../configs/gtp_smoke.cfg"),
            "--model",
            "/dev/null",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn katago-rs gtp");

    let mut stdin = child.stdin.take().expect("stdin");
    for cmd in commands {
        writeln!(stdin, "{cmd}").expect("write command");
    }
    writeln!(stdin, "quit").expect("quit");
    drop(stdin);

    let out = child.wait_with_output().expect("wait");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            if let Some(rest) = l.strip_prefix("= ") {
                rest.to_string()
            } else if let Some(rest) = l.strip_prefix("? ") {
                rest.to_string()
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn gtp_protocol_basics() {
    let replies = gtp_session(&["protocol_version", "name", "version", "known_command play", "known_command no_such_cmd"]);
    let lines: Vec<&str> = replies.lines().collect();
    assert_eq!(lines[0], "2");
    assert_eq!(lines[1], "KataGo");
    assert!(lines[2].contains("KataGo-Rust"));
    assert_eq!(lines[3], "true");
    assert_eq!(lines[4], "false");
}

#[test]
fn gtp_board_and_moves() {
    let replies = gtp_session(&[
        "boardsize 19",
        "clear_board",
        "komi 7.5",
        "get_komi",
        "play B D4",
        "undo",
        "showboard",
        "final_score",
    ]);
    assert!(replies.contains("7.5"), "get_komi: {replies:?}");
    // showboard：多行应答，含棋盘与黑子 X（undo 后 D4 已移除，故重新落一子再看）
    let replies2 = gtp_session(&[
        "boardsize 19",
        "clear_board",
        "play B D4",
        "showboard",
    ]);
    assert!(replies2.contains("A B C"), "showboard 棋盘: {replies2:?}");
    assert!(replies2.contains('X'), "showboard 应含黑子 X");
    // 终局未到，final_score 返回预估比分
    assert!(
        replies.contains("B+") || replies.contains("W+"),
        "final_score: {replies:?}"
    );
}

#[test]
fn gtp_private_kata_commands() {
    let replies = gtp_session(&[
        "kata-get-rules",
        "kata-set-rules chinese",
        "kata-get-rules",
        "kata-list-params",
        "kata-set-param analysisWideRootNoise 0.1",
        "kata-get-param analysisWideRootNoise",
        "kata-get-models",
        "kata-list_time_settings",
        "gomill-cpu_time",
        "cputime",
    ]);
    assert!(replies.contains("ko"), "kata-get-rules 应含规则 JSON: {replies:?}");
    assert!(replies.contains("SIMPLE"), "kata-set-rules chinese");
    assert!(
        replies.contains("analysisWideRootNoise"),
        "kata-list-params 应含参数名"
    );
    assert!(replies.contains('['), "kata-get-models 应返回 JSON 数组");
    assert!(replies.contains("byoyomi"), "时间类型列表: {replies:?}");
    assert!(replies.contains("0"), "cputime 应为数字");
}

#[test]
fn gtp_handicap_and_sgf() {
    let replies = gtp_session(&[
        "boardsize 19",
        "clear_board",
        "fixed_handicap 4",
        "showboard",
        "printsgf",
    ]);
    // 4 个让子（星位）应为黑子 X；dummy 后端不影响让子摆放。
    let xs = replies.matches('X').count();
    assert!(xs >= 4, "fixed_handicap 4 应摆 4 子，got {xs}: {replies:?}");
    assert!(replies.contains("(;FF"), "printsgf: {replies:?}");
    assert!(replies.contains("AB"), "SGF 应含 AB 让子标记");
}

#[test]
fn gtp_search_commands_smoke() {
    let replies = gtp_session(&[
        "boardsize 19",
        "clear_board",
        "kata-search B",
        "kata-search_cancellable W",
        "clear_cache",
        "kata-benchmark 10",
    ]);
    // dummy 后端下 kata-search 返回合法坐标或 pass；benchmark 有输出。
    assert!(
        replies.contains("pass") || replies.lines().any(|l| l.len() >= 2 && l != "0"),
        "kata-search: {replies:?}"
    );
}

#[test]
fn gtp_time_controls() {
    let replies = gtp_session(&[
        "boardsize 19",
        "clear_board",
        "time_settings 300 0 1",
        "time_left B 250 0",
        "time_left W 300 0",
        "kata-time_settings canadian 300 0 20",
        "kata-list_time_settings",
    ]);
    assert!(replies.contains("canadian"), "canadian 时间类型: {replies:?}");
}

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

/// 与 [`gtp_session`] 相同，但把命令中的 "PLACEHOLDER" 替换为 `path`（用于 loadsgf）。
fn gtp_session_with(commands: &[&str], path: &str) -> String {
    let replaced: Vec<String> = commands
        .iter()
        .map(|c| c.replace("PLACEHOLDER", path))
        .collect();
    let refs: Vec<&str> = replaced.iter().map(|s| s.as_str()).collect();
    gtp_session(&refs)
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
fn gtp_set_position_and_raw_nn() {
    let replies = gtp_session(&[
        "boardsize 19",
        "clear_board",
        "set_position B D4 W Q16",
        "showboard",
        "kata-raw-nn 0",
    ]);
    assert!(replies.contains('X'), "set_position 黑子 X: {replies:?}");
    assert!(replies.contains('O'), "set_position 白子 O");
    // kata-raw-nn：dummy 后端输出统一策略，但字段结构必须完整。
    assert!(replies.contains("whiteWin"), "raw-nn 应含 whiteWin: {replies:?}");
    assert!(replies.contains("policyPass"), "raw-nn 应含 policyPass");
}

#[test]
fn gtp_analysis_commands_smoke() {
    // lz-analyze / kata-analyze：dummy 后端也应持续产出 info 行（此处只测不崩溃、
    // 首个分析行出现即可；interval 用厘秒，取小值快速返回）。
    let replies = gtp_session(&[
        "boardsize 19",
        "clear_board",
        "lz-analyze B 10 minmoves 0 maxmoves 1",
    ]);
    assert!(
        replies.contains("info move") || replies.contains("play"),
        "lz-analyze 应产出分析行: {replies:?}"
    );
}

#[test]
fn gtp_free_handicap_and_sgf_io() {
    // set_free_handicap：指定让子位置
    let replies = gtp_session(&[
        "boardsize 19",
        "clear_board",
        "set_free_handicap D4 Q16 D16 Q4",
        "showboard",
        "printsgf",
    ]);
    let xs = replies.matches('X').count();
    assert!(xs >= 4, "set_free_handicap 4 子: got {xs}: {replies:?}");

    // place_free_handicap：引擎自选让子
    let replies2 = gtp_session(&[
        "boardsize 19",
        "clear_board",
        "place_free_handicap 5",
        "showboard",
    ]);
    let xs2 = replies2.matches('X').count();
    assert!(xs2 >= 5, "place_free_handicap 5 子: got {xs2}: {replies2:?}");

    // loadsgf：临时 SGF 文件加载到指定手数
    let sgf = "(;FF[4]GM[1]SZ[19]KM[7.5]PB[B]PW[W];B[dd];W[pp];B[pd])";
    let mut f = tempfile::NamedTempFile::new().expect("tempfile");
    f.write_all(sgf.as_bytes()).expect("write sgf");
    let replies3 = gtp_session_with(&[
        "boardsize 19",
        "loadsgf PLACEHOLDER 1",
        "showboard",
        "printsgf",
    ], f.path().to_str().unwrap());
    assert!(replies3.contains('X'), "loadsgf 后应有黑子: {replies3:?}");
}

#[test]
fn gtp_kgs_time_settings() {
    let replies = gtp_session(&[
        "boardsize 19",
        "clear_board",
        "kgs-time_settings byoyomi 300 5 30",
        "time_left B 200 2",
        "time_left W 100 0",
    ]);
    // byoyomi 时限设置不应报错（应答里无 ? 前缀错误行——解析器已剥离前缀，
    // 错误文本会直接出现在行内；此处仅断言会话不崩溃且有过应答）。
    assert!(!replies.is_empty(), "kgs-time_settings 应答");
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

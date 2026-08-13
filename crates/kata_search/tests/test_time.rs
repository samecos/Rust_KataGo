use kata_game::board::{Board, P_BLACK};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_search::time_control::TimeControls;

fn parse_annotated_board(x_size: i32, y_size: i32, s: &str) -> Board {
    // Remove the optional header line (e.g. "   A B C D ..."), y-coordinate
    // prefixes, inter-cell spaces, and move-number digits so that the diagram
    // reduces to one character per intersection.
    let mut lines: Vec<&str> = s.trim().split('\n').collect();
    if !lines.is_empty() && lines[0].trim().starts_with('A') {
        lines.remove(0);
    }
    assert_eq!(lines.len(), y_size as usize, "wrong number of board rows");
    let cleaned_rows: Vec<String> = lines
        .iter()
        .map(|line| {
            let trimmed = line.trim();
            // Strip leading y-coordinate digits, then remove spaces and digits.
            let after_coords = trimmed
                .chars()
                .skip_while(|c| c.is_ascii_digit())
                .collect::<String>();
            after_coords
                .chars()
                .filter(|c| !c.is_whitespace() && !c.is_ascii_digit())
                .collect()
        })
        .collect();
    for row in &cleaned_rows {
        assert_eq!(
            row.len(),
            x_size as usize,
            "row length mismatch: got {}",
            row.len()
        );
    }
    Board::parse_board(x_size, y_size, &cleaned_rows.join("\n"), '\n').unwrap()
}

fn board9_early() -> (Board, BoardHistory) {
    let board = Board::parse_board(
        9,
        9,
        ".........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n.........",
        '\n',
    )
    .unwrap();
    let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
    (board, hist)
}

fn board9_late() -> (Board, BoardHistory) {
    let board = Board::parse_board(
        9,
        9,
        "..xoo..x.\n.x.x.ox.x\n..xoxo.x.\nxx.oooo..\noxx..oxo.\noox.ox...\n..o.ooxx.\n.o..ox.x.\n...oxxx..",
        '\n',
    )
    .unwrap();
    let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
    (board, hist)
}

fn board19_early() -> (Board, BoardHistory) {
    let rows: Vec<String> = (0..19).map(|_| ".".repeat(19)).collect();
    let board = Board::parse_board(19, 19, &rows.join("\n"), '\n').unwrap();
    let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
    (board, hist)
}

fn board19_late() -> (Board, BoardHistory) {
    let board = parse_annotated_board(
        19,
        19,
        "   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . O . O . . X X . . . . . . . X O .
17 . O . X O . O O . X . X . . . . X X O
16 O X X . O X X O . . . . . X . X X O .
15 . . . . X O O O X . . . . X . . . O .
14 . . X X X O . O . . . . O X O X X O .
13 . . . . X X O O . . . . O . O O O O .
12 . . O . O O X . . . X . . O . O X . .
11 X X X . . X X . X X O O . O X . X . .
10 O O X . X . . . X O . . . . X . . . .
 9 O X . X . . . . . O . X . O O X . X .
 8 O O O O O O O . O . O X . . O X X O .
 7 . X . . . X O . . . O O O . . X O O .
 6 X . X X . X X . X . . O X X X . X O .
 5 . X O O X X X X . . O X O . . X . . .
 4 . O . O X O O O X X X X . . X . O O .
 3 . O . O O X O O O O . X . . X O O X .
 2 . O . O X . X O O X1X . . . . X X O O
 1 . . O . . X .2X3. O . . . . . . . X .",
    );
    let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
    (board, hist)
}

fn try_time_controls_on_board(
    name: &str,
    time_controls: &TimeControls,
    board: &Board,
    hist: &BoardHistory,
    lag_buffer: f64,
) -> String {
    let (min_time, recommended_time, max_time) =
        time_controls.get_time(board, hist, lag_buffer).unwrap();
    let rrec0 = time_controls.round_up_time_limit_if_needed(lag_buffer, 0.0, recommended_time);
    let rreclimit = time_controls.round_up_time_limit_if_needed(
        lag_buffer,
        recommended_time - 0.000_001,
        recommended_time,
    );
    let rreclimit2 =
        time_controls.round_up_time_limit_if_needed(lag_buffer, rrec0 - 0.000_001, rrec0);

    format!(
        "{} min rec max = {} {} {} roundedrec(used0) {} roundedrec(usedlimit) {} roundedrec(usedlimit2) {}",
        name, min_time, recommended_time, max_time, rrec0, rreclimit, rreclimit2
    )
}

fn try_time_controls_on_boards(time_controls: &TimeControls, lag_buffer: f64) -> Vec<String> {
    let (board9e, hist9e) = board9_early();
    let (board9l, hist9l) = board9_late();
    let (board19e, hist19e) = board19_early();
    let (board19l, hist19l) = board19_late();
    [
        ("board9Early", board9e, hist9e),
        ("board9Late", board9l, hist9l),
        ("board19Early", board19e, hist19e),
        ("board19Late", board19l, hist19l),
    ]
    .iter()
    .map(|(name, board, hist)| {
        try_time_controls_on_board(name, time_controls, board, hist, lag_buffer)
    })
    .collect()
}

fn run_scenario(header: &str, time_controls: &TimeControls, lag_buffer: f64) -> String {
    let mut lines = vec![
        "===================================================================".to_string(),
        header.to_string(),
        "===================================================================".to_string(),
    ];
    lines.extend(try_time_controls_on_boards(time_controls, lag_buffer));
    lines.join("\n")
}

fn absolute_time_controls(main_time: f64, main_time_left: f64, increment: f64) -> TimeControls {
    let mut tc = TimeControls::absolute_time(main_time);
    tc.increment = increment;
    tc.original_num_periods = 0;
    tc.num_stones_per_period = 0;
    tc.per_period_time = 0.0;
    tc.main_time_left = main_time_left;
    tc.in_overtime = false;
    tc.num_periods_left_including_current = 0;
    tc.num_stones_left_in_period = 0;
    tc.time_left_in_period = 0.0;
    tc
}

#[allow(clippy::too_many_arguments)]
fn byo_yomi_time_controls(
    main_time: f64,
    main_time_left: f64,
    per_period_time: f64,
    num_periods: i32,
    num_stones_per_period: i32,
    in_overtime: bool,
    time_left_in_period: f64,
    num_stones_left_in_period: i32,
) -> TimeControls {
    let mut tc = TimeControls::canadian_or_byo_yomi_time(
        main_time,
        per_period_time,
        num_periods,
        num_stones_per_period,
    );
    tc.main_time_left = main_time_left;
    tc.in_overtime = in_overtime;
    tc.num_periods_left_including_current = num_periods;
    tc.num_stones_left_in_period = num_stones_left_in_period;
    tc.time_left_in_period = time_left_in_period;
    tc
}

const EXPECTED_TEST_BASIC_10M_FISCHER_10M_LEFT_10S_INCREMENT_MAIN_TIME_LIMIT_10M: &str = r#"===================================================================
Basic 10m fischer time controls, 10m left, 10s increment, main time limit 10m
===================================================================
board9Early min rec max = 9 24.551155115511552 126.8 roundedrec(used0) 24.551155115511552 roundedrec(usedlimit) 24.551155115511552 roundedrec(usedlimit2) 24.551155115511552
board9Late min rec max = 9 46.10236220472441 126.8 roundedrec(used0) 46.10236220472441 roundedrec(usedlimit) 46.10236220472441 roundedrec(usedlimit2) 46.10236220472441
board19Early min rec max = 9 13.122484689413824 126.8 roundedrec(used0) 13.122484689413824 roundedrec(usedlimit) 13.122484689413824 roundedrec(usedlimit2) 13.122484689413824
board19Late min rec max = 9 20.03512880562061 126.8 roundedrec(used0) 20.03512880562061 roundedrec(usedlimit) 20.03512880562061 roundedrec(usedlimit2) 20.03512880562061"#;

const EXPECTED_TEST_BASIC_10M_FISCHER_10M_LEFT_10S_INCREMENT_MAIN_TIME_LIMIT_10M_5S: &str = r#"===================================================================
Basic 10m fischer time controls, 10m left, 10s increment, main time limit 10m+5s
===================================================================
board9Early min rec max = 4 24.551155115511552 126.8 roundedrec(used0) 24.551155115511552 roundedrec(usedlimit) 24.551155115511552 roundedrec(usedlimit2) 24.551155115511552
board9Late min rec max = 4 46.10236220472441 126.8 roundedrec(used0) 46.10236220472441 roundedrec(usedlimit) 46.10236220472441 roundedrec(usedlimit2) 46.10236220472441
board19Early min rec max = 4 13.122484689413824 126.8 roundedrec(used0) 13.122484689413824 roundedrec(usedlimit) 13.122484689413824 roundedrec(usedlimit2) 13.122484689413824
board19Late min rec max = 4 20.03512880562061 126.8 roundedrec(used0) 20.03512880562061 roundedrec(usedlimit) 20.03512880562061 roundedrec(usedlimit2) 20.03512880562061"#;

const EXPECTED_TEST_BASIC_10S_FISCHER_MINUS1S_LEFT_10S_INCREMENT_10S_MAIN_TIME_LIMIT: &str = r#"===================================================================
Basic 10s fischer time controls, -1s left, 10s increment, 10s main time limit
===================================================================
board9Early min rec max = 0 0 0 roundedrec(used0) 0 roundedrec(usedlimit) 0 roundedrec(usedlimit2) 0
board9Late min rec max = 0 0 0 roundedrec(used0) 0 roundedrec(usedlimit) 0 roundedrec(usedlimit2) 0
board19Early min rec max = 0 0 0 roundedrec(used0) 0 roundedrec(usedlimit) 0 roundedrec(usedlimit2) 0
board19Late min rec max = 0 0 0 roundedrec(used0) 0 roundedrec(usedlimit) 0 roundedrec(usedlimit2) 0"#;

const EXPECTED_TEST_BASIC_1H_ABSOLUTE_10M_LEFT: &str = r#"===================================================================
Basic 1h absolute time controls, 10m left
===================================================================
board9Early min rec max = 0 11.356884992264053 118.8 roundedrec(used0) 11.356884992264053 roundedrec(usedlimit) 11.356884992264053 roundedrec(usedlimit2) 11.356884992264053
board9Late min rec max = 0 21.625118035882906 118.8 roundedrec(used0) 21.625118035882906 roundedrec(usedlimit) 21.625118035882906 roundedrec(usedlimit2) 21.625118035882906
board19Early min rec max = 0 2.3007301281168204 118.8 roundedrec(used0) 2.3007301281168204 roundedrec(usedlimit) 2.3007301281168204 roundedrec(usedlimit2) 2.3007301281168204
board19Late min rec max = 0 5.512639304158739 118.8 roundedrec(used0) 5.512639304158739 roundedrec(usedlimit) 5.512639304158739 roundedrec(usedlimit2) 5.512639304158739"#;

const EXPECTED_TEST_BASIC_1H_ABSOLUTE_ALL_TIME_LEFT: &str = r#"===================================================================
Basic 1h absolute time controls, all time left
===================================================================
board9Early min rec max = 0 73.24445590510572 718.8 roundedrec(used0) 73.24445590510572 roundedrec(usedlimit) 73.24445590510572 roundedrec(usedlimit2) 73.24445590510572
board9Late min rec max = 0 134.93956562795088 718.8 roundedrec(used0) 134.93956562795088 roundedrec(usedlimit) 134.93956562795088 roundedrec(usedlimit2) 134.93956562795088
board19Early min rec max = 0 18.831932773109244 718.8 roundedrec(used0) 18.831932773109244 roundedrec(usedlimit) 18.831932773109244 roundedrec(usedlimit2) 18.831932773109244
board19Late min rec max = 0 38.13019842348464 718.8 roundedrec(used0) 38.13019842348464 roundedrec(usedlimit) 38.13019842348464 roundedrec(usedlimit2) 38.13019842348464"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_15S_LEFT_1_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, 15s left, 1 periods of 30s
===================================================================
board9Early min rec max = 0 43 43 roundedrec(used0) 43 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44
board9Late min rec max = 0 43 43 roundedrec(used0) 43 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44
board19Early min rec max = 0 43 43 roundedrec(used0) 43 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44
board19Late min rec max = 0 43 43 roundedrec(used0) 43 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_15S_LEFT_2_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, 15s left, 2 periods of 30s
===================================================================
board9Early min rec max = 0 44 44 roundedrec(used0) 44 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44
board9Late min rec max = 0 44 44 roundedrec(used0) 44 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44
board19Early min rec max = 0 44 44 roundedrec(used0) 44 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44
board19Late min rec max = 0 44 44 roundedrec(used0) 44 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_15S_LEFT_3_MOVES_CANADIAN_IN_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, 15s left, 3 moves canadian in 30s
===================================================================
board9Early min rec max = 0 14 26 roundedrec(used0) 14 roundedrec(usedlimit) 16 roundedrec(usedlimit2) 16
board9Late min rec max = 0 14 26 roundedrec(used0) 14 roundedrec(usedlimit) 16 roundedrec(usedlimit2) 16
board19Early min rec max = 0 14 26 roundedrec(used0) 14 roundedrec(usedlimit) 16 roundedrec(usedlimit2) 16
board19Late min rec max = 0 14 26 roundedrec(used0) 14 roundedrec(usedlimit) 16 roundedrec(usedlimit2) 16"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_1_PERIODS_30S_ENTERED_OVERTIME_15S_USED: &str = r#"===================================================================
Basic 1h byo yomi time controls, 1 periods of 30s, entered overtime 15s used
===================================================================
board9Early min rec max = 13 13 13 roundedrec(used0) 14 roundedrec(usedlimit) 14 roundedrec(usedlimit2) 14
board9Late min rec max = 13 13 13 roundedrec(used0) 14 roundedrec(usedlimit) 14 roundedrec(usedlimit2) 14
board19Early min rec max = 13 13 13 roundedrec(used0) 14 roundedrec(usedlimit) 14 roundedrec(usedlimit2) 14
board19Late min rec max = 13 13 13 roundedrec(used0) 14 roundedrec(usedlimit) 14 roundedrec(usedlimit2) 14"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_1_PERIODS_30S_JUST_ENTERED_OVERTIME: &str = r#"===================================================================
Basic 1h byo yomi time controls, 1 periods of 30s, just entered overtime
===================================================================
board9Early min rec max = 28 28 28 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board9Late min rec max = 28 28 28 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board19Early min rec max = 28 28 28 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board19Late min rec max = 28 28 28 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_3_MOVES_CANADIAN_IN_30S_ENTERED_OVERTIME_15S_USED: &str = r#"===================================================================
Basic 1h byo yomi time controls, 3 moves canadian in 30s, entered overtime 15s used
===================================================================
board9Early min rec max = 0 4 5 roundedrec(used0) 4 roundedrec(usedlimit) 4 roundedrec(usedlimit2) 4
board9Late min rec max = 0 4 5 roundedrec(used0) 4 roundedrec(usedlimit) 4 roundedrec(usedlimit2) 4
board19Early min rec max = 0 4 5 roundedrec(used0) 4 roundedrec(usedlimit) 4 roundedrec(usedlimit2) 4
board19Late min rec max = 0 4 5 roundedrec(used0) 4 roundedrec(usedlimit) 4 roundedrec(usedlimit2) 4"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_3_MOVES_CANADIAN_IN_30S_ENTERED_OVERTIME_15S_USED_1_MOVES_LEFT: &str = r#"===================================================================
Basic 1h byo yomi time controls, 3 moves canadian in 30s, entered overtime 15s used, 1 moves left
===================================================================
board9Early min rec max = 13 13 13 roundedrec(used0) 14 roundedrec(usedlimit) 14 roundedrec(usedlimit2) 14
board9Late min rec max = 13 13 13 roundedrec(used0) 14 roundedrec(usedlimit) 14 roundedrec(usedlimit2) 14
board19Early min rec max = 13 13 13 roundedrec(used0) 14 roundedrec(usedlimit) 14 roundedrec(usedlimit2) 14
board19Late min rec max = 13 13 13 roundedrec(used0) 14 roundedrec(usedlimit) 14 roundedrec(usedlimit2) 14"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_3_MOVES_CANADIAN_IN_30S_ENTERED_OVERTIME_15S_USED_2_MOVES_LEFT: &str = r#"===================================================================
Basic 1h byo yomi time controls, 3 moves canadian in 30s, entered overtime 15s used, 2 moves left
===================================================================
board9Early min rec max = 0 6.5 7.571428571428571 roundedrec(used0) 6.5 roundedrec(usedlimit) 6.5 roundedrec(usedlimit2) 6.5
board9Late min rec max = 0 6.5 7.571428571428571 roundedrec(used0) 6.5 roundedrec(usedlimit) 6.5 roundedrec(usedlimit2) 6.5
board19Early min rec max = 0 6.5 7.571428571428571 roundedrec(used0) 6.5 roundedrec(usedlimit) 6.5 roundedrec(usedlimit2) 6.5
board19Late min rec max = 0 6.5 7.571428571428571 roundedrec(used0) 6.5 roundedrec(usedlimit) 6.5 roundedrec(usedlimit2) 6.5"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_3_MOVES_CANADIAN_IN_30S_JUST_ENTERED_OVERTIME: &str = r#"===================================================================
Basic 1h byo yomi time controls, 3 moves canadian in 30s, just entered overtime
===================================================================
board9Early min rec max = 0 9 11 roundedrec(used0) 9 roundedrec(usedlimit) 9 roundedrec(usedlimit2) 9
board9Late min rec max = 0 9 11 roundedrec(used0) 9 roundedrec(usedlimit) 9 roundedrec(usedlimit2) 9
board19Early min rec max = 0 9 11 roundedrec(used0) 9 roundedrec(usedlimit) 9 roundedrec(usedlimit2) 9
board19Late min rec max = 0 9 11 roundedrec(used0) 9 roundedrec(usedlimit) 9 roundedrec(usedlimit2) 9"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_3_PERIODS_30S_JUST_ENTERED_OVERTIME: &str = r#"===================================================================
Basic 1h byo yomi time controls, 3 periods of 30s, just entered overtime
===================================================================
board9Early min rec max = 29 29 29 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board9Late min rec max = 29 29 29 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board19Early min rec max = 29 29 29 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board19Late min rec max = 29 29 29 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_45S_LEFT_1_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, 45s left, 1 periods of 30s
===================================================================
board9Early min rec max = 0 43 73 roundedrec(used0) 43 roundedrec(usedlimit) 74 roundedrec(usedlimit2) 74
board9Late min rec max = 0 43 73 roundedrec(used0) 43 roundedrec(usedlimit) 74 roundedrec(usedlimit2) 74
board19Early min rec max = 0 43 73 roundedrec(used0) 43 roundedrec(usedlimit) 74 roundedrec(usedlimit2) 74
board19Late min rec max = 0 43 73 roundedrec(used0) 43 roundedrec(usedlimit) 74 roundedrec(usedlimit2) 74"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_45S_LEFT_2_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, 45s left, 2 periods of 30s
===================================================================
board9Early min rec max = 0 44 74 roundedrec(used0) 44 roundedrec(usedlimit) 74 roundedrec(usedlimit2) 74
board9Late min rec max = 0 44 74 roundedrec(used0) 44 roundedrec(usedlimit) 74 roundedrec(usedlimit2) 74
board19Early min rec max = 0 44 74 roundedrec(used0) 44 roundedrec(usedlimit) 74 roundedrec(usedlimit2) 74
board19Late min rec max = 0 44 74 roundedrec(used0) 44 roundedrec(usedlimit) 74 roundedrec(usedlimit2) 74"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_45S_LEFT_2_PERIODS_30S_MAX_TIME_PER_MOVE_57S: &str = r#"===================================================================
Basic 1h byo yomi time controls, 45s left, 2 periods of 30s, max time per move 57s
===================================================================
board9Early min rec max = 0 44 56 roundedrec(used0) 44 roundedrec(usedlimit) 74 roundedrec(usedlimit2) 74
board9Late min rec max = 0 44 56 roundedrec(used0) 44 roundedrec(usedlimit) 74 roundedrec(usedlimit2) 74
board19Early min rec max = 0 44 56 roundedrec(used0) 44 roundedrec(usedlimit) 74 roundedrec(usedlimit2) 74
board19Late min rec max = 0 44 56 roundedrec(used0) 44 roundedrec(usedlimit) 74 roundedrec(usedlimit2) 74"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_45S_LEFT_3_MOVES_CANADIAN_IN_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, 45s left, 3 moves canadian in 30s
===================================================================
board9Early min rec max = 0 14.533039019765972 32 roundedrec(used0) 14.533039019765972 roundedrec(usedlimit) 14.533039019765972 roundedrec(usedlimit2) 14.533039019765972
board9Late min rec max = 0 14.533039019765972 32 roundedrec(used0) 14.533039019765972 roundedrec(usedlimit) 14.533039019765972 roundedrec(usedlimit2) 14.533039019765972
board19Early min rec max = 0 14.533039019765972 32 roundedrec(used0) 14.533039019765972 roundedrec(usedlimit) 14.533039019765972 roundedrec(usedlimit2) 14.533039019765972
board19Late min rec max = 0 14.533039019765972 32 roundedrec(used0) 14.533039019765972 roundedrec(usedlimit) 14.533039019765972 roundedrec(usedlimit2) 14.533039019765972"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_5_PERIODS_30S_ENTERED_OVERTIME_15S_USED: &str = r#"===================================================================
Basic 1h byo yomi time controls, 5 periods of 30s, entered overtime 15s used
===================================================================
board9Early min rec max = 14 14 14 roundedrec(used0) 14 roundedrec(usedlimit) 14 roundedrec(usedlimit2) 14
board9Late min rec max = 14 14 14 roundedrec(used0) 14 roundedrec(usedlimit) 14 roundedrec(usedlimit2) 14
board19Early min rec max = 14 14 14 roundedrec(used0) 14 roundedrec(usedlimit) 14 roundedrec(usedlimit2) 14
board19Late min rec max = 14 14 14 roundedrec(used0) 14 roundedrec(usedlimit) 14 roundedrec(usedlimit2) 14"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_5_PERIODS_30S_JUST_ENTERED_OVERTIME: &str = r#"===================================================================
Basic 1h byo yomi time controls, 5 periods of 30s, just entered overtime
===================================================================
board9Early min rec max = 29 29 29 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board9Late min rec max = 29 29 29 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board19Early min rec max = 29 29 29 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board19Late min rec max = 29 29 29 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_61S_LEFT_1_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, 61s left, 1 periods of 30s
===================================================================
board9Early min rec max = 0 44.59911705929792 89 roundedrec(used0) 44.59911705929792 roundedrec(usedlimit) 44.59911705929792 roundedrec(usedlimit2) 44.59911705929792
board9Late min rec max = 0 44.59911705929792 89 roundedrec(used0) 44.59911705929792 roundedrec(usedlimit) 44.59911705929792 roundedrec(usedlimit2) 44.59911705929792
board19Early min rec max = 0 44.59911705929792 89 roundedrec(used0) 44.59911705929792 roundedrec(usedlimit) 44.59911705929792 roundedrec(usedlimit2) 44.59911705929792
board19Late min rec max = 0 44.59911705929792 89 roundedrec(used0) 44.59911705929792 roundedrec(usedlimit) 44.59911705929792 roundedrec(usedlimit2) 44.59911705929792"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_61S_LEFT_2_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, 61s left, 2 periods of 30s
===================================================================
board9Early min rec max = 0 45.59911705929792 90 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 45.59911705929792 roundedrec(usedlimit2) 45.59911705929792
board9Late min rec max = 0 45.59911705929792 90 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 45.59911705929792 roundedrec(usedlimit2) 45.59911705929792
board19Early min rec max = 0 45.59911705929792 90 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 45.59911705929792 roundedrec(usedlimit2) 45.59911705929792
board19Late min rec max = 0 45.59911705929792 90 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 45.59911705929792 roundedrec(usedlimit2) 45.59911705929792"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_61S_LEFT_3_MOVES_CANADIAN_IN_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, 61s left, 3 moves canadian in 30s
===================================================================
board9Early min rec max = 0 14.533039019765974 32 roundedrec(used0) 14.533039019765974 roundedrec(usedlimit) 14.533039019765974 roundedrec(usedlimit2) 14.533039019765974
board9Late min rec max = 0 17.76923076923077 32 roundedrec(used0) 17.76923076923077 roundedrec(usedlimit) 17.76923076923077 roundedrec(usedlimit2) 17.76923076923077
board19Early min rec max = 0 14.533039019765974 32 roundedrec(used0) 14.533039019765974 roundedrec(usedlimit) 14.533039019765974 roundedrec(usedlimit2) 14.533039019765974
board19Late min rec max = 0 14.533039019765974 32 roundedrec(used0) 14.533039019765974 roundedrec(usedlimit) 14.533039019765974 roundedrec(usedlimit2) 14.533039019765974"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_6_PERIODS_30S_ENTERED_OVERTIME_15S_USED: &str = r#"===================================================================
Basic 1h byo yomi time controls, 6 periods of 30s, entered overtime 15s used
===================================================================
board9Early min rec max = 0 44 44 roundedrec(used0) 44 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44
board9Late min rec max = 0 44 44 roundedrec(used0) 44 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44
board19Early min rec max = 0 44 44 roundedrec(used0) 44 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44
board19Late min rec max = 0 44 44 roundedrec(used0) 44 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_6_PERIODS_30S_JUST_ENTERED_OVERTIME: &str = r#"===================================================================
Basic 1h byo yomi time controls, 6 periods of 30s, just entered overtime
===================================================================
board9Early min rec max = 0 59 59 roundedrec(used0) 59 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59
board9Late min rec max = 0 59 59 roundedrec(used0) 59 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59
board19Early min rec max = 0 59 59 roundedrec(used0) 59 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59
board19Late min rec max = 0 59 59 roundedrec(used0) 59 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_70S_LEFT_1_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, 70s left, 1 periods of 30s
===================================================================
board9Early min rec max = 0 44.59911705929791 98 roundedrec(used0) 44.59911705929791 roundedrec(usedlimit) 44.59911705929791 roundedrec(usedlimit2) 44.59911705929791
board9Late min rec max = 0 44.59911705929791 98 roundedrec(used0) 44.59911705929791 roundedrec(usedlimit) 44.59911705929791 roundedrec(usedlimit2) 44.59911705929791
board19Early min rec max = 0 44.59911705929791 98 roundedrec(used0) 44.59911705929791 roundedrec(usedlimit) 44.59911705929791 roundedrec(usedlimit2) 44.59911705929791
board19Late min rec max = 0 44.59911705929791 98 roundedrec(used0) 44.59911705929791 roundedrec(usedlimit) 44.59911705929791 roundedrec(usedlimit2) 44.59911705929791"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_70S_LEFT_2_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, 70s left, 2 periods of 30s
===================================================================
board9Early min rec max = 0 45.59911705929791 99 roundedrec(used0) 45.59911705929791 roundedrec(usedlimit) 45.59911705929791 roundedrec(usedlimit2) 45.59911705929791
board9Late min rec max = 0 45.59911705929791 99 roundedrec(used0) 45.59911705929791 roundedrec(usedlimit) 45.59911705929791 roundedrec(usedlimit2) 45.59911705929791
board19Early min rec max = 0 45.59911705929791 99 roundedrec(used0) 45.59911705929791 roundedrec(usedlimit) 45.59911705929791 roundedrec(usedlimit2) 45.59911705929791
board19Late min rec max = 0 45.59911705929791 99 roundedrec(used0) 45.59911705929791 roundedrec(usedlimit) 45.59911705929791 roundedrec(usedlimit2) 45.59911705929791"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_70S_LEFT_3_MOVES_CANADIAN_IN_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, 70s left, 3 moves canadian in 30s
===================================================================
board9Early min rec max = 0 14.533039019765972 32 roundedrec(used0) 14.533039019765972 roundedrec(usedlimit) 14.533039019765972 roundedrec(usedlimit2) 14.533039019765972
board9Late min rec max = 0 20.53846153846154 32 roundedrec(used0) 20.53846153846154 roundedrec(usedlimit) 20.53846153846154 roundedrec(usedlimit2) 20.53846153846154
board19Early min rec max = 0 14.533039019765972 32 roundedrec(used0) 14.533039019765972 roundedrec(usedlimit) 14.533039019765972 roundedrec(usedlimit2) 14.533039019765972
board19Late min rec max = 0 14.533039019765972 32 roundedrec(used0) 14.533039019765972 roundedrec(usedlimit) 14.533039019765972 roundedrec(usedlimit2) 14.533039019765972"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_7_PERIODS_30S_ENTERED_OVERTIME_15S_USED: &str = r#"===================================================================
Basic 1h byo yomi time controls, 7 periods of 30s, entered overtime 15s used
===================================================================
board9Early min rec max = 0 44 74 roundedrec(used0) 44 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44
board9Late min rec max = 0 44 74 roundedrec(used0) 44 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44
board19Early min rec max = 0 44 74 roundedrec(used0) 44 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44
board19Late min rec max = 0 44 74 roundedrec(used0) 44 roundedrec(usedlimit) 44 roundedrec(usedlimit2) 44"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_7_PERIODS_30S_JUST_ENTERED_OVERTIME: &str = r#"===================================================================
Basic 1h byo yomi time controls, 7 periods of 30s, just entered overtime
===================================================================
board9Early min rec max = 0 45.59911705929792 89 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59
board9Late min rec max = 0 45.59911705929792 89 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59
board19Early min rec max = 0 45.59911705929792 89 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59
board19Late min rec max = 0 45.59911705929792 89 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_ALL_TIME_LEFT_1_PERIOD_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, all time left, 1 period of 30s
===================================================================
board9Early min rec max = 0 80.54845485377136 749 roundedrec(used0) 80.54845485377136 roundedrec(usedlimit) 80.54845485377136 roundedrec(usedlimit2) 80.54845485377136
board9Late min rec max = 0 134.97733711048159 749 roundedrec(used0) 134.97733711048159 roundedrec(usedlimit) 134.97733711048159 roundedrec(usedlimit2) 134.97733711048159
board19Early min rec max = 0 45.59911705929792 749 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 45.59911705929792 roundedrec(usedlimit2) 45.59911705929792
board19Late min rec max = 0 80.54845485377136 749 roundedrec(used0) 80.54845485377136 roundedrec(usedlimit) 80.54845485377136 roundedrec(usedlimit2) 80.54845485377136"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_ALL_TIME_LEFT_3_MOVES_CANADIAN_IN_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, all time left, 3 moves canadian in 30s
===================================================================
board9Early min rec max = 0 73.26508509541 731 roundedrec(used0) 73.26508509541 roundedrec(usedlimit) 73.26508509541 roundedrec(usedlimit2) 73.26508509541
board9Late min rec max = 0 134.97733711048159 731 roundedrec(used0) 134.97733711048159 roundedrec(usedlimit) 134.97733711048159 roundedrec(usedlimit2) 134.97733711048159
board19Early min rec max = 0 26.182818284590454 731 roundedrec(used0) 26.182818284590454 roundedrec(usedlimit) 26.182818284590454 roundedrec(usedlimit2) 26.182818284590454
board19Late min rec max = 0 38.14107094319109 731 roundedrec(used0) 38.14107094319109 roundedrec(usedlimit) 38.14107094319109 roundedrec(usedlimit2) 38.14107094319109"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_ALL_TIME_LEFT_3_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, all time left, 3 periods of 30s
===================================================================
board9Early min rec max = 0 80.54845485377136 749 roundedrec(used0) 80.54845485377136 roundedrec(usedlimit) 80.54845485377136 roundedrec(usedlimit2) 80.54845485377136
board9Late min rec max = 0 134.97733711048159 749 roundedrec(used0) 134.97733711048159 roundedrec(usedlimit) 134.97733711048159 roundedrec(usedlimit2) 134.97733711048159
board19Early min rec max = 0 45.59911705929792 749 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 45.59911705929792 roundedrec(usedlimit2) 45.59911705929792
board19Late min rec max = 0 80.54845485377136 749 roundedrec(used0) 80.54845485377136 roundedrec(usedlimit) 80.54845485377136 roundedrec(usedlimit2) 80.54845485377136"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_ALL_TIME_LEFT_5_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, all time left, 5 periods of 30s
===================================================================
board9Early min rec max = 0 80.54845485377136 749 roundedrec(used0) 80.54845485377136 roundedrec(usedlimit) 80.54845485377136 roundedrec(usedlimit2) 80.54845485377136
board9Late min rec max = 0 134.97733711048159 749 roundedrec(used0) 134.97733711048159 roundedrec(usedlimit) 134.97733711048159 roundedrec(usedlimit2) 134.97733711048159
board19Early min rec max = 0 45.59911705929792 749 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 45.59911705929792 roundedrec(usedlimit2) 45.59911705929792
board19Late min rec max = 0 80.54845485377136 749 roundedrec(used0) 80.54845485377136 roundedrec(usedlimit) 80.54845485377136 roundedrec(usedlimit2) 80.54845485377136"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_ALL_TIME_LEFT_6_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, all time left, 6 periods of 30s
===================================================================
board9Early min rec max = 0 80.54845485377135 755 roundedrec(used0) 80.54845485377135 roundedrec(usedlimit) 80.54845485377135 roundedrec(usedlimit2) 80.54845485377135
board9Late min rec max = 0 136.11048158640227 755 roundedrec(used0) 136.11048158640227 roundedrec(usedlimit) 136.11048158640227 roundedrec(usedlimit2) 136.11048158640227
board19Early min rec max = 0 45.59911705929791 755 roundedrec(used0) 45.59911705929791 roundedrec(usedlimit) 45.59911705929791 roundedrec(usedlimit2) 45.59911705929791
board19Late min rec max = 0 80.54845485377135 755 roundedrec(used0) 80.54845485377135 roundedrec(usedlimit) 80.54845485377135 roundedrec(usedlimit2) 80.54845485377135"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_ALL_TIME_LEFT_7_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, all time left, 7 periods of 30s
===================================================================
board9Early min rec max = 0 80.54845485377136 761 roundedrec(used0) 80.54845485377136 roundedrec(usedlimit) 80.54845485377136 roundedrec(usedlimit2) 80.54845485377136
board9Late min rec max = 0 137.24362606232293 761 roundedrec(used0) 137.24362606232293 roundedrec(usedlimit) 137.24362606232293 roundedrec(usedlimit2) 137.24362606232293
board19Early min rec max = 0 45.59911705929792 761 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 45.59911705929792 roundedrec(usedlimit2) 45.59911705929792
board19Late min rec max = 0 80.54845485377136 761 roundedrec(used0) 80.54845485377136 roundedrec(usedlimit) 80.54845485377136 roundedrec(usedlimit2) 80.54845485377136"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_NO_TIME_LEFT_1_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, no time left, 1 periods of 30s
===================================================================
board9Early min rec max = 0 28 28 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board9Late min rec max = 0 28 28 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board19Early min rec max = 0 28 28 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board19Late min rec max = 0 28 28 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_NO_TIME_LEFT_3_MOVES_CANADIAN_IN_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, no time left, 3 moves canadian in 30s
===================================================================
board9Early min rec max = 0 9 11 roundedrec(used0) 9 roundedrec(usedlimit) 9 roundedrec(usedlimit2) 9
board9Late min rec max = 0 9 11 roundedrec(used0) 9 roundedrec(usedlimit) 9 roundedrec(usedlimit2) 9
board19Early min rec max = 0 9 11 roundedrec(used0) 9 roundedrec(usedlimit) 9 roundedrec(usedlimit2) 9
board19Late min rec max = 0 9 11 roundedrec(used0) 9 roundedrec(usedlimit) 9 roundedrec(usedlimit2) 9"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_NO_TIME_LEFT_5_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, no time left, 5 periods of 30s
===================================================================
board9Early min rec max = 0 29 29 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board9Late min rec max = 0 29 29 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board19Early min rec max = 0 29 29 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board19Late min rec max = 0 29 29 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_NO_TIME_LEFT_6_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, no time left, 6 periods of 30s
===================================================================
board9Early min rec max = 0 59 59 roundedrec(used0) 59 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59
board9Late min rec max = 0 59 59 roundedrec(used0) 59 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59
board19Early min rec max = 0 59 59 roundedrec(used0) 59 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59
board19Late min rec max = 0 59 59 roundedrec(used0) 59 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59"#;

const EXPECTED_TEST_BASIC_1H_BYO_YOMI_NO_TIME_LEFT_7_PERIODS_30S: &str = r#"===================================================================
Basic 1h byo yomi time controls, no time left, 7 periods of 30s
===================================================================
board9Early min rec max = 0 45.59911705929792 89 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59
board9Late min rec max = 0 45.59911705929792 89 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59
board19Early min rec max = 0 45.59911705929792 89 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59
board19Late min rec max = 0 45.59911705929792 89 roundedrec(used0) 45.59911705929792 roundedrec(usedlimit) 59 roundedrec(usedlimit2) 59"#;

const EXPECTED_TEST_BASIC_1H_FISCHER_10M_LEFT_10S_INCREMENT: &str = r#"===================================================================
Basic 1h fischer time controls, 10m left, 10s increment
===================================================================
board9Early min rec max = 0 24.551155115511552 126.8 roundedrec(used0) 24.551155115511552 roundedrec(usedlimit) 24.551155115511552 roundedrec(usedlimit2) 24.551155115511552
board9Late min rec max = 0 46.10236220472441 126.8 roundedrec(used0) 46.10236220472441 roundedrec(usedlimit) 46.10236220472441 roundedrec(usedlimit2) 46.10236220472441
board19Early min rec max = 0 13.122484689413824 126.8 roundedrec(used0) 13.122484689413824 roundedrec(usedlimit) 13.122484689413824 roundedrec(usedlimit2) 13.122484689413824
board19Late min rec max = 0 20.03512880562061 126.8 roundedrec(used0) 20.03512880562061 roundedrec(usedlimit) 20.03512880562061 roundedrec(usedlimit2) 20.03512880562061"#;

const EXPECTED_TEST_BASIC_1H_FISCHER_10M_LEFT_10S_INCREMENT_LARGER_LAG_BUFFER: &str = r#"===================================================================
Basic 1h fischer time controls, 10m left, 10s increment, larger lag buffer
===================================================================
board9Early min rec max = 0 20.445544554455445 122 roundedrec(used0) 20.445544554455445 roundedrec(usedlimit) 20.445544554455445 roundedrec(usedlimit2) 20.445544554455445
board9Late min rec max = 0 41.8503937007874 122 roundedrec(used0) 41.8503937007874 roundedrec(usedlimit) 41.8503937007874 roundedrec(usedlimit2) 41.8503937007874
board19Early min rec max = 0 9.094488188976378 122 roundedrec(used0) 9.094488188976378 roundedrec(usedlimit) 9.094488188976378 roundedrec(usedlimit2) 9.094488188976378
board19Late min rec max = 0 15.960187353629976 122 roundedrec(used0) 15.960187353629976 roundedrec(usedlimit) 15.960187353629976 roundedrec(usedlimit2) 15.960187353629976"#;

const EXPECTED_TEST_BASIC_1H_FISCHER_10M_LEFT_10S_INCREMENT_MAX_TIME_PER_MOVE_20S: &str = r#"===================================================================
Basic 1h fischer time controls, 10m left, 10s increment, max time per move 20s
===================================================================
board9Early min rec max = 0 19 19 roundedrec(used0) 19 roundedrec(usedlimit) 19 roundedrec(usedlimit2) 19
board9Late min rec max = 0 19 19 roundedrec(used0) 19 roundedrec(usedlimit) 19 roundedrec(usedlimit2) 19
board19Early min rec max = 0 13.122484689413824 19 roundedrec(used0) 13.122484689413824 roundedrec(usedlimit) 13.122484689413824 roundedrec(usedlimit2) 13.122484689413824
board19Late min rec max = 0 19 19 roundedrec(used0) 19 roundedrec(usedlimit) 19 roundedrec(usedlimit2) 19"#;

const EXPECTED_TEST_BASIC_1H_FISCHER_10M_LEFT_10S_INCREMENT_MAX_TIME_PER_MOVE_30S: &str = r#"===================================================================
Basic 1h fischer time controls, 10m left, 10s increment, max time per move 30s
===================================================================
board9Early min rec max = 0 24.551155115511552 29 roundedrec(used0) 24.551155115511552 roundedrec(usedlimit) 24.551155115511552 roundedrec(usedlimit2) 24.551155115511552
board9Late min rec max = 0 29 29 roundedrec(used0) 29 roundedrec(usedlimit) 29 roundedrec(usedlimit2) 29
board19Early min rec max = 0 13.122484689413824 29 roundedrec(used0) 13.122484689413824 roundedrec(usedlimit) 13.122484689413824 roundedrec(usedlimit2) 13.122484689413824
board19Late min rec max = 0 20.03512880562061 29 roundedrec(used0) 20.03512880562061 roundedrec(usedlimit) 20.03512880562061 roundedrec(usedlimit2) 20.03512880562061"#;

const EXPECTED_TEST_BASIC_1H_FISCHER_15S_LEFT_10S_INCREMENT: &str = r#"===================================================================
Basic 1h fischer time controls, 15s left, 10s increment
===================================================================
board9Early min rec max = 0 9.105610561056105 9.8 roundedrec(used0) 9.105610561056105 roundedrec(usedlimit) 9.105610561056105 roundedrec(usedlimit2) 9.105610561056105
board9Late min rec max = 0 9.251968503937007 9.8 roundedrec(used0) 9.251968503937007 roundedrec(usedlimit) 9.251968503937007 roundedrec(usedlimit2) 9.251968503937007
board19Early min rec max = 0 9.027996500437446 9.8 roundedrec(used0) 9.027996500437446 roundedrec(usedlimit) 9.027996500437446 roundedrec(usedlimit2) 9.027996500437446
board19Late min rec max = 0 9.074941451990632 9.8 roundedrec(used0) 9.074941451990632 roundedrec(usedlimit) 9.074941451990632 roundedrec(usedlimit2) 9.074941451990632"#;

const EXPECTED_TEST_BASIC_1H_FISCHER_15S_LEFT_10S_INCREMENT_MAX_TIME_PER_MOVE_11S: &str = r#"===================================================================
Basic 1h fischer time controls, 15s left, 10s increment, max time per move 11s
===================================================================
board9Early min rec max = 0 9.105610561056105 9.8 roundedrec(used0) 9.105610561056105 roundedrec(usedlimit) 9.105610561056105 roundedrec(usedlimit2) 9.105610561056105
board9Late min rec max = 0 9.251968503937007 9.8 roundedrec(used0) 9.251968503937007 roundedrec(usedlimit) 9.251968503937007 roundedrec(usedlimit2) 9.251968503937007
board19Early min rec max = 0 9.027996500437446 9.8 roundedrec(used0) 9.027996500437446 roundedrec(usedlimit) 9.027996500437446 roundedrec(usedlimit2) 9.027996500437446
board19Late min rec max = 0 9.074941451990632 9.8 roundedrec(used0) 9.074941451990632 roundedrec(usedlimit) 9.074941451990632 roundedrec(usedlimit2) 9.074941451990632"#;

const EXPECTED_TEST_BASIC_1H_FISCHER_15S_LEFT_10S_INCREMENT_MAX_TIME_PER_MOVE_7S: &str = r#"===================================================================
Basic 1h fischer time controls, 15s left, 10s increment, max time per move 7s
===================================================================
board9Early min rec max = 0 6 6 roundedrec(used0) 6 roundedrec(usedlimit) 6 roundedrec(usedlimit2) 6
board9Late min rec max = 0 6 6 roundedrec(used0) 6 roundedrec(usedlimit) 6 roundedrec(usedlimit2) 6
board19Early min rec max = 0 6 6 roundedrec(used0) 6 roundedrec(usedlimit) 6 roundedrec(usedlimit2) 6
board19Late min rec max = 0 6 6 roundedrec(used0) 6 roundedrec(usedlimit) 6 roundedrec(usedlimit2) 6"#;

const EXPECTED_TEST_BASIC_1H_FISCHER_5S_LEFT_10S_INCREMENT: &str = r#"===================================================================
Basic 1h fischer time controls, 5s left, 10s increment
===================================================================
board9Early min rec max = 0 3 4 roundedrec(used0) 3 roundedrec(usedlimit) 3 roundedrec(usedlimit2) 3
board9Late min rec max = 0 3 4 roundedrec(used0) 3 roundedrec(usedlimit) 3 roundedrec(usedlimit2) 3
board19Early min rec max = 0 3 4 roundedrec(used0) 3 roundedrec(usedlimit) 3 roundedrec(usedlimit2) 3
board19Late min rec max = 0 3 4 roundedrec(used0) 3 roundedrec(usedlimit) 3 roundedrec(usedlimit2) 3"#;

const EXPECTED_TEST_BASIC_1H_FISCHER_MINUS1S_LEFT_10S_INCREMENT: &str = r#"===================================================================
Basic 1h fischer time controls, -1s left, 10s increment
===================================================================
board9Early min rec max = 0 0 0 roundedrec(used0) 0 roundedrec(usedlimit) 0 roundedrec(usedlimit2) 0
board9Late min rec max = 0 0 0 roundedrec(used0) 0 roundedrec(usedlimit) 0 roundedrec(usedlimit2) 0
board19Early min rec max = 0 0 0 roundedrec(used0) 0 roundedrec(usedlimit) 0 roundedrec(usedlimit2) 0
board19Late min rec max = 0 0 0 roundedrec(used0) 0 roundedrec(usedlimit) 0 roundedrec(usedlimit2) 0"#;

const EXPECTED_TEST_BASIC_5S_FISCHER_5S_LEFT_10S_INCREMENT_6S_MAIN_TIME_LIMIT: &str = r#"===================================================================
Basic 5s fischer time controls, 5s left, 10s increment, 6s main time limit
===================================================================
board9Early min rec max = 1.5 3 4 roundedrec(used0) 3 roundedrec(usedlimit) 3 roundedrec(usedlimit2) 3
board9Late min rec max = 1.5 3 4 roundedrec(used0) 3 roundedrec(usedlimit) 3 roundedrec(usedlimit2) 3
board19Early min rec max = 1.5 3 4 roundedrec(used0) 3 roundedrec(usedlimit) 3 roundedrec(usedlimit2) 3
board19Late min rec max = 1.5 3 4 roundedrec(used0) 3 roundedrec(usedlimit) 3 roundedrec(usedlimit2) 3"#;

const EXPECTED_TEST_UNLIMITED_TIME_CONTROLS: &str = r#"===================================================================
Unlimited time controls
===================================================================
board9Early min rec max = 0 20629190304280557000000000000 200000000000000020000000000000 roundedrec(used0) 20629190304280557000000000000 roundedrec(usedlimit) 20629190304280557000000000000 roundedrec(usedlimit2) 20629190304280557000000000000
board9Late min rec max = 0 37771482530689330000000000000 200000000000000020000000000000 roundedrec(used0) 37771482530689330000000000000 roundedrec(usedlimit) 37771482530689330000000000000 roundedrec(usedlimit2) 37771482530689330000000000000
board19Early min rec max = 0 5510400881664141000000000000 200000000000000020000000000000 roundedrec(used0) 5510400881664141000000000000 roundedrec(usedlimit) 5510400881664141000000000000 roundedrec(usedlimit2) 5510400881664141000000000000
board19Late min rec max = 0 10872519706441970000000000000 200000000000000020000000000000 roundedrec(used0) 10872519706441970000000000000 roundedrec(usedlimit) 10872519706441970000000000000 roundedrec(usedlimit2) 10872519706441970000000000000"#;

fn assert_output_matches(actual: &str, expected: &str) {
    const EPS: f64 = 1e-9;
    let actual_lines: Vec<&str> = actual.lines().collect();
    let expected_lines: Vec<&str> = expected.lines().collect();
    assert_eq!(
        actual_lines.len(),
        expected_lines.len(),
        "line count mismatch\nactual:\n{}\nexpected:\n{}",
        actual,
        expected
    );
    for (i, (actual_line, expected_line)) in
        actual_lines.iter().zip(expected_lines.iter()).enumerate()
    {
        let actual_tokens: Vec<&str> = actual_line.split_whitespace().collect();
        let expected_tokens: Vec<&str> = expected_line.split_whitespace().collect();
        assert_eq!(
            actual_tokens.len(),
            expected_tokens.len(),
            "token count mismatch on line {}\nactual: {}\nexpected: {}",
            i,
            actual_line,
            expected_line
        );
        for (a, e) in actual_tokens.iter().zip(expected_tokens.iter()) {
            if let (Ok(av), Ok(ev)) = (a.parse::<f64>(), e.parse::<f64>()) {
                assert!(
                    (av - ev).abs() < EPS,
                    "float mismatch on line {}: actual {} vs expected {} (diff {})",
                    i,
                    av,
                    ev,
                    (av - ev).abs()
                );
            } else {
                assert_eq!(*a, *e, "token mismatch on line {}", i);
            }
        }
    }
}

#[test]
fn test_unlimited_time_controls() {
    let tc = TimeControls::new();
    let actual = run_scenario("Unlimited time controls", &tc, 0.0);
    assert_output_matches(&actual, EXPECTED_TEST_UNLIMITED_TIME_CONTROLS);
}

#[test]
fn test_basic_1h_absolute_all_time_left() {
    let tc = absolute_time_controls(3600.0, 3600.0, 0.0);
    let actual = run_scenario("Basic 1h absolute time controls, all time left", &tc, 1.0);
    assert_output_matches(&actual, EXPECTED_TEST_BASIC_1H_ABSOLUTE_ALL_TIME_LEFT);
}

#[test]
fn test_basic_1h_absolute_10m_left() {
    let tc = absolute_time_controls(3600.0, 600.0, 0.0);
    let actual = run_scenario("Basic 1h absolute time controls, 10m left", &tc, 1.0);
    assert_output_matches(&actual, EXPECTED_TEST_BASIC_1H_ABSOLUTE_10M_LEFT);
}

#[test]
fn test_basic_1h_fischer_10m_left_10s_increment() {
    let tc = absolute_time_controls(3600.0, 600.0, 10.0);
    let actual = run_scenario(
        "Basic 1h fischer time controls, 10m left, 10s increment",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_FISCHER_10M_LEFT_10S_INCREMENT,
    );
}

#[test]
fn test_basic_1h_fischer_10m_left_10s_increment_larger_lag_buffer() {
    let tc = absolute_time_controls(3600.0, 600.0, 10.0);
    let actual = run_scenario(
        "Basic 1h fischer time controls, 10m left, 10s increment, larger lag buffer",
        &tc,
        5.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_FISCHER_10M_LEFT_10S_INCREMENT_LARGER_LAG_BUFFER,
    );
}

#[test]
fn test_basic_10m_fischer_10m_left_10s_increment_main_time_limit_10m() {
    let mut tc = absolute_time_controls(600.0, 600.0, 10.0);
    tc.main_time_limit = 600.0;
    let actual = run_scenario(
        "Basic 10m fischer time controls, 10m left, 10s increment, main time limit 10m",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_10M_FISCHER_10M_LEFT_10S_INCREMENT_MAIN_TIME_LIMIT_10M,
    );
}

#[test]
fn test_basic_10m_fischer_10m_left_10s_increment_main_time_limit_10m_5s() {
    let mut tc = absolute_time_controls(600.0, 600.0, 10.0);
    tc.main_time_limit = 605.0;
    let actual = run_scenario(
        "Basic 10m fischer time controls, 10m left, 10s increment, main time limit 10m+5s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_10M_FISCHER_10M_LEFT_10S_INCREMENT_MAIN_TIME_LIMIT_10M_5S,
    );
}

#[test]
fn test_basic_1h_fischer_10m_left_10s_increment_max_time_per_move_20s() {
    let mut tc = absolute_time_controls(3600.0, 600.0, 10.0);
    tc.max_time_per_move = 20.0;
    let actual = run_scenario(
        "Basic 1h fischer time controls, 10m left, 10s increment, max time per move 20s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_FISCHER_10M_LEFT_10S_INCREMENT_MAX_TIME_PER_MOVE_20S,
    );
}

#[test]
fn test_basic_1h_fischer_10m_left_10s_increment_max_time_per_move_30s() {
    let mut tc = absolute_time_controls(3600.0, 600.0, 10.0);
    tc.max_time_per_move = 30.0;
    let actual = run_scenario(
        "Basic 1h fischer time controls, 10m left, 10s increment, max time per move 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_FISCHER_10M_LEFT_10S_INCREMENT_MAX_TIME_PER_MOVE_30S,
    );
}

#[test]
fn test_basic_1h_fischer_15s_left_10s_increment() {
    let tc = absolute_time_controls(3600.0, 15.0, 10.0);
    let actual = run_scenario(
        "Basic 1h fischer time controls, 15s left, 10s increment",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_FISCHER_15S_LEFT_10S_INCREMENT,
    );
}

#[test]
fn test_basic_1h_fischer_15s_left_10s_increment_max_time_per_move_11s() {
    let mut tc = absolute_time_controls(3600.0, 15.0, 10.0);
    tc.max_time_per_move = 11.0;
    let actual = run_scenario(
        "Basic 1h fischer time controls, 15s left, 10s increment, max time per move 11s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_FISCHER_15S_LEFT_10S_INCREMENT_MAX_TIME_PER_MOVE_11S,
    );
}

#[test]
fn test_basic_1h_fischer_15s_left_10s_increment_max_time_per_move_7s() {
    let mut tc = absolute_time_controls(3600.0, 15.0, 10.0);
    tc.max_time_per_move = 7.0;
    let actual = run_scenario(
        "Basic 1h fischer time controls, 15s left, 10s increment, max time per move 7s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_FISCHER_15S_LEFT_10S_INCREMENT_MAX_TIME_PER_MOVE_7S,
    );
}

#[test]
fn test_basic_1h_fischer_5s_left_10s_increment() {
    let tc = absolute_time_controls(3600.0, 5.0, 10.0);
    let actual = run_scenario(
        "Basic 1h fischer time controls, 5s left, 10s increment",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_FISCHER_5S_LEFT_10S_INCREMENT,
    );
}

#[test]
fn test_basic_5s_fischer_5s_left_10s_increment_6s_main_time_limit() {
    let mut tc = absolute_time_controls(5.0, 5.0, 10.0);
    tc.main_time_limit = 6.0;
    let actual = run_scenario(
        "Basic 5s fischer time controls, 5s left, 10s increment, 6s main time limit",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_5S_FISCHER_5S_LEFT_10S_INCREMENT_6S_MAIN_TIME_LIMIT,
    );
}

#[test]
fn test_basic_1h_fischer_minus1s_left_10s_increment() {
    let tc = absolute_time_controls(3600.0, -1.0, 10.0);
    let actual = run_scenario(
        "Basic 1h fischer time controls, -1s left, 10s increment",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_FISCHER_MINUS1S_LEFT_10S_INCREMENT,
    );
}

#[test]
fn test_basic_10s_fischer_minus1s_left_10s_increment_10s_main_time_limit() {
    let mut tc = absolute_time_controls(10.0, -1.0, 10.0);
    tc.main_time_limit = 10.0;
    let actual = run_scenario(
        "Basic 10s fischer time controls, -1s left, 10s increment, 10s main time limit",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_10S_FISCHER_MINUS1S_LEFT_10S_INCREMENT_10S_MAIN_TIME_LIMIT,
    );
}

#[test]
fn test_basic_1h_byo_yomi_all_time_left_1_period_30s() {
    let tc = byo_yomi_time_controls(3600.0, 3600.0, 30.0, 1, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, all time left, 1 period of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_ALL_TIME_LEFT_1_PERIOD_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_all_time_left_3_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 3600.0, 30.0, 3, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, all time left, 3 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_ALL_TIME_LEFT_3_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_all_time_left_5_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 3600.0, 30.0, 5, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, all time left, 5 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_ALL_TIME_LEFT_5_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_all_time_left_6_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 3600.0, 30.0, 6, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, all time left, 6 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_ALL_TIME_LEFT_6_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_all_time_left_7_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 3600.0, 30.0, 7, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, all time left, 7 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_ALL_TIME_LEFT_7_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_all_time_left_3_moves_canadian_in_30s() {
    let tc = byo_yomi_time_controls(3600.0, 3600.0, 30.0, 1, 3, false, 0.0, 3);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, all time left, 3 moves canadian in 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_ALL_TIME_LEFT_3_MOVES_CANADIAN_IN_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_no_time_left_1_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 1, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, no time left, 1 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_NO_TIME_LEFT_1_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_no_time_left_5_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 5, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, no time left, 5 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_NO_TIME_LEFT_5_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_no_time_left_6_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 6, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, no time left, 6 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_NO_TIME_LEFT_6_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_no_time_left_7_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 7, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, no time left, 7 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_NO_TIME_LEFT_7_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_no_time_left_3_moves_canadian_in_30s() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 1, 3, false, 0.0, 3);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, no time left, 3 moves canadian in 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_NO_TIME_LEFT_3_MOVES_CANADIAN_IN_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_15s_left_1_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 15.0, 30.0, 1, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 15s left, 1 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_15S_LEFT_1_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_15s_left_2_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 15.0, 30.0, 2, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 15s left, 2 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_15S_LEFT_2_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_15s_left_3_moves_canadian_in_30s() {
    let tc = byo_yomi_time_controls(3600.0, 15.0, 30.0, 1, 3, false, 0.0, 3);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 15s left, 3 moves canadian in 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_15S_LEFT_3_MOVES_CANADIAN_IN_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_45s_left_1_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 45.0, 30.0, 1, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 45s left, 1 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_45S_LEFT_1_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_45s_left_2_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 45.0, 30.0, 2, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 45s left, 2 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_45S_LEFT_2_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_45s_left_2_periods_30s_max_time_per_move_57s() {
    let mut tc = byo_yomi_time_controls(3600.0, 45.0, 30.0, 2, 1, false, 0.0, 1);
    tc.max_time_per_move = 57.0;
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 45s left, 2 periods of 30s, max time per move 57s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_45S_LEFT_2_PERIODS_30S_MAX_TIME_PER_MOVE_57S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_45s_left_3_moves_canadian_in_30s() {
    let tc = byo_yomi_time_controls(3600.0, 45.0, 30.0, 1, 3, false, 0.0, 3);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 45s left, 3 moves canadian in 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_45S_LEFT_3_MOVES_CANADIAN_IN_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_61s_left_1_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 61.0, 30.0, 1, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 61s left, 1 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_61S_LEFT_1_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_61s_left_2_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 61.0, 30.0, 2, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 61s left, 2 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_61S_LEFT_2_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_61s_left_3_moves_canadian_in_30s() {
    let tc = byo_yomi_time_controls(3600.0, 61.0, 30.0, 1, 3, false, 0.0, 3);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 61s left, 3 moves canadian in 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_61S_LEFT_3_MOVES_CANADIAN_IN_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_70s_left_1_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 70.0, 30.0, 1, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 70s left, 1 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_70S_LEFT_1_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_70s_left_2_periods_30s() {
    let tc = byo_yomi_time_controls(3600.0, 70.0, 30.0, 2, 1, false, 0.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 70s left, 2 periods of 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_70S_LEFT_2_PERIODS_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_70s_left_3_moves_canadian_in_30s() {
    let tc = byo_yomi_time_controls(3600.0, 70.0, 30.0, 1, 3, false, 0.0, 3);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 70s left, 3 moves canadian in 30s",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_70S_LEFT_3_MOVES_CANADIAN_IN_30S,
    );
}

#[test]
fn test_basic_1h_byo_yomi_1_periods_30s_just_entered_overtime() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 1, 1, true, 30.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 1 periods of 30s, just entered overtime",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_1_PERIODS_30S_JUST_ENTERED_OVERTIME,
    );
}

#[test]
fn test_basic_1h_byo_yomi_3_periods_30s_just_entered_overtime() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 3, 1, true, 30.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 3 periods of 30s, just entered overtime",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_3_PERIODS_30S_JUST_ENTERED_OVERTIME,
    );
}

#[test]
fn test_basic_1h_byo_yomi_5_periods_30s_just_entered_overtime() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 5, 1, true, 30.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 5 periods of 30s, just entered overtime",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_5_PERIODS_30S_JUST_ENTERED_OVERTIME,
    );
}

#[test]
fn test_basic_1h_byo_yomi_6_periods_30s_just_entered_overtime() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 6, 1, true, 30.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 6 periods of 30s, just entered overtime",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_6_PERIODS_30S_JUST_ENTERED_OVERTIME,
    );
}

#[test]
fn test_basic_1h_byo_yomi_7_periods_30s_just_entered_overtime() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 7, 1, true, 30.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 7 periods of 30s, just entered overtime",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_7_PERIODS_30S_JUST_ENTERED_OVERTIME,
    );
}

#[test]
fn test_basic_1h_byo_yomi_3_moves_canadian_in_30s_just_entered_overtime() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 1, 3, true, 30.0, 3);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 3 moves canadian in 30s, just entered overtime",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_3_MOVES_CANADIAN_IN_30S_JUST_ENTERED_OVERTIME,
    );
}

#[test]
fn test_basic_1h_byo_yomi_1_periods_30s_entered_overtime_15s_used() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 1, 1, true, 15.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 1 periods of 30s, entered overtime 15s used",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_1_PERIODS_30S_ENTERED_OVERTIME_15S_USED,
    );
}

#[test]
fn test_basic_1h_byo_yomi_5_periods_30s_entered_overtime_15s_used() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 5, 1, true, 15.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 5 periods of 30s, entered overtime 15s used",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_5_PERIODS_30S_ENTERED_OVERTIME_15S_USED,
    );
}

#[test]
fn test_basic_1h_byo_yomi_6_periods_30s_entered_overtime_15s_used() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 6, 1, true, 15.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 6 periods of 30s, entered overtime 15s used",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_6_PERIODS_30S_ENTERED_OVERTIME_15S_USED,
    );
}

#[test]
fn test_basic_1h_byo_yomi_7_periods_30s_entered_overtime_15s_used() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 7, 1, true, 15.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 7 periods of 30s, entered overtime 15s used",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_7_PERIODS_30S_ENTERED_OVERTIME_15S_USED,
    );
}

#[test]
fn test_basic_1h_byo_yomi_3_moves_canadian_in_30s_entered_overtime_15s_used() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 1, 3, true, 15.0, 3);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 3 moves canadian in 30s, entered overtime 15s used",
        &tc,
        1.0,
    );
    assert_output_matches(
        &actual,
        EXPECTED_TEST_BASIC_1H_BYO_YOMI_3_MOVES_CANADIAN_IN_30S_ENTERED_OVERTIME_15S_USED,
    );
}

#[test]
fn test_basic_1h_byo_yomi_3_moves_canadian_in_30s_entered_overtime_15s_used_2_moves_left() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 1, 3, true, 15.0, 2);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 3 moves canadian in 30s, entered overtime 15s used, 2 moves left",
        &tc,
        1.0,
    );
    assert_output_matches(&actual, EXPECTED_TEST_BASIC_1H_BYO_YOMI_3_MOVES_CANADIAN_IN_30S_ENTERED_OVERTIME_15S_USED_2_MOVES_LEFT);
}

#[test]
fn test_basic_1h_byo_yomi_3_moves_canadian_in_30s_entered_overtime_15s_used_1_moves_left() {
    let tc = byo_yomi_time_controls(3600.0, 0.0, 30.0, 1, 3, true, 15.0, 1);
    let actual = run_scenario(
        "Basic 1h byo yomi time controls, 3 moves canadian in 30s, entered overtime 15s used, 1 moves left",
        &tc,
        1.0,
    );
    assert_output_matches(&actual, EXPECTED_TEST_BASIC_1H_BYO_YOMI_3_MOVES_CANADIAN_IN_30S_ENTERED_OVERTIME_15S_USED_1_MOVES_LEFT);
}

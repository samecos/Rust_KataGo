//! Throwaway probe: dump the engine's v7 features for the empty 19x19 board
//! with the EXACT GTP genmove setup (chinese rules, komi 7.5).
use kata_core::logger::{Logger, LoggerOptions};
use kata_game::board::{Board, P_BLACK};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_nn::inputs::{MiscNNInputParams, fill_row_v7};

#[test]
fn dump_empty_board_features() {
    let board = Board::new(19, 19);
    let rules = Rules::try_parse_rules_without_komi("chinese", 7.5).expect("chinese");
    let hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
    let mut params = MiscNNInputParams::default();
    params.conservative_pass_and_is_root = true;
    params.enable_passing_hacks = true;
    let mut spatial = vec![0.0f32; 22 * 19 * 19];
    let mut global = vec![0.0f32; 19];
    fill_row_v7(&board, &hist, P_BLACK, &params, 19, 19, false, &mut spatial, &mut global);
    println!("SPATIAL_START");
    for v in &spatial {
        print!("{:.4} ", v);
    }
    println!();
    println!("GLOBAL_START");
    for v in &global {
        print!("{:.4} ", v);
    }
    println!();
    println!("DONE");
}

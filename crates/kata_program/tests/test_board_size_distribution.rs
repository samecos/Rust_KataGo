//! Port of the "Board size distribution" block from
//! `KataGo/cpp/tests/testsearchnonn.cpp`.
//!
//! This exercises `GameInitializer`'s randomized board-size selection,
//! including rectangular boards and komi auto-adjustment.

use std::collections::HashMap;

use kata_core::config::ConfigParser;
use kata_core::logger::{Logger, LoggerOptions};
use kata_game::board::Board;
use kata_game::history::BoardHistory;
use kata_program::play::{GameInitializer, OtherGameProperties};
use kata_program::play_settings::PlaySettings;
use kata_program::play_utils::ExtraBlackAndKomi;

#[test]
fn board_size_distribution() {
    let logger = Logger::new(LoggerOptions::default(), None);
    let mut cfg = ConfigParser::new(false, false);
    cfg.override_key("koRules", "SIMPLE");
    cfg.override_key("scoringRules", "AREA");
    cfg.override_key("taxRules", "SEKI");
    cfg.override_key("multiStoneSuicideLegals", "false");
    cfg.override_key("hasButtons", "false");
    cfg.override_key("bSizes", "2,4,6,8");
    cfg.override_key("bSizeRelProbs", "1,2,3,4");
    cfg.override_key("allowRectangleProb", "0.3");
    cfg.override_key("komiAuto", "true");

    let game_init =
        GameInitializer::new(&cfg, &logger).expect("GameInitializer should parse config");

    let mut counts: HashMap<(i32, i32), usize> = HashMap::new();
    const SAMPLES: usize = 100_000;

    for _ in 0..SAMPLES {
        let mut board = Board::default();
        let mut pla = 0;
        let mut hist = BoardHistory::default();
        let mut extra_black_and_komi = ExtraBlackAndKomi::default();
        let mut other_props = OtherGameProperties::default();
        game_init.create_game(
            &mut board,
            &mut pla,
            &mut hist,
            &mut extra_black_and_komi,
            None,
            &PlaySettings::default(),
            &mut other_props,
            None,
        );
        *counts.entry((board.x_size, board.y_size)).or_insert(0) += 1;
    }

    let mut total = 0;
    let mut rectangular_count = 0;
    for x in (2..=8).step_by(2) {
        for y in (2..=8).step_by(2) {
            let count = counts.get(&(x, y)).copied().unwrap_or(0);
            total += count;
            if x != y {
                rectangular_count += count;
            }
            assert!(
                count > 0,
                "board size {}x{} should appear at least once in {} samples",
                x,
                y,
                SAMPLES
            );
        }
    }

    assert_eq!(total, SAMPLES, "all samples should be accounted for");
    assert!(
        rectangular_count > 0,
        "rectangular boards should appear when allowRectangleProb > 0"
    );
}

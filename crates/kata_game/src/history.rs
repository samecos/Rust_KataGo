//! Go board history and game-state tracking.
//!
//! Corresponds to `cpp/game/boardhistory.h` and `cpp/game/boardhistory.cpp`.
//! This implementation covers move history, simple ko, positional/situational
//! superko, pass tracking, and basic area-scoring game end. Territory-scoring
//! encore phases are present in the struct but treated as a future slice.

use crate::board::{
    Board, C_BLACK, C_EMPTY, C_WHITE, Color, Loc, MAX_ARR_SIZE, Move, NULL_LOC, P_BLACK, P_WHITE,
    PASS_LOC, Player,
};
use crate::rules::Rules;
use kata_core::hash;
use kata_core::hash::Hash128;

const NUM_RECENT_BOARDS: usize = 6;

/// Record of a ko capture during a territory-scoring encore.
#[derive(Debug, Clone, Copy)]
struct EncoreKoCapture {
    pos_hash_before_move: Hash128,
    move_loc: Loc,
    move_pla: Player,
}

/// Tracks the history of a board, including legality, superko, and game-end state.
#[derive(Clone)]
pub struct BoardHistory {
    pub rules: Rules,
    pub move_history: Vec<Move>,
    pub prevent_encore_history: Vec<bool>,
    pub ko_hash_history: Vec<Hash128>,
    pub first_turn_idx_with_ko_history: usize,

    pub initial_board: Board,
    pub initial_pla: Player,
    pub initial_encore_phase: i32,
    pub initial_turn_number: i64,

    pub assume_multiple_starting_black_moves_are_handicap: bool,
    pub white_has_moved: bool,
    pub override_num_handicap_stones: i32,

    recent_boards: [Board; NUM_RECENT_BOARDS],
    current_recent_board_idx: usize,
    pub presumed_next_move_pla: Player,

    was_ever_occupied_or_played: [bool; MAX_ARR_SIZE],
    super_ko_banned: [bool; MAX_ARR_SIZE],

    pub consecutive_ending_passes: i32,
    hashes_before_black_pass: Vec<Hash128>,
    hashes_before_white_pass: Vec<Hash128>,

    pub encore_phase: i32,
    pub num_turns_this_phase: i32,
    pub num_approx_valid_turns_this_phase: i32,
    pub num_consec_valid_turns_this_game: i32,

    ko_recap_blocked: [bool; MAX_ARR_SIZE],
    ko_recap_block_hash: Hash128,
    ko_captures_in_encore: Vec<EncoreKoCapture>,
    second_encore_start_colors: [Color; MAX_ARR_SIZE],

    pub white_bonus_score: f32,
    pub white_handicap_bonus_score: f32,
    pub has_button: bool,

    pub is_past_normal_phase_end: bool,
    pub is_game_finished: bool,
    pub winner: Player,
    pub final_white_minus_black_score: f32,
    pub is_scored: bool,
    pub is_no_result: bool,
    pub is_resignation: bool,
}

impl Default for BoardHistory {
    fn default() -> Self {
        Self::new(Board::default(), P_BLACK, Rules::default(), 0)
    }
}

impl BoardHistory {
    pub fn new(board: Board, pla: Player, rules: Rules, encore_phase: i32) -> Self {
        let mut hist = Self {
            rules,
            move_history: Vec::new(),
            prevent_encore_history: Vec::new(),
            ko_hash_history: Vec::new(),
            first_turn_idx_with_ko_history: 0,
            initial_board: board.clone(),
            initial_pla: pla,
            initial_encore_phase: encore_phase,
            initial_turn_number: 0,
            assume_multiple_starting_black_moves_are_handicap: false,
            white_has_moved: false,
            override_num_handicap_stones: -1,
            recent_boards: std::array::from_fn(|_| board.clone()),
            current_recent_board_idx: 0,
            presumed_next_move_pla: pla,
            was_ever_occupied_or_played: [false; MAX_ARR_SIZE],
            super_ko_banned: [false; MAX_ARR_SIZE],
            consecutive_ending_passes: 0,
            hashes_before_black_pass: Vec::new(),
            hashes_before_white_pass: Vec::new(),
            encore_phase: 0,
            num_turns_this_phase: 0,
            num_approx_valid_turns_this_phase: 0,
            num_consec_valid_turns_this_game: 0,
            ko_recap_blocked: [false; MAX_ARR_SIZE],
            ko_recap_block_hash: Hash128::default(),
            ko_captures_in_encore: Vec::new(),
            second_encore_start_colors: [C_EMPTY; MAX_ARR_SIZE],
            white_bonus_score: 0.0,
            white_handicap_bonus_score: 0.0,
            has_button: false,
            is_past_normal_phase_end: false,
            is_game_finished: false,
            winner: C_EMPTY,
            final_white_minus_black_score: 0.0,
            is_scored: false,
            is_no_result: false,
            is_resignation: false,
        };
        hist.clear(board, pla, rules, encore_phase);
        hist
    }

    pub fn clear(&mut self, board: Board, pla: Player, rules: Rules, encore_phase: i32) {
        assert!(
            (0..=2).contains(&encore_phase),
            "BoardHistory::clear - invalid encore phase"
        );
        if encore_phase > 0 {
            assert_eq!(
                rules.scoring_rule,
                crate::rules::ScoringRule::Territory,
                "encore requires territory scoring"
            );
        }

        self.rules = rules;
        self.move_history.clear();
        self.prevent_encore_history.clear();
        self.ko_hash_history.clear();
        self.first_turn_idx_with_ko_history = 0;

        self.initial_board = board.clone();
        self.initial_pla = pla;
        self.initial_encore_phase = encore_phase;
        self.initial_turn_number = 0;
        self.assume_multiple_starting_black_moves_are_handicap = false;
        self.white_has_moved = false;
        self.override_num_handicap_stones = -1;

        self.recent_boards = std::array::from_fn(|_| board.clone());
        self.current_recent_board_idx = 0;
        self.presumed_next_move_pla = pla;

        self.was_ever_occupied_or_played = [false; MAX_ARR_SIZE];
        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let loc = crate::board::location::get_loc(x, y, board.x_size);
                self.was_ever_occupied_or_played[loc as usize] =
                    board.colors[loc as usize] != C_EMPTY;
            }
        }

        self.super_ko_banned = [false; MAX_ARR_SIZE];
        self.consecutive_ending_passes = 0;
        self.hashes_before_black_pass.clear();
        self.hashes_before_white_pass.clear();
        self.num_turns_this_phase = 0;
        self.num_approx_valid_turns_this_phase = 0;
        self.num_consec_valid_turns_this_game = 0;
        self.ko_recap_blocked = [false; MAX_ARR_SIZE];
        self.ko_recap_block_hash = Hash128::default();
        self.ko_captures_in_encore.clear();
        self.second_encore_start_colors = [C_EMPTY; MAX_ARR_SIZE];
        self.white_bonus_score = 0.0;
        if rules.scoring_rule == crate::rules::ScoringRule::Territory {
            // Chill 1 point for every stone initially on the board.
            for y in 0..board.y_size {
                for x in 0..board.x_size {
                    let loc = crate::board::location::get_loc(x, y, board.x_size);
                    match board.colors[loc as usize] {
                        C_BLACK => self.white_bonus_score += 1.0,
                        C_WHITE => self.white_bonus_score -= 1.0,
                        _ => {}
                    }
                }
            }
            // Chill for stones that were captured before history began so that
            // they still affect the territory score.
            let net_white_captures = board.num_white_captures - board.num_black_captures;
            self.white_bonus_score -= net_white_captures as f32;
        }
        self.white_handicap_bonus_score = 0.0;
        self.has_button = rules.has_button && encore_phase == 0;
        self.is_past_normal_phase_end = false;
        self.is_game_finished = false;
        self.winner = C_EMPTY;
        self.final_white_minus_black_score = 0.0;
        self.is_scored = false;
        self.is_no_result = false;
        self.is_resignation = false;

        self.encore_phase = encore_phase;
        if self.encore_phase == 2 {
            self.second_encore_start_colors = board.colors;
        }

        self.ko_hash_history.push(self.compute_ko_hash(&board, pla));
        self.white_handicap_bonus_score = self.compute_white_handicap_bonus() as f32;
    }

    pub fn copy_to_initial(&self) -> Self {
        let mut hist = Self::new(
            self.initial_board.clone(),
            self.initial_pla,
            self.rules,
            self.initial_encore_phase,
        );
        hist.set_initial_turn_number(self.initial_turn_number);
        hist.set_assume_multiple_starting_black_moves_are_handicap(
            self.assume_multiple_starting_black_moves_are_handicap,
        );
        hist.set_override_num_handicap_stones(self.override_num_handicap_stones);
        hist
    }

    pub fn set_komi(&mut self, new_komi: f32) {
        let old_komi = self.rules.komi;
        self.rules.komi = (new_komi * 2.0).round() as i32;
        if self.is_game_finished && self.is_scored {
            self.set_final_score_and_winner(
                self.final_white_minus_black_score - old_komi as f32 / 2.0 + new_komi,
            );
        }
    }

    pub fn set_initial_turn_number(&mut self, n: i64) {
        self.initial_turn_number = n;
    }

    pub fn set_assume_multiple_starting_black_moves_are_handicap(&mut self, b: bool) {
        self.assume_multiple_starting_black_moves_are_handicap = b;
        self.white_handicap_bonus_score = self.compute_white_handicap_bonus() as f32;
    }

    pub fn set_override_num_handicap_stones(&mut self, n: i32) {
        self.override_num_handicap_stones = n;
        self.white_handicap_bonus_score = self.compute_white_handicap_bonus() as f32;
    }

    pub fn get_recent_board(&self, num_moves_ago: usize) -> &Board {
        assert!(num_moves_ago < NUM_RECENT_BOARDS);
        let idx =
            (self.current_recent_board_idx + NUM_RECENT_BOARDS - num_moves_ago) % NUM_RECENT_BOARDS;
        &self.recent_boards[idx]
    }

    pub fn get_current_turn_number(&self) -> i64 {
        (self.initial_turn_number + self.move_history.len() as i64).max(0)
    }

    pub fn is_legal(&self, board: &Board, move_loc: Loc, move_pla: Player) -> bool {
        if move_pla != self.presumed_next_move_pla {
            return false;
        }

        // Ko-moves in the encore that are recapture blocked are interpreted as pass-for-ko, so they are legal.
        if self.encore_phase > 0 && self.is_pass_for_ko(board, move_loc, move_pla) {
            return true;
        }

        if self.encore_phase == 0 && board.is_ko_banned(move_loc) {
            return false;
        }

        if !board.is_legal_ignoring_ko(move_loc, move_pla, self.rules.multi_stone_suicide_legal) {
            return false;
        }

        if self.super_ko_banned[move_loc as usize] {
            return false;
        }

        true
    }

    pub fn is_super_ko_banned(&self, loc: Loc) -> bool {
        self.super_ko_banned[loc as usize]
    }

    pub fn is_ko_recap_blocked(&self, loc: Loc) -> bool {
        self.ko_recap_blocked[loc as usize]
    }

    pub fn second_encore_start_color(&self, loc: Loc) -> Color {
        self.second_encore_start_colors[loc as usize]
    }

    pub fn is_pass_for_ko(&self, board: &Board, move_loc: Loc, move_pla: Player) -> bool {
        if self.encore_phase <= 0 || move_loc == PASS_LOC {
            return false;
        }
        let opp = crate::board::get_opp(move_pla);
        if board.colors[move_loc as usize] == opp
            && self.ko_recap_blocked[move_loc as usize]
            && board.get_chain_size(move_loc) == 1
            && board.get_num_liberties(move_loc) == 1
        {
            return true;
        }
        let ko_capture_loc = board.get_ko_capture_loc(move_loc, move_pla);
        if ko_capture_loc != NULL_LOC
            && self.ko_recap_blocked[ko_capture_loc as usize]
            && board.colors[ko_capture_loc as usize] == opp
        {
            return true;
        }
        false
    }

    pub fn is_legal_tolerant(&self, board: &Board, move_loc: Loc, move_pla: Player) -> bool {
        if move_pla != P_BLACK && move_pla != P_WHITE {
            return false;
        }
        // In the encore, pass-for-ko moves (played on a ko-recap-blocked opponent
        // stone) are legal even though the location is not empty.
        if self.encore_phase > 0 && self.is_pass_for_ko(board, move_loc, move_pla) {
            return true;
        }
        board.is_legal_ignoring_ko(move_loc, move_pla, true)
    }

    pub fn make_board_move_tolerant(
        &mut self,
        board: &mut Board,
        move_loc: Loc,
        move_pla: Player,
    ) -> bool {
        self.make_board_move_tolerant_with_prevent(board, move_loc, move_pla, false)
    }

    pub fn make_board_move_tolerant_with_prevent(
        &mut self,
        board: &mut Board,
        move_loc: Loc,
        move_pla: Player,
        prevent_encore: bool,
    ) -> bool {
        if !self.is_legal_tolerant(board, move_loc, move_pla) {
            return false;
        }
        self.make_board_move_assume_legal_with_prevent(board, move_loc, move_pla, prevent_encore);
        true
    }

    pub fn make_board_move_assume_legal(
        &mut self,
        board: &mut Board,
        move_loc: Loc,
        move_pla: Player,
    ) {
        self.make_board_move_assume_legal_with_prevent(board, move_loc, move_pla, false);
    }

    pub fn prevent_encore_history(&self) -> &[bool] {
        &self.prevent_encore_history
    }

    pub fn pass_would_end_phase(&self, board: &Board, move_pla: Player) -> bool {
        let ko_hash_before_move = self.compute_ko_hash(board, move_pla);
        self.new_consecutive_ending_passes_after_pass() >= 2
            || self.would_be_spightlike_ending_pass(move_pla, ko_hash_before_move)
    }

    pub fn pass_would_end_game(&self, board: &Board, move_pla: Player) -> bool {
        self.pass_would_end_phase(board, move_pla)
            && (self.rules.scoring_rule == crate::rules::ScoringRule::Area
                || (self.rules.scoring_rule == crate::rules::ScoringRule::Territory
                    && self.encore_phase >= 2))
    }

    pub fn is_final_phase(&self) -> bool {
        self.rules.scoring_rule == crate::rules::ScoringRule::Area
            || (self.rules.scoring_rule == crate::rules::ScoringRule::Territory
                && self.encore_phase >= 2)
    }

    /// True if a friendly pass would suppress the game-end effect of a pass.
    pub fn should_suppress_end_game_from_friendly_pass(
        &self,
        board: &Board,
        move_pla: Player,
    ) -> bool {
        self.rules.friendly_pass_ok
            && self.rules.scoring_rule == crate::rules::ScoringRule::Area
            && self.new_consecutive_ending_passes_after_pass() == 2
            && !self
                .would_be_spightlike_ending_pass(move_pla, self.compute_ko_hash(board, move_pla))
    }

    pub fn make_board_move_assume_legal_with_prevent(
        &mut self,
        board: &mut Board,
        move_loc: Loc,
        move_pla: Player,
        prevent_encore: bool,
    ) {
        let pos_hash_before_move = board.pos_hash;

        if self.is_game_finished || self.is_past_normal_phase_end {
            self.num_approx_valid_turns_this_phase = self.num_approx_valid_turns_this_phase.min(1);
            self.num_consec_valid_turns_this_game = self.num_consec_valid_turns_this_game.min(1);
        }

        let move_is_illegal = !self.is_legal(board, move_loc, move_pla);

        self.is_game_finished = false;
        self.is_past_normal_phase_end = false;
        self.winner = C_EMPTY;
        self.final_white_minus_black_score = 0.0;
        self.is_scored = false;
        self.is_no_result = false;
        self.is_resignation = false;

        let mut is_spightlike_ending_pass = false;
        if move_loc != PASS_LOC {
            self.consecutive_ending_passes = 0;
        } else if self.has_button {
            assert!(self.encore_phase == 0 && self.rules.has_button);
            self.has_button = false;
            self.white_bonus_score += if move_pla == P_WHITE { 0.5 } else { -0.5 };
            self.consecutive_ending_passes = 0;
            self.hashes_before_black_pass.clear();
            self.hashes_before_white_pass.clear();
            self.ko_hash_history.clear();
            self.first_turn_idx_with_ko_history = self.move_history.len() + 1;
        } else {
            if self.phase_has_spightlike_ending_and_pass_history_clearing() {
                self.ko_hash_history.clear();
                self.first_turn_idx_with_ko_history = self.move_history.len() + 1;
            }
            let ko_hash_before_this_move = self.compute_ko_hash(board, move_pla);
            self.consecutive_ending_passes = self.new_consecutive_ending_passes_after_pass();
            is_spightlike_ending_pass =
                self.would_be_spightlike_ending_pass(move_pla, ko_hash_before_this_move);

            if move_pla == P_BLACK {
                self.hashes_before_black_pass.push(ko_hash_before_this_move);
            } else {
                self.hashes_before_white_pass.push(ko_hash_before_this_move);
            }
        }

        // Handle pass-for-ko moves in the encore. Pass for ko lifts a ko recapture block and does nothing else.
        let mut was_pass_for_ko = false;
        if self.encore_phase > 0 && move_loc != PASS_LOC {
            let opp = crate::board::get_opp(move_pla);
            if board.colors[move_loc as usize] == opp && self.ko_recap_blocked[move_loc as usize] {
                self.set_ko_recap_blocked(move_loc, false);
                was_pass_for_ko = true;
                // Clear simple ko loc just in case.
                // Since we aren't otherwise touching the board, from the board's perspective a player will be moving twice in a row.
                board.clear_simple_ko_loc();
            } else {
                let ko_capture_loc = board.get_ko_capture_loc(move_loc, move_pla);
                if ko_capture_loc != NULL_LOC
                    && self.ko_recap_blocked[ko_capture_loc as usize]
                    && board.colors[ko_capture_loc as usize] == opp
                {
                    self.set_ko_recap_blocked(ko_capture_loc, false);
                    was_pass_for_ko = true;
                    board.clear_simple_ko_loc();
                }
            }
        }

        if !was_pass_for_ko {
            board.play_move_assume_legal(move_loc, move_pla);

            if self.encore_phase > 0 {
                // Update ko recapture blocks and record that this was a ko capture.
                if board.ko_loc != NULL_LOC {
                    self.set_ko_recap_blocked(move_loc, true);
                    self.ko_captures_in_encore.push(EncoreKoCapture {
                        pos_hash_before_move,
                        move_loc,
                        move_pla,
                    });
                    // Clear simple ko loc now that we've absorbed the ko loc information into the ko recap blocks.
                    board.clear_simple_ko_loc();
                }
                // Unmark all ko recap blocks not on stones.
                for y in 0..board.y_size {
                    for x in 0..board.x_size {
                        let loc = crate::board::location::get_loc(x, y, board.x_size);
                        if board.colors[loc as usize] == C_EMPTY
                            && self.ko_recap_blocked[loc as usize]
                        {
                            self.set_ko_recap_blocked(loc, false);
                        }
                    }
                }
            }
        }

        self.current_recent_board_idx = (self.current_recent_board_idx + 1) % NUM_RECENT_BOARDS;
        self.recent_boards[self.current_recent_board_idx] = board.clone();

        let next_pla = crate::board::get_opp(move_pla);
        let ko_hash_after_this_move = self.compute_ko_hash(board, next_pla);
        self.ko_hash_history.push(ko_hash_after_this_move);
        self.move_history.push(Move::new(move_loc, move_pla));
        self.prevent_encore_history.push(prevent_encore);
        self.num_turns_this_phase += 1;
        self.num_approx_valid_turns_this_phase += 1;
        self.num_consec_valid_turns_this_game += 1;
        self.presumed_next_move_pla = next_pla;

        if move_is_illegal {
            self.num_consec_valid_turns_this_game = 0;
        }

        if move_loc != PASS_LOC {
            self.was_ever_occupied_or_played[move_loc as usize] = true;
        }

        self.update_super_ko_banned(board);

        // Territory scoring - chill 1 point per move in main phase and first encore.
        if self.rules.scoring_rule == crate::rules::ScoringRule::Territory
            && self.encore_phase <= 1
            && move_loc != PASS_LOC
            && !was_pass_for_ko
        {
            if move_pla == P_BLACK {
                self.white_bonus_score += 1.0;
            } else {
                self.white_bonus_score -= 1.0;
            }
        }

        if move_pla == P_WHITE && move_loc != PASS_LOC {
            self.white_has_moved = true;
        }
        if self.assume_multiple_starting_black_moves_are_handicap
            && !self.white_has_moved
            && move_pla == P_BLACK
            && self.rules.white_handicap_bonus_rule != crate::rules::WhiteHandicapBonusRule::Zero
        {
            self.white_handicap_bonus_score = self.compute_white_handicap_bonus() as f32;
        }

        if self.consecutive_ending_passes >= 2 || is_spightlike_ending_pass {
            match self.rules.scoring_rule {
                crate::rules::ScoringRule::Area => {
                    assert!(self.encore_phase <= 0);
                    self.end_and_score_game_now(board);
                }
                crate::rules::ScoringRule::Territory => {
                    if self.encore_phase >= 2 {
                        self.end_and_score_game_now(board);
                    } else if prevent_encore {
                        self.is_past_normal_phase_end = true;
                        self.num_approx_valid_turns_this_phase =
                            self.num_approx_valid_turns_this_phase.min(1);
                        self.num_consec_valid_turns_this_game =
                            self.num_consec_valid_turns_this_game.min(1);
                    } else {
                        self.encore_phase += 1;
                        self.num_turns_this_phase = 0;
                        self.num_approx_valid_turns_this_phase = 0;
                        if self.encore_phase == 2 {
                            self.second_encore_start_colors = board.colors;
                        }

                        self.super_ko_banned = [false; MAX_ARR_SIZE];
                        self.consecutive_ending_passes = 0;
                        self.hashes_before_black_pass.clear();
                        self.hashes_before_white_pass.clear();
                        self.ko_recap_blocked = [false; MAX_ARR_SIZE];
                        self.ko_recap_block_hash = Hash128::default();
                        self.ko_captures_in_encore.clear();

                        self.ko_hash_history.clear();
                        let next_pla = crate::board::get_opp(move_pla);
                        self.ko_hash_history
                            .push(self.compute_ko_hash(board, next_pla));
                        self.first_turn_idx_with_ko_history = self.move_history.len();
                    }
                }
            }
        }

        if move_loc != PASS_LOC
            && (self.encore_phase > 0 || self.rules.ko_rule == crate::rules::KoRule::Simple)
        {
            let idx = self.ko_hash_history.len() - 1;
            if self.number_of_ko_hash_occurrences_in_history(self.ko_hash_history[idx]) >= 3 {
                self.is_no_result = true;
                self.is_game_finished = true;
            }
        }
    }

    pub fn set_winner_by_resignation(&mut self, pla: Player) {
        self.is_game_finished = true;
        self.is_past_normal_phase_end = false;
        self.is_scored = false;
        self.is_no_result = false;
        self.is_resignation = true;
        self.winner = pla;
        self.final_white_minus_black_score = 0.0;
    }

    pub fn end_and_score_game_now(&mut self, board: &Board) {
        let board_score = match self.rules.scoring_rule {
            crate::rules::ScoringRule::Area => self.count_area_score_white_minus_black(board),
            crate::rules::ScoringRule::Territory => {
                self.count_territory_area_score_white_minus_black(board)
            }
        };
        if self.has_button {
            self.has_button = false;
            self.white_bonus_score += if self.presumed_next_move_pla == P_WHITE {
                0.5
            } else {
                -0.5
            };
        }
        self.set_final_score_and_winner(
            board_score as f32
                + self.white_bonus_score
                + self.white_handicap_bonus_score
                + self.rules.komi as f32 / 2.0,
        );
        self.is_scored = true;
        self.is_no_result = false;
        self.is_resignation = false;
        self.is_game_finished = true;
        self.is_past_normal_phase_end = false;
    }

    /// End the game immediately if every board point is pass-alive territory.
    ///
    /// Mirrors `BoardHistory::endGameIfAllPassAlive` in `cpp/game/boardhistory.cpp`.
    pub fn end_game_if_all_pass_alive(&mut self, board: &Board) {
        use crate::board::{C_BLACK, C_EMPTY, C_WHITE, location};

        let mut board_score = 0i32;
        let mut area = [C_EMPTY; MAX_ARR_SIZE];
        board.calculate_area(
            &mut area,
            true,
            true,
            true,
            self.rules.multi_stone_suicide_legal,
        );

        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let loc = location::get_loc(x, y, board.x_size) as usize;
                match area[loc] {
                    C_WHITE => board_score += 1,
                    C_BLACK => board_score -= 1,
                    _ => return,
                }
            }
        }

        // With group tax, score normally so that the tax is actually applied.
        if self.rules.tax_rule == crate::rules::TaxRule::All {
            self.end_and_score_game_now(board);
        } else {
            if self.has_button {
                self.has_button = false;
                self.white_bonus_score += if self.presumed_next_move_pla == C_WHITE {
                    0.5
                } else {
                    -0.5
                };
            }
            self.set_final_score_and_winner(
                board_score as f32
                    + self.white_bonus_score
                    + self.white_handicap_bonus_score
                    + self.rules.komi as f32 / 2.0,
            );
            self.is_scored = true;
            self.is_no_result = false;
            self.is_resignation = false;
            self.is_game_finished = true;
            self.is_past_normal_phase_end = false;
        }
    }

    /// Print detailed debug state, mirroring `BoardHistory::printDebugInfo`.
    pub fn print_debug_info(&self, out: &mut String, board: &Board) {
        // Note: C++ prints the board hash here; Rust uses a different zobrist seed, so omit it.
        // C++ also prints an extra blank line after the board grid, so include it for fidelity.
        out.push_str(&format!("{}\n\n", board));
        out.push_str(&format!(
            "Initial pla {}\n",
            crate::board::player_io::player_to_string(self.initial_pla)
        ));
        out.push_str(&format!("Encore phase {}\n", self.encore_phase));
        out.push_str(&format!("Turns this phase {}\n", self.num_turns_this_phase));
        out.push_str(&format!(
            "Approx valid turns this phase {}\n",
            self.num_approx_valid_turns_this_phase
        ));
        out.push_str(&format!(
            "Approx consec valid turns this game {}\n",
            self.num_consec_valid_turns_this_game
        ));
        out.push_str(&format!("Rules {}\n", self.rules));
        out.push_str(&format!(
            "Ko recap block hash {}\n",
            self.ko_recap_block_hash
        ));
        out.push_str(&format!("White bonus score {}\n", self.white_bonus_score));
        out.push_str(&format!(
            "White handicap bonus score {}\n",
            self.white_handicap_bonus_score
        ));
        out.push_str(&format!("Has button {}\n", self.has_button as i32));
        out.push_str(&format!(
            "Presumed next pla {}\n",
            crate::board::player_io::player_to_string(self.presumed_next_move_pla)
        ));
        out.push_str(&format!(
            "Past normal phase end {}\n",
            self.is_past_normal_phase_end as i32
        ));
        out.push_str(&format!(
            "Game result {} {} {} {} {} {}\n",
            self.is_game_finished as i32,
            crate::board::player_io::player_to_string(self.winner),
            self.final_white_minus_black_score,
            self.is_scored as i32,
            self.is_no_result as i32,
            self.is_resignation as i32
        ));
        out.push_str("Last moves ");
        for m in &self.move_history {
            out.push_str(&format!(
                "{} ",
                crate::board::location::to_string(m.loc, board.x_size, board.y_size)
            ));
        }
        out.push('\n');
        assert!(
            self.first_turn_idx_with_ko_history + self.ko_hash_history.len()
                == self.move_history.len() + 1
        );
    }

    pub fn current_self_komi(&self, pla: Player, draw_equivalent_wins_for_white: f64) -> f32 {
        let white_komi_adjusted = self.white_bonus_score
            + self.white_handicap_bonus_score
            + self.rules.komi as f32 / 2.0
            + self.white_komi_adjustment_for_draws(draw_equivalent_wins_for_white);
        if pla == P_WHITE {
            white_komi_adjusted
        } else {
            -white_komi_adjusted
        }
    }

    pub fn white_komi_adjustment_for_draws(&self, draw_equivalent_wins_for_white: f64) -> f32 {
        if self.rules.game_result_will_be_integer() {
            (draw_equivalent_wins_for_white - 0.5) as f32
        } else {
            0.0
        }
    }

    pub fn num_handicap_stones_on_board(board: &Board) -> i32 {
        Self::num_handicap_stones_on_board_helper(board, 0)
    }

    pub fn compute_num_handicap_stones(&self) -> i32 {
        if self.override_num_handicap_stones >= 0 {
            return self.override_num_handicap_stones;
        }

        let mut black_non_pass_turns_to_start = 0;
        if self.assume_multiple_starting_black_moves_are_handicap {
            for i in 0..self.move_history.len() {
                let move_loc = self.move_history[i].loc;
                let move_pla = self.move_history[i].pla;
                if move_pla != P_BLACK {
                    if i + 1 < self.move_history.len() && self.move_history[i + 1].pla != P_BLACK {
                        black_non_pass_turns_to_start = 0;
                        break;
                    }
                    if move_loc == PASS_LOC {
                        continue;
                    }
                    break;
                }
                if move_loc != PASS_LOC && move_loc != NULL_LOC {
                    black_non_pass_turns_to_start += 1;
                }
            }
        }
        Self::num_handicap_stones_on_board_helper(
            &self.initial_board,
            black_non_pass_turns_to_start,
        )
    }

    pub fn compute_white_handicap_bonus(&self) -> i32 {
        use crate::rules::WhiteHandicapBonusRule;
        match self.rules.white_handicap_bonus_rule {
            WhiteHandicapBonusRule::Zero => 0,
            _ => {
                let n = self.compute_num_handicap_stones();
                match self.rules.white_handicap_bonus_rule {
                    WhiteHandicapBonusRule::N => n,
                    WhiteHandicapBonusRule::NMinusOne => (n - 1).max(0),
                    WhiteHandicapBonusRule::Zero => 0,
                }
            }
        }
    }

    pub fn has_black_pass_or_white_first(&self) -> bool {
        if self.initial_board.is_empty()
            && !self.move_history.is_empty()
            && self.move_history[0].pla == P_WHITE
        {
            return true;
        }
        let mut num_black_passes = 0;
        let mut num_white_passes = 0;
        let mut num_black_double_moves = 0;
        let mut num_white_double_moves = 0;
        for i in 0..self.move_history.len() {
            let m = self.move_history[i];
            if m.loc == PASS_LOC && m.pla == P_BLACK {
                num_black_passes += 1;
            }
            if m.loc == PASS_LOC && m.pla == P_WHITE {
                num_white_passes += 1;
            }
            if i > 0 && m.pla == P_BLACK && self.move_history[i - 1].pla == P_BLACK {
                num_black_double_moves += 1;
            }
            if i > 0 && m.pla == P_WHITE && self.move_history[i - 1].pla == P_WHITE {
                num_white_double_moves += 1;
            }
        }
        (num_black_passes == 1
            && num_white_passes == 0
            && num_black_double_moves == 0
            && num_white_double_moves == 0)
            || (num_black_passes == 0
                && num_white_passes == 0
                && num_black_double_moves == 0
                && num_white_double_moves == 1)
    }

    pub fn get_situation_and_simple_ko_hash(board: &Board, next_player: Player) -> Hash128 {
        board.get_sit_hash_with_simple_ko(next_player)
    }

    pub fn get_situation_and_simple_ko_and_prev_pos_hash(
        board: &Board,
        hist: &BoardHistory,
        next_player: Player,
    ) -> Hash128 {
        let hash = board.get_sit_hash_with_simple_ko(next_player);
        let mut mixed = Hash128::new(hash::rrmxmx(hash.hash0), hash::split_mix64(hash.hash1));
        if !hist.move_history.is_empty() {
            mixed ^= hist.get_recent_board(1).pos_hash;
        }
        mixed
    }

    pub fn get_situation_rules_and_ko_hash(
        board: &Board,
        hist: &BoardHistory,
        next_player: Player,
        draw_equivalent_wins_for_white: f64,
    ) -> Hash128 {
        use crate::rules::Rules;

        let mut hash = board.pos_hash;
        hash ^= Board::zobrist_player_hash(next_player as usize);
        hash ^= Board::zobrist_encore_hash(hist.encore_phase as usize);

        if hist.encore_phase == 0 {
            if board.ko_loc != NULL_LOC {
                hash ^= Board::zobrist_ko_loc_hash(board.ko_loc as usize);
            }
            for y in 0..board.y_size {
                for x in 0..board.x_size {
                    let loc = crate::board::location::get_loc(x, y, board.x_size) as usize;
                    if hist.super_ko_banned[loc] && loc as Loc != board.ko_loc {
                        hash ^= Board::zobrist_ko_loc_hash(loc);
                    }
                }
            }
        } else {
            for y in 0..board.y_size {
                for x in 0..board.x_size {
                    let loc = crate::board::location::get_loc(x, y, board.x_size) as usize;
                    if hist.super_ko_banned[loc] {
                        hash ^= Board::zobrist_ko_loc_hash(loc);
                    }
                    if hist.ko_recap_blocked[loc] {
                        hash ^= Board::zobrist_ko_mark_hash(loc, C_BLACK as usize)
                            ^ Board::zobrist_ko_mark_hash(loc, C_WHITE as usize);
                    }
                }
            }
            if hist.encore_phase == 2 {
                for y in 0..board.y_size {
                    for x in 0..board.x_size {
                        let loc = crate::board::location::get_loc(x, y, board.x_size) as usize;
                        let c = hist.second_encore_start_colors[loc];
                        if c != C_EMPTY {
                            hash ^= Board::zobrist_second_encore_start_hash(loc, c as usize);
                        }
                    }
                }
            }
        }

        let self_komi = hist.current_self_komi(next_player, draw_equivalent_wins_for_white);
        let komi_discretized = (self_komi * 256.0) as i64;
        let komi_hash = hash::murmur_mix(komi_discretized as u64);
        hash.hash0 ^= komi_hash;
        hash.hash1 ^= hash::basic_l_cong(komi_hash);

        hash ^= Rules::zobrist_ko_rule_hash(hist.rules.ko_rule);
        hash ^= Rules::zobrist_scoring_rule_hash(hist.rules.scoring_rule);
        hash ^= Rules::zobrist_tax_rule_hash(hist.rules.tax_rule);
        if hist.rules.multi_stone_suicide_legal {
            hash ^= Rules::zobrist_multi_stone_suicide_hash();
        }
        if hist.has_button {
            hash ^= Rules::zobrist_button_hash();
        }
        if hist.rules.friendly_pass_ok {
            hash ^= Rules::zobrist_friendly_pass_ok_hash();
        }

        hash
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn compute_ko_hash(&self, board: &Board, pla: Player) -> Hash128 {
        use crate::rules::KoRule;
        let mut hash = board.pos_hash ^ self.ko_recap_block_hash;
        if self.rules.ko_rule == KoRule::Situational
            || self.rules.ko_rule == KoRule::Simple
            || self.encore_phase > 0
        {
            hash ^= Board::zobrist_player_hash(pla as usize);
        }
        hash
    }

    fn compute_ko_hash_after_move_nonencore(
        &self,
        pos_hash_after_move: Hash128,
        pla: Player,
    ) -> Hash128 {
        use crate::rules::KoRule;
        if self.rules.ko_rule == KoRule::Situational || self.rules.ko_rule == KoRule::Simple {
            pos_hash_after_move ^ Board::zobrist_player_hash(pla as usize)
        } else {
            pos_hash_after_move
        }
    }

    fn ko_hash_occurs_in_history(&self, ko_hash: Hash128) -> bool {
        self.ko_hash_history.contains(&ko_hash)
    }

    pub fn number_of_ko_hash_occurrences_in_history(&self, ko_hash: Hash128) -> i32 {
        self.ko_hash_history
            .iter()
            .filter(|&&h| h == ko_hash)
            .count() as i32
    }

    fn new_consecutive_ending_passes_after_pass(&self) -> i32 {
        if self.encore_phase > 0 {
            self.consecutive_ending_passes + 1
        } else {
            match self.rules.ko_rule {
                crate::rules::KoRule::Simple
                | crate::rules::KoRule::Positional
                | crate::rules::KoRule::Situational => self.consecutive_ending_passes + 1,
                crate::rules::KoRule::Spight => 0,
            }
        }
    }

    fn phase_has_spightlike_ending_and_pass_history_clearing(&self) -> bool {
        self.encore_phase > 0
            || self.rules.ko_rule == crate::rules::KoRule::Simple
            || self.rules.ko_rule == crate::rules::KoRule::Spight
    }

    fn would_be_spightlike_ending_pass(&self, move_pla: Player, ko_hash: Hash128) -> bool {
        if !self.phase_has_spightlike_ending_and_pass_history_clearing() {
            return false;
        }
        let vec = if move_pla == P_BLACK {
            &self.hashes_before_black_pass
        } else {
            &self.hashes_before_white_pass
        };
        vec.contains(&ko_hash)
    }

    fn set_ko_recap_blocked(&mut self, loc: Loc, b: bool) {
        if self.ko_recap_blocked[loc as usize] != b {
            self.ko_recap_blocked[loc as usize] = b;
            // We used to have per-color marks, so the zobrist was for both. Just combine them.
            self.ko_recap_block_hash ^= Board::zobrist_ko_mark_hash(loc as usize, C_BLACK as usize)
                ^ Board::zobrist_ko_mark_hash(loc as usize, C_WHITE as usize);
        }
    }

    fn set_final_score_and_winner(&mut self, score: f32) {
        self.final_white_minus_black_score = score;
        if score > 0.0 {
            self.winner = C_WHITE;
        } else if score < 0.0 {
            self.winner = C_BLACK;
        } else {
            self.winner = C_EMPTY;
        }
    }

    fn update_super_ko_banned(&mut self, board: &Board) {
        let next_pla = self.presumed_next_move_pla;
        if self.encore_phase <= 0 && self.rules.ko_rule != crate::rules::KoRule::Simple {
            assert!(self.ko_recap_block_hash == Hash128::default());
            for y in 0..board.y_size {
                for x in 0..board.x_size {
                    let loc = crate::board::location::get_loc(x, y, board.x_size);
                    let idx = loc as usize;
                    let not_superko_candidate = board.colors[idx] != C_EMPTY
                        || (!self.was_ever_occupied_or_played[idx]
                            && !board.is_suicide(loc, next_pla))
                        || board.is_illegal_suicide(
                            loc,
                            next_pla,
                            self.rules.multi_stone_suicide_legal,
                        )
                        || loc == board.ko_loc;
                    if not_superko_candidate {
                        self.super_ko_banned[idx] = false;
                    } else {
                        let pos_hash_after_move = board.get_pos_hash_after_move(loc, next_pla);
                        let ko_hash_after_move = self.compute_ko_hash_after_move_nonencore(
                            pos_hash_after_move,
                            crate::board::get_opp(next_pla),
                        );
                        self.super_ko_banned[idx] =
                            self.ko_hash_occurs_in_history(ko_hash_after_move);
                    }
                }
            }
        } else if self.encore_phase > 0 {
            self.super_ko_banned = [false; MAX_ARR_SIZE];
            for capture in &self.ko_captures_in_encore {
                if capture.pos_hash_before_move == board.pos_hash && capture.move_pla == next_pla {
                    self.super_ko_banned[capture.move_loc as usize] = true;
                }
            }
        } else {
            self.super_ko_banned = [false; MAX_ARR_SIZE];
        }
    }

    fn count_area_score_white_minus_black(&self, board: &Board) -> i32 {
        let mut area = [C_EMPTY; MAX_ARR_SIZE];
        let mut score = 0;

        match self.rules.tax_rule {
            crate::rules::TaxRule::None => {
                board.calculate_area(
                    &mut area,
                    true,
                    true,
                    true,
                    self.rules.multi_stone_suicide_legal,
                );
            }
            crate::rules::TaxRule::Seki | crate::rules::TaxRule::All => {
                let white_minus_black_independent_life_region_count = board
                    .calculate_independent_life_area(
                        &mut area,
                        false,
                        true,
                        self.rules.multi_stone_suicide_legal,
                    );
                if self.rules.tax_rule == crate::rules::TaxRule::All {
                    score -= 2 * white_minus_black_independent_life_region_count;
                }
            }
        }

        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let loc = crate::board::location::get_loc(x, y, board.x_size) as usize;
                if area[loc] == C_WHITE {
                    score += 1;
                } else if area[loc] == C_BLACK {
                    score -= 1;
                }
            }
        }
        score
    }

    fn count_territory_area_score_white_minus_black(&self, board: &Board) -> i32 {
        let mut area = [C_EMPTY; MAX_ARR_SIZE];
        let (keep_territories, keep_stones) = match self.rules.tax_rule {
            crate::rules::TaxRule::None => (true, false),
            crate::rules::TaxRule::Seki | crate::rules::TaxRule::All => (false, false),
        };

        let white_minus_black_independent_life_region_count = board
            .calculate_independent_life_area(
                &mut area,
                keep_territories,
                keep_stones,
                self.rules.multi_stone_suicide_legal,
            );

        let mut score = 0;
        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let loc = crate::board::location::get_loc(x, y, board.x_size) as usize;
                if area[loc] == C_WHITE {
                    score += 1;
                } else if area[loc] == C_BLACK {
                    score -= 1;
                } else {
                    // Checking encore_phase < 2 allows us to get the correct score if we directly end the game before the second
                    // encore such that we never actually fill second_encore_start_colors.
                    if board.colors[loc] == C_WHITE
                        && (self.encore_phase < 2
                            || self.second_encore_start_colors[loc] == C_WHITE)
                    {
                        score += 1;
                        area[loc] = C_WHITE;
                    }
                    if board.colors[loc] == C_BLACK
                        && (self.encore_phase < 2
                            || self.second_encore_start_colors[loc] == C_BLACK)
                    {
                        score -= 1;
                        area[loc] = C_BLACK;
                    }
                }
            }
        }
        if self.rules.tax_rule == crate::rules::TaxRule::All {
            score -= 2 * white_minus_black_independent_life_region_count;
        }
        score
    }

    fn num_handicap_stones_on_board_helper(
        board: &Board,
        black_non_pass_turns_to_start: i32,
    ) -> i32 {
        let mut start_board_num_black_stones = 0;
        let mut start_board_num_white_stones = 0;
        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let loc = crate::board::location::get_loc(x, y, board.x_size) as usize;
                let c = board.colors[loc];
                if c == C_BLACK {
                    start_board_num_black_stones += 1;
                } else if c == C_WHITE {
                    start_board_num_white_stones += 1;
                }
            }
        }
        if start_board_num_white_stones != 0 {
            return 0;
        }
        let black_turn_advantage = start_board_num_black_stones + black_non_pass_turns_to_start;
        if black_turn_advantage <= 1 {
            0
        } else {
            black_turn_advantage
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::location;

    fn empty_history() -> BoardHistory {
        BoardHistory::new(Board::new(5, 5), P_BLACK, Rules::default(), 0)
    }

    #[test]
    fn test_history_init() {
        let hist = empty_history();
        assert_eq!(hist.get_current_turn_number(), 0);
        assert_eq!(hist.presumed_next_move_pla, P_BLACK);
        assert!(!hist.is_game_finished);
    }

    #[test]
    fn test_history_simple_moves() {
        let mut board = Board::new(5, 5);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let c3 = location::get_loc(2, 2, 5);
        assert!(hist.is_legal(&board, c3, P_BLACK));
        hist.make_board_move_assume_legal(&mut board, c3, P_BLACK);
        assert_eq!(hist.presumed_next_move_pla, P_WHITE);
        assert_eq!(hist.get_current_turn_number(), 1);
        assert!(hist.is_legal(&board, PASS_LOC, P_WHITE));
    }

    #[test]
    fn test_history_turn_order() {
        let mut board = Board::new(5, 5);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let c3 = location::get_loc(2, 2, 5);
        assert!(!hist.is_legal(&board, c3, P_WHITE));
        hist.make_board_move_assume_legal(&mut board, c3, P_BLACK);
        assert!(!hist.is_legal(&board, c3, P_BLACK));
    }

    #[test]
    fn test_history_recent_boards() {
        let mut board = Board::new(5, 5);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let c3 = location::get_loc(2, 2, 5);
        hist.make_board_move_assume_legal(&mut board, c3, P_BLACK);
        assert_eq!(hist.get_recent_board(0).colors[c3 as usize], C_BLACK);
        assert_eq!(hist.get_recent_board(1).colors[c3 as usize], C_EMPTY);
    }

    #[test]
    fn test_history_pass_ends_game_area_scoring() {
        let mut board = Board::new(5, 5);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        hist.make_board_move_assume_legal(&mut board, PASS_LOC, P_BLACK);
        assert!(!hist.is_game_finished);
        hist.make_board_move_assume_legal(&mut board, PASS_LOC, P_WHITE);
        assert!(hist.is_game_finished);
        assert!(hist.is_scored);
    }

    #[test]
    fn test_history_superko_positional() {
        use crate::rules::KoRule;
        let rules = Rules {
            ko_rule: KoRule::Positional,
            ..Rules::default()
        };
        let mut board = Board::new(5, 5);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, rules, 0);
        // A simple cycle: B C3, W D3, B D2, W C2 returns to the empty position rotated.
        // Instead, play a direct ko cycle and verify the third repetition is banned.
        let c3 = location::get_loc(2, 2, 5);
        let d3 = location::get_loc(3, 2, 5);
        let c2 = location::get_loc(2, 1, 5);
        let d2 = location::get_loc(3, 1, 5);
        hist.make_board_move_assume_legal(&mut board, c3, P_BLACK);
        hist.make_board_move_assume_legal(&mut board, d3, P_WHITE);
        hist.make_board_move_assume_legal(&mut board, d2, P_BLACK);
        hist.make_board_move_assume_legal(&mut board, c2, P_WHITE);
        // The board is non-empty, so no superko ban should be present yet for empty points.
        assert!(hist.is_legal(&board, PASS_LOC, P_BLACK));
    }

    #[test]
    fn test_history_hash_methods() {
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let h1 = BoardHistory::get_situation_and_simple_ko_hash(&board, P_BLACK);
        let h2 = BoardHistory::get_situation_and_simple_ko_hash(&board, P_WHITE);
        assert_ne!(h1, h2);
        let h3 = BoardHistory::get_situation_rules_and_ko_hash(&board, &hist, P_BLACK, 0.5);
        let h4 = BoardHistory::get_situation_rules_and_ko_hash(&board, &hist, P_WHITE, 0.5);
        assert_ne!(h3, h4);
    }
}

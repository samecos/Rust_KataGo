//! Neural-network input/output types and basic helpers.
//!
//! Corresponds to parts of `cpp/neuralnet/nninputs.h` and `cpp/neuralnet/nninputs.cpp`.
//! This slice covers the `NNPos` helpers, `MiscNNInputParams`, and the `NNOutput`
//! data structure. The feature encoding (`fillRowV*`) and `ScoreValue` tables are
//! left for later slices.

use kata_core::hash::{self, Hash128};
use kata_game::board::{
    Board, C_BLACK, C_EMPTY, C_WHITE, Color, Loc, MAX_ARR_SIZE, MAX_LEN, Move, NULL_LOC, P_BLACK,
    P_WHITE, PASS_LOC, Player,
};
use kata_game::history::BoardHistory;
use kata_game::rules::{KoRule, ScoringRule, TaxRule};

pub const NUM_FEATURES_SPATIAL_V3: i32 = 22;
pub const NUM_FEATURES_GLOBAL_V3: i32 = 14;

pub const NUM_FEATURES_SPATIAL_V4: i32 = 22;
pub const NUM_FEATURES_GLOBAL_V4: i32 = 14;

pub const NUM_FEATURES_SPATIAL_V5: i32 = 13;
pub const NUM_FEATURES_GLOBAL_V5: i32 = 12;

pub const NUM_FEATURES_SPATIAL_V6: i32 = 22;
pub const NUM_FEATURES_GLOBAL_V6: i32 = 16;

pub const NUM_FEATURES_SPATIAL_V7: i32 = 22;
pub const NUM_FEATURES_GLOBAL_V7: i32 = 19;

/// Helpers for converting between board locations and neural-network positions.
pub mod nn_pos {
    use super::{Loc, NULL_LOC, PASS_LOC};
    use kata_game::board::location;

    pub const MAX_BOARD_LEN: usize = super::MAX_LEN;
    pub const MAX_BOARD_AREA: usize = MAX_BOARD_LEN * MAX_BOARD_LEN;
    /// Policy output adds +1 for the pass move.
    pub const MAX_NN_POLICY_SIZE: usize = MAX_BOARD_AREA + 1;
    /// Extra score distribution radius, used for writing score in data rows and for the neural net score belief output.
    pub const EXTRA_SCORE_DISTR_RADIUS: i32 = 60;
    /// Used various places we clip komi beyond board area.
    pub const KOMI_CLIP_RADIUS: f32 = 20.0f32;

    /// Convert `(x, y)` to a flat position index.
    pub fn xy_to_pos(x: i32, y: i32, nn_x_len: i32) -> i32 {
        y * nn_x_len + x
    }

    /// Convert a board location to a flat neural-network position.
    pub fn loc_to_pos(loc: Loc, board_x_size: i32, nn_x_len: i32, nn_y_len: i32) -> i32 {
        if loc == PASS_LOC {
            nn_x_len * nn_y_len
        } else if loc == NULL_LOC {
            nn_x_len * (nn_y_len + 1)
        } else {
            location::get_y(loc, board_x_size) * nn_x_len + location::get_x(loc, board_x_size)
        }
    }

    /// Convert a flat neural-network position back to a board location.
    pub fn pos_to_loc(
        pos: i32,
        board_x_size: i32,
        board_y_size: i32,
        nn_x_len: i32,
        nn_y_len: i32,
    ) -> Loc {
        if pos == nn_x_len * nn_y_len {
            return PASS_LOC;
        }
        let x = pos % nn_x_len;
        let y = pos / nn_x_len;
        if x < 0 || x >= board_x_size || y < 0 || y >= board_y_size {
            return NULL_LOC;
        }
        location::get_loc(x, y, board_x_size)
    }

    /// Flat position index of the pass output.
    pub fn pass_pos(nn_x_len: i32, nn_y_len: i32) -> i32 {
        nn_x_len * nn_y_len
    }

    /// Returns true if `pos` is the pass output position.
    pub fn is_pass_pos(pos: i32, nn_x_len: i32, nn_y_len: i32) -> bool {
        pos == nn_x_len * nn_y_len
    }

    /// Total policy output size for a board of `(nn_x_len, nn_y_len)`.
    pub fn policy_size(nn_x_len: i32, nn_y_len: i32) -> i32 {
        nn_x_len * nn_y_len + 1
    }
}

/// Miscellanous parameters that affect how a neural-net input row is constructed.
pub mod nn_inputs {
    pub const SYMMETRY_NOT_SPECIFIED: i32 = -1;
    pub const SYMMETRY_ALL: i32 = -2;
}

/// Parameters that can alter the neural-net hash and input features for a position.
#[derive(Debug, Clone, Copy)]
pub struct MiscNNInputParams {
    pub draw_equivalent_wins_for_white: f64,
    pub conservative_pass_and_is_root: bool,
    pub enable_passing_hacks: bool,
    pub playout_doubling_advantage: f64,
    pub nn_policy_temperature: f32,
    pub avoid_mytdagger_hack: bool,
    /// If no symmetry is specified, it will use default or random based on config, unless node is already cached.
    pub symmetry: i32,
    pub policy_optimism: f64,
    pub max_history: i32,
}

impl Default for MiscNNInputParams {
    fn default() -> Self {
        Self {
            draw_equivalent_wins_for_white: 0.5,
            conservative_pass_and_is_root: false,
            enable_passing_hacks: false,
            playout_doubling_advantage: 0.0,
            nn_policy_temperature: 1.0,
            avoid_mytdagger_hack: false,
            symmetry: nn_inputs::SYMMETRY_NOT_SPECIFIED,
            policy_optimism: 0.0,
            max_history: 1000,
        }
    }
}

impl MiscNNInputParams {
    /// Zobrist salt for the conservative-pass flag.
    pub const ZOBRIST_CONSERVATIVE_PASS: Hash128 =
        Hash128::new(0x0c2b96f4b8ae2da9, 0x5a14dee208fec0ed);
    /// Zobrist salt for the friendly-pass flag.
    pub const ZOBRIST_FRIENDLY_PASS: Hash128 = Hash128::new(0xe750505a66f7c5c2, 0x7a83139bf632d6c4);
    /// Zobrist salt for passing-hacks enablement.
    pub const ZOBRIST_PASSING_HACKS: Hash128 = Hash128::new(0x9c89f4fd3ce5a92c, 0x268c9aff79c64d00);
    /// Zobrist salt for playout-doubling advantage.
    pub const ZOBRIST_PLAYOUT_DOUBLINGS: Hash128 =
        Hash128::new(0xa5e6114d380bfc1d, 0x4160557f1222f4ad);
    /// Zobrist salt for NN policy temperature.
    pub const ZOBRIST_NN_POLICY_TEMP: Hash128 =
        Hash128::new(0xebcbdfeec6f4334b, 0xb85e43ee243b5ad2);
    /// Zobrist salt for the avoid-MYTDagger hack.
    pub const ZOBRIST_AVOID_MYTDAGGER_HACK: Hash128 =
        Hash128::new(0x612d22ec402ce054, 0x0db915c49de527ae);
    /// Zobrist salt for policy optimism.
    pub const ZOBRIST_POLICY_OPTIMISM: Hash128 =
        Hash128::new(0x88415c85c2801955, 0x39bdf76b2aaa5eb1);
    /// Zobrist salt for zero-history masking.
    pub const ZOBRIST_ZERO_HISTORY: Hash128 = Hash128::new(0x78f02afdd1aa4910, 0xda78d550486fe978);
}

/// The output of a neural-net evaluation for a single position.
///
/// Values are stored from white's perspective after `nneval.cpp`-style fixup.
#[derive(Debug, Clone)]
pub struct NNOutput {
    /// Hash of the inputs that produced this output.
    pub nn_hash: Hash128,

    /// Categorical probabilities for each game outcome.
    pub white_win_prob: f32,
    pub white_loss_prob: f32,
    pub white_no_result_prob: f32,

    /// First two moments of the believed final score distribution, from white's perspective.
    pub white_score_mean: f32,
    pub white_score_mean_sq: f32,
    /// Points to make the game fair.
    pub white_lead: f32,
    /// Expected arrival time of remaining game variance, in turns (model version >= 9).
    pub var_time_left: f32,
    /// Short-term winloss error estimate.
    pub shortterm_winloss_error: f32,
    /// Short-term score error estimate.
    pub shortterm_score_error: f32,

    /// Policy logits indexed by neural-network position. Negative values mark illegal moves.
    pub policy_probs: [f32; nn_pos::MAX_NN_POLICY_SIZE],
    /// The optimism value used for this evaluation.
    pub policy_optimism_used: f32,

    pub nn_x_len: i32,
    pub nn_y_len: i32,

    /// Optional ownership map of length `nn_x_len * nn_y_len`.
    pub white_owner_map: Option<Box<[f32]>>,
    /// Optional policy with dirichlet or other noise adjustments.
    pub noised_policy_probs: Option<Box<[f32]>>,
}

impl Default for NNOutput {
    fn default() -> Self {
        Self {
            nn_hash: Hash128::default(),
            white_win_prob: 0.0,
            white_loss_prob: 0.0,
            white_no_result_prob: 0.0,
            white_score_mean: 0.0,
            white_score_mean_sq: 0.0,
            white_lead: 0.0,
            var_time_left: 0.0,
            shortterm_winloss_error: 0.0,
            shortterm_score_error: 0.0,
            policy_probs: [0.0f32; nn_pos::MAX_NN_POLICY_SIZE],
            policy_optimism_used: 0.0,
            nn_x_len: 0,
            nn_y_len: 0,
            white_owner_map: None,
            noised_policy_probs: None,
        }
    }
}

/// Set a single value in a spatial feature row.
///
/// `row_bin` is laid out according to `pos_stride` and `feature_stride`.
/// This matches `setRowBin` in `cpp/neuralnet/nninputs.cpp`.
pub fn set_row_bin(
    row_bin: &mut [f32],
    pos: i32,
    feature: i32,
    value: f32,
    pos_stride: i32,
    feature_stride: i32,
) {
    row_bin[(pos * pos_stride + feature * feature_stride) as usize] = value;
}

/// Compute the neural-net input hash for a position.
///
/// Mirrors `NNInputs::getHash` in `cpp/neuralnet/nninputs.cpp`.
pub fn get_hash(
    board: &Board,
    hist: &BoardHistory,
    next_player: Player,
    nn_input_params: &MiscNNInputParams,
) -> Hash128 {
    let mut hash = BoardHistory::get_situation_rules_and_ko_hash(
        board,
        hist,
        next_player,
        nn_input_params.draw_equivalent_wins_for_white,
    );

    if hist.pass_would_end_phase(board, next_player) {
        hash ^= Board::ZOBRIST_PASS_ENDS_PHASE;
        if nn_input_params.conservative_pass_and_is_root {
            hash ^= MiscNNInputParams::ZOBRIST_CONSERVATIVE_PASS;
        }
        if hist.should_suppress_end_game_from_friendly_pass(board, next_player) {
            hash ^= MiscNNInputParams::ZOBRIST_FRIENDLY_PASS;
        }
        if nn_input_params.enable_passing_hacks {
            hash ^= MiscNNInputParams::ZOBRIST_PASSING_HACKS;
        }
    }

    if hist.is_game_finished || hist.is_past_normal_phase_end {
        hash ^= Board::ZOBRIST_GAME_IS_OVER;
    }

    if nn_input_params.max_history <= 0 {
        hash.hash0 = hash
            .hash0
            .wrapping_add(MiscNNInputParams::ZOBRIST_ZERO_HISTORY.hash0);
        hash.hash1 = hash
            .hash1
            .wrapping_add(MiscNNInputParams::ZOBRIST_ZERO_HISTORY.hash1);
    }

    if nn_input_params.playout_doubling_advantage != 0.0 {
        let playout_doublings_discretized =
            (nn_input_params.playout_doubling_advantage * 256.0) as i64;
        hash.hash0 = hash
            .hash0
            .wrapping_add(hash::split_mix64(playout_doublings_discretized as u64));
        hash.hash1 = hash
            .hash1
            .wrapping_add(hash::basic_l_cong(playout_doublings_discretized as u64));
        hash ^= MiscNNInputParams::ZOBRIST_PLAYOUT_DOUBLINGS;
    }

    if nn_input_params.nn_policy_temperature != 1.0f32 {
        let nn_policy_temperature_discretized =
            (nn_input_params.nn_policy_temperature * 2048.0f32) as i64;
        hash.hash0 ^= hash::basic_l_cong2(nn_policy_temperature_discretized as u64);
        hash.hash1 = hash::split_mix64(
            hash.hash1
                .wrapping_add(nn_policy_temperature_discretized as u64),
        );
        hash.hash0 = hash.hash0.wrapping_add(hash.hash1);
        hash ^= MiscNNInputParams::ZOBRIST_NN_POLICY_TEMP;
    }

    if nn_input_params.avoid_mytdagger_hack {
        hash ^= MiscNNInputParams::ZOBRIST_AVOID_MYTDAGGER_HACK;
    }

    if nn_input_params.policy_optimism > 0.0 {
        hash ^= MiscNNInputParams::ZOBRIST_POLICY_OPTIMISM;
        let policy_optimism_discretized = (nn_input_params.policy_optimism * 1024.0) as i64;
        hash.hash0 = hash::rrmxmx(
            hash::split_mix64(hash.hash0).wrapping_add(policy_optimism_discretized as u64),
        );
        hash.hash1 = hash::rrmxmx(
            hash.hash1
                .wrapping_add(hash.hash0)
                .wrapping_add(policy_optimism_discretized as u64),
        );
    }

    hash
}

/// Fill a per-location scoring array from an ownership/area estimate.
///
/// Mirrors `NNInputs::fillScoring` in `cpp/neuralnet/nninputs.cpp`.
/// `area` and `scoring` must both have length [`MAX_ARR_SIZE`].
/// When `group_tax` is false, scoring is simply +1 for white, -1 for black,
/// 0 for empty. When `group_tax` is true, small groups of empty/opponent
/// points attached to a colored group are scored fractionally.
pub fn fill_scoring(board: &Board, area: &[Color], group_tax: bool, scoring: &mut [f32]) {
    assert_eq!(area.len(), MAX_ARR_SIZE);
    assert_eq!(scoring.len(), MAX_ARR_SIZE);

    if !group_tax {
        scoring.fill(0.0f32);
        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let loc = kata_game::board::location::get_loc(x, y, board.x_size) as usize;
                let area_color = area[loc];
                scoring[loc] = if area_color == C_BLACK {
                    -1.0f32
                } else if area_color == C_WHITE {
                    1.0f32
                } else {
                    assert_eq!(area_color, C_EMPTY);
                    0.0f32
                };
            }
        }
    } else {
        let mut visited = [false; MAX_ARR_SIZE];
        scoring.fill(0.0f32);

        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let loc = kata_game::board::location::get_loc(x, y, board.x_size) as usize;
                if visited[loc] {
                    continue;
                }
                let area_color = area[loc];
                if area_color == C_BLACK || area_color == C_WHITE {
                    let full_value = if area_color == C_WHITE {
                        1.0f32
                    } else {
                        -1.0f32
                    };
                    let mut queue = Vec::with_capacity(MAX_ARR_SIZE);
                    queue.push(loc as Loc);
                    visited[loc] = true;

                    // First pass: count non-group points in the connected area.
                    let mut territory_count = 0i32;
                    let mut head = 0usize;
                    while head < queue.len() {
                        let current = queue[head];
                        let current_idx = current as usize;
                        head += 1;
                        if board.colors[current_idx] != area_color {
                            territory_count += 1;
                        }
                        for i in 0..4 {
                            let adj = (current + board.adj_offsets()[i]) as usize;
                            if area[adj] == area_color && !visited[adj] {
                                visited[adj] = true;
                                queue.push(adj as Loc);
                            }
                        }
                    }

                    // Second pass: write fractional territory value or full value.
                    let territory_value = if territory_count <= 2 {
                        0.0f32
                    } else {
                        full_value * (territory_count as f32 - 2.0f32) / territory_count as f32
                    };
                    for &next in &queue {
                        let next = next as usize;
                        if board.colors[next] != area_color {
                            scoring[next] = territory_value;
                        } else {
                            scoring[next] = full_value;
                        }
                    }
                } else {
                    assert_eq!(area_color, C_EMPTY);
                    scoring[loc] = 0.0f32;
                }
            }
        }
    }
}

/// Enumerate ladder-captured groups on `board`.
///
/// Calls `f(loc, pos, working_moves)` for every chain with 1 or 2 liberties
/// that is found to be in inescapable atari. `pos` is the flat neural-network
/// position of `loc`. For two-liberty ladders, `working_moves` contains the
/// attacker moves that succeed.
///
/// Mirrors `iterLadders` in `cpp/neuralnet/nninputs.cpp`.
pub fn iter_ladders(board: &Board, nn_x_len: i32, f: &mut dyn FnMut(Loc, i32, &[Loc])) {
    let x_size = board.x_size;
    let y_size = board.y_size;

    let mut chain_heads_solved: Vec<Loc> = Vec::new();
    let mut chain_heads_solved_value: Vec<bool> = Vec::new();
    let copy = board.clone();
    let mut working_moves = Vec::new();

    for y in 0..y_size {
        for x in 0..x_size {
            let loc = kata_game::board::location::get_loc(x, y, x_size);
            let stone = board.colors[loc as usize];
            if stone == C_BLACK || stone == C_WHITE {
                let libs = board.get_num_liberties(loc);
                if libs == 1 || libs == 2 {
                    let pos = nn_pos::xy_to_pos(x, y, nn_x_len);
                    let head = board.chain_head[loc as usize];

                    let mut already_solved = false;
                    for i in 0..chain_heads_solved.len() {
                        if chain_heads_solved[i] == head {
                            already_solved = true;
                            if chain_heads_solved_value[i] {
                                working_moves.clear();
                                f(loc, pos, &working_moves);
                            }
                            break;
                        }
                    }

                    if !already_solved {
                        let laddered = if libs == 1 {
                            working_moves.clear();
                            copy.search_is_ladder_captured(loc, true)
                        } else {
                            working_moves.clear();
                            let (captured, moves) =
                                copy.search_is_ladder_captured_attacker_first_2_libs(loc);
                            working_moves.extend_from_slice(&moves);
                            captured
                        };

                        chain_heads_solved.push(head);
                        chain_heads_solved_value.push(laddered);
                        if laddered {
                            f(loc, pos, &working_moves);
                        }
                    }
                }
            }
        }
    }
}

/// Fill a neural-net input row for input version 5.
///
/// Mirrors `NNInputs::fillRowV5` in `cpp/neuralnet/nninputs.cpp`.
/// `row_bin` must have length `NUM_FEATURES_SPATIAL_V5 * nn_x_len * nn_y_len`
/// and `row_global` must have length `NUM_FEATURES_GLOBAL_V5`.
#[allow(clippy::too_many_arguments)]
pub fn fill_row_v5(
    board: &Board,
    hist: &BoardHistory,
    next_player: Player,
    nn_input_params: &MiscNNInputParams,
    nn_x_len: i32,
    nn_y_len: i32,
    use_nhwc: bool,
    row_bin: &mut [f32],
    row_global: &mut [f32],
) {
    assert!(nn_x_len <= nn_pos::MAX_BOARD_LEN as i32);
    assert!(nn_y_len <= nn_pos::MAX_BOARD_LEN as i32);
    assert!(board.x_size <= nn_x_len);
    assert!(board.y_size <= nn_y_len);
    row_bin.fill(0.0f32);
    row_global.fill(0.0f32);

    let pla = next_player;
    let opp = if pla == P_BLACK { P_WHITE } else { P_BLACK };
    let x_size = board.x_size;
    let y_size = board.y_size;

    let (feature_stride, pos_stride) = if use_nhwc {
        (1, NUM_FEATURES_SPATIAL_V5)
    } else {
        (nn_x_len * nn_y_len, 1)
    };

    for y in 0..y_size {
        for x in 0..x_size {
            let pos = nn_pos::xy_to_pos(x, y, nn_x_len);
            let loc = kata_game::board::location::get_loc(x, y, x_size);

            // Feature 0 - on board
            set_row_bin(row_bin, pos, 0, 1.0f32, pos_stride, feature_stride);

            let stone = board.colors[loc as usize];

            // Features 1,2 - pla,opp stone
            if stone == pla as Color {
                set_row_bin(row_bin, pos, 1, 1.0f32, pos_stride, feature_stride);
            } else if stone == opp as Color {
                set_row_bin(row_bin, pos, 2, 1.0f32, pos_stride, feature_stride);
            }
        }
    }

    // Feature 3 - ko-ban locations, including possibly superko.
    if hist.encore_phase == 0 {
        if board.ko_loc != NULL_LOC {
            let pos = nn_pos::loc_to_pos(board.ko_loc, x_size, nn_x_len, nn_y_len);
            set_row_bin(row_bin, pos, 3, 1.0f32, pos_stride, feature_stride);
        }
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                if hist.is_super_ko_banned(loc) && loc != board.ko_loc {
                    let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                    set_row_bin(row_bin, pos, 3, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    } else {
        // Features 3,4,5 - in the encore, ko-related prohibitions.
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                if hist.is_super_ko_banned(loc) {
                    set_row_bin(row_bin, pos, 3, 1.0f32, pos_stride, feature_stride);
                }
                if hist.is_ko_recap_blocked(loc) {
                    set_row_bin(row_bin, pos, 4, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    }

    // Hide history from the net if a pass would end things and we're behaving as if it won't.
    let hide_history = hist.is_game_finished
        || hist.is_past_normal_phase_end
        || (hist.pass_would_end_game(board, next_player)
            && (nn_input_params.conservative_pass_and_is_root
                || hist.should_suppress_end_game_from_friendly_pass(board, next_player)));

    // Features 6,7,8,9,10 - recent move history.
    if !hide_history {
        let move_history = &hist.move_history;
        let move_history_len = move_history.len();
        if move_history_len >= 1 && move_history[move_history_len - 1].pla == opp {
            fill_history_move(
                row_bin,
                row_global,
                &move_history[move_history_len - 1],
                0,
                6,
                x_size,
                nn_x_len,
                nn_y_len,
                pos_stride,
                feature_stride,
            );
            if move_history_len >= 2 && move_history[move_history_len - 2].pla == pla {
                fill_history_move(
                    row_bin,
                    row_global,
                    &move_history[move_history_len - 2],
                    1,
                    7,
                    x_size,
                    nn_x_len,
                    nn_y_len,
                    pos_stride,
                    feature_stride,
                );
                if move_history_len >= 3 && move_history[move_history_len - 3].pla == opp {
                    fill_history_move(
                        row_bin,
                        row_global,
                        &move_history[move_history_len - 3],
                        2,
                        8,
                        x_size,
                        nn_x_len,
                        nn_y_len,
                        pos_stride,
                        feature_stride,
                    );
                    if move_history_len >= 4 && move_history[move_history_len - 4].pla == pla {
                        fill_history_move(
                            row_bin,
                            row_global,
                            &move_history[move_history_len - 4],
                            3,
                            9,
                            x_size,
                            nn_x_len,
                            nn_y_len,
                            pos_stride,
                            feature_stride,
                        );
                        if move_history_len >= 5 && move_history[move_history_len - 5].pla == opp {
                            fill_history_move(
                                row_bin,
                                row_global,
                                &move_history[move_history_len - 5],
                                4,
                                10,
                                x_size,
                                nn_x_len,
                                nn_y_len,
                                pos_stride,
                                feature_stride,
                            );
                        }
                    }
                }
            }
        }
    }

    // Features 11, 12 - second encore starting stones.
    if hist.encore_phase >= 2 {
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                let color = hist.second_encore_start_color(loc);
                if color == pla as Color {
                    set_row_bin(row_bin, pos, 11, 1.0f32, pos_stride, feature_stride);
                } else if color == opp as Color {
                    set_row_bin(row_bin, pos, 12, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    }

    // Global features.
    // Komi and any score adjustments.
    let mut self_komi =
        hist.current_self_komi(next_player, nn_input_params.draw_equivalent_wins_for_white);
    let b_area = (x_size * y_size) as f32;
    self_komi = self_komi.clamp(-b_area - 1.0f32, b_area + 1.0f32);
    row_global[5] = self_komi / 15.0f32;

    // Ko rule.
    match hist.rules.ko_rule {
        KoRule::Simple => {}
        KoRule::Positional | KoRule::Spight => {
            row_global[6] = 1.0f32;
            row_global[7] = 0.5f32;
        }
        KoRule::Situational => {
            row_global[6] = 1.0f32;
            row_global[7] = -0.5f32;
        }
    }

    // Suicide.
    if hist.rules.multi_stone_suicide_legal {
        row_global[8] = 1.0f32;
    }

    // Scoring.
    if hist.rules.scoring_rule == ScoringRule::Territory {
        row_global[9] = 1.0f32;
    }

    // Encore phase.
    if hist.encore_phase > 0 {
        row_global[10] = 1.0f32;
    }
    if hist.encore_phase > 1 {
        row_global[11] = 1.0f32;
    }
}

/// Fill a neural-net input row for input version 7.
///
/// Mirrors `NNInputs::fillRowV7` in `cpp/neuralnet/nninputs.cpp`.
/// `row_bin` must have length `NUM_FEATURES_SPATIAL_V7 * nn_x_len * nn_y_len`
/// and `row_global` must have length `NUM_FEATURES_GLOBAL_V7`.
#[allow(clippy::too_many_arguments)]
pub fn fill_row_v7(
    board: &Board,
    hist: &BoardHistory,
    next_player: Player,
    nn_input_params: &MiscNNInputParams,
    nn_x_len: i32,
    nn_y_len: i32,
    use_nhwc: bool,
    row_bin: &mut [f32],
    row_global: &mut [f32],
) {
    assert!(nn_x_len <= nn_pos::MAX_BOARD_LEN as i32);
    assert!(nn_y_len <= nn_pos::MAX_BOARD_LEN as i32);
    assert!(board.x_size <= nn_x_len);
    assert!(board.y_size <= nn_y_len);
    row_bin.fill(0.0f32);
    row_global.fill(0.0f32);

    let pla = next_player;
    let opp = if pla == P_BLACK { P_WHITE } else { P_BLACK };
    let x_size = board.x_size;
    let y_size = board.y_size;

    let (feature_stride, pos_stride) = if use_nhwc {
        (1, NUM_FEATURES_SPATIAL_V7)
    } else {
        (nn_x_len * nn_y_len, 1)
    };

    for y in 0..y_size {
        for x in 0..x_size {
            let pos = nn_pos::xy_to_pos(x, y, nn_x_len);
            let loc = kata_game::board::location::get_loc(x, y, x_size);

            // Feature 0 - on board.
            set_row_bin(row_bin, pos, 0, 1.0f32, pos_stride, feature_stride);

            let stone = board.colors[loc as usize];

            // Features 1, 2 - pla, opp stone.
            if stone == pla as Color {
                set_row_bin(row_bin, pos, 1, 1.0f32, pos_stride, feature_stride);
            } else if stone == opp as Color {
                set_row_bin(row_bin, pos, 2, 1.0f32, pos_stride, feature_stride);
            }

            // Features 3, 4, 5 - 1, 2, 3 libs.
            if stone == pla as Color || stone == opp as Color {
                let libs = board.get_num_liberties(loc);
                if libs == 1 {
                    set_row_bin(row_bin, pos, 3, 1.0f32, pos_stride, feature_stride);
                } else if libs == 2 {
                    set_row_bin(row_bin, pos, 4, 1.0f32, pos_stride, feature_stride);
                } else if libs == 3 {
                    set_row_bin(row_bin, pos, 5, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    }

    // Feature 6 - ko-ban locations, including possibly superko.
    if hist.encore_phase == 0 {
        if board.ko_loc != NULL_LOC {
            let pos = nn_pos::loc_to_pos(board.ko_loc, x_size, nn_x_len, nn_y_len);
            set_row_bin(row_bin, pos, 6, 1.0f32, pos_stride, feature_stride);
        }
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                if hist.is_super_ko_banned(loc) && loc != board.ko_loc {
                    let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                    set_row_bin(row_bin, pos, 6, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    } else {
        // Features 6, 7, 8 - in the encore, ko-related prohibitions.
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                if hist.is_super_ko_banned(loc) {
                    set_row_bin(row_bin, pos, 6, 1.0f32, pos_stride, feature_stride);
                }
                if hist.is_ko_recap_blocked(loc) {
                    set_row_bin(row_bin, pos, 7, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    }

    // Features 18, 19 - current territory, not counting group tax.
    let mut area = [C_EMPTY; MAX_ARR_SIZE];
    let mut has_area_feature = false;
    let mut group_tax_adjustment_for_pla = 0;

    if hist.rules.scoring_rule == ScoringRule::Area && hist.rules.tax_rule == TaxRule::None {
        has_area_feature = true;
        board.calculate_area(
            &mut area,
            true, // non_pass_alive_stones
            true, // safe_big_territories
            true, // unsafe_big_territories
            hist.rules.multi_stone_suicide_legal,
        );
    } else {
        let mut keep_territories = false;
        let mut keep_stones = false;
        if hist.rules.scoring_rule == ScoringRule::Area
            && (hist.rules.tax_rule == TaxRule::Seki || hist.rules.tax_rule == TaxRule::All)
        {
            has_area_feature = true;
            keep_territories = false;
            keep_stones = true;
        } else if hist.rules.scoring_rule == ScoringRule::Territory
            && hist.rules.tax_rule == TaxRule::None
        {
            if hist.encore_phase >= 2 {
                has_area_feature = true;
                keep_territories = true;
                keep_stones = false;
            }
        } else if hist.rules.scoring_rule == ScoringRule::Territory
            && (hist.rules.tax_rule == TaxRule::Seki || hist.rules.tax_rule == TaxRule::All)
        {
            if hist.encore_phase >= 2 {
                has_area_feature = true;
                keep_territories = false;
                keep_stones = false;
            }
        } else {
            unreachable!("fill_row_v7: unsupported rule combination");
        }

        if has_area_feature {
            let white_minus_black_independent_life_region_count = board
                .calculate_independent_life_area(
                    &mut area,
                    keep_territories,
                    keep_stones,
                    hist.rules.multi_stone_suicide_legal,
                );
            if hist.rules.tax_rule == TaxRule::All {
                group_tax_adjustment_for_pla = if pla == P_WHITE {
                    -2 * white_minus_black_independent_life_region_count
                } else {
                    2 * white_minus_black_independent_life_region_count
                };
            }
        }
    }

    let mut final_phase_and_game_end_would_not_be_win = false;
    if has_area_feature {
        let mut board_score_for_pla = group_tax_adjustment_for_pla;
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                if area[loc as usize] == pla as Color {
                    set_row_bin(row_bin, pos, 18, 1.0f32, pos_stride, feature_stride);
                    board_score_for_pla += 1;
                } else if area[loc as usize] == opp as Color {
                    set_row_bin(row_bin, pos, 19, 1.0f32, pos_stride, feature_stride);
                    board_score_for_pla -= 1;
                } else {
                    if hist.rules.scoring_rule == ScoringRule::Territory {
                        let stone = board.colors[loc as usize];
                        let start_color = hist.second_encore_start_color(loc);
                        if stone == pla as Color && start_color == pla as Color {
                            set_row_bin(row_bin, pos, 18, 1.0f32, pos_stride, feature_stride);
                            board_score_for_pla += 1;
                        } else if stone == opp as Color && start_color == opp as Color {
                            set_row_bin(row_bin, pos, 19, 1.0f32, pos_stride, feature_stride);
                            board_score_for_pla -= 1;
                        }
                    }
                }
            }
        }

        let self_komi = hist.current_self_komi(pla, nn_input_params.draw_equivalent_wins_for_white);
        let final_score_pla = board_score_for_pla as f32 + self_komi;
        if final_score_pla <= 0.0f32 {
            final_phase_and_game_end_would_not_be_win = true;
        }
    }

    // Determine how much history to include.
    let mut max_turns_of_history_to_include = 5;
    let mut suppress_pass_would_end_phase = false;
    if hist.pass_would_end_game(board, next_player)
        && (nn_input_params.conservative_pass_and_is_root
            || hist.should_suppress_end_game_from_friendly_pass(board, next_player)
            || (nn_input_params.enable_passing_hacks && final_phase_and_game_end_would_not_be_win))
    {
        max_turns_of_history_to_include = 0;
        suppress_pass_would_end_phase = true;
    } else if hist.is_game_finished || hist.is_past_normal_phase_end {
        max_turns_of_history_to_include = 1;
    }
    max_turns_of_history_to_include =
        max_turns_of_history_to_include.min(nn_input_params.max_history);

    let mut num_turns_of_history_included = 0;

    // Features 9, 10, 11, 12, 13 - recent move history.
    if max_turns_of_history_to_include > 0 {
        let move_history = &hist.move_history;
        let move_history_len = move_history.len();
        assert!(move_history_len as i32 >= hist.num_approx_valid_turns_this_phase);

        let amount_of_history_to_try_to_use =
            max_turns_of_history_to_include.min(hist.num_approx_valid_turns_this_phase);

        if amount_of_history_to_try_to_use >= 1 && move_history[move_history_len - 1].pla == opp {
            let prev1_loc = move_history[move_history_len - 1].loc;
            num_turns_of_history_included = 1;
            if prev1_loc == PASS_LOC {
                row_global[0] = 1.0f32;
            } else if prev1_loc != NULL_LOC {
                let pos = nn_pos::loc_to_pos(prev1_loc, x_size, nn_x_len, nn_y_len);
                set_row_bin(row_bin, pos, 9, 1.0f32, pos_stride, feature_stride);
            }
            if amount_of_history_to_try_to_use >= 2 && move_history[move_history_len - 2].pla == pla
            {
                let prev2_loc = move_history[move_history_len - 2].loc;
                num_turns_of_history_included = 2;
                if prev2_loc == PASS_LOC {
                    row_global[1] = 1.0f32;
                } else if prev2_loc != NULL_LOC {
                    let pos = nn_pos::loc_to_pos(prev2_loc, x_size, nn_x_len, nn_y_len);
                    set_row_bin(row_bin, pos, 10, 1.0f32, pos_stride, feature_stride);
                }
                if amount_of_history_to_try_to_use >= 3
                    && move_history[move_history_len - 3].pla == opp
                {
                    let prev3_loc = move_history[move_history_len - 3].loc;
                    num_turns_of_history_included = 3;
                    if prev3_loc == PASS_LOC {
                        row_global[2] = 1.0f32;
                    } else if prev3_loc != NULL_LOC {
                        let pos = nn_pos::loc_to_pos(prev3_loc, x_size, nn_x_len, nn_y_len);
                        set_row_bin(row_bin, pos, 11, 1.0f32, pos_stride, feature_stride);
                    }
                    if amount_of_history_to_try_to_use >= 4
                        && move_history[move_history_len - 4].pla == pla
                    {
                        let prev4_loc = move_history[move_history_len - 4].loc;
                        num_turns_of_history_included = 4;
                        if prev4_loc == PASS_LOC {
                            row_global[3] = 1.0f32;
                        } else if prev4_loc != NULL_LOC {
                            let pos = nn_pos::loc_to_pos(prev4_loc, x_size, nn_x_len, nn_y_len);
                            set_row_bin(row_bin, pos, 12, 1.0f32, pos_stride, feature_stride);
                        }
                        if amount_of_history_to_try_to_use >= 5
                            && move_history[move_history_len - 5].pla == opp
                        {
                            let prev5_loc = move_history[move_history_len - 5].loc;
                            num_turns_of_history_included = 5;
                            if prev5_loc == PASS_LOC {
                                row_global[4] = 1.0f32;
                            } else if prev5_loc != NULL_LOC {
                                let pos = nn_pos::loc_to_pos(prev5_loc, x_size, nn_x_len, nn_y_len);
                                set_row_bin(row_bin, pos, 13, 1.0f32, pos_stride, feature_stride);
                            }
                        }
                    }
                }
            }
        }
    }

    // Ladder features 14, 15, 16, 17.
    let mut add_ladder_feature = |loc: Loc, pos: i32, working_moves: &[Loc]| {
        debug_assert!(
            board.colors[loc as usize] == C_BLACK || board.colors[loc as usize] == C_WHITE
        );
        debug_assert!(pos >= 0 && pos < nn_pos::MAX_BOARD_AREA as i32);
        set_row_bin(row_bin, pos, 14, 1.0f32, pos_stride, feature_stride);
        if board.colors[loc as usize] == opp as Color && board.get_num_liberties(loc) > 1 {
            for &working_move in working_moves {
                let working_pos = nn_pos::loc_to_pos(working_move, x_size, nn_x_len, nn_y_len);
                set_row_bin(row_bin, working_pos, 17, 1.0f32, pos_stride, feature_stride);
            }
        }
    };
    iter_ladders(board, nn_x_len, &mut add_ladder_feature);

    let prev_board = if num_turns_of_history_included < 1 {
        board
    } else {
        hist.get_recent_board(1)
    };
    let mut add_prev_ladder_feature = |loc: Loc, pos: i32, _working_moves: &[Loc]| {
        debug_assert!(
            prev_board.colors[loc as usize] == C_BLACK
                || prev_board.colors[loc as usize] == C_WHITE
        );
        debug_assert!(pos >= 0 && pos < nn_pos::MAX_BOARD_AREA as i32);
        set_row_bin(row_bin, pos, 15, 1.0f32, pos_stride, feature_stride);
    };
    iter_ladders(prev_board, nn_x_len, &mut add_prev_ladder_feature);

    let prev_prev_board = if num_turns_of_history_included < 2 {
        prev_board
    } else {
        hist.get_recent_board(2)
    };
    let mut add_prev_prev_ladder_feature = |loc: Loc, pos: i32, _working_moves: &[Loc]| {
        debug_assert!(
            prev_prev_board.colors[loc as usize] == C_BLACK
                || prev_prev_board.colors[loc as usize] == C_WHITE
        );
        debug_assert!(pos >= 0 && pos < nn_pos::MAX_BOARD_AREA as i32);
        set_row_bin(row_bin, pos, 16, 1.0f32, pos_stride, feature_stride);
    };
    iter_ladders(prev_prev_board, nn_x_len, &mut add_prev_prev_ladder_feature);

    // Features 20, 21 - second encore starting stones.
    if hist.encore_phase >= 2 {
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                let color = hist.second_encore_start_color(loc);
                if color == pla as Color {
                    set_row_bin(row_bin, pos, 20, 1.0f32, pos_stride, feature_stride);
                } else if color == opp as Color {
                    set_row_bin(row_bin, pos, 21, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    }

    // Global features.
    // The first 5 were set already above to flag which of the past 5 moves were passes.

    // Komi and any score adjustments.
    let mut self_komi =
        hist.current_self_komi(next_player, nn_input_params.draw_equivalent_wins_for_white);
    let b_area = (x_size * y_size) as f32;
    let clip = b_area + nn_pos::KOMI_CLIP_RADIUS;
    self_komi = self_komi.clamp(-clip, clip);
    row_global[5] = self_komi / 20.0f32;

    // Ko rule.
    match hist.rules.ko_rule {
        KoRule::Simple => {}
        KoRule::Positional | KoRule::Spight => {
            row_global[6] = 1.0f32;
            row_global[7] = 0.5f32;
        }
        KoRule::Situational => {
            row_global[6] = 1.0f32;
            row_global[7] = -0.5f32;
        }
    }

    // Suicide.
    if hist.rules.multi_stone_suicide_legal {
        row_global[8] = 1.0f32;
    }

    // Scoring.
    if hist.rules.scoring_rule == ScoringRule::Territory {
        row_global[9] = 1.0f32;
    }

    // Tax.
    match hist.rules.tax_rule {
        TaxRule::None => {}
        TaxRule::Seki => {
            row_global[10] = 1.0f32;
        }
        TaxRule::All => {
            row_global[10] = 1.0f32;
            row_global[11] = 1.0f32;
        }
    }

    // Encore phase.
    if hist.encore_phase > 0 {
        row_global[12] = 1.0f32;
    }
    if hist.encore_phase > 1 {
        row_global[13] = 1.0f32;
    }

    // Does a pass end the current phase given the ruleset and history?
    let pass_would_end_phase = if suppress_pass_would_end_phase {
        false
    } else {
        hist.pass_would_end_phase(board, next_player)
    };
    row_global[14] = if pass_would_end_phase { 1.0f32 } else { 0.0f32 };

    // Handicap play.
    if nn_input_params.playout_doubling_advantage != 0.0 {
        row_global[15] = 1.0f32;
        row_global[16] = (0.5 * nn_input_params.playout_doubling_advantage) as f32;
    }

    // Button.
    if hist.has_button {
        row_global[17] = 1.0f32;
    }

    // Score-belief parity wave.
    if hist.rules.scoring_rule == ScoringRule::Area || hist.encore_phase >= 2 {
        let board_area_is_even = (x_size * y_size) % 2 == 0;
        let drawable_komis_are_even = board_area_is_even;

        let komi_floor = if drawable_komis_are_even {
            (self_komi / 2.0f32).floor() * 2.0f32
        } else {
            ((self_komi - 1.0f32) / 2.0f32).floor() * 2.0f32 + 1.0f32
        };

        let delta = (self_komi - komi_floor).clamp(0.0f32, 2.0f32);

        let wave = if delta < 0.5f32 {
            delta
        } else if delta < 1.5f32 {
            1.0f32 - delta
        } else {
            delta - 2.0f32
        };

        row_global[18] = wave;
    }
}

/// Fill a neural-net input row for input version 3.
///
/// Mirrors `NNInputs::fillRowV3` in `cpp/neuralnet/nninputs.cpp`.
#[allow(clippy::too_many_arguments)]
pub fn fill_row_v3(
    board: &Board,
    hist: &BoardHistory,
    next_player: Player,
    nn_input_params: &MiscNNInputParams,
    nn_x_len: i32,
    nn_y_len: i32,
    use_nhwc: bool,
    row_bin: &mut [f32],
    row_global: &mut [f32],
) {
    assert!(nn_x_len <= nn_pos::MAX_BOARD_LEN as i32);
    assert!(nn_y_len <= nn_pos::MAX_BOARD_LEN as i32);
    assert!(board.x_size <= nn_x_len);
    assert!(board.y_size <= nn_y_len);
    row_bin.fill(0.0f32);
    row_global.fill(0.0f32);

    let pla = next_player;
    let opp = if pla == P_BLACK { P_WHITE } else { P_BLACK };
    let x_size = board.x_size;
    let y_size = board.y_size;

    let (feature_stride, pos_stride) = if use_nhwc {
        (1, NUM_FEATURES_SPATIAL_V3)
    } else {
        (nn_x_len * nn_y_len, 1)
    };

    for y in 0..y_size {
        for x in 0..x_size {
            let pos = nn_pos::xy_to_pos(x, y, nn_x_len);
            let loc = kata_game::board::location::get_loc(x, y, x_size);

            set_row_bin(row_bin, pos, 0, 1.0f32, pos_stride, feature_stride);

            let stone = board.colors[loc as usize];
            if stone == pla as Color {
                set_row_bin(row_bin, pos, 1, 1.0f32, pos_stride, feature_stride);
            } else if stone == opp as Color {
                set_row_bin(row_bin, pos, 2, 1.0f32, pos_stride, feature_stride);
            }

            if stone == pla as Color || stone == opp as Color {
                let libs = board.get_num_liberties(loc);
                if libs == 1 {
                    set_row_bin(row_bin, pos, 3, 1.0f32, pos_stride, feature_stride);
                } else if libs == 2 {
                    set_row_bin(row_bin, pos, 4, 1.0f32, pos_stride, feature_stride);
                } else if libs == 3 {
                    set_row_bin(row_bin, pos, 5, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    }

    // Feature 6 - ko-ban locations, including possibly superko.
    if hist.encore_phase == 0 {
        if board.ko_loc != NULL_LOC {
            let pos = nn_pos::loc_to_pos(board.ko_loc, x_size, nn_x_len, nn_y_len);
            set_row_bin(row_bin, pos, 6, 1.0f32, pos_stride, feature_stride);
        }
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                if hist.is_super_ko_banned(loc) && loc != board.ko_loc {
                    let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                    set_row_bin(row_bin, pos, 6, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    } else {
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                if hist.is_super_ko_banned(loc) {
                    set_row_bin(row_bin, pos, 6, 1.0f32, pos_stride, feature_stride);
                }
                if hist.is_ko_recap_blocked(loc) {
                    set_row_bin(row_bin, pos, 7, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    }

    // Hide history if a pass would end things and we're behaving as if it won't.
    let hide_history = hist.is_game_finished
        || hist.is_past_normal_phase_end
        || (hist.pass_would_end_game(board, next_player)
            && (nn_input_params.conservative_pass_and_is_root
                || hist.should_suppress_end_game_from_friendly_pass(board, next_player)));
    let mut num_turns_of_history_included = 0;

    // Features 9,10,11,12,13 - recent move history.
    if !hide_history {
        let move_history = &hist.move_history;
        let move_history_len = move_history.len();
        if move_history_len >= 1 && move_history[move_history_len - 1].pla == opp {
            let prev1_loc = move_history[move_history_len - 1].loc;
            num_turns_of_history_included = 1;
            if prev1_loc == PASS_LOC {
                row_global[0] = 1.0f32;
            } else if prev1_loc != NULL_LOC {
                let pos = nn_pos::loc_to_pos(prev1_loc, x_size, nn_x_len, nn_y_len);
                set_row_bin(row_bin, pos, 9, 1.0f32, pos_stride, feature_stride);
            }
            if move_history_len >= 2 && move_history[move_history_len - 2].pla == pla {
                let prev2_loc = move_history[move_history_len - 2].loc;
                num_turns_of_history_included = 2;
                if prev2_loc == PASS_LOC {
                    row_global[1] = 1.0f32;
                } else if prev2_loc != NULL_LOC {
                    let pos = nn_pos::loc_to_pos(prev2_loc, x_size, nn_x_len, nn_y_len);
                    set_row_bin(row_bin, pos, 10, 1.0f32, pos_stride, feature_stride);
                }
                if move_history_len >= 3 && move_history[move_history_len - 3].pla == opp {
                    let prev3_loc = move_history[move_history_len - 3].loc;
                    num_turns_of_history_included = 3;
                    if prev3_loc == PASS_LOC {
                        row_global[2] = 1.0f32;
                    } else if prev3_loc != NULL_LOC {
                        let pos = nn_pos::loc_to_pos(prev3_loc, x_size, nn_x_len, nn_y_len);
                        set_row_bin(row_bin, pos, 11, 1.0f32, pos_stride, feature_stride);
                    }
                    if move_history_len >= 4 && move_history[move_history_len - 4].pla == pla {
                        let prev4_loc = move_history[move_history_len - 4].loc;
                        num_turns_of_history_included = 4;
                        if prev4_loc == PASS_LOC {
                            row_global[3] = 1.0f32;
                        } else if prev4_loc != NULL_LOC {
                            let pos = nn_pos::loc_to_pos(prev4_loc, x_size, nn_x_len, nn_y_len);
                            set_row_bin(row_bin, pos, 12, 1.0f32, pos_stride, feature_stride);
                        }
                        if move_history_len >= 5 && move_history[move_history_len - 5].pla == opp {
                            let prev5_loc = move_history[move_history_len - 5].loc;
                            num_turns_of_history_included = 5;
                            if prev5_loc == PASS_LOC {
                                row_global[4] = 1.0f32;
                            } else if prev5_loc != NULL_LOC {
                                let pos = nn_pos::loc_to_pos(prev5_loc, x_size, nn_x_len, nn_y_len);
                                set_row_bin(row_bin, pos, 13, 1.0f32, pos_stride, feature_stride);
                            }
                        }
                    }
                }
            }
        }
    }

    // Ladder features 14,15,16,17.
    let mut add_ladder_feature = |loc: Loc, pos: i32, working_moves: &[Loc]| {
        debug_assert!(
            board.colors[loc as usize] == C_BLACK || board.colors[loc as usize] == C_WHITE
        );
        debug_assert!(pos >= 0 && pos < nn_pos::MAX_BOARD_AREA as i32);
        set_row_bin(row_bin, pos, 14, 1.0f32, pos_stride, feature_stride);
        if board.colors[loc as usize] == opp as Color && board.get_num_liberties(loc) > 1 {
            for &working_move in working_moves {
                let working_pos = nn_pos::loc_to_pos(working_move, x_size, nn_x_len, nn_y_len);
                set_row_bin(row_bin, working_pos, 17, 1.0f32, pos_stride, feature_stride);
            }
        }
    };
    iter_ladders(board, nn_x_len, &mut add_ladder_feature);

    let prev_board = if hide_history || num_turns_of_history_included < 1 {
        board
    } else {
        hist.get_recent_board(1)
    };
    let mut add_prev_ladder_feature = |loc: Loc, pos: i32, _working_moves: &[Loc]| {
        debug_assert!(
            prev_board.colors[loc as usize] == C_BLACK
                || prev_board.colors[loc as usize] == C_WHITE
        );
        debug_assert!(pos >= 0 && pos < nn_pos::MAX_BOARD_AREA as i32);
        set_row_bin(row_bin, pos, 15, 1.0f32, pos_stride, feature_stride);
    };
    iter_ladders(prev_board, nn_x_len, &mut add_prev_ladder_feature);

    let prev_prev_board = if hide_history || num_turns_of_history_included < 2 {
        prev_board
    } else {
        hist.get_recent_board(2)
    };
    let mut add_prev_prev_ladder_feature = |loc: Loc, pos: i32, _working_moves: &[Loc]| {
        debug_assert!(
            prev_prev_board.colors[loc as usize] == C_BLACK
                || prev_prev_board.colors[loc as usize] == C_WHITE
        );
        debug_assert!(pos >= 0 && pos < nn_pos::MAX_BOARD_AREA as i32);
        set_row_bin(row_bin, pos, 16, 1.0f32, pos_stride, feature_stride);
    };
    iter_ladders(prev_prev_board, nn_x_len, &mut add_prev_prev_ladder_feature);

    // Features 18,19 - current territory.
    let mut area = [C_EMPTY; MAX_ARR_SIZE];
    let (non_pass_alive_stones, safe_big_territories, unsafe_big_territories) =
        if hist.rules.scoring_rule == ScoringRule::Area {
            (true, true, true)
        } else if hist.rules.scoring_rule == ScoringRule::Territory {
            (false, true, false)
        } else {
            unreachable!("fill_row_v3: unsupported scoring rule")
        };
    board.calculate_area(
        &mut area,
        non_pass_alive_stones,
        safe_big_territories,
        unsafe_big_territories,
        hist.rules.multi_stone_suicide_legal,
    );

    for y in 0..y_size {
        for x in 0..x_size {
            let loc = kata_game::board::location::get_loc(x, y, x_size);
            let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
            if area[loc as usize] == pla as Color {
                set_row_bin(row_bin, pos, 18, 1.0f32, pos_stride, feature_stride);
            } else if area[loc as usize] == opp as Color {
                set_row_bin(row_bin, pos, 19, 1.0f32, pos_stride, feature_stride);
            }
        }
    }

    // Features 20, 21 - second encore starting stones.
    if hist.encore_phase >= 2 {
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                let color = hist.second_encore_start_color(loc);
                if color == pla as Color {
                    set_row_bin(row_bin, pos, 20, 1.0f32, pos_stride, feature_stride);
                } else if color == opp as Color {
                    set_row_bin(row_bin, pos, 21, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    }

    // Global features.
    let mut self_komi =
        hist.current_self_komi(next_player, nn_input_params.draw_equivalent_wins_for_white);
    let b_area = (x_size * y_size) as f32;
    self_komi = self_komi.clamp(-b_area - 1.0f32, b_area + 1.0f32);
    row_global[5] = self_komi / 15.0f32;

    match hist.rules.ko_rule {
        KoRule::Simple => {}
        KoRule::Positional | KoRule::Spight => {
            row_global[6] = 1.0f32;
            row_global[7] = 0.5f32;
        }
        KoRule::Situational => {
            row_global[6] = 1.0f32;
            row_global[7] = -0.5f32;
        }
    }

    if hist.rules.multi_stone_suicide_legal {
        row_global[8] = 1.0f32;
    }

    if hist.rules.scoring_rule == ScoringRule::Territory {
        row_global[9] = 1.0f32;
    }

    if hist.encore_phase > 0 {
        row_global[10] = 1.0f32;
    }
    if hist.encore_phase > 1 {
        row_global[11] = 1.0f32;
    }

    let pass_would_end_phase = if hide_history {
        false
    } else {
        hist.pass_would_end_phase(board, next_player)
    };
    row_global[12] = if pass_would_end_phase { 1.0f32 } else { 0.0f32 };

    if hist.rules.scoring_rule == ScoringRule::Area || hist.encore_phase >= 2 {
        let board_area_is_even = (x_size * y_size) % 2 == 0;
        let drawable_komis_are_even = board_area_is_even;
        let komi_floor = if drawable_komis_are_even {
            (self_komi / 2.0f32).floor() * 2.0f32
        } else {
            ((self_komi - 1.0f32) / 2.0f32).floor() * 2.0f32 + 1.0f32
        };
        let delta = (self_komi - komi_floor).clamp(0.0f32, 2.0f32);
        let wave = if delta < 0.5f32 {
            delta
        } else if delta < 1.5f32 {
            1.0f32 - delta
        } else {
            delta - 2.0f32
        };
        row_global[13] = wave;
    }
}

/// Fill a neural-net input row for input version 4.
///
/// Mirrors `NNInputs::fillRowV4` in `cpp/neuralnet/nninputs.cpp`.
#[allow(clippy::too_many_arguments)]
pub fn fill_row_v4(
    board: &Board,
    hist: &BoardHistory,
    next_player: Player,
    nn_input_params: &MiscNNInputParams,
    nn_x_len: i32,
    nn_y_len: i32,
    use_nhwc: bool,
    row_bin: &mut [f32],
    row_global: &mut [f32],
) {
    assert!(nn_x_len <= nn_pos::MAX_BOARD_LEN as i32);
    assert!(nn_y_len <= nn_pos::MAX_BOARD_LEN as i32);
    assert!(board.x_size <= nn_x_len);
    assert!(board.y_size <= nn_y_len);
    row_bin.fill(0.0f32);
    row_global.fill(0.0f32);

    let pla = next_player;
    let opp = if pla == P_BLACK { P_WHITE } else { P_BLACK };
    let x_size = board.x_size;
    let y_size = board.y_size;

    let (feature_stride, pos_stride) = if use_nhwc {
        (1, NUM_FEATURES_SPATIAL_V4)
    } else {
        (nn_x_len * nn_y_len, 1)
    };

    for y in 0..y_size {
        for x in 0..x_size {
            let pos = nn_pos::xy_to_pos(x, y, nn_x_len);
            let loc = kata_game::board::location::get_loc(x, y, x_size);

            set_row_bin(row_bin, pos, 0, 1.0f32, pos_stride, feature_stride);

            let stone = board.colors[loc as usize];
            if stone == pla as Color {
                set_row_bin(row_bin, pos, 1, 1.0f32, pos_stride, feature_stride);
            } else if stone == opp as Color {
                set_row_bin(row_bin, pos, 2, 1.0f32, pos_stride, feature_stride);
            }

            if stone == pla as Color || stone == opp as Color {
                let libs = board.get_num_liberties(loc);
                if libs == 1 {
                    set_row_bin(row_bin, pos, 3, 1.0f32, pos_stride, feature_stride);
                } else if libs == 2 {
                    set_row_bin(row_bin, pos, 4, 1.0f32, pos_stride, feature_stride);
                } else if libs == 3 {
                    set_row_bin(row_bin, pos, 5, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    }

    // Feature 6 - ko-ban locations, including possibly superko.
    if hist.encore_phase == 0 {
        if board.ko_loc != NULL_LOC {
            let pos = nn_pos::loc_to_pos(board.ko_loc, x_size, nn_x_len, nn_y_len);
            set_row_bin(row_bin, pos, 6, 1.0f32, pos_stride, feature_stride);
        }
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                if hist.is_super_ko_banned(loc) && loc != board.ko_loc {
                    let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                    set_row_bin(row_bin, pos, 6, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    } else {
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                if hist.is_super_ko_banned(loc) {
                    set_row_bin(row_bin, pos, 6, 1.0f32, pos_stride, feature_stride);
                }
                if hist.is_ko_recap_blocked(loc) {
                    set_row_bin(row_bin, pos, 7, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    }

    // Hide history if a pass would end things and we're behaving as if it won't.
    let hide_history = hist.is_game_finished
        || hist.is_past_normal_phase_end
        || (hist.pass_would_end_game(board, next_player)
            && (nn_input_params.conservative_pass_and_is_root
                || hist.should_suppress_end_game_from_friendly_pass(board, next_player)));
    let mut num_turns_of_history_included = 0;

    // Features 9,10,11,12,13 - recent move history.
    if !hide_history {
        let move_history = &hist.move_history;
        let move_history_len = move_history.len();
        if move_history_len >= 1 && move_history[move_history_len - 1].pla == opp {
            let prev1_loc = move_history[move_history_len - 1].loc;
            num_turns_of_history_included = 1;
            if prev1_loc == PASS_LOC {
                row_global[0] = 1.0f32;
            } else if prev1_loc != NULL_LOC {
                let pos = nn_pos::loc_to_pos(prev1_loc, x_size, nn_x_len, nn_y_len);
                set_row_bin(row_bin, pos, 9, 1.0f32, pos_stride, feature_stride);
            }
            if move_history_len >= 2 && move_history[move_history_len - 2].pla == pla {
                let prev2_loc = move_history[move_history_len - 2].loc;
                num_turns_of_history_included = 2;
                if prev2_loc == PASS_LOC {
                    row_global[1] = 1.0f32;
                } else if prev2_loc != NULL_LOC {
                    let pos = nn_pos::loc_to_pos(prev2_loc, x_size, nn_x_len, nn_y_len);
                    set_row_bin(row_bin, pos, 10, 1.0f32, pos_stride, feature_stride);
                }
                if move_history_len >= 3 && move_history[move_history_len - 3].pla == opp {
                    let prev3_loc = move_history[move_history_len - 3].loc;
                    num_turns_of_history_included = 3;
                    if prev3_loc == PASS_LOC {
                        row_global[2] = 1.0f32;
                    } else if prev3_loc != NULL_LOC {
                        let pos = nn_pos::loc_to_pos(prev3_loc, x_size, nn_x_len, nn_y_len);
                        set_row_bin(row_bin, pos, 11, 1.0f32, pos_stride, feature_stride);
                    }
                    if move_history_len >= 4 && move_history[move_history_len - 4].pla == pla {
                        let prev4_loc = move_history[move_history_len - 4].loc;
                        num_turns_of_history_included = 4;
                        if prev4_loc == PASS_LOC {
                            row_global[3] = 1.0f32;
                        } else if prev4_loc != NULL_LOC {
                            let pos = nn_pos::loc_to_pos(prev4_loc, x_size, nn_x_len, nn_y_len);
                            set_row_bin(row_bin, pos, 12, 1.0f32, pos_stride, feature_stride);
                        }
                        if move_history_len >= 5 && move_history[move_history_len - 5].pla == opp {
                            let prev5_loc = move_history[move_history_len - 5].loc;
                            num_turns_of_history_included = 5;
                            if prev5_loc == PASS_LOC {
                                row_global[4] = 1.0f32;
                            } else if prev5_loc != NULL_LOC {
                                let pos = nn_pos::loc_to_pos(prev5_loc, x_size, nn_x_len, nn_y_len);
                                set_row_bin(row_bin, pos, 13, 1.0f32, pos_stride, feature_stride);
                            }
                        }
                    }
                }
            }
        }
    }

    // Ladder features 14,15,16,17.
    let mut add_ladder_feature = |loc: Loc, pos: i32, working_moves: &[Loc]| {
        debug_assert!(
            board.colors[loc as usize] == C_BLACK || board.colors[loc as usize] == C_WHITE
        );
        debug_assert!(pos >= 0 && pos < nn_pos::MAX_BOARD_AREA as i32);
        set_row_bin(row_bin, pos, 14, 1.0f32, pos_stride, feature_stride);
        if board.colors[loc as usize] == opp as Color && board.get_num_liberties(loc) > 1 {
            for &working_move in working_moves {
                let working_pos = nn_pos::loc_to_pos(working_move, x_size, nn_x_len, nn_y_len);
                set_row_bin(row_bin, working_pos, 17, 1.0f32, pos_stride, feature_stride);
            }
        }
    };
    iter_ladders(board, nn_x_len, &mut add_ladder_feature);

    let prev_board = if hide_history || num_turns_of_history_included < 1 {
        board
    } else {
        hist.get_recent_board(1)
    };
    let mut add_prev_ladder_feature = |loc: Loc, pos: i32, _working_moves: &[Loc]| {
        debug_assert!(
            prev_board.colors[loc as usize] == C_BLACK
                || prev_board.colors[loc as usize] == C_WHITE
        );
        debug_assert!(pos >= 0 && pos < nn_pos::MAX_BOARD_AREA as i32);
        set_row_bin(row_bin, pos, 15, 1.0f32, pos_stride, feature_stride);
    };
    iter_ladders(prev_board, nn_x_len, &mut add_prev_ladder_feature);

    let prev_prev_board = if hide_history || num_turns_of_history_included < 2 {
        prev_board
    } else {
        hist.get_recent_board(2)
    };
    let mut add_prev_prev_ladder_feature = |loc: Loc, pos: i32, _working_moves: &[Loc]| {
        debug_assert!(
            prev_prev_board.colors[loc as usize] == C_BLACK
                || prev_prev_board.colors[loc as usize] == C_WHITE
        );
        debug_assert!(pos >= 0 && pos < nn_pos::MAX_BOARD_AREA as i32);
        set_row_bin(row_bin, pos, 16, 1.0f32, pos_stride, feature_stride);
    };
    iter_ladders(prev_prev_board, nn_x_len, &mut add_prev_prev_ladder_feature);

    // Features 18,19 - pass-alive territory and stones.
    let mut area = [C_EMPTY; MAX_ARR_SIZE];
    board.calculate_area(
        &mut area,
        false, // non_pass_alive_stones
        true,  // safe_big_territories
        false, // unsafe_big_territories
        hist.rules.multi_stone_suicide_legal,
    );

    for y in 0..y_size {
        for x in 0..x_size {
            let loc = kata_game::board::location::get_loc(x, y, x_size);
            let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
            if area[loc as usize] == pla as Color {
                set_row_bin(row_bin, pos, 18, 1.0f32, pos_stride, feature_stride);
            } else if area[loc as usize] == opp as Color {
                set_row_bin(row_bin, pos, 19, 1.0f32, pos_stride, feature_stride);
            }
        }
    }

    // Features 20, 21 - second encore starting stones.
    if hist.encore_phase >= 2 {
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                let color = hist.second_encore_start_color(loc);
                if color == pla as Color {
                    set_row_bin(row_bin, pos, 20, 1.0f32, pos_stride, feature_stride);
                } else if color == opp as Color {
                    set_row_bin(row_bin, pos, 21, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    }

    // Global features.
    let mut self_komi =
        hist.current_self_komi(next_player, nn_input_params.draw_equivalent_wins_for_white);
    let b_area = (x_size * y_size) as f32;
    self_komi = self_komi.clamp(-b_area - 1.0f32, b_area + 1.0f32);
    row_global[5] = self_komi / 15.0f32;

    match hist.rules.ko_rule {
        KoRule::Simple => {}
        KoRule::Positional | KoRule::Spight => {
            row_global[6] = 1.0f32;
            row_global[7] = 0.5f32;
        }
        KoRule::Situational => {
            row_global[6] = 1.0f32;
            row_global[7] = -0.5f32;
        }
    }

    if hist.rules.multi_stone_suicide_legal {
        row_global[8] = 1.0f32;
    }

    if hist.rules.scoring_rule == ScoringRule::Territory {
        row_global[9] = 1.0f32;
    }

    if hist.encore_phase > 0 {
        row_global[10] = 1.0f32;
    }
    if hist.encore_phase > 1 {
        row_global[11] = 1.0f32;
    }

    let pass_would_end_phase = if hide_history {
        false
    } else {
        hist.pass_would_end_phase(board, next_player)
    };
    row_global[12] = if pass_would_end_phase { 1.0f32 } else { 0.0f32 };

    if hist.rules.scoring_rule == ScoringRule::Area || hist.encore_phase >= 2 {
        let board_area_is_even = (x_size * y_size) % 2 == 0;
        let drawable_komis_are_even = board_area_is_even;
        let komi_floor = if drawable_komis_are_even {
            (self_komi / 2.0f32).floor() * 2.0f32
        } else {
            ((self_komi - 1.0f32) / 2.0f32).floor() * 2.0f32 + 1.0f32
        };
        let delta = (self_komi - komi_floor).clamp(0.0f32, 2.0f32);
        let wave = if delta < 0.5f32 {
            delta
        } else if delta < 1.5f32 {
            1.0f32 - delta
        } else {
            delta - 2.0f32
        };
        row_global[13] = wave;
    }
}

/// Fill a neural-net input row for input version 6.
///
/// Mirrors `NNInputs::fillRowV6` in `cpp/neuralnet/nninputs.cpp`.
#[allow(clippy::too_many_arguments)]
pub fn fill_row_v6(
    board: &Board,
    hist: &BoardHistory,
    next_player: Player,
    nn_input_params: &MiscNNInputParams,
    nn_x_len: i32,
    nn_y_len: i32,
    use_nhwc: bool,
    row_bin: &mut [f32],
    row_global: &mut [f32],
) {
    assert!(nn_x_len <= nn_pos::MAX_BOARD_LEN as i32);
    assert!(nn_y_len <= nn_pos::MAX_BOARD_LEN as i32);
    assert!(board.x_size <= nn_x_len);
    assert!(board.y_size <= nn_y_len);
    row_bin.fill(0.0f32);
    row_global.fill(0.0f32);

    let pla = next_player;
    let opp = if pla == P_BLACK { P_WHITE } else { P_BLACK };
    let x_size = board.x_size;
    let y_size = board.y_size;

    let (feature_stride, pos_stride) = if use_nhwc {
        (1, NUM_FEATURES_SPATIAL_V6)
    } else {
        (nn_x_len * nn_y_len, 1)
    };

    for y in 0..y_size {
        for x in 0..x_size {
            let pos = nn_pos::xy_to_pos(x, y, nn_x_len);
            let loc = kata_game::board::location::get_loc(x, y, x_size);

            set_row_bin(row_bin, pos, 0, 1.0f32, pos_stride, feature_stride);

            let stone = board.colors[loc as usize];
            if stone == pla as Color {
                set_row_bin(row_bin, pos, 1, 1.0f32, pos_stride, feature_stride);
            } else if stone == opp as Color {
                set_row_bin(row_bin, pos, 2, 1.0f32, pos_stride, feature_stride);
            }

            if stone == pla as Color || stone == opp as Color {
                let libs = board.get_num_liberties(loc);
                if libs == 1 {
                    set_row_bin(row_bin, pos, 3, 1.0f32, pos_stride, feature_stride);
                } else if libs == 2 {
                    set_row_bin(row_bin, pos, 4, 1.0f32, pos_stride, feature_stride);
                } else if libs == 3 {
                    set_row_bin(row_bin, pos, 5, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    }

    // Feature 6 - ko-ban locations, including possibly superko.
    if hist.encore_phase == 0 {
        if board.ko_loc != NULL_LOC {
            let pos = nn_pos::loc_to_pos(board.ko_loc, x_size, nn_x_len, nn_y_len);
            set_row_bin(row_bin, pos, 6, 1.0f32, pos_stride, feature_stride);
        }
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                if hist.is_super_ko_banned(loc) && loc != board.ko_loc {
                    let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                    set_row_bin(row_bin, pos, 6, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    } else {
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                if hist.is_super_ko_banned(loc) {
                    set_row_bin(row_bin, pos, 6, 1.0f32, pos_stride, feature_stride);
                }
                if hist.is_ko_recap_blocked(loc) {
                    set_row_bin(row_bin, pos, 7, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    }

    // Features 18, 19 - current territory, not counting group tax.
    let mut area = [C_EMPTY; MAX_ARR_SIZE];
    let mut has_area_feature = false;
    let mut group_tax_adjustment_for_pla = 0;

    if hist.rules.scoring_rule == ScoringRule::Area && hist.rules.tax_rule == TaxRule::None {
        has_area_feature = true;
        board.calculate_area(
            &mut area,
            true, // non_pass_alive_stones
            true, // safe_big_territories
            true, // unsafe_big_territories
            hist.rules.multi_stone_suicide_legal,
        );
    } else {
        let mut keep_territories = false;
        let mut keep_stones = false;
        if hist.rules.scoring_rule == ScoringRule::Area
            && (hist.rules.tax_rule == TaxRule::Seki || hist.rules.tax_rule == TaxRule::All)
        {
            has_area_feature = true;
            keep_territories = false;
            keep_stones = true;
        } else if hist.rules.scoring_rule == ScoringRule::Territory
            && hist.rules.tax_rule == TaxRule::None
        {
            if hist.encore_phase >= 2 {
                has_area_feature = true;
                keep_territories = true;
                keep_stones = false;
            }
        } else if hist.rules.scoring_rule == ScoringRule::Territory
            && (hist.rules.tax_rule == TaxRule::Seki || hist.rules.tax_rule == TaxRule::All)
        {
            if hist.encore_phase >= 2 {
                has_area_feature = true;
                keep_territories = false;
                keep_stones = false;
            }
        } else {
            unreachable!("fill_row_v6: unsupported rule combination");
        }

        if has_area_feature {
            let white_minus_black_independent_life_region_count = board
                .calculate_independent_life_area(
                    &mut area,
                    keep_territories,
                    keep_stones,
                    hist.rules.multi_stone_suicide_legal,
                );
            if hist.rules.tax_rule == TaxRule::All {
                group_tax_adjustment_for_pla = if pla == P_WHITE {
                    -2 * white_minus_black_independent_life_region_count
                } else {
                    2 * white_minus_black_independent_life_region_count
                };
            }
        }
    }

    let mut final_phase_and_game_end_would_not_be_win = false;
    if has_area_feature {
        let mut board_score_for_pla = group_tax_adjustment_for_pla;
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                if area[loc as usize] == pla as Color {
                    set_row_bin(row_bin, pos, 18, 1.0f32, pos_stride, feature_stride);
                    board_score_for_pla += 1;
                } else if area[loc as usize] == opp as Color {
                    set_row_bin(row_bin, pos, 19, 1.0f32, pos_stride, feature_stride);
                    board_score_for_pla -= 1;
                } else if hist.rules.scoring_rule == ScoringRule::Territory {
                    let stone = board.colors[loc as usize];
                    let start_color = hist.second_encore_start_color(loc);
                    if stone == pla as Color && start_color == pla as Color {
                        set_row_bin(row_bin, pos, 18, 1.0f32, pos_stride, feature_stride);
                        board_score_for_pla += 1;
                    } else if stone == opp as Color && start_color == opp as Color {
                        set_row_bin(row_bin, pos, 19, 1.0f32, pos_stride, feature_stride);
                        board_score_for_pla -= 1;
                    }
                }
            }
        }

        let self_komi = hist.current_self_komi(pla, nn_input_params.draw_equivalent_wins_for_white);
        let final_score_pla = board_score_for_pla as f32 + self_komi;
        if final_score_pla <= 0.0f32 {
            final_phase_and_game_end_would_not_be_win = true;
        }
    }

    // Determine how much history to include.
    let mut max_turns_of_history_to_include = 5;
    let mut suppress_pass_would_end_phase = false;
    if hist.pass_would_end_game(board, next_player)
        && (nn_input_params.conservative_pass_and_is_root
            || hist.should_suppress_end_game_from_friendly_pass(board, next_player)
            || (nn_input_params.enable_passing_hacks && final_phase_and_game_end_would_not_be_win))
    {
        max_turns_of_history_to_include = 0;
        suppress_pass_would_end_phase = true;
    } else if hist.is_game_finished || hist.is_past_normal_phase_end {
        max_turns_of_history_to_include = 1;
    }
    max_turns_of_history_to_include =
        max_turns_of_history_to_include.min(nn_input_params.max_history);

    let mut num_turns_of_history_included = 0;

    // Features 9, 10, 11, 12, 13 - recent move history.
    if max_turns_of_history_to_include > 0 {
        let move_history = &hist.move_history;
        let move_history_len = move_history.len();
        assert!(move_history_len as i32 >= hist.num_approx_valid_turns_this_phase);
        assert!(move_history_len as i32 >= hist.num_consec_valid_turns_this_game);

        let amount_of_history_to_try_to_use = max_turns_of_history_to_include
            .min(hist.num_approx_valid_turns_this_phase)
            .min(hist.num_consec_valid_turns_this_game);

        if amount_of_history_to_try_to_use >= 1 && move_history[move_history_len - 1].pla == opp {
            let prev1_loc = move_history[move_history_len - 1].loc;
            num_turns_of_history_included = 1;
            if prev1_loc == PASS_LOC {
                row_global[0] = 1.0f32;
            } else if prev1_loc != NULL_LOC {
                let pos = nn_pos::loc_to_pos(prev1_loc, x_size, nn_x_len, nn_y_len);
                set_row_bin(row_bin, pos, 9, 1.0f32, pos_stride, feature_stride);
            }
            if amount_of_history_to_try_to_use >= 2 && move_history[move_history_len - 2].pla == pla
            {
                let prev2_loc = move_history[move_history_len - 2].loc;
                num_turns_of_history_included = 2;
                if prev2_loc == PASS_LOC {
                    row_global[1] = 1.0f32;
                } else if prev2_loc != NULL_LOC {
                    let pos = nn_pos::loc_to_pos(prev2_loc, x_size, nn_x_len, nn_y_len);
                    set_row_bin(row_bin, pos, 10, 1.0f32, pos_stride, feature_stride);
                }
                if amount_of_history_to_try_to_use >= 3
                    && move_history[move_history_len - 3].pla == opp
                {
                    let prev3_loc = move_history[move_history_len - 3].loc;
                    num_turns_of_history_included = 3;
                    if prev3_loc == PASS_LOC {
                        row_global[2] = 1.0f32;
                    } else if prev3_loc != NULL_LOC {
                        let pos = nn_pos::loc_to_pos(prev3_loc, x_size, nn_x_len, nn_y_len);
                        set_row_bin(row_bin, pos, 11, 1.0f32, pos_stride, feature_stride);
                    }
                    if amount_of_history_to_try_to_use >= 4
                        && move_history[move_history_len - 4].pla == pla
                    {
                        let prev4_loc = move_history[move_history_len - 4].loc;
                        num_turns_of_history_included = 4;
                        if prev4_loc == PASS_LOC {
                            row_global[3] = 1.0f32;
                        } else if prev4_loc != NULL_LOC {
                            let pos = nn_pos::loc_to_pos(prev4_loc, x_size, nn_x_len, nn_y_len);
                            set_row_bin(row_bin, pos, 12, 1.0f32, pos_stride, feature_stride);
                        }
                        if amount_of_history_to_try_to_use >= 5
                            && move_history[move_history_len - 5].pla == opp
                        {
                            let prev5_loc = move_history[move_history_len - 5].loc;
                            num_turns_of_history_included = 5;
                            if prev5_loc == PASS_LOC {
                                row_global[4] = 1.0f32;
                            } else if prev5_loc != NULL_LOC {
                                let pos = nn_pos::loc_to_pos(prev5_loc, x_size, nn_x_len, nn_y_len);
                                set_row_bin(row_bin, pos, 13, 1.0f32, pos_stride, feature_stride);
                            }
                        }
                    }
                }
            }
        }
    }

    // Ladder features 14, 15, 16, 17.
    let mut add_ladder_feature = |loc: Loc, pos: i32, working_moves: &[Loc]| {
        debug_assert!(
            board.colors[loc as usize] == C_BLACK || board.colors[loc as usize] == C_WHITE
        );
        debug_assert!(pos >= 0 && pos < nn_pos::MAX_BOARD_AREA as i32);
        set_row_bin(row_bin, pos, 14, 1.0f32, pos_stride, feature_stride);
        if board.colors[loc as usize] == opp as Color && board.get_num_liberties(loc) > 1 {
            for &working_move in working_moves {
                let working_pos = nn_pos::loc_to_pos(working_move, x_size, nn_x_len, nn_y_len);
                set_row_bin(row_bin, working_pos, 17, 1.0f32, pos_stride, feature_stride);
            }
        }
    };
    iter_ladders(board, nn_x_len, &mut add_ladder_feature);

    let prev_board = if num_turns_of_history_included < 1 {
        board
    } else {
        hist.get_recent_board(1)
    };
    let mut add_prev_ladder_feature = |loc: Loc, pos: i32, _working_moves: &[Loc]| {
        debug_assert!(
            prev_board.colors[loc as usize] == C_BLACK
                || prev_board.colors[loc as usize] == C_WHITE
        );
        debug_assert!(pos >= 0 && pos < nn_pos::MAX_BOARD_AREA as i32);
        set_row_bin(row_bin, pos, 15, 1.0f32, pos_stride, feature_stride);
    };
    iter_ladders(prev_board, nn_x_len, &mut add_prev_ladder_feature);

    let prev_prev_board = if num_turns_of_history_included < 2 {
        prev_board
    } else {
        hist.get_recent_board(2)
    };
    let mut add_prev_prev_ladder_feature = |loc: Loc, pos: i32, _working_moves: &[Loc]| {
        debug_assert!(
            prev_prev_board.colors[loc as usize] == C_BLACK
                || prev_prev_board.colors[loc as usize] == C_WHITE
        );
        debug_assert!(pos >= 0 && pos < nn_pos::MAX_BOARD_AREA as i32);
        set_row_bin(row_bin, pos, 16, 1.0f32, pos_stride, feature_stride);
    };
    iter_ladders(prev_prev_board, nn_x_len, &mut add_prev_prev_ladder_feature);

    // Features 20, 21 - second encore starting stones.
    if hist.encore_phase >= 2 {
        for y in 0..y_size {
            for x in 0..x_size {
                let loc = kata_game::board::location::get_loc(x, y, x_size);
                let pos = nn_pos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
                let color = hist.second_encore_start_color(loc);
                if color == pla as Color {
                    set_row_bin(row_bin, pos, 20, 1.0f32, pos_stride, feature_stride);
                } else if color == opp as Color {
                    set_row_bin(row_bin, pos, 21, 1.0f32, pos_stride, feature_stride);
                }
            }
        }
    }

    // Global features.
    let mut self_komi =
        hist.current_self_komi(next_player, nn_input_params.draw_equivalent_wins_for_white);
    let b_area = (x_size * y_size) as f32;
    self_komi = self_komi.clamp(-b_area - 1.0f32, b_area + 1.0f32);
    row_global[5] = self_komi / 20.0f32;

    match hist.rules.ko_rule {
        KoRule::Simple => {}
        KoRule::Positional | KoRule::Spight => {
            row_global[6] = 1.0f32;
            row_global[7] = 0.5f32;
        }
        KoRule::Situational => {
            row_global[6] = 1.0f32;
            row_global[7] = -0.5f32;
        }
    }

    if hist.rules.multi_stone_suicide_legal {
        row_global[8] = 1.0f32;
    }

    if hist.rules.scoring_rule == ScoringRule::Territory {
        row_global[9] = 1.0f32;
    }

    match hist.rules.tax_rule {
        TaxRule::None => {}
        TaxRule::Seki => {
            row_global[10] = 1.0f32;
        }
        TaxRule::All => {
            row_global[10] = 1.0f32;
            row_global[11] = 1.0f32;
        }
    }

    if hist.encore_phase > 0 {
        row_global[12] = 1.0f32;
    }
    if hist.encore_phase > 1 {
        row_global[13] = 1.0f32;
    }

    let pass_would_end_phase = if suppress_pass_would_end_phase {
        false
    } else {
        hist.pass_would_end_phase(board, next_player)
    };
    row_global[14] = if pass_would_end_phase { 1.0f32 } else { 0.0f32 };

    // Score-belief parity wave.
    if hist.rules.scoring_rule == ScoringRule::Area || hist.encore_phase >= 2 {
        let board_area_is_even = (x_size * y_size) % 2 == 0;
        let drawable_komis_are_even = board_area_is_even;
        let komi_floor = if drawable_komis_are_even {
            (self_komi / 2.0f32).floor() * 2.0f32
        } else {
            ((self_komi - 1.0f32) / 2.0f32).floor() * 2.0f32 + 1.0f32
        };
        let delta = (self_komi - komi_floor).clamp(0.0f32, 2.0f32);
        let wave = if delta < 0.5f32 {
            delta
        } else if delta < 1.5f32 {
            1.0f32 - delta
        } else {
            delta - 2.0f32
        };
        row_global[15] = wave;
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_history_move(
    row_bin: &mut [f32],
    row_global: &mut [f32],
    mv: &Move,
    global_idx: usize,
    bin_feature: i32,
    x_size: i32,
    nn_x_len: i32,
    nn_y_len: i32,
    pos_stride: i32,
    feature_stride: i32,
) {
    if mv.loc == PASS_LOC {
        row_global[global_idx] = 1.0f32;
    } else if mv.loc != NULL_LOC {
        let pos = nn_pos::loc_to_pos(mv.loc, x_size, nn_x_len, nn_y_len);
        set_row_bin(
            row_bin,
            pos,
            bin_feature,
            1.0f32,
            pos_stride,
            feature_stride,
        );
    }
}

impl NNOutput {
    /// Return the policy logits, using the noised version if one is present.
    pub fn get_policy_probs_maybe_noised(&self) -> &[f32] {
        self.noised_policy_probs
            .as_deref()
            .unwrap_or(&self.policy_probs)
    }

    /// Convert a board location to a flat position index for this output.
    pub fn get_pos(&self, loc: Loc, board: &Board) -> i32 {
        nn_pos::loc_to_pos(loc, board.x_size, self.nn_x_len, self.nn_y_len)
    }

    /// Average several `NNOutput`s that share the same `nn_hash`.
    ///
    /// Panics if `others` is empty. Does not carry over `noised_policy_probs`.
    pub fn average(others: &[NNOutput]) -> Self {
        assert!(
            !others.is_empty(),
            "NNOutput::average requires at least one output"
        );
        let first = &others[0];
        let n = others.len() as f32;
        let policy_size = nn_pos::policy_size(first.nn_x_len, first.nn_y_len) as usize;

        let mut out = NNOutput {
            nn_hash: first.nn_hash,
            white_win_prob: others.iter().map(|o| o.white_win_prob).sum::<f32>() / n,
            white_loss_prob: others.iter().map(|o| o.white_loss_prob).sum::<f32>() / n,
            white_no_result_prob: others.iter().map(|o| o.white_no_result_prob).sum::<f32>() / n,
            white_score_mean: others.iter().map(|o| o.white_score_mean).sum::<f32>() / n,
            white_score_mean_sq: others.iter().map(|o| o.white_score_mean_sq).sum::<f32>() / n,
            white_lead: others.iter().map(|o| o.white_lead).sum::<f32>() / n,
            var_time_left: others.iter().map(|o| o.var_time_left).sum::<f32>() / n,
            shortterm_winloss_error: others
                .iter()
                .map(|o| o.shortterm_winloss_error)
                .sum::<f32>()
                / n,
            shortterm_score_error: others.iter().map(|o| o.shortterm_score_error).sum::<f32>() / n,
            policy_probs: [0.0f32; nn_pos::MAX_NN_POLICY_SIZE],
            policy_optimism_used: first.policy_optimism_used,
            nn_x_len: first.nn_x_len,
            nn_y_len: first.nn_y_len,
            white_owner_map: None,
            noised_policy_probs: None,
        };

        for i in 0..policy_size {
            out.policy_probs[i] = others.iter().map(|o| o.policy_probs[i]).sum::<f32>() / n;
        }

        if others.iter().all(|o| o.white_owner_map.is_some()) {
            let area = (first.nn_x_len * first.nn_y_len) as usize;
            let mut owner_map = vec![0.0f32; area].into_boxed_slice();
            for i in 0..area {
                owner_map[i] = others
                    .iter()
                    .map(|o| o.white_owner_map.as_ref().unwrap()[i])
                    .sum::<f32>()
                    / n;
            }
            out.white_owner_map = Some(owner_map);
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kata_game::board::{P_BLACK, P_WHITE, location};
    use kata_game::history::BoardHistory;
    use kata_game::rules::Rules;

    #[test]
    fn test_loc_to_pos_pass_and_null() {
        assert_eq!(nn_pos::loc_to_pos(PASS_LOC, 19, 19, 19), 19 * 19);
        assert_eq!(nn_pos::loc_to_pos(NULL_LOC, 19, 19, 19), 19 * 20);
    }

    #[test]
    fn test_pos_to_loc_roundtrip() {
        let board = Board::new(9, 9);
        let loc = location::get_loc(3, 4, board.x_size);
        let pos = nn_pos::loc_to_pos(loc, board.x_size, 19, 19);
        assert_eq!(nn_pos::pos_to_loc(pos, 9, 9, 19, 19), loc);
    }

    #[test]
    fn test_pos_to_loc_pass() {
        assert_eq!(nn_pos::pos_to_loc(19 * 19, 9, 9, 19, 19), PASS_LOC);
    }

    #[test]
    fn test_pos_to_loc_out_of_board() {
        assert_eq!(nn_pos::pos_to_loc(10, 9, 9, 19, 19), NULL_LOC);
    }

    #[test]
    fn test_misc_nn_input_params_default() {
        let p = MiscNNInputParams::default();
        assert_eq!(p.draw_equivalent_wins_for_white, 0.5);
        assert!(!p.conservative_pass_and_is_root);
        assert_eq!(p.symmetry, nn_inputs::SYMMETRY_NOT_SPECIFIED);
        assert_eq!(p.max_history, 1000);
    }

    fn dummy_output(value: f32) -> NNOutput {
        NNOutput {
            nn_hash: Hash128::new(1, 2),
            white_win_prob: value,
            white_score_mean: value * 10.0,
            nn_x_len: 9,
            nn_y_len: 9,
            policy_probs: {
                let mut p = [0.0f32; nn_pos::MAX_NN_POLICY_SIZE];
                p[0] = value;
                p
            },
            ..NNOutput::default()
        }
    }

    #[test]
    fn test_nn_output_clone() {
        let mut out = dummy_output(0.5);
        out.white_owner_map = Some(vec![1.0f32; 81].into_boxed_slice());
        let cloned = out.clone();
        assert_eq!(cloned.white_win_prob, 0.5);
        assert!(cloned.white_owner_map.is_some());
        assert_eq!(cloned.white_owner_map.unwrap()[0], 1.0);
    }

    #[test]
    fn test_nn_output_average() {
        let a = dummy_output(0.0);
        let b = dummy_output(2.0);
        let avg = NNOutput::average(&[a, b]);
        assert!((avg.white_win_prob - 1.0).abs() < 1e-6);
        assert!((avg.white_score_mean - 10.0).abs() < 1e-6);
        assert!((avg.policy_probs[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_policy_probs_maybe_noised() {
        let mut out = dummy_output(0.0);
        out.policy_probs[0] = 1.0;
        assert_eq!(out.get_policy_probs_maybe_noised()[0], 1.0);

        out.noised_policy_probs = Some(vec![2.0f32; nn_pos::MAX_NN_POLICY_SIZE].into_boxed_slice());
        assert_eq!(out.get_policy_probs_maybe_noised()[0], 2.0);
    }

    #[test]
    fn test_set_row_bin_nchw() {
        let mut row = vec![0.0f32; 6]; // 2 pos * 3 features
        set_row_bin(&mut row, 1, 2, 7.0, 1, 2); // pos_stride=1, feature_stride=2
        assert_eq!(row[1 + 2 * 2], 7.0);
    }

    #[test]
    fn test_set_row_bin_nhwc() {
        let mut row = vec![0.0f32; 6]; // 2 pos * 3 features
        set_row_bin(&mut row, 1, 2, 7.0, 3, 1); // pos_stride=3, feature_stride=1
        assert_eq!(row[5], 7.0);
    }

    #[test]
    fn test_get_hash_changes_with_next_player() {
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let params = MiscNNInputParams::default();
        let h_black = get_hash(&board, &hist, P_BLACK, &params);
        let h_white = get_hash(&board, &hist, P_WHITE, &params);
        assert_ne!(h_black, h_white);
    }

    #[test]
    fn test_get_hash_changes_with_params() {
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let mut params = MiscNNInputParams::default();
        let h_default = get_hash(&board, &hist, P_BLACK, &params);
        params.conservative_pass_and_is_root = true;
        params.enable_passing_hacks = true;
        params.avoid_mytdagger_hack = true;
        let h_changed = get_hash(&board, &hist, P_BLACK, &params);
        assert_ne!(h_default, h_changed);
    }

    #[test]
    fn test_fill_scoring_no_group_tax() {
        let mut board = Board::new(5, 5);
        let b = location::get_loc(1, 1, 5);
        let w = location::get_loc(3, 3, 5);
        board.set_stone_fail_if_no_libs(b, C_BLACK);
        board.set_stone_fail_if_no_libs(w, C_WHITE);

        let mut area = [C_EMPTY; MAX_ARR_SIZE];
        area[b as usize] = C_BLACK;
        area[w as usize] = C_WHITE;

        let mut scoring = [0.0f32; MAX_ARR_SIZE];
        fill_scoring(&board, &area, false, &mut scoring);
        assert_eq!(scoring[b as usize], -1.0);
        assert_eq!(scoring[w as usize], 1.0);
        assert_eq!(scoring[location::get_loc(0, 0, 5) as usize], 0.0);
    }

    #[test]
    fn test_fill_scoring_group_tax_small_territory() {
        // A single white stone with one empty neighbor: under group tax the
        // empty point should score 0 because the territory count is <= 2.
        let mut board = Board::new(5, 5);
        let w = location::get_loc(2, 2, 5);
        board.set_stone_fail_if_no_libs(w, C_WHITE);

        let mut area = [C_EMPTY; MAX_ARR_SIZE];
        area[w as usize] = C_WHITE;
        area[location::get_loc(2, 3, 5) as usize] = C_WHITE;

        let mut scoring = [0.0f32; MAX_ARR_SIZE];
        fill_scoring(&board, &area, true, &mut scoring);
        assert_eq!(scoring[w as usize], 1.0);
        assert_eq!(scoring[location::get_loc(2, 3, 5) as usize], 0.0);
    }

    #[test]
    fn test_fill_scoring_group_tax_large_territory() {
        // A white stone surrounded by many empty points in the same area:
        // empty points should get a positive fractional score.
        let mut board = Board::new(7, 7);
        let w = location::get_loc(3, 3, 7);
        board.set_stone_fail_if_no_libs(w, C_WHITE);

        let mut area = [C_EMPTY; MAX_ARR_SIZE];
        let mut empties = Vec::new();
        for y in 2..5 {
            for x in 2..5 {
                if x == 3 && y == 3 {
                    continue;
                }
                let loc = location::get_loc(x, y, 7);
                area[loc as usize] = C_WHITE;
                empties.push(loc);
            }
        }
        area[w as usize] = C_WHITE;

        let mut scoring = [0.0f32; MAX_ARR_SIZE];
        fill_scoring(&board, &area, true, &mut scoring);
        assert_eq!(scoring[w as usize], 1.0);
        let territory_value = scoring[empties[0] as usize];
        assert!(territory_value > 0.0 && territory_value < 1.0);
    }

    fn index_nchw(
        _nn_x_len: i32,
        _nn_y_len: i32,
        pos: i32,
        feature: i32,
        num_features: i32,
    ) -> usize {
        (pos * num_features + feature) as usize
    }

    #[test]
    fn test_fill_row_v5_basic_stones_and_on_board() {
        let mut board = Board::new(5, 5);
        let b = location::get_loc(1, 1, 5);
        let w = location::get_loc(3, 3, 5);
        board.set_stone_fail_if_no_libs(b, C_BLACK);
        board.set_stone_fail_if_no_libs(w, C_WHITE);

        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let params = MiscNNInputParams::default();
        let nn_x_len = 5;
        let nn_y_len = 5;
        let mut row_bin = vec![0.0f32; (NUM_FEATURES_SPATIAL_V5 * nn_x_len * nn_y_len) as usize];
        let mut row_global = vec![0.0f32; NUM_FEATURES_GLOBAL_V5 as usize];

        fill_row_v5(
            &board,
            &hist,
            P_BLACK,
            &params,
            nn_x_len,
            nn_y_len,
            true, // NHWC
            &mut row_bin,
            &mut row_global,
        );

        let pos_b = nn_pos::loc_to_pos(b, 5, nn_x_len, nn_y_len);
        let pos_w = nn_pos::loc_to_pos(w, 5, nn_x_len, nn_y_len);
        let idx =
            |pos, feature| index_nchw(nn_x_len, nn_y_len, pos, feature, NUM_FEATURES_SPATIAL_V5);

        // Feature 0 - on board
        assert_eq!(row_bin[idx(pos_b, 0)], 1.0);
        // Feature 1 - black stone (pla)
        assert_eq!(row_bin[idx(pos_b, 1)], 1.0);
        // Feature 2 - white stone (opp)
        assert_eq!(row_bin[idx(pos_w, 2)], 1.0);
    }

    #[test]
    fn test_fill_row_v5_global_features() {
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let params = MiscNNInputParams::default();
        let nn_x_len = 5;
        let nn_y_len = 5;
        let mut row_bin = vec![0.0f32; (NUM_FEATURES_SPATIAL_V5 * nn_x_len * nn_y_len) as usize];
        let mut row_global = vec![0.0f32; NUM_FEATURES_GLOBAL_V5 as usize];

        fill_row_v5(
            &board,
            &hist,
            P_BLACK,
            &params,
            nn_x_len,
            nn_y_len,
            false,
            &mut row_bin,
            &mut row_global,
        );

        assert!(row_global[5].abs() > 0.0); // komi
        assert_eq!(row_global[6], 1.0); // positional ko rule
        assert_eq!(row_global[7], 0.5);
        assert_eq!(row_global[8], 1.0); // suicide legal
        assert_eq!(row_global[9], 0.0); // area scoring
        assert_eq!(row_global[10], 0.0); // no encore
        assert_eq!(row_global[11], 0.0);
    }

    #[test]
    fn test_fill_row_v5_history() {
        let mut board = Board::new(5, 5);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let move_loc = location::get_loc(2, 2, 5);
        assert!(hist.make_board_move_tolerant(&mut board, move_loc, P_BLACK));

        let params = MiscNNInputParams::default();
        let nn_x_len = 5;
        let nn_y_len = 5;
        let mut row_bin = vec![0.0f32; (NUM_FEATURES_SPATIAL_V5 * nn_x_len * nn_y_len) as usize];
        let mut row_global = vec![0.0f32; NUM_FEATURES_GLOBAL_V5 as usize];

        fill_row_v5(
            &board,
            &hist,
            P_WHITE,
            &params,
            nn_x_len,
            nn_y_len,
            true,
            &mut row_bin,
            &mut row_global,
        );

        let pos = nn_pos::loc_to_pos(move_loc, 5, nn_x_len, nn_y_len);
        let idx = (pos * NUM_FEATURES_SPATIAL_V5 + 6) as usize;
        assert_eq!(row_bin[idx], 1.0);
    }

    #[test]
    fn test_iter_ladders_finds_ladder() {
        // White chain A2-B2 in the corner with a two-liberty ladder capture.
        let mut board = Board::new(5, 5);
        let placements = vec![
            Move::new(location::get_loc(0, 1, 5), P_WHITE),
            Move::new(location::get_loc(1, 1, 5), P_WHITE),
            Move::new(location::get_loc(0, 2, 5), P_BLACK),
            Move::new(location::get_loc(1, 2, 5), P_BLACK),
            Move::new(location::get_loc(2, 1, 5), P_BLACK),
            Move::new(location::get_loc(2, 0, 5), P_BLACK),
        ];
        assert!(board.set_stones_tolerant(&placements) == 0);

        let mut found = Vec::new();
        iter_ladders(&board, 5, &mut |loc, pos, _moves| {
            found.push((loc, pos));
        });

        assert!(!found.is_empty());
        let white_a2 = location::get_loc(0, 1, 5);
        let white_b2 = location::get_loc(1, 1, 5);
        assert!(
            found
                .iter()
                .any(|&(loc, _pos)| loc == white_a2 || loc == white_b2)
        );
    }

    fn idx_v7(nn_x_len: i32, nn_y_len: i32, pos: i32, feature: i32) -> usize {
        let _ = (nn_x_len, nn_y_len);
        (pos * NUM_FEATURES_SPATIAL_V7 + feature) as usize
    }

    #[test]
    fn test_fill_row_v7_basic_stones_and_libs() {
        let mut board = Board::new(5, 5);
        let black_1_lib = location::get_loc(0, 0, 5);
        let white_2_libs = location::get_loc(3, 3, 5);
        board.set_stone_fail_if_no_libs(black_1_lib, C_BLACK);
        board.set_stone_fail_if_no_libs(white_2_libs, C_WHITE);
        // Reduce black's liberties to 1.
        board.set_stone_fail_if_no_libs(location::get_loc(1, 0, 5), C_WHITE);
        board.set_stone_fail_if_no_libs(location::get_loc(0, 1, 5), C_WHITE);
        // Reduce white's liberties to 2.
        board.set_stone_fail_if_no_libs(location::get_loc(3, 2, 5), C_BLACK);
        board.set_stone_fail_if_no_libs(location::get_loc(2, 3, 5), C_BLACK);

        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let params = MiscNNInputParams::default();
        let nn_x_len = 5;
        let nn_y_len = 5;
        let mut row_bin = vec![0.0f32; (NUM_FEATURES_SPATIAL_V7 * nn_x_len * nn_y_len) as usize];
        let mut row_global = vec![0.0f32; NUM_FEATURES_GLOBAL_V7 as usize];

        fill_row_v7(
            &board,
            &hist,
            P_BLACK,
            &params,
            nn_x_len,
            nn_y_len,
            true, // NHWC
            &mut row_bin,
            &mut row_global,
        );

        let pos_b = nn_pos::loc_to_pos(black_1_lib, 5, nn_x_len, nn_y_len);
        let pos_w = nn_pos::loc_to_pos(white_2_libs, 5, nn_x_len, nn_y_len);

        // Feature 0 - on board.
        assert_eq!(row_bin[idx_v7(nn_x_len, nn_y_len, pos_b, 0)], 1.0);
        // Feature 1 - pla stone.
        assert_eq!(row_bin[idx_v7(nn_x_len, nn_y_len, pos_b, 1)], 1.0);
        // Feature 3 - one liberty.
        assert_eq!(row_bin[idx_v7(nn_x_len, nn_y_len, pos_b, 3)], 1.0);
        // Feature 2 - opp stone.
        assert_eq!(row_bin[idx_v7(nn_x_len, nn_y_len, pos_w, 2)], 1.0);
        // Feature 4 - two liberties.
        assert_eq!(row_bin[idx_v7(nn_x_len, nn_y_len, pos_w, 4)], 1.0);
    }

    #[test]
    fn test_fill_row_v7_history_and_pass_flag() {
        let mut board = Board::new(5, 5);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let move_loc = location::get_loc(2, 2, 5);
        assert!(hist.make_board_move_tolerant(&mut board, move_loc, P_BLACK));
        assert!(hist.make_board_move_tolerant(&mut board, PASS_LOC, P_WHITE));

        let params = MiscNNInputParams::default();
        let nn_x_len = 5;
        let nn_y_len = 5;
        let mut row_bin = vec![0.0f32; (NUM_FEATURES_SPATIAL_V7 * nn_x_len * nn_y_len) as usize];
        let mut row_global = vec![0.0f32; NUM_FEATURES_GLOBAL_V7 as usize];

        fill_row_v7(
            &board,
            &hist,
            P_BLACK,
            &params,
            nn_x_len,
            nn_y_len,
            true,
            &mut row_bin,
            &mut row_global,
        );

        // Most recent move was a pass by opponent -> global[0].
        assert_eq!(row_global[0], 1.0);
        // Move before that was a pla stone at move_loc -> spatial feature 10.
        let pos = nn_pos::loc_to_pos(move_loc, 5, nn_x_len, nn_y_len);
        assert_eq!(row_bin[idx_v7(nn_x_len, nn_y_len, pos, 10)], 1.0);
    }

    #[test]
    fn test_fill_row_v7_ladder_features() {
        // White chain A2-B2 in the corner with a two-liberty ladder capture.
        let mut board = Board::new(5, 5);
        let placements = vec![
            Move::new(location::get_loc(0, 1, 5), P_WHITE),
            Move::new(location::get_loc(1, 1, 5), P_WHITE),
            Move::new(location::get_loc(0, 2, 5), P_BLACK),
            Move::new(location::get_loc(1, 2, 5), P_BLACK),
            Move::new(location::get_loc(2, 1, 5), P_BLACK),
            Move::new(location::get_loc(2, 0, 5), P_BLACK),
        ];
        assert!(board.set_stones_tolerant(&placements) == 0);

        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let params = MiscNNInputParams::default();
        let nn_x_len = 5;
        let nn_y_len = 5;
        let mut row_bin = vec![0.0f32; (NUM_FEATURES_SPATIAL_V7 * nn_x_len * nn_y_len) as usize];
        let mut row_global = vec![0.0f32; NUM_FEATURES_GLOBAL_V7 as usize];

        fill_row_v7(
            &board,
            &hist,
            P_BLACK,
            &params,
            nn_x_len,
            nn_y_len,
            true,
            &mut row_bin,
            &mut row_global,
        );

        let white_a2 = location::get_loc(0, 1, 5);
        let pos_a2 = nn_pos::loc_to_pos(white_a2, 5, nn_x_len, nn_y_len);

        // Current, previous and previous-previous ladder features are all set
        // because there is no move history, so all three boards are the same.
        assert_eq!(row_bin[idx_v7(nn_x_len, nn_y_len, pos_a2, 14)], 1.0);
        assert_eq!(row_bin[idx_v7(nn_x_len, nn_y_len, pos_a2, 15)], 1.0);
        assert_eq!(row_bin[idx_v7(nn_x_len, nn_y_len, pos_a2, 16)], 1.0);
        // Feature 17 should be set for at least one working attacker move.
        let has_working_move =
            (0..(nn_x_len * nn_y_len)).any(|p| row_bin[idx_v7(nn_x_len, nn_y_len, p, 17)] > 0.0);
        assert!(has_working_move);
    }

    #[test]
    fn test_fill_row_v7_global_features() {
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let params = MiscNNInputParams::default();
        let nn_x_len = 5;
        let nn_y_len = 5;
        let mut row_bin = vec![0.0f32; (NUM_FEATURES_SPATIAL_V7 * nn_x_len * nn_y_len) as usize];
        let mut row_global = vec![0.0f32; NUM_FEATURES_GLOBAL_V7 as usize];

        fill_row_v7(
            &board,
            &hist,
            P_BLACK,
            &params,
            nn_x_len,
            nn_y_len,
            false,
            &mut row_bin,
            &mut row_global,
        );

        // Komi from Black's perspective: -7.5 / 20 = -0.375.
        assert!((row_global[5] - -0.375).abs() < 1e-6);
        assert_eq!(row_global[6], 1.0); // positional ko rule
        assert_eq!(row_global[7], 0.5);
        assert_eq!(row_global[8], 1.0); // suicide legal
        assert_eq!(row_global[9], 0.0); // area scoring
        assert_eq!(row_global[10], 0.0); // no tax
        assert_eq!(row_global[11], 0.0);
        assert_eq!(row_global[12], 0.0); // no encore
        assert_eq!(row_global[13], 0.0);
        assert_eq!(row_global[14], 0.0); // pass does not end phase
        assert_eq!(row_global[15], 0.0); // no playout doubling
        assert_eq!(row_global[16], 0.0);
        assert_eq!(row_global[17], 0.0); // no button
        // Parity wave is defined for area scoring.
        assert!(row_global[18].abs() <= 1.0);
    }

    fn idx_v3(pos: i32, feature: i32) -> usize {
        (pos * NUM_FEATURES_SPATIAL_V3 + feature) as usize
    }

    fn idx_v4(pos: i32, feature: i32) -> usize {
        (pos * NUM_FEATURES_SPATIAL_V4 + feature) as usize
    }

    #[test]
    fn test_fill_row_v3_basic_stones_and_area() {
        let mut board = Board::new(5, 5);
        let black_stone = location::get_loc(2, 2, 5);
        board.set_stone_fail_if_no_libs(black_stone, C_BLACK);

        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let params = MiscNNInputParams::default();
        let nn_x_len = 5;
        let nn_y_len = 5;
        let mut row_bin = vec![0.0f32; (NUM_FEATURES_SPATIAL_V3 * nn_x_len * nn_y_len) as usize];
        let mut row_global = vec![0.0f32; NUM_FEATURES_GLOBAL_V3 as usize];

        fill_row_v3(
            &board,
            &hist,
            P_BLACK,
            &params,
            nn_x_len,
            nn_y_len,
            true,
            &mut row_bin,
            &mut row_global,
        );

        let pos = nn_pos::loc_to_pos(black_stone, 5, nn_x_len, nn_y_len);
        assert_eq!(row_bin[idx_v3(pos, 0)], 1.0); // on board
        assert_eq!(row_bin[idx_v3(pos, 1)], 1.0); // pla stone
        // Area feature for the stone itself should be set (area scoring, non-pass-alive stones).
        assert_eq!(row_bin[idx_v3(pos, 18)], 1.0);
    }

    #[test]
    fn test_fill_row_v3_global_features() {
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let params = MiscNNInputParams::default();
        let nn_x_len = 5;
        let nn_y_len = 5;
        let mut row_bin = vec![0.0f32; (NUM_FEATURES_SPATIAL_V3 * nn_x_len * nn_y_len) as usize];
        let mut row_global = vec![0.0f32; NUM_FEATURES_GLOBAL_V3 as usize];

        fill_row_v3(
            &board,
            &hist,
            P_BLACK,
            &params,
            nn_x_len,
            nn_y_len,
            false,
            &mut row_bin,
            &mut row_global,
        );

        // Komi from Black's perspective: -7.5 / 15 = -0.5.
        assert!((row_global[5] - -0.5).abs() < 1e-6);
        assert_eq!(row_global[6], 1.0); // positional ko
        assert_eq!(row_global[7], 0.5);
        assert_eq!(row_global[8], 1.0); // suicide legal
        assert_eq!(row_global[9], 0.0); // area scoring
        assert_eq!(row_global[10], 0.0); // no encore
        assert_eq!(row_global[11], 0.0);
        assert_eq!(row_global[12], 0.0); // pass does not end phase
        assert!(row_global[13].abs() <= 1.0); // parity wave
    }

    #[test]
    fn test_fill_row_v4_pass_alive_area() {
        let mut board = Board::new(5, 5);
        let black_stone = location::get_loc(2, 2, 5);
        board.set_stone_fail_if_no_libs(black_stone, C_BLACK);

        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let params = MiscNNInputParams::default();
        let nn_x_len = 5;
        let nn_y_len = 5;
        let mut row_bin = vec![0.0f32; (NUM_FEATURES_SPATIAL_V4 * nn_x_len * nn_y_len) as usize];
        let mut row_global = vec![0.0f32; NUM_FEATURES_GLOBAL_V4 as usize];

        fill_row_v4(
            &board,
            &hist,
            P_BLACK,
            &params,
            nn_x_len,
            nn_y_len,
            true,
            &mut row_bin,
            &mut row_global,
        );

        let pos = nn_pos::loc_to_pos(black_stone, 5, nn_x_len, nn_y_len);
        assert_eq!(row_bin[idx_v4(pos, 1)], 1.0); // pla stone
        // V4 uses pass-alive territory; the stone itself is not pass-alive by itself,
        // so feature 18 should be 0, but the stone feature is present.
        assert_eq!(row_bin[idx_v4(pos, 18)], 0.0);
    }

    #[test]
    fn test_fill_row_v4_global_features() {
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let params = MiscNNInputParams::default();
        let nn_x_len = 5;
        let nn_y_len = 5;
        let mut row_bin = vec![0.0f32; (NUM_FEATURES_SPATIAL_V4 * nn_x_len * nn_y_len) as usize];
        let mut row_global = vec![0.0f32; NUM_FEATURES_GLOBAL_V4 as usize];

        fill_row_v4(
            &board,
            &hist,
            P_BLACK,
            &params,
            nn_x_len,
            nn_y_len,
            false,
            &mut row_bin,
            &mut row_global,
        );

        assert!((row_global[5] - -0.5).abs() < 1e-6);
        assert_eq!(row_global[9], 0.0); // area scoring
        assert_eq!(row_global[12], 0.0); // pass does not end phase
        assert!(row_global[13].abs() <= 1.0);
    }

    fn idx_v6(pos: i32, feature: i32) -> usize {
        (pos * NUM_FEATURES_SPATIAL_V6 + feature) as usize
    }

    #[test]
    fn test_fill_row_v6_basic_stones_and_libs() {
        let mut board = Board::new(5, 5);
        let black_1_lib = location::get_loc(0, 0, 5);
        let white_2_libs = location::get_loc(3, 3, 5);
        board.set_stone_fail_if_no_libs(black_1_lib, C_BLACK);
        board.set_stone_fail_if_no_libs(white_2_libs, C_WHITE);
        board.set_stone_fail_if_no_libs(location::get_loc(1, 0, 5), C_WHITE);
        board.set_stone_fail_if_no_libs(location::get_loc(0, 1, 5), C_WHITE);
        board.set_stone_fail_if_no_libs(location::get_loc(3, 2, 5), C_BLACK);
        board.set_stone_fail_if_no_libs(location::get_loc(2, 3, 5), C_BLACK);

        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let params = MiscNNInputParams::default();
        let nn_x_len = 5;
        let nn_y_len = 5;
        let mut row_bin = vec![0.0f32; (NUM_FEATURES_SPATIAL_V6 * nn_x_len * nn_y_len) as usize];
        let mut row_global = vec![0.0f32; NUM_FEATURES_GLOBAL_V6 as usize];

        fill_row_v6(
            &board,
            &hist,
            P_BLACK,
            &params,
            nn_x_len,
            nn_y_len,
            true,
            &mut row_bin,
            &mut row_global,
        );

        let pos_b = nn_pos::loc_to_pos(black_1_lib, 5, nn_x_len, nn_y_len);
        let pos_w = nn_pos::loc_to_pos(white_2_libs, 5, nn_x_len, nn_y_len);

        assert_eq!(row_bin[idx_v6(pos_b, 0)], 1.0);
        assert_eq!(row_bin[idx_v6(pos_b, 1)], 1.0);
        assert_eq!(row_bin[idx_v6(pos_b, 3)], 1.0);
        assert_eq!(row_bin[idx_v6(pos_w, 2)], 1.0);
        assert_eq!(row_bin[idx_v6(pos_w, 4)], 1.0);
    }

    #[test]
    fn test_fill_row_v6_global_features() {
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::get_tromp_taylorish(), 0);
        let params = MiscNNInputParams::default();
        let nn_x_len = 5;
        let nn_y_len = 5;
        let mut row_bin = vec![0.0f32; (NUM_FEATURES_SPATIAL_V6 * nn_x_len * nn_y_len) as usize];
        let mut row_global = vec![0.0f32; NUM_FEATURES_GLOBAL_V6 as usize];

        fill_row_v6(
            &board,
            &hist,
            P_BLACK,
            &params,
            nn_x_len,
            nn_y_len,
            false,
            &mut row_bin,
            &mut row_global,
        );

        // Komi from Black's perspective: -7.5 / 20 = -0.375.
        assert!((row_global[5] - -0.375).abs() < 1e-6);
        assert_eq!(row_global[6], 1.0); // positional ko
        assert_eq!(row_global[7], 0.5);
        assert_eq!(row_global[8], 1.0); // suicide legal
        assert_eq!(row_global[9], 0.0); // area scoring
        assert_eq!(row_global[10], 0.0); // no tax
        assert_eq!(row_global[11], 0.0);
        assert_eq!(row_global[12], 0.0); // no encore
        assert_eq!(row_global[13], 0.0);
        assert_eq!(row_global[14], 0.0); // pass does not end phase
        // Parity wave is defined for area scoring.
        assert!(row_global[15].abs() <= 1.0);
        // V6 has no button or playout-doubling global features.
        assert_eq!(row_global.len(), 16);
    }
}

//! Training data output structures and buffers.
//!
//! Corresponds to `cpp/dataio/trainingwrite.h` and `cpp/dataio/trainingwrite.cpp`.
//! This slice covers the value types and `TrainingWriteBuffers` construction/clear.
//! The full `addRow` / `writeToZipFile` / `writeToTextOstream` logic is left for a
//! later slice.

use kata_core::global::IOError;
use kata_core::hash::Hash128;
use kata_core::rng::Rand;
use kata_game::board::{Board, C_EMPTY, Color, Loc, P_WHITE, Player, get_opp, location};
use kata_game::history::BoardHistory;
use kata_game::rules::ScoringRule;
use kata_nn::inputs::{
    MiscNNInputParams, NUM_FEATURES_GLOBAL_V3, NUM_FEATURES_GLOBAL_V4, NUM_FEATURES_GLOBAL_V5,
    NUM_FEATURES_GLOBAL_V6, NUM_FEATURES_GLOBAL_V7, NUM_FEATURES_SPATIAL_V3,
    NUM_FEATURES_SPATIAL_V4, NUM_FEATURES_SPATIAL_V5, NUM_FEATURES_SPATIAL_V6,
    NUM_FEATURES_SPATIAL_V7, fill_row_v3, fill_row_v4, fill_row_v5, fill_row_v6, fill_row_v7,
    nn_pos,
};
use kata_nn::sgf_meta::{METADATA_INPUT_NUM_CHANNELS, SgfMetadata};

use crate::numpy::{NumpyElement, ZipFile};
use std::io::Write;

use crate::numpy::NumpyBuffer;

/// Number of channels for the policy target tensor.
pub const POLICY_TARGET_NUM_CHANNELS: i64 = 2;
/// Number of channels for the global target tensor.
pub const GLOBAL_TARGET_NUM_CHANNELS: i64 = 64;
/// Number of channels for the spatial value target tensor.
pub const VALUE_SPATIAL_TARGET_NUM_CHANNELS: i64 = 5;
/// Number of channels for the spatial Q-value target tensor.
pub const QVALUE_SPATIAL_TARGET_NUM_CHANNELS: i64 = 3;

/// A single move in a policy target distribution.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PolicyTargetMove {
    pub loc: Loc,
    pub policy_target: i16,
}

/// Policy target for a turn, possibly reduced.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PolicyTarget {
    pub policy_targets: Vec<PolicyTargetMove>,
    pub unreduced_num_visits: i64,
}

/// A single move in a Q-value target distribution.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QValueTargetMove {
    pub loc: Loc,
    pub win_loss: f32,
    pub score: f32,
    pub visits: i64,
}

/// Q-value targets for a turn.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QValueTargets {
    pub targets: Vec<QValueTargetMove>,
}

/// Value-head-related training targets from the perspective of white.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ValueTargets {
    pub win: f32,
    pub loss: f32,
    pub no_result: f32,
    pub score: f32,
    pub has_lead: bool,
    pub lead: f32,
}

/// Raw neural-net stats for a position, from the perspective of white.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NNRawStats {
    pub white_win_loss: f64,
    pub white_score_mean: f64,
    pub policy_entropy: f64,
}

/// A side position searched off the main line of a game.
#[derive(Clone, Default)]
pub struct SidePosition {
    pub board: Board,
    pub hist: BoardHistory,
    pub pla: Player,
    pub unreduced_num_visits: i64,
    pub policy_target: Vec<PolicyTargetMove>,
    pub policy_surprise: f64,
    pub policy_entropy: f64,
    pub search_entropy: f64,
    pub white_value_targets: ValueTargets,
    pub white_q_value_targets: QValueTargets,
    pub nn_raw_stats: NNRawStats,
    pub target_weight: f32,
    pub target_weight_unrounded: f32,
    pub num_neural_net_changes_so_far: i32,
    pub playout_doubling_advantage_pla: Player,
    pub playout_doubling_advantage: f64,
}

/// Record that the neural net changed at a particular turn.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChangedNeuralNet {
    pub name: String,
    pub turn_idx: i32,
}

/// Game initialization mode constants for [`FinishedGameData`].
pub mod mode {
    /// Normal self-play game.
    pub const NORMAL: i32 = 0;
    /// Cleanup-phase training game.
    pub const CLEANUP_TRAINING: i32 = 1;
    /// Fork from another self-play game.
    pub const FORK: i32 = 2;
    /// Handicap game.
    pub const HANDICAP: i32 = 3;
    /// Sampled from an external SGF position.
    pub const SGF_POS: i32 = 4;
    /// Sampled from a hint position.
    pub const HINT_POS: i32 = 5;
    /// Forked from a hint position.
    pub const HINT_FORK: i32 = 6;
    /// Asymmetric playouts game (nonzero PDA).
    pub const ASYM: i32 = 7;
    /// Total number of modes.
    pub const COUNT: i32 = 8;
}

/// Data accumulated for a finished game, before it is written to training rows.
#[derive(Clone, Default)]
pub struct FinishedGameData {
    pub b_name: String,
    pub w_name: String,
    pub b_idx: i32,
    pub w_idx: i32,

    pub start_board: Board,
    pub start_hist: BoardHistory,
    pub end_hist: BoardHistory,
    pub start_pla: Player,
    pub game_hash: Hash128,

    pub draw_equivalent_wins_for_white: f64,
    pub playout_doubling_advantage_pla: Player,
    pub playout_doubling_advantage: f64,
    pub hit_turn_limit: bool,

    pub num_extra_black: i32,
    pub mode: i32,
    pub began_in_encore_phase: i32,
    pub used_initial_position: i32,
    pub handicap_for_sgf: i32,

    pub has_full_data: bool,
    pub target_weight_by_turn: Vec<f32>,
    pub target_weight_by_turn_unrounded: Vec<f32>,
    pub policy_targets_by_turn: Vec<PolicyTarget>,
    pub policy_surprise_by_turn: Vec<f64>,
    pub policy_entropy_by_turn: Vec<f64>,
    pub search_entropy_by_turn: Vec<f64>,
    pub white_value_targets_by_turn: Vec<ValueTargets>,
    pub white_q_value_targets_by_turn: Vec<QValueTargets>,
    pub nn_raw_stats_by_turn: Vec<NNRawStats>,
    pub final_full_area: Vec<Color>,
    pub final_ownership: Vec<Color>,
    pub final_seki_areas: Vec<bool>,
    pub final_white_scoring: Vec<f32>,

    pub training_weight: f64,

    pub side_positions: Vec<SidePosition>,
    pub changed_neural_nets: Vec<ChangedNeuralNet>,

    pub b_time_used: f64,
    pub w_time_used: f64,
    pub b_move_count: i32,
    pub w_move_count: i32,
}

/// Pre-allocated numpy-style buffers for assembling training rows.
pub struct TrainingWriteBuffers {
    pub inputs_version: i32,
    pub max_rows: i32,
    pub num_binary_channels: i32,
    pub num_global_channels: i32,
    pub data_x_len: i32,
    pub data_y_len: i32,
    pub packed_board_area: i32,

    pub has_metadata_input: bool,

    pub cur_rows: i32,

    /// Unpacked binary input scratch buffer, not written to disk.
    pub binary_input_nchw_unpacked: Vec<f32>,

    /// Packed binary input feature planes.
    pub binary_input_nchw_packed: NumpyBuffer<u8>,
    /// Global input features.
    pub global_input_nc: NumpyBuffer<f32>,

    /// Policy targets (current turn and next turn).
    pub policy_targets_nc_move: NumpyBuffer<i16>,
    /// Global (value / metadata) targets.
    pub global_targets_nc: NumpyBuffer<f32>,
    /// Score distribution target.
    pub score_distr_n: NumpyBuffer<i8>,
    /// Spatial value-related targets.
    pub value_targets_nchw: NumpyBuffer<i8>,
    /// Spatial Q-value targets.
    pub q_value_targets_nc_move: NumpyBuffer<i16>,
    /// Metadata input features.
    pub metadata_input_nc: NumpyBuffer<f32>,
}

impl TrainingWriteBuffers {
    /// Create a set of empty buffers sized for `max_rows` rows.
    pub fn new(
        inputs_version: i32,
        max_rows: i32,
        num_binary_channels: i32,
        num_global_channels: i32,
        data_x_len: i32,
        data_y_len: i32,
        has_metadata_input: bool,
    ) -> Result<Self, IOError> {
        let packed_board_area = (data_x_len * data_y_len + 7) / 8;
        let policy_size = data_x_len * data_y_len + 1;
        let score_distr_len = data_x_len * data_y_len * 2 + nn_pos::EXTRA_SCORE_DISTR_RADIUS * 2;

        let binary_input_nchw_unpacked =
            vec![0.0f32; (num_binary_channels * data_x_len * data_y_len) as usize];

        Ok(Self {
            inputs_version,
            max_rows,
            num_binary_channels,
            num_global_channels,
            data_x_len,
            data_y_len,
            packed_board_area,
            has_metadata_input,
            cur_rows: 0,
            binary_input_nchw_unpacked,
            binary_input_nchw_packed: NumpyBuffer::new(vec![
                max_rows as i64,
                num_binary_channels as i64,
                packed_board_area as i64,
            ])?,
            global_input_nc: NumpyBuffer::new(vec![max_rows as i64, num_global_channels as i64])?,
            policy_targets_nc_move: NumpyBuffer::new(vec![
                max_rows as i64,
                POLICY_TARGET_NUM_CHANNELS,
                policy_size as i64,
            ])?,
            global_targets_nc: NumpyBuffer::new(vec![max_rows as i64, GLOBAL_TARGET_NUM_CHANNELS])?,
            score_distr_n: NumpyBuffer::new(vec![max_rows as i64, score_distr_len as i64])?,
            value_targets_nchw: NumpyBuffer::new(vec![
                max_rows as i64,
                VALUE_SPATIAL_TARGET_NUM_CHANNELS,
                data_y_len as i64,
                data_x_len as i64,
            ])?,
            q_value_targets_nc_move: NumpyBuffer::new(vec![
                max_rows as i64,
                QVALUE_SPATIAL_TARGET_NUM_CHANNELS,
                policy_size as i64,
            ])?,
            metadata_input_nc: NumpyBuffer::new(vec![
                if has_metadata_input {
                    max_rows as i64
                } else {
                    1
                },
                METADATA_INPUT_NUM_CHANNELS as i64,
            ])?,
        })
    }

    /// Reset the row counter. The underlying buffers are not zeroed.
    pub fn clear(&mut self) {
        self.cur_rows = 0;
    }

    /// Add a single training row to the buffers.
    ///
    /// Mirrors `TrainingWriteBuffers::addRow` in `cpp/dataio/trainingwrite.cpp`.
    /// This is a large, mostly mechanical translation of the C++ routine; the
    /// channel layout of `global_targets_nc` matches the original exactly.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    pub fn add_row(
        &mut self,
        board: &Board,
        hist: &BoardHistory,
        next_player: Player,
        start_hist: &BoardHistory,
        actual_game_end_hist: &BoardHistory,
        turn_idx: i32,
        target_weight: f32,
        unreduced_num_visits: i64,
        policy_target0: Option<&[PolicyTargetMove]>,
        policy_target1: Option<&[PolicyTargetMove]>,
        policy_surprise: f64,
        policy_entropy: f64,
        search_entropy: f64,
        white_value_targets: &[ValueTargets],
        white_q_value_targets: &[QValueTargets],
        white_value_targets_idx: usize,
        value_target_weight: f32,
        td_value_target_weight: f32,
        lead_target_weight_factor: f32,
        nn_raw_stats: &NNRawStats,
        final_board: Option<&Board>,
        final_full_area: Option<&[Color]>,
        final_ownership: Option<&[Color]>,
        final_white_scoring: Option<&[f32]>,
        pos_hist_for_future_boards: Option<&[Board]>,
        is_side_position: bool,
        num_neural_nets_behind_latest: i32,
        draw_equivalent_wins_for_white: f64,
        playout_doubling_advantage_pla: Player,
        playout_doubling_advantage: f64,
        game_hash: Hash128,
        changed_neural_nets: &[ChangedNeuralNet],
        hit_turn_limit: bool,
        num_extra_black: i32,
        mode: i32,
        sgf_meta: Option<&SgfMetadata>,
        rand: &mut Rand,
    ) {
        assert!((3..=7).contains(&self.inputs_version));
        assert!(self.cur_rows < self.max_rows);

        let pos_area = (self.data_x_len * self.data_y_len) as usize;
        let policy_size = nn_pos::policy_size(self.data_x_len, self.data_y_len) as usize;

        // Fill binary and global input features.
        {
            let playout_doubling_advantage_value = if !is_side_position {
                if get_opp(next_player) == playout_doubling_advantage_pla {
                    -playout_doubling_advantage
                } else {
                    playout_doubling_advantage
                }
            } else {
                assert_eq!(playout_doubling_advantage_pla, C_EMPTY);
                assert_eq!(playout_doubling_advantage, 0.0);
                0.0
            };
            let nn_input_params = MiscNNInputParams {
                draw_equivalent_wins_for_white,
                playout_doubling_advantage: playout_doubling_advantage_value,
                ..MiscNNInputParams::default()
            };

            let row_bin = &mut self.binary_input_nchw_unpacked;
            let row_global_base = (self.cur_rows as i64 * self.num_global_channels as i64) as usize;
            let row_global = &mut self.global_input_nc.data
                [row_global_base..row_global_base + self.num_global_channels as usize];

            match self.inputs_version {
                3 => {
                    assert_eq!(NUM_FEATURES_SPATIAL_V3, self.num_binary_channels);
                    assert_eq!(NUM_FEATURES_GLOBAL_V3, self.num_global_channels);
                    fill_row_v3(
                        board,
                        hist,
                        next_player,
                        &nn_input_params,
                        self.data_x_len,
                        self.data_y_len,
                        false,
                        row_bin,
                        row_global,
                    );
                }
                4 => {
                    assert_eq!(NUM_FEATURES_SPATIAL_V4, self.num_binary_channels);
                    assert_eq!(NUM_FEATURES_GLOBAL_V4, self.num_global_channels);
                    fill_row_v4(
                        board,
                        hist,
                        next_player,
                        &nn_input_params,
                        self.data_x_len,
                        self.data_y_len,
                        false,
                        row_bin,
                        row_global,
                    );
                }
                5 => {
                    assert_eq!(NUM_FEATURES_SPATIAL_V5, self.num_binary_channels);
                    assert_eq!(NUM_FEATURES_GLOBAL_V5, self.num_global_channels);
                    fill_row_v5(
                        board,
                        hist,
                        next_player,
                        &nn_input_params,
                        self.data_x_len,
                        self.data_y_len,
                        false,
                        row_bin,
                        row_global,
                    );
                }
                6 => {
                    assert_eq!(NUM_FEATURES_SPATIAL_V6, self.num_binary_channels);
                    assert_eq!(NUM_FEATURES_GLOBAL_V6, self.num_global_channels);
                    fill_row_v6(
                        board,
                        hist,
                        next_player,
                        &nn_input_params,
                        self.data_x_len,
                        self.data_y_len,
                        false,
                        row_bin,
                        row_global,
                    );
                }
                7 => {
                    assert_eq!(NUM_FEATURES_SPATIAL_V7, self.num_binary_channels);
                    assert_eq!(NUM_FEATURES_GLOBAL_V7, self.num_global_channels);
                    fill_row_v7(
                        board,
                        hist,
                        next_player,
                        &nn_input_params,
                        self.data_x_len,
                        self.data_y_len,
                        false,
                        row_bin,
                        row_global,
                    );
                }
                _ => unreachable!(),
            }

            // Pack bools bitwise into uint8_t.
            let packed_base = (self.cur_rows as i64
                * self.num_binary_channels as i64
                * self.packed_board_area as i64) as usize;
            let row_bin_packed = &mut self.binary_input_nchw_packed.data[packed_base..];
            for c in 0..self.num_binary_channels {
                let plane_start = (c * pos_area as i32) as usize;
                let bits_start = (c * self.packed_board_area) as usize;
                pack_bits(
                    &row_bin[plane_start..plane_start + pos_area],
                    &mut row_bin_packed[bits_start..bits_start + pos_area.div_ceil(8)],
                );
            }
        }

        // Global targets and metadata.
        let global_base = (self.cur_rows as i64 * GLOBAL_TARGET_NUM_CHANNELS) as usize;
        let row_global = &mut self.global_targets_nc.data
            [global_base..global_base + GLOBAL_TARGET_NUM_CHANNELS as usize];

        // Target weight for the whole row.
        row_global[25] = target_weight;

        // Policy targets.
        let policy_base =
            (self.cur_rows as i64 * POLICY_TARGET_NUM_CHANNELS * policy_size as i64) as usize;
        let row_policy = &mut self.policy_targets_nc_move.data
            [policy_base..policy_base + (POLICY_TARGET_NUM_CHANNELS * policy_size as i64) as usize];

        if let Some(targets) = policy_target0 {
            fill_policy_target(
                targets,
                self.data_x_len,
                self.data_y_len,
                board.x_size,
                &mut row_policy[0..policy_size],
            );
            row_global[26] = 1.0;
        } else {
            uniform_policy_target(policy_size as i32, &mut row_policy[0..policy_size]);
            row_global[26] = 0.0;
        }

        if let Some(targets) = policy_target1 {
            fill_policy_target(
                targets,
                self.data_x_len,
                self.data_y_len,
                board.x_size,
                &mut row_policy[policy_size..policy_size * 2],
            );
            row_global[28] = 1.0;
        } else {
            uniform_policy_target(
                policy_size as i32,
                &mut row_policy[policy_size..policy_size * 2],
            );
            row_global[28] = 0.0;
        }

        // TD-like value targets.
        let board_area = (board.x_size * board.y_size) as f64;
        fill_value_td_targets(
            white_value_targets,
            white_value_targets_idx,
            next_player,
            0.0,
            &mut row_global[0..4],
        );
        fill_value_td_targets(
            white_value_targets,
            white_value_targets_idx,
            next_player,
            1.0 / (1.0 + board_area * 0.176),
            &mut row_global[4..8],
        );
        fill_value_td_targets(
            white_value_targets,
            white_value_targets_idx,
            next_player,
            1.0 / (1.0 + board_area * 0.056),
            &mut row_global[8..12],
        );
        fill_value_td_targets(
            white_value_targets,
            white_value_targets_idx,
            next_player,
            1.0 / (1.0 + board_area * 0.016),
            &mut row_global[12..16],
        );
        fill_value_td_targets(
            white_value_targets,
            white_value_targets_idx,
            next_player,
            1.0,
            &mut row_global[16..20],
        );

        // Lead.
        row_global[21] = 0.0;
        row_global[29] = 0.0;
        let this_targets = &white_value_targets[white_value_targets_idx];
        if this_targets.has_lead
            && !(actual_game_end_hist.is_game_finished && actual_game_end_hist.is_no_result)
        {
            let mut lead = if next_player == P_WHITE {
                this_targets.lead
            } else {
                -this_targets.lead
            };
            let score_target_cap =
                (nn_pos::MAX_BOARD_AREA as i32 + nn_pos::EXTRA_SCORE_DISTR_RADIUS) as f32;
            lead = lead.clamp(-score_target_cap, score_target_cap);
            row_global[21] = lead;
            row_global[29] = value_target_weight * lead_target_weight_factor;
        }

        // Expected time of arrival of win/loss variance.
        let mut sum = 0.0f64;
        for i in (white_value_targets_idx + 1)..white_value_targets.len() {
            let turns_from_now = i - white_value_targets_idx;
            let prev_wl = white_value_targets[i - 1].win - white_value_targets[i - 1].loss;
            let next_wl = white_value_targets[i].win - white_value_targets[i].loss;
            let variance = (next_wl - prev_wl) * (next_wl - prev_wl);
            sum += turns_from_now as f64 * variance as f64;
        }
        row_global[22] = sum as f32;

        // Unused and various other data.
        row_global[23] = 0.0;
        row_global[24] = 1.0 - td_value_target_weight;
        row_global[30] = policy_surprise as f32;
        row_global[31] = policy_entropy as f32;
        row_global[32] = search_entropy as f32;
        // Value weight.
        row_global[35] = 1.0 - value_target_weight;

        // History-use flags.
        let use_hist0 = rand.next_double() < 0.98;
        let use_hist1 = use_hist0 && rand.next_double() < 0.98;
        let use_hist2 = use_hist1 && rand.next_double() < 0.98;
        let use_hist3 = use_hist2 && rand.next_double() < 0.98;
        let use_hist4 = use_hist3 && rand.next_double() < 0.98;
        row_global[36] = if use_hist0 { 1.0 } else { 0.0 };
        row_global[37] = if use_hist1 { 1.0 } else { 0.0 };
        row_global[38] = if use_hist2 { 1.0 } else { 0.0 };
        row_global[39] = if use_hist3 { 1.0 } else { 0.0 };
        row_global[40] = if use_hist4 { 1.0 } else { 0.0 };

        // Hash of game.
        row_global[41] = (game_hash.hash0 & 0x3FFFFF) as f32;
        row_global[42] = ((game_hash.hash0 >> 22) & 0x3FFFFF) as f32;
        row_global[43] = ((game_hash.hash0 >> 44) & 0xFFFFF) as f32;
        row_global[44] = (game_hash.hash1 & 0x3FFFFF) as f32;
        row_global[45] = ((game_hash.hash1 >> 22) & 0x3FFFFF) as f32;
        row_global[46] = ((game_hash.hash1 >> 44) & 0xFFFFF) as f32;

        // Various other data.
        row_global[47] = hist.current_self_komi(next_player, draw_equivalent_wins_for_white);
        row_global[48] = if hist.encore_phase == 2 || hist.rules.scoring_rule == ScoringRule::Area {
            1.0
        } else {
            0.0
        };

        // Earlier neural net metadata.
        row_global[49] = if changed_neural_nets.is_empty() {
            0.0
        } else {
            1.0
        };
        row_global[50] = num_neural_nets_behind_latest as f32;

        // Misc metadata.
        row_global[51] = turn_idx as f32;
        row_global[52] = if hit_turn_limit { 1.0 } else { 0.0 };
        row_global[53] = start_hist.move_history.len() as f32;
        row_global[54] = num_extra_black as f32;

        // Game initialization.
        row_global[55] = mode as f32;
        row_global[56] = hist.initial_turn_number as f32;

        // Stats.
        row_global[57] = (if next_player == P_WHITE {
            nn_raw_stats.white_win_loss
        } else {
            -nn_raw_stats.white_win_loss
        }) as f32;
        row_global[58] = (if next_player == P_WHITE {
            nn_raw_stats.white_score_mean
        } else {
            -nn_raw_stats.white_score_mean
        }) as f32;
        row_global[59] = nn_raw_stats.policy_entropy as f32;

        // Original number of visits.
        row_global[60] = unreduced_num_visits as f32;

        // Bonus points.
        if !is_side_position {
            let white_bonus_points =
                actual_game_end_hist.white_bonus_score - hist.white_bonus_score;
            let self_bonus_points = if next_player == P_WHITE {
                white_bonus_points
            } else {
                -white_bonus_points
            };
            row_global[61] = if self_bonus_points != 0.0 {
                self_bonus_points
            } else {
                0.0
            };
        } else {
            row_global[61] = 0.0;
        }

        // Game finished.
        row_global[62] =
            if !is_side_position && actual_game_end_hist.is_game_finished && !hit_turn_limit {
                1.0
            } else {
                0.0
            };

        // Version.
        row_global[63] = 2.0;

        assert_eq!(64, GLOBAL_TARGET_NUM_CHANNELS);

        // Score distribution and ownership targets.
        let score_distr_len = pos_area * 2 + (nn_pos::EXTRA_SCORE_DISTR_RADIUS * 2) as usize;
        let score_distr_mid = pos_area + nn_pos::EXTRA_SCORE_DISTR_RADIUS as usize;
        let score_distr_base = (self.cur_rows as i64 * score_distr_len as i64) as usize;
        let row_score_distr =
            &mut self.score_distr_n.data[score_distr_base..score_distr_base + score_distr_len];
        let value_base =
            (self.cur_rows as i64 * VALUE_SPATIAL_TARGET_NUM_CHANNELS * pos_area as i64) as usize;
        let row_ownership = &mut self.value_targets_nchw.data[value_base
            ..value_base + (VALUE_SPATIAL_TARGET_NUM_CHANNELS * pos_area as i64) as usize];

        let no_ownership = final_ownership.is_none()
            || (actual_game_end_hist.is_game_finished && actual_game_end_hist.is_no_result);
        if no_ownership {
            row_global[27] = 0.0;
            row_global[20] = 0.0;
            row_ownership.fill(0);
            row_score_distr.fill(0);
            // Dummy value to make sure it still sums to 100.
            row_score_distr[score_distr_mid - 1] = 50;
            row_score_distr[score_distr_mid] = 50;
        } else if let Some(final_ownership) = final_ownership {
            let final_full_area =
                final_full_area.expect("finalFullArea required with finalOwnership");
            assert!(final_board.is_some());

            // Ownership weight scales by value weight.
            row_global[27] = value_target_weight;
            let last_targets = &white_value_targets[white_value_targets.len() - 1];
            let score = if next_player == P_WHITE {
                last_targets.score
            } else {
                -last_targets.score
            };
            row_global[20] = score;

            row_ownership.fill(0);

            let opp = get_opp(next_player);
            for y in 0..board.y_size {
                for x in 0..board.x_size {
                    let pos = nn_pos::xy_to_pos(x, y, self.data_x_len) as usize;
                    let loc = location::get_loc(x, y, board.x_size) as usize;
                    if final_ownership[loc] == next_player {
                        row_ownership[pos] = 1;
                    } else if final_ownership[loc] == opp {
                        row_ownership[pos] = -1;
                    }
                    if final_full_area[loc] != C_EMPTY && final_ownership[loc] == C_EMPTY {
                        row_ownership[pos + pos_area] = if final_full_area[loc] == next_player {
                            1
                        } else {
                            -1
                        };
                    }
                }
            }

            // Score vector "onehot"-like.
            row_score_distr.fill(0);
            let center_score = score.round() as i32;
            let lower_idx = center_score + score_distr_mid as i32 - 1;
            let upper_idx = center_score + score_distr_mid as i32;
            if upper_idx <= 0 {
                row_score_distr[0] = 100;
            } else if lower_idx >= score_distr_len as i32 - 1 {
                row_score_distr[score_distr_len - 1] = 100;
            } else {
                let lambda = score - (center_score as f32 - 0.5f32);
                let upper_prop = (lambda * 100.0f32).round() as i32;
                row_score_distr[lower_idx as usize] = (100 - upper_prop) as i8;
                row_score_distr[upper_idx as usize] = upper_prop as i8;
            }
        }

        // Future board states.
        if let Some(boards) = pos_hist_for_future_boards {
            assert_eq!(boards.len(), white_value_targets.len());
            assert!(!boards.is_empty());

            row_global[33] = 1.0;
            let end_idx = boards.len() - 1;
            let board2 = &boards[(white_value_targets_idx + 8).min(end_idx)];
            let board3 = &boards[(white_value_targets_idx + 32).min(end_idx)];
            assert_eq!(board2.y_size, board.y_size);
            assert_eq!(board2.x_size, board.x_size);
            assert_eq!(board3.y_size, board.y_size);
            assert_eq!(board3.x_size, board.x_size);

            row_ownership[pos_area * 2..pos_area * 3].fill(0);
            row_ownership[pos_area * 3..pos_area * 4].fill(0);

            let pla = next_player;
            let opp = get_opp(next_player);
            for y in 0..board.y_size {
                for x in 0..board.x_size {
                    let pos = nn_pos::xy_to_pos(x, y, self.data_x_len) as usize;
                    let loc = location::get_loc(x, y, board.x_size) as usize;
                    if board2.colors[loc] == pla {
                        row_ownership[pos + pos_area * 2] = 1;
                    } else if board2.colors[loc] == opp {
                        row_ownership[pos + pos_area * 2] = -1;
                    }
                    if board3.colors[loc] == pla {
                        row_ownership[pos + pos_area * 3] = 1;
                    } else if board3.colors[loc] == opp {
                        row_ownership[pos + pos_area * 3] = -1;
                    }
                }
            }
        } else {
            row_global[33] = 0.0;
            row_ownership[pos_area * 2..pos_area * 3].fill(0);
            row_ownership[pos_area * 3..pos_area * 4].fill(0);
        }

        // Final white scoring.
        if let Some(scoring) = final_white_scoring {
            if !(actual_game_end_hist.is_game_finished && actual_game_end_hist.is_no_result) {
                row_global[34] = value_target_weight;
                row_ownership[pos_area * 4..pos_area * 5].fill(0);

                for y in 0..board.y_size {
                    for x in 0..board.x_size {
                        let pos = nn_pos::xy_to_pos(x, y, self.data_x_len) as usize;
                        let loc = location::get_loc(x, y, board.x_size) as usize;
                        let s = if next_player == P_WHITE {
                            scoring[loc]
                        } else {
                            -scoring[loc]
                        };
                        assert!((-1.0f32..=1.0f32).contains(&s));
                        row_ownership[pos + pos_area * 4] = clamp_to_radius120(s * 120.0f32, rand);
                    }
                }
            } else {
                row_global[34] = 0.0;
                row_ownership[pos_area * 4..pos_area * 5].fill(0);
            }
        } else {
            row_global[34] = 0.0;
            row_ownership[pos_area * 4..pos_area * 5].fill(0);
        }

        // Q values.
        {
            assert!(white_value_targets_idx < white_q_value_targets.len());
            let q_base = (self.cur_rows as i64
                * QVALUE_SPATIAL_TARGET_NUM_CHANNELS
                * policy_size as i64) as usize;
            let row_q_values = &mut self.q_value_targets_nc_move.data[q_base
                ..q_base + (QVALUE_SPATIAL_TARGET_NUM_CHANNELS * policy_size as i64) as usize];
            fill_q_value_target(
                &white_q_value_targets[white_value_targets_idx].targets,
                next_player,
                self.data_x_len,
                self.data_y_len,
                board.x_size,
                row_q_values,
                rand,
            );
        }

        // Metadata input.
        if self.has_metadata_input {
            let sgf_meta = sgf_meta.expect("sgfMeta required when hasMetadataInput is true");
            let meta_base = (self.cur_rows as i64 * METADATA_INPUT_NUM_CHANNELS as i64) as usize;
            let row_metadata = &mut self.metadata_input_nc.data
                [meta_base..meta_base + METADATA_INPUT_NUM_CHANNELS];
            sgf_meta.fill_metadata_row(row_metadata, next_player, board.x_size * board.y_size);
        }

        self.cur_rows += 1;
    }

    /// Write all buffers to a `.npz` zip file.
    ///
    /// Mirrors `TrainingWriteBuffers::writeToZipFile` in `cpp/dataio/trainingwrite.cpp`.
    pub fn write_to_zip_file(&self, file_name: &str) -> Result<(), IOError> {
        let mut zip_file = ZipFile::new(file_name)?;
        let bytes = self
            .binary_input_nchw_packed
            .prepare_header_with_num_rows(self.cur_rows as i64)?;
        zip_file.write_buffer("binaryInputNCHWPacked", &bytes)?;

        let bytes = self
            .global_input_nc
            .prepare_header_with_num_rows(self.cur_rows as i64)?;
        zip_file.write_buffer("globalInputNC", &bytes)?;

        let bytes = self
            .policy_targets_nc_move
            .prepare_header_with_num_rows(self.cur_rows as i64)?;
        zip_file.write_buffer("policyTargetsNCMove", &bytes)?;

        let bytes = self
            .global_targets_nc
            .prepare_header_with_num_rows(self.cur_rows as i64)?;
        zip_file.write_buffer("globalTargetsNC", &bytes)?;

        let bytes = self
            .score_distr_n
            .prepare_header_with_num_rows(self.cur_rows as i64)?;
        zip_file.write_buffer("scoreDistrN", &bytes)?;

        let bytes = self
            .value_targets_nchw
            .prepare_header_with_num_rows(self.cur_rows as i64)?;
        zip_file.write_buffer("valueTargetsNCHW", &bytes)?;

        let bytes = self
            .q_value_targets_nc_move
            .prepare_header_with_num_rows(self.cur_rows as i64)?;
        zip_file.write_buffer("qValueTargetsNCMove", &bytes)?;

        if self.has_metadata_input {
            let bytes = self
                .metadata_input_nc
                .prepare_header_with_num_rows(self.cur_rows as i64)?;
            zip_file.write_buffer("metadataInputNC", &bytes)?;
        }

        zip_file.close()
    }

    /// Write all buffers to a human-readable text stream for debugging.
    ///
    /// Mirrors `TrainingWriteBuffers::writeToTextOstream` in `cpp/dataio/trainingwrite.cpp`.
    pub fn write_to_text_ostream(&self, out: &mut dyn Write) -> Result<(), IOError> {
        fn write_buffer_text<T: NumpyElement + Copy>(
            out: &mut dyn Write,
            name: &str,
            buffer: &NumpyBuffer<T>,
            cur_rows: i64,
            format_elem: &dyn Fn(T) -> String,
        ) -> Result<(), IOError> {
            fn io_err(e: std::io::Error) -> IOError {
                IOError(format!("TrainingWriteBuffers text output failed: {}", e))
            }

            let bytes = buffer.prepare_header_with_num_rows(cur_rows)?;
            writeln!(out, "{}", name).map_err(io_err)?;
            for &b in bytes.iter().take(10) {
                write!(out, "{} ", b as i32).map_err(io_err)?;
            }
            for &b in bytes
                .iter()
                .take(crate::numpy::TOTAL_HEADER_BYTES / 2)
                .skip(10)
            {
                out.write_all(&[b]).map_err(io_err)?;
            }
            writeln!(out).map_err(io_err)?;

            let len = buffer.get_actual_data_len(cur_rows) as usize;
            let row_len = if cur_rows > 0 {
                len / cur_rows as usize
            } else {
                0
            };
            for (i, &v) in buffer.data[..len].iter().enumerate() {
                write!(out, "{} ", format_elem(v)).map_err(io_err)?;
                if row_len > 0 && (i + 1) % row_len == 0 {
                    writeln!(out).map_err(io_err)?;
                }
            }
            writeln!(out).map_err(io_err)?;
            Ok(())
        }

        write_buffer_text(
            out,
            "binaryInputNCHWPacked",
            &self.binary_input_nchw_packed,
            self.cur_rows as i64,
            &|v: u8| format!("{:02X}", v),
        )?;
        write_buffer_text(
            out,
            "globalInputNC",
            &self.global_input_nc,
            self.cur_rows as i64,
            &|v: f32| format!("{}", v),
        )?;
        write_buffer_text(
            out,
            "policyTargetsNCMove",
            &self.policy_targets_nc_move,
            self.cur_rows as i64,
            &|v: i16| format!("{}", v),
        )?;
        write_buffer_text(
            out,
            "globalTargetsNC",
            &self.global_targets_nc,
            self.cur_rows as i64,
            &|v: f32| format!("{}", v),
        )?;
        write_buffer_text(
            out,
            "scoreDistrN",
            &self.score_distr_n,
            self.cur_rows as i64,
            &|v: i8| format!("{}", v as i32),
        )?;
        write_buffer_text(
            out,
            "valueTargetsNCHW",
            &self.value_targets_nchw,
            self.cur_rows as i64,
            &|v: i8| format!("{}", v as i32),
        )?;
        write_buffer_text(
            out,
            "qValueTargetsNCMove",
            &self.q_value_targets_nc_move,
            self.cur_rows as i64,
            &|v: i16| format!("{}", v as i32),
        )?;
        if self.has_metadata_input {
            write_buffer_text(
                out,
                "metadataInputNC",
                &self.metadata_input_nc,
                self.cur_rows as i64,
                &|v: f32| format!("{}", v as i32),
            )?;
        }
        Ok(())
    }
}

/// Pack 0/1 floats into bits, 8 per byte, big-endian-style within each byte.
///
/// Mirrors the static `packBits` helper in `cpp/dataio/trainingwrite.cpp`.
/// `bits` must have length at least `(binary_floats.len() + 7) / 8`.
pub fn pack_bits(binary_floats: &[f32], bits: &mut [u8]) {
    let len = binary_floats.len();
    assert!(bits.len() * 8 >= len);

    for i in (0..len).step_by(8) {
        if i + 8 <= len {
            bits[i >> 3] = ((binary_floats[i] != 0.0) as u8) << 7
                | ((binary_floats[i + 1] != 0.0) as u8) << 6
                | ((binary_floats[i + 2] != 0.0) as u8) << 5
                | ((binary_floats[i + 3] != 0.0) as u8) << 4
                | ((binary_floats[i + 4] != 0.0) as u8) << 3
                | ((binary_floats[i + 5] != 0.0) as u8) << 2
                | ((binary_floats[i + 6] != 0.0) as u8) << 1
                | ((binary_floats[i + 7] != 0.0) as u8);
        } else {
            let mut b = 0u8;
            for di in 0..(len - i) {
                if binary_floats[i + di] != 0.0 {
                    b |= 1u8 << (7 - di);
                }
            }
            bits[i >> 3] = b;
        }
    }
}

/// Clamp `x` to an integer in `[-120, 120]`, randomizing to keep the expectation exact.
///
/// Mirrors the static `clampToRadius120` helper in `cpp/dataio/trainingwrite.cpp`.
pub fn clamp_to_radius120(x: f32, rand: &mut Rand) -> i8 {
    let low = x.floor() as i32;
    let high = low + 1;
    if low < -120 {
        return -120;
    }
    if high > 120 {
        return 120;
    }
    let lambda = (x - low as f32) as f64;
    if lambda == 0.0 {
        low as i8
    } else if rand.next_bool(lambda) {
        high as i8
    } else {
        low as i8
    }
}

/// Clamp `x` to an integer in `[-32000, 32000]`, randomizing to keep the expectation exact.
///
/// Mirrors the static `clampToRadius32000` helper in `cpp/dataio/trainingwrite.cpp`.
pub fn clamp_to_radius32000(x: f32, rand: &mut Rand) -> i16 {
    let low = x.floor() as i32;
    let high = low + 1;
    if low < -32000 {
        return -32000;
    }
    if high > 32000 {
        return 32000;
    }
    let lambda = (x - low as f32) as f64;
    if lambda == 0.0 {
        low as i16
    } else if rand.next_bool(lambda) {
        high as i16
    } else {
        low as i16
    }
}

/// Expand sparse policy target moves into a dense target plane.
///
/// Mirrors the static `fillPolicyTarget` helper in `cpp/dataio/trainingwrite.cpp`.
/// `target` must have length `data_x_len * data_y_len + 1`.
pub fn fill_policy_target(
    policy_target_moves: &[PolicyTargetMove],
    data_x_len: i32,
    data_y_len: i32,
    board_x_size: i32,
    target: &mut [i16],
) {
    let policy_size = data_x_len * data_y_len + 1;
    assert_eq!(target.len(), policy_size as usize);
    target.fill(0);

    for mv in policy_target_moves {
        let pos = nn_pos::loc_to_pos(mv.loc, board_x_size, data_x_len, data_y_len);
        assert!(pos >= 0 && pos < policy_size);
        target[pos as usize] = mv.policy_target;
    }
}

/// Expand sparse Q-value target moves into dense planes.
///
/// Mirrors the static `fillQValueTarget` helper in `cpp/dataio/trainingwrite.cpp`.
/// `target` must have length `3 * (data_x_len * data_y_len + 1)`.
pub fn fill_q_value_target(
    white_q_value_targets: &[QValueTargetMove],
    next_player: Player,
    data_x_len: i32,
    data_y_len: i32,
    board_x_size: i32,
    target: &mut [i16],
    rand: &mut Rand,
) {
    let policy_size = data_x_len * data_y_len + 1;
    assert_eq!(
        target.len(),
        (QVALUE_SPATIAL_TARGET_NUM_CHANNELS * policy_size as i64) as usize
    );
    target.fill(0);

    let score_target_cap =
        (nn_pos::MAX_BOARD_AREA as i32 + nn_pos::EXTRA_SCORE_DISTR_RADIUS) as f32;

    for entry in white_q_value_targets {
        let pos = nn_pos::loc_to_pos(entry.loc, board_x_size, data_x_len, data_y_len);
        assert!(pos >= 0 && pos < policy_size);

        let win_loss = if next_player == P_WHITE {
            entry.win_loss
        } else {
            -entry.win_loss
        };
        let mut score = if next_player == P_WHITE {
            entry.score
        } else {
            -entry.score
        };
        if score > score_target_cap {
            score = score_target_cap;
        }
        if score < -score_target_cap {
            score = -score_target_cap;
        }

        target[pos as usize] = clamp_to_radius32000(win_loss * 32000.0f32, rand);
        target[(pos + policy_size) as usize] = clamp_to_radius32000(score * 60.0f32, rand);
        target[(pos + policy_size * 2) as usize] = entry.visits.clamp(0, 32000) as i16;
    }
}

/// Fill a policy target plane with a uniform distribution.
///
/// Mirrors the static `uniformPolicyTarget` helper in `cpp/dataio/trainingwrite.cpp`.
/// `target` must have length `policy_size`.
pub fn uniform_policy_target(policy_size: i32, target: &mut [i16]) {
    assert_eq!(target.len(), policy_size as usize);
    target.fill(1);
}

/// Fill TD-like value targets by exponentially weighting future targets.
///
/// Mirrors the static `fillValueTDTargets` helper in `cpp/dataio/trainingwrite.cpp`.
/// `buf` must have length at least 4; it receives `[win, loss, no_result, score]`
/// from the perspective of `next_player`.
pub fn fill_value_td_targets(
    white_value_targets: &[ValueTargets],
    idx: usize,
    next_player: Player,
    now_factor: f64,
    buf: &mut [f32],
) {
    assert!(buf.len() >= 4);
    assert!(idx < white_value_targets.len());

    let mut win_value = 0.0f64;
    let mut loss_value = 0.0f64;
    let mut no_result_value = 0.0f64;
    let mut score = 0.0f64;

    let mut weight_left = 1.0f64;
    for i in idx..white_value_targets.len() {
        let weight_now = if i == white_value_targets.len() - 1 {
            weight_left
        } else {
            let w = weight_left * now_factor;
            weight_left *= 1.0 - now_factor;
            w
        };

        let targets = &white_value_targets[i];
        win_value += weight_now
            * if next_player == P_WHITE {
                targets.win as f64
            } else {
                targets.loss as f64
            };
        loss_value += weight_now
            * if next_player == P_WHITE {
                targets.loss as f64
            } else {
                targets.win as f64
            };
        no_result_value += weight_now * targets.no_result as f64;
        score += weight_now
            * if next_player == P_WHITE {
                targets.score as f64
            } else {
                -targets.score as f64
            };
    }

    let score_target_cap =
        (nn_pos::MAX_BOARD_AREA as i32 + nn_pos::EXTRA_SCORE_DISTR_RADIUS) as f64;
    score = score.clamp(-score_target_cap, score_target_cap);

    buf[0] = win_value as f32;
    buf[1] = loss_value as f32;
    buf[2] = no_result_value as f32;
    buf[3] = score as f32;
}

/// Top-level writer that batches finished games into `.npz` training files.
///
/// Mirrors `TrainingDataWriter` in `cpp/dataio/trainingwrite.cpp`.
pub struct TrainingDataWriter {
    pub output_dir: String,
    pub inputs_version: i32,
    pub rand: Rand,
    pub write_buffers: TrainingWriteBuffers,

    pub debug_out: Option<Box<dyn std::io::Write + Send>>,
    pub debug_only_write_every: i32,
    pub row_count: i64,

    pub is_first_file: bool,
    pub first_file_max_rows: i32,
}

impl TrainingDataWriter {
    /// Create a writer that outputs `.npz` files under `output_dir`.
    pub fn new(
        output_dir: &str,
        inputs_version: i32,
        max_rows_per_file: i32,
        first_file_min_rand_prop: f64,
        data_x_len: i32,
        data_y_len: i32,
        rand_seed: &str,
    ) -> Result<Self, IOError> {
        Self::new_with_options(
            output_dir,
            None,
            inputs_version,
            max_rows_per_file,
            first_file_min_rand_prop,
            data_x_len,
            data_y_len,
            1,
            rand_seed,
        )
    }

    /// Create a writer that emits human-readable debug rows to `debug_out`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_debug(
        debug_out: Box<dyn std::io::Write + Send>,
        inputs_version: i32,
        max_rows_per_file: i32,
        first_file_min_rand_prop: f64,
        data_x_len: i32,
        data_y_len: i32,
        only_every: i32,
        rand_seed: &str,
    ) -> Result<Self, IOError> {
        Self::new_with_options(
            "",
            Some(debug_out),
            inputs_version,
            max_rows_per_file,
            first_file_min_rand_prop,
            data_x_len,
            data_y_len,
            only_every,
            rand_seed,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_options(
        output_dir: &str,
        debug_out: Option<Box<dyn std::io::Write + Send>>,
        inputs_version: i32,
        max_rows_per_file: i32,
        first_file_min_rand_prop: f64,
        data_x_len: i32,
        data_y_len: i32,
        only_every: i32,
        rand_seed: &str,
    ) -> Result<Self, IOError> {
        let (num_binary_channels, num_global_channels) = match inputs_version {
            3 => (NUM_FEATURES_SPATIAL_V3, NUM_FEATURES_GLOBAL_V3),
            4 => (NUM_FEATURES_SPATIAL_V4, NUM_FEATURES_GLOBAL_V4),
            5 => (NUM_FEATURES_SPATIAL_V5, NUM_FEATURES_GLOBAL_V5),
            6 => (NUM_FEATURES_SPATIAL_V6, NUM_FEATURES_GLOBAL_V6),
            7 => (NUM_FEATURES_SPATIAL_V7, NUM_FEATURES_GLOBAL_V7),
            _ => {
                return Err(IOError(format!(
                    "TrainingDataWriter: Unsupported inputs version: {}",
                    inputs_version
                )));
            }
        };

        if !(0.0..=1.0).contains(&first_file_min_rand_prop) {
            return Err(IOError(format!(
                "TrainingDataWriter: first_file_min_rand_prop not in [0,1]: {}",
                first_file_min_rand_prop
            )));
        }

        let mut rand = Rand::new_from_seed(rand_seed);
        let first_file_max_rows = if first_file_min_rand_prop >= 1.0 {
            max_rows_per_file
        } else {
            max_rows_per_file
                - (max_rows_per_file as f64 * (1.0 - first_file_min_rand_prop) * rand.next_double())
                    as i32
        };

        let write_buffers = TrainingWriteBuffers::new(
            inputs_version,
            max_rows_per_file,
            num_binary_channels,
            num_global_channels,
            data_x_len,
            data_y_len,
            false,
        )?;

        Ok(Self {
            output_dir: output_dir.to_string(),
            inputs_version,
            rand,
            write_buffers,
            debug_out,
            debug_only_write_every: only_every,
            row_count: 0,
            is_first_file: true,
            first_file_max_rows,
        })
    }

    /// True if no rows have been buffered yet.
    pub fn is_empty(&self) -> bool {
        self.write_buffers.cur_rows <= 0
    }

    /// Number of rows currently buffered.
    pub fn num_rows_in_buffer(&self) -> i64 {
        self.write_buffers.cur_rows as i64
    }

    /// Flush the current buffers if they are full.
    pub fn write_and_clear_if_full(&mut self) -> Result<(), IOError> {
        if self.write_buffers.cur_rows >= self.write_buffers.max_rows
            || (self.is_first_file && self.write_buffers.cur_rows >= self.first_file_max_rows)
        {
            self.flush_if_nonempty()?;
        }
        Ok(())
    }

    /// Flush the current buffers to disk if they contain any rows.
    ///
    /// In debug mode the returned string is empty; in file mode it is the
    /// path of the produced `.npz` file.
    pub fn flush_if_nonempty(&mut self) -> Result<Option<String>, IOError> {
        if self.write_buffers.cur_rows <= 0 {
            return Ok(None);
        }

        self.is_first_file = false;

        if let Some(mut debug_out) = self.debug_out.take() {
            self.write_buffers.write_to_text_ostream(&mut *debug_out)?;
            self.write_buffers.clear();
            self.debug_out = Some(debug_out);
            Ok(Some(String::new()))
        } else {
            let file_name = format!("{}/{:016X}.npz", self.output_dir, self.rand.next_u64());
            let tmp_file_name = format!("{}.tmp", file_name);
            self.write_buffers.write_to_zip_file(&tmp_file_name)?;
            self.write_buffers.clear();
            std::fs::rename(&tmp_file_name, &file_name).map_err(|e| {
                IOError(format!(
                    "Could not rename {} to {}: {}",
                    tmp_file_name, file_name, e
                ))
            })?;
            Ok(Some(file_name))
        }
    }

    /// Write all rows for a finished game into the buffer, flushing as needed.
    pub fn write_game(&mut self, data: &FinishedGameData) -> Result<(), IOError> {
        let num_moves =
            data.end_hist.move_history.len() as i32 - data.start_hist.move_history.len() as i32;
        assert!(num_moves >= 0);
        assert!(data.start_hist.move_history.len() <= data.end_hist.move_history.len());
        assert!(data.end_hist.move_history.len() <= 100_000_000);
        assert_eq!(data.target_weight_by_turn.len(), num_moves as usize);
        assert_eq!(
            data.target_weight_by_turn_unrounded.len(),
            num_moves as usize
        );
        assert_eq!(data.policy_targets_by_turn.len(), num_moves as usize);
        assert_eq!(data.policy_surprise_by_turn.len(), num_moves as usize);
        assert_eq!(data.policy_entropy_by_turn.len(), num_moves as usize);
        assert_eq!(data.search_entropy_by_turn.len(), num_moves as usize);
        assert_eq!(
            data.white_value_targets_by_turn.len(),
            num_moves as usize + 1
        );
        assert_eq!(data.white_q_value_targets_by_turn.len(), num_moves as usize);
        assert_eq!(data.nn_raw_stats_by_turn.len(), num_moves as usize);

        // Sanity checks on the terminal position.
        {
            let last_targets =
                &data.white_value_targets_by_turn[data.white_value_targets_by_turn.len() - 1];
            if !data.end_hist.is_game_finished {
                assert!(data.hit_turn_limit);
            } else if data.end_hist.is_no_result {
                assert_eq!(last_targets.win, 0.0f32);
                assert_eq!(last_targets.loss, 0.0f32);
                assert_eq!(last_targets.no_result, 1.0f32);
            } else if data.end_hist.winner == P_WHITE {
                assert_eq!(last_targets.win, 1.0f32);
                assert_eq!(last_targets.loss, 0.0f32);
                assert_eq!(last_targets.no_result, 0.0f32);
            } else {
                assert_eq!(last_targets.no_result, 0.0f32);
            }

            assert!(!data.final_full_area.is_empty());
            assert!(!data.final_ownership.is_empty());
            assert!(!data.final_seki_areas.is_empty());
            assert!(!data.final_white_scoring.is_empty());
            assert!(!data.end_hist.is_resignation);
        }

        assert!(data.has_full_data);

        // Replay the game once to collect future board states.
        let mut pos_hist_for_future_boards = Vec::with_capacity(num_moves as usize + 1);
        {
            let mut board = data.start_board.clone();
            let mut hist = data.start_hist.clone();
            let mut next_player = data.start_pla;
            pos_hist_for_future_boards.push(board.clone());

            let start_turn_idx = data.start_hist.move_history.len() as i32;
            for turn_after_start in 0..num_moves {
                let turn_idx = turn_after_start + start_turn_idx;
                let m = data.end_hist.move_history[turn_idx as usize];
                assert_eq!(m.pla, next_player);
                assert!(hist.is_legal(&board, m.loc, m.pla));
                hist.make_board_move_assume_legal(&mut board, m.loc, m.pla);
                next_player = get_opp(next_player);

                pos_hist_for_future_boards.push(board.clone());
            }
        }

        // Write main game rows.
        {
            let mut board = data.start_board.clone();
            let mut hist = data.start_hist.clone();
            let mut next_player = data.start_pla;
            let start_turn_idx = data.start_hist.move_history.len() as i32;

            for turn_after_start in 0..num_moves {
                let mut target_weight = data.target_weight_by_turn[turn_after_start as usize];
                let turn_idx = turn_after_start + start_turn_idx;

                let unreduced_num_visits =
                    data.policy_targets_by_turn[turn_after_start as usize].unreduced_num_visits;
                let policy_target0 = data.policy_targets_by_turn[turn_after_start as usize]
                    .policy_targets
                    .as_slice();
                let policy_target1 = if turn_after_start + 1 < num_moves {
                    Some(
                        data.policy_targets_by_turn[turn_after_start as usize + 1]
                            .policy_targets
                            .as_slice(),
                    )
                } else {
                    None
                };
                let is_side_position = false;
                let value_target_weight = 1.0f32;
                let td_value_target_weight = 1.0f32;
                let lead_target_weight_factor = 1.0f32;

                let mut num_neural_nets_behind_latest = 0;
                for (i, net) in data.changed_neural_nets.iter().enumerate() {
                    if net.turn_idx > turn_idx {
                        num_neural_nets_behind_latest = (data.changed_neural_nets.len() - i) as i32;
                        break;
                    }
                }

                while target_weight > 0.0 {
                    if target_weight >= 1.0f32 || self.rand.next_bool(target_weight as f64) {
                        if self.debug_out.is_none()
                            || self.row_count % self.debug_only_write_every as i64 == 0
                        {
                            self.write_buffers.add_row(
                                &board,
                                &hist,
                                next_player,
                                &data.start_hist,
                                &data.end_hist,
                                turn_idx,
                                data.training_weight as f32,
                                unreduced_num_visits,
                                Some(policy_target0),
                                policy_target1,
                                data.policy_surprise_by_turn[turn_after_start as usize],
                                data.policy_entropy_by_turn[turn_after_start as usize],
                                data.search_entropy_by_turn[turn_after_start as usize],
                                &data.white_value_targets_by_turn,
                                &data.white_q_value_targets_by_turn,
                                turn_after_start as usize,
                                value_target_weight,
                                td_value_target_weight,
                                lead_target_weight_factor,
                                &data.nn_raw_stats_by_turn[turn_after_start as usize],
                                Some(data.end_hist.get_recent_board(0)),
                                Some(&data.final_full_area),
                                Some(&data.final_ownership),
                                Some(&data.final_white_scoring),
                                Some(&pos_hist_for_future_boards),
                                is_side_position,
                                num_neural_nets_behind_latest,
                                data.draw_equivalent_wins_for_white,
                                data.playout_doubling_advantage_pla,
                                data.playout_doubling_advantage,
                                data.game_hash,
                                &data.changed_neural_nets,
                                data.hit_turn_limit,
                                data.num_extra_black,
                                data.mode,
                                None,
                                &mut self.rand,
                            );
                            self.write_and_clear_if_full()?;
                        }
                        self.row_count += 1;
                    }
                    target_weight -= 1.0f32;
                }

                let m = data.end_hist.move_history[turn_idx as usize];
                assert_eq!(m.pla, next_player);
                assert!(hist.is_legal(&board, m.loc, m.pla));
                hist.make_board_move_assume_legal(&mut board, m.loc, m.pla);
                next_player = get_opp(next_player);
            }
        }

        // Write side positions.
        let mut white_value_targets_buf = vec![ValueTargets::default()];
        let mut white_q_value_targets_buf = vec![QValueTargets::default()];
        for sp in &data.side_positions {
            let mut target_weight = sp.target_weight;
            while target_weight > 0.0 {
                if target_weight >= 1.0f32 || self.rand.next_bool(target_weight as f64) {
                    if self.debug_out.is_none()
                        || self.row_count % self.debug_only_write_every as i64 == 0
                    {
                        let turn_idx = sp.hist.move_history.len() as i32;
                        assert!(turn_idx >= data.start_hist.move_history.len() as i32);
                        white_value_targets_buf[0] = sp.white_value_targets.clone();
                        white_q_value_targets_buf[0] = sp.white_q_value_targets.clone();
                        let is_side_position = true;
                        let num_neural_nets_behind_latest = data.changed_neural_nets.len() as i32
                            - sp.num_neural_net_changes_so_far;
                        let value_target_weight = 1.0f32;
                        let td_value_target_weight = 1.0f32;
                        let lead_target_weight_factor = 1.0f32;

                        self.write_buffers.add_row(
                            &sp.board,
                            &sp.hist,
                            sp.pla,
                            &data.start_hist,
                            &data.end_hist,
                            turn_idx,
                            data.training_weight as f32,
                            sp.unreduced_num_visits,
                            Some(&sp.policy_target),
                            None,
                            sp.policy_surprise,
                            sp.policy_entropy,
                            sp.search_entropy,
                            &white_value_targets_buf,
                            &white_q_value_targets_buf,
                            0,
                            value_target_weight,
                            td_value_target_weight,
                            lead_target_weight_factor,
                            &sp.nn_raw_stats,
                            None,
                            None,
                            None,
                            None,
                            None,
                            is_side_position,
                            num_neural_nets_behind_latest,
                            data.draw_equivalent_wins_for_white,
                            sp.playout_doubling_advantage_pla,
                            sp.playout_doubling_advantage,
                            data.game_hash,
                            &data.changed_neural_nets,
                            data.hit_turn_limit,
                            data.num_extra_black,
                            data.mode,
                            None,
                            &mut self.rand,
                        );
                        self.write_and_clear_if_full()?;
                    }
                    self.row_count += 1;
                }
                target_weight -= 1.0f32;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kata_game::board::P_BLACK;
    use kata_game::rules::Rules;

    #[test]
    fn test_mode_constants() {
        assert_eq!(mode::NORMAL, 0);
        assert_eq!(mode::ASYM, 7);
        assert_eq!(mode::COUNT, 8);
    }

    #[test]
    fn test_training_write_buffers_shapes() {
        let buffers = TrainingWriteBuffers::new(7, 16, 22, 27, 19, 19, true).unwrap();
        assert_eq!(buffers.cur_rows, 0);
        assert_eq!(buffers.data_x_len, 19);
        assert_eq!(buffers.data_y_len, 19);
        assert_eq!(buffers.packed_board_area, (19 * 19 + 7) / 8);

        assert_eq!(
            buffers.binary_input_nchw_packed.shape,
            vec![16i64, 22, (19 * 19 + 7) / 8]
        );
        assert_eq!(buffers.global_input_nc.shape, vec![16i64, 27]);
        assert_eq!(
            buffers.policy_targets_nc_move.shape,
            vec![16i64, 2, 19 * 19 + 1]
        );
        assert_eq!(buffers.global_targets_nc.shape, vec![16i64, 64]);
        assert_eq!(
            buffers.score_distr_n.shape,
            vec![
                16i64,
                19 * 19 * 2 + (nn_pos::EXTRA_SCORE_DISTR_RADIUS * 2) as i64
            ]
        );
        assert_eq!(buffers.value_targets_nchw.shape, vec![16i64, 5, 19, 19]);
        assert_eq!(
            buffers.q_value_targets_nc_move.shape,
            vec![16i64, 3, 19 * 19 + 1]
        );
        assert_eq!(buffers.metadata_input_nc.shape, vec![16i64, 192]);
    }

    #[test]
    fn test_training_write_buffers_no_metadata_uses_single_row() {
        let buffers = TrainingWriteBuffers::new(7, 16, 22, 27, 19, 19, false).unwrap();
        assert_eq!(buffers.metadata_input_nc.shape, vec![1i64, 192]);
    }

    #[test]
    fn test_training_write_buffers_clear_resets_counter() {
        let mut buffers = TrainingWriteBuffers::new(7, 16, 22, 27, 19, 19, true).unwrap();
        buffers.cur_rows = 5;
        buffers.clear();
        assert_eq!(buffers.cur_rows, 0);
    }

    #[test]
    fn test_finished_game_data_default() {
        let data = FinishedGameData::default();
        assert!(!data.hit_turn_limit);
        assert!(data.side_positions.is_empty());
        assert!(data.changed_neural_nets.is_empty());
    }

    #[test]
    fn test_pack_bits() {
        let floats = vec![
            1.0f32, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, // first byte: 10110010 = 0xB2
            1.0f32, 1.0, 0.0, // second byte partial: 11000000 = 0xC0
        ];
        let mut bits = vec![0u8; 2];
        pack_bits(&floats, &mut bits);
        assert_eq!(bits[0], 0b1011_0010);
        assert_eq!(bits[1], 0b1100_0000);
    }

    #[test]
    fn test_clamp_to_radius120() {
        let mut rand = Rand::new_from_u64(12345);
        assert_eq!(clamp_to_radius120(0.0f32, &mut rand), 0);
        assert_eq!(clamp_to_radius120(-120.5f32, &mut rand), -120);
        assert_eq!(clamp_to_radius120(120.5f32, &mut rand), 120);
        // Integer values should be exact without randomness.
        assert_eq!(clamp_to_radius120(7.0f32, &mut rand), 7);
    }

    #[test]
    fn test_clamp_to_radius32000() {
        let mut rand = Rand::new_from_u64(12345);
        assert_eq!(clamp_to_radius32000(0.0f32, &mut rand), 0);
        assert_eq!(clamp_to_radius32000(-40000.0f32, &mut rand), -32000);
        assert_eq!(clamp_to_radius32000(40000.0f32, &mut rand), 32000);
        assert_eq!(clamp_to_radius32000(7.0f32, &mut rand), 7);
    }

    #[test]
    fn test_fill_policy_target() {
        let mut target = vec![0i16; 5 * 5 + 1];
        // Use a loc that maps to a known pos on a 5x5 board with data 5x5.
        let loc = kata_game::board::location::get_loc(0, 0, 5);
        let moves = vec![PolicyTargetMove {
            loc,
            policy_target: 1234,
        }];
        fill_policy_target(&moves, 5, 5, 5, &mut target);
        assert_eq!(target[0], 1234);
        assert!(target.iter().skip(1).all(|&v| v == 0));
    }

    #[test]
    fn test_fill_q_value_target() {
        let mut target = vec![0i16; 3 * (5 * 5 + 1)];
        let loc = kata_game::board::location::get_loc(0, 0, 5);
        let entries = vec![QValueTargetMove {
            loc,
            win_loss: 0.5f32,
            score: 12.0f32,
            visits: 7,
        }];
        let mut rand = Rand::new_from_u64(42);
        fill_q_value_target(&entries, P_WHITE, 5, 5, 5, &mut target, &mut rand);

        assert!(target[0] > 0);
        assert!(target[5 * 5 + 1] > 0);
        assert_eq!(target[2 * (5 * 5 + 1)], 7);
    }

    #[test]
    fn test_uniform_policy_target() {
        let mut target = vec![0i16; 5 * 5 + 1];
        uniform_policy_target(5 * 5 + 1, &mut target);
        assert!(target.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_fill_value_td_targets_now_factor_zero() {
        let targets = vec![
            ValueTargets {
                win: 0.5,
                loss: 0.3,
                no_result: 0.2,
                score: 2.0,
                has_lead: false,
                lead: 0.0,
            },
            ValueTargets {
                win: 0.6,
                loss: 0.2,
                no_result: 0.2,
                score: 3.0,
                has_lead: false,
                lead: 0.0,
            },
        ];
        let mut buf = [0.0f32; 4];
        fill_value_td_targets(&targets, 0, P_WHITE, 0.0, &mut buf);
        // With nowFactor 0, all weight lands on the final entry.
        assert_eq!(buf[0], targets[1].win);
        assert_eq!(buf[1], targets[1].loss);
        assert_eq!(buf[2], targets[1].no_result);
        assert_eq!(buf[3], targets[1].score);
    }

    #[test]
    fn test_fill_value_td_targets_perspective_flip() {
        let targets = vec![ValueTargets {
            win: 0.5,
            loss: 0.3,
            no_result: 0.2,
            score: 2.0,
            has_lead: false,
            lead: 0.0,
        }];
        let mut buf_white = [0.0f32; 4];
        let mut buf_black = [0.0f32; 4];
        fill_value_td_targets(&targets, 0, P_WHITE, 0.0, &mut buf_white);
        fill_value_td_targets(&targets, 0, P_BLACK, 0.0, &mut buf_black);
        assert_eq!(buf_white[0], 0.5);
        assert_eq!(buf_black[0], 0.3); // win from black's perspective is white's loss
        assert_eq!(buf_white[3], 2.0);
        assert_eq!(buf_black[3], -2.0);
    }

    #[test]
    fn test_add_row_increments_counter_and_sets_global_targets() {
        let mut buffers = TrainingWriteBuffers::new(7, 16, 22, 19, 7, 7, false).unwrap();
        let board = Board::new(7, 7);
        let hist = BoardHistory::new(board.clone(), P_WHITE, Rules::default(), 0);
        let white_value_targets = vec![ValueTargets {
            win: 0.5,
            loss: 0.3,
            no_result: 0.2,
            score: 2.5,
            has_lead: true,
            lead: 2.5,
        }];
        let white_q_value_targets = vec![QValueTargets::default()];
        let nn_raw_stats = NNRawStats::default();
        let mut rand = Rand::new_from_u64(42);

        buffers.add_row(
            &board,
            &hist,
            P_WHITE,
            &hist,
            &hist,
            0,
            1.0,
            100,
            None,
            None,
            0.5,
            0.4,
            0.3,
            &white_value_targets,
            &white_q_value_targets,
            0,
            1.0,
            1.0,
            1.0,
            &nn_raw_stats,
            None,
            None,
            None,
            None,
            None,
            false,
            0,
            0.5,
            C_EMPTY,
            0.0,
            Hash128::new(0x123456789abcdef0, 0xfedcba9876543210),
            &[],
            false,
            0,
            mode::NORMAL,
            None,
            &mut rand,
        );

        assert_eq!(buffers.cur_rows, 1);

        // Spot-check a few global target channels.
        let row = buffers.global_targets_nc.data.as_slice();
        let row_start = 0;
        let row_slice = &row[row_start..row_start + GLOBAL_TARGET_NUM_CHANNELS as usize];
        assert_eq!(row_slice[25], 1.0); // target_weight
        assert_eq!(row_slice[26], 0.0); // no policyTarget0
        assert_eq!(row_slice[28], 0.0); // no policyTarget1
        assert_eq!(row_slice[0], white_value_targets[0].win); // TD nowFactor 0
        assert_eq!(row_slice[21], white_value_targets[0].lead); // lead
        assert_eq!(row_slice[29], 1.0); // lead weight
        assert_eq!(row_slice[30], 0.5); // policySurprise
        assert_eq!(row_slice[31], 0.4); // policyEntropy
        assert_eq!(row_slice[32], 0.3); // searchEntropy
        assert_eq!(row_slice[63], 2.0); // version
    }

    #[test]
    fn test_write_to_zip_file_creates_valid_npz() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test.npz");

        let mut buffers = TrainingWriteBuffers::new(7, 2, 22, 19, 7, 7, false).unwrap();
        let board = Board::new(7, 7);
        let hist = BoardHistory::new(board.clone(), P_WHITE, Rules::default(), 0);
        let white_value_targets = vec![ValueTargets {
            win: 0.5,
            loss: 0.3,
            no_result: 0.2,
            score: 2.5,
            has_lead: false,
            lead: 0.0,
        }];
        let white_q_value_targets = vec![QValueTargets::default()];
        let nn_raw_stats = NNRawStats::default();
        let mut rand = Rand::new_from_u64(42);

        buffers.add_row(
            &board,
            &hist,
            P_WHITE,
            &hist,
            &hist,
            0,
            1.0,
            100,
            None,
            None,
            0.0,
            0.0,
            0.0,
            &white_value_targets,
            &white_q_value_targets,
            0,
            1.0,
            1.0,
            1.0,
            &nn_raw_stats,
            None,
            None,
            None,
            None,
            None,
            false,
            0,
            0.5,
            C_EMPTY,
            0.0,
            Hash128::default(),
            &[],
            false,
            0,
            mode::NORMAL,
            None,
            &mut rand,
        );

        buffers.write_to_zip_file(path.to_str().unwrap()).unwrap();

        assert!(path.exists());
        let file = std::fs::File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        assert_eq!(archive.len(), 7);
        assert!(archive.by_name("globalTargetsNC").is_ok());
        assert!(archive.by_name("valueTargetsNCHW").is_ok());
    }

    #[test]
    fn test_training_data_writer_new_and_empty() {
        let writer = TrainingDataWriter::new(".", 7, 16, 1.0, 7, 7, "testseed").unwrap();
        assert!(writer.is_empty());
        assert_eq!(writer.num_rows_in_buffer(), 0);
        assert_eq!(writer.first_file_max_rows, 16);
    }

    #[test]
    fn test_training_data_writer_rejects_bad_version() {
        assert!(TrainingDataWriter::new(".", 2, 16, 1.0, 7, 7, "testseed").is_err());
        assert!(TrainingDataWriter::new(".", 8, 16, 1.0, 7, 7, "testseed").is_err());
    }

    #[test]
    fn test_training_data_writer_rejects_bad_rand_prop() {
        assert!(TrainingDataWriter::new(".", 7, 16, -0.1, 7, 7, "testseed").is_err());
        assert!(TrainingDataWriter::new(".", 7, 16, 1.1, 7, 7, "testseed").is_err());
    }
}

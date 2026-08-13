//! Regression test for low-visit root move selection with a scripted NN
//! backend.
//!
//! The engine's search previously ran only against the dummy backend (uniform
//! legal policy + neutral value), which masked a selection bug that showed up
//! only with a real neural net: at maxVisits=5 from an empty board the chosen
//! move was `pass` even though the policy peak was at a normal move.
//!
//! Root cause: GTP runs with `rootSymmetryPruning = true` (the C++ GTP
//! default), which restricts the root's selectable moves to one representative
//! per symmetry orbit. When the policy head is not perfectly symmetric, the
//! policy peak can be a symmetry duplicate whose orbit representative has far
//! lower policy - even lower than pass. With very few visits the first playout
//! then goes to that low-policy representative (pass), and the low-visit
//! chosen move is wrong. Fix: the first visit at the root (and the
//! zero-children direct-policy fallback) selects the global policy argmax,
//! exempting it from the symmetry-duplicate restriction.
//!
//! This test injects a scripted backend whose raw logits mimic a real model
//! (policy peak at a non-pass position, pass probability very low, neutral
//! value) and asserts that low-visit searches pick the policy-peak move
//! rather than pass, both with default params and with the GTP
//! `rootSymmetryPruning` profile.

use std::sync::Arc;

use kata_core::config::ConfigParser;
use kata_core::logger::{Logger, LoggerOptions};
use kata_game::board::{Board, P_BLACK, PASS_LOC};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_nn::backend::{
    Backend, ComputeContext, ComputeHandle, Enabled, InputBuffers, LoadedModel, NNResultBuf,
    NeuralNetError,
};
use kata_nn::desc::ModelDesc;
use kata_nn::eval::NnEvaluator;
use kata_nn::inputs::{NNOutput, nn_pos};
use kata_search::params::SearchParams;
use kata_search::search::Search;

// ---------------------------------------------------------------------------
// Scripted backend
// ---------------------------------------------------------------------------

/// A backend that fills in raw NN logits from a script, mimicking a real
/// model's output distribution (the TRT b11fix model used in the GTP repro).
#[derive(Clone)]
struct ScriptedBackend {
    /// Policy logit for the peak move(s).
    peak_logit: f32,
    /// Policy logit for pass.
    pass_logit: f32,
    /// Policy logit for every other move.
    other_logit: f32,
    /// Logit for `white_win_prob` (loss logit is 0). `0.0` => neutral 0.5.
    win_logit: f32,
    /// Board locations receiving `peak_logit` (the policy-peak set).
    peak_locs: Vec<i16>,
}

impl ScriptedBackend {
    fn new() -> Self {
        Self {
            // Softmax over 361 legal moves: peak ~5.3%, pass ~0.43%, rest ~0.26%.
            peak_logit: 3.0,
            pass_logit: 0.5,
            other_logit: 0.0,
            // Neutral value: P(win) = P(loss) = 0.5.
            win_logit: 0.0,
            peak_locs: vec![PEAK_LOC],
        }
    }

    /// A policy with a symmetric set of eight high-probability peaks (one per
    /// symmetry orbit, so all are selectable under root symmetry pruning) and
    /// a very low pass probability, mimicking the real model's opening policy.
    fn with_eight_peaks() -> Self {
        Self {
            peak_logit: 3.0,
            pass_logit: -2.0,
            other_logit: -4.0,
            win_logit: 0.0,
            peak_locs: vec![
                1 * 19 + 1, // (1,1)
                1 * 19 + 2, // (2,1)
                1 * 19 + 3, // (3,1)
                1 * 19 + 4, // (4,1)
                1 * 19 + 5, // (5,1)
                2 * 19 + 2, // (2,2)
                2 * 19 + 3, // (3,2)
                2 * 19 + 4, // (4,2)
            ],
        }
    }

    /// The peak board locations for this backend.
    fn peak_locs(&self) -> &[i16] {
        &self.peak_locs
    }
}

impl Default for ScriptedBackend {
    fn default() -> Self {
        Self::new()
    }
}

struct ScriptedLoadedModel {
    model_desc: ModelDesc,
}

impl LoadedModel for ScriptedLoadedModel {
    fn model_desc(&self) -> &ModelDesc {
        &self.model_desc
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

struct ScriptedComputeContext;

impl ComputeContext for ScriptedComputeContext {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

struct ScriptedComputeHandle {
    is_warmup: bool,
}

impl ScriptedComputeHandle {
    fn new() -> Self {
        Self { is_warmup: false }
    }
}

impl ComputeHandle for ScriptedComputeHandle {
    fn is_using_fp16(&self) -> bool {
        false
    }
    fn set_is_warmup(&mut self, is_warmup: bool) -> bool {
        let prev = self.is_warmup;
        self.is_warmup = is_warmup;
        prev
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

struct ScriptedInputBuffers;

impl InputBuffers for ScriptedInputBuffers {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Backend for ScriptedBackend {
    fn global_initialize(&self) {}
    fn global_cleanup(&self) {}
    fn print_devices(&self) {}

    fn load_model_file(
        &self,
        _file: &str,
        _expected_sha256: &str,
    ) -> Result<Box<dyn LoadedModel>, NeuralNetError> {
        Ok(Box::new(ScriptedLoadedModel {
            model_desc: ModelDesc {
                model_version: 8,
                ..ModelDesc::default()
            },
        }))
    }

    fn create_compute_context(
        &self,
        _gpu_idxs: &[i32],
        _logger: &Logger,
        _nn_x_len: i32,
        _nn_y_len: i32,
        _home_data_dir_override: &str,
        _use_fp16_mode: Enabled,
        _loaded_model: &dyn LoadedModel,
        _cfg: &kata_core::config::ConfigParser,
    ) -> Result<Box<dyn ComputeContext>, NeuralNetError> {
        Ok(Box::new(ScriptedComputeContext))
    }

    fn create_compute_handle(
        &self,
        _ctx: &dyn ComputeContext,
        _loaded_model: &dyn LoadedModel,
        _logger: &Logger,
        _max_batch_size: i32,
        _require_exact_nn_len: bool,
        _inputs_use_nhwc: bool,
        _gpu_idx_for_this_thread: i32,
        _server_thread_idx: i32,
    ) -> Result<Box<dyn ComputeHandle>, NeuralNetError> {
        Ok(Box::new(ScriptedComputeHandle::new()))
    }

    fn is_using_fp16(&self, handle: &dyn ComputeHandle) -> bool {
        handle.is_using_fp16()
    }

    fn set_is_warmup(&self, handle: &mut dyn ComputeHandle, is_warmup: bool) -> bool {
        handle.set_is_warmup(is_warmup)
    }

    fn create_input_buffers(
        &self,
        _loaded_model: &dyn LoadedModel,
        _max_batch_size: i32,
        _nn_x_len: i32,
        _nn_y_len: i32,
    ) -> Result<Box<dyn InputBuffers>, NeuralNetError> {
        Ok(Box::new(ScriptedInputBuffers))
    }

    fn get_output(
        &self,
        _handle: &dyn ComputeHandle,
        _buffers: &dyn InputBuffers,
        num_batch_elts: i32,
        input_bufs: &mut [&mut NNResultBuf],
        outputs: &mut [&mut NNOutput],
    ) -> Result<(), NeuralNetError> {
        let n = num_batch_elts as usize;
        let pass_pos = nn_pos::loc_to_pos(PASS_LOC, 19, 19, 19) as usize;
        let peak_positions: Vec<usize> = self
            .peak_locs
            .iter()
            .map(|&loc| nn_pos::loc_to_pos(loc, 19, 19, 19) as usize)
            .collect();
        for (_input_buf, output) in input_bufs
            .iter_mut()
            .take(n)
            .zip(outputs.iter_mut().take(n))
        {
            for pos in 0..output.policy_probs.len() {
                output.policy_probs[pos] = if peak_positions.contains(&pos) {
                    self.peak_logit
                } else if pos == pass_pos {
                    self.pass_logit
                } else {
                    self.other_logit
                };
            }
            output.white_win_prob = self.win_logit;
            output.white_loss_prob = 0.0;
            output.white_no_result_prob = -30.0;
        }
        Ok(())
    }
}

/// Policy peak location. Chosen so that it is *not* the symmetry
/// representative of its orbit (GTP rootSymmetryPruning marks it as a
/// duplicate), which is exactly the situation that used to make the search
/// pick pass at low visit counts.
const PEAK_LOC: i16 = 15 * 19 + 3;

/// Build an `NnEvaluator` backed by the scripted backend.
fn scripted_evaluator() -> NnEvaluator {
    scripted_evaluator_with(ScriptedBackend::new())
}

/// Build an `NnEvaluator` backed by a custom scripted backend.
fn scripted_evaluator_with(backend: ScriptedBackend) -> NnEvaluator {
    let logger = Arc::new(Logger::new(LoggerOptions::default(), None));
    let cfg = ConfigParser::new(false, false);
    let mut nn_eval = NnEvaluator::new(
        "scripted-model".to_string(),
        "/dev/null".to_string(),
        String::new(),
        logger.clone(),
        16,  // max_batch_size
        19,  // nn_x_len
        19,  // nn_y_len
        false,
        false,
        16,  // nn_cache_size_power_of_two
        12,  // nn_mutex_pool_size_power_of_two
        false, // debug_skip_neural_net: use the scripted backend path
        String::new(),
        Enabled::False,
        1,
        vec![0],
        "scriptedBackendRandSeed".to_string(),
        false,
        0,
        true,
        &cfg,
    );
    nn_eval.set_backend(Arc::new(backend));
    nn_eval
        .load_model()
        .expect("scripted backend should load the model");
    nn_eval.spawn_server_threads();
    nn_eval
}

/// Run a low-visit search from the empty 19x19 board and return the chosen move.
fn run_low_visit_search(nn_eval: &NnEvaluator, max_visits: i64, root_symmetry_pruning: bool) -> i16 {
    let params = SearchParams {
        max_visits,
        // Existing tests avoid the value-weight distribution plumbing by
        // disabling value weighting; the GTP path initializes it separately.
        value_weight_exponent: 0.0,
        root_symmetry_pruning,
        ..SearchParams::default()
    };
    let logger = Arc::new(Logger::new(LoggerOptions::default(), None));
    let mut search = Search::new(params, nn_eval, logger.as_ref(), "lowVisitSearchSeed");

    let board = Board::new(19, 19);
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);
    search.get_chosen_move_loc()
}

#[test]
fn low_visit_search_chooses_policy_peak_not_pass() {
    let nn_eval = scripted_evaluator();

    let chosen = run_low_visit_search(&nn_eval, 5, false);

    assert_ne!(
        chosen,
        PASS_LOC,
        "low-visit search must not choose pass when the policy peak is a normal move"
    );
    assert_eq!(
        chosen, PEAK_LOC,
        "low-visit search should choose the policy peak move, got {}",
        chosen
    );
}

/// GTP runs with rootSymmetryPruning=true; with an asymmetric policy the peak
/// can be a symmetry duplicate. The first visit must still go to the policy
/// peak rather than to pass (which previously won because its policy exceeded
/// every symmetry representative).
#[test]
fn low_visit_search_with_root_symmetry_pruning_chooses_policy_peak_not_pass() {
    let nn_eval = scripted_evaluator();

    for max_visits in [1i64, 2, 5, 10] {
        let chosen = run_low_visit_search(&nn_eval, max_visits, true);
        assert_ne!(
            chosen,
            PASS_LOC,
            "search with maxVisits={} and rootSymmetryPruning must not choose pass",
            max_visits
        );
        assert_eq!(
            chosen, PEAK_LOC,
            "search with maxVisits={} and rootSymmetryPruning should choose the policy peak, got {}",
            max_visits, chosen
        );
    }
}

// ---------------------------------------------------------------------------
// Regression: GTP-style parameters with a symmetric eight-peak policy must
// pick one of the policy peaks, never pass.
//
// Reported failure: `boardsize 19; clear_board; genmove B` returned pass for
// maxVisits = 5/50/500/1000 even though the search tree was healthy (8
// symmetric first moves each with roughly equal visits, pass with 0 visits).
// Root cause was in the GTP layer (crates/katago/src/cmd/gtp.rs):
// `launch_gen_move` set the search parameters but never launched the search,
// so `get_chosen_move_loc` ran on an empty tree and returned NULL_LOC, which
// the GTP handler turned into "pass". This test pins down the search-side
// contract that the GTP layer relies on: with GTP-style parameters and a
// scripted policy that looks like the real model's (eight symmetric peaks,
// very low pass), the chosen move must be one of the peaks at every visit
// count the bug report covered.
// ---------------------------------------------------------------------------

/// The GTP search-parameter profile, mirroring
/// `kata_program::setup::load_params_impl(SetupFor::Gtp)` plus the GTPEngine
/// defaults (conservativePass/fillDameBeforePass forced on, passing hacks on,
/// temperature 0.1/0.5, LCB selection on, etc.).
fn gtp_like_params(max_visits: i64) -> SearchParams {
    let mut p = SearchParams::new();
    p.max_visits = max_visits;
    p.win_loss_utility_factor = 1.0;
    p.static_score_utility_factor = 0.1;
    p.dynamic_score_utility_factor = 0.3;
    p.dynamic_score_center_zero_weight = 0.2;
    p.dynamic_score_center_scale = 0.75;
    p.cpuct_exploration = 1.0;
    p.cpuct_exploration_log = 0.45;
    p.cpuct_exploration_base = 500.0;
    p.cpuct_utility_stdev_prior = 0.40;
    p.cpuct_utility_stdev_prior_weight = 2.0;
    p.cpuct_utility_stdev_scale = 0.85;
    p.fpu_reduction_max = 0.2;
    p.fpu_loss_prop = 0.0;
    p.fpu_parent_weight_by_visited_policy = true;
    p.fpu_parent_weight_by_visited_policy_pow = 2.0;
    p.policy_optimism = 1.0;
    p.value_weight_exponent = 0.25;
    p.use_noise_pruning = true;
    p.noise_prune_utility_scale = 0.15;
    p.use_uncertainty = true;
    p.uncertainty_coeff = 0.25;
    p.uncertainty_exponent = 1.0;
    p.uncertainty_max_weight = 8.0;
    p.use_graph_search = true;
    p.graph_search_rep_bound = 11;
    p.root_noise_enabled = false;
    p.root_dirichlet_noise_total_concentration = 10.83;
    p.root_dirichlet_noise_weight = 0.25;
    p.root_policy_temperature = 1.0;
    p.root_policy_temperature_early = 1.0;
    p.root_fpu_reduction_max = 0.1;
    p.root_fpu_loss_prop = 0.0;
    p.root_num_symmetries_to_sample = 1;
    p.root_symmetry_pruning = true;
    p.root_policy_optimism = 0.2;
    p.chosen_move_temperature = 0.1;
    p.chosen_move_temperature_early = 0.5;
    p.chosen_move_temperature_halflife = 19.0;
    p.chosen_move_temperature_only_below_prob = 1.0;
    p.chosen_move_subtract = 0.0;
    p.chosen_move_prune = 1.0;
    p.use_lcb_for_selection = true;
    p.lcb_stdevs = 5.0;
    p.min_visit_prop_for_lcb = 0.15;
    p.use_non_buggy_lcb = true;
    p.root_ending_bonus_points = 0.5;
    p.root_prune_useless_moves = true;
    p.conservative_pass = true;
    p.fill_dame_before_pass = true;
    p.enable_passing_hacks = true;
    p.enable_more_passing_hacks = true;
    p.anti_mirror = true;
    p.avoid_repeated_pattern_utility = 0.0;
    p
}

/// Run a search with GTP-style parameters and return the chosen move.
fn run_gtp_like_search(nn_eval: &NnEvaluator, max_visits: i64) -> i16 {
    let params = gtp_like_params(max_visits);
    let logger = Arc::new(Logger::new(LoggerOptions::default(), None));
    let mut search = Search::new(params, nn_eval, logger.as_ref(), "gtpLikeSearchSeed");

    let board = Board::new(19, 19);
    let next_pla = P_BLACK;
    let hist = BoardHistory::new(board.clone(), next_pla, Rules::get_tromp_taylorish(), 0);

    search.set_position(next_pla, &board, &hist);
    search.run_whole_search(next_pla);
    search.get_chosen_move_loc()
}

/// The chosen move must be one of the policy peaks, never pass, at every
/// visit count from the bug report (5/50/500). This is the search-side
/// contract that the fixed GTP `launch_gen_move` now relies on.
#[test]
fn gtp_like_search_with_symmetric_peaks_never_chooses_pass() {
    let backend = ScriptedBackend::with_eight_peaks();
    let nn_eval = scripted_evaluator_with(backend.clone());

    for max_visits in [5i64, 50, 500] {
        let chosen = run_gtp_like_search(&nn_eval, max_visits);
        assert_ne!(
            chosen, PASS_LOC,
            "GTP-like search with maxVisits={} must not choose pass when the policy peaks are normal moves",
            max_visits
        );
        assert!(
            backend.peak_locs().contains(&chosen),
            "GTP-like search with maxVisits={} should choose one of the policy peaks, got {}",
            max_visits,
            chosen
        );
    }
}

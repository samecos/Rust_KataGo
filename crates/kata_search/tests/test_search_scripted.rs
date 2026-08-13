//! Regression test for low-visit root move selection with a scripted NN
//! backend.
//!
//! The engine's search previously ran only against the dummy backend (uniform
//! legal policy + neutral value), which masked a selection bug that showed up
//! only with a real neural net: at maxVisits=5 from an empty board the chosen
//! move was `pass` even though the policy peak was at a normal move.
//!
//! This test injects a scripted backend whose raw logits mimic the real model
//! used in the GTP repro (policy peak ~5-6% at a star point, pass ~0.5%, value
//! strongly favoring white), and asserts that a 5-visit search picks the
//! policy-peak move rather than pass.

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
struct ScriptedBackend {
    /// Policy logit for the peak move (a star point).
    peak_logit: f32,
    /// Policy logit for pass.
    pass_logit: f32,
    /// Policy logit for every other move.
    other_logit: f32,
    /// Logit for `white_win_prob` (loss logit is 0), producing ~white 0.96.
    win_logit: f32,
}

impl ScriptedBackend {
    fn new() -> Self {
        Self {
            // Softmax over 361 legal moves: peak ~5.3%, pass ~0.43%, rest ~0.26%.
            peak_logit: 3.0,
            pass_logit: 0.5,
            other_logit: 0.0,
            // e^3.18 / (e^3.18 + 1) ~= 0.96.
            win_logit: 3.18,
        }
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
        let peak_pos = nn_pos::loc_to_pos(PEAK_LOC, 19, 19, 19) as usize;
        let pass_pos = nn_pos::loc_to_pos(PASS_LOC, 19, 19, 19) as usize;        for (input_buf, output) in input_bufs
            .iter_mut()
            .take(n)
            .zip(outputs.iter_mut().take(n))
        {
            for pos in 0..output.policy_probs.len() {
                output.policy_probs[pos] = if pos == peak_pos {
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
            input_buf.has_result = true;
            input_buf.result = Some(Arc::new(NNOutput::default()));
        }
        Ok(())
    }
}

/// Policy peak: the upper-left star point D16 (x=3, y=15).
const PEAK_LOC: i16 = 15 * 19 + 3;

/// Build an `NnEvaluator` backed by the scripted backend.
fn scripted_evaluator() -> NnEvaluator {
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
    nn_eval.set_backend(Arc::new(ScriptedBackend::new()));
    nn_eval
        .load_model()
        .expect("scripted backend should load the model");
    nn_eval.spawn_server_threads();
    nn_eval
}

/// Run a low-visit search from the empty 19x19 board and return the chosen move.
fn run_low_visit_search(nn_eval: &NnEvaluator, max_visits: i64) -> i16 {
    let params = SearchParams {
        max_visits,
        // Existing tests avoid the value-weight distribution plumbing by
        // disabling value weighting; the GTP path initializes it separately.
        value_weight_exponent: 0.0,
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

    let chosen = run_low_visit_search(&nn_eval, 5);

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

#[test]
fn moderate_visit_search_also_chooses_policy_peak_not_pass() {
    let nn_eval = scripted_evaluator();

    for max_visits in [5i64, 10, 20, 50] {
        let chosen = run_low_visit_search(&nn_eval, max_visits);
        assert_ne!(
            chosen,
            PASS_LOC,
            "search with maxVisits={} must not choose pass",
            max_visits
        );
    }
}

//! Temporary probe: sweep value-space to find regimes where pass is chosen.
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

struct ScriptedBackend {
    peak_logit: f32,
    pass_logit: f32,
    other_logit: f32,
    /// Backend (player-to-move) win logit.
    win_logit: f32,
    /// Loss logit.
    loss_logit: f32,
    /// White score mean logit (player-to-move perspective, like the model).
    score_mean_logit: f32,
}

impl ScriptedBackend {
    fn new(win_logit: f32, loss_logit: f32, score_mean_logit: f32) -> Self {
        Self {
            peak_logit: 3.0,
            pass_logit: 0.5,
            other_logit: 0.0,
            win_logit,
            loss_logit,
            score_mean_logit,
        }
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
        let pass_pos = nn_pos::loc_to_pos(PASS_LOC, 19, 19, 19) as usize;
        for (_input_buf, output) in input_bufs.iter_mut().take(n).zip(outputs.iter_mut().take(n)) {
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
            output.white_loss_prob = self.loss_logit;
            output.white_no_result_prob = -30.0;
            output.white_score_mean = self.score_mean_logit;
        }
        Ok(())
    }
}

const PEAK_LOC: i16 = 15 * 19 + 3;

/// Logit such that p(win) = w (with loss logit 0).
fn win_logit_for(w: f32) -> f32 {
    (w / (1.0 - w)).ln()
}

fn scripted_evaluator(win_logit: f32, loss_logit: f32, score_mean: f32) -> NnEvaluator {
    let logger = Arc::new(Logger::new(LoggerOptions::default(), None));
    let cfg = ConfigParser::new(false, false);
    let mut nn_eval = NnEvaluator::new(
        "scripted-model".to_string(),
        "/dev/null".to_string(),
        String::new(),
        logger.clone(),
        16,
        19,
        19,
        false,
        false,
        16,
        12,
        false,
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
    nn_eval.set_backend(Arc::new(ScriptedBackend::new(win_logit, loss_logit, score_mean)));
    nn_eval
        .load_model()
        .expect("scripted backend should load the model");
    nn_eval.spawn_server_threads();
    nn_eval
}

fn run_search(nn_eval: &NnEvaluator, max_visits: i64) -> i16 {
    let params = SearchParams {
        max_visits,
        value_weight_exponent: 0.0,
        // Mimic the GTP genmove param profile.
        conservative_pass: true,
        fill_dame_before_pass: true,
        enable_passing_hacks: true,
        enable_more_passing_hacks: true,
        anti_mirror: true,
        root_symmetry_pruning: true,
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

/// Print a grid: rows = white win at black root (after flip), cols = white win at
/// white-to-move children (after flip). Mark P if chosen == PASS.
#[test]
fn sweep_value_space() {
    let wins = [0.02f32, 0.04, 0.1, 0.2, 0.5, 0.8, 0.9, 0.96, 0.98];
    for &max_visits in &[5i64, 10] {
        println!("=== maxVisits={} ===", max_visits);
        // Headers
        print!("root\\child");
        for &cw in &wins {
            print!(" {:>6}", cw);
        }
        println!();
        for &rw in &wins {
            print!("{:>10}", rw);
            for &cw in &wins {
                // Backend is player-to-move: root is black -> white_win after flip = loss_prob.
                // To get white_win=rw at black root: loss logit = win_logit_for(rw), win logit = 0.
                // At white children: white_win = win_prob = win_logit_for(cw).
                let nn_eval = scripted_evaluator(win_logit_for(cw), win_logit_for(rw), 0.0);
                let chosen = run_search(&nn_eval, max_visits);
                if chosen == PASS_LOC {
                    print!(" {:>6}", "PASS");
                } else if chosen == PEAK_LOC {
                    print!(" {:>6}", "peak");
                } else {
                    print!(" {:>6}", chosen);
                }
            }
            println!();
        }
    }
}

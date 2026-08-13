//! Temporary probe: dump low-visit search internals with the scripted backend.
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
}

impl ScriptedBackend {
    fn new(win_logit: f32, loss_logit: f32) -> Self {
        Self {
            peak_logit: 3.0,
            pass_logit: 0.5,
            other_logit: 0.0,
            win_logit,
            loss_logit,
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
        }
        Ok(())
    }
}

const PEAK_LOC: i16 = 15 * 19 + 3;

fn scripted_evaluator(win_logit: f32, loss_logit: f32) -> NnEvaluator {
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
    nn_eval.set_backend(Arc::new(ScriptedBackend::new(win_logit, loss_logit)));
    nn_eval
        .load_model()
        .expect("scripted backend should load the model");
    nn_eval.spawn_server_threads();
    nn_eval
}

fn dump_search(nn_eval: &NnEvaluator, max_visits: i64) {
    let params = SearchParams {
        max_visits,
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

    let root = search.root_node.as_deref().unwrap();
    if let Some(nn) = root.get_nn_output() {
        let p = nn.policy_probs.clone();
        println!(
            "root policy: pos0={:.6} pos1={:.6} pos288={:.6} pos360={:.6} pos361(pass)={:.6}",
            p[0], p[1], p[288], p[360], p[361]
        );
    }
    println!(
        "root: visits={} weight_sum={} utility_avg={:.4}",
        root.stats.visits.load(std::sync::atomic::Ordering::Acquire),
        root.stats.weight_sum.load(std::sync::atomic::Ordering::Acquire),
        root.stats.utility_avg.load(std::sync::atomic::Ordering::Acquire)
    );
    let children = root.get_children();
    for i in 0..children.get_capacity() {
        let cp = children.get(i);
        let child = match cp.get_if_allocated() {
            Some(c) => c,
            None => break,
        };
        let loc = cp.get_move_loc_relaxed();
        let ev = cp.get_edge_visits();
        let cv = child.stats.visits.load(std::sync::atomic::Ordering::Acquire);
        let u = child.stats.utility_avg.load(std::sync::atomic::Ordering::Acquire);
        let w = child.stats.weight_sum.load(std::sync::atomic::Ordering::Acquire);
        println!(
            "  child loc={} edgeVisits={} childVisits={} utility={:.4} weightSum={:.4}",
            loc, ev, cv, u, w
        );
    }
    let mut locs = Vec::new();
    let mut vals = Vec::new();
    let ok = search.get_play_selection_values(&mut locs, &mut vals, 0.0);
    println!("playSelection ok={}", ok);
    for (l, v) in locs.iter().zip(vals.iter()) {
        println!("  psv loc={} val={:.4}", l, v);
    }
    let chosen = search.get_chosen_move_loc();
    println!("chosen={}", chosen);
}

#[test]
fn probe_scenarios() {
    // Side-to-move perspective (like a sane net): black to move -> white ~0.04.
    let nn_eval = scripted_evaluator(3.18, 0.0);
    println!("=== scenario A: side-to-move wins 0.96 (white_win=0.04 at black root) ===");
    dump_search(&nn_eval, 5);

    // White-perspective bug: net always says white 0.96 -> black root white_win=0.96.
    let nn_eval = scripted_evaluator(-3.18, 0.0);
    println!("=== scenario B: net says white 0.96 always (white_win=0.96 at black root) ===");
    dump_search(&nn_eval, 5);
}

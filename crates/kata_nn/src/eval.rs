//! Neural net evaluator and cache.
//!
//! Corresponds to `cpp/neuralnet/nneval.h` and `cpp/neuralnet/nneval.cpp`.
//! `NNCacheTable` is fully ported; `NNEvaluator` is present as a structural
//! skeleton with getters and stats, while evaluation / server-thread logic is
//! left for a later slice.

use std::cell::UnsafeCell;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use parking_lot::{Condvar, Mutex as AsyncMutex};

use kata_core::config::ConfigParser;
use kata_core::global::StringError;
use kata_core::hash::Hash128;
use kata_core::logger::Logger;
use kata_core::rng::Rand;
use kata_core::thread::queue::ThreadSafeQueue;
use kata_game::board::location;
use kata_game::board::{Board, P_BLACK, PASS_LOC, Player, P_WHITE};
use kata_game::history::BoardHistory;
use kata_game::rules::{KoRule, Rules, ScoringRule};
use kata_game::symmetry::NUM_SYMMETRIES;

use crate::backend::dummy::DummyInputBuffers;
use crate::backend::{
    Backend, ComputeContext, ComputeHandle, Enabled, InputBuffers, LoadedModel, NNResultBuf,
};
use crate::desc::ModelPostProcessParams;
use crate::inputs::{MiscNNInputParams, NNOutput, get_hash, nn_pos};
use crate::sgf_meta::SgfMetadata;
use crate::version::{get_inputs_version, get_num_global_features, get_num_spatial_features};

/// A simple hash table caching `NNOutput` values by `nn_hash`.
///
/// The implementation mirrors C++ `NNCacheTable`: a flat array of entries with
/// a smaller mutex pool guarding access.
pub struct NnCacheTable {
    table_mask: u64,
    entries: Vec<UnsafeCell<Option<Arc<NNOutput>>>>,
    mutexes: Vec<Mutex<()>>,
    mutex_mask: u32,
}

// Safety: entries are only accessed while holding the corresponding mutex.
unsafe impl Send for NnCacheTable {}
unsafe impl Sync for NnCacheTable {}

impl NnCacheTable {
    /// Create a cache with `2^size_power_of_two` entries and
    /// `2^mutex_pool_size_power_of_two` mutexes.
    pub fn new(size_power_of_two: u32, mutex_pool_size_power_of_two: u32) -> Self {
        assert!(size_power_of_two <= 63, "size_power_of_two too large");
        assert!(
            mutex_pool_size_power_of_two <= 31,
            "mutex_pool_size_power_of_two too large"
        );
        let mutex_power = mutex_pool_size_power_of_two.min(size_power_of_two);

        let table_size = 1u64 << size_power_of_two;
        let mutex_count = 1u32 << mutex_power;

        let mut entries = Vec::with_capacity(table_size as usize);
        for _ in 0..table_size {
            entries.push(UnsafeCell::new(None));
        }
        let mutexes: Vec<Mutex<()>> = (0..mutex_count).map(|_| Mutex::new(())).collect();

        Self {
            table_mask: table_size - 1,
            entries,
            mutexes,
            mutex_mask: mutex_count - 1,
        }
    }

    /// Look up an output by hash. Returns `None` if not present.
    pub fn get(&self, nn_hash: Hash128) -> Option<Arc<NNOutput>> {
        let idx = (nn_hash.hash0 & self.table_mask) as usize;
        let mutex_idx = (idx as u32 & self.mutex_mask) as usize;
        let _lock = self.mutexes[mutex_idx].lock().unwrap();

        // Safety: we hold the mutex for this entry.
        let entry = unsafe { &*self.entries[idx].get() };
        entry.as_ref().and_then(|ptr| {
            if ptr.nn_hash == nn_hash {
                Some(Arc::clone(ptr))
            } else {
                None
            }
        })
    }

    /// Store `output` in the cache, indexed by `output.nn_hash`.
    pub fn set(&self, output: &Arc<NNOutput>) {
        let idx = (output.nn_hash.hash0 & self.table_mask) as usize;
        let mutex_idx = (idx as u32 & self.mutex_mask) as usize;
        let _lock = self.mutexes[mutex_idx].lock().unwrap();

        // Safety: we hold the mutex for this entry. The old value is swapped
        // out and dropped after the lock is released, matching the C++ design.
        let old = unsafe { (*self.entries[idx].get()).replace(Arc::clone(output)) };
        drop(_lock);
        drop(old);
    }

    /// Remove all entries from the cache.
    pub fn clear(&self) {
        let mut old_value = None;
        for idx in 0..self.entries.len() {
            let mutex_idx = (idx as u32 & self.mutex_mask) as usize;
            let _lock = self.mutexes[mutex_idx].lock().unwrap();
            // Safety: we hold the mutex for this entry.
            std::mem::swap(unsafe { &mut *self.entries[idx].get() }, &mut old_value);
            drop(_lock);
            old_value = None;
        }
    }
}

impl Drop for NnCacheTable {
    fn drop(&mut self) {
        for entry in &self.entries {
            // Safety: no other thread can be accessing the table during drop.
            unsafe {
                *entry.get() = None;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NNEvaluator skeleton
// ---------------------------------------------------------------------------

/// Neural-network evaluator.
///
/// Mirrors the C++ `NNEvaluator` class. This slice provides the field layout,
/// constructor, getters, cache clearing, and statistics, plus a first slice of
/// the async server-thread evaluation pipeline.
#[allow(dead_code)]
pub struct NnEvaluator {
    model_name: String,
    model_file_name: String,
    expected_sha256: String,
    require_exact_nn_len: bool,
    inputs_use_nhwc: bool,
    using_fp16_mode: Enabled,
    num_threads: i32,
    gpu_idx_by_server_thread: Vec<i32>,
    rand_seed: String,
    debug_skip_neural_net: bool,
    disable_warmup: bool,
    cfg: ConfigParser,

    compute_context: Option<Box<dyn ComputeContext>>,
    loaded_model: Option<Box<dyn LoadedModel>>,
    compute_handle: Option<Box<dyn ComputeHandle>>,
    backend: Option<Arc<dyn Backend>>,
    logger: Arc<Logger>,

    internal_model_name: String,
    model_version: i32,
    inputs_version: i32,
    num_input_meta_channels: i32,

    post_process_params: ModelPostProcessParams,

    num_server_threads_ever_spawned: i32,
    server_threads: Vec<JoinHandle<()>>,

    max_batch_size: i32,

    buffer_mutex: AsyncMutex<()>,
    num_server_threads_starting_up: i32,
    main_thread_waiting_for_spawn: Condvar,
    server_threads_is_using_fp16: Vec<bool>,
    num_ongoing_evals: i32,
    num_waiting_evals: i32,
    num_evals_to_awaken: i32,

    current_do_randomize: AtomicBool,
    current_default_symmetry: AtomicI32,

    shared: Arc<SharedState>,
}

/// State shared between the evaluator and its server threads.
///
/// Keeping shared state in an `Arc` allows server threads to outlive any
/// particular `NnEvaluator` borrow and makes the evaluator safe to move after
/// threads are spawned (the heap-allocated `SharedState` stays put).
struct SharedState {
    query_queue: ThreadSafeQueue<Arc<EvalRequest>>,
    nn_cache_table: Option<Arc<NnCacheTable>>,
    is_killed: AtomicBool,
    current_batch_size: AtomicI32,
    waiting_for_finish: Condvar,
    num_rows_processed: AtomicU64,
    num_batches_processed: AtomicU64,
    nn_x_len: i32,
    nn_y_len: i32,
    policy_size: i32,
    backend: Mutex<Option<Arc<dyn Backend>>>,
    compute_handles: Mutex<Vec<Box<dyn ComputeHandle>>>,
    input_buffers: Mutex<Option<Box<dyn InputBuffers>>>,
    /// Model version and post-processing parameters, set by `load_model`
    /// before any server thread is spawned (then read-only).
    postprocess_cfg: Mutex<(i32, ModelPostProcessParams)>,
}

impl SharedState {
    /// Server-thread entry point.
    fn serve(
        self: &Arc<Self>,
        handle: &Mutex<Box<dyn ComputeHandle>>,
        _server_buf: &mut NnServerBuf,
        _rand: &mut Rand,
        _gpu_idx: i32,
        _thread_idx: i32,
    ) {
        loop {
            let mut request = dummy_request();
            if !self.query_queue.wait_pop(&mut request) {
                break;
            }

            // Collect a small batch without blocking.
            let mut batch = vec![request];
            let target_batch_size = self.current_batch_size.load(Ordering::Relaxed) as usize;
            while batch.len() < target_batch_size.max(1) {
                let mut extra = dummy_request();
                if !self.query_queue.try_pop(&mut extra) {
                    break;
                }
                batch.push(extra);
            }

            self.process_batch(&batch, handle);
            self.num_batches_processed.fetch_add(1, Ordering::Relaxed);
            self.num_rows_processed
                .fetch_add(batch.len() as u64, Ordering::Relaxed);
            self.waiting_for_finish.notify_all();
        }
    }

    /// Compute and store results for a batch of requests.
    fn process_batch(&self, batch: &[Arc<EvalRequest>], handle: &Mutex<Box<dyn ComputeHandle>>) {
        let backend_guard = self.backend.lock().unwrap();
        let handle_guard = handle.lock().unwrap();

        if let Some(backend) = backend_guard.as_deref() {
            // Real backend path
            let n = batch.len();
            let mut outputs: Vec<NNOutput> = (0..n).map(|_| NNOutput::default()).collect();

            {
                let mut bufs: Vec<parking_lot::MutexGuard<'_, NNResultBuf>> =
                    batch.iter().map(|r| r.buf.lock()).collect();
                let mut input_refs: Vec<&mut NNResultBuf> =
                    bufs.iter_mut().map(|g| &mut **g).collect();
                let mut output_refs: Vec<&mut NNOutput> = outputs.iter_mut().collect();

                let buffers_guard = self.input_buffers.lock().unwrap();
                let dummy_buffers;
                let input_buffers: &dyn InputBuffers = match buffers_guard.as_deref() {
                    Some(b) => b,
                    None => {
                        dummy_buffers = DummyInputBuffers;
                        &dummy_buffers
                    }
                };
                backend
                    .get_output(
                        handle_guard.as_ref(),
                        input_buffers,
                        n as i32,
                        &mut input_refs,
                        &mut output_refs,
                    )
                    .unwrap_or_else(|e| {
                        eprintln!("WARNING: Backend get_output failed: {e}");
                    });
            }

            for (i, mut output) in outputs.into_iter().enumerate() {
                // Turn raw backend logits into probabilities / points,
                // mirroring the C++ NNEvaluator postprocessing.
                self.postprocess_output(
                    &batch[i].board,
                    &batch[i].history,
                    batch[i].next_player,
                    &batch[i].nn_input_params,
                    &mut output,
                );
                let output = Arc::new(output);
                if let Some(cache) = &self.nn_cache_table {
                    cache.set(&output);
                }
                let mut buf = batch[i].buf.lock();
                buf.result = Some(output);
                buf.has_result = true;
                drop(buf);
                batch[i].result_ready.notify_one();
            }
        } else {
            // No backend — fall back to dummy uniform policy.
            for request in batch {
                let output = compute_output(
                    self.nn_x_len,
                    self.nn_y_len,
                    self.policy_size,
                    &request.board,
                    &request.history,
                    request.next_player,
                    &request.nn_input_params,
                    request.nn_hash,
                    request.include_owner_map,
                );
                let output = Arc::new(output);
                if let Some(cache) = &self.nn_cache_table {
                    cache.set(&output);
                }
                let mut buf = request.buf.lock();
                buf.result = Some(output);
                buf.has_result = true;
                drop(buf);
                request.result_ready.notify_one();
            }
        }
    }

    /// Postprocess a raw backend output into probabilities / points.
    ///
    /// Mirrors the postprocessing in C++ `nneval.cpp` (`NNEvaluator::evaluate`):
    /// policy logits are softmaxed over legal moves (with policy temperature
    /// and optional passing hacks), value logits are softmaxed and flipped to
    /// white's perspective, score outputs get their multipliers / softplus,
    /// and the ownership map is tanh-ed and flipped.
    fn postprocess_output(
        &self,
        board: &Board,
        history: &BoardHistory,
        next_player: Player,
        nn_input_params: &MiscNNInputParams,
        output: &mut NNOutput,
    ) {
        let (model_version, pp) = *self.postprocess_cfg.lock().unwrap();
        let policy_size = self.policy_size as usize;
        let x_size = board.x_size;
        let y_size = board.y_size;

        // --- Policy ---------------------------------------------------------
        let policy_output_scaling = pp.output_scale_multiplier
            / nn_input_params.nn_policy_temperature.clamp(1e-6, 1e6);

        let mut is_legal = vec![false; policy_size];
        let mut legal_count = 0usize;
        for i in 0..policy_size {
            let loc = nn_pos::pos_to_loc(
                i as i32,
                x_size,
                y_size,
                self.nn_x_len,
                self.nn_y_len,
            );
            is_legal[i] = history.is_legal(board, loc, next_player);
        }
        // TODO(selfplay): the C++ avoidMYTDaggerHack dagger-match ban is not
        // ported here; it only affects selfplay training, not GTP play.

        let mut max_policy = -1e25f32;
        for i in 0..policy_size {
            let v = if is_legal[i] {
                legal_count += 1;
                output.policy_probs[i] * policy_output_scaling
            } else {
                -1e30f32
            };
            output.policy_probs[i] = v;
            if v > max_policy {
                max_policy = v;
            }
        }

        let mut policy_sum = 0.0f32;
        if nn_input_params.enable_passing_hacks {
            // Cap passing prior policy at 95% (19x other moves).
            let max_pass_policy_sum_factor = 19.0f32;
            for i in 0..policy_size - 1 {
                let v = (output.policy_probs[i] - max_policy).exp();
                output.policy_probs[i] = v;
                policy_sum += v;
            }
            let i = policy_size - 1;
            let v = (output.policy_probs[i] - max_policy)
                .exp()
                .clamp(1e-20, policy_sum * max_pass_policy_sum_factor);
            output.policy_probs[i] = v;
            policy_sum += v;
        } else {
            for v in output.policy_probs.iter_mut().take(policy_size) {
                *v = (*v - max_policy).exp();
                policy_sum += *v;
            }
        }

        if policy_sum <= 0.0 {
            // Somehow all legal moves rounded to 0 probability.
            let uniform = 1.0f32 / legal_count.max(1) as f32;
            for i in 0..policy_size {
                output.policy_probs[i] = if is_legal[i] { uniform } else { -1.0 };
            }
        } else {
            for i in 0..policy_size {
                output.policy_probs[i] = if is_legal[i] {
                    output.policy_probs[i] / policy_sum
                } else {
                    -1.0
                };
            }
        }
        for v in output.policy_probs.iter_mut().skip(policy_size) {
            *v = -1.0f32;
        }
        output.policy_optimism_used = nn_input_params.policy_optimism as f32;

        // --- Value / score (model version >= 4) ------------------------------
        if model_version >= 4 {
            let win_logits = output.white_win_prob as f64 * pp.output_scale_multiplier as f64;
            let loss_logits = output.white_loss_prob as f64 * pp.output_scale_multiplier as f64;
            let mut no_result_logits =
                output.white_no_result_prob as f64 * pp.output_scale_multiplier as f64;
            let score_mean_pre = output.white_score_mean as f64 * pp.output_scale_multiplier as f64;
            let score_stdev_pre =
                output.white_score_mean_sq as f64 * pp.output_scale_multiplier as f64;
            let lead_pre = output.white_lead as f64 * pp.output_scale_multiplier as f64;
            let var_time_pre = output.var_time_left as f64 * pp.output_scale_multiplier as f64;
            let swin_pre =
                output.shortterm_winloss_error as f64 * pp.output_scale_multiplier as f64;
            let sscore_pre =
                output.shortterm_score_error as f64 * pp.output_scale_multiplier as f64;

            if history.rules.ko_rule != KoRule::Simple
                && history.rules.scoring_rule != ScoringRule::Territory
            {
                no_result_logits -= 100000.0;
            }

            let max_logits = win_logits.max(loss_logits).max(no_result_logits);
            let mut win_prob = (win_logits - max_logits).exp();
            let mut loss_prob = (loss_logits - max_logits).exp();
            let mut no_result_prob = (no_result_logits - max_logits).exp();
            if history.rules.ko_rule != KoRule::Simple
                && history.rules.scoring_rule != ScoringRule::Territory
            {
                no_result_prob = 0.0;
            }
            let prob_sum = win_prob + loss_prob + no_result_prob;
            win_prob /= prob_sum;
            loss_prob /= prob_sum;
            no_result_prob /= prob_sum;

            let mut score_mean = score_mean_pre * pp.score_mean_multiplier;
            let score_stdev = softplus(score_stdev_pre) * pp.score_stdev_multiplier;
            let mut score_mean_sq = score_mean * score_mean + score_stdev * score_stdev;
            let mut lead = lead_pre * pp.lead_multiplier;
            let var_time_left = softplus(var_time_pre) * pp.variance_time_multiplier;
            // No-result counts as 0 score for score-value purposes.
            score_mean *= 1.0 - no_result_prob;
            score_mean_sq *= 1.0 - no_result_prob;
            lead *= 1.0 - no_result_prob;

            let (shortterm_winloss_error, shortterm_score_error) = if model_version >= 14 {
                let s1 = softplus(swin_pre * 0.5);
                let s2 = softplus(sscore_pre * 0.5);
                (
                    (s1 * s1 * pp.shortterm_value_error_multiplier).sqrt(),
                    (s2 * s2 * pp.shortterm_score_error_multiplier).sqrt(),
                )
            } else if model_version >= 10 {
                (
                    (softplus(swin_pre) * pp.shortterm_value_error_multiplier).sqrt(),
                    (softplus(sscore_pre) * pp.shortterm_score_error_multiplier).sqrt(),
                )
            } else {
                (softplus(swin_pre), softplus(sscore_pre) * 10.0)
            };

            // Flip from player-to-move to white's perspective.
            if next_player == P_WHITE {
                output.white_win_prob = win_prob as f32;
                output.white_loss_prob = loss_prob as f32;
                output.white_no_result_prob = no_result_prob as f32;
                output.white_score_mean = score_mean as f32;
                output.white_score_mean_sq = score_mean_sq as f32;
                output.white_lead = lead as f32;
            } else {
                output.white_win_prob = loss_prob as f32;
                output.white_loss_prob = win_prob as f32;
                output.white_no_result_prob = no_result_prob as f32;
                output.white_score_mean = -(score_mean as f32);
                output.white_score_mean_sq = score_mean_sq as f32;
                output.white_lead = -(lead as f32);
            }
            if model_version >= 9 {
                output.var_time_left = var_time_left as f32;
                output.shortterm_winloss_error = shortterm_winloss_error as f32;
                output.shortterm_score_error = shortterm_score_error as f32;
            } else {
                output.var_time_left = -1.0;
                output.shortterm_winloss_error = -1.0;
                output.shortterm_score_error = -1.0;
            }
        }

        // --- Ownership ---------------------------------------------------------
        if let Some(map) = &mut output.white_owner_map {
            let s = pp.output_scale_multiplier;
            for pos in 0..(self.nn_x_len * self.nn_y_len) as usize {
                let y = pos as i32 / self.nn_x_len;
                let x = pos as i32 % self.nn_x_len;
                if y >= board.y_size || x >= board.x_size {
                    map[pos] = 0.0f32;
                } else {
                    // Same as value: flip player-to-move → white and tanh.
                    let v = map[pos] * s;
                    map[pos] = if next_player == P_WHITE { v.tanh() } else { -v.tanh() };
                }
            }
        }
    }
}

/// Softplus: `log(1 + exp(x))`, linear for large x (mirrors C++ `softPlus`).
fn softplus(x: f64) -> f64 {
    if x > 40.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// A single pending NN evaluation request passed from a client thread to a
/// server thread.
///
/// The server thread fills `buf` with the computed result and signals
/// `result_ready` to wake the waiting client.
struct EvalRequest {
    buf: AsyncMutex<NNResultBuf>,
    result_ready: Condvar,
    board: Board,
    history: BoardHistory,
    next_player: Player,
    #[allow(dead_code)]
    sgf_meta: Option<SgfMetadata>,
    nn_input_params: MiscNNInputParams,
    include_owner_map: bool,
    nn_hash: Hash128,
}

#[allow(clippy::too_many_arguments)]
impl NnEvaluator {
    /// Create a new evaluator skeleton.
    ///
    /// This does **not** load a model or spawn server threads; it only stores
    /// the configuration and allocates the cache table when requested.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_name: String,
        model_file_name: String,
        expected_sha256: String,
        logger: Arc<Logger>,
        max_batch_size: i32,
        nn_x_len: i32,
        nn_y_len: i32,
        require_exact_nn_len: bool,
        inputs_use_nhwc: bool,
        nn_cache_size_power_of_two: i32,
        nn_mutex_pool_size_power_of_two: i32,
        debug_skip_neural_net: bool,
        _home_data_dir_override: String,
        using_fp16_mode: Enabled,
        num_threads: i32,
        gpu_idx_by_server_thread: Vec<i32>,
        rand_seed: String,
        do_randomize: bool,
        default_symmetry: i32,
        disable_warmup: bool,
        cfg: &ConfigParser,
    ) -> Self {
        assert!(nn_x_len > 0, "nn_x_len must be positive");
        assert!(nn_y_len > 0, "nn_y_len must be positive");
        assert!(max_batch_size > 0, "max_batch_size must be positive");

        let nn_cache_table = if nn_cache_size_power_of_two >= 0 {
            Some(Arc::new(NnCacheTable::new(
                nn_cache_size_power_of_two as u32,
                nn_mutex_pool_size_power_of_two as u32,
            )))
        } else {
            None
        };

        let shared = Arc::new(SharedState {
            query_queue: ThreadSafeQueue::new(),
            nn_cache_table,
            is_killed: AtomicBool::new(false),
            current_batch_size: AtomicI32::new(max_batch_size),
            waiting_for_finish: Condvar::new(),
            num_rows_processed: AtomicU64::new(0),
            num_batches_processed: AtomicU64::new(0),
            nn_x_len,
            nn_y_len,
            policy_size: nn_x_len * nn_y_len + 1,
            backend: Mutex::new(None),
            compute_handles: Mutex::new(Vec::new()),
            input_buffers: Mutex::new(None),
            postprocess_cfg: Mutex::new((0, ModelPostProcessParams::default())),
        });

        Self {
            model_name,
            model_file_name,
            expected_sha256,
            require_exact_nn_len,
            inputs_use_nhwc,
            using_fp16_mode,
            num_threads,
            gpu_idx_by_server_thread,
            rand_seed,
            debug_skip_neural_net,
            disable_warmup,
            cfg: cfg.clone(),
            compute_context: None,
            loaded_model: None,
            compute_handle: None,
            backend: None,
            logger,
            internal_model_name: String::new(),
            model_version: 0,
            inputs_version: 0,
            num_input_meta_channels: 0,
            post_process_params: ModelPostProcessParams::default(),
            num_server_threads_ever_spawned: 0,
            server_threads: Vec::new(),
            max_batch_size,
            buffer_mutex: AsyncMutex::new(()),
            num_server_threads_starting_up: 0,
            main_thread_waiting_for_spawn: Condvar::new(),
            server_threads_is_using_fp16: Vec::new(),
            num_ongoing_evals: 0,
            num_waiting_evals: 0,
            num_evals_to_awaken: 0,
            current_do_randomize: AtomicBool::new(do_randomize),
            current_default_symmetry: AtomicI32::new(default_symmetry),
            shared,
        }
    }

    /// Name of the model.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// File name of the model.
    pub fn model_file_name(&self) -> &str {
        &self.model_file_name
    }

    /// Internal model name reported by the loader.
    pub fn internal_model_name(&self) -> &str {
        &self.internal_model_name
    }

    /// Short form of the internal model name.
    pub fn abbrev_internal_model_name(&self) -> String {
        self.internal_model_name
            .split_whitespace()
            .next()
            .unwrap_or(&self.internal_model_name)
            .to_string()
    }

    /// Logger used by this evaluator.
    pub fn logger(&self) -> Arc<Logger> {
        Arc::clone(&self.logger)
    }

    /// True if the evaluator runs without an actual neural net.
    pub fn is_neural_net_less(&self) -> bool {
        self.debug_skip_neural_net
    }

    /// Maximum batch size.
    pub fn max_batch_size(&self) -> i32 {
        self.max_batch_size
    }

    /// Current batch size, which may be smaller than the maximum.
    pub fn current_batch_size(&self) -> i32 {
        self.shared.current_batch_size.load(Ordering::Relaxed)
    }

    /// Set the current batch size. Must not exceed the maximum.
    pub fn set_current_batch_size(&self, batch_size: i32) {
        assert!(
            batch_size > 0 && batch_size <= self.max_batch_size,
            "batch_size out of range"
        );
        self.shared
            .current_batch_size
            .store(batch_size, Ordering::Relaxed);
    }

    /// True if the model expects SGF metadata input channels.
    pub fn requires_sgf_metadata(&self) -> bool {
        self.num_input_meta_channels > 0
    }

    /// Number of distinct GPUs referenced by server threads.
    pub fn num_gpus(&self) -> i32 {
        self.gpu_idxs().len() as i32
    }

    /// Number of configured server threads.
    pub fn num_server_threads(&self) -> i32 {
        self.gpu_idx_by_server_thread.len() as i32
    }

    /// Set of distinct GPU indices.
    pub fn gpu_idxs(&self) -> BTreeSet<i32> {
        self.gpu_idx_by_server_thread.iter().copied().collect()
    }

    /// Neural-net X length.
    pub fn nn_x_len(&self) -> i32 {
        self.shared.nn_x_len
    }

    /// Neural-net Y length.
    pub fn nn_y_len(&self) -> i32 {
        self.shared.nn_y_len
    }

    /// Whether the input board size must exactly match the NN size.
    pub fn require_exact_nn_len(&self) -> bool {
        self.require_exact_nn_len
    }

    /// Model version reported by the loader.
    pub fn model_version(&self) -> i32 {
        self.model_version
    }

    /// Spatial convolution depth of the trunk.
    pub fn trunk_spatial_conv_depth(&self) -> f64 {
        // TODO: derive from ModelDesc once model loading is ported.
        0.0
    }

    /// FP16 mode setting.
    pub fn using_fp16_mode(&self) -> Enabled {
        self.using_fp16_mode
    }

    /// True if the loaded model exposes shortterm-error fields.
    pub fn supports_shortterm_error(&self) -> bool {
        // TODO: implement once model loading is ported.
        false
    }

    /// Return the nearest ruleset supported by the model.
    ///
    /// In this skeleton the desired rules are returned unchanged and
    /// `supported` is set to `false`.
    pub fn supported_rules(&self, desired_rules: Rules, supported: &mut bool) -> Rules {
        *supported = false;
        desired_rules
    }

    /// Clear the NN output cache.
    pub fn clear_cache(&self) {
        if let Some(cache) = &self.shared.nn_cache_table {
            cache.clear();
        }
    }

    /// Evaluate a position.
    ///
    /// In this slice the evaluation is a minimal placeholder that produces a
    /// uniform policy over legal moves and a neutral value distribution. It is
    /// sufficient to exercise the search loop and AsyncBot end-to-end, while the
    /// real neural-net server threads remain a future slice. Cache lookup and
    /// storage are implemented.
    pub fn evaluate(
        &self,
        board: &Board,
        history: &BoardHistory,
        next_player: Player,
        nn_input_params: &MiscNNInputParams,
        buf: &mut NNResultBuf,
        skip_cache: bool,
        include_owner_map: bool,
    ) {
        self.evaluate_with_sgf_meta(
            board,
            history,
            next_player,
            None,
            nn_input_params,
            buf,
            skip_cache,
            include_owner_map,
        );
    }

    /// Evaluate a position with optional SGF metadata.
    ///
    /// Mirrors the cache-aware evaluation path in
    /// `cpp/neuralnet/nneval.cpp`. The skeleton still uses a uniform policy +
    /// neutral value as the fallback computation.
    pub fn evaluate_with_sgf_meta(
        &self,
        board: &Board,
        history: &BoardHistory,
        next_player: Player,
        sgf_meta: Option<&SgfMetadata>,
        nn_input_params: &MiscNNInputParams,
        buf: &mut NNResultBuf,
        skip_cache: bool,
        include_owner_map: bool,
    ) {
        buf.has_result = false;

        let nn_x_len = self.shared.nn_x_len;
        let nn_y_len = self.shared.nn_y_len;
        if board.x_size > nn_x_len || board.y_size > nn_y_len {
            panic!(
                "NNEvaluator was configured with nnXLen = {} nnYLen = {} but was asked to evaluate board with larger x or y size",
                nn_x_len, nn_y_len
            );
        }
        if self.require_exact_nn_len && (board.x_size != nn_x_len || board.y_size != nn_y_len) {
            panic!(
                "NNEvaluator was configured with nnXLen = {} nnYLen = {} and requireExactNNLen, but was asked to evaluate board with different x or y size",
                nn_x_len, nn_y_len
            );
        }

        let mut nn_input_params = *nn_input_params;
        if self.num_input_meta_channels > 0 {
            nn_input_params.policy_optimism = 0.0;
        }

        let mut nn_hash = get_hash(board, history, next_player, &nn_input_params);
        if self.num_input_meta_channels > 0 {
            let sgf_meta = sgf_meta.expect("SGFMetadata is required but was not provided");
            assert!(
                sgf_meta.initialized,
                "SGFMetadata is required but was not initialized"
            );
            nn_hash ^= sgf_meta.get_hash(next_player);
        }

        if let Some(cache) = &self.shared.nn_cache_table {
            if !skip_cache {
                if let Some(cached) = cache.get(nn_hash) {
                    if !(include_owner_map && cached.white_owner_map.is_none()) {
                        buf.result = Some(cached);
                        buf.has_result = true;
                        return;
                    }
                }
            }
        }

        if self.server_threads.is_empty() {
            let output = compute_output(
                nn_x_len,
                nn_y_len,
                self.shared.policy_size,
                board,
                history,
                next_player,
                &nn_input_params,
                nn_hash,
                include_owner_map,
            );
            self.store_output(output, buf);
        } else {
            self.evaluate_async(
                board,
                history,
                next_player,
                sgf_meta,
                &nn_input_params,
                nn_hash,
                buf,
                include_owner_map,
            );
        }
    }

    /// Store a freshly-computed output in the cache and in `buf`.
    fn store_output(&self, output: NNOutput, buf: &mut NNResultBuf) {
        let output = Arc::new(output);
        if let Some(cache) = &self.shared.nn_cache_table {
            cache.set(&output);
        }
        buf.result = Some(output);
        buf.has_result = true;
    }

    /// Fill a result buffer with NN input features from a position.
    ///
    /// Resizes the row buffers to the feature-count requirements for the
    /// model version before filling (mirrors the C++ `fillRowV*` callers).
    fn fill_nn_input(
        board: &Board,
        history: &BoardHistory,
        next_player: Player,
        nn_input_params: &MiscNNInputParams,
        nn_x_len: i32,
        nn_y_len: i32,
        use_nhwc: bool,
        inputs_version: i32,
        model_version: i32,
        buf: &mut NNResultBuf,
    ) {
        let num_spatial =
            get_num_spatial_features(model_version).unwrap_or(22).max(0) as usize;
        let num_global = get_num_global_features(model_version).unwrap_or(19).max(0) as usize;
        buf.row_spatial_buf
            .resize(num_spatial * nn_x_len as usize * nn_y_len as usize, 0.0);
        buf.row_global_buf.resize(num_global, 0.0);

        let (row_spatial, row_global) = (&mut buf.row_spatial_buf, &mut buf.row_global_buf);
        match inputs_version {
            3 => crate::inputs::fill_row_v3(
                board,
                history,
                next_player,
                nn_input_params,
                nn_x_len,
                nn_y_len,
                use_nhwc,
                row_spatial,
                row_global,
            ),
            4 => crate::inputs::fill_row_v4(
                board,
                history,
                next_player,
                nn_input_params,
                nn_x_len,
                nn_y_len,
                use_nhwc,
                row_spatial,
                row_global,
            ),
            5 => crate::inputs::fill_row_v5(
                board,
                history,
                next_player,
                nn_input_params,
                nn_x_len,
                nn_y_len,
                use_nhwc,
                row_spatial,
                row_global,
            ),
            6 => crate::inputs::fill_row_v6(
                board,
                history,
                next_player,
                nn_input_params,
                nn_x_len,
                nn_y_len,
                use_nhwc,
                row_spatial,
                row_global,
            ),
            _ => crate::inputs::fill_row_v7(
                board,
                history,
                next_player,
                nn_input_params,
                nn_x_len,
                nn_y_len,
                use_nhwc,
                row_spatial,
                row_global,
            ),
        }
    }

    /// Enqueue a request for the server threads and block until it is processed.
    fn evaluate_async(
        &self,
        board: &Board,
        history: &BoardHistory,
        next_player: Player,
        sgf_meta: Option<&SgfMetadata>,
        nn_input_params: &MiscNNInputParams,
        nn_hash: Hash128,
        buf: &mut NNResultBuf,
        include_owner_map: bool,
    ) {
        // Fill input feature buffers before queueing so the server thread
        // only needs to read them (matching C++ semantics).
        let mut filled_buf = NNResultBuf::new();
        Self::fill_nn_input(
            board,
            history,
            next_player,
            nn_input_params,
            self.shared.nn_x_len,
            self.shared.nn_y_len,
            self.inputs_use_nhwc,
            self.inputs_version,
            self.model_version,
            &mut filled_buf,
        );

        let request = Arc::new(EvalRequest {
            buf: AsyncMutex::new(filled_buf),
            result_ready: Condvar::new(),
            board: board.clone(),
            history: history.clone(),
            next_player,
            sgf_meta: sgf_meta.cloned(),
            nn_input_params: *nn_input_params,
            include_owner_map,
            nn_hash,
        });

        {
            let mut req_buf = request.buf.lock();
            req_buf.board_x_size_for_server = board.x_size;
            req_buf.board_y_size_for_server = board.y_size;
        }

        self.shared.query_queue.wait_push(request.clone());

        let mut req_buf = request.buf.lock();
        req_buf.client_waiting_for_result = true;
        let condvar = &request.result_ready;
        // Wait until the result is actually stored. The backend's get_output
        // is allowed to flip `has_result` on the request buffers while the
        // result Arc is still pending, so keying the wait on `has_result`
        // alone could let the client leave early with `result == None`.
        while req_buf.result.is_none() && !self.shared.is_killed.load(Ordering::Relaxed) {
            condvar.wait(&mut req_buf);
        }

        if req_buf.result.is_some() {
            buf.result = req_buf.result.clone();
            buf.has_result = true;
        } else {
            // The evaluator was killed while we were waiting. Fall back to a
            // synchronous compute so the caller still gets a valid result.
            drop(req_buf);
            let output = compute_output(
                self.shared.nn_x_len,
                self.shared.nn_y_len,
                self.shared.policy_size,
                board,
                history,
                next_player,
                nn_input_params,
                nn_hash,
                include_owner_map,
            );
            self.store_output(output, buf);
        }
    }

    /// Evaluate a position under multiple symmetries and return the individual
    /// outputs so that callers can average them.
    ///
    /// Mirrors `NNEvaluator::averageMultipleSymmetries` in
    /// `cpp/neuralnet/nneval.cpp`. In this slice the evaluator is not backed by
    /// real server threads, so each symmetry is evaluated directly via
    /// [`Self::evaluate_with_sgf_meta`].
    pub fn average_multiple_symmetries(
        &self,
        board: &Board,
        history: &BoardHistory,
        next_player: Player,
        sgf_meta: Option<&SgfMetadata>,
        base_nn_input_params: &MiscNNInputParams,
        buf: &mut NNResultBuf,
        include_owner_map: bool,
        rand: &mut Rand,
        num_symmetries_to_sample: i32,
    ) -> Vec<Arc<NNOutput>> {
        let num_symmetries_to_sample = num_symmetries_to_sample.clamp(0, NUM_SYMMETRIES) as usize;
        let mut symmetry_indexes: Vec<i32> = (0..NUM_SYMMETRIES).collect();
        let mut outputs = Vec::with_capacity(num_symmetries_to_sample);

        for i in 0..num_symmetries_to_sample {
            let j = rand.next_i32_range(i as i32, NUM_SYMMETRIES - 1) as usize;
            symmetry_indexes.swap(i, j);
            let symmetry = symmetry_indexes[i];

            let mut nn_input_params = *base_nn_input_params;
            nn_input_params.symmetry = symmetry;
            self.evaluate_with_sgf_meta(
                board,
                history,
                next_player,
                sgf_meta,
                &nn_input_params,
                buf,
                true, // skip cache
                include_owner_map,
            );
            if let Some(output) = buf.result.take() {
                outputs.push(output);
            }
        }
        outputs
    }

    /// Wait until at least one in-flight evaluation completes.
    ///
    /// If server threads are running and the queue is non-empty, blocks until
    /// the queue size drops (i.e. at least one pending request was processed).
    pub fn wait_for_next_nn_eval_if_any(&self) {
        if self.server_threads.is_empty() {
            return;
        }
        let initial_size = self.shared.query_queue.size();
        if initial_size == 0 {
            return;
        }
        let mut lock = self.buffer_mutex.lock();
        while self.shared.query_queue.size() >= initial_size
            && !self.shared.is_killed.load(Ordering::Relaxed)
        {
            self.shared.waiting_for_finish.wait(&mut lock);
        }
    }

    /// Set the backend used to load models and perform inference.
    pub fn set_backend(&mut self, backend: Arc<dyn Backend>) {
        self.backend = Some(backend);
    }

    /// Load the configured model file using the currently set backend.
    ///
    /// Mirrors part of the model-loading path in
    /// `cpp/program/setup.cpp`. The model is loaded synchronously and the
    /// per-server-thread compute handles and input buffers are prepared in
    /// `SharedState` so that `spawn_server_threads` can pick them up.
    pub fn load_model(&mut self) -> Result<(), StringError> {
        if self.debug_skip_neural_net {
            return Ok(());
        }
        let backend = self
            .backend
            .as_ref()
            .ok_or_else(|| StringError::new("No backend set for NnEvaluator".to_string()))?;
        let model = backend
            .load_model_file(&self.model_file_name, &self.expected_sha256)
            .map_err(|e| StringError::new(format!("Could not load model file: {e}")))?;
        self.internal_model_name = model.model_desc().get_short_info_string();
        self.model_version = model.model_desc().model_version;
        self.inputs_version = get_inputs_version(self.model_version)
            .map_err(|e| StringError::new(format!("Could not determine inputs version: {e}")))?;
        self.num_input_meta_channels = model.model_desc().num_input_meta_channels;
        self.post_process_params = model.model_desc().post_process_params;
        // Mirror into SharedState so server threads can postprocess raw
        // backend outputs. Runs before spawn_server_threads, so a plain
        // mutex-guarded pair is fine.
        *self.shared.postprocess_cfg.lock().unwrap() =
            (self.model_version, self.post_process_params);

        let cfg = &self.cfg;
        let compute_context = backend
            .create_compute_context(
                &self.gpu_idx_by_server_thread,
                &self.logger,
                self.shared.nn_x_len,
                self.shared.nn_y_len,
                "",
                self.using_fp16_mode,
                &*model,
                cfg,
            )
            .map_err(|e| StringError::new(format!("Could not create compute context: {e}")))?;
        let compute_handle = backend
            .create_compute_handle(
                &*compute_context,
                &*model,
                &self.logger,
                self.max_batch_size,
                self.require_exact_nn_len,
                self.inputs_use_nhwc,
                self.gpu_idx_by_server_thread.first().copied().unwrap_or(-1),
                0,
            )
            .map_err(|e| StringError::new(format!("Could not create compute handle: {e}")))?;
        let input_buffers = backend
            .create_input_buffers(
                &*model,
                self.max_batch_size,
                self.shared.nn_x_len,
                self.shared.nn_y_len,
            )
            .map_err(|e| StringError::new(format!("Could not create input buffers: {e}")))?;

        self.compute_context = Some(compute_context);
        self.compute_handle = Some(compute_handle);
        // Store the input buffers in SharedState so server threads can use
        // them when processing batches.
        *self.shared.input_buffers.lock().unwrap() = Some(input_buffers);
        self.loaded_model = Some(model);

        // Also store a clone of the backend and one compute handle per
        // server thread in SharedState so that each server thread gets its
        // own dedicated handle.
        {
            let mut b = self.shared.backend.lock().unwrap();
            *b = Some(Arc::clone(backend));
        }
        {
            let mut handles: Vec<Box<dyn ComputeHandle>> = Vec::new();
            let num_threads = self.gpu_idx_by_server_thread.len().max(1);
            for thread_idx in 0..num_threads {
                let gpu_idx = self
                    .gpu_idx_by_server_thread
                    .get(thread_idx)
                    .copied()
                    .unwrap_or(-1);
                let h = backend
                    .create_compute_handle(
                        self.compute_context.as_deref().unwrap(),
                        self.loaded_model.as_deref().unwrap(),
                        &self.logger,
                        self.max_batch_size,
                        self.require_exact_nn_len,
                        self.inputs_use_nhwc,
                        gpu_idx,
                        thread_idx as i32,
                    )
                    .map_err(|e| {
                        StringError::new(format!(
                            "Could not create server compute handle for thread {thread_idx}: {e}"
                        ))
                    })?;
                handles.push(h);
            }
            *self.shared.compute_handles.lock().unwrap() = handles;
        }

        Ok(())
    }

    /// Spawn server threads.
    ///
    /// For this slice a single server thread is spawned regardless of the
    /// configured number of threads. It reads pending requests from
    /// `query_queue`, evaluates them, and signals waiting clients.
    pub fn spawn_server_threads(&mut self) {
        if self.shared.is_killed.load(Ordering::Relaxed) || !self.server_threads.is_empty() {
            return;
        }

        let handles: Vec<Box<dyn ComputeHandle>> =
            std::mem::take(&mut *self.shared.compute_handles.lock().unwrap());
        if handles.is_empty() {
            // No backend loaded — no GPU threads to spawn. The evaluator
            // will use the synchronous compute_output fallback.
            return;
        }

        self.num_server_threads_ever_spawned += 1;
        for (thread_idx, handle) in handles.into_iter().enumerate() {
            let shared = Arc::clone(&self.shared);
            let handle = Arc::new(Mutex::new(handle));
            let gpu_idx = self
                .gpu_idx_by_server_thread
                .get(thread_idx)
                .copied()
                .unwrap_or(-1);
            let h = std::thread::spawn(move || {
                let mut server_buf = NnServerBuf::new_empty();
                let mut rand = Rand::new_from_seed("server-thread");
                shared.serve(
                    &handle,
                    &mut server_buf,
                    &mut rand,
                    gpu_idx,
                    thread_idx as i32,
                );
            });
            self.server_threads.push(h);
        }
    }

    /// Kill server threads and wait for them to finish.
    pub fn kill_server_threads(&mut self) {
        self.shared.is_killed.store(true, Ordering::Relaxed);
        self.shared.query_queue.close();
        self.shared.waiting_for_finish.notify_all();

        for handle in self.server_threads.drain(..) {
            let _ = handle.join();
        }
    }

    /// Set the GPU indices used by server threads.
    ///
    /// Currently a no-op; the skeleton evaluator ignores thread configuration.
    pub fn set_num_threads(&mut self, _gpu_idx_by_server_thread: &[i32]) {
        // No-op until server-thread spawning is implemented.
    }

    /// Check whether any spawned thread is using FP16.
    ///
    /// Always returns `false` in this slice.
    pub fn is_any_thread_using_fp16(&self) -> bool {
        false
    }

    /// Whether symmetry randomization is enabled.
    pub fn do_randomize(&self) -> bool {
        self.current_do_randomize.load(Ordering::Relaxed)
    }

    /// Default symmetry index.
    pub fn default_symmetry(&self) -> i32 {
        self.current_default_symmetry.load(Ordering::Relaxed)
    }

    /// Enable or disable symmetry randomization.
    pub fn set_do_randomize(&self, value: bool) {
        self.current_do_randomize.store(value, Ordering::Relaxed);
    }

    /// Set the default symmetry index.
    pub fn set_default_symmetry(&self, symmetry: i32) {
        self.current_default_symmetry
            .store(symmetry, Ordering::Relaxed);
    }

    /// Number of rows processed since construction or the last stats clear.
    pub fn num_rows_processed(&self) -> u64 {
        self.shared.num_rows_processed.load(Ordering::Relaxed)
    }

    /// Number of batches processed since construction or the last stats clear.
    pub fn num_batches_processed(&self) -> u64 {
        self.shared.num_batches_processed.load(Ordering::Relaxed)
    }

    /// Average number of rows per processed batch.
    pub fn average_processed_batch_size(&self) -> f64 {
        let batches = self.num_batches_processed();
        if batches == 0 {
            0.0
        } else {
            self.num_rows_processed() as f64 / batches as f64
        }
    }

    /// Reset row/batch counters.
    pub fn clear_stats(&self) {
        self.shared.num_rows_processed.store(0, Ordering::Relaxed);
        self.shared
            .num_batches_processed
            .store(0, Ordering::Relaxed);
    }

    /// Server-thread entry point.
    ///
    /// Delegates to the shared state so that callers can run the server loop
    /// synchronously for testing.
    pub fn serve(
        &self,
        server_buf: &mut NnServerBuf,
        rand: &mut Rand,
        gpu_idx: i32,
        thread_idx: i32,
    ) {
        use crate::backend::dummy::DummyComputeHandle;
        // For synchronous testing — use a dummy handle. The queue is
        // typically already closed in test scenarios, so serve exits
        // immediately without ever accessing the handle.
        let dummy_handle = Box::new(DummyComputeHandle::new()) as Box<dyn ComputeHandle>;
        let handle = Mutex::new(dummy_handle);
        self.shared
            .serve(&handle, server_buf, rand, gpu_idx, thread_idx);
    }
}

/// Compute a dummy uniform-policy output for a position.
///
/// This is a placeholder for real neural-net inference. It produces a uniform
/// distribution over legal moves and a neutral value distribution.
#[allow(clippy::too_many_arguments)]
fn compute_output(
    nn_x_len: i32,
    nn_y_len: i32,
    policy_size: i32,
    board: &Board,
    history: &BoardHistory,
    next_player: Player,
    nn_input_params: &MiscNNInputParams,
    nn_hash: Hash128,
    include_owner_map: bool,
) -> NNOutput {
    let _ = policy_size;
    let pass_pos = nn_pos::loc_to_pos(PASS_LOC, board.x_size, nn_x_len, nn_y_len);

    let mut policy_probs = [-1.0f32; nn_pos::MAX_NN_POLICY_SIZE];
    let mut legal_positions = Vec::new();
    for y in 0..board.y_size {
        for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size);
            if history.is_legal(board, loc, next_player) {
                let pos = nn_pos::loc_to_pos(loc, board.x_size, nn_x_len, nn_y_len);
                legal_positions.push(pos);
            }
        }
    }
    if history.is_legal(board, PASS_LOC, next_player) {
        legal_positions.push(pass_pos);
    }

    if !legal_positions.is_empty() {
        let uniform_prob = 1.0f32 / legal_positions.len() as f32;
        for pos in legal_positions {
            policy_probs[pos as usize] = uniform_prob;
        }
    }

    let white_owner_map = if include_owner_map {
        let area = (nn_x_len * nn_y_len) as usize;
        Some(vec![0.0f32; area].into_boxed_slice())
    } else {
        None
    };

    NNOutput {
        nn_hash,
        nn_x_len,
        nn_y_len,
        white_win_prob: 0.5,
        white_loss_prob: 0.5,
        white_no_result_prob: 0.0,
        white_score_mean: 0.0,
        white_score_mean_sq: 0.0,
        white_lead: 0.0,
        var_time_left: 0.0,
        shortterm_winloss_error: 0.0,
        shortterm_score_error: 0.0,
        policy_probs,
        policy_optimism_used: nn_input_params.policy_optimism as f32,
        white_owner_map,
        noised_policy_probs: None,
    }
}

/// Create a placeholder request used to receive values from `ThreadSafeQueue`.
fn dummy_request() -> Arc<EvalRequest> {
    Arc::new(EvalRequest {
        buf: AsyncMutex::new(NNResultBuf::new()),
        result_ready: Condvar::new(),
        board: Board::new(1, 1),
        history: BoardHistory::new(Board::new(1, 1), P_BLACK, Rules::default(), 0),
        next_player: P_BLACK,
        sgf_meta: None,
        nn_input_params: MiscNNInputParams::default(),
        include_owner_map: false,
        nn_hash: Hash128::new(0, 0),
    })
}

impl Drop for NnEvaluator {
    fn drop(&mut self) {
        self.kill_server_threads();
    }
}

/// Per-server-thread buffers.
///
/// Mirrors C++ `NNServerBuf`. The actual input buffer type is backend-specific.
pub struct NnServerBuf {
    pub input_buffers: Option<Box<dyn crate::backend::InputBuffers>>,
}

impl NnServerBuf {
    /// Create a new server buffer. In this skeleton no input buffers are
    /// allocated yet.
    pub fn new(_nneval: &NnEvaluator, _model: Option<&dyn LoadedModel>) -> Self {
        Self::new_empty()
    }

    /// Create a new empty server buffer for use by a server thread.
    pub fn new_empty() -> Self {
        Self {
            input_buffers: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use kata_core::logger::LoggerOptions;

    fn output_with_hash(hash: Hash128) -> Arc<NNOutput> {
        Arc::new(NNOutput {
            nn_hash: hash,
            ..NNOutput::default()
        })
    }

    fn test_logger() -> Arc<Logger> {
        Arc::new(Logger::new(LoggerOptions::default(), None))
    }

    fn empty_cfg() -> ConfigParser {
        ConfigParser::new(false, false)
    }

    #[test]
    fn test_nn_evaluator_new_with_cache() {
        let eval = NnEvaluator::new(
            "test-model".to_string(),
            "test-model.bin".to_string(),
            String::new(),
            test_logger(),
            8,
            19,
            19,
            false,
            false,
            12,
            4,
            false,
            String::new(),
            Enabled::Auto,
            1,
            vec![0],
            "test-seed".to_string(),
            false,
            0,
            false,
            &empty_cfg(),
        );
        assert_eq!(eval.model_name(), "test-model");
        assert_eq!(eval.model_file_name(), "test-model.bin");
        assert_eq!(eval.max_batch_size(), 8);
        assert_eq!(eval.current_batch_size(), 8);
        assert_eq!(eval.nn_x_len(), 19);
        assert_eq!(eval.nn_y_len(), 19);
        assert!(!eval.is_neural_net_less());
        assert_eq!(eval.num_gpus(), 1);
        assert!(eval.gpu_idxs().contains(&0));
        assert_eq!(eval.num_server_threads(), 1);
        assert!(!eval.requires_sgf_metadata());

        eval.clear_cache();
        eval.set_current_batch_size(4);
        assert_eq!(eval.current_batch_size(), 4);

        eval.set_do_randomize(true);
        assert!(eval.do_randomize());
        eval.set_default_symmetry(3);
        assert_eq!(eval.default_symmetry(), 3);

        eval.clear_stats();
        assert_eq!(eval.num_rows_processed(), 0);
        assert_eq!(eval.num_batches_processed(), 0);
        assert_eq!(eval.average_processed_batch_size(), 0.0);
    }

    #[test]
    fn test_nn_evaluator_without_cache() {
        let eval = NnEvaluator::new(
            "m".to_string(),
            "m.bin".to_string(),
            String::new(),
            test_logger(),
            1,
            13,
            13,
            true,
            true,
            -1,
            0,
            true,
            String::new(),
            Enabled::False,
            0,
            Vec::new(),
            String::new(),
            false,
            0,
            true,
            &empty_cfg(),
        );
        assert!(eval.is_neural_net_less());
        assert!(eval.require_exact_nn_len());
        assert_eq!(eval.using_fp16_mode(), Enabled::False);
        assert_eq!(eval.num_gpus(), 0);
        assert_eq!(eval.num_server_threads(), 0);
        // Calling clear_cache without a cache should be a no-op.
        eval.clear_cache();
    }

    #[test]
    fn test_cache_set_and_get() {
        let cache = NnCacheTable::new(8, 4);
        let hash = Hash128::new(12345, 67890);
        let output = output_with_hash(hash);
        assert!(cache.get(hash).is_none());
        cache.set(&output);
        let got = cache.get(hash).unwrap();
        assert_eq!(got.nn_hash, hash);
    }

    #[test]
    fn test_cache_collision_checks_full_hash() {
        let cache = NnCacheTable::new(4, 2);
        // Two hashes that collide in the low bits used for indexing.
        let hash0 = Hash128::new(0, 0);
        let hash1 = Hash128::new(16, 0);
        let out0 = output_with_hash(hash0);
        let out1 = output_with_hash(hash1);
        cache.set(&out0);
        cache.set(&out1);
        // The table stores only one entry per bucket, so out1 overwrites out0.
        // The full-hash check must prevent returning out1 for hash0.
        assert!(cache.get(hash0).is_none());
        assert_eq!(cache.get(hash1).unwrap().nn_hash, hash1);

        // Re-store out0 and verify it replaces out1 in the same bucket.
        cache.set(&out0);
        assert!(cache.get(hash1).is_none());
        assert_eq!(cache.get(hash0).unwrap().nn_hash, hash0);
    }

    #[test]
    fn test_cache_clear() {
        let cache = NnCacheTable::new(8, 4);
        let hash = Hash128::new(1, 2);
        let output = output_with_hash(hash);
        cache.set(&output);
        assert!(cache.get(hash).is_some());
        cache.clear();
        assert!(cache.get(hash).is_none());
    }

    #[test]
    fn test_evaluate_produces_uniform_policy() {
        let eval = NnEvaluator::new(
            "m".to_string(),
            "m.bin".to_string(),
            String::new(),
            test_logger(),
            1,
            9,
            9,
            false,
            false,
            -1,
            0,
            true,
            String::new(),
            Enabled::False,
            0,
            Vec::new(),
            String::new(),
            false,
            0,
            true,
            &empty_cfg(),
        );
        let board = Board::new(9, 9);
        let hist = BoardHistory::new(
            board.clone(),
            kata_game::board::P_BLACK,
            Rules::default(),
            0,
        );
        let mut buf = NNResultBuf::new();
        eval.evaluate(
            &board,
            &hist,
            kata_game::board::P_BLACK,
            &MiscNNInputParams::default(),
            &mut buf,
            false,
            false,
        );
        let output = buf.result.expect("evaluate should produce a result");
        assert_eq!(output.nn_x_len, 9);
        assert_eq!(output.nn_y_len, 9);
        assert!((output.white_win_prob - 0.5).abs() < 1e-6);
        assert!((output.white_loss_prob - 0.5).abs() < 1e-6);

        let pass_idx = (9 * 9) as usize;
        assert!(output.policy_probs[pass_idx] > 0.0);

        let positive: Vec<f32> = output
            .policy_probs
            .iter()
            .copied()
            .filter(|&p| p > 0.0)
            .collect();
        assert!(!positive.is_empty());
        let first = positive[0];
        for p in &positive {
            assert!((p - first).abs() < 1e-6);
        }
    }

    #[test]
    fn test_average_multiple_symmetries_returns_requested_count() {
        let eval = NnEvaluator::new(
            "m".to_string(),
            "m.bin".to_string(),
            String::new(),
            test_logger(),
            1,
            9,
            9,
            false,
            false,
            -1,
            0,
            true,
            String::new(),
            Enabled::False,
            0,
            Vec::new(),
            String::new(),
            false,
            0,
            true,
            &empty_cfg(),
        );
        let board = Board::new(9, 9);
        let hist = BoardHistory::new(
            board.clone(),
            kata_game::board::P_BLACK,
            Rules::default(),
            0,
        );
        let mut buf = NNResultBuf::new();
        let mut rand = Rand::new_from_seed("sym-test");
        let outputs = eval.average_multiple_symmetries(
            &board,
            &hist,
            kata_game::board::P_BLACK,
            None,
            &MiscNNInputParams::default(),
            &mut buf,
            false,
            &mut rand,
            4,
        );
        assert_eq!(outputs.len(), 4);
    }

    #[test]
    fn test_stub_methods_do_not_panic() {
        let mut eval = NnEvaluator::new(
            "m".to_string(),
            "m.bin".to_string(),
            String::new(),
            test_logger(),
            1,
            9,
            9,
            false,
            false,
            -1,
            0,
            true,
            String::new(),
            Enabled::False,
            0,
            Vec::new(),
            String::new(),
            false,
            0,
            true,
            &empty_cfg(),
        );
        eval.wait_for_next_nn_eval_if_any();
        eval.spawn_server_threads();
        eval.kill_server_threads();
        eval.set_num_threads(&[0, 1]);
        assert!(!eval.is_any_thread_using_fp16());
        let mut server_buf = NnServerBuf::new(&eval, None);
        let mut rand = Rand::new_from_seed("serve-test");
        eval.serve(&mut server_buf, &mut rand, 0, 0);
    }

    #[test]
    fn test_load_model_requires_backend() {
        let mut eval = NnEvaluator::new(
            "m".to_string(),
            "m.bin".to_string(),
            String::new(),
            test_logger(),
            1,
            9,
            9,
            false,
            false,
            -1,
            0,
            false,
            String::new(),
            Enabled::False,
            0,
            Vec::new(),
            String::new(),
            false,
            0,
            true,
            &empty_cfg(),
        );
        assert!(eval.load_model().is_err());
    }

    #[test]
    fn test_load_model_with_dummy_backend() {
        use crate::backend::dummy::DummyBackend;

        let mut eval = NnEvaluator::new(
            "m".to_string(),
            "m.bin".to_string(),
            String::new(),
            test_logger(),
            1,
            9,
            9,
            false,
            false,
            -1,
            0,
            false,
            String::new(),
            Enabled::False,
            0,
            Vec::new(),
            String::new(),
            false,
            0,
            true,
            &empty_cfg(),
        );
        eval.set_backend(Arc::new(DummyBackend));
        eval.load_model().unwrap();
        assert!(!eval.internal_model_name().is_empty());
        assert!(eval.model_version() > 0);
        assert!(eval.compute_context.is_some());
        assert!(eval.compute_handle.is_some());
        assert!(eval.shared.input_buffers.lock().unwrap().is_some());
        assert!(eval.loaded_model.is_some());
    }

    #[test]
    fn test_load_model_skips_when_neural_net_less() {
        use crate::backend::dummy::DummyBackend;

        let mut eval = NnEvaluator::new(
            "m".to_string(),
            "/dev/null".to_string(),
            String::new(),
            test_logger(),
            1,
            9,
            9,
            false,
            false,
            -1,
            0,
            true, // debug_skip_neural_net is true for /dev/null
            String::new(),
            Enabled::False,
            0,
            Vec::new(),
            String::new(),
            false,
            0,
            true,
            &empty_cfg(),
        );
        eval.set_backend(Arc::new(DummyBackend));
        eval.load_model().unwrap();
        assert!(eval.loaded_model.is_none());
    }

    #[test]
    fn test_evaluate_uses_cache() {
        let eval = NnEvaluator::new(
            "m".to_string(),
            "m.bin".to_string(),
            String::new(),
            test_logger(),
            1,
            9,
            9,
            false,
            false,
            8,
            4,
            true,
            String::new(),
            Enabled::False,
            0,
            Vec::new(),
            String::new(),
            false,
            0,
            true,
            &empty_cfg(),
        );
        let board = Board::new(9, 9);
        let hist = BoardHistory::new(
            board.clone(),
            kata_game::board::P_BLACK,
            Rules::default(),
            0,
        );
        let mut buf1 = NNResultBuf::new();
        eval.evaluate(
            &board,
            &hist,
            kata_game::board::P_BLACK,
            &MiscNNInputParams::default(),
            &mut buf1,
            false,
            false,
        );
        let hash1 = buf1.result.as_ref().unwrap().nn_hash;

        let mut buf2 = NNResultBuf::new();
        eval.evaluate(
            &board,
            &hist,
            kata_game::board::P_BLACK,
            &MiscNNInputParams::default(),
            &mut buf2,
            false,
            false,
        );
        let hash2 = buf2.result.as_ref().unwrap().nn_hash;
        assert_eq!(hash1, hash2);
        assert!(Arc::ptr_eq(
            buf1.result.as_ref().unwrap(),
            buf2.result.as_ref().unwrap()
        ));
    }

    #[test]
    fn test_evaluate_skip_cache_bypasses_cache() {
        let eval = NnEvaluator::new(
            "m".to_string(),
            "m.bin".to_string(),
            String::new(),
            test_logger(),
            1,
            9,
            9,
            false,
            false,
            8,
            4,
            true,
            String::new(),
            Enabled::False,
            0,
            Vec::new(),
            String::new(),
            false,
            0,
            true,
            &empty_cfg(),
        );
        let board = Board::new(9, 9);
        let hist = BoardHistory::new(
            board.clone(),
            kata_game::board::P_BLACK,
            Rules::default(),
            0,
        );
        let mut buf1 = NNResultBuf::new();
        eval.evaluate(
            &board,
            &hist,
            kata_game::board::P_BLACK,
            &MiscNNInputParams::default(),
            &mut buf1,
            true,
            false,
        );
        let mut buf2 = NNResultBuf::new();
        eval.evaluate(
            &board,
            &hist,
            kata_game::board::P_BLACK,
            &MiscNNInputParams::default(),
            &mut buf2,
            true,
            false,
        );
        assert!(!Arc::ptr_eq(
            buf1.result.as_ref().unwrap(),
            buf2.result.as_ref().unwrap()
        ));
    }

    #[test]
    fn test_evaluate_async_server_thread_produces_result() {
        use crate::backend::dummy::DummyBackend;

        let mut eval = NnEvaluator::new(
            "m".to_string(),
            "m.bin".to_string(),
            String::new(),
            test_logger(),
            4,
            9,
            9,
            false,
            false,
            8,
            4,
            false, // must be false for load_model to create handles
            String::new(),
            Enabled::False,
            1,
            vec![0],
            "async-test".to_string(),
            false,
            0,
            true,
            &empty_cfg(),
        );
        eval.set_backend(Arc::new(DummyBackend));
        eval.load_model().unwrap();
        eval.spawn_server_threads();

        let board = Board::new(9, 9);
        let hist = BoardHistory::new(
            board.clone(),
            kata_game::board::P_BLACK,
            Rules::default(),
            0,
        );
        let mut buf = NNResultBuf::new();
        eval.evaluate(
            &board,
            &hist,
            kata_game::board::P_BLACK,
            &MiscNNInputParams::default(),
            &mut buf,
            false,
            false,
        );
        // Async path should produce a result via the server thread.
        assert!(buf.result.is_some(), "evaluate should produce a result");
        assert!(buf.has_result);

        eval.kill_server_threads();
        assert!(eval.num_batches_processed() > 0);
        assert!(eval.num_rows_processed() > 0);
    }
}

//! Self-play model manager.
//!
//! Corresponds to `cpp/program/selfplaymanager.h` and
//! `cpp/program/selfplaymanager.cpp`. This slice provides the struct layout,
//! constructors, destructor, public query/acquire methods, and the background
//! data-writing loops that drain finished games to `TrainingDataWriter`.

use std::io::Write;
use std::sync::{Arc, Weak};

use parking_lot::{Condvar, Mutex};

use kata_core::global::StringError;
use kata_core::logger::Logger;
use kata_core::thread::queue::ThreadSafeQueue;
use kata_core::time::timer::ClockTimer;
use kata_data::training::{FinishedGameData, TrainingDataWriter};
use kata_nn::eval::NnEvaluator;

/// Per-model state tracked by [`SelfplayManager`].
///
/// Mirrors the public `SelfplayManager::ModelData` struct declared in the C++
/// header. The finished-game queue is wrapped in an `Arc` so that the manager
/// and the detached data-writing thread can wait on it without holding the
/// per-model mutex for the whole pop. The rest of the fields are protected by
/// that mutex (`Arc<Mutex<ModelData>>`).
pub struct ModelData {
    pub model_name: String,
    pub nn_eval: Box<NnEvaluator>,
    pub game_started_count: i64,
    pub last_release_time: f64,
    pub has_data_write_loop: bool,
    pub finished_game_queue: Arc<ThreadSafeQueue<FinishedGameData>>,
    pub acquire_count: i32,
    pub tdata_writer: Box<TrainingDataWriter>,
    pub sgf_out: Option<Box<dyn Write + Send + Sync>>,
}

impl ModelData {
    /// Create a new `ModelData` entry.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_name: String,
        nn_eval: NnEvaluator,
        max_data_queue_size: usize,
        tdata_writer: TrainingDataWriter,
        sgf_out: Option<Box<dyn Write + Send + Sync>>,
        initial_last_release_time: f64,
        has_data_write_loop: bool,
    ) -> Self {
        Self {
            model_name,
            nn_eval: Box::new(nn_eval),
            game_started_count: 0,
            last_release_time: initial_last_release_time,
            has_data_write_loop,
            finished_game_queue: Arc::new(ThreadSafeQueue::with_max_size(max_data_queue_size)),
            acquire_count: 0,
            tdata_writer: Box::new(tdata_writer),
            sgf_out,
        }
    }
}

struct Inner {
    model_datas: Vec<Arc<Mutex<ModelData>>>,
    num_data_write_loops_active: i32,
    total_num_rows_processed: u64,
}

/// Manages a sequence of loaded neural-net models for self-play.
///
/// Models are loaded in order and can be acquired/released by worker threads.
/// The manager optionally cleans up older unused models and coordinates
/// background threads that write finished-game data to disk.
///
/// Because data-writing loops are detached threads that may outlive individual
/// method calls, the manager must always be created via [`SelfplayManager::new`],
/// which returns an `Arc<Self>`.
pub struct SelfplayManager {
    max_data_queue_size: usize,
    logger: Option<Arc<Logger>>,
    log_games_every: i64,
    auto_cleanup_all_but_latest_if_unused: bool,
    timer: ClockTimer,
    inner: Mutex<Inner>,
    data_write_loops_are_done: Condvar,
    self_ref: Mutex<Option<Weak<SelfplayManager>>>,
}

impl SelfplayManager {
    /// Create a new manager.
    pub fn new(
        max_data_queue_size: usize,
        logger: Option<Arc<Logger>>,
        log_games_every: i64,
        auto_cleanup_all_but_latest_if_unused: bool,
    ) -> Arc<Self> {
        Arc::new_cyclic(|weak| Self {
            max_data_queue_size,
            logger,
            log_games_every,
            auto_cleanup_all_but_latest_if_unused,
            timer: ClockTimer::new(),
            inner: Mutex::new(Inner {
                model_datas: Vec::new(),
                num_data_write_loops_active: 0,
                total_num_rows_processed: 0,
            }),
            data_write_loops_are_done: Condvar::new(),
            self_ref: Mutex::new(Some(weak.clone())),
        })
    }

    fn upgrade_self(&self) -> Arc<Self> {
        let guard = self.self_ref.lock();
        guard
            .as_ref()
            .and_then(Weak::upgrade)
            .expect("SelfplayManager must be created via SelfplayManager::new")
    }

    /// Total rows processed across all models ever managed, summed from each
    /// evaluator's live counter plus rows already accumulated from cleaned-up
    /// models.
    pub fn get_total_num_rows_processed(&self) -> u64 {
        let inner = self.inner.lock();
        let mut total = inner.total_num_rows_processed;
        for model_data in &inner.model_datas {
            total += model_data.lock().nn_eval.num_rows_processed();
        }
        total
    }

    /// Load a model and start a background data-writing loop.
    pub fn load_model_and_start_data_writing(
        &self,
        nn_eval: NnEvaluator,
        tdata_writer: TrainingDataWriter,
        sgf_out: Option<Box<dyn Write + Send + Sync>>,
    ) -> Result<(), StringError> {
        let manager_arc = self.upgrade_self();
        let model_name = nn_eval.model_name().to_string();
        let mut inner = self.inner.lock();
        if inner
            .model_datas
            .iter()
            .any(|m| m.lock().model_name == model_name)
        {
            return Err(StringError::new(format!(
                "SelfplayManager::loadModelAndStartDataWriting: Duplicate model name: {}",
                model_name
            )));
        }
        let initial_time = self.timer.get_seconds();
        let new_model = ModelData::new(
            model_name,
            nn_eval,
            self.max_data_queue_size,
            tdata_writer,
            sgf_out,
            initial_time,
            true,
        );
        let model_arc: Arc<Mutex<ModelData>> = Arc::new(Mutex::new(new_model));
        inner.model_datas.push(model_arc.clone());
        inner.num_data_write_loops_active += 1;
        self.maybe_auto_cleanup_already_locked(&mut inner);
        drop(inner);

        std::thread::spawn(move || {
            Self::run_data_write_loop_thread(manager_arc, model_arc);
        });

        Ok(())
    }

    /// Load a model without starting a background data-writing loop.
    pub fn load_model_no_data_writing_loop(
        &self,
        nn_eval: NnEvaluator,
        tdata_writer: TrainingDataWriter,
        sgf_out: Option<Box<dyn Write + Send + Sync>>,
    ) -> Result<(), StringError> {
        let model_name = nn_eval.model_name().to_string();
        let mut inner = self.inner.lock();
        if inner
            .model_datas
            .iter()
            .any(|m| m.lock().model_name == model_name)
        {
            return Err(StringError::new(format!(
                "SelfplayManager::loadModelNoDataWritingLoop: Duplicate model name: {}",
                model_name
            )));
        }
        let initial_time = self.timer.get_seconds();
        let new_model = ModelData::new(
            model_name,
            nn_eval,
            self.max_data_queue_size,
            tdata_writer,
            sgf_out,
            initial_time,
            false,
        );
        inner.model_datas.push(Arc::new(Mutex::new(new_model)));
        self.maybe_auto_cleanup_already_locked(&mut inner);
        Ok(())
    }

    /// Number of currently-loaded models.
    pub fn num_models(&self) -> usize {
        let inner = self.inner.lock();
        inner.model_datas.len()
    }

    /// Model names from earliest to latest.
    pub fn model_names(&self) -> Vec<String> {
        let inner = self.inner.lock();
        inner
            .model_datas
            .iter()
            .map(|m| m.lock().model_name.clone())
            .collect()
    }

    /// Name of the most recently loaded model.
    pub fn get_latest_model_name(&self) -> Result<String, StringError> {
        let inner = self.inner.lock();
        if let Some(model_data) = inner.model_datas.last() {
            Ok(model_data.lock().model_name.clone())
        } else {
            Err(StringError::new(
                "SelfplayManager::getLatestModelName: no models loaded",
            ))
        }
    }

    /// Whether a model with the given name is currently loaded.
    pub fn has_model(&self, model_name: &str) -> bool {
        let inner = self.inner.lock();
        inner
            .model_datas
            .iter()
            .any(|m| m.lock().model_name == model_name)
    }

    /// Acquire a model by name.
    ///
    /// Returns a raw pointer to the evaluator, or `None` if the model is not
    /// loaded. The caller must call one of the `release` methods when done.
    pub fn acquire_model(&self, model_name: &str) -> Option<*const NnEvaluator> {
        let inner = self.inner.lock();
        inner
            .model_datas
            .iter()
            .find(|m| m.lock().model_name == model_name)
            .map(|model_data| {
                let mut model = model_data.lock();
                model.acquire_count += 1;
                model.nn_eval.as_ref() as *const NnEvaluator
            })
    }

    /// Acquire the latest loaded model.
    pub fn acquire_latest(&self) -> Option<*const NnEvaluator> {
        let inner = self.inner.lock();
        inner.model_datas.last().map(|model_data| {
            let mut model = model_data.lock();
            model.acquire_count += 1;
            model.nn_eval.as_ref() as *const NnEvaluator
        })
    }

    /// Release a model by name.
    pub fn release(&self, model_name: &str) {
        let mut inner = self.inner.lock();
        if let Some(model_data) = inner
            .model_datas
            .iter()
            .find(|m| m.lock().model_name == model_name)
        {
            self.release_already_locked(model_data);
            self.maybe_auto_cleanup_already_locked(&mut inner);
        }
    }

    /// Release a model by the evaluator pointer returned from `acquire_model`
    /// or `acquire_latest`.
    pub fn release_by_nn_eval(&self, nn_eval: *const NnEvaluator) {
        let mut inner = self.inner.lock();
        if let Some(model_data) = inner
            .model_datas
            .iter()
            .find(|m| std::ptr::eq(m.lock().nn_eval.as_ref(), nn_eval))
        {
            self.release_already_locked(model_data);
            self.maybe_auto_cleanup_already_locked(&mut inner);
        }
    }

    /// Clean up any currently-unused models whose last release was older than
    /// `seconds` ago.
    pub fn cleanup_unused_models_older_than(&self, seconds: f64) {
        let mut inner = self.inner.lock();
        let now = self.timer.get_seconds();
        let mut i = 0;
        while i < inner.model_datas.len() {
            let should_remove = {
                let model = inner.model_datas[i].lock();
                model.acquire_count <= 0 && now - model.last_release_time > seconds
            };
            if should_remove {
                let model_data = inner.model_datas.remove(i);
                assert_eq!(model_data.lock().acquire_count, 0);
                if let Some(logger) = &self.logger {
                    logger.write(&format!(
                        "Unloading network that hasn't been used in a while: {}",
                        model_data.lock().model_name
                    ));
                }
                model_data.lock().finished_game_queue.set_read_only();
                inner.total_num_rows_processed += model_data.lock().nn_eval.num_rows_processed();
                // model_data Arc may still be held by a data-writing thread.
            } else {
                i += 1;
            }
        }
    }

    /// Clear the evaluation caches of any models that are currently unused.
    pub fn clear_unused_model_caches(&self) {
        let inner = self.inner.lock();
        for model_data in &inner.model_datas {
            let model = model_data.lock();
            if model.acquire_count <= 0 {
                model.nn_eval.clear_cache();
            }
        }
    }

    /// Increment the per-model game-started counter and maybe log stats.
    ///
    /// # Safety
    /// `nn_eval` must be a pointer previously returned by this manager's
    /// `acquire_model` or `acquire_latest` and the model must still be loaded.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn count_one_game_started(&self, nn_eval: *const NnEvaluator) -> Result<(), StringError> {
        let inner = self.inner.lock();
        let model_data = inner
            .model_datas
            .iter()
            .find(|m| std::ptr::eq(m.lock().nn_eval.as_ref(), nn_eval))
            .ok_or_else(|| {
                StringError::new(
                    "SelfplayManager::countOneGameStarted: could not find model. Possible bug - client did not acquire model?",
                )
            })?;
        let mut model = model_data.lock();
        model.game_started_count += 1;
        let game_started_count = model.game_started_count;
        drop(model);
        drop(inner);

        if let Some(logger) = &self.logger {
            if game_started_count % self.log_games_every == 0 {
                // SAFETY: caller must hold an active acquire on this model.
                let model_name = unsafe { &*nn_eval }.model_name();
                logger.write(&format!(
                    "Started {} games with {}",
                    game_started_count, model_name
                ));
            }
            let log_nn_every = std::cmp::max(self.log_games_every * 100, 1000);
            if game_started_count % log_nn_every == 0 {
                // SAFETY: caller must hold an active acquire on this model.
                let eval = unsafe { &*nn_eval };
                logger.write(eval.model_file_name());
                logger.write(&format!("NN rows: {}", eval.num_rows_processed()));
                logger.write(&format!("NN batches: {}", eval.num_batches_processed()));
                logger.write(&format!(
                    "NN avg batch size: {}",
                    eval.average_processed_batch_size()
                ));
            }
        }
        Ok(())
    }

    /// Enqueue a finished game to be written by a model's data-writing loop.
    pub fn enqueue_data_to_write_by_name(
        &self,
        model_name: &str,
        game_data: FinishedGameData,
    ) -> Result<(), StringError> {
        let inner = self.inner.lock();
        let model_data = inner
            .model_datas
            .iter()
            .find(|m| m.lock().model_name == model_name)
            .ok_or_else(|| {
                StringError::new(
                    "SelfplayManager::enqueueDataToWrite: could not find model. Possible bug - client did not acquire model?",
                )
            })?;
        if !model_data.lock().has_data_write_loop {
            return Err(StringError::new(
                "SelfplayManager::enqueueDataToWrite: model has no data write loop",
            ));
        }
        let queue = Arc::clone(&model_data.lock().finished_game_queue);
        drop(inner);
        queue.wait_push(game_data);
        Ok(())
    }

    /// Enqueue a finished game to be written by a model's data-writing loop.
    pub fn enqueue_data_to_write_by_nn_eval(
        &self,
        nn_eval: *const NnEvaluator,
        game_data: FinishedGameData,
    ) -> Result<(), StringError> {
        let inner = self.inner.lock();
        let model_data = inner
            .model_datas
            .iter()
            .find(|m| std::ptr::eq(m.lock().nn_eval.as_ref(), nn_eval))
            .ok_or_else(|| {
                StringError::new(
                    "SelfplayManager::enqueueDataToWrite: could not find model. Possible bug - client did not acquire model?",
                )
            })?;
        if !model_data.lock().has_data_write_loop {
            return Err(StringError::new(
                "SelfplayManager::enqueueDataToWrite: model has no data write loop",
            ));
        }
        let queue = Arc::clone(&model_data.lock().finished_game_queue);
        drop(inner);
        queue.wait_push(game_data);
        Ok(())
    }

    /// Call a closure with the model's writer and optional SGF output stream.
    ///
    /// This is intended for models loaded with
    /// [`SelfplayManager::load_model_no_data_writing_loop`].
    pub fn with_data_writers<F, R>(
        &self,
        nn_eval: *const NnEvaluator,
        f: F,
    ) -> Result<R, StringError>
    where
        F: FnOnce(&mut TrainingDataWriter, Option<&mut Box<dyn Write + Send + Sync>>) -> R,
    {
        struct SgfOutGuard<'a> {
            slot: &'a mut Option<Box<dyn Write + Send + Sync>>,
            taken: Option<Box<dyn Write + Send + Sync>>,
        }
        impl<'a> SgfOutGuard<'a> {
            fn new(slot: &'a mut Option<Box<dyn Write + Send + Sync>>) -> Self {
                let taken = slot.take();
                Self { slot, taken }
            }
            fn as_mut(&mut self) -> Option<&mut Box<dyn Write + Send + Sync>> {
                self.taken.as_mut()
            }
        }
        impl<'a> Drop for SgfOutGuard<'a> {
            fn drop(&mut self) {
                *self.slot = self.taken.take();
            }
        }

        let inner = self.inner.lock();
        let model_data = inner
            .model_datas
            .iter()
            .find(|m| std::ptr::eq(m.lock().nn_eval.as_ref(), nn_eval))
            .ok_or_else(|| {
                StringError::new(
                    "SelfplayManager::withDataWriters: could not find model. Possible bug - client did not acquire model?",
                )
            })?;
        if model_data.lock().has_data_write_loop {
            return Err(StringError::new(
                "SelfplayManager::withDataWriters: model has data write loop",
            ));
        }
        let model = &mut *model_data.lock();
        let sgf_out = &mut model.sgf_out;
        let tdata_writer = &mut model.tdata_writer;
        let mut sgf_guard = SgfOutGuard::new(sgf_out);
        Ok(f(tdata_writer, sgf_guard.as_mut()))
    }

    /// For internal use: entry point for a data-writing loop thread.
    pub fn run_data_write_loop(&self, model_data: &Arc<Mutex<ModelData>>) {
        // The thread wrapper already catches panics and ensures cleanup.
        self.run_data_write_loop_impl(model_data);
    }

    fn release_already_locked(&self, model_data: &Arc<Mutex<ModelData>>) {
        let mut model = model_data.lock();
        model.last_release_time = self.timer.get_seconds();
        model.acquire_count -= 1;
    }

    fn maybe_auto_cleanup_already_locked(&self, inner: &mut Inner) {
        if self.auto_cleanup_all_but_latest_if_unused && !inner.model_datas.is_empty() {
            let mut i = 0;
            while i < inner.model_datas.len() - 1 {
                let should_remove = inner.model_datas[i].lock().acquire_count <= 0;
                if should_remove {
                    let model_data = inner.model_datas.remove(i);
                    assert_eq!(model_data.lock().acquire_count, 0);
                    model_data.lock().finished_game_queue.set_read_only();
                    inner.total_num_rows_processed +=
                        model_data.lock().nn_eval.num_rows_processed();
                    // model_data Arc may still be held by a data-writing thread.
                } else {
                    i += 1;
                }
            }
        }
    }

    fn run_data_write_loop_impl(&self, model_data: &Arc<Mutex<ModelData>>) {
        let model_name = {
            let model = model_data.lock();
            model.model_name.clone()
        };

        if let Some(logger) = &self.logger {
            logger.write(&format!(
                "Data write loop starting for neural net: {}",
                model_name
            ));
        }

        let queue = {
            let model = model_data.lock();
            Arc::clone(&model.finished_game_queue)
        };
        let max_data_queue_size = self.max_data_queue_size;

        loop {
            let queue_size = queue.size();
            if queue_size > max_data_queue_size / 2 {
                if let Some(logger) = &self.logger {
                    logger.write(&format!(
                        "WARNING: Struggling to keep up writing data, {} games enqueued out of {} max",
                        queue_size, max_data_queue_size
                    ));
                }
            }

            let mut game_data = FinishedGameData::default();
            if !queue.wait_pop(&mut game_data) {
                break;
            }

            let write_result = {
                let mut model = model_data.lock();
                model.tdata_writer.write_game(&game_data)
            };
            if let Err(e) = write_result {
                if let Some(logger) = &self.logger {
                    logger.write(&format!("ERROR writing game for {}: {}", model_name, e.0));
                }
            }

            // Full SGF serialization is not yet ported; flush the stream at the
            // end of the loop instead of per game.
            {
                let _sgf = model_data.lock();
                // TODO: write SGF for `game_data` when a Rust serializer exists.
            }
        }

        {
            let mut model = model_data.lock();
            if let Err(e) = model.tdata_writer.flush_if_nonempty() {
                if let Some(logger) = &self.logger {
                    logger.write(&format!(
                        "ERROR flushing training data for {}: {}",
                        model_name, e.0
                    ));
                }
            }
        }

        {
            let mut model = model_data.lock();
            if let Some(sgf_out) = model.sgf_out.as_mut() {
                let _ = sgf_out.flush();
            }
        }

        {
            let model = model_data.lock();
            assert_eq!(
                model.acquire_count, 0,
                "SelfplayManager data write loop finished while model is still acquired"
            );
        }

        if let Some(logger) = &self.logger {
            let (model_file_name, num_rows, num_batches, avg_batch) = {
                let model = model_data.lock();
                (
                    model.nn_eval.model_file_name().to_string(),
                    model.nn_eval.num_rows_processed(),
                    model.nn_eval.num_batches_processed(),
                    model.nn_eval.average_processed_batch_size(),
                )
            };
            logger.write(&format!("Final cleanup of net: {}", model_file_name));
            logger.write(&format!("Final NN rows: {}", num_rows));
            logger.write(&format!("Final NN batches: {}", num_batches));
            logger.write(&format!("Final NN avg batch size: {}", avg_batch));
        }
    }

    fn run_data_write_loop_thread(
        manager: Arc<SelfplayManager>,
        model_data: Arc<Mutex<ModelData>>,
    ) {
        struct DoneGuard {
            manager: Arc<SelfplayManager>,
        }
        impl Drop for DoneGuard {
            fn drop(&mut self) {
                let mut inner = self.manager.inner.lock();
                inner.num_data_write_loops_active -= 1;
                if inner.num_data_write_loops_active <= 0 {
                    self.manager.data_write_loops_are_done.notify_all();
                }
            }
        }

        let _guard = DoneGuard {
            manager: manager.clone(),
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            manager.run_data_write_loop(&model_data);
        }));

        if let Some(logger) = &manager.logger {
            match result {
                Ok(()) => {
                    let name = model_data.lock().model_name.clone();
                    logger.write(&format!(
                        "Data write loop cleaned up and terminating for {}",
                        name
                    ));
                }
                Err(_) => {
                    let name = model_data.lock().model_name.clone();
                    logger.write(&format!(
                        "ERROR: Data write loop panicked for neural net: {}",
                        name
                    ));
                }
            }
        }
    }
}

impl Drop for SelfplayManager {
    fn drop(&mut self) {
        let mut inner = self.inner.lock();
        for model_data in &inner.model_datas {
            assert_eq!(
                model_data.lock().acquire_count,
                0,
                "SelfplayManager dropped while a model is still acquired"
            );
            model_data.lock().finished_game_queue.set_read_only();
        }
        let rows_to_accumulate: u64 = inner
            .model_datas
            .iter()
            .map(|m| m.lock().nn_eval.num_rows_processed())
            .sum();
        inner.total_num_rows_processed += rows_to_accumulate;
        inner.model_datas.clear();

        while inner.num_data_write_loops_active > 0 {
            self.data_write_loops_are_done.wait(&mut inner);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use kata_core::config::ConfigParser;
    use kata_core::logger::{Logger, LoggerOptions};
    use kata_data::training::{
        FinishedGameData, NNRawStats, PolicyTarget, PolicyTargetMove, QValueTargets,
        TrainingDataWriter, ValueTargets,
    };
    use kata_game::board::{Board, C_EMPTY, MAX_ARR_SIZE, P_BLACK, P_WHITE};
    use kata_game::history::BoardHistory;
    use kata_game::rules::Rules;
    use kata_nn::backend::Enabled;
    use kata_nn::backend::dummy::DummyBackend;
    use kata_nn::eval::NnEvaluator;

    fn test_logger() -> Arc<Logger> {
        Arc::new(Logger::new(LoggerOptions::default(), None))
    }

    fn empty_cfg() -> ConfigParser {
        ConfigParser::new(false, false)
    }

    fn dummy_nn_eval(model_name: &str) -> NnEvaluator {
        let _backend = DummyBackend;
        NnEvaluator::new(
            model_name.to_string(),
            format!("{}.bin", model_name),
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
        )
    }

    fn dummy_writer() -> TrainingDataWriter {
        TrainingDataWriter::new(".", 7, 16, 1.0, 19, 19, "testseed").unwrap()
    }

    fn dummy_debug_writer() -> TrainingDataWriter {
        TrainingDataWriter::new_debug(Box::new(Vec::new()), 7, 16, 1.0, 5, 5, 1, "testseed")
            .unwrap()
    }

    fn finished_game_one_move() -> FinishedGameData {
        use kata_game::board::location;

        let x_size = 5;
        let y_size = 5;
        let board = Board::new(x_size, y_size);
        let start_hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let move_loc = location::get_loc(2, 2, x_size);
        let mut end_board = board.clone();
        let mut end_hist = start_hist.clone();
        end_hist.make_board_move_assume_legal(&mut end_board, move_loc, P_BLACK);
        end_hist.is_game_finished = true;
        end_hist.winner = P_WHITE;
        end_hist.is_scored = true;
        end_hist.final_white_minus_black_score = 1.0;

        FinishedGameData {
            start_board: board,
            start_hist,
            start_pla: P_BLACK,
            end_hist,
            has_full_data: true,
            target_weight_by_turn: vec![1.0f32],
            target_weight_by_turn_unrounded: vec![1.0f32],
            policy_targets_by_turn: vec![PolicyTarget {
                policy_targets: vec![PolicyTargetMove {
                    loc: move_loc,
                    policy_target: 1,
                }],
                unreduced_num_visits: 1,
            }],
            policy_surprise_by_turn: vec![0.0],
            policy_entropy_by_turn: vec![0.0],
            search_entropy_by_turn: vec![0.0],
            white_value_targets_by_turn: vec![
                ValueTargets {
                    win: 0.5,
                    loss: 0.5,
                    no_result: 0.0,
                    score: 0.0,
                    has_lead: false,
                    lead: 0.0,
                },
                ValueTargets {
                    win: 1.0,
                    loss: 0.0,
                    no_result: 0.0,
                    score: 0.0,
                    has_lead: false,
                    lead: 0.0,
                },
            ],
            white_q_value_targets_by_turn: vec![QValueTargets { targets: vec![] }],
            nn_raw_stats_by_turn: vec![NNRawStats {
                white_win_loss: 0.0,
                white_score_mean: 0.0,
                policy_entropy: 0.0,
            }],
            final_full_area: vec![C_EMPTY; MAX_ARR_SIZE],
            final_ownership: vec![C_EMPTY; MAX_ARR_SIZE],
            final_seki_areas: vec![false; MAX_ARR_SIZE],
            final_white_scoring: vec![0.0f32; MAX_ARR_SIZE],
            ..FinishedGameData::default()
        }
    }

    #[test]
    fn test_selfplay_manager_basics() {
        let manager = SelfplayManager::new(16, None, 100, false);
        assert_eq!(manager.num_models(), 0);

        let nn_eval = dummy_nn_eval("model-a");
        let writer = dummy_writer();
        manager
            .load_model_no_data_writing_loop(nn_eval, writer, None)
            .unwrap();

        assert_eq!(manager.num_models(), 1);
        assert!(manager.has_model("model-a"));
        assert!(!manager.has_model("model-b"));
        assert_eq!(manager.model_names(), vec!["model-a"]);
        assert_eq!(manager.get_latest_model_name().unwrap(), "model-a");
        assert_eq!(manager.get_total_num_rows_processed(), 0);
    }

    #[test]
    fn test_acquire_latest_release_roundtrip() {
        let manager = SelfplayManager::new(16, None, 100, false);
        let nn_eval = dummy_nn_eval("model-a");
        let writer = dummy_writer();
        manager
            .load_model_no_data_writing_loop(nn_eval, writer, None)
            .unwrap();

        let ptr = manager
            .acquire_latest()
            .expect("acquire_latest should succeed");
        assert!(!ptr.is_null());
        // SAFETY: pointer came from acquire_latest and we have not released it.
        assert_eq!(unsafe { &*ptr }.model_name(), "model-a");

        manager.release("model-a");
        assert_eq!(manager.num_models(), 1);
        assert!(manager.has_model("model-a"));
    }

    #[test]
    fn test_duplicate_model_name_rejected() {
        let manager = SelfplayManager::new(16, None, 100, false);
        let nn_eval = dummy_nn_eval("model-a");
        let writer = dummy_writer();
        manager
            .load_model_no_data_writing_loop(nn_eval, writer, None)
            .unwrap();

        let nn_eval2 = dummy_nn_eval("model-a");
        let writer2 = dummy_writer();
        assert!(
            manager
                .load_model_no_data_writing_loop(nn_eval2, writer2, None)
                .is_err()
        );
    }

    #[test]
    fn test_data_write_loop_drains_queue_and_exits() {
        let manager = SelfplayManager::new(16, None, 100, false);
        let nn_eval = dummy_nn_eval("model-loop");
        let writer = dummy_debug_writer();
        manager
            .load_model_and_start_data_writing(nn_eval, writer, None)
            .unwrap();

        let game = finished_game_one_move();
        manager
            .enqueue_data_to_write_by_name("model-loop", game)
            .unwrap();

        // Dropping the manager sets the queue read-only and waits for the loop.
        drop(manager);
    }

    #[test]
    fn test_with_data_writers_flushes_no_loop_model() {
        let manager = SelfplayManager::new(16, None, 100, false);
        let nn_eval = dummy_nn_eval("model-no-loop");
        let writer = dummy_debug_writer();
        manager
            .load_model_no_data_writing_loop(nn_eval, writer, None)
            .unwrap();

        let ptr = manager.acquire_latest().unwrap();
        manager
            .with_data_writers(ptr, |writer, _sgf| {
                writer.write_game(&finished_game_one_move()).unwrap();
                let flushed = writer.flush_if_nonempty().unwrap();
                assert!(flushed.is_some());
            })
            .unwrap();
        manager.release_by_nn_eval(ptr);
    }
}

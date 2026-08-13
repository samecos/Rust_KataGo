//! Asynchronous search bot.
//!
//! Corresponds to `cpp/search/asyncbot.h` and `cpp/search/asyncbot.cpp`.
//! The background search thread runs in a dedicated `std::thread` and
//! communicates with the foreground thread through a `parking_lot` mutex,
//! condition variables and an atomic stop flag.

#![allow(
    clippy::all,
    dead_code,
    missing_docs,
    unsafe_op_in_unsafe_fn,
    unused_variables,
    clippy::too_many_arguments
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI16, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use kata_core::logger::Logger;
use kata_game::board::{Board, Loc, NULL_LOC, Player};
use kata_game::history::BoardHistory;
use kata_nn::eval::NnEvaluator;
use parking_lot::{Condvar, Mutex};

use crate::eval_cache::EvalCacheTable;
use crate::params::SearchParams;
use crate::pattern_bonus::PatternBonusTable;
use crate::reported_values::ReportedSearchValues;
use crate::search::Search;
use crate::time_control::TimeControls;

/// Mutable foreground/background coordination state.
struct ControlState {
    is_running: bool,
    is_pondering: bool,
    is_killed: bool,
    is_search_begun: bool,
    callback_loop_should_stop: bool,
    queued_search_id: i32,
    queued_on_move: Option<Box<dyn Fn(Loc, i32, &Search) + Send + Sync>>,
    time_controls: TimeControls,
    search_factor: f64,
    analyze_callback_period: f64,
    analyze_first_callback_after: f64,
    analyze_callback: Option<Box<dyn Fn(&Search) + Send + Sync>>,
    search_begun_callback: Option<Box<dyn Fn() + Send + Sync>>,
}

impl ControlState {
    fn new() -> Self {
        Self {
            is_running: false,
            is_pondering: false,
            is_killed: false,
            is_search_begun: false,
            callback_loop_should_stop: false,
            queued_search_id: 0,
            queued_on_move: None,
            time_controls: TimeControls::new(),
            search_factor: 1.0,
            analyze_callback_period: -1.0,
            analyze_first_callback_after: -1.0,
            analyze_callback: None,
            search_begun_callback: None,
        }
    }
}

/// All state shared between the foreground API and the background search thread.
struct AsyncBotInner<'a> {
    search: Search<'a>,
    control: Mutex<ControlState>,
    thread_waiting_to_search: Condvar,
    user_waiting_for_stop: Condvar,
    callback_loop_waiting: Condvar,
    callback_loop_waiting_for_search_begun: Condvar,
    should_stop_now: AtomicBool,
}

impl<'a> AsyncBotInner<'a> {
    fn new(search: Search<'a>) -> Self {
        Self {
            search,
            control: Mutex::new(ControlState::new()),
            thread_waiting_to_search: Condvar::new(),
            user_waiting_for_stop: Condvar::new(),
            callback_loop_waiting: Condvar::new(),
            callback_loop_waiting_for_search_begun: Condvar::new(),
            should_stop_now: AtomicBool::new(false),
        }
    }

    /// Background thread entry point.
    ///
    /// # Safety
    /// `inner` must be a valid pointer produced by `Arc::into_raw` for this bot,
    /// and it must remain valid until this function returns.
    unsafe fn search_thread_loop(inner: *mut AsyncBotInner<'a>) {
        let inner_ref = &*inner;
        let mut guard = inner_ref.control.lock();

        loop {
            while !guard.is_running && !guard.is_killed {
                inner_ref.thread_waiting_to_search.wait(&mut guard);
            }
            if guard.is_killed {
                guard.is_running = false;
                guard.is_pondering = false;
                inner_ref.user_waiting_for_stop.notify_all();
                break;
            }

            let pondering = guard.is_pondering;
            let tc = guard.time_controls.clone();
            let search_factor = guard.search_factor;
            let mut callback_period = guard.analyze_callback_period;
            let mut first_callback_after = guard.analyze_first_callback_after;
            let analyze_callback_local = guard.analyze_callback.take();
            let search_begun_callback_local = guard.search_begun_callback.take();
            guard.is_search_begun = false;
            guard.callback_loop_should_stop = false;
            drop(guard);

            // Avoid absurdly large callback timeouts that can cause wait_for to hang.
            if callback_period >= 1e7 {
                callback_period = -1.0;
            }
            if first_callback_after >= 1e7 {
                first_callback_after = -1.0;
                callback_period = -1.0;
            }

            let using_callback_loop = (first_callback_after >= 0.0 || callback_period >= 0.0)
                && analyze_callback_local.is_some();

            let search_begun = {
                let inner_raw = inner;
                let cb = search_begun_callback_local;
                move || {
                    if let Some(ref f) = cb {
                        f();
                    }
                    if using_callback_loop {
                        let inner_ref = &*inner_raw;
                        let mut g = inner_ref.control.lock();
                        g.is_search_begun = true;
                        inner_ref
                            .callback_loop_waiting_for_search_begun
                            .notify_all();
                    }
                }
            };

            let callback_handle = if using_callback_loop {
                let inner_addr = inner as usize;
                let cb = analyze_callback_local.unwrap();
                Some(std::thread::spawn(move || {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        AsyncBotInner::callback_loop(
                            inner_addr as *mut _,
                            first_callback_after,
                            callback_period,
                            cb,
                        );
                    }));
                }))
            } else {
                None
            };

            let search = &mut (*inner).search;
            let logger = search.logger;
            let should_stop_early = || inner_ref.should_stop_now.load(Ordering::Acquire);

            let search_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                search.run_whole_search_full(
                    Some(&search_begun),
                    Some(&should_stop_early),
                    pondering,
                    &tc,
                    search_factor,
                );
                search.get_chosen_move_loc()
            }));

            let (move_loc, search_ok) = match search_result {
                Ok(loc) => (loc, true),
                Err(payload) => {
                    let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                        (*s).to_string()
                    } else if let Some(s) = payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    let full = format!("ERROR: Async bot search thread failed: {}", msg);
                    if let Some(logger) = logger {
                        logger.write(&full);
                    } else {
                        eprintln!("{}", full);
                    }
                    (NULL_LOC, false)
                }
            };

            if let Some(handle) = callback_handle {
                {
                    let mut g = inner_ref.control.lock();
                    g.callback_loop_should_stop = true;
                    inner_ref
                        .callback_loop_waiting_for_search_begun
                        .notify_all();
                    inner_ref.callback_loop_waiting.notify_all();
                }
                let _ = handle.join();
            }

            guard = inner_ref.control.lock();
            if search_ok {
                if let Some(ref f) = guard.queued_on_move {
                    f(move_loc, guard.queued_search_id, &(*inner).search);
                }
            }
            guard.is_running = false;
            guard.is_pondering = false;
            inner_ref.user_waiting_for_stop.notify_all();
        }

        // Restore the Arc owned by the raw pointer so its strong count is dropped.
        let _ = Arc::from_raw(inner);
    }

    /// Periodic analysis callback loop.
    ///
    /// # Safety
    /// `inner` must be valid for the lifetime of this function. The calling
    /// thread must be the one currently running the search, otherwise the
    /// `Search` reference passed to the callback may alias with a mutable
    /// borrow.
    unsafe fn callback_loop(
        inner: *mut AsyncBotInner<'a>,
        first_callback_after: f64,
        callback_period: f64,
        callback: Box<dyn Fn(&Search) + Send + Sync>,
    ) {
        let inner_ref = &*inner;
        let mut guard = inner_ref.control.lock();

        while !guard.is_search_begun && !guard.callback_loop_should_stop {
            inner_ref
                .callback_loop_waiting_for_search_begun
                .wait(&mut guard);
        }
        if guard.callback_loop_should_stop {
            return;
        }

        let mut got_report_yet = false;
        let mut period_to_wait = first_callback_after;

        loop {
            if period_to_wait < 0.0 {
                return;
            }

            let timeout = Duration::from_secs_f64(period_to_wait);
            inner_ref
                .callback_loop_waiting
                .wait_for(&mut guard, timeout);
            if guard.callback_loop_should_stop {
                return;
            }

            if !got_report_yet {
                let search = &(*inner).search;
                let mut vals = ReportedSearchValues::new();
                if !search.get_root_values(&mut vals) {
                    period_to_wait = period_to_wait * 1.25 + 0.001;
                    let cap = first_callback_after.max(if callback_period < 0.0 {
                        1.0
                    } else {
                        callback_period
                    });
                    period_to_wait = period_to_wait.min(cap);
                    continue;
                }
                got_report_yet = true;
                period_to_wait = callback_period;
            }

            let search = &(*inner).search;
            drop(guard);
            callback(search);
            guard = inner_ref.control.lock();
        }
    }
}

/// Asynchronous wrapper around a `Search`.
///
/// Mirrors the C++ `AsyncBot` API. The background search thread is spawned in
/// the constructor and joined in `Drop`. All foreground methods that mutate the
/// position or parameters first stop any running search and wait for it to end.
pub struct AsyncBot<'a> {
    inner: Arc<AsyncBotInner<'a>>,
    search_thread: Option<JoinHandle<()>>,
}

impl<'a> AsyncBot<'a> {
    /// Construct a bot with a single neural-net evaluator.
    pub fn new(
        params: SearchParams,
        nn_eval: &'a NnEvaluator,
        logger: &'a Logger,
        rand_seed: &str,
    ) -> Self {
        Self::new_with_human(params, nn_eval, None, logger, rand_seed)
    }

    /// Construct a bot with an optional separate human-style evaluator.
    pub fn new_with_human(
        params: SearchParams,
        nn_eval: &'a NnEvaluator,
        human_eval: Option<&'a NnEvaluator>,
        logger: &'a Logger,
        rand_seed: &str,
    ) -> Self {
        let search = Search::new_with_human(params, nn_eval, human_eval, logger, rand_seed);
        let inner = Arc::new(AsyncBotInner::new(search));
        let inner_raw = Arc::into_raw(Arc::clone(&inner));
        let inner_addr = inner_raw as usize;

        let search_thread = Some(std::thread::spawn(move || {
            unsafe { AsyncBotInner::search_thread_loop(inner_addr as *mut _) };
        }));

        Self {
            inner,
            search_thread,
        }
    }

    fn inner_ptr(&self) -> *mut AsyncBotInner<'a> {
        Arc::as_ptr(&self.inner) as *mut _
    }

    fn search_ref(&self) -> &Search<'a> {
        unsafe { &(*self.inner_ptr()).search }
    }

    fn search_mut(&mut self) -> &mut Search<'a> {
        unsafe { &mut (*self.inner_ptr()).search }
    }
}

impl<'a> AsyncBot<'a> {
    pub fn get_root_board(&self) -> &Board {
        &self.search_ref().root_board
    }

    pub fn get_root_hist(&self) -> &BoardHistory {
        &self.search_ref().root_history
    }

    pub fn get_root_pla(&self) -> Player {
        self.search_ref().root_pla
    }

    pub fn get_playout_doubling_advantage_pla(&self) -> Player {
        self.search_ref().get_playout_doubling_advantage_pla()
    }

    pub fn get_params(&self) -> &SearchParams {
        &self.search_ref().search_params
    }

    /// Get the search directly.
    ///
    /// If the bot is doing anything asynchronous, the search may still be
    /// running.
    pub fn get_search(&self) -> &Search<'a> {
        self.search_ref()
    }

    /// Get the search after stopping and waiting for any existing search.
    pub fn get_search_stop_and_wait(&mut self) -> &mut Search<'a> {
        self.stop_and_wait();
        self.search_mut()
    }
}

impl<'a> AsyncBot<'a> {
    pub fn set_position(&mut self, pla: Player, board: &Board, history: &BoardHistory) {
        self.stop_and_wait();
        self.search_mut().set_position(pla, board, history);
    }

    pub fn set_player_and_clear_history(&mut self, pla: Player) {
        self.stop_and_wait();
        self.search_mut().set_player_and_clear_history(pla);
    }

    pub fn set_player_if_new(&mut self, pla: Player) {
        self.stop_and_wait();
        self.search_mut().set_player_if_new(pla);
    }

    pub fn set_komi_if_new(&mut self, new_komi: f64) {
        self.stop_and_wait();
        self.search_mut().set_komi_if_new(new_komi);
    }

    pub fn set_root_hint_loc(&mut self, loc: Loc) {
        self.stop_and_wait();
        self.search_mut().set_root_hint_loc(loc);
    }

    pub fn set_avoid_move_until_by_loc(&mut self, b_vec: &[i32], w_vec: &[i32]) {
        self.stop_and_wait();
        self.search_mut().set_avoid_move_until_by_loc(b_vec, w_vec);
    }

    pub fn set_avoid_move_until_rescale_root(&mut self, b: bool) {
        self.stop_and_wait();
        self.search_mut().set_avoid_move_until_rescale_root(b);
    }

    pub fn set_always_include_owner_map(&mut self, b: bool) {
        self.stop_and_wait();
        self.search_mut().set_always_include_owner_map(b);
    }

    pub fn set_params(&mut self, params: &SearchParams) {
        self.stop_and_wait();
        self.search_mut().set_params(params);
    }

    pub fn set_params_no_clearing(&mut self, params: &SearchParams) {
        self.stop_and_wait();
        self.search_mut().set_params_no_clearing(params);
    }

    pub fn set_external_pattern_bonus_table(&mut self, table: Option<Box<PatternBonusTable>>) {
        self.stop_and_wait();
        self.search_mut().set_external_pattern_bonus_table(table);
    }

    pub fn set_copy_of_external_pattern_bonus_table(
        &mut self,
        table: &Option<Box<PatternBonusTable>>,
    ) {
        self.stop_and_wait();
        self.search_mut()
            .set_copy_of_external_pattern_bonus_table(table);
    }

    pub fn set_external_eval_cache(&mut self, cache: Option<Arc<EvalCacheTable>>) {
        self.stop_and_wait();
        self.search_mut().set_external_eval_cache(cache);
    }

    pub fn clear_search(&mut self) {
        self.stop_and_wait();
        self.search_mut().clear_search();
    }

    pub fn clear_eval_cache(&mut self) {
        self.stop_and_wait();
        if let Some(cache) = self.search_ref().eval_cache.as_ref() {
            cache.clear();
        }
    }
}

impl<'a> AsyncBot<'a> {
    pub fn make_move(&mut self, move_loc: Loc, move_pla: Player) -> bool {
        self.make_move_with_prevent(move_loc, move_pla, false)
    }

    pub fn make_move_with_prevent(
        &mut self,
        move_loc: Loc,
        move_pla: Player,
        prevent_encore: bool,
    ) -> bool {
        self.stop_and_wait();
        self.search_mut()
            .make_move_with_prevent(move_loc, move_pla, prevent_encore)
    }

    pub fn is_legal_tolerant(&self, move_loc: Loc, move_pla: Player) -> bool {
        self.search_ref().is_legal_tolerant(move_loc, move_pla)
    }

    pub fn is_legal_strict(&self, move_loc: Loc, move_pla: Player) -> bool {
        self.search_ref().is_legal_strict(move_loc, move_pla)
    }
}

impl<'a> AsyncBot<'a> {
    pub fn gen_move_async(
        &mut self,
        move_pla: Player,
        search_id: i32,
        tc: &TimeControls,
        on_move: Box<dyn Fn(Loc, i32, &Search) + Send + Sync>,
    ) {
        self.gen_move_async_with_factor_and_begun(move_pla, search_id, tc, 1.0, on_move, None);
    }

    pub fn gen_move_async_with_factor(
        &mut self,
        move_pla: Player,
        search_id: i32,
        tc: &TimeControls,
        search_factor: f64,
        on_move: Box<dyn Fn(Loc, i32, &Search) + Send + Sync>,
    ) {
        self.gen_move_async_with_factor_and_begun(
            move_pla,
            search_id,
            tc,
            search_factor,
            on_move,
            None,
        );
    }

    pub fn gen_move_async_with_factor_and_begun(
        &mut self,
        move_pla: Player,
        search_id: i32,
        tc: &TimeControls,
        search_factor: f64,
        on_move: Box<dyn Fn(Loc, i32, &Search) + Send + Sync>,
        on_search_begun: Option<Box<dyn Fn() + Send + Sync>>,
    ) {
        let inner = Arc::clone(&self.inner);
        let mut guard = inner.control.lock();
        Self::stop_and_wait_already_locked(&inner, &mut guard);
        if guard.is_killed {
            return;
        }
        if move_pla != self.search_ref().root_pla {
            self.search_mut().set_player_and_clear_history(move_pla);
        }
        guard.queued_search_id = search_id;
        guard.queued_on_move = Some(on_move);
        guard.is_running = true;
        guard.is_pondering = false;
        inner.should_stop_now.store(false, Ordering::Release);
        guard.time_controls = tc.clone();
        guard.search_factor = search_factor;
        guard.analyze_callback_period = -1.0;
        guard.analyze_first_callback_after = -1.0;
        guard.analyze_callback = None;
        guard.search_begun_callback = on_search_begun;
        drop(guard);
        inner.thread_waiting_to_search.notify_all();
    }

    pub fn gen_move_synchronous(&mut self, move_pla: Player, tc: &TimeControls) -> Loc {
        self.gen_move_synchronous_with_factor_and_begun(move_pla, tc, 1.0, None)
    }

    pub fn gen_move_synchronous_with_factor(
        &mut self,
        move_pla: Player,
        tc: &TimeControls,
        search_factor: f64,
    ) -> Loc {
        self.gen_move_synchronous_with_factor_and_begun(move_pla, tc, search_factor, None)
    }

    pub fn gen_move_synchronous_with_factor_and_begun(
        &mut self,
        move_pla: Player,
        tc: &TimeControls,
        search_factor: f64,
        on_search_begun: Option<Box<dyn Fn() + Send + Sync>>,
    ) -> Loc {
        let move_loc = Arc::new(AtomicI16::new(NULL_LOC));
        let move_loc_clone = Arc::clone(&move_loc);
        let on_move = Box::new(move |loc: Loc, search_id: i32, _search: &Search| {
            assert_eq!(search_id, 0);
            move_loc_clone.store(loc, Ordering::Relaxed);
        });
        self.gen_move_async_with_factor_and_begun(
            move_pla,
            0,
            tc,
            search_factor,
            on_move,
            on_search_begun,
        );
        self.wait_for_search_to_end();
        move_loc.load(Ordering::Relaxed)
    }

    pub fn ponder(&mut self) {
        self.ponder_with_factor(1.0);
    }

    pub fn ponder_with_factor(&mut self, search_factor: f64) {
        let inner = Arc::clone(&self.inner);
        let mut guard = inner.control.lock();
        if guard.is_running || guard.is_killed {
            return;
        }
        guard.queued_search_id = 0;
        guard.queued_on_move = None;
        guard.is_running = true;
        guard.is_pondering = true;
        inner.should_stop_now.store(false, Ordering::Release);
        guard.time_controls = TimeControls::new();
        guard.search_factor = search_factor;
        guard.analyze_callback_period = -1.0;
        guard.analyze_first_callback_after = -1.0;
        guard.analyze_callback = None;
        guard.search_begun_callback = None;
        drop(guard);
        inner.thread_waiting_to_search.notify_all();
    }

    pub fn analyze_async(
        &mut self,
        move_pla: Player,
        search_factor: f64,
        callback_period: f64,
        first_callback_after: f64,
        callback: Box<dyn Fn(&Search) + Send + Sync>,
    ) {
        let inner = Arc::clone(&self.inner);
        let mut guard = inner.control.lock();
        Self::stop_and_wait_already_locked(&inner, &mut guard);
        if guard.is_killed {
            return;
        }
        if move_pla != self.search_ref().root_pla {
            self.search_mut().set_player_and_clear_history(move_pla);
        }
        guard.queued_search_id = 0;
        guard.queued_on_move = None;
        guard.is_running = true;
        guard.is_pondering = false;
        inner.should_stop_now.store(false, Ordering::Release);
        guard.time_controls = TimeControls::new();
        guard.search_factor = search_factor;
        guard.analyze_callback_period = callback_period;
        guard.analyze_first_callback_after = first_callback_after;
        guard.analyze_callback = Some(callback);
        guard.search_begun_callback = None;
        drop(guard);
        inner.thread_waiting_to_search.notify_all();
    }

    pub fn gen_move_async_analyze(
        &mut self,
        move_pla: Player,
        search_id: i32,
        tc: &TimeControls,
        search_factor: f64,
        on_move: Box<dyn Fn(Loc, i32, &Search) + Send + Sync>,
        callback_period: f64,
        first_callback_after: f64,
        callback: Box<dyn Fn(&Search) + Send + Sync>,
    ) {
        self.gen_move_async_analyze_with_begun(
            move_pla,
            search_id,
            tc,
            search_factor,
            on_move,
            callback_period,
            first_callback_after,
            callback,
            None,
        );
    }

    pub fn gen_move_async_analyze_with_begun(
        &mut self,
        move_pla: Player,
        search_id: i32,
        tc: &TimeControls,
        search_factor: f64,
        on_move: Box<dyn Fn(Loc, i32, &Search) + Send + Sync>,
        callback_period: f64,
        first_callback_after: f64,
        callback: Box<dyn Fn(&Search) + Send + Sync>,
        on_search_begun: Option<Box<dyn Fn() + Send + Sync>>,
    ) {
        let inner = Arc::clone(&self.inner);
        let mut guard = inner.control.lock();
        Self::stop_and_wait_already_locked(&inner, &mut guard);
        if guard.is_killed {
            return;
        }
        if move_pla != self.search_ref().root_pla {
            self.search_mut().set_player_and_clear_history(move_pla);
        }
        guard.queued_search_id = search_id;
        guard.queued_on_move = Some(on_move);
        guard.is_running = true;
        guard.is_pondering = false;
        inner.should_stop_now.store(false, Ordering::Release);
        guard.time_controls = tc.clone();
        guard.search_factor = search_factor;
        guard.analyze_callback_period = callback_period;
        guard.analyze_first_callback_after = first_callback_after;
        guard.analyze_callback = Some(callback);
        guard.search_begun_callback = on_search_begun;
        drop(guard);
        inner.thread_waiting_to_search.notify_all();
    }

    pub fn gen_move_synchronous_analyze(
        &mut self,
        move_pla: Player,
        tc: &TimeControls,
        search_factor: f64,
        callback_period: f64,
        first_callback_after: f64,
        callback: Box<dyn Fn(&Search) + Send + Sync>,
    ) -> Loc {
        self.gen_move_synchronous_analyze_with_begun(
            move_pla,
            tc,
            search_factor,
            callback_period,
            first_callback_after,
            callback,
            None,
        )
    }

    pub fn gen_move_synchronous_analyze_with_begun(
        &mut self,
        move_pla: Player,
        tc: &TimeControls,
        search_factor: f64,
        callback_period: f64,
        first_callback_after: f64,
        callback: Box<dyn Fn(&Search) + Send + Sync>,
        on_search_begun: Option<Box<dyn Fn() + Send + Sync>>,
    ) -> Loc {
        let move_loc = Arc::new(AtomicI16::new(NULL_LOC));
        let move_loc_clone = Arc::clone(&move_loc);
        let on_move = Box::new(move |loc: Loc, search_id: i32, _search: &Search| {
            assert_eq!(search_id, 0);
            move_loc_clone.store(loc, Ordering::Relaxed);
        });
        self.gen_move_async_analyze_with_begun(
            move_pla,
            0,
            tc,
            search_factor,
            on_move,
            callback_period,
            first_callback_after,
            callback,
            on_search_begun,
        );
        self.wait_for_search_to_end();
        move_loc.load(Ordering::Relaxed)
    }
}

impl<'a> AsyncBot<'a> {
    pub fn wait_for_search_to_end(&self) {
        let mut guard = self.inner.control.lock();
        while guard.is_running {
            self.inner.user_waiting_for_stop.wait(&mut guard);
        }
    }

    pub fn stop_and_wait(&mut self) {
        self.inner.should_stop_now.store(true, Ordering::Relaxed);
        self.wait_for_search_to_end();
    }

    pub fn stop_without_wait(&self) {
        self.inner.should_stop_now.store(true, Ordering::Relaxed);
    }

    pub fn set_killed(&mut self) {
        {
            let mut guard = self.inner.control.lock();
            guard.is_killed = true;
            self.inner.should_stop_now.store(true, Ordering::Relaxed);
        }
        self.inner.thread_waiting_to_search.notify_all();
    }

    fn stop_and_wait_already_locked(
        inner: &Arc<AsyncBotInner<'a>>,
        guard: &mut parking_lot::MutexGuard<ControlState>,
    ) {
        inner.should_stop_now.store(true, Ordering::Relaxed);
        while guard.is_running {
            inner.user_waiting_for_stop.wait(guard);
        }
    }
}

impl<'a> Drop for AsyncBot<'a> {
    fn drop(&mut self) {
        self.stop_and_wait();
        self.set_killed();
        if let Some(handle) = self.search_thread.take() {
            let _ = handle.join();
        }
    }
}

impl<'a> AsyncBot<'a> {
    /// Internal search driver; exposed only because the C++ header exposes it.
    ///
    /// # Safety
    /// This runs the search loop on the current thread. The background thread
    /// spawned by this bot must not be active concurrently.
    pub unsafe fn internal_search_thread_loop(&mut self) {
        let raw = Arc::into_raw(Arc::clone(&self.inner));
        unsafe { AsyncBotInner::search_thread_loop(raw as *mut _) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kata_core::config::ConfigParser;
    use kata_core::logger::{Logger, LoggerOptions};
    use kata_game::board::{Board, C_EMPTY, P_BLACK, P_WHITE, PASS_LOC, location};
    use kata_game::history::BoardHistory;
    use kata_game::rules::Rules;
    use kata_nn::backend::Enabled;
    use std::sync::Arc;

    fn test_logger() -> Logger {
        Logger::new(LoggerOptions::default(), None)
    }

    fn dummy_evaluator() -> NnEvaluator {
        NnEvaluator::new(
            "dummy".to_string(),
            "dummy.bin".to_string(),
            String::new(),
            Arc::new(test_logger()),
            1,
            19,
            19,
            false,
            false,
            -1,
            0,
            true,
            String::new(),
            Enabled::False,
            0,
            Vec::new(),
            "seed".to_string(),
            false,
            0,
            true,
            &ConfigParser::new(false, false),
        )
    }

    fn async_bot_with_dummy() -> AsyncBot<'static> {
        let eval: &'static NnEvaluator = Box::leak(Box::new(dummy_evaluator()));
        let logger: &'static Logger = Box::leak(Box::new(test_logger()));
        AsyncBot::new(SearchParams::new(), eval, logger, "test-seed")
    }

    #[test]
    fn test_async_bot_construct_and_getters() {
        let bot = async_bot_with_dummy();
        assert_eq!(bot.get_root_pla(), C_EMPTY);
        assert_eq!(bot.get_root_board().x_size, 19);
        assert_eq!(bot.get_root_hist().initial_pla, C_EMPTY);
        assert_eq!(bot.get_params().num_threads, 1);
        assert_eq!(bot.get_playout_doubling_advantage_pla(), C_EMPTY);
        assert!(bot.get_search().nn_evaluator.is_some());
    }

    #[test]
    fn test_set_position_and_clear() {
        let mut bot = async_bot_with_dummy();
        let board = Board::new(9, 9);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        bot.set_position(P_BLACK, &board, &hist);
        assert_eq!(bot.get_root_pla(), P_BLACK);
        assert_eq!(bot.get_root_board().x_size, 9);
        assert_eq!(bot.get_root_hist().initial_pla, P_BLACK);

        bot.clear_search();
        assert_eq!(bot.get_root_pla(), P_BLACK);
        assert_eq!(bot.get_search().get_root_visits(), 0);
    }

    #[test]
    fn test_make_move_and_is_legal_tolerant() {
        let mut bot = async_bot_with_dummy();
        let loc = location::get_loc(3, 3, bot.get_root_board().x_size);
        assert!(bot.is_legal_tolerant(loc, P_BLACK));

        // (3,3) is a legal empty intersection, so the move is accepted.
        assert!(bot.make_move(loc, P_BLACK));

        // The same intersection is now occupied, so further moves are illegal.
        assert!(!bot.make_move_with_prevent(loc, P_WHITE, false));
        assert!(!bot.make_move(loc, P_BLACK));
    }

    #[test]
    fn test_is_legal_strict() {
        let mut bot = async_bot_with_dummy();
        let loc = location::get_loc(3, 3, bot.get_root_board().x_size);

        // With no player set, the presumed next player is C_EMPTY, so Black's move
        // is not strictly legal yet.
        assert!(!bot.is_legal_strict(loc, P_BLACK));

        bot.set_player_and_clear_history(P_BLACK);
        assert!(bot.is_legal_strict(loc, P_BLACK));
        assert!(!bot.is_legal_strict(loc, C_EMPTY));
    }

    #[test]
    fn test_set_player_and_params() {
        let mut bot = async_bot_with_dummy();
        bot.set_player_and_clear_history(P_BLACK);
        assert_eq!(bot.get_root_pla(), P_BLACK);

        let mut params = SearchParams::new();
        params.num_threads = 4;
        bot.set_params(&params);
        assert_eq!(bot.get_params().num_threads, 4);

        params.num_threads = 8;
        bot.set_params_no_clearing(&params);
        assert_eq!(bot.get_params().num_threads, 8);
    }

    #[test]
    fn test_get_search_stop_and_wait() {
        let mut bot = async_bot_with_dummy();
        let search = bot.get_search_stop_and_wait();
        assert_eq!(search.get_root_pla(), C_EMPTY);
    }

    #[test]
    fn test_wait_and_stop_when_idle() {
        let mut bot = async_bot_with_dummy();
        bot.wait_for_search_to_end();
        bot.stop_and_wait();
        assert_eq!(bot.get_search().get_root_pla(), C_EMPTY);
    }

    #[test]
    fn test_killed_bot_rejects_new_searches() {
        let mut bot = async_bot_with_dummy();
        bot.set_killed();

        let tc = TimeControls::new();
        // A killed bot should not start a real search; synchronous genMove returns
        // NULL immediately because genMoveAsync sees isKilled under the lock.
        let loc = bot.gen_move_synchronous(P_BLACK, &tc);
        assert_eq!(loc, NULL_LOC);
    }

    #[test]
    fn test_gen_move_synchronous_runs_search() {
        let eval: &'static NnEvaluator = Box::leak(Box::new(dummy_evaluator()));
        let logger: &'static Logger = Box::leak(Box::new(test_logger()));
        let mut params = SearchParams::new();
        params.max_visits = 10;
        params.max_playouts = 10;
        params.num_threads = 1;
        params.value_weight_exponent = 0.0;
        let mut bot = AsyncBot::new(params, eval, logger, "test-seed");

        let board = Board::new(9, 9);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        bot.set_position(P_BLACK, &board, &hist);

        let tc = TimeControls::new();
        let loc = bot.gen_move_synchronous(P_BLACK, &tc);

        // With the dummy uniform evaluator the chosen move should be legal or pass.
        assert!(loc == PASS_LOC || bot.is_legal_tolerant(loc, P_BLACK));
    }
}

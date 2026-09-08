//! Engine bridge: runs a KataGo `AsyncBot` on a dedicated thread.
//!
//! The UI owns a `Sender<EngineCmd>`; the engine thread owns the bot and
//! streams back `Snap` snapshots over an mpsc channel (polled once per UI
//! frame — the shell already repaints at ~30fps for the breathing dot).
//!
//! Flow: UI click -> `Play{c,r}` -> `is_legal_tolerant` -> `make_move` ->
//! `Snap::Board` -> white to move -> `gen_move_async_analyze` (analysis
//! streams during the reply search) -> on_move -> `make_move` -> `Snap::Board`
//! -> `analyze_async` again (always-on analysis of the latest position).

use std::path::Path;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::Duration;

use kata_core::config::ConfigParser;
use kata_core::logger::{Logger, LoggerOptions};
use kata_core::rng::Rand;
use kata_game::board::{Board, Loc, NULL_LOC, P_BLACK, P_WHITE, PASS_LOC, location};
use kata_game::history::BoardHistory;
use kata_nn::eval::NnEvaluator;
use kata_nn::inputs::nn_pos;
use kata_program::setup::{self, SetupFor};
use kata_search::async_bot::AsyncBot;
use kata_search::params::SearchParams;
use kata_search::search::Search;
use kata_search::time_control::TimeControls;

/// Board edge; the shell only supports 19x19.
pub const N: usize = 19;

/// Visits cap for the engine's reply moves (analysis itself runs unbounded).
const GENMOVE_VISITS: i64 = 300;
/// Analysis snapshot period (seconds).
const ANALYSIS_PERIOD: f64 = 0.25;

/// Commands from the UI to the engine thread.
pub enum EngineCmd {
    /// User (black) plays at column/row.
    Play { c: u8, r: u8 },
    /// User (black) passes. No shell control yet; exercised by the pipeline
    /// test (dual-stop detection belongs to the engine's friendly-pass path).
    #[allow(dead_code)]
    Pass,
    /// Reset to an empty board. Reserved: no shell control sends it yet;
    /// exercised by the engine pipeline test.
    #[allow(dead_code)]
    NewGame,
    /// Shut the engine thread down. The process exit drops the sender and
    /// the loop returns on disconnect; used explicitly by tests.
    #[allow(dead_code)]
    Quit,
}

/// One candidate move in an analysis snapshot (winrate in percent, black's
/// perspective, matching the right-column display).
pub struct CandSnap {
    pub c: u8,
    pub r: u8,
    pub visits: i64,
    pub winrate: f32,
}

/// Snapshots from the engine thread to the UI.
pub enum Snap {
    /// Model loaded, bot ready. `label` is e.g. "cudabackend · b11fix".
    Ready { label: String },
    /// Board state after every position change. `stones[r][c]`: 0 empty,
    /// 1 black, 2 white. `move_no` counts moves played (0 = empty board).
    Board {
        stones: [[u8; N]; N],
        black_to_move: bool,
        last_move: Option<(u8, u8)>,
        move_no: usize,
    },
    /// Periodic analysis frame. `move_no` tags the position it belongs to;
    /// stale frames (superseded by a newer Board) are dropped by the UI.
    Analysis {
        move_no: usize,
        candidates: Vec<CandSnap>,
        /// White-positive ownership, [r][c], in [-1, 1].
        ownership: Box<[[f32; N]; N]>,
        /// Root winrate in percent, black's perspective.
        root_wr: f32,
        /// Root score lead in points, black's perspective.
        lead: f32,
        total_visits: i64,
    },
    /// Fatal engine error (model load failure etc.).
    Error(String),
}

/// Which NN backend the engine should use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendKind {
    Dummy,
    Cuda,
}

impl BackendKind {
    pub fn cfg_key(self) -> &'static str {
        match self {
            Self::Dummy => "dummybackend",
            Self::Cuda => "cudabackend",
        }
    }

    /// Index into the Settings「器」backend rows (0 dummy / 1 TRT / 2 CUDA).
    pub fn settings_row(self) -> usize {
        match self {
            Self::Dummy => 0,
            Self::Cuda => 2,
        }
    }
}

/// Engine startup configuration (CLI/env parsed by main.rs).
pub struct EngineConfig {
    pub model: String,
    pub backend: BackendKind,
    pub threads: i32,
}

/// Spawn the engine thread. Returns the command sender and snapshot receiver.
pub fn spawn(cfg: EngineConfig) -> (Sender<EngineCmd>, Receiver<Snap>) {
    let (cmd_tx, cmd_rx) = channel::<EngineCmd>();
    let (snap_tx, snap_rx) = channel::<Snap>();
    std::thread::Builder::new()
        .name("xuanping-engine".into())
        .spawn(move || run(cfg, cmd_rx, snap_tx))
        .expect("spawn engine thread");
    (cmd_tx, snap_rx)
}

fn run(cfg: EngineConfig, cmd_rx: Receiver<EngineCmd>, snap_tx: Sender<Snap>) {
    let send_err = |msg: String| {
        let _ = snap_tx.send(Snap::Error(msg));
    };

    // Config string mirrors configs/gtp_cuda.cfg (minus the tactic plan).
    let cfg_str = format!(
        "rules = chinese\nkomi = 7.5\nnumSearchThreads = {}\nnnBackend = {}\n\
         nnMaxBatchSize = 16\nnnCacheSizePowerOfTwo = 18\nnnMutexPoolSizePowerOfTwo = 14\n",
        cfg.threads,
        cfg.backend.cfg_key()
    );
    let kcfg = match ConfigParser::from_str(&cfg_str, false, false) {
        Ok(c) => c,
        Err(e) => return send_err(format!("config: {e}")),
    };

    let logger: &'static Logger = Box::leak(Box::new(Logger::new(
        LoggerOptions {
            log_to_stdout: false,
            log_to_stderr: true,
            log_time: false,
        },
        None,
    )));

    // The dummy backend needs no model file ("/dev/null" marks
    // debugSkipNeuralNet, mirroring the GTP test construction).
    let model_file = if cfg.backend == BackendKind::Dummy {
        "/dev/null".to_string()
    } else {
        cfg.model.clone()
    };
    let eval = setup::initialize_nn_evaluator(
        "xuanping".to_string(),
        model_file,
        String::new(),
        &kcfg,
        logger,
        &mut Rand::new(),
        cfg.threads,
        N as i32,
        N as i32,
        16,
        true,
        false,
        SetupFor::Gtp,
    );
    let eval: &'static NnEvaluator = Box::leak(Box::new(match eval {
        Ok(e) => e,
        Err(e) => return send_err(format!("model: {}", e.message)),
    }));

    let base_params = match setup::load_single_params_with_human(&kcfg, SetupFor::Gtp, false) {
        Ok(p) => p,
        Err(e) => return send_err(format!("params: {}", e.message)),
    };
    let rules = match setup::load_single_rules(&kcfg, false) {
        Ok(r) => r,
        Err(e) => return send_err(format!("rules: {}", e.message)),
    };

    // GTP defaults (load_genmove_and_analysis_params): friendly pass handling.
    let mut analysis_params = base_params.clone();
    analysis_params.conservative_pass = true;
    analysis_params.fill_dame_before_pass = true;
    // Replies are visit-capped so the GUI stays snappy; analysis is unbounded.
    let mut genmove_params = analysis_params.clone();
    genmove_params.max_visits = GENMOVE_VISITS;
    genmove_params.max_playouts = GENMOVE_VISITS;

    let mut bot = AsyncBot::new_with_human(genmove_params.clone(), eval, None, logger, "xuanping");
    bot.set_always_include_owner_map(true);

    // Genmove completions arrive here from the bot's search thread.
    let (gen_tx, gen_rx) = channel::<Loc>();

    let mut moves: Vec<Loc> = Vec::new();
    reset_position(&mut bot, &rules);
    let label = engine_label(&cfg);
    if snap_tx.send(Snap::Ready { label }).is_err() {
        return;
    }
    if !send_board(&bot, &moves, &snap_tx) {
        return;
    }
    start_analysis(&mut bot, &analysis_params, moves.len(), &snap_tx);

    loop {
        // A finished reply (if any) is applied before the next command.
        while let Ok(loc) = gen_rx.try_recv() {
            let loc = if loc == NULL_LOC { PASS_LOC } else { loc };
            bot.make_move(loc, P_WHITE);
            moves.push(loc);
            if !send_board(&bot, &moves, &snap_tx) {
                return;
            }
            start_analysis(&mut bot, &analysis_params, moves.len(), &snap_tx);
        }

        match cmd_rx.recv_timeout(Duration::from_millis(25)) {
            Ok(EngineCmd::Play { c, r }) => {
                let loc = location::get_loc(c as i32, r as i32, N as i32);
                if bot.get_root_pla() != P_BLACK || !bot.is_legal_tolerant(loc, P_BLACK) {
                    continue;
                }
                bot.make_move(loc, P_BLACK);
                moves.push(loc);
                if !send_board(&bot, &moves, &snap_tx) {
                    return;
                }
                start_genmove(&mut bot, &genmove_params, moves.len(), &gen_tx, &snap_tx);
            }
            Ok(EngineCmd::Pass) => {
                if bot.get_root_pla() != P_BLACK {
                    continue;
                }
                bot.make_move(PASS_LOC, P_BLACK);
                moves.push(PASS_LOC);
                if !send_board(&bot, &moves, &snap_tx) {
                    return;
                }
                start_genmove(&mut bot, &genmove_params, moves.len(), &gen_tx, &snap_tx);
            }
            Ok(EngineCmd::NewGame) => {
                moves.clear();
                // set_position stops a running reply first; its on_move then
                // fires synchronously before is_running clears, so the stale
                // completion is already queued — drain it.
                reset_position(&mut bot, &rules);
                while gen_rx.try_recv().is_ok() {}
                if !send_board(&bot, &moves, &snap_tx) {
                    return;
                }
                start_analysis(&mut bot, &analysis_params, moves.len(), &snap_tx);
            }
            Ok(EngineCmd::Quit) | Err(RecvTimeoutError::Disconnected) => return,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}

fn reset_position(bot: &mut AsyncBot, rules: &kata_game::rules::Rules) {
    let board = Board::new(N as i32, N as i32);
    let hist = BoardHistory::new(board.clone(), P_BLACK, rules.clone(), 0);
    bot.set_position(P_BLACK, &board, &hist);
}

/// Build the status-line label, e.g. "cudabackend · b11fix".
fn engine_label(cfg: &EngineConfig) -> String {
    let stem = if cfg.backend == BackendKind::Dummy {
        "dummy".to_string()
    } else {
        Path::new(&cfg.model)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| cfg.model.clone())
    };
    format!("{} · {}", cfg.backend.cfg_key(), stem)
}

/// Extract the board from the bot and send a `Snap::Board`.
fn send_board(bot: &AsyncBot, moves: &[Loc], snap_tx: &Sender<Snap>) -> bool {
    let board = bot.get_root_board();
    let mut stones = [[0u8; N]; N];
    for r in 0..N {
        for c in 0..N {
            let loc = location::get_loc(c as i32, r as i32, N as i32);
            let color = board.colors[loc as usize];
            stones[r][c] = if (0..=2).contains(&color) { color as u8 } else { 0 };
        }
    }
    let last_move = moves.last().and_then(|&loc| {
        if loc == PASS_LOC || loc == NULL_LOC {
            None
        } else {
            Some((
                location::get_x(loc, N as i32) as u8,
                location::get_y(loc, N as i32) as u8,
            ))
        }
    });
    snap_tx
        .send(Snap::Board {
            stones,
            black_to_move: bot.get_root_pla() == P_BLACK,
            last_move,
            move_no: moves.len(),
        })
        .is_ok()
}

/// Always-on analysis of the current position (search factor 1e40 = until
/// superseded), streaming `Snap::Analysis` every ANALYSIS_PERIOD seconds.
fn start_analysis(
    bot: &mut AsyncBot,
    analysis_params: &SearchParams,
    move_no: usize,
    snap_tx: &Sender<Snap>,
) {
    bot.set_params(analysis_params);
    let tx = snap_tx.clone();
    let cb: Box<dyn Fn(&Search) + Send + Sync> = Box::new(move |search: &Search| {
        if let Some(snap) = extract_analysis(search, move_no) {
            let _ = tx.send(snap);
        }
    });
    let pla = bot.get_root_pla();
    bot.analyze_async(pla, 1e40, ANALYSIS_PERIOD, ANALYSIS_PERIOD, cb);
}

/// Reply search for white: capped by genmove visit limits, analysis frames
/// stream while it runs, completion is reported over `gen_tx`.
fn start_genmove(
    bot: &mut AsyncBot,
    genmove_params: &SearchParams,
    move_no: usize,
    gen_tx: &Sender<Loc>,
    snap_tx: &Sender<Snap>,
) {
    bot.set_params(genmove_params);
    let done_tx = gen_tx.clone();
    let on_move: Box<dyn Fn(Loc, i32, &Search) + Send + Sync> =
        Box::new(move |loc: Loc, _id: i32, _search: &Search| {
            let _ = done_tx.send(loc);
        });
    let tx = snap_tx.clone();
    let cb: Box<dyn Fn(&Search) + Send + Sync> = Box::new(move |search: &Search| {
        if let Some(snap) = extract_analysis(search, move_no) {
            let _ = tx.send(snap);
        }
    });
    bot.gen_move_async_analyze(
        P_WHITE,
        0,
        &TimeControls::new(),
        1.0,
        on_move,
        ANALYSIS_PERIOD,
        ANALYSIS_PERIOD,
        cb,
    );
}

/// Pull one analysis frame out of a running search. Values are converted to
/// black's perspective (the UI curve/rows are black-centric); ownership is
/// white-positive, matching the shell's wash convention.
fn extract_analysis(search: &Search, move_no: usize) -> Option<Snap> {
    let mut buf = Vec::new();
    search.get_analysis_data(&mut buf, 3, false, 13, false);
    buf.retain(|d| d.child_visits > 0);
    if buf.is_empty() {
        return None;
    }
    let values = search.get_root_values_require_success();
    let flip = search.root_pla == P_WHITE;

    let mut candidates = Vec::new();
    for d in &buf {
        if d.move_loc == PASS_LOC || d.move_loc == NULL_LOC {
            continue;
        }
        let mut wr = 0.5 * (1.0 + d.win_loss_value);
        if flip {
            wr = 1.0 - wr;
        }
        candidates.push(CandSnap {
            c: location::get_x(d.move_loc, N as i32) as u8,
            r: location::get_y(d.move_loc, N as i32) as u8,
            visits: d.child_visits,
            winrate: (wr * 100.0) as f32,
        });
        if candidates.len() == 3 {
            break;
        }
    }

    let mut root_wr = 0.5 * (1.0 + values.win_loss_value);
    let mut lead = values.lead;
    if flip {
        root_wr = 1.0 - root_wr;
        lead = -lead;
    }

    let mut ownership = Box::new([[0.0f32; N]; N]);
    let raw = search.get_average_tree_ownership(None);
    if !raw.is_empty() {
        for r in 0..N {
            for c in 0..N {
                let pos = nn_pos::xy_to_pos(c as i32, r as i32, search.nn_x_len) as usize;
                ownership[r][c] = raw.get(pos).copied().unwrap_or(0.0) as f32;
            }
        }
    }

    Some(Snap::Analysis {
        move_no,
        candidates,
        ownership,
        root_wr: (root_wr * 100.0) as f32,
        lead: lead as f32,
        total_visits: values.visits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recv_until<T>(rx: &Receiver<Snap>, timeout: Duration, mut f: impl FnMut(&Snap) -> Option<T>) -> T {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            assert!(!left.is_zero(), "timed out waiting for engine snapshot");
            match rx.recv_timeout(left) {
                Ok(snap) => {
                    if let Some(v) = f(&snap) {
                        return v;
                    }
                }
                Err(e) => panic!("engine channel closed while waiting: {e}"),
            }
        }
    }

    /// Full engine pipeline over the dummy backend (no model file): Ready,
    /// Play -> Board, engine reply -> Board, streaming Analysis, Pass, NewGame.
    #[test]
    fn dummy_engine_pipeline() {
        let (tx, rx) = spawn(EngineConfig {
            model: "/dev/null".into(),
            backend: BackendKind::Dummy,
            threads: 1,
        });

        // Ready.
        recv_until(&rx, Duration::from_secs(30), |s| match s {
            Snap::Ready { label } => Some(label.clone()),
            Snap::Error(e) => panic!("engine error: {e}"),
            _ => None,
        });

        // First board: empty, black to move.
        let (stones, btm, move_no) = recv_until(&rx, Duration::from_secs(30), |s| match s {
            Snap::Board {
                stones,
                black_to_move,
                move_no,
                ..
            } => Some((*stones, *black_to_move, *move_no)),
            _ => None,
        });
        assert_eq!(move_no, 0);
        assert!(btm);
        assert!(stones.iter().flatten().all(|&s| s == 0));

        // Analysis of the empty board streams in.
        let a0 = recv_until(&rx, Duration::from_secs(30), |s| match s {
            Snap::Analysis {
                move_no,
                candidates,
                total_visits,
                ..
            } => Some((*move_no, candidates.len(), *total_visits)),
            _ => None,
        });
        assert_eq!(a0.0, 0);
        assert!(a0.1 >= 1);
        assert!(a0.2 > 0);

        // User plays (3,3): board with the black stone, then the white reply.
        tx.send(EngineCmd::Play { c: 3, r: 3 }).unwrap();
        let (stones, move_no) = recv_until(&rx, Duration::from_secs(30), |s| match s {
            Snap::Board {
                stones, move_no, ..
            } if *move_no == 1 => Some((*stones, *move_no)),
            _ => None,
        });
        assert_eq!(stones[3][3], 1);
        let _ = move_no;
        let move_no = recv_until(&rx, Duration::from_secs(30), |s| match s {
            Snap::Board { move_no, .. } if *move_no == 2 => Some(*move_no),
            _ => None,
        });
        assert_eq!(move_no, 2);

        // Analysis of the post-reply position belongs to ply 2.
        let a2 = recv_until(&rx, Duration::from_secs(30), |s| match s {
            Snap::Analysis {
                move_no,
                candidates,
                ..
            } if *move_no == 2 => Some(candidates.len()),
            _ => None,
        });
        assert!(a2 >= 1);

        // Pass: black passes (ply 3), white replies (ply 4).
        tx.send(EngineCmd::Pass).unwrap();
        let move_no = recv_until(&rx, Duration::from_secs(30), |s| match s {
            Snap::Board { move_no, .. } if *move_no == 4 => Some(*move_no),
            _ => None,
        });
        assert_eq!(move_no, 4);

        // NewGame resets to an empty board.
        tx.send(EngineCmd::NewGame).unwrap();
        let move_no = recv_until(&rx, Duration::from_secs(30), |s| match s {
            Snap::Board { move_no, .. } if *move_no == 0 => Some(*move_no),
            _ => None,
        });
        assert_eq!(move_no, 0);

        tx.send(EngineCmd::Quit).unwrap();
    }
}

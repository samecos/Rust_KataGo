//! Strict protocol context replay and direct, postprocessed NN evaluation.
//!
//! This module never constructs a search tree. Friendly-pass permissions match
//! the reference worker's pre-move predicate; they do not relax move legality.

use std::fs::File;
use std::io::Read;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use kata_core::config::ConfigParser;
use kata_core::logger::{Logger, LoggerOptions};
use kata_core::rng::Rand;
use kata_game::board::{Board, Loc, P_BLACK, P_WHITE, PASS_LOC, Player, get_opp, location};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_nn::backend::NNResultBuf;
use kata_nn::eval::NnEvaluator;
use kata_nn::inputs::{MiscNNInputParams, NNOutput};
use kata_program::setup::{self, SetupFor};
use sha2::{Digest, Sha256};

use crate::{EvalFailure, EvaluationReport, Evaluator, Metadata, wire};

const SIZE: i32 = 19;
const AREA: usize = 361;
const SYNTHETIC_MODEL: &[u8] = b"Rust_KataGo synthetic dummy worker v1";

pub struct Engine {
    evaluator: ConcurrentEvaluator,
    metadata: Metadata,
}

/// A fully initialized evaluator used exclusively through its concurrent client
/// path. Construction, backend/model inspection and warmup precede this wrapper;
/// destruction happens only after the final Engine owner has released it.
struct ConcurrentEvaluator(NnEvaluator);

// SAFETY: NnEvaluator's runtime evaluate path reads immutable scalar settings
// and uses its Arc<SharedState>, atomic counters and synchronized NN cache/queue.
// The non-Sync model/context/backend handle objects are accessed only during
// exclusive initialization or destruction, never by evaluate or these stats
// getters. Keep this wrapper private and do not expose the inner evaluator or
// add concurrent methods that inspect those handles. This has the same bounded
// sharing requirement as the existing SearchPlayoutRef implementation.
unsafe impl Sync for ConcurrentEvaluator {}

impl ConcurrentEvaluator {
    fn evaluate(&self, context: &Prepared, buf: &mut NNResultBuf) {
        self.0.evaluate(
            &context.board,
            &context.history,
            context.next,
            &context.params,
            buf,
            context.skip_cache,
            context.include_ownership,
        );
    }

    fn stats(&self) -> (u64, u64) {
        (self.0.num_rows_processed(), self.0.num_batches_processed())
    }
}

impl Engine {
    /// Load the deployment-selected backend once. Capacity is an application
    /// concurrency hint; batch size and GPU selection remain configuration-owned.
    pub fn load(
        model: &str,
        expected_sha256: Option<&str>,
        cfg: &ConfigParser,
        capacity: u32,
        allow_dummy: bool,
    ) -> Result<Self> {
        if !(1..=4096).contains(&capacity) {
            bail!("capacity must be in [1,4096]");
        }
        let backend = selected_backend(cfg)?;
        match backend {
            "trtbackend" => bail!(
                "Go Server worker currently supports CUDA; TensorRT model-bound cache not yet validated"
            ),
            "eigenbackend" => {
                bail!("Go Server worker currently supports CUDA; eigenbackend is not supported")
            }
            _ => {}
        }
        let debug_skip =
            cfg.contains("debugSkipNeuralNet") && cfg.get_bool("debugSkipNeuralNet")?;
        let synthetic = backend == "dummybackend" || debug_skip || model == "/dev/null";
        if synthetic && !allow_dummy {
            bail!(
                "nnworker requires a real model/backend; synthetic evaluation requires --allow-dummy"
            );
        }
        if allow_dummy && (backend != "dummybackend" || model != "/dev/null") {
            bail!(
                "--allow-dummy is restricted to nnBackend=dummybackend and model=/dev/null; synthetic evaluation cannot advertise a real model identity"
            );
        }
        let model_sha256 = if synthetic {
            hex::encode(Sha256::digest(SYNTHETIC_MODEL))
        } else {
            hash_model_file(model)?
        };
        if let Some(expected) = expected_sha256 {
            if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
                bail!("expected model SHA256 must contain exactly 64 hexadecimal characters");
            }
            if !expected.eq_ignore_ascii_case(&model_sha256) {
                bail!("model SHA256 does not match expected value");
            }
        }
        let logger = Logger::new(
            LoggerOptions {
                log_to_stderr: true,
                ..LoggerOptions::default()
            },
            None,
        );
        setup::initialize_session(cfg);
        let evaluator = setup::initialize_nn_evaluator(
            model.to_owned(),
            model.to_owned(),
            model_sha256.clone(),
            cfg,
            &logger,
            &mut Rand::new(),
            capacity as i32,
            SIZE,
            SIZE,
            -1,
            false,
            false,
            SetupFor::Other,
        )
        .context("initialize NN evaluator")?;
        if evaluator.requires_sgf_metadata() {
            bail!("nnworker does not accept human models requiring SGF metadata");
        }
        if !synthetic && evaluator.is_neural_net_less() {
            bail!("real worker unexpectedly loaded a neural-net-less evaluator");
        }
        if evaluator.nn_x_len() != SIZE || evaluator.nn_y_len() != SIZE {
            bail!("this worker requires an exact 19x19 NN buffer");
        }
        if !synthetic && evaluator.model_version() <= 0 {
            bail!("real evaluator did not report a loaded model version");
        }
        if !synthetic {
            let mut supported = false;
            evaluator.supported_rules(Rules::parse_rules("chinese")?, &mut supported);
            if !supported {
                bail!("loaded model does not exactly support the chinese rules profile");
            }
        }
        let revision =
            option_env!("KATAGO_WORKER_ENGINE_REVISION").unwrap_or("revision-unavailable");
        let engine_commit = format!("Rust_KataGo/{revision}");
        let mut backend_info = format!(
            "Rust_KataGo {} {}; backend={backend}; model={}; model-version={}; history-modes=false,false{}",
            env!("CARGO_PKG_VERSION"),
            revision,
            std::path::Path::new(model)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            evaluator.model_version(),
            if synthetic {
                "; SYNTHETIC TEST ONLY - not real NN inference"
            } else {
                ""
            },
        );
        if backend == "cudaint8backend" {
            let scope = if cfg.contains("cudaInt8Scope") { cfg.get_string("cudaInt8Scope")? } else { "ffn".into() };
            let min_width = if cfg.contains("cudaInt8MinFfnWidth") { cfg.get_string("cudaInt8MinFfnWidth")? } else { "0".into() };
            backend_info.push_str(&format!("; precision=W8A8-mixed; int8-scope={scope}; int8-min-ffn-width={min_width}; quantization=w8a8-row-out-rne-v1"));
        }
        let metadata = Metadata {
            model_sha256,
            model_version: evaluator.model_version().max(0) as u32,
            engine_commit,
            backend_info,
            supports_shortterm_error: !synthetic && evaluator.supports_shortterm_error(),
            default_always_compute_pass_alive: evaluator
                .model_prefer_pass_alive_under_suicide_rules(),
            default_exclude_territory_adjacent_to_atari: evaluator
                .model_prefer_exclude_territory_adjacent_to_atari(),
        };
        Ok(Self {
            evaluator: ConcurrentEvaluator(evaluator),
            metadata,
        })
    }

    fn prepare(&self, request: &wire::EvalRequest) -> Result<Prepared, EvalFailure> {
        if !request
            .model_sha256
            .eq_ignore_ascii_case(&self.metadata.model_sha256)
        {
            return Err(invalid_context(
                "request model_sha256 differs from loaded model",
            ));
        }
        prepare_context(request)
    }
}

impl Evaluator for Engine {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn evaluate(&self, request: &wire::EvalRequest) -> EvaluationReport {
        let context_start = Instant::now();
        let prepared = catch_unwind(AssertUnwindSafe(|| self.prepare(request)));
        let context_us = Some(micros(context_start));
        let prepared = match prepared {
            Ok(Ok(context)) => context,
            Ok(Err(error)) => {
                return EvaluationReport {
                    result: Err(error),
                    context_us,
                    evaluator_us: None,
                };
            }
            Err(panic) => {
                return EvaluationReport {
                    result: Err(EvalFailure::new(
                        "WORKER_INTERNAL_ERROR",
                        panic_message(panic),
                    )),
                    context_us,
                    evaluator_us: None,
                };
            }
        };
        let mut buf = NNResultBuf::new();
        let evaluator_start = Instant::now();
        let evaluated = catch_unwind(AssertUnwindSafe(|| {
            self.evaluator.evaluate(&prepared, &mut buf);
        }));
        let evaluator_us = Some(micros(evaluator_start));
        let result = match evaluated {
            Err(panic) => Err(EvalFailure::new("EVALUATOR_ERROR", panic_message(panic))),
            Ok(()) => match buf.result.as_deref() {
                None => Err(EvalFailure::new(
                    "EVALUATOR_ERROR",
                    "NN evaluator returned no output",
                )),
                Some(nn) => match catch_unwind(AssertUnwindSafe(|| {
                    convert_output(
                        nn,
                        prepared.include_ownership,
                        self.metadata.supports_shortterm_error,
                    )
                })) {
                    Ok(output) => output,
                    Err(panic) => Err(EvalFailure::new("INVALID_OUTPUT", panic_message(panic))),
                },
            },
        };
        EvaluationReport {
            result,
            context_us,
            evaluator_us,
        }
    }

    fn stats(&self) -> (u64, u64) {
        self.evaluator.stats()
    }
}

fn hash_model_file(model: &str) -> Result<String> {
    let mut file = File::open(model).with_context(|| format!("open model {model}"))?;
    let mut hasher = Sha256::new();
    let mut bytes = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut bytes).context("read model for SHA256")?;
        if count == 0 {
            break;
        }
        hasher.update(&bytes[..count]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn selected_backend(cfg: &ConfigParser) -> Result<&'static str> {
    let value = if cfg.contains("nnBackend0") {
        cfg.get_string("nnBackend0")?
    } else if cfg.contains("nnBackend") {
        cfg.get_string("nnBackend")?
    } else {
        "dummybackend".to_owned()
    };
    match value.as_str() {
        "dummy" | "dummybackend" => Ok("dummybackend"),
        "cuda" | "cudabackend" => Ok("cudabackend"),
        "cudaint8" | "cudaint8backend" => Ok("cudaint8backend"),
        "trt" | "tensorrt" | "trtbackend" => Ok("trtbackend"),
        "eigen" | "cpu" | "eigenbackend" => Ok("eigenbackend"),
        _ => bail!("unknown nnBackend: {value}"),
    }
}

struct Prepared {
    board: Board,
    history: BoardHistory,
    next: Player,
    params: MiscNNInputParams,
    skip_cache: bool,
    include_ownership: bool,
}

fn invalid_context(message: impl Into<String>) -> EvalFailure {
    EvalFailure::new("INVALID_CONTEXT", message)
}

fn finite_range(value: f64, min: f64, max: f64, name: &str) -> Result<(), EvalFailure> {
    if !value.is_finite() || !(min..=max).contains(&value) {
        return Err(invalid_context(format!("invalid {name}")));
    }
    Ok(())
}

fn player(color: i32) -> Result<Player, EvalFailure> {
    match color {
        1 => Ok(P_BLACK),
        2 => Ok(P_WHITE),
        _ => Err(invalid_context("color must be BLACK or WHITE")),
    }
}

fn vertex(value: i32) -> Result<Loc, EvalFailure> {
    if value == -1 {
        Ok(PASS_LOC)
    } else if (0..AREA as i32).contains(&value) {
        Ok(location::get_loc(value % SIZE, value / SIZE, SIZE))
    } else {
        Err(invalid_context(
            "vertex must be -1 or a board point in [0,360]",
        ))
    }
}

fn prepare_context(request: &wire::EvalRequest) -> Result<Prepared, EvalFailure> {
    let pos = request
        .position
        .as_ref()
        .ok_or_else(|| invalid_context("missing position"))?;
    let p = request
        .parameters
        .as_ref()
        .ok_or_else(|| invalid_context("missing parameters"))?;
    if pos.board_size != SIZE as u32 || pos.rules != "chinese" {
        return Err(invalid_context(
            "katago-eval-v1 requires board_size=19 and rules=chinese",
        ));
    }
    if pos.moves.len() > 10_000 || pos.initial_stones.len() > AREA {
        return Err(invalid_context("position context too large"));
    }
    finite_range(pos.komi, -150.0, 150.0, "komi")?;
    if (pos.komi * 2.0).round() != pos.komi * 2.0 {
        return Err(invalid_context("komi must be an integer or half-integer"));
    }
    finite_range(p.policy_temperature, 0.001, 100.0, "policy_temperature")?;
    finite_range(p.policy_optimism, 0.0, 1.0, "policy_optimism")?;
    finite_range(
        p.draw_equivalent_wins_for_white,
        0.0,
        1.0,
        "draw_equivalent_wins_for_white",
    )?;
    finite_range(
        p.playout_doubling_advantage,
        -100.0,
        100.0,
        "playout_doubling_advantage",
    )?;
    if !(0..=7).contains(&p.symmetry) || p.max_history > 10_000 {
        return Err(invalid_context("invalid symmetry or max_history"));
    }
    // The current Rust history implementation exposes only these modes. Reject
    // unsupported semantics explicitly instead of silently ignoring the flags.
    if p.always_compute_pass_alive || p.exclude_territory_adjacent_to_atari {
        return Err(invalid_context(
            "this Rust worker supports only false/false history modes",
        ));
    }
    let mut board = Board::new(SIZE, SIZE);
    let mut occupied = [false; AREA];
    for stone in &pos.initial_stones {
        let index = stone.vertex as usize;
        if index >= AREA || occupied[index] {
            return Err(invalid_context("invalid or duplicate initial stone"));
        }
        occupied[index] = true;
        let loc = vertex(stone.vertex as i32)?;
        board.colors[loc as usize] = player(stone.color)?;
    }
    // Setup stones are simultaneous, so their input order cannot change whether
    // a group is accepted. Rebuild chains before checking final group liberties.
    board.regen_chains_from_colors();
    for stone in &pos.initial_stones {
        if board.get_num_liberties(vertex(stone.vertex as i32)?) <= 0 {
            return Err(invalid_context(
                "initial stones contain a chain without liberties",
            ));
        }
    }
    let mut rules = Rules::parse_rules("chinese")
        .map_err(|e| EvalFailure::new("WORKER_INTERNAL_ERROR", e.to_string()))?;
    rules.set_komi(pos.komi as f32);
    let mut next = player(pos.initial_player)?;
    let mut history = BoardHistory::new(board.clone(), next, rules, 0);
    history.set_assume_multiple_starting_black_moves_are_handicap(false);
    let mut last_force_friendly = false;
    for (index, movement) in pos.moves.iter().enumerate() {
        if history.is_game_finished && !(p.allow_terminal_search_history && last_force_friendly) {
            return Err(invalid_context(format!(
                "move {index} follows a terminal position without valid friendly-pass search history"
            )));
        }
        let pla = player(movement.color)?;
        let loc = vertex(movement.vertex)?;
        if pla != next || !history.is_legal(&board, loc, pla) {
            return Err(invalid_context(format!(
                "illegal or out-of-turn move at index {index}"
            )));
        }
        let can_force_friendly =
            loc == PASS_LOC && history.should_suppress_end_game_from_friendly_pass(&board, pla);
        history.make_board_move_assume_legal(&mut board, loc, pla);
        last_force_friendly =
            can_force_friendly && history.is_game_finished && !history.is_no_result;
        next = get_opp(next);
    }
    if next != player(pos.next_player)? || next != history.presumed_next_move_pla {
        return Err(invalid_context("next_player does not match replay"));
    }
    if p.force_non_terminal && !(history.is_game_finished && last_force_friendly) {
        return Err(invalid_context(
            "force_non_terminal requires an ordinary second friendly pass",
        ));
    }
    if history.is_game_finished && !(p.force_non_terminal && last_force_friendly) {
        return Err(invalid_context(
            "terminal positions must be evaluated by service rules",
        ));
    }
    Ok(Prepared {
        board,
        history,
        next,
        params: MiscNNInputParams {
            symmetry: p.symmetry,
            nn_policy_temperature: p.policy_temperature as f32,
            policy_optimism: p.policy_optimism,
            draw_equivalent_wins_for_white: p.draw_equivalent_wins_for_white,
            playout_doubling_advantage: p.playout_doubling_advantage,
            max_history: p.max_history as i32,
            conservative_pass_and_is_root: p.conservative_pass,
            enable_passing_hacks: p.enable_passing_hacks,
            avoid_mytdagger_hack: p.avoid_mytdagger_hack,
        },
        skip_cache: p.skip_cache,
        include_ownership: p.include_ownership,
    })
}

fn convert_output(
    nn: &NNOutput,
    include_ownership: bool,
    has_shortterm_error: bool,
) -> Result<wire::NnOutput, EvalFailure> {
    let invalid = |message| EvalFailure::new("INVALID_OUTPUT", message);
    if nn.nn_x_len != SIZE || nn.nn_y_len != SIZE {
        return Err(invalid("NN output dimensions must be 19x19"));
    }
    let ownership = if include_ownership {
        let owner = nn
            .white_owner_map
            .as_ref()
            .ok_or_else(|| invalid("NN evaluator omitted requested ownership"))?;
        if owner.len() != AREA {
            return Err(invalid("invalid ownership shape"));
        }
        owner.to_vec()
    } else {
        Vec::new()
    };
    let policy = nn.policy_probs[..=AREA].to_vec();
    if policy
        .iter()
        .any(|p| !p.is_finite() || !(-1.0001..=1.0001).contains(p))
    {
        return Err(invalid("invalid policy probabilities"));
    }
    let outcome = [
        nn.white_win_prob as f64,
        nn.white_loss_prob as f64,
        nn.white_no_result_prob as f64,
    ];
    if outcome
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0001).contains(p))
        || (outcome.iter().sum::<f64>() - 1.0).abs() > 0.001
    {
        return Err(invalid("invalid outcome probabilities"));
    }
    let mean = nn.white_score_mean as f64;
    let mean_sq = nn.white_score_mean_sq as f64;
    let lead = nn.white_lead as f64;
    let variance_time = nn.var_time_left as f64;
    let shortterm_winloss = nn.shortterm_winloss_error as f64;
    let shortterm_score = nn.shortterm_score_error as f64;
    if [
        mean,
        mean_sq,
        lead,
        variance_time,
        shortterm_winloss,
        shortterm_score,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(invalid("nonfinite NN value"));
    }
    if mean.abs() > 10_000.0
        || lead.abs() > 10_000.0
        || mean_sq > 100_000_000.0
        || mean_sq < mean * mean - 0.02
    {
        return Err(invalid("invalid score moments"));
    }
    if has_shortterm_error
        && (!(0.0..=10.0).contains(&shortterm_winloss)
            || !(0.0..=10_000.0).contains(&shortterm_score))
    {
        return Err(invalid("uncertainty outside supported range"));
    }
    if ownership.iter().any(|x| !x.is_finite() || x.abs() > 1.0001) {
        return Err(invalid("ownership outside supported range"));
    }
    Ok(wire::NnOutput {
        policy,
        white_win_prob: outcome[0],
        white_loss_prob: outcome[1],
        white_no_result_prob: outcome[2],
        white_score_mean: mean,
        white_score_mean_sq: mean_sq,
        white_lead: lead,
        var_time_left: variance_time,
        shortterm_winloss_error: shortterm_winloss,
        shortterm_score_error: shortterm_score,
        ownership,
        has_shortterm_error,
    })
}

fn micros(start: Instant) -> u64 {
    start.elapsed().as_micros().min(u64::MAX as u128) as u64
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "worker stage panicked".to_owned()
    }
}

#[cfg(test)]
#[path = "input_fixture_export.rs"]
mod input_fixture_export;

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> wire::EvalRequest {
        wire::EvalRequest {
            position: Some(wire::Position {
                board_size: 19,
                komi: 7.5,
                rules: "chinese".into(),
                initial_player: 1,
                next_player: 1,
                ..Default::default()
            }),
            parameters: Some(wire::EvalParameters {
                policy_temperature: 1.0,
                draw_equivalent_wins_for_white: 0.5,
                max_history: 1000,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn with_moves(moves: &[(i32, i32)]) -> wire::EvalRequest {
        let mut r = request();
        let pos = r.position.as_mut().unwrap();
        pos.moves = moves
            .iter()
            .map(|&(color, vertex)| wire::Move { color, vertex })
            .collect();
        pos.next_player = if moves.len() % 2 == 0 { 1 } else { 2 };
        r
    }

    fn assert_context_error(r: &wire::EvalRequest) {
        let error = prepare_context(r).err().expect("invalid context must fail");
        assert_eq!(error.code, "INVALID_CONTEXT", "{}", error.message);
    }

    #[test]
    fn rejects_wrong_color_illegal_point_and_mismatched_next_player() {
        assert_context_error(&with_moves(&[(2, 0)]));
        assert_context_error(&with_moves(&[(1, 0), (1, 1)]));
        assert_context_error(&with_moves(&[(1, 0), (2, 0)]));
        assert_context_error(&with_moves(&[(1, 361)]));
        assert_context_error(&with_moves(&[(1, -2)]));
        assert_context_error(&with_moves(&[(0, 0)]));
        let mut r = with_moves(&[(1, 0)]);
        r.position.as_mut().unwrap().next_player = 1;
        assert_context_error(&r);
        let valid = prepare_context(&with_moves(&[(1, 0), (2, 360)])).unwrap();
        assert_eq!(
            valid.board.colors[location::get_loc(0, 0, 19) as usize],
            P_BLACK
        );
        assert_eq!(
            valid.board.colors[location::get_loc(18, 18, 19) as usize],
            P_WHITE
        );
        let mut suicide = with_moves(&[(1, 0)]);
        suicide.position.as_mut().unwrap().initial_stones = vec![
            wire::Stone {
                color: 2,
                vertex: 1,
            },
            wire::Stone {
                color: 2,
                vertex: 19,
            },
        ];
        assert_context_error(&suicide);
    }

    #[test]
    fn validates_initial_stones_simultaneously_and_liberties() {
        let mut r = request();
        r.position.as_mut().unwrap().initial_stones = vec![
            wire::Stone {
                color: 1,
                vertex: 0,
            },
            wire::Stone {
                color: 2,
                vertex: 0,
            },
        ];
        assert_context_error(&r);
        r.position.as_mut().unwrap().initial_stones = vec![
            wire::Stone {
                color: 1,
                vertex: 0,
            },
            wire::Stone {
                color: 2,
                vertex: 1,
            },
            wire::Stone {
                color: 2,
                vertex: 19,
            },
        ];
        assert_context_error(&r);
        r.position.as_mut().unwrap().initial_stones = vec![
            wire::Stone {
                color: 2,
                vertex: 1,
            },
            wire::Stone {
                color: 2,
                vertex: 19,
            },
            wire::Stone {
                color: 1,
                vertex: 20,
            },
            wire::Stone {
                color: 1,
                vertex: 21,
            },
        ];
        let first = prepare_context(&r).unwrap().board.pos_hash;
        r.position.as_mut().unwrap().initial_stones.reverse();
        assert_eq!(prepare_context(&r).unwrap().board.pos_hash, first);
    }

    #[test]
    fn friendly_pass_requires_precise_final_and_intermediate_permissions() {
        let mut r = with_moves(&[(1, -1), (2, -1)]);
        assert_context_error(&r);
        r.parameters.as_mut().unwrap().force_non_terminal = true;
        let terminal = prepare_context(&r).unwrap();
        assert!(terminal.history.is_game_finished);
        assert!(!terminal.history.is_no_result);
        let mut crossed = with_moves(&[(1, -1), (2, -1), (1, 180)]);
        assert_context_error(&crossed);
        crossed
            .parameters
            .as_mut()
            .unwrap()
            .allow_terminal_search_history = true;
        assert!(!prepare_context(&crossed).unwrap().history.is_game_finished);
        crossed.parameters.as_mut().unwrap().force_non_terminal = true;
        assert_context_error(&crossed);
        let mut triple = with_moves(&[(1, -1), (2, -1), (1, -1)]);
        let p = triple.parameters.as_mut().unwrap();
        p.allow_terminal_search_history = true;
        p.force_non_terminal = true;
        assert_context_error(&triple);
        let mut fourth = with_moves(&[(1, -1), (2, -1), (1, -1), (2, 180)]);
        fourth
            .parameters
            .as_mut()
            .unwrap()
            .allow_terminal_search_history = true;
        assert_context_error(&fourth);
        let mut live = request();
        live.parameters.as_mut().unwrap().force_non_terminal = true;
        assert_context_error(&live);
    }

    #[test]
    fn all_misc_parameters_map_without_search_defaults() {
        let mut r = request();
        let p = r.parameters.as_mut().unwrap();
        p.symmetry = 7;
        p.policy_temperature = 0.25;
        p.policy_optimism = 0.75;
        p.draw_equivalent_wins_for_white = 0.125;
        p.playout_doubling_advantage = -3.5;
        p.max_history = 2;
        p.conservative_pass = true;
        p.enable_passing_hacks = true;
        p.avoid_mytdagger_hack = true;
        p.skip_cache = true;
        p.include_ownership = true;
        let prepared = prepare_context(&r).unwrap();
        let m = prepared.params;
        assert_eq!(m.symmetry, 7);
        assert_eq!(m.nn_policy_temperature, 0.25);
        assert_eq!(m.policy_optimism, 0.75);
        assert_eq!(m.draw_equivalent_wins_for_white, 0.125);
        assert_eq!(m.playout_doubling_advantage, -3.5);
        assert_eq!(m.max_history, 2);
        assert!(
            m.conservative_pass_and_is_root && m.enable_passing_hacks && m.avoid_mytdagger_hack
        );
        assert!(prepared.skip_cache && prepared.include_ownership);
        for komi in [f64::NAN, f64::INFINITY, 150.5, 7.25, 7.50000001] {
            let mut bad = request();
            bad.position.as_mut().unwrap().komi = komi;
            assert_context_error(&bad);
        }
        for temperature in [f64::NAN, f64::INFINITY, 0.0, 0.0001, 100.1] {
            let mut bad = request();
            bad.parameters.as_mut().unwrap().policy_temperature = temperature;
            assert_context_error(&bad);
        }
        r.parameters.as_mut().unwrap().always_compute_pass_alive = true;
        assert_context_error(&r);
    }

    #[test]
    fn rejects_missing_context_and_every_out_of_range_semantic_field() {
        let mut missing = request();
        missing.parameters = None;
        assert_context_error(&missing);
        missing = request();
        missing.position = None;
        assert_context_error(&missing);
        let invalid_parameters: [fn(&mut wire::EvalParameters); 10] = [
            |p| p.symmetry = -1,
            |p| p.symmetry = 8,
            |p| p.max_history = 10_001,
            |p| p.policy_optimism = -0.1,
            |p| p.policy_optimism = f64::NAN,
            |p| p.draw_equivalent_wins_for_white = 1.1,
            |p| p.draw_equivalent_wins_for_white = f64::INFINITY,
            |p| p.playout_doubling_advantage = -100.1,
            |p| p.playout_doubling_advantage = f64::NAN,
            |p| p.exclude_territory_adjacent_to_atari = true,
        ];
        for change in invalid_parameters {
            let mut r = request();
            change(r.parameters.as_mut().unwrap());
            assert_context_error(&r);
        }
        let mut r = request();
        r.position.as_mut().unwrap().initial_player = 0;
        assert_context_error(&r);
        r = request();
        r.position.as_mut().unwrap().rules = "japanese".into();
        assert_context_error(&r);
    }

    #[test]
    fn hashes_actual_file_bytes_and_honors_per_model_backend() {
        let path =
            std::env::temp_dir().join(format!("kata-worker-hash-{}.bin", uuid::Uuid::new_v4()));
        std::fs::write(&path, b"abc").unwrap();
        let hashed = hash_model_file(path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();
        assert_eq!(
            hashed.unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let cfg =
            ConfigParser::from_str("nnBackend=cuda\nnnBackend0=dummy\n", false, false).unwrap();
        assert_eq!(selected_backend(&cfg).unwrap(), "dummybackend");
        assert!(Engine::load("/dev/null", None, &cfg, 2, false).is_err());
    }

    #[test]
    fn rejects_unvalidated_production_backends_before_loading_any_model() {
        for backend in ["trt", "tensorrt", "trtbackend"] {
            let cfg =
                ConfigParser::from_str(&format!("nnBackend={backend}\n"), false, false).unwrap();
            for allow_dummy in [false, true] {
                let error = Engine::load("unread-model.onnx", None, &cfg, 2, allow_dummy)
                    .err()
                    .expect("TRT worker must be rejected");
                assert!(
                    error
                        .to_string()
                        .contains("TensorRT model-bound cache not yet validated")
                );
            }
        }
        let cfg =
            ConfigParser::from_str("nnBackend=cuda\nnnBackend0=eigen\n", false, false).unwrap();
        let error = Engine::load("unread-model.onnx", None, &cfg, 2, false)
            .err()
            .expect("Eigen worker must be rejected");
        assert!(error.to_string().contains("eigenbackend is not supported"));
    }

    #[test]
    fn synthetic_permission_cannot_load_or_advertise_a_real_model() {
        for configuration in [
            "nnBackend=dummybackend\n",
            "nnBackend=dummybackend\ndebugSkipNeuralNet=true\n",
            "nnBackend=cudabackend\ndebugSkipNeuralNet=true\n",
            "nnBackend=cudabackend\n",
        ] {
            let cfg = ConfigParser::from_str(configuration, false, false).unwrap();
            let error = Engine::load("real-model.onnx", None, &cfg, 2, true)
                .err()
                .expect("allow-dummy must be limited to the named synthetic fixture");
            assert!(error.to_string().contains("--allow-dummy is restricted"));
        }
        let cfg = ConfigParser::from_str("nnBackend=cudabackend\n", false, false).unwrap();
        assert!(Engine::load("/dev/null", None, &cfg, 2, true).is_err());
    }

    fn output() -> NNOutput {
        let mut nn = NNOutput {
            nn_x_len: 19,
            nn_y_len: 19,
            white_win_prob: 0.75,
            white_loss_prob: 0.125,
            white_no_result_prob: 0.125,
            white_score_mean: -2.0,
            white_score_mean_sq: 9.0,
            white_lead: -1.5,
            var_time_left: 42.0,
            shortterm_winloss_error: 0.25,
            shortterm_score_error: 1.25,
            white_owner_map: Some(vec![-0.5; AREA].into_boxed_slice()),
            ..Default::default()
        };
        nn.policy_probs[..=AREA].fill(1.0 / AREA as f32);
        nn.policy_probs[0] = -1.0;
        nn
    }

    #[test]
    fn output_preserves_white_perspective_orientation_and_legacy_errors() {
        let mut nn = output();
        nn.policy_probs[360] = 0.3;
        nn.policy_probs[361] = 0.2;
        let out = convert_output(&nn, true, true).unwrap();
        assert_eq!(out.policy.len(), 362);
        assert_eq!(out.policy[0], -1.0);
        assert_eq!(out.policy[360], 0.3);
        assert_eq!(out.policy[361], 0.2);
        assert_eq!(out.white_win_prob, 0.75);
        assert_eq!(out.white_score_mean, -2.0);
        assert_eq!(out.white_lead, -1.5);
        assert_eq!(out.ownership, vec![-0.5; AREA]);
        assert!(out.has_shortterm_error);
        nn.shortterm_winloss_error = -1.0;
        nn.shortterm_score_error = -1.0;
        let legacy = convert_output(&nn, false, false).unwrap();
        assert!(!legacy.has_shortterm_error);
        assert_eq!(legacy.shortterm_score_error, -1.0);
        assert!(legacy.ownership.is_empty());
        assert!(convert_output(&nn, false, true).is_err());
    }

    #[test]
    fn malformed_output_is_classified_as_worker_fault() {
        let mut nn = output();
        nn.white_score_mean_sq = 3.0;
        assert_eq!(
            convert_output(&nn, false, true).unwrap_err().code,
            "INVALID_OUTPUT"
        );
        nn = output();
        nn.white_win_prob = 0.1;
        assert!(convert_output(&nn, false, true).is_err());
        nn = output();
        nn.policy_probs[5] = f32::NAN;
        assert!(convert_output(&nn, false, true).is_err());
        nn = output();
        nn.white_owner_map = None;
        assert!(convert_output(&nn, true, true).is_err());
        nn.white_owner_map = Some(vec![0.0; AREA - 1].into_boxed_slice());
        assert!(convert_output(&nn, true, true).is_err());
        nn = output();
        nn.white_owner_map.as_mut().unwrap()[4] = 1.1;
        assert!(convert_output(&nn, true, true).is_err());
    }

    #[test]
    fn synthetic_engine_is_explicit_and_records_executed_stages() {
        let cfg = ConfigParser::from_str(
            "nnBackend=dummybackend\nnnMaxBatchSize=4\nnnCacheSizePowerOfTwo=4\nnnMutexPoolSizePowerOfTwo=2\n",
            false, false,
        ).unwrap();
        assert!(Engine::load("/dev/null", None, &cfg, 2, false).is_err());
        assert!(Engine::load("/dev/null", Some(&"0".repeat(64)), &cfg, 2, true).is_err());
        let engine = Engine::load("/dev/null", None, &cfg, 2, true).unwrap();
        assert!(engine.metadata.backend_info.contains("SYNTHETIC TEST ONLY"));
        assert!(engine.metadata.engine_commit.starts_with("Rust_KataGo/"));
        assert_eq!(
            engine.metadata.model_sha256,
            hex::encode(Sha256::digest(SYNTHETIC_MODEL))
        );
        assert!(!engine.metadata.supports_shortterm_error);
        let mut r = request();
        r.model_sha256.clone_from(&engine.metadata.model_sha256);
        r.parameters.as_mut().unwrap().include_ownership = true;
        let report = engine.evaluate(&r);
        assert!(report.context_us.is_some() && report.evaluator_us.is_some());
        assert_eq!(report.result.unwrap().ownership.len(), AREA);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let engine = &engine;
                let r = &r;
                scope.spawn(move || {
                    let output = engine.evaluate(r).result.unwrap();
                    assert_eq!(output.policy.len(), AREA + 1);
                    assert_eq!(output.ownership.len(), AREA);
                });
            }
        });
        r.position.as_mut().unwrap().board_size = 9;
        let invalid = engine.evaluate(&r);
        assert_eq!(invalid.result.unwrap_err().code, "INVALID_CONTEXT");
        assert!(invalid.context_us.is_some() && invalid.evaluator_us.is_none());
    }
}

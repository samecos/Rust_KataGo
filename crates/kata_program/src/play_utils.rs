//! Higher-level helpers for selecting moves, setting komi, and initializing games.
//!
//! Corresponds to the self-contained parts of `cpp/program/playutils.h` and
//! `cpp/program/playutils.cpp`. Functions that require `Search`, `AsyncBot`,
//! or `NNOutput` are intentionally left for later slices.

use kata_core::global::StringError;
use kata_core::rng::Rand;
use kata_game::board::{
    Board, C_EMPTY, Loc, MAX_ARR_SIZE, NULL_LOC, P_BLACK, P_WHITE, PASS_LOC, Player, get_opp,
    location,
};
use kata_game::history::BoardHistory;
use kata_game::rules::{KoRule, Rules, ScoringRule, TaxRule};
use kata_game::symmetry::NUM_SYMMETRIES;
use kata_nn::backend::NNResultBuf;
use kata_nn::inputs::nn_pos::KOMI_CLIP_RADIUS;
use kata_nn::inputs::{MiscNNInputParams, NNOutput, nn_pos};
use kata_search::params::SearchParams;
use kata_search::reported_values::ReportedSearchValues;
use kata_search::search::Search;

/// Komi and handicap information used when initializing a game.
///
/// Mirrors `ExtraBlackAndKomi` in `cpp/program/play.h`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ExtraBlackAndKomi {
    pub extra_black: i32,
    pub komi_mean: f32,
    pub komi_stdev: f32,
    pub make_game_fair: bool,
    pub make_game_fair_for_empty_board: bool,
    pub allow_integer: bool,
    pub interp_zero: bool,
}

fn get_default_max_extra_black(sqrt_board_area: f64) -> i32 {
    if sqrt_board_area <= 10.000_01 {
        0
    } else if sqrt_board_area <= 14.000_01 {
        1
    } else if sqrt_board_area <= 16.000_01 {
        2
    } else if sqrt_board_area <= 17.000_01 {
        3
    } else if sqrt_board_area <= 18.000_01 {
        4
    } else {
        5
    }
}

/// Choose a random number of handicap stones and a random komi distribution.
///
/// Mirrors `PlayUtils::chooseExtraBlackAndKomi`.
#[allow(clippy::too_many_arguments)]
pub fn choose_extra_black_and_komi(
    base: f32,
    stdev: f32,
    allow_integer_prob: f64,
    handicap_prob: f64,
    num_extra_black_fixed: i32,
    big_stdev_prob: f64,
    big_stdev: f32,
    bigger_stdev_prob: f64,
    bigger_stdev: f32,
    sqrt_board_area: f64,
    rand: &mut Rand,
) -> ExtraBlackAndKomi {
    let mut extra_black = 0;
    let komi = base;

    let mut stdev_to_use = 0.0f32;
    if stdev > 0.0 {
        stdev_to_use = stdev;
    }
    if big_stdev > 0.0 && rand.next_bool(big_stdev_prob) {
        stdev_to_use = big_stdev;
    }
    if bigger_stdev > 0.0 && bigger_stdev_prob > 0.0 && rand.next_bool(bigger_stdev_prob) {
        stdev_to_use = bigger_stdev;
    }
    // Adjust for board size, so that we don't give the same massive komis on smaller boards.
    stdev_to_use *= (sqrt_board_area / 19.0) as f32;

    // Add handicap stones.
    let default_max_extra_black = get_default_max_extra_black(sqrt_board_area);
    if (num_extra_black_fixed > 0 || default_max_extra_black > 0) && rand.next_bool(handicap_prob) {
        if num_extra_black_fixed > 0 {
            extra_black = num_extra_black_fixed;
        } else {
            extra_black = 1 + rand.next_u32_bounded(default_max_extra_black as u32) as i32;
        }
    }

    let allow_integer = rand.next_bool(allow_integer_prob);

    ExtraBlackAndKomi {
        extra_black,
        komi_mean: komi,
        komi_stdev: stdev_to_use,
        make_game_fair: false,
        make_game_fair_for_empty_board: false,
        interp_zero: false,
        allow_integer,
    }
}

/// Round `komi` to the nearest half-integer using a linear probability between
/// the two nearest half-integers.
fn round_komi_with_linear_prob(komi: f32, rand: &mut Rand) -> f32 {
    let lower = (komi * 2.0).floor() / 2.0;
    let upper = (komi * 2.0).ceil() / 2.0;

    if lower == upper {
        lower
    } else {
        assert!(upper > lower);
        if rand.next_double() < f64::from(komi - lower) / f64::from(upper - lower) {
            upper
        } else {
            lower
        }
    }
}

/// Clamp and round a komi value to a half-integer within a board-dependent range.
///
/// Mirrors `PlayUtils::roundAndClipKomi`.
pub fn round_and_clip_komi(unrounded: f64, board: &Board) -> f32 {
    let range = f64::from(KOMI_CLIP_RADIUS) + f64::from(board.x_size * board.y_size);
    let mut unrounded = unrounded.clamp(-range, range);
    unrounded = 0.5 * (2.0 * unrounded).round();
    unrounded as f32
}

/// Set komi from `extra_black_and_komi` without noise. Also ignores `allow_integer`.
///
/// Mirrors `PlayUtils::setKomiWithoutNoise`.
pub fn set_komi_without_noise(extra_black_and_komi: &ExtraBlackAndKomi, hist: &mut BoardHistory) {
    let komi = round_and_clip_komi(
        f64::from(extra_black_and_komi.komi_mean),
        hist.get_recent_board(0),
    );
    assert!(Rules::komi_is_int_or_half_int(komi));
    hist.set_komi(komi);
}

/// Set komi from `extra_black_and_komi` with Gaussian noise and rounding.
///
/// Mirrors `PlayUtils::setKomiWithNoise`.
pub fn set_komi_with_noise(
    extra_black_and_komi: &ExtraBlackAndKomi,
    hist: &mut BoardHistory,
    rand: &mut Rand,
) {
    let mut komi = extra_black_and_komi.komi_mean;
    if extra_black_and_komi.komi_stdev > 0.0 {
        komi += extra_black_and_komi.komi_stdev * rand.next_gaussian_truncated(3.0) as f32;
    }
    if extra_black_and_komi.interp_zero {
        komi *= rand.next_double() as f32;
    }
    let mut komi = round_komi_with_linear_prob(komi, rand);
    komi = round_and_clip_komi(f64::from(komi), hist.get_recent_board(0));
    assert!(Rules::komi_is_int_or_half_int(komi));
    if !extra_black_and_komi.allow_integer && komi.fract() == 0.0 {
        komi += if rand.next_bool(0.5) { -0.5 } else { 0.5 };
    }
    hist.set_komi(komi);
}

/// Choose a uniformly random legal move, optionally banning one location.
///
/// Mirrors `PlayUtils::chooseRandomLegalMove`.
pub fn choose_random_legal_move(
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
    game_rand: &mut Rand,
    ban_move: Loc,
) -> Loc {
    assert_eq!(pla, hist.presumed_next_move_pla);
    let mut locs = [NULL_LOC; MAX_ARR_SIZE];
    let mut num_legal_moves = 0usize;
    for loc in 0..MAX_ARR_SIZE as Loc {
        if hist.is_legal(board, loc, pla) && loc != ban_move {
            locs[num_legal_moves] = loc;
            num_legal_moves += 1;
        }
    }
    if num_legal_moves > 0 {
        let n = game_rand.next_u32_bounded(num_legal_moves as u32) as usize;
        locs[n]
    } else {
        NULL_LOC
    }
}

/// Fill `buf` with uniformly random legal moves (sampled with replacement).
///
/// Returns the number of moves written (either `buf.len()` or 0 if there are
/// no legal moves). Mirrors `PlayUtils::chooseRandomLegalMoves`.
pub fn choose_random_legal_moves(
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
    game_rand: &mut Rand,
    buf: &mut [Loc],
) -> usize {
    assert_eq!(pla, hist.presumed_next_move_pla);
    let mut locs = [NULL_LOC; MAX_ARR_SIZE];
    let mut num_legal_moves = 0usize;
    for loc in 0..MAX_ARR_SIZE as Loc {
        if hist.is_legal(board, loc, pla) {
            locs[num_legal_moves] = loc;
            num_legal_moves += 1;
        }
    }
    if num_legal_moves > 0 {
        for out in buf.iter_mut() {
            let n = game_rand.next_u32_bounded(num_legal_moves as u32) as usize;
            *out = locs[n];
        }
        buf.len()
    } else {
        0
    }
}

/// Place `n` fixed handicap stones on an empty board of the given size.
///
/// Mirrors `PlayUtils::placeFixedHandicap`.
pub fn place_fixed_handicap(board: &mut Board, n: i32) -> Result<(), StringError> {
    let x_size = board.x_size;
    let y_size = board.y_size;
    if x_size < 7 || y_size < 7 {
        return Err(StringError::new(
            "Board is too small for fixed handicap".to_string(),
        ));
    }
    if (x_size % 2 == 0 || y_size % 2 == 0) && n > 4 {
        return Err(StringError::new(
            "Fixed handicap > 4 is not allowed on boards with even dimensions".to_string(),
        ));
    }
    if (x_size <= 7 || y_size <= 7) && n > 4 {
        return Err(StringError::new(
            "Fixed handicap > 4 is not allowed on boards with size 7".to_string(),
        ));
    }
    if n < 2 {
        return Err(StringError::new(
            "Fixed handicap < 2 is not allowed".to_string(),
        ));
    }
    if n > 9 {
        return Err(StringError::new(
            "Fixed handicap > 9 is not allowed".to_string(),
        ));
    }

    *board = Board::new(x_size, y_size);

    let x_coords = if x_size <= 12 {
        [2, x_size - 3, x_size / 2]
    } else {
        [3, x_size - 4, x_size / 2]
    };
    let y_coords = if y_size <= 12 {
        [2, y_size - 3, y_size / 2]
    } else {
        [3, y_size - 4, y_size / 2]
    };

    let set = |board: &mut Board, xi: usize, yi: usize| {
        let loc = location::get_loc(x_coords[xi], y_coords[yi], board.x_size);
        board.set_stone(loc, P_BLACK);
    };

    match n {
        2 => {
            set(board, 0, 1);
            set(board, 1, 0);
        }
        3 => {
            set(board, 0, 1);
            set(board, 1, 0);
            set(board, 0, 0);
        }
        4 => {
            set(board, 0, 1);
            set(board, 1, 0);
            set(board, 0, 0);
            set(board, 1, 1);
        }
        5 => {
            set(board, 0, 1);
            set(board, 1, 0);
            set(board, 0, 0);
            set(board, 1, 1);
            set(board, 2, 2);
        }
        6 => {
            set(board, 0, 1);
            set(board, 1, 0);
            set(board, 0, 0);
            set(board, 1, 1);
            set(board, 0, 2);
            set(board, 1, 2);
        }
        7 => {
            set(board, 0, 1);
            set(board, 1, 0);
            set(board, 0, 0);
            set(board, 1, 1);
            set(board, 0, 2);
            set(board, 1, 2);
            set(board, 2, 2);
        }
        8 => {
            set(board, 0, 1);
            set(board, 1, 0);
            set(board, 0, 0);
            set(board, 1, 1);
            set(board, 0, 2);
            set(board, 1, 2);
            set(board, 2, 0);
            set(board, 2, 1);
        }
        9 => {
            set(board, 0, 1);
            set(board, 1, 0);
            set(board, 0, 0);
            set(board, 1, 1);
            set(board, 0, 2);
            set(board, 1, 2);
            set(board, 2, 0);
            set(board, 2, 1);
            set(board, 2, 2);
        }
        _ => unreachable!(),
    }

    Ok(())
}

/// Generate a random `Rules` with uniformly distributed rule options.
///
/// Mirrors `PlayUtils::genRandomRules`.
pub fn gen_random_rules(rand: &mut Rand) -> Rules {
    let allowed_ko_rules = [KoRule::Simple, KoRule::Positional, KoRule::Situational];
    let allowed_scoring_rules = [ScoringRule::Area, ScoringRule::Territory];
    let allowed_tax_rules = [TaxRule::None, TaxRule::Seki, TaxRule::All];

    let ko_rule = allowed_ko_rules[rand.next_u32_bounded(allowed_ko_rules.len() as u32) as usize];
    let scoring_rule =
        allowed_scoring_rules[rand.next_u32_bounded(allowed_scoring_rules.len() as u32) as usize];
    let tax_rule =
        allowed_tax_rules[rand.next_u32_bounded(allowed_tax_rules.len() as u32) as usize];
    let multi_stone_suicide_legal = rand.next_bool(0.5);

    let has_button = scoring_rule == ScoringRule::Area && rand.next_bool(0.5);

    Rules {
        ko_rule,
        scoring_rule,
        tax_rule,
        multi_stone_suicide_legal,
        has_button,
        ..Rules::default()
    }
}

/// Determine all living and dead stones using a simple Tromp-Taylor-like area
/// scoring that recognizes pass-dead stones.
///
/// Mirrors `PlayUtils::computeAnticipatedStatusesSimple`.
pub fn compute_anticipated_statuses_simple(board: &Board, hist: &BoardHistory) -> Vec<bool> {
    let mut is_alive = vec![false; MAX_ARR_SIZE];

    // Treat all stones as alive under a no-result.
    if hist.is_game_finished && hist.is_no_result {
        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let loc = location::get_loc(x, y, board.x_size) as usize;
                if board.colors[loc] != C_EMPTY {
                    is_alive[loc] = true;
                }
            }
        }
    } else {
        let mut area = [C_EMPTY; MAX_ARR_SIZE];
        board.calculate_area(
            &mut area,
            true,
            true,
            true,
            hist.rules.multi_stone_suicide_legal,
        );
        for y in 0..board.y_size {
            for x in 0..board.x_size {
                let loc = location::get_loc(x, y, board.x_size) as usize;
                if board.colors[loc] != C_EMPTY {
                    is_alive[loc] = board.colors[loc] == area[loc];
                }
            }
        }
    }

    is_alive
}

/// Compute a search factor multiplier when a player is heavily winning.
///
/// Mirrors `PlayUtils::getSearchFactor`. The factor is 1.0 unless the last
/// three win/loss values all exceed `search_factor_when_winning_threshold`,
/// in which case it interpolates up to `search_factor_when_winning`.
pub fn get_search_factor(
    search_factor_when_winning_threshold: f64,
    search_factor_when_winning: f64,
    params: &SearchParams,
    recent_win_loss_values: &[f64],
    pla: Player,
) -> f64 {
    let mut search_factor = 1.0;
    if recent_win_loss_values.len() >= 3
        && params.win_loss_utility_factor - search_factor_when_winning_threshold > 1e-10
    {
        let mut recent_least_winning = if pla == P_BLACK {
            -params.win_loss_utility_factor
        } else {
            params.win_loss_utility_factor
        };
        let start = recent_win_loss_values.len() - 3;
        for &wl in &recent_win_loss_values[start..] {
            if pla == P_BLACK && wl > recent_least_winning {
                recent_least_winning = wl;
            }
            if pla == P_WHITE && wl < recent_least_winning {
                recent_least_winning = wl;
            }
        }
        let excess_winning = if pla == P_BLACK {
            -search_factor_when_winning_threshold - recent_least_winning
        } else {
            recent_least_winning - search_factor_when_winning_threshold
        };
        if excess_winning > 0.0 {
            let lambda = excess_winning
                / (params.win_loss_utility_factor - search_factor_when_winning_threshold);
            search_factor = 1.0 + lambda * (search_factor_when_winning - 1.0);
        }
    }
    search_factor
}

/// Sample a legal move from a policy distribution, applying temperature.
///
/// Mirrors `PlayUtils::chooseRandomPolicyMove`. `allow_pass` controls whether
/// passing is among the candidate moves and `ban_move` excludes one location.
#[allow(clippy::too_many_arguments)]
pub fn choose_random_policy_move(
    nn_output: &NNOutput,
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
    rand: &mut Rand,
    temperature: f64,
    allow_pass: bool,
    ban_move: Loc,
) -> Loc {
    assert_eq!(pla, hist.presumed_next_move_pla);
    let mut locs = Vec::new();
    let mut probs = Vec::new();
    for loc in 0..MAX_ARR_SIZE as Loc {
        if loc == ban_move {
            continue;
        }
        if loc == PASS_LOC && !allow_pass {
            continue;
        }
        if hist.is_legal(board, loc, pla) {
            let pos = nn_pos::loc_to_pos(loc, board.x_size, nn_output.nn_x_len, nn_output.nn_y_len);
            let p = nn_output.policy_probs[pos as usize];
            if p >= 0.0 {
                locs.push(loc);
                probs.push(f64::from(p));
            }
        }
    }
    if locs.is_empty() {
        return NULL_LOC;
    }

    if temperature != 1.0 {
        for p in &mut probs {
            *p = p.powf(temperature);
        }
    }
    let sum: f64 = probs.iter().sum();
    if sum <= 0.0 {
        return NULL_LOC;
    }
    for p in &mut probs {
        *p /= sum;
    }
    let idx = rand.next_u32_from_probs(&probs) as usize;
    locs[idx]
}

/// Sample a single move for game initialization from the bot's policy.
///
/// Mirrors `PlayUtils::getGameInitializationMove`. This is the move-sampling
/// helper used by `evalrandominits` and by policy-based game initialization.
///
/// The C++ version takes separate `botB` and `botW` pointers; the Rust port
/// accepts a single bot because the evaluator and parameters are drawn from the
/// player to move, and the caller can pass the same search for both colors when
/// needed.
#[allow(clippy::too_many_arguments)]
pub fn get_game_initialization_move(
    bot: &mut Search,
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
    game_rand: &mut Rand,
    temperature: f64,
) -> Loc {
    assert_eq!(pla, hist.presumed_next_move_pla);

    let nn_eval = bot
        .nn_evaluator
        .expect("get_game_initialization_move called with a Search that has no NNEvaluator");

    let mut buf = NNResultBuf::new();
    let nn_input_params = MiscNNInputParams {
        draw_equivalent_wins_for_white: bot.search_params.draw_equivalent_wins_for_white,
        ..Default::default()
    };
    nn_eval.evaluate(board, hist, pla, &nn_input_params, &mut buf, false, false);
    let nn_output = buf.result.expect("NN evaluation produced no output");

    let policy_size = nn_output.nn_x_len as usize * nn_output.nn_y_len as usize + 1;
    assert!(nn_output.nn_x_len >= board.x_size);
    assert!(nn_output.nn_y_len >= board.y_size);
    assert!(nn_output.nn_x_len > 0 && nn_output.nn_x_len < 100);
    assert!(nn_output.nn_y_len > 0 && nn_output.nn_y_len < 100);

    let mut locs = Vec::new();
    let mut play_selection_values = Vec::new();
    for move_pos in 0..policy_size {
        let move_loc = nn_pos::pos_to_loc(
            move_pos as i32,
            board.x_size,
            board.y_size,
            nn_output.nn_x_len,
            nn_output.nn_y_len,
        );
        let policy_prob = nn_output.policy_probs[move_pos];
        if !hist.is_legal(board, move_loc, pla) || policy_prob <= 0.0 {
            continue;
        }
        locs.push(move_loc);
        play_selection_values.push(f64::from(policy_prob).powf(1.0 / temperature));
    }

    if play_selection_values.is_empty() {
        panic!("get_game_initialization_move: playSelectionValues.size() <= 0");
    }

    let idx_chosen = if game_rand.next_bool(0.0002) {
        game_rand.next_u32_bounded(locs.len() as u32) as usize
    } else {
        game_rand.next_u32_from_probs(&play_selection_values) as usize
    };
    locs[idx_chosen]
}

/// Place `num_extra_black` free handicap stones using the bot's policy.
///
/// Mirrors `PlayUtils::playExtraBlack`. This implementation is a best-effort
/// port: it samples policy moves and clears the history after each stone.
pub fn play_extra_black(
    bot: &mut Search,
    num_extra_black: i32,
    board: &mut Board,
    hist: &mut BoardHistory,
    temperature: f64,
    game_rand: &mut Rand,
) {
    let pla = P_BLACK;
    if hist.is_game_finished {
        return;
    }

    let nn_eval = bot
        .nn_evaluator
        .expect("playExtraBlack called with a Search that has no NNEvaluator");

    for _ in 0..num_extra_black {
        let mut buf = NNResultBuf::new();
        let nn_input_params = MiscNNInputParams {
            draw_equivalent_wins_for_white: bot.search_params.draw_equivalent_wins_for_white,
            ..Default::default()
        };
        nn_eval.evaluate(board, hist, pla, &nn_input_params, &mut buf, false, false);
        let nn_output = buf.result.expect("NN evaluation produced no output");

        let loc = choose_random_policy_move(
            &nn_output,
            board,
            hist,
            pla,
            game_rand,
            temperature,
            false,
            NULL_LOC,
        );
        if loc == NULL_LOC {
            break;
        }
        hist.make_board_move_assume_legal(board, loc, pla);
        hist.clear(board.clone(), pla, hist.rules, 0);
    }

    bot.set_position(pla, board, hist);
}

/// Run a short search and return the reported root values from White's perspective.
///
/// Mirrors `PlayUtils::getWhiteScoreValues` in `cpp/program/playutils.cpp`.
#[allow(clippy::too_many_arguments)]
fn get_white_score_values(
    bot: &mut Search,
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
    num_visits: i64,
    other_game_props: &super::play::OtherGameProperties,
) -> ReportedSearchValues {
    assert!(num_visits > 0);
    let old_params = bot.search_params.clone();
    let mut new_params = get_noiseless_params(&old_params, num_visits);

    if other_game_props.playout_doubling_advantage != 0.0
        && other_game_props.playout_doubling_advantage_pla != C_EMPTY
    {
        // Don't actually adjust playouts, but DO tell the bot what it's up
        // against, so that it gives estimates appropriate to the asymmetric
        // game about to be played.
        new_params.playout_doubling_advantage_pla = other_game_props.playout_doubling_advantage_pla;
        new_params.playout_doubling_advantage = other_game_props.playout_doubling_advantage;
    }

    bot.set_params(&new_params);
    bot.set_position(pla, board, hist);
    bot.run_whole_search(pla);

    let values = bot.get_root_values_require_success();
    bot.set_params(&old_params);
    values
}

/// Evaluate a candidate komi and return `(lead, win_loss)` for White.
///
/// Mirrors the local `evalKomi` helper in `cpp/program/playutils.cpp`. Caches
/// results per rounded-and-clipped komi value and restores the original komi
/// before returning.
#[allow(clippy::too_many_arguments)]
fn eval_komi(
    score_wl_cache: &mut Vec<(f32, (f64, f64))>,
    bot_b: &mut Search,
    bot_w: &mut Search,
    board: &Board,
    hist: &mut BoardHistory,
    pla: Player,
    num_visits: i64,
    other_game_props: &super::play::OtherGameProperties,
    rounded_clipped_komi: f32,
) -> (f64, f64) {
    if let Some(&cached) = score_wl_cache
        .iter()
        .find(|(k, _)| *k == rounded_clipped_komi)
        .map(|(_, v)| v)
    {
        return cached;
    }

    let old_komi = hist.rules.komi as f32 / 2.0;
    hist.set_komi(rounded_clipped_komi);

    let values0 = get_white_score_values(bot_b, board, hist, pla, num_visits, other_game_props);
    let mut lead = values0.lead;
    let mut win_loss = values0.win_loss_value;

    // If we have a second bot, average the two.
    if !std::ptr::eq(bot_w, bot_b) {
        let values1 = get_white_score_values(bot_w, board, hist, pla, num_visits, other_game_props);
        lead = 0.5 * (values0.lead + values1.lead);
        win_loss = 0.5 * (values0.win_loss_value + values1.win_loss_value);
    }

    let result = (lead, win_loss);
    score_wl_cache.push((rounded_clipped_komi, result));

    hist.set_komi(old_komi);
    result
}

/// Search for a komi value that makes the game roughly even.
///
/// Mirrors the local `getNaiveEvenKomiHelper` helper in
/// `cpp/program/playutils.cpp`. Returns the new komi as `f64`; the caller is
/// responsible for restoring or applying it.
#[allow(clippy::too_many_arguments)]
fn get_naive_even_komi_helper(
    score_wl_cache: &mut Vec<(f32, (f64, f64))>,
    bot_b: &mut Search,
    bot_w: &mut Search,
    board: &Board,
    hist: &mut BoardHistory,
    pla: Player,
    num_visits: i64,
    other_game_props: &super::play::OtherGameProperties,
) -> f64 {
    let old_komi = hist.rules.komi as f32 / 2.0;

    // A few times iterate based on expected score to hopefully get a value
    // close to fair.
    let mut last_shift = 0.0;
    let mut last_win_loss = 0.0;
    let mut last_lead = 0.0;
    for i in 0..3 {
        let (lead, win_loss) = eval_komi(
            score_wl_cache,
            bot_b,
            bot_w,
            board,
            hist,
            pla,
            num_visits,
            other_game_props,
            hist.rules.komi as f32 / 2.0,
        );

        // If the last shift made stats go the WRONG way, and by a nontrivial
        // amount, then revert half of it and stop immediately.
        if i > 0
            && ((last_lead > 0.0 && lead > last_lead + 5.0 && win_loss < 0.75)
                || (last_lead < 0.0 && lead < last_lead - 5.0 && win_loss > -0.75)
                || (last_win_loss > 0.0 && win_loss > last_win_loss + 0.1)
                || (last_win_loss < 0.0 && win_loss < last_win_loss - 0.1))
        {
            let fair_komi =
                round_and_clip_komi(f64::from(hist.rules.komi) / 2.0 - last_shift * 0.5, board);
            hist.set_komi(fair_komi);
            break;
        }
        last_lead = lead;
        last_win_loss = win_loss;

        // Shift by the predicted lead.
        let mut shift = -lead;
        // Under no situations should the shift be bigger in absolute value than
        // the last shift.
        if i > 0 && shift.abs() > last_shift.abs() {
            if shift < 0.0 {
                shift = -last_shift.abs();
            } else if shift > 0.0 {
                shift = last_shift.abs();
            }
        }
        last_shift = shift;

        // If the score and winrate would like to move in opposite directions,
        // quit immediately.
        if (shift > 0.0 && win_loss > 0.0) || (shift < 0.0 && lead < 0.0) {
            break;
        }

        let fair_komi = round_and_clip_komi(f64::from(hist.rules.komi) / 2.0 + shift, board);
        hist.set_komi(fair_komi);

        // After a small shift, break out to the binary search.
        if shift.abs() < 16.0 {
            break;
        }
    }

    // Try a small window and do a binary search.
    let mut eval_win_loss = |delta: f64| -> f64 {
        let new_komi = f64::from(hist.rules.komi) / 2.0 + delta;
        let rounded = round_and_clip_komi(new_komi, board);
        eval_komi(
            score_wl_cache,
            bot_b,
            bot_w,
            board,
            hist,
            pla,
            num_visits,
            other_game_props,
            rounded,
        )
        .1
    };

    let mut lower_delta = 0.0;
    let mut upper_delta = 0.0;
    let mut lower_win_loss = 0.0;
    let mut upper_win_loss = 0.0;

    // Grow window outward.
    {
        let win_loss_zero = eval_win_loss(0.0);
        if win_loss_zero < 0.0 {
            // Losing, so this is the lower bound.
            lower_delta = 0.0;
            lower_win_loss = win_loss_zero;
            for i in 0..=5 {
                upper_delta = 2.0_f64.powi(i).round();
                upper_win_loss = eval_win_loss(upper_delta);
                if upper_win_loss >= 0.0 {
                    break;
                }
            }
        } else {
            // Winning, so this is the upper bound.
            upper_delta = 0.0;
            upper_win_loss = win_loss_zero;
            for i in 0..=5 {
                lower_delta = -2.0_f64.powi(i).round();
                lower_win_loss = eval_win_loss(lower_delta);
                if lower_win_loss <= 0.0 {
                    break;
                }
            }
        }
    }

    while upper_delta - lower_delta > 0.500_01 {
        let mid_delta = 0.5 * (lower_delta + upper_delta);
        let mid_win_loss = eval_win_loss(mid_delta);
        if mid_win_loss < 0.0 {
            lower_delta = mid_delta;
            lower_win_loss = mid_win_loss;
        } else {
            upper_delta = mid_delta;
            upper_win_loss = mid_win_loss;
        }
    }
    // Floating point math should be exact to multiples of 0.5 so this should
    // hold *exactly*.
    assert!((upper_delta - lower_delta - 0.5).abs() < 1e-9);

    let final_delta = if lower_win_loss >= upper_win_loss - 1e-30 {
        // Crossed, potentially due to noise: just pick the average.
        0.5 * (lower_delta + upper_delta)
    } else if upper_win_loss <= 0.0 {
        // 0 is outside of the range: choose the endpoint.
        upper_delta
    } else if lower_win_loss >= 0.0 {
        lower_delta
    } else {
        // Interpolate.
        lower_delta
            + (upper_delta - lower_delta) * (0.0 - lower_win_loss)
                / (upper_win_loss - lower_win_loss)
    };

    let new_komi = f64::from(hist.rules.komi) / 2.0 + final_delta;
    hist.set_komi(old_komi);
    new_komi
}

/// Adjust komi to make the game roughly even.
///
/// Mirrors `PlayUtils::adjustKomiToEven`.
#[allow(clippy::too_many_arguments)]
pub fn adjust_komi_to_even(
    bot_b: &mut Search,
    bot_w: &mut Search,
    board: &Board,
    hist: &mut BoardHistory,
    pla: Player,
    num_visits: i32,
    other_game_props: &super::play::OtherGameProperties,
    rand: &mut Rand,
) {
    let mut score_wl_cache = Vec::new();
    let mut new_komi = get_naive_even_komi_helper(
        &mut score_wl_cache,
        bot_b,
        bot_w,
        board,
        hist,
        pla,
        i64::from(num_visits),
        other_game_props,
    );
    let lower = (new_komi * 2.0).floor() * 0.5;
    let upper = lower + 0.5;
    if rand.next_bool((new_komi - lower) / (upper - lower)) {
        new_komi = upper;
    } else {
        new_komi = lower;
    }
    hist.set_komi(round_and_clip_komi(new_komi, board));
}

/// Initialize the early game by playing pure policy moves.
///
/// Mirrors `PlayUtils::initializeGameUsingPolicy`. Samples a number of opening
/// moves from the bot's policy and plays them out, switching sides each move.
#[allow(clippy::too_many_arguments)]
pub fn initialize_game_using_policy(
    bot_b: &mut Search,
    bot_w: &mut Search,
    board: &mut Board,
    hist: &mut BoardHistory,
    pla: &mut Player,
    game_rand: &mut Rand,
    do_end_game_if_all_pass_alive: bool,
    proportion_of_board_area: f64,
    policy_init_gamma_shape: f64,
    temperature: f64,
) {
    if hist.is_game_finished {
        return;
    }

    let mean = board.x_size as f64 * board.y_size as f64 * proportion_of_board_area;
    let num_initial_moves_to_play = if (policy_init_gamma_shape - 1.0).abs() > 1e-10 {
        (game_rand.next_gamma(policy_init_gamma_shape) * (mean / policy_init_gamma_shape)).floor()
            as i32
    } else {
        (game_rand.next_exponential() * mean).floor() as i32
    };

    assert!(num_initial_moves_to_play >= 0);
    for _ in 0..num_initial_moves_to_play {
        let loc = if *pla == P_BLACK {
            get_game_initialization_move(bot_b, board, hist, *pla, game_rand, temperature)
        } else {
            get_game_initialization_move(bot_w, board, hist, *pla, game_rand, temperature)
        };

        assert!(hist.is_legal(board, loc, *pla));
        hist.make_board_move_assume_legal(board, loc, *pla);
        *pla = get_opp(*pla);

        if do_end_game_if_all_pass_alive {
            hist.end_game_if_all_pass_alive(board);
        }
        if hist.is_game_finished {
            break;
        }
    }
}

/// Estimate the lead (score difference from even) for a position.
///
/// Mirrors `PlayUtils::computeLead`.
#[allow(clippy::too_many_arguments)]
pub fn compute_lead(
    bot_b: &mut Search,
    bot_w: &mut Search,
    board: &Board,
    hist: &mut BoardHistory,
    pla: Player,
    num_visits: i32,
    other_game_props: &super::play::OtherGameProperties,
) -> f32 {
    let mut score_wl_cache = Vec::new();
    let old_komi_halfpoints = hist.rules.komi;
    let old_komi = old_komi_halfpoints as f32 / 2.0;
    let naive_komi = get_naive_even_komi_helper(
        &mut score_wl_cache,
        bot_b,
        bot_w,
        board,
        hist,
        pla,
        i64::from(num_visits),
        other_game_props,
    );

    let granularity_is_coarse =
        hist.rules.scoring_rule == ScoringRule::Area && !hist.rules.has_button;
    if !granularity_is_coarse {
        assert_eq!(hist.rules.komi, old_komi_halfpoints);
        return (f64::from(old_komi) - naive_komi) as f32;
    }

    let mut eval_win_loss = |new_komi: f64| -> f64 {
        let rounded = round_and_clip_komi(new_komi, board);
        eval_komi(
            &mut score_wl_cache,
            bot_b,
            bot_w,
            board,
            hist,
            pla,
            i64::from(num_visits),
            other_game_props,
            rounded,
        )
        .1
    };

    // If komi is exactly an integer, then we're good.
    if naive_komi == naive_komi.round() {
        assert_eq!(hist.rules.komi, old_komi_halfpoints);
        return (f64::from(old_komi) - naive_komi) as f32;
    }

    let lower = (naive_komi * 2.0).floor() * 0.5;
    let upper = lower + 0.5;

    // Average out the oscillation.
    let lower_win_loss = 0.5 * (eval_win_loss(upper) + eval_win_loss(lower - 0.5));
    let upper_win_loss = 0.5 * (eval_win_loss(upper + 0.5) + eval_win_loss(lower));

    let result = if lower_win_loss >= upper_win_loss - 1e-30 {
        0.5 * (lower + upper)
    } else {
        let mut r =
            lower + (upper - lower) * (0.0 - lower_win_loss) / (upper_win_loss - lower_win_loss);
        // Bound the result to be within lower-0.5 and upper+0.5.
        if r < lower - 0.5 {
            r = lower - 0.5;
        }
        if r > upper + 0.5 {
            r = upper + 0.5;
        }
        r
    };

    assert_eq!(hist.rules.komi, old_komi_halfpoints);
    (f64::from(old_komi) - result) as f32
}

/// Build noiseless search parameters suitable for ownership / score estimation.
///
/// Mirrors the local `getNoiselessParams` helper in `cpp/program/playutils.cpp`.
fn get_noiseless_params(old_params: &SearchParams, num_visits: i64) -> SearchParams {
    assert!(num_visits > 0);
    let mut new_params = old_params.clone();
    new_params.max_visits = num_visits;
    new_params.max_playouts = num_visits;
    new_params.root_noise_enabled = false;
    new_params.root_policy_temperature = 1.0;
    new_params.root_policy_temperature_early = 1.0;
    new_params.root_fpu_reduction_max = new_params.fpu_reduction_max;
    new_params.root_fpu_loss_prop = new_params.fpu_loss_prop;
    new_params.root_desired_per_child_visits_coeff = 0.0;
    new_params.root_num_symmetries_to_sample = 1;
    new_params.search_factor_after_one_pass = 1.0;
    new_params.search_factor_after_two_pass = 1.0;
    let max_threads = (num_visits + 7) / 8;
    if new_params.num_threads > max_threads as i32 {
        new_params.num_threads = max_threads as i32;
    }
    new_params
}

/// Run a short search and return the average tree ownership vector.
///
/// Mirrors `PlayUtils::computeOwnership` in `cpp/program/playutils.cpp`.
pub fn compute_ownership(
    bot: &mut Search,
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
    num_visits: i64,
) -> Vec<f64> {
    assert!(num_visits > 0);
    let old_always_include_owner_map = bot.always_include_owner_map;
    bot.set_always_include_owner_map(true);

    let old_params = bot.search_params.clone();
    let mut new_params = get_noiseless_params(&old_params, num_visits);
    new_params.playout_doubling_advantage_pla = C_EMPTY;
    new_params.playout_doubling_advantage = 0.0;
    new_params.conservative_pass = true;

    bot.set_params(&new_params);
    bot.set_position(pla, board, hist);
    bot.run_whole_search(pla);

    let ownership = bot.get_average_tree_ownership(None);

    bot.set_params(&old_params);
    bot.set_always_include_owner_map(old_always_include_owner_map);
    bot.clear_search();

    ownership
}

/// Evaluate a position under all 8 symmetries and return the averaged output.
///
/// Mirrors `PlayUtils::getFullSymmetryNNOutput` in `cpp/program/playutils.cpp`.
/// SGF metadata is omitted because the skeleton evaluator ignores it.
pub fn get_full_symmetry_nn_output(
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
    include_owner_map: bool,
    nn_eval: &kata_nn::eval::NnEvaluator,
) -> NNOutput {
    let mut outputs = Vec::with_capacity(NUM_SYMMETRIES as usize);
    for sym in 0..NUM_SYMMETRIES {
        let mut buf = NNResultBuf::new();
        let nn_input_params = MiscNNInputParams {
            symmetry: sym,
            ..MiscNNInputParams::default()
        };
        nn_eval.evaluate(
            board,
            hist,
            pla,
            &nn_input_params,
            &mut buf,
            true,
            include_owner_map,
        );
        let arc = buf
            .result
            .expect("full symmetry evaluation should produce a result");
        outputs.push((*arc).clone());
    }
    NNOutput::average(&outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kata_game::board::{C_WALL, P_WHITE};

    #[test]
    fn test_round_and_clip_komi() {
        let board = Board::new(19, 19);
        assert_eq!(round_and_clip_komi(7.3, &board), 7.5);
        assert_eq!(round_and_clip_komi(7.2, &board), 7.0);
        assert_eq!(round_and_clip_komi(-3.0, &board), -3.0);
        let huge = f64::from(board.x_size * board.y_size) + f64::from(KOMI_CLIP_RADIUS) + 10.0;
        let clipped = round_and_clip_komi(huge, &board);
        assert!(clipped <= huge as f32);
        assert!(Rules::komi_is_int_or_half_int(clipped));
    }

    #[test]
    fn test_choose_extra_black_and_komi_no_handicap() {
        let mut rand = Rand::new_from_seed("test-no-handicap");
        let e =
            choose_extra_black_and_komi(7.5, 0.5, 1.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 19.0, &mut rand);
        assert_eq!(e.extra_black, 0);
        assert_eq!(e.komi_mean, 7.5);
        assert!(e.allow_integer);
    }

    #[test]
    fn test_choose_extra_black_and_komi_fixed_handicap() {
        let mut rand = Rand::new_from_seed("test-fixed");
        let e =
            choose_extra_black_and_komi(7.5, 0.5, 1.0, 1.0, 3, 0.0, 0.0, 0.0, 0.0, 19.0, &mut rand);
        assert_eq!(e.extra_black, 3);
    }

    #[test]
    fn test_set_komi_without_noise() {
        let board = Board::new(19, 19);
        let mut hist = BoardHistory::new(board, P_BLACK, Rules::default(), 0);
        let e = ExtraBlackAndKomi {
            komi_mean: 6.75,
            ..Default::default()
        };
        set_komi_without_noise(&e, &mut hist);
        assert_eq!(hist.rules.komi, 14);
    }

    #[test]
    fn test_set_komi_with_noise_deterministic() {
        let board = Board::new(19, 19);
        let mut hist = BoardHistory::new(board, P_BLACK, Rules::default(), 0);
        let mut rand = Rand::new_from_seed("komi-noise");
        let e = ExtraBlackAndKomi {
            komi_mean: 7.5,
            komi_stdev: 0.0,
            allow_integer: true,
            ..Default::default()
        };
        set_komi_with_noise(&e, &mut hist, &mut rand);
        assert!(Rules::komi_is_int_or_half_int(hist.rules.komi as f32 / 2.0));
    }

    #[test]
    fn test_get_search_factor() {
        let params = SearchParams {
            win_loss_utility_factor: 1.0,
            ..Default::default()
        };

        // Not enough history: factor stays 1.0.
        assert_eq!(
            get_search_factor(0.5, 2.0, &params, &[0.9, 0.95], P_BLACK),
            1.0
        );

        // Black is winning heavily in the last three values.
        let recent = vec![-0.9, -0.95, -0.99];
        let factor = get_search_factor(0.5, 2.0, &params, &recent, P_BLACK);
        assert!(factor > 1.0);
        assert!(factor <= 2.0);

        // White is winning heavily.
        let recent = vec![0.9, 0.95, 0.99];
        let factor = get_search_factor(0.5, 2.0, &params, &recent, P_WHITE);
        assert!(factor > 1.0);
        assert!(factor <= 2.0);

        // Values below threshold: factor stays 1.0.
        let recent = vec![0.1, 0.1, 0.1];
        assert_eq!(get_search_factor(0.5, 2.0, &params, &recent, P_WHITE), 1.0);
    }

    #[test]
    fn test_choose_random_legal_move_on_empty_board() {
        let board = Board::new(5, 5);
        let hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let mut rand = Rand::new_from_seed("empty-board");
        let m = choose_random_legal_move(&board, &hist, P_BLACK, &mut rand, NULL_LOC);
        assert_ne!(m, NULL_LOC);
        assert!(hist.is_legal(&board, m, P_BLACK));

        let mut buf = [NULL_LOC; 10];
        let count = choose_random_legal_moves(&board, &hist, P_BLACK, &mut rand, &mut buf);
        assert_eq!(count, 10);
        for &loc in &buf {
            assert!(hist.is_legal(&board, loc, P_BLACK));
        }

        // Banning the chosen move should never return that move on a fresh call.
        let banned = m;
        let m2 = choose_random_legal_move(&board, &hist, P_BLACK, &mut rand, banned);
        assert_ne!(m2, banned);
        assert!(hist.is_legal(&board, m2, P_BLACK));
    }

    #[test]
    fn test_place_fixed_handicap() {
        let mut board = Board::new(19, 19);
        place_fixed_handicap(&mut board, 9).unwrap();
        let mut black_count = 0;
        for y in 0..19 {
            for x in 0..19 {
                let loc = location::get_loc(x, y, 19) as usize;
                if board.colors[loc] == P_BLACK {
                    black_count += 1;
                }
            }
        }
        assert_eq!(black_count, 9);

        let mut small = Board::new(5, 5);
        assert!(place_fixed_handicap(&mut small, 5).is_err());
    }

    #[test]
    fn test_gen_random_rules_deterministic() {
        let mut rand = Rand::new_from_seed("rules");
        let rules = gen_random_rules(&mut rand);
        assert!([KoRule::Simple, KoRule::Positional, KoRule::Situational].contains(&rules.ko_rule));
        assert!([ScoringRule::Area, ScoringRule::Territory].contains(&rules.scoring_rule));
        assert!([TaxRule::None, TaxRule::Seki, TaxRule::All].contains(&rules.tax_rule));
    }

    #[test]
    fn test_compute_anticipated_statuses_simple_alive() {
        let s = ".....\n.XX..\n.XO..\n.XX..\n.....\n";
        let board = Board::parse_board(5, 5, s, '\n').unwrap();
        let hist = BoardHistory::new(board.clone(), P_WHITE, Rules::default(), 0);
        let alive = compute_anticipated_statuses_simple(&board, &hist);
        for y in 0..5 {
            for x in 0..5 {
                let loc = location::get_loc(x, y, 5) as usize;
                if board.colors[loc] == C_EMPTY || board.colors[loc] == C_WALL {
                    continue;
                }
                // All stones in this closed position are treated as alive
                // under the simple area heuristic.
                assert!(alive[loc], "loc {} should be alive", loc);
            }
        }
    }

    #[test]
    fn test_compute_anticipated_statuses_simple_no_result() {
        let mut board = Board::new(5, 5);
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        hist.is_no_result = true;
        hist.is_game_finished = true;
        board.set_stone(location::get_loc(2, 2, 5), P_BLACK);
        let alive = compute_anticipated_statuses_simple(&board, &hist);
        assert!(alive[location::get_loc(2, 2, 5) as usize]);
    }
}

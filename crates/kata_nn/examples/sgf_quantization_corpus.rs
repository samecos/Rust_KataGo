//! CPU-only, model-independent PTQ corpus preparation from local SGFs.
//!
//! cargo run -p kata_nn --example sgf_quantization_corpus -- SGF_DIRECTORY
//!   --output NEW_DIRECTORY --seed rustgo-ptq-v1
//! Optional: --games-per-split 128,128,128 --positions-per-game 32
//!           --assume-rules chinese --assume-komi 7.5
//!
//! Missing RU/KM are rejected unless their defaults were explicitly supplied.
//! Only the current Worker Chinese rules profile is admitted. Initial stones,
//! either initial player and half-point komi are preserved in complete wire
//! position descriptions. Nonroot setup is limited to move-free AB nodes before
//! the first move; later setup and nonroot player/rule changes are rejected.
//! Separate legacy fixtures contain ONLY empty-board,
//! initial-Black, Chinese-7.5 positions, because the old Python fixture reader
//! hardcodes those fields. This tool does not download data, load a model,
//! evaluate labels, expand symmetry samples, or certify numerical accuracy.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use kata_data::sgf::{Sgf, SgfNode};
use kata_game::board::{
    Board, C_EMPTY, Loc, P_BLACK, P_WHITE, PASS_LOC, Player, get_opp, location,
};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use kata_game::symmetry::get_sym_loc_xy;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const SPLITS: [&str; 3] = ["calibration", "selection", "holdout"];
const FIRST_PLY: usize = 8;
const HOLDOUT_GAMES: usize = 128;
const HOLDOUT_POSITIONS: usize = 4096;

#[derive(Debug, Clone, Serialize)]
struct Options {
    input: PathBuf,
    output: PathBuf,
    seed: String,
    games_per_split: [usize; 3],
    positions_per_game: usize,
    assume_rules: Option<String>,
    assume_komi: Option<f32>,
}

fn options(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let mut args = args.into_iter();
    let mut result = Options {
        input: PathBuf::new(),
        output: PathBuf::new(),
        seed: "rustgo-ptq-v1".into(),
        games_per_split: [128; 3],
        positions_per_game: 32,
        assume_rules: None,
        assume_komi: None,
    };
    let mut seen = BTreeSet::new();
    while let Some(key) = args.next() {
        if !key.starts_with('-') && result.input.as_os_str().is_empty() {
            result.input = key.into();
            continue;
        }
        if !seen.insert(key.clone()) {
            return Err(format!("duplicate argument {key}"));
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value after {key}"))?;
        match key.as_str() {
            "--input" if result.input.as_os_str().is_empty() => result.input = value.into(),
            "--output" => result.output = value.into(),
            "--seed" if !value.is_empty() => result.seed = value,
            "--games-per-split" => {
                let counts: Vec<usize> = value
                    .split(',')
                    .map(|s| s.parse::<usize>())
                    .collect::<Result<_, _>>()
                    .map_err(|e| format!("games-per-split: {e}"))?;
                result.games_per_split = counts
                    .try_into()
                    .map_err(|_| "games-per-split requires calibration,selection,holdout counts")?;
                if result.games_per_split.contains(&0)
                    || result.games_per_split.iter().any(|&n| n > 1_000_000)
                {
                    return Err("games-per-split counts must be in 1..=1000000".into());
                }
            }
            "--positions-per-game" => {
                result.positions_per_game = value
                    .parse()
                    .map_err(|e| format!("positions-per-game: {e}"))?;
                if !(2..=10000).contains(&result.positions_per_game) {
                    return Err("positions-per-game must be in 2..=10000".into());
                }
            }
            "--assume-rules" => {
                let parsed = Rules::parse_rules(&value).map_err(|e| e.to_string())?;
                if parsed != chinese()? {
                    return Err("assume-rules must be the Chinese Worker profile".into());
                }
                result.assume_rules = Some("chinese".into());
            }
            "--assume-komi" => {
                let value = value
                    .parse::<f32>()
                    .map_err(|e| format!("assume-komi: {e}"))?;
                check_komi(value)?;
                result.assume_komi = Some(value);
            }
            _ => return Err(format!("unknown, duplicate or invalid argument {key}")),
        }
    }
    if result.input.as_os_str().is_empty() || result.output.as_os_str().is_empty() {
        return Err("usage: sgf_quantization_corpus SGF_DIRECTORY --output NEW_DIRECTORY [--seed STRING] [--games-per-split 128,128,128] [--positions-per-game 32] [--assume-rules chinese] [--assume-komi 7.5]".into());
    }
    Ok(result)
}

#[derive(Debug, Clone, Serialize)]
struct Source {
    path: String,
    sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Rejection {
    source: Source,
    code: String,
    detail: String,
}

#[derive(Debug)]
struct Invalid {
    code: &'static str,
    detail: String,
}

fn invalid(code: &'static str, detail: impl Into<String>) -> Invalid {
    Invalid {
        code,
        detail: detail.into(),
    }
}

type Movement = (Player, i32);

/// Only replay semantics participate in game identity, never comments, player
/// names, result tags or source filenames. Vertices are Worker row-major 0..360.
#[derive(Debug, Clone, Serialize)]
struct GameSemantics {
    board_size: usize,
    rules: Value,
    komi_half_points: i32,
    initial_player: Player,
    initial_stones: Vec<Movement>,
    moves: Vec<Movement>,
}

#[derive(Debug, Clone, Serialize)]
struct WireMove {
    color: Player,
    vertex: i32,
}

impl From<&Movement> for WireMove {
    fn from(&(color, vertex): &Movement) -> Self {
        Self { color, vertex }
    }
}

#[derive(Debug, Clone, Serialize)]
struct WirePosition {
    board_size: usize,
    rules: &'static str,
    komi: f32,
    initial_player: Player,
    next_player: Player,
    initial_stones: Vec<WireMove>,
    moves: Vec<WireMove>,
}

#[derive(Debug, Clone, Serialize)]
struct PositionRecord {
    name: String,
    game_id: String,
    split: String,
    ply: usize,
    phase: &'static str,
    semantic_position_sha256: String,
    /// Conservative leakage key: current stones/player/rules/komi, without
    /// history or ko rights. This can discard distinct histories, never merges
    /// them into one request or claims they are numerically interchangeable.
    board_state_sha256: String,
    position: WirePosition,
    parameters: Value,
}

#[derive(Debug, Clone, Serialize)]
struct GameRecord {
    game_id: String,
    sources: Vec<Source>,
    split: Option<String>,
    assumptions: Vec<String>,
    branch_policy: &'static str,
    selected_branch: Vec<usize>,
    canonical_symmetry: i32,
    canonical_semantics: GameSemantics,
    semantics: GameSemantics,
    sampled_plies: Vec<usize>,
    retained_plies: Vec<usize>,
    legacy_fixture_exclusions: Vec<String>,
}

struct Candidate {
    record: GameRecord,
    positions: Vec<PositionRecord>,
}

struct Corpus {
    games: Vec<GameRecord>,
    requests: [Vec<PositionRecord>; 3],
    rejections: Vec<Rejection>,
    duplicates: Vec<Value>,
}

fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn json_bytes(value: &impl Serialize) -> Vec<u8> {
    serde_json::to_vec(value).expect("only finite, validated corpus values are serialized")
}

fn chinese() -> Result<Rules, String> {
    Rules::parse_rules("chinese").map_err(|e| e.to_string())
}

fn check_komi(komi: f32) -> Result<(), String> {
    if !komi.is_finite() || !(-150.0..=150.0).contains(&komi) || (komi * 2.0).round() != komi * 2.0
    {
        return Err("Worker komi must be a finite half-integer in -150..=150".into());
    }
    Ok(())
}

fn vertex(loc: Loc) -> i32 {
    if loc == PASS_LOC {
        -1
    } else {
        location::get_y(loc, 19) * 19 + location::get_x(loc, 19)
    }
}

fn loc(vertex: i32) -> Loc {
    if vertex == -1 {
        PASS_LOC
    } else {
        location::get_loc(vertex % 19, vertex / 19, 19)
    }
}

fn transform_vertex(vertex: i32, symmetry: i32) -> i32 {
    if vertex == -1 {
        return -1;
    }
    let (x, y) = get_sym_loc_xy(vertex % 19, vertex / 19, 19, 19, symmetry);
    y * 19 + x
}

fn transform_semantics(semantics: &GameSemantics, symmetry: i32) -> GameSemantics {
    let mut result = semantics.clone();
    for (_, vertex) in result.initial_stones.iter_mut().chain(&mut result.moves) {
        *vertex = transform_vertex(*vertex, symmetry);
    }
    result
        .initial_stones
        .sort_unstable_by_key(|&(color, vertex)| (vertex, color));
    result
}

fn canonical_semantics(semantics: &GameSemantics) -> (i32, GameSemantics) {
    (0..8)
        .map(|symmetry| {
            let transformed = transform_semantics(semantics, symmetry);
            (json_bytes(&transformed), symmetry, transformed)
        })
        .min_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, sym, record)| (sym, record))
        .unwrap()
}

fn semantic_hash(domain: &str, semantics: &GameSemantics) -> String {
    let (_, canonical) = canonical_semantics(semantics);
    sha(&json_bytes(&json!({"schema":domain,"semantics":canonical})))
}

fn board_state_hash(board: &Board, next: Player, rules: &Value) -> String {
    let mut variants = Vec::with_capacity(8);
    for symmetry in 0..8 {
        let mut stones = vec![C_EMPTY; 361];
        for vertex in 0..361 {
            stones[transform_vertex(vertex, symmetry) as usize] =
                board.colors[loc(vertex) as usize];
        }
        variants.push(json_bytes(&json!({"schema":"rustgo-board-leakage-v1",
            "stones":stones,"next_player":next,"rules":rules})));
    }
    sha(&variants.into_iter().min().unwrap())
}

/// Native Sgf::parse reads one tree. Reject collections/trailing data instead
/// of silently dropping other games. This only checks framing, never parses
/// moves from text; escaped brackets and parentheses inside properties are inert.
fn single_tree(text: &str) -> Result<(), Invalid> {
    let (mut in_value, mut escaped, mut depth, mut roots) = (false, false, 0usize, 0usize);
    for byte in text.bytes() {
        if in_value {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b']' {
                in_value = false;
            }
            continue;
        }
        match byte {
            b'[' if depth > 0 => in_value = true,
            b'(' => {
                if depth == 0 {
                    roots += 1;
                }
                depth += 1;
            }
            b')' if depth > 0 => depth -= 1,
            b if depth == 0 && !b.is_ascii_whitespace() => {
                return Err(invalid(
                    "sgf_framing",
                    "unexpected content outside SGF tree",
                ));
            }
            _ => {}
        }
    }
    if roots != 1 || depth != 0 || in_value {
        return Err(invalid(
            "sgf_framing",
            "exactly one balanced SGF game tree per file is required",
        ));
    }
    Ok(())
}

fn longest_branch(sgf: &Sgf) -> Result<(Vec<&SgfNode>, Vec<usize>), Invalid> {
    let mut tree = sgf;
    let mut nodes = Vec::new();
    let mut branch = Vec::new();
    loop {
        // Keep the old get_moves_helper rejection of empty selected tree
        // sequences, even when an empty wrapper has a nonempty child.
        if tree.nodes.is_empty() {
            return Err(invalid(
                "sgf_parse",
                "empty tree sequence on selected branch",
            ));
        }
        nodes.extend(tree.nodes.iter());
        if tree.children.is_empty() {
            break;
        }
        // Same tie rule as Sgf::get_moves: first child wins equal depths.
        let mut index = 0;
        for i in 1..tree.children.len() {
            if tree.children[i].depth() > tree.children[index].depth() {
                index = i;
            }
        }
        branch.push(index);
        tree = &tree.children[index];
    }
    Ok((nodes, branch))
}

fn replay_start(
    semantics: &GameSemantics,
    rules: Rules,
) -> Result<(Board, BoardHistory, Player), Invalid> {
    let mut board = Board::new(19, 19);
    let mut occupied = BTreeSet::new();
    for &(color, vertex) in &semantics.initial_stones {
        if !matches!(color, P_BLACK | P_WHITE)
            || !(0..361).contains(&vertex)
            || !occupied.insert(vertex)
        {
            return Err(invalid(
                "invalid_initial_stones",
                "initial stones must be distinct Black/White board points",
            ));
        }
        board.colors[loc(vertex) as usize] = color;
    }
    board.regen_chains_from_colors();
    if semantics
        .initial_stones
        .iter()
        .any(|&(_, vertex)| board.get_num_liberties(loc(vertex)) <= 0)
    {
        return Err(invalid(
            "invalid_initial_stones",
            "initial chain has no liberties",
        ));
    }
    let mut history = BoardHistory::new(board.clone(), semantics.initial_player, rules, 0);
    history.set_assume_multiple_starting_black_moves_are_handicap(false);
    Ok((board, history, semantics.initial_player))
}

fn sample_plies(available: &[(usize, Player)], limit: usize) -> Vec<usize> {
    let count = limit.min(available.len());
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![available[0].0];
    }
    let mut indices: BTreeSet<usize> = (0..count)
        .map(|i| i * (available.len() - 1) / (count - 1))
        .collect();
    let players: BTreeSet<_> = indices.iter().map(|&i| available[i].1).collect();
    if players.len() == 1
        && let Some(opposite) = (0..available.len()).find(|i| !players.contains(&available[*i].1))
    {
        // Preserve both-player coverage even when evenly spaced plies all have
        // the same parity. An adjacent replacement is still uniformly nearby.
        let replace = *indices
            .iter()
            .min_by_key(|&&i| i.abs_diff(opposite))
            .unwrap();
        indices.remove(&replace);
        indices.insert(opposite);
    }
    indices.into_iter().map(|i| available[i].0).collect()
}

fn request_parameters() -> Value {
    json!({"symmetry":0,"policy_temperature":1.0,"policy_optimism":0.0,
        "draw_equivalent_wins_for_white":0.5,"playout_doubling_advantage":0.0,
        "max_history":10000,"include_ownership":true,"skip_cache":true,
        "conservative_pass":false,"enable_passing_hacks":false,
        "always_compute_pass_alive":false,"exclude_territory_adjacent_to_atari":false,
        "avoid_mytdagger_hack":false,"allow_terminal_search_history":false,"force_non_terminal":false})
}

fn parse_game(bytes: &[u8], source: Source, options: &Options) -> Result<Candidate, Invalid> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| invalid("encoding", e.to_string()))?
        .trim_start_matches('\u{feff}');
    single_tree(text)?;
    let tree = Sgf::parse(text).map_err(|e| invalid("sgf_parse", e.to_string()))?;
    let (nodes, selected_branch) = longest_branch(&tree)?;
    let root = nodes
        .first()
        .ok_or_else(|| invalid("empty_sgf", "no nodes"))?;
    let size = tree
        .get_xy_size()
        .map_err(|e| invalid("non_19_board", e.to_string()))?;
    if (size.x, size.y) != (19, 19) {
        return Err(invalid(
            "non_19_board",
            format!("board is {}x{}", size.x, size.y),
        ));
    }
    if root.has_property("GM")
        && root
            .get_single_property("GM")
            .map_err(|e| invalid("sgf_parse", e.to_string()))?
            != "1"
    {
        return Err(invalid("unsupported_game", "GM must be 1 (Go)"));
    }
    // CompactSgf only collects outer-root placements. Extract both inputs from
    // the same selected branch so a child-tree's initial AB cannot be lost.
    let mut placements = Vec::new();
    let mut extracted_moves = Vec::new();
    let mut has_premove_ab = false;
    for (index, node) in nodes.iter().enumerate() {
        if index > 0
            && ["AW", "AE", "PL", "RU", "KM", "SZ", "HA"]
                .iter()
                .any(|key| node.has_property(key))
        {
            return Err(invalid(
                "midgame_setup_or_rules",
                format!("selected node {index} changes setup/player/rules"),
            ));
        }
        let mut node_moves = Vec::new();
        node.accum_moves(&mut node_moves, 19, 19)
            .map_err(|e| invalid("sgf_parse", e.to_string()))?;
        if node_moves.len() > 1 {
            return Err(invalid(
                "multiple_moves_in_node",
                format!("selected node {index} contains multiple moves"),
            ));
        }
        // Preserve the existing root setup+move interpretation. Only newly
        // admitted nonroot setup must occupy a move-free node.
        if index > 0 && node.has_placements() && !node_moves.is_empty() {
            return Err(invalid(
                "midgame_setup_or_rules",
                format!("selected node {index} combines setup and a move"),
            ));
        }
        if index > 0 && node.has_property("AB") {
            if !extracted_moves.is_empty() {
                return Err(invalid(
                    "midgame_setup_or_rules",
                    format!("selected node {index} places stones after the first move"),
                ));
            }
            has_premove_ab = true;
        }
        node.accum_placements(&mut placements, 19, 19)
            .map_err(|e| invalid("sgf_parse", e.to_string()))?;
        extracted_moves.extend(node_moves);
    }
    if extracted_moves.len() > 10000 {
        return Err(invalid(
            "too_many_moves",
            "Worker accepts at most 10000 moves",
        ));
    }
    let mut assumptions = Vec::new();
    let chinese = chinese().map_err(|e| invalid("rules", e))?;
    let mut rules = if root.has_property("RU") {
        root.get_rules_from_ru_tag_or_fail()
            .map_err(|e| invalid("unsupported_rules", e.to_string()))?
    } else if options.assume_rules.is_some() {
        assumptions.push("missing RU: explicitly assumed chinese".into());
        chinese
    } else {
        return Err(invalid(
            "missing_rules",
            "RU is absent; --assume-rules is required to assume a profile",
        ));
    };
    let komi = if root.has_property("KM") {
        root.get_komi_or_fail()
            .map_err(|e| invalid("unsupported_komi", e.to_string()))?
    } else if let Some(komi) = options.assume_komi {
        assumptions.push(format!("missing KM: explicitly assumed {komi}"));
        komi
    } else {
        return Err(invalid(
            "missing_komi",
            "KM is absent; --assume-komi is required to assume a value",
        ));
    };
    check_komi(komi).map_err(|e| invalid("unsupported_komi", e))?;
    rules.set_komi(chinese.komi_f32());
    if rules != chinese {
        return Err(invalid(
            "unsupported_rules",
            "current Worker requires the Chinese rules profile",
        ));
    }
    rules.set_komi(komi);
    let initial_player = if root.has_property("PL") {
        let player = root.get_pl_specified_color();
        if !matches!(player, P_BLACK | P_WHITE) {
            return Err(invalid(
                "invalid_initial_player",
                "PL is not Black or White",
            ));
        }
        player
    } else if let Some(first) = extracted_moves.first() {
        first.pla
    } else if !placements.is_empty() && placements.iter().all(|m| m.pla == P_BLACK) {
        P_WHITE
    } else {
        P_BLACK
    };
    let mut initial_stones: Vec<_> = placements.iter().map(|m| (m.pla, vertex(m.loc))).collect();
    initial_stones.sort_unstable_by_key(|&(color, vertex)| (vertex, color));
    let semantics = GameSemantics {
        board_size: 19,
        rules: rules.to_json(),
        komi_half_points: rules.komi,
        initial_player,
        initial_stones,
        moves: extracted_moves
            .iter()
            .map(|m| (m.pla, vertex(m.loc)))
            .collect(),
    };
    let (mut board, mut history, mut next) = replay_start(&semantics, rules)?;
    // Keep the old root-only interpretation intact. For the newly supported
    // prefix, an explicit handicap must describe the complete initial Black
    // setup; it must not cause us to invent stones or change the next player.
    if has_premove_ab && root.has_property("HA") {
        let handicap = tree
            .get_handicap_value()
            .map_err(|e| invalid("invalid_initial_handicap", e.to_string()))?;
        let black_stones = semantics
            .initial_stones
            .iter()
            .filter(|&&(color, _)| color == P_BLACK)
            .count();
        if handicap < 0 || handicap as usize != black_stones {
            return Err(invalid(
                "invalid_initial_handicap",
                "root HA must equal the initial Black stone count when nonroot AB is used",
            ));
        }
    }
    let mut available = Vec::new();
    for (index, &(color, vertex)) in semantics.moves.iter().enumerate() {
        if history.is_game_finished {
            return Err(invalid(
                "move_after_terminal",
                format!("ply {} follows a terminal position", index + 1),
            ));
        }
        if color != next || !history.is_legal(&board, loc(vertex), color) {
            return Err(invalid(
                "illegal_or_out_of_turn",
                format!("illegal or out-of-turn move at ply {}", index + 1),
            ));
        }
        history.make_board_move_assume_legal(&mut board, loc(vertex), color);
        next = get_opp(next);
        if index + 1 >= FIRST_PLY && !history.is_game_finished {
            available.push((index + 1, next));
        }
    }
    let sampled_plies = sample_plies(&available, options.positions_per_game);
    if sampled_plies.is_empty() {
        return Err(invalid(
            "no_sample_positions",
            "no legal nonterminal position at or after ply 8",
        ));
    }
    let (canonical_symmetry, canonical) = canonical_semantics(&semantics);
    let game_id = semantic_hash("rustgo-sgf-game-v1", &semantics);
    let selected: BTreeSet<_> = sampled_plies.iter().copied().collect();
    let (mut board, mut history, mut next) = replay_start(&semantics, rules)?;
    let mut positions = Vec::new();
    for (index, &(color, vertex)) in semantics.moves.iter().enumerate() {
        history.make_board_move_assume_legal(&mut board, loc(vertex), color);
        next = get_opp(next);
        let ply = index + 1;
        if !selected.contains(&ply) {
            continue;
        }
        let mut prefix = semantics.clone();
        prefix.moves.truncate(ply);
        positions.push(PositionRecord {
            name: format!("game-{game_id}-ply-{ply:04}"),
            game_id: game_id.clone(),
            split: String::new(),
            ply,
            phase: if ply * 3 <= semantics.moves.len() {
                "early"
            } else if ply * 3 <= semantics.moves.len() * 2 {
                "middle"
            } else {
                "late"
            },
            semantic_position_sha256: semantic_hash("rustgo-sgf-position-v1", &prefix),
            board_state_sha256: board_state_hash(&board, next, &semantics.rules),
            position: WirePosition {
                board_size: 19,
                rules: "chinese",
                komi,
                initial_player,
                next_player: next,
                initial_stones: semantics
                    .initial_stones
                    .iter()
                    .map(WireMove::from)
                    .collect(),
                moves: prefix.moves.iter().map(WireMove::from).collect(),
            },
            parameters: request_parameters(),
        });
    }
    let mut legacy_fixture_exclusions = Vec::new();
    if !semantics.initial_stones.is_empty() {
        legacy_fixture_exclusions
            .push("initial stones are not supported by the legacy fixture reader".into());
    }
    if initial_player != P_BLACK {
        legacy_fixture_exclusions.push("initial player is not Black".into());
    }
    if komi != 7.5 {
        legacy_fixture_exclusions.push("komi is not 7.5".into());
    }
    Ok(Candidate {
        record: GameRecord {
            game_id,
            sources: vec![source],
            split: None,
            assumptions,
            branch_policy: "native longest node-depth branch, first child on ties",
            selected_branch,
            canonical_symmetry,
            canonical_semantics: canonical,
            semantics,
            sampled_plies,
            retained_plies: Vec::new(),
            legacy_fixture_exclusions,
        },
        positions,
    })
}

fn assemble(
    mut candidates: Vec<Candidate>,
    rejections: Vec<Rejection>,
    options: &Options,
) -> Corpus {
    // Source sorting chooses a reproducible representative without affecting a
    // game's split. Original and symmetry-transformed game copies stay together.
    candidates.sort_by(|a, b| a.record.sources[0].path.cmp(&b.record.sources[0].path));
    let mut unique: BTreeMap<String, Candidate> = BTreeMap::new();
    let mut duplicates = Vec::new();
    for candidate in candidates {
        if let Some(first) = unique.get_mut(&candidate.record.game_id) {
            duplicates.push(
                json!({"kind":"duplicate_game","game_id":candidate.record.game_id,
                "kept_source":first.record.sources[0],"dropped_source":candidate.record.sources[0],
                "dropped_source_assumptions":candidate.record.assumptions,
                "normalization":"full game replay semantics across all 8 spatial symmetries"}),
            );
            first.record.sources.extend(candidate.record.sources);
        } else {
            unique.insert(candidate.record.game_id.clone(), candidate);
        }
    }
    let rank = |game_id: &str| {
        sha(&json_bytes(
            &json!({"schema":"rustgo-corpus-split-v1","seed":options.seed,"game_id":game_id}),
        ))
    };
    let mut candidates: Vec<_> = unique.into_values().collect();
    candidates.sort_by_key(|g| (rank(&g.record.game_id), g.record.game_id.clone()));
    let mut requests: [Vec<PositionRecord>; 3] = std::array::from_fn(|_| Vec::new());
    let mut position_owners: BTreeMap<String, Value> = BTreeMap::new();
    let mut board_owners: BTreeMap<String, Value> = BTreeMap::new();
    let mut games = Vec::new();
    let first_end = options.games_per_split[0];
    let second_end = first_end + options.games_per_split[1];
    let third_end = second_end + options.games_per_split[2];
    for (index, mut candidate) in candidates.into_iter().enumerate() {
        let split_index = if index < first_end {
            Some(0)
        } else if index < second_end {
            Some(1)
        } else if index < third_end {
            Some(2)
        } else {
            None
        };
        if let Some(split_index) = split_index {
            let split = SPLITS[split_index];
            candidate.record.split = Some(split.into());
            for mut position in candidate.positions {
                let collision = position_owners.get(&position.semantic_position_sha256)
                    .map(|owner| ("same canonical history", owner))
                    .or_else(|| board_owners.get(&position.board_state_sha256).map(|owner| ("same board/player/rules/komi; conservative history-independent exclusion", owner)));
                if let Some((reason, owner)) = collision {
                    duplicates.push(json!({"kind":"duplicate_position","reason":reason,"game_id":position.game_id,
                        "ply":position.ply,"split":split,"semantic_position_sha256":position.semantic_position_sha256,
                        "board_state_sha256":position.board_state_sha256,"kept":owner}));
                    continue;
                }
                position.split = split.into();
                let owner = json!({"game_id":position.game_id,"ply":position.ply,"split":split});
                position_owners.insert(position.semantic_position_sha256.clone(), owner.clone());
                board_owners.insert(position.board_state_sha256.clone(), owner);
                candidate.record.retained_plies.push(position.ply);
                requests[split_index].push(position);
            }
        }
        games.push(candidate.record);
    }
    Corpus {
        games,
        requests,
        rejections,
        duplicates,
    }
}

fn discover(root: &Path) -> Result<Vec<PathBuf>, String> {
    if !root.is_dir() {
        return Err(format!("input is not a directory: {}", root.display()));
    }
    let mut files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)
            .map_err(|e| format!("read directory {}: {e}", directory.display()))?
        {
            let entry = entry.map_err(|e| e.to_string())?;
            let kind = entry.file_type().map_err(|e| e.to_string())?;
            // Symlinked files are recorded/rejected below; directories are not
            // traversed, avoiding cycles and silently importing external data.
            if kind.is_dir() {
                pending.push(entry.path());
            } else if entry
                .path()
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("sgf"))
            {
                files.push(entry.path());
            }
        }
    }
    files.sort();
    Ok(files)
}

fn read_corpus(options: &Options) -> Result<Corpus, String> {
    let files = discover(&options.input)?;
    let mut candidates = Vec::new();
    let mut rejections = Vec::new();
    for path in files {
        let mut source = Source {
            path: path
                .strip_prefix(&options.input)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/"),
            sha256: None,
        };
        if fs::symlink_metadata(&path)
            .map_err(|e| e.to_string())?
            .file_type()
            .is_symlink()
        {
            rejections.push(Rejection {
                source,
                code: "symlink_not_followed".into(),
                detail: "source SGF symlinks are not followed".into(),
            });
            continue;
        }
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(err) => {
                rejections.push(Rejection {
                    source,
                    code: "read_error".into(),
                    detail: err.to_string(),
                });
                continue;
            }
        };
        source.sha256 = Some(sha(&bytes));
        match parse_game(&bytes, source.clone(), options) {
            Ok(game) => candidates.push(game),
            Err(error) => rejections.push(Rejection {
                source,
                code: error.code.into(),
                detail: error.detail,
            }),
        }
    }
    Ok(assemble(candidates, rejections, options))
}

fn readiness(corpus: &Corpus, options: &Options) -> Value {
    let mut counts = Vec::new();
    let mut reasons = Vec::new();
    for (index, split) in SPLITS.iter().enumerate() {
        let games = corpus
            .games
            .iter()
            .filter(|g| g.split.as_deref() == Some(split) && !g.retained_plies.is_empty())
            .count();
        let assigned = corpus
            .games
            .iter()
            .filter(|g| g.split.as_deref() == Some(split))
            .count();
        let positions = corpus.requests[index].len();
        let distinct: BTreeSet<_> = corpus.requests[index]
            .iter()
            .map(|p| p.semantic_position_sha256.as_str())
            .collect();
        assert_eq!(positions, distinct.len());
        counts.push(json!({"split":split,"requested_games":options.games_per_split[index],"assigned_games":assigned,
            "games_with_retained_positions":games,"distinct_semantic_positions":positions,
            "black_to_play":corpus.requests[index].iter().filter(|p| p.position.next_player == P_BLACK).count(),
            "white_to_play":corpus.requests[index].iter().filter(|p| p.position.next_player == P_WHITE).count(),
            "recorded_game_first_third":corpus.requests[index].iter().filter(|p| p.phase == "early").count(),
            "recorded_game_middle_third":corpus.requests[index].iter().filter(|p| p.phase == "middle").count(),
            "recorded_game_last_third":corpus.requests[index].iter().filter(|p| p.phase == "late").count()}));
        if games < options.games_per_split[index] {
            reasons.push(format!(
                "{split}: only {games}/{} requested games retain positions",
                options.games_per_split[index]
            ));
        }
        if index == 2 {
            if games < HOLDOUT_GAMES {
                reasons.push(format!(
                    "holdout requires at least {HOLDOUT_GAMES} independent games, got {games}"
                ));
            }
            if positions < HOLDOUT_POSITIONS {
                reasons.push(format!("holdout requires at least {HOLDOUT_POSITIONS} distinct positions, got {positions}"));
            }
        }
    }
    json!({"status":if reasons.is_empty(){"READY"}else{"NOT_READY"},"scope":"corpus composition only; accuracy, search-leaf coverage and playing strength are not validated",
        "holdout_minimum_games":HOLDOUT_GAMES,"holdout_minimum_distinct_positions":HOLDOUT_POSITIONS,
        "symmetry_multiplicity_counted":false,"splits":counts,"reasons":reasons})
}

fn write_new(output: &Path, name: &str, bytes: &[u8]) -> Result<Value, String> {
    let path = output.join(name);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| format!("create {}: {e}", path.display()))?;
    file.write_all(bytes)
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(json!({"file":name,"sha256":sha(bytes),"bytes":bytes.len()}))
}

fn lines<T: Serialize>(records: &[T]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for record in records {
        bytes.extend(json_bytes(record));
        bytes.push(b'\n');
    }
    bytes
}

fn export(corpus: &Corpus, options: &Options) -> Result<Value, String> {
    // create_dir (not create_dir_all) refuses an existing output, including an
    // empty directory. Partial failures leave evidence and are never overwritten.
    fs::create_dir(&options.output).map_err(|e| {
        format!(
            "output must be a new directory with an existing parent: {}: {e}",
            options.output.display()
        )
    })?;
    let mut artifacts = Vec::new();
    artifacts.push(write_new(
        &options.output,
        "games.jsonl",
        &lines(&corpus.games),
    )?);
    artifacts.push(write_new(
        &options.output,
        "rejections.jsonl",
        &lines(&corpus.rejections),
    )?);
    artifacts.push(write_new(
        &options.output,
        "duplicates.jsonl",
        &lines(&corpus.duplicates),
    )?);
    let mut legacy_counts = Vec::new();
    for (index, split) in SPLITS.iter().enumerate() {
        artifacts.push(write_new(
            &options.output,
            &format!("{split}.requests.jsonl"),
            &lines(&corpus.requests[index]),
        )?);
        let mut positions = Vec::new();
        let mut exclusions = Vec::new();
        for position in &corpus.requests[index] {
            let game = corpus
                .games
                .iter()
                .find(|g| g.game_id == position.game_id)
                .unwrap();
            if !game.legacy_fixture_exclusions.is_empty() {
                exclusions.push(json!({"name":position.name,"game_id":position.game_id,"reasons":game.legacy_fixture_exclusions}));
                continue;
            }
            positions.push(json!({"name":position.name,"game_id":position.game_id,"ply":position.ply,
                "semantic_position_sha256":position.semantic_position_sha256,
                "moves":position.position.moves.iter().map(|m| (m.color,m.vertex)).collect::<Vec<_>>(),
                "parameters":position.parameters}));
        }
        let fixture = json!({"schema":1,"description":"STRICT LEGACY SUBSET: empty initial board, initial Black, Chinese rules, komi 7.5. Symmetry expansion by consumers does not increase independent sample counts.",
            "split":split,"positions":positions,"excluded_positions":exclusions});
        artifacts.push(write_new(
            &options.output,
            &format!("{split}.worker-fixture.json"),
            &json_bytes(&fixture),
        )?);
        legacy_counts.push(json!({"split":split,"positions":positions.len(),"excluded_positions":exclusions.len()}));
    }
    let readiness = readiness(corpus, options);
    let manifest = json!({"schema":"rustgo-quantization-corpus-v1","options":options,
        "status":readiness["status"],"readiness":readiness,
        "source_policy":"local UTF-8 SGFs; native parser; longest branch; comments, player names and results never become labels",
        "rule_scope":"current Worker Chinese profile; half-integer komi; explicit initial stones and player",
        "sampling":{"first_ply":FIRST_PLY,"max_positions_per_game":options.positions_per_game,
            "phase_definition":"relative thirds of the recorded move sequence, not an assertion of opening/middlegame/endgame or game completion",
            "method":"uniform nonterminal plies, both-player correction; no synthetic symmetries or duplicated padding"},
        "split_policy":"SHA256(schema,seed,canonical full-game ID) order; fixed calibration/selection/holdout quotas; remaining games are reserve",
        "leakage_policy":"globally reject duplicate 8-symmetry canonical replay histories and duplicate board/player/rules/komi keys; latter deliberately excludes some distinct histories conservatively",
        "request_format":"*.requests.jsonl carries complete Position and EvalParameters fields; caller binds model/task/session/input_hash. No model output or result labels are present.",
        "legacy_fixture_scope":"only *.worker-fixture.json may be passed to the old hardcoded build_requests; each records exclusions",
        "legacy_counts":legacy_counts,"unique_games":corpus.games.len(),"rejected_sources":corpus.rejections.len(),
        "duplicate_records":corpus.duplicates.len(),"artifacts":artifacts,
        "request_set_sha256":sha(&json_bytes(&corpus.requests)),"production_certified":false});
    write_new(
        &options.output,
        "manifest.json",
        &serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
    )?;
    Ok(manifest)
}

fn run(options: &Options) -> Result<Value, String> {
    if options.output.exists() {
        return Err(format!(
            "refuse existing output {}",
            options.output.display()
        ));
    }
    let corpus = read_corpus(options)?;
    export(&corpus, options)
}

fn main() {
    let result = options(std::env::args().skip(1)).and_then(|options| {
        // Match the engine CLI's explicit stack: Windows main starts with only
        // 1 MiB, insufficient for debug Board/BoardHistory and Zobrist setup.
        let worker = std::thread::Builder::new()
            .name("sgf-corpus".into())
            .stack_size(256 * 1024 * 1024)
            .spawn(move || run(&options))
            .map_err(|e| format!("start corpus worker: {e}"))?;
        worker
            .join()
            .map_err(|_| "corpus worker panicked".to_owned())?
    });
    match result {
        Ok(manifest) => println!(
            "{}",
            serde_json::to_string_pretty(
                &json!({"status":manifest["status"],"readiness":manifest["readiness"]})
            )
            .unwrap()
        ),
        Err(error) => {
            eprintln!("sgf_quantization_corpus: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_options() -> Options {
        options(["sgfs", "--output", "corpus", "--games-per-split", "1,1,1"].map(str::to_owned))
            .unwrap()
    }

    fn moves(count: usize) -> Vec<Movement> {
        (0..count)
            .map(|i| {
                (
                    if i % 2 == 0 { P_BLACK } else { P_WHITE },
                    ((i % 9) * 2 + (i / 9) * 2 * 19) as i32,
                )
            })
            .collect()
    }

    fn sgf(moves: &[Movement], root_extra: &str) -> String {
        let mut text = format!("(;GM[1]FF[4]SZ[19]RU[Chinese]KM[7.5]{root_extra}");
        for &(color, vertex) in moves {
            let vertex = if vertex == -1 {
                String::new()
            } else {
                format!(
                    "{}{}",
                    (b'a' + (vertex % 19) as u8) as char,
                    (b'a' + (vertex / 19) as u8) as char
                )
            };
            text.push_str(&format!(
                ";{}[{vertex}]",
                if color == P_BLACK { "B" } else { "W" }
            ));
        }
        text.push(')');
        text
    }

    fn candidate(text: &str, path: &str, options: &Options) -> Result<Candidate, Invalid> {
        parse_game(
            text.as_bytes(),
            Source {
                path: path.into(),
                sha256: Some(sha(text.as_bytes())),
            },
            options,
        )
    }

    #[test]
    fn full_game_identity_ignores_annotations_and_normalizes_symmetry() {
        let options = test_options();
        let a = candidate(
            &sgf(
                &moves(36),
                "PB[A]PW[B]RE[B+R]C[Text B\\[ss\\] is not a move]",
            ),
            "a.sgf",
            &options,
        )
        .unwrap();
        let rotated: Vec<_> = moves(36)
            .into_iter()
            .map(|(p, v)| (p, transform_vertex(v, 5)))
            .collect();
        let b = candidate(
            &sgf(&rotated, "PB[Different]PW[Other]RE[W+1.5]C[No labels]"),
            "b.sgf",
            &options,
        )
        .unwrap();
        assert_eq!(a.record.game_id, b.record.game_id);
        assert_eq!(
            a.positions[0].semantic_position_sha256,
            b.positions[0].semantic_position_sha256
        );
        assert_eq!(
            a.positions[0].board_state_sha256,
            b.positions[0].board_state_sha256
        );
        assert_ne!(a.record.sources[0].sha256, b.record.sources[0].sha256);
        let corpus = assemble(vec![a, b], Vec::new(), &options);
        assert_eq!(corpus.games.len(), 1);
        assert_eq!(corpus.games[0].sources.len(), 2);
        assert_eq!(corpus.duplicates[0]["kind"], "duplicate_game");
    }

    #[test]
    fn illegal_tail_rejects_entire_game_and_terminal_history_is_not_forged() {
        let options = test_options();
        let mut illegal = moves(20);
        illegal.push((P_BLACK, illegal[0].1));
        assert_eq!(
            candidate(&sgf(&illegal, ""), "bad.sgf", &options)
                .err()
                .unwrap()
                .code,
            "illegal_or_out_of_turn"
        );
        let mut terminal = moves(20);
        terminal.extend([(P_BLACK, -1), (P_WHITE, -1), (P_BLACK, 360)]);
        assert_eq!(
            candidate(&sgf(&terminal, ""), "terminal.sgf", &options)
                .err()
                .unwrap()
                .code,
            "move_after_terminal"
        );
        let mut out_of_turn = moves(20);
        out_of_turn[10].0 = P_WHITE;
        assert_eq!(
            candidate(&sgf(&out_of_turn, ""), "turn.sgf", &options)
                .err()
                .unwrap()
                .code,
            "illegal_or_out_of_turn"
        );
    }

    #[test]
    fn longest_branch_and_midgame_setup_boundaries_match_native_rules() {
        let options = test_options();
        let full = sgf(&moves(20), "");
        let root = full.trim_end_matches(')');
        let tree = format!("{root}(;B[rr])(;B[ss];W[qs]))");
        let game = candidate(&tree, "variation.sgf", &options).unwrap();
        assert_eq!(game.record.selected_branch, vec![1]);
        assert_eq!(game.record.semantics.moves.len(), 22);
        let setup = format!("{root}(;AB[ss];B[rs]))");
        assert_eq!(
            candidate(&setup, "setup.sgf", &options).err().unwrap().code,
            "midgame_setup_or_rules"
        );
        let collection = format!("{full}{full}");
        assert_eq!(
            candidate(&collection, "collection.sgf", &options)
                .err()
                .unwrap()
                .code,
            "sgf_framing"
        );
    }

    #[test]
    fn rules_komi_and_initial_state_are_preserved_or_explicitly_rejected() {
        let options = test_options();
        let baseline = sgf(&moves(20), "");
        let missing = baseline.replace("RU[Chinese]KM[7.5]", "");
        assert_eq!(
            candidate(&missing, "missing.sgf", &options)
                .err()
                .unwrap()
                .code,
            "missing_rules"
        );
        let mut assumed = options.clone();
        assumed.assume_rules = Some("chinese".into());
        assumed.assume_komi = Some(7.5);
        let accepted = candidate(&missing, "assumed.sgf", &assumed).unwrap();
        assert_eq!(accepted.record.assumptions.len(), 2);
        assert_eq!(
            accepted.record.game_id,
            candidate(&baseline, "explicit.sgf", &options)
                .unwrap()
                .record
                .game_id
        );
        assert_eq!(
            candidate(
                &baseline.replace("Chinese", "Japanese"),
                "japanese.sgf",
                &options
            )
            .err()
            .unwrap()
            .code,
            "unsupported_rules"
        );
        assert_eq!(
            candidate(&baseline.replace("SZ[19]", "SZ[13]"), "13.sgf", &options)
                .err()
                .unwrap()
                .code,
            "non_19_board"
        );
        let white_moves: Vec<_> = moves(20)
            .into_iter()
            .map(|(p, v)| (get_opp(p), v))
            .collect();
        let setup = sgf(&white_moves, "AB[ss]PL[W]").replace("KM[7.5]", "KM[0.5]");
        let game = candidate(&setup, "handicap.sgf", &options).unwrap();
        assert_eq!(game.positions[0].position.initial_player, P_WHITE);
        assert_eq!(game.positions[0].position.komi, 0.5);
        assert_eq!(game.positions[0].position.initial_stones[0].vertex, 360);
        assert_eq!(game.record.legacy_fixture_exclusions.len(), 3);
    }

    #[test]
    fn premove_ab_matches_root_setup_and_preserves_root_white_stones() {
        let options = test_options();
        let sequence: Vec<_> = moves(20)
            .into_iter()
            .map(|(p, v)| (get_opp(p), v))
            .collect();
        let root = candidate(
            &sgf(&sequence, "HA[2]AB[ss][rs]PL[W]"),
            "root.sgf",
            &options,
        )
        .unwrap();
        for extra in [
            "HA[2]PL[W];AB[ss][rs]",
            "HA[2]AB[ss]PL[W];AB[rs]",
            "HA[2]PL[W];AB[ss];C[no move];AB[rs]",
            // As before, absent PL derives the initial player from the first move.
            "HA[2];AB[ss][rs]",
        ] {
            let prefix = candidate(&sgf(&sequence, extra), "prefix.sgf", &options).unwrap();
            assert_eq!(root.record.game_id, prefix.record.game_id);
            assert_eq!(
                json_bytes(&root.record.semantics),
                json_bytes(&prefix.record.semantics)
            );
            assert_eq!(json_bytes(&root.positions), json_bytes(&prefix.positions));
        }

        let mixed_root = candidate(
            &sgf(&sequence, "HA[1]AB[ss]AW[rs]PL[W]"),
            "mixed-root.sgf",
            &options,
        )
        .unwrap();
        let mixed_prefix = candidate(
            &sgf(&sequence, "HA[1]AW[rs]PL[W];AB[ss]"),
            "mixed-prefix.sgf",
            &options,
        )
        .unwrap();
        assert_eq!(mixed_root.record.game_id, mixed_prefix.record.game_id);
        assert_eq!(
            json_bytes(&mixed_root.positions),
            json_bytes(&mixed_prefix.positions)
        );

        // The new HA consistency gate applies only to newly admitted nonroot AB.
        let old_root_ha = candidate(
            &sgf(&sequence, "HA[99]AB[ss][rs]PL[W]"),
            "old-root-ha.sgf",
            &options,
        )
        .unwrap();
        assert_eq!(root.record.game_id, old_root_ha.record.game_id);
        let old_root_move = candidate(
            &sgf(&sequence[1..], "HA[2]AB[ss][rs]PL[W]W[aa]"),
            "old-root-move.sgf",
            &options,
        )
        .unwrap();
        assert_eq!(root.record.game_id, old_root_move.record.game_id);
        assert_eq!(
            json_bytes(&root.positions),
            json_bytes(&old_root_move.positions)
        );
    }

    #[test]
    fn premove_ab_in_child_root_uses_longest_branch_and_first_child_ties() {
        let options = test_options();
        let sequence: Vec<_> = moves(20)
            .into_iter()
            .map(|(p, v)| (get_opp(p), v))
            .collect();
        let header = "(;GM[1]FF[4]SZ[19]RU[Chinese]KM[7.5]";
        let sequence_text = sgf(&sequence, "");
        let tail = &sequence_text[header.len()..sequence_text.len() - 1];
        let short_text = sgf(&sequence[..18], "");
        let short_tail = &short_text[header.len()..short_text.len() - 1];
        let expected =
            candidate(&sgf(&sequence, "HA[2]AB[ss][rs]"), "expected.sgf", &options).unwrap();
        let longest = candidate(
            &format!("{header}HA[2](;AB[ss][qs]{short_tail})(;AB[ss][rs]{tail}))"),
            "longest.sgf",
            &options,
        )
        .unwrap();
        assert_eq!(longest.record.selected_branch, vec![1]);
        assert_eq!(longest.record.game_id, expected.record.game_id);
        assert_eq!(
            json_bytes(&longest.positions),
            json_bytes(&expected.positions)
        );
        let tie = candidate(
            &format!("{header}HA[2](;AB[ss][rs]{tail})(;AB[ss][qs]{tail}))"),
            "tie.sgf",
            &options,
        )
        .unwrap();
        assert_eq!(tie.record.selected_branch, vec![0]);
        assert_eq!(tie.record.game_id, expected.record.game_id);
        let empty_wrapper = format!("{header}HA[2]((;AB[ss][rs]{tail})))");
        assert_eq!(
            candidate(&empty_wrapper, "empty-wrapper.sgf", &options)
                .err()
                .unwrap()
                .code,
            "sgf_parse",
        );
    }

    #[test]
    fn premove_ab_rejects_conflicting_invalid_and_inconsistent_setup() {
        let options = test_options();
        let sequence: Vec<_> = moves(20)
            .into_iter()
            .map(|(p, v)| (get_opp(p), v))
            .collect();
        for extra in [
            ";AB[ss][ss]",
            "AB[ss];AB[ss]",
            "AW[ss];AB[ss]",
            ";AB[ss];AB[ss]",
            ";AB[rr:ss][ss]",
            // Both White neighbors give the added corner Black stone no liberties.
            "AW[rs][sr];AB[ss]",
        ] {
            assert_eq!(
                candidate(&sgf(&sequence, extra), "conflict.sgf", &options)
                    .err()
                    .unwrap()
                    .code,
                "invalid_initial_stones",
                "{extra}",
            );
        }
        assert_eq!(
            candidate(&sgf(&sequence, ";AB[tt]"), "point.sgf", &options)
                .err()
                .unwrap()
                .code,
            "sgf_parse",
        );
        for extra in [
            "HA[3];AB[ss][rs]",
            "HA[-1];AB[ss]",
            "HA[bad];AB[ss]",
            "HA[1][1];AB[ss]",
        ] {
            assert_eq!(
                candidate(&sgf(&sequence, extra), "ha.sgf", &options)
                    .err()
                    .unwrap()
                    .code,
                "invalid_initial_handicap",
                "{extra}",
            );
        }
        assert_eq!(
            candidate(&sgf(&sequence, "PL[B];AB[ss]"), "pl-conflict.sgf", &options)
                .err()
                .unwrap()
                .code,
            "illegal_or_out_of_turn",
        );
    }

    #[test]
    fn premove_ab_keeps_nonroot_edits_and_entire_illegal_tail_rejected() {
        let options = test_options();
        let sequence: Vec<_> = moves(20)
            .into_iter()
            .map(|(p, v)| (get_opp(p), v))
            .collect();
        for extra in [
            ";AW[ss]",
            ";AE[ss]",
            ";PL[W]",
            ";RU[Chinese]",
            ";KM[7.5]",
            ";SZ[19]",
            ";HA[1]",
            ";AB[ss]W[qq]",
            ";W[];AB[ss]", // A pass still marks the start of play.
        ] {
            assert_eq!(
                candidate(&sgf(&sequence, extra), "edit.sgf", &options)
                    .err()
                    .unwrap()
                    .code,
                "midgame_setup_or_rules",
                "{extra}",
            );
        }
        let full = sgf(&sequence, ";AB[ss]");
        let after = format!("{};AB[rs])", full.trim_end_matches(')'));
        assert_eq!(
            candidate(&after, "after.sgf", &options).err().unwrap().code,
            "midgame_setup_or_rules",
        );
        let mut illegal = sequence.clone();
        illegal.push((P_WHITE, sequence[0].1));
        assert_eq!(
            candidate(&sgf(&illegal, ";AB[ss]"), "illegal-tail.sgf", &options)
                .err()
                .unwrap()
                .code,
            "illegal_or_out_of_turn",
        );
        let mut terminal = sequence;
        terminal.extend([(P_WHITE, -1), (P_BLACK, -1), (P_WHITE, 360)]);
        assert_eq!(
            candidate(&sgf(&terminal, ";AB[rs]"), "terminal-tail.sgf", &options)
                .err()
                .unwrap()
                .code,
            "move_after_terminal",
        );
    }

    #[test]
    fn sampling_has_both_players_and_no_symmetry_padding() {
        let available: Vec<_> = (8..=256)
            .map(|ply| (ply, if ply % 2 == 0 { P_BLACK } else { P_WHITE }))
            .collect();
        let selected = sample_plies(&available, 32);
        assert_eq!(selected.len(), 32);
        assert!(selected.iter().any(|p| p % 2 == 0));
        assert!(selected.iter().any(|p| p % 2 == 1));
        assert!(*selected.first().unwrap() <= 9);
        assert_eq!(*selected.last().unwrap(), 256);
        assert_eq!(sample_plies(&available[..3], 32), vec![8, 9, 10]);
    }

    #[test]
    fn splitting_is_stable_and_duplicate_positions_never_cross_splits() {
        let mut options = test_options();
        options.positions_per_game = 2;
        let make = |reverse: bool| {
            let mut candidates = Vec::new();
            for i in 0..3 {
                let mut sequence = moves(20);
                sequence[19].1 = 360 - i;
                candidates
                    .push(candidate(&sgf(&sequence, ""), &format!("{i}.sgf"), &options).unwrap());
            }
            if reverse {
                candidates.reverse();
            }
            assemble(candidates, Vec::new(), &options)
        };
        let a = make(false);
        let b = make(true);
        assert_eq!(json_bytes(&a.requests), json_bytes(&b.requests));
        assert_eq!(a.games.iter().filter(|g| g.split.is_some()).count(), 3);
        assert!(
            a.duplicates
                .iter()
                .any(|d| d["kind"] == "duplicate_position")
        );
        let mut all = BTreeSet::new();
        for request in a.requests.iter().flatten() {
            assert!(all.insert(&request.board_state_sha256));
        }
        assert_eq!(readiness(&a, &options)["status"], "NOT_READY");
        assert_eq!(
            readiness(&a, &options)["symmetry_multiplicity_counted"],
            false
        );
    }

    #[test]
    fn same_board_different_history_is_excluded_conservatively() {
        let options = test_options();
        let sequence = moves(20);
        let mut transposed = sequence.clone();
        transposed.swap(0, 2);
        let a = candidate(&sgf(&sequence, ""), "a.sgf", &options).unwrap();
        let b = candidate(&sgf(&transposed, ""), "b.sgf", &options).unwrap();
        assert_ne!(a.record.game_id, b.record.game_id);
        assert_ne!(
            a.positions[0].semantic_position_sha256,
            b.positions[0].semantic_position_sha256
        );
        assert_eq!(
            a.positions[0].board_state_sha256,
            b.positions[0].board_state_sha256
        );
        let corpus = assemble(vec![a, b], Vec::new(), &options);
        assert!(corpus.duplicates.iter().any(|d| {
            d["reason"]
                .as_str()
                .is_some_and(|r| r.contains("conservative"))
        }));
    }
}

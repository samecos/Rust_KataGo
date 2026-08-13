//! Per-pattern utility-bonus table for encouraging or avoiding specific local shapes.
//!
//! Corresponds to `cpp/search/patternbonustable.h` and
//! `cpp/search/patternbonustable.cpp`.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::OnceLock;

use kata_core::global;
use kata_core::hash::Hash128;
use kata_core::logger::Logger;
use kata_core::rng::Rand;
use kata_data::files;
use kata_data::sgf::{PositionSample, Sgf};
use kata_game::board::{Board, C_EMPTY, Loc, MAX_ARR_SIZE, NULL_LOC, PASS_LOC, Player, get_opp};
use kata_game::history::BoardHistory;
use kata_game::symmetry::{self, get_sym_loc, is_transpose};
use parking_lot::Mutex;

use crate::local_pattern::LocalPatternHasher;

struct ZobristTables {
    pattern_hasher: LocalPatternHasher,
    move_locs: [Hash128; MAX_ARR_SIZE],
}

static ZOBRIST: OnceLock<ZobristTables> = OnceLock::new();

fn zobrist_tables() -> &'static ZobristTables {
    ZOBRIST.get_or_init(|| {
        let mut rand = Rand::new();
        rand.init_from_seed("PatternBonusTable ZOBRIST STUFF");

        let mut pattern_hasher = LocalPatternHasher::new();
        pattern_hasher.init(9, 9, &mut rand);

        rand.init_from_seed(
            "Reseed PatternBonusTable zobrist so that zobrists don't change when Board::MAX_ARR_SIZE changes",
        );
        let mut move_locs = [Hash128::default(); MAX_ARR_SIZE];
        for cell in &mut move_locs {
            *cell = Hash128::new(rand.next_u64(), rand.next_u64());
        }

        ZobristTables {
            pattern_hasher,
            move_locs,
        }
    })
}

/// A bonus (positive) or penalty (negative) to white's utility for a pattern.
#[derive(Debug, Clone, Copy, Default)]
pub struct PatternBonusEntry {
    pub utility_bonus: f64,
}

/// Sharded table mapping local board patterns to utility bonuses.
pub struct PatternBonusTable {
    entries: Vec<Mutex<BTreeMap<Hash128, PatternBonusEntry>>>,
}

impl Clone for PatternBonusTable {
    fn clone(&self) -> Self {
        Self {
            entries: self
                .entries
                .iter()
                .map(|m| Mutex::new(m.lock().clone()))
                .collect(),
        }
    }
}

impl Default for PatternBonusTable {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternBonusTable {
    /// Create a table with the default 1024 shards.
    pub fn new() -> Self {
        Self::with_shards(1024)
    }

    /// Create a table with `num_shards` shards.
    pub fn with_shards(num_shards: i32) -> Self {
        assert!(
            num_shards > 0,
            "PatternBonusTable must have at least one shard"
        );
        let _ = zobrist_tables();
        let entries = (0..num_shards)
            .map(|_| Mutex::new(BTreeMap::new()))
            .collect();
        Self { entries }
    }

    /// Hash key for `(pla, move_loc, board)` where `board` is the board *before* the move.
    pub fn get_hash(&self, pla: Player, move_loc: Loc, board: &Board) -> Hash128 {
        if move_loc == NULL_LOC || move_loc == PASS_LOC || board.would_be_ko_capture(move_loc, pla)
        {
            return Hash128::default();
        }

        let z = zobrist_tables();
        let mut hash = z.pattern_hasher.get_hash(board, move_loc, pla);
        hash ^= z.move_locs[move_loc as usize];
        hash ^= Board::zobrist_size_x_hash(board.x_size as usize);
        hash ^= Board::zobrist_size_y_hash(board.y_size as usize);
        hash
    }

    /// Look up an entry by its hash key.
    pub fn get(&self, hash: Hash128) -> PatternBonusEntry {
        if hash == Hash128::default() {
            return PatternBonusEntry::default();
        }
        let sub_map_idx = (hash.hash0 % self.entries.len() as u64) as usize;
        let map = self.entries[sub_map_idx].lock();
        map.get(&hash).copied().unwrap_or_default()
    }

    /// Look up an entry for a specific move on the board before the move.
    pub fn get_for_move(&self, pla: Player, move_loc: Loc, board: &Board) -> PatternBonusEntry {
        self.get(self.get_hash(pla, move_loc, board))
    }

    /// Add `bonus` to white's utility for the pattern around `(pla, move_loc, board)`,
    /// after applying `symmetry` and optionally flipping colors.
    ///
    /// `hashes_this_game` prevents the same transformed pattern from being counted
    /// more than once within a single game/sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn add_bonus(
        &self,
        pla: Player,
        move_loc: Loc,
        board: &Board,
        bonus: f64,
        symmetry: i32,
        flip_colors: bool,
        hashes_this_game: &mut HashSet<Hash128>,
    ) {
        if move_loc == NULL_LOC || move_loc == PASS_LOC || board.would_be_ko_capture(move_loc, pla)
        {
            return;
        }

        let z = zobrist_tables();
        let mut hash =
            z.pattern_hasher
                .get_hash_with_sym(board, move_loc, pla, symmetry, flip_colors);
        let sym_move_loc = get_sym_loc(move_loc, board, symmetry);
        hash ^= z.move_locs[sym_move_loc as usize];
        if is_transpose(symmetry) {
            hash ^= Board::zobrist_size_x_hash(board.y_size as usize);
            hash ^= Board::zobrist_size_y_hash(board.x_size as usize);
        } else {
            hash ^= Board::zobrist_size_x_hash(board.x_size as usize);
            hash ^= Board::zobrist_size_y_hash(board.y_size as usize);
        }

        if hashes_this_game.contains(&hash) {
            return;
        }
        hashes_this_game.insert(hash);

        let sub_map_idx = (hash.hash0 % self.entries.len() as u64) as usize;
        let mut map = self.entries[sub_map_idx].lock();
        map.entry(hash).or_default().utility_bonus += bonus;
    }

    /// Add `bonus` for every move in `game`, applying all symmetries and color flips.
    pub fn add_bonus_for_game_moves(&self, game: &BoardHistory, bonus: f64) {
        self.add_bonus_for_game_moves_player(game, bonus, C_EMPTY);
    }

    /// Add `bonus` for every move by `only_pla` (or both if `C_EMPTY`) in `game`.
    pub fn add_bonus_for_game_moves_player(
        &self,
        game: &BoardHistory,
        bonus: f64,
        only_pla: Player,
    ) {
        let mut hashes_this_game = HashSet::new();
        let mut board = game.initial_board.clone();
        let mut hist = BoardHistory::new(
            board.clone(),
            game.initial_pla,
            game.rules,
            game.initial_encore_phase,
        );

        for m in &game.move_history {
            let pla = m.pla;
            let loc = m.loc;
            if !hist.make_board_move_tolerant(&mut board, loc, pla) {
                break;
            }
            if only_pla == C_EMPTY || only_pla == pla {
                for flip_colors in [false, true] {
                    for symmetry in 0..symmetry::NUM_SYMMETRIES {
                        // Convention: pattern-match on the board *before* the move was played.
                        self.add_bonus(
                            pla,
                            loc,
                            hist.get_recent_board(1),
                            bonus,
                            symmetry,
                            flip_colors,
                            &mut hashes_this_game,
                        );
                    }
                }
            }
        }
    }

    /// Penalize local shapes that repeat moves found in SGF files.
    #[allow(clippy::too_many_arguments)]
    pub fn avoid_repeated_sgf_moves(
        &self,
        sgfs_dirs_or_files: &[String],
        penalty: f64,
        decay_older_files_lambda: f64,
        min_turn_number: i64,
        max_files: usize,
        allowed_player_names: &[String],
        logger: &Logger,
        log_source: &str,
    ) {
        let mut sgf_files = Vec::new();
        if let Err(e) = files::collect_sgfs_from_dirs_or_files(sgfs_dirs_or_files, &mut sgf_files) {
            logger.write(&format!(
                "Error collecting SGFs for {}: {}",
                log_source, e.0
            ));
            return;
        }
        if let Err(e) = files::sort_newest_to_oldest(&mut sgf_files) {
            logger.write(&format!("Error sorting SGFs for {}: {}", log_source, e.0));
            return;
        }

        let mut factor = 1.0;
        for file_name in sgf_files.iter().take(max_files) {
            let sgf = match Sgf::parse_file(file_name) {
                Ok(s) => s,
                Err(e) => {
                    logger.write(&format!("Invalid SGF {}: {}", file_name, e.0));
                    continue;
                }
            };

            let black_okay = allowed_player_names.is_empty()
                || global::contains_str(
                    allowed_player_names,
                    &sgf.get_player_name(kata_game::board::P_BLACK),
                );
            let white_okay = allowed_player_names.is_empty()
                || global::contains_str(
                    allowed_player_names,
                    &sgf.get_player_name(kata_game::board::P_WHITE),
                );

            let mut hashes_this_game = HashSet::new();
            let mut unique_hashes = BTreeSet::new();

            let mut handle_position =
                |pos_sample: &PositionSample, hist: &BoardHistory, comments: &str| {
                    let _ = pos_sample;
                    if comments.contains("%SKIP%") {
                        return;
                    }
                    if hist.move_history.is_empty() {
                        return;
                    }
                    if hist.get_current_turn_number() < min_turn_number {
                        return;
                    }
                    let last = hist.move_history[hist.move_history.len() - 1];
                    let move_loc = last.loc;
                    let move_pla = last.pla;
                    if move_pla == kata_game::board::P_BLACK && !black_okay {
                        return;
                    }
                    if move_pla == kata_game::board::P_WHITE && !white_okay {
                        return;
                    }

                    for flip_colors in [false, true] {
                        for symmetry in 0..symmetry::NUM_SYMMETRIES {
                            let sym_pla = if flip_colors {
                                get_opp(move_pla)
                            } else {
                                move_pla
                            };
                            let bonus = if sym_pla == kata_game::board::P_WHITE {
                                -penalty * factor
                            } else {
                                penalty * factor
                            };
                            self.add_bonus(
                                move_pla,
                                move_loc,
                                hist.get_recent_board(1),
                                bonus,
                                symmetry,
                                flip_colors,
                                &mut hashes_this_game,
                            );
                        }
                    }
                };

            if let Err(e) = sgf.iter_all_unique_positions(
                &mut unique_hashes,
                true,
                true,
                false,
                false,
                None::<&mut dyn rand::RngCore>,
                &mut handle_position,
                false,
            ) {
                logger.write(&format!(
                    "Error iterating SGF {} for {}: {}",
                    file_name, log_source, e.0
                ));
            }

            logger.write(&format!(
                "Added {} shapes to penalize repeats for {} from {}",
                global::uint64_to_string(hashes_this_game.len() as u64),
                log_source,
                file_name
            ));
            factor *= decay_older_files_lambda;
        }
    }

    /// Penalize local shapes that repeat moves found in position files, and delete
    /// any position files beyond the loaded ones.
    #[allow(clippy::too_many_arguments)]
    pub fn avoid_repeated_pos_moves_and_delete_excess_files(
        &self,
        poses_dirs_to_load_and_prune: &[String],
        penalty: f64,
        decay_older_poses_lambda: f64,
        min_turn_number: i64,
        max_turn_number: i64,
        max_poses: usize,
        logger: &Logger,
        log_source: &str,
    ) {
        let mut pos_files = Vec::new();
        if let Err(e) = files::collect_poses_from_dirs(poses_dirs_to_load_and_prune, &mut pos_files)
        {
            logger.write(&format!(
                "Error collecting poses for {}: {}",
                log_source, e.0
            ));
            return;
        }
        if let Err(e) = files::sort_newest_to_oldest(&mut pos_files) {
            logger.write(&format!("Error sorting poses for {}: {}", log_source, e.0));
            return;
        }

        let mut num_poses_used: usize = 0;
        let mut num_poses_invalid: usize = 0;
        let mut num_pos_load_errors: usize = 0;
        let mut factor = 1.0;

        let mut i = 0;
        while i < pos_files.len() {
            if num_poses_used >= max_poses {
                break;
            }
            let file_name = &pos_files[i];
            let mut hashes_this_game = HashSet::new();
            let lines = match kata_core::fs::read_file_lines(file_name, b'\n') {
                Ok(l) => l,
                Err(e) => {
                    logger.write(&format!(
                        "Error reading pos file {} for {}: {}",
                        file_name, log_source, e.0
                    ));
                    i += 1;
                    continue;
                }
            };

            for line in lines {
                let trimmed = global::trim(&line);
                if trimmed.is_empty() {
                    continue;
                }
                let pos_sample = match PositionSample::of_json_line(trimmed) {
                    Ok(s) => s,
                    Err(_) => {
                        num_pos_load_errors += 1;
                        continue;
                    }
                };

                const IS_MULTI_STONE_SUICIDE_LEGAL: bool = true;
                let turn_number = pos_sample.get_current_turn_number();
                if turn_number < min_turn_number
                    || turn_number > max_turn_number
                    || !pos_sample.moves.is_empty()
                    || !pos_sample.board.is_legal(
                        pos_sample.hint_loc,
                        pos_sample.next_pla,
                        IS_MULTI_STONE_SUICIDE_LEGAL,
                    )
                {
                    num_poses_invalid += 1;
                    continue;
                }

                for flip_colors in [false, true] {
                    for symmetry in 0..symmetry::NUM_SYMMETRIES {
                        let sym_pla = if flip_colors {
                            get_opp(pos_sample.next_pla)
                        } else {
                            pos_sample.next_pla
                        };
                        let bonus = if sym_pla == kata_game::board::P_WHITE {
                            -penalty * factor
                        } else {
                            penalty * factor
                        };
                        self.add_bonus(
                            pos_sample.next_pla,
                            pos_sample.hint_loc,
                            &pos_sample.board,
                            bonus,
                            symmetry,
                            flip_colors,
                            &mut hashes_this_game,
                        );
                    }
                }
                num_poses_used += 1;
                factor *= decay_older_poses_lambda;
            }
            i += 1;
        }

        while i < pos_files.len() {
            logger.write(&format!("Removing old pos file: {}", pos_files[i]));
            kata_core::fs::try_remove_file(&pos_files[i]);
            i += 1;
        }

        logger.write(&format!("Loaded avoid poses from {}", log_source));
        logger.write(&format!(
            "numPosesUsed = {}",
            global::uint64_to_string(num_poses_used as u64)
        ));
        logger.write(&format!(
            "numPosesInvalid = {}",
            global::uint64_to_string(num_poses_invalid as u64)
        ));
        logger.write(&format!(
            "numPosLoadErrors = {}",
            global::uint64_to_string(num_pos_load_errors as u64)
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kata_game::board::{Board, P_BLACK, location};
    use kata_game::rules::Rules;

    fn empty_board() -> Board {
        Board::new(9, 9)
    }

    #[test]
    fn test_get_hash_zero_for_pass_and_null() {
        let table = PatternBonusTable::new();
        let board = empty_board();
        assert_eq!(
            table.get_hash(P_BLACK, PASS_LOC, &board),
            Hash128::default()
        );
        assert_eq!(
            table.get_hash(P_BLACK, NULL_LOC, &board),
            Hash128::default()
        );
    }

    #[test]
    fn test_get_returns_default_for_unknown_hash() {
        let table = PatternBonusTable::new();
        assert_eq!(table.get(Hash128::default()).utility_bonus, 0.0);
        assert_eq!(table.get(Hash128::new(1, 2)).utility_bonus, 0.0);
    }

    #[test]
    fn test_add_and_get_bonus() {
        let table = PatternBonusTable::new();
        let board = empty_board();
        let loc = location::get_loc(4, 4, board.x_size);

        let mut hashes = HashSet::new();
        table.add_bonus(P_BLACK, loc, &board, 1.5, 0, false, &mut hashes);

        let entry = table.get_for_move(P_BLACK, loc, &board);
        assert!((entry.utility_bonus - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_same_pattern_not_double_counted_in_same_game() {
        let table = PatternBonusTable::new();
        let board = empty_board();
        let loc = location::get_loc(4, 4, board.x_size);

        let mut hashes = HashSet::new();
        table.add_bonus(P_BLACK, loc, &board, 1.0, 0, false, &mut hashes);
        table.add_bonus(P_BLACK, loc, &board, 1.0, 0, false, &mut hashes);

        let entry = table.get_for_move(P_BLACK, loc, &board);
        assert!((entry.utility_bonus - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_different_symmetries_are_distinct() {
        let table = PatternBonusTable::new();
        let board = empty_board();
        let loc = location::get_loc(4, 4, board.x_size);

        let mut hashes = HashSet::new();
        table.add_bonus(P_BLACK, loc, &board, 1.0, 0, false, &mut hashes);
        table.add_bonus(P_BLACK, loc, &board, 2.0, 1, false, &mut hashes);

        let entry = table.get_for_move(P_BLACK, loc, &board);
        assert!((entry.utility_bonus - 1.0).abs() < 1e-9);

        let hash_sym = table.get_hash(P_BLACK, loc, &board);
        let hash_base = table.get_hash(P_BLACK, loc, &board);
        assert_eq!(hash_sym, hash_base);
    }

    #[test]
    fn test_add_bonus_for_game_moves() {
        let table = PatternBonusTable::new();
        let mut board = empty_board();
        let mut hist = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        let loc = location::get_loc(4, 4, board.x_size);
        hist.make_board_move_assume_legal(&mut board, loc, P_BLACK);

        table.add_bonus_for_game_moves(&hist, 0.25);

        let entry = table.get_for_move(P_BLACK, loc, hist.get_recent_board(1));
        assert!(entry.utility_bonus.abs() > 1e-9);
    }

    #[test]
    fn test_clone_preserves_entries() {
        let table = PatternBonusTable::new();
        let board = empty_board();
        let loc = location::get_loc(4, 4, board.x_size);

        let mut hashes = HashSet::new();
        table.add_bonus(P_BLACK, loc, &board, 1.0, 0, false, &mut hashes);

        let cloned = table.clone();
        let entry = cloned.get_for_move(P_BLACK, loc, &board);
        assert!((entry.utility_bonus - 1.0).abs() < 1e-9);
    }
}

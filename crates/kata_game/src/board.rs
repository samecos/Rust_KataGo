//! Go board types and core board logic.
//!
//! Corresponds to `cpp/game/board.h` and `cpp/game/board.cpp`.
//! This module currently provides the foundational types and the main `Board`
//! struct with initialization, chain management, legal move checking, playing
//! moves, and undo. More advanced tactical helpers (ladders, area scoring) are
//! deferred to later slices.

use kata_core::global::IOError;
use kata_core::hash;
use kata_core::hash::Hash128;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::fmt;

/// Maximum edge length allowed for the board.
pub const MAX_LEN: usize = 19;
/// Default edge length if unspecified.
pub const DEFAULT_LEN: usize = 19;
/// Maximum number of playable spaces.
pub const MAX_PLAY_SIZE: usize = MAX_LEN * MAX_LEN;
/// Maximum size of arrays needed (includes walls and padding).
pub const MAX_ARR_SIZE: usize = (MAX_LEN + 1) * (MAX_LEN + 2) + 1;

/// Player identifier.
pub type Player = i8;
pub const P_BLACK: Player = 1;
pub const P_WHITE: Player = 2;

/// Color of a point on the board.
pub type Color = i8;
pub const C_EMPTY: Color = 0;
pub const C_BLACK: Color = 1;
pub const C_WHITE: Color = 2;
pub const C_WALL: Color = 3;
pub const NUM_BOARD_COLORS: usize = 4;

/// Get the opposite color/player.
pub const fn get_opp(c: Color) -> Color {
    c ^ 3
}

/// Convert a player to the corresponding stone color.
pub const fn player_to_color(p: Player) -> Color {
    p
}

/// Convert a stone color to the player that owns it.
pub const fn color_to_player(c: Color) -> Player {
    c
}

/// Player/color I/O helpers.
pub mod player_io {
    use super::{C_BLACK, C_EMPTY, C_WHITE, Color, P_BLACK, P_WHITE, Player};
    use kata_core::global::IOError;

    pub fn color_to_char(c: Color) -> char {
        match c {
            C_EMPTY => '.',
            C_BLACK => 'X',
            C_WHITE => 'O',
            _ => '?',
        }
    }

    pub fn player_to_string_short(p: Player) -> &'static str {
        match p {
            P_BLACK => "B",
            P_WHITE => "W",
            _ => "?",
        }
    }

    pub fn player_to_string(p: Player) -> &'static str {
        match p {
            P_BLACK => "Black",
            P_WHITE => "White",
            C_EMPTY => "Empty",
            _ => "?",
        }
    }

    pub fn try_parse_player(s: &str) -> Option<Player> {
        match kata_core::global::to_upper(s).as_str() {
            "B" | "BLACK" => Some(P_BLACK),
            "W" | "WHITE" => Some(P_WHITE),
            _ => None,
        }
    }

    pub fn parse_player(s: &str) -> Result<Player, IOError> {
        try_parse_player(s).ok_or_else(|| IOError(format!("Could not parse player: {}", s)))
    }
}

/// Location of a point on the board.
///
/// `(x, y)` is represented as `(x + 1) + (y + 1) * (x_size + 1)`.
pub type Loc = i16;

/// Special location indicating an invalid spot on the board.
pub const NULL_LOC: Loc = 0;
/// Special location indicating a pass move.
pub const PASS_LOC: Loc = 1;

/// Location arithmetic helpers.
pub mod location {
    use super::{Loc, NULL_LOC, PASS_LOC};

    pub fn get_loc(x: i32, y: i32, x_size: i32) -> Loc {
        ((x + 1) + (y + 1) * (x_size + 1)) as Loc
    }

    pub fn get_x(loc: Loc, x_size: i32) -> i32 {
        (loc % (x_size + 1) as Loc) as i32 - 1
    }

    pub fn get_y(loc: Loc, x_size: i32) -> i32 {
        (loc / (x_size + 1) as Loc) as i32 - 1
    }

    /// Fill `adj_offsets` with the 8 neighboring offsets for a board of width `x_size`.
    /// Indices 0-3 are orthogonal, 4-7 are diagonal.
    pub fn get_adjacent_offsets(adj_offsets: &mut [Loc; 8], x_size: i32) {
        let xs = x_size as Loc + 1;
        adj_offsets[0] = -xs; // up
        adj_offsets[1] = -1; // left
        adj_offsets[2] = 1; // right
        adj_offsets[3] = xs; // down
        adj_offsets[4] = -xs - 1; // up-left
        adj_offsets[5] = -xs + 1; // up-right
        adj_offsets[6] = xs - 1; // down-left
        adj_offsets[7] = xs + 1; // down-right
    }

    pub fn is_adjacent(loc0: Loc, loc1: Loc, x_size: i32) -> bool {
        let dx = (get_x(loc0, x_size) - get_x(loc1, x_size)).abs();
        let dy = (get_y(loc0, x_size) - get_y(loc1, x_size)).abs();
        (dx == 1 && dy == 0) || (dx == 0 && dy == 1)
    }

    pub fn get_mirror_loc(loc: Loc, x_size: i32, y_size: i32) -> Loc {
        if loc == NULL_LOC || loc == PASS_LOC {
            return loc;
        }
        get_loc(
            x_size - 1 - get_x(loc, x_size),
            y_size - 1 - get_y(loc, x_size),
            x_size,
        )
    }

    pub fn get_center_loc(x_size: i32, y_size: i32) -> Loc {
        if x_size % 2 == 0 || y_size % 2 == 0 {
            return NULL_LOC;
        }
        get_loc(x_size / 2, y_size / 2, x_size)
    }

    pub fn is_central(loc: Loc, x_size: i32, y_size: i32) -> bool {
        let x = get_x(loc, x_size);
        let y = get_y(loc, x_size);
        x >= (x_size - 1) / 2 && x <= x_size / 2 && y >= (y_size - 1) / 2 && y <= y_size / 2
    }

    pub fn is_near_central(loc: Loc, x_size: i32, y_size: i32) -> bool {
        let x = get_x(loc, x_size);
        let y = get_y(loc, x_size);
        x >= (x_size - 1) / 2 - 1
            && x <= x_size / 2 + 1
            && y >= (y_size - 1) / 2 - 1
            && y <= y_size / 2 + 1
    }

    pub fn distance(loc0: Loc, loc1: Loc, x_size: i32) -> i32 {
        let dx = get_x(loc1, x_size) - get_x(loc0, x_size);
        let dy = ((loc1 - loc0 - dx as Loc) / (x_size as Loc + 1)) as i32;
        dx.abs() + dy.abs()
    }

    pub fn euclidean_distance_squared(loc0: Loc, loc1: Loc, x_size: i32) -> i32 {
        let dx = get_x(loc1, x_size) - get_x(loc0, x_size);
        let dy = ((loc1 - loc0 - dx as Loc) / (x_size as Loc + 1)) as i32;
        dx * dx + dy * dy
    }

    /// Convert a location to human-readable GTP notation (e.g., "D4").
    pub fn to_string(loc: Loc, x_size: i32, y_size: i32) -> String {
        if loc == PASS_LOC {
            return "pass".to_string();
        }
        if loc == NULL_LOC {
            return "null".to_string();
        }
        if !is_on_board(loc, x_size, y_size) {
            return "offboard".to_string();
        }
        let x = get_x(loc, x_size);
        let y = get_y(loc, x_size);
        let col_letters = column_letters(x);
        format!("{}{}", col_letters, y_size - y)
    }

    fn column_letters(x: i32) -> String {
        if x <= 24 {
            column_letter(x)
        } else {
            format!("{}{}", column_letter(x / 25 - 1), column_letter(x % 25))
        }
    }

    fn column_letter(x: i32) -> String {
        // GTP skips 'I'.
        let mut c = x;
        if c >= 8 {
            c += 1;
        }
        ((b'A' + c as u8) as char).to_string()
    }

    fn parse_column_letter(ch: char) -> Option<i32> {
        let upper = ch.to_ascii_uppercase();
        match upper {
            'A'..='H' => Some((upper as u8 - b'A') as i32),
            'J'..='Z' => Some((upper as u8 - b'A' - 1) as i32),
            _ => None,
        }
    }

    pub fn is_on_board(loc: Loc, x_size: i32, y_size: i32) -> bool {
        let x = get_x(loc, x_size);
        let y = get_y(loc, x_size);
        x >= 0 && x < x_size && y >= 0 && y < y_size
    }

    /// Parse a GTP-style coordinate string into a location.
    pub fn try_of_string(s: &str, x_size: i32, y_size: i32) -> Option<Loc> {
        let trimmed = kata_core::global::trim(s);
        if trimmed.eq_ignore_ascii_case("pass") {
            return Some(PASS_LOC);
        }
        let mut chars = trimmed.chars();
        let col = chars.next()?;
        let mut x = parse_column_letter(col)?;

        // Extended two-letter column for very large boards.
        if let Some(next) = chars.clone().next() {
            if next.is_ascii_alphabetic() {
                let x1 = parse_column_letter(next)?;
                chars.next();
                x = (x + 1) * 25 + x1;
            }
        }

        let rest: String = chars.collect();
        let y: i32 = rest.parse().ok()?;
        if y < 1 || y > y_size {
            return None;
        }
        let y = y_size - y;
        if x < 0 || x >= x_size || y < 0 || y >= y_size {
            return None;
        }
        let loc = get_loc(x, y, x_size);
        if !is_on_board(loc, x_size, y_size) {
            return None;
        }
        Some(loc)
    }

    pub fn of_string(s: &str, x_size: i32, y_size: i32) -> Result<Loc, kata_core::global::IOError> {
        try_of_string(s, x_size, y_size)
            .ok_or_else(|| kata_core::global::IOError(format!("Could not parse location: {}", s)))
    }

    /// Parse a coordinate string, allowing "null" to represent `NULL_LOC`.
    pub fn try_of_string_allow_null(s: &str, x_size: i32, y_size: i32) -> Option<Loc> {
        let trimmed = kata_core::global::trim(s);
        if trimmed.eq_ignore_ascii_case("null") {
            return Some(NULL_LOC);
        }
        try_of_string(s, x_size, y_size)
    }

    pub fn of_string_allow_null(
        s: &str,
        x_size: i32,
        y_size: i32,
    ) -> Result<Loc, kata_core::global::IOError> {
        try_of_string_allow_null(s, x_size, y_size)
            .ok_or_else(|| kata_core::global::IOError(format!("Could not parse location: {}", s)))
    }

    /// Parse a comma/space-separated sequence of coordinates.
    pub fn parse_sequence(
        s: &str,
        x_size: i32,
        y_size: i32,
    ) -> Result<Vec<Loc>, kata_core::global::IOError> {
        let mut result = Vec::new();
        for token in s.split(|c: char| c == ',' || c.is_whitespace()) {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            result.push(of_string_allow_null(token, x_size, y_size)?);
        }
        Ok(result)
    }
}

/// A move consists of a location and a player.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move {
    pub loc: Loc,
    pub pla: Player,
}

impl Move {
    pub const fn new(loc: Loc, pla: Player) -> Self {
        Self { loc, pla }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}",
            player_io::player_to_string_short(self.pla),
            location::to_string(self.loc, 19, 19)
        )
    }
}

/// Tracks a chain/string/group of stones.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChainData {
    pub owner: Player,
    pub num_locs: i16,
    pub num_liberties: i16,
}

/// Record of a move, sufficient to allow undo.
#[derive(Debug, Clone, Copy)]
pub struct MoveRecord {
    pub pla: Player,
    pub loc: Loc,
    pub ko_loc: Loc,
    /// First 4 bits indicate directions of capture; fifth bit indicates suicide.
    pub cap_dirs: u8,
}

/// A Go board with chain tracking and Zobrist hashing.
#[derive(Debug, Clone)]
pub struct Board {
    pub x_size: i32,
    pub y_size: i32,
    pub colors: [Color; MAX_ARR_SIZE],

    pub chain_data: [ChainData; MAX_ARR_SIZE],
    pub chain_head: [Loc; MAX_ARR_SIZE],
    pub next_in_chain: [Loc; MAX_ARR_SIZE],

    pub ko_loc: Loc,
    pub pos_hash: Hash128,

    pub num_black_captures: i32,
    pub num_white_captures: i32,

    adj_offsets: [Loc; 8],
}

impl Default for Board {
    fn default() -> Self {
        Self::new(DEFAULT_LEN as i32, DEFAULT_LEN as i32)
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let x_chars = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
        write!(f, "  ")?;
        for x in 0..self.x_size {
            write!(f, " {}", x_chars.chars().nth(x as usize).unwrap())?;
        }
        writeln!(f)?;
        for y in 0..self.y_size {
            write!(f, "{:2} ", self.y_size - y)?;
            for x in 0..self.x_size {
                let loc = location::get_loc(x, y, self.x_size) as usize;
                write!(f, "{}", player_io::color_to_char(self.colors[loc]))?;
                if x < self.x_size - 1 {
                    write!(f, " ")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Board {
    pub fn new(x_size: i32, y_size: i32) -> Self {
        init_hash();
        let mut board = Self {
            x_size: 0,
            y_size: 0,
            colors: [C_WALL; MAX_ARR_SIZE],
            chain_data: [ChainData::default(); MAX_ARR_SIZE],
            chain_head: [NULL_LOC; MAX_ARR_SIZE],
            next_in_chain: [NULL_LOC; MAX_ARR_SIZE],
            ko_loc: NULL_LOC,
            pos_hash: Hash128::default(),
            num_black_captures: 0,
            num_white_captures: 0,
            adj_offsets: [0; 8],
        };
        board.init(x_size, y_size);
        board
    }

    fn init(&mut self, x_size: i32, y_size: i32) {
        assert!(
            x_size >= 0 && y_size >= 0 && x_size <= MAX_LEN as i32 && y_size <= MAX_LEN as i32,
            "Board::init - invalid board size"
        );

        self.x_size = x_size;
        self.y_size = y_size;
        self.colors = [C_WALL; MAX_ARR_SIZE];

        for y in 0..y_size {
            for x in 0..x_size {
                let loc = location::get_loc(x, y, x_size);
                self.colors[loc as usize] = C_EMPTY;
            }
        }

        self.ko_loc = NULL_LOC;
        self.pos_hash =
            Self::zobrist_size_x_hash(x_size as usize) ^ Self::zobrist_size_y_hash(y_size as usize);
        self.num_black_captures = 0;
        self.num_white_captures = 0;

        let xs = x_size as Loc + 1;
        self.adj_offsets[0] = -xs;
        self.adj_offsets[1] = -1;
        self.adj_offsets[2] = 1;
        self.adj_offsets[3] = xs;
        self.adj_offsets[4] = -xs - 1;
        self.adj_offsets[5] = -xs + 1;
        self.adj_offsets[6] = xs - 1;
        self.adj_offsets[7] = xs + 1;
    }

    pub fn sqrt_board_area(&self) -> f64 {
        if self.x_size == self.y_size {
            self.x_size as f64
        } else {
            ((self.x_size * self.y_size) as f64).sqrt()
        }
    }

    pub fn adj_offsets(&self) -> &[Loc; 8] {
        &self.adj_offsets
    }

    pub fn get_chain_size(&self, loc: Loc) -> i32 {
        i32::from(self.chain_data[self.chain_head[loc as usize] as usize].num_locs)
    }

    pub fn get_num_liberties(&self, loc: Loc) -> i32 {
        i32::from(self.chain_data[self.chain_head[loc as usize] as usize].num_liberties)
    }

    pub fn get_num_immediate_liberties(&self, loc: Loc) -> i32 {
        let mut num_libs = 0;
        if self.colors[(loc + self.adj_offsets[0]) as usize] == C_EMPTY {
            num_libs += 1;
        }
        if self.colors[(loc + self.adj_offsets[1]) as usize] == C_EMPTY {
            num_libs += 1;
        }
        if self.colors[(loc + self.adj_offsets[2]) as usize] == C_EMPTY {
            num_libs += 1;
        }
        if self.colors[(loc + self.adj_offsets[3]) as usize] == C_EMPTY {
            num_libs += 1;
        }
        num_libs
    }

    /// Fast lower and upper bounds on the number of liberties a new stone at
    /// `loc` for `pla` would have.
    ///
    /// Mirrors `Board::getBoundNumLibertiesAfterPlay` in `cpp/game/board.cpp`.
    pub fn get_bound_num_liberties_after_play(&self, loc: Loc, pla: Player) -> (i32, i32) {
        let opp = get_opp(pla);

        let mut num_immediate_libs = 0;
        let mut num_caps = 0;
        let mut potential_libs_from_caps = 0;
        let mut num_connection_libs = 0;
        let mut max_connection_libs = 0;

        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == C_EMPTY {
                num_immediate_libs += 1;
            } else if self.colors[adj] == opp {
                let libs = self.chain_data[self.chain_head[adj] as usize].num_liberties;
                if libs == 1 {
                    num_caps += 1;
                    potential_libs_from_caps +=
                        self.chain_data[self.chain_head[adj] as usize].num_locs as i32;
                }
            } else if self.colors[adj] == pla {
                let libs = self.chain_data[self.chain_head[adj] as usize].num_liberties;
                let conn_libs = libs as i32 - 1;
                num_connection_libs += conn_libs;
                if conn_libs > max_connection_libs {
                    max_connection_libs = conn_libs;
                }
            }
        }

        let lower_bound = num_caps
            + if max_connection_libs > num_immediate_libs {
                max_connection_libs
            } else {
                num_immediate_libs
            };
        let upper_bound = num_immediate_libs + potential_libs_from_caps + num_connection_libs;
        (lower_bound, upper_bound)
    }

    /// Returns the exact number of liberties a new `pla` stone at `loc` would
    /// have, capped at `max`.
    ///
    /// Mirrors `Board::getNumLibertiesAfterPlay` in `cpp/game/board.cpp`.
    pub fn get_num_liberties_after_play(&self, loc: Loc, pla: Player, max: i32) -> i32 {
        let opp = get_opp(pla);

        let mut libs: Vec<Loc> = Vec::new();
        let mut captured_group_heads: [Loc; 4] = [NULL_LOC; 4];
        let mut num_captured_groups: usize = 0;

        // Count immediate liberties and groups that would be captured.
        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == C_EMPTY {
                libs.push(adj as Loc);
                if libs.len() as i32 >= max {
                    return max;
                }
            } else if self.colors[adj] == opp && self.get_num_liberties(adj as Loc) == 1 {
                libs.push(adj as Loc);
                if libs.len() as i32 >= max {
                    return max;
                }

                let head = self.chain_head[adj];
                let mut already_found = false;
                for &h in captured_group_heads.iter().take(num_captured_groups) {
                    if h == head {
                        already_found = true;
                        break;
                    }
                }
                if !already_found {
                    captured_group_heads[num_captured_groups] = head;
                    num_captured_groups += 1;
                }
            }
        }

        let would_be_empty = |lc: Loc| {
            if self.colors[lc as usize] == C_EMPTY {
                return true;
            }
            if self.colors[lc as usize] == opp {
                let head = self.chain_head[lc as usize];
                for &h in captured_group_heads.iter().take(num_captured_groups) {
                    if h == head {
                        return true;
                    }
                }
            }
            false
        };

        // Walk surrounding friendly groups and count their remaining liberties.
        let mut connecting_group_heads: [Loc; 4] = [NULL_LOC; 4];
        let mut num_connecting_groups: usize = 0;
        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == pla {
                let head = self.chain_head[adj];
                let mut already_found = false;
                for &h in connecting_group_heads.iter().take(num_connecting_groups) {
                    if h == head {
                        already_found = true;
                        break;
                    }
                }
                if !already_found {
                    connecting_group_heads[num_connecting_groups] = head;
                    num_connecting_groups += 1;

                    let mut cur = adj as Loc;
                    loop {
                        for k in 0..4 {
                            let possible_lib = cur + self.adj_offsets[k];
                            if possible_lib != loc
                                && would_be_empty(possible_lib)
                                && !libs.contains(&possible_lib)
                            {
                                libs.push(possible_lib);
                                if libs.len() as i32 >= max {
                                    return max;
                                }
                            }
                        }
                        cur = self.next_in_chain[cur as usize];
                        if cur == adj as Loc {
                            break;
                        }
                    }
                }
            }
        }

        libs.len() as i32
    }

    /// Find the liberties of the chain at `loc`, writing them into `buf` and
    /// returning how many were found.
    ///
    /// `buf_start` is the first index used for duplicate checking; `buf_idx` is
    /// where writing begins. Mirrors `Board::findLiberties` in `cpp/game/board.cpp`.
    pub fn find_liberties(
        &self,
        loc: Loc,
        buf: &mut Vec<Loc>,
        buf_start: usize,
        buf_idx: usize,
    ) -> usize {
        let mut num_found = 0usize;
        let mut cur = loc;
        loop {
            for i in 0..4 {
                let lib = cur + self.adj_offsets[i];
                if self.colors[lib as usize] == C_EMPTY {
                    let mut found_dup = false;
                    let end = buf_idx + num_found;
                    for &existing in buf.iter().take(end).skip(buf_start) {
                        if existing == lib {
                            found_dup = true;
                            break;
                        }
                    }
                    if !found_dup {
                        if buf_idx + num_found >= buf.len() {
                            buf.resize(buf_idx + num_found + 1, NULL_LOC);
                        }
                        buf[buf_idx + num_found] = lib;
                        num_found += 1;
                    }
                }
            }
            cur = self.next_in_chain[cur as usize];
            if cur == loc {
                break;
            }
        }
        num_found
    }

    /// Find captures that would gain liberties for the chain at `loc`.
    ///
    /// Returns the number of capturing moves found and fills `buf` with their
    /// locations. Mirrors `Board::findLibertyGainingCaptures` in `cpp/game/board.cpp`.
    pub fn find_liberty_gaining_captures(
        &self,
        loc: Loc,
        buf: &mut Vec<Loc>,
        buf_start: usize,
        buf_idx: usize,
    ) -> usize {
        let opp = get_opp(self.colors[loc as usize]);

        let mut chain_heads_checked: Vec<Loc> = Vec::new();
        let mut num_found = 0usize;
        let mut cur = loc;
        loop {
            for i in 0..4 {
                let adj = cur + self.adj_offsets[i];
                if self.colors[adj as usize] == opp {
                    let head = self.chain_head[adj as usize];
                    if self.chain_data[head as usize].num_liberties == 1
                        && !chain_heads_checked.contains(&head)
                    {
                        num_found += self.find_liberties(adj, buf, buf_start, buf_idx + num_found);
                        chain_heads_checked.push(head);
                    }
                }
            }
            cur = self.next_in_chain[cur as usize];
            if cur == loc {
                break;
            }
        }
        num_found
    }

    /// Returns true if the chain at `loc` has at least one adjacent opponent
    /// group in atari.
    ///
    /// Mirrors `Board::hasLibertyGainingCaptures` in `cpp/game/board.cpp`.
    pub fn has_liberty_gaining_captures(&self, loc: Loc) -> bool {
        let opp = get_opp(self.colors[loc as usize]);
        let mut cur = loc;
        loop {
            for i in 0..4 {
                let adj = cur + self.adj_offsets[i];
                if self.colors[adj as usize] == opp {
                    let head = self.chain_head[adj as usize];
                    if self.chain_data[head as usize].num_liberties == 1 {
                        return true;
                    }
                }
            }
            cur = self.next_in_chain[cur as usize];
            if cur == loc {
                break;
            }
        }
        false
    }

    /// Heuristic count of connection liberties (times two) for a `pla` stone at
    /// `loc`.
    ///
    /// Mirrors `Board::countHeuristicConnectionLibertiesX2` in `cpp/game/board.cpp`.
    pub fn count_heuristic_connection_liberties_x2(&self, loc: Loc, pla: Player) -> i32 {
        let mut num_libs_x2 = 0;
        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == pla {
                let libs = self.chain_data[self.chain_head[adj] as usize].num_liberties;
                if libs > 1 {
                    num_libs_x2 += libs as i32 * 2 - 3;
                }
            }
        }
        num_libs_x2
    }

    /// Returns true if playing at `loc` would be a self-capture.
    pub fn is_suicide(&self, loc: Loc, pla: Player) -> bool {
        if loc == PASS_LOC {
            return false;
        }
        let opp = get_opp(pla);
        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == C_EMPTY {
                return false;
            } else if self.colors[adj] == pla {
                if self.get_num_liberties(adj as Loc) > 1 {
                    return false;
                }
            } else if self.colors[adj] == opp && self.get_num_liberties(adj as Loc) == 1 {
                return false;
            }
        }
        true
    }

    pub fn is_illegal_suicide(
        &self,
        loc: Loc,
        pla: Player,
        is_multi_stone_suicide_legal: bool,
    ) -> bool {
        let opp = get_opp(pla);
        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == C_EMPTY {
                return false;
            } else if self.colors[adj] == pla {
                if is_multi_stone_suicide_legal || self.get_num_liberties(adj as Loc) > 1 {
                    return false;
                }
            } else if self.colors[adj] == opp && self.get_num_liberties(adj as Loc) == 1 {
                return false;
            }
        }
        true
    }

    pub fn is_ko_banned(&self, loc: Loc) -> bool {
        loc == self.ko_loc
    }

    pub fn is_legal(&self, loc: Loc, pla: Player, is_multi_stone_suicide_legal: bool) -> bool {
        if pla != P_BLACK && pla != P_WHITE {
            return false;
        }
        loc == PASS_LOC
            || (loc >= 0
                && (loc as usize) < MAX_ARR_SIZE
                && self.colors[loc as usize] == C_EMPTY
                && !self.is_ko_banned(loc)
                && !self.is_illegal_suicide(loc, pla, is_multi_stone_suicide_legal))
    }

    pub fn is_legal_ignoring_ko(
        &self,
        loc: Loc,
        pla: Player,
        is_multi_stone_suicide_legal: bool,
    ) -> bool {
        if pla != P_BLACK && pla != P_WHITE {
            return false;
        }
        loc == PASS_LOC
            || (loc >= 0
                && (loc as usize) < MAX_ARR_SIZE
                && self.colors[loc as usize] == C_EMPTY
                && !self.is_illegal_suicide(loc, pla, is_multi_stone_suicide_legal))
    }

    pub fn is_on_board(&self, loc: Loc) -> bool {
        loc >= 0 && (loc as usize) < MAX_ARR_SIZE && self.colors[loc as usize] != C_WALL
    }

    pub fn is_simple_eye(&self, loc: Loc, pla: Player) -> bool {
        if self.colors[loc as usize] != C_EMPTY {
            return false;
        }
        let mut against_wall = false;
        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == C_WALL {
                against_wall = true;
            } else if self.colors[adj] != pla {
                return false;
            }
        }
        let opp = get_opp(pla);
        let mut num_opp_corners = 0;
        for i in 4..8 {
            let corner = (loc + self.adj_offsets[i]) as usize;
            if self.colors[corner] == opp {
                num_opp_corners += 1;
            }
        }
        !(num_opp_corners >= 2 || (against_wall && num_opp_corners >= 1))
    }

    pub fn would_be_capture(&self, loc: Loc, pla: Player) -> bool {
        if self.colors[loc as usize] != C_EMPTY {
            return false;
        }
        let opp = get_opp(pla);
        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == opp && self.get_num_liberties(adj as Loc) == 1 {
                return true;
            }
        }
        false
    }

    pub fn would_be_ko_capture(&self, loc: Loc, pla: Player) -> bool {
        if self.colors[loc as usize] != C_EMPTY {
            return false;
        }
        let opp = get_opp(pla);
        let mut opp_capturable_loc = NULL_LOC;
        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] != C_WALL && self.colors[adj] != opp {
                return false;
            }
            if self.colors[adj] == opp && self.get_num_liberties(adj as Loc) == 1 {
                if opp_capturable_loc != NULL_LOC {
                    return false;
                }
                opp_capturable_loc = adj as Loc;
            }
        }
        if opp_capturable_loc == NULL_LOC {
            return false;
        }
        self.chain_data[self.chain_head[opp_capturable_loc as usize] as usize].num_locs == 1
    }

    pub fn get_ko_capture_loc(&self, loc: Loc, pla: Player) -> Loc {
        if self.colors[loc as usize] != C_EMPTY {
            return NULL_LOC;
        }
        let opp = get_opp(pla);
        let mut opp_capturable_loc = NULL_LOC;
        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] != C_WALL && self.colors[adj] != opp {
                return NULL_LOC;
            }
            if self.colors[adj] == opp && self.get_num_liberties(adj as Loc) == 1 {
                if opp_capturable_loc != NULL_LOC {
                    return NULL_LOC;
                }
                opp_capturable_loc = adj as Loc;
            }
        }
        if opp_capturable_loc == NULL_LOC {
            return NULL_LOC;
        }
        if self.chain_data[self.chain_head[opp_capturable_loc as usize] as usize].num_locs != 1 {
            return NULL_LOC;
        }
        opp_capturable_loc
    }

    pub fn is_adjacent_to_pla(&self, loc: Loc, pla: Player) -> bool {
        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == pla {
                return true;
            }
        }
        false
    }

    pub fn is_adjacent_or_diagonal_to_pla(&self, loc: Loc, pla: Player) -> bool {
        for i in 0..8 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == pla {
                return true;
            }
        }
        false
    }

    pub fn is_adjacent_to_chain(&self, loc: Loc, chain: Loc) -> bool {
        if self.colors[chain as usize] == C_EMPTY {
            return false;
        }
        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == self.colors[chain as usize]
                && self.chain_head[adj] == self.chain_head[chain as usize]
            {
                return true;
            }
        }
        false
    }

    /// True if any orthogonal neighbor of `loc` is a `pla` stone belonging to `head`.
    pub fn is_adjacent_to_pla_head(&self, loc: Loc, pla: Player, head: Loc) -> bool {
        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == pla as Color && self.chain_head[adj] == head {
                return true;
            }
        }
        false
    }

    pub fn is_empty(&self) -> bool {
        for y in 0..self.y_size {
            for x in 0..self.x_size {
                let loc = location::get_loc(x, y, self.x_size);
                if self.colors[loc as usize] != C_EMPTY {
                    return false;
                }
            }
        }
        true
    }

    pub fn num_stones_on_board(&self) -> i32 {
        let mut num = 0;
        for y in 0..self.y_size {
            for x in 0..self.x_size {
                let loc = location::get_loc(x, y, self.x_size);
                let c = self.colors[loc as usize];
                if c == C_BLACK || c == C_WHITE {
                    num += 1;
                }
            }
        }
        num
    }

    pub fn num_pla_stones_on_board(&self, pla: Player) -> i32 {
        let mut num = 0;
        for y in 0..self.y_size {
            for x in 0..self.x_size {
                let loc = location::get_loc(x, y, self.x_size);
                if self.colors[loc as usize] == pla {
                    num += 1;
                }
            }
        }
        num
    }

    pub fn get_sit_hash_with_simple_ko(&self, pla: Player) -> Hash128 {
        let mut h = self.pos_hash;
        if self.ko_loc != NULL_LOC {
            h ^= Self::zobrist_ko_loc_hash(self.ko_loc as usize);
        }
        h ^= Self::zobrist_player_hash(pla as usize);
        h
    }

    /// Compute what `pos_hash` would be after playing `pla` at `loc`, without
    /// mutating the board. Mirrors C++ `Board::getPosHashAfterMove`.
    pub fn get_pos_hash_after_move(&self, loc: Loc, pla: Player) -> Hash128 {
        if loc == PASS_LOC {
            return self.pos_hash;
        }
        assert!(loc != NULL_LOC);

        let mut hash = self.pos_hash;
        hash ^= Self::zobrist_board_hash(loc as usize, pla as usize);

        let opp = get_opp(pla);
        let mut would_be_suicide = true;
        let mut captured_heads: [Loc; 4] = [NULL_LOC; 4];

        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == C_EMPTY
                || (self.colors[adj] == pla && self.get_num_liberties(adj as Loc) > 1)
            {
                would_be_suicide = false;
            } else if self.colors[adj] == opp && self.get_num_liberties(adj as Loc) == 1 {
                let head = self.chain_head[adj];
                if !captured_heads.iter().take(i).any(|&h| h == head) {
                    captured_heads[i] = head;
                    would_be_suicide = false;
                    let mut cur = adj as Loc;
                    loop {
                        hash ^= Self::zobrist_board_hash(cur as usize, opp as usize);
                        cur = self.next_in_chain[cur as usize];
                        if cur == adj as Loc {
                            break;
                        }
                    }
                }
            }
        }

        if would_be_suicide {
            for i in 0..4 {
                let adj = (loc + self.adj_offsets[i]) as usize;
                if self.colors[adj] == pla && self.get_num_liberties(adj as Loc) == 1 {
                    let head = self.chain_head[adj];
                    if !captured_heads.iter().take(i).any(|&h| h == head) {
                        captured_heads[i] = head;
                        let mut cur = adj as Loc;
                        loop {
                            hash ^= Self::zobrist_board_hash(cur as usize, pla as usize);
                            cur = self.next_in_chain[cur as usize];
                            if cur == adj as Loc {
                                break;
                            }
                        }
                    }
                }
            }
            hash ^= Self::zobrist_board_hash(loc as usize, pla as usize);
        }

        hash
    }

    /// Returns true if, for a move just played at `loc`, the sum of the number
    /// of stones in loc's group and the sizes of the empty regions it touches
    /// is greater than `bound`. Used by the graph hash to decide when to fully
    /// recompute the hash to guard against long cycles.
    pub fn simple_repetition_bound_gt(&self, loc: Loc, bound: i32) -> bool {
        if loc == NULL_LOC || loc == PASS_LOC {
            return false;
        }

        let mut count = 0;
        let color = self.colors[loc as usize];

        if color != C_EMPTY {
            let head = self.chain_head[loc as usize];
            count += self.chain_data[head as usize].num_locs as i32;
            if count + self.chain_data[head as usize].num_liberties as i32 > bound {
                return true;
            }
        }

        let mut empty_counted = [false; MAX_ARR_SIZE];

        if color == C_EMPTY {
            self.count_empty_helper(&mut empty_counted, loc as usize, &mut count, bound)
        } else {
            let mut cur = loc;
            loop {
                for i in 0..4 {
                    let lib = (cur + self.adj_offsets[i]) as usize;
                    if self.colors[lib] == C_EMPTY
                        && self.count_empty_helper(&mut empty_counted, lib, &mut count, bound)
                    {
                        return true;
                    }
                }
                cur = self.next_in_chain[cur as usize];
                if cur == loc {
                    break;
                }
            }
            false
        }
    }

    fn count_empty_helper(
        &self,
        empty_counted: &mut [bool],
        initial_loc: usize,
        count: &mut i32,
        bound: i32,
    ) -> bool {
        if empty_counted[initial_loc] {
            return false;
        }
        *count += 1;
        empty_counted[initial_loc] = true;
        if *count > bound {
            return true;
        }

        let mut to_expand = vec![initial_loc];
        let mut expanded = 0;
        while expanded < to_expand.len() {
            let loc = to_expand[expanded];
            expanded += 1;
            for i in 0..4 {
                let adj = (loc as Loc + self.adj_offsets[i]) as usize;
                if self.colors[adj] == C_EMPTY && !empty_counted[adj] {
                    *count += 1;
                    empty_counted[adj] = true;
                    if *count > bound {
                        return true;
                    }
                    to_expand.push(adj);
                }
            }
        }
        false
    }

    pub fn clear_simple_ko_loc(&mut self) {
        self.ko_loc = NULL_LOC;
    }

    pub fn set_simple_ko_loc(&mut self, loc: Loc) {
        self.ko_loc = loc;
    }

    pub fn set_stone(&mut self, loc: Loc, color: Color) -> bool {
        if loc < 0 || (loc as usize) >= MAX_ARR_SIZE || self.colors[loc as usize] == C_WALL {
            return false;
        }
        if color != C_BLACK && color != C_WHITE && color != C_EMPTY {
            return false;
        }

        if self.colors[loc as usize] == color {
            // nothing
        } else if self.colors[loc as usize] == C_EMPTY {
            if !self.is_illegal_suicide(loc, color, true) {
                self.play_move_assume_legal(loc, color);
            }
        } else if color == C_EMPTY {
            self.remove_single_stone(loc);
        } else {
            self.remove_single_stone(loc);
            if !self.is_suicide(loc, color) {
                self.play_move_assume_legal(loc, color);
            }
        }

        self.ko_loc = NULL_LOC;
        true
    }

    pub fn play_move(&mut self, loc: Loc, pla: Player, is_multi_stone_suicide_legal: bool) -> bool {
        if self.is_legal(loc, pla, is_multi_stone_suicide_legal) {
            self.play_move_assume_legal(loc, pla);
            true
        } else {
            false
        }
    }

    pub fn play_move_assume_legal(&mut self, loc: Loc, pla: Player) {
        if loc == PASS_LOC {
            self.ko_loc = NULL_LOC;
            return;
        }

        let opp = get_opp(pla);

        self.colors[loc as usize] = pla;
        self.pos_hash ^= Self::zobrist_board_hash(loc as usize, pla as usize);
        self.chain_data[loc as usize].owner = pla;
        self.chain_data[loc as usize].num_locs = 1;
        self.chain_data[loc as usize].num_liberties = self.get_num_immediate_liberties(loc) as i16;
        self.chain_head[loc as usize] = loc;
        self.next_in_chain[loc as usize] = loc;

        let mut num_captured = 0;
        let mut possible_ko_loc = NULL_LOC;
        let mut num_opps_seen: usize = 0;
        let mut opp_heads_seen: [Loc; 4] = [NULL_LOC; 4];

        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;

            if self.colors[adj] == pla {
                if self.chain_head[adj] == self.chain_head[loc as usize] {
                    continue;
                }
                self.chain_data[self.chain_head[adj] as usize].num_liberties -= 1;
                self.merge_chains(adj as Loc, loc);
            } else if self.colors[adj] == opp {
                let opp_head = self.chain_head[adj];

                if opp_heads_seen
                    .iter()
                    .take(num_opps_seen)
                    .any(|&h| h == opp_head)
                {
                    continue;
                }

                self.chain_data[opp_head as usize].num_liberties -= 1;
                opp_heads_seen[num_opps_seen] = opp_head;
                num_opps_seen += 1;

                if self.get_num_liberties(adj as Loc) == 0 {
                    num_captured += self.remove_chain(adj as Loc);
                    possible_ko_loc = adj as Loc;
                }
            }
        }

        if num_captured == 1
            && self.chain_data[self.chain_head[loc as usize] as usize].num_locs == 1
            && self.chain_data[self.chain_head[loc as usize] as usize].num_liberties == 1
        {
            self.ko_loc = possible_ko_loc;
        } else {
            self.ko_loc = NULL_LOC;
        }

        if pla == P_BLACK {
            self.num_white_captures += num_captured;
        } else {
            self.num_black_captures += num_captured;
        }

        if self.get_num_liberties(loc) == 0 {
            let num_suicided =
                self.chain_data[self.chain_head[loc as usize] as usize].num_locs as i32;
            self.remove_chain(loc);
            if pla == P_BLACK {
                self.num_black_captures += num_suicided;
            } else {
                self.num_white_captures += num_suicided;
            }
        }
    }

    /// Play a move and record enough information to allow `undo`.
    ///
    /// Mirrors `Board::playMoveRecorded` in `cpp/game/board.cpp`.
    pub fn play_move_recorded(&mut self, loc: Loc, pla: Player) -> MoveRecord {
        let mut record = MoveRecord {
            loc,
            pla,
            ko_loc: self.ko_loc,
            cap_dirs: 0,
        };

        if loc != PASS_LOC {
            let opp = get_opp(pla);
            for i in 0..4 {
                let adj = (loc + self.adj_offsets[i]) as usize;
                if self.colors[adj] == opp && self.get_num_liberties(adj as Loc) == 1 {
                    record.cap_dirs |= 1 << i;
                }
            }
            if record.cap_dirs == 0 && self.is_suicide(loc, pla) {
                record.cap_dirs = 0x10;
            }
        }

        self.play_move_assume_legal(loc, pla);
        record
    }

    /// Undo a move previously recorded by `play_move_recorded`.
    ///
    /// Moves MUST be undone in the order they were made. The internal chain
    /// representation after undo may differ from before the move (heads and
    /// circular-list order can change), but the board state is restored.
    ///
    /// Mirrors `Board::undo` in `cpp/game/board.cpp`.
    pub fn undo(&mut self, record: MoveRecord) {
        self.ko_loc = record.ko_loc;

        let loc = record.loc;
        if loc == PASS_LOC {
            return;
        }

        let opp = get_opp(record.pla);

        // Re-fill stones in all captured directions.
        for i in 0..4 {
            if record.cap_dirs & (1 << i) != 0 {
                let adj = (loc + self.adj_offsets[i]) as usize;
                if self.colors[adj] == C_EMPTY {
                    self.add_chain(adj as Loc, opp);
                    let num_uncaptured =
                        self.chain_data[self.chain_head[adj] as usize].num_locs as i32;
                    if record.pla == P_BLACK {
                        self.num_white_captures -= num_uncaptured;
                    } else {
                        self.num_black_captures -= num_uncaptured;
                    }
                }
            }
        }

        // Re-fill suicided stones.
        if record.cap_dirs == 0x10 {
            assert_eq!(self.colors[loc as usize], C_EMPTY);
            self.add_chain(loc, record.pla);
            let num_uncaptured =
                self.chain_data[self.chain_head[loc as usize] as usize].num_locs as i32;
            if record.pla == P_BLACK {
                self.num_black_captures -= num_uncaptured;
            } else {
                self.num_white_captures -= num_uncaptured;
            }
        }

        // Delete the stone played here.
        self.pos_hash ^= Self::zobrist_board_hash(loc as usize, self.colors[loc as usize] as usize);
        self.colors[loc as usize] = C_EMPTY;

        // Restore opponent liberties around the removed stone.
        self.change_surrounding_liberties(loc, opp, 1);

        // If this was not a single stone, we may need to recompute the chain.
        if self.chain_data[self.chain_head[loc as usize] as usize].num_locs > 1 {
            let mut num_neighbors = 0;
            for i in 0..4 {
                let adj = (loc + self.adj_offsets[i]) as usize;
                if self.colors[adj] == record.pla {
                    num_neighbors += 1;
                }
            }

            if num_neighbors <= 1 {
                // Undoing didn't disconnect the group.
                let mut head = self.chain_head[loc as usize];
                if head == loc {
                    let new_head = self.next_in_chain[loc as usize];
                    let mut cur = loc;
                    loop {
                        self.chain_head[cur as usize] = new_head;
                        cur = self.next_in_chain[cur as usize];
                        if cur == loc {
                            break;
                        }
                    }
                    self.chain_data[new_head as usize] = self.chain_data[head as usize];
                    head = new_head;
                }

                // Extract this move out of the circular list.
                let mut cur = head;
                while self.next_in_chain[cur as usize] != loc {
                    cur = self.next_in_chain[cur as usize];
                }
                self.next_in_chain[cur as usize] = self.next_in_chain[loc as usize];

                // Fix up liberties.
                let mut liberty_delta = 0;
                for i in 0..4 {
                    let adj = (loc + self.adj_offsets[i]) as usize;
                    if self.colors[adj] == C_EMPTY && !self.is_liberty_of(adj as Loc, head) {
                        liberty_delta -= 1;
                    }
                }
                liberty_delta += 1; // the removed point itself becomes a liberty
                self.chain_data[head as usize].num_liberties += liberty_delta as i16;
                self.chain_data[head as usize].num_locs -= 1;
            } else {
                // Potentially disconnected: rebuild each adjacent chain.
                let mut cur = loc;
                loop {
                    self.chain_head[cur as usize] = NULL_LOC;
                    cur = self.next_in_chain[cur as usize];
                    if cur == loc {
                        break;
                    }
                }
                for i in 0..4 {
                    let adj = (loc + self.adj_offsets[i]) as usize;
                    if self.colors[adj] == record.pla && self.chain_head[adj] == NULL_LOC {
                        self.rebuild_chain(adj as Loc, record.pla);
                    }
                }
            }
        }
    }

    fn merge_chains(&mut self, loc1: Loc, loc2: Loc) {
        let mut head1 = self.chain_head[loc1 as usize];
        let mut head2 = self.chain_head[loc2 as usize];

        assert!(head1 != head2);
        assert_eq!(
            self.chain_data[head1 as usize].owner,
            self.chain_data[head2 as usize].owner
        );

        if self.chain_data[head1 as usize].num_locs < self.chain_data[head2 as usize].num_locs {
            std::mem::swap(&mut head1, &mut head2);
        }

        self.chain_data[head1 as usize].num_locs += self.chain_data[head2 as usize].num_locs;
        let mut num_new_liberties = 0;
        let mut loc = head2;
        loop {
            for i in 0..4 {
                let adj = (loc + self.adj_offsets[i]) as usize;
                if self.colors[adj] == C_EMPTY && !self.is_liberty_of(adj as Loc, head1) {
                    num_new_liberties += 1;
                }
            }
            self.chain_head[loc as usize] = head1;
            if self.next_in_chain[loc as usize] != head2 {
                loc = self.next_in_chain[loc as usize];
            } else {
                break;
            }
        }

        self.chain_data[head1 as usize].num_liberties += num_new_liberties;

        let old_head1_next = self.next_in_chain[head1 as usize];
        self.next_in_chain[head1 as usize] = head2;
        self.next_in_chain[loc as usize] = old_head1_next;
    }

    /// Flood-fill a connected region of empty points with `pla` stones.
    ///
    /// Mirrors `Board::addChain` in `cpp/game/board.cpp`. Used by `undo` to
    /// restore captured stones.
    fn add_chain(&mut self, loc: Loc, pla: Player) {
        self.chain_data[loc as usize].num_liberties = 0;
        self.chain_data[loc as usize].num_locs = 0;
        self.chain_data[loc as usize].owner = pla;
        let front = self.add_chain_helper(loc, loc, loc, pla);
        self.next_in_chain[loc as usize] = front;
    }

    fn add_chain_helper(&mut self, head: Loc, tail_target: Loc, loc: Loc, pla: Player) -> Loc {
        self.colors[loc as usize] = pla;
        self.pos_hash ^= Self::zobrist_board_hash(loc as usize, pla as usize);
        self.chain_head[loc as usize] = head;
        self.chain_data[head as usize].num_locs += 1;
        self.next_in_chain[loc as usize] = tail_target;

        self.change_surrounding_liberties(loc, get_opp(pla), -1);

        let mut next_tail_target = loc;
        for i in 0..4 {
            let adj = (loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == C_EMPTY {
                next_tail_target = self.add_chain_helper(head, next_tail_target, adj as Loc, pla);
            }
        }
        next_tail_target
    }

    fn remove_chain(&mut self, loc: Loc) -> i32 {
        let mut num_stones_removed = 0;
        let opp = get_opp(self.colors[loc as usize]);

        let mut cur = loc;
        loop {
            self.pos_hash ^=
                Self::zobrist_board_hash(cur as usize, self.colors[cur as usize] as usize);
            self.colors[cur as usize] = C_EMPTY;
            num_stones_removed += 1;

            self.change_surrounding_liberties(cur, opp, 1);

            cur = self.next_in_chain[cur as usize];
            if cur == loc {
                break;
            }
        }

        num_stones_removed
    }

    fn remove_single_stone(&mut self, loc: Loc) {
        let pla = self.colors[loc as usize];
        let num_locs = self.chain_data[self.chain_head[loc as usize] as usize].num_locs;
        let mut locs = vec![0i32; num_locs as usize];
        let mut idx = 0;
        let mut cur = loc;
        loop {
            locs[idx] = cur as i32;
            idx += 1;
            cur = self.next_in_chain[cur as usize];
            if cur == loc {
                break;
            }
        }
        assert_eq!(idx, num_locs as usize);

        self.remove_chain(loc);

        for &l in &locs {
            if l != loc as i32 {
                self.play_move_assume_legal(l as Loc, pla);
            }
        }
    }

    fn is_liberty_of(&self, loc: Loc, head: Loc) -> bool {
        let adj0 = (loc + self.adj_offsets[0]) as usize;
        if self.colors[adj0] == self.colors[head as usize] && self.chain_head[adj0] == head {
            return true;
        }
        let adj1 = (loc + self.adj_offsets[1]) as usize;
        if self.colors[adj1] == self.colors[head as usize] && self.chain_head[adj1] == head {
            return true;
        }
        let adj2 = (loc + self.adj_offsets[2]) as usize;
        if self.colors[adj2] == self.colors[head as usize] && self.chain_head[adj2] == head {
            return true;
        }
        let adj3 = (loc + self.adj_offsets[3]) as usize;
        if self.colors[adj3] == self.colors[head as usize] && self.chain_head[adj3] == head {
            return true;
        }
        false
    }

    fn change_surrounding_liberties(&mut self, loc: Loc, pla: Player, delta: i32) {
        let adj0 = (loc + self.adj_offsets[0]) as usize;
        let adj1 = (loc + self.adj_offsets[1]) as usize;
        let adj2 = (loc + self.adj_offsets[2]) as usize;
        let adj3 = (loc + self.adj_offsets[3]) as usize;

        if self.colors[adj0] == pla {
            self.chain_data[self.chain_head[adj0] as usize].num_liberties += delta as i16;
        }
        if self.colors[adj1] == pla
            && !(self.colors[adj0] == pla && self.chain_head[adj0] == self.chain_head[adj1])
        {
            self.chain_data[self.chain_head[adj1] as usize].num_liberties += delta as i16;
        }
        if self.colors[adj2] == pla
            && !(self.colors[adj0] == pla && self.chain_head[adj0] == self.chain_head[adj2])
            && !(self.colors[adj1] == pla && self.chain_head[adj1] == self.chain_head[adj2])
        {
            self.chain_data[self.chain_head[adj2] as usize].num_liberties += delta as i16;
        }
        if self.colors[adj3] == pla
            && !(self.colors[adj0] == pla && self.chain_head[adj0] == self.chain_head[adj3])
            && !(self.colors[adj1] == pla && self.chain_head[adj1] == self.chain_head[adj3])
            && !(self.colors[adj2] == pla && self.chain_head[adj2] == self.chain_head[adj3])
        {
            self.chain_data[self.chain_head[adj3] as usize].num_liberties += delta as i16;
        }
    }

    /// Attempt to place or remove a stone at `loc`, returning `false` if the
    /// resulting placement would have zero liberties or be a capture.
    pub fn set_stone_fail_if_no_libs(&mut self, loc: Loc, color: Color) -> bool {
        if loc < 0 || (loc as usize) >= MAX_ARR_SIZE || self.colors[loc as usize] == C_WALL {
            return false;
        }
        if color != C_BLACK && color != C_WHITE && color != C_EMPTY {
            return false;
        }

        let old_ko_loc = self.ko_loc;
        if self.colors[loc as usize] == color {
            // nothing
        } else if self.colors[loc as usize] == C_EMPTY {
            if self.is_suicide(loc, color) || self.would_be_capture(loc, color) {
                return false;
            }
            self.play_move_assume_legal(loc, color);
        } else if color == C_EMPTY {
            self.remove_single_stone(loc);
        } else {
            assert_eq!(self.colors[loc as usize], get_opp(color));
            self.remove_single_stone(loc);
            if self.is_suicide(loc, color) || self.would_be_capture(loc, color) {
                self.play_move_assume_legal(loc, get_opp(color));
                self.ko_loc = old_ko_loc;
                return false;
            }
            self.play_move_assume_legal(loc, color);
        }

        self.ko_loc = NULL_LOC;
        true
    }

    /// Apply a batch of setup stones atomically, returning `false` if any
    /// placement would create a zero-liberty group.
    pub fn set_stones_fail_if_no_libs(&mut self, placements: &[Move]) -> bool {
        let mut locs = HashSet::new();
        for placement in placements {
            if !locs.insert(placement.loc) {
                return false;
            }
        }
        // First empty out all locations that we plan to set.
        for placement in placements {
            if !self.set_stone_fail_if_no_libs(placement.loc, C_EMPTY) {
                return false;
            }
        }
        // Now set all the stones we wanted.
        for placement in placements {
            if !self.set_stone_fail_if_no_libs(placement.loc, placement.pla) {
                return false;
            }
        }
        true
    }

    /// Recompute chain links, liberty counts, and the position hash from the
    /// raw `colors` array. Used after direct color manipulation (e.g. SGF setups).
    pub fn regen_chains_from_colors(&mut self) {
        self.pos_hash = Self::zobrist_size_x_hash(self.x_size as usize)
            ^ Self::zobrist_size_y_hash(self.y_size as usize);
        self.chain_head = [NULL_LOC; MAX_ARR_SIZE];
        for y in 0..self.y_size {
            for x in 0..self.x_size {
                let loc = location::get_loc(x, y, self.x_size) as usize;
                let c = self.colors[loc];
                if c == C_BLACK || c == C_WHITE {
                    self.pos_hash ^= Self::zobrist_board_hash(loc, c as usize);
                }
            }
        }
        for y in 0..self.y_size {
            for x in 0..self.x_size {
                let loc = location::get_loc(x, y, self.x_size);
                let c = self.colors[loc as usize];
                if (c == C_BLACK || c == C_WHITE) && self.chain_head[loc as usize] == NULL_LOC {
                    self.rebuild_chain(loc, c);
                }
            }
        }
    }

    fn rebuild_chain(&mut self, loc: Loc, pla: Player) {
        self.chain_data[loc as usize].num_liberties = 0;
        self.chain_data[loc as usize].num_locs = 0;
        self.chain_data[loc as usize].owner = pla;
        let front = self.rebuild_chain_helper(loc, loc, loc, pla);
        self.next_in_chain[loc as usize] = front;
    }

    fn rebuild_chain_helper(&mut self, head: Loc, tail_target: Loc, loc: Loc, pla: Player) -> Loc {
        let adj0 = (loc + self.adj_offsets[0]) as usize;
        let adj1 = (loc + self.adj_offsets[1]) as usize;
        let adj2 = (loc + self.adj_offsets[2]) as usize;
        let adj3 = (loc + self.adj_offsets[3]) as usize;

        let mut num_head_liberties = 0;
        if self.colors[adj0] == C_EMPTY && !self.is_liberty_of(adj0 as Loc, head) {
            num_head_liberties += 1;
        }
        if self.colors[adj1] == C_EMPTY && !self.is_liberty_of(adj1 as Loc, head) {
            num_head_liberties += 1;
        }
        if self.colors[adj2] == C_EMPTY && !self.is_liberty_of(adj2 as Loc, head) {
            num_head_liberties += 1;
        }
        if self.colors[adj3] == C_EMPTY && !self.is_liberty_of(adj3 as Loc, head) {
            num_head_liberties += 1;
        }
        self.chain_data[head as usize].num_liberties += num_head_liberties;

        self.chain_head[loc as usize] = head;
        self.chain_data[head as usize].num_locs += 1;
        self.next_in_chain[loc as usize] = tail_target;

        let mut next_tail_target = loc;
        if self.colors[adj0] == pla && self.chain_head[adj0] != head {
            next_tail_target = self.rebuild_chain_helper(head, next_tail_target, adj0 as Loc, pla);
        }
        if self.colors[adj1] == pla && self.chain_head[adj1] != head {
            next_tail_target = self.rebuild_chain_helper(head, next_tail_target, adj1 as Loc, pla);
        }
        if self.colors[adj2] == pla && self.chain_head[adj2] != head {
            next_tail_target = self.rebuild_chain_helper(head, next_tail_target, adj2 as Loc, pla);
        }
        if self.colors[adj3] == pla && self.chain_head[adj3] != head {
            next_tail_target = self.rebuild_chain_helper(head, next_tail_target, adj3 as Loc, pla);
        }
        next_tail_target
    }

    /// Faithfully apply setup stones to the raw colors array, then remove any
    /// zero-liberty stones simultaneously. Returns the number of stones removed.
    pub fn set_stones_tolerant(&mut self, placements: &[Move]) -> i32 {
        for placement in placements {
            let loc = placement.loc;
            let color = placement.pla;
            if loc < 0 || (loc as usize) >= MAX_ARR_SIZE || self.colors[loc as usize] == C_WALL {
                continue;
            }
            if color != C_EMPTY && color != C_BLACK && color != C_WHITE {
                continue;
            }
            self.colors[loc as usize] = color;
        }
        self.regen_chains_from_colors();

        let mut num_removed = 0;
        for y in 0..self.y_size {
            for x in 0..self.x_size {
                let loc = location::get_loc(x, y, self.x_size);
                let c = self.colors[loc as usize];
                if (c == C_BLACK || c == C_WHITE)
                    && self.chain_data[self.chain_head[loc as usize] as usize].num_liberties == 0
                {
                    self.colors[loc as usize] = C_EMPTY;
                    num_removed += 1;
                }
            }
        }
        if num_removed > 0 {
            self.regen_chains_from_colors();
        }
        self.ko_loc = NULL_LOC;
        num_removed
    }

    /// Equality for testing; chain internal order is allowed to differ.
    pub fn is_equal_for_testing(
        &self,
        other: &Board,
        check_num_captures: bool,
        check_simple_ko: bool,
    ) -> bool {
        if self.x_size != other.x_size || self.y_size != other.y_size {
            return false;
        }
        if check_simple_ko && self.ko_loc != other.ko_loc {
            return false;
        }
        if check_num_captures && self.num_black_captures != other.num_black_captures {
            return false;
        }
        if check_num_captures && self.num_white_captures != other.num_white_captures {
            return false;
        }
        if self.pos_hash != other.pos_hash {
            return false;
        }
        self.colors == other.colors
    }

    /// Serialize the board to a simple character grid.
    pub fn to_string_simple(&self, line_delimiter: char) -> String {
        let mut s = String::new();
        for y in 0..self.y_size {
            for x in 0..self.x_size {
                let loc = location::get_loc(x, y, self.x_size) as usize;
                s.push(player_io::color_to_char(self.colors[loc]));
            }
            s.push(line_delimiter);
        }
        s
    }

    /// Parse a simple character grid into a board.
    pub fn parse_board(
        x_size: i32,
        y_size: i32,
        s: &str,
        line_delimiter: char,
    ) -> Result<Self, IOError> {
        let mut board = Self::new(x_size, y_size);
        let trimmed = kata_core::global::trim(s);
        let mut lines: Vec<&str> = trimmed.split(line_delimiter).collect();
        while let Some(last) = lines.last() {
            if last.is_empty() {
                lines.pop();
            } else {
                break;
            }
        }
        if lines.len() == (y_size + 1) as usize {
            let first = kata_core::global::trim(lines[0]);
            if !first.is_empty() && first.starts_with('A') {
                lines.remove(0);
            }
        }
        if lines.len() != y_size as usize {
            return Err(IOError(format!(
                "Board::parse_board - string has different number of board rows than y_size: expected {}, got {}",
                y_size,
                lines.len()
            )));
        }
        for y in 0..y_size {
            let line = kata_core::global::trim(lines[y as usize]);
            let first_non_digit_idx = line.find(|c: char| !c.is_ascii_digit()).unwrap_or(0);
            let mut line = &line[first_non_digit_idx..];
            line = kata_core::global::trim(line);
            if line.len() != x_size as usize && line.len() != (2 * x_size - 1) as usize {
                return Err(IOError(format!(
                    "Board::parse_board - line length not compatible with x_size: {}",
                    line.len()
                )));
            }
            for x in 0..x_size {
                let c = if line.len() == x_size as usize {
                    line.as_bytes()[x as usize]
                } else {
                    line.as_bytes()[(x * 2) as usize]
                } as char;
                let loc = location::get_loc(x, y, board.x_size);
                if ". ,*`".contains(c) {
                    continue;
                } else if c == 'o' || c == 'O' {
                    if !board.set_stone_fail_if_no_libs(loc, P_WHITE) {
                        return Err(IOError(format!(
                            "Board::parse_board - zero-liberty group near {}",
                            location::to_string(loc, board.x_size, board.y_size)
                        )));
                    }
                } else if c == 'x' || c == 'X' {
                    if !board.set_stone_fail_if_no_libs(loc, P_BLACK) {
                        return Err(IOError(format!(
                            "Board::parse_board - zero-liberty group near {}",
                            location::to_string(loc, board.x_size, board.y_size)
                        )));
                    }
                } else {
                    return Err(IOError(format!(
                        "Board::parse_board - could not parse board character: {}",
                        c
                    )));
                }
            }
        }
        Ok(board)
    }

    /// Serialize the board to a JSON object matching the C++ `Board::toJson` format.
    pub fn to_json(&self) -> Value {
        json!({
            "xSize": self.x_size,
            "ySize": self.y_size,
            "stones": self.to_string_simple('|'),
            "koLoc": location::to_string(self.ko_loc, self.x_size, self.y_size),
            "numBlackCaptures": self.num_black_captures,
            "numWhiteCaptures": self.num_white_captures,
        })
    }

    /// Parse a JSON object produced by [`Self::to_json`] or C++ `Board::toJson`.
    pub fn of_json(data: &Value) -> Result<Self, IOError> {
        let x_size = data["xSize"]
            .as_i64()
            .ok_or_else(|| IOError("Board::of_json missing xSize".to_string()))?
            as i32;
        let y_size = data["ySize"]
            .as_i64()
            .ok_or_else(|| IOError("Board::of_json missing ySize".to_string()))?
            as i32;
        let stones = data["stones"]
            .as_str()
            .ok_or_else(|| IOError("Board::of_json missing stones".to_string()))?;
        let mut board = Self::parse_board(x_size, y_size, stones, '|')?;
        let ko_loc_str = data["koLoc"]
            .as_str()
            .ok_or_else(|| IOError("Board::of_json missing koLoc".to_string()))?;
        board.set_simple_ko_loc(location::of_string_allow_null(ko_loc_str, x_size, y_size)?);
        board.num_black_captures = data["numBlackCaptures"]
            .as_i64()
            .ok_or_else(|| IOError("Board::of_json missing numBlackCaptures".to_string()))?
            as i32;
        board.num_white_captures = data["numWhiteCaptures"]
            .as_i64()
            .ok_or_else(|| IOError("Board::of_json missing numWhiteCaptures".to_string()))?
            as i32;
        Ok(board)
    }

    // Zobrist hash tables.
    pub fn zobrist_player_hash(idx: usize) -> Hash128 {
        ZOBRIST_HASHES.get().unwrap().player[idx]
    }

    fn zobrist_board_hash(loc: usize, color: usize) -> Hash128 {
        ZOBRIST_HASHES.get().unwrap().board[loc][color]
    }

    /// Secondary independent board Zobrist table, used by the opening book.
    pub fn zobrist_board_hash2(loc: usize, color: usize) -> Hash128 {
        ZOBRIST_HASHES.get().unwrap().board2[loc][color]
    }

    pub fn zobrist_ko_loc_hash(loc: usize) -> Hash128 {
        ZOBRIST_HASHES.get().unwrap().ko_loc[loc]
    }

    pub fn zobrist_ko_mark_hash(loc: usize, color: usize) -> Hash128 {
        ZOBRIST_HASHES.get().unwrap().ko_mark[loc][color]
    }

    pub fn zobrist_encore_hash(phase: usize) -> Hash128 {
        ZOBRIST_HASHES.get().unwrap().encore[phase]
    }

    pub fn zobrist_second_encore_start_hash(loc: usize, color: usize) -> Hash128 {
        ZOBRIST_HASHES.get().unwrap().second_encore_start[loc][color]
    }

    pub const ZOBRIST_PASS_ENDS_PHASE: Hash128 =
        Hash128::new(0x853E097C279EBF4E, 0xE3153DEF9E14A62C);
    pub const ZOBRIST_GAME_IS_OVER: Hash128 = Hash128::new(0xb6f9e465597a77ee, 0xf1d583d960a4ce7f);

    pub fn zobrist_size_x_hash(size: usize) -> Hash128 {
        ZOBRIST_HASHES.get().unwrap().size_x[size]
    }

    /// Compute pass-alive area and territory for `pla` using Benson's algorithm.
    ///
    /// Mirrors `Board::calculateAreaForPla` in `cpp/game/board.cpp`.
    /// Results are written into `result` for locations owned by `pla`.
    pub fn calculate_area_for_pla(
        &self,
        pla: Player,
        safe_big_territories: bool,
        unsafe_big_territories: bool,
        is_multi_stone_suicide_legal: bool,
        result: &mut [Color],
    ) {
        let opp = get_opp(pla as Color);
        let max_regions = (MAX_LEN * MAX_LEN).div_ceil(2) + 1;
        let vital_for_pla_heads_lists_max_len = max_regions * 4;

        let mut region_idx_by_loc = [-1i16; MAX_ARR_SIZE];
        let mut next_empty_or_opp = [NULL_LOC; MAX_ARR_SIZE];
        let mut borders_non_pass_alive_pla_by_head = [false; MAX_ARR_SIZE];
        // Pre-sized buffer, matching C++'s fixed array. Each region writes into
        // its slice starting at vital_start[region_idx]; using direct indexing
        // instead of push is required because regions are packed by final length
        // but previous regions may have left stale entries beyond their final len.
        let mut vital_for_pla_heads_lists = vec![NULL_LOC; vital_for_pla_heads_lists_max_len];
        let mut region_heads = Vec::with_capacity(max_regions);
        let mut vital_start = Vec::with_capacity(max_regions);
        let mut vital_len = Vec::with_capacity(max_regions);
        let mut num_internal_spaces_max2 = Vec::with_capacity(max_regions);
        let mut contains_opp = Vec::with_capacity(max_regions);

        let mut build_region_queue = [NULL_LOC; MAX_ARR_SIZE];

        let mut num_regions = 0usize;
        let mut vital_for_pla_heads_lists_total = 0usize;
        let mut at_least_one_pla = false;

        for y in 0..self.y_size {
            for x in 0..self.x_size {
                let loc = location::get_loc(x, y, self.x_size) as usize;
                if region_idx_by_loc[loc] != -1 {
                    continue;
                }
                if self.colors[loc] != C_EMPTY {
                    at_least_one_pla |= self.colors[loc] == pla as Color;
                    continue;
                }

                let region_idx = num_regions;
                num_regions += 1;
                assert!(num_regions <= max_regions, "too many regions");

                region_heads.push(loc as Loc);
                vital_start.push(vital_for_pla_heads_lists_total as u16);
                vital_len.push(0u16);
                num_internal_spaces_max2.push(0u8);
                contains_opp.push(false);

                // Fill in all adjacent pla heads as vital.
                {
                    let v_start = vital_for_pla_heads_lists_total;
                    assert!(
                        v_start + 4 <= vital_for_pla_heads_lists_max_len,
                        "vital list overflow"
                    );
                    let mut initial_v_len = 0usize;
                    for i in 0..4 {
                        let adj = (loc as Loc + self.adj_offsets[i]) as usize;
                        if self.colors[adj] == pla as Color {
                            let pla_head = self.chain_head[adj];
                            let mut already_present = false;
                            for j in 0..initial_v_len {
                                if vital_for_pla_heads_lists[v_start + j] == pla_head {
                                    already_present = true;
                                    break;
                                }
                            }
                            if !already_present {
                                vital_for_pla_heads_lists[v_start + initial_v_len] = pla_head;
                                initial_v_len += 1;
                            }
                        }
                    }
                    vital_len[region_idx] = initial_v_len as u16;
                }

                // Build the region via BFS, forming a circular linked list through next_empty_or_opp.
                let mut tail_target = loc as Loc;
                let mut is_v_len_non_zero = vital_len[region_idx] > 0;
                let mut queue_head = 0usize;
                let mut queue_tail = 1usize;
                build_region_queue[0] = loc as Loc;
                region_idx_by_loc[loc] = region_idx as i16;

                while queue_head != queue_tail {
                    let current = build_region_queue[queue_head];
                    let current_idx = current as usize;
                    queue_head += 1;

                    // Filter out pla heads we're not actually adjacent to.
                    if is_v_len_non_zero
                        && (is_multi_stone_suicide_legal || self.colors[current_idx] == C_EMPTY)
                    {
                        let v_start = vital_start[region_idx] as usize;
                        let old_v_len = vital_len[region_idx] as usize;
                        let mut new_v_len = 0usize;
                        for i in 0..old_v_len {
                            if self.is_adjacent_to_pla_head(
                                current,
                                pla,
                                vital_for_pla_heads_lists[v_start + i],
                            ) {
                                vital_for_pla_heads_lists[v_start + new_v_len] =
                                    vital_for_pla_heads_lists[v_start + i];
                                new_v_len += 1;
                            }
                        }
                        vital_len[region_idx] = new_v_len as u16;
                        is_v_len_non_zero = new_v_len > 0;
                    }

                    // Track internal spaces (max 2).
                    if num_internal_spaces_max2[region_idx] < 2
                        && !self.is_adjacent_to_pla(current, pla)
                    {
                        num_internal_spaces_max2[region_idx] += 1;
                    }

                    if self.colors[current_idx] == opp {
                        contains_opp[region_idx] = true;
                    }

                    next_empty_or_opp[current_idx] = tail_target;
                    tail_target = current;

                    // Push adjacent empty/opp locations.
                    for i in 0..4 {
                        let adj = (current + self.adj_offsets[i]) as usize;
                        if (self.colors[adj] == C_EMPTY || self.colors[adj] == opp)
                            && region_idx_by_loc[adj] == -1
                        {
                            build_region_queue[queue_tail] = adj as Loc;
                            queue_tail += 1;
                            region_idx_by_loc[adj] = region_idx as i16;
                        }
                    }
                }

                assert!(queue_tail < MAX_ARR_SIZE, "region queue overflow");
                next_empty_or_opp[loc] = tail_target;
                vital_for_pla_heads_lists_total += vital_len[region_idx] as usize;
            }
        }

        // Collect all pla heads.
        let mut all_pla_heads = Vec::with_capacity(MAX_PLAY_SIZE);
        for loc in 0..MAX_ARR_SIZE {
            if self.colors[loc] == pla as Color && self.chain_head[loc] == loc as Loc {
                all_pla_heads.push(loc as Loc);
            }
        }
        let num_pla_heads = all_pla_heads.len();
        let mut pla_has_been_killed = vec![false; num_pla_heads];

        let mut vital_count_by_pla_head = [0u16; MAX_ARR_SIZE];
        for i in 0..num_pla_heads {
            vital_count_by_pla_head[all_pla_heads[i] as usize] = 0;
        }

        // Accumulate vital liberties per pla head.
        for i in 0..num_regions {
            let v_start = vital_start[i] as usize;
            let v_len = vital_len[i] as usize;
            for j in 0..v_len {
                let pla_head = vital_for_pla_heads_lists[v_start + j];
                vital_count_by_pla_head[pla_head as usize] += 1;
            }
        }

        // Benson iteration: kill pla heads with fewer than 2 vital liberties.
        loop {
            let mut killed_anything = false;
            for i in 0..num_pla_heads {
                if pla_has_been_killed[i] {
                    continue;
                }
                let pla_head = all_pla_heads[i];
                if vital_count_by_pla_head[pla_head as usize] < 2 {
                    pla_has_been_killed[i] = true;
                    killed_anything = true;

                    let mut cur = pla_head;
                    loop {
                        for j in 0..4 {
                            let adj = (cur + self.adj_offsets[j]) as usize;
                            let region_idx = region_idx_by_loc[adj];
                            if region_idx < 0
                                || !(self.colors[adj] == C_EMPTY || self.colors[adj] == opp)
                            {
                                continue;
                            }
                            let head = region_heads[region_idx as usize];
                            if !borders_non_pass_alive_pla_by_head[head as usize] {
                                borders_non_pass_alive_pla_by_head[head as usize] = true;
                                let v_start = vital_start[region_idx as usize] as usize;
                                let v_len = vital_len[region_idx as usize] as usize;
                                for k in 0..v_len {
                                    let pla_h = vital_for_pla_heads_lists[v_start + k];
                                    vital_count_by_pla_head[pla_h as usize] -= 1;
                                }
                            }
                        }
                        cur = self.next_in_chain[cur as usize];
                        if cur == pla_head {
                            break;
                        }
                    }
                }
            }
            if !killed_anything {
                break;
            }
        }

        // Mark pass-alive groups in result.
        for i in 0..num_pla_heads {
            if !pla_has_been_killed[i] {
                let pla_head = all_pla_heads[i];
                let mut cur = pla_head;
                loop {
                    result[cur as usize] = pla as Color;
                    cur = self.next_in_chain[cur as usize];
                    if cur == pla_head {
                        break;
                    }
                }
            }
        }

        // Mark territory.
        for i in 0..num_regions {
            let head = region_heads[i];
            let head_idx = head as usize;

            let mut should_mark = num_internal_spaces_max2[i] <= 1
                && !borders_non_pass_alive_pla_by_head[head_idx]
                && at_least_one_pla;
            should_mark = should_mark
                || (safe_big_territories
                    && !contains_opp[i]
                    && !borders_non_pass_alive_pla_by_head[head_idx]
                    && at_least_one_pla);

            if should_mark {
                let mut cur = head;
                loop {
                    result[cur as usize] = pla as Color;
                    cur = next_empty_or_opp[cur as usize];
                    if cur == head {
                        break;
                    }
                }
            } else {
                let should_mark_if_empty =
                    unsafe_big_territories && !contains_opp[i] && at_least_one_pla;
                if should_mark_if_empty {
                    let mut cur = head;
                    loop {
                        if result[cur as usize] == C_EMPTY {
                            result[cur as usize] = pla as Color;
                        }
                        cur = next_empty_or_opp[cur as usize];
                        if cur == head {
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Compute pass-alive area and territory for both players.
    ///
    /// Mirrors `Board::calculateArea` in `cpp/game/board.cpp`.
    pub fn calculate_area(
        &self,
        result: &mut [Color],
        non_pass_alive_stones: bool,
        safe_big_territories: bool,
        unsafe_big_territories: bool,
        is_multi_stone_suicide_legal: bool,
    ) {
        assert_eq!(result.len(), MAX_ARR_SIZE);
        result.fill(C_EMPTY);
        self.calculate_area_for_pla(
            P_BLACK,
            safe_big_territories,
            unsafe_big_territories,
            is_multi_stone_suicide_legal,
            result,
        );
        self.calculate_area_for_pla(
            P_WHITE,
            safe_big_territories,
            unsafe_big_territories,
            is_multi_stone_suicide_legal,
            result,
        );

        if non_pass_alive_stones {
            for y in 0..self.y_size {
                for x in 0..self.x_size {
                    let loc = location::get_loc(x, y, self.x_size) as usize;
                    if result[loc] == C_EMPTY {
                        result[loc] = self.colors[loc];
                    }
                }
            }
        }
    }

    /// Helper for `calculate_independent_life_area`.
    ///
    /// Mirrors `Board::calculateIndependentLifeAreaHelper` in `cpp/game/board.cpp`.
    fn calculate_independent_life_area_helper(
        &self,
        basic_area: &[Color],
        result: &mut [Color],
    ) -> i32 {
        let mut is_seki = [false; MAX_ARR_SIZE];
        let mut queue = [NULL_LOC; MAX_ARR_SIZE];
        let mut queue_head = 0usize;
        let mut queue_tail = 0usize;

        // Mark all regions touching dame or containing an atari stone as seki.
        for y in 0..self.y_size {
            for x in 0..self.x_size {
                let loc = location::get_loc(x, y, self.x_size) as usize;
                if basic_area[loc] != C_EMPTY && !is_seki[loc] {
                    let is_atari_stone = self.colors[loc] == basic_area[loc]
                        && self.get_num_liberties(loc as Loc) == 1;
                    let touches_dame = (0..4).any(|i| {
                        let adj = (loc as Loc + self.adj_offsets[i]) as usize;
                        self.colors[adj] == C_EMPTY && basic_area[adj] == C_EMPTY
                    });
                    if is_atari_stone || touches_dame {
                        let pla = basic_area[loc];
                        is_seki[loc] = true;
                        queue[queue_tail] = loc as Loc;
                        queue_tail += 1;
                        while queue_head != queue_tail {
                            let next_loc = queue[queue_head] as usize;
                            queue_head += 1;
                            for i in 0..4 {
                                let adj = (next_loc as Loc + self.adj_offsets[i]) as usize;
                                if basic_area[adj] == pla && !is_seki[adj] {
                                    is_seki[adj] = true;
                                    queue[queue_tail] = adj as Loc;
                                    queue_tail += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Copy non-seki basic areas into result, counting regions.
        let mut white_minus_black_independent_life_region_count = 0i32;
        queue_head = 0;
        queue_tail = 0;
        for y in 0..self.y_size {
            for x in 0..self.x_size {
                let loc = location::get_loc(x, y, self.x_size) as usize;
                if basic_area[loc] != C_EMPTY && !is_seki[loc] && result[loc] != basic_area[loc] {
                    let pla = basic_area[loc];
                    white_minus_black_independent_life_region_count +=
                        if pla == C_WHITE { 1 } else { -1 };
                    result[loc] = basic_area[loc];
                    queue[queue_tail] = loc as Loc;
                    queue_tail += 1;
                    while queue_head != queue_tail {
                        let next_loc = queue[queue_head] as usize;
                        queue_head += 1;
                        for i in 0..4 {
                            let adj = (next_loc as Loc + self.adj_offsets[i]) as usize;
                            if basic_area[adj] == pla && result[adj] != basic_area[adj] {
                                result[adj] = basic_area[adj];
                                queue[queue_tail] = adj as Loc;
                                queue_tail += 1;
                            }
                        }
                    }
                }
            }
        }

        white_minus_black_independent_life_region_count
    }

    /// Compute independent life area used for territory scoring with tax.
    ///
    /// Mirrors `Board::calculateIndependentLifeArea` in `cpp/game/board.cpp`.
    pub fn calculate_independent_life_area(
        &self,
        result: &mut [Color],
        keep_territories: bool,
        keep_stones: bool,
        is_multi_stone_suicide_legal: bool,
    ) -> i32 {
        assert_eq!(result.len(), MAX_ARR_SIZE);
        let mut basic_area = [C_EMPTY; MAX_ARR_SIZE];
        result.fill(C_EMPTY);
        self.calculate_area_for_pla(
            P_BLACK,
            true,
            true,
            is_multi_stone_suicide_legal,
            &mut basic_area,
        );
        self.calculate_area_for_pla(
            P_WHITE,
            true,
            true,
            is_multi_stone_suicide_legal,
            &mut basic_area,
        );

        for y in 0..self.y_size {
            for x in 0..self.x_size {
                let loc = location::get_loc(x, y, self.x_size) as usize;
                if basic_area[loc] == C_EMPTY {
                    basic_area[loc] = self.colors[loc];
                }
            }
        }

        let count = self.calculate_independent_life_area_helper(&basic_area, result);

        if keep_territories {
            for y in 0..self.y_size {
                for x in 0..self.x_size {
                    let loc = location::get_loc(x, y, self.x_size) as usize;
                    if basic_area[loc] != C_EMPTY && basic_area[loc] != self.colors[loc] {
                        result[loc] = basic_area[loc];
                    }
                }
            }
        }
        if keep_stones {
            for y in 0..self.y_size {
                for x in 0..self.x_size {
                    let loc = location::get_loc(x, y, self.x_size) as usize;
                    if basic_area[loc] != C_EMPTY && basic_area[loc] == self.colors[loc] {
                        result[loc] = basic_area[loc];
                    }
                }
            }
        }

        count
    }

    /// Does this connect two pla distinct groups that are not both pass-alive
    /// and not within opponent pass-alive area either?
    ///
    /// Mirrors `Board::isNonPassAliveSelfConnection` in `cpp/game/board.cpp`.
    pub fn is_non_pass_alive_self_connection(
        &self,
        loc: Loc,
        pla: Player,
        pass_alive_area: &[Color],
    ) -> bool {
        assert_eq!(pass_alive_area.len(), MAX_ARR_SIZE);
        let loc = loc as usize;
        if self.colors[loc] != C_EMPTY || pass_alive_area[loc] == pla {
            return false;
        }

        let mut non_pass_alive_adj_head = NULL_LOC;
        for i in 0..4 {
            let adj = (loc as Loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == pla && pass_alive_area[adj] == C_EMPTY {
                non_pass_alive_adj_head = self.chain_head[adj];
                break;
            }
        }

        if non_pass_alive_adj_head == NULL_LOC {
            return false;
        }

        for i in 0..4 {
            let adj = (loc as Loc + self.adj_offsets[i]) as usize;
            if self.colors[adj] == pla && self.chain_head[adj] != non_pass_alive_adj_head {
                return true;
            }
        }

        false
    }

    pub fn zobrist_size_y_hash(size: usize) -> Hash128 {
        ZOBRIST_HASHES.get().unwrap().size_y[size]
    }

    /// Maximum number of nodes searched by the ladder helpers.
    const MAX_LADDER_SEARCH_NODE_BUDGET: i32 = 25000;

    /// Returns true if the group at `loc` is ladder-captured.
    ///
    /// When `defender_first` is true, the defender (the owner of the group) is
    /// to move and the group must have exactly one liberty. When false, the
    /// attacker is to move and the group may have one or two liberties.
    ///
    /// This implementation uses board cloning rather than C++-style play/undo.
    /// Mirrors `Board::searchIsLadderCaptured` in `cpp/game/board.cpp`.
    pub fn search_is_ladder_captured(&self, loc: Loc, defender_first: bool) -> bool {
        if !self.is_on_board(loc) {
            return false;
        }
        let c = self.colors[loc as usize];
        if c != C_BLACK && c != C_WHITE {
            return false;
        }
        let libs = self.get_num_liberties(loc);
        if libs > 2 || (defender_first && libs > 1) {
            return false;
        }

        let mut board = self.clone();
        if defender_first {
            board.ko_loc = NULL_LOC;
        }
        let mut node_count = 0;
        board.ladder_search_rec(loc, defender_first, &mut node_count)
    }

    /// Returns true if the two-liberty group at `loc` is ladder-captured, along
    /// with the attacker moves that succeed.
    ///
    /// Mirrors `Board::searchIsLadderCapturedAttackerFirst2Libs` in
    /// `cpp/game/board.cpp`.
    pub fn search_is_ladder_captured_attacker_first_2_libs(&self, loc: Loc) -> (bool, Vec<Loc>) {
        if !self.is_on_board(loc) {
            return (false, Vec::new());
        }
        let c = self.colors[loc as usize];
        if c != C_BLACK && c != C_WHITE {
            return (false, Vec::new());
        }
        if self.get_num_liberties(loc) != 2 {
            return (false, Vec::new());
        }

        let pla = c;
        let opp = get_opp(pla);
        let mut buf = Vec::new();
        let num_libs = self.find_liberties(loc, &mut buf, 0, 0);
        assert_eq!(num_libs, 2);

        let move0 = buf[0];
        let move1 = buf[1];
        let mut working_moves = Vec::new();
        let is_multi_stone_suicide_legal = false;

        if self.is_legal(move0, opp, is_multi_stone_suicide_legal) {
            let mut copy = self.clone();
            copy.play_move_assume_legal(move0, opp);
            if copy.search_is_ladder_captured(loc, true) {
                working_moves.push(move0);
            }
        }
        if self.is_legal(move1, opp, is_multi_stone_suicide_legal) {
            let mut copy = self.clone();
            copy.play_move_assume_legal(move1, opp);
            if copy.search_is_ladder_captured(loc, true) {
                working_moves.push(move1);
            }
        }

        (!working_moves.is_empty(), working_moves)
    }

    fn ladder_search_rec(&self, loc: Loc, is_defender_turn: bool, node_count: &mut i32) -> bool {
        if *node_count >= Self::MAX_LADDER_SEARCH_NODE_BUDGET {
            return false;
        }

        let libs = self.get_num_liberties(loc);
        if is_defender_turn {
            if libs >= 2 {
                return false;
            }
            if self.ko_loc != NULL_LOC {
                return false;
            }
        } else {
            if libs <= 1 {
                return true;
            }
            if libs >= 3 {
                return false;
            }
        }

        let pla = self.colors[loc as usize];
        let opp = get_opp(pla);
        let mut buf = Vec::new();
        let mut move_list: Vec<Loc> = Vec::new();

        if is_defender_turn {
            let num_captures = self.find_liberty_gaining_captures(loc, &mut buf, 0, 0);
            let num_libs_found = self.find_liberties(loc, &mut buf, 0, num_captures);
            let move_list_len = num_captures + num_libs_found;
            if move_list_len == 0 {
                return true;
            }

            let last_move = buf[move_list_len - 1];
            let (lower_bound, upper_bound) =
                self.get_bound_num_liberties_after_play(last_move, pla);
            if lower_bound >= 3 {
                return false;
            }
            if move_list_len == 1 && upper_bound <= 1 {
                return true;
            }

            move_list.extend_from_slice(&buf[..move_list_len]);
        } else {
            let num_libs_found = self.find_liberties(loc, &mut buf, 0, 0);
            assert_eq!(num_libs_found, 2);

            let mut move0 = buf[0];
            let mut move1 = buf[1];
            let mut libs0 = self.get_num_immediate_liberties(move0);
            let libs1 = self.get_num_immediate_liberties(move1);

            // Double-ko death heuristic.
            if libs0 == 0
                && libs1 == 0
                && self.would_be_ko_capture(move0, opp)
                && self.would_be_ko_capture(move1, opp)
                && self.get_num_liberties_after_play(move0, pla, 3) <= 2
                && self.get_num_liberties_after_play(move1, pla, 3) <= 2
                && !self.has_liberty_gaining_captures(loc)
            {
                return true;
            }

            let mut move_list_len = 2usize;

            // Early quitouts if the two liberties are not adjacent.
            if !location::is_adjacent(move0, move1, self.x_size) {
                if libs0 >= 3 && libs1 >= 3 {
                    return false;
                } else if libs0 >= 3 {
                    move_list_len = 1;
                } else if libs1 >= 3 {
                    move0 = move1;
                    libs0 = libs1;
                    move_list_len = 1;
                }
            }

            // Order the moves using a simple connection-liberty heuristic.
            if move_list_len > 1 {
                let heuristic0 =
                    libs0 * 2 + self.count_heuristic_connection_liberties_x2(move0, pla);
                let heuristic1 =
                    libs1 * 2 + self.count_heuristic_connection_liberties_x2(move1, pla);
                if heuristic1 > heuristic0 {
                    std::mem::swap(&mut move0, &mut move1);
                }
            }

            move_list.push(move0);
            if move_list_len > 1 {
                move_list.push(move1);
            }
        }

        let p = if is_defender_turn { pla } else { opp };
        for &mv in &move_list {
            if !self.is_legal(mv, p, false) {
                // Illegal moves count as a failed branch for the player to move.
                continue;
            }

            *node_count += 1;
            if *node_count >= Self::MAX_LADDER_SEARCH_NODE_BUDGET {
                return false;
            }

            let mut next_board = self.clone();
            next_board.play_move_assume_legal(mv, p);
            let result = next_board.ladder_search_rec(loc, !is_defender_turn, node_count);
            if is_defender_turn && !result {
                return false;
            }
            if !is_defender_turn && result {
                return true;
            }
        }

        // No move escaped/captured.
        is_defender_turn
    }
}

struct ZobristHashes {
    player: [Hash128; 4],
    board: [[Hash128; 4]; MAX_ARR_SIZE],
    board2: [[Hash128; 4]; MAX_ARR_SIZE],
    ko_loc: [Hash128; MAX_ARR_SIZE],
    ko_mark: [[Hash128; 4]; MAX_ARR_SIZE],
    encore: [Hash128; 3],
    second_encore_start: [[Hash128; 4]; MAX_ARR_SIZE],
    size_x: [Hash128; MAX_LEN + 1],
    size_y: [Hash128; MAX_LEN + 1],
}

impl Default for ZobristHashes {
    fn default() -> Self {
        let mut h = Self {
            player: [Hash128::default(); 4],
            board: [[Hash128::default(); 4]; MAX_ARR_SIZE],
            board2: [[Hash128::default(); 4]; MAX_ARR_SIZE],
            ko_loc: [Hash128::default(); MAX_ARR_SIZE],
            ko_mark: [[Hash128::default(); 4]; MAX_ARR_SIZE],
            encore: [Hash128::default(); 3],
            second_encore_start: [[Hash128::default(); 4]; MAX_ARR_SIZE],
            size_x: [Hash128::default(); MAX_LEN + 1],
            size_y: [Hash128::default(); MAX_LEN + 1],
        };

        let mut seed: u64 = 0x9e3779b97f4a7c15;
        let mut next_hash = || {
            seed = seed.wrapping_add(0x9e3779b97f4a7c15);
            let h0 = hash::split_mix64(seed);
            let h1 = hash::split_mix64(h0);
            Hash128::new(h0, h1)
        };

        for i in 0..4 {
            h.player[i] = next_hash();
        }

        for i in 0..3 {
            h.encore[i] = next_hash();
        }

        for i in 0..MAX_ARR_SIZE {
            for j in 0..4 {
                if j == C_EMPTY as usize || j == C_WALL as usize {
                    h.board[i][j] = Hash128::default();
                    h.ko_mark[i][j] = Hash128::default();
                } else {
                    h.board[i][j] = next_hash();
                    h.ko_mark[i][j] = next_hash();
                }
            }
            h.ko_loc[i] = next_hash();
        }

        for i in 0..MAX_ARR_SIZE {
            for j in 0..4 {
                if j == C_EMPTY as usize || j == C_WALL as usize {
                    h.second_encore_start[i][j] = Hash128::default();
                } else {
                    h.second_encore_start[i][j] = next_hash();
                }
            }
        }

        for i in 0..=MAX_LEN {
            h.size_x[i] = next_hash();
            h.size_y[i] = next_hash();
        }

        for i in 0..MAX_ARR_SIZE {
            for j in 0..4 {
                if j == C_EMPTY as usize || j == C_WALL as usize {
                    h.board2[i][j] = Hash128::default();
                } else {
                    h.board2[i][j] = next_hash();
                }
            }
        }

        h
    }
}

use std::sync::OnceLock;
static ZOBRIST_HASHES: OnceLock<ZobristHashes> = OnceLock::new();

fn init_hash() {
    ZOBRIST_HASHES.get_or_init(ZobristHashes::default);
}

#[cfg(test)]
mod tests {
    use super::location;
    use super::*;

    #[test]
    fn test_get_opp() {
        assert_eq!(get_opp(C_BLACK), C_WHITE);
        assert_eq!(get_opp(C_WHITE), C_BLACK);
        assert_eq!(get_opp(C_EMPTY), C_WALL);
    }

    #[test]
    fn test_player_io() {
        assert_eq!(player_io::color_to_char(C_BLACK), 'X');
        assert_eq!(player_io::player_to_string_short(P_WHITE), "W");
        assert_eq!(player_io::parse_player("b").unwrap(), P_BLACK);
        assert!(player_io::try_parse_player("red").is_none());
    }

    #[test]
    fn test_location_helpers() {
        let x_size = 19;
        let y_size = 19;
        let loc = location::get_loc(3, 4, x_size);
        assert_eq!(location::get_x(loc, x_size), 3);
        assert_eq!(location::get_y(loc, x_size), 4);
        assert!(location::is_on_board(loc, x_size, y_size));

        let mut offsets = [0; 8];
        location::get_adjacent_offsets(&mut offsets, x_size);
        assert_eq!(offsets[3], x_size as Loc + 1);

        assert!(location::is_adjacent(
            location::get_loc(0, 0, x_size),
            location::get_loc(1, 0, x_size),
            x_size
        ));
        assert!(!location::is_adjacent(
            location::get_loc(0, 0, x_size),
            location::get_loc(1, 1, x_size),
            x_size
        ));

        assert_eq!(
            location::distance(
                location::get_loc(0, 0, x_size),
                location::get_loc(3, 4, x_size),
                x_size
            ),
            7
        );
    }

    #[test]
    fn test_location_strings() {
        let x_size = 19;
        let y_size = 19;
        // GTP row numbers count from the top of the board.
        assert_eq!(
            location::to_string(location::get_loc(3, 3, x_size), x_size, y_size),
            "D16"
        );
        assert_eq!(location::to_string(PASS_LOC, x_size, y_size), "pass");
        assert_eq!(location::to_string(NULL_LOC, x_size, y_size), "null");

        let loc = location::of_string("D4", x_size, y_size).unwrap();
        assert_eq!(location::get_x(loc, x_size), 3);
        assert_eq!(location::get_y(loc, x_size), y_size - 4);

        // 'I' is skipped in GTP, so 'J' is the 9th letter but column index 8.
        let j10 = location::of_string("J10", x_size, y_size).unwrap();
        assert_eq!(location::get_x(j10, x_size), 8);
        assert_eq!(location::get_y(j10, x_size), y_size - 10);
    }

    #[test]
    fn test_parse_sequence() {
        let seq = location::parse_sequence("D4, J10 pass", 19, 19).unwrap();
        assert_eq!(seq.len(), 3);
        assert_eq!(seq[2], PASS_LOC);
    }

    #[test]
    fn test_move() {
        let m = Move::new(PASS_LOC, P_BLACK);
        assert_eq!(m.loc, PASS_LOC);
        assert_eq!(m.pla, P_BLACK);
    }

    #[test]
    fn test_board_init() {
        let b = Board::new(19, 19);
        assert_eq!(b.x_size, 19);
        assert_eq!(b.y_size, 19);
        assert!(b.is_empty());
        assert_eq!(b.num_stones_on_board(), 0);
    }

    #[test]
    fn test_play_move_simple() {
        let mut b = Board::new(5, 5);
        assert!(b.play_move(location::get_loc(2, 2, 5), P_BLACK, true));
        assert_eq!(b.colors[location::get_loc(2, 2, 5) as usize], C_BLACK);
        assert_eq!(b.get_num_liberties(location::get_loc(2, 2, 5)), 4);
    }

    #[test]
    fn test_capture() {
        let mut b = Board::new(5, 5);
        let c3 = location::get_loc(2, 2, 5);
        let d3 = location::get_loc(3, 2, 5);
        let e3 = location::get_loc(4, 2, 5);
        let d2 = location::get_loc(3, 1, 5);
        let d4 = location::get_loc(3, 3, 5);

        // Surround d3 with black stones and capture it.
        b.play_move(d2, P_BLACK, true);
        b.play_move(d3, P_WHITE, true);
        b.play_move(c3, P_BLACK, true);
        b.play_move(d4, P_BLACK, true);
        b.play_move(e3, P_BLACK, true);

        assert_eq!(b.colors[d3 as usize], C_EMPTY);
        assert_eq!(b.num_black_captures, 0);
        assert_eq!(b.num_white_captures, 1);
    }

    #[test]
    fn test_ko() {
        let mut b = Board::new(5, 5);
        let c2 = location::get_loc(2, 1, 5);
        let d1 = location::get_loc(3, 0, 5);
        let e2 = location::get_loc(4, 1, 5);
        let d2 = location::get_loc(3, 1, 5);
        let c3 = location::get_loc(2, 2, 5);
        let e3 = location::get_loc(4, 2, 5);
        let d3 = location::get_loc(3, 2, 5);
        let d4 = location::get_loc(3, 3, 5);

        // Black stone at d2 with all liberties but d3 filled by white.
        b.play_move(d2, P_BLACK, true);
        // Surround the capturing point d3 with black so the white stone there
        // will be an isolated single stone with only d2 as a liberty.
        b.play_move(c3, P_BLACK, true);
        b.play_move(e3, P_BLACK, true);
        b.play_move(d4, P_BLACK, true);
        // Fill d2's other liberties with white (not adjacent to d3).
        b.play_move(c2, P_WHITE, true);
        b.play_move(e2, P_WHITE, true);
        b.play_move(d1, P_WHITE, true);

        // White captures the black stone at d2, creating a simple ko at d2.
        assert!(b.play_move(d3, P_WHITE, true));
        assert!(b.is_ko_banned(d2));
        assert!(!b.is_legal(d2, P_BLACK, true));
        assert!(b.is_legal_ignoring_ko(d2, P_BLACK, true));

        // A pass clears the ko ban.
        b.play_move(PASS_LOC, P_BLACK, true);
        assert!(!b.is_ko_banned(d2));
        assert!(b.is_legal(d2, P_BLACK, true));
    }

    #[test]
    fn test_suicide() {
        let mut b = Board::new(5, 5);
        let b3 = location::get_loc(1, 2, 5);
        let c3 = location::get_loc(2, 2, 5);
        let d3 = location::get_loc(3, 2, 5);
        let c2 = location::get_loc(2, 1, 5);
        let d2 = location::get_loc(3, 1, 5);
        let e3 = location::get_loc(4, 2, 5);
        let c4 = location::get_loc(2, 3, 5);
        let e4 = location::get_loc(4, 3, 5);
        let d5 = location::get_loc(3, 4, 5);
        let d4 = location::get_loc(3, 3, 5);

        // White group c3-d3 with exactly one liberty at d4, which is also
        // surrounded on its other sides by black.
        b.play_move(c3, P_WHITE, true);
        b.play_move(d3, P_WHITE, true);
        b.play_move(b3, P_BLACK, true);
        b.play_move(c2, P_BLACK, true);
        b.play_move(d2, P_BLACK, true);
        b.play_move(e3, P_BLACK, true);
        b.play_move(c4, P_BLACK, true);
        b.play_move(e4, P_BLACK, true);
        b.play_move(d5, P_BLACK, true);

        // Playing d4 connects to the white group but the combined group has no liberties.
        assert!(!b.is_legal(d4, P_WHITE, false));
        assert!(b.is_legal(d4, P_WHITE, true));
    }

    #[test]
    fn test_set_stone_fail_if_no_libs() {
        let mut b = Board::new(5, 5);
        let c3 = location::get_loc(2, 2, 5);
        assert!(b.set_stone_fail_if_no_libs(c3, P_BLACK));
        assert_eq!(b.colors[c3 as usize], C_BLACK);
        assert!(b.set_stone_fail_if_no_libs(c3, C_EMPTY));
        assert_eq!(b.colors[c3 as usize], C_EMPTY);
    }

    #[test]
    fn test_set_stones_tolerant_removes_no_liberty() {
        let mut b = Board::new(5, 5);
        let c3 = location::get_loc(2, 2, 5);
        let d3 = location::get_loc(3, 2, 5);
        let e3 = location::get_loc(4, 2, 5);
        let d2 = location::get_loc(3, 1, 5);
        let d4 = location::get_loc(3, 3, 5);

        // Create a white stone with zero liberties via tolerant setup.
        let placements = vec![
            Move::new(c3, P_BLACK),
            Move::new(e3, P_BLACK),
            Move::new(d2, P_BLACK),
            Move::new(d4, P_BLACK),
            Move::new(d3, P_WHITE),
        ];
        let removed = b.set_stones_tolerant(&placements);
        assert_eq!(removed, 1);
        assert_eq!(b.colors[d3 as usize], C_EMPTY);
        assert_eq!(b.colors[c3 as usize], C_BLACK);
    }

    #[test]
    fn test_parse_board_roundtrip() {
        let s = "X.O..\n.OX..\n.....\n.....\n.....\n";
        let b = Board::parse_board(5, 5, s, '\n').unwrap();
        let out = b.to_string_simple('\n');
        assert_eq!(out, s);
    }

    #[test]
    fn test_is_equal_for_testing() {
        let b1 = Board::parse_board(5, 5, "X.O..\n.....\n.....\n.....\n.....\n", '\n').unwrap();
        let b2 = Board::parse_board(5, 5, "X.O..\n.....\n.....\n.....\n.....\n", '\n').unwrap();
        assert!(b1.is_equal_for_testing(&b2, true, true));
    }

    #[test]
    fn test_calculate_area_empty_board() {
        let b = Board::new(5, 5);
        let mut area = [C_EMPTY; MAX_ARR_SIZE];
        b.calculate_area(&mut area, true, true, true, true);
        for y in 0..b.y_size {
            for x in 0..b.x_size {
                let loc = location::get_loc(x, y, b.x_size) as usize;
                assert_eq!(area[loc], C_EMPTY);
            }
        }
    }

    #[test]
    fn test_calculate_area_two_eyes_alive() {
        // A 3x3 black group with two eyes in the center of a 7x7 board.
        let s = ".......\n.......\n..XXX..\n..X.X..\n..XXX..\n.......\n.......\n";
        let b = Board::parse_board(7, 7, s, '\n').unwrap();
        let mut area = [C_EMPTY; MAX_ARR_SIZE];
        b.calculate_area(&mut area, true, true, true, true);

        // The black stones and the two eyes should be marked as black's area.
        for y in 2..5 {
            for x in 2..5 {
                let loc = location::get_loc(x, y, 7) as usize;
                assert_eq!(area[loc], C_BLACK, "loc ({},{}) should be black area", x, y);
            }
        }
    }

    #[test]
    fn test_calculate_area_dead_stone() {
        // White stone completely surrounded by black stones (zero-liberty setup).
        let mut b = Board::new(7, 7);
        let placements = vec![
            Move::new(location::get_loc(2, 2, 7), P_BLACK),
            Move::new(location::get_loc(3, 2, 7), P_BLACK),
            Move::new(location::get_loc(4, 2, 7), P_BLACK),
            Move::new(location::get_loc(2, 3, 7), P_BLACK),
            Move::new(location::get_loc(4, 3, 7), P_BLACK),
            Move::new(location::get_loc(2, 4, 7), P_BLACK),
            Move::new(location::get_loc(3, 4, 7), P_BLACK),
            Move::new(location::get_loc(4, 4, 7), P_BLACK),
            Move::new(location::get_loc(3, 3, 7), P_WHITE),
        ];
        b.set_stones_tolerant(&placements);
        let mut area = [C_EMPTY; MAX_ARR_SIZE];
        b.calculate_area(&mut area, true, true, true, true);

        // Everything inside the black wall is black area (the white stone is dead).
        for y in 2..5 {
            for x in 2..5 {
                let loc = location::get_loc(x, y, 7) as usize;
                assert_eq!(area[loc], C_BLACK, "loc ({},{}) should be black area", x, y);
            }
        }
    }

    #[test]
    fn test_calculate_independent_life_area_basic() {
        let s = ".......\n.......\n..XXX..\n..X.X..\n..XXX..\n.......\n.......\n";
        let b = Board::parse_board(7, 7, s, '\n').unwrap();
        let mut area = [C_EMPTY; MAX_ARR_SIZE];
        let count = b.calculate_independent_life_area(&mut area, true, true, true);
        assert_eq!(count, -1);
        for y in 2..5 {
            for x in 2..5 {
                let loc = location::get_loc(x, y, 7) as usize;
                assert_eq!(area[loc], C_BLACK);
            }
        }
    }

    #[test]
    fn test_ladder_corner_capture() {
        // White chain A2-B2 with two liberties along the bottom edge, surrounded
        // by black so that either liberty is a working ladder capture.
        let mut board = Board::new(5, 5);
        let placements = vec![
            Move::new(location::get_loc(0, 1, 5), P_WHITE),
            Move::new(location::get_loc(1, 1, 5), P_WHITE),
            Move::new(location::get_loc(0, 2, 5), P_BLACK),
            Move::new(location::get_loc(1, 2, 5), P_BLACK),
            Move::new(location::get_loc(2, 1, 5), P_BLACK),
            Move::new(location::get_loc(2, 0, 5), P_BLACK),
        ];
        assert!(board.set_stones_tolerant(&placements) == 0);
        let white_loc = location::get_loc(0, 1, 5);
        assert_eq!(board.get_num_liberties(white_loc), 2);

        let (captured, working_moves) =
            board.search_is_ladder_captured_attacker_first_2_libs(white_loc);
        assert!(captured);
        assert_eq!(working_moves.len(), 2);
    }

    #[test]
    fn test_ladder_escapes() {
        // A white stone with two non-adjacent liberties in the open center.
        // Each liberty has three immediate empty neighbors, so the attacker
        // cannot ladder-capture it.
        let mut board = Board::new(9, 9);
        let placements = vec![
            Move::new(location::get_loc(3, 3, 9), P_WHITE),
            Move::new(location::get_loc(2, 3, 9), P_BLACK),
            Move::new(location::get_loc(4, 3, 9), P_BLACK),
        ];
        assert!(board.set_stones_tolerant(&placements) == 0);
        let white_loc = location::get_loc(3, 3, 9);
        assert_eq!(board.get_num_liberties(white_loc), 2);

        let (captured, working_moves) =
            board.search_is_ladder_captured_attacker_first_2_libs(white_loc);
        assert!(!captured);
        assert!(working_moves.is_empty());
    }

    #[test]
    fn test_one_liberty_inescapable_atari() {
        // White stone B2 has a single liberty at B1 and no captures available.
        let mut board = Board::new(5, 5);
        let placements = vec![
            Move::new(location::get_loc(1, 1, 5), P_WHITE),
            Move::new(location::get_loc(0, 1, 5), P_BLACK),
            Move::new(location::get_loc(2, 1, 5), P_BLACK),
            Move::new(location::get_loc(1, 2, 5), P_BLACK),
        ];
        assert!(board.set_stones_tolerant(&placements) == 0);
        let white_loc = location::get_loc(1, 1, 5);
        assert_eq!(board.get_num_liberties(white_loc), 1);

        assert!(board.search_is_ladder_captured(white_loc, true));
    }
}

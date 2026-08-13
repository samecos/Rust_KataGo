//! SGF parsing and basic metadata extraction.
//!
//! Corresponds to `cpp/dataio/sgf.h` and `cpp/dataio/sgf.cpp`.
//! This slice implements the tree structure, property parsing, root-node metadata,
//! move/placement extraction along the longest branch, and `CompactSgf` construction.
//! Position sampling and full board-history setup are intentionally left as stubs
//! for later slices.

use kata_core::global::{self, IOError};
use kata_core::hash::Hash128;
use kata_game::board::{
    Board, C_BLACK, C_EMPTY, C_WHITE, Color, Loc, MAX_LEN, Move, NULL_LOC, P_BLACK, P_WHITE,
    PASS_LOC, Player, get_opp, location, player_io,
};
use kata_game::history::BoardHistory;
use kata_game::rules::Rules;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

/// Sentinel used inside `SgfNode::move_no_size` to mean "pass".
const COORD_MAX: u8 = 128;

const SGF_HASH_SALT: u64 = 0x9e3779b97f4a7c15;

/// Board size returned by `Sgf::get_xy_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XYSize {
    pub x: i32,
    pub y: i32,
}

impl XYSize {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

impl fmt::Display for XYSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.x == self.y {
            write!(f, "{}", self.x)
        } else {
            write!(f, "{}:{}", self.x, self.y)
        }
    }
}

/// Internal move representation before the board size is known.
///
/// A coordinate of `(COORD_MAX, COORD_MAX)` encodes a pass; `(19, 19)` is the
/// legacy `tt` pass for boards of size at most 19.
#[derive(Debug, Clone, Copy, Default)]
struct SgfMove {
    x: u8,
    y: u8,
    pla: Player,
}

impl SgfMove {
    fn is_pass(self, x_size: i32, y_size: i32) -> bool {
        (self.x == COORD_MAX && self.y == COORD_MAX)
            || (self.x == 19 && self.y == 19 && (x_size <= 19 || y_size <= 19))
    }

    fn to_move(self, x_size: i32, y_size: i32) -> Result<Move, IOError> {
        if self.is_pass(x_size, y_size) {
            return Ok(Move::new(PASS_LOC, self.pla));
        }
        if self.x as i32 >= x_size || self.y as i32 >= y_size {
            return Err(IOError(format!(
                "Move out of bounds: {},{}",
                self.x, self.y
            )));
        }
        let loc = location::get_loc(self.x as i32, self.y as i32, x_size);
        Ok(Move::new(loc, self.pla))
    }
}

fn sgf_fail(message: impl Into<String>, pos: usize) -> IOError {
    IOError(format!("{} (pos {})", message.into(), pos))
}

fn prop_fail(message: impl Into<String>) -> IOError {
    IOError(message.into())
}

fn is_alpha_byte(b: u8) -> bool {
    b.is_ascii_alphabetic()
}

fn is_whitespace_byte(b: u8) -> bool {
    global::is_whitespace_char(b as char)
}

fn skip_bom_and_ws(bytes: &[u8], pos: &mut usize) {
    if *pos == 0 && bytes.len() >= 3 && bytes[0..3] == [0xEF, 0xBB, 0xBF] {
        *pos += 3;
    }
    while *pos < bytes.len() && is_whitespace_byte(bytes[*pos]) {
        *pos += 1;
    }
}

fn peek_sgf_char(bytes: &[u8], pos: &mut usize) -> Result<u8, IOError> {
    skip_bom_and_ws(bytes, pos);
    if *pos >= bytes.len() {
        return Err(sgf_fail("Unexpected end of SGF", *pos));
    }
    let c = bytes[*pos];
    *pos += 1;
    Ok(c)
}

fn parse_text_value(bytes: &[u8], pos: &mut usize) -> Result<String, IOError> {
    let mut acc: Vec<u8> = Vec::new();
    let mut escaping = false;
    loop {
        if *pos >= bytes.len() {
            return Err(sgf_fail(
                "Unexpected end of SGF while reading property value",
                *pos,
            ));
        }
        let c = bytes[*pos];

        if !escaping && c == b']' {
            break;
        }
        *pos += 1;

        if !escaping && c == b'\\' {
            escaping = true;
            continue;
        }

        if c == b'\n' || c == b'\r' {
            while *pos < bytes.len() && (bytes[*pos] == b'\n' || bytes[*pos] == b'\r') {
                *pos += 1;
            }
            if !escaping {
                acc.push(b'\n');
            }
            escaping = false;
            continue;
        }

        if c == b'\t' || c == b'\x0B' || c == b'\x0C' {
            escaping = false;
            acc.push(b' ');
            continue;
        }

        escaping = false;
        acc.push(c);
    }
    Ok(String::from_utf8_lossy(&acc).into_owned())
}

fn parse_sgf_coord(c: u8) -> Option<i32> {
    if c.is_ascii_lowercase() {
        Some(i32::from(c - b'a'))
    } else if c.is_ascii_uppercase() {
        Some(i32::from(c - b'A') + 26)
    } else {
        None
    }
}

fn parse_sgf_move_or_pass_no_size(s: &str, pla: Player) -> Result<SgfMove, IOError> {
    if s.is_empty() {
        return Ok(SgfMove {
            x: COORD_MAX,
            y: COORD_MAX,
            pla,
        });
    }
    if s.len() != 2 {
        return Err(prop_fail(format!("Invalid location: {}", s)));
    }
    let bytes = s.as_bytes();
    let x =
        parse_sgf_coord(bytes[0]).ok_or_else(|| prop_fail(format!("Invalid location: {}", s)))?;
    let y =
        parse_sgf_coord(bytes[1]).ok_or_else(|| prop_fail(format!("Invalid location: {}", s)))?;
    if x < 0 || y < 0 || x >= i32::from(COORD_MAX) || y >= i32::from(COORD_MAX) {
        return Err(prop_fail(format!("Invalid location: {}", s)));
    }
    Ok(SgfMove {
        x: x as u8,
        y: y as u8,
        pla,
    })
}

fn parse_sgf_loc(s: &str, x_size: i32, y_size: i32) -> Result<Loc, IOError> {
    if s.len() != 2 {
        return Err(prop_fail(format!("Invalid location: {}", s)));
    }
    let bytes = s.as_bytes();
    let x =
        parse_sgf_coord(bytes[0]).ok_or_else(|| prop_fail(format!("Invalid location: {}", s)))?;
    let y =
        parse_sgf_coord(bytes[1]).ok_or_else(|| prop_fail(format!("Invalid location: {}", s)))?;
    if x < 0 || x >= x_size || y < 0 || y >= y_size {
        return Err(prop_fail(format!("Invalid location: {}", s)));
    }
    Ok(location::get_loc(x, y, x_size))
}

fn parse_sgf_loc_rectangle(
    s: &str,
    x_size: i32,
    y_size: i32,
) -> Result<(i32, i32, i32, i32), IOError> {
    let (x1, y1, x2, y2) = if s.contains(':') {
        if s.len() != 5 {
            return Err(prop_fail(format!("Invalid location rect: {}", s)));
        }
        let bytes = s.as_bytes();
        if bytes[2] != b':' {
            return Err(prop_fail(format!("Invalid location rect: {}", s)));
        }
        let x1 = parse_sgf_coord(bytes[0])
            .ok_or_else(|| prop_fail(format!("Invalid location rect: {}", s)))?;
        let y1 = parse_sgf_coord(bytes[1])
            .ok_or_else(|| prop_fail(format!("Invalid location rect: {}", s)))?;
        let x2 = parse_sgf_coord(bytes[3])
            .ok_or_else(|| prop_fail(format!("Invalid location rect: {}", s)))?;
        let y2 = parse_sgf_coord(bytes[4])
            .ok_or_else(|| prop_fail(format!("Invalid location rect: {}", s)))?;
        (x1, y1, x2, y2)
    } else {
        let loc = parse_sgf_loc(s, x_size, y_size)?;
        let x = location::get_x(loc, x_size);
        let y = location::get_y(loc, x_size);
        (x, y, x, y)
    };
    if x1 < 0
        || x1 >= x_size
        || y1 < 0
        || y1 >= y_size
        || x2 < 0
        || x2 >= x_size
        || y2 < 0
        || y2 >= y_size
        || x1 > x2
        || y1 > y2
    {
        return Err(prop_fail(format!(
            "Invalid location or location rect: {}",
            s
        )));
    }
    Ok((x1, y1, x2, y2))
}

/// Parse an SGF coordinate string into a `Loc`, treating an empty value or the
/// legacy `tt` value as a pass.
pub fn parse_sgf_loc_or_pass(s: &str, x_size: i32, y_size: i32) -> Result<Loc, IOError> {
    if s.is_empty() || (s == "tt" && (x_size <= 19 || y_size <= 19)) {
        return Ok(PASS_LOC);
    }
    parse_sgf_loc(s, x_size, y_size)
}

/// Write a `Loc` to SGF coordinate notation. Passes and `NULL_LOC` become the
/// empty string.
pub fn write_sgf_loc(loc: Loc, x_size: i32, y_size: i32) -> Result<String, IOError> {
    if x_size >= 53 || y_size >= 53 {
        return Err(prop_fail(
            "Writing coordinates for SGF files for board sizes >= 53 is not implemented",
        ));
    }
    if loc == PASS_LOC || loc == NULL_LOC {
        return Ok(String::new());
    }
    let x = location::get_x(loc, x_size);
    let y = location::get_y(loc, x_size);
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    if x < 0 || x >= CHARS.len() as i32 || y < 0 || y >= CHARS.len() as i32 {
        return Err(prop_fail(format!(
            "Location out of SGF coord range: {}",
            loc
        )));
    }
    let mut out = String::with_capacity(2);
    out.push(CHARS[x as usize] as char);
    out.push(CHARS[y as usize] as char);
    Ok(out)
}

fn maybe_parse_property(
    node: &mut SgfNode,
    bytes: &[u8],
    pos: &mut usize,
) -> Result<bool, IOError> {
    let mut key = String::new();
    loop {
        let saved = *pos;
        let c = peek_sgf_char(bytes, pos)?;
        if is_alpha_byte(c) {
            key.push(c as char);
        } else {
            *pos = saved;
            break;
        }
    }
    if key.is_empty() {
        return Ok(false);
    }

    let mut parsed_at_least_one = false;
    loop {
        let saved = *pos;
        let c = peek_sgf_char(bytes, pos)?;
        if c != b'[' {
            *pos = saved;
            break;
        }

        let value = parse_text_value(bytes, pos)?;
        if node.move_no_size.pla == C_EMPTY && key == "B" {
            node.move_no_size = parse_sgf_move_or_pass_no_size(&value, P_BLACK)?;
        } else if node.move_no_size.pla == C_EMPTY && key == "W" {
            node.move_no_size = parse_sgf_move_or_pass_no_size(&value, P_WHITE)?;
        } else {
            node.add_property(&key, value);
        }

        let c2 = peek_sgf_char(bytes, pos)?;
        if c2 != b']' {
            return Err(sgf_fail("Expected closing bracket", *pos));
        }
        parsed_at_least_one = true;
    }

    if !parsed_at_least_one {
        return Err(sgf_fail(
            format!("No property values for property {}", key),
            *pos,
        ));
    }
    Ok(true)
}

fn maybe_parse_node(bytes: &[u8], pos: &mut usize) -> Result<Option<SgfNode>, IOError> {
    let saved = *pos;
    let c = peek_sgf_char(bytes, pos)?;
    if c != b';' {
        *pos = saved;
        return Ok(None);
    }
    let mut node = SgfNode::default();
    loop {
        if !maybe_parse_property(&mut node, bytes, pos)? {
            break;
        }
    }
    Ok(Some(node))
}

fn maybe_parse_sgf(bytes: &[u8], pos: &mut usize) -> Result<Option<Sgf>, IOError> {
    if *pos >= bytes.len() {
        return Ok(None);
    }
    let saved = *pos;
    let c = peek_sgf_char(bytes, pos)?;
    if c != b'(' {
        *pos = saved;
        return Ok(None);
    }

    let mut stack: Vec<Sgf> = vec![Sgf::default()];
    let mut entry_pos_stack: Vec<usize> = vec![*pos];
    let mut returned_child: Option<Sgf> = None;

    loop {
        let sgf = stack.last_mut().expect("sgf stack is non-empty");
        if returned_child.is_none() {
            while let Some(node) = maybe_parse_node(bytes, pos)? {
                sgf.nodes.push(node);
            }
        } else {
            sgf.children
                .push(returned_child.take().expect("just checked"));
        }

        let c2 = peek_sgf_char(bytes, pos)?;
        if c2 == b'(' {
            stack.push(Sgf::default());
            entry_pos_stack.push(*pos);
        } else if c2 == b')' {
            returned_child = stack.pop();
            entry_pos_stack.pop();
            if stack.is_empty() {
                break;
            }
        } else {
            return Err(sgf_fail(
                "Expected closing paren for sgf tree",
                entry_pos_stack.last().copied().unwrap_or(*pos),
            ));
        }
    }

    Ok(returned_child)
}

fn compute_sgf_hash(s: &str) -> Hash128 {
    let bytes = s.as_bytes();
    let mut h0: u64 = 0xcbf29ce484222325;
    let mut h1: u64 = 0x84222225cbf29ce4;
    for &b in bytes {
        h0 ^= u64::from(b);
        h0 = h0.wrapping_mul(0x100000001b3);
        h1 ^= u64::from(b);
        h1 = h1.wrapping_mul(0x100000001b3);
    }
    h1 ^= h0 ^ SGF_HASH_SALT;
    Hash128::new(h0, h1)
}

/// A single SGF node.
#[derive(Debug, Clone, Default)]
pub struct SgfNode {
    props: BTreeMap<String, Vec<String>>,
    move_no_size: SgfMove,
}

impl SgfNode {
    pub fn has_property(&self, key: &str) -> bool {
        self.props.contains_key(key)
    }

    pub fn get_single_property(&self, key: &str) -> Result<String, IOError> {
        let vals = self
            .props
            .get(key)
            .ok_or_else(|| prop_fail(format!("SGF does not contain property: {}", key)))?;
        if vals.len() != 1 {
            return Err(prop_fail(format!(
                "SGF property is not a singleton: {}",
                key
            )));
        }
        Ok(vals[0].clone())
    }

    pub fn get_properties(&self, key: &str) -> Result<Vec<String>, IOError> {
        self.props
            .get(key)
            .cloned()
            .ok_or_else(|| prop_fail(format!("SGF does not contain property: {}", key)))
    }

    pub fn add_property(&mut self, key: &str, value: String) {
        self.props.entry(key.to_string()).or_default().push(value);
    }

    pub fn append_comment(&mut self, value: &str) {
        let comments = self.props.entry("C".to_string()).or_default();
        if comments.is_empty() {
            comments.push(value.to_string());
        } else {
            let last = comments.len() - 1;
            comments[last].push_str(value);
        }
    }

    pub fn has_placements(&self) -> bool {
        self.has_property("AB") || self.has_property("AW") || self.has_property("AE")
    }

    pub fn accum_placements(
        &self,
        moves: &mut Vec<Move>,
        x_size: i32,
        y_size: i32,
    ) -> Result<(), IOError> {
        let handle_rect = |props: &BTreeMap<String, Vec<String>>,
                           key: &str,
                           color: Color,
                           moves: &mut Vec<Move>|
         -> Result<(), IOError> {
            if let Some(values) = props.get(key) {
                for value in values {
                    let (x1, y1, x2, y2) = parse_sgf_loc_rectangle(value, x_size, y_size)?;
                    for x in x1..=x2 {
                        for y in y1..=y2 {
                            let loc = location::get_loc(x, y, x_size);
                            moves.push(Move::new(loc, color));
                        }
                    }
                }
            }
            Ok(())
        };

        handle_rect(&self.props, "AB", P_BLACK, moves)?;
        handle_rect(&self.props, "AW", P_WHITE, moves)?;
        handle_rect(&self.props, "AE", C_EMPTY, moves)?;
        Ok(())
    }

    pub fn accum_moves(
        &self,
        moves: &mut Vec<Move>,
        x_size: i32,
        y_size: i32,
    ) -> Result<(), IOError> {
        if self.move_no_size.pla == P_BLACK {
            moves.push(self.move_no_size.to_move(x_size, y_size)?);
        }
        if let Some(values) = self.props.get("B") {
            for value in values {
                let loc = parse_sgf_loc_or_pass(value, x_size, y_size)?;
                moves.push(Move::new(loc, P_BLACK));
            }
        }
        if self.move_no_size.pla == P_WHITE {
            moves.push(self.move_no_size.to_move(x_size, y_size)?);
        }
        if let Some(values) = self.props.get("W") {
            for value in values {
                let loc = parse_sgf_loc_or_pass(value, x_size, y_size)?;
                moves.push(Move::new(loc, P_WHITE));
            }
        }
        Ok(())
    }

    pub fn get_pl_specified_color(&self) -> Color {
        if !self.has_property("PL") {
            return C_EMPTY;
        }
        match self.get_single_property("PL") {
            Ok(s) => match global::to_lower(&s).as_str() {
                "b" | "black" => C_BLACK,
                "w" | "white" => C_WHITE,
                _ => C_EMPTY,
            },
            Err(_) => C_EMPTY,
        }
    }

    pub fn get_rules_from_ru_tag_or_fail(&self) -> Result<Rules, IOError> {
        if !self.has_property("RU") {
            return Err(prop_fail("SGF file does not specify rules"));
        }
        let s = self.get_single_property("RU")?;
        Rules::try_parse_rules(&s)
            .ok_or_else(|| prop_fail(format!("Could not parse rules in sgf: {}", s)))
    }

    pub fn get_sgf_winner(&self) -> Player {
        if !self.has_property("RE") {
            return C_EMPTY;
        }
        match self.get_single_property("RE") {
            Ok(s) => {
                let lower = global::to_lower(&s);
                if global::is_prefix(&lower, "b+") || global::is_prefix(&lower, "black+") {
                    P_BLACK
                } else if global::is_prefix(&lower, "w+") || global::is_prefix(&lower, "white+") {
                    P_WHITE
                } else {
                    C_EMPTY
                }
            }
            Err(_) => C_EMPTY,
        }
    }

    pub fn get_komi_or_fail(&self) -> Result<f32, IOError> {
        if !self.has_property("KM") {
            return Err(prop_fail("Sgf does not specify komi"));
        }
        self.get_komi_or_default(0.0)
    }

    pub fn get_komi_or_default(&self, default_komi: f32) -> Result<f32, IOError> {
        if !self.has_property("KM") {
            return Ok(default_komi);
        }
        let mut komi = global::try_string_to_float(&self.get_single_property("KM")?)
            .ok_or_else(|| prop_fail("Could not parse komi in sgf"))?;

        if !Rules::komi_is_int_or_half_int(komi) {
            if Rules::komi_is_int_or_half_int(komi * 2.0)
                && self.has_property("US")
                && self.has_property("RU")
            {
                let us = global::to_lower(&self.get_single_property("US")?);
                let ru = global::to_lower(&self.get_single_property("RU")?);
                if global::is_prefix(&us, "gogod") && (ru == "chinese" || ru == "chinese, pair go")
                {
                    komi *= 2.0;
                } else {
                    return Err(prop_fail("Komi in sgf is not integer or half-integer"));
                }
            } else {
                return Err(prop_fail("Komi in sgf is not integer or half-integer"));
            }
        }

        if self.has_property("AP") {
            let ap = self.get_properties("AP")?;
            if ap.iter().any(|s| s == "foxwq" || s == "SGFC:2.0") {
                komi = match komi {
                    550.0 | 275.0 => 5.5,
                    325.0 | 650.0 => 6.5,
                    375.0 | 750.0 => 7.5,
                    350.0 | 700.0 => 7.0,
                    0.0 => 0.0,
                    6.5 | 7.5 | 7.0 => komi,
                    _ => {
                        return Err(prop_fail(format!(
                            "Currently no case implemented for foxwq or SGFC komi: {}",
                            komi
                        )));
                    }
                };
            }
        }

        Ok(komi)
    }

    pub fn get_player_name(&self, pla: Player) -> String {
        let key = if pla == P_BLACK { "PB" } else { "PW" };
        self.get_single_property(key).unwrap_or_default()
    }
}

/// An SGF tree.
#[derive(Debug, Clone, Default)]
pub struct Sgf {
    pub file_name: String,
    pub nodes: Vec<SgfNode>,
    pub children: Vec<Sgf>,
    pub hash: Hash128,
}

impl Sgf {
    pub const RANK_UNKNOWN: i32 = -100_000;

    pub fn parse(s: &str) -> Result<Self, IOError> {
        let bytes = s.as_bytes();
        let mut pos = 0usize;
        let sgf = maybe_parse_sgf(bytes, &mut pos)?;
        let mut sgf = sgf.ok_or_else(|| {
            sgf_fail(
                "Empty or invalid sgf (is the opening parenthesis missing?)",
                0,
            )
        })?;
        if sgf.nodes.is_empty() {
            return Err(sgf_fail("Empty or invalid sgf", 0));
        }
        sgf.hash = compute_sgf_hash(s);
        Ok(sgf)
    }

    pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Self, IOError> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)
            .map_err(|e| IOError(format!("Could not read SGF file {}: {}", path.display(), e)))?;
        let mut sgf = Self::parse(&contents)?;
        sgf.file_name = path.to_string_lossy().into_owned();
        Ok(sgf)
    }

    fn check_non_empty(&self) -> Result<(), IOError> {
        if self.nodes.is_empty() {
            Err(prop_fail("Empty sgf"))
        } else {
            Ok(())
        }
    }

    pub fn get_xy_size(&self) -> Result<XYSize, IOError> {
        self.check_non_empty()?;
        if !self.nodes[0].has_property("SZ") {
            return Ok(XYSize::new(19, 19));
        }
        let s = self.nodes[0].get_single_property("SZ")?;
        if s.contains(':') {
            let pieces: Vec<&str> = s.split(':').collect();
            if pieces.len() != 2 {
                return Err(prop_fail(format!(
                    "Could not parse board size in sgf: {}",
                    s
                )));
            }
            let x_size = global::try_string_to_int(pieces[0])
                .ok_or_else(|| prop_fail(format!("Could not parse board size in sgf: {}", s)))?;
            let y_size = global::try_string_to_int(pieces[1])
                .ok_or_else(|| prop_fail(format!("Could not parse board size in sgf: {}", s)))?;
            if x_size <= 1 || y_size <= 1 {
                return Err(prop_fail(format!("Board size in sgf is <= 1: {}", s)));
            }
            if x_size > MAX_LEN as i32 || y_size > MAX_LEN as i32 {
                return Err(prop_fail(format!(
                    "Board size in sgf is > Board::MAX_LEN = {}",
                    MAX_LEN
                )));
            }
            Ok(XYSize::new(x_size, y_size))
        } else {
            let size = global::try_string_to_int(&s)
                .ok_or_else(|| prop_fail(format!("Could not parse board size in sgf: {}", s)))?;
            if size <= 1 {
                return Err(prop_fail(format!("Board size in sgf is <= 1: {}", s)));
            }
            if size > MAX_LEN as i32 {
                return Err(prop_fail(format!(
                    "Board size in sgf is > Board::MAX_LEN = {}",
                    MAX_LEN
                )));
            }
            Ok(XYSize::new(size, size))
        }
    }

    pub fn get_komi_or_fail(&self) -> Result<f32, IOError> {
        self.check_non_empty()?;
        self.nodes[0].get_komi_or_fail()
    }

    pub fn get_komi_or_default(&self, default_komi: f32) -> Result<f32, IOError> {
        self.check_non_empty()?;
        self.nodes[0].get_komi_or_default(default_komi)
    }

    pub fn has_rules(&self) -> bool {
        !self.nodes.is_empty() && self.nodes[0].has_property("RU")
    }

    pub fn get_rules_or_fail(&self) -> Result<Rules, IOError> {
        self.check_non_empty()?;
        let mut rules = self.nodes[0].get_rules_from_ru_tag_or_fail()?;
        rules.set_komi(self.nodes[0].get_komi_or_fail()?);
        Ok(rules)
    }

    pub fn get_handicap_value(&self) -> Result<i32, IOError> {
        self.check_non_empty()?;
        if !self.nodes[0].has_property("HA") {
            return Ok(0);
        }
        global::try_string_to_int(&self.nodes[0].get_single_property("HA")?)
            .ok_or_else(|| prop_fail("Could not parse handicap value in sgf"))
    }

    pub fn get_sgf_winner(&self) -> Player {
        if self.nodes.is_empty() {
            return C_EMPTY;
        }
        self.nodes[0].get_sgf_winner()
    }

    pub fn get_first_player_color(&self) -> Result<Player, IOError> {
        self.check_non_empty()?;
        let pl_color = self.nodes[0].get_pl_specified_color();
        if pl_color == P_BLACK || pl_color == P_WHITE {
            return Ok(pl_color);
        }
        let size = self.get_xy_size()?;
        let mut moves = Vec::new();
        self.get_moves(&mut moves, size.x, size.y)?;
        if let Some(first) = moves.first() {
            Ok(first.pla)
        } else {
            Ok(P_BLACK)
        }
    }

    pub fn get_rank(&self, pla: Player) -> Result<i32, IOError> {
        self.check_non_empty()?;
        let key = if pla == P_BLACK { "BR" } else { "WR" };
        if !self.nodes[0].has_property(key) {
            return Ok(Self::RANK_UNKNOWN);
        }
        let rank_str = self.nodes[0].get_single_property(key)?;
        parse_rank(&rank_str)
    }

    pub fn get_rating(&self, pla: Player) -> Result<i32, IOError> {
        self.check_non_empty()?;
        let key = if pla == P_BLACK { "BR" } else { "WR" };
        if !self.nodes[0].has_property(key) {
            return Err(prop_fail("Could not parse rating in sgf"));
        }
        let rank_str = self.nodes[0].get_single_property(key)?;
        global::try_string_to_int(&rank_str)
            .ok_or_else(|| prop_fail(format!("Could not parse rating in sgf: {}", rank_str)))
    }

    pub fn get_player_name(&self, pla: Player) -> String {
        if self.nodes.is_empty() {
            return String::new();
        }
        self.nodes[0].get_player_name(pla)
    }

    pub fn has_root_property(&self, property: &str) -> bool {
        !self.nodes.is_empty() && self.nodes[0].has_property(property)
    }

    pub fn get_root_property_with_default(&self, property: &str, default_ret: &str) -> String {
        if self.nodes.is_empty() {
            return default_ret.to_string();
        }
        self.nodes[0]
            .get_single_property(property)
            .unwrap_or_else(|_| default_ret.to_string())
    }

    pub fn get_root_properties(&self, property: &str) -> Vec<String> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        self.nodes[0].get_properties(property).unwrap_or_default()
    }

    pub fn add_root_property(&mut self, key: &str, value: &str) {
        if !self.nodes.is_empty() {
            self.nodes[0].add_property(key, value.to_string());
        }
    }

    pub fn get_placements(
        &self,
        moves: &mut Vec<Move>,
        x_size: i32,
        y_size: i32,
    ) -> Result<(), IOError> {
        moves.clear();
        self.check_non_empty()?;
        self.nodes[0].accum_placements(moves, x_size, y_size)
    }

    pub fn get_moves(
        &self,
        moves: &mut Vec<Move>,
        x_size: i32,
        y_size: i32,
    ) -> Result<(), IOError> {
        moves.clear();
        self.get_moves_helper(moves, x_size, y_size)
    }

    fn get_moves_helper(
        &self,
        moves: &mut Vec<Move>,
        x_size: i32,
        y_size: i32,
    ) -> Result<(), IOError> {
        let mut sgf = self;
        loop {
            sgf.check_non_empty()?;
            for (i, node) in sgf.nodes.iter().enumerate() {
                if i > 0 && node.has_placements() {
                    return Err(prop_fail(
                        "Found stone placements after the root, game records that are not simply ordinary play not currently supported",
                    ));
                }
                node.accum_moves(moves, x_size, y_size)?;
            }

            match sgf.children.len() {
                0 => return Ok(()),
                1 => {
                    sgf = &sgf.children[0];
                    continue;
                }
                _ => {
                    let mut max_child_depth = sgf.children[0].depth();
                    let mut max_index = 0;
                    for (i, child) in sgf.children.iter().enumerate().skip(1) {
                        let d = child.depth();
                        if d > max_child_depth {
                            max_child_depth = d;
                            max_index = i;
                        }
                    }
                    sgf = &sgf.children[max_index];
                }
            }
        }
    }

    pub fn depth(&self) -> i64 {
        self.traverse(
            0i64,
            |max_child_depth, child_value| max_child_depth.max(child_value),
            |sgf, max_child_depth| max_child_depth + sgf.nodes.len() as i64,
        )
    }

    pub fn node_count(&self) -> i64 {
        self.traverse(
            0i64,
            |count, child_value| count + child_value,
            |sgf, count| count + sgf.nodes.len() as i64,
        )
    }

    pub fn branch_count(&self) -> i64 {
        1 + self.traverse(
            0i64,
            |count, child_value| count + child_value,
            |sgf, count| count + (sgf.children.len() as i64 - 1).max(0),
        )
    }

    fn traverse<T: Clone>(
        &self,
        initial_value: T,
        reduce: impl Fn(T, T) -> T,
        transform: impl Fn(&Self, T) -> T,
    ) -> T {
        let mut stack: Vec<&Sgf> = vec![self];
        let mut next_child_idx: Vec<usize> = vec![0];
        let mut value_stack: Vec<T> = vec![initial_value.clone()];

        loop {
            let sgf = *stack.last().expect("stack non-empty");
            let next_idx = *next_child_idx.last().expect("idx stack non-empty");

            if next_idx >= sgf.children.len() {
                let value = transform(sgf, value_stack.pop().expect("value stack non-empty"));
                stack.pop();
                next_child_idx.pop();
                if stack.is_empty() {
                    return value;
                }
                let parent_value = value_stack.last_mut().expect("value stack non-empty");
                *parent_value = reduce(parent_value.clone(), value);
            } else {
                if let Some(i) = next_child_idx.last_mut() {
                    *i += 1;
                }
                stack.push(&sgf.children[next_idx]);
                next_child_idx.push(0);
                value_stack.push(initial_value.clone());
            }
        }
    }

    /// Load all unique positions from this SGF into `samples`.
    #[allow(clippy::too_many_arguments)]
    pub fn load_all_unique_positions(
        &self,
        unique_hashes: &mut BTreeSet<Hash128>,
        hash_comments: bool,
        hash_parent: bool,
        flip_if_pass_or_w_first: bool,
        allow_game_over: bool,
        rng: Option<&mut dyn rand::RngCore>,
        tolerate_illegal_moves: bool,
    ) -> Result<Vec<PositionSample>, IOError> {
        let mut samples = Vec::new();
        self.iter_all_unique_positions(
            unique_hashes,
            hash_comments,
            hash_parent,
            flip_if_pass_or_w_first,
            allow_game_over,
            rng,
            &mut |sample, _hist, _comments| {
                samples.push(sample.clone());
            },
            tolerate_illegal_moves,
        )?;
        Ok(samples)
    }

    /// Iterate all unique positions in this SGF, calling `f` for each.
    #[allow(clippy::too_many_arguments)]
    pub fn iter_all_unique_positions<F>(
        &self,
        unique_hashes: &mut BTreeSet<Hash128>,
        hash_comments: bool,
        hash_parent: bool,
        flip_if_pass_or_w_first: bool,
        allow_game_over: bool,
        rng: Option<&mut dyn rand::RngCore>,
        f: &mut F,
        tolerate_illegal_moves: bool,
    ) -> Result<(), IOError>
    where
        F: FnMut(&PositionSample, &BoardHistory, &str),
    {
        let size = self.get_xy_size()?;
        let mut board = Board::new(size.x, size.y);
        let mut next_pla = if !self.nodes.is_empty() {
            self.nodes[0].get_pl_specified_color()
        } else {
            C_EMPTY
        };
        if next_pla != P_BLACK && next_pla != P_WHITE {
            next_pla = P_BLACK;
        }
        let mut rules = Rules::get_tromp_taylorish();
        rules.ko_rule = kata_game::rules::KoRule::Situational;
        rules.multi_stone_suicide_legal = true;
        let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

        let mut sample_buf = PositionSample::default();
        let mut variation_trace_nodes_branch: Vec<(i64, i64)> = Vec::new();
        let is_root = true;
        let require_unique = true;
        self.iter_all_positions_helper(
            &mut board,
            &mut hist,
            next_pla,
            rules,
            size.x,
            size.y,
            &mut sample_buf,
            unique_hashes,
            require_unique,
            hash_comments,
            hash_parent,
            flip_if_pass_or_w_first,
            allow_game_over,
            tolerate_illegal_moves,
            is_root,
            rng,
            &mut variation_trace_nodes_branch,
            f,
        )?;
        Ok(())
    }

    /// Iterate all positions in this SGF (not just unique ones).
    pub fn iter_all_positions<F>(
        &self,
        flip_if_pass_or_w_first: bool,
        allow_game_over: bool,
        rng: Option<&mut dyn rand::RngCore>,
        f: &mut F,
        tolerate_illegal_moves: bool,
    ) -> Result<(), IOError>
    where
        F: FnMut(&PositionSample, &BoardHistory, &str),
    {
        let size = self.get_xy_size()?;
        let mut board = Board::new(size.x, size.y);
        let mut next_pla = if !self.nodes.is_empty() {
            self.nodes[0].get_pl_specified_color()
        } else {
            C_EMPTY
        };
        if next_pla != P_BLACK && next_pla != P_WHITE {
            next_pla = P_BLACK;
        }
        let mut rules = Rules::get_tromp_taylorish();
        rules.ko_rule = kata_game::rules::KoRule::Situational;
        rules.multi_stone_suicide_legal = true;
        let mut hist = BoardHistory::new(board.clone(), next_pla, rules, 0);

        let mut sample_buf = PositionSample::default();
        let mut variation_trace_nodes_branch: Vec<(i64, i64)> = Vec::new();
        let mut unique_hashes = BTreeSet::new();
        let is_root = true;
        let require_unique = false;
        let hash_comments = false;
        let hash_parent = false;
        self.iter_all_positions_helper(
            &mut board,
            &mut hist,
            next_pla,
            rules,
            size.x,
            size.y,
            &mut sample_buf,
            &mut unique_hashes,
            require_unique,
            hash_comments,
            hash_parent,
            flip_if_pass_or_w_first,
            allow_game_over,
            tolerate_illegal_moves,
            is_root,
            rng,
            &mut variation_trace_nodes_branch,
            f,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn iter_all_positions_helper<F>(
        &self,
        board: &mut Board,
        hist: &mut BoardHistory,
        mut next_pla: Player,
        rules: Rules,
        x_size: i32,
        y_size: i32,
        sample_buf: &mut PositionSample,
        unique_hashes: &mut BTreeSet<Hash128>,
        require_unique: bool,
        hash_comments: bool,
        hash_parent: bool,
        flip_if_pass_or_w_first: bool,
        allow_game_over: bool,
        tolerate_illegal_moves: bool,
        is_root: bool,
        rng: Option<&mut dyn rand::RngCore>,
        variation_trace_nodes_branch: &mut Vec<(i64, i64)>,
        f: &mut F,
    ) -> Result<(), IOError>
    where
        F: FnMut(&PositionSample, &BoardHistory, &str),
    {
        let build_trace = |node_idx: usize| -> String {
            let mut trace = String::new();
            for (nodes, branch) in variation_trace_nodes_branch.iter() {
                trace.push_str(&format!("forward {} branch {} ", nodes, branch));
            }
            trace.push_str(&format!("forward {}", node_idx));
            trace
        };

        let interrupt_history_like_setup =
            |b: &mut Board, h: &mut BoardHistory, next_pla_for_setup: Player| {
                b.clear_simple_ko_loc();
                let mut initial_turn_number = h.initial_turn_number + h.move_history.len() as i64;
                if i64::from(b.num_stones_on_board()) > initial_turn_number {
                    initial_turn_number = i64::from(b.num_stones_on_board());
                }
                h.clear(b.clone(), next_pla_for_setup, rules, 0);
                h.set_initial_turn_number(initial_turn_number);
            };

        let mut buf = Vec::new();
        for i in 0..self.nodes.len() {
            let comments = if self.nodes[i].has_property("C") {
                self.nodes[i].get_single_property("C").unwrap_or_default()
            } else {
                String::new()
            };

            if is_root && i == 0 && !self.nodes[i].has_placements() {
                self.sample_position_helper(
                    board,
                    hist,
                    next_pla,
                    sample_buf,
                    unique_hashes,
                    require_unique,
                    hash_comments,
                    hash_parent,
                    flip_if_pass_or_w_first,
                    allow_game_over,
                    &comments,
                    f,
                );
            }

            let pl_color = self.nodes[i].get_pl_specified_color();

            // Handle placements and player changes as setup nodes.
            if self.nodes[i].has_placements() || (pl_color != C_EMPTY && pl_color != next_pla) {
                buf.clear();
                self.nodes[i].accum_placements(&mut buf, x_size, y_size)?;

                let mut net_stones_added = 0;
                if !buf.is_empty() {
                    for m in &buf {
                        if board.colors[m.loc as usize] != C_EMPTY && m.pla == C_EMPTY {
                            net_stones_added -= 1;
                        }
                        if board.colors[m.loc as usize] == C_EMPTY && m.pla != C_EMPTY {
                            net_stones_added += 1;
                        }
                    }
                    if tolerate_illegal_moves {
                        let num_removed = board.set_stones_tolerant(&buf);
                        if num_removed > 0 {
                            eprintln!(
                                "WARNING: Removed {} zero-liberty stone(s) from illegal setup in {} SGF trace (branches 0-indexed): {}",
                                num_removed,
                                self.file_name,
                                build_trace(i)
                            );
                        }
                    } else {
                        let suc = board.set_stones_fail_if_no_libs(&buf);
                        if !suc {
                            return Err(prop_fail(format!(
                                "Illegal placements in {} SGF trace (branches 0-indexed): {}",
                                self.file_name,
                                build_trace(i)
                            )));
                        }
                    }
                }

                if !buf.is_empty() || (pl_color != C_EMPTY && pl_color != next_pla) {
                    board.clear_simple_ko_loc();
                    let mut initial_turn_number = hist.initial_turn_number;
                    initial_turn_number += hist.move_history.len() as i64;
                    if net_stones_added > 0 {
                        initial_turn_number += (net_stones_added + 1) / 2;
                    }
                    if i64::from(board.num_stones_on_board()) > initial_turn_number {
                        initial_turn_number = i64::from(board.num_stones_on_board());
                    }
                    if pl_color != C_EMPTY && pl_color != next_pla {
                        next_pla = pl_color;
                    }
                    hist.clear(board.clone(), next_pla, rules, 0);
                    hist.set_initial_turn_number(initial_turn_number);
                }
                self.sample_position_helper(
                    board,
                    hist,
                    next_pla,
                    sample_buf,
                    unique_hashes,
                    require_unique,
                    hash_comments,
                    hash_parent,
                    flip_if_pass_or_w_first,
                    allow_game_over,
                    &comments,
                    f,
                );
            }

            // Handle actual moves.
            buf.clear();
            self.nodes[i].accum_moves(&mut buf, x_size, y_size)?;

            for m in &buf {
                let move_loc = m.loc;
                let move_pla = m.pla;

                let tolerant_legal = hist.is_legal_tolerant(board, move_loc, move_pla);
                let simple_ko_banned = board.is_ko_banned(move_loc);
                let super_ko_illegal = hist.is_super_ko_banned(move_loc);
                let rule_suicide_illegal = move_loc != PASS_LOC
                    && board.is_illegal_suicide(
                        move_loc,
                        move_pla,
                        rules.multi_stone_suicide_legal,
                    );

                let interrupt_history_before_move =
                    simple_ko_banned || super_ko_illegal || move_pla != hist.presumed_next_move_pla;
                let interrupt_history_after_move = rule_suicide_illegal;

                if !tolerate_illegal_moves && (!tolerant_legal || simple_ko_banned) {
                    return Err(prop_fail(format!(
                        "Illegal move in {} effective turn {} move {} SGF trace (branches 0-indexed): {}",
                        self.file_name,
                        hist.get_current_turn_number(),
                        location::to_string(move_loc, board.x_size, board.y_size),
                        build_trace(i)
                    )));
                }

                if tolerate_illegal_moves && !tolerant_legal {
                    eprintln!(
                        "WARNING: Skipping illegal move in {} effective turn {} move {} SGF trace (branches 0-indexed): {}",
                        self.file_name,
                        hist.get_current_turn_number(),
                        location::to_string(move_loc, board.x_size, board.y_size),
                        build_trace(i)
                    );
                    interrupt_history_like_setup(board, hist, next_pla);
                    continue;
                }

                if tolerate_illegal_moves && simple_ko_banned {
                    eprintln!(
                        "WARNING: Tolerating simple ko violation in {} effective turn {} move {} SGF trace (branches 0-indexed): {}",
                        self.file_name,
                        hist.get_current_turn_number(),
                        location::to_string(move_loc, board.x_size, board.y_size),
                        build_trace(i)
                    );
                }

                if interrupt_history_before_move {
                    interrupt_history_like_setup(board, hist, next_pla);
                }

                let suc = hist.make_board_move_tolerant(board, move_loc, move_pla);
                assert!(suc);
                if hist.move_history.len() > 0x3FFFFFFF {
                    return Err(prop_fail("too many moves in sgf"));
                }
                next_pla = get_opp(move_pla);

                if interrupt_history_after_move {
                    interrupt_history_like_setup(board, hist, next_pla);
                }

                self.sample_position_helper(
                    board,
                    hist,
                    next_pla,
                    sample_buf,
                    unique_hashes,
                    require_unique,
                    hash_comments,
                    hash_parent,
                    flip_if_pass_or_w_first,
                    allow_game_over,
                    &comments,
                    f,
                );
            }
        }

        let mut permutation: Vec<usize> = (0..self.children.len()).collect();
        if let Some(rng) = rng {
            use rand::seq::SliceRandom;
            permutation.shuffle(rng);
        }

        for &child_idx in &permutation {
            let mut board_copy = board.clone();
            let mut hist_copy = hist.clone();
            variation_trace_nodes_branch.push((self.nodes.len() as i64, child_idx as i64));
            self.children[child_idx].iter_all_positions_helper(
                &mut board_copy,
                &mut hist_copy,
                next_pla,
                rules,
                x_size,
                y_size,
                sample_buf,
                unique_hashes,
                require_unique,
                hash_comments,
                hash_parent,
                flip_if_pass_or_w_first,
                allow_game_over,
                tolerate_illegal_moves,
                false,
                None,
                variation_trace_nodes_branch,
                f,
            )?;
            assert!(!variation_trace_nodes_branch.is_empty());
            variation_trace_nodes_branch.pop();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_position_helper<F>(
        &self,
        board: &Board,
        hist: &BoardHistory,
        next_pla: Player,
        sample_buf: &mut PositionSample,
        unique_hashes: &mut BTreeSet<Hash128>,
        require_unique: bool,
        hash_comments: bool,
        hash_parent: bool,
        flip_if_pass_or_w_first: bool,
        allow_game_over: bool,
        comments: &str,
        f: &mut F,
    ) where
        F: FnMut(&PositionSample, &BoardHistory, &str),
    {
        if !allow_game_over
            && (hist.is_game_finished
                || (hist.move_history.len() >= 2
                    && hist.move_history[hist.move_history.len() - 1].loc == PASS_LOC
                    && hist.move_history[hist.move_history.len() - 2].loc == PASS_LOC))
        {
            return;
        }

        let mut situation_hash = board.pos_hash;
        situation_hash ^= Board::zobrist_player_hash(next_pla as usize);
        assert_eq!(hist.encore_phase, 0);
        if board.ko_loc != NULL_LOC {
            situation_hash ^= Board::zobrist_ko_loc_hash(board.ko_loc as usize);
        }

        if hash_comments {
            situation_hash.hash0 = situation_hash
                .hash0
                .wrapping_add(kata_core::hash::simple_hash_str(comments));
        }

        if hash_parent {
            let mut parent_hash = Hash128::default();
            if !hist.move_history.is_empty() {
                let prev_board = hist.get_recent_board(1);
                parent_hash = prev_board.pos_hash;
                if prev_board.ko_loc != NULL_LOC {
                    parent_hash ^= Board::zobrist_ko_loc_hash(prev_board.ko_loc as usize);
                }
            }
            let mixed = Hash128::new(
                kata_core::hash::murmur_mix(parent_hash.hash1),
                kata_core::hash::split_mix64(parent_hash.hash0),
            );
            situation_hash ^= mixed;
        }

        if require_unique && unique_hashes.contains(&situation_hash) {
            return;
        }
        unique_hashes.insert(situation_hash);

        PositionSample::write_pos_of_hist(sample_buf, hist, next_pla);

        if flip_if_pass_or_w_first && hist.has_black_pass_or_white_first() {
            if let Ok(flipped) = sample_buf.get_color_flipped() {
                *sample_buf = flipped;
            }
        }

        f(sample_buf, hist, comments);
    }
}

/// A compact, flattened view of an SGF.
#[derive(Debug, Clone)]
pub struct CompactSgf {
    pub file_name: String,
    pub root_node: SgfNode,
    pub placements: Vec<Move>,
    pub moves: Vec<Move>,
    pub x_size: i32,
    pub y_size: i32,
    pub depth: i64,
    pub sgf_winner: Player,
    pub hash: Hash128,
}

impl CompactSgf {
    pub fn parse(s: &str) -> Result<Self, IOError> {
        Self::from_sgf(Sgf::parse(s)?)
    }

    pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Self, IOError> {
        Self::from_sgf(Sgf::parse_file(path)?)
    }

    fn from_sgf(sgf: Sgf) -> Result<Self, IOError> {
        let size = sgf.get_xy_size()?;
        let mut placements = Vec::new();
        let mut moves = Vec::new();
        sgf.get_placements(&mut placements, size.x, size.y)?;
        sgf.get_moves(&mut moves, size.x, size.y)?;
        let depth = sgf.depth();
        let hash = sgf.hash;
        let file_name = sgf.file_name.clone();

        if sgf.nodes.is_empty() {
            return Err(prop_fail("Empty sgf"));
        }
        let root_node = sgf.nodes[0].clone();
        let sgf_winner = root_node.get_sgf_winner();

        Ok(Self {
            file_name,
            root_node,
            placements,
            moves,
            x_size: size.x,
            y_size: size.y,
            depth,
            sgf_winner,
            hash,
        })
    }

    pub fn has_rules(&self) -> bool {
        self.root_node.has_property("RU")
    }

    pub fn get_rules_or_fail(&self) -> Result<Rules, IOError> {
        let mut rules = self.root_node.get_rules_from_ru_tag_or_fail()?;
        rules.set_komi(self.root_node.get_komi_or_fail()?);
        Ok(rules)
    }

    pub fn get_rules_or_fail_allow_unspecified(
        &self,
        default_rules: &Rules,
    ) -> Result<Rules, IOError> {
        let mut rules = if !self.has_rules() {
            *default_rules
        } else {
            self.root_node.get_rules_from_ru_tag_or_fail()?
        };
        if self.root_node.has_property("KM") {
            rules.set_komi(self.root_node.get_komi_or_fail()?);
        }
        Ok(rules)
    }

    /// Set up the initial board and history from this compact SGF.
    ///
    /// Corresponds to `CompactSgf::setupInitialBoardAndHist` in C++.
    pub fn setup_initial_board_and_hist(
        &self,
        initial_rules: &Rules,
        board: &mut Board,
        next_pla: &mut Player,
        hist: &mut BoardHistory,
    ) -> Result<(), IOError> {
        let pl_color = self.root_node.get_pl_specified_color();
        if pl_color == P_BLACK || pl_color == P_WHITE {
            *next_pla = pl_color;
        } else {
            let mut has_black = false;
            let mut all_black = true;
            for m in &self.placements {
                if m.pla == P_BLACK {
                    has_black = true;
                } else {
                    all_black = false;
                }
            }
            if has_black && all_black {
                *next_pla = P_WHITE;
            } else {
                *next_pla = P_BLACK;
            }
        }
        if !self.moves.is_empty() {
            *next_pla = self.moves[0].pla;
        }

        *board = Board::new(self.x_size, self.y_size);
        let suc = board.set_stones_fail_if_no_libs(&self.placements);
        if !suc {
            return Err(prop_fail(
                "setupInitialBoardAndHist: initial board position contains invalid stones or zero-liberty stones",
            ));
        }
        *hist = BoardHistory::new(board.clone(), *next_pla, *initial_rules, 0);
        if hist.initial_turn_number < i64::from(board.num_stones_on_board()) {
            hist.set_initial_turn_number(i64::from(board.num_stones_on_board()));
        }
        Ok(())
    }

    fn play_moves_assume_legal(
        &self,
        board: &mut Board,
        next_pla: &mut Player,
        hist: &mut BoardHistory,
        turn_idx: i64,
    ) -> Result<(), IOError> {
        if turn_idx < 0 || turn_idx > self.moves.len() as i64 {
            return Err(prop_fail(format!(
                "Attempting to set up position from SGF for invalid turn idx {}, valid values are {} to {}",
                turn_idx,
                0,
                self.moves.len()
            )));
        }
        for i in 0..turn_idx as usize {
            let m = self.moves[i];
            let suc = hist.make_board_move_tolerant(board, m.loc, m.pla);
            if !suc {
                return Err(prop_fail(format!(
                    "Illegal move in {} turn {} move {}",
                    self.file_name,
                    i,
                    location::to_string(m.loc, board.x_size, board.y_size)
                )));
            }
            *next_pla = get_opp(m.pla);
        }
        Ok(())
    }

    /// Set up the board and history to the requested turn index, assuming all
    /// moves in the SGF are legal.
    ///
    /// Corresponds to `CompactSgf::setupBoardAndHistAssumeLegal` in C++.
    pub fn setup_board_and_hist_assume_legal(
        &self,
        initial_rules: &Rules,
        board: &mut Board,
        next_pla: &mut Player,
        hist: &mut BoardHistory,
        turn_idx: i64,
    ) -> Result<(), IOError> {
        self.setup_initial_board_and_hist(initial_rules, board, next_pla, hist)?;
        self.play_moves_assume_legal(board, next_pla, hist, turn_idx)
    }
}

/// A sampled board position extracted from an SGF, together with the moves
/// that followed it and metadata needed to reconstruct the game history.
#[derive(Debug, Clone)]
pub struct PositionSample {
    pub board: Board,
    pub next_pla: Player,
    pub moves: Vec<Move>,
    pub initial_turn_number: i64,
    pub hint_loc: Loc,
    pub weight: f64,
    pub metadata: String,
    pub training_weight: f64,
}

impl Default for PositionSample {
    fn default() -> Self {
        Self {
            board: Board::default(),
            next_pla: P_BLACK,
            moves: Vec::new(),
            initial_turn_number: 0,
            hint_loc: NULL_LOC,
            weight: 1.0,
            metadata: String::new(),
            training_weight: 1.0,
        }
    }
}

impl PositionSample {
    fn write_pos_of_hist(sample_buf: &mut PositionSample, hist: &BoardHistory, next_pla: Player) {
        // Snap the position up to 5 turns ago so as to include up to 5 moves of history.
        let mut turns_ago_to_snap = 0;
        while turns_ago_to_snap < 5 {
            if turns_ago_to_snap >= hist.move_history.len() {
                break;
            }
            if turns_ago_to_snap > 0
                && hist.move_history[hist.move_history.len() - turns_ago_to_snap - 1].pla
                    == hist.move_history[hist.move_history.len() - turns_ago_to_snap].pla
            {
                break;
            }
            if turns_ago_to_snap == 0
                && hist.move_history[hist.move_history.len() - turns_ago_to_snap - 1].pla
                    == next_pla
            {
                break;
            }
            turns_ago_to_snap += 1;
        }
        let start_turn_idx = hist.move_history.len() - turns_ago_to_snap;

        sample_buf.board = hist.get_recent_board(turns_ago_to_snap).clone();
        sample_buf.next_pla = if start_turn_idx < hist.move_history.len() {
            hist.move_history[start_turn_idx].pla
        } else {
            next_pla
        };
        sample_buf.moves.clear();
        for i in start_turn_idx..hist.move_history.len() {
            sample_buf.moves.push(hist.move_history[i]);
        }
        sample_buf.initial_turn_number = hist.initial_turn_number + start_turn_idx as i64;
        sample_buf.hint_loc = NULL_LOC;
        sample_buf.weight = 1.0;
    }

    /// Serialize the sample as a single-line JSON object.
    pub fn to_json_line(sample: &PositionSample) -> String {
        let mut data = serde_json::Map::new();
        data.insert("xSize".to_string(), Value::from(sample.board.x_size));
        data.insert("ySize".to_string(), Value::from(sample.board.y_size));
        data.insert(
            "board".to_string(),
            Value::String(sample.board.to_string_simple('/')),
        );
        data.insert(
            "nextPla".to_string(),
            Value::String(player_io::player_to_string_short(sample.next_pla).to_string()),
        );
        let move_locs: Vec<Value> = sample
            .moves
            .iter()
            .map(|m| {
                Value::String(location::to_string(
                    m.loc,
                    sample.board.x_size,
                    sample.board.y_size,
                ))
            })
            .collect();
        let move_plas: Vec<Value> = sample
            .moves
            .iter()
            .map(|m| Value::String(player_io::player_to_string_short(m.pla).to_string()))
            .collect();
        data.insert("moveLocs".to_string(), Value::Array(move_locs));
        data.insert("movePlas".to_string(), Value::Array(move_plas));
        data.insert(
            "initialTurnNumber".to_string(),
            Value::from(sample.initial_turn_number),
        );
        data.insert(
            "hintLoc".to_string(),
            Value::String(location::to_string(
                sample.hint_loc,
                sample.board.x_size,
                sample.board.y_size,
            )),
        );
        data.insert("weight".to_string(), Value::from(sample.weight));
        if !sample.metadata.is_empty() {
            data.insert(
                "metadata".to_string(),
                Value::String(sample.metadata.clone()),
            );
        }
        if (sample.training_weight - 1.0).abs() > 1e-12 {
            data.insert(
                "trainingWeight".to_string(),
                Value::from(sample.training_weight),
            );
        }
        Value::Object(data).to_string()
    }

    /// Parse a single-line JSON object into a sample.
    pub fn of_json_line(s: &str) -> Result<Self, IOError> {
        let data: Value = serde_json::from_str(s)
            .map_err(|e| IOError(format!("Error parsing position sample json: {}\n{}", e, s)))?;
        let mut sample = PositionSample::default();

        let x_size = get_json_i32(&data, "xSize")?;
        let y_size = get_json_i32(&data, "ySize")?;
        let board_str = get_json_string(&data, "board")?;
        sample.board = Board::parse_board(x_size, y_size, &board_str, '/').map_err(|e| {
            IOError(format!(
                "Error parsing position sample json: {}\n{}",
                e.0, s
            ))
        })?;
        sample.next_pla = player_io::parse_player(&get_json_string(&data, "nextPla")?)?;

        let move_locs = get_json_array_of_strings(&data, "moveLocs")?;
        let move_plas = get_json_array_of_strings(&data, "movePlas")?;
        if move_locs.len() != move_plas.len() {
            return Err(IOError(format!(
                "Error parsing position sample json: moveLocs.len() != movePlas.size()\n{}",
                s
            )));
        }
        for i in 0..move_locs.len() {
            let move_loc =
                location::of_string(&move_locs[i], sample.board.x_size, sample.board.y_size)?;
            let move_pla = player_io::parse_player(&move_plas[i])?;
            sample.moves.push(Move::new(move_loc, move_pla));
        }

        sample.initial_turn_number = get_json_i64(&data, "initialTurnNumber")?;

        let hint_loc_str = global::to_lower(global::trim(&get_json_string(&data, "hintLoc")?));
        if hint_loc_str.is_empty()
            || hint_loc_str == "''"
            || hint_loc_str == "\"\""
            || hint_loc_str == "null"
            || hint_loc_str == "'null'"
            || hint_loc_str == "\"null\""
        {
            sample.hint_loc = NULL_LOC;
        } else {
            sample.hint_loc = location::of_string(
                &get_json_string(&data, "hintLoc")?,
                sample.board.x_size,
                sample.board.y_size,
            )?;
        }

        sample.weight = get_json_f64(&data, "weight").unwrap_or(1.0);
        sample.metadata = get_json_string(&data, "metadata").unwrap_or_default();
        sample.training_weight = get_json_f64(&data, "trainingWeight").unwrap_or(1.0);

        Ok(sample)
    }

    /// Return a copy with black and white swapped.
    pub fn get_color_flipped(&self) -> Result<Self, IOError> {
        let mut other = self.clone();
        let mut new_board = Board::new(other.board.x_size, other.board.y_size);
        for y in 0..other.board.y_size {
            for x in 0..other.board.x_size {
                let loc = location::get_loc(x, y, other.board.x_size);
                let c = other.board.colors[loc as usize];
                if (c == C_BLACK || c == C_WHITE)
                    && !new_board.set_stone_fail_if_no_libs(loc, get_opp(c))
                {
                    return Err(IOError(
                        "Color-flipped position from SGF has a stone with no liberties".to_string(),
                    ));
                }
            }
        }
        other.board = new_board;
        other.next_pla = get_opp(other.next_pla);
        for m in &mut other.moves {
            m.pla = get_opp(m.pla);
        }
        Ok(other)
    }

    pub fn has_previous_positions(&self, num_previous: usize) -> bool {
        self.moves.len() >= num_previous
    }

    pub fn previous_position(&self, new_weight: f64) -> Self {
        let mut other = self.clone();
        if !other.moves.is_empty() {
            other.moves.pop();
            other.hint_loc = NULL_LOC;
            other.weight = new_weight;
        }
        other
    }

    /// Reconstruct a board history by replaying the stored moves.
    pub fn try_get_current_board_history(
        &self,
        rules: &Rules,
        next_pla_to_move: &mut Player,
        hist: &mut BoardHistory,
    ) -> bool {
        let mut pla = self.next_pla;
        let mut board_copy = self.board.clone();
        hist.clear(board_copy.clone(), pla, *rules, 0);
        for m in &self.moves {
            if !hist.is_legal(&board_copy, m.loc, m.pla) {
                return false;
            }
            assert_eq!(m.pla, pla);
            hist.make_board_move_assume_legal(&mut board_copy, m.loc, m.pla);
            pla = get_opp(pla);
        }
        *next_pla_to_move = pla;
        true
    }

    pub fn get_current_turn_number(&self) -> i64 {
        (self.initial_turn_number + self.moves.len() as i64).max(0)
    }

    pub fn is_equal_for_testing(
        &self,
        other: &PositionSample,
        check_num_captures: bool,
        check_simple_ko: bool,
    ) -> bool {
        if !self
            .board
            .is_equal_for_testing(&other.board, check_num_captures, check_simple_ko)
        {
            return false;
        }
        if self.next_pla != other.next_pla {
            return false;
        }
        if self.moves.len() != other.moves.len() {
            return false;
        }
        for i in 0..self.moves.len() {
            if self.moves[i].pla != other.moves[i].pla {
                return false;
            }
            if self.moves[i].loc != other.moves[i].loc {
                return false;
            }
        }
        if self.initial_turn_number != other.initial_turn_number {
            return false;
        }
        if self.hint_loc != other.hint_loc {
            return false;
        }
        if (self.weight - other.weight).abs() > 1e-12 {
            return false;
        }
        true
    }
}

fn get_json_i32(data: &Value, key: &str) -> Result<i32, IOError> {
    data.get(key)
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .ok_or_else(|| IOError(format!("Missing or invalid integer field: {}", key)))
}

fn get_json_i64(data: &Value, key: &str) -> Result<i64, IOError> {
    data.get(key)
        .and_then(|v| v.as_i64())
        .ok_or_else(|| IOError(format!("Missing or invalid integer field: {}", key)))
}

fn get_json_f64(data: &Value, key: &str) -> Result<f64, IOError> {
    data.get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| IOError(format!("Missing or invalid float field: {}", key)))
}

fn get_json_string(data: &Value, key: &str) -> Result<String, IOError> {
    data.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| IOError(format!("Missing or invalid string field: {}", key)))
}

fn get_json_array_of_strings(data: &Value, key: &str) -> Result<Vec<String>, IOError> {
    data.get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect()
        })
        .ok_or_else(|| IOError(format!("Missing or invalid array field: {}", key)))
}

fn parse_rank(rank_str: &str) -> Result<i32, IOError> {
    const TOP_DAN: i32 = 13;
    const BOTTOM_KYU: i32 = 50;
    let rank_str_lower = global::to_lower(rank_str);

    if let Some(stripped) = rank_str_lower.strip_suffix("d") {
        if let Some(rank) = global::try_string_to_int(stripped) {
            if (1..=TOP_DAN).contains(&rank) {
                return Ok(rank - 1);
            }
        }
    }
    if let Some(stripped) = rank_str_lower.strip_suffix(" d") {
        if let Some(rank) = global::try_string_to_int(stripped) {
            if (1..=TOP_DAN).contains(&rank) {
                return Ok(rank - 1);
            }
        }
    }
    if let Some(stripped) = rank_str_lower.strip_suffix("dan") {
        if let Some(rank) = global::try_string_to_int(stripped) {
            if (1..=TOP_DAN).contains(&rank) {
                return Ok(rank - 1);
            }
        }
    }
    if let Some(stripped) = rank_str_lower.strip_suffix(" dan") {
        if let Some(rank) = global::try_string_to_int(stripped) {
            if (1..=TOP_DAN).contains(&rank) {
                return Ok(rank - 1);
            }
        }
    }
    // Chinese "段" suffix.
    if let Some(stripped) = rank_str.strip_suffix("\u{6BB5}") {
        if let Some(rank) = global::try_string_to_int(stripped) {
            if (1..=TOP_DAN).contains(&rank) {
                return Ok(rank - 1);
            }
        }
    }
    if let Some(stripped) = rank_str_lower.strip_suffix("p") {
        if let Some(rank) = global::try_string_to_int(stripped) {
            if (1..=TOP_DAN).contains(&rank) {
                return Ok(rank.max(9) - 1);
            }
        }
    }
    if let Some(stripped) = rank_str_lower.strip_suffix(" p") {
        if let Some(rank) = global::try_string_to_int(stripped) {
            if (1..=TOP_DAN).contains(&rank) {
                return Ok(rank.max(9) - 1);
            }
        }
    }
    if let Some(stripped) = rank_str_lower.strip_suffix("pro") {
        if let Some(rank) = global::try_string_to_int(stripped) {
            if (1..=TOP_DAN).contains(&rank) {
                return Ok(rank.max(9) - 1);
            }
        }
    }
    if let Some(stripped) = rank_str_lower.strip_suffix(" pro") {
        if let Some(rank) = global::try_string_to_int(stripped) {
            if (1..=TOP_DAN).contains(&rank) {
                return Ok(rank.max(9) - 1);
            }
        }
    }
    if let Some(stripped) = rank_str.strip_prefix("P") {
        if let Some(stripped) = stripped.strip_suffix("\u{6BB5}") {
            if let Some(rank) = global::try_string_to_int(stripped) {
                if (1..=TOP_DAN).contains(&rank) {
                    return Ok(rank.max(9) - 1);
                }
            }
        }
    }
    if let Some(stripped) = rank_str_lower.strip_suffix("k") {
        if let Some(rank) = global::try_string_to_int(stripped) {
            if (1..=BOTTOM_KYU).contains(&rank) {
                return Ok(-rank);
            }
        }
    }
    if let Some(stripped) = rank_str_lower.strip_suffix(" k") {
        if let Some(rank) = global::try_string_to_int(stripped) {
            if (1..=BOTTOM_KYU).contains(&rank) {
                return Ok(-rank);
            }
        }
    }
    if let Some(stripped) = rank_str_lower.strip_suffix("kyu") {
        if let Some(rank) = global::try_string_to_int(stripped) {
            if (1..=BOTTOM_KYU).contains(&rank) {
                return Ok(-rank);
            }
        }
    }
    if let Some(stripped) = rank_str_lower.strip_suffix(" kyu") {
        if let Some(rank) = global::try_string_to_int(stripped) {
            if (1..=BOTTOM_KYU).contains(&rank) {
                return Ok(-rank);
            }
        }
    }
    Err(prop_fail(format!(
        "Could not parse rank in sgf: {}",
        rank_str
    )))
}

fn game_result_no_sgf_tag(hist: &BoardHistory) -> String {
    if !hist.is_game_finished {
        return String::new();
    }
    if hist.is_no_result {
        return "Void".to_string();
    }
    if hist.is_resignation && hist.winner == C_BLACK {
        return "B+R".to_string();
    }
    if hist.is_resignation && hist.winner == C_WHITE {
        return "W+R".to_string();
    }
    if hist.winner == C_BLACK {
        format!("B+{}", -hist.final_white_minus_black_score)
    } else if hist.winner == C_WHITE {
        format!("W+{}", hist.final_white_minus_black_score)
    } else if hist.winner == C_EMPTY {
        "0".to_string()
    } else {
        String::new()
    }
}

fn print_game_result(out: &mut String, hist: &BoardHistory) {
    if hist.is_game_finished {
        out.push_str("RE[");
        out.push_str(&game_result_no_sgf_tag(hist));
        out.push(']');
    }
}

/// Minimal SGF writer for round-trip testing.
///
/// Corresponds to the `WriteSgf::writeSgf` overload used in the C++ SGF tests
/// (`gameData == NULL`, `tryNicerRulesString == false`,
/// `omitResignPlayerMove == false`). It emits enough metadata and move
/// notation for `CompactSgf::parse` to reconstruct the same board/history.
pub fn write_sgf(out: &mut String, b_name: &str, w_name: &str, end_hist: &BoardHistory) {
    let initial_board = &end_hist.initial_board;
    let rules = &end_hist.rules;
    let x_size = initial_board.x_size;
    let y_size = initial_board.y_size;

    out.push_str("(;FF[4]GM[1]");
    if x_size == y_size {
        out.push_str(&format!("SZ[{}]", x_size));
    } else {
        out.push_str(&format!("SZ[{}:{}]", x_size, y_size));
    }
    out.push_str(&format!("PB[{}]", b_name));
    out.push_str(&format!("PW[{}]", w_name));

    let mut hist_copy = end_hist.clone();
    hist_copy.set_assume_multiple_starting_black_moves_are_handicap(true);
    out.push_str(&format!("HA[{}]", hist_copy.compute_num_handicap_stones()));

    out.push_str(&format!("KM[{}]", rules.komi_f32()));
    out.push_str(&format!("RU[{}]", rules.to_legacy_string_no_komi()));
    print_game_result(out, end_hist);

    let mut has_ab = false;
    for y in 0..y_size {
        for x in 0..x_size {
            let loc = location::get_loc(x, y, x_size);
            if initial_board.colors[loc as usize] == C_BLACK {
                if !has_ab {
                    out.push_str("AB");
                    has_ab = true;
                }
                out.push('[');
                out.push_str(&write_sgf_loc(loc, x_size, y_size).unwrap_or_default());
                out.push(']');
            }
        }
    }

    let mut has_aw = false;
    for y in 0..y_size {
        for x in 0..x_size {
            let loc = location::get_loc(x, y, x_size);
            if initial_board.colors[loc as usize] == C_WHITE {
                if !has_aw {
                    out.push_str("AW");
                    has_aw = true;
                }
                out.push('[');
                out.push_str(&write_sgf_loc(loc, x_size, y_size).unwrap_or_default());
                out.push(']');
            }
        }
    }

    let mut board = initial_board.clone();
    let mut hist = BoardHistory::new(
        board.clone(),
        end_hist.initial_pla,
        end_hist.rules,
        end_hist.initial_encore_phase,
    );
    for m in &end_hist.move_history {
        out.push(';');
        let loc = m.loc;
        let pla = m.pla;
        if pla == P_BLACK {
            out.push_str("B[");
        } else {
            out.push_str("W[");
        }
        out.push_str(&write_sgf_loc(loc, x_size, y_size).unwrap_or_default());
        out.push(']');
        hist.make_board_move_assume_legal(&mut board, loc, pla);
    }
    out.push(')');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc(x: i32, y: i32, x_size: i32) -> Loc {
        location::get_loc(x, y, x_size)
    }

    #[test]
    fn test_parse_simple_sgf() {
        let s = "(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Black]PW[White];B[dd];W[qq])";
        let sgf = Sgf::parse(s).unwrap();
        assert_eq!(sgf.nodes.len(), 3);
        assert!(sgf.nodes[0].has_property("SZ"));
        assert_eq!(sgf.get_xy_size().unwrap(), XYSize::new(19, 19));
        assert!((sgf.get_komi_or_fail().unwrap() - 7.5).abs() < 1e-6);
        assert!(sgf.has_rules());
        let rules = sgf.get_rules_or_fail().unwrap();
        assert_eq!(rules.scoring_rule, kata_game::rules::ScoringRule::Area);
        assert_eq!(sgf.get_player_name(P_BLACK), "Black");
        assert_eq!(sgf.get_player_name(P_WHITE), "White");

        let mut moves = Vec::new();
        sgf.get_moves(&mut moves, 19, 19).unwrap();
        assert_eq!(moves.len(), 2);
        assert_eq!(moves[0], Move::new(loc(3, 3, 19), P_BLACK));
        assert_eq!(moves[1], Move::new(loc(16, 16, 19), P_WHITE));
    }

    #[test]
    fn test_parse_pass_and_tt() {
        let s = "(;SZ[19];B[];W[tt];B[pd])";
        let sgf = Sgf::parse(s).unwrap();
        let mut moves = Vec::new();
        sgf.get_moves(&mut moves, 19, 19).unwrap();
        assert_eq!(moves.len(), 3);
        assert_eq!(moves[0].loc, PASS_LOC);
        assert_eq!(moves[0].pla, P_BLACK);
        assert_eq!(moves[1].loc, PASS_LOC);
        assert_eq!(moves[1].pla, P_WHITE);
        assert_eq!(moves[2].loc, loc(15, 3, 19));
    }

    #[test]
    fn test_placements_and_rectangle() {
        let s = "(;SZ[9]AB[aa:ac]AW[bb];B[cc])";
        let sgf = Sgf::parse(s).unwrap();
        let mut placements = Vec::new();
        sgf.get_placements(&mut placements, 9, 9).unwrap();
        assert_eq!(placements.len(), 4);
        assert!(placements.contains(&Move::new(loc(0, 0, 9), P_BLACK)));
        assert!(placements.contains(&Move::new(loc(0, 1, 9), P_BLACK)));
        assert!(placements.contains(&Move::new(loc(0, 2, 9), P_BLACK)));
        assert!(placements.contains(&Move::new(loc(1, 1, 9), P_WHITE)));
    }

    #[test]
    fn test_branches_longest_child() {
        let s = "(;SZ[19];B[aa](;W[bb];B[cc])(;W[dd]))";
        let sgf = Sgf::parse(s).unwrap();
        let mut moves = Vec::new();
        sgf.get_moves(&mut moves, 19, 19).unwrap();
        assert_eq!(moves.len(), 3);
        assert_eq!(moves[0], Move::new(loc(0, 0, 19), P_BLACK));
        assert_eq!(moves[1], Move::new(loc(1, 1, 19), P_WHITE));
        assert_eq!(moves[2], Move::new(loc(2, 2, 19), P_BLACK));
    }

    #[test]
    fn test_rectangular_board() {
        let s = "(;SZ[13:9];B[aa])";
        let sgf = Sgf::parse(s).unwrap();
        assert_eq!(sgf.get_xy_size().unwrap(), XYSize::new(13, 9));
    }

    #[test]
    fn test_escaped_value() {
        let s = "(;SZ[19]C[hello \\] world];B[aa])";
        let sgf = Sgf::parse(s).unwrap();
        assert_eq!(
            sgf.nodes[0].get_single_property("C").unwrap(),
            "hello ] world"
        );
    }

    #[test]
    fn test_winner_and_first_player() {
        let s = "(;SZ[19]RE[W+R]PL[W];B[aa];W[bb])";
        let sgf = Sgf::parse(s).unwrap();
        assert_eq!(sgf.get_sgf_winner(), P_WHITE);
        assert_eq!(sgf.get_first_player_color().unwrap(), P_WHITE);

        let s2 = "(;SZ[19]RE[B+3.5];B[aa];W[bb])";
        let sgf2 = Sgf::parse(s2).unwrap();
        assert_eq!(sgf2.get_sgf_winner(), P_BLACK);
        assert_eq!(sgf2.get_first_player_color().unwrap(), P_BLACK);
    }

    #[test]
    fn test_ranks() {
        let s = "(;SZ[19]BR[3d]WR[5k])";
        let sgf = Sgf::parse(s).unwrap();
        assert_eq!(sgf.get_rank(P_BLACK).unwrap(), 2);
        assert_eq!(sgf.get_rank(P_WHITE).unwrap(), -5);
    }

    #[test]
    fn test_compact_sgf() {
        let s = "(;SZ[19]KM[6.5]RU[Japanese];B[dd];W[qq])";
        let compact = CompactSgf::parse(s).unwrap();
        assert_eq!(compact.x_size, 19);
        assert_eq!(compact.y_size, 19);
        assert_eq!(compact.moves.len(), 2);
        assert_eq!(compact.placements.len(), 0);
        let rules = compact.get_rules_or_fail().unwrap();
        assert_eq!(rules.scoring_rule, kata_game::rules::ScoringRule::Territory);
        assert!((rules.komi_f32() - 6.5).abs() < 1e-6);
    }

    #[test]
    fn test_write_sgf_loc() {
        assert_eq!(write_sgf_loc(PASS_LOC, 19, 19).unwrap(), "");
        assert_eq!(write_sgf_loc(loc(0, 0, 19), 19, 19).unwrap(), "aa");
        assert_eq!(write_sgf_loc(loc(3, 3, 19), 19, 19).unwrap(), "dd");
    }

    #[test]
    fn test_position_sample_json_roundtrip() {
        let s = "(;SZ[5]KM[7.5]RU[Chinese];B[cc];W[bb];B[cd])";
        let sgf = Sgf::parse(s).unwrap();
        let mut samples = Vec::new();
        sgf.iter_all_positions(
            false,
            false,
            None,
            &mut |sample, _hist, _comments| {
                samples.push(sample.clone());
            },
            true,
        )
        .unwrap();
        assert!(!samples.is_empty());
        let last = samples.last().unwrap();
        let json = PositionSample::to_json_line(last);
        let parsed = PositionSample::of_json_line(&json).unwrap();
        assert!(last.is_equal_for_testing(&parsed, false, false));
    }

    #[test]
    fn test_position_sample_color_flip() {
        let s = "(;SZ[5]KM[7.5]RU[Chinese];B[cc];W[bb])";
        let sgf = Sgf::parse(s).unwrap();
        let mut samples = Vec::new();
        sgf.iter_all_positions(
            false,
            false,
            None,
            &mut |sample, _hist, _comments| {
                samples.push(sample.clone());
            },
            true,
        )
        .unwrap();
        let sample = samples.last().unwrap();
        let flipped = sample.get_color_flipped().unwrap();
        assert_eq!(flipped.next_pla, get_opp(sample.next_pla));
        assert_eq!(flipped.moves.len(), sample.moves.len());
    }

    #[test]
    fn test_iter_all_positions_simple_sgf() {
        let s = "(;SZ[5]KM[7.5]RU[Chinese];B[cc];W[bb];B[cd])";
        let sgf = Sgf::parse(s).unwrap();
        let mut count = 0;
        sgf.iter_all_positions(
            false,
            false,
            None,
            &mut |_sample, _hist, _comments| {
                count += 1;
            },
            true,
        )
        .unwrap();
        // One sample per node (root + each move node).
        assert_eq!(count, 4);
    }

    #[test]
    fn test_load_all_unique_positions_uniqueness() {
        let s = "(;SZ[5]KM[7.5]RU[Chinese];B[cc];W[bb];B[cd])";
        let sgf = Sgf::parse(s).unwrap();
        let mut unique_hashes = BTreeSet::new();
        let samples = sgf
            .load_all_unique_positions(&mut unique_hashes, false, false, false, false, None, true)
            .unwrap();
        assert!(!samples.is_empty());
        assert_eq!(samples.len(), unique_hashes.len());
    }

    #[test]
    fn test_position_sample_replay_history() {
        let s = "(;SZ[5]KM[7.5]RU[Chinese];B[cc];W[bb];B[cd])";
        let sgf = Sgf::parse(s).unwrap();
        let mut samples = Vec::new();
        sgf.iter_all_positions(
            false,
            false,
            None,
            &mut |sample, _hist, _comments| {
                samples.push(sample.clone());
            },
            true,
        )
        .unwrap();
        let sample = samples.last().unwrap();
        let rules = Rules::get_tromp_taylorish();
        let mut next_pla = C_EMPTY;
        let mut hist = BoardHistory::default();
        assert!(sample.try_get_current_board_history(&rules, &mut next_pla, &mut hist));
        assert!(next_pla == P_BLACK || next_pla == P_WHITE);
    }
}

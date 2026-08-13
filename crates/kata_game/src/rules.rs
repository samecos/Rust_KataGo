//! Go game rules.
//!
//! Corresponds to `cpp/game/rules.h` and `cpp/game/rules.cpp`.

use kata_core::global;
use kata_core::global::IOError;
use kata_core::hash::Hash128;
use serde_json::json;
use std::collections::BTreeSet;
use std::fmt;

/// Ko rule variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KoRule {
    /// Only immediate recapture of a single stone is illegal.
    #[default]
    Simple,
    /// Any repeated board position is illegal.
    Positional,
    /// A repeated board position with the same player to move is illegal.
    Situational,
    /// Spight-style situational superko (not widely used).
    Spight,
}

impl KoRule {
    pub const SIMPLE: i32 = 0;
    pub const POSITIONAL: i32 = 1;
    pub const SITUATIONAL: i32 = 2;
    pub const SPIGHT: i32 = 3;

    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Simple),
            1 => Some(Self::Positional),
            2 => Some(Self::Situational),
            3 => Some(Self::Spight),
            _ => None,
        }
    }

    pub fn to_i32(self) -> i32 {
        match self {
            Self::Simple => 0,
            Self::Positional => 1,
            Self::Situational => 2,
            Self::Spight => 3,
        }
    }

    pub fn all() -> BTreeSet<Self> {
        let mut s = BTreeSet::new();
        s.insert(Self::Simple);
        s.insert(Self::Positional);
        s.insert(Self::Situational);
        s.insert(Self::Spight);
        s
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Simple => "SIMPLE",
            Self::Positional => "POSITIONAL",
            Self::Situational => "SITUATIONAL",
            Self::Spight => "SPIGHT",
        }
    }
}

impl std::str::FromStr for KoRule {
    type Err = IOError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "SIMPLE" => Ok(Self::Simple),
            "POSITIONAL" => Ok(Self::Positional),
            "SITUATIONAL" => Ok(Self::Situational),
            "SPIGHT" => Ok(Self::Spight),
            _ => Err(IOError(format!(
                "Rules::parseKoRule: Invalid ko rule: {}",
                s
            ))),
        }
    }
}

impl fmt::Display for KoRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Ord for KoRule {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_i32().cmp(&other.to_i32())
    }
}

impl PartialOrd for KoRule {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Scoring rule variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ScoringRule {
    /// Area scoring: stones + territory.
    #[default]
    Area,
    /// Territory scoring: territory - captures.
    Territory,
}

impl ScoringRule {
    pub const AREA: i32 = 0;
    pub const TERRITORY: i32 = 1;

    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Area),
            1 => Some(Self::Territory),
            _ => None,
        }
    }

    pub fn to_i32(self) -> i32 {
        match self {
            Self::Area => 0,
            Self::Territory => 1,
        }
    }

    pub fn all() -> BTreeSet<Self> {
        let mut s = BTreeSet::new();
        s.insert(Self::Area);
        s.insert(Self::Territory);
        s
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Area => "AREA",
            Self::Territory => "TERRITORY",
        }
    }
}

impl std::str::FromStr for ScoringRule {
    type Err = IOError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "AREA" => Ok(Self::Area),
            "TERRITORY" => Ok(Self::Territory),
            _ => Err(IOError(format!(
                "Rules::parseScoringRule: Invalid scoring rule: {}",
                s
            ))),
        }
    }
}

impl fmt::Display for ScoringRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Ord for ScoringRule {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_i32().cmp(&other.to_i32())
    }
}

impl PartialOrd for ScoringRule {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Tax rule variants (only relevant for territory scoring).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TaxRule {
    /// No tax.
    #[default]
    None,
    /// Seki groups are taxed.
    Seki,
    /// All groups are taxed (ancient scoring).
    All,
}

impl TaxRule {
    pub const NONE: i32 = 0;
    pub const SEKI: i32 = 1;
    pub const ALL: i32 = 2;

    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Seki),
            2 => Some(Self::All),
            _ => None,
        }
    }

    pub fn to_i32(self) -> i32 {
        match self {
            Self::None => 0,
            Self::Seki => 1,
            Self::All => 2,
        }
    }

    pub fn all() -> BTreeSet<Self> {
        let mut s = BTreeSet::new();
        s.insert(Self::None);
        s.insert(Self::Seki);
        s.insert(Self::All);
        s
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Seki => "SEKI",
            Self::All => "ALL",
        }
    }
}

impl std::str::FromStr for TaxRule {
    type Err = IOError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "NONE" => Ok(Self::None),
            "SEKI" => Ok(Self::Seki),
            "ALL" => Ok(Self::All),
            _ => Err(IOError(format!(
                "Rules::parseTaxRule: Invalid tax rule: {}",
                s
            ))),
        }
    }
}

impl fmt::Display for TaxRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Ord for TaxRule {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_i32().cmp(&other.to_i32())
    }
}

impl PartialOrd for TaxRule {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// White handicap bonus rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WhiteHandicapBonusRule {
    /// No bonus.
    #[default]
    Zero,
    /// White receives N points bonus per handicap stone.
    N,
    /// White receives N-1 points bonus per handicap stone.
    NMinusOne,
}

impl WhiteHandicapBonusRule {
    pub const ZERO: i32 = 0;
    pub const N: i32 = 1;
    pub const N_MINUS_ONE: i32 = 2;

    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Zero),
            1 => Some(Self::N),
            2 => Some(Self::NMinusOne),
            _ => None,
        }
    }

    pub fn to_i32(self) -> i32 {
        match self {
            Self::Zero => 0,
            Self::N => 1,
            Self::NMinusOne => 2,
        }
    }

    pub fn all() -> BTreeSet<Self> {
        let mut s = BTreeSet::new();
        s.insert(Self::Zero);
        s.insert(Self::N);
        s.insert(Self::NMinusOne);
        s
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Zero => "0",
            Self::N => "N",
            Self::NMinusOne => "N-1",
        }
    }
}

impl std::str::FromStr for WhiteHandicapBonusRule {
    type Err = IOError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" => Ok(Self::Zero),
            "N" => Ok(Self::N),
            "N-1" => Ok(Self::NMinusOne),
            _ => Err(IOError(format!(
                "Rules::parseWhiteHandicapBonusRule: Invalid whiteHandicapBonus rule: {}",
                s
            ))),
        }
    }
}

impl fmt::Display for WhiteHandicapBonusRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Ord for WhiteHandicapBonusRule {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_i32().cmp(&other.to_i32())
    }
}

impl PartialOrd for WhiteHandicapBonusRule {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A complete set of game rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rules {
    pub ko_rule: KoRule,
    pub scoring_rule: ScoringRule,
    pub tax_rule: TaxRule,
    pub multi_stone_suicide_legal: bool,
    pub has_button: bool,
    pub white_handicap_bonus_rule: WhiteHandicapBonusRule,
    /// Informational flag used by GTP/Analysis to adjust passing behavior.
    pub friendly_pass_ok: bool,
    pub komi: i32, // Komi stored in half-points to avoid float issues.
}

impl Default for Rules {
    fn default() -> Self {
        Self {
            ko_rule: KoRule::Positional,
            scoring_rule: ScoringRule::Area,
            tax_rule: TaxRule::None,
            multi_stone_suicide_legal: true,
            has_button: false,
            white_handicap_bonus_rule: WhiteHandicapBonusRule::Zero,
            friendly_pass_ok: false,
            komi: 15, // 7.5 in half-points.
        }
    }
}

impl Rules {
    pub const MIN_USER_KOMI: f32 = -150.0;
    pub const MAX_USER_KOMI: f32 = 150.0;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ko_rule: KoRule,
        scoring_rule: ScoringRule,
        tax_rule: TaxRule,
        multi_stone_suicide_legal: bool,
        has_button: bool,
        white_handicap_bonus_rule: WhiteHandicapBonusRule,
        friendly_pass_ok: bool,
        komi: f32,
    ) -> Self {
        Self {
            ko_rule,
            scoring_rule,
            tax_rule,
            multi_stone_suicide_legal,
            has_button,
            white_handicap_bonus_rule,
            friendly_pass_ok,
            komi: Self::komi_to_half_points(komi),
        }
    }

    pub fn get_tromp_taylorish() -> Self {
        Self {
            ko_rule: KoRule::Positional,
            scoring_rule: ScoringRule::Area,
            tax_rule: TaxRule::None,
            multi_stone_suicide_legal: true,
            has_button: false,
            white_handicap_bonus_rule: WhiteHandicapBonusRule::Zero,
            friendly_pass_ok: false,
            komi: Self::komi_to_half_points(7.5),
        }
    }

    pub fn get_simple_territory() -> Self {
        Self {
            ko_rule: KoRule::Simple,
            scoring_rule: ScoringRule::Territory,
            tax_rule: TaxRule::Seki,
            multi_stone_suicide_legal: false,
            has_button: false,
            white_handicap_bonus_rule: WhiteHandicapBonusRule::Zero,
            friendly_pass_ok: false,
            komi: Self::komi_to_half_points(7.5),
        }
    }

    /// Returns true if all rule fields except komi match.
    pub fn equals_ignoring_komi(&self, other: &Self) -> bool {
        self.ko_rule == other.ko_rule
            && self.scoring_rule == other.scoring_rule
            && self.tax_rule == other.tax_rule
            && self.multi_stone_suicide_legal == other.multi_stone_suicide_legal
            && self.has_button == other.has_button
            && self.white_handicap_bonus_rule == other.white_handicap_bonus_rule
            && self.friendly_pass_ok == other.friendly_pass_ok
    }

    /// Returns true if the final game result will be an integer.
    pub fn game_result_will_be_integer(&self) -> bool {
        let komi_is_integer = self.komi % 2 == 0;
        komi_is_integer != self.has_button
    }

    pub fn komi_f32(&self) -> f32 {
        self.komi as f32 / 2.0
    }

    pub fn set_komi(&mut self, komi: f32) {
        self.komi = Self::komi_to_half_points(komi);
    }

    fn komi_to_half_points(komi: f32) -> i32 {
        (komi * 2.0).round() as i32
    }

    pub fn komi_is_int_or_half_int(komi: f32) -> bool {
        komi.is_finite() && (komi * 2.0).round() == komi * 2.0
    }

    pub fn ko_rule_strings() -> BTreeSet<&'static str> {
        KoRule::all().into_iter().map(KoRule::as_str).collect()
    }

    pub fn scoring_rule_strings() -> BTreeSet<&'static str> {
        ScoringRule::all()
            .into_iter()
            .map(ScoringRule::as_str)
            .collect()
    }

    pub fn tax_rule_strings() -> BTreeSet<&'static str> {
        TaxRule::all().into_iter().map(TaxRule::as_str).collect()
    }

    pub fn white_handicap_bonus_rule_strings() -> BTreeSet<&'static str> {
        WhiteHandicapBonusRule::all()
            .into_iter()
            .map(WhiteHandicapBonusRule::as_str)
            .collect()
    }

    /// Parse a rules string that includes komi.
    pub fn parse_rules(s: &str) -> Result<Self, IOError> {
        parse_rules_helper(s, true)
    }

    /// Parse a rules string without komi, using the supplied value.
    pub fn parse_rules_without_komi(s: &str, komi: f32) -> Result<Self, IOError> {
        let mut rules = parse_rules_helper(s, false)?;
        rules.set_komi(komi);
        Ok(rules)
    }

    pub fn try_parse_rules(s: &str) -> Option<Self> {
        Self::parse_rules(s).ok()
    }

    pub fn try_parse_rules_without_komi(s: &str, komi: f32) -> Option<Self> {
        Self::parse_rules_without_komi(s, komi).ok()
    }

    /// Update a single key/value pair on top of existing rules.
    pub fn update_rules(key: &str, value: &str, prior: &Self) -> Result<Self, IOError> {
        let mut rules = *prior;
        let k = global::trim(key);
        let upper = global::to_upper(value);
        let v = global::trim(&upper);
        match k {
            "ko" => rules.ko_rule = v.parse()?,
            "score" | "scoring" => rules.scoring_rule = v.parse()?,
            "tax" => rules.tax_rule = v.parse()?,
            "suicide" => rules.multi_stone_suicide_legal = global::string_to_bool(v)?,
            "hasButton" => rules.has_button = global::string_to_bool(v)?,
            "whiteHandicapBonus" => rules.white_handicap_bonus_rule = v.parse()?,
            "friendlyPassOk" => rules.friendly_pass_ok = global::string_to_bool(v)?,
            _ => return Err(IOError(format!("Unknown rules option: {}", k))),
        }
        Ok(rules)
    }

    /// Serialize to the compact legacy string format.
    pub fn to_legacy_string_no_komi(&self) -> String {
        let bool_str = |b: bool| if b { "1" } else { "0" };
        let mut out = format!(
            "ko{}score{}tax{}sui{}",
            self.ko_rule,
            self.scoring_rule,
            self.tax_rule,
            bool_str(self.multi_stone_suicide_legal)
        );
        if self.has_button {
            out.push_str(&format!("button{}", bool_str(self.has_button)));
        }
        if self.white_handicap_bonus_rule != WhiteHandicapBonusRule::Zero {
            out.push_str(&format!("whb{}", self.white_handicap_bonus_rule));
        }
        if self.friendly_pass_ok {
            out.push_str(&format!("fpok{}", bool_str(self.friendly_pass_ok)));
        }
        out
    }

    pub fn to_legacy_string(&self) -> String {
        format!("{}komi{}", self.to_legacy_string_no_komi(), self.komi_f32())
    }

    /// Return a nice name if the rules match a known preset, else the compact form.
    pub fn to_legacy_string_no_komi_maybe_nice(&self) -> String {
        let presets = [
            ("TrompTaylor", Self::get_tromp_taylorish()),
            ("Japanese", Self::preset_japanese()),
            ("Chinese", Self::preset_chinese()),
            ("Chinese-OGS", Self::preset_chinese_ogs()),
            ("AGA", Self::preset_aga()),
            ("StoneScoring", Self::preset_stone_scoring()),
            ("NewZealand", Self::preset_new_zealand()),
        ];
        for (name, preset) in presets {
            if self.equals_ignoring_komi(&preset) {
                return name.to_string();
            }
        }
        self.to_legacy_string_no_komi()
    }

    fn json_helper(&self, omit_komi: bool, omit_defaults: bool) -> serde_json::Value {
        let mut ret = serde_json::Map::new();
        ret.insert("ko".to_string(), json!(self.ko_rule.as_str()));
        ret.insert("scoring".to_string(), json!(self.scoring_rule.as_str()));
        ret.insert("tax".to_string(), json!(self.tax_rule.as_str()));
        ret.insert("suicide".to_string(), json!(self.multi_stone_suicide_legal));
        if !omit_defaults || self.has_button {
            ret.insert("hasButton".to_string(), json!(self.has_button));
        }
        if !omit_defaults || self.white_handicap_bonus_rule != WhiteHandicapBonusRule::Zero {
            ret.insert(
                "whiteHandicapBonus".to_string(),
                json!(self.white_handicap_bonus_rule.as_str()),
            );
        }
        if !omit_defaults || self.friendly_pass_ok {
            ret.insert("friendlyPassOk".to_string(), json!(self.friendly_pass_ok));
        }
        if !omit_komi {
            ret.insert("komi".to_string(), json!(self.komi_f32()));
        }
        serde_json::Value::Object(ret)
    }

    pub fn to_json(&self) -> serde_json::Value {
        self.json_helper(false, false)
    }

    pub fn to_json_no_komi(&self) -> serde_json::Value {
        self.json_helper(true, false)
    }

    pub fn to_json_no_komi_maybe_omit_stuff(&self) -> serde_json::Value {
        self.json_helper(true, true)
    }

    pub fn to_json_string(&self) -> String {
        self.to_json().to_string()
    }

    pub fn to_json_string_no_komi(&self) -> String {
        self.to_json_no_komi().to_string()
    }

    pub fn to_json_string_no_komi_maybe_omit_stuff(&self) -> String {
        self.to_json_no_komi_maybe_omit_stuff().to_string()
    }

    // Zobrist hashes for rule components.
    pub fn zobrist_ko_rule_hash(ko_rule: KoRule) -> Hash128 {
        Self::ZOBRIST_KO_RULE_HASH[ko_rule.to_i32() as usize]
    }

    pub fn zobrist_scoring_rule_hash(scoring_rule: ScoringRule) -> Hash128 {
        Self::ZOBRIST_SCORING_RULE_HASH[scoring_rule.to_i32() as usize]
    }

    pub fn zobrist_tax_rule_hash(tax_rule: TaxRule) -> Hash128 {
        Self::ZOBRIST_TAX_RULE_HASH[tax_rule.to_i32() as usize]
    }

    pub fn zobrist_multi_stone_suicide_hash() -> Hash128 {
        Self::ZOBRIST_MULTI_STONE_SUICIDE_HASH
    }

    pub fn zobrist_button_hash() -> Hash128 {
        Self::ZOBRIST_BUTTON_HASH
    }

    pub fn zobrist_friendly_pass_ok_hash() -> Hash128 {
        Self::ZOBRIST_FRIENDLY_PASS_OK_HASH
    }

    pub const ZOBRIST_KO_RULE_HASH: [Hash128; 4] = [
        Hash128::new(0x3cc7e0bf846820f6, 0x1fb7fbde5fc6ba4e),
        Hash128::new(0xcc18f5d47188554a, 0x3a63152c23e4128d),
        Hash128::new(0x3bc55e42b23b35bf, 0xc75fa1e615621dcd),
        Hash128::new(0x5b2096e48241d21b, 0x23cc18d4e85cd67f),
    ];

    pub const ZOBRIST_SCORING_RULE_HASH: [Hash128; 2] = [
        Hash128::new(
            0x8b3ed7598f901494 ^ 0x72eeccc72c82a5e7,
            0x1dfd47ac77bce5f8 ^ 0x0d1265e413623e2b,
        ),
        Hash128::new(
            0x381345dc357ec982 ^ 0x125bfe48a41042d5,
            0x03ba55c026026b56 ^ 0x061866b5f2b98a79,
        ),
    ];

    pub const ZOBRIST_TAX_RULE_HASH: [Hash128; 3] = [
        Hash128::new(0x72eeccc72c82a5e7, 0x0d1265e413623e2b),
        Hash128::new(0x125bfe48a41042d5, 0x061866b5f2b98a79),
        Hash128::new(0xa384ece9d8ee713c, 0xfdc9f3b5d1f3732b),
    ];

    pub const ZOBRIST_MULTI_STONE_SUICIDE_HASH: Hash128 =
        Hash128::new(0xf9b475b3bbf35e37, 0xefa19d8b1e5b3e5a);

    pub const ZOBRIST_BUTTON_HASH: Hash128 = Hash128::new(0xb8b914c9234ece84, 0x3d759cddebe29c14);

    pub const ZOBRIST_FRIENDLY_PASS_OK_HASH: Hash128 =
        Hash128::new(0x0113655998ef0a25, 0x99c9d04ecd964874);

    // Known presets (without komi).
    fn preset_japanese() -> Self {
        Self {
            ko_rule: KoRule::Simple,
            scoring_rule: ScoringRule::Territory,
            tax_rule: TaxRule::Seki,
            multi_stone_suicide_legal: false,
            has_button: false,
            white_handicap_bonus_rule: WhiteHandicapBonusRule::Zero,
            friendly_pass_ok: false,
            komi: Self::komi_to_half_points(6.5),
        }
    }

    fn preset_chinese() -> Self {
        Self {
            ko_rule: KoRule::Simple,
            scoring_rule: ScoringRule::Area,
            tax_rule: TaxRule::None,
            multi_stone_suicide_legal: false,
            has_button: false,
            white_handicap_bonus_rule: WhiteHandicapBonusRule::N,
            friendly_pass_ok: true,
            komi: Self::komi_to_half_points(7.5),
        }
    }

    fn preset_chinese_ogs() -> Self {
        Self {
            ko_rule: KoRule::Positional,
            scoring_rule: ScoringRule::Area,
            tax_rule: TaxRule::None,
            multi_stone_suicide_legal: false,
            has_button: false,
            white_handicap_bonus_rule: WhiteHandicapBonusRule::N,
            friendly_pass_ok: true,
            komi: Self::komi_to_half_points(7.5),
        }
    }

    fn preset_aga() -> Self {
        Self {
            ko_rule: KoRule::Situational,
            scoring_rule: ScoringRule::Area,
            tax_rule: TaxRule::None,
            multi_stone_suicide_legal: false,
            has_button: false,
            white_handicap_bonus_rule: WhiteHandicapBonusRule::NMinusOne,
            friendly_pass_ok: true,
            komi: Self::komi_to_half_points(7.5),
        }
    }

    fn preset_stone_scoring() -> Self {
        Self {
            ko_rule: KoRule::Simple,
            scoring_rule: ScoringRule::Area,
            tax_rule: TaxRule::All,
            multi_stone_suicide_legal: false,
            has_button: false,
            white_handicap_bonus_rule: WhiteHandicapBonusRule::Zero,
            friendly_pass_ok: true,
            komi: Self::komi_to_half_points(7.5),
        }
    }

    fn preset_new_zealand() -> Self {
        Self {
            ko_rule: KoRule::Situational,
            scoring_rule: ScoringRule::Area,
            tax_rule: TaxRule::None,
            multi_stone_suicide_legal: true,
            has_button: false,
            white_handicap_bonus_rule: WhiteHandicapBonusRule::Zero,
            friendly_pass_ok: true,
            komi: Self::komi_to_half_points(7.5),
        }
    }
}

impl fmt::Display for Rules {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_legacy_string())
    }
}

fn parse_rules_helper(s_orig: &str, allow_komi: bool) -> Result<Rules, IOError> {
    let lowercased = global::to_lower(global::trim(s_orig));

    if lowercased == "japanese" || lowercased == "korean" {
        return Ok(Rules::preset_japanese());
    }
    if lowercased == "chinese" {
        return Ok(Rules::preset_chinese());
    }
    if matches!(
        lowercased.as_str(),
        "chineseogs" | "chinese_ogs" | "chinese-ogs" | "chinesekgs" | "chinese_kgs" | "chinese-kgs"
    ) {
        return Ok(Rules::preset_chinese_ogs());
    }
    if matches!(
        lowercased.as_str(),
        "ancientarea"
            | "ancient-area"
            | "ancient_area"
            | "ancient area"
            | "stonescoring"
            | "stone-scoring"
            | "stone_scoring"
            | "stone scoring"
    ) {
        return Ok(Rules::preset_stone_scoring());
    }
    if matches!(
        lowercased.as_str(),
        "ancientterritory" | "ancient-territory" | "ancient_territory" | "ancient territory"
    ) {
        let mut r = Rules::preset_stone_scoring();
        r.scoring_rule = ScoringRule::Territory;
        r.komi = Rules::komi_to_half_points(6.5);
        return Ok(r);
    }
    if matches!(
        lowercased.as_str(),
        "agabutton" | "aga-button" | "aga_button" | "aga button"
    ) {
        let mut r = Rules::preset_aga();
        r.has_button = true;
        r.komi = Rules::komi_to_half_points(7.0);
        return Ok(r);
    }
    if lowercased == "aga" || lowercased == "bga" || lowercased == "french" {
        return Ok(Rules::preset_aga());
    }
    if matches!(
        lowercased.as_str(),
        "nz" | "newzealand" | "new zealand" | "new-zealand" | "new_zealand"
    ) {
        return Ok(Rules::preset_new_zealand());
    }
    if matches!(
        lowercased.as_str(),
        "tromp-taylor" | "tromp_taylor" | "tromp taylor" | "tromptaylor"
    ) {
        return Ok(Rules::get_tromp_taylorish());
    }
    if lowercased == "goe" || lowercased == "ing" {
        return Ok(Rules::get_tromp_taylorish());
    }

    if !s_orig.is_empty() && s_orig.starts_with('{') {
        return parse_json_rules(s_orig, allow_komi);
    }

    parse_legacy_rules(s_orig, allow_komi)
}

fn parse_json_rules(s_orig: &str, allow_komi: bool) -> Result<Rules, IOError> {
    let mut rules = Rules::get_tromp_taylorish();
    let value: serde_json::Value = serde_json::from_str(s_orig)
        .map_err(|_| IOError(format!("Could not parse rules: {}", s_orig)))?;
    let obj = value
        .as_object()
        .ok_or_else(|| IOError(format!("Could not parse rules: {}", s_orig)))?;

    let mut komi_specified = false;
    let mut tax_specified = false;

    for (key, val) in obj {
        match key.as_str() {
            "ko" => {
                rules.ko_rule = val
                    .as_str()
                    .ok_or_else(|| IOError(format!("Invalid ko value in rules: {}", s_orig)))?
                    .parse()?;
            }
            "score" | "scoring" => {
                rules.scoring_rule = val
                    .as_str()
                    .ok_or_else(|| IOError(format!("Invalid scoring value in rules: {}", s_orig)))?
                    .parse()?;
            }
            "tax" => {
                rules.tax_rule = val
                    .as_str()
                    .ok_or_else(|| IOError(format!("Invalid tax value in rules: {}", s_orig)))?
                    .parse()?;
                tax_specified = true;
            }
            "suicide" => {
                rules.multi_stone_suicide_legal = val.as_bool().ok_or_else(|| {
                    IOError(format!("Invalid suicide value in rules: {}", s_orig))
                })?;
            }
            "hasButton" => {
                rules.has_button = val.as_bool().ok_or_else(|| {
                    IOError(format!("Invalid hasButton value in rules: {}", s_orig))
                })?;
            }
            "whiteHandicapBonus" => {
                rules.white_handicap_bonus_rule = val
                    .as_str()
                    .ok_or_else(|| {
                        IOError(format!(
                            "Invalid whiteHandicapBonus value in rules: {}",
                            s_orig
                        ))
                    })?
                    .parse()?;
            }
            "friendlyPassOk" => {
                rules.friendly_pass_ok = val.as_bool().ok_or_else(|| {
                    IOError(format!("Invalid friendlyPassOk value in rules: {}", s_orig))
                })?;
            }
            "komi" => {
                if !allow_komi {
                    return Err(IOError(format!("Unknown rules option: {}", key)));
                }
                let komi = val
                    .as_f64()
                    .ok_or_else(|| IOError(format!("Invalid komi value in rules: {}", s_orig)))?
                    as f32;
                if !(Rules::MIN_USER_KOMI..=Rules::MAX_USER_KOMI).contains(&komi)
                    || !Rules::komi_is_int_or_half_int(komi)
                {
                    return Err(IOError(
                        "Komi value is not a half-integer or is too extreme".to_string(),
                    ));
                }
                rules.set_komi(komi);
                komi_specified = true;
            }
            _ => return Err(IOError(format!("Unknown rules option: {}", key))),
        }
    }

    if !tax_specified {
        rules.tax_rule = if rules.scoring_rule == ScoringRule::Territory {
            TaxRule::Seki
        } else {
            TaxRule::None
        };
    }
    if !komi_specified {
        if rules.scoring_rule == ScoringRule::Territory {
            rules.set_komi(6.5);
        } else if rules.has_button {
            rules.set_komi(7.0);
        }
    }

    Ok(rules)
}

fn parse_legacy_rules(s_orig: &str, allow_komi: bool) -> Result<Rules, IOError> {
    let mut rules = Rules::get_tromp_taylorish();
    let mut s = global::trim(s_orig).to_string();

    if s.is_empty() {
        return Err(IOError(format!("Could not parse rules: {}", s_orig)));
    }

    let mut komi_specified = false;
    let mut tax_specified = false;

    loop {
        if s.is_empty() {
            break;
        }

        if starts_with_and_strip(&mut s, "komi") {
            if !allow_komi {
                return Err(IOError(format!("Could not parse rules: {}", s_orig)));
            }
            let end_idx = s
                .find(|c: char| c.is_ascii_alphabetic() || global::is_whitespace_char(c))
                .unwrap_or(s.len());
            let komi_str = &s[..end_idx];
            let komi = global::try_string_to_float(komi_str)
                .ok_or_else(|| IOError(format!("Could not parse rules: {}", s_orig)))?;
            if !komi.is_finite() || !(-1e5..=1e5).contains(&komi) {
                return Err(IOError(format!("Could not parse rules: {}", s_orig)));
            }
            rules.set_komi(komi);
            komi_specified = true;
            s = s[end_idx..].to_string();
            s = global::trim(&s).to_string();
            continue;
        }
        if starts_with_and_strip(&mut s, "ko") {
            if starts_with_and_strip(&mut s, "SIMPLE") {
                rules.ko_rule = KoRule::Simple;
            } else if starts_with_and_strip(&mut s, "POSITIONAL") {
                rules.ko_rule = KoRule::Positional;
            } else if starts_with_and_strip(&mut s, "SITUATIONAL") {
                rules.ko_rule = KoRule::Situational;
            } else if starts_with_and_strip(&mut s, "SPIGHT") {
                rules.ko_rule = KoRule::Spight;
            } else {
                return Err(IOError(format!("Could not parse rules: {}", s_orig)));
            }
            continue;
        }
        if starts_with_and_strip(&mut s, "scoring") {
            if starts_with_and_strip(&mut s, "AREA") {
                rules.scoring_rule = ScoringRule::Area;
            } else if starts_with_and_strip(&mut s, "TERRITORY") {
                rules.scoring_rule = ScoringRule::Territory;
            } else {
                return Err(IOError(format!("Could not parse rules: {}", s_orig)));
            }
            continue;
        }
        if starts_with_and_strip(&mut s, "score") {
            if starts_with_and_strip(&mut s, "AREA") {
                rules.scoring_rule = ScoringRule::Area;
            } else if starts_with_and_strip(&mut s, "TERRITORY") {
                rules.scoring_rule = ScoringRule::Territory;
            } else {
                return Err(IOError(format!("Could not parse rules: {}", s_orig)));
            }
            continue;
        }
        if starts_with_and_strip(&mut s, "tax") {
            if starts_with_and_strip(&mut s, "NONE") {
                rules.tax_rule = TaxRule::None;
                tax_specified = true;
            } else if starts_with_and_strip(&mut s, "SEKI") {
                rules.tax_rule = TaxRule::Seki;
                tax_specified = true;
            } else if starts_with_and_strip(&mut s, "ALL") {
                rules.tax_rule = TaxRule::All;
                tax_specified = true;
            } else {
                return Err(IOError(format!("Could not parse rules: {}", s_orig)));
            }
            continue;
        }
        if starts_with_and_strip(&mut s, "sui") {
            if starts_with_and_strip(&mut s, "1") {
                rules.multi_stone_suicide_legal = true;
            } else if starts_with_and_strip(&mut s, "0") {
                rules.multi_stone_suicide_legal = false;
            } else {
                return Err(IOError(format!("Could not parse rules: {}", s_orig)));
            }
            continue;
        }
        if starts_with_and_strip(&mut s, "button") {
            if starts_with_and_strip(&mut s, "1") {
                rules.has_button = true;
            } else if starts_with_and_strip(&mut s, "0") {
                rules.has_button = false;
            } else {
                return Err(IOError(format!("Could not parse rules: {}", s_orig)));
            }
            continue;
        }
        if starts_with_and_strip(&mut s, "whb") {
            if starts_with_and_strip(&mut s, "0") {
                rules.white_handicap_bonus_rule = WhiteHandicapBonusRule::Zero;
            } else if starts_with_and_strip(&mut s, "N-1") {
                rules.white_handicap_bonus_rule = WhiteHandicapBonusRule::NMinusOne;
            } else if starts_with_and_strip(&mut s, "N") {
                rules.white_handicap_bonus_rule = WhiteHandicapBonusRule::N;
            } else {
                return Err(IOError(format!("Could not parse rules: {}", s_orig)));
            }
            continue;
        }
        if starts_with_and_strip(&mut s, "fpok") {
            if starts_with_and_strip(&mut s, "1") {
                rules.friendly_pass_ok = true;
            } else if starts_with_and_strip(&mut s, "0") {
                rules.friendly_pass_ok = false;
            } else {
                return Err(IOError(format!("Could not parse rules: {}", s_orig)));
            }
            continue;
        }

        return Err(IOError(format!("Could not parse rules: {}", s_orig)));
    }

    if !tax_specified {
        rules.tax_rule = if rules.scoring_rule == ScoringRule::Territory {
            TaxRule::Seki
        } else {
            TaxRule::None
        };
    }
    if !komi_specified {
        if rules.scoring_rule == ScoringRule::Territory {
            rules.set_komi(6.5);
        } else if rules.has_button {
            rules.set_komi(7.0);
        }
    }

    Ok(rules)
}

fn starts_with_and_strip(s: &mut String, prefix: &str) -> bool {
    if s.starts_with(prefix) {
        *s = s[prefix.len()..].to_string();
        *s = global::trim(s).to_string();
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_rules() {
        let r = Rules::default();
        assert_eq!(r.ko_rule, KoRule::Positional);
        assert_eq!(r.scoring_rule, ScoringRule::Area);
        assert!((r.komi_f32() - 7.5).abs() < 1e-6);
    }

    #[test]
    fn test_parse_presets() {
        let j = Rules::parse_rules("Japanese").unwrap();
        assert_eq!(j.scoring_rule, ScoringRule::Territory);
        assert_eq!(j.ko_rule, KoRule::Simple);
        assert!((j.komi_f32() - 6.5).abs() < 1e-6);

        let c = Rules::parse_rules("Chinese").unwrap();
        assert_eq!(c.scoring_rule, ScoringRule::Area);
        assert_eq!(c.white_handicap_bonus_rule, WhiteHandicapBonusRule::N);
        assert!(c.friendly_pass_ok);

        let tt = Rules::parse_rules("TrompTaylor").unwrap();
        assert_eq!(tt, Rules::get_tromp_taylorish());
    }

    #[test]
    fn test_parse_json() {
        let json = r#"{"ko":"SIMPLE","scoring":"TERRITORY","komi":6.5}"#;
        let r = Rules::parse_rules(json).unwrap();
        assert_eq!(r.ko_rule, KoRule::Simple);
        assert_eq!(r.scoring_rule, ScoringRule::Territory);
        assert_eq!(r.tax_rule, TaxRule::Seki);
        assert!((r.komi_f32() - 6.5).abs() < 1e-6);
    }

    #[test]
    fn test_parse_legacy() {
        let s = "koSIMPLEscoreTERRITORYsui0komi6.5";
        let r = Rules::parse_rules(s).unwrap();
        assert_eq!(r.ko_rule, KoRule::Simple);
        assert_eq!(r.scoring_rule, ScoringRule::Territory);
        assert!(!r.multi_stone_suicide_legal);
        assert!((r.komi_f32() - 6.5).abs() < 1e-6);
    }

    #[test]
    fn test_to_string_roundtrip() {
        let r = Rules::get_tromp_taylorish();
        let s = r.to_legacy_string();
        let r2 = Rules::parse_rules(&s).unwrap();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_update_rules() {
        let r = Rules::default();
        let r2 = Rules::update_rules("ko", "SIMPLE", &r).unwrap();
        assert_eq!(r2.ko_rule, KoRule::Simple);
    }

    #[test]
    fn test_nice_names() {
        let j = Rules::parse_rules("Japanese").unwrap();
        assert_eq!(j.to_legacy_string_no_komi_maybe_nice(), "Japanese");

        let custom = Rules::new(
            KoRule::Spight,
            ScoringRule::Area,
            TaxRule::All,
            true,
            true,
            WhiteHandicapBonusRule::NMinusOne,
            true,
            7.5,
        );
        assert!(!custom.to_legacy_string_no_komi_maybe_nice().contains(' '));
    }

    #[test]
    fn test_game_result_integer() {
        let mut r = Rules::default();
        assert!(!r.game_result_will_be_integer());
        r.set_komi(7.0);
        assert!(r.game_result_will_be_integer());
    }

    #[test]
    fn test_zobrist_hashes() {
        let h = Rules::zobrist_ko_rule_hash(KoRule::Simple);
        assert_ne!(h, Hash128::default());
    }
}

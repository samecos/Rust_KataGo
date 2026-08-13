//! SGF-derived metadata for neural-net input.
//!
//! Corresponds to `cpp/neuralnet/sgfmetadata.h` and `cpp/neuralnet/sgfmetadata.cpp`.

use kata_core::global::{StringError, chop_prefix, is_prefix, split_by};
use kata_core::hash::{Hash128, combine, nasam};
use kata_core::time::SimpleDate;
use kata_game::board::{P_BLACK, P_WHITE, Player};

/// Number of channels produced by [`SgfMetadata::fill_metadata_row`].
pub const METADATA_INPUT_NUM_CHANNELS: usize = 192;

pub const SOURCE_OGS: i32 = 1;
pub const SOURCE_KGS: i32 = 2;
pub const SOURCE_FOX: i32 = 3;
pub const SOURCE_TYGEM: i32 = 4;
pub const SOURCE_GOGOD: i32 = 5;
pub const SOURCE_GO4GO: i32 = 6;

/// Metadata describing the source game of a training position.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SgfMetadata {
    pub initialized: bool,

    /// KG = 0, 9d = 1, 8d = 2, ..., 1d = 9, 1k = 10, ...
    pub inverse_b_rank: i32,
    pub inverse_w_rank: i32,
    pub b_is_unranked: bool,
    pub w_is_unranked: bool,
    pub b_rank_is_unknown: bool,
    pub w_rank_is_unknown: bool,
    pub b_is_human: bool,
    pub w_is_human: bool,

    pub game_is_unrated: bool,
    pub game_ratedness_is_unknown: bool,

    pub tc_is_unknown: bool,
    pub tc_is_none: bool,
    pub tc_is_absolute: bool,
    pub tc_is_simple: bool,
    pub tc_is_byo_yomi: bool,
    pub tc_is_canadian: bool,
    pub tc_is_fischer: bool,

    pub main_time_seconds: f64,
    pub period_time_seconds: f64,
    pub byo_yomi_periods: i32,
    pub canadian_moves: i32,

    pub game_date: SimpleDate,

    pub source: i32,
}

impl SgfMetadata {
    /// Compute a 128-bit hash of this metadata for the given player to move.
    pub fn get_hash(&self, next_player: Player) -> Hash128 {
        if !self.initialized
            || !(0..128).contains(&self.inverse_b_rank)
            || !(0..128).contains(&self.inverse_w_rank)
            || !(0..128).contains(&self.source)
            || self.main_time_seconds < 0.0
            || self.period_time_seconds < 0.0
            || self.byo_yomi_periods < 0
            || self.canadian_moves < 0
        {
            panic!("Invalid or uninitialized SGFMetadata for hash");
        }

        let b = self.inverse_b_rank as u32
            + self.b_is_unranked as u32 * 128
            + self.b_rank_is_unknown as u32 * 256
            + self.b_is_human as u32 * 512;
        let w = self.inverse_w_rank as u32
            + self.w_is_unranked as u32 * 128
            + self.w_rank_is_unknown as u32 * 256
            + self.w_is_human as u32 * 512;

        let mut x0: u32 = 0;
        let mut x1: u32 = 0;
        let mut x2: u32 = 0;
        let mut x3: u32 = 0;

        if next_player == P_BLACK {
            x0 = x0.wrapping_add(b.wrapping_add(w << 10));
        } else {
            x0 = x0.wrapping_add(w.wrapping_add(b << 10));
        }

        x0 = x0.wrapping_add((self.game_is_unrated as u32) << 20);
        x0 = x0.wrapping_add((self.game_ratedness_is_unknown as u32) << 21);

        let which_tc: u32 = if self.tc_is_unknown {
            1
        } else if self.tc_is_none {
            2
        } else if self.tc_is_absolute {
            3
        } else if self.tc_is_simple {
            4
        } else if self.tc_is_byo_yomi {
            5
        } else if self.tc_is_canadian {
            6
        } else if self.tc_is_fischer {
            7
        } else {
            0
        };

        x0 = x0.wrapping_add(which_tc << 22);
        // 7 bits left for source going up to 128.
        x0 = x0.wrapping_add((self.source as u32) << 25);

        if !self.main_time_seconds.is_finite() || !self.period_time_seconds.is_finite() {
            panic!(
                "Invalid SGFMetadata: non-finite time values, mainTimeSeconds={} periodTimeSeconds={}",
                self.main_time_seconds, self.period_time_seconds
            );
        }
        let main_time_capped = self.main_time_seconds.clamp(0.0, 3.0 * 86400.0);
        let period_time_capped = self.period_time_seconds.clamp(0.0, 1.0 * 86400.0);
        x1 = x1.wrapping_add((main_time_capped * 4.0) as u32);
        x2 = x2.wrapping_add((period_time_capped * 32.0) as u32);

        let byo_yomi_capped = self.byo_yomi_periods.clamp(0, 50);
        let canadian_capped = self.canadian_moves.clamp(0, 50);
        x1 = x1.wrapping_add((byo_yomi_capped as u32) << 24);
        x2 = x2.wrapping_add((canadian_capped as u32) << 24);

        let epoch = SimpleDate::new(1970, 1, 1).expect("valid epoch");
        let days_difference = self.game_date.num_days_after(&epoch);
        x3 = x3.wrapping_add(days_difference as u32);

        let mut h0 = combine(x0, x1);
        let mut h1 = combine(x2, x3);
        h0 = nasam(h0);
        h1 = h1.wrapping_add(h0);
        h1 = nasam(h1);
        h0 = h0.wrapping_add(h1);

        Hash128::new(h0, h1)
    }

    /// Fill the 192-channel metadata row for `next_player` on a board of `board_area` points.
    ///
    /// # Panics
    /// Panics if `row` is shorter than [`METADATA_INPUT_NUM_CHANNELS`] or if the metadata is
    /// uninitialized / has invalid flags.
    pub fn fill_metadata_row(&self, row: &mut [f32], next_player: Player, board_area: i32) {
        assert!(
            row.len() >= METADATA_INPUT_NUM_CHANNELS,
            "metadata row too short"
        );
        if !self.initialized {
            panic!("Invalid or uninitialized SGFMetadata");
        }

        row[..METADATA_INPUT_NUM_CHANNELS].fill(0.0);

        let pla_is_human = if next_player == P_WHITE {
            self.w_is_human
        } else {
            self.b_is_human
        };
        let opp_is_human = if next_player == P_WHITE {
            self.b_is_human
        } else {
            self.w_is_human
        };
        row[0] = if pla_is_human { 1.0 } else { 0.0 };
        row[1] = if opp_is_human { 1.0 } else { 0.0 };

        let pla_is_unranked = if next_player == P_WHITE {
            self.w_is_unranked
        } else {
            self.b_is_unranked
        };
        let opp_is_unranked = if next_player == P_WHITE {
            self.b_is_unranked
        } else {
            self.w_is_unranked
        };
        row[2] = if pla_is_unranked { 1.0 } else { 0.0 };
        row[3] = if opp_is_unranked { 1.0 } else { 0.0 };

        let pla_rank_is_unknown = if next_player == P_WHITE {
            self.w_rank_is_unknown
        } else {
            self.b_rank_is_unknown
        };
        let opp_rank_is_unknown = if next_player == P_WHITE {
            self.b_rank_is_unknown
        } else {
            self.w_rank_is_unknown
        };
        row[4] = if pla_rank_is_unknown { 1.0 } else { 0.0 };
        row[5] = if opp_rank_is_unknown { 1.0 } else { 0.0 };

        const RANK_START_IDX: usize = 6;
        const RANK_LEN_PER_PLA: usize = 34;
        let inv_pla_rank = if next_player == P_WHITE {
            self.inverse_w_rank
        } else {
            self.inverse_b_rank
        };
        let inv_opp_rank = if next_player == P_WHITE {
            self.inverse_b_rank
        } else {
            self.inverse_w_rank
        };
        if !pla_is_unranked {
            for i in 0..inv_pla_rank.min(RANK_LEN_PER_PLA as i32) as usize {
                row[RANK_START_IDX + i] = 1.0;
            }
        }
        if !opp_is_unranked {
            for i in 0..inv_opp_rank.min(RANK_LEN_PER_PLA as i32) as usize {
                row[RANK_START_IDX + RANK_LEN_PER_PLA + i] = 1.0;
            }
        }

        row[74] = if self.game_ratedness_is_unknown {
            0.5
        } else if self.game_is_unrated {
            1.0
        } else {
            0.0
        };

        row[75] = if self.tc_is_unknown { 1.0 } else { 0.0 };
        row[76] = if self.tc_is_none { 1.0 } else { 0.0 };
        row[77] = if self.tc_is_absolute { 1.0 } else { 0.0 };
        row[78] = if self.tc_is_simple { 1.0 } else { 0.0 };
        row[79] = if self.tc_is_byo_yomi { 1.0 } else { 0.0 };
        row[80] = if self.tc_is_canadian { 1.0 } else { 0.0 };
        row[81] = if self.tc_is_fischer { 1.0 } else { 0.0 };

        let tc_sum = row[75] + row[76] + row[77] + row[78] + row[79] + row[80] + row[81];
        if (tc_sum - 1.0).abs() > 1e-6 {
            panic!("SGFMetadata has invalid time control flags - exactly one must be set");
        }

        let main_time_capped = self.main_time_seconds.clamp(0.0, 3.0 * 86400.0);
        let period_time_capped = self.period_time_seconds.clamp(0.0, 1.0 * 86400.0);
        row[82] = (0.4 * ((main_time_capped + 60.0).ln() - 6.5)) as f32;
        row[83] = (0.3 * ((period_time_capped + 1.0).ln() - 3.0)) as f32;
        let byo_yomi_capped = self.byo_yomi_periods.clamp(0, 50);
        let canadian_capped = self.canadian_moves.clamp(0, 50);
        row[84] = (0.5 * ((byo_yomi_capped as f64 + 2.0).ln() - 1.5)) as f32;
        row[85] = (0.25 * ((canadian_capped as f64 + 2.0).ln() - 1.5)) as f32;

        row[86] = (0.5 * ((board_area as f64 / 361.0).ln())) as f32;

        let epoch = SimpleDate::new(1970, 1, 1).expect("valid epoch");
        let days_difference = self.game_date.num_days_after(&epoch) as f64;
        const DATE_START_IDX: usize = 87;
        const DATE_LEN: usize = 32;
        let mut period: f64 = 7.0;
        let factor: f64 = 80000_f64.powf(1.0 / (DATE_LEN as f64 - 1.0));
        const TWO_PI: f64 = std::f64::consts::TAU;
        for i in 0..DATE_LEN {
            let num_revolutions = days_difference / period;
            row[DATE_START_IDX + i * 2] = (num_revolutions * TWO_PI).cos() as f32;
            row[DATE_START_IDX + i * 2 + 1] = (num_revolutions * TWO_PI).sin() as f32;
            period *= factor;
        }

        if !(0..16).contains(&self.source) {
            panic!("SGFMetadata has invalid source: {}", self.source);
        }
        row[151 + self.source as usize] = 1.0;
    }

    /// Convenience wrapper that allocates and returns the 192-channel row.
    pub fn metadata_row(&self, next_player: Player, board_area: i32) -> Vec<f32> {
        let mut row = vec![0.0f32; METADATA_INPUT_NUM_CHANNELS];
        self.fill_metadata_row(&mut row, next_player, board_area);
        row
    }
}

fn inverse_rank_from_str(rank_str: &str) -> Option<i32> {
    match rank_str {
        "9d" => Some(1),
        "8d" => Some(2),
        "7d" => Some(3),
        "6d" => Some(4),
        "5d" => Some(5),
        "4d" => Some(6),
        "3d" => Some(7),
        "2d" => Some(8),
        "1d" => Some(9),
        "1k" => Some(10),
        "2k" => Some(11),
        "3k" => Some(12),
        "4k" => Some(13),
        "5k" => Some(14),
        "6k" => Some(15),
        "7k" => Some(16),
        "8k" => Some(17),
        "9k" => Some(18),
        "10k" => Some(19),
        "11k" => Some(20),
        "12k" => Some(21),
        "13k" => Some(22),
        "14k" => Some(23),
        "15k" => Some(24),
        "16k" => Some(25),
        "17k" => Some(26),
        "18k" => Some(27),
        "19k" => Some(28),
        "20k" => Some(29),
        _ => None,
    }
}

fn make_basic_rank_profile(
    inverse_rank_black: i32,
    inverse_rank_white: i32,
    pre_az: bool,
) -> SgfMetadata {
    SgfMetadata {
        initialized: true,
        inverse_b_rank: inverse_rank_black,
        inverse_w_rank: inverse_rank_white,
        b_is_human: true,
        w_is_human: true,
        game_ratedness_is_unknown: true,
        tc_is_byo_yomi: true,
        main_time_seconds: 1200.0,
        period_time_seconds: 30.0,
        byo_yomi_periods: 5,
        game_date: if pre_az {
            SimpleDate::new(2016, 9, 1).expect("valid date")
        } else {
            SimpleDate::new(2020, 3, 1).expect("valid date")
        },
        source: SOURCE_KGS,
        ..SgfMetadata::default()
    }
}

fn make_historical_pro_profile(date: SimpleDate) -> SgfMetadata {
    SgfMetadata {
        initialized: true,
        inverse_b_rank: 1,
        inverse_w_rank: 1,
        b_is_human: true,
        w_is_human: true,
        tc_is_unknown: true,
        game_date: date,
        source: SOURCE_GOGOD,
        ..SgfMetadata::default()
    }
}

fn make_modern_pro_profile(date: SimpleDate) -> SgfMetadata {
    SgfMetadata {
        initialized: true,
        inverse_b_rank: 1,
        inverse_w_rank: 1,
        b_is_human: true,
        w_is_human: true,
        tc_is_unknown: true,
        game_date: date,
        source: SOURCE_GO4GO,
        ..SgfMetadata::default()
    }
}

/// Parse a human-SL profile name into an initialized metadata profile.
pub fn get_profile(human_sl_profile_name: &str) -> Result<SgfMetadata, StringError> {
    if human_sl_profile_name.is_empty()
        || human_sl_profile_name == "_"
        || human_sl_profile_name == "\"\""
    {
        return Ok(SgfMetadata::default());
    }

    if is_prefix(human_sl_profile_name, "proyear_") {
        let year_str = chop_prefix(human_sl_profile_name, "proyear_")
            .map_err(|e| StringError { message: e.message })?;
        if let Ok(year) = year_str.parse::<i32>() {
            if (1800..=2020).contains(&year) {
                return Ok(make_historical_pro_profile(
                    SimpleDate::new(year, 6, 1).expect("valid date"),
                ));
            }
            if (2021..=2023).contains(&year) {
                return Ok(make_modern_pro_profile(
                    SimpleDate::new(year, 6, 1).expect("valid date"),
                ));
            }
        }
    }

    if is_prefix(human_sl_profile_name, "rank_") || is_prefix(human_sl_profile_name, "preaz_") {
        let (ranks_str, pre_az) = if is_prefix(human_sl_profile_name, "rank_") {
            (
                chop_prefix(human_sl_profile_name, "rank_")
                    .map_err(|e| StringError { message: e.message })?,
                false,
            )
        } else {
            (
                chop_prefix(human_sl_profile_name, "preaz_")
                    .map_err(|e| StringError { message: e.message })?,
                true,
            )
        };

        if let Some(inverse_rank) = inverse_rank_from_str(ranks_str) {
            return Ok(make_basic_rank_profile(inverse_rank, inverse_rank, pre_az));
        }

        let pieces = split_by(ranks_str, '_');
        if pieces.len() == 2 {
            if let (Some(b), Some(w)) = (
                inverse_rank_from_str(&pieces[0]),
                inverse_rank_from_str(&pieces[1]),
            ) {
                return Ok(make_basic_rank_profile(b, w, pre_az));
            }
        }
    }

    Err(StringError {
        message: format!(
            "Unknown human SL network profile: {}",
            human_sl_profile_name
        ),
    })
}

/// Returns an arbitrary valid profile for neural-net warmup.
pub fn make_dummy_warmup_profile() -> SgfMetadata {
    make_modern_pro_profile(SimpleDate::new(2020, 1, 1).expect("valid date"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_meta() -> SgfMetadata {
        SgfMetadata {
            initialized: true,
            inverse_b_rank: 1,
            inverse_w_rank: 10,
            b_is_human: true,
            w_is_human: false,
            tc_is_byo_yomi: true,
            main_time_seconds: 1200.0,
            period_time_seconds: 30.0,
            byo_yomi_periods: 5,
            game_date: SimpleDate::new(2020, 6, 15).expect("valid date"),
            source: SOURCE_KGS,
            ..SgfMetadata::default()
        }
    }

    #[test]
    fn test_equality() {
        let a = sample_meta();
        let mut b = sample_meta();
        assert_eq!(a, b);
        b.b_is_human = false;
        assert_ne!(a, b);
    }

    #[test]
    fn test_hash_depends_on_next_player() {
        let meta = sample_meta();
        let h_black = meta.get_hash(P_BLACK);
        let h_white = meta.get_hash(P_WHITE);
        assert_ne!(h_black, h_white);
    }

    #[test]
    fn test_fill_row_length() {
        let meta = sample_meta();
        let row = meta.metadata_row(P_BLACK, 361);
        assert_eq!(row.len(), METADATA_INPUT_NUM_CHANNELS);
    }

    #[test]
    fn test_fill_row_human_channel() {
        let meta = sample_meta();
        let row_black = meta.metadata_row(P_BLACK, 361);
        assert_eq!(row_black[0], 1.0); // black (pla) is human
        assert_eq!(row_black[1], 0.0); // white (opp) is not human

        let row_white = meta.metadata_row(P_WHITE, 361);
        assert_eq!(row_white[0], 0.0); // black (opp) is human, white (pla) is not
        assert_eq!(row_white[1], 1.0);
    }

    #[test]
    fn test_rank_encoding_cumulative() {
        let mut meta = sample_meta();
        meta.inverse_b_rank = 3;
        meta.b_is_unranked = false;
        meta.b_rank_is_unknown = false;
        let row = meta.metadata_row(P_BLACK, 361);
        assert_eq!(row[6], 1.0);
        assert_eq!(row[7], 1.0);
        assert_eq!(row[8], 1.0);
        assert_eq!(row[9], 0.0);
    }

    #[test]
    fn test_time_control_one_hot() {
        let row = sample_meta().metadata_row(P_BLACK, 361);
        assert_eq!(row[75], 0.0);
        assert_eq!(row[76], 0.0);
        assert_eq!(row[77], 0.0);
        assert_eq!(row[78], 0.0);
        assert_eq!(row[79], 1.0);
        assert_eq!(row[80], 0.0);
        assert_eq!(row[81], 0.0);
    }

    #[test]
    fn test_source_one_hot() {
        let row = sample_meta().metadata_row(P_BLACK, 361);
        assert_eq!(row[151 + SOURCE_KGS as usize], 1.0);
    }

    #[test]
    fn test_get_profile_rank() {
        let profile = get_profile("rank_3d").unwrap();
        assert!(profile.initialized);
        assert_eq!(profile.inverse_b_rank, 7);
        assert_eq!(profile.source, SOURCE_KGS);
    }

    #[test]
    fn test_get_profile_proyear() {
        let profile = get_profile("proyear_2015").unwrap();
        assert!(profile.initialized);
        assert_eq!(profile.source, SOURCE_GOGOD);
    }

    #[test]
    fn test_get_profile_empty_is_uninitialized() {
        let profile = get_profile("").unwrap();
        assert!(!profile.initialized);
    }

    #[test]
    fn test_get_profile_unknown() {
        assert!(get_profile("rank_30d").is_err());
    }

    #[test]
    fn test_dummy_warmup_profile_is_initialized() {
        let profile = make_dummy_warmup_profile();
        assert!(profile.initialized);
    }
}

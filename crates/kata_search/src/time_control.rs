//! Go time-control logic.
//!
//! Corresponds to `cpp/search/timecontrols.h` and `cpp/search/timecontrols.cpp`.

use kata_core::global::StringError;
use kata_game::board::Board;
use kata_game::history::BoardHistory;

/// Supported time controls: Fischer, absolute, byo-yomi, or Canadian.
#[derive(Debug, Clone)]
pub struct TimeControls {
    pub original_main_time: f64,
    pub increment: f64,
    pub main_time_limit: f64,
    pub max_time_per_move: f64,
    pub original_num_periods: i32,
    pub num_stones_per_period: i32,
    pub per_period_time: f64,

    pub main_time_left: f64,
    pub in_overtime: bool,
    pub num_periods_left_including_current: i32,
    pub num_stones_left_in_period: i32,
    pub time_left_in_period: f64,
}

impl TimeControls {
    /// Threshold at which time is considered unlimited.
    pub const UNLIMITED_TIME_THRESHOLD: f64 = 1e20;
    /// The max time tolerated from a user input.
    pub const MAX_USER_INPUT_TIME: f64 = 1e25;
    /// Default "unlimited" value for fields that need it.
    pub const UNLIMITED_TIME_DEFAULT: f64 = 1e30;
    /// Default "unlimited" value that must be larger than other values.
    pub const UNLIMITED_TIME_DEFAULT_LARGE: f64 = 1e40;

    /// Create an unlimited-time control.
    pub fn new() -> Self {
        Self {
            original_main_time: Self::UNLIMITED_TIME_DEFAULT,
            increment: 0.0,
            main_time_limit: Self::UNLIMITED_TIME_DEFAULT_LARGE,
            max_time_per_move: Self::UNLIMITED_TIME_DEFAULT_LARGE,
            original_num_periods: 0,
            num_stones_per_period: 0,
            per_period_time: 0.0,
            main_time_left: Self::UNLIMITED_TIME_DEFAULT,
            in_overtime: false,
            num_periods_left_including_current: 0,
            num_stones_left_in_period: 0,
            time_left_in_period: 0.0,
        }
    }

    /// Absolute time: a fixed main time, no increment.
    pub fn absolute_time(main_time: f64) -> Self {
        let mut tc = Self::new();
        tc.original_main_time = main_time;
        tc.main_time_left = main_time;
        tc
    }

    /// Fischer time: main time plus an increment each move.
    pub fn fischer_time(main_time: f64, increment: f64) -> Self {
        let mut tc = Self::new();
        tc.original_main_time = main_time;
        tc.increment = increment;
        tc.main_time_left = main_time;
        tc
    }

    /// Fischer with a cap on the main time and a per-move maximum.
    pub fn fischer_capped_time(
        main_time: f64,
        increment: f64,
        main_time_limit: f64,
        max_time_per_move: f64,
    ) -> Result<Self, StringError> {
        if main_time_limit < main_time {
            return Err(StringError {
                message: "TimeControls: mainTimeLimit is smaller than mainTime".to_string(),
            });
        }
        let mut tc = Self::new();
        tc.original_main_time = main_time;
        tc.increment = increment;
        tc.main_time_limit = main_time_limit;
        tc.max_time_per_move = max_time_per_move;
        tc.main_time_left = main_time;
        Ok(tc)
    }

    /// Byo-yomi or Canadian time.
    pub fn canadian_or_byo_yomi_time(
        main_time: f64,
        per_period_time: f64,
        num_periods: i32,
        num_stones_per_period: i32,
    ) -> Self {
        let mut tc = Self::new();
        tc.original_main_time = main_time;
        tc.original_num_periods = num_periods;
        tc.num_stones_per_period = num_stones_per_period;
        tc.per_period_time = per_period_time;
        tc.main_time_left = main_time;
        tc.num_periods_left_including_current = num_periods;
        tc
    }

    /// Returns true if this time control is effectively unlimited.
    pub fn is_effectively_unlimited_time(&self) -> bool {
        (self.main_time_left > Self::UNLIMITED_TIME_THRESHOLD
            || (self.in_overtime && self.time_left_in_period > Self::UNLIMITED_TIME_THRESHOLD))
            && self.max_time_per_move > Self::UNLIMITED_TIME_THRESHOLD
    }

    /// Compute recommended time limits for the current position.
    ///
    /// Returns `(min_time, recommended_time, max_time)`.
    pub fn get_time(
        &self,
        board: &Board,
        _hist: &BoardHistory,
        lag_buffer: f64,
    ) -> Result<(f64, f64, f64), StringError> {
        let board_area = board.x_size * board.y_size;
        let num_stones_on_board = board.num_stones_on_board();

        let typical_game_length_absolute = 0.95 * board_area as f64 + 20.0;
        let typical_game_length_increment = 0.75 * board_area as f64 + 15.0;
        let typical_game_length_byo_yomi = 0.50 * board_area as f64 + 10.0;

        let min_approx_turns_left_absolute = 0.15 * board_area as f64 + 30.0;
        let min_approx_turns_left_increment = 0.10 * board_area as f64 + 20.0;
        let min_approx_turns_left_byo_yomi = 0.02 * board_area as f64 + 4.0;

        let approx_turns_left_absolute = (typical_game_length_absolute
            - num_stones_on_board as f64)
            .max(min_approx_turns_left_absolute)
            * 0.5;
        let approx_turns_left_increment = (typical_game_length_increment
            - num_stones_on_board as f64)
            .max(min_approx_turns_left_increment)
            * 0.5;
        let approx_turns_left_byo_yomi = (typical_game_length_byo_yomi
            - num_stones_on_board as f64)
            .max(min_approx_turns_left_byo_yomi)
            * 0.5;

        let divide_time_evenly_for_game =
            |time: f64, is_increment_or_abs: bool, is_byo_yomi: bool| {
                let main_time_to_use_if_absolute = time / approx_turns_left_absolute;

                if is_increment_or_abs {
                    if time <= 0.0 {
                        time
                    } else {
                        let mut main_time_to_use = time / approx_turns_left_increment;
                        main_time_to_use = main_time_to_use
                            .min(main_time_to_use_if_absolute + 2.0 * self.increment);
                        main_time_to_use
                    }
                } else if is_byo_yomi {
                    if self.per_period_time <= 0.0 || self.num_stones_per_period <= 0 {
                        main_time_to_use_if_absolute
                    } else {
                        let byo_yomi_time_per_move =
                            self.per_period_time / self.num_stones_per_period as f64;
                        let theoretical_optimal_turns_to_spend_our_time =
                            (time / byo_yomi_time_per_move) * std::f64::consts::E.recip();
                        let mut approx_turns_left_to_use =
                            theoretical_optimal_turns_to_spend_our_time;

                        if approx_turns_left_byo_yomi > theoretical_optimal_turns_to_spend_our_time
                        {
                            approx_turns_left_to_use = approx_turns_left_byo_yomi
                                .min(theoretical_optimal_turns_to_spend_our_time * 1.75);
                        }
                        if approx_turns_left_to_use > approx_turns_left_absolute {
                            approx_turns_left_to_use = approx_turns_left_absolute;
                        }
                        if approx_turns_left_to_use < 1.0 {
                            approx_turns_left_to_use = 1.0;
                        }

                        let mut main_time_to_use = time / approx_turns_left_to_use;
                        main_time_to_use = main_time_to_use
                            .min(main_time_to_use_if_absolute + 3.0 * byo_yomi_time_per_move);
                        if main_time_to_use < byo_yomi_time_per_move {
                            main_time_to_use = byo_yomi_time_per_move;
                        }
                        if main_time_to_use < byo_yomi_time_per_move * 1.5
                            && time < byo_yomi_time_per_move * 1.5
                        {
                            main_time_to_use = time + byo_yomi_time_per_move;
                        }
                        main_time_to_use
                    }
                } else {
                    main_time_to_use_if_absolute
                }
            };

        let mut min_time;
        let mut recommended_time;
        let mut max_time;
        let mut lag_buffer_to_use = lag_buffer;

        if self.increment > 0.0 || self.num_periods_left_including_current <= 0 {
            if self.in_overtime {
                return Err(StringError {
                    message: "TimeControls: inOvertime with Fischer or absolute time, inconsistent time control?".to_string(),
                });
            }
            if self.num_periods_left_including_current != 0 {
                return Err(StringError {
                    message: "TimeControls: numPeriodsLeftIncludingCurrent != 0 with Fischer or absolute time, inconsistent time control?".to_string(),
                });
            }
            if self.main_time_limit < self.original_main_time {
                return Err(StringError {
                    message: "TimeControls: mainTimeLimit is smaller than original mainTime"
                        .to_string(),
                });
            }

            if self.main_time_left <= self.increment {
                min_time = (self.main_time_left * 0.5)
                    .max(0.0)
                    .min((self.main_time_left + self.increment - self.main_time_limit).max(0.0));
                recommended_time = apply_lag_buffer(self.main_time_left, lag_buffer_to_use);
                max_time = self.main_time_left;
            } else {
                let excess_main_time =
                    apply_lag_buffer(self.main_time_left - self.increment, lag_buffer_to_use);
                min_time = (self.main_time_left * 0.5)
                    .max(0.0)
                    .min((self.main_time_left + self.increment - self.main_time_limit).max(0.0));
                recommended_time =
                    self.increment + divide_time_evenly_for_game(excess_main_time, true, false);
                max_time = self
                    .main_time_left
                    .min(self.increment + excess_main_time / 5.0);
            }
        } else {
            if self.main_time_limit < Self::UNLIMITED_TIME_THRESHOLD {
                return Err(StringError {
                    message: "TimeControls: mainTimeLimit is used with byo-yomiish periods, inconsistent time control?".to_string(),
                });
            }
            if self.num_stones_per_period <= 0 {
                return Err(StringError {
                    message: "TimeControls: numStonesPerPeriod <= 0 with byo-yomiish periods, inconsistent time control?".to_string(),
                });
            }
            if !self.in_overtime
                && self.num_periods_left_including_current != self.original_num_periods
            {
                return Err(StringError {
                    message: "TimeControls: not in overtime, but numPeriodsLeftIncludingCurrent != originalNumPeriods".to_string(),
                });
            }
            if self.in_overtime && self.num_stones_left_in_period < 1 {
                return Err(StringError {
                    message: "TimeControls: numStonesLeftInPeriod < 1 while in overtime, inconsistent time control?".to_string(),
                });
            }

            let mut effective_main_time_left = self.main_time_left;
            let mut effectively_in_overtime = self.in_overtime;
            let mut effective_num_periods_left_including_current =
                self.num_periods_left_including_current;
            let mut effective_time_left_in_period = self.time_left_in_period;
            let mut effective_num_stones_left_in_period = self.num_stones_left_in_period;

            if effective_main_time_left < 0.0 && !effectively_in_overtime {
                effectively_in_overtime = true;
                effective_time_left_in_period = effective_main_time_left + self.per_period_time;
                effective_num_stones_left_in_period = self.num_stones_per_period;
            }
            if effectively_in_overtime {
                while effective_time_left_in_period < 0.0
                    && effective_num_periods_left_including_current > 1
                {
                    effective_num_periods_left_including_current -= 1;
                    effective_time_left_in_period += self.per_period_time;
                }
            }

            const NUM_RESERVED_PERIODS: i32 = 5;
            if effective_num_periods_left_including_current > NUM_RESERVED_PERIODS {
                effectively_in_overtime = false;
                if !self.in_overtime {
                    effective_main_time_left += self.per_period_time
                        * (effective_num_periods_left_including_current - NUM_RESERVED_PERIODS)
                            as f64;
                } else {
                    effective_main_time_left += effective_time_left_in_period
                        + self.per_period_time
                            * (effective_num_periods_left_including_current
                                - NUM_RESERVED_PERIODS
                                - 1) as f64;
                }
            }

            if !effectively_in_overtime {
                let large_byo_yomi_time_per_move =
                    self.per_period_time / (0.75 * self.num_stones_per_period as f64 + 0.25);

                min_time = 0.0;
                recommended_time =
                    divide_time_evenly_for_game(effective_main_time_left, false, true);
                max_time = large_byo_yomi_time_per_move
                    + (large_byo_yomi_time_per_move * 1.75)
                        .min(effective_main_time_left)
                        .max(effective_main_time_left / 5.0);

                if max_time > effective_main_time_left
                    && max_time < effective_main_time_left + large_byo_yomi_time_per_move
                {
                    max_time = effective_main_time_left + large_byo_yomi_time_per_move;
                }

                if max_time > effective_main_time_left
                    && effective_num_periods_left_including_current <= 1
                    && self.num_stones_per_period <= 1
                {
                    lag_buffer_to_use *= 2.0;
                }
            } else {
                if effective_num_stones_left_in_period < 1 {
                    return Err(StringError {
                        message: "TimeControls: effectiveNumStonesLeftInPeriod < 1 while in overtime, inconsistent time control?".to_string(),
                    });
                }

                if effective_num_periods_left_including_current > 1
                    && apply_lag_buffer(effective_time_left_in_period, lag_buffer_to_use)
                        < apply_lag_buffer(0.5 * self.per_period_time, lag_buffer_to_use)
                            * (effective_num_periods_left_including_current - 1) as f64
                            / (NUM_RESERVED_PERIODS - 1) as f64
                {
                    effective_num_periods_left_including_current -= 1;
                    effective_time_left_in_period += self.per_period_time;
                }

                min_time = if effective_num_stones_left_in_period <= 1 {
                    effective_time_left_in_period
                } else {
                    0.0
                };
                recommended_time =
                    effective_time_left_in_period / effective_num_stones_left_in_period as f64;
                max_time = effective_time_left_in_period
                    / (0.75 * effective_num_stones_left_in_period as f64 + 0.25);

                if effective_num_periods_left_including_current <= 1
                    && effective_num_stones_left_in_period <= 1
                {
                    lag_buffer_to_use *= 2.0;
                }
            }
        }

        max_time = max_time.min(self.max_time_per_move);

        min_time = apply_lag_buffer(min_time, lag_buffer_to_use);
        recommended_time = apply_lag_buffer(recommended_time, lag_buffer_to_use);
        max_time = apply_lag_buffer(max_time, lag_buffer_to_use);

        if max_time < 0.0 {
            max_time = 0.0;
        }
        if min_time < 0.0 {
            min_time = 0.0;
        }
        if recommended_time < 0.0 {
            recommended_time = 0.0;
        }
        if min_time > max_time {
            min_time = max_time;
        }
        if recommended_time > max_time {
            recommended_time = max_time;
        }

        Ok((min_time, recommended_time, max_time))
    }

    /// If thinking for `time_limit` would waste time (e.g. a byo-yomi period), bump it up.
    pub fn round_up_time_limit_if_needed(
        &self,
        lag_buffer: f64,
        time_used: f64,
        mut time_limit: f64,
    ) -> f64 {
        if self.increment > 0.0 || self.num_periods_left_including_current <= 0 {
            return time_limit;
        }

        let mut effective_main_time_left = self.main_time_left;
        let mut effectively_in_overtime = self.in_overtime;
        let mut effective_num_periods_left_including_current =
            self.num_periods_left_including_current;
        let mut effective_time_left_in_period = self.time_left_in_period;
        let mut effective_num_stones_left_in_period = self.num_stones_left_in_period as f64;

        if !effectively_in_overtime {
            effective_main_time_left -= time_used;
        } else {
            effective_time_left_in_period -= time_used;
        }

        if effective_main_time_left < 0.0 && !effectively_in_overtime {
            effectively_in_overtime = true;
            effective_time_left_in_period = effective_main_time_left + self.per_period_time;
            effective_num_stones_left_in_period = self.num_stones_per_period as f64;
        }

        if effectively_in_overtime {
            while effective_time_left_in_period < 0.0
                && effective_num_periods_left_including_current > 1
            {
                effective_num_periods_left_including_current -= 1;
                effective_time_left_in_period += self.per_period_time;
            }
        }

        let rounded_up_time_usage;
        let byo_yomi_time_per_move = self.per_period_time / self.num_stones_per_period as f64;
        let byo_yomi_time_per_move_buffered = apply_lag_buffer(
            self.per_period_time / self.num_stones_per_period as f64,
            lag_buffer,
        );

        let bit_of_time = lag_buffer
            .max(byo_yomi_time_per_move_buffered * 0.01)
            .min(byo_yomi_time_per_move_buffered);

        if !effectively_in_overtime {
            if effective_main_time_left < byo_yomi_time_per_move * 0.5 {
                if self.num_stones_per_period <= 1 {
                    rounded_up_time_usage =
                        time_used + effective_main_time_left + byo_yomi_time_per_move_buffered;
                } else {
                    rounded_up_time_usage = time_used + effective_main_time_left + bit_of_time;
                }
            } else {
                return time_limit;
            }
        } else {
            if effective_time_left_in_period <= 0.0 {
                return time_limit;
            }
            if effective_num_stones_left_in_period > 1.0 {
                if !self.in_overtime
                    && (self.per_period_time - effective_time_left_in_period) < bit_of_time
                {
                    rounded_up_time_usage = time_used + bit_of_time
                        - (self.per_period_time - effective_time_left_in_period);
                } else {
                    return time_limit;
                }
            } else {
                rounded_up_time_usage =
                    apply_lag_buffer(time_used + effective_time_left_in_period, lag_buffer);
            }
        }

        if rounded_up_time_usage < time_used {
            return time_limit;
        }
        if time_limit < rounded_up_time_usage {
            time_limit = rounded_up_time_usage;
        }
        time_limit
    }

    /// Debug string without a board-specific recommendation.
    pub fn to_debug_string(&self) -> String {
        let mut parts = Vec::new();
        parts.push(format!("originalMainTime {}", self.original_main_time));
        if self.increment != 0.0 {
            parts.push(format!("increment {}", self.increment));
        }
        if self.main_time_limit < Self::UNLIMITED_TIME_THRESHOLD {
            parts.push(format!("mainTimeLimit {}", self.main_time_limit));
        }
        if self.max_time_per_move < Self::UNLIMITED_TIME_THRESHOLD {
            parts.push(format!("maxTimePerMove {}", self.max_time_per_move));
        }
        if self.original_num_periods != 0 {
            parts.push(format!("originalNumPeriods {}", self.original_num_periods));
        }
        if self.num_stones_per_period != 0 {
            parts.push(format!("numStonesPerPeriod {}", self.num_stones_per_period));
        }
        if self.per_period_time != 0.0 {
            parts.push(format!("perPeriodTime {}", self.per_period_time));
        }
        parts.push(format!("mainTimeLeft {}", self.main_time_left));
        parts.push(format!("inOvertime {}", self.in_overtime));
        if self.num_periods_left_including_current != 0 {
            parts.push(format!(
                "numPeriodsLeftIncludingCurrent {}",
                self.num_periods_left_including_current
            ));
        }
        if self.num_stones_left_in_period != 0 {
            parts.push(format!(
                "numStonesLeftInPeriod {}",
                self.num_stones_left_in_period
            ));
        }
        if self.time_left_in_period != 0.0 {
            parts.push(format!("timeLeftInPeriod {}", self.time_left_in_period));
        }
        parts.join("")
    }

    /// Debug string including recommended time limits for the current position.
    pub fn to_debug_string_with_board(
        &self,
        board: &Board,
        hist: &BoardHistory,
        lag_buffer: f64,
    ) -> Result<String, StringError> {
        let mut s = self.to_debug_string();
        let (min_time, recommended_time, max_time) = self.get_time(board, hist, lag_buffer)?;
        s.push_str(&format!(
            " minRecMax {} {} {}",
            min_time, recommended_time, max_time
        ));
        let rrec0 = self.round_up_time_limit_if_needed(lag_buffer, 0.0, recommended_time);
        let rrec_limit = self.round_up_time_limit_if_needed(
            lag_buffer,
            recommended_time - 0.000_001,
            recommended_time,
        );
        let rrec_limit2 =
            self.round_up_time_limit_if_needed(lag_buffer, rrec_limit - 0.000_001, rrec_limit);
        s.push_str(&format!(
            " rrec0 {} rreclimit {} rreclimit2 {}",
            rrec0, rrec_limit, rrec_limit2
        ));
        Ok(s)
    }
}

impl Default for TimeControls {
    fn default() -> Self {
        Self::new()
    }
}

fn apply_lag_buffer(time: f64, lag_buffer: f64) -> f64 {
    if time < 0.0 {
        time
    } else if time < 2.0 * lag_buffer {
        time * 0.5
    } else {
        time - lag_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kata_game::board::P_BLACK;
    use kata_game::rules::Rules;

    fn empty_board() -> Board {
        Board::new(19, 19)
    }

    fn empty_history(board: &Board) -> BoardHistory {
        BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0)
    }

    #[test]
    fn test_default_is_unlimited() {
        let tc = TimeControls::new();
        assert!(tc.is_effectively_unlimited_time());
    }

    #[test]
    fn test_absolute_time() {
        let tc = TimeControls::absolute_time(60.0);
        assert!(!tc.is_effectively_unlimited_time());
        assert_eq!(tc.main_time_left, 60.0);
    }

    #[test]
    fn test_fischer_capped_rejects_invalid_limit() {
        assert!(TimeControls::fischer_capped_time(60.0, 1.0, 30.0, 10.0).is_err());
    }

    #[test]
    fn test_get_time_absolute() {
        let tc = TimeControls::absolute_time(60.0);
        let board = empty_board();
        let hist = empty_history(&board);
        let (min, rec, max) = tc.get_time(&board, &hist, 0.5).unwrap();
        assert!(min >= 0.0 && rec >= min && max >= rec);
    }

    #[test]
    fn test_get_time_byo_yomi() {
        let tc = TimeControls::canadian_or_byo_yomi_time(300.0, 30.0, 5, 1);
        let board = empty_board();
        let hist = empty_history(&board);
        let (min, rec, max) = tc.get_time(&board, &hist, 0.5).unwrap();
        assert!(min >= 0.0 && rec >= min && max >= rec);
    }

    #[test]
    fn test_round_up_time_limit_byo_yomi() {
        let tc = TimeControls::canadian_or_byo_yomi_time(10.0, 30.0, 3, 1);
        let limit = tc.round_up_time_limit_if_needed(0.5, 0.0, 5.0);
        assert!(limit >= 5.0);
    }

    #[test]
    fn test_debug_string_with_board() {
        let tc = TimeControls::absolute_time(60.0);
        let board = empty_board();
        let hist = empty_history(&board);
        let s = tc.to_debug_string_with_board(&board, &hist, 0.5).unwrap();
        assert!(s.contains("minRecMax"));
    }
}

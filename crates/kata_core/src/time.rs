//! Date/time helpers and `SimpleDate`.
//!
//! Corresponds to `cpp/core/datetime.h` and `cpp/core/datetime.cpp`.

pub mod timer;

use std::cmp::Ordering;
use std::fmt;
use std::io::{self, Write};

use chrono::{Datelike, Local, TimeZone, Timelike, Utc};

use crate::global;

/// Default log timestamp format: `YYYY-MM-DD HH:MM:SS+ZZZZ: `.
pub const TIME_FORMAT: &str = "%F %T%z: ";

/// Wall-clock time components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tm {
    pub year: i32,
    pub month: i32,
    pub day: i32,
    pub hour: i32,
    pub minute: i32,
    pub second: i32,
}

/// Current time as seconds since the Unix epoch.
pub fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Convert seconds since the Unix epoch to UTC components.
pub fn gm_time(seconds: i64) -> Tm {
    let dt = Utc
        .timestamp_opt(seconds, 0)
        .single()
        .unwrap_or_else(Utc::now);
    Tm {
        year: dt.year(),
        month: dt.month() as i32,
        day: dt.day() as i32,
        hour: dt.hour() as i32,
        minute: dt.minute() as i32,
        second: dt.second() as i32,
    }
}

/// Convert seconds since the Unix epoch to local-time components.
pub fn local_time(seconds: i64) -> Tm {
    let dt = Local
        .timestamp_opt(seconds, 0)
        .single()
        .unwrap_or_else(Local::now);
    Tm {
        year: dt.year(),
        month: dt.month() as i32,
        day: dt.day() as i32,
        hour: dt.hour() as i32,
        minute: dt.minute() as i32,
        second: dt.second() as i32,
    }
}

/// Format a UTC timestamp with the given `strftime`-style format string.
pub fn write_time_to_stream<W: Write>(out: &mut W, fmt: &str, seconds: i64) -> io::Result<()> {
    let dt = Local
        .timestamp_opt(seconds, 0)
        .single()
        .unwrap_or_else(Local::now);
    write!(out, "{}", dt.format(fmt))
}

/// Return the current UTC date as `YYYY-MM-DD`.
pub fn get_date_string() -> String {
    let tm = gm_time(now());
    format!("{}-{}-{}", tm.year, tm.month, tm.day)
}

/// Return the current local date/time as `YYYYMMDD-HHMMSS`.
pub fn get_compact_date_time_string() -> String {
    let tm = local_time(now());
    format!(
        "{:04}{:02}{:02}-{:02}{:02}{:02}",
        tm.year, tm.month, tm.day, tm.hour, tm.minute, tm.second
    )
}

// ---------------------------------------------------------------------------
// SimpleDate
// ---------------------------------------------------------------------------

/// A proleptic Gregorian calendar date.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SimpleDate {
    pub year: i32,
    pub month: i32,
    pub day: i32,
}

impl Default for SimpleDate {
    fn default() -> Self {
        Self {
            year: 1970,
            month: 1,
            day: 1,
        }
    }
}

impl SimpleDate {
    pub fn new(year: i32, month: i32, day: i32) -> Result<Self, global::StringError> {
        let d = Self { year, month, day };
        if !d.is_valid() {
            Err(global::StringError::new(format!(
                "SimpleDate: Invalid year month day: {},{},{}",
                year, month, day
            )))
        } else {
            Ok(d)
        }
    }

    fn is_valid(&self) -> bool {
        self.month >= 1
            && self.month <= 12
            && self.day >= 1
            && self.day <= days_in_month(self.year, self.month)
    }

    /// Parse an ISO-8601 date string (`YYYY-MM-DD`).
    pub fn from_string(s: &str) -> Result<Self, global::StringError> {
        if s.len() != 10 || s.as_bytes()[4] != b'-' || s.as_bytes()[7] != b'-' {
            return Err(global::StringError::new(format!(
                "SimpleDate: Unable to parse as ISO8601 date: {}",
                s
            )));
        }
        for (i, c) in s.bytes().enumerate() {
            if i != 4 && i != 7 && !c.is_ascii_digit() {
                return Err(global::StringError::new(format!(
                    "SimpleDate: Unable to parse as ISO8601 date: {}",
                    s
                )));
            }
        }
        let year = global::parse_digits(&s[0..4]).map_err(|_| {
            global::StringError::new(format!(
                "SimpleDate: Unable to parse as ISO8601 date: {}",
                s
            ))
        })?;
        let month = global::parse_digits(&s[5..7]).map_err(|_| {
            global::StringError::new(format!(
                "SimpleDate: Unable to parse as ISO8601 date: {}",
                s
            ))
        })?;
        let day = global::parse_digits(&s[8..10]).map_err(|_| {
            global::StringError::new(format!(
                "SimpleDate: Unable to parse as ISO8601 date: {}",
                s
            ))
        })?;
        Self::new(year, month, day)
    }

    /// Format as `YYYY-MM-DD`.
    #[allow(clippy::inherent_to_string_shadow_display)]
    pub fn to_string(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }

    /// Returns `true` if the date falls in a leap year.
    pub fn is_leap_year(&self) -> bool {
        is_leap_year(self.year)
    }

    /// Number of days into the year (Jan 1 is 0).
    pub fn num_days_into_year(&self) -> i32 {
        let cum = if self.is_leap_year() {
            CUMULATIVE_DAYS_UNTIL_MONTH_LEAP
        } else {
            CUMULATIVE_DAYS_UNTIL_MONTH
        };
        cum[self.month as usize] + (self.day - 1)
    }

    /// Number of days from `other` to `self` (`self - other`).
    pub fn num_days_after(&self, other: &Self) -> i32 {
        let flip = matches!(self.cmp(other), Ordering::Less);
        let d_later = if flip { other } else { self };
        let d_early = if flip { self } else { other };

        let mut day_count = 365 * (d_later.year - d_early.year);
        day_count += num_leap_years_up_to_and_including(d_later.year - 1)
            - num_leap_years_up_to_and_including(d_early.year - 1);
        day_count -= d_early.num_days_into_year();
        day_count += d_later.num_days_into_year();

        if flip { -day_count } else { day_count }
    }

    /// Add `n` days in place.
    pub fn add_days(&mut self, n: i32) {
        let mut n = n + self.num_days_into_year();
        self.month = 1;
        self.day = 1;

        if n < 0 {
            let approx_years_to_sub = 1 + (-n) / 365;
            let mut date2 = *self;
            date2.year -= approx_years_to_sub;
            n += self.num_days_after(&date2);
            self.year -= approx_years_to_sub;
            debug_assert!(n >= 0);
        }

        while n >= 366 {
            let approx_years_to_add = n / 366;
            let mut date2 = *self;
            date2.year += approx_years_to_add;
            n -= date2.num_days_after(self);
            self.year += approx_years_to_add;
            debug_assert!(n >= 0);
        }

        if n == 365 {
            if self.is_leap_year() {
                self.month = 12;
                self.day = 31;
                return;
            }
            self.year += 1;
            return;
        }

        debug_assert!((0..=364).contains(&n));
        let cum = if self.is_leap_year() {
            CUMULATIVE_DAYS_UNTIL_MONTH_LEAP
        } else {
            CUMULATIVE_DAYS_UNTIL_MONTH
        };
        while n >= cum[(self.month + 1) as usize] {
            self.month += 1;
        }
        self.day = 1 + n - cum[self.month as usize];
    }

    /// Subtract `n` days in place.
    pub fn sub_days(&mut self, n: i32) {
        self.add_days(-n);
    }
}

impl fmt::Display for SimpleDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

impl PartialOrd for SimpleDate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SimpleDate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.year
            .cmp(&other.year)
            .then_with(|| self.month.cmp(&other.month))
            .then_with(|| self.day.cmp(&other.day))
    }
}

impl std::ops::Add<i32> for SimpleDate {
    type Output = Self;
    fn add(mut self, rhs: i32) -> Self {
        self.add_days(rhs);
        self
    }
}

impl std::ops::Add<SimpleDate> for i32 {
    type Output = SimpleDate;
    fn add(self, mut rhs: SimpleDate) -> SimpleDate {
        rhs.add_days(self);
        rhs
    }
}

impl std::ops::Sub<i32> for SimpleDate {
    type Output = Self;
    fn sub(mut self, rhs: i32) -> Self {
        self.sub_days(rhs);
        self
    }
}

impl std::ops::AddAssign<i32> for SimpleDate {
    fn add_assign(&mut self, rhs: i32) {
        self.add_days(rhs);
    }
}

impl std::ops::SubAssign<i32> for SimpleDate {
    fn sub_assign(&mut self, rhs: i32) {
        self.sub_days(rhs);
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn days_in_month(year: i32, month: i32) -> i32 {
    match month {
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

fn div_floor(n: i32, d: i32) -> i32 {
    if n >= 0 { n / d } else { -1 - (-1 - n) / d }
}

fn num_leap_years_up_to_and_including(year: i32) -> i32 {
    div_floor(year, 4) - div_floor(year, 100) + div_floor(year, 400)
}

const CUMULATIVE_DAYS_UNTIL_MONTH: [i32; 14] = [
    0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365,
];
const CUMULATIVE_DAYS_UNTIL_MONTH_LEAP: [i32; 14] = [
    0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366,
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rand;

    #[test]
    fn test_simple_date_comparisons() {
        let a = SimpleDate::new(1999, 1, 11).unwrap();
        assert_eq!(a, SimpleDate::new(1999, 1, 11).unwrap());
        assert_ne!(a, SimpleDate::new(1999, 2, 11).unwrap());
        assert_ne!(a, SimpleDate::new(1999, 1, 21).unwrap());
        assert_ne!(a, SimpleDate::new(1998, 1, 11).unwrap());

        let b = SimpleDate::new(1965, 6, 21).unwrap();
        assert!(b > SimpleDate::new(1945, 8, 25).unwrap());
        assert!(b > SimpleDate::new(1965, 4, 25).unwrap());
        assert!(b > SimpleDate::new(1965, 6, 20).unwrap());
        assert!(b >= SimpleDate::new(1965, 6, 21).unwrap());

        assert!(b < SimpleDate::new(1985, 4, 20).unwrap());
        assert!(b < SimpleDate::new(1965, 8, 20).unwrap());
        assert!(b < SimpleDate::new(1965, 6, 22).unwrap());
    }

    #[test]
    fn test_simple_date_parse_and_display() {
        assert_eq!(
            SimpleDate::from_string("1234-05-07").unwrap(),
            SimpleDate::new(1234, 5, 7).unwrap()
        );
        assert_eq!(
            SimpleDate::default(),
            SimpleDate::from_string("1970-01-01").unwrap()
        );
        assert_eq!(
            SimpleDate::new(475, 1, 31).unwrap().to_string(),
            "0475-01-31"
        );
        assert_eq!(
            SimpleDate::new(1234, 5, 7).unwrap().to_string(),
            "1234-05-07"
        );
    }

    #[test]
    fn test_simple_date_add() {
        assert_eq!(
            SimpleDate::default() + 365,
            SimpleDate::from_string("1971-01-01").unwrap()
        );
        assert_eq!(
            SimpleDate::default() + 365 * 2,
            SimpleDate::from_string("1972-01-01").unwrap()
        );
        assert_eq!(
            SimpleDate::default() + 365 * 3,
            SimpleDate::from_string("1972-12-31").unwrap()
        );

        assert_eq!(
            SimpleDate::from_string("1895-01-01").unwrap() + 365,
            SimpleDate::from_string("1896-01-01").unwrap()
        );
        assert_eq!(
            SimpleDate::from_string("1895-01-01").unwrap() + 365 * 2,
            SimpleDate::from_string("1896-12-31").unwrap()
        );
        assert_eq!(
            SimpleDate::from_string("1899-01-01").unwrap() + 365 * 3,
            SimpleDate::from_string("1902-01-01").unwrap()
        );
        assert_eq!(
            SimpleDate::from_string("1999-01-01").unwrap() + 365 * 3,
            SimpleDate::from_string("2001-12-31").unwrap()
        );
    }

    #[test]
    fn test_simple_date_random_properties() {
        let mut rand = Rand::new_from_seed("SimpleDate tests");
        for _ in 0..100_000 {
            let d1 = rand.next_i32_range(-10_000_000, 10_000_000);
            let mut d2 = rand.next_i32_range(-20, 20);
            if rand.next_bool(0.1) {
                d2 *= 100;
            }
            let mut d3 = rand.next_i32_range(-20, 20);
            d3 = d3 * d3 * if rand.next_bool(0.5) { -1 } else { 1 };
            if rand.next_bool(0.1) {
                d3 *= 37;
            }

            let mut date1 = SimpleDate::default();
            date1 += d1;
            assert_eq!(SimpleDate::default() + d1, date1);
            assert_eq!(d1 + SimpleDate::default(), date1);
            assert_eq!(SimpleDate::default() - (-d1), date1);
            assert_eq!(date1 - d1, SimpleDate::default());
            assert_eq!(date1.num_days_after(&SimpleDate::default()), d1);

            let mut date2 = date1;
            date2 += d2;
            assert_eq!(date1 + d2, date2);
            assert_eq!(d2 + date1, date2);
            assert_eq!(date1 - (-d2), date2);
            assert_eq!(date2 - d2, date1);
            assert_eq!(date2.num_days_after(&date1), d2);

            let mut date3 = date2;
            date3 += d3;
            assert_eq!(date3.num_days_after(&date2), d3);
            assert_eq!((date1 + d2) + d3, date3);
            assert_eq!((date1 + d3) + d2, date3);
            assert_eq!(date1 + (d2 + d3), date3);

            let roundtrip = (((((SimpleDate::default() + d1) + d2) - d1) + d3) - d2) - d3;
            assert_eq!(roundtrip, SimpleDate::default());
        }
    }

    #[test]
    fn test_counting_forward_dates() {
        let mut out = String::new();
        let mut i = 0;
        while i < 370 {
            if i == 80 {
                i = 340;
            }
            out.push_str(&format!(
                "{} {}\n",
                (SimpleDate::from_string("2011-12-20").unwrap() + i).to_string(),
                (SimpleDate::from_string("2012-12-20").unwrap() + i).to_string()
            ));
            i += 1;
        }

        let expected = r"2011-12-20 2012-12-20
2011-12-21 2012-12-21
2011-12-22 2012-12-22
2011-12-23 2012-12-23
2011-12-24 2012-12-24
2011-12-25 2012-12-25
2011-12-26 2012-12-26
2011-12-27 2012-12-27
2011-12-28 2012-12-28
2011-12-29 2012-12-29
2011-12-30 2012-12-30
2011-12-31 2012-12-31
2012-01-01 2013-01-01
2012-01-02 2013-01-02
2012-01-03 2013-01-03
2012-01-04 2013-01-04
2012-01-05 2013-01-05
2012-01-06 2013-01-06
2012-01-07 2013-01-07
2012-01-08 2013-01-08
2012-01-09 2013-01-09
2012-01-10 2013-01-10
2012-01-11 2013-01-11
2012-01-12 2013-01-12
2012-01-13 2013-01-13
2012-01-14 2013-01-14
2012-01-15 2013-01-15
2012-01-16 2013-01-16
2012-01-17 2013-01-17
2012-01-18 2013-01-18
2012-01-19 2013-01-19
2012-01-20 2013-01-20
2012-01-21 2013-01-21
2012-01-22 2013-01-22
2012-01-23 2013-01-23
2012-01-24 2013-01-24
2012-01-25 2013-01-25
2012-01-26 2013-01-26
2012-01-27 2013-01-27
2012-01-28 2013-01-28
2012-01-29 2013-01-29
2012-01-30 2013-01-30
2012-01-31 2013-01-31
2012-02-01 2013-02-01
2012-02-02 2013-02-02
2012-02-03 2013-02-03
2012-02-04 2013-02-04
2012-02-05 2013-02-05
2012-02-06 2013-02-06
2012-02-07 2013-02-07
2012-02-08 2013-02-08
2012-02-09 2013-02-09
2012-02-10 2013-02-10
2012-02-11 2013-02-11
2012-02-12 2013-02-12
2012-02-13 2013-02-13
2012-02-14 2013-02-14
2012-02-15 2013-02-15
2012-02-16 2013-02-16
2012-02-17 2013-02-17
2012-02-18 2013-02-18
2012-02-19 2013-02-19
2012-02-20 2013-02-20
2012-02-21 2013-02-21
2012-02-22 2013-02-22
2012-02-23 2013-02-23
2012-02-24 2013-02-24
2012-02-25 2013-02-25
2012-02-26 2013-02-26
2012-02-27 2013-02-27
2012-02-28 2013-02-28
2012-02-29 2013-03-01
2012-03-01 2013-03-02
2012-03-02 2013-03-03
2012-03-03 2013-03-04
2012-03-04 2013-03-05
2012-03-05 2013-03-06
2012-03-06 2013-03-07
2012-03-07 2013-03-08
2012-03-08 2013-03-09
2012-11-24 2013-11-25
2012-11-25 2013-11-26
2012-11-26 2013-11-27
2012-11-27 2013-11-28
2012-11-28 2013-11-29
2012-11-29 2013-11-30
2012-11-30 2013-12-01
2012-12-01 2013-12-02
2012-12-02 2013-12-03
2012-12-03 2013-12-04
2012-12-04 2013-12-05
2012-12-05 2013-12-06
2012-12-06 2013-12-07
2012-12-07 2013-12-08
2012-12-08 2013-12-09
2012-12-09 2013-12-10
2012-12-10 2013-12-11
2012-12-11 2013-12-12
2012-12-12 2013-12-13
2012-12-13 2013-12-14
2012-12-14 2013-12-15
2012-12-15 2013-12-16
2012-12-16 2013-12-17
2012-12-17 2013-12-18
2012-12-18 2013-12-19
2012-12-19 2013-12-20
2012-12-20 2013-12-21
2012-12-21 2013-12-22
2012-12-22 2013-12-23
2012-12-23 2013-12-24
";
        assert_eq!(out, expected);
    }
}

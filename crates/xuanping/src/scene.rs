//! Demo scene data, transcribed from `scripts/design/xuanping_mockup.py`.
//!
//! Everything here is hardcoded presentation data for the shell; the structs
//! are shaped so an engine-backed implementation can replace the constructors
//! without touching the painting code.

use std::time::Instant;

use eframe::egui::ColorImage;

use crate::theme::Palette;

/// Reference canvas the design is authored against.
pub const REF_W: f32 = 1440.0;
pub const REF_H: f32 = 900.0;

// ---- board geometry ----
pub const BX: f32 = 140.0;
pub const BY: f32 = 162.0;
pub const BS: f32 = 576.0;
pub const PAD: f32 = 18.0;
pub const GX: f32 = BX + PAD;
pub const GY: f32 = BY + PAD;
pub const CELL: f32 = (BS - 2.0 * PAD) / 18.0; // 30
pub const STONE_R: f32 = 13.5;
pub const CAND_R: f32 = 17.5;

/// Grid intersection (0-based column, row) -> reference pixels.
pub fn gp(c: u8, r: u8) -> (f32, f32) {
    (GX + c as f32 * CELL, GY + r as f32 * CELL)
}

// ---- views ----

/// Top-level view. The analysis overlay is a boolean inside `Play`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum View {
    /// 对局
    Play,
    /// 复盘
    Review,
    /// 设置「器」
    Settings,
}

// ---- demo position (mockup stone lists) ----
pub const DEMO_BLACK: &[(u8, u8)] = &[
    (3, 3),
    (9, 3),
    (15, 3),
    (15, 9),
    (16, 4),
    (14, 6),
    (16, 14),
    (5, 4),
    (10, 10),
    (13, 13),
    (9, 9),
    (16, 9),
];
pub const DEMO_WHITE: &[(u8, u8)] = &[
    (3, 9),
    (4, 14),
    (9, 15),
    (15, 15),
    (3, 15),
    (9, 16),
    (14, 15),
    (16, 16),
    (4, 5),
    (14, 4),
];

/// Moves covered by the baked demo history.
pub const DEMO_MOVES: usize = 45;

/// Hardcoded white reply points for the simulated engine (8 moves).
pub const REPLY_POINTS: &[(u8, u8)] = &[
    (14, 10),
    (10, 13),
    (5, 9),
    (12, 15),
    (7, 5),
    (16, 12),
    (4, 11),
    (11, 14),
];

// ---- analysis payload ----

/// One engine candidate move (board point + right-column row data).
pub struct Candidate {
    pub pos: (u8, u8),
    pub seq: String,
    pub coord: String,
    pub winrate_text: String,
    pub visits: String,
    pub ring_alpha: f32,
    pub ring_width: f32,
}

impl Candidate {
    /// Rank-styled ring (alpha/width tiers 72%/50%/34% from the mockup).
    pub fn ring_style(rank: usize) -> (f32, f32) {
        match rank {
            0 => (0.72, 1.5),
            1 => (0.50, 1.25),
            _ => (0.34, 1.0),
        }
    }
}

/// Analysis overlay data: winrate curve, candidates, ownership.
pub struct AnalysisData {
    /// Black winrate (percent) indexed by ply (`history[i]` = after move i).
    pub winrate_history: Vec<f32>,
    pub candidates: [Candidate; 3],
    pub score_lead_text: String,
    pub visits_text: String,
    /// Key move highlighted in red on the curve.
    pub key_move: usize,
    /// 19x19 ownership, positive = white; used to build the wash texture.
    pub ownership: [[f32; 19]; 19],
}

impl AnalysisData {
    pub fn demo() -> Self {
        let mut wrs = Vec::with_capacity(DEMO_MOVES + 1);
        for m in 0..=DEMO_MOVES {
            let m = m as f32;
            let mut v = if m <= 10.0 {
                50.0 - 0.2 * m
            } else if m <= 20.0 {
                48.0 - 0.2 * (m - 10.0)
            } else if m <= 30.0 {
                46.0 + 0.1 * (m - 20.0)
            } else if m <= 37.0 {
                47.0 - 0.4 * (m - 30.0)
            } else if m == 38.0 {
                61.8
            } else {
                61.8 + 0.086 * (m - 38.0)
            };
            v += 1.1 * (m * 2.3).sin() + 0.7 * (m * 0.71).sin();
            wrs.push(v);
        }
        Self {
            winrate_history: wrs,
            candidates: [
                Candidate {
                    pos: (8, 12),
                    seq: "一".into(),
                    coord: "J07".into(),
                    winrate_text: "62.4".into(),
                    visits: "2.1k".into(),
                    ring_alpha: 0.72,
                    ring_width: 1.5,
                },
                Candidate {
                    pos: (11, 8),
                    seq: "二".into(),
                    coord: "M11".into(),
                    winrate_text: "61.1".into(),
                    visits: "1.6k".into(),
                    ring_alpha: 0.50,
                    ring_width: 1.25,
                },
                Candidate {
                    pos: (6, 6),
                    seq: "三".into(),
                    coord: "G13".into(),
                    winrate_text: "60.3".into(),
                    visits: "1.1k".into(),
                    ring_alpha: 0.34,
                    ring_width: 1.0,
                },
            ],
            score_lead_text: "+3.5".into(),
            visits_text: "12.8k".into(),
            key_move: 38,
            ownership: gen_ownership(),
        }
    }

    /// Fresh engine-mode payload: empty candidate texts (rings skipped until
    /// the first analysis frame), curve seeded at 50% for plies 0..=1.
    pub fn live_empty() -> Self {
        let blank = |(alpha, width): (f32, f32)| Candidate {
            pos: (9, 9),
            seq: String::new(),
            coord: String::new(),
            winrate_text: String::new(),
            visits: String::new(),
            ring_alpha: alpha,
            ring_width: width,
        };
        Self {
            winrate_history: vec![50.0, 50.0],
            candidates: [
                blank(Candidate::ring_style(0)),
                blank(Candidate::ring_style(1)),
                blank(Candidate::ring_style(2)),
            ],
            score_lead_text: String::new(),
            visits_text: "—".into(),
            key_move: 1,
            ownership: [[0.0; 19]; 19],
        }
    }

    /// Winrate at move `m`; user-placed moves extend flat at the last value.
    pub fn winrate_at(&self, m: usize) -> f32 {
        self.winrate_history[m.min(self.winrate_history.len() - 1)]
    }

    /// Key moves for the Review scroll: swings |Δ|>8%, top `max_n` by
    /// magnitude, returned ascending by move number.
    pub fn key_moves(&self, max_n: usize) -> Vec<usize> {
        let mut swings: Vec<(usize, f32)> = (1..self.winrate_history.len())
            .map(|m| (m, (self.winrate_history[m] - self.winrate_history[m - 1]).abs()))
            .filter(|&(_, d)| d > 8.0)
            .collect();
        swings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        swings.truncate(max_n);
        let mut moves: Vec<usize> = swings.into_iter().map(|(m, _)| m).collect();
        moves.sort_unstable();
        moves
    }
}

// ---- ownership wash ----

/// (column, row, radius in px) territory clusters from the mockup.
const WHITE_CLUSTERS: &[(u8, u8, f32)] = &[
    (4, 11, 95.0),
    (5, 14, 115.0),
    (7, 16, 85.0),
    (2, 7, 70.0),
    (3, 17, 60.0),
];
const BLACK_CLUSTERS: &[(u8, u8, f32)] = &[
    (6, 2, 105.0),
    (11, 2, 110.0),
    (15, 6, 95.0),
    (16, 10, 75.0),
    (9, 1, 70.0),
];
const WHITE_AMP: f32 = 0.30;
const BLACK_AMP: f32 = 0.45;

fn add_cluster(g: &mut [[f32; 19]; 19], cc: u8, rr: u8, rad_px: f32, amp: f32) {
    // Gaussian decay in cell units; sigma ~ half the mockup radius.
    let sigma = rad_px / CELL / 2.0;
    let denom = 2.0 * sigma * sigma;
    for (r, row) in g.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            let d2 = (c as f32 - cc as f32).powi(2) + (r as f32 - rr as f32).powi(2);
            *v += amp * (-d2 / denom).exp();
        }
    }
}

fn gen_ownership() -> [[f32; 19]; 19] {
    let mut g = [[0.0f32; 19]; 19];
    for &(c, r, rad) in WHITE_CLUSTERS {
        add_cluster(&mut g, c, r, rad, WHITE_AMP);
    }
    for &(c, r, rad) in BLACK_CLUSTERS {
        add_cluster(&mut g, c, r, rad, -BLACK_AMP);
    }
    g
}

/// Build the 19x19 wash texture: white-side texels carry the palette's
/// wash-white, black-side its wash-black, alpha tiered by |value| (5 steps).
/// Bilinear magnification on the GPU turns this into the ink-wash gradient
/// (egui has no gaussian blur). Rebuilt on theme switch.
pub fn ownership_image(ownership: &[[f32; 19]; 19], palette: &Palette) -> ColorImage {
    let mut img = ColorImage::new(
        [19, 19],
        vec![eframe::egui::Color32::TRANSPARENT; 19 * 19],
    );
    for (r, row) in ownership.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            let (rgb, amp) = if v >= 0.0 {
                (palette.wash_white, WHITE_AMP)
            } else {
                (palette.wash_black, BLACK_AMP)
            };
            let tier = ((v.abs() / amp).clamp(0.0, 1.0) * 5.0).round() / 5.0;
            let a = (tier * amp * 255.0).round() as u8;
            img[(c, r)] =
                eframe::egui::Color32::from_rgba_unmultiplied(rgb.r(), rgb.g(), rgb.b(), a);
        }
    }
    img
}

// ---- shell game state ----

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StoneColor {
    Black,
    White,
}

impl StoneColor {
    pub fn other(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }
}

/// A stone placed on top of the demo position (user or simulated reply).
pub struct PlacedMove {
    pub pos: (u8, u8),
    pub color: StoneColor,
}

/// Engine-backed board state (`--model` mode): the authoritative stones come
/// verbatim from `Snap::Board` captures included; the demo position lists do
/// not apply. The engine stays white, the user black.
pub struct LiveState {
    /// 0 empty / 1 black / 2 white, row-major `[r][c]`.
    pub stones: [[u8; 19]; 19],
    pub black_to_move: bool,
    /// Plies played (0 = empty board); analysis snapshots tag off this.
    pub move_no: usize,
}

/// First empty intersection on an expanding ring spiral from the centre
/// (9,9); fallback when the hardcoded reply list is exhausted.
pub fn spiral_empty(mut occupied: impl FnMut(u8, u8) -> bool) -> Option<(u8, u8)> {
    for k in 0..=9i32 {
        for dy in -k..=k {
            for dx in -k..=k {
                if dx.abs().max(dy.abs()) != k {
                    continue;
                }
                let (c, r) = (9 + dx, 9 + dy);
                if (0..19).contains(&c) && (0..19).contains(&r) && !occupied(c as u8, r as u8) {
                    return Some((c as u8, r as u8));
                }
            }
        }
    }
    None
}

/// Shell-level game state. Demo mode (default): demo position + simulated
/// replies. Engine mode (`live = Some`): every field below is driven by the
/// engine thread's snapshots and the placement machinery is bypassed.
pub struct GameState {
    pub placed: Vec<PlacedMove>,
    /// Move number currently in view (1-based). Starts at the demo horizon.
    pub cursor: usize,
    pub show_analysis: bool,
    pub show_kifu: bool,
    /// Most recent placement, for the fade/drop/ripple animation.
    pub last_placed: Option<(u8, u8, Instant)>,
    /// Simulated engine thinking since this instant (after a user move).
    thinking_since: Option<Instant>,
    reply_idx: usize,
    next: StoneColor,
    /// Engine-driven state; `None` in demo mode.
    pub live: Option<LiveState>,
}

impl Default for GameState {
    fn default() -> Self {
        Self {
            placed: Vec::new(),
            cursor: DEMO_MOVES,
            show_analysis: true,
            show_kifu: true,
            last_placed: None,
            thinking_since: None,
            reply_idx: 0,
            next: StoneColor::Black,
            live: None,
        }
    }
}

impl GameState {
    /// A fresh engine-backed game (empty board, cursor at head).
    pub fn new_live() -> Self {
        let mut s = Self::default();
        s.live = Some(LiveState {
            stones: [[0; 19]; 19],
            black_to_move: true,
            move_no: 0,
        });
        s.cursor = 1;
        s
    }

    /// Current head ply in engine mode (>= 1 so the kifu/cursor stay valid).
    pub fn live_head(&self) -> Option<usize> {
        self.live.as_ref().map(|l| l.move_no.max(1))
    }

    /// Apply an authoritative `Snap::Board`. The view follows to the head
    /// unless the user had scrubbed back; snapshrink (新局) snaps home.
    pub fn apply_live_board(
        &mut self,
        stones: [[u8; 19]; 19],
        black_to_move: bool,
        last_move: Option<(u8, u8)>,
        move_no: usize,
        now: Instant,
    ) {
        if self.live.is_none() {
            return;
        }
        let head_before = self.total_moves();
        let at_head = self.cursor >= head_before;
        let head = move_no.max(1);
        {
            let live = self.live.as_mut().expect("checked above");
            *live = LiveState {
                stones,
                black_to_move,
                move_no,
            };
        }
        if let Some((c, r)) = last_move {
            // Placement animation rides on every newly landed stone.
            self.last_placed = Some((c, r, now));
        }
        if at_head || head < head_before {
            self.cursor = head;
        }
    }

    pub fn total_moves(&self) -> usize {
        match &self.live {
            Some(l) => l.move_no.max(1),
            None => DEMO_MOVES + self.placed.len(),
        }
    }

    pub fn step(&mut self, delta: i64) {
        let cur = self.cursor as i64 + delta;
        self.cursor = cur.clamp(1, self.total_moves() as i64) as usize;
    }

    pub fn set_cursor(&mut self, m: usize) {
        self.cursor = m.clamp(1, self.total_moves());
    }

    pub fn is_occupied(&self, c: u8, r: u8) -> bool {
        match &self.live {
            Some(l) => l.stones[r as usize][c as usize] != 0,
            None => {
                DEMO_BLACK.contains(&(c, r))
                    || DEMO_WHITE.contains(&(c, r))
                    || self.placed.iter().any(|m| m.pos == (c, r))
            }
        }
    }

    pub fn is_thinking(&self) -> bool {
        match &self.live {
            // Engine mode: analysis runs continuously once the bot is up.
            Some(_) => true,
            None => self.thinking_since.is_some(),
        }
    }

    /// User's turn (plays black): not waiting for the simulated reply.
    pub fn user_turn(&self) -> bool {
        match &self.live {
            Some(l) => {
                l.black_to_move && self.cursor == self.total_moves()
            }
            None => self.next == StoneColor::Black && !self.is_thinking(),
        }
    }

    /// Place a user stone. Returns true when the simulated engine should
    /// start thinking (fresh-head placement with reply stock, not a scrub).
    pub fn place_user(&mut self, c: u8, r: u8, now: Instant) -> bool {
        if self.is_occupied(c, r) || self.is_thinking() {
            return false;
        }
        let at_head = self.cursor == self.total_moves();
        self.placed.push(PlacedMove {
            pos: (c, r),
            color: self.next,
        });
        self.next = self.next.other();
        self.cursor = self.total_moves();
        self.last_placed = Some((c, r, now));
        at_head
    }

    /// Enter the thinking state (fast breathing, 思考 status).
    pub fn start_thinking(&mut self, now: Instant) {
        self.thinking_since = Some(now);
    }

    /// After ~0.9s of thinking, play the next white reply (hardcoded list,
    /// spiral fallback). Returns true when a stone was placed. Demo mode only.
    pub fn maybe_reply(&mut self, now: Instant) -> bool {
        if self.live.is_some() {
            return false;
        }
        let Some(since) = self.thinking_since else {
            return false;
        };
        if now.duration_since(since).as_secs_f32() < 0.9 {
            return false;
        }
        self.thinking_since = None;
        let point = self.next_reply_point();
        if let Some((c, r)) = point {
            self.placed.push(PlacedMove {
                pos: (c, r),
                color: StoneColor::White,
            });
            self.next = StoneColor::Black;
            self.cursor = self.total_moves();
            self.last_placed = Some((c, r, now));
            true
        } else {
            false
        }
    }

    fn next_reply_point(&mut self) -> Option<(u8, u8)> {
        while self.reply_idx < REPLY_POINTS.len() {
            let (c, r) = REPLY_POINTS[self.reply_idx];
            self.reply_idx += 1;
            if !self.is_occupied(c, r) {
                return Some((c, r));
            }
        }
        spiral_empty(|c, r| self.is_occupied(c, r))
    }
}

// ---- settings state (demo, local only) ----

/// 「器」 view state. Backend picks are pure local presentation state.
pub struct SettingsState {
    /// 0 = dummy, 1 = TRT, 2 = CUDA (default selected).
    pub backend: usize,
    /// 动效: reduced motion (constant breath dot, no ripple/drop).
    pub reduced_motion: bool,
}

impl Default for SettingsState {
    fn default() -> Self {
        Self {
            backend: 2,
            reduced_motion: false,
        }
    }
}

// ---- chinese numerals for the kifu strip ----

const DIGITS: [char; 10] = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];

/// 1..=361 -> 汉字数字 (e.g. 45 -> "四十五").
pub fn cn_num(n: usize) -> String {
    debug_assert!((1..=361).contains(&n));
    if n < 10 {
        return DIGITS[n].to_string();
    }
    if n < 20 {
        return if n == 10 {
            "十".into()
        } else {
            format!("十{}", DIGITS[n % 10])
        };
    }
    if n < 100 {
        let (t, u) = (n / 10, n % 10);
        return if u == 0 {
            format!("{}十", DIGITS[t])
        } else {
            format!("{}十{}", DIGITS[t], DIGITS[u])
        };
    }
    let (h, rem) = (n / 100, n % 100);
    let mut s = format!("{}百", DIGITS[h]);
    match rem {
        0 => {}
        1..=9 => {
            s.push('零');
            s.push(DIGITS[rem]);
        }
        10..=19 => {
            s.push('一');
            s.push('十');
            if rem % 10 > 0 {
                s.push(DIGITS[rem % 10]);
            }
        }
        _ => {
            s.push(DIGITS[rem / 10]);
            s.push('十');
            if rem % 10 > 0 {
                s.push(DIGITS[rem % 10]);
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cn_num_basic() {
        assert_eq!(cn_num(1), "一");
        assert_eq!(cn_num(9), "九");
        assert_eq!(cn_num(10), "十");
        assert_eq!(cn_num(15), "十五");
        assert_eq!(cn_num(20), "二十");
        assert_eq!(cn_num(42), "四十二");
        assert_eq!(cn_num(45), "四十五");
        assert_eq!(cn_num(100), "一百");
        assert_eq!(cn_num(101), "一百零一");
        assert_eq!(cn_num(110), "一百一十");
        assert_eq!(cn_num(361), "三百六十一");
    }

    #[test]
    fn ownership_peaks_at_clusters() {
        let g = gen_ownership();
        assert!(g[14][5] > 0.25); // white cluster centre
        assert!(g[2][11] < -0.40); // black cluster centre
        assert!(g[18][18].abs() < 0.01); // far corner stays neutral
    }

    #[test]
    fn key_moves_top_swings() {
        let a = AnalysisData::demo();
        // Only the move-38 jump clears the 8% bar in the demo series.
        assert_eq!(a.key_moves(3), vec![38]);
        // Synthetic series: three swings above the bar, top three kept,
        // returned ascending.
        let wr = vec![50.0, 60.0, 59.0, 40.0, 41.0, 30.0, 31.0];
        let mut custom = AnalysisData::demo();
        custom.winrate_history = wr;
        assert_eq!(custom.key_moves(3), vec![1, 3, 5]);
        // Fewer than three qualifying moves -> as many as exist.
        let mut quiet = AnalysisData::demo();
        quiet.winrate_history = vec![50.0, 51.0, 51.5];
        assert!(quiet.key_moves(3).is_empty());
    }

    #[test]
    fn spiral_finds_nearest_empty() {
        // Empty board -> centre.
        assert_eq!(spiral_empty(|_, _| false), Some((9, 9)));
        // Centre occupied -> a ring-1 point.
        let p = spiral_empty(|c, r| (c, r) == (9, 9)).unwrap();
        assert!((8..=10).contains(&p.0) && (8..=10).contains(&p.1) && p != (9, 9));
        // Everything occupied except (0, 0) -> that point.
        assert_eq!(spiral_empty(|c, r| (c, r) != (0, 0)), Some((0, 0)));
        // Full board -> none.
        assert_eq!(spiral_empty(|_, _| true), None);
    }
}

//! All painting, in reference coordinates (1440x900) scaled to the window.
//!
//! Layout constants are transcribed from `scripts/design/xuanping_mockup.py`
//! (Play view) or from the design spec (Review / Settings / Help). Geometry
//! rule: rects and circles only — no rounded corners anywhere. Colors come
//! from the active `Palette` exclusively.

use std::time::Instant;

use eframe::egui::{
    self, Align2, Color32, FontId, Mesh, Painter, Pos2, Rect, Shape, Stroke, StrokeKind, TextureId,
    pos2,
};

use crate::scene::{
    self, AnalysisData, DEMO_BLACK, DEMO_MOVES, DEMO_WHITE, GameState, SettingsState, StoneColor,
    View,
};
use crate::theme::{self, Fonts, Palette, ThemeKind};

// ---- right column layout (reference px) ----
const COL_X: f32 = 904.0;
const COL_W: f32 = 376.0;
const COL_R: f32 = COL_X + COL_W;
const GRAPH_Y: f32 = 128.0;
const GRAPH_H: f32 = 72.0;
const WR_MIN: f32 = 35.0;
const WR_MAX: f32 = 70.0;
const ROW_Y: f32 = 232.0;
const ROW_H: f32 = 36.0;

// ---- review long-scroll layout ----
pub const SCROLL_X0: f32 = 96.0;
pub const SCROLL_X1: f32 = 1344.0;
pub const SCROLL_Y0: f32 = 720.0;
pub const SCROLL_Y1: f32 = 848.0;

// ---- settings layout ----
pub const SET_X: f32 = 480.0;
pub const SET_W: f32 = 480.0;
const SET_TITLE_Y: f32 = 128.0;
pub const SET_ROW_Y: f32 = 176.0;
pub const SET_ROW_H: f32 = 48.0;
pub const SET_ROWS: usize = 6;

// ---- stone animation ----
const FADE_MS: f32 = 160.0;
const RIPPLE_MS: f32 = 300.0;

/// Reference -> window coordinate mapper.
pub struct Scaled<'a> {
    pub p: &'a Painter,
    s: f32,
    ox: f32,
    oy: f32,
}

impl<'a> Scaled<'a> {
    pub fn new(p: &'a Painter, viewport: Rect) -> Self {
        let s = (viewport.width() / scene::REF_W).min(viewport.height() / scene::REF_H);
        let ox = viewport.min.x + (viewport.width() - scene::REF_W * s) / 2.0;
        let oy = viewport.min.y + (viewport.height() - scene::REF_H * s) / 2.0;
        Self { p, s, ox, oy }
    }

    /// Reference point -> window point.
    pub fn pt(&self, x: f32, y: f32) -> Pos2 {
        pos2(self.ox + x * self.s, self.oy + y * self.s)
    }

    /// Window point -> reference point (inverse of `pt`).
    pub fn unpt(&self, pos: Pos2) -> (f32, f32) {
        ((pos.x - self.ox) / self.s, (pos.y - self.oy) / self.s)
    }

    /// Reference length -> window length.
    pub fn sp(&self, v: f32) -> f32 {
        v * self.s
    }

    fn rrect(&self, x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::from_min_max(self.pt(x, y), self.pt(x + w, y + h))
    }

    fn hline(&self, x0: f32, x1: f32, y: f32, color: Color32, alpha: f32, width: f32) {
        self.p.line_segment(
            [self.pt(x0, y), self.pt(x1, y)],
            Stroke::new(self.sp(width), theme::with_alpha(color, alpha)),
        );
    }

    fn vline(&self, x: f32, y0: f32, y1: f32, color: Color32, alpha: f32, width: f32) {
        self.p.line_segment(
            [self.pt(x, y0), self.pt(x, y1)],
            Stroke::new(self.sp(width), theme::with_alpha(color, alpha)),
        );
    }

    fn text(
        &self,
        x: f32,
        y: f32,
        align: Align2,
        text: &str,
        size: f32,
        family: egui::FontFamily,
        color: Color32,
        alpha: f32,
    ) {
        self.p.text(
            self.pt(x, y),
            align,
            text,
            FontId::new(self.sp(size), family),
            theme::with_alpha(color, alpha),
        );
    }

    /// Width of `text` at `size`, in reference px.
    fn text_width(&self, text: &str, size: f32, family: egui::FontFamily) -> f32 {
        let galley = self.p.layout_no_wrap(
            text.to_owned(),
            FontId::new(self.sp(size), family),
            Color32::WHITE,
        );
        galley.rect.width() / self.s
    }

    /// Palette hairline (ink, theme-dependent alpha).
    fn hairline_h(&self, x0: f32, x1: f32, y: f32, pal: &Palette) {
        self.hline(x0, x1, y, pal.ink, pal.hairline_alpha, 1.0);
    }

    fn hairline_v(&self, x: f32, y0: f32, y1: f32, pal: &Palette) {
        self.vline(x, y0, y1, pal.ink, pal.hairline_alpha, 1.0);
    }
}

pub struct FrameInput<'a> {
    pub fonts: &'a Fonts,
    pub palette: &'a Palette,
    pub theme: ThemeKind,
    pub analysis: &'a AnalysisData,
    pub state: &'a GameState,
    pub settings: &'a SettingsState,
    pub view: View,
    pub help_open: bool,
    pub wash_tex: Option<TextureId>,
    /// Hovered empty intersection, if any (Play only).
    pub hover: Option<(u8, u8)>,
    /// Pointer is inside the grid area (drives the A-T / 1-19 rulers).
    pub pointer_in_grid: bool,
    /// Breathing-dot alpha.
    pub breath: f32,
    pub now: Instant,
    /// Play status-line override in engine mode: (left engine label, right
    /// 「思考/载 + 访问」 text). `None` keeps the demo's fixed line.
    pub status_line: Option<(String, String)>,
    /// Engine mode: backend rows are locked (青点仍指实际后端, 灰注不可切).
    pub backend_locked: bool,
}

pub fn paint_all(sc: &Scaled, input: &FrameInput) {
    title(sc, input);
    match input.view {
        View::Play => play(sc, input),
        View::Review => review(sc, input),
        View::Settings => settings(sc, input),
    }
    if input.help_open {
        help_overlay(sc, input);
    }
}

fn title(sc: &Scaled, input: &FrameInput) {
    // 玄枰 at (32, 30), SimSun 15, letter spacing 0.12em.
    let mut x = 32.0;
    for ch in ["玄", "枰"] {
        sc.text(x, 30.0, Align2::LEFT_TOP, ch, 15.0, input.fonts.song.clone(), input.palette.ink, 0.40);
        x += 15.0 * 1.12;
    }
}

// ---- Play view ----

fn play(sc: &Scaled, input: &FrameInput) {
    let pal = input.palette;
    board(sc, input, input.state.show_analysis);
    if input.state.show_analysis {
        sc.hairline_v(872.0, 64.0, 836.0, pal);
        sc.hairline_v(1304.0, 64.0, 836.0, pal);
        right_column(sc, input);
    } else if input.state.user_turn() {
        turn_seal(sc, input);
    }
    if input.state.show_kifu {
        kifu_strip(sc, input);
    }
    if input.pointer_in_grid {
        rulers(sc, input);
    }
}

/// 轮着朱印: 14px red outline square just outside the board's bottom-right
/// corner — only in pure play (analysis off) on the user's turn. When the
/// analysis overlay is up, red belongs to the winrate graph's key tick.
fn turn_seal(sc: &Scaled, input: &FrameInput) {
    let pal = input.palette;
    let x = scene::BX + scene::BS + 12.0;
    let y = scene::BY + scene::BS + 12.0;
    sc.p.rect_stroke(
        sc.rrect(x, y, 14.0, 14.0),
        0.0,
        Stroke::new(sc.sp(1.0), theme::with_alpha(pal.zhu, 0.90)),
        StrokeKind::Middle,
    );
}

/// A-T / 1-19 rulers, shown while the pointer is inside the grid area.
fn rulers(sc: &Scaled, input: &FrameInput) {
    let pal = input.palette;
    let f = input.fonts;
    const LETTERS: &[u8] = b"ABCDEFGHJKLMNOPQRST";
    for i in 0..19u8 {
        let (x, _) = scene::gp(i, 0);
        sc.text(
            x,
            scene::GY + 18.0 * scene::CELL + 8.0,
            Align2::CENTER_TOP,
            &(LETTERS[i as usize] as char).to_string(),
            9.0,
            f.hei.clone(),
            pal.ink,
            0.25,
        );
        let (_, y) = scene::gp(0, i);
        sc.text(
            scene::GX - 8.0,
            y,
            Align2::RIGHT_CENTER,
            &(19 - i).to_string(),
            9.0,
            f.hei.clone(),
            pal.ink,
            0.25,
        );
    }
}

fn board(sc: &Scaled, input: &FrameInput, with_overlay: bool) {
    let pal = input.palette;
    let f = input.fonts;
    // Surface with a hairline border.
    let rect = sc.rrect(scene::BX, scene::BY, scene::BS, scene::BS);
    sc.p.rect_filled(rect, 0.0, pal.surface);
    sc.p.rect_stroke(
        rect,
        0.0,
        Stroke::new(sc.sp(1.0), theme::with_alpha(pal.ink, pal.hairline_alpha)),
        StrokeKind::Middle,
    );

    // Territory wash: above the surface, below the grid.
    if with_overlay
        && let Some(tex) = input.wash_tex
    {
        let grid = sc.rrect(scene::GX, scene::GY, 18.0 * scene::CELL, 18.0 * scene::CELL);
        sc.p.image(
            tex,
            grid,
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    }

    // Grid lines (ink 22%) and star points (ink 35%).
    for i in 0..19 {
        let v = i as f32 * scene::CELL;
        sc.vline(scene::GX + v, scene::GY, scene::GY + 18.0 * scene::CELL, pal.ink, 0.22, 1.0);
        sc.hline(scene::GX, scene::GX + 18.0 * scene::CELL, scene::GY + v, pal.ink, 0.22, 1.0);
    }
    for c in [3u8, 9, 15] {
        for r in [3u8, 9, 15] {
            let (x, y) = scene::gp(c, r);
            sc.p.circle_filled(sc.pt(x, y), sc.sp(1.5), theme::with_alpha(pal.ink, 0.35));
        }
    }

    // Stones: demo position + placed moves in demo mode; the authoritative
    // engine board (captures included) in engine mode.
    match &input.state.live {
        Some(live) => {
            for r in 0..19u8 {
                for c in 0..19u8 {
                    match live.stones[r as usize][c as usize] {
                        1 => stone(sc, input, c, r, StoneColor::Black),
                        2 => stone(sc, input, c, r, StoneColor::White),
                        _ => {}
                    }
                }
            }
        }
        None => {
            for &(c, r) in DEMO_BLACK {
                stone(sc, input, c, r, StoneColor::Black);
            }
            for &(c, r) in DEMO_WHITE {
                stone(sc, input, c, r, StoneColor::White);
            }
            for m in &input.state.placed {
                stone(sc, input, m.pos.0, m.pos.1, m.color);
            }
        }
    }

    // Ripple + ghost stone (Play only).
    if input.view == View::Play {
        ripple(sc, input);
        if let Some((c, r)) = input.hover {
            let (x, y) = scene::gp(c, r);
            sc.p.circle_filled(sc.pt(x, y), sc.sp(scene::STONE_R), theme::with_alpha(pal.ink, 0.10));
        }
    }

    // Candidate rings (rank 1/2/3) with winrate labels.
    if with_overlay {
        for cand in &input.analysis.candidates {
            if cand.winrate_text.is_empty() {
                continue; // engine mode before the first analysis frame
            }
            let (x, y) = scene::gp(cand.pos.0, cand.pos.1);
            sc.p.circle_stroke(
                sc.pt(x, y),
                sc.sp(scene::CAND_R),
                Stroke::new(sc.sp(cand.ring_width), theme::with_alpha(pal.ink, cand.ring_alpha)),
            );
            sc.text(
                x,
                y + 24.0,
                Align2::CENTER_TOP,
                &cand.winrate_text,
                11.0,
                f.hei.clone(),
                pal.ink,
                0.60,
            );
        }
    }
}

/// Ripple around the most recent placement: frost ring growing to 1.8x the
/// stone diameter, alpha 25% -> 0 over 300ms. Off under reduced motion.
fn ripple(sc: &Scaled, input: &FrameInput) {
    if input.settings.reduced_motion {
        return;
    }
    let Some((c, r, t0)) = input.state.last_placed else {
        return;
    };
    let ms = input.now.duration_since(t0).as_secs_f32() * 1000.0;
    if !(0.0..RIPPLE_MS).contains(&ms) {
        return;
    }
    let t = ms / RIPPLE_MS;
    let (x, y) = scene::gp(c, r);
    sc.p.circle_stroke(
        sc.pt(x, y),
        sc.sp(scene::STONE_R * (1.0 + 0.8 * t)),
        Stroke::new(sc.sp(1.0), theme::with_alpha(input.palette.ink, 0.25 * (1.0 - t))),
    );
}

fn stone(sc: &Scaled, input: &FrameInput, c: u8, r: u8, color: StoneColor) {
    let pal = input.palette;
    let (x, y) = scene::gp(c, r);

    // Placement animation: 160ms fade-in + 2px settle (drop). Reduced
    // motion keeps the fade, drops the settle.
    let mut alpha = 1.0f32;
    let mut yoff = 0.0f32;
    if let Some((lc, lr, t0)) = input.state.last_placed
        && (lc, lr) == (c, r)
    {
        let ms = input.now.duration_since(t0).as_secs_f32() * 1000.0;
        if ms < FADE_MS {
            let t = ms / FADE_MS;
            alpha = t;
            if !input.settings.reduced_motion {
                yoff = -2.0 * (1.0 - (1.0 - t).powi(3));
            }
        }
    }

    let center = sc.pt(x, y + yoff);
    let radius = sc.sp(scene::STONE_R);
    match color {
        StoneColor::Black => {
            sc.p.circle_filled(center, radius, theme::with_alpha(pal.stone_black, alpha));
            if let Some(rim) = pal.black_rim {
                // Full-circle 0.75px rim light.
                sc.p.circle_stroke(
                    center,
                    radius,
                    Stroke::new(sc.sp(0.75), theme::with_alpha(pal.ink, rim * alpha)),
                );
            }
        }
        StoneColor::White => {
            sc.p.circle_filled(center, radius, theme::with_alpha(pal.stone_white, alpha));
            if let Some(rim) = pal.white_rim {
                sc.p.circle_stroke(
                    center,
                    radius,
                    Stroke::new(sc.sp(0.75), theme::with_alpha(pal.ink, rim * alpha)),
                );
            }
        }
    }
}

// ---- right column (analysis overlay) ----

fn right_column(sc: &Scaled, input: &FrameInput) {
    let pal = input.palette;
    let f = input.fonts;
    sc.text(COL_X, 92.0, Align2::LEFT_TOP, "胜率", 11.0, f.hei.clone(), pal.gray, 0.90);
    sc.text(COL_R, 90.0, Align2::RIGHT_TOP, "析", 13.0, f.hei.clone(), pal.ink, 0.90);
    winrate_graph(sc, input);
    candidate_rows(sc, input);
    readings(sc, input);
    status_line(sc, input);
}

/// Curve x/y for move `m` in reference px.
fn wpt(a: &AnalysisData, m: usize) -> (f32, f32) {
    let v = a.winrate_at(m);
    // Denominator floored at 1 so a live game's first ply (len-1 == 0) is safe.
    let denom = ((a.winrate_history.len() - 1).max(1)) as f32;
    (
        COL_X + COL_W * m as f32 / denom,
        GRAPH_Y + GRAPH_H * (WR_MAX - v) / (WR_MAX - WR_MIN),
    )
}

fn winrate_graph(sc: &Scaled, input: &FrameInput) {
    let pal = input.palette;
    let a = input.analysis;
    let n = a.winrate_history.len();

    // Vertical gradient fill under the curve (top 12% -> bottom 0).
    let mut mesh = Mesh::default();
    for m in 0..n {
        let (x, y) = wpt(a, m);
        let top_alpha = 0.12 * (1.0 - (y - GRAPH_Y) / GRAPH_H).clamp(0.0, 1.0);
        mesh.colored_vertex(sc.pt(x, y), theme::with_alpha(pal.ink, top_alpha));
        mesh.colored_vertex(
            sc.pt(x, GRAPH_Y + GRAPH_H),
            theme::with_alpha(pal.ink, 0.0),
        );
        if m > 0 {
            let b = 2 * m as u32;
            mesh.add_triangle(b - 2, b - 1, b);
            mesh.add_triangle(b - 1, b + 1, b);
        }
    }
    sc.p.add(Shape::mesh(mesh));

    // 50% hairline.
    let y50 = GRAPH_Y + GRAPH_H * (WR_MAX - 50.0) / (WR_MAX - WR_MIN);
    sc.hline(COL_X, COL_R, y50, pal.ink, pal.hairline_alpha, 1.0);

    // The curve itself.
    let pts: Vec<Pos2> = (0..n).map(|m| { let (x, y) = wpt(a, m); sc.pt(x, y) }).collect();
    sc.p.add(Shape::line(pts, Stroke::new(sc.sp(1.0), theme::with_alpha(pal.ink, 0.85))));

    // Key move: this view's single red accent (tick + small dot above).
    let (kx, ky) = wpt(a, a.key_move);
    sc.vline(kx, ky - 4.5, ky + 4.5, pal.zhu, 0.95, 1.5);
    sc.p.circle_filled(sc.pt(kx, ky - 7.0), sc.sp(1.5), theme::with_alpha(pal.zhu, 0.95));

    // Current-move tick.
    let cur = input.state.cursor.min(n - 1);
    let (cx, cy) = wpt(a, cur);
    sc.vline(cx, cy - 3.0, cy + 3.0, pal.ink, 0.80, 1.5);
}

fn candidate_rows(sc: &Scaled, input: &FrameInput) {
    let pal = input.palette;
    let f = input.fonts;
    for (i, cand) in input.analysis.candidates.iter().enumerate() {
        let y = ROW_Y + i as f32 * ROW_H;
        sc.text(COL_X, y + 8.0, Align2::LEFT_TOP, &cand.seq, 13.0, f.song.clone(), pal.ink, 0.85);
        sc.text(COL_X + 36.0, y + 8.0, Align2::LEFT_TOP, &cand.coord, 13.0, f.hei.clone(), pal.ink, 0.90);
        sc.text(COL_X + 236.0, y + 8.0, Align2::RIGHT_TOP, &cand.winrate_text, 13.0, f.hei.clone(), pal.ink, 0.90);
        sc.text(COL_R, y + 10.0, Align2::RIGHT_TOP, &cand.visits, 11.0, f.hei.clone(), pal.gray, 0.90);
        if i > 0 {
            sc.hairline_h(COL_X, COL_R, y, pal);
        }
    }
    sc.hairline_h(COL_X, COL_R, ROW_Y + 3.0 * ROW_H, pal);
}

fn readings(sc: &Scaled, input: &FrameInput) {
    let pal = input.palette;
    let f = input.fonts;
    let top = &input.analysis.candidates[0];
    let items = [
        ("胜率", top.winrate_text.as_str()),
        ("目差", input.analysis.score_lead_text.as_str()),
        ("访问", input.analysis.visits_text.as_str()),
    ];
    let mut x = COL_X;
    for (label, val) in items {
        sc.text(x, 388.0, Align2::LEFT_TOP, label, 11.0, f.hei.clone(), pal.gray, 0.90);
        x += sc.text_width(label, 11.0, f.hei.clone()) + 8.0;
        sc.text(x, 384.0, Align2::LEFT_TOP, val, 15.0, f.hei.clone(), pal.ink, 0.92);
        x += sc.text_width(val, 15.0, f.hei.clone()) + 28.0;
    }
}

fn status_line(sc: &Scaled, input: &FrameInput) {
    let pal = input.palette;
    let f = input.fonts;
    let (left, status) = match &input.status_line {
        Some((l, r)) => (l.clone(), r.clone()),
        None => {
            let status_text = if input.state.is_thinking() { "思考" } else { "待机" };
            (
                "KataGo · cudabackend · b11".to_string(),
                format!("{status_text}\u{3000}访问 {}", input.analysis.visits_text),
            )
        }
    };
    sc.text(COL_X, 850.0, Align2::LEFT_TOP, &left, 10.0, f.hei.clone(), pal.gray, 0.70);
    let tw = sc.text_width(&status, 11.0, f.hei.clone());
    sc.text(COL_R, 850.0, Align2::RIGHT_TOP, &status, 11.0, f.hei.clone(), pal.gray, 0.90);
    // Breathing dot (青), left of the status text.
    let dot_x = COL_R - tw - 14.0;
    sc.p.circle_filled(sc.pt(dot_x, 855.5), sc.sp(2.0), theme::with_alpha(pal.qing, input.breath));
}

// ---- right-edge vertical kifu strip ----

fn kifu_strip(sc: &Scaled, input: &FrameInput) {
    let pal = input.palette;
    let f = input.fonts;
    sc.text(1398.0, 830.0, Align2::RIGHT_TOP, "谱", 11.0, f.song.clone(), pal.gray, 0.60);
    // Most recent 4 moves ending at the cursor; newest at the right edge.
    let cur = input.state.cursor;
    let from = cur.saturating_sub(3).max(1);
    for m in from..=cur {
        let x = 1398.0 - (cur - m) as f32 * 27.0;
        let alpha = if m == cur { 1.0 } else { 0.40 };
        vtext(sc, x, 120.0, &scene::cn_num(m), 12.5, f.song.clone(), pal.ink, alpha, 16.0);
    }
}

/// Vertical text: chars stacked downward, x = column centre.
#[allow(clippy::too_many_arguments)]
fn vtext(
    sc: &Scaled,
    x: f32,
    y: f32,
    text: &str,
    size: f32,
    family: egui::FontFamily,
    color: Color32,
    alpha: f32,
    pitch: f32,
) {
    let mut cy = y;
    for ch in text.chars() {
        sc.p.text(
            sc.pt(x, cy),
            Align2::CENTER_TOP,
            ch.to_string(),
            FontId::new(sc.sp(size), family.clone()),
            theme::with_alpha(color, alpha),
        );
        cy += pitch;
    }
}

// ---- Review view ----

fn review(sc: &Scaled, input: &FrameInput) {
    // Bare board (no overlay), same geometry as Play.
    board(sc, input, false);
    review_scroll(sc, input);
    // Comment strip, top-right: vertical SimSun, gray.
    vtext(
        sc,
        1230.0,
        140.0,
        "此处的得失,当从全局看。",
        13.0,
        input.fonts.song.clone(),
        input.palette.gray,
        0.80,
        13.0 * 1.9,
    );
}

/// Scroll x for move `m`; x domain spans 96..1344 over 0..=total.
fn scroll_x(m: usize, total: usize) -> f32 {
    SCROLL_X0 + (SCROLL_X1 - SCROLL_X0) * m as f32 / total as f32
}

/// Scroll y for a winrate value (35..70 mapped onto 720..848).
fn scroll_y(v: f32) -> f32 {
    SCROLL_Y0 + (SCROLL_Y1 - SCROLL_Y0) * (WR_MAX - v) / (WR_MAX - WR_MIN)
}

fn review_scroll(sc: &Scaled, input: &FrameInput) {
    let pal = input.palette;
    let f = input.fonts;
    let a = input.analysis;
    let total = input.state.total_moves().max(DEMO_MOVES);

    // Winrate pen: 1.5px, ends fading out over the first/last 24px (收锋).
    for m in 1..=total {
        let (x0, y0) = (scroll_x(m - 1, total), scroll_y(a.winrate_at(m - 1)));
        let (x1, y1) = (scroll_x(m, total), scroll_y(a.winrate_at(m)));
        let fade = ((x0 - SCROLL_X0) / 24.0)
            .min((SCROLL_X1 - x1) / 24.0)
            .clamp(0.0, 1.0);
        sc.p.line_segment(
            [sc.pt(x0, y0), sc.pt(x1, y1)],
            Stroke::new(sc.sp(1.5), theme::with_alpha(pal.ink, 0.85 * fade)),
        );
    }

    // Frost ticks + move numbers every 10 moves.
    for m in (10..=total).step_by(10) {
        let x = scroll_x(m, total);
        sc.vline(x, 822.0, 828.0, pal.ink, pal.hairline_alpha, 1.0);
        sc.text(x, 832.0, Align2::CENTER_TOP, &m.to_string(), 9.0, f.hei.clone(), pal.gray, 0.90);
    }

    // Key moves: this view's red — 6px ticks, top-3 swings above 8%.
    for m in a.key_moves(3) {
        let x = scroll_x(m, total);
        let y = scroll_y(a.winrate_at(m));
        sc.vline(x, y - 3.0, y + 3.0, pal.zhu, 0.95, 1.5);
    }

    // Cursor marker (scrub position).
    let cx = scroll_x(input.state.cursor, total);
    sc.vline(cx, SCROLL_Y0, SCROLL_Y1, pal.ink, 0.25, 1.0);
}

// ---- Settings view「器」 ----

fn settings(sc: &Scaled, input: &FrameInput) {
    let pal = input.palette;
    let f = input.fonts;
    sc.text(
        SET_X + SET_W / 2.0,
        SET_TITLE_Y,
        Align2::CENTER_TOP,
        "器",
        18.0,
        f.song.clone(),
        pal.ink,
        0.90,
    );

    let backends = [("dummy", "未接"), ("TRT", "兜底"), ("CUDA", "认证 plan")];
    for (i, &(label, note)) in backends.iter().enumerate() {
        let cy = SET_ROW_Y + i as f32 * SET_ROW_H + SET_ROW_H / 2.0;
        let current = input.settings.backend == i;
        // Engine mode: rows stay put; the qing dot marks the real backend.
        let note = if input.backend_locked && i < 3 { "运行中不可切" } else { note };
        status_dot(sc, SET_X + 2.0, cy, current, pal);
        sc.text(SET_X + 16.0, cy, Align2::LEFT_CENTER, label, 13.0, f.hei.clone(), pal.ink, 0.90);
        sc.text(SET_X + SET_W, cy, Align2::RIGHT_CENTER, note, 11.0, f.hei.clone(), pal.gray, 0.90);
    }

    // Theme row: dots 玄墨/雪宣, linked to `t`.
    let cy = SET_ROW_Y + 3.0 * SET_ROW_H + SET_ROW_H / 2.0;
    sc.text(SET_X + 16.0, cy, Align2::LEFT_CENTER, "主题", 13.0, f.hei.clone(), pal.ink, 0.90);
    option_pair(sc, input, cy, "玄墨", "雪宣", input.theme == ThemeKind::Xuan);

    // Motion row: dots 开/减.
    let cy = SET_ROW_Y + 4.0 * SET_ROW_H + SET_ROW_H / 2.0;
    sc.text(SET_X + 16.0, cy, Align2::LEFT_CENTER, "动效", 13.0, f.hei.clone(), pal.ink, 0.90);
    option_pair(sc, input, cy, "开", "减", input.settings.reduced_motion);

    // Sound row: gray, not clickable.
    let cy = SET_ROW_Y + 5.0 * SET_ROW_H + SET_ROW_H / 2.0;
    sc.text(SET_X + 16.0, cy, Align2::LEFT_CENTER, "声音", 13.0, f.hei.clone(), pal.gray, 0.90);
    sc.text(SET_X + SET_W, cy, Align2::RIGHT_CENTER, "未装", 11.0, f.hei.clone(), pal.gray, 0.90);

    // Hairlines between rows and under the last one.
    for i in 1..=SET_ROWS {
        sc.hairline_h(SET_X, SET_X + SET_W, SET_ROW_Y + i as f32 * SET_ROW_H, pal);
    }
}

/// r=2 status dot: qing = current, gray = the rest. Never red here.
fn status_dot(sc: &Scaled, x: f32, y: f32, current: bool, pal: &Palette) {
    let (color, alpha) = if current { (pal.qing, 1.0) } else { (pal.gray, 0.50) };
    sc.p.circle_filled(sc.pt(x, y), sc.sp(2.0), theme::with_alpha(color, alpha));
}

/// Two options with status dots, laid out from the right edge.
fn option_pair(sc: &Scaled, input: &FrameInput, cy: f32, first: &str, second: &str, second_on: bool) {
    let pal = input.palette;
    let f = input.fonts;
    let mut x = SET_X + SET_W;
    for (label, on) in [(second, second_on), (first, !second_on)] {
        sc.text(x, cy, Align2::RIGHT_CENTER, label, 11.0, f.hei.clone(), pal.ink, 0.90);
        x -= sc.text_width(label, 11.0, f.hei.clone()) + 10.0;
        status_dot(sc, x, cy, on, pal);
        x -= 24.0;
    }
}

// ---- help overlay「键」 ----

const HELP_W: f32 = 420.0;
const HELP_ROWS: &[(&str, &str)] = &[
    ("a", "析"),
    ("k", "谱"),
    ("r", "盘"),
    ("e", "器"),
    ("t", "玄宣"),
    ("← →", "步进"),
    ("?", "此页"),
    ("Esc", "归"),
];

fn help_overlay(sc: &Scaled, input: &FrameInput) {
    let pal = input.palette;
    let f = input.fonts;
    // Dim the scene underneath.
    let full = sc.rrect(0.0, 0.0, scene::REF_W, scene::REF_H);
    sc.p.rect_filled(full, 0.0, theme::with_alpha(pal.dim, 0.40));

    // Panel: centred, 420 wide, surface 96% + hairline border.
    let row_h = 28.0;
    let h = 24.0 + 15.0 + 34.0 + HELP_ROWS.len() as f32 * row_h + 20.0;
    let x0 = (scene::REF_W - HELP_W) / 2.0;
    let y0 = (scene::REF_H - h) / 2.0;
    let panel = sc.rrect(x0, y0, HELP_W, h);
    sc.p.rect_filled(panel, 0.0, theme::with_alpha(pal.surface, 0.96));
    sc.p.rect_stroke(
        panel,
        0.0,
        Stroke::new(sc.sp(1.0), theme::with_alpha(pal.ink, pal.hairline_alpha)),
        StrokeKind::Middle,
    );

    sc.text(x0 + 28.0, y0 + 24.0, Align2::LEFT_TOP, "键", 15.0, f.song.clone(), pal.ink, 0.90);
    for (i, &(key, meaning)) in HELP_ROWS.iter().enumerate() {
        let y = y0 + 24.0 + 34.0 + i as f32 * row_h;
        sc.text(x0 + 30.0, y, Align2::LEFT_TOP, key, 12.0, f.hei.clone(), pal.ink, 0.85);
        sc.text(x0 + 150.0, y, Align2::LEFT_TOP, meaning, 12.0, f.hei.clone(), pal.gray, 0.90);
    }
}

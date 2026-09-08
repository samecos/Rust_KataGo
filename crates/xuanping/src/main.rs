//! 玄枰 (xuanping) — minimal eastern-aesthetic shell for KataGo.
//!
//! Dual mode. Demo mode (default): hardcoded mockup data, simulated replies —
//! no engine touched. Engine mode (`--model <path>` / `XUANPING_MODEL`, plus
//! optional `--backend cuda|dummy` and `--threads N`): a KataGo `AsyncBot`
//! runs on its own thread (`crate::engine`) and streams `Snap` snapshots that
//! drive stones, turn, candidates, territory wash and the winrate curve; the
//! user takes black, the engine white.
//!
//! Views: Play / Review / Settings (`r` / `e` / Esc), analysis overlay inside
//! Play (`a`), kifu strip (`k`), theme (`t`), help overlay (`?` / `h`).

mod engine;
mod paint;
mod scene;
mod theme;

use std::time::{Duration, Instant};

use eframe::egui::{self, FontData, FontDefinitions, FontFamily, Key, TextureOptions, Visuals};
use scene::{AnalysisData, Candidate, GameState, SettingsState, View};
use theme::{Fonts, Palette, ThemeKind};

use crate::engine::{BackendKind, EngineCmd, EngineConfig, Snap};

/// CLI/env launch flags collected before the window opens.
struct Launch {
    model: Option<String>,
    backend: Option<String>,
    threads: Option<i32>,
}

impl Launch {
    /// `--model/--backend/--threads` take precedence over their XUANPING_*
    /// environment equivalents (used for headless verification).
    fn parse() -> Self {
        let mut out = Self {
            model: None,
            backend: None,
            threads: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            let mut val = |prefix: &str| -> Option<String> {
                if let Some(rest) = a.strip_prefix(prefix) {
                    return Some(rest.to_string());
                }
                args.next()
            };
            if a == "--model" {
                out.model = val("");
            } else if let Some(m) = a.strip_prefix("--model=") {
                out.model = Some(m.to_string());
            } else if a == "--backend" {
                out.backend = val("");
            } else if let Some(b) = a.strip_prefix("--backend=") {
                out.backend = Some(b.to_string());
            } else if a == "--threads" {
                out.threads = val("").and_then(|v| v.parse().ok());
            } else if let Some(t) = a.strip_prefix("--threads=") {
                out.threads = t.parse().ok();
            }
        }
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        out.model = out.model.or_else(|| env("XUANPING_MODEL"));
        out.backend = out.backend.or_else(|| env("XUANPING_BACKEND"));
        out.threads =
            out.threads.or_else(|| env("XUANPING_THREADS").and_then(|v| v.parse().ok()));
        out
    }

    /// Engine mode iff a model was named. Backend defaults to CUDA (the
    /// project's primary路线); threads default to 8.
    fn engine_config(&self) -> Option<EngineConfig> {
        let model = self.model.clone()?;
        let backend = match self.backend.as_deref() {
            Some("dummy") => BackendKind::Dummy,
            _ => BackendKind::Cuda,
        };
        Some(EngineConfig {
            model,
            backend,
            threads: self.threads.unwrap_or(8),
        })
    }
}

fn main() -> eframe::Result<()> {
    let launch = Launch::parse();
    // Engine mode spawns its thread before the window opens so model loading
    // starts immediately (状态行「载」 until `Snap::Ready` lands).
    let backend_row;
    let engine = match launch.engine_config() {
        Some(cfg) => {
            backend_row = Some(cfg.backend.settings_row());
            let (tx, rx) = engine::spawn(cfg);
            Some(EngineUi { tx, rx })
        }
        None => {
            backend_row = None;
            None
        }
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("玄枰")
            .with_maximized(true)
            .with_min_inner_size([640.0, 400.0]),
        ..Default::default()
    };
    eframe::run_native(
        "玄枰",
        options,
        Box::new(move |cc| Ok(Box::new(XuanpingApp::new(cc, engine, backend_row)))),
    )
}

/// Live side of the bridge: command sender + snapshot receiver polled in ui().
struct EngineUi {
    tx: std::sync::mpsc::Sender<EngineCmd>,
    rx: std::sync::mpsc::Receiver<Snap>,
}

struct XuanpingApp {
    fonts: Fonts,
    state: GameState,
    analysis: AnalysisData,
    settings: SettingsState,
    view: View,
    help_open: bool,
    theme: ThemeKind,
    palette: Palette,
    wash_tex: Option<egui::TextureHandle>,
    t0: Instant,
    frames: u32,
    /// `Some` in engine mode.
    engine: Option<EngineUi>,
    /// From `Snap::Ready`; empty until the bot is up (状态行「载」期).
    engine_label: String,
    /// Fatal engine-side failure (model load, bad config, ...).
    engine_error: Option<String>,
    /// First `Snap::Analysis` landed (gates XUANPING_SHOT in engine mode).
    analysis_seen: bool,
    /// Ownership texture needs re-upload (live analysis updates it).
    wash_dirty: bool,
    /// When XUANPING_SHOT started waiting for the analysis gate.
    shot_since: Option<Instant>,
}

impl XuanpingApp {
    fn new(
        cc: &eframe::CreationContext<'_>,
        engine: Option<EngineUi>,
        backend_row: Option<usize>,
    ) -> Self {
        let fonts = load_fonts(&cc.egui_ctx);

        // Headless-capture overrides: XUANPING_VIEW, XUANPING_THEME.
        let mut view = View::Play;
        let mut show_analysis = true;
        match std::env::var("XUANPING_VIEW").as_deref() {
            Ok("play") => show_analysis = false,
            Ok("review") => view = View::Review,
            Ok("settings") => view = View::Settings,
            _ => {}
        }
        let theme = match std::env::var("XUANPING_THEME").as_deref() {
            Ok("xuan") => ThemeKind::Xuan,
            _ => ThemeKind::Mo,
        };
        cc.egui_ctx.set_theme(egui_theme(theme));

        let live = engine.is_some();
        let mut state = if live {
            GameState::new_live()
        } else {
            GameState::default()
        };
        state.show_analysis = show_analysis;
        let mut settings = SettingsState::default();
        if let Some(row) = backend_row {
            // 引擎模式: 青点固定在实际后端, 行点击被忽略(运行中不可切)。
            settings.backend = row;
        }
        Self {
            fonts,
            state,
            analysis: if live {
                AnalysisData::live_empty()
            } else {
                AnalysisData::demo()
            },
            settings,
            view,
            help_open: std::env::var_os("XUANPING_HELP").is_some(),
            theme,
            palette: theme.palette(),
            wash_tex: None,
            t0: Instant::now(),
            frames: 0,
            engine,
            engine_label: String::new(),
            engine_error: None,
            analysis_seen: false,
            wash_dirty: false,
            shot_since: None,
        }
    }

    /// Drain every pending engine snapshot. Board snaps update the game
    /// state; analysis snaps only apply to the newest position (stale ones
    /// are dropped, review scrubbing is purely local).
    fn pump_engine(&mut self, now: Instant) {
        let snaps = self.collect_snaps();
        for snap in snaps {
            self.ingest(snap, now);
        }
    }

    /// Drain pending snapshots. While the bot is still loading (no Ready yet,
    /// or no applicable analysis frame landed) short-block up to 25ms per
    /// frame so XUANPING_SHOT locks onto real output without stalling the UI.
    fn collect_snaps(&mut self) -> Vec<Snap> {
        let rx = match &self.engine {
            Some(e) => &e.rx,
            None => return Vec::new(),
        };
        let mut out = Vec::new();
        let want_block =
            self.engine_error.is_none() && (self.engine_label.is_empty() || !self.analysis_seen);
        if want_block {
            let deadline = Instant::now() + Duration::from_millis(25);
            loop {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break;
                }
                match rx.recv_timeout(left) {
                    Ok(snap) => {
                        let done = matches!(snap, Snap::Analysis { .. });
                        out.push(snap);
                        if done {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        }
        while let Ok(snap) = rx.try_recv() {
            out.push(snap);
        }
        out
    }

    fn ingest(&mut self, snap: Snap, now: Instant) {
        match snap {
            Snap::Ready { label } => {
                self.engine_label = label;
            }
            Snap::Error(e) => {
                self.engine_error = Some(e);
            }
            Snap::Board {
                stones,
                black_to_move,
                last_move,
                move_no,
            } => {
                self.state
                    .apply_live_board(stones, black_to_move, last_move, move_no, now);
                // Winrate history grows with the authoritative ply count;
                // fresh plies hold the previous value until their first
                // analysis frame lands.
                let want = (move_no.max(1) + 1).max(2);
                let hist = &mut self.analysis.winrate_history;
                let tail = *hist.last().unwrap_or(&50.0);
                if hist.len() > want {
                    hist.truncate(want);
                } else if hist.len() < want {
                    hist.resize(want, tail);
                }
            }
            Snap::Analysis {
                move_no,
                candidates,
                ownership,
                root_wr,
                lead,
                total_visits,
            } => {
                let applies = self
                    .state
                    .live
                    .as_ref()
                    .is_some_and(|l| l.move_no == move_no);
                if !applies {
                    return;
                }
                self.analysis_seen = true;
                self.apply_analysis(move_no, candidates, ownership, root_wr, lead, total_visits);
            }
        }
    }

    fn apply_analysis(
        &mut self,
        move_no: usize,
        candidates: Vec<engine::CandSnap>,
        ownership: Box<[[f32; 19]; 19]>,
        root_wr: f32,
        lead: f32,
        total_visits: i64,
    ) {
        let hist = &mut self.analysis.winrate_history;
        // history[ply] exists thanks to the Board-driven resize above.
        let idx = move_no.min(hist.len().saturating_sub(1));
        if let Some(slot) = hist.get_mut(idx) {
            *slot = root_wr;
        }
        self.analysis.score_lead_text = format!("{lead:+.1}");
        self.analysis.visits_text = fmt_visits(total_visits);
        self.analysis.key_move = key_move_of(hist);

        const SEQ: [&str; 3] = ["一", "二", "三"];
        let mut next: [Option<Candidate>; 3] = [None, None, None];
        for (i, cand) in candidates.iter().take(3).enumerate() {
            let (alpha, width) = Candidate::ring_style(i);
            next[i] = Some(Candidate {
                pos: (cand.c, cand.r),
                seq: SEQ[i].to_string(),
                coord: coord_name(cand.c, cand.r),
                winrate_text: format!("{:.1}", cand.winrate),
                visits: fmt_visits(cand.visits),
                ring_alpha: alpha,
                ring_width: width,
            });
        }
        for (dst, src) in self.analysis.candidates.iter_mut().zip(next.into_iter()) {
            if let Some(cand) = src {
                *dst = cand;
            }
        }
        self.analysis.ownership = *ownership;
        self.wash_dirty = true;
    }

    fn toggle_theme(&mut self, ctx: &egui::Context) {
        self.theme = match self.theme {
            ThemeKind::Mo => ThemeKind::Xuan,
            ThemeKind::Xuan => ThemeKind::Mo,
        };
        self.palette = self.theme.palette();
        // Wash texels are palette-bound; force a re-upload.
        self.wash_tex = None;
        ctx.set_theme(egui_theme(self.theme));
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        let (left, right, a, k, r, e, t, slash, h, esc) = ctx.input(|i| {
            (
                i.key_pressed(Key::ArrowLeft),
                i.key_pressed(Key::ArrowRight),
                i.key_pressed(Key::A),
                i.key_pressed(Key::K),
                i.key_pressed(Key::R),
                i.key_pressed(Key::E),
                i.key_pressed(Key::T),
                i.key_pressed(Key::Slash),
                i.key_pressed(Key::H),
                i.key_pressed(Key::Escape),
            )
        });

        // Help overlay is topmost: only its own dismiss keys act.
        if self.help_open {
            if esc || slash || h {
                self.help_open = false;
            }
            return;
        }
        if slash || h {
            self.help_open = true;
            return;
        }
        if t {
            self.toggle_theme(ctx);
        }
        if esc {
            self.view = View::Play;
        }
        if r {
            self.view = if self.view == View::Review { View::Play } else { View::Review };
        }
        if e {
            self.view = if self.view == View::Settings { View::Play } else { View::Settings };
        }
        // `a` / `k` are Play-scoped; stepping works in Play and Review.
        match self.view {
            View::Play => {
                if a {
                    self.state.show_analysis = !self.state.show_analysis;
                    // 引擎模式: 覆层随最新局面打开(scrub 不受影响)。
                    if self.state.show_analysis
                        && let Some(head) = self.state.live_head()
                    {
                        self.state.set_cursor(head);
                    }
                }
                if k {
                    self.state.show_kifu = !self.state.show_kifu;
                }
            }
            View::Review | View::Settings => {}
        }
        if self.view != View::Settings {
            // 引擎模式: 只能在已分析的历史内回看, 最新手永远可回。
            let floor = match self.state.live.as_ref() {
                Some(l) => l.move_no.max(1),
                None => 1,
            };
            if left && self.state.cursor > floor {
                self.state.step(-1);
            }
            if right {
                self.state.step(1);
            }
        }
    }

    /// Board placement (Play), scroll scrub (Review), row clicks (Settings).
    fn handle_pointer(&mut self, ctx: &egui::Context, sc: &paint::Scaled) {
        let (hover_pos, clicked, down) = ctx.input(|i| {
            (
                i.pointer.hover_pos(),
                i.pointer.primary_clicked(),
                i.pointer.primary_down(),
            )
        });
        let ref_pos = hover_pos.map(|p| sc.unpt(p));

        match self.view {
            View::Play => {
                let editable = self.state.user_turn();
                let hover = ref_pos
                    .and_then(|(rx, ry)| self.snap_point(rx, ry))
                    .filter(|&(c, r)| !self.state.is_occupied(c, r));
                if clicked
                    && editable
                    && let Some((c, r)) = hover
                {
                    self.play_user_move(c, r);
                }
            }
            View::Review => {
                // Click / drag scrub on the long scroll.
                if down
                    && let Some((rx, ry)) = ref_pos
                    && (paint::SCROLL_X0 - 8.0..=paint::SCROLL_X1 + 8.0).contains(&rx)
                    && (paint::SCROLL_Y0 - 12.0..=paint::SCROLL_Y1 + 12.0).contains(&ry)
                {
                    let total = self.state.total_moves();
                    let frac = ((rx - paint::SCROLL_X0) / (paint::SCROLL_X1 - paint::SCROLL_X0))
                        .clamp(0.0, 1.0);
                    self.state.set_cursor((frac * total as f32).round() as usize);
                }
            }
            View::Settings => {
                if clicked
                    && let Some((rx, ry)) = ref_pos
                    && (paint::SET_X..=paint::SET_X + paint::SET_W).contains(&rx)
                {
                    let row = ((ry - paint::SET_ROW_Y) / paint::SET_ROW_H) as i32;
                    if (0..paint::SET_ROWS as i32).contains(&row) {
                        match row {
                            // 引擎模式: 后端行禁切(灰注「运行中不可切」,
                            // 青点固定在实际后端)。
                            0..=2 if self.state.live.is_none() => {
                                self.settings.backend = row as usize
                            }
                            0..=2 => {}
                            3 => self.toggle_theme(ctx),
                            4 => self.settings.reduced_motion = !self.settings.reduced_motion,
                            _ => {} // 声音: not installed
                        }
                    }
                }
            }
        }
    }

    /// Snap a reference-space point to a board intersection.
    fn snap_point(&self, rx: f32, ry: f32) -> Option<(u8, u8)> {
        let c = ((rx - scene::GX) / scene::CELL).round();
        let r = ((ry - scene::GY) / scene::CELL).round();
        if !(0.0..19.0).contains(&c) || !(0.0..19.0).contains(&r) {
            return None;
        }
        let (gx, gy) = scene::gp(c as u8, r as u8);
        let dist = ((rx - gx).powi(2) + (ry - gy).powi(2)).sqrt();
        if dist > scene::CELL * 0.45 {
            return None;
        }
        Some((c as u8, r as u8))
    }

    /// Route a board click. Demo mode keeps the mockup placement machinery;
    /// engine mode forwards Play to the engine thread — legality and turn
    /// are decided by the bot and echoed back via `Snap::Board`.
    fn play_user_move(&mut self, c: u8, r: u8) {
        if self.state.live.is_some() {
            if let Some(e) = &self.engine {
                let _ = e.tx.send(EngineCmd::Play { c, r });
            }
            return;
        }
        let now = Instant::now();
        if self.state.place_user(c, r, now) {
            self.state.start_thinking(now);
        }
    }

    /// Engine-driven Play status line: (engine label, 思考/载 + 访问).
    /// `None` in demo mode (paint falls back to its fixed line).
    fn status_override(&self) -> Option<(String, String)> {
        self.state.live.as_ref()?;
        if self.engine_error.is_some() {
            return Some(("KataGo · 引擎错误".into(), "错误".into()));
        }
        if self.engine_label.is_empty() {
            return Some(("KataGo · 模型加载中".into(), "载".into()));
        }
        // 分析常开: 引擎始终在搜索最新局面。
        Some((
            format!("KataGo · {}", self.engine_label),
            format!("思考\u{3000}访问 {}", self.analysis.visits_text),
        ))
    }
}

/// Winrate-curve ply whose swing exceeds 8% the most; the latest ply when no
/// qualifying swing exists (mirrors the demo's key-move rule).
fn key_move_of(hist: &[f32]) -> usize {
    let mut best = (hist.len().saturating_sub(1), 0.0f32);
    for m in 1..hist.len() {
        let d = (hist[m] - hist[m - 1]).abs();
        if d > best.1 {
            best = (m, d);
        }
    }
    if best.1 > 8.0 {
        best.0
    } else {
        hist.len().saturating_sub(1)
    }
}

/// GTP-style coordinate, e.g. (8, 12) -> "J07" (letters skip I, rows count
/// down from 19).
pub fn coord_name(c: u8, r: u8) -> String {
    const LETTERS: &[u8] = b"ABCDEFGHJKLMNOPQRST";
    format!("{}{:02}", LETTERS[c as usize % 19] as char, 19 - r as i32)
}

/// 1234 -> "1.2k", 450 -> "450", 152300 -> "152k" (right-column conventions).
pub fn fmt_visits(v: i64) -> String {
    if v < 1000 {
        return v.to_string();
    }
    if v < 100_000 {
        let k = v as f64 / 1000.0;
        let s = format!("{k:.1}k");
        return s.strip_suffix(".0k").map(|s| format!("{s}k")).unwrap_or(s);
    }
    format!("{}k", v / 1000)
}

fn egui_theme(kind: ThemeKind) -> egui::Theme {
    match kind {
        ThemeKind::Mo => egui::Theme::Dark,
        ThemeKind::Xuan => egui::Theme::Light,
    }
}

impl eframe::App for XuanpingApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let now = Instant::now();
        self.handle_keys(&ctx);
        // Engine snapshots drive this frame's stones/readings.
        self.pump_engine(now);

        static ONCE: std::sync::Once = std::sync::Once::new();
        if std::env::var_os("XUANPING_DEBUG").is_some() {
            ONCE.call_once(|| {
                eprintln!(
                    "[dbg] content_rect={:?} screen_rect={:?} ppp={:?}",
                    ctx.content_rect(),
                    ctx.input(|i| i.raw.screen_rect),
                    ctx.pixels_per_point()
                );
            });
        }

        // Debug hook: XUANPING_SHOT=<ppm path> captures one frame and exits.
        // The request is delayed until the wgpu surface has settled at its
        // final (maximized) size — shooting on frame 0 catches the initial
        // min-size surface.
        self.frames += 1;
        if let Ok(path) = std::env::var("XUANPING_SHOT") {
            let shot = ctx.input(|i| {
                i.events.iter().find_map(|e| match e {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            if let Some(img) = shot {
                save_ppm(&path, &img);
                std::process::exit(0);
            }
            // Engine mode: hold fire until the first analysis frame landed,
            // bounded by a 15s timeout so a wedged engine still captures.
            let ready = self.engine.is_none() || self.analysis_seen;
            let t0 = *self.shot_since.get_or_insert(now);
            let timed_out = now.duration_since(t0) >= Duration::from_secs(15);
            if (ready || timed_out)
                && self.frames > 15
                && self.frames % 10 == 0
            {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(
                    egui::UserData::default(),
                ));
            }
        }

        // Simulated engine: play the white reply ~0.9s after the user move.
        self.state.maybe_reply(now);

        let sc = paint::Scaled::new(ui.painter(), ctx.content_rect());
        self.handle_pointer(&ctx, &sc);

        // Pointer context for hover ghost / rulers.
        let ref_pos = ctx.input(|i| i.pointer.hover_pos()).map(|p| sc.unpt(p));
        let pointer_in_grid = ref_pos.is_some_and(|(rx, ry)| {
            (scene::GX..=scene::GX + 18.0 * scene::CELL).contains(&rx)
                && (scene::GY..=scene::GY + 18.0 * scene::CELL).contains(&ry)
        });
        let hover = if self.view == View::Play && !self.help_open {
            ref_pos
                .and_then(|(rx, ry)| self.snap_point(rx, ry))
                .filter(|&(c, r)| !self.state.is_occupied(c, r))
        } else {
            None
        };

        // Wash texture: uploaded once, re-uploaded on theme switch or fresh
        // live ownership data.
        if self.view == View::Play
            && self.state.show_analysis
            && (self.wash_tex.is_none() || self.wash_dirty)
        {
            let img = scene::ownership_image(&self.analysis.ownership, &self.palette);
            self.wash_tex = Some(ctx.load_texture(
                "ownership-wash",
                img,
                TextureOptions {
                    magnification: egui::TextureFilter::Linear,
                    minification: egui::TextureFilter::Linear,
                    ..Default::default()
                },
            ));
            self.wash_dirty = false;
        }

        // Breathing dot: 2.4s fast breath while thinking, 5s idle;
        // constant 0.4 under reduced motion.
        let breath = if self.settings.reduced_motion {
            0.4
        } else {
            let period = if self.state.is_thinking() { 2.4 } else { 5.0 };
            let t = self.t0.elapsed().as_secs_f32();
            0.375 + 0.175 * (t * std::f32::consts::TAU / period).sin()
        };
        ctx.request_repaint_after(Duration::from_millis(33));

        let input = paint::FrameInput {
            fonts: &self.fonts,
            palette: &self.palette,
            theme: self.theme,
            analysis: &self.analysis,
            state: &self.state,
            settings: &self.settings,
            view: self.view,
            help_open: self.help_open,
            wash_tex: self.wash_tex.as_ref().map(|t| t.id()),
            hover,
            pointer_in_grid: pointer_in_grid && self.view == View::Play && !self.help_open,
            breath,
            now,
            status_line: self.status_override(),
            backend_locked: self.state.live.is_some(),
        };
        paint::paint_all(&sc, &input);
    }

    fn clear_color(&self, _visuals: &Visuals) -> [f32; 4] {
        self.palette.base.to_normalized_gamma_f32()
    }
}

/// Write a ColorImage as binary PPM (P6) for debug capture.
fn save_ppm(path: &str, img: &egui::ColorImage) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("create ppm");
    write!(f, "P6\n{} {}\n255\n", img.width(), img.height()).unwrap();
    for p in &img.pixels {
        f.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    eprintln!("[dbg] shot saved: {} ({}x{})", path, img.width(), img.height());
}

/// Register SimSun / YaHei Light from the Windows font directory as the
/// `song` / `hei` families. Missing files fall back to egui defaults
/// silently (index 0 selects the first face inside each .ttc).
fn load_fonts(ctx: &egui::Context) -> Fonts {
    let mut defs = FontDefinitions::default();
    let mut install = |key: &str, path: &str| -> Option<FontFamily> {
        let bytes = std::fs::read(path).ok()?;
        defs.font_data.insert(
            key.to_owned(),
            FontData {
                font: bytes.into(),
                index: 0,
                tweak: Default::default(),
            }
            .into(),
        );
        let family = FontFamily::Name(key.into());
        defs.families.insert(family.clone(), vec![key.to_owned()]);
        Some(family)
    };
    let song = install("song", r"C:\Windows\Fonts\simsun.ttc");
    let hei = install("hei", r"C:\Windows\Fonts\msyhl.ttc");
    if song.is_some() || hei.is_some() {
        ctx.set_fonts(defs);
    }
    let mut fonts = Fonts::fallback();
    if let Some(f) = song {
        fonts.song = f;
    }
    if let Some(f) = hei {
        fonts.hei = f;
    }
    fonts
}

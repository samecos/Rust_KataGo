//! Design tokens and font family handles.
//!
//! Two palettes: 玄墨 (dark, mirrored from `scripts/design/xuanping_mockup.py`)
//! and 雪宣 (light). All painting goes through `Palette`; no hardcoded colors
//! outside this file.

use eframe::egui::{Color32, FontFamily};

/// Theme selection (`t` toggles, `XUANPING_THEME=mo|xuan` overrides at boot).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ThemeKind {
    /// 玄墨 · dark
    Mo,
    /// 雪宣 · light
    Xuan,
}

impl ThemeKind {
    pub fn palette(self) -> Palette {
        match self {
            Self::Mo => Palette::mo(),
            Self::Xuan => Palette::xuan(),
        }
    }
}

/// Color tokens for one theme. Alphas that differ per theme are fields;
/// fixed alphas (grid 22%, star 35%, rings 72/50/34, ...) stay at call sites.
#[derive(Clone, Copy)]
pub struct Palette {
    /// Window base (玄 / 宣底)
    pub base: Color32,
    /// Board surface (黛 / 纸面)
    pub surface: Color32,
    /// Primary text and line work (月白 / 墨)
    pub ink: Color32,
    /// Secondary text (灰)
    pub gray: Color32,
    /// 朱 · seal red; at most one accent per frame, owned per view
    pub zhu: Color32,
    /// 青 · engine status only
    pub qing: Color32,
    pub stone_black: Color32,
    pub stone_white: Color32,
    /// Rim-light alpha on black stones (ink); dark theme only.
    pub black_rim: Option<f32>,
    /// Rim alpha on white stones (ink); light theme only.
    pub white_rim: Option<f32>,
    /// White-territory wash texel (月白 / 纯白)
    pub wash_white: Color32,
    /// Black-territory wash texel (黑 / 墨)
    pub wash_black: Color32,
    /// Hairline alpha: 8% on dark, 10% on light (color is always `ink`).
    pub hairline_alpha: f32,
    /// Full-screen dim under the help overlay (玄 40% / 墨 40%).
    pub dim: Color32,
}

impl Palette {
    /// 玄墨 (dark) — the original mockup values.
    pub fn mo() -> Self {
        Self {
            base: Color32::from_rgb(11, 12, 14),
            surface: Color32::from_rgb(20, 22, 26),
            ink: Color32::from_rgb(232, 236, 239),
            gray: Color32::from_rgb(138, 146, 155),
            zhu: Color32::from_rgb(201, 59, 46),
            qing: Color32::from_rgb(127, 166, 163),
            stone_black: Color32::from_rgb(32, 35, 40),
            stone_white: Color32::from_rgb(233, 237, 239),
            black_rim: Some(0.28),
            white_rim: None,
            wash_white: Color32::from_rgb(232, 236, 239),
            wash_black: Color32::BLACK,
            hairline_alpha: 0.08,
            dim: Color32::from_rgb(11, 12, 14),
        }
    }

    /// 雪宣 (light).
    pub fn xuan() -> Self {
        Self {
            base: Color32::from_rgb(244, 243, 239),
            surface: Color32::from_rgb(235, 234, 228),
            ink: Color32::from_rgb(28, 30, 34),
            gray: Color32::from_rgb(107, 112, 118),
            zhu: Color32::from_rgb(184, 53, 39),
            qing: Color32::from_rgb(78, 122, 119),
            stone_black: Color32::from_rgb(28, 30, 34),
            stone_white: Color32::from_rgb(252, 252, 249),
            black_rim: None,
            white_rim: Some(0.12),
            wash_white: Color32::WHITE,
            wash_black: Color32::from_rgb(28, 30, 34),
            hairline_alpha: 0.10,
            dim: Color32::from_rgb(28, 30, 34),
        }
    }
}

/// `color` with alpha replaced by `a` (0.0..=1.0).
pub fn with_alpha(color: Color32, a: f32) -> Color32 {
    let byte = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), byte)
}

/// Resolved font families; falls back to egui defaults when the Windows
/// font files are missing.
#[derive(Clone)]
pub struct Fonts {
    /// SimSun (`simsun.ttc`): title, candidate ordinals, kifu strip.
    pub song: FontFamily,
    /// Microsoft YaHei Light (`msyhl.ttc`): body text and digits.
    pub hei: FontFamily,
}

impl Fonts {
    pub fn fallback() -> Self {
        Self {
            song: FontFamily::Proportional,
            hei: FontFamily::Proportional,
        }
    }
}

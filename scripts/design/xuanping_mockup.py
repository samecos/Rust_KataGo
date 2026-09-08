# -*- coding: utf-8 -*-
"""玄枰 UI 静态视觉稿 —— 分析视图(玄墨主题)。

2x 超采样绘制后 LANCZOS 降采样,保证发丝线与细字可用。
仅依赖 Pillow;字体用系统 SimSun(代宋体)与 Microsoft YaHei Light(代黑体 Light)。
"""

import math
import os

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

S = 2  # 超采样倍率
W, H = 1440, 900


def P(v):
    return int(round(v * S))


# ---- 令牌(玄墨) ----
XUAN = (11, 12, 14)      # 玄 · 底
DAI = (20, 22, 26)       # 黛 · 枰面
YUE = (232, 236, 239)    # 月白 · 正文
HUI = (138, 146, 155)    # 灰 · 次级
ZHU = (201, 59, 46)      # 朱 · 印(全局唯一)
QING = (127, 166, 163)   # 青 · 机息

FONT_DIR = r"C:\Windows\Fonts"


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), P(size))


def SONG(s):
    return font("simsun.ttc", s)


def HEI(s):
    return font("msyhl.ttc", s)


def layer():
    return Image.new("RGBA", base.size, (0, 0, 0, 0))


def A(a):
    """0..1 透明度 -> 0..255"""
    return int(round(a * 255))


def text(l, x, y, s, f, color, alpha=1.0, anchor="la", spacing=0.0):
    d = ImageDraw.Draw(l)
    fill = color + (A(alpha),)
    if spacing:
        cx = P(x)
        for ch in s:
            d.text((cx, P(y)), ch, font=f, fill=fill, anchor=anchor)
            cx += int(f.size * (1 + spacing))
    else:
        d.text((P(x), P(y)), s, font=f, fill=fill, anchor=anchor)


def vtext(l, x, y, s, f, color, alpha, pitch):
    """竖排:逐字堆叠,x 为列中心。"""
    d = ImageDraw.Draw(l)
    cy = P(y)
    for ch in s:
        d.text((P(x), cy), ch, font=f, fill=color + (A(alpha),), anchor="ma")
        cy += P(pitch)


base = Image.new("RGBA", (P(W), P(H)), XUAN + (255,))

# ---- 棋盘几何 ----
bx, by, bs = 140, 162, 576          # 黛色枰面
pad = 18
gx, gy = bx + pad, by + pad         # 格线原点
cell = (bs - 2 * pad) / 18          # 30px
LETTERS = "ABCDEFGHJKLMNOPQRST"


def gp(c, r):
    """交叉点(0 起始列,行) -> 像素(final 尺度)"""
    return gx + c * cell, gy + r * cell


# ---- 枰面 ----
l = layer()
d = ImageDraw.Draw(l)
d.rectangle([P(bx), P(by), P(bx + bs), P(by + bs)], fill=DAI + (255,),
            outline=YUE + (A(0.08),), width=P(1))
base = Image.alpha_composite(base, l)

# ---- 墨五色 · 领地晕染(格线下、子上) ----
wash = layer()
dw = ImageDraw.Draw(wash)


def blob(cx, cy, r, color, a_max):
    """五阶同心晕染:最大最淡 -> 最小最浓。"""
    for i, (kr, ka) in enumerate(zip([1.0, 0.82, 0.64, 0.46, 0.30],
                                     [0.18, 0.36, 0.55, 0.75, 1.0])):
        rr = r * kr
        dw.ellipse([P(cx - rr), P(cy - rr), P(cx + rr), P(cy + rr)],
                   fill=color + (A(a_max * ka),))


# 白方领地:左下月白纱;黑方领地:上边与右侧玄染
for (c, r, rad) in [(4, 11, 95), (5, 14, 115), (7, 16, 85), (2, 7, 70), (3, 17, 60)]:
    cx, cy = gp(c, r)
    blob(cx, cy, rad, YUE, 0.30)
for (c, r, rad) in [(6, 2, 105), (11, 2, 110), (15, 6, 95), (16, 10, 75), (9, 1, 70)]:
    cx, cy = gp(c, r)
    blob(cx, cy, rad, (0, 0, 0), 0.45)
wash = wash.filter(ImageFilter.GaussianBlur(P(22)))
base = Image.alpha_composite(base, wash)

# ---- 格线与星位 ----
l = layer()
d = ImageDraw.Draw(l)
for i in range(19):
    x = gx + i * cell
    y = gy + i * cell
    d.line([P(x), P(gy), P(x), P(gy + 18 * cell)], fill=YUE + (A(0.22),), width=P(1))
    d.line([P(gx), P(y), P(gx + 18 * cell), P(y)], fill=YUE + (A(0.22),), width=P(1))
for c in (3, 9, 15):
    for r in (3, 9, 15):
        cx, cy = gp(c, r)
        d.ellipse([P(cx - 1.5), P(cy - 1.5), P(cx + 1.5), P(cy + 1.5)],
                  fill=YUE + (A(0.35),))
base = Image.alpha_composite(base, l)

# ---- 子 ----
black = [(3, 3), (9, 3), (15, 3), (15, 9), (16, 4), (14, 6), (16, 14),
         (5, 4), (10, 10), (13, 13), (9, 9), (16, 9)]
white = [(3, 9), (4, 14), (9, 15), (15, 15), (3, 15), (9, 16),
         (14, 15), (16, 16), (4, 5), (14, 4)]
l = layer()
d = ImageDraw.Draw(l)
SR = 13.5
for (c, r) in black:
    cx, cy = gp(c, r)
    bb = [P(cx - SR), P(cy - SR), P(cx + SR), P(cy + SR)]
    d.ellipse(bb, fill=(32, 35, 40, 255))
    d.ellipse(bb, outline=YUE + (A(0.28),), width=P(0.75))  # 整圈 0.75px 冷光
for (c, r) in white:
    cx, cy = gp(c, r)
    bb = [P(cx - SR), P(cy - SR), P(cx + SR), P(cy + SR)]
    d.ellipse(bb, fill=(233, 237, 239, 255))
base = Image.alpha_composite(base, l)

# ---- 候选点(焦/浓/重) ----
cands = [((8, 12), "62.4", 0.72, 1.5),
         ((11, 8), "61.1", 0.50, 1.25),
         ((6, 6), "60.3", 0.34, 1.0)]
l = layer()
d = ImageDraw.Draw(l)
for (c, r), wr, alpha, wd in cands:
    cx, cy = gp(c, r)
    d.ellipse([P(cx - 17.5), P(cy - 17.5), P(cx + 17.5), P(cy + 17.5)],
              outline=YUE + (A(alpha),), width=P(wd))
    text(l, cx, cy + 24, wr, HEI(11), YUE, 0.60, anchor="ma")
base = Image.alpha_composite(base, l)

# ---- 分隔发丝线 ----
l = layer()
d = ImageDraw.Draw(l)
d.line([P(872), P(64), P(872), P(836)], fill=YUE + (A(0.08),), width=P(1))
d.line([P(1304), P(64), P(1304), P(836)], fill=YUE + (A(0.08),), width=P(1))
base = Image.alpha_composite(base, l)

# ---- 远山胜率图 ----
ggx, ggy, gw, gh = 904, 128, 376, 72
wmin, wmax = 35.0, 70.0
wrs = []
for m in range(46):
    if m <= 10:
        v = 50 - 0.2 * m
    elif m <= 20:
        v = 48 - 0.2 * (m - 10)
    elif m <= 30:
        v = 46 + 0.1 * (m - 20)
    elif m <= 37:
        v = 47 - 0.4 * (m - 30)
    elif m == 38:
        v = 61.8
    else:
        v = 61.8 + 0.086 * (m - 38)
    v += 1.1 * math.sin(m * 2.3) + 0.7 * math.sin(m * 0.71)
    wrs.append(v)


def wpt(m):
    v = wrs[m]
    return ggx + gw * m / 45, ggy + gh * (wmax - v) / (wmax - wmin)


# 线下渐变填充(12% -> 0)
mask = Image.new("L", base.size, 0)
dm = ImageDraw.Draw(mask)
poly = [(P(wpt(m)[0]), P(wpt(m)[1])) for m in range(46)]
poly += [(P(ggx + gw), P(ggy + gh)), (P(ggx), P(ggy + gh))]
dm.polygon(poly, fill=255)
grad = Image.new("L", base.size, 0)
dg = ImageDraw.Draw(grad)
for yy in range(P(ggy), P(ggy + gh)):
    a = int(A(0.12) * (1 - (yy - P(ggy)) / P(gh)))
    dg.line([(P(ggx), yy), (P(ggx + gw), yy)], fill=a)
fill_img = layer()
fill_img.paste(Image.new("RGBA", base.size, YUE + (255,)), (0, 0),
               ImageChops.multiply(mask, grad))
base = Image.alpha_composite(base, fill_img)

l = layer()
d = ImageDraw.Draw(l)
y50 = ggy + gh * (wmax - 50) / (wmax - wmin)
d.line([P(ggx), P(y50), P(ggx + gw), P(y50)], fill=YUE + (A(0.08),), width=P(1))
d.line([(P(wpt(m)[0]), P(wpt(m)[1])) for m in range(46)],
       fill=YUE + (A(0.85),), width=P(1), joint="curve")
# 关键手:全局唯一一点朱(9px 刻度 + 顶点小粒)
kx, ky = wpt(38)
d.line([P(kx), P(ky - 4.5), P(kx), P(ky + 4.5)], fill=ZHU + (A(0.95),), width=P(1.5))
d.ellipse([P(kx - 1.5), P(ky - 8.5), P(kx + 1.5), P(ky - 5.5)], fill=ZHU + (A(0.95),))
# 当前手竖刻
cx_, cy_ = wpt(45)
d.line([P(cx_), P(cy_ - 3), P(cx_), P(cy_ + 3)], fill=YUE + (A(0.80),), width=P(1.5))
base = Image.alpha_composite(base, l)

# ---- 右列文字 ----
l = layer()
text(l, ggx, 92, "胜率", HEI(11), HUI, 0.9)
text(l, ggx + gw, 90, "析", HEI(13), YUE, 0.9, anchor="ra")

rows = [("一", "J07", "62.4", "2.1k"),
        ("二", "M11", "61.1", "1.6k"),
        ("三", "G13", "60.3", "1.1k")]
ry = 232
d = ImageDraw.Draw(l)
for i, (seq, coord, wr, vis) in enumerate(rows):
    y = ry + i * 36
    text(l, ggx, y + 8, seq, SONG(13), YUE, 0.85)
    text(l, ggx + 36, y + 8, coord, HEI(13), YUE, 0.9)
    text(l, ggx + 236, y + 8, wr, HEI(13), YUE, 0.9, anchor="ra")
    text(l, ggx + gw, y + 10, vis, HEI(11), HUI, 0.9, anchor="ra")
    if i:
        d.line([P(ggx), P(y), P(ggx + gw), P(y)], fill=YUE + (A(0.08),), width=P(1))
d.line([P(ggx), P(ry + 3 * 36), P(ggx + gw), P(ry + 3 * 36)],
       fill=YUE + (A(0.08),), width=P(1))

# 引擎读数行
xcur = ggx
for label, val in [("胜率", "62.4"), ("目差", "+3.5"), ("访问", "12.8k")]:
    text(l, xcur, 388, label, HEI(11), HUI, 0.9)
    xcur += ImageDraw.Draw(l).textlength(label, font=HEI(11)) / S + 8
    text(l, xcur, 384, val, HEI(15), YUE, 0.92)
    xcur += ImageDraw.Draw(l).textlength(val, font=HEI(15)) / S + 28

# 列底状态行
text(l, ggx, 850, "KataGo · cudabackend · b11", HEI(10), HUI, 0.7)
status = "思考　访问 12.8k"
tw = ImageDraw.Draw(l).textlength(status, font=HEI(11)) / S
text(l, ggx + gw, 850, status, HEI(11), HUI, 0.9, anchor="ra")
dotx = ggx + gw - tw - 14
d.ellipse([P(dotx - 2), P(855.5 - 2), P(dotx + 2), P(855.5 + 2)],
          fill=QING + (A(0.55),))  # 呼吸点(一息中段)
base = Image.alpha_composite(base, l)

# ---- 右缘竖排棋谱 ----
l = layer()
text(l, 1398, 830, "谱", SONG(11), HUI, 0.6, anchor="ra")
kifu = [("四十五", 1.0), ("四十四", 0.4), ("四十三", 0.4), ("四十二", 0.4)]
for i, (num, alpha) in enumerate(kifu):
    x = 1398 - i * 27
    vtext(l, x, 120, num, SONG(12.5), YUE, alpha, 16)
base = Image.alpha_composite(base, l)

# ---- 题名 ----
l = layer()
text(l, 32, 30, "玄枰", SONG(15), YUE, 0.4, spacing=0.12)
base = Image.alpha_composite(base, l)

# ---- 降采样输出 ----
out = base.convert("RGB").resize((W, H), Image.LANCZOS)
os.makedirs("docs/design", exist_ok=True)
path = "docs/design/xuanping-analysis-v1.png"
out.save(path)
print(path)

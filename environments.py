"""
environments.py — Multi-Layer Procedural Painterly Ecosystem Generator
======================================================================

Procedurally paints and caches high-resolution, painterly-textured environment
layers into `food_chain_project/assets/`. Inspired by Studio Ghibli nature
backgrounds and BBC illustrated wildlife documentaries:
  - Soft golden atmospheric sunrise/morning skies (no clipped white glare)
  - Sculpted billowing backlit cumulus clouds with warm gold rims & cool undersides
  - Layered rolling hills with organic woodland belts & misty valley haze
  - Rich stratified subterranean soil with pebbles, humus, and root filaments
  - Lush freshwater pond with grounded mossy wet boulders, cattails & lily pads
  - Sun-dappled rocky meadow clearing with crevices, fallen leaves & grass tufts
  - Sweeping valley panorama with perspective-widening river & forested hills
  - Serene luminous botanical backdrop for the educational food-chain finale
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import cairo
import cv2
import numpy as np

from animation import clamp, hash1d, lerp, lerp_rgb
from drawing_engine import (
    CairoLayer,
    apply_painterly_finish,
    composite_straight_bgra,
    draw_organic_poly,
    draw_soft_shadow,
    draw_tapered_ribbon,
    fbm_noise_2d,
    set_rgb,
)


def _paint_sky_gradient(
    w: int,
    h: int,
    stops: List[Tuple[float, Tuple[int, int, int]]],
    sun_xy: Tuple[float, float] = (0.78, 0.24),
    sun_glow_strength: float = 0.72,
) -> np.ndarray:
    """Create a rich, unclipped atmospheric sky gradient with warm golden solar aureole."""
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    sky = np.zeros((h, w, 3), dtype=np.float32)

    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        mask = (yy >= p0) & (yy <= p1)
        if not np.any(mask):
            continue
        u = np.clip((yy[mask] - p0) / max(p1 - p0, 1e-6), 0.0, 1.0)
        u = u * u * (3.0 - 2.0 * u)
        c0_bgr = np.array([c0[2], c0[1], c0[0]], dtype=np.float32)
        c1_bgr = np.array([c1[2], c1[1], c1[0]], dtype=np.float32)
        col = c0_bgr[None, :] * (1.0 - u[:, None]) + c1_bgr[None, :] * u[:, None]
        sky[mask, :, :] = col[:, None, :]

    # Delicate warm golden Mie haze + small realistic solar disc (soft-rolled so it never clips ugly!)
    xg, yg = np.meshgrid(
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        np.linspace(0.0, 1.0, h, dtype=np.float32),
    )
    aspect = w / float(h)
    dx = (xg - sun_xy[0]) * aspect
    dy = yg - sun_xy[1]
    dist = np.sqrt(dx * dx + dy * dy)

    wide_haze = np.exp(-dist * 1.95) * 0.42 * sun_glow_strength
    inner_aura = np.exp(-dist * dist * 28.0) * 0.44 * sun_glow_strength
    sun_disc = np.exp(-dist * dist * 950.0) * 0.75 * sun_glow_strength

    # Screen-like soft additive blend that approaches warm gold instead of harsh flat white
    warm_gold = np.array([150.0, 212.0, 252.0], dtype=np.float32)[None, None, :]
    sun_cream = np.array([218.0, 244.0, 255.0], dtype=np.float32)[None, None, :]

    w_mix = np.clip(wide_haze + inner_aura, 0.0, 0.85)[:, :, None]
    sky = sky * (1.0 - w_mix * 0.55) + warm_gold * (w_mix * 0.65)
    c_mix = np.clip(sun_disc, 0.0, 0.92)[:, :, None]
    sky = sky * (1.0 - c_mix) + sun_cream * c_mix

    return np.clip(sky, 0.0, 255.0).astype(np.uint8)


def _draw_sculpted_clouds(
    bgr: np.ndarray,
    sun_xy: Tuple[float, float] = (0.78, 0.24),
    seed: int = 101,
    y_min_frac: float = 0.08,
    y_max_frac: float = 0.48,
    count: int = 7,
) -> np.ndarray:
    """
    Paint lush Studio Ghibli-inspired billowing cumulus clouds with multi-lobed
    sculpted shapes, warm cream-gold sunlit tops, and soft periwinkle-lavender bases.
    """
    h, w = bgr.shape[:2]
    layer = CairoLayer(w, h)
    ctx = layer.ctx

    for c_i in range(count):
        cx = w * (0.06 + 0.88 * ((c_i + 0.25 + 0.5 * hash1d(seed * 19 + c_i * 7)) / float(count)))
        cy = h * lerp(y_min_frac, y_max_frac, hash1d(seed * 29 + c_i * 11))
        cw = w * (0.14 + 0.14 * hash1d(seed * 37 + c_i * 13))
        ch = cw * (0.28 + 0.14 * hash1d(seed * 41 + c_i * 17))
        sun_dir = 1.0 if (sun_xy[0] * w - cx) >= 0 else -1.0

        # Build billowing cloud puff hierarchy (cool base -> warm sunlit puffs -> golden rim)
        n_puffs = 12
        puff_data = []
        for p in range(n_puffs):
            u = (p / float(n_puffs - 1)) * 2.0 - 1.0  # -1..1 across width
            px = cx + u * cw * 0.48 + (hash1d(seed * 53 + c_i * 31 + p) - 0.5) * cw * 0.12
            dome = math.sqrt(max(0.05, 1.0 - u * u * 0.85))
            py = cy - dome * ch * (0.35 + 0.55 * hash1d(seed * 61 + c_i * 37 + p)) + ch * 0.22
            pr = ch * (0.38 + 0.42 * dome * (0.65 + 0.35 * hash1d(seed * 67 + c_i * 41 + p)))
            puff_data.append((px, py, pr))

        # Pass 1: Cool soft lavender-blue shadowed cloud undersides
        for px, py, pr in puff_data:
            pat = cairo.RadialGradient(px, py + pr * 0.35, pr * 0.15, px, py + pr * 0.2, pr * 1.12)
            pat.add_color_stop_rgba(0.0, 0.72, 0.78, 0.88, 0.52)
            pat.add_color_stop_rgba(0.65, 0.76, 0.81, 0.90, 0.34)
            pat.add_color_stop_rgba(1.0, 0.76, 0.81, 0.90, 0.0)
            ctx.set_source(pat)
            ctx.save()
            ctx.translate(px, py + ch * 0.14)
            ctx.scale(1.22, 0.72)
            ctx.arc(0, 0, pr, 0, math.tau)
            ctx.fill()
            ctx.restore()

        # Pass 2: Main luminous ivory-white sculpted cumulus body
        for px, py, pr in puff_data:
            lx = px + sun_dir * pr * 0.25
            ly = py - pr * 0.35
            pat = cairo.RadialGradient(lx, ly, pr * 0.08, px, py, pr * 1.05)
            pat.add_color_stop_rgba(0.0, 1.00, 0.98, 0.93, 0.82)
            pat.add_color_stop_rgba(0.55, 0.96, 0.95, 0.92, 0.62)
            pat.add_color_stop_rgba(0.88, 0.84, 0.86, 0.91, 0.26)
            pat.add_color_stop_rgba(1.0, 0.84, 0.86, 0.91, 0.0)
            ctx.set_source(pat)
            ctx.save()
            ctx.translate(px, py)
            ctx.scale(1.16, 0.84)
            ctx.arc(0, 0, pr, 0, math.tau)
            ctx.fill()
            ctx.restore()

        # Pass 3: Sun-facing warm golden-peach rim highlight
        for px, py, pr in puff_data[::2]:
            rx = px + sun_dir * pr * 0.22
            ry = py - pr * 0.32
            pat = cairo.RadialGradient(rx, ry, pr * 0.05, rx, ry, pr * 0.75)
            pat.add_color_stop_rgba(0.0, 1.00, 0.95, 0.82, 0.62)
            pat.add_color_stop_rgba(0.60, 1.00, 0.93, 0.80, 0.22)
            pat.add_color_stop_rgba(1.0, 1.00, 0.93, 0.80, 0.0)
            ctx.set_source(pat)
            ctx.arc(rx, ry, pr * 0.75, 0, math.tau)
            ctx.fill()

    # Subtle FBM organic wisp modulation so cloud edges feel genuinely painted, never geometric circles
    cloud_bgra = layer.to_straight_bgra()
    wisp = fbm_noise_2d(h, w, octaves=4, base_grid=(10, 18), persistence=0.55, seed=seed + 303)
    a_mod = np.clip((cloud_bgra[:, :, 3].astype(np.float32) / 255.0) * (0.72 + 0.45 * wisp), 0.0, 1.0)
    a_mod = cv2.GaussianBlur(a_mod, (0, 0), 3.5)
    cloud_bgra[:, :, 3] = np.clip(a_mod * 255.0, 0, 255).astype(np.uint8)
    composite_straight_bgra(bgr, cloud_bgra, opacity=0.92)
    return bgr


def _draw_painted_rock(
    ctx: cairo.Context,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    seed: int = 1,
    base_rgb: Tuple[float, float, float] = (0.43, 0.45, 0.41),
    moss_amount: float = 0.45,
    wet_bottom: bool = False,
    sun_right: bool = True,
) -> None:
    """Paint a sculpted 3D-lit weathered boulder with mineral planes, cracks, moss & soft rim."""
    pts = []
    n_pts = 16
    for i in range(n_pts):
        ang = (i / float(n_pts)) * math.tau
        rad_jit = 0.85 + 0.26 * hash1d(seed * 41 + i * 7)
        y_scale = 0.68 if math.sin(ang) > 0.22 else 1.06
        px = cx + math.cos(ang) * rx * rad_jit
        py = cy + math.sin(ang) * ry * rad_jit * y_scale
        pts.append((px, py))

    # Ground contact shadow
    draw_soft_shadow(
        ctx,
        cx - (rx * 0.14 if sun_right else -rx * 0.14),
        cy + ry * 0.62,
        rx * 1.15,
        ry * 0.36,
        alpha=0.45,
    )

    ctx.save()
    draw_organic_poly(ctx, pts, closed=True)
    ctx.clip_preserve()

    # Directional spherical lighting
    lx = cx + (rx * 0.40 if sun_right else -rx * 0.40)
    ly = cy - ry * 0.48
    pat = cairo.RadialGradient(lx, ly, rx * 0.06, cx, cy + ry * 0.15, max(rx, ry) * 1.50)
    lit_c = (min(1.0, base_rgb[0] * 1.42 + 0.08), min(1.0, base_rgb[1] * 1.38 + 0.06), min(1.0, base_rgb[2] * 1.24 + 0.03))
    mid_c = base_rgb
    shd_c = (base_rgb[0] * 0.36, base_rgb[1] * 0.42, base_rgb[2] * 0.48)
    pat.add_color_stop_rgb(0.0, *lit_c)
    pat.add_color_stop_rgb(0.46, *mid_c)
    pat.add_color_stop_rgb(1.0, *shd_c)
    ctx.set_source(pat)
    ctx.fill()

    # Chiseled rock strata & mineral facets
    for k in range(24):
        fx = cx + (hash1d(seed * 97 + k * 5) - 0.5) * rx * 1.7
        fy = cy + (hash1d(seed * 53 + k * 11) - 0.5) * ry * 1.4
        frx = rx * (0.12 + 0.24 * hash1d(seed * 19 + k * 3))
        fry = ry * (0.07 + 0.14 * hash1d(seed * 23 + k * 9))
        bright = hash1d(seed * 71 + k * 13) > 0.48
        ctx.save()
        ctx.translate(fx, fy)
        ctx.rotate((hash1d(seed * 31 + k) - 0.5) * 1.1)
        ctx.scale(frx, fry)
        ctx.arc(0, 0, 1.0, 0, math.tau)
        ctx.set_source_rgba(*(lit_c if bright else shd_c), 0.22 if bright else 0.28)
        ctx.fill()
        ctx.restore()

    # Crisp dark rock crevices
    ctx.set_line_width(max(1.1, rx * 0.011))
    ctx.set_source_rgba(0.10, 0.12, 0.10, 0.38)
    for c_i in range(4):
        sx = cx + (hash1d(seed * 131 + c_i * 7) - 0.5) * rx * 0.95
        sy = cy + (hash1d(seed * 149 + c_i * 11) - 0.5) * ry * 0.65
        ctx.move_to(sx, sy)
        for seg in range(3):
            sx += (hash1d(seed * 17 + c_i * 29 + seg * 5) - 0.5) * rx * 0.42
            sy += (hash1d(seed * 23 + c_i * 37 + seg * 7) - 0.2) * ry * 0.32
            ctx.line_to(sx, sy)
        ctx.stroke()

    # Soft sunlit olive-emerald moss blankets clinging to upper shoulders
    if moss_amount > 0.05:
        n_moss = int(6 + moss_amount * 14)
        for m in range(n_moss):
            mx = cx + (hash1d(seed * 211 + m * 7) - 0.5) * rx * 1.3
            my = cy - ry * (0.10 + 0.58 * hash1d(seed * 223 + m * 13))
            mr = rx * (0.08 + 0.14 * hash1d(seed * 229 + m * 17))
            mpat = cairo.RadialGradient(mx + mr * 0.2, my - mr * 0.25, mr * 0.1, mx, my, mr)
            mpat.add_color_stop_rgba(0.0, 0.44, 0.62, 0.22, 0.78 * moss_amount)
            mpat.add_color_stop_rgba(0.6, 0.24, 0.40, 0.15, 0.55 * moss_amount)
            mpat.add_color_stop_rgba(1.0, 0.16, 0.28, 0.11, 0.0)
            ctx.set_source(mpat)
            ctx.save()
            ctx.translate(mx, my)
            ctx.scale(1.35, 0.62)
            ctx.arc(0, 0, mr, 0, math.tau)
            ctx.fill()
            ctx.restore()

    # Wet glistening darkened bottom band & specular sheen on pond rocks
    if wet_bottom:
        wgrad = cairo.LinearGradient(cx, cy + ry * 0.06, cx, cy + ry * 0.85)
        wgrad.add_color_stop_rgba(0.0, 0.04, 0.09, 0.08, 0.0)
        wgrad.add_color_stop_rgba(0.45, 0.04, 0.09, 0.08, 0.48)
        wgrad.add_color_stop_rgba(1.0, 0.03, 0.07, 0.06, 0.70)
        ctx.set_source(wgrad)
        ctx.rectangle(cx - rx * 1.5, cy, rx * 3.0, ry * 1.3)
        ctx.fill()

        spat = cairo.RadialGradient(lx, ly + ry * 0.15, rx * 0.02, lx, ly + ry * 0.15, rx * 0.55)
        spat.add_color_stop_rgba(0.0, 0.96, 0.99, 0.97, 0.48)
        spat.add_color_stop_rgba(0.45, 0.82, 0.92, 0.90, 0.16)
        spat.add_color_stop_rgba(1.0, 0.82, 0.92, 0.90, 0.0)
        ctx.set_source(spat)
        ctx.arc(lx, ly + ry * 0.15, rx * 0.55, 0, math.tau)
        ctx.fill()

    # Warm sunlit upper-right rim edge
    ctx.restore()
    draw_organic_poly(ctx, pts, closed=True)
    ctx.set_source_rgba(0.14, 0.15, 0.12, 0.34)
    ctx.set_line_width(max(1.0, rx * 0.008))
    ctx.stroke()


def _draw_woodland_treeline(
    ctx: cairo.Context,
    ridge_pts: List[Tuple[float, float]],
    height_scale: float,
    lit_rgb: Tuple[float, float, float],
    mid_rgb: Tuple[float, float, float],
    shd_rgb: Tuple[float, float, float],
    seed: int,
    alpha: float = 0.92,
) -> None:
    """Paint continuous, organic silhouetted deciduous woodland canopies along a hillside."""
    n = len(ridge_pts)
    # Build several natural grove stretches along the ridge (not isolated round balls!)
    groves = [
        (0.02, 0.24),
        (0.32, 0.52),
        (0.62, 0.96),
    ]
    for g_idx, (u0, u1) in enumerate(groves):
        i0 = int(u0 * (n - 1))
        i1 = int(u1 * (n - 1))
        # Subtle irregular overlapping canopy puffs forming rich forest silhouettes
        n_trees = max(8, int((u1 - u0) * 48))
        # First pass: shaded forest mass
        for t_i in range(n_trees):
            u = lerp(u0, u1, t_i / float(max(1, n_trees - 1)))
            ri = min(n - 1, int(u * (n - 1)))
            bx, by = ridge_pts[ri]
            # Taper grove edges naturally
            edge_fade = math.sin(clamp((u - u0) / max(u1 - u0, 1e-4)) * math.pi) ** 0.45
            th = (34.0 + 46.0 * hash1d(seed * 31 + g_idx * 97 + t_i * 7)) * height_scale * (0.45 + 0.55 * edge_fade)
            tw = th * (0.72 + 0.45 * hash1d(seed * 41 + g_idx * 89 + t_i * 11))

            # Multiple clustered sub-foliage lobes per tree crown
            for lobe in range(4):
                lx = bx + (hash1d(seed * 53 + t_i * 17 + lobe * 3) - 0.5) * tw * 1.15
                ly = by + 6.0 - th * (0.35 + 0.65 * hash1d(seed * 59 + t_i * 19 + lobe * 5))
                lr = tw * (0.40 + 0.30 * hash1d(seed * 61 + t_i * 23 + lobe * 7))
                pat = cairo.RadialGradient(lx + lr * 0.28, ly - lr * 0.32, lr * 0.08, lx, ly + lr * 0.15, lr)
                pat.add_color_stop_rgba(0.0, *lit_rgb, alpha * 0.85)
                pat.add_color_stop_rgba(0.55, *mid_rgb, alpha * 0.92)
                pat.add_color_stop_rgba(1.0, *shd_rgb, 0.0)
                ctx.set_source(pat)
                ctx.save()
                ctx.translate(lx, ly)
                ctx.scale(1.15, 0.88)
                ctx.arc(0, 0, lr, 0, math.tau)
                ctx.fill()
                ctx.restore()


def _draw_wildflower(ctx: cairo.Context, x: float, y: float, scale: float, kind: int) -> None:
    """Tiny natural meadow daisy / buttercup / clover blossom."""
    ctx.save()
    ctx.translate(x, y)
    if kind == 0:
        for i in range(8):
            a = i * (math.tau / 8.0)
            ctx.save()
            ctx.rotate(a)
            ctx.scale(1.0, 0.42)
            ctx.arc(5.5 * scale, 0, 4.2 * scale, 0, math.tau)
            ctx.set_source_rgba(0.98, 0.97, 0.91, 0.88)
            ctx.fill()
            ctx.restore()
        ctx.arc(0, 0, 3.0 * scale, 0, math.tau)
        ctx.set_source_rgba(0.94, 0.74, 0.16, 0.95)
        ctx.fill()
    elif kind == 1:
        for i in range(5):
            a = i * (math.tau / 5.0)
            ctx.arc(math.cos(a) * 3.8 * scale, math.sin(a) * 3.8 * scale, 3.3 * scale, 0, math.tau)
            ctx.set_source_rgba(0.98, 0.82, 0.20, 0.92)
            ctx.fill()
        ctx.arc(0, 0, 2.0 * scale, 0, math.tau)
        ctx.set_source_rgba(0.76, 0.48, 0.08, 0.94)
        ctx.fill()
    else:
        for i in range(6):
            a = i * (math.tau / 6.0)
            ctx.arc(math.cos(a) * 3.0 * scale, math.sin(a) * 3.0 * scale, 2.6 * scale, 0, math.tau)
            ctx.set_source_rgba(0.86, 0.58, 0.72, 0.84)
            ctx.fill()
    ctx.restore()


def _draw_fallen_leaf(
    ctx: cairo.Context,
    x: float,
    y: float,
    length: float,
    angle: float,
    hue_mix: float,
    alpha: float = 0.92,
) -> None:
    """Paint a realistic curled dry grassland leaf with central midrib and soft shadow."""
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(angle)
    draw_soft_shadow(ctx, 2.0, 4.5, length * 0.52, length * 0.18, alpha=0.32 * alpha)

    ctx.new_path()
    ctx.move_to(-length * 0.5, 0.0)
    ctx.curve_to(-length * 0.24, -length * 0.34, length * 0.22, -length * 0.30, length * 0.52, 0.0)
    ctx.curve_to(length * 0.20, length * 0.28, -length * 0.26, length * 0.30, -length * 0.5, 0.0)
    ctx.close_path()

    c_lit = lerp_rgb((0.74, 0.48, 0.20), (0.56, 0.32, 0.14), hue_mix)
    c_shd = lerp_rgb((0.38, 0.21, 0.09), (0.26, 0.15, 0.07), hue_mix)
    pat = cairo.LinearGradient(-length * 0.4, -length * 0.25, length * 0.4, length * 0.25)
    pat.add_color_stop_rgba(0.0, *c_lit, alpha)
    pat.add_color_stop_rgba(1.0, *c_shd, alpha)
    ctx.set_source(pat)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.22, 0.12, 0.06, 0.55 * alpha)
    ctx.set_line_width(1.0)
    ctx.stroke()

    ctx.move_to(-length * 0.55, 0.0)
    ctx.line_to(length * 0.44, 0.0)
    ctx.set_source_rgba(0.86, 0.68, 0.42, 0.52 * alpha)
    ctx.set_line_width(1.0)
    ctx.stroke()
    ctx.restore()


# =============================================================================
# Environment Builders (Saved & Cached in assets/)
# =============================================================================

def build_morning_sky(w: int = 2400, h: int = 1350) -> np.ndarray:
    """Early morning golden-cerulean grassland sky with sculpted backlit cumulus clouds."""
    sky = _paint_sky_gradient(
        w,
        h,
        stops=[
            (0.00, (66, 122, 184)),
            (0.26, (118, 170, 214)),
            (0.48, (186, 212, 224)),
            (0.66, (244, 224, 184)),
            (0.84, (248, 206, 152)),
            (1.00, (236, 194, 144)),
        ],
        sun_xy=(0.76, 0.26),
        sun_glow_strength=0.78,
    )
    sky = _draw_sculpted_clouds(sky, sun_xy=(0.76, 0.26), seed=105, y_min_frac=0.14, y_max_frac=0.44, count=8)
    return apply_painterly_finish(sky, brush_strength=5.5, angle_deg=-5.0, seed=31)


def build_distant_valley(w: int = 2400, h: int = 1350) -> np.ndarray:
    """Multi-tier misty rolling grassland hills & organic woodland belts (straight BGRA)."""
    layer = CairoLayer(w, h)
    ctx = layer.ctx

    tiers = [
        # (base_y, amp1, amp2, freq1, phase, top_rgb, bot_rgb, alpha, woodland_scale)
        (0.55, 68, 26, 2.2, 0.4, (0.46, 0.60, 0.61), (0.37, 0.52, 0.50), 0.88, 0.0),
        (0.62, 58, 22, 2.9, 2.1, (0.38, 0.55, 0.36), (0.25, 0.42, 0.25), 0.96, 0.72),
        (0.70, 44, 16, 2.5, 4.0, (0.44, 0.62, 0.27), (0.21, 0.38, 0.15), 1.00, 1.05),
    ]

    for idx, (by_frac, a1, a2, f1, ph, c_top, c_bot, alpha, tree_sc) in enumerate(tiers):
        by = h * by_frac
        ridge_pts = []
        steps = 90
        for i in range(steps + 1):
            u = i / float(steps)
            x = u * w
            y = by - a1 * math.sin(u * math.tau * f1 * 0.35 + ph) - a2 * math.cos(u * math.tau * f1 * 0.9 + ph * 1.7)
            ridge_pts.append((x, y))

        # Woodland silhouette behind/along crest before filling hill slope
        if tree_sc > 0.0:
            t_lit = lerp_rgb(c_top, (0.68, 0.76, 0.34), 0.42)
            t_mid = lerp_rgb(c_top, c_bot, 0.45)
            t_shd = lerp_rgb(c_bot, (0.12, 0.22, 0.14), 0.50)
            _draw_woodland_treeline(ctx, ridge_pts, tree_sc, t_lit, t_mid, t_shd, seed=300 + idx * 101, alpha=0.90)

        ctx.new_path()
        ctx.move_to(0, h)
        for rx, ry in ridge_pts:
            ctx.line_to(rx, ry)
        ctx.line_to(w, h)
        ctx.close_path()

        pat = cairo.LinearGradient(0, by - a1 - 20, 0, h)
        pat.add_color_stop_rgba(0.0, *c_top, alpha)
        pat.add_color_stop_rgba(0.45, *lerp_rgb(c_top, c_bot, 0.5), alpha)
        pat.add_color_stop_rgba(1.0, *c_bot, alpha)
        ctx.set_source(pat)
        ctx.fill()

        # Soft golden sunlit crest highlight
        ctx.new_path()
        ctx.move_to(ridge_pts[0][0], ridge_pts[0][1])
        for rx, ry in ridge_pts[1:]:
            ctx.line_to(rx, ry)
        ctx.set_source_rgba(0.96, 0.93, 0.66, 0.24 if idx < 2 else 0.34)
        ctx.set_line_width(4.5 - idx)
        ctx.stroke()

        # Luminous morning valley mist floating in the hollow
        if idx < 2:
            mgrad = cairo.LinearGradient(0, by - 30, 0, by + 115)
            mgrad.add_color_stop_rgba(0.0, 0.97, 0.94, 0.84, 0.0)
            mgrad.add_color_stop_rgba(0.48, 0.97, 0.94, 0.84, 0.28 - idx * 0.08)
            mgrad.add_color_stop_rgba(1.0, 0.97, 0.94, 0.84, 0.0)
            ctx.set_source(mgrad)
            ctx.rectangle(0, by - 30, w, 145)
            ctx.fill()

    bgra = layer.to_straight_bgra()
    bgra[:, :, :3] = apply_painterly_finish(bgra[:, :, :3], brush_strength=9.5, angle_deg=-8.0, seed=44)
    return bgra


def build_grassland_ground(w: int = 2400, h: int = 1350) -> np.ndarray:
    """Rich painted foreground meadow floor with wildflowers, dappled sun & grass tufts (BGRA)."""
    layer = CairoLayer(w, h)
    ctx = layer.ctx

    horizon_y = int(h * 0.67)
    ctx.new_path()
    ctx.move_to(0, h)
    for i in range(61):
        u = i / 60.0
        x = u * w
        y = horizon_y + 18.0 * math.sin(u * math.tau * 1.4 + 0.7) + 8.0 * math.cos(u * math.tau * 3.1)
        ctx.line_to(x, y)
    ctx.line_to(w, h)
    ctx.close_path()

    pat = cairo.LinearGradient(0, horizon_y - 20, 0, h)
    pat.add_color_stop_rgba(0.0, 0.48, 0.65, 0.25, 1.0)
    pat.add_color_stop_rgba(0.32, 0.34, 0.53, 0.19, 1.0)
    pat.add_color_stop_rgba(0.68, 0.22, 0.38, 0.13, 1.0)
    pat.add_color_stop_rgba(1.0, 0.14, 0.25, 0.09, 1.0)
    ctx.set_source(pat)
    ctx.fill()

    # Warm dappled golden sunlight pools across the meadow
    for i in range(32):
        px = hash1d(i * 19 + 3) * w
        py = horizon_y + 20 + hash1d(i * 29 + 7) * (h - horizon_y - 35)
        rx = 120 + 210 * hash1d(i * 37 + 11)
        ry = 22 + 42 * hash1d(i * 41 + 13)
        g = cairo.RadialGradient(px, py, 10, px, py, rx)
        g.add_color_stop_rgba(0.0, 0.88, 0.92, 0.40, 0.28)
        g.add_color_stop_rgba(1.0, 0.88, 0.92, 0.40, 0.0)
        ctx.save()
        ctx.translate(px, py)
        ctx.scale(1.0, ry / rx)
        ctx.set_source(g)
        ctx.arc(0, 0, rx, 0, math.tau)
        ctx.fill()
        ctx.restore()

    # Lush painterly background grass blades
    for i in range(520):
        gx = hash1d(i * 13 + 5) * w
        gy = horizon_y + 12 + (hash1d(i * 17 + 9) ** 0.85) * (h - horizon_y - 20)
        depth = clamp((gy - horizon_y) / float(h - horizon_y))
        gh = lerp(24.0, 86.0, depth) * (0.7 + 0.6 * hash1d(i * 23 + 3))
        lean = (hash1d(i * 31 + 11) - 0.35) * gh * 0.55
        c = lerp_rgb((0.38, 0.58, 0.20), (0.17, 0.34, 0.11), hash1d(i * 43 + 7))
        if hash1d(i * 59 + 2) > 0.68:
            c = lerp_rgb(c, (0.78, 0.82, 0.32), 0.48)
        ctx.move_to(gx, gy)
        ctx.curve_to(gx + lean * 0.3, gy - gh * 0.5, gx + lean * 0.75, gy - gh * 0.85, gx + lean, gy - gh)
        ctx.set_source_rgba(c[0], c[1], c[2], 0.72)
        ctx.set_line_width(lerp(1.5, 4.4, depth))
        ctx.stroke()

    for i in range(62):
        fx = hash1d(i * 71 + 13) * w
        fy = horizon_y + 34 + (hash1d(i * 79 + 17) ** 1.1) * (h - horizon_y - 110)
        depth = clamp((fy - horizon_y) / float(h - horizon_y))
        _draw_wildflower(ctx, fx, fy, scale=lerp(0.55, 1.4, depth), kind=i % 3)

    bgra = layer.to_straight_bgra()
    bgra[:, :, :3] = apply_painterly_finish(bgra[:, :, :3], brush_strength=9.5, angle_deg=-18.0, seed=55)
    return bgra


def build_soil_cross_section(w: int = 2400, h: int = 1400, surface_y: int = 560) -> np.ndarray:
    """Rich macro subterranean soil cross-section for Scene 1 (Life Begins)."""
    sky = build_morning_sky(w, h)
    valley = build_distant_valley(w, h)
    composite_straight_bgra(sky, valley, opacity=0.92)
    sky = cv2.GaussianBlur(sky, (0, 0), 7.5)

    soil_h = h - surface_y + 80
    n_coarse = fbm_noise_2d(soil_h, w, octaves=5, base_grid=(8, 14), persistence=0.56, seed=303)
    n_fine = fbm_noise_2d(soil_h, w, octaves=4, base_grid=(28, 48), persistence=0.60, seed=409)
    n_speck = fbm_noise_2d(soil_h, w, octaves=3, base_grid=(90, 150), persistence=0.65, seed=509)

    yy = np.linspace(0.0, 1.0, soil_h, dtype=np.float32)[:, None]
    top_bgr = np.array([30.0, 46.0, 72.0], dtype=np.float32)
    mid_bgr = np.array([36.0, 60.0, 94.0], dtype=np.float32)
    deep_bgr = np.array([22.0, 34.0, 54.0], dtype=np.float32)

    t1 = np.clip(yy * 2.2, 0.0, 1.0)[:, :, None]
    t2 = np.clip((yy - 0.42) / 0.58, 0.0, 1.0)[:, :, None]
    soil_col = top_bgr * (1.0 - t1) + mid_bgr * t1
    soil_col = soil_col * (1.0 - t2) + deep_bgr * t2

    clod = (n_coarse * 0.55 + n_fine * 0.30 + n_speck * 0.15 - 0.5)
    soil_col = soil_col + clod[:, :, None] * np.array([28.0, 44.0, 66.0], dtype=np.float32)
    fleck_mask = (n_speck > 0.72).astype(np.float32)[:, :, None]
    soil_col = soil_col + fleck_mask * np.array([30.0, 48.0, 68.0], dtype=np.float32)
    pore_mask = (n_fine < 0.28).astype(np.float32)[:, :, None]
    soil_col = soil_col * (1.0 - 0.32 * pore_mask)

    soil_u8 = np.clip(soil_col, 0.0, 255.0).astype(np.uint8)

    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    surf_curve = (
        surface_y
        + 16.0 * np.sin(xx * math.tau * 1.8 + 0.5)
        + 7.0 * np.cos(xx * math.tau * 5.2)
        + 3.5 * np.sin(xx * math.tau * 11.7)
    )

    out = sky.copy()
    y_grid = np.arange(h, dtype=np.float32)[:, None]
    mask = np.clip((y_grid - surf_curve[None, :]) / 4.5, 0.0, 1.0)
    y0 = max(0, surface_y - 40)
    sh = min(soil_u8.shape[0], h - y0)
    roi_m = mask[y0 : y0 + sh, :, None]
    out[y0 : y0 + sh] = np.clip(
        out[y0 : y0 + sh].astype(np.float32) * (1.0 - roi_m) + soil_u8[:sh].astype(np.float32) * roi_m,
        0.0,
        255.0,
    ).astype(np.uint8)

    layer = CairoLayer(w, h)
    ctx = layer.ctx

    # Sunlit earthy crust along top soil ridge
    ctx.new_path()
    ctx.move_to(0, float(surf_curve[0]))
    for x_i in range(0, w, 12):
        ctx.line_to(x_i, float(surf_curve[x_i]))
    ctx.line_to(w, float(surf_curve[-1]))
    ctx.set_source_rgba(0.66, 0.50, 0.30, 0.58)
    ctx.set_line_width(7.0)
    ctx.stroke()

    # Scattered small smooth pebbles & stones embedded in soil (keeping hero seed pocket open!)
    for i in range(95):
        px = hash1d(i * 37 + 11) * w
        sy_at = float(surf_curve[min(w - 1, int(px))])
        py = sy_at + 28 + (hash1d(i * 43 + 17) ** 0.9) * (h - sy_at - 50)
        if math.hypot(px - w * 0.5, py - (surface_y + 165)) < 145:
            continue
        rx = 8.0 + 25.0 * (hash1d(i * 53 + 7) ** 1.6)
        ry = rx * (0.55 + 0.35 * hash1d(i * 61 + 3))
        stone_rgb = lerp_rgb((0.46, 0.40, 0.34), (0.30, 0.27, 0.24), hash1d(i * 71 + 19))
        _draw_painted_rock(ctx, px, py, rx, ry, seed=900 + i, base_rgb=stone_rgb, moss_amount=0.0)

    # Fine ambient soil roots
    for i in range(130):
        rx = hash1d(i * 89 + 5) * w
        sy_at = float(surf_curve[min(w - 1, int(rx))])
        ry = sy_at + 10 + hash1d(i * 97 + 13) * 380.0
        if math.hypot(rx - w * 0.5, ry - (surface_y + 165)) < 160:
            continue
        ctx.move_to(rx, ry)
        cx1, cy1 = rx + (hash1d(i * 101) - 0.5) * 40, ry + 35 + hash1d(i * 103) * 40
        cx2, cy2 = cx1 + (hash1d(i * 107) - 0.5) * 55, cy1 + 30 + hash1d(i * 109) * 55
        ctx.curve_to(cx1, cy1, cx2, cy2, cx2 + (hash1d(i * 113) - 0.5) * 35, cy2 + 45)
        ctx.set_source_rgba(0.64, 0.54, 0.38, 0.28)
        ctx.set_line_width(1.2 + 1.2 * hash1d(i * 127))
        ctx.stroke()

    # Crumbly soil clods along the top surface
    for i in range(180):
        cx = hash1d(i * 131 + 2) * w
        sy_at = float(surf_curve[min(w - 1, int(cx))])
        cy = sy_at + (hash1d(i * 137 + 7) - 0.35) * 26.0
        r = 2.5 + 6.5 * hash1d(i * 139 + 11)
        ctx.arc(cx, cy, r, 0, math.tau)
        c = lerp_rgb((0.26, 0.16, 0.09), (0.48, 0.34, 0.20), hash1d(i * 149 + 13))
        ctx.set_source_rgba(c[0], c[1], c[2], 0.85)
        ctx.fill()

    layer.composite_over(out, opacity=1.0)
    return out


def build_pond_environment(w: int = 2400, h: int = 1350) -> Tuple[np.ndarray, np.ndarray]:
    """Freshwater pond nestled in the sunlit grassland ecosystem (Scene 3)."""
    sky = _paint_sky_gradient(
        w,
        h,
        stops=[
            (0.00, (58, 126, 192)),
            (0.28, (112, 174, 220)),
            (0.50, (186, 218, 230)),
            (0.68, (238, 228, 196)),
            (1.00, (194, 212, 190)),
        ],
        sun_xy=(0.74, 0.20),
        sun_glow_strength=0.72,
    )
    sky = _draw_sculpted_clouds(sky, sun_xy=(0.74, 0.20), seed=312, y_min_frac=0.10, y_max_frac=0.38, count=7)
    valley = build_distant_valley(w, h)
    M_shift = np.array([[1, 0, 0], [0, 1, -135]], dtype=np.float32)
    valley_shifted = cv2.warpAffine(valley, M_shift, (w, h), borderMode=cv2.BORDER_CONSTANT)
    composite_straight_bgra(sky, valley_shifted, opacity=0.95)

    layer = CairoLayer(w, h)
    ctx = layer.ctx
    water_line = int(h * 0.61)

    # Lush grassy pond bank behind waterline
    ctx.new_path()
    ctx.move_to(0, water_line + 18)
    for i in range(61):
        u = i / 60.0
        x = u * w
        y = water_line - 42.0 - 24.0 * math.sin(u * math.tau * 1.1 + 0.3) - 10.0 * math.cos(u * math.tau * 3.0)
        ctx.line_to(x, y)
    ctx.line_to(w, water_line + 18)
    ctx.close_path()
    bgrad = cairo.LinearGradient(0, water_line - 95, 0, water_line + 20)
    bgrad.add_color_stop_rgb(0.0, 0.36, 0.56, 0.22)
    bgrad.add_color_stop_rgb(0.65, 0.22, 0.39, 0.15)
    bgrad.add_color_stop_rgb(1.0, 0.13, 0.22, 0.11)
    ctx.set_source(bgrad)
    ctx.fill()

    # Lush clustered cattails, tall sedges & shoreline wildflowers
    for i in range(135):
        u = hash1d(i * 31 + 7)
        if 0.37 < u < 0.65 and hash1d(i * 41) > 0.20:
            continue
        rx = u * w
        ry = water_line + (hash1d(i * 47 + 3) - 0.5) * 18.0
        rh = 95.0 + 150.0 * hash1d(i * 53 + 11)
        lean = (hash1d(i * 59 + 13) - 0.42) * 65.0
        col = lerp_rgb((0.18, 0.38, 0.14), (0.46, 0.60, 0.22), hash1d(i * 61 + 17))
        ctx.move_to(rx, ry)
        ctx.curve_to(rx + lean * 0.3, ry - rh * 0.45, rx + lean * 0.75, ry - rh * 0.8, rx + lean, ry - rh)
        ctx.set_source_rgba(col[0], col[1], col[2], 0.88)
        ctx.set_line_width(3.0 + 2.8 * hash1d(i * 67))
        ctx.stroke()

        if i % 7 == 0:
            tx = rx + lean * 0.78
            ty = ry - rh * 0.78
            ang = math.atan2(-rh * 0.6, lean * 0.6)
            ctx.save()
            ctx.translate(tx, ty)
            ctx.rotate(ang + math.pi * 0.5)
            cpat = cairo.LinearGradient(-8, 0, 8, 0)
            cpat.add_color_stop_rgb(0.0, 0.22, 0.12, 0.06)
            cpat.add_color_stop_rgb(0.45, 0.48, 0.27, 0.12)
            cpat.add_color_stop_rgb(1.0, 0.18, 0.10, 0.05)
            ctx.set_source(cpat)
            ctx.scale(1.0, 4.2)
            ctx.arc(0, 0, 7.2, 0, math.tau)
            ctx.fill()
            ctx.restore()

    # Shoreline background boulders
    _draw_painted_rock(ctx, 270, water_line + 12, 185, 78, seed=601, base_rgb=(0.40, 0.43, 0.40), moss_amount=0.76, wet_bottom=True)
    _draw_painted_rock(ctx, 495, water_line + 25, 118, 52, seed=602, base_rgb=(0.45, 0.44, 0.39), moss_amount=0.58, wet_bottom=True)
    _draw_painted_rock(ctx, 1960, water_line + 18, 225, 86, seed=603, base_rgb=(0.42, 0.45, 0.42), moss_amount=0.70, wet_bottom=True)
    _draw_painted_rock(ctx, 2195, water_line + 5, 185, 92, seed=604, base_rgb=(0.37, 0.40, 0.38), moss_amount=0.82, wet_bottom=True)

    # Base translucent pond water bed
    wgrad = cairo.LinearGradient(0, water_line, 0, h)
    wgrad.add_color_stop_rgb(0.0, 0.24, 0.40, 0.36)
    wgrad.add_color_stop_rgb(0.5, 0.14, 0.28, 0.24)
    wgrad.add_color_stop_rgb(1.0, 0.08, 0.17, 0.15)
    ctx.set_source(wgrad)
    ctx.rectangle(0, water_line, w, h - water_line)
    ctx.fill()

    layer.composite_over(sky, opacity=1.0)
    sky = apply_painterly_finish(sky, brush_strength=7.0, angle_deg=-5.0, seed=77)

    # Foreground pond elements (lily pads + hero frog boulders + water reeds) composited OVER live water reflection
    fg = CairoLayer(w, h)
    fctx = fg.ctx

    lily_specs = [
        (1145, water_line + 120, 88, 26, 0.4),
        (1255, water_line + 155, 64, 19, -0.7),
        (1585, water_line + 98, 74, 22, 1.2),
        (540, water_line + 235, 98, 30, 0.2),
        (1715, water_line + 215, 108, 34, -0.5),
    ]
    for lx, ly, lrx, lry, notch_ang in lily_specs:
        draw_soft_shadow(fctx, lx + 4, ly + 8, lrx * 1.04, lry * 1.08, alpha=0.34, color=(0.03, 0.10, 0.08))
        fctx.save()
        fctx.translate(lx, ly)
        fctx.scale(1.0, lry / lrx)
        fctx.new_path()
        fctx.move_to(0, 0)
        fctx.arc(0, 0, lrx, notch_ang + 0.18, notch_ang + math.tau - 0.18)
        fctx.close_path()
        lpat = cairo.RadialGradient(lrx * 0.2, -lrx * 0.2, 6, 0, 0, lrx * 1.1)
        lpat.add_color_stop_rgb(0.0, 0.44, 0.66, 0.28)
        lpat.add_color_stop_rgb(0.75, 0.23, 0.44, 0.18)
        lpat.add_color_stop_rgb(1.0, 0.15, 0.30, 0.13)
        fctx.set_source(lpat)
        fctx.fill_preserve()
        fctx.set_source_rgba(0.56, 0.74, 0.34, 0.48)
        fctx.set_line_width(1.8)
        fctx.stroke()
        for v_i in range(7):
            va = notch_ang + 0.65 + v_i * ((math.tau - 1.3) / 6.0)
            fctx.move_to(0, 0)
            fctx.line_to(math.cos(va) * lrx * 0.85, math.sin(va) * lrx * 0.85)
        fctx.set_source_rgba(0.64, 0.80, 0.40, 0.26)
        fctx.set_line_width(1.2)
        fctx.stroke()
        fctx.restore()

    # Hero wet mossy boulders (left frog perch + right target landing stone)
    _draw_painted_rock(fctx, 840, water_line + 168, 230, 108, seed=701, base_rgb=(0.41, 0.45, 0.42), moss_amount=0.68, wet_bottom=True)
    _draw_painted_rock(fctx, 635, water_line + 212, 134, 68, seed=702, base_rgb=(0.38, 0.41, 0.39), moss_amount=0.48, wet_bottom=True)
    _draw_painted_rock(fctx, 1495, water_line + 154, 178, 78, seed=703, base_rgb=(0.44, 0.46, 0.41), moss_amount=0.62, wet_bottom=True)
    _draw_painted_rock(fctx, 1690, water_line + 188, 138, 64, seed=704, base_rgb=(0.39, 0.42, 0.39), moss_amount=0.52, wet_bottom=True)

    # Subtle water-meniscus highlights hugging the wet rock bases
    for bx, by, brx, bry in [(840, water_line + 220, 236, 34), (1495, water_line + 198, 185, 26)]:
        fctx.save()
        fctx.translate(bx, by)
        fctx.scale(1.0, bry / brx)
        fctx.arc(0, 0, brx, 0.15, math.pi - 0.15)
        fctx.set_source_rgba(0.86, 0.96, 0.94, 0.40)
        fctx.set_line_width(2.8)
        fctx.stroke()
        fctx.restore()

    fg_bgra = fg.to_straight_bgra()
    return sky, fg_bgra


def build_rocky_meadow(w: int = 2400, h: int = 1350) -> np.ndarray:
    """Sun-dappled rocky section of the grassland for Scene 4 (The Snake)."""
    sky = _paint_sky_gradient(
        w,
        h,
        stops=[
            (0.00, (74, 134, 188)),
            (0.28, (142, 186, 214)),
            (0.48, (224, 222, 196)),
            (1.00, (194, 196, 164)),
        ],
        sun_xy=(0.74, 0.18),
        sun_glow_strength=0.62,
    )
    sky = _draw_sculpted_clouds(sky, sun_xy=(0.74, 0.18), seed=418, y_min_frac=0.08, y_max_frac=0.34, count=5)
    valley = build_distant_valley(w, h)
    M_up = np.array([[1, 0, 0], [0, 1, -240]], dtype=np.float32)
    valley_up = cv2.warpAffine(valley, M_up, (w, h), borderMode=cv2.BORDER_CONSTANT)
    composite_straight_bgra(sky, valley_up, opacity=0.90)

    ground_y = int(h * 0.38)
    gh = h - ground_y + 40
    n1 = fbm_noise_2d(gh, w, octaves=5, base_grid=(7, 12), persistence=0.55, seed=811)
    n2 = fbm_noise_2d(gh, w, octaves=4, base_grid=(22, 36), persistence=0.58, seed=823)

    yy = np.linspace(0.0, 1.0, gh, dtype=np.float32)[:, None]
    far_bgr = np.array([58.0, 98.0, 72.0], dtype=np.float32)
    mid_bgr = np.array([56.0, 84.0, 92.0], dtype=np.float32)
    near_bgr = np.array([40.0, 62.0, 78.0], dtype=np.float32)

    t1 = np.clip(yy * 1.7, 0.0, 1.0)[:, :, None]
    t2 = np.clip((yy - 0.5) * 2.0, 0.0, 1.0)[:, :, None]
    gcol = far_bgr * (1.0 - t1) + mid_bgr * t1
    gcol = gcol * (1.0 - t2) + near_bgr * t2
    gcol = gcol + ((n1 * 0.6 + n2 * 0.4) - 0.5)[:, :, None] * np.array([34.0, 44.0, 52.0], dtype=np.float32)

    # Dappled warm sunlight pools vs cool leafy shadows
    n_dapple = fbm_noise_2d(gh, w, octaves=3, base_grid=(5, 9), persistence=0.5, seed=839)
    sun_pool = np.clip((n_dapple - 0.40) * 2.4, 0.0, 1.0)[:, :, None]
    gcol = gcol * (0.72 + 0.45 * sun_pool) + sun_pool * np.array([14.0, 26.0, 34.0], dtype=np.float32)

    g_u8 = np.clip(gcol, 0.0, 255.0).astype(np.uint8)
    out = sky.copy()
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    curve = ground_y + 22.0 * np.sin(xx * math.tau * 1.2 + 0.8) + 9.0 * np.cos(xx * math.tau * 2.9)
    yg = np.arange(h, dtype=np.float32)[:, None]
    mask = np.clip((yg - curve[None, :]) / 14.0, 0.0, 1.0)
    y0 = max(0, ground_y - 35)
    sh = min(g_u8.shape[0], h - y0)
    rm = mask[y0 : y0 + sh, :, None]
    out[y0 : y0 + sh] = np.clip(
        out[y0 : y0 + sh].astype(np.float32) * (1.0 - rm) + g_u8[:sh].astype(np.float32) * rm,
        0.0,
        255.0,
    ).astype(np.uint8)

    layer = CairoLayer(w, h)
    ctx = layer.ctx

    bg_rocks = [
        (290, 560, 230, 118, (0.48, 0.45, 0.39), 0.56),
        (595, 522, 165, 82, (0.52, 0.49, 0.42), 0.44),
        (1065, 542, 150, 70, (0.46, 0.45, 0.39), 0.38),
        (1785, 545, 215, 106, (0.49, 0.46, 0.40), 0.62),
        (2130, 595, 255, 132, (0.44, 0.42, 0.37), 0.66),
        # Frog's weathered boulder on right (~x=1690, y=815)
        (1690, 815, 195, 92, (0.50, 0.48, 0.42), 0.55),
        (1910, 860, 135, 64, (0.45, 0.44, 0.38), 0.44),
        # Left rock shelter where the snake emerges
        (255, 830, 240, 118, (0.44, 0.42, 0.37), 0.58),
        (505, 775, 128, 58, (0.49, 0.46, 0.40), 0.48),
    ]
    for idx, (rx, ry, rrx, rry, rgb, moss) in enumerate(bg_rocks):
        _draw_painted_rock(ctx, rx, ry, rrx, rry, seed=850 + idx, base_rgb=rgb, moss_amount=moss)
        # Grounding grass blades hugging the rock base so every boulder sits organically in the earth!
        for g_i in range(18):
            gx = rx + (hash1d(idx * 91 + g_i * 7) - 0.5) * rrx * 2.1
            gy = ry + rry * (0.45 + 0.32 * hash1d(idx * 97 + g_i * 11))
            gh_t = 28.0 + 52.0 * hash1d(idx * 101 + g_i * 13)
            lean = (hash1d(idx * 103 + g_i * 17) - 0.45) * 42.0
            col = lerp_rgb((0.28, 0.46, 0.16), (0.52, 0.54, 0.22), hash1d(idx * 107 + g_i))
            ctx.move_to(gx, gy)
            ctx.curve_to(gx + lean * 0.3, gy - gh_t * 0.5, gx + lean * 0.75, gy - gh_t * 0.85, gx + lean, gy - gh_t)
            ctx.set_source_rgba(col[0], col[1], col[2], 0.82)
            ctx.set_line_width(2.4 + 1.8 * hash1d(idx * 109 + g_i))
            ctx.stroke()

    # Small gravel & pebbles
    for i in range(110):
        px = hash1d(i * 41 + 9) * w
        py = ground_y + 120 + hash1d(i * 47 + 13) * (h - ground_y - 160)
        prx = 6.0 + 15.0 * (hash1d(i * 53 + 7) ** 1.5)
        sc = lerp_rgb((0.52, 0.48, 0.41), (0.35, 0.33, 0.29), hash1d(i * 61))
        _draw_painted_rock(ctx, px, py, prx, prx * 0.55, seed=1200 + i, base_rgb=sc, moss_amount=0.05)

    # Curled dry leaves scattered naturally
    for i in range(95):
        lx = hash1d(i * 73 + 5) * w
        ly = ground_y + 145 + hash1d(i * 79 + 11) * (h - ground_y - 195)
        depth = clamp((ly - ground_y) / float(h - ground_y))
        llen = lerp(24.0, 56.0, depth) * (0.75 + 0.5 * hash1d(i * 83))
        ang = (hash1d(i * 89) - 0.5) * math.tau
        _draw_fallen_leaf(ctx, lx, ly, llen, ang, hue_mix=hash1d(i * 97), alpha=0.88)

    # Dry & green grass tufts flanking the rocky clearing
    for i in range(170):
        gx = hash1d(i * 101 + 3) * w
        gy = ground_y + 55 + hash1d(i * 107 + 7) * (h - ground_y - 125)
        if 540 < gx < 1560 and 780 < gy < 1040:
            continue
        depth = clamp((gy - ground_y) / float(h - ground_y))
        gh_b = lerp(35.0, 96.0, depth)
        lean = (hash1d(i * 113) - 0.45) * 45.0
        col = lerp_rgb((0.34, 0.48, 0.19), (0.58, 0.50, 0.24), hash1d(i * 127))
        ctx.move_to(gx, gy)
        ctx.curve_to(gx + lean * 0.3, gy - gh_b * 0.5, gx + lean * 0.8, gy - gh_b * 0.85, gx + lean, gy - gh_b)
        ctx.set_source_rgba(col[0], col[1], col[2], 0.80)
        ctx.set_line_width(lerp(1.8, 3.8, depth))
        ctx.stroke()

    layer.composite_over(out, opacity=1.0)
    return apply_painterly_finish(out, brush_strength=8.5, angle_deg=-12.0, seed=91)


def build_sky_panorama(w: int = 2600, h: int = 1600) -> np.ndarray:
    """Grand wide sunlit valley sky & sweeping grassland vista for Scene 5 (The Hawk)."""
    sky = _paint_sky_gradient(
        w,
        h,
        stops=[
            (0.00, (38, 96, 172)),
            (0.24, (80, 148, 212)),
            (0.46, (154, 202, 232)),
            (0.62, (230, 230, 210)),
            (0.76, (244, 222, 180)),
            (1.00, (192, 204, 152)),
        ],
        sun_xy=(0.78, 0.18),
        sun_glow_strength=0.76,
    )
    sky = _draw_sculpted_clouds(sky, sun_xy=(0.78, 0.18), seed=505, y_min_frac=0.10, y_max_frac=0.44, count=10)

    layer = CairoLayer(w, h)
    ctx = layer.ctx

    ridges = [
        (0.61, 96, 36, 1.9, 0.6, (0.42, 0.56, 0.67), (0.33, 0.48, 0.58), 0.0),
        (0.68, 72, 26, 2.5, 2.0, (0.37, 0.55, 0.41), (0.27, 0.45, 0.31), 0.55),
        (0.76, 52, 18, 2.2, 3.8, (0.44, 0.62, 0.29), (0.28, 0.47, 0.18), 0.72),
        (0.84, 34, 14, 1.8, 1.2, (0.50, 0.65, 0.28), (0.22, 0.39, 0.14), 0.0),
    ]
    for idx, (by_f, a1, a2, f1, ph, c1, c2, tree_sc) in enumerate(ridges):
        by = h * by_f
        pts = []
        for i in range(91):
            u = i / 90.0
            x = u * w
            y = by - a1 * math.sin(u * math.tau * f1 * 0.4 + ph) - a2 * math.cos(u * math.tau * f1 * 0.95 + ph)
            pts.append((x, y))
        if tree_sc > 0.0:
            _draw_woodland_treeline(
                ctx, pts, tree_sc,
                lerp_rgb(c1, (0.64, 0.74, 0.34), 0.38),
                lerp_rgb(c1, c2, 0.5),
                lerp_rgb(c2, (0.12, 0.22, 0.14), 0.48),
                seed=700 + idx * 83,
                alpha=0.88,
            )
        ctx.new_path()
        ctx.move_to(0, h)
        for rx, ry in pts:
            ctx.line_to(rx, ry)
        ctx.line_to(w, h)
        ctx.close_path()
        pat = cairo.LinearGradient(0, by - a1, 0, h)
        pat.add_color_stop_rgb(0.0, *c1)
        pat.add_color_stop_rgb(1.0, *c2)
        ctx.set_source(pat)
        ctx.fill()

        if idx < 3:
            mg = cairo.LinearGradient(0, by - 25, 0, by + 95)
            mg.add_color_stop_rgba(0.0, 0.97, 0.95, 0.87, 0.0)
            mg.add_color_stop_rgba(0.5, 0.97, 0.95, 0.87, 0.26 - idx * 0.05)
            mg.add_color_stop_rgba(1.0, 0.97, 0.95, 0.87, 0.0)
            ctx.set_source(mg)
            ctx.rectangle(0, by - 25, w, 120)
            ctx.fill()

    # Perspective-widening meandering river & pond gleaming across the sunlit valley floor!
    river_pts = []
    river_hw = []
    for i in range(42):
        u = i / 41.0
        rx = lerp(w * 0.44, w * 0.72, u) + math.sin(u * math.pi * 2.4 + 0.5) * w * (0.04 + 0.14 * u)
        ry = lerp(h * 0.785, h * 0.99, u ** 1.35)
        river_pts.append((rx, ry))
        # Natural widening pond-pool near u=0.55 and perspective widening toward foreground
        pond_bulge = 22.0 * math.exp(-((u - 0.55) ** 2) / 0.018)
        river_hw.append(3.5 + 34.0 * (u ** 1.7) + pond_bulge)

    # Dark green riverside foliage edge
    draw_tapered_ribbon(ctx, river_pts, [hw + 12.0 for hw in river_hw], close_tip=False)
    ctx.set_source_rgba(0.16, 0.30, 0.13, 0.52)
    ctx.fill()

    # Sun-reflecting cerulean-gold water surface
    draw_tapered_ribbon(ctx, river_pts, river_hw, close_tip=False)
    rpat = cairo.LinearGradient(0, h * 0.78, 0, h)
    rpat.add_color_stop_rgba(0.0, 0.86, 0.93, 0.94, 0.75)
    rpat.add_color_stop_rgba(0.5, 0.58, 0.78, 0.84, 0.82)
    rpat.add_color_stop_rgba(1.0, 0.34, 0.58, 0.64, 0.88)
    ctx.set_source(rpat)
    ctx.fill()

    # Sunlit meadow patchwork highlights
    for i in range(28):
        px = hash1d(i * 43 + 7) * w
        py = h * (0.80 + 0.18 * hash1d(i * 53 + 11))
        rx = 140 + 220 * hash1d(i * 61)
        ry = 22 + 34 * hash1d(i * 67)
        g = cairo.RadialGradient(px, py, 10, px, py, rx)
        g.add_color_stop_rgba(0.0, 0.88, 0.87, 0.42, 0.24)
        g.add_color_stop_rgba(1.0, 0.88, 0.87, 0.42, 0.0)
        ctx.save()
        ctx.translate(px, py)
        ctx.scale(1.0, ry / rx)
        ctx.set_source(g)
        ctx.arc(0, 0, rx, 0, math.tau)
        ctx.fill()
        ctx.restore()

    layer.composite_over(sky, opacity=1.0)
    return apply_painterly_finish(sky, brush_strength=6.5, angle_deg=-5.0, seed=112)


def build_finale_backdrop(w: int = 1920, h: int = 1080) -> np.ndarray:
    """Serene, clean, museum-grade sunlit meadow backdrop for Scene 6."""
    sky = build_morning_sky(w, h)
    valley = build_distant_valley(w, h)
    composite_straight_bgra(sky, valley, opacity=0.94)
    bg = cv2.GaussianBlur(sky, (0, 0), 8.5).astype(np.float32)

    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    warm_veil = np.array([175.0, 212.0, 232.0], dtype=np.float32)[None, None, :]
    deep_base = np.array([22.0, 42.0, 24.0], dtype=np.float32)[None, None, :]

    bg = bg * 0.74 + warm_veil * 0.26
    low_band = np.clip((yy - 0.56) / 0.44, 0.0, 1.0) ** 1.2
    bg = bg * (1.0 - 0.62 * low_band) + deep_base * (0.62 * low_band)

    # Gentle central readability cushion behind the 5 animated organisms & glowing arrows
    center_band = np.exp(-((yy - 0.50) ** 2) / 0.08) * 0.26
    bg = bg * (1.0 - center_band) + np.array([18.0, 32.0, 22.0], dtype=np.float32) * center_band

    out = np.clip(bg, 0.0, 255.0).astype(np.uint8)

    layer = CairoLayer(w, h)
    ctx = layer.ctx
    pad = 36
    ctx.rectangle(pad, pad, w - 2 * pad, h - 2 * pad)
    ctx.set_source_rgba(0.98, 0.90, 0.66, 0.36)
    ctx.set_line_width(2.0)
    ctx.stroke()

    ctx.rectangle(pad + 8, pad + 8, w - 2 * (pad + 8), h - 2 * (pad + 8))
    ctx.set_source_rgba(0.98, 0.90, 0.66, 0.16)
    ctx.set_line_width(1.0)
    ctx.stroke()

    layer.composite_over(out, opacity=1.0)
    return out


# =============================================================================
# Asset Cache Manager
# =============================================================================

ASSET_FILES = {
    "morning_sky": "env_morning_sky.png",
    "distant_valley": "env_distant_valley.png",
    "grassland_ground": "env_grassland_ground.png",
    "soil_cross_section": "env_soil_cross_section.png",
    "pond_bg": "env_pond_bg.png",
    "pond_fg": "env_pond_fg.png",
    "rocky_meadow": "env_rocky_meadow.png",
    "sky_panorama": "env_sky_panorama.png",
    "finale_backdrop": "env_finale_backdrop.png",
}


def ensure_environment_assets(assets_dir: str, force: bool = False) -> Dict[str, str]:
    os.makedirs(assets_dir, exist_ok=True)
    paths = {k: os.path.join(assets_dir, v) for k, v in ASSET_FILES.items()}

    if force or not os.path.exists(paths["morning_sky"]):
        print("[environments] Painting morning sky...")
        cv2.imwrite(paths["morning_sky"], build_morning_sky())

    if force or not os.path.exists(paths["distant_valley"]):
        print("[environments] Painting distant valley...")
        cv2.imwrite(paths["distant_valley"], build_distant_valley())

    if force or not os.path.exists(paths["grassland_ground"]):
        print("[environments] Painting grassland ground...")
        cv2.imwrite(paths["grassland_ground"], build_grassland_ground())

    if force or not os.path.exists(paths["soil_cross_section"]):
        print("[environments] Painting subterranean soil cross-section...")
        cv2.imwrite(paths["soil_cross_section"], build_soil_cross_section())

    if force or not os.path.exists(paths["pond_bg"]) or not os.path.exists(paths["pond_fg"]):
        print("[environments] Painting freshwater pond layers...")
        p_bg, p_fg = build_pond_environment()
        cv2.imwrite(paths["pond_bg"], p_bg)
        cv2.imwrite(paths["pond_fg"], p_fg)

    if force or not os.path.exists(paths["rocky_meadow"]):
        print("[environments] Painting sun-dappled rocky meadow...")
        cv2.imwrite(paths["rocky_meadow"], build_rocky_meadow())

    if force or not os.path.exists(paths["sky_panorama"]):
        print("[environments] Painting grand valley sky panorama...")
        cv2.imwrite(paths["sky_panorama"], build_sky_panorama())

    if force or not os.path.exists(paths["finale_backdrop"]):
        print("[environments] Painting serene finale backdrop...")
        cv2.imwrite(paths["finale_backdrop"], build_finale_backdrop())

    return paths


def load_environment_assets(assets_dir: str) -> Dict[str, np.ndarray]:
    paths = ensure_environment_assets(assets_dir, force=False)
    loaded: Dict[str, np.ndarray] = {}
    for k, p in paths.items():
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to load environment asset: {p}")
        loaded[k] = img
    return loaded


if __name__ == "__main__":
    assets_out = os.path.join(os.path.dirname(__file__), "assets")
    paths = ensure_environment_assets(assets_out, force=True)
    imgs = []
    for k in ["soil_cross_section", "morning_sky", "pond_bg", "rocky_meadow", "sky_panorama", "finale_backdrop"]:
        im = cv2.imread(paths[k], cv2.IMREAD_COLOR)
        if k == "morning_sky":
            v = cv2.imread(paths["distant_valley"], cv2.IMREAD_UNCHANGED)
            g = cv2.imread(paths["grassland_ground"], cv2.IMREAD_UNCHANGED)
            composite_straight_bgra(im, v)
            composite_straight_bgra(im, g)
        if k == "pond_bg":
            pf = cv2.imread(paths["pond_fg"], cv2.IMREAD_UNCHANGED)
            composite_straight_bgra(im, pf)
        imgs.append(cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA))
    row1 = np.concatenate(imgs[:3], axis=1)
    row2 = np.concatenate(imgs[3:], axis=1)
    sheet = np.concatenate([row1, row2], axis=0)
    sheet_path = os.path.join(assets_out, "environments_sheet.png")
    cv2.imwrite(sheet_path, sheet)
    print(f"Saved upgraded environment contact sheet: {sheet_path}")

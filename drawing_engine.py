"""
drawing_engine.py — Painterly PyCairo + NumPy + OpenCV + Pillow Rendering Engine
================================================================================

Core visual engine delivering a premium hand-painted animated nature-film look:
  - Zero-copy premultiplied BGRA PyCairo canvases for crisp anti-aliased anatomy
  - Multi-octave FBM & anisotropic oil-brush painterly textures
  - Soft grounded contact shadows, dewdrops, and tapered organic Bezier ribbons
  - Real-time distorted water reflection & concentric ripple shader
  - Volumetric golden sunbeams, atmospheric bloom, subtle canvas grain & color grading
  - Crisp Pillow documentary typography and optional Pygame surface bridge
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import cairo
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from animation import HEIGHT, WIDTH, clamp, lerp, polyline_frames

# =============================================================================
# Color Utilities (RGB 0..1 <-> BGR uint8)
# =============================================================================

def rgb01(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Normalize either 0..1 or 0..255 RGB tuples into 0..1 floats."""
    if max(r, g, b) > 1.001:
        return (r / 255.0, g / 255.0, b / 255.0)
    return (float(r), float(g), float(b))


def set_rgb(ctx: cairo.Context, col: Tuple[float, float, float], alpha: float = 1.0) -> None:
    r, g, b = rgb01(*col)
    ctx.set_source_rgba(r, g, b, clamp(alpha))


# =============================================================================
# Zero-Copy PyCairo Surface backed by NumPy BGRA Buffer
# =============================================================================

class CairoLayer:
    """
    Fast PyCairo drawing layer backed directly by a NumPy (H, W, 4) uint8 array
    in Cairo native little-endian premultiplied BGRA format.
    """

    def __init__(self, w: int = WIDTH, h: int = HEIGHT):
        self.w = int(w)
        self.h = int(h)
        self.buf = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        self.surface = cairo.ImageSurface.create_for_data(
            self.buf, cairo.FORMAT_ARGB32, self.w, self.h, self.w * 4
        )
        self.ctx = cairo.Context(self.surface)
        self.ctx.set_antialias(cairo.ANTIALIAS_GOOD)
        self.ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)

    def clear(self) -> None:
        self.buf.fill(0)

    def composite_over(self, dst_bgr: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        """
        Composite this premultiplied BGRA layer onto a uint8/float32 BGR image.
        Uses pure OpenCV C++ SIMD ops when opacity == 1.0 on uint8 canvases (~6ms at 1080p!).
        """
        self.surface.flush()
        src = self.buf
        if opacity <= 0.001:
            return dst_bgr
        if dst_bgr.dtype == np.uint8 and opacity >= 0.999:
            inv_a = cv2.subtract(255, src[:, :, 3])
            inv_a3 = cv2.merge([inv_a, inv_a, inv_a])
            cv2.multiply(dst_bgr, inv_a3, dst=dst_bgr, scale=1.0 / 255.0)
            cv2.add(dst_bgr, src[:, :, :3], dst=dst_bgr)
            return dst_bgr
        alpha = src[:, :, 3:4].astype(np.float32) * (opacity / 255.0)
        src_bgr = src[:, :, :3].astype(np.float32) * opacity
        if dst_bgr.dtype == np.uint8:
            out = src_bgr + dst_bgr.astype(np.float32) * (1.0 - alpha)
            np.clip(out, 0.0, 255.0, out=out)
            dst_bgr[:] = out.astype(np.uint8)
        else:
            dst_bgr[:] = src_bgr + dst_bgr * (1.0 - alpha)
        return dst_bgr

    def to_straight_bgra(self) -> np.ndarray:
        """Convert premultiplied Cairo BGRA to straight-alpha BGRA uint8 for PNG saving."""
        self.surface.flush()
        a = self.buf[:, :, 3:4].astype(np.float32)
        safe_a = np.maximum(a, 1.0)
        bgr = np.clip(self.buf[:, :, :3].astype(np.float32) * (255.0 / safe_a), 0, 255).astype(np.uint8)
        bgr[self.buf[:, :, 3] == 0] = 0
        return np.concatenate([bgr, self.buf[:, :, 3:4]], axis=2)


def composite_straight_bgra(dst_bgr: np.ndarray, src_bgra: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Alpha-composite a straight-alpha uint8 BGRA image over a uint8 BGR canvas."""
    if opacity <= 0.001:
        return dst_bgr
    if dst_bgr.dtype == np.uint8 and opacity >= 0.999:
        a = src_bgra[:, :, 3]
        a3 = cv2.merge([a, a, a])
        inv_a3 = cv2.subtract(255, a3)
        fg_part = cv2.multiply(src_bgra[:, :, :3], a3, scale=1.0 / 255.0)
        cv2.multiply(dst_bgr, inv_a3, dst=dst_bgr, scale=1.0 / 255.0)
        cv2.add(dst_bgr, fg_part, dst=dst_bgr)
        return dst_bgr
    a = src_bgra[:, :, 3:4].astype(np.float32) * (opacity / 255.0)
    out = src_bgra[:, :, :3].astype(np.float32) * a + dst_bgr.astype(np.float32) * (1.0 - a)
    np.clip(out, 0.0, 255.0, out=out)
    dst_bgr[:] = out.astype(np.uint8)
    return dst_bgr


# =============================================================================
# Procedural Painterly Textures (FBM, Brush Strokes, Canvas Grain)
# =============================================================================

def fbm_noise_2d(
    h: int,
    w: int,
    octaves: int = 4,
    base_grid: Tuple[int, int] = (8, 14),
    persistence: float = 0.52,
    seed: int = 42,
) -> np.ndarray:
    """Fast multi-octave smooth 2D fractal noise normalized to [0, 1] float32."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    norm = 0.0
    gh, gw = base_grid
    for _ in range(octaves):
        grid = rng.random((max(2, gh), max(2, gw)), dtype=np.float32)
        up = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        acc += up * amp
        norm += amp
        amp *= persistence
        gh = min(h, int(gh * 2.15))
        gw = min(w, int(gw * 2.15))
    acc /= max(norm, 1e-6)
    lo, hi = float(acc.min()), float(acc.max())
    if hi - lo > 1e-6:
        acc = (acc - lo) / (hi - lo)
    return np.clip(acc, 0.0, 1.0)


def directional_brush_texture(
    h: int,
    w: int,
    angle_deg: float = -12.0,
    stretch_x: int = 27,
    stretch_y: int = 5,
    seed: int = 7,
) -> np.ndarray:
    """
    Generate an anisotropic hand-painted gouache/oil brush texture in [-1, 1].
    Gives skies, rolling hills, rocks, and foliage rich tactile brush energy.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (h // 2, w // 2)).astype(np.float32)
    # Build oriented elongated Gaussian/box brush kernel
    ks = max(stretch_x, stretch_y) | 1
    kern = np.zeros((ks, ks), dtype=np.float32)
    c = ks // 2
    cv2.ellipse(kern, (c, c), (max(1, stretch_x // 2), max(1, stretch_y // 2)), angle_deg, 0, 360, 1.0, -1)
    kern = cv2.GaussianBlur(kern, (5, 5), 1.2)
    kern /= max(float(kern.sum()), 1e-6)
    stroked = cv2.filter2D(base, -1, kern, borderType=cv2.BORDER_REFLECT)
    stroked = cv2.resize(stroked, (w, h), interpolation=cv2.INTER_LINEAR)
    std = float(stroked.std())
    if std > 1e-6:
        stroked /= std
    return np.clip(stroked * 0.45, -1.0, 1.0)


def apply_painterly_finish(
    img_bgr: np.ndarray,
    brush_strength: float = 9.0,
    angle_deg: float = -10.0,
    seed: int = 11,
) -> np.ndarray:
    """Apply subtle tactile gouache brushstroke modulation to a baked background layer."""
    h, w = img_bgr.shape[:2]
    b1 = directional_brush_texture(h, w, angle_deg=angle_deg, stretch_x=29, stretch_y=5, seed=seed)
    b2 = directional_brush_texture(h, w, angle_deg=angle_deg + 35.0, stretch_x=17, stretch_y=3, seed=seed + 99)
    tex = (0.65 * b1 + 0.35 * b2)[:, :, None] * brush_strength
    out = np.clip(img_bgr.astype(np.float32) + tex, 0.0, 255.0).astype(np.uint8)
    return out


@lru_cache(maxsize=4)
def _cached_canvas_grain(h: int, w: int) -> np.ndarray:
    """Cached subtle warm fine illustration-paper texture in [-1, 1] float32."""
    rng = np.random.default_rng(2026)
    fine = rng.normal(0.0, 1.0, (h // 2, w // 2)).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (3, 3), 0.7)
    fine = cv2.resize(fine, (w, h), interpolation=cv2.INTER_LINEAR)
    # Subtle horizontal + vertical woven linen fibers
    hy = np.sin(np.linspace(0, h * 0.45, h, dtype=np.float32))[:, None] * 0.12
    vx = np.sin(np.linspace(0, w * 0.45, w, dtype=np.float32))[None, :] * 0.08
    grain = np.clip(fine * 0.42 + hy + vx, -1.0, 1.0)
    return grain[:, :, None]


@lru_cache(maxsize=4)
def _cached_vignette(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / float(w) - 0.5) * 2.0
    ny = (yy / float(h) - 0.5) * 2.0
    r2 = nx * nx * 0.78 + ny * ny * 1.05
    vig = np.clip(1.0 - 0.24 * np.power(np.clip(r2 - 0.28, 0.0, 2.0), 1.45), 0.68, 1.0)
    return vig[:, :, None]


@lru_cache(maxsize=4)
def _cached_filmic_lut() -> np.ndarray:
    """Warm nature-documentary filmic S-curve + sunlit highlight / cool shadow grade LUT (256x3 BGR)."""
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    # Gentle filmic S-curve (lifted shadows, silky shoulder highlights)
    sc = x + 0.055 * np.sin(math.pi * (2.0 * x - 1.0)) * np.sin(math.pi * x)
    sc = np.clip(sc * 0.985 + 0.012, 0.0, 1.0)

    # Channel tints: warm golden highlights, subtly richer cool-teal deep shadows
    r = np.clip(sc + 0.024 * np.power(sc, 1.35) - 0.008 * (1.0 - sc) ** 2, 0.0, 1.0)
    g = np.clip(sc + 0.012 * np.sin(math.pi * sc), 0.0, 1.0)
    b = np.clip(sc - 0.018 * np.power(sc, 1.5) + 0.014 * (1.0 - sc) ** 2.1, 0.0, 1.0)

    lut = np.stack([b, g, r], axis=1)
    return np.clip(lut * 255.0 + 0.5, 0, 255).astype(np.uint8)


# =============================================================================
# Rich PyCairo Painterly Primitives
# =============================================================================

def draw_tapered_ribbon(
    ctx: cairo.Context,
    pts: Sequence[Tuple[float, float]],
    half_widths: Sequence[float],
    close_tip: bool = True,
) -> None:
    """Build a smooth closed Cairo path along a polyline with arbitrary half-width profile."""
    if len(pts) < 2:
        return
    arr, _, normals = polyline_frames(pts)
    n = len(arr)
    hw = np.asarray(half_widths, dtype=np.float64)
    left = arr + normals * hw[:, None]
    right = arr - normals * hw[:, None]

    ctx.new_path()
    ctx.move_to(float(left[0, 0]), float(left[0, 1]))
    for i in range(1, n):
        ctx.line_to(float(left[i, 0]), float(left[i, 1]))
    if close_tip:
        ctx.line_to(float(arr[-1, 0]), float(arr[-1, 1]))
    for i in range(n - 1, -1, -1):
        ctx.line_to(float(right[i, 0]), float(right[i, 1]))
    ctx.close_path()


def draw_soft_shadow(
    ctx: cairo.Context,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle: float = 0.0,
    alpha: float = 0.34,
    color: Tuple[float, float, float] = (0.06, 0.10, 0.07),
) -> None:
    """Soft radial ambient-occlusion ground shadow to anchor creatures and stones."""
    if rx <= 0.5 or ry <= 0.5 or alpha <= 0.005:
        return
    ctx.save()
    ctx.translate(cx, cy)
    if abs(angle) > 1e-4:
        ctx.rotate(angle)
    ctx.scale(rx, ry)
    pat = cairo.RadialGradient(0.0, 0.0, 0.08, 0.0, 0.0, 1.0)
    r, g, b = rgb01(*color)
    pat.add_color_stop_rgba(0.0, r, g, b, alpha)
    pat.add_color_stop_rgba(0.55, r, g, b, alpha * 0.55)
    pat.add_color_stop_rgba(1.0, r, g, b, 0.0)
    ctx.set_source(pat)
    ctx.arc(0.0, 0.0, 1.0, 0.0, math.tau)
    ctx.fill()
    ctx.restore()


def draw_dewdrop(ctx: cairo.Context, x: float, y: float, r: float, alpha: float = 0.92) -> None:
    """Draw a photorealistic translucent morning dewdrop with caustic bounce & specular glint."""
    ctx.save()
    ctx.translate(x, y)
    # Soft drop shadow underneath
    ctx.arc(r * 0.15, r * 0.25, r * 1.05, 0, math.tau)
    ctx.set_source_rgba(0.08, 0.18, 0.06, 0.22 * alpha)
    ctx.fill()

    # Translucent refractive body (darker upper, bright golden-green caustic bottom)
    pat = cairo.RadialGradient(-r * 0.25, -r * 0.28, r * 0.1, 0.0, r * 0.18, r * 1.1)
    pat.add_color_stop_rgba(0.0, 0.92, 0.98, 0.95, 0.45 * alpha)
    pat.add_color_stop_rgba(0.45, 0.58, 0.78, 0.58, 0.52 * alpha)
    pat.add_color_stop_rgba(0.82, 0.88, 0.97, 0.72, 0.78 * alpha)
    pat.add_color_stop_rgba(1.0, 0.30, 0.46, 0.28, 0.58 * alpha)
    ctx.arc(0, 0, r, 0, math.tau)
    ctx.set_source(pat)
    ctx.fill()

    # Crisp sun glint upper-left
    ctx.arc(-r * 0.34, -r * 0.34, r * 0.30, 0, math.tau)
    ctx.set_source_rgba(1.0, 1.0, 0.96, 0.95 * alpha)
    ctx.fill()
    # Secondary tiny caustic dot lower-right
    ctx.arc(r * 0.28, r * 0.36, r * 0.16, 0, math.tau)
    ctx.set_source_rgba(0.95, 1.0, 0.82, 0.75 * alpha)
    ctx.fill()
    ctx.restore()


def draw_organic_poly(
    ctx: cairo.Context,
    pts: Sequence[Tuple[float, float]],
    closed: bool = True,
) -> None:
    """Draw a smooth Catmull-Rom-like quadratic midpoint smoothed curve through control points."""
    n = len(pts)
    if n < 2:
        return
    ctx.new_path()
    if closed and n >= 3:
        m_last = ((pts[-1][0] + pts[0][0]) * 0.5, (pts[-1][1] + pts[0][1]) * 0.5)
        ctx.move_to(m_last[0], m_last[1])
        for i in range(n):
            p = pts[i]
            p_next = pts[(i + 1) % n]
            mid = ((p[0] + p_next[0]) * 0.5, (p[1] + p_next[1]) * 0.5)
            # Quadratic Bezier through p from current to mid converted to cubic
            cx, cy = ctx.get_current_point()
            c1x = cx + (2.0 / 3.0) * (p[0] - cx)
            c1y = cy + (2.0 / 3.0) * (p[1] - cy)
            c2x = mid[0] + (2.0 / 3.0) * (p[0] - mid[0])
            c2y = mid[1] + (2.0 / 3.0) * (p[1] - mid[1])
            ctx.curve_to(c1x, c1y, c2x, c2y, mid[0], mid[1])
        ctx.close_path()
    else:
        ctx.move_to(pts[0][0], pts[0][1])
        for i in range(1, n):
            ctx.line_to(pts[i][0], pts[i][1])


# =============================================================================
# Real-Time Freshwater Pond Reflection & Ripple Shader
# =============================================================================

def apply_pond_water_reflection(
    frame_bgr: np.ndarray,
    water_y_px: int,
    t: float,
    ripples: Sequence[Tuple[float, float, float, float]] = (),
    depth_tint_bgr: Tuple[float, float, float] = (54.0, 74.0, 38.0),
) -> np.ndarray:
    """
    Create realistic animated freshwater pond reflection below `water_y_px`:
      - Mirrors the above-water world vertically with subtle perspective compression
      - Applies gentle wind-driven water wave refraction + localized expanding splash ripples
      - Blends deep translucent emerald-teal water absorption, Fresnel sky shimmer,
        shoreline foam/meniscus highlight, and dancing sunlight caustics.
    `ripples`: list of (cx_px, cy_px, age_s, strength)
    """
    h, w = frame_bgr.shape[:2]
    wy = int(clamp(water_y_px, 4, h - 4))
    water_h = h - wy
    if water_h <= 2:
        return frame_bgr

    # Grab above-water strip and mirror vertically
    src_h = min(wy, int(water_h * 1.15))
    above = frame_bgr[wy - src_h : wy, :, :]
    mirrored = above[::-1, :, :]
    if mirrored.shape[0] != water_h:
        mirrored = cv2.resize(mirrored, (w, water_h), interpolation=cv2.INTER_LINEAR)

    # Coordinate grids for wave displacement (computed on half-res for speed)
    hw, hh = w // 2, max(2, water_h // 2)
    yy, xx = np.mgrid[0:hh, 0:hw].astype(np.float32)
    depth_norm = yy / float(max(hh - 1, 1))  # 0 at waterline -> 1 at bottom foreground

    # Natural wind ripples (shorter near horizon, wider toward foreground)
    wave_x = (
        np.sin(yy * (0.55 - 0.25 * depth_norm) + t * 2.9 + xx * 0.025) * (1.8 + 4.2 * depth_norm)
        + np.sin(yy * 1.15 - t * 4.1 - xx * 0.04) * (0.9 + 1.8 * depth_norm)
    )
    wave_y = np.cos(yy * 0.72 + t * 2.5 + xx * 0.018) * (0.8 + 1.9 * depth_norm)

    ripple_highlight = np.zeros((hh, hw), dtype=np.float32)
    for rcx, rcy, age, strength in ripples:
        if age <= 0.0 or age > 3.2 or strength <= 0.01:
            continue
        rx = (rcx * 0.5)
        ry = ((rcy - wy) * 0.5)
        dx = xx - rx
        dy = (yy - ry) * 2.4  # elliptical perspective foreshortening
        dist = np.sqrt(dx * dx + dy * dy + 1e-4)
        radius = age * 95.0
        env = np.exp(-((dist - radius) ** 2) / (260.0 + age * 280.0)) * math.exp(-age * 1.05) * strength
        osc = np.sin((dist - radius) * 0.42)
        wave_x += (dx / dist) * osc * env * 12.0
        wave_y += (dy / dist) * osc * env * 6.0
        ripple_highlight += np.maximum(0.0, osc) * env * 0.85

    map_x = np.clip(xx * 2.0 + wave_x * 2.0, 0, w - 1)
    map_y = np.clip(yy * 2.0 + wave_y * 2.0, 0, water_h - 1)
    map_x_full = cv2.resize(map_x, (w, water_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    map_y_full = cv2.resize(map_y, (w, water_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    refl = cv2.remap(mirrored, map_x_full, map_y_full, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Blend reflection with deep translucent pond color + subtle sunlight caustics
    d_full = np.linspace(0.0, 1.0, water_h, dtype=np.float32)[:, None, None]
    fresnel = 0.58 - 0.24 * d_full  # stronger mirror reflection near shoreline/grazing angle
    tint = np.array(depth_tint_bgr, dtype=np.float32)[None, None, :]
    # Subtle underwater green-teal depth gradient
    water_body = tint * (1.15 - 0.35 * d_full)

    # Shimmering horizontal sunlight glints along wave crests
    shimmer_half = np.maximum(
        0.0,
        np.sin(yy * 0.85 + t * 3.1 + xx * 0.03) * np.cos(xx * 0.055 - t * 1.4 + yy * 0.22) - 0.68,
    ) * 2.4
    shimmer_half = shimmer_half * np.exp(-((xx / float(hw) - 0.62) ** 2) / 0.16)
    shimmer_half = np.clip(shimmer_half + ripple_highlight, 0.0, 1.3)
    shimmer = cv2.resize(shimmer_half, (w, water_h), interpolation=cv2.INTER_LINEAR)[:, :, None]

    # Compose water surface
    water = refl.astype(np.float32) * fresnel + water_body * (1.0 - fresnel)
    water += shimmer * np.array([155.0, 215.0, 245.0], dtype=np.float32)[None, None, :] * 0.38

    # Soft waterline transition meniscus (4px blend)
    meniscus_h = min(6, water_h)
    blend_m = np.linspace(0.25, 1.0, meniscus_h, dtype=np.float32)[:, None, None]
    orig_top = frame_bgr[wy : wy + meniscus_h].astype(np.float32)
    water[:meniscus_h] = orig_top * (1.0 - blend_m) + water[:meniscus_h] * blend_m

    frame_bgr[wy:, :, :] = np.clip(water, 0.0, 255.0).astype(np.uint8)
    return frame_bgr


# =============================================================================
# Cinematic Lighting, Volumetric Sunbeams, Bloom & Color Grading
# =============================================================================

def add_volumetric_sunbeams(
    frame_bgr: np.ndarray,
    sun_norm: Tuple[float, float],
    strength: float,
    t: float,
    warm_bgr: Tuple[float, float, float] = (145.0, 212.0, 255.0),
) -> np.ndarray:
    """Fast half-res volumetric sunlight shafts radiating diagonally from the sun."""
    if strength <= 0.01:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    hh, hw = h // 4, w // 4
    yy, xx = np.mgrid[0:hh, 0:hw].astype(np.float32)
    sx = sun_norm[0] * hw
    sy = sun_norm[1] * hh
    dx = xx - sx
    dy = yy - sy
    dist = np.sqrt(dx * dx + dy * dy) / float(hw)
    ang = np.arctan2(dy, dx)

    # Soft multi-frequency ray shafts drifting slowly like light through clouds/canopy
    rays = (
        0.50
        + 0.24 * np.sin(ang * 13.0 + t * 0.16)
        + 0.16 * np.sin(ang * 23.0 - t * 0.11 + 1.4)
        + 0.10 * np.cos(ang * 37.0 + t * 0.07)
    )
    rays = np.power(np.clip(rays, 0.0, 1.0), 1.75)
    hub_fade = np.clip(dist / 0.18, 0.0, 1.0) ** 1.4
    radial = np.exp(-dist * 1.95) * hub_fade
    # Soft solar halo near sun center (never harsh spoke hub!)
    halo = np.exp(-dist * dist * 14.0) * 0.42
    mask = np.clip((rays * radial * 0.56 + halo) * strength, 0.0, 0.58)
    tint = np.array(warm_bgr, dtype=np.float32)[None, None, :]
    rays_small_u8 = np.clip(mask[:, :, None] * tint, 0.0, 255.0).astype(np.uint8)
    rays_full = cv2.resize(rays_small_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.add(frame_bgr, rays_full, dst=frame_bgr)
    return frame_bgr


def apply_cinematic_post(
    frame_bgr: np.ndarray,
    t: float,
    sun_norm: Tuple[float, float] = (0.78, 0.15),
    sun_strength: float = 0.28,
    bloom_strength: float = 0.22,
    grain_strength: float = 3.2,
    vignette_strength: float = 1.0,
) -> np.ndarray:
    """
    Apply full cinema-grade post-processing (SIMD-accelerated for fast 1080p):
      1. Volumetric warm sunbeams
      2. Soft atmospheric bloom on sunlit highlights
      3. Filmic warm-golden/cool-shadow LUT grade
      4. Subtle illustration-paper micro-grain & organic vignette
    """
    h, w = frame_bgr.shape[:2]

    if sun_strength > 0.01:
        add_volumetric_sunbeams(frame_bgr, sun_norm, sun_strength, t)

    if bloom_strength > 0.01:
        small = cv2.resize(frame_bgr, (w // 4, h // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
        lum = small[:, :, 2] * 0.35 + small[:, :, 1] * 0.50 + small[:, :, 0] * 0.15
        bright = np.clip((lum - 175.0) / 80.0, 0.0, 1.0)[:, :, None]
        glow = cv2.GaussianBlur(small * bright, (0, 0), 9.0) * bloom_strength
        glow_u8 = np.clip(glow, 0.0, 255.0).astype(np.uint8)
        glow_full = cv2.resize(glow_u8, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.add(frame_bgr, glow_full, dst=frame_bgr)

    # Single-call SIMD 3-channel filmic LUT grade!
    lut = _cached_filmic_lut().reshape(1, 256, 3)
    cv2.LUT(frame_bgr, lut, dst=frame_bgr)

    if vignette_strength > 0.01 or grain_strength > 0.01:
        _apply_vignette_grain_fast(frame_bgr, vignette_strength, grain_strength)
    return frame_bgr


@lru_cache(maxsize=12)
def _cached_vignette_u8(h: int, w: int, strength_q: int) -> np.ndarray:
    strength = strength_q / 100.0
    vig = _cached_vignette(h, w)
    if abs(strength - 1.0) > 1e-3:
        vig = 1.0 - (1.0 - vig) * strength
    v1 = np.clip(vig * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.repeat(v1, 3, axis=2)


@lru_cache(maxsize=8)
def _cached_grain_u8(h: int, w: int, strength_q: int) -> Tuple[np.ndarray, np.ndarray]:
    g = _cached_canvas_grain(h, w) * (strength_q / 10.0)
    pos = np.clip(g, 0.0, 255.0).astype(np.uint8)
    neg = np.clip(-g, 0.0, 255.0).astype(np.uint8)
    return np.repeat(pos, 3, axis=2), np.repeat(neg, 3, axis=2)


def _apply_vignette_grain_fast(frame_bgr: np.ndarray, vignette_strength: float, grain_strength: float) -> None:
    h, w = frame_bgr.shape[:2]
    if vignette_strength > 0.01:
        vq = int(round(vignette_strength * 100.0))
        cv2.multiply(frame_bgr, _cached_vignette_u8(h, w, vq), dst=frame_bgr, scale=1.0 / 255.0)
    if grain_strength > 0.01:
        gq = int(round(grain_strength * 10.0))
        gpos, gneg = _cached_grain_u8(h, w, gq)
        cv2.add(frame_bgr, gpos, dst=frame_bgr)
        cv2.subtract(frame_bgr, gneg, dst=frame_bgr)


# =============================================================================
# High-DPI Pillow Documentary Typography (for Scene 6 Educational Finale)
# =============================================================================

_FONT_CANDIDATES = {
    "serif_bold": [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
    ],
    "serif": [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ],
    "sans_bold": [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ],
    "sans": [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
}


@lru_cache(maxsize=32)
def load_font(style: str, size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES.get(style, _FONT_CANDIDATES["sans_bold"]):
        if os.path.exists(path):
            return ImageFont.truetype(path, size=max(8, int(size)))
    return ImageFont.load_default()


def draw_text_overlay(
    frame_bgr: np.ndarray,
    texts: Sequence[Dict[str, object]],
) -> np.ndarray:
    """
    Render crisp anti-aliased Pillow typography with fast OpenCV-blurred
    drop shadows and subtle golden glow onto a uint8 BGR frame.
    """
    if not texts:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    scale = w / float(WIDTH)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    glow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    d_main = ImageDraw.Draw(overlay)
    d_shad = ImageDraw.Draw(shadow_layer)
    d_glow = ImageDraw.Draw(glow_layer)

    has_shadow = False
    has_glow = False

    for item in texts:
        alpha = float(item.get("alpha", 1.0))
        if alpha <= 0.01:
            continue
        txt = str(item["text"])
        x = float(item["x"]) * scale
        y = float(item["y"]) * scale
        size = max(10, int(round(float(item.get("size", 32)) * scale)))
        style = str(item.get("style", "sans_bold"))
        col = item.get("color_rgb", (255, 250, 238))
        align = str(item.get("align", "center"))
        font = load_font(style, size)

        bbox = d_main.textbbox((0, 0), txt, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x - tw * 0.5 if align == "center" else x
        ty = y - th * 0.5 - bbox[1]

        a255 = int(clamp(alpha) * 255)
        if item.get("shadow", True):
            has_shadow = True
            off = max(1.5, 2.5 * scale)
            d_shad.text((tx + off, ty + off), txt, font=font, fill=(14, 20, 12, int(a255 * 0.68)))
        if item.get("glow", False):
            has_glow = True
            gcol = item.get("glow_rgb", (255, 224, 130))
            d_glow.text((tx, ty), txt, font=font, fill=(int(gcol[0]), int(gcol[1]), int(gcol[2]), int(a255 * 0.58)))
        d_main.text((tx, ty), txt, font=font, fill=(int(col[0]), int(col[1]), int(col[2]), a255))

    if has_glow:
        g_bgra = cv2.cvtColor(np.asarray(glow_layer, dtype=np.uint8), cv2.COLOR_RGBA2BGRA)
        g_sm = cv2.resize(g_bgra, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        g_sm = cv2.GaussianBlur(g_sm, (0, 0), max(1.5, 3.2 * scale))
        composite_straight_bgra(frame_bgr, cv2.resize(g_sm, (w, h), interpolation=cv2.INTER_LINEAR), opacity=1.0)
    if has_shadow:
        s_bgra = cv2.cvtColor(np.asarray(shadow_layer, dtype=np.uint8), cv2.COLOR_RGBA2BGRA)
        s_sm = cv2.resize(s_bgra, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        s_sm = cv2.GaussianBlur(s_sm, (0, 0), max(1.0, 1.8 * scale))
        composite_straight_bgra(frame_bgr, cv2.resize(s_sm, (w, h), interpolation=cv2.INTER_LINEAR), opacity=1.0)

    main_bgra = cv2.cvtColor(np.asarray(overlay, dtype=np.uint8), cv2.COLOR_RGBA2BGRA)
    return composite_straight_bgra(frame_bgr, main_bgra, opacity=1.0)


# =============================================================================
# Pygame Surface Interop Utility
# =============================================================================

def bgr_to_pygame_surface(frame_bgr: np.ndarray):
    """Convert a uint8 BGR NumPy frame to a Pygame Surface for optional preview/inspection."""
    import pygame

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")

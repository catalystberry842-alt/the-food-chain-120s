"""
animation.py — Shared Master Timeline, Kinematics, Splines, IK & Wind Physics
===========================================================================

Provides the shared synchronization timeline between video and audio modules,
smooth easing curves, parametric Bezier/Catmull-Rom splines, analytical 2-bone
inverse kinematics, traveling grassland wind physics, and organic motion noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

# =============================================================================
# Master Video & Audio Specifications
# =============================================================================

WIDTH: int = 1920
HEIGHT: int = 1080
FPS: int = 30
SAMPLE_RATE: int = 44100

# Scene timing (seconds). Scene 6 finishes cleanly around 124.0s (~2:04, well under 2:15 max)
# giving comfortable breathing room for all 5 organisms, arrows, narration, and final title hold.
SCENE_SPECS: List[Dict[str, object]] = [
    {"id": 1, "slug": "scene1_life_begins",   "title": "Scene 1: Life Begins",            "start": 0.0,   "end": 22.0},
    {"id": 2, "slug": "scene2_grasshopper",   "title": "Scene 2: The Grasshopper",        "start": 22.0,  "end": 42.0},
    {"id": 3, "slug": "scene3_frog_hunts",    "title": "Scene 3: The Frog Hunts",         "start": 42.0,  "end": 62.0},
    {"id": 4, "slug": "scene4_snake",         "title": "Scene 4: The Snake",              "start": 62.0,  "end": 82.0},
    {"id": 5, "slug": "scene5_hawk",          "title": "Scene 5: The Hawk",               "start": 82.0,  "end": 107.0},
    {"id": 6, "slug": "scene6_food_chain",    "title": "Scene 6: The Complete Food Chain","start": 107.0, "end": 124.0},
]

TOTAL_DURATION: float = float(SCENE_SPECS[-1]["end"])
TOTAL_FRAMES: int = int(round(TOTAL_DURATION * FPS))

# Transition crossfade overlap durations between consecutive scenes (seconds)
TRANSITION_OVERLAP: Dict[Tuple[int, int], float] = {
    (1, 2): 1.1,
    (2, 3): 1.1,
    (3, 4): 1.1,
    (4, 5): 1.1,
    (5, 6): 1.4,
}


def get_scene_spec(scene_id: int) -> Dict[str, object]:
    for s in SCENE_SPECS:
        if int(s["id"]) == scene_id:
            return s
    raise ValueError(f"Unknown scene_id: {scene_id}")


# =============================================================================
# Core Math & Easing Primitives
# =============================================================================

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def inv_lerp(a: float, b: float, x: float) -> float:
    if abs(b - a) < 1e-9:
        return 0.0
    return clamp((x - a) / (b - a), 0.0, 1.0)


def remap(x: float, a0: float, a1: float, b0: float, b1: float, ease=None) -> float:
    u = inv_lerp(a0, a1, x)
    if ease is not None:
        u = ease(u)
    return lerp(b0, b1, u)


def smoothstep(u: float) -> float:
    u = clamp(u)
    return u * u * (3.0 - 2.0 * u)


def smootherstep(u: float) -> float:
    u = clamp(u)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def ease_in_quad(u: float) -> float:
    u = clamp(u)
    return u * u


def ease_out_quad(u: float) -> float:
    u = clamp(u)
    return 1.0 - (1.0 - u) * (1.0 - u)


def ease_in_out_quad(u: float) -> float:
    u = clamp(u)
    return 2.0 * u * u if u < 0.5 else 1.0 - ((-2.0 * u + 2.0) ** 2) * 0.5


def ease_in_cubic(u: float) -> float:
    u = clamp(u)
    return u * u * u


def ease_out_cubic(u: float) -> float:
    u = clamp(u)
    return 1.0 - (1.0 - u) ** 3


def ease_in_out_cubic(u: float) -> float:
    u = clamp(u)
    return 4.0 * u * u * u if u < 0.5 else 1.0 - ((-2.0 * u + 2.0) ** 3) * 0.5


def ease_in_quart(u: float) -> float:
    u = clamp(u)
    return u ** 4


def ease_out_quart(u: float) -> float:
    u = clamp(u)
    return 1.0 - (1.0 - u) ** 4


def ease_out_back(u: float, overshoot: float = 1.35) -> float:
    u = clamp(u)
    c3 = overshoot + 1.0
    return 1.0 + c3 * ((u - 1.0) ** 3) + overshoot * ((u - 1.0) ** 2)


def pulse_bell(t: float, t0: float, t1: float, sharp: float = 1.0) -> float:
    """Smooth 0 -> 1 -> 0 bell pulse between t0 and t1."""
    if t <= t0 or t >= t1:
        return 0.0
    u = (t - t0) / (t1 - t0)
    val = math.sin(math.pi * u)
    return val ** sharp


def lerp_rgb(
    c0: Tuple[float, float, float],
    c1: Tuple[float, float, float],
    u: float,
) -> Tuple[float, float, float]:
    u = clamp(u)
    return (
        c0[0] + (c1[0] - c0[0]) * u,
        c0[1] + (c1[1] - c0[1]) * u,
        c0[2] + (c1[2] - c0[2]) * u,
    )


# =============================================================================
# Deterministic Organic Noise & Grassland Wind Physics
# =============================================================================

def hash1d(n: int) -> float:
    """Deterministic pseudo-random float in [0, 1] from an integer."""
    x = math.sin(n * 127.1 + 311.7) * 43758.5453123
    return x - math.floor(x)


def value_noise_1d(x: float, seed: int = 0) -> float:
    i = math.floor(x)
    f = x - i
    u = f * f * (3.0 - 2.0 * f)
    a = hash1d(i + seed * 997)
    b = hash1d(i + 1 + seed * 997)
    return lerp(a, b, u) * 2.0 - 1.0


def fbm_1d(x: float, octaves: int = 3, seed: int = 0) -> float:
    amp = 0.55
    freq = 1.0
    total = 0.0
    norm = 0.0
    for o in range(octaves):
        total += amp * value_noise_1d(x * freq, seed=seed + o * 31)
        norm += amp
        freq *= 2.07
        amp *= 0.5
    return total / max(norm, 1e-6)


def wind_field(x: float, t: float, strength: float = 1.0) -> float:
    """
    Traveling natural meadow breeze combining a broad slow gust wave,
    mid-frequency swaying ripples, and subtle blade-scale flutter.
    Returns signed deflection magnitude (~[-1.2, +1.5] * strength).
    """
    # Slow prevailing warm breeze biased slightly right (+0.28)
    gust = 0.28 + 0.45 * math.sin(t * 0.65 - x * 0.0018)
    wave1 = 0.36 * math.sin(t * 1.95 - x * 0.0046 + 0.7)
    wave2 = 0.16 * math.sin(t * 3.6 - x * 0.0105 + 1.9)
    flutter = 0.08 * fbm_1d(t * 2.4 + x * 0.015, octaves=2, seed=17)
    return (gust + wave1 + wave2 + flutter) * strength


# =============================================================================
# Parametric Splines, Tangents & Analytical 2-Bone IK
# =============================================================================

def quad_bezier(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    u: float,
) -> Tuple[float, float]:
    om = 1.0 - u
    return (
        om * om * p0[0] + 2.0 * om * u * p1[0] + u * u * p2[0],
        om * om * p0[1] + 2.0 * om * u * p1[1] + u * u * p2[1],
    )


def cubic_bezier(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    u: float,
) -> Tuple[float, float]:
    om = 1.0 - u
    om2 = om * om
    u2 = u * u
    return (
        om2 * om * p0[0] + 3.0 * om2 * u * p1[0] + 3.0 * om * u2 * p2[0] + u2 * u * p3[0],
        om2 * om * p0[1] + 3.0 * om2 * u * p1[1] + 3.0 * om * u2 * p2[1] + u2 * u * p3[1],
    )


def sample_cubic_bezier(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    steps: int = 24,
) -> List[Tuple[float, float]]:
    return [cubic_bezier(p0, p1, p2, p3, i / float(steps)) for i in range(steps + 1)]


def parabolic_arc(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    apex_height: float,
    u: float,
) -> Tuple[float, float]:
    """Ballistic jump trajectory from p0 to p1 lifting upward by apex_height (screen -Y)."""
    u = clamp(u)
    x = lerp(p0[0], p1[0], u)
    y_base = lerp(p0[1], p1[1], u)
    lift = 4.0 * apex_height * u * (1.0 - u)
    return (x, y_base - lift)


def polyline_frames(
    pts: Sequence[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a sequence of 2D points, return arrays of shape (N, 2) for:
      (positions, unit_tangents, unit_normals)
    where normal is rotated +90 deg (pointing 'left/down' depending on orientation).
    """
    arr = np.asarray(pts, dtype=np.float64)
    n = len(arr)
    tangents = np.zeros_like(arr)
    if n >= 2:
        tangents[0] = arr[1] - arr[0]
        tangents[-1] = arr[-1] - arr[-2]
        if n > 2:
            tangents[1:-1] = arr[2:] - arr[:-2]
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(lengths, 1e-8)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    return arr, tangents, normals


def solve_ik_2bone(
    hip: Tuple[float, float],
    target: Tuple[float, float],
    l1: float,
    l2: float,
    bend_sign: float = 1.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Analytical 2-link cosine-law Inverse Kinematics solver.
    Returns (knee_xy, foot_xy) with safe clamping for full extension / folding.
    """
    hx, hy = hip
    tx, ty = target
    dx = tx - hx
    dy = ty - hy
    dist = math.hypot(dx, dy)
    min_d = abs(l1 - l2) + 1e-3
    max_d = (l1 + l2) - 1e-3
    clamped_d = clamp(dist, min_d, max_d)

    base_ang = math.atan2(dy, dx)
    cos_a = clamp((l1 * l1 + clamped_d * clamped_d - l2 * l2) / (2.0 * l1 * clamped_d), -1.0, 1.0)
    ang_a = math.acos(cos_a)
    knee_ang = base_ang + bend_sign * ang_a
    kx = hx + l1 * math.cos(knee_ang)
    ky = hy + l1 * math.sin(knee_ang)

    if dist > max_d and dist > 1e-6:
        fx = hx + dx * (max_d / dist)
        fy = hy + dy * (max_d / dist)
    else:
        fx, fy = tx, ty
    return (kx, ky), (fx, fy)


def blink_envelope(t: float, period: float = 3.6, offset: float = 0.8, dur: float = 0.16) -> float:
    """Returns 0.0 (open) to 1.0 (closed) for natural periodic wildlife eye blinking."""
    phase = (t + offset) % period
    if phase < dur:
        u = phase / dur
        return math.sin(math.pi * u) ** 1.4
    return 0.0

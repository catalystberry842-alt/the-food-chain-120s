"""
camera.py — Cinematic 2.5D Virtual Camera Rig & Parallax Projection
===================================================================

Supports smooth push-ins, dolly pull-backs, wildlife tracking shots, macro
close-ups, subtle handheld/breathing sway, impact camera shake, and multi-plane
parallax projection for foreground, midground, subject, and sky layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from animation import HEIGHT, WIDTH, clamp, fbm_1d, lerp, smootherstep


@dataclass
class CameraState:
    """
    Camera state in 1920x1080 logical world coordinates:
      cx, cy     : world point anchored at screen center
      zoom       : 1.0 = native 1920x1080 view, >1.0 = push-in/macro, <1.0 = wide
      roll_deg   : subtle banking tilt (used sparingly during hawk dive/soar)
      shake_amp  : micro shake amplitude in pixels
      dof_fg     : foreground blur intensity (0.0..1.0)
      dof_bg     : subtle background softness (0.0..1.0)
      sun_screen : (x, y) normalized [0..1] screen position of sun for god rays
      sun_strength: intensity of volumetric sunbeams
    """
    cx: float = WIDTH * 0.5
    cy: float = HEIGHT * 0.5
    zoom: float = 1.0
    roll_deg: float = 0.0
    shake_amp: float = 0.0
    dof_fg: float = 0.45
    dof_bg: float = 0.20
    sun_screen: Tuple[float, float] = (0.78, 0.16)
    sun_strength: float = 0.35

    def with_organic_motion(self, t: float, breath_amp: float = 1.6) -> "CameraState":
        """Apply very subtle documentary lens breathing + impact shake if active."""
        bx = fbm_1d(t * 0.42, octaves=2, seed=101) * breath_amp
        by = fbm_1d(t * 0.37 + 13.0, octaves=2, seed=202) * breath_amp * 0.7
        sx = 0.0
        sy = 0.0
        if self.shake_amp > 1e-3:
            sx = math.sin(t * 47.0) * math.cos(t * 19.3) * self.shake_amp
            sy = math.cos(t * 53.0) * math.sin(t * 23.7) * self.shake_amp
        return CameraState(
            cx=self.cx + bx + sx,
            cy=self.cy + by + sy,
            zoom=self.zoom,
            roll_deg=self.roll_deg,
            shake_amp=0.0,
            dof_fg=self.dof_fg,
            dof_bg=self.dof_bg,
            sun_screen=self.sun_screen,
            sun_strength=self.sun_strength,
        )

    def world_to_screen(
        self,
        wx: float,
        wy: float,
        parallax: float = 1.0,
        out_w: int = WIDTH,
        out_h: int = HEIGHT,
    ) -> Tuple[float, float]:
        """Project a world coordinate at a given depth parallax factor onto screen pixels."""
        scale_res = out_w / float(WIDTH)
        eff_zoom = 1.0 + (self.zoom - 1.0) * (0.45 + 0.55 * parallax)
        dx = (wx - WIDTH * 0.5) - (self.cx - WIDTH * 0.5) * parallax
        dy = (wy - HEIGHT * 0.5) - (self.cy - HEIGHT * 0.5) * parallax
        if abs(self.roll_deg) > 1e-3:
            rad = math.radians(self.roll_deg)
            cr, sr = math.cos(rad), math.sin(rad)
            dx, dy = dx * cr - dy * sr, dx * sr + dy * cr
        sx = (WIDTH * 0.5 + dx * eff_zoom) * scale_res
        sy = (HEIGHT * 0.5 + dy * eff_zoom) * scale_res
        return (sx, sy)

    def apply_to_cairo(
        self,
        ctx,
        parallax: float = 1.0,
        out_w: int = WIDTH,
        out_h: int = HEIGHT,
    ) -> None:
        """Configure a PyCairo context transform matrix for drawing in world space."""
        scale_res = out_w / float(WIDTH)
        eff_zoom = 1.0 + (self.zoom - 1.0) * (0.45 + 0.55 * parallax)
        ctx.scale(scale_res, scale_res)
        ctx.translate(WIDTH * 0.5, HEIGHT * 0.5)
        if abs(self.roll_deg) > 1e-3:
            ctx.rotate(math.radians(self.roll_deg))
        ctx.scale(eff_zoom, eff_zoom)
        ctx.translate(
            -(WIDTH * 0.5 + (self.cx - WIDTH * 0.5) * parallax),
            -(HEIGHT * 0.5 + (self.cy - HEIGHT * 0.5) * parallax),
        )

    def sample_layer(
        self,
        layer: np.ndarray,
        parallax: float = 1.0,
        out_w: int = WIDTH,
        out_h: int = HEIGHT,
        world_origin: Tuple[float, float] = (0.0, 0.0),
        border_mode: int = cv2.BORDER_REPLICATE,
    ) -> np.ndarray:
        """
        Warp a pre-baked environment canvas (BGR or BGRA) through the camera
        transform with smooth bilinear filtering. `world_origin` is the top-left
        world coordinate of `layer`.
        """
        scale_res = out_w / float(WIDTH)
        eff_zoom = 1.0 + (self.zoom - 1.0) * (0.45 + 0.55 * parallax)

        # Build 2x3 affine matrix mapping source image pixel -> output screen pixel
        # p_world = p_src + world_origin
        # p_rel   = p_world - (WIDTH/2 + (cx - WIDTH/2)*parallax, HEIGHT/2 + (cy - HEIGHT/2)*parallax)
        # p_out   = scale_res * ( (WIDTH/2, HEIGHT/2) + eff_zoom * R(roll) * p_rel )
        rad = math.radians(self.roll_deg)
        c = math.cos(rad) * eff_zoom * scale_res
        s = math.sin(rad) * eff_zoom * scale_res

        ref_x = WIDTH * 0.5 + (self.cx - WIDTH * 0.5) * parallax - world_origin[0]
        ref_y = HEIGHT * 0.5 + (self.cy - HEIGHT * 0.5) * parallax - world_origin[1]

        tx = (WIDTH * 0.5) * scale_res - (c * ref_x - s * ref_y)
        ty = (HEIGHT * 0.5) * scale_res - (s * ref_x + c * ref_y)

        M = np.array([[c, -s, tx], [s, c, ty]], dtype=np.float32)
        return cv2.warpAffine(
            layer,
            M,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=border_mode,
        )


def interpolate_camera(a: CameraState, b: CameraState, u: float) -> CameraState:
    e = smootherstep(clamp(u))
    return CameraState(
        cx=lerp(a.cx, b.cx, e),
        cy=lerp(a.cy, b.cy, e),
        zoom=lerp(a.zoom, b.zoom, e),
        roll_deg=lerp(a.roll_deg, b.roll_deg, e),
        shake_amp=lerp(a.shake_amp, b.shake_amp, e),
        dof_fg=lerp(a.dof_fg, b.dof_fg, e),
        dof_bg=lerp(a.dof_bg, b.dof_bg, e),
        sun_screen=(lerp(a.sun_screen[0], b.sun_screen[0], e), lerp(a.sun_screen[1], b.sun_screen[1], e)),
        sun_strength=lerp(a.sun_strength, b.sun_strength, e),
    )

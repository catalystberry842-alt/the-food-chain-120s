"""
characters.py — Anatomically Realistic Articulated Wildlife Animation Systems
=============================================================================

Every organism in the food chain has its own independent procedural anatomy,
painterly multi-stop shading, organic surface details, and biomechanical rigging:

  1. GRASS & SEED  : Germinating caryopsis seed (water absorption, splitting coat,
                     branching downward roots + root hairs, pushing shoot), and
                     individually articulated bending blades with midribs, dewdrops,
                     weight deflection, and live chewed bite notches.
  2. GRASSHOPPER   : Hypognathous vertical-faced head, faceted bronze compound eye,
                     chewing mandibles & maxillary palps, saddle pronotum, veined
                     folded tegmina wings, 7-segmented breathing abdomen, curved
                     whip antennae, and 6 articulated legs with iconic chevron-
                     striated saltatorial hind femurs & spiny tibias.
  3. FROG          : Streamlined emerald-olive pond frog with bronze tympanum,
                     protruding metallic copper-gold eyes (horizontal pupil,
                     blinking nictitating membrane, swallowing eye retraction),
                     pulsing throat sac, golden dorsolateral ridges, moist skin
                     speculars, Z-folded/extending webbed hind legs, and ballistic
                     fleshy tongue strike (+ macro eye close-up renderer).
  4. SNAKE         : Continuous 58-segment lateral-undulation spine with S-coil
                     gathering, lightning strike lunge, and non-graphic swallow
                     bulge; hundreds of individually oriented overlapping olive-
                     ochre-umber diamond scales, dorsal sheen, grounded coil
                     shadow, independent scaly head, and flicking forked tongue.
  5. HAWK          : Majestic Red-tailed Hawk with articulated cambered wings
                     (7 individually fanned slotted primaries, barred secondaries,
                     mottled coverts), spreadable 9-feather rufous tail, speckled
                     belly-band, fierce overhanging brow & golden eye, hooked
                     beak, feathered legs & yellow scaly talons that snatch and
                     carry the snake (+ dramatic raptor eye close-up renderer).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cairo
import numpy as np

from animation import (
    clamp,
    cubic_bezier,
    ease_in_out_cubic,
    ease_out_cubic,
    fbm_1d,
    hash1d,
    lerp,
    lerp_rgb,
    polyline_frames,
    solve_ik_2bone,
    wind_field,
)
from drawing_engine import (
    CairoLayer,
    draw_dewdrop,
    draw_organic_poly,
    draw_soft_shadow,
    draw_tapered_ribbon,
)


# =============================================================================
# 1. GERMINATING SEED & ARTICULATED GRASS SYSTEM
# =============================================================================

def eval_grass_blade(
    base_x: float,
    base_y: float,
    height: float,
    base_lean: float,
    wind_val: float,
    grow_u: float,
    s: float,
    bend_load: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Evaluate world position (x, y) and tangent angle (rad) at normalized arc s in [0, 1]
    along an individually growing, wind-swaying, weight-bending grass blade.
    """
    gu = clamp(grow_u)
    eff_h = height * (gu ** 0.85)
    # Young shoots start erect and gradually arch gracefully as they mature
    arch_factor = 0.35 + 0.65 * gu
    sway = (base_lean + wind_val * 0.42 * eff_h * (0.35 + 0.65 * gu)) * arch_factor
    load_dx = bend_load * eff_h * 0.38
    load_dy = abs(bend_load) * eff_h * 0.24

    p0 = (base_x, base_y)
    p1 = (base_x + sway * 0.14 + load_dx * 0.15, base_y - eff_h * 0.38)
    p2 = (base_x + sway * 0.56 + load_dx * 0.62, base_y - eff_h * 0.78 + load_dy * 0.45)
    p3 = (base_x + sway + load_dx, base_y - eff_h * (0.96 - 0.10 * min(1.2, abs(sway) / max(eff_h, 1.0))) + load_dy)

    s_c = clamp(s)
    pt = cubic_bezier(p0, p1, p2, p3, s_c)
    ds = 0.015
    pt_a = cubic_bezier(p0, p1, p2, p3, clamp(s_c - ds))
    pt_b = cubic_bezier(p0, p1, p2, p3, clamp(s_c + ds))
    ang = math.atan2(pt_b[1] - pt_a[1], pt_b[0] - pt_a[0])
    return pt[0], pt[1], ang


def draw_grass_blade(
    ctx: cairo.Context,
    base_x: float,
    base_y: float,
    height: float,
    base_width: float,
    base_lean: float,
    wind_val: float,
    grow_u: float = 1.0,
    color_mix: float = 0.5,
    bend_load: float = 0.0,
    bite_notches: Sequence[Tuple[float, float, float]] = (),
    dew_s: Optional[float] = None,
    alpha: float = 1.0,
    steps: int = 22,
) -> None:
    """
    Draw a single shaded, tapered grass blade with central midrib highlight,
    sunlit & shaded faces, optional morning dewdrop, and real bite cutouts
    `bite_notches`: list of (s_center, bite_depth_0to1, half_width_s) carved into the right/upper edge.
    """
    gu = clamp(grow_u)
    if gu <= 0.02 or alpha <= 0.01:
        return

    pts = []
    hw_left = []
    hw_right = []
    eff_w = base_width * (0.45 + 0.55 * gu)

    for i in range(steps + 1):
        s = i / float(steps)
        px, py, _ = eval_grass_blade(base_x, base_y, height, base_lean, wind_val, gu, s, bend_load)
        pts.append((px, py))
        # Natural lanceolate grass taper (wider near base/mid, crisp graceful spear tip!)
        profile = (0.35 + 0.65 * math.sin(clamp(s * 0.90 + 0.10) * math.pi * 0.5)) * ((1.0 - s) ** 0.84)
        hw = eff_w * 0.55 * profile
        hw_l = hw
        hw_r = hw
        # Carve organic crescent chew notches along the feeding margin!
        for bs, bdepth, bwidth in bite_notches:
            if bdepth <= 0.01:
                continue
            d = abs(s - bs) / max(bwidth, 1e-4)
            if d < 1.0:
                cut = math.cos(d * math.pi * 0.5) ** 1.35 * bdepth
                hw_l = max(-hw * 0.35, hw_l - hw * 1.45 * cut)
        hw_left.append(hw_l)
        hw_right.append(hw_r)

    arr, _, normals = polyline_frames(pts)
    n = len(arr)
    left_pts = arr + normals * np.asarray(hw_left)[:, None]
    right_pts = arr - normals * np.asarray(hw_right)[:, None]

    # Colors: rich botanical meadow greens transitioning to warm sunlit chartreuse near tip
    c_base = lerp_rgb((0.12, 0.28, 0.08), (0.18, 0.34, 0.10), color_mix)
    c_mid = lerp_rgb((0.26, 0.52, 0.14), (0.38, 0.60, 0.18), color_mix)
    c_tip = lerp_rgb((0.62, 0.78, 0.24), (0.76, 0.84, 0.30), color_mix)

    # Full blade polygon
    ctx.new_path()
    ctx.move_to(float(left_pts[0, 0]), float(left_pts[0, 1]))
    for i in range(1, n):
        ctx.line_to(float(left_pts[i, 0]), float(left_pts[i, 1]))
    ctx.line_to(float(arr[-1, 0]), float(arr[-1, 1]))
    for i in range(n - 1, -1, -1):
        ctx.line_to(float(right_pts[i, 0]), float(right_pts[i, 1]))
    ctx.close_path()

    pat = cairo.LinearGradient(float(arr[0, 0]), float(arr[0, 1]), float(arr[-1, 0]), float(arr[-1, 1]))
    pat.add_color_stop_rgba(0.0, *c_base, alpha)
    pat.add_color_stop_rgba(0.55, *c_mid, alpha)
    pat.add_color_stop_rgba(1.0, *c_tip, alpha)
    ctx.set_source(pat)
    ctx.fill_preserve()

    # Subtle shaded half-fold along the right side of the central midrib for 3D botanical depth
    ctx.save()
    ctx.clip()
    ctx.new_path()
    ctx.move_to(float(arr[0, 0]), float(arr[0, 1]))
    for i in range(1, n):
        ctx.line_to(float(arr[i, 0]), float(arr[i, 1]))
    for i in range(n - 1, -1, -1):
        ctx.line_to(float(right_pts[i, 0]), float(right_pts[i, 1]))
    ctx.close_path()
    ctx.set_source_rgba(0.06, 0.16, 0.04, 0.26 * alpha)
    ctx.fill()

    # Pale sunlit central midrib vein
    ctx.new_path()
    ctx.move_to(float(arr[0, 0]), float(arr[0, 1]))
    for i in range(1, n - 1):
        ctx.line_to(float(arr[i, 0]), float(arr[i, 1]))
    ctx.set_source_rgba(0.84, 0.94, 0.52, 0.42 * alpha)
    ctx.set_line_width(max(0.8, eff_w * 0.11))
    ctx.stroke()

    # Fresh pale chewed edge rim if bitten
    if bite_notches:
        ctx.new_path()
        ctx.move_to(float(left_pts[0, 0]), float(left_pts[0, 1]))
        for i in range(1, n):
            ctx.line_to(float(left_pts[i, 0]), float(left_pts[i, 1]))
        ctx.set_source_rgba(0.88, 0.96, 0.64, 0.35 * alpha)
        ctx.set_line_width(1.6)
        ctx.stroke()
    ctx.restore()

    # Delicate dark forest painterly outline
    ctx.new_path()
    ctx.move_to(float(left_pts[0, 0]), float(left_pts[0, 1]))
    for i in range(1, n):
        ctx.line_to(float(left_pts[i, 0]), float(left_pts[i, 1]))
    ctx.line_to(float(arr[-1, 0]), float(arr[-1, 1]))
    for i in range(n - 1, -1, -1):
        ctx.line_to(float(right_pts[i, 0]), float(right_pts[i, 1]))
    ctx.close_path()
    ctx.set_source_rgba(0.08, 0.18, 0.05, 0.42 * alpha)
    ctx.set_line_width(max(0.9, eff_w * 0.07))
    ctx.stroke()

    # Optional sparkling morning dewdrop clinging to the blade
    if dew_s is not None and gu > 0.55:
        di = min(n - 1, int(clamp(dew_s) * (n - 1)))
        dx, dy = float(left_pts[di, 0]), float(left_pts[di, 1])
        draw_dewdrop(ctx, dx, dy + 2.0, r=max(3.0, eff_w * 0.28), alpha=alpha * 0.9)


def draw_germinating_seed(
    ctx: cairo.Context,
    cx: float,
    cy: float,
    soil_surface_y: float,
    absorb_u: float,
    split_u: float,
    root_u: float,
    shoot_u: float,
    t: float,
    scale: float = 1.0,
) -> None:
    """
    Animate a realistic meadow grass seed beneath the soil:
      - absorb_u : moisture droplets percolate down, seed swells & glistens
      - split_u  : golden husk splits at apex revealing pale embryo
      - root_u   : branching creamy-white radicle + fine root hairs grow downward
      - shoot_u  : pale-green coleoptile pushes up through soil clods into morning light
    """
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(scale, scale)

    # 1. Percolating subterranean moisture halo & droplets (absorb_u)
    if absorb_u > 0.01:
        halo_r = lerp(24.0, 74.0, absorb_u)
        hpat = cairo.RadialGradient(0, 0, 8, 0, 0, halo_r)
        hpat.add_color_stop_rgba(0.0, 0.28, 0.56, 0.72, 0.28 * math.sin(clamp(absorb_u) * math.pi))
        hpat.add_color_stop_rgba(0.6, 0.18, 0.36, 0.48, 0.15 * absorb_u)
        hpat.add_color_stop_rgba(1.0, 0.10, 0.20, 0.26, 0.0)
        ctx.set_source(hpat)
        ctx.arc(0, 0, halo_r, 0, math.tau)
        ctx.fill()

        # Tiny shimmering water beads soaking downward into the seed
        for d_i in range(6):
            du = clamp((absorb_u * 1.45) - d_i * 0.10)
            if 0.02 < du < 0.98:
                ang = -math.pi * 0.5 + (d_i - 2.5) * 0.36
                dist = lerp(115.0, 14.0, ease_in_out_cubic(du))
                wx = math.cos(ang) * dist * 0.9
                wy = math.sin(ang) * dist
                draw_dewdrop(ctx, wx, wy, r=4.2 * (1.0 - 0.35 * du), alpha=math.sin(du * math.pi))

    # 2. Growing branching root system downward (root_u)
    if root_u > 0.01:
        ru = ease_out_cubic(clamp(root_u))
        root_len = 235.0 * ru
        steps = 26
        r_pts = []
        r_hw = []
        for i in range(steps + 1):
            s = i / float(steps)
            rx = math.sin(s * 5.2 + 0.4) * 16.0 * s + math.cos(s * 9.0) * 4.5 * s
            ry = 14.0 + s * root_len
            r_pts.append((rx, ry))
            r_hw.append(max(0.7, (5.8 * (1.0 - s * 0.85)) * (0.5 + 0.5 * ru)))

        # Primary taproot (radicle)
        draw_tapered_ribbon(ctx, r_pts, r_hw, close_tip=True)
        rpat = cairo.LinearGradient(0, 12, 0, 14 + root_len)
        rpat.add_color_stop_rgb(0.0, 0.78, 0.71, 0.52)
        rpat.add_color_stop_rgb(0.6, 0.88, 0.84, 0.68)
        rpat.add_color_stop_rgb(1.0, 0.94, 0.91, 0.80)
        ctx.set_source(rpat)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.32, 0.24, 0.14, 0.45)
        ctx.set_line_width(1.1)
        ctx.stroke()

        # Branching secondary lateral roots & delicate root hairs
        branches = [
            (0.18, -1.0, 68.0, 0.22),
            (0.26,  1.0, 74.0, 0.28),
            (0.38, -1.0, 82.0, 0.36),
            (0.46,  1.0, 78.0, 0.42),
            (0.58, -1.0, 64.0, 0.52),
            (0.68,  1.0, 56.0, 0.60),
            (0.78, -1.0, 40.0, 0.70),
        ]
        for b_idx, (bs, side, blen, delay) in enumerate(branches):
            bu = clamp((ru - delay) / max(1.0 - delay, 0.15))
            if bu <= 0.02:
                continue
            ri = min(len(r_pts) - 1, int(bs * (len(r_pts) - 1)))
            bx, by = r_pts[ri]
            cur_l = blen * ease_out_cubic(bu)
            b_pts = []
            b_hw = []
            for j in range(12):
                u = j / 11.0
                px = bx + side * cur_l * (0.2 + 0.8 * u) + math.sin(u * 4.0 + b_idx) * 6.0 * u
                py = by + cur_l * 0.62 * (u ** 1.15)
                b_pts.append((px, py))
                b_hw.append(max(0.45, 2.5 * (1.0 - u * 0.85) * bu))
            draw_tapered_ribbon(ctx, b_pts, b_hw, close_tip=True)
            ctx.set_source_rgba(0.88, 0.84, 0.68, 0.88)
            ctx.fill()

            # Fine translucent root hairs sprouting off each lateral root
            if bu > 0.35:
                ctx.set_source_rgba(0.94, 0.90, 0.78, 0.46 * bu)
                ctx.set_line_width(0.85)
                for h_i in range(2, len(b_pts) - 1, 2):
                    hx, hy = b_pts[h_i]
                    for sgn in (-1, 1):
                        ctx.move_to(hx, hy)
                        ctx.line_to(hx + sgn * 8.0 * bu, hy + 7.5 * bu)
                ctx.stroke()

    # 3. Emerging green shoot (coleoptile) pushing upward through soil (shoot_u)
    if shoot_u > 0.01:
        su = ease_in_out_cubic(clamp(shoot_u))
        dist_to_surf = (cy - soil_surface_y) / max(scale, 1e-4)
        shoot_len = lerp(18.0, dist_to_surf + 165.0, su)
        s_pts = []
        s_hw = []
        wind_tip = wind_field(cx, t, strength=0.65) * clamp((shoot_u - 0.55) / 0.45)
        for i in range(22):
            u = i / 21.0
            sx = math.sin(u * 2.8) * 5.0 * (1.0 - u * 0.5) + wind_tip * 28.0 * (u ** 2.1)
            sy = -12.0 - u * shoot_len
            s_pts.append((sx, sy))
            s_hw.append(max(1.0, (6.5 * (1.0 - u * 0.78)) * (0.55 + 0.45 * su)))

        draw_tapered_ribbon(ctx, s_pts, s_hw, close_tip=True)
        spat = cairo.LinearGradient(0, -10, 0, -12 - shoot_len)
        spat.add_color_stop_rgb(0.0, 0.82, 0.84, 0.54)   # pale subterranean sheath
        spat.add_color_stop_rgb(0.45, 0.46, 0.72, 0.24)  # emerging chlorophyll green
        spat.add_color_stop_rgb(1.0, 0.64, 0.85, 0.28)   # sunlit fresh shoot tip
        ctx.set_source(spat)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.14, 0.30, 0.08, 0.55)
        ctx.set_line_width(1.2)
        ctx.stroke()

        # Once shoot nears/breaks the soil surface, displaced soil crumbs push outward!
        break_u = clamp((shoot_u - 0.44) / 0.30)
        if break_u > 0.01:
            sy_local = -dist_to_surf
            for c_i in range(9):
                side = -1.0 if c_i % 2 == 0 else 1.0
                spread = (10.0 + c_i * 5.5) * ease_out_cubic(break_u)
                lift = math.sin(clamp(break_u) * math.pi) * (10.0 + (c_i % 3) * 6.0)
                clod_x = side * spread
                clod_y = sy_local - lift + (c_i % 3) * 3.5
                ctx.arc(clod_x, clod_y, 3.8 + (c_i % 3) * 1.4, 0, math.tau)
                ctx.set_source_rgba(0.32, 0.20, 0.12, 0.90 * (1.0 - 0.35 * break_u))
                ctx.fill()

    # 4. Sculpted meadow grass seed (caryopsis) with splitting husk
    swell = 1.0 + 0.16 * clamp(absorb_u)
    ctx.save()
    ctx.scale(swell, swell)
    # Soft pocket shadow behind seed
    draw_soft_shadow(ctx, 2, 5, 28, 17, alpha=0.52, color=(0.06, 0.04, 0.02))

    # Left & right halves separate slightly as split_u increases
    gap = 4.2 * clamp(split_u)
    for side in (-1.0, 1.0):
        ctx.save()
        ctx.translate(side * gap * 0.45, 0)
        ctx.rotate(side * 0.16 * clamp(split_u))
        ctx.new_path()
        ctx.move_to(0, -24)
        ctx.curve_to(side * 20, -18, side * 22, 16, 0, 24)
        ctx.curve_to(side * 4, 8, side * 4, -8, 0, -24)
        ctx.close_path()

        g = cairo.RadialGradient(side * 6 - 5, -8, 2, 0, 2, 28)
        g.add_color_stop_rgb(0.0, 0.88, 0.72, 0.42)
        g.add_color_stop_rgb(0.55, 0.68, 0.48, 0.24)
        g.add_color_stop_rgb(1.0, 0.40, 0.26, 0.12)
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.24, 0.14, 0.06, 0.78)
        ctx.set_line_width(1.6)
        ctx.stroke()
        ctx.restore()

    # Pale embryo visible inside the crack when split
    if split_u > 0.05:
        ctx.new_path()
        ctx.move_to(0, -22)
        ctx.curve_to(-gap * 0.8, -4, -gap * 0.7, 10, 0, 21)
        ctx.curve_to(gap * 0.7, 10, gap * 0.8, -4, 0, -22)
        ctx.close_path()
        ctx.set_source_rgba(0.92, 0.90, 0.66, 0.92 * clamp(split_u * 1.4))
        ctx.fill()

    # Moist specular sheen on seed husk during water absorption
    if absorb_u > 0.15:
        ctx.save()
        ctx.translate(-5, -8)
        ctx.rotate(-0.35)
        ctx.scale(4.5, 9.5)
        ctx.arc(0, 0, 1.0, 0, math.tau)
        ctx.set_source_rgba(1.0, 0.97, 0.86, 0.40 * clamp(absorb_u))
        ctx.fill()
        ctx.restore()

    ctx.restore()
    ctx.restore()


# =============================================================================
# 2. ARTICULATED REALISTIC GRASSHOPPER RIG
# =============================================================================

def _draw_insect_leg_segment(
    ctx: cairo.Context,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    w0: float,
    w1: float,
    col_lit: Tuple[float, float, float],
    col_shd: Tuple[float, float, float],
    alpha: float = 1.0,
) -> None:
    pts = [p0, ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5), p1]
    draw_tapered_ribbon(ctx, pts, [w0 * 0.5, (w0 + w1) * 0.28, w1 * 0.5], close_tip=False)
    pat = cairo.LinearGradient(p0[0], p0[1], p1[0], p1[1])
    pat.add_color_stop_rgba(0.0, *col_lit, alpha)
    pat.add_color_stop_rgba(1.0, *col_shd, alpha)
    ctx.set_source(pat)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.14, 0.16, 0.06, 0.65 * alpha)
    ctx.set_line_width(1.1)
    ctx.stroke()


def _draw_grasshopper_hind_leg(
    ctx: cairo.Context,
    hip: Tuple[float, float],
    hop_extend: float,
    walk_offset: float,
    is_far: bool,
    alpha: float = 1.0,
) -> None:
    """
    Iconic saltatorial grasshopper hind leg:
      - Large muscular chevron-striated femur angled UP-and-BACK
      - Long slender spiny tibia angled DOWN-and-BACK (unfolds smoothly during hop_extend)
      - Articulated tarsus & gripping claws
    """
    shade = 0.68 if is_far else 1.0
    hx, hy = hip

    # Femur angle sweeps from steep folded (-2.38 rad) toward extended backward kick (-2.95 rad)
    femur_ang = lerp(-2.34, -2.92, clamp(hop_extend)) + walk_offset * 0.18
    femur_len = 94.0
    kx = hx + math.cos(femur_ang) * femur_len
    ky = hy + math.sin(femur_ang) * femur_len

    # Tibia folds sharply downward from knee (+1.88 rad) and kicks out straight on hop (+2.85 rad)
    tibia_ang = lerp(1.92, 2.95, clamp(hop_extend)) + walk_offset * 0.14
    tibia_len = 92.0
    tx = kx + math.cos(tibia_ang) * tibia_len
    ty = ky + math.sin(tibia_ang) * tibia_len

    # Tarsus foot resting/gripping flat forward
    tarsus_ang = lerp(0.02, 2.6, clamp(hop_extend))
    fx = tx + math.cos(tarsus_ang) * 26.0
    fy = ty + math.sin(tarsus_ang) * 26.0

    # Draw muscular sculpted hind femur (bulging near proximal third, tapering to knee)
    dx, dy = kx - hx, ky - hy
    ux, uy = dx / femur_len, dy / femur_len
    nx, ny = -uy, ux

    ctx.new_path()
    ctx.move_to(hx - nx * 4.5, hy - ny * 4.5)
    ctx.curve_to(
        hx + dx * 0.25 - nx * 16.5, hy + dy * 0.25 - ny * 16.5,
        hx + dx * 0.65 - nx * 11.0, hy + dy * 0.65 - ny * 11.0,
        kx - nx * 4.2, ky - ny * 4.2,
    )
    ctx.line_to(kx + nx * 4.2, ky + ny * 4.2)
    ctx.curve_to(
        hx + dx * 0.65 + nx * 8.5, hy + dy * 0.65 + ny * 8.5,
        hx + dx * 0.25 + nx * 13.0, hy + dy * 0.25 + ny * 13.0,
        hx + nx * 4.5, hy + ny * 4.5,
    )
    ctx.close_path()

    fpat = cairo.LinearGradient(hx - nx * 14, hy - ny * 14, hx + nx * 12, hy + ny * 12)
    fpat.add_color_stop_rgba(0.0, 0.66 * shade, 0.72 * shade, 0.28 * shade, alpha)
    fpat.add_color_stop_rgba(0.45, 0.44 * shade, 0.56 * shade, 0.20 * shade, alpha)
    fpat.add_color_stop_rgba(1.0, 0.22 * shade, 0.30 * shade, 0.11 * shade, alpha)
    ctx.set_source(fpat)
    ctx.fill_preserve()

    # Signature herringbone / chevron muscle striations on near femur!
    if not is_far:
        ctx.save()
        ctx.clip_preserve()
        ctx.set_source_rgba(0.18, 0.24, 0.08, 0.55 * alpha)
        ctx.set_line_width(1.6)
        for s_i in range(2, 10):
            u = s_i / 11.0
            cx_s = hx + dx * u
            cy_s = hy + dy * u
            w_s = 11.5 * math.sin(u * math.pi)
            ctx.move_to(cx_s - ux * 5.0 - nx * w_s, cy_s - uy * 5.0 - ny * w_s)
            ctx.line_to(cx_s + ux * 3.0, cy_s + uy * 3.0)
            ctx.line_to(cx_s - ux * 5.0 + nx * w_s, cy_s - uy * 5.0 + ny * w_s)
        ctx.stroke()
        ctx.restore()

    ctx.set_source_rgba(0.12, 0.16, 0.06, 0.72 * alpha)
    ctx.set_line_width(1.5)
    ctx.stroke()

    # Warm amber-brown knee cap joint
    ctx.arc(kx, ky, 5.2, 0, math.tau)
    ctx.set_source_rgba(0.56 * shade, 0.34 * shade, 0.14 * shade, alpha)
    ctx.fill()

    # Slender spiny tibia (warm golden-ochre/amber-olive)
    _draw_insect_leg_segment(
        ctx, (kx, ky), (tx, ty), 6.2, 3.6,
        (0.62 * shade, 0.48 * shade, 0.22 * shade),
        (0.42 * shade, 0.30 * shade, 0.12 * shade),
        alpha=alpha,
    )
    # Tiny sharp rear tibial spines!
    if not is_far:
        tdx, tdy = tx - kx, ty - ky
        tux, tuy = tdx / tibia_len, tdy / tibia_len
        tnx, tny = -tuy, tux
        ctx.set_source_rgba(0.16, 0.12, 0.05, 0.80 * alpha)
        ctx.set_line_width(1.3)
        for sp in range(2, 9):
            u = sp / 10.0
            sx = kx + tdx * u
            sy = ky + tdy * u
            ctx.move_to(sx, sy)
            ctx.line_to(sx - tnx * 5.5 + tux * 3.5, sy - tny * 5.5 + tuy * 3.5)
        ctx.stroke()

    # Segmented gripping tarsus + paired tarsal claws
    _draw_insect_leg_segment(
        ctx, (tx, ty), (fx, fy), 3.4, 2.0,
        (0.54 * shade, 0.42 * shade, 0.18 * shade),
        (0.30 * shade, 0.22 * shade, 0.10 * shade),
        alpha=alpha,
    )


def draw_grasshopper(
    ctx: cairo.Context,
    x: float,
    y: float,
    scale: float = 1.0,
    facing: float = 1.0,
    body_angle: float = 0.0,
    t: float = 0.0,
    walk_phase: float = 0.0,
    hop_extend: float = 0.0,
    chew_phase: float = 0.0,
    head_nod: float = 0.0,
    wing_flutter: float = 0.0,
    draw_shadow: bool = True,
    alpha: float = 1.0,
) -> None:
    """
    Render a richly illustrated, anatomically realistic field grasshopper.
    Origin `(x, y)` sits at the center of the thorax (facing=+1 -> head on right).
    """
    if alpha <= 0.01:
        return
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(body_angle)
    ctx.scale(scale * facing, scale)

    if draw_shadow:
        draw_soft_shadow(ctx, -18, 46, 95, 18, alpha=0.28 * alpha)

    breath = math.sin(t * 3.4) * 1.8
    w1 = math.sin(walk_phase)
    w2 = math.sin(walk_phase + math.pi)

    # 1. FAR LEGS (3 articulated background legs)
    _draw_grasshopper_hind_leg(ctx, (-16, -2), hop_extend=hop_extend, walk_offset=w2 * 0.4, is_far=True, alpha=alpha)
    # Far midleg
    m_hip_f = (4, 6)
    m_tgt_f = (10 + w1 * 14, 48 - max(0.0, w1) * 8)
    m_kn_f, m_ft_f = solve_ik_2bone(m_hip_f, m_tgt_f, 34, 34, bend_sign=-1.0)
    _draw_insect_leg_segment(ctx, m_hip_f, m_kn_f, 5.5, 4.0, (0.34, 0.44, 0.16), (0.24, 0.32, 0.12), alpha)
    _draw_insect_leg_segment(ctx, m_kn_f, m_ft_f, 3.8, 2.2, (0.36, 0.38, 0.15), (0.22, 0.24, 0.10), alpha)
    # Far foreleg
    f_hip_f = (26, 8)
    f_tgt_f = (52 + w2 * 12, 46 - max(0.0, w2) * 8)
    f_kn_f, f_ft_f = solve_ik_2bone(f_hip_f, f_tgt_f, 28, 28, bend_sign=-1.0)
    _draw_insect_leg_segment(ctx, f_hip_f, f_kn_f, 5.0, 3.6, (0.34, 0.44, 0.16), (0.24, 0.32, 0.12), alpha)
    _draw_insect_leg_segment(ctx, f_kn_f, f_ft_f, 3.4, 2.0, (0.36, 0.38, 0.15), (0.22, 0.24, 0.10), alpha)

    # 2. SEGMENTED ABDOMEN (7 tapered overlapping plates, golden-cream belly + olive back)
    abd_segs = 7
    for i in range(abd_segs - 1, -1, -1):
        u = i / float(abd_segs - 1)  # 0 at thorax -> 1 at tail tip
        sx = lerp(-18.0, -118.0, u)
        sy = lerp(2.0, 12.0, u ** 1.3) + breath * 0.45 * math.sin(u * math.pi)
        rx = lerp(20.0, 8.5, u)
        ry = lerp(15.5, 5.2, u) + breath * 0.25 * (1.0 - u)

        ctx.save()
        ctx.translate(sx, sy)
        ctx.rotate(-0.10 * u)
        ctx.scale(rx, ry)
        ctx.arc(0, 0, 1.0, 0, math.tau)
        apat = cairo.LinearGradient(0, -1.0, 0, 1.0)
        apat.add_color_stop_rgba(0.0, 0.46, 0.60, 0.20, alpha)
        apat.add_color_stop_rgba(0.50, 0.56, 0.64, 0.24, alpha)
        apat.add_color_stop_rgba(0.85, 0.78, 0.72, 0.34, alpha)
        apat.add_color_stop_rgba(1.0, 0.48, 0.44, 0.20, alpha)
        ctx.set_source(apat)
        ctx.fill()
        ctx.restore()

        # Segment seam line & tiny dark spiracle dot
        ctx.save()
        ctx.translate(sx, sy)
        ctx.set_source_rgba(0.18, 0.24, 0.08, 0.48 * alpha)
        ctx.set_line_width(1.2)
        ctx.arc(0, 0, ry * 0.96, -1.1, 1.1)
        ctx.stroke()
        if 0 < i < abd_segs - 1:
            ctx.arc(-rx * 0.15, ry * 0.22, 1.5, 0, math.tau)
            ctx.set_source_rgba(0.16, 0.18, 0.08, 0.65 * alpha)
            ctx.fill()
        ctx.restore()

    # 3. FOLDED VEINED FOREWINGS (TEGMINA) + SUBTLE FLUTTERING HINDWING
    if wing_flutter > 0.02:
        ctx.save()
        ctx.translate(2, -12)
        ctx.rotate(-0.35 - 0.55 * wing_flutter * (0.5 + 0.5 * math.sin(t * 48.0)))
        ctx.new_path()
        ctx.move_to(0, 0)
        ctx.curve_to(-35, -30, -95, -24, -125, -4)
        ctx.curve_to(-92, 14, -34, 14, 0, 4)
        ctx.close_path()
        ctx.set_source_rgba(0.86, 0.90, 0.68, 0.45 * alpha)
        ctx.fill()
        ctx.restore()

    ctx.save()
    ctx.translate(8, -11)
    ctx.rotate(-0.08 - 0.18 * wing_flutter)
    ctx.new_path()
    ctx.move_to(0, 0)
    ctx.curve_to(-32, -13, -96, -10, -142, 6)
    ctx.curve_to(-124, 18, -56, 17, 2, 11)
    ctx.close_path()

    wpat = cairo.LinearGradient(-10, -12, -120, 14)
    wpat.add_color_stop_rgba(0.0, 0.50, 0.64, 0.22, 0.92 * alpha)
    wpat.add_color_stop_rgba(0.50, 0.58, 0.66, 0.28, 0.88 * alpha)
    wpat.add_color_stop_rgba(1.0, 0.46, 0.48, 0.24, 0.84 * alpha)
    ctx.set_source(wpat)
    ctx.fill_preserve()

    # Delicate longitudinal & cross wing venation + organic speckles
    ctx.save()
    ctx.clip_preserve()
    ctx.set_source_rgba(0.24, 0.34, 0.10, 0.50 * alpha)
    ctx.set_line_width(1.1)
    for vy in (-5.0, -1.0, 3.5, 7.5):
        ctx.move_to(-4, vy * 0.6)
        ctx.curve_to(-48, vy - 2.0, -96, vy + 1.0, -136, 7.0)
        ctx.stroke()
    ctx.set_line_width(0.8)
    for vx in range(-18, -126, -12):
        ctx.move_to(vx, -8)
        ctx.line_to(vx - 5, 14)
    ctx.stroke()
    # Natural mottled wing spots
    for sp in range(7):
        spx = -24 - sp * 15.0
        spy = -2.0 + math.sin(sp * 2.1) * 5.0
        ctx.arc(spx, spy, 2.8 + (sp % 2), 0, math.tau)
        ctx.set_source_rgba(0.24, 0.26, 0.12, 0.32 * alpha)
        ctx.fill()
    ctx.restore()

    ctx.set_source_rgba(0.16, 0.24, 0.08, 0.68 * alpha)
    ctx.set_line_width(1.3)
    ctx.stroke()
    ctx.restore()

    # 4. SADDLE-SHAPED PRONOTUM (THORACIC ARMOR SHIELD)
    ctx.new_path()
    ctx.move_to(38, -16)
    ctx.curve_to(24, -22, 6, -20, -12, -13)
    ctx.curve_to(-14, 4, -8, 18, 6, 19)
    ctx.curve_to(20, 19, 32, 13, 39, 4)
    ctx.close_path()
    ppat = cairo.LinearGradient(14, -22, 14, 20)
    ppat.add_color_stop_rgba(0.0, 0.58, 0.72, 0.25, alpha)
    ppat.add_color_stop_rgba(0.42, 0.38, 0.56, 0.18, alpha)
    ppat.add_color_stop_rgba(0.75, 0.28, 0.42, 0.13, alpha)
    ppat.add_color_stop_rgba(1.0, 0.52, 0.58, 0.24, alpha)
    ctx.set_source(ppat)
    ctx.fill_preserve()

    # Dark olive lateral racing stripe & dorsal keel highlight on pronotum
    ctx.save()
    ctx.clip_preserve()
    ctx.move_to(38, -8)
    ctx.curve_to(20, -11, 2, -8, -12, -2)
    ctx.set_source_rgba(0.18, 0.26, 0.09, 0.52 * alpha)
    ctx.set_line_width(6.0)
    ctx.stroke()
    ctx.move_to(35, -15)
    ctx.curve_to(20, -18, 4, -17, -10, -12)
    ctx.set_source_rgba(0.84, 0.90, 0.46, 0.52 * alpha)
    ctx.set_line_width(2.2)
    ctx.stroke()
    ctx.restore()

    ctx.set_source_rgba(0.14, 0.20, 0.07, 0.72 * alpha)
    ctx.set_line_width(1.5)
    ctx.stroke()

    # 5. HYPOGNATHOUS HEAD CAPSULE, COMPOUND EYE, CHEWING MANDIBLES & ANTENNAE
    ctx.save()
    ctx.translate(36, -4)
    ctx.rotate(head_nod)

    # Vertical sculpted insect head capsule
    ctx.new_path()
    ctx.move_to(-2, -14)
    ctx.curve_to(12, -18, 25, -10, 26, 4)
    ctx.curve_to(26, 18, 18, 28, 8, 27)
    ctx.curve_to(-2, 26, -6, 12, -4, -4)
    ctx.close_path()
    hpat = cairo.LinearGradient(12, -16, 4, 28)
    hpat.add_color_stop_rgba(0.0, 0.62, 0.76, 0.28, alpha)
    hpat.add_color_stop_rgba(0.55, 0.42, 0.58, 0.20, alpha)
    hpat.add_color_stop_rgba(1.0, 0.30, 0.42, 0.14, alpha)
    ctx.set_source(hpat)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.14, 0.20, 0.07, 0.74 * alpha)
    ctx.set_line_width(1.5)
    ctx.stroke()

    # Articulated chewing mandibles & maxillary palps at bottom of head
    bite = 0.5 - 0.5 * math.cos(chew_phase * math.tau)
    # Left/front mandible
    ctx.save()
    ctx.translate(16, 24)
    ctx.rotate(0.28 * bite)
    ctx.new_path()
    ctx.move_to(-4, -2)
    ctx.line_to(5, 8)
    ctx.line_to(-2, 11)
    ctx.line_to(-6, 3)
    ctx.close_path()
    ctx.set_source_rgba(0.28, 0.18, 0.08, alpha)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.12, 0.08, 0.04, 0.8 * alpha)
    ctx.set_line_width(1.0)
    ctx.stroke()
    ctx.restore()
    # Rear mandible
    ctx.save()
    ctx.translate(9, 25)
    ctx.rotate(-0.30 * bite)
    ctx.new_path()
    ctx.move_to(-3, -2)
    ctx.line_to(4, 9)
    ctx.line_to(-3, 10)
    ctx.close_path()
    ctx.set_source_rgba(0.22, 0.14, 0.06, alpha)
    ctx.fill()
    ctx.restore()
    # Articulated sensory palps tapping the leaf edge
    for p_idx, (px0, py0) in enumerate([(19, 21), (13, 24)]):
        p_sw = math.sin(chew_phase * math.tau + p_idx * 1.4 + t * 4.5) * 4.5
        ctx.move_to(px0, py0)
        ctx.curve_to(px0 + 7, py0 + 6 + p_sw * 0.4, px0 + 11, py0 + 13 + p_sw, px0 + 8, py0 + 17 + p_sw)
        ctx.set_source_rgba(0.46, 0.40, 0.18, 0.90 * alpha)
        ctx.set_line_width(1.7)
        ctx.stroke()

    # Realistic large vertical-oval faceted bronze-amber COMPOUND EYE (NO cartoon white eyeball!)
    ctx.save()
    ctx.translate(14, -4)
    ctx.rotate(0.14)
    ctx.scale(7.5, 11.2)
    ctx.arc(0, 0, 1.0, 0, math.tau)
    epat = cairo.RadialGradient(0.25, -0.30, 0.08, 0.0, 0.1, 1.1)
    epat.add_color_stop_rgba(0.0, 0.72, 0.48, 0.22, alpha)
    epat.add_color_stop_rgba(0.45, 0.40, 0.24, 0.10, alpha)
    epat.add_color_stop_rgba(0.85, 0.20, 0.12, 0.06, alpha)
    epat.add_color_stop_rgba(1.0, 0.10, 0.07, 0.04, alpha)
    ctx.set_source(epat)
    ctx.fill_preserve()
    # Subtle compound eye vertical band / pseudopupil glint
    ctx.set_source_rgba(0.10, 0.08, 0.04, 0.72 * alpha)
    ctx.set_line_width(0.14)
    ctx.stroke()
    # Crisp sky reflection highlight dot
    ctx.arc(0.26, -0.35, 0.20, 0, math.tau)
    ctx.set_source_rgba(1.0, 0.97, 0.88, 0.82 * alpha)
    ctx.fill()
    ctx.restore()

    # Two independently moving segmented whip antennae
    for a_i in range(2):
        sw1 = math.sin(t * (3.1 + a_i * 0.8) + a_i * 2.1) * 10.0 + fbm_1d(t * 2.2 + a_i * 7, seed=41 + a_i) * 8.0
        sw2 = math.cos(t * (4.2 + a_i * 0.6) + a_i * 1.3) * 14.0
        ax0, ay0 = (22, -10) if a_i == 0 else (18, -12)
        c1x, c1y = ax0 + 22, ay0 - 24 - a_i * 6 + sw1 * 0.5
        c2x, c2y = ax0 + 48, ay0 - 36 - a_i * 8 + sw1
        ex, ey = ax0 + 74, ay0 - 28 - a_i * 10 + sw2
        a_pts = [cubic_bezier((ax0, ay0), (c1x, c1y), (c2x, c2y), (ex, ey), k / 10.0) for k in range(11)]
        draw_tapered_ribbon(ctx, a_pts, [lerp(2.2, 0.45, k / 10.0) for k in range(11)], close_tip=True)
        ctx.set_source_rgba(0.36, 0.30, 0.14, (0.90 if a_i == 0 else 0.65) * alpha)
        ctx.fill()

    ctx.restore()  # end head

    # 6. NEAR FOREGROUND LEGS (foreleg, midleg, and iconic hero jumping hindleg!)
    # Near foreleg
    f_hip = (28, 10)
    f_tgt = (56 + w1 * 13, 47 - max(0.0, w1) * 9)
    f_kn, f_ft = solve_ik_2bone(f_hip, f_tgt, 30, 30, bend_sign=-1.0)
    _draw_insect_leg_segment(ctx, f_hip, f_kn, 6.2, 4.4, (0.52, 0.64, 0.22), (0.34, 0.46, 0.16), alpha)
    _draw_insect_leg_segment(ctx, f_kn, f_ft, 4.2, 2.4, (0.56, 0.50, 0.20), (0.34, 0.32, 0.12), alpha)
    # Near midleg
    m_hip = (6, 9)
    m_tgt = (16 + w2 * 14, 49 - max(0.0, w2) * 9)
    m_kn, m_ft = solve_ik_2bone(m_hip, m_tgt, 36, 36, bend_sign=-1.0)
    _draw_insect_leg_segment(ctx, m_hip, m_kn, 6.8, 4.8, (0.52, 0.64, 0.22), (0.34, 0.46, 0.16), alpha)
    _draw_insect_leg_segment(ctx, m_kn, m_ft, 4.4, 2.5, (0.56, 0.50, 0.20), (0.34, 0.32, 0.12), alpha)
    # Near HERO hind jumping leg
    _draw_grasshopper_hind_leg(ctx, (-12, 3), hop_extend=hop_extend, walk_offset=w1 * 0.4, is_far=False, alpha=alpha)

    ctx.restore()


# =============================================================================
# 3. ARTICULATED REALISTIC GREEN POND FROG RIG
# =============================================================================

def _draw_webbed_foot(
    ctx: cairo.Context,
    ankle: Tuple[float, float],
    angle: float,
    scale: float,
    spread: float,
    shade: float,
    alpha: float,
) -> None:
    ctx.save()
    ctx.translate(ankle[0], ankle[1])
    ctx.rotate(angle)
    ctx.scale(scale, scale)

    toe_angles = [-0.46 * spread, -0.18 * spread, 0.10 * spread, 0.36 * spread]
    toe_lens = [24.0, 32.0, 36.0, 26.0]
    tips = [(math.cos(a) * l, math.sin(a) * l) for a, l in zip(toe_angles, toe_lens)]

    # Translucent olive-gold interdigital webbing
    ctx.new_path()
    ctx.move_to(0, 0)
    ctx.line_to(tips[0][0] * 0.78, tips[0][1] * 0.78)
    for i in range(len(tips) - 1):
        mx = (tips[i][0] + tips[i + 1][0]) * 0.36
        my = (tips[i][1] + tips[i + 1][1]) * 0.36
        ctx.curve_to(mx, my, mx, my, tips[i + 1][0] * 0.78, tips[i + 1][1] * 0.78)
    ctx.close_path()
    ctx.set_source_rgba(0.36 * shade, 0.48 * shade, 0.20 * shade, 0.58 * alpha)
    ctx.fill()

    # Individual slender amphibian toes with rounded toe pads
    for tx, ty in tips:
        _draw_insect_leg_segment(
            ctx, (0, 0), (tx, ty), 4.4, 2.2,
            (0.42 * shade, 0.56 * shade, 0.20 * shade),
            (0.24 * shade, 0.36 * shade, 0.13 * shade),
            alpha=alpha,
        )
        ctx.arc(tx, ty, 2.1, 0, math.tau)
        ctx.set_source_rgba(0.48 * shade, 0.58 * shade, 0.24 * shade, alpha)
        ctx.fill()
    ctx.restore()


def _draw_rounded_limb(
    ctx: cairo.Context,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    r0: float,
    r1: float,
    bulge: float,
    col_lit: Tuple[float, float, float],
    col_mid: Tuple[float, float, float],
    col_shd: Tuple[float, float, float],
    alpha: float = 1.0,
    stroke_alpha: float = 0.52,
) -> None:
    """Draw a smooth, rounded anatomical muscle capsule between two joints (zero boxy corners!)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    dist = max(1e-4, math.hypot(dx, dy))
    ang = math.atan2(dy, dx)

    ctx.save()
    ctx.translate(p0[0], p0[1])
    ctx.rotate(ang)

    rm = max(r0, r1) * (1.0 + bulge)
    ctx.new_path()
    # Rounded cap at p0 -> muscular top contour -> rounded cap at p1 -> lower contour
    ctx.arc(0.0, 0.0, r0, math.pi * 0.5, math.pi * 1.5)
    ctx.curve_to(dist * 0.35, -rm, dist * 0.72, -rm * 0.92, dist, -r1)
    ctx.arc(dist, 0.0, r1, -math.pi * 0.5, math.pi * 0.5)
    ctx.curve_to(dist * 0.72, rm * 0.78, dist * 0.35, rm * 0.85, 0.0, r0)
    ctx.close_path()

    pat = cairo.LinearGradient(dist * 0.5, -rm, dist * 0.5, rm)
    pat.add_color_stop_rgba(0.0, *col_lit, alpha)
    pat.add_color_stop_rgba(0.55, *col_mid, alpha)
    pat.add_color_stop_rgba(1.0, *col_shd, alpha)
    ctx.set_source(pat)
    ctx.fill_preserve()

    ctx.set_source_rgba(0.10, 0.18, 0.07, stroke_alpha * alpha)
    ctx.set_line_width(1.5)
    ctx.stroke()
    ctx.restore()


def _draw_frog_hind_leg(
    ctx: cairo.Context,
    hip: Tuple[float, float],
    crouch: float,
    leap_stretch: float,
    is_far: bool,
    alpha: float,
) -> None:
    """Smooth biomechanical Z-folded -> streamlined mid-leap amphibian hind leg."""
    shade = 0.72 if is_far else 1.0
    hx, hy = hip

    # Thigh points forward-up when folded, sweeps cleanly backward during leap
    thigh_ang = lerp(lerp(-0.34, -0.18, crouch), 2.74, leap_stretch)
    thigh_len = lerp(46.0, 58.0, leap_stretch)
    kx = hx + math.cos(thigh_ang) * thigh_len
    ky = hy + math.sin(thigh_ang) * thigh_len

    # Shank folds tucked cleanly underneath thigh when sitting, unfolds during leap
    shank_ang = lerp(lerp(2.74, 2.86, crouch), 2.88, leap_stretch)
    shank_len = lerp(52.0, 64.0, leap_stretch)
    ax = kx + math.cos(shank_ang) * shank_len
    ay = ky + math.sin(shank_ang) * shank_len

    # Elongated ankle/tarsus + webbed foot
    foot_ang = lerp(0.10, 2.94, leap_stretch)
    fx = ax + math.cos(foot_ang) * 24.0
    fy = ay + math.sin(foot_ang) * 24.0

    c_lit = (0.48 * shade, 0.68 * shade, 0.24 * shade)
    c_mid = (0.28 * shade, 0.50 * shade, 0.17 * shade)
    c_shd = (0.15 * shade, 0.29 * shade, 0.11 * shade)

    # 1. Webbed foot & rounded tarsus
    _draw_webbed_foot(ctx, (fx, fy), foot_ang, scale=0.92, spread=lerp(0.85, 1.25, leap_stretch), shade=shade, alpha=alpha)
    _draw_rounded_limb(ctx, (ax, ay), (fx, fy), 6.2, 4.0, 0.12, c_lit, c_mid, c_shd, alpha=alpha)

    # 2. Muscular calf (shank)
    _draw_rounded_limb(ctx, (kx, ky), (ax, ay), 11.5, 6.2, 0.22, c_lit, c_mid, c_shd, alpha=alpha)

    # 3. Powerful rounded muscular thigh with tiger crossbars & wet specular gleam
    _draw_rounded_limb(ctx, (hx, hy), (kx, ky), 16.5, 11.5, 0.24, c_lit, c_mid, c_shd, alpha=alpha)

    if not is_far:
        # Organic dark leopard-frog crossbars on muscular thigh
        for u in (0.30, 0.56, 0.78):
            bx = lerp(hx, kx, u)
            by = lerp(hy, ky, u)
            ctx.save()
            ctx.translate(bx, by - 3)
            ctx.rotate(thigh_ang + 1.35)
            ctx.scale(3.8, 10.5)
            ctx.arc(0, 0, 1.0, 0, math.tau)
            ctx.set_source_rgba(0.10, 0.20, 0.08, 0.40 * alpha)
            ctx.fill()
            ctx.restore()
        # Glossy wet specular highlight along upper thigh ridge
        ctx.move_to(hx + 4, hy - 11)
        ctx.curve_to((hx + kx) * 0.5, (hy + ky) * 0.5 - 16, kx - 8, ky - 10, kx - 2, ky - 5)
        ctx.set_source_rgba(0.92, 0.99, 0.84, 0.36 * alpha)
        ctx.set_line_width(3.8)
        ctx.stroke()


def draw_frog(
    ctx: cairo.Context,
    x: float,
    y: float,
    scale: float = 1.0,
    facing: float = 1.0,
    body_pitch: float = 0.0,
    t: float = 0.0,
    crouch: float = 0.0,
    leap_stretch: float = 0.0,
    mouth_open: float = 0.0,
    tongue_ext: float = 0.0,
    tongue_target_world: Optional[Tuple[float, float]] = None,
    blink: float = 0.0,
    throat_swell: float = 0.0,
    swallow_u: float = 0.0,
    draw_shadow: bool = True,
    alpha: float = 1.0,
) -> Tuple[float, float]:
    """
    Draw an anatomically realistic emerald green pond frog.
    Returns world (x, y) of the tongue tip when `tongue_ext > 0` (or mouth tip otherwise).
    """
    if alpha <= 0.01:
        return (x, y)

    ctx.save()
    ctx.translate(x, y)
    # Crouch lowers torso closer to the rock and tilts snout forward
    crouch_dy = 12.0 * crouch * (1.0 - leap_stretch)
    eff_pitch = body_pitch + lerp(0.14 * crouch, -0.18, leap_stretch)
    ctx.translate(0, crouch_dy * scale)
    ctx.rotate(eff_pitch)
    ctx.scale(scale * facing, scale)

    if draw_shadow and leap_stretch < 0.7:
        draw_soft_shadow(ctx, -6, 38 - crouch_dy * 0.6, 78, 18, alpha=0.36 * (1.0 - leap_stretch * 0.8) * alpha)

    breath = math.sin(t * 5.2) * 0.5 + 0.5
    throat_val = clamp(breath * 0.45 + throat_swell * 0.85 + math.sin(clamp(swallow_u) * math.pi) * 0.95)
    eye_retract = math.sin(clamp(swallow_u) * math.pi) * 7.5  # Authentic frog swallowing eye depression!

    # 1. FAR LIMBS
    _draw_frog_hind_leg(ctx, (-36, 12), crouch=crouch, leap_stretch=leap_stretch, is_far=True, alpha=alpha)
    # Far foreleg
    fa_hip = (22, 12)
    fa_ft = (42, lerp(40, 26, leap_stretch))
    fa_kn, _ = solve_ik_2bone(fa_hip, fa_ft, 22, 22, bend_sign=1.0)
    _draw_rounded_limb(ctx, fa_hip, fa_kn, 6.4, 4.4, 0.12, (0.28, 0.46, 0.16), (0.20, 0.34, 0.12), (0.12, 0.22, 0.08), alpha)
    _draw_rounded_limb(ctx, fa_kn, fa_ft, 4.4, 3.0, 0.08, (0.26, 0.44, 0.15), (0.18, 0.32, 0.11), (0.12, 0.20, 0.07), alpha)

    # 2. SUPPLE AMPHIBIAN BODY, THROAT SAC & SCULPTED HEAD
    stretch_x = 1.0 + 0.24 * leap_stretch
    stretch_y = 1.0 - 0.16 * leap_stretch - 0.08 * crouch
    ctx.save()
    ctx.scale(stretch_x, stretch_y)

    # Pulsing pale vocal/breathing throat pouch beneath chin
    ctx.new_path()
    ctx.move_to(68, 4)
    ctx.curve_to(56, 22 + throat_val * 13.0, 26, 26 + throat_val * 14.0, 8, 18)
    ctx.line_to(32, -2)
    ctx.close_path()
    th_pat = cairo.RadialGradient(42, 14 + throat_val * 5, 4, 38, 12, 34)
    th_pat.add_color_stop_rgba(0.0, 0.92, 0.93, 0.74, alpha)
    th_pat.add_color_stop_rgba(0.65, 0.74, 0.80, 0.56, alpha)
    th_pat.add_color_stop_rgba(1.0, 0.44, 0.56, 0.32, alpha)
    ctx.set_source(th_pat)
    ctx.fill()

    # Main streamlined dorsal torso & snout silhouette
    # If mouth is open, hinge lower jaw downward!
    jaw_drop = 22.0 * clamp(mouth_open)
    if mouth_open > 0.02:
        # Warm rosy inner mouth cavity
        ctx.new_path()
        ctx.move_to(26, 3)
        ctx.line_to(78, -2)
        ctx.line_to(68, 6 + jaw_drop)
        ctx.line_to(26, 12 + jaw_drop * 0.3)
        ctx.close_path()
        mpat = cairo.LinearGradient(30, 0, 74, 12)
        mpat.add_color_stop_rgba(0.0, 0.45, 0.16, 0.18, alpha)
        mpat.add_color_stop_rgba(1.0, 0.78, 0.40, 0.44, alpha)
        ctx.set_source(mpat)
        ctx.fill()

        # Lower jaw rim
        ctx.new_path()
        ctx.move_to(24, 10)
        ctx.curve_to(44, 15 + jaw_drop * 0.7, 62, 13 + jaw_drop, 72, 6 + jaw_drop)
        ctx.curve_to(58, 22 + jaw_drop * 0.9, 36, 24 + jaw_drop * 0.5, 16, 18)
        ctx.close_path()
        ctx.set_source_rgba(0.76, 0.82, 0.56, alpha)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.14, 0.24, 0.10, 0.6 * alpha)
        ctx.set_line_width(1.4)
        ctx.stroke()

    ctx.new_path()
    # Start at snout tip -> brow -> subtle pelvic urostyle hump -> rump -> creamy belly -> throat -> jaw
    ctx.move_to(82, -2)
    ctx.curve_to(72, -16, 56, -24, 38, -24)    # sloping pointed snout to eye ridge
    ctx.curve_to(16, -24, -10, -34, -34, -26)  # muscular arched back & pelvic hump
    ctx.curve_to(-58, -18, -68, 2, -56, 18)    # rounded rump
    ctx.curve_to(-40, 30, -8, 29, 24, 19)      # supple belly
    ctx.curve_to(46, 14, 68, 8, 82, -2)        # mouth line back to snout
    ctx.close_path()

    bpat = cairo.LinearGradient(12, -34, -6, 30)
    bpat.add_color_stop_rgba(0.0, 0.46, 0.70, 0.22, alpha)  # sunlit emerald-chartreuse back
    bpat.add_color_stop_rgba(0.42, 0.26, 0.52, 0.16, alpha) # rich pond-frog emerald
    bpat.add_color_stop_rgba(0.72, 0.34, 0.52, 0.20, alpha) # olive flank
    bpat.add_color_stop_rgba(0.90, 0.84, 0.86, 0.64, alpha) # pale creamy underside
    bpat.add_color_stop_rgba(1.0, 0.66, 0.70, 0.48, alpha)
    ctx.set_source(bpat)
    ctx.fill_preserve()

    # Interior organic details clipped inside the frog's skin
    ctx.save()
    ctx.clip_preserve()

    # Organic dark-olive/charcoal leopard spots across back & flanks
    spots = [
        (-42, -12, 9, 6, 0.2), (-24, -20, 11, 7, -0.1), (-4, -17, 10, 6, 0.15),
        (14, -14, 8, 5, -0.2), (-32, -2, 8, 5, 0.3), (-12, -5, 9, 6, 0.0),
        (6, -3, 7, 4.5, -0.15), (-48, 2, 7, 5, 0.4),
    ]
    for sx, sy, srx, sry, sang in spots:
        ctx.save()
        ctx.translate(sx, sy)
        ctx.rotate(sang)
        ctx.scale(srx, sry)
        ctx.arc(0, 0, 1.0, 0, math.tau)
        # Subtle pale ring around dark amphibian spot
        ctx.set_source_rgba(0.56, 0.72, 0.28, 0.32 * alpha)
        ctx.fill()
        ctx.scale(0.76, 0.76)
        ctx.arc(0, 0, 1.0, 0, math.tau)
        ctx.set_source_rgba(0.11, 0.21, 0.08, 0.54 * alpha)
        ctx.fill()
        ctx.restore()

    # Signature golden-bronze dorsolateral skin ridge running from behind eye down to rump
    ctx.new_path()
    ctx.move_to(38, -16)
    ctx.curve_to(12, -17, -18, -22, -52, -8)
    ctx.set_source_rgba(0.78, 0.72, 0.32, 0.62 * alpha)
    ctx.set_line_width(3.4)
    ctx.stroke()

    # Glossy wet sunlit specular ribbon along upper back & snout
    ctx.new_path()
    ctx.move_to(72, -8)
    ctx.curve_to(48, -20, 10, -28, -32, -22)
    ctx.set_source_rgba(0.92, 0.99, 0.84, 0.42 * alpha)
    ctx.set_line_width(5.2)
    ctx.stroke()

    # Circular bronze tympanum (ear drum) behind eye!
    ctx.arc(31, -8, 7.6, 0, math.tau)
    tymp = cairo.RadialGradient(33, -10, 1.2, 31, -8, 7.8)
    tymp.add_color_stop_rgba(0.0, 0.66, 0.52, 0.26, 0.88 * alpha)
    tymp.add_color_stop_rgba(0.75, 0.38, 0.30, 0.14, 0.88 * alpha)
    tymp.add_color_stop_rgba(1.0, 0.16, 0.20, 0.09, 0.90 * alpha)
    ctx.set_source(tymp)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.12, 0.18, 0.07, 0.65 * alpha)
    ctx.set_line_width(1.2)
    ctx.stroke()

    # Nostril pore & crisp mouth commissure line
    ctx.arc(71, -8.5, 1.8, 0, math.tau)
    ctx.set_source_rgba(0.10, 0.16, 0.06, 0.78 * alpha)
    ctx.fill()
    if mouth_open <= 0.02:
        ctx.move_to(81, -2)
        ctx.curve_to(62, 5, 42, 7, 25, 3)
        ctx.set_source_rgba(0.12, 0.20, 0.08, 0.72 * alpha)
        ctx.set_line_width(1.8)
        ctx.stroke()

    ctx.restore()  # end clip

    # Soft painterly torso outline
    ctx.set_source_rgba(0.10, 0.19, 0.07, 0.62 * alpha)
    ctx.set_line_width(1.8)
    ctx.stroke()

    # 3. HIGH-SET PROTRUDING METALLIC COPPER-GOLD EYE + NICTITATING BLINK
    ex, ey = 47.0, -20.0 + eye_retract
    er = 11.8
    # Green orbital eyelid dome behind eye
    ctx.arc(ex - 1.0, ey + 1.5, er + 3.2, math.pi * 0.92, math.tau + 0.15)
    ctx.set_source_rgba(0.28, 0.52, 0.16, alpha)
    ctx.fill()

    # Copper-gold metallic iris
    ctx.save()
    ctx.arc(ex, ey, er, 0, math.tau)
    ctx.clip_preserve()
    ipat = cairo.RadialGradient(ex + 2.5, ey - 3.5, 1.2, ex, ey, er * 1.05)
    ipat.add_color_stop_rgba(0.0, 0.98, 0.84, 0.36, alpha)
    ipat.add_color_stop_rgba(0.50, 0.82, 0.56, 0.16, alpha)
    ipat.add_color_stop_rgba(0.88, 0.44, 0.26, 0.08, alpha)
    ipat.add_color_stop_rgba(1.0, 0.18, 0.12, 0.05, alpha)
    ctx.set_source(ipat)
    ctx.fill()

    # Sleek dark horizontal oval amphibian pupil
    ctx.save()
    ctx.translate(ex + 1.2, ey + 0.4)
    ctx.scale(6.8, 4.4)
    ctx.arc(0, 0, 1.0, 0, math.tau)
    ctx.set_source_rgba(0.04, 0.06, 0.05, alpha)
    ctx.fill()
    ctx.restore()

    # Moist curved sky-dome specular highlight
    ctx.arc(ex + 3.8, ey - 4.2, 3.0, 0, math.tau)
    ctx.set_source_rgba(1.0, 0.99, 0.92, 0.92 * alpha)
    ctx.fill()
    ctx.arc(ex - 4.0, ey + 4.5, 1.6, 0, math.tau)
    ctx.set_source_rgba(0.92, 0.96, 0.80, 0.45 * alpha)
    ctx.fill()

    # Translucent blinking lower eyelid / nictitating membrane rising smoothly over eye
    if blink > 0.01:
        lid_y = lerp(ey + er + 2.0, ey - er - 2.0, clamp(blink))
        ctx.rectangle(ex - er - 2, lid_y, er * 2 + 4, er * 2 + 6)
        ctx.set_source_rgba(0.36, 0.56, 0.22, 0.92 * alpha)
        ctx.fill()
        ctx.move_to(ex - er, lid_y)
        ctx.line_to(ex + er, lid_y)
        ctx.set_source_rgba(0.82, 0.82, 0.46, 0.72 * alpha)
        ctx.set_line_width(1.6)
        ctx.stroke()

    ctx.restore()
    ctx.arc(ex, ey, er, 0, math.tau)
    ctx.set_source_rgba(0.12, 0.18, 0.07, 0.78 * alpha)
    ctx.set_line_width(1.6)
    ctx.stroke()

    ctx.restore()  # end body stretch

    # 4. NEAR FORELEG (smooth rounded amphibian arm with 4 splayed toes!)
    f_hip = (18, 12)
    f_tgt = (44, lerp(42, 24, leap_stretch))
    f_kn, f_ft = solve_ik_2bone(f_hip, f_tgt, 23, 23, bend_sign=1.0)
    _draw_rounded_limb(ctx, f_hip, f_kn, 7.8, 5.4, 0.14, (0.44, 0.64, 0.22), (0.28, 0.48, 0.16), (0.16, 0.30, 0.11), alpha)
    _draw_rounded_limb(ctx, f_kn, f_ft, 5.4, 3.8, 0.10, (0.42, 0.60, 0.21), (0.26, 0.44, 0.15), (0.16, 0.28, 0.10), alpha)
    for ang_t in (-0.35, 0.0, 0.35, 0.72):
        tx_t = f_ft[0] + math.cos(ang_t) * 13.5
        ty_t = f_ft[1] + math.sin(ang_t) * 7.5 + 2.0
        _draw_rounded_limb(ctx, f_ft, (tx_t, ty_t), 2.1, 1.3, 0.05, (0.46, 0.62, 0.22), (0.30, 0.46, 0.16), (0.18, 0.30, 0.10), alpha, stroke_alpha=0.35)

    # 5. NEAR POWERFUL HIND LEG
    _draw_frog_hind_leg(ctx, (-30, 15), crouch=crouch, leap_stretch=leap_stretch, is_far=False, alpha=alpha)

    # 6. BALLISTIC FLESHY TONGUE EXTENSION (hinged cleanly inside open lower jaw!)
    mouth_local = (58.0 * stretch_x, 9.0 * stretch_y)
    c_p, s_p = math.cos(eff_pitch), math.sin(eff_pitch)
    mx_w = x + (mouth_local[0] * facing * c_p - mouth_local[1] * s_p) * scale
    my_w = y + crouch_dy * scale + (mouth_local[0] * facing * s_p + mouth_local[1] * c_p) * scale
    tip_world = (mx_w, my_w)

    if tongue_ext > 0.01:
        if tongue_target_world is not None:
            dw_x = (tongue_target_world[0] - x) / max(scale, 1e-4)
            dw_y = (tongue_target_world[1] - (y + crouch_dy * scale)) / max(scale, 1e-4)
            loc_tx = (dw_x * c_p + dw_y * s_p) / facing
            loc_ty = (-dw_x * s_p + dw_y * c_p)
        else:
            loc_tx = mouth_local[0] + 220.0
            loc_ty = mouth_local[1] - 18.0

        cur_x = lerp(mouth_local[0], loc_tx, clamp(tongue_ext))
        cur_y = lerp(mouth_local[1], loc_ty, clamp(tongue_ext))
        tip_world = (
            x + (cur_x * facing * c_p - cur_y * s_p) * scale,
            y + crouch_dy * scale + (cur_x * facing * s_p + cur_y * c_p) * scale,
        )

        mid_x = (mouth_local[0] + cur_x) * 0.5
        mid_y = (mouth_local[1] + cur_y) * 0.5 - 12.0 * math.sin(clamp(tongue_ext) * math.pi)
        t_pts = [
            cubic_bezier(mouth_local, (mid_x, mid_y), (mid_x, mid_y), (cur_x, cur_y), k / 12.0)
            for k in range(13)
        ]
        t_hw = [lerp(4.8, 3.6, k / 12.0) for k in range(13)]

        # Subtle motion-blur echo along rapid strike
        if 0.15 < tongue_ext < 0.92:
            draw_tapered_ribbon(ctx, t_pts, [w * 1.45 for w in t_hw], close_tip=False)
            ctx.set_source_rgba(0.94, 0.54, 0.58, 0.20 * alpha)
            ctx.fill()

        draw_tapered_ribbon(ctx, t_pts, t_hw, close_tip=False)
        tng_pat = cairo.LinearGradient(mouth_local[0], mouth_local[1], cur_x, cur_y)
        tng_pat.add_color_stop_rgba(0.0, 0.84, 0.38, 0.44, alpha)
        tng_pat.add_color_stop_rgba(0.7, 0.94, 0.52, 0.58, alpha)
        tng_pat.add_color_stop_rgba(1.0, 0.97, 0.60, 0.65, alpha)
        ctx.set_source(tng_pat)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.56, 0.20, 0.25, 0.45 * alpha)
        ctx.set_line_width(1.1)
        ctx.stroke()

        # Smooth rounded fleshy adhesive tongue pad at tip
        ang_tip = math.atan2(cur_y - t_pts[-2][1], cur_x - t_pts[-2][0])
        ctx.save()
        ctx.translate(cur_x, cur_y)
        ctx.rotate(ang_tip)
        ctx.scale(8.5, 6.2)
        ctx.arc(0, 0, 1.0, 0, math.tau)
        pad_p = cairo.RadialGradient(-0.2, -0.3, 0.1, 0, 0, 1.0)
        pad_p.add_color_stop_rgba(0.0, 1.0, 0.72, 0.76, alpha)
        pad_p.add_color_stop_rgba(0.7, 0.93, 0.50, 0.56, alpha)
        pad_p.add_color_stop_rgba(1.0, 0.76, 0.32, 0.38, alpha)
        ctx.set_source(pad_p)
        ctx.fill()
        ctx.restore()

        # Wet specular glint on tongue pad
        ctx.arc(cur_x - 1.5, cur_y - 2.0, 2.4, 0, math.tau)
        ctx.set_source_rgba(1.0, 0.95, 0.96, 0.80 * alpha)
        ctx.fill()

    ctx.restore()
    return tip_world


def draw_frog_eye_closeup(
    ctx: cairo.Context,
    cx: float,
    cy: float,
    radius: float,
    blink: float = 0.0,
    focus_u: float = 0.5,
    t: float = 0.0,
) -> None:
    """Hyper-detailed wildlife documentary macro shot of the frog's gleaming eye locking on."""
    ctx.save()
    # Surrounding textured emerald-olive moist amphibian brow & skin
    skin_pat = cairo.RadialGradient(cx + radius * 0.25, cy - radius * 0.3, radius * 0.3, cx, cy, radius * 3.8)
    skin_pat.add_color_stop_rgb(0.0, 0.36, 0.58, 0.19)
    skin_pat.add_color_stop_rgb(0.45, 0.22, 0.42, 0.14)
    skin_pat.add_color_stop_rgb(1.0, 0.10, 0.20, 0.08)
    ctx.set_source(skin_pat)
    ctx.rectangle(cx - radius * 4.5, cy - radius * 3.0, radius * 9.0, radius * 6.0)
    ctx.fill()

    # Subtle amphibian skin pores & dappled moss spots around orbital ridge
    for i in range(42):
        ang = hash1d(i * 37 + 11) * math.tau
        dist = radius * (1.28 + 1.8 * hash1d(i * 43 + 17))
        px = cx + math.cos(ang) * dist
        py = cy + math.sin(ang) * dist
        pr = radius * (0.04 + 0.09 * hash1d(i * 53 + 7))
        ctx.arc(px, py, pr, 0, math.tau)
        if i % 3 == 0:
            ctx.set_source_rgba(0.11, 0.21, 0.08, 0.34)
        else:
            ctx.set_source_rgba(0.66, 0.74, 0.28, 0.20)
        ctx.fill()

    # Sculpted fleshy orbital eyelid fold surrounding the eye
    ctx.arc(cx, cy, radius * 1.15, 0, math.tau)
    fold = cairo.RadialGradient(cx + radius * 0.2, cy - radius * 0.3, radius * 0.8, cx, cy, radius * 1.2)
    fold.add_color_stop_rgb(0.0, 0.48, 0.68, 0.24)
    fold.add_color_stop_rgb(0.75, 0.22, 0.38, 0.13)
    fold.add_color_stop_rgb(1.0, 0.10, 0.18, 0.07)
    ctx.set_source(fold)
    ctx.fill()

    # Clip to circular corneal dome
    ctx.save()
    ctx.arc(cx, cy, radius, 0, math.tau)
    ctx.clip_preserve()

    # Rich metallic gold-copper iris with intricate radial filaments
    ipat = cairo.RadialGradient(cx + radius * 0.14, cy - radius * 0.18, radius * 0.08, cx, cy, radius * 1.05)
    ipat.add_color_stop_rgb(0.0, 1.00, 0.88, 0.38)
    ipat.add_color_stop_rgb(0.42, 0.90, 0.66, 0.18)
    ipat.add_color_stop_rgb(0.76, 0.56, 0.32, 0.08)
    ipat.add_color_stop_rgb(1.0, 0.18, 0.10, 0.04)
    ctx.set_source(ipat)
    ctx.fill()

    # Fine radial iris fibers & metallic flecks
    for i in range(64):
        a = (i / 64.0) * math.tau
        r0 = radius * 0.38
        r1 = radius * (0.86 + 0.12 * hash1d(i * 19 + 5))
        ctx.move_to(cx + math.cos(a) * r0, cy + math.sin(a) * r0)
        ctx.line_to(cx + math.cos(a) * r1, cy + math.sin(a) * r1)
        if i % 2 == 0:
            ctx.set_source_rgba(1.0, 0.92, 0.52, 0.26)
        else:
            ctx.set_source_rgba(0.24, 0.14, 0.04, 0.34)
        ctx.set_line_width(radius * 0.025)
        ctx.stroke()

    # Horizontal oval pupil tightening slightly as focus_u -> 1.0 (predator target lock!)
    saccade = math.sin(t * 2.1) * radius * 0.015
    pw = radius * lerp(0.58, 0.50, clamp(focus_u))
    ph = radius * lerp(0.39, 0.31, clamp(focus_u))
    ctx.save()
    ctx.translate(cx + saccade, cy)
    ctx.scale(pw, ph)
    ctx.arc(0, 0, 1.0, 0, math.tau)
    ctx.set_source_rgb(0.03, 0.05, 0.04)
    ctx.fill()
    ctx.restore()

    # Delicate sky & swaying reed reflection gleaming on the wet cornea!
    ctx.save()
    ctx.translate(cx + radius * 0.28, cy - radius * 0.32)
    ctx.rotate(-0.35)
    ctx.scale(radius * 0.34, radius * 0.22)
    ctx.arc(0, 0, 1.0, 0, math.tau)
    r_pat = cairo.RadialGradient(0, 0, 0.1, 0, 0, 1.0)
    r_pat.add_color_stop_rgba(0.0, 1.0, 1.0, 0.96, 0.92)
    r_pat.add_color_stop_rgba(0.65, 0.84, 0.94, 0.98, 0.45)
    r_pat.add_color_stop_rgba(1.0, 0.84, 0.94, 0.98, 0.0)
    ctx.set_source(r_pat)
    ctx.fill()
    ctx.restore()

    # Lower green meadow caustic bounce inside bottom cornea
    ctx.arc(cx - radius * 0.26, cy + radius * 0.42, radius * 0.22, 0, math.tau)
    ctx.set_source_rgba(0.72, 0.90, 0.46, 0.28)
    ctx.fill()

    # Translucent golden-veined nictitating eyelid during blink
    if blink > 0.01:
        lid_y = lerp(cy + radius + 5, cy - radius - 5, clamp(blink))
        ctx.rectangle(cx - radius - 4, lid_y, radius * 2 + 8, radius * 2 + 12)
        ctx.set_source_rgba(0.36, 0.56, 0.22, 0.94)
        ctx.fill()
        ctx.move_to(cx - radius, lid_y)
        ctx.line_to(cx + radius, lid_y)
        ctx.set_source_rgba(0.86, 0.82, 0.42, 0.80)
        ctx.set_line_width(radius * 0.05)
        ctx.stroke()

    ctx.restore()
    ctx.arc(cx, cy, radius, 0, math.tau)
    ctx.set_source_rgba(0.08, 0.14, 0.06, 0.82)
    ctx.set_line_width(radius * 0.045)
    ctx.stroke()
    ctx.restore()


# =============================================================================
# 4. CONTINUOUS SCALED SNAKE RIG (LATERAL UNDULATION + COIL + STRIKE)
# =============================================================================

def compute_snake_spine(
    head_xy: Tuple[float, float],
    heading_rad: float = 0.0,
    total_len: float = 520.0,
    n_segs: int = 58,
    slither_phase: float = 0.0,
    wave_amp: float = 34.0,
    coil_u: float = 0.0,
    strike_u: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Generate a continuous 58-point biomechanical snake spine from head (index 0)
    to tail tip (index n_segs-1):
      - `slither_phase` drives traveling lateral sinusoidal undulation pushing along terrain
      - `coil_u` gathers the neck & midbody into a tight poised S-coil before striking
      - `strike_u` shoots the anterior third straight forward while the rear anchors!
    """
    hx, hy = head_xy
    ch, sh = math.cos(heading_rad), math.sin(heading_rad)
    # Backward unit vector (from head toward tail) & perpendicular lateral axis
    bx, by = -ch, -sh
    nx, ny = -sh, ch

    pts: List[Tuple[float, float]] = []
    # Compact body longitudinally when coiling, elongate forward when striking
    len_scale = lerp(1.0, 0.66, clamp(coil_u) * (1.0 - clamp(strike_u)))
    eff_len = total_len * len_scale

    for i in range(n_segs):
        s = i / float(n_segs - 1)  # 0 = head, 1 = tail tip
        # Traveling wave envelope: stabilized near head (s<0.08), full undulation midbody, subtle tail whip
        head_stab = clamp(s / 0.10) ** 1.3
        # Smooth lateral undulation
        slither_lat = math.sin(s * 12.5 - slither_phase) * wave_amp * head_stab * (1.0 - 0.35 * s)
        # Tense S-coil shape (tight S behind head + anchored loop posteriorly)
        coil_lat = (
            math.sin(s * 11.2 + 0.5) * 62.0 * math.sin(clamp(s * 1.15) * math.pi)
            + math.cos(s * 6.4) * 24.0 * clamp(s * 1.8)
        )
        # Straighten anterior half dramatically during rapid strike!
        strike_straight = clamp(1.0 - s * 1.65) * clamp(strike_u)
        lat = lerp(slither_lat, coil_lat, clamp(coil_u)) * (1.0 - 0.85 * strike_straight)

        # Forward/backward longitudinal offset along heading
        long_d = s * eff_len
        if coil_u > 0.01:
            long_d += math.sin(s * 9.5) * 16.0 * clamp(coil_u) * (1.0 - strike_straight)

        px = hx + bx * long_d + nx * lat
        py = hy + by * long_d + ny * lat
        pts.append((px, py))

    return pts


def draw_snake(
    ctx: cairo.Context,
    spine_pts: Sequence[Tuple[float, float]],
    max_radius: float = 20.0,
    head_angle: Optional[float] = None,
    mouth_open: float = 0.0,
    tongue_flick: float = 0.0,
    swallow_bulge_s: Optional[float] = None,
    swallow_bulge_amp: float = 0.0,
    draw_shadow: bool = True,
    alpha: float = 1.0,
    t: float = 0.0,
) -> None:
    """
    Render a richly scaled olive-amber-umber meadow snake along `spine_pts` (head=0 -> tail=-1):
      - Grounded soft shadow beneath every undulating curve
      - Realistic anatomical taper (narrow neck -> muscular midbody -> slender whip tail)
      - Hundreds of overlapping oriented diamond/hexagonal dorsal scales & saddle markings
      - Cylindrical specular ridge gleam + pale creamy ventral scales
      - Sculpted scaly colubrid head with glossy golden-amber eye & flicking forked crimson tongue
    """
    if len(spine_pts) < 4 or alpha <= 0.01:
        return

    arr, tangents, normals = polyline_frames(spine_pts)
    n = len(arr)

    # Anatomical radius profile along spine (0=head/neck -> 1=tail tip)
    radii = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = i / float(n - 1)
        if s < 0.08:
            # Slightly narrower neck behind head
            r = max_radius * lerp(0.58, 0.78, s / 0.08)
        elif s < 0.45:
            r = max_radius * lerp(0.78, 1.0, (s - 0.08) / 0.37)
        else:
            r = max_radius * max(0.08, (1.0 - (s - 0.45) / 0.55) ** 0.78)
        if swallow_bulge_s is not None and swallow_bulge_amp > 0.01:
            bulge = math.exp(-((s - swallow_bulge_s) ** 2) / 0.0045) * max_radius * 0.52 * swallow_bulge_amp
            r += bulge
        radii[i] = r

    # 1. CONTINUOUS GROUND CONTACT SHADOW BENEATH EVERY COIL
    if draw_shadow:
        shad_pts = [(float(p[0]) + 4.0, float(p[1]) + max_radius * 0.52) for p in arr]
        draw_tapered_ribbon(ctx, shad_pts, radii * 1.12, close_tip=True)
        ctx.set_source_rgba(0.05, 0.07, 0.04, 0.34 * alpha)
        ctx.fill()

    # 2. BASE CYLINDRICAL BODY FILL (pale cream underside -> olive-ochre sides -> deep moss back)
    draw_tapered_ribbon(ctx, spine_pts, radii, close_tip=True)
    ctx.set_source_rgba(0.32, 0.36, 0.19, alpha)
    ctx.fill_preserve()

    # Clip inner painterly scales & dorsal cylindrical lighting cleanly inside the body!
    ctx.save()
    ctx.clip_preserve()

    # Pale warm cream ventral belly edge along lower side of spine
    belly_pts = [(float(arr[i, 0] + normals[i, 0] * radii[i] * 0.55), float(arr[i, 1] + normals[i, 1] * radii[i] * 0.55)) for i in range(n)]
    draw_tapered_ribbon(ctx, belly_pts, radii * 0.52, close_tip=True)
    ctx.set_source_rgba(0.84, 0.79, 0.56, 0.55 * alpha)
    ctx.fill()

    # Paint overlapping diamond scales from TAIL to HEAD so scales imbricate naturally!
    for i in range(n - 2, 0, -1):
        s = i / float(n - 1)
        px, py = float(arr[i, 0]), float(arr[i, 1])
        # Tangent pointing from tail -> head is -tangents[i]
        ang = math.atan2(-float(tangents[i, 1]), -float(tangents[i, 0]))
        r = float(radii[i])
        if r < 2.2:
            continue

        # Alternating natural diamond/saddle camouflage pattern along length
        saddle = 0.5 + 0.5 * math.sin(s * 42.0)
        is_dark_saddle = saddle > 0.54

        ctx.save()
        ctx.translate(px, py)
        ctx.rotate(ang)

        # Draw 3 staggered scale rows across cylindrical girth (dorsal, upper flank, lower flank)
        offsets = [(-0.48, 0.92), (0.0, 1.05), (0.48, 0.88)]
        for row_idx, (lat_f, sc_f) in enumerate(offsets):
            sy_off = lat_f * r
            sx_off = (3.5 if (i + row_idx) % 2 == 0 else -1.5) * (r / max_radius)
            sw = r * 0.62 * sc_f
            sh = r * 0.46 * sc_f

            ctx.new_path()
            ctx.move_to(sx_off + sw, sy_off)
            ctx.line_to(sx_off, sy_off - sh)
            ctx.line_to(sx_off - sw * 0.9, sy_off)
            ctx.line_to(sx_off, sy_off + sh)
            ctx.close_path()

            if row_idx == 1 and is_dark_saddle:
                col = (0.24, 0.21, 0.11)  # deep umber-chocolate dorsal saddle
            elif row_idx == 0:
                col = (0.46, 0.50, 0.24) if not is_dark_saddle else (0.34, 0.30, 0.15)  # sunlit upper olive-ochre
            elif row_idx == 1:
                col = (0.58, 0.54, 0.26)  # warm golden-olive diamond scale
            else:
                col = (0.36, 0.36, 0.19)  # shaded lower flank scale

            ctx.set_source_rgba(col[0], col[1], col[2], 0.90 * alpha)
            ctx.fill_preserve()
            ctx.set_source_rgba(0.14, 0.13, 0.07, 0.45 * alpha)
            ctx.set_line_width(max(0.7, r * 0.055))
            ctx.stroke()

        ctx.restore()

    # Glossy cylindrical dorsal sunlit highlight ribbon running smoothly along upper ridge
    hi_pts = [(float(arr[i, 0] - normals[i, 0] * radii[i] * 0.34), float(arr[i, 1] - normals[i, 1] * radii[i] * 0.34)) for i in range(n - 2)]
    draw_tapered_ribbon(ctx, hi_pts, radii[:-2] * 0.24, close_tip=True)
    ctx.set_source_rgba(0.95, 0.94, 0.72, 0.28 * alpha)
    ctx.fill()

    ctx.restore()  # end body clip

    # Crisp dark-umber painterly body rim outline
    draw_tapered_ribbon(ctx, spine_pts, radii, close_tip=True)
    ctx.set_source_rgba(0.13, 0.12, 0.06, 0.65 * alpha)
    ctx.set_line_width(1.6)
    ctx.stroke()

    # 3. SCULPTED INDEPENDENT COLUBRID HEAD & FLICKING FORKED TONGUE
    hx, hy = float(arr[0, 0]), float(arr[0, 1])
    if head_angle is None:
        head_angle = math.atan2(float(arr[0, 1] - arr[2, 1]), float(arr[0, 0] - arr[2, 0]))
    hs = max_radius / 20.0

    ctx.save()
    ctx.translate(hx, hy)
    ctx.rotate(head_angle)
    ctx.scale(hs, hs)

    # Flicking forked crimson tongue (extends & oscillates rapidly!)
    if tongue_flick > 0.02:
        t_len = 46.0 * clamp(tongue_flick)
        wig = math.sin(t * 58.0) * 7.5 * clamp(tongue_flick)
        ctx.new_path()
        ctx.move_to(28, 0)
        ctx.curve_to(28 + t_len * 0.4, wig * 0.3, 28 + t_len * 0.75, wig, 28 + t_len * 0.82, wig)
        ctx.set_source_rgba(0.76, 0.14, 0.18, 0.92 * alpha)
        ctx.set_line_width(2.0)
        ctx.stroke()
        # Forked tips
        fx0, fy0 = 28 + t_len * 0.80, wig
        ctx.move_to(fx0, fy0)
        ctx.line_to(fx0 + 11.0 * clamp(tongue_flick), fy0 - 6.5)
        ctx.move_to(fx0, fy0)
        ctx.line_to(fx0 + 11.0 * clamp(tongue_flick), fy0 + 6.5)
        ctx.set_line_width(1.4)
        ctx.stroke()

    # Open jaws during strike
    if mouth_open > 0.05:
        mo = clamp(mouth_open)
        ctx.new_path()
        ctx.move_to(-2, 0)
        ctx.line_to(32, -16 * mo)
        ctx.line_to(28, 18 * mo)
        ctx.close_path()
        ctx.set_source_rgba(0.74, 0.34, 0.38, 0.92 * alpha)
        ctx.fill()
        # Lower jaw
        ctx.new_path()
        ctx.move_to(-6, 4)
        ctx.curve_to(8, 12 + 12 * mo, 22, 16 + 14 * mo, 30, 12 + 10 * mo)
        ctx.line_to(26, 4 + 6 * mo)
        ctx.close_path()
        ctx.set_source_rgba(0.76, 0.72, 0.48, alpha)
        ctx.fill()

    # Sleek diamond/almond-shaped scaly snake head
    ctx.new_path()
    ctx.move_to(-16, -9)
    ctx.curve_to(-6, -18, 10, -16, 24, -8)
    ctx.curve_to(34, -4, 34, 4, 24, 8)
    ctx.curve_to(10, 16, -6, 17, -16, 9)
    ctx.close_path()

    hpat = cairo.LinearGradient(4, -18, 4, 18)
    hpat.add_color_stop_rgba(0.0, 0.48, 0.50, 0.24, alpha)
    hpat.add_color_stop_rgba(0.45, 0.32, 0.35, 0.16, alpha)
    hpat.add_color_stop_rgba(0.82, 0.64, 0.60, 0.34, alpha)
    hpat.add_color_stop_rgba(1.0, 0.80, 0.75, 0.50, alpha)
    ctx.set_source(hpat)
    ctx.fill_preserve()

    # Crown head-plate pattern & dark postocular eye mask stripe
    ctx.save()
    ctx.clip_preserve()
    ctx.move_to(12, -5)
    ctx.line_to(-14, -10)
    ctx.set_source_rgba(0.14, 0.12, 0.06, 0.55 * alpha)
    ctx.set_line_width(5.2)
    ctx.stroke()
    # Sculpted crown scale suture hint
    ctx.new_path()
    ctx.move_to(-8, -5)
    ctx.line_to(6, -7)
    ctx.line_to(14, -2)
    ctx.line_to(6, 2)
    ctx.line_to(-8, 1)
    ctx.close_path()
    ctx.set_source_rgba(0.24, 0.23, 0.11, 0.48 * alpha)
    ctx.fill()
    ctx.restore()

    ctx.set_source_rgba(0.12, 0.11, 0.06, 0.78 * alpha)
    ctx.set_line_width(1.6)
    ctx.stroke()

    # Glossy golden-amber reptilian eye under overhanging supraocular brow scale
    ctx.arc(13, -5.5, 4.4, 0, math.tau)
    epat = cairo.RadialGradient(14, -6.5, 0.6, 13, -5.5, 4.5)
    epat.add_color_stop_rgba(0.0, 1.00, 0.82, 0.26, alpha)
    epat.add_color_stop_rgba(0.75, 0.76, 0.46, 0.10, alpha)
    epat.add_color_stop_rgba(1.0, 0.22, 0.14, 0.04, alpha)
    ctx.set_source(epat)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.10, 0.08, 0.04, 0.85 * alpha)
    ctx.set_line_width(1.0)
    ctx.stroke()

    # Crisp dark pupil & bright specular dot
    ctx.arc(13.4, -5.5, 2.0, 0, math.tau)
    ctx.set_source_rgba(0.04, 0.04, 0.03, alpha)
    ctx.fill()
    ctx.arc(14.6, -6.8, 1.0, 0, math.tau)
    ctx.set_source_rgba(1.0, 0.99, 0.92, 0.95 * alpha)
    ctx.fill()

    # Brow scale ridge & nostril
    ctx.move_to(7, -9.8)
    ctx.curve_to(12, -11.0, 17, -10.2, 20, -7.8)
    ctx.set_source_rgba(0.56, 0.56, 0.28, 0.85 * alpha)
    ctx.set_line_width(2.0)
    ctx.stroke()
    ctx.arc(25, -4.2, 1.2, 0, math.tau)
    ctx.set_source_rgba(0.10, 0.09, 0.05, 0.78 * alpha)
    ctx.fill()

    ctx.restore()


# =============================================================================
# 5. ARTICULATED REALISTIC RED-TAILED HAWK RIG
# =============================================================================

def _draw_single_feather(
    ctx: cairo.Context,
    base: Tuple[float, float],
    angle: float,
    length: float,
    width: float,
    col_base: Tuple[float, float, float],
    col_tip: Tuple[float, float, float],
    barred: bool = True,
    alpha: float = 1.0,
) -> None:
    """Paint an individually barbed, tapered raptor feather with central rachis shaft."""
    ctx.save()
    ctx.translate(base[0], base[1])
    ctx.rotate(angle)

    ctx.new_path()
    ctx.move_to(0, 0)
    ctx.curve_to(length * 0.25, -width * 0.58, length * 0.75, -width * 0.50, length, 0)
    ctx.curve_to(length * 0.75, width * 0.52, length * 0.25, width * 0.56, 0, 0)
    ctx.close_path()

    fpat = cairo.LinearGradient(0, 0, length, 0)
    fpat.add_color_stop_rgba(0.0, *col_base, alpha)
    fpat.add_color_stop_rgba(0.70, *lerp_rgb(col_base, col_tip, 0.65), alpha)
    fpat.add_color_stop_rgba(1.0, *col_tip, alpha)
    ctx.set_source(fpat)
    ctx.fill_preserve()

    if barred and length > 24:
        ctx.save()
        ctx.clip_preserve()
        ctx.set_source_rgba(0.14, 0.09, 0.05, 0.36 * alpha)
        ctx.set_line_width(max(1.8, length * 0.055))
        for b_u in (0.28, 0.48, 0.68, 0.84):
            bx = length * b_u
            ctx.move_to(bx - 2, -width)
            ctx.line_to(bx + 2, width)
        ctx.stroke()
        ctx.restore()

    # Subtle feather outline & pale central quill shaft (rachis)
    ctx.set_source_rgba(0.12, 0.08, 0.05, 0.55 * alpha)
    ctx.set_line_width(1.0)
    ctx.stroke()

    ctx.move_to(2, 0)
    ctx.line_to(length * 0.90, 0)
    ctx.set_source_rgba(0.86, 0.78, 0.62, 0.42 * alpha)
    ctx.set_line_width(1.1)
    ctx.stroke()
    ctx.restore()


def _draw_hawk_wing(
    ctx: cairo.Context,
    shoulder: Tuple[float, float],
    flap_val: float,
    dive_fold: float,
    is_far: bool,
    scale_y: float = 1.0,
    alpha: float = 1.0,
) -> None:
    """
    Anatomically accurate cambered raptor wing:
      - Leading wing spar sweeps UP and BACK from shoulder -> elbow -> wrist
      - All 11 barred secondaries and 7 slotted primaries fan smoothly BACKWARD
        toward the trailing edge (never blocking the hawk's head, zero gaps!).
      - Layered mottled chestnut-umber coverts bridge the entire leading airfoil.
    """
    shade = 0.74 if is_far else 1.0
    sx, sy = shoulder
    fold_u = clamp(dive_fold)

    # Leading edge spar angles (in screen coords: -pi/2 = straight UP, -pi = LEFT/BACK)
    if is_far:
        # Far wing lifts up-and-slightly-forward in 3/4 view so both majestic wings read clearly!
        inner_ang = lerp(-1.55 + flap_val * 0.45, -2.35, fold_u)
        outer_ang = lerp(-1.42 + flap_val * 0.62, -2.68, fold_u)
        inner_len = lerp(72.0, 52.0, fold_u)
        outer_len = lerp(74.0, 56.0, fold_u)
    else:
        # Near wing sweeps broadly UP-and-BACK during upstroke, DOWN-and-BACK on power downstroke
        inner_ang = lerp(-2.02 + flap_val * 0.65, -2.48, fold_u)
        outer_ang = lerp(-1.90 + flap_val * 0.85, -2.78, fold_u)
        inner_len = lerp(78.0, 56.0, fold_u)
        outer_len = lerp(80.0, 62.0, fold_u)

    elbow = (sx + math.cos(inner_ang) * inner_len, sy + math.sin(inner_ang) * inner_len * scale_y)
    wrist = (elbow[0] + math.cos(outer_ang) * outer_len, elbow[1] + math.sin(outer_ang) * outer_len * scale_y)

    # Trailing direction points smoothly BACKWARD (more negative angle, toward -pi / tail!)
    # Helper: point along continuous wing spar u in [0..1] (0=shoulder, 0.5=elbow, 1=wrist)
    def spar_pt(u: float) -> Tuple[float, float]:
        if u <= 0.5:
            t_s = u * 2.0
            return (lerp(sx, elbow[0], t_s), lerp(sy, elbow[1], t_s))
        t_s = (u - 0.5) * 2.0
        return (lerp(elbow[0], wrist[0], t_s), lerp(elbow[1], wrist[1], t_s))

    # 1. 7 Individually articulated slotted PRIMARY wingtip feathers fanning seamlessly from wrist -> mid-outer wing!
    n_prim = 7
    for i in range(n_prim - 1, -1, -1):
        u = i / float(n_prim - 1)  # 0 = leading tip primary -> 1 = inner primary meeting secondaries
        base = spar_pt(lerp(1.0, 0.64, u))
        fan_spread = lerp(0.88, 0.28, fold_u)
        # Leading primary extends along outer_ang; subsequent primaries fan BACKWARD (-angle)!
        p_ang = outer_ang - lerp(-0.04, fan_spread, u) - 0.06 * math.sin(flap_val * 1.5)
        p_len = lerp(96.0, 74.0, u) * lerp(1.0, 0.86, fold_u)
        _draw_single_feather(
            ctx,
            base,
            p_ang,
            p_len,
            width=16.5,
            col_base=(0.76 * shade, 0.64 * shade, 0.46 * shade),
            col_tip=(0.16 * shade, 0.11 * shade, 0.08 * shade),
            barred=True,
            alpha=alpha,
        )

    # 2. 11 Overlapping barred SECONDARY feathers bridging seamlessly from inner primary all the way to torso!
    n_sec = 11
    for i in range(n_sec - 1, -1, -1):
        u = i / float(n_sec - 1)  # 0 = shoulder -> 1 = outer elbow overlapping inner primary!
        base = spar_pt(lerp(0.02, 0.68, u))
        # Sweep smoothly backward toward tail (-2.95 near body -> blending cleanly into primary fan near wrist!)
        s_ang = lerp(inner_ang - 1.18, outer_ang - lerp(0.86, 0.30, fold_u), u)
        s_len = lerp(64.0, 78.0, math.sin(clamp(u * 0.85 + 0.15) * math.pi * 0.6)) * lerp(1.0, 0.80, fold_u)
        _draw_single_feather(
            ctx,
            base,
            s_ang,
            s_len,
            width=19.0,
            col_base=(0.54 * shade, 0.36 * shade, 0.19 * shade),
            col_tip=(0.24 * shade, 0.15 * shade, 0.09 * shade),
            barred=True,
            alpha=alpha,
        )

    # 3. Continuous aerodynamic mottled chestnut-umber COVERT plumage along the leading airfoil
    # Compute trailing edge of coverts (offset backward along secondary/primary direction)
    trail_ang_in = inner_ang - 1.25
    trail_ang_mid = (inner_ang + outer_ang) * 0.5 - 1.05
    trail_ang_out = outer_ang - 0.85
    c_in = (sx + math.cos(trail_ang_in) * 34.0, sy + math.sin(trail_ang_in) * 34.0)
    c_mid = (elbow[0] + math.cos(trail_ang_mid) * 38.0, elbow[1] + math.sin(trail_ang_mid) * 38.0)
    c_out = (wrist[0] + math.cos(trail_ang_out) * 26.0, wrist[1] + math.sin(trail_ang_out) * 26.0)

    ctx.new_path()
    ctx.move_to(sx + 6, sy + 6)
    ctx.curve_to(sx + 10, sy - 10, elbow[0] + 10, elbow[1] - 8, wrist[0] + 4, wrist[1] - 2)
    ctx.curve_to(c_out[0], c_out[1], c_mid[0], c_mid[1], c_in[0], c_in[1])
    ctx.close_path()

    cpat = cairo.LinearGradient(sx + 8, sy - 12, c_mid[0], c_mid[1])
    cpat.add_color_stop_rgba(0.0, 0.46 * shade, 0.29 * shade, 0.15 * shade, alpha)
    cpat.add_color_stop_rgba(0.50, 0.34 * shade, 0.20 * shade, 0.10 * shade, alpha)
    cpat.add_color_stop_rgba(1.0, 0.22 * shade, 0.13 * shade, 0.07 * shade, alpha)
    ctx.set_source(cpat)
    ctx.fill_preserve()

    # Layered covert feather scallops along the wing!
    ctx.save()
    ctx.clip_preserve()
    for row in range(3):
        for k in range(8):
            u = (k + 0.25) / 8.0
            sp = spar_pt(u)
            cx_f = sp[0] + math.cos(trail_ang_mid) * (8.0 + row * 10.5)
            cy_f = sp[1] + math.sin(trail_ang_mid) * (8.0 + row * 10.5)
            ctx.arc(cx_f, cy_f, 10.5, 0.2, math.pi - 0.2)
            ctx.set_source_rgba(0.70 * shade, 0.50 * shade, 0.30 * shade, 0.38 * alpha)
            ctx.set_line_width(1.6)
            ctx.stroke()
    ctx.restore()

    ctx.set_source_rgba(0.14, 0.08, 0.05, 0.62 * alpha)
    ctx.set_line_width(1.6)
    ctx.stroke()


def _draw_hawk_talon_foot(
    ctx: cairo.Context,
    leg_base: Tuple[float, float],
    extend_u: float,
    grip_u: float,
    alpha: float,
) -> Tuple[float, float]:
    """Feathered raptor thigh, scaly golden-yellow tarsus, and 4 curved black talons."""
    lx, ly = leg_base
    leg_ang = lerp(2.78, 0.92, clamp(extend_u))
    knee = (lx + math.cos(leg_ang) * 24.0, ly + math.sin(leg_ang) * 24.0)
    foot_ang = lerp(2.88, 1.32, clamp(extend_u))
    foot = (knee[0] + math.cos(foot_ang) * 26.0, knee[1] + math.sin(foot_ang) * 26.0)

    draw_tapered_ribbon(ctx, [(lx, ly), knee], [13.0, 8.5], close_tip=False)
    ctx.set_source_rgba(0.84, 0.76, 0.60, alpha)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.32, 0.22, 0.12, 0.55 * alpha)
    ctx.set_line_width(1.2)
    ctx.stroke()

    _draw_insect_leg_segment(ctx, knee, foot, 6.2, 5.0, (0.94, 0.76, 0.22), (0.72, 0.52, 0.12), alpha)

    spread = lerp(1.0, 0.38, clamp(grip_u))
    toe_specs = [(-0.55 * spread, 18.0), (0.05, 22.0), (0.55 * spread, 17.0), (2.45 - 0.5 * clamp(grip_u), 16.0)]
    for t_ang_rel, t_len in toe_specs:
        ta = foot_ang + t_ang_rel
        tx = foot[0] + math.cos(ta) * t_len
        ty = foot[1] + math.sin(ta) * t_len
        _draw_insect_leg_segment(ctx, foot, (tx, ty), 4.0, 2.5, (0.95, 0.78, 0.24), (0.70, 0.50, 0.12), alpha)

        claw_curl = ta + lerp(0.42, 0.95, clamp(grip_u)) * (1.0 if t_ang_rel < 1.5 else -1.0)
        cx_c = tx + math.cos(claw_curl) * 11.5
        cy_c = ty + math.sin(claw_curl) * 11.5
        ctx.new_path()
        ctx.move_to(tx, ty)
        mx = tx + math.cos(ta) * 6.5
        my = ty + math.sin(ta) * 6.5
        ctx.curve_to(mx, my, mx, my, cx_c, cy_c)
        ctx.set_source_rgba(0.08, 0.07, 0.07, alpha)
        ctx.set_line_width(2.6)
        ctx.stroke()

    return foot


def draw_hawk(
    ctx: cairo.Context,
    x: float,
    y: float,
    scale: float = 1.0,
    facing: float = 1.0,
    body_pitch: float = 0.0,
    t: float = 0.0,
    wing_flap_phase: float = 0.0,
    flap_amp: float = 0.25,
    dive_fold: float = 0.0,
    tail_spread: float = 0.75,
    talon_extend: float = 0.0,
    talon_grip: float = 0.0,
    head_pitch: float = 0.0,
    carrying_snake: bool = False,
    alpha: float = 1.0,
) -> Tuple[float, float]:
    """
    Render a majestic articulated Red-tailed Hawk (Buteo jamaicensis).
    Returns world (x, y) of the talons for precise snake-snatch alignment.
    """
    if alpha <= 0.01:
        return (x, y)

    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(body_pitch)
    ctx.scale(scale * facing, scale)

    flap_val = math.sin(wing_flap_phase) * flap_amp

    # 1. FAR WING (sweeps up cleanly behind torso)
    _draw_hawk_wing(ctx, (14, -14), flap_val=flap_val * 0.92 + 0.10, dive_fold=dive_fold, is_far=True, scale_y=0.92, alpha=alpha)

    # 2. FANNED RUFOUS-CHESTNUT RED-TAILED HAWK TAIL (9 overlapping rectrices!)
    n_tail = 9
    fan = lerp(0.20, 0.74, clamp(tail_spread)) * lerp(1.0, 0.35, clamp(dive_fold))
    for i in range(n_tail):
        u = (i / float(n_tail - 1)) * 2.0 - 1.0
        t_ang = math.pi + 0.16 + u * fan
        t_len = 76.0 - abs(u) * 8.0
        _draw_single_feather(
            ctx,
            (-48, 4),
            t_ang,
            t_len,
            width=15.0,
            col_base=(0.68, 0.28, 0.14),
            col_tip=(0.82, 0.42, 0.20),
            barred=False,
            alpha=alpha,
        )
        bx = -48 + math.cos(t_ang) * (t_len * 0.84)
        by = 4 + math.sin(t_ang) * (t_len * 0.84)
        ctx.arc(bx, by, 3.6, 0, math.tau)
        ctx.set_source_rgba(0.16, 0.09, 0.06, 0.55 * alpha)
        ctx.fill()

    # 3. IF CARRYING SNAKE: Render gracefully curved captured snake grasped in talons!
    talon_pos_local = (-4.0, 20.0)
    if carrying_snake:
        leg_ang = lerp(2.78, 0.92, clamp(talon_extend))
        kx_t = -8.0 + math.cos(leg_ang) * 24.0
        ky_t = 16.0 + math.sin(leg_ang) * 24.0
        f_ang = lerp(2.88, 1.32, clamp(talon_extend))
        fx_t = kx_t + math.cos(f_ang) * 26.0
        fy_t = ky_t + math.sin(f_ang) * 26.0
        talon_pos_local = (fx_t, fy_t)

        sn_pts = []
        for k in range(32):
            u = k / 31.0
            rel = u - 0.42
            sx_s = fx_t - rel * 140.0 + math.sin(u * 7.5 - t * 4.5) * 16.0 * abs(rel)
            sy_s = fy_t + abs(rel) * 82.0 + math.cos(u * 5.2 - t * 3.8) * 11.0 * abs(rel)
            sn_pts.append((sx_s, sy_s))
        draw_snake(ctx, sn_pts, max_radius=8.5, draw_shadow=False, alpha=alpha, t=t)

    # 4. FEATHERED LEGS & SCALY YELLOW TALONS
    _draw_hawk_talon_foot(ctx, (-12, 16), extend_u=talon_extend, grip_u=talon_grip, alpha=alpha)
    foot_near = _draw_hawk_talon_foot(ctx, (-4, 18), extend_u=talon_extend, grip_u=talon_grip, alpha=alpha)
    talon_pos_local = foot_near

    # 5. STREAMLINED MUSCULAR RAPTOR TORSO
    ctx.new_path()
    ctx.move_to(54, -8)
    ctx.curve_to(36, -24, -4, -22, -52, 0)
    ctx.curve_to(-32, 22, 8, 28, 38, 18)
    ctx.curve_to(52, 12, 58, 2, 54, -8)
    ctx.close_path()

    bpat = cairo.LinearGradient(12, -24, 6, 28)
    bpat.add_color_stop_rgba(0.0, 0.36, 0.22, 0.12, alpha)
    bpat.add_color_stop_rgba(0.45, 0.46, 0.30, 0.16, alpha)
    bpat.add_color_stop_rgba(0.68, 0.88, 0.82, 0.68, alpha)
    bpat.add_color_stop_rgba(1.0, 0.76, 0.68, 0.54, alpha)
    ctx.set_source(bpat)
    ctx.fill_preserve()

    ctx.save()
    ctx.clip_preserve()
    for b_i in range(12):
        bx = lerp(-24, 24, b_i / 11.0)
        by = 12 + (b_i % 3) * 4.5
        ctx.save()
        ctx.translate(bx, by)
        ctx.rotate(0.25)
        ctx.scale(5.5, 2.2)
        ctx.arc(0, 0, 1.0, 0, math.tau)
        ctx.set_source_rgba(0.26, 0.15, 0.08, 0.62 * alpha)
        ctx.fill()
        ctx.restore()
    ctx.restore()

    ctx.set_source_rgba(0.14, 0.09, 0.05, 0.68 * alpha)
    ctx.set_line_width(1.6)
    ctx.stroke()

    # 6. NEAR HERO ARTICULATED WING (anchored cleanly behind the shoulder, sweeping UP & BACK!)
    _draw_hawk_wing(ctx, (10, -10), flap_val=flap_val, dive_fold=dive_fold, is_far=False, scale_y=1.0, alpha=alpha)

    # 7. FIERCE SCULPTED RAPTOR HEAD (drawn crisp & unobstructed in front of wing shoulder!)
    ctx.save()
    ctx.translate(52, -7)
    ctx.rotate(head_pitch)

    ctx.new_path()
    ctx.move_to(-12, -10)
    ctx.curve_to(-2, -20, 14, -19, 22, -9)
    ctx.curve_to(24, 2, 16, 12, 4, 13)
    ctx.curve_to(-8, 13, -16, 4, -12, -10)
    ctx.close_path()
    hpat = cairo.LinearGradient(2, -20, 2, 14)
    hpat.add_color_stop_rgba(0.0, 0.30, 0.18, 0.10, alpha)
    hpat.add_color_stop_rgba(0.55, 0.44, 0.28, 0.15, alpha)
    hpat.add_color_stop_rgba(1.0, 0.84, 0.78, 0.64, alpha)
    ctx.set_source(hpat)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.14, 0.09, 0.05, 0.72 * alpha)
    ctx.set_line_width(1.4)
    ctx.stroke()

    # Yellow waxy cere at base of beak
    ctx.new_path()
    ctx.move_to(19, -9)
    ctx.curve_to(26, -9, 27, -2, 21, -1)
    ctx.close_path()
    ctx.set_source_rgba(0.94, 0.78, 0.20, alpha)
    ctx.fill()

    # Sharply hooked dark slate-charcoal raptor beak!
    ctx.new_path()
    ctx.move_to(23, -8)
    ctx.curve_to(35, -7, 40, 1, 35, 11)
    ctx.curve_to(32, 5, 28, 2, 20, 2)
    ctx.close_path()
    bk = cairo.LinearGradient(22, -8, 36, 10)
    bk.add_color_stop_rgba(0.0, 0.40, 0.40, 0.38, alpha)
    bk.add_color_stop_rgba(0.5, 0.22, 0.22, 0.22, alpha)
    bk.add_color_stop_rgba(1.0, 0.08, 0.08, 0.09, alpha)
    ctx.set_source(bk)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.06, 0.06, 0.06, 0.85 * alpha)
    ctx.set_line_width(1.2)
    ctx.stroke()

    # Piercing golden-amber raptor eye beneath fierce overhanging brow ridge
    ctx.arc(12.5, -6.5, 4.5, 0, math.tau)
    ctx.set_source_rgba(0.98, 0.78, 0.16, alpha)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.10, 0.07, 0.04, 0.88 * alpha)
    ctx.set_line_width(1.0)
    ctx.stroke()
    ctx.arc(13.2, -6.5, 2.2, 0, math.tau)
    ctx.set_source_rgba(0.04, 0.04, 0.04, alpha)
    ctx.fill()
    ctx.arc(14.3, -7.6, 1.0, 0, math.tau)
    ctx.set_source_rgba(1.0, 1.0, 0.94, 0.95 * alpha)
    ctx.fill()

    ctx.move_to(6, -11.5)
    ctx.line_to(19.5, -8.5)
    ctx.set_source_rgba(0.24, 0.14, 0.08, alpha)
    ctx.set_line_width(3.0)
    ctx.stroke()

    ctx.restore()  # end head

    ctx.restore()

    # Return approximate talon world coordinate
    c_p, s_p = math.cos(body_pitch), math.sin(body_pitch)
    tw_x = x + (talon_pos_local[0] * facing * c_p - talon_pos_local[1] * s_p) * scale
    tw_y = y + (talon_pos_local[0] * facing * s_p + talon_pos_local[1] * c_p) * scale
    return (tw_x, tw_y)


def draw_hawk_eye_closeup(
    ctx: cairo.Context,
    cx: float,
    cy: float,
    radius: float,
    focus_u: float = 0.5,
    t: float = 0.0,
) -> None:
    """Dramatic BBC-documentary macro portrait of the Red-tailed Hawk locking onto the snake."""
    ctx.save()
    # Layered dark-chocolate & warm chestnut head feathers filling the frame
    bg = cairo.RadialGradient(cx + radius * 0.4, cy - radius * 0.4, radius * 0.4, cx, cy, radius * 4.5)
    bg.add_color_stop_rgb(0.0, 0.42, 0.26, 0.13)
    bg.add_color_stop_rgb(0.55, 0.24, 0.14, 0.08)
    bg.add_color_stop_rgb(1.0, 0.12, 0.07, 0.04)
    ctx.set_source(bg)
    ctx.rectangle(cx - radius * 5.0, cy - radius * 3.5, radius * 10.0, radius * 7.0)
    ctx.fill()

    # Hundreds of fine painterly overlapping facial feather strokes sweeping back from beak!
    for i in range(130):
        fx = cx + (hash1d(i * 31 + 7) - 0.48) * radius * 8.2
        fy = cy + (hash1d(i * 41 + 13) - 0.50) * radius * 5.2
        if math.hypot(fx - cx, fy - cy) < radius * 1.15:
            continue
        ang = math.atan2(fy - cy, fx - cx) * 0.22 + math.pi
        flen = radius * (0.45 + 0.45 * hash1d(i * 53 + 3))
        col = lerp_rgb((0.22, 0.13, 0.07), (0.56, 0.36, 0.18), hash1d(i * 61 + 11))
        ctx.move_to(fx, fy)
        ctx.line_to(fx + math.cos(ang) * flen, fy + math.sin(ang) * flen)
        ctx.set_source_rgba(col[0], col[1], col[2], 0.55)
        ctx.set_line_width(radius * 0.08)
        ctx.stroke()

    # Organic curved yellow cere & lethal dark slate hooked beak entering on the right!
    ctx.new_path()
    ctx.move_to(cx + radius * 1.50, cy - radius * 0.44)
    ctx.curve_to(cx + radius * 2.15, cy - radius * 0.56, cx + radius * 2.55, cy - radius * 0.25, cx + radius * 2.52, cy + radius * 0.35)
    ctx.curve_to(cx + radius * 2.45, cy + radius * 0.92, cx + radius * 1.92, cy + radius * 1.25, cx + radius * 1.48, cy + radius * 0.98)
    ctx.curve_to(cx + radius * 1.32, cy + radius * 0.45, cx + radius * 1.32, cy - radius * 0.08, cx + radius * 1.50, cy - radius * 0.44)
    ctx.close_path()
    cpat = cairo.RadialGradient(cx + radius * 1.95, cy - radius * 0.10, radius * 0.12, cx + radius * 1.95, cy + radius * 0.30, radius * 1.05)
    cpat.add_color_stop_rgb(0.0, 0.98, 0.84, 0.28)
    cpat.add_color_stop_rgb(0.65, 0.82, 0.62, 0.14)
    cpat.add_color_stop_rgb(1.0, 0.46, 0.34, 0.08)
    ctx.set_source(cpat)
    ctx.fill()

    # Authentic oval raptor nostril (nares) set inside the waxy yellow cere!
    ctx.save()
    ctx.translate(cx + radius * 2.02, cy + radius * 0.05)
    ctx.rotate(-0.22)
    ctx.scale(radius * 0.16, radius * 0.10)
    ctx.arc(0, 0, 1.0, 0, math.tau)
    ctx.set_source_rgb(0.14, 0.10, 0.05)
    ctx.fill()
    ctx.restore()

    # Curved dark slate hooked beak emerging seamlessly from cere
    ctx.new_path()
    ctx.move_to(cx + radius * 2.32, cy - radius * 0.42)
    ctx.curve_to(cx + radius * 3.85, cy - radius * 0.22, cx + radius * 4.65, cy + radius * 1.25, cx + radius * 4.10, cy + radius * 2.55)
    ctx.curve_to(cx + radius * 3.15, cy + radius * 1.85, cx + radius * 2.65, cy + radius * 1.35, cx + radius * 1.96, cy + radius * 1.18)
    ctx.curve_to(cx + radius * 2.38, cy + radius * 0.72, cx + radius * 2.48, cy + radius * 0.05, cx + radius * 2.32, cy - radius * 0.42)
    ctx.close_path()
    bpat = cairo.LinearGradient(cx + radius * 2.3, cy - radius * 0.3, cx + radius * 4.4, cy + radius * 2.2)
    bpat.add_color_stop_rgb(0.0, 0.44, 0.45, 0.46)
    bpat.add_color_stop_rgb(0.45, 0.20, 0.21, 0.22)
    bpat.add_color_stop_rgb(1.0, 0.06, 0.06, 0.07)
    ctx.set_source(bpat)
    ctx.fill()

    # Fine dark bristled lores feathers bridging eye socket smoothly into cere!
    for l_i in range(14):
        ly = cy + lerp(-0.35, 0.85, l_i / 13.0) * radius
        lx = cx + radius * (1.18 + 0.22 * hash1d(l_i * 19 + 7))
        ctx.move_to(lx, ly)
        ctx.line_to(lx + radius * 0.34, ly + (hash1d(l_i * 23) - 0.5) * radius * 0.14)
        ctx.set_source_rgba(0.14, 0.09, 0.05, 0.65)
        ctx.set_line_width(radius * 0.032)
        ctx.stroke()

    # Dark orbital eye socket ring
    ctx.arc(cx, cy, radius * 1.16, 0, math.tau)
    ctx.set_source_rgb(0.11, 0.08, 0.06)
    ctx.fill()

    # Piercing golden-amber raptor eye
    ctx.save()
    ctx.arc(cx, cy, radius, 0, math.tau)
    ctx.clip_preserve()
    ipat = cairo.RadialGradient(cx + radius * 0.18, cy - radius * 0.18, radius * 0.06, cx, cy, radius * 1.05)
    ipat.add_color_stop_rgb(0.0, 1.00, 0.90, 0.32)
    ipat.add_color_stop_rgb(0.55, 0.94, 0.68, 0.14)
    ipat.add_color_stop_rgb(0.88, 0.60, 0.32, 0.06)
    ipat.add_color_stop_rgb(1.0, 0.24, 0.12, 0.03)
    ctx.set_source(ipat)
    ctx.fill()

    # Radial iris striations
    for i in range(72):
        a = (i / 72.0) * math.tau
        ctx.move_to(cx + math.cos(a) * radius * 0.35, cy + math.sin(a) * radius * 0.35)
        ctx.line_to(cx + math.cos(a) * radius * 0.95, cy + math.sin(a) * radius * 0.95)
        ctx.set_source_rgba(0.45, 0.22, 0.04, 0.24)
        ctx.set_line_width(radius * 0.022)
        ctx.stroke()

    # Pinpoint predator pupil tightening dramatically on focus_u -> 1.0!
    pr = radius * lerp(0.46, 0.35, clamp(focus_u))
    ctx.arc(cx + radius * 0.05, cy + radius * 0.02, pr, 0, math.tau)
    ctx.set_source_rgb(0.03, 0.03, 0.04)
    ctx.fill()

    # Crisp sunlit sky reflection gleaming in the raptor's eye
    ctx.arc(cx + radius * 0.28, cy - radius * 0.30, radius * 0.22, 0, math.tau)
    ctx.set_source_rgba(1.0, 0.99, 0.94, 0.94)
    ctx.fill()
    ctx.arc(cx - radius * 0.30, cy + radius * 0.35, radius * 0.10, 0, math.tau)
    ctx.set_source_rgba(0.85, 0.92, 0.70, 0.45)
    ctx.fill()

    # Fierce overhanging feathered supraorbital brow ridge casting a dark shadow across top of eye!
    ctx.new_path()
    ctx.move_to(cx - radius * 1.45, cy - radius * 1.35)
    ctx.line_to(cx + radius * 1.45, cy - radius * 0.55)
    ctx.line_to(cx + radius * 1.45, cy - radius * 1.75)
    ctx.line_to(cx - radius * 1.45, cy - radius * 1.75)
    ctx.close_path()
    ctx.set_source_rgba(0.06, 0.04, 0.02, 0.42)
    ctx.fill()
    ctx.restore()

    # Thick sculptural feathered supraorbital brow crest across upper eye
    ctx.new_path()
    ctx.move_to(cx - radius * 1.45, cy - radius * 1.05)
    ctx.curve_to(cx - radius * 0.2, cy - radius * 1.28, cx + radius * 0.85, cy - radius * 0.95, cx + radius * 1.65, cy - radius * 0.42)
    ctx.curve_to(cx + radius * 0.90, cy - radius * 0.42, cx - radius * 0.15, cy - radius * 0.72, cx - radius * 1.35, cy - radius * 0.55)
    ctx.close_path()
    brp = cairo.LinearGradient(cx, cy - radius * 1.3, cx, cy - radius * 0.4)
    brp.add_color_stop_rgb(0.0, 0.46, 0.29, 0.15)
    brp.add_color_stop_rgb(1.0, 0.18, 0.10, 0.05)
    ctx.set_source(brp)
    ctx.fill()
    ctx.restore()


# =============================================================================
# Visual QA Studio Character Turnaround / Design Sheet
# =============================================================================

if __name__ == "__main__":
    import os
    import cv2

    W, H = 1920, 1080
    bg = np.full((H, W, 3), (235, 243, 246), dtype=np.uint8)
    layer = CairoLayer(W, H)
    ctx = layer.ctx

    # Subtle studio parchment grid for inspection
    ctx.set_source_rgba(0.75, 0.78, 0.72, 0.45)
    ctx.set_line_width(2.0)
    ctx.move_to(640, 0); ctx.line_to(640, H)
    ctx.move_to(1280, 0); ctx.line_to(1280, H)
    ctx.move_to(0, 540); ctx.line_to(W, 540)
    ctx.stroke()

    # Cell 1: Germinating Seed + Growing Grass Blades (with bite notch & dewdrop!)
    draw_germinating_seed(ctx, 210, 310, soil_surface_y=210, absorb_u=0.9, split_u=0.95, root_u=0.9, shoot_u=0.85, t=2.0, scale=0.85)
    for idx in range(7):
        draw_grass_blade(
            ctx,
            base_x=380 + idx * 32,
            base_y=470,
            height=220 + (idx % 3) * 45,
            base_width=18,
            base_lean=(idx - 3) * 18,
            wind_val=0.45,
            grow_u=1.0,
            color_mix=idx / 6.0,
            bite_notches=[(0.62, 0.75, 0.08), (0.48, 0.60, 0.07)] if idx == 3 else (),
            dew_s=0.72 if idx == 2 else None,
        )

    # Cell 2: Realistic Articulated Grasshopper
    draw_grasshopper(ctx, 960, 290, scale=2.2, facing=1.0, t=1.5, chew_phase=0.35, hop_extend=0.05)

    # Cell 3: Realistic Emerald Green Pond Frog (+ tongue strike!)
    draw_frog(ctx, 1520, 310, scale=2.0, facing=1.0, t=2.2, crouch=0.2, mouth_open=0.75, tongue_ext=0.85, tongue_target_world=(1840, 250))

    # Cell 4: Realistic Scaled Undulating Snake
    spine = compute_snake_spine((470, 810), heading_rad=-0.12, total_len=410, slither_phase=1.8, wave_amp=42, coil_u=0.25)
    draw_snake(ctx, spine, max_radius=20.0, tongue_flick=0.95, t=2.4)

    # Cell 5: Majestic Red-tailed Hawk carrying snake
    draw_hawk(ctx, 980, 790, scale=1.55, facing=1.0, body_pitch=-0.14, t=1.2, wing_flap_phase=0.5, flap_amp=0.35, talon_extend=0.95, talon_grip=0.95, carrying_snake=True)

    # Cell 6: Macro Frog Eye & Macro Hawk Eye side-by-side
    ctx.save()
    ctx.rectangle(1285, 545, 310, 530); ctx.clip()
    draw_frog_eye_closeup(ctx, 1440, 810, radius=95, focus_u=0.75, t=1.0)
    ctx.restore()
    ctx.save()
    ctx.rectangle(1605, 545, 310, 530); ctx.clip()
    draw_hawk_eye_closeup(ctx, 1730, 810, radius=85, focus_u=0.85, t=1.0)
    ctx.restore()

    layer.composite_over(bg, opacity=1.0)
    out_path = os.path.join(os.path.dirname(__file__), "assets", "character_sheet.png")
    cv2.imwrite(out_path, bg)
    print(f"Saved character design sheet: {out_path}")

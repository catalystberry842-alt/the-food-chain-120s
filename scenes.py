"""
scenes.py — Complete 6-Scene Cinematic Choreography & Seamless Transitions
========================================================================

Orchestrates the entire 124-second animated nature film frame-by-frame across
all 6 connected scenes, maintaining ecosystem continuity, living atmospheric
particles, dynamic water reflections, real-time grass bending & bite erosion,
wildlife macro eye close-ups, seamless transitions, and the animated finale.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cairo
import cv2
import numpy as np

from animation import (
    HEIGHT,
    SCENE_SPECS,
    TRANSITION_OVERLAP,
    WIDTH,
    blink_envelope,
    clamp,
    ease_in_cubic,
    ease_in_out_cubic,
    ease_in_quad,
    ease_out_back,
    ease_out_cubic,
    ease_out_quad,
    fbm_1d,
    hash1d,
    inv_lerp,
    lerp,
    lerp_rgb,
    parabolic_arc,
    pulse_bell,
    remap,
    smootherstep,
    smoothstep,
    wind_field,
)
from camera import CameraState
from characters import (
    compute_snake_spine,
    draw_frog,
    draw_frog_eye_closeup,
    draw_germinating_seed,
    draw_grass_blade,
    draw_grasshopper,
    draw_hawk,
    draw_hawk_eye_closeup,
    draw_snake,
    eval_grass_blade,
)
from drawing_engine import (
    CairoLayer,
    apply_cinematic_post,
    apply_pond_water_reflection,
    composite_straight_bgra,
    draw_soft_shadow,
    draw_text_overlay,
)


# =============================================================================
# Reusable Living Atmospheric Elements (Birds, Insects, Pollen, Wind Streaks)
# =============================================================================

def _draw_distant_bird_flock(ctx: cairo.Context, t: float, base_x: float, base_y: float, count: int = 7) -> None:
    """Graceful flapping silhouettes of distant morning birds crossing the sunlit sky."""
    for i in range(count):
        bx = base_x + i * 38.0 + math.sin(i * 1.9) * 26.0 + t * (26.0 + (i % 3) * 4.5)
        by = base_y + (i - count * 0.5) * 14.0 + math.sin(t * 2.0 + i * 1.3) * 8.0
        sc = 0.72 + 0.35 * hash1d(i * 41 + 9)
        flap = math.sin(t * (7.2 + (i % 3) * 0.7) + i * 1.1)
        wing_y = flap * 7.5 * sc
        span = 14.5 * sc

        ctx.new_path()
        ctx.move_to(bx - span, by - wing_y)
        ctx.curve_to(bx - span * 0.45, by - wing_y * 0.8 - 4.0 * sc, bx - 2.0, by - 2.0, bx, by)
        ctx.curve_to(bx + 2.0, by - 2.0, bx + span * 0.45, by - wing_y * 0.8 - 4.0 * sc, bx + span, by - wing_y)
        ctx.set_source_rgba(0.18, 0.24, 0.30, 0.62)
        ctx.set_line_width(2.0 * sc)
        ctx.stroke()


def _draw_pollen_and_insects(
    ctx: cairo.Context,
    t: float,
    count: int = 38,
    y_min: float = 180.0,
    y_max: float = 960.0,
    warm_alpha: float = 0.62,
    seed: int = 500,
) -> None:
    """Luminous sunlit floating pollen motes and delicate darting meadow gnats."""
    for i in range(count):
        base_x = hash1d(seed + i * 17) * (WIDTH + 200) - 100
        base_y = lerp(y_min, y_max, hash1d(seed + i * 29))
        drift_x = (base_x + t * (16.0 + 22.0 * hash1d(seed + i * 37))) % (WIDTH + 180) - 90
        drift_y = base_y + math.sin(t * (0.9 + hash1d(seed + i * 43)) + i * 2.1) * 24.0
        r = 1.6 + 2.8 * hash1d(seed + i * 53)
        twinkle = 0.55 + 0.45 * math.sin(t * 2.8 + i * 3.7)

        if i % 6 == 0:
            # Tiny darting sunlit meadow midge/insect
            ix = drift_x + math.sin(t * 6.5 + i) * 18.0
            iy = drift_y + math.cos(t * 8.2 + i * 1.4) * 12.0
            ctx.arc(ix, iy, 2.0, 0, math.tau)
            ctx.set_source_rgba(0.20, 0.18, 0.10, 0.52)
            ctx.fill()
            # Delicate glinting wing flutter
            wf = 3.0 + 2.2 * abs(math.sin(t * 42.0 + i))
            ctx.arc(ix - 1.5, iy - 2.0, wf * 0.65, 0, math.tau)
            ctx.set_source_rgba(1.0, 0.98, 0.85, 0.42)
            ctx.fill()
        else:
            # Soft glowing golden pollen mote
            pat = cairo.RadialGradient(drift_x, drift_y, 0.2, drift_x, drift_y, r * 2.4)
            pat.add_color_stop_rgba(0.0, 1.0, 0.96, 0.76, warm_alpha * twinkle)
            pat.add_color_stop_rgba(0.5, 1.0, 0.90, 0.58, warm_alpha * 0.42 * twinkle)
            pat.add_color_stop_rgba(1.0, 1.0, 0.88, 0.50, 0.0)
            ctx.set_source(pat)
            ctx.arc(drift_x, drift_y, r * 2.4, 0, math.tau)
            ctx.fill()


def _draw_dragonfly(ctx: cairo.Context, x: float, y: float, t: float, scale: float = 1.0, facing: float = 1.0) -> None:
    """Delicate iridescent azure pond dragonfly skimming above the water."""
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(scale * facing, scale)

    # Shimmering translucent fluttering wings
    wf = math.sin(t * 58.0) * 0.35
    for w_idx, (ang0, wlen) in enumerate([(-0.45 + wf, 22.0), (-0.75 - wf, 19.0)]):
        ctx.save()
        ctx.translate(4, -2)
        ctx.rotate(ang0)
        ctx.scale(wlen, 3.6)
        ctx.arc(0.5, 0, 0.5, 0, math.tau)
        ctx.set_source_rgba(0.88, 0.96, 1.0, 0.48)
        ctx.fill()
        ctx.restore()

    # Slender iridescent turquoise-blue abdomen & thorax
    ctx.move_to(-26, 2)
    ctx.line_to(10, -1)
    ctx.set_source_rgba(0.14, 0.54, 0.68, 0.92)
    ctx.set_line_width(3.2)
    ctx.stroke()
    ctx.arc(11, -1.5, 3.2, 0, math.tau)
    ctx.set_source_rgba(0.18, 0.44, 0.52, 0.95)
    ctx.fill()
    ctx.restore()


def _draw_foreground_dof_blades(
    frame_bgr: np.ndarray,
    cam: CameraState,
    t: float,
    out_w: int,
    out_h: int,
    blade_count: int = 14,
    seed: int = 700,
    color_tint: str = "green",
) -> None:
    """Render softly blurred out-of-focus foreground framing blades for cinema lens depth."""
    if cam.dof_fg <= 0.02:
        return
    # Draw on half-res Cairo surface then Gaussian-blur for ultra-fast, gorgeous bokeh DOF!
    hw, hh = max(160, out_w // 2), max(90, out_h // 2)
    fg = CairoLayer(hw, hh)
    ctx = fg.ctx
    cam.apply_to_cairo(ctx, parallax=1.45, out_w=hw, out_h=hh)

    for i in range(blade_count):
        # Concentrate foreground framing near left/right & bottom margins (never obstructing center hero!)
        u = i / float(max(1, blade_count - 1))
        if 0.24 < u < 0.76 and i % 3 != 0:
            continue
        bx = lerp(-240.0, WIDTH + 240.0, u) + (hash1d(seed + i * 19) - 0.5) * 90.0
        by = HEIGHT + 95.0
        bh = lerp(240.0, 440.0, hash1d(seed + i * 29))
        if 0.24 < u < 0.76:
            bh *= 0.52  # shorter framing tufts along center bottom
        lean = (hash1d(seed + i * 37) - 0.45) * 140.0
        wval = wind_field(bx, t, strength=1.15)
        cmix = 0.85 if color_tint == "dry" else hash1d(seed + i * 47)
        draw_grass_blade(
            ctx,
            base_x=bx,
            base_y=by,
            height=bh,
            base_width=24.0 + 12.0 * hash1d(seed + i * 53),
            base_lean=lean,
            wind_val=wval,
            grow_u=1.0,
            color_mix=cmix,
            alpha=0.78,
            steps=14,
        )

    fg.surface.flush()
    k = max(3, int(round(13 * cam.dof_fg * (out_w / float(WIDTH)))) | 1)
    blurred = cv2.GaussianBlur(fg.buf, (k, k), 0)
    blurred_full = cv2.resize(blurred, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # Pure OpenCV SIMD premultiplied alpha composite (~6ms at 1080p!)
    inv_a = cv2.subtract(255, blurred_full[:, :, 3])
    inv_a3 = cv2.merge([inv_a, inv_a, inv_a])
    cv2.multiply(frame_bgr, inv_a3, dst=frame_bgr, scale=1.0 / 255.0)
    cv2.add(frame_bgr, blurred_full[:, :, :3], dst=frame_bgr)


# =============================================================================
# SCENE 1: LIFE BEGINS (00:00 - 00:22)
# =============================================================================

def _build_meadow_blades_table(seed: int = 100, count: int = 88, x0: float = -180.0, x1: float = 2120.0) -> List[Tuple]:
    """Deterministic table of individually articulated meadow grass blades sorted back-to-front."""
    blades = []
    for i in range(count):
        depth = hash1d(seed + i * 17) ** 1.15  # 0=midground far, 1=foreground close
        bx = lerp(x0, x1, (i + 0.25 + 0.5 * hash1d(seed + i * 23)) / float(count))
        by = lerp(740.0, 1110.0, depth)
        bh = lerp(110.0, 390.0, depth) * (0.72 + 0.52 * hash1d(seed + i * 31))
        bw = lerp(8.0, 25.0, depth) * (0.8 + 0.4 * hash1d(seed + i * 41))
        lean = (hash1d(seed + i * 47) - 0.38) * bh * 0.42
        cmix = hash1d(seed + i * 53)
        delay = hash1d(seed + i * 61)
        dew = 0.68 if (i % 13 == 0 and depth > 0.45) else None
        blades.append((by, bx, bh, bw, lean, cmix, delay, dew, depth))
    blades.sort(key=lambda b: b[0])
    return blades


_SCENE1_MEADOW_BLADES = _build_meadow_blades_table(seed=101, count=92, x0=-160.0, x1=2100.0)
_SCENE2_MEADOW_BLADES = _build_meadow_blades_table(seed=202, count=84, x0=-220.0, x1=2240.0)


def render_scene1(
    t: float,
    lt: float,
    env: Dict[str, np.ndarray],
    out_w: int = WIDTH,
    out_h: int = HEIGHT,
) -> np.ndarray:
    """
    SCENE 1: LIFE BEGINS (00:00 - 00:22, lt in [0..22])
      - 0.0 -> 9.8s: Subterranean soil macro cutaway: seed absorbs water, splits,
        grows branching downward roots, pushes green shoot through soil into morning light,
        first individual blades emerge and unfold around it.
      - 9.2 -> 11.4s: Seamless atmospheric blend up into the sunlit early-morning grassland.
      - 10.5 -> 22.0s: Lush individual blades grow & bend in gentle breeze, camera pulls back
        to reveal vast morning grassland, sunbeams, drifting insects & flying birds!
    """
    # Part A: Subterranean soil & sprouting seed (active lt <= 11.6s)
    frame_a: Optional[np.ndarray] = None
    blend_to_meadow = smootherstep(inv_lerp(9.0, 11.4, lt))

    if blend_to_meadow < 0.995:
        # Soil cross-section canvas is 2400x1400 with world origin (-240, -160)
        # Seed sits at world (960, 565), soil surface is at world y=400!
        soil_surf_y = 400.0
        seed_x, seed_y = 960.0, 565.0

        # Smooth choreographed macro camera: tight on seed -> cranes up following shoot breaking soil!
        if lt < 3.6:
            u = smootherstep(lt / 3.6)
            cam_a = CameraState(
                cx=lerp(960.0, 960.0, u),
                cy=lerp(560.0, 550.0, u),
                zoom=lerp(1.65, 1.82, u),
                sun_screen=(0.76, 0.14),
                sun_strength=lerp(0.12, 0.22, u),
                dof_fg=0.0,
            )
        elif lt < 6.8:
            u = smootherstep((lt - 3.6) / 3.2)
            cam_a = CameraState(
                cx=960.0,
                cy=lerp(550.0, 485.0, u),
                zoom=lerp(1.82, 1.32, u),
                sun_screen=(0.76, 0.16),
                sun_strength=lerp(0.22, 0.30, u),
                dof_fg=0.0,
            )
        else:
            u = smootherstep((lt - 6.8) / 4.6)
            cam_a = CameraState(
                cx=lerp(960.0, 985.0, u),
                cy=lerp(485.0, 355.0, u),
                zoom=lerp(1.32, 1.04, u),
                sun_screen=(0.76, 0.18),
                sun_strength=lerp(0.30, 0.42, u),
                dof_fg=0.0,
            )
        cam_a = cam_a.with_organic_motion(t, breath_amp=1.2)
        frame_a = cam_a.sample_layer(env["soil_cross_section"], parallax=1.0, out_w=out_w, out_h=out_h, world_origin=(-240.0, -160.0))

        layer = CairoLayer(out_w, out_h)
        ctx = layer.ctx
        cam_a.apply_to_cairo(ctx, parallax=1.0, out_w=out_w, out_h=out_h)

        # Biological germination timeline!
        absorb_u = smootherstep(inv_lerp(0.4, 3.6, lt))
        split_u = smootherstep(inv_lerp(2.8, 5.2, lt))
        root_u = smootherstep(inv_lerp(3.5, 9.8, lt))
        shoot_u = smootherstep(inv_lerp(4.4, 9.6, lt))

        draw_germinating_seed(
            ctx,
            cx=seed_x,
            cy=seed_y,
            soil_surface_y=soil_surf_y,
            absorb_u=absorb_u,
            split_u=split_u,
            root_u=root_u,
            shoot_u=shoot_u,
            t=t,
            scale=1.05,
        )

        # Individual emerging grass blades unfolding around the first sprout after shoot breaks soil!
        if lt > 6.6:
            tuft_specs = [
                (-16, 215, 16, -42, 0.15, 6.9, 0.72),
                ( 14, 240, 17,  46, 0.45, 7.1, 0.65),
                (-38, 175, 14, -68, 0.65, 7.5, None),
                ( 40, 190, 15,  72, 0.30, 7.7, None),
                (-74, 155, 13, -55, 0.80, 8.1, None),
                ( 78, 165, 13,  62, 0.25, 8.3, 0.62),
                (-125, 135, 12, -38, 0.50, 8.6, None),
                ( 130, 145, 12,  48, 0.60, 8.7, None),
            ]
            for dx, bh, bw, lean, cmix, t_start, dew in tuft_specs:
                gu = smootherstep(inv_lerp(t_start, t_start + 2.8, lt))
                if gu > 0.01:
                    bx = seed_x + dx
                    wval = wind_field(bx, t, strength=0.85)
                    draw_grass_blade(
                        ctx, bx, soil_surf_y + 12.0, bh, bw, lean, wval,
                        grow_u=gu, color_mix=cmix, dew_s=dew,
                    )

        layer.composite_over(frame_a, opacity=1.0)
        apply_cinematic_post(frame_a, t, sun_norm=cam_a.sun_screen, sun_strength=cam_a.sun_strength, bloom_strength=0.18)

    # Part B: Sweeping early-morning grassland reveal (active lt >= 8.9s)
    frame_b: Optional[np.ndarray] = None
    if blend_to_meadow > 0.005:
        u_reveal = smootherstep(inv_lerp(9.2, 22.0, lt))
        cam_b = CameraState(
            cx=lerp(920.0, 1080.0, u_reveal),
            cy=lerp(665.0, 525.0, ease_out_cubic(u_reveal)),
            zoom=lerp(1.42, 1.00, ease_out_cubic(u_reveal)),
            sun_screen=(0.76, 0.22),
            sun_strength=lerp(0.32, 0.44, u_reveal),
            dof_fg=lerp(0.30, 0.52, u_reveal),
        ).with_organic_motion(t, breath_amp=1.5)

        # Multi-plane parallax ecosystem layers
        frame_b = cam_b.sample_layer(env["morning_sky"], parallax=0.22, out_w=out_w, out_h=out_h, world_origin=(-240.0, -135.0))
        val_w = cam_b.sample_layer(env["distant_valley"], parallax=0.52, out_w=out_w, out_h=out_h, world_origin=(-240.0, -135.0), border_mode=cv2.BORDER_CONSTANT)
        composite_straight_bgra(frame_b, val_w, opacity=1.0)
        gnd_w = cam_b.sample_layer(env["grassland_ground"], parallax=0.84, out_w=out_w, out_h=out_h, world_origin=(-240.0, -135.0), border_mode=cv2.BORDER_CONSTANT)
        composite_straight_bgra(frame_b, gnd_w, opacity=1.0)

        # Distant flying birds across the golden sky + live animated swaying individual grass blades!
        layer_b = CairoLayer(out_w, out_h)
        ctx = layer_b.ctx

        ctx.save()
        cam_b.apply_to_cairo(ctx, parallax=0.35, out_w=out_w, out_h=out_h)
        _draw_distant_bird_flock(ctx, lt - 9.0, base_x=380.0, base_y=275.0, count=7)
        ctx.restore()

        ctx.save()
        cam_b.apply_to_cairo(ctx, parallax=1.0, out_w=out_w, out_h=out_h)
        for by, bx, bh, bw, lean, cmix, delay, dew, depth in _SCENE1_MEADOW_BLADES:
            # Central tuft starts already grown; surrounding meadow sprouts organically outward!
            dist_center = abs(bx - 960.0) / 1100.0
            t_start = 8.4 + dist_center * 3.8 + delay * 1.4
            gu = smootherstep(inv_lerp(t_start, t_start + 3.2, lt))
            if gu <= 0.01:
                continue
            wval = wind_field(bx, t, strength=lerp(0.65, 1.15, depth))
            draw_grass_blade(
                ctx, bx, by, bh, bw, lean, wval,
                grow_u=gu, color_mix=cmix, dew_s=dew,
            )
        _draw_pollen_and_insects(ctx, t, count=44, y_min=260.0, y_max=960.0, warm_alpha=0.68, seed=111)
        ctx.restore()

        layer_b.composite_over(frame_b, opacity=1.0)
        _draw_foreground_dof_blades(frame_b, cam_b, t, out_w, out_h, blade_count=14, seed=190)
        apply_cinematic_post(frame_b, t, sun_norm=cam_b.sun_screen, sun_strength=cam_b.sun_strength, bloom_strength=0.25)

    if frame_a is None:
        out = frame_b
    elif frame_b is None:
        out = frame_a
    else:
        out = cv2.addWeighted(frame_a, 1.0 - blend_to_meadow, frame_b, blend_to_meadow, 0.0)

    # Gentle warm fade-in from soft dark soil dawn on first 0.9s of the film
    if lt < 0.9:
        fade = smootherstep(lt / 0.9)
        out = np.clip(out.astype(np.float32) * fade, 0, 255).astype(np.uint8)
    return out


# =============================================================================
# SCENE 2: THE GRASSHOPPER (00:22 - 00:42)
# =============================================================================

def render_scene2(
    t: float,
    lt: float,
    env: Dict[str, np.ndarray],
    out_w: int = WIDTH,
    out_h: int = HEIGHT,
) -> np.ndarray:
    """
    SCENE 2: THE GRASSHOPPER (00:22 - 00:42, lt in [0..20])
      - 0.0 -> 4.6s: Camera tracks across sunlit grassland; grasshopper enters from the distance
        walking & hopping gracefully through stalks (never popping in!).
      - 4.6 -> 9.2s: Low-angle wildlife shot as grasshopper climbs onto a fresh green hero blade
        which bends realistically under its weight!
      - 9.2 -> 15.8s: Macro close-up of feeding: articulated mandibles chew, visible crescent
        bite notches progressively disappear from the blade with tiny chewed leaf flecks!
      - 15.8 -> 20.0s: Smooth pull-back to wider meadow shot gliding toward the pond reeds.
    """
    # Hero feeding grass blade anchored at (1040, 995) arching gracefully across sunlit sky
    hero_bx, hero_by = 1040.0, 995.0
    hero_h, hero_w, hero_lean = 565.0, 42.0, 125.0

    # Grasshopper choreography across 20s!
    # Compute visible downward weight-bend as the grasshopper climbs & perches along the blade!
    climb_u = smootherstep(inv_lerp(4.6, 8.8, lt))
    on_blade = smootherstep(inv_lerp(4.4, 5.2, lt))
    bend_load = 0.42 * on_blade * lerp(0.35, 1.0, climb_u)
    w_hero = wind_field(hero_bx, t, strength=0.68)

    # Progressive real bite notches carved into the blade during feeding (lt = 9.4 -> 15.8s)!
    bite_specs = [
        (0.66, smootherstep(inv_lerp(9.5, 10.7, lt)) * 0.78, 0.050),
        (0.61, smootherstep(inv_lerp(10.8, 12.1, lt)) * 0.84, 0.054),
        (0.56, smootherstep(inv_lerp(12.2, 13.6, lt)) * 0.88, 0.056),
        (0.51, smootherstep(inv_lerp(13.8, 15.3, lt)) * 0.82, 0.052),
    ]
    active_bites = [b for b in bite_specs if b[1] > 0.01]

    # Determine Grasshopper exact world pose (x, y, scale, angle, walk, hop, chew, head_nod)
    chew_phase = 0.0
    head_nod = 0.0
    hop_extend = 0.0
    wing_flutter = 0.0
    walk_phase = 0.0

    if lt < 2.5:
        u = lt / 2.5
        gx = lerp(290.0, 600.0, ease_in_out_cubic(u))
        gy = lerp(805.0, 865.0, u)
        g_scale = lerp(0.46, 0.72, u)
        g_ang = 0.05
        walk_phase = lt * 10.5
    elif lt < 3.45:
        u = (lt - 2.5) / 0.95
        gx, gy = parabolic_arc((600.0, 865.0), (885.0, 915.0), apex_height=165.0, u=u)
        g_scale = lerp(0.72, 1.04, u)
        g_ang = lerp(-0.34, 0.24, u)
        hop_extend = pulse_bell(u, 0.0, 0.72, sharp=0.8)
        wing_flutter = pulse_bell(u, 0.12, 0.82, sharp=1.0) * 0.85
    elif lt < 4.65:
        u = (lt - 3.45) / 1.20
        bx0, by0, ang0 = eval_grass_blade(hero_bx, hero_by, hero_h, hero_lean, w_hero, 1.0, 0.14, bend_load)
        gx = lerp(885.0, bx0 - 40.0, ease_in_out_cubic(u))
        gy = lerp(915.0, by0 - 16.0, ease_in_out_cubic(u))
        g_scale = lerp(1.04, 1.18, u)
        g_ang = lerp(0.04, ang0 + 0.12, u)
        walk_phase = lt * 12.0
    else:
        if lt < 8.8:
            s_pos = lerp(0.14, 0.56, climb_u)
            walk_phase = lt * 11.0 * (1.0 - smoothstep(inv_lerp(8.2, 8.8, lt)))
        else:
            eat_slide = smootherstep(inv_lerp(9.5, 15.5, lt))
            s_pos = lerp(0.56, 0.45, eat_slide)
            walk_phase = eat_slide * 5.0

        bx, by, b_ang = eval_grass_blade(hero_bx, hero_by, hero_h, hero_lean, w_hero, 1.0, s_pos, bend_load)
        nx, ny = -math.sin(b_ang), math.cos(b_ang)
        gx = bx - nx * 40.0
        gy = by - ny * 40.0
        g_scale = 1.22
        g_ang = b_ang + 0.15

        if 9.0 <= lt <= 16.0:
            feed_env = smootherstep(inv_lerp(9.0, 9.6, lt)) * (1.0 - smootherstep(inv_lerp(15.3, 16.0, lt)))
            chew_phase = (lt - 9.0) * 3.1 * feed_env
            head_nod = (0.32 + 0.10 * math.sin(lt * 19.0)) * feed_env

    # Camera choreography: low-angle tracking -> macro feeding close-up -> wider meadow glide
    if lt < 5.0:
        u = smootherstep(lt / 5.0)
        cam = CameraState(
            cx=lerp(880.0, 1030.0, u),
            cy=lerp(565.0, 645.0, u),
            zoom=lerp(1.04, 1.24, u),
            sun_screen=(0.78, 0.20),
            sun_strength=0.38,
            dof_fg=0.50,
        )
    elif lt < 9.5:
        u = smootherstep((lt - 5.0) / 4.5)
        cam = CameraState(
            cx=lerp(1030.0, gx + 48.0, u),
            cy=lerp(645.0, gy - 20.0, u),
            zoom=lerp(1.24, 1.88, u),
            sun_screen=(0.80, 0.18),
            sun_strength=0.32,
            dof_fg=0.62,
        )
    elif lt < 15.8:
        # Intimate macro close-up locked smoothly onto the feeding head & notched blade edge!
        cam = CameraState(
            cx=gx + 46.0,
            cy=gy - 22.0,
            zoom=1.88 + 0.06 * math.sin((lt - 9.5) * 0.5),
            sun_screen=(0.80, 0.18),
            sun_strength=0.30,
            dof_fg=0.68,
        )
    else:
        u = smootherstep((lt - 15.8) / 4.2)
        cam = CameraState(
            cx=lerp(gx + 46.0, 1180.0, u),
            cy=lerp(gy - 22.0, 560.0, u),
            zoom=lerp(1.88, 1.04, u),
            sun_screen=(0.76, 0.20),
            sun_strength=0.38,
            dof_fg=0.48,
        )
    cam = cam.with_organic_motion(t, breath_amp=1.3)

    # Background ecosystem layers (identical palette & hills as Scene 1!)
    frame = cam.sample_layer(env["morning_sky"], parallax=0.22, out_w=out_w, out_h=out_h, world_origin=(-240.0, -135.0))
    val_w = cam.sample_layer(env["distant_valley"], parallax=0.52, out_w=out_w, out_h=out_h, world_origin=(-240.0, -135.0), border_mode=cv2.BORDER_CONSTANT)
    composite_straight_bgra(frame, val_w, opacity=1.0)
    gnd_w = cam.sample_layer(env["grassland_ground"], parallax=0.84, out_w=out_w, out_h=out_h, world_origin=(-240.0, -135.0), border_mode=cv2.BORDER_CONSTANT)
    composite_straight_bgra(frame, gnd_w, opacity=1.0)

    layer = CairoLayer(out_w, out_h)
    ctx = layer.ctx
    cam.apply_to_cairo(ctx, parallax=1.0, out_w=out_w, out_h=out_h)

    # Back-layer meadow grass blades (behind grasshopper & hero stalk)
    for by, bx, bh, bw, lean, cmix, delay, dew, depth in _SCENE2_MEADOW_BLADES:
        if by > 965.0:
            continue
        if abs(bx - hero_bx) < 75.0 and by > 860.0:
            continue
        wval = wind_field(bx, t, strength=lerp(0.6, 1.05, depth))
        draw_grass_blade(ctx, bx, by, bh, bw, lean, wval, grow_u=1.0, color_mix=cmix, dew_s=dew)

    # HERO BENDING & BITTEN GRASS BLADE!
    draw_grass_blade(
        ctx,
        base_x=hero_bx,
        base_y=hero_by,
        height=hero_h,
        base_width=hero_w,
        base_lean=hero_lean,
        wind_val=w_hero,
        grow_u=1.0,
        color_mix=0.35,
        bend_load=bend_load,
        bite_notches=active_bites,
        dew_s=0.84,
        steps=32,
    )

    # HERO ARTICULATED GRASSHOPPER!
    draw_grasshopper(
        ctx,
        x=gx,
        y=gy,
        scale=g_scale,
        facing=1.0,
        body_angle=g_ang,
        t=t,
        walk_phase=walk_phase,
        hop_extend=hop_extend,
        chew_phase=chew_phase,
        head_nod=head_nod,
        wing_flutter=wing_flutter,
        draw_shadow=(lt < 4.5),
    )

    # Tiny fresh green bitten leaf flecks drifting down during feeding bites!
    if 9.4 <= lt <= 15.8:
        for k in range(10):
            bt = (lt - 9.4 - k * 0.58) % 1.35
            if 0.0 < bt < 0.95:
                fx = gx + 65.0 + math.sin(bt * 7.0 + k * 2.1) * 18.0
                fy = gy + 24.0 + bt * 110.0
                a_f = (1.0 - bt / 0.95) * 0.85
                ctx.save()
                ctx.translate(fx, fy)
                ctx.rotate(bt * 5.0 + k)
                ctx.rectangle(-3.5, -2.0, 7.0, 4.0)
                ctx.set_source_rgba(0.46, 0.72, 0.22, a_f)
                ctx.fill()
                ctx.restore()

    # Foreground sharp grass stalks framing lower edges
    for by, bx, bh, bw, lean, cmix, delay, dew, depth in _SCENE2_MEADOW_BLADES:
        if by <= 965.0:
            continue
        if abs(bx - hero_bx) < 260.0:
            continue
        wval = wind_field(bx, t, strength=1.15)
        draw_grass_blade(ctx, bx, by, bh, bw, lean, wval, grow_u=1.0, color_mix=cmix)

    _draw_pollen_and_insects(ctx, t, count=34, y_min=200.0, y_max=960.0, warm_alpha=0.58, seed=222)
    layer.composite_over(frame, opacity=1.0)
    _draw_foreground_dof_blades(frame, cam, t, out_w, out_h, blade_count=12, seed=290)
    apply_cinematic_post(frame, t, sun_norm=cam.sun_screen, sun_strength=cam.sun_strength, bloom_strength=0.22)
    return frame


# =============================================================================
# SCENE 3: THE FROG HUNTS (00:42 - 01:02)
# =============================================================================

def render_scene3(
    t: float,
    lt: float,
    env: Dict[str, np.ndarray],
    out_w: int = WIDTH,
    out_h: int = HEIGHT,
) -> np.ndarray:
    """
    SCENE 3: THE FROG HUNTS (00:42 - 01:02, lt in [0..20])
      - 0.0 -> 5.2s: Tranquil freshwater pond, animated water reflections, wet boulders,
        reeds, dragonflies, green frog breathing, throat pulsing, eyes blinking on wet rock.
      - 5.2 -> 9.2s: Grasshopper hops onto a reed arching near the pond; frog spots it!
      - 9.2 -> 11.7s: Tense wildlife macro close-up of the frog's metallic golden eye locking on!
      - 11.7 -> 16.5s: Frog crouches low, springs in realistic parabolic leap, fires rapid
        fleshy tongue catching the grasshopper cleanly, lands with water splash ripples!
      - 16.5 -> 20.0s: Frog visibly swallows prey (throat bulge + amphibian eye retraction!).
    """
    # Eye macro insert window (9.2 -> 11.7s with sleek 0.22s cinematic crossfade)
    eye_in = smootherstep(inv_lerp(9.15, 9.42, lt))
    eye_out = 1.0 - smootherstep(inv_lerp(11.45, 11.75, lt))
    eye_mix = eye_in * eye_out

    # Render main pond shot whenever visible
    frame_pond: Optional[np.ndarray] = None
    if eye_mix < 0.995:
        # Frog & Grasshopper world coordinates aligned precisely onto our painted wet boulders in pond_fg!
        # pond_bg/pond_fg (2400x1350) placed with world_origin=(-240, -135) ->
        # Left frog perch rock top is at world (600, 748); right landing rock top at world (1255, 738);
        # waterline in world space is y = 688!
        frog_start = (610.0, 725.0)
        frog_land = (1190.0, 718.0)

        # Target reed stalk arching near right rock where the grasshopper perches
        reed_bx, reed_by = 1495.0, 920.0
        reed_h, reed_lean = 375.0, -145.0
        w_reed = wind_field(reed_bx, t, strength=0.65)
        rx_tip, ry_tip, r_ang = eval_grass_blade(reed_bx, reed_by, reed_h, reed_lean, w_reed, 1.0, 0.72, -0.15)
        gh_perch = (rx_tip - 8.0, ry_tip - 24.0)

        # Frog kinematic state across Scene 3
        crouch = 0.0
        leap_stretch = 0.0
        body_pitch = -0.06
        mouth_open = 0.0
        tongue_ext = 0.0
        throat_swell = pulse_bell(lt, 2.6, 3.6, sharp=1.2) * 0.85 + pulse_bell(lt, 4.1, 4.9, sharp=1.2) * 0.65
        swallow_u = 0.0
        blink = blink_envelope(t, period=3.4, offset=0.5, dur=0.18)

        if lt < 11.7:
            fx, fy = frog_start
            # Slight alert head lean after spotting grasshopper (lt > 6.8s)
            crouch = smootherstep(inv_lerp(7.2, 9.2, lt)) * 0.35
        elif lt < 13.25:
            # Deep predator crouch coiling hind legs tightly before launch!
            fx, fy = frog_start
            crouch = lerp(0.35, 1.0, smootherstep(inv_lerp(11.7, 13.15, lt)))
            body_pitch = lerp(-0.06, -0.18, crouch)
        elif lt < 14.35:
            # Explosive realistic parabolic jump from left boulder to right boulder!
            ju = (lt - 13.25) / 1.10
            fx, fy = parabolic_arc(frog_start, frog_land, apex_height=195.0, u=ju)
            leap_stretch = math.sin(clamp(ju) * math.pi) ** 0.65
            crouch = 0.0
            body_pitch = lerp(-0.42, 0.28, ju)
            # Mid-leap tongue strike (lt = 13.68 -> 14.22s)!
            if 13.62 <= lt <= 14.28:
                mouth_open = pulse_bell(lt, 13.62, 14.28, sharp=0.65)
                if lt < 13.92:
                    tongue_ext = ease_out_quad(inv_lerp(13.68, 13.92, lt))
                else:
                    tongue_ext = 1.0 - ease_in_quad(inv_lerp(13.92, 14.22, lt))
        else:
            # Landed cleanly on right wet boulder; subtle landing compression then natural swallowing!
            fx, fy = frog_land
            land_squash = pulse_bell(lt, 14.35, 14.85, sharp=1.1) * 0.55
            crouch = land_squash
            body_pitch = -0.05
            # Two visible authentic swallowing gulps (with frog eye depression!)
            if 15.1 <= lt <= 16.5:
                swallow_u = (lt - 15.1) / 1.4
            elif 16.9 <= lt <= 18.1:
                swallow_u = (lt - 16.9) / 1.2

        # Grasshopper state (hops onto reed at lt=5.2 -> 6.4s, sits until caught at lt=13.92s!)
        gh_visible = True
        gh_scale = 0.72
        gh_hop = 0.0
        gh_ang = 0.12
        if lt < 5.0:
            gh_visible = False
            gh_x, gh_y = gh_perch[0] + 240.0, gh_perch[1] - 40.0
        elif lt < 6.15:
            hu = (lt - 5.0) / 1.15
            gh_x, gh_y = parabolic_arc((gh_perch[0] + 240.0, gh_perch[1] - 20.0), gh_perch, apex_height=115.0, u=hu)
            gh_hop = pulse_bell(hu, 0.0, 0.75)
            gh_ang = lerp(-0.35, 0.12, hu)
        elif lt < 13.92:
            gh_x, gh_y = gh_perch
        elif lt < 14.22:
            # Snagged cleanly on the retracting tongue tip — pulled right into the frog's mouth!
            gh_x, gh_y = gh_perch  # overridden immediately by actual tongue tip below!
            gh_ang = (lt - 13.92) * 5.0
        else:
            gh_visible = False
            gh_x, gh_y = frog_land

        # Camera choreography across the pond
        if lt < 5.2:
            u = smootherstep(lt / 5.2)
            cam = CameraState(
                cx=lerp(760.0, 880.0, u),
                cy=lerp(540.0, 565.0, u),
                zoom=lerp(1.04, 1.12, u),
                sun_screen=(0.74, 0.18),
                sun_strength=0.34,
                dof_fg=0.42,
            )
        elif lt < 9.2:
            u = smootherstep((lt - 5.2) / 4.0)
            cam = CameraState(
                cx=lerp(880.0, 995.0, u),
                cy=575.0,
                zoom=lerp(1.12, 1.06, u),
                sun_screen=(0.74, 0.18),
                sun_strength=0.34,
                dof_fg=0.40,
            )
        elif lt < 15.2:
            u = smootherstep(inv_lerp(12.2, 14.8, lt))
            cam = CameraState(
                cx=lerp(945.0, 1140.0, u),
                cy=lerp(590.0, 580.0, u),
                zoom=lerp(1.16, 1.24, u),
                shake_amp=pulse_bell(lt, 14.32, 14.62) * 3.8,
                sun_screen=(0.74, 0.18),
                sun_strength=0.34,
                dof_fg=0.44,
            )
        else:
            u = smootherstep((lt - 15.2) / 4.8)
            cam = CameraState(
                cx=lerp(1140.0, 1230.0, u),
                cy=lerp(580.0, 565.0, u),
                zoom=lerp(1.24, 1.38, u),
                sun_screen=(0.74, 0.18),
                sun_strength=0.32,
                dof_fg=0.45,
            )
        cam = cam.with_organic_motion(t, breath_amp=1.3)

        # 1. Sample painted pond background (sky, hills, far cattails & shoreline)
        frame_pond = cam.sample_layer(env["pond_bg"], parallax=0.92, out_w=out_w, out_h=out_h, world_origin=(-240.0, -135.0))

        # 2. Compute screen-space waterline & live splash ripples for real-time water reflection shader!
        _, wy_px = cam.world_to_screen(960.0, 688.0, parallax=0.92, out_w=out_w, out_h=out_h)
        ripples = []
        # Gentle periodic ambient insect/water-drop ripples across pond
        for r_i in range(3):
            age = (t + r_i * 1.15) % 3.2
            rx_w = 560.0 + r_i * 420.0
            ry_w = 815.0 + (r_i % 2) * 85.0
            sx_r, sy_r = cam.world_to_screen(rx_w, ry_w, parallax=1.0, out_w=out_w, out_h=out_h)
            ripples.append((sx_r, sy_r, age, 0.45))
        # Takeoff ripple off left rock (lt >= 13.25s) & landing splash ripple at right rock (lt >= 14.35s)!
        if lt > 13.25:
            sx_l, sy_l = cam.world_to_screen(frog_start[0] + 60, 820.0, parallax=1.0, out_w=out_w, out_h=out_h)
            ripples.append((sx_l, sy_l, lt - 13.25, 1.25))
        if lt > 14.35:
            sx_r, sy_r = cam.world_to_screen(frog_land[0] - 25, 815.0, parallax=1.0, out_w=out_w, out_h=out_h)
            ripples.append((sx_r, sy_r, lt - 14.35, 1.55))

        apply_pond_water_reflection(frame_pond, int(wy_px), t=t, ripples=ripples)

        # 3. Composite wet mossy boulders & floating lily pads cleanly OVER the reflective pond surface!
        p_fg = cam.sample_layer(env["pond_fg"], parallax=1.0, out_w=out_w, out_h=out_h, world_origin=(-240.0, -135.0), border_mode=cv2.BORDER_CONSTANT)
        composite_straight_bgra(frame_pond, p_fg, opacity=1.0)

        # 4. Draw live swaying pond reeds, dragonflies, hero frog, prey grasshopper & water splash droplets!
        layer = CairoLayer(out_w, out_h)
        ctx = layer.ctx
        cam.apply_to_cairo(ctx, parallax=1.0, out_w=out_w, out_h=out_h)

        # Swaying shoreline grass/reeds near right rock (including grasshopper's perch reed!)
        for r_idx, (rbx, rby, rbh, rbw, rlean) in enumerate([
            (1560, 960, 420, 20, -90),
            (1620, 980, 450, 22, -55),
            (1440, 940, 310, 17, -110),
            (reed_bx, reed_by, reed_h, 21, reed_lean),
            (280, 970, 390, 20, 85),
            (340, 990, 340, 18, 120),
        ]):
            wv = wind_field(rbx, t, strength=0.70)
            bl = -0.15 if (r_idx == 3 and 6.1 <= lt < 13.92) else 0.0
            draw_grass_blade(ctx, rbx, rby, rbh, rbw, rlean, wv, grow_u=1.0, color_mix=0.25 + (r_idx % 3) * 0.2, bend_load=bl)

        # Skimming iridescent pond dragonflies
        df1_x = 480.0 + math.sin(t * 0.72) * 260.0
        df1_y = 535.0 + math.sin(t * 2.1) * 32.0
        _draw_dragonfly(ctx, df1_x, df1_y, t, scale=1.05, facing=1.0 if math.cos(t * 0.72) >= 0 else -1.0)
        df2_x = 1280.0 + math.cos(t * 0.58 + 1.4) * 220.0
        df2_y = 485.0 + math.sin(t * 2.4 + 1.0) * 24.0
        _draw_dragonfly(ctx, df2_x, df2_y, t + 0.7, scale=0.82, facing=-1.0 if math.sin(t * 0.58 + 1.4) >= 0 else 1.0)

        # Draw HERO FROG (returns exact world tongue tip coordinate!)
        tip_world = draw_frog(
            ctx,
            x=fx,
            y=fy,
            scale=1.32,
            facing=1.0,
            body_pitch=body_pitch,
            t=t,
            crouch=crouch,
            leap_stretch=leap_stretch,
            mouth_open=mouth_open,
            tongue_ext=tongue_ext,
            tongue_target_world=gh_perch,
            blink=blink,
            throat_swell=throat_swell,
            swallow_u=swallow_u,
            draw_shadow=(leap_stretch < 0.35),
        )

        # Draw PREY GRASSHOPPER (on reed before catch, or attached cleanly to retracting tongue tip!)
        if gh_visible:
            if lt >= 13.92:
                gh_x, gh_y = tip_world
                gh_scale = lerp(0.72, 0.38, inv_lerp(13.92, 14.22, lt))
            draw_grasshopper(
                ctx,
                x=gh_x,
                y=gh_y,
                scale=gh_scale,
                facing=-1.0,
                body_angle=gh_ang,
                t=t,
                hop_extend=gh_hop,
                wing_flutter=0.9 if lt >= 13.92 else 0.0,
                draw_shadow=False,
            )

        # Sparkling water splash droplets erupting on leap takeoff & rock landing!
        for ev_t, sx0, sy0 in [(13.28, frog_start[0] - 20, 765.0), (14.35, frog_land[0] - 10, 760.0)]:
            dt_s = lt - ev_t
            if 0.0 < dt_s < 0.78:
                for d_i in range(11):
                    vx = (hash1d(d_i * 31 + 7) - 0.5) * 210.0
                    vy0 = -(95.0 + 135.0 * hash1d(d_i * 43 + 3))
                    dx = sx0 + vx * dt_s
                    dy = sy0 + vy0 * dt_s + 360.0 * dt_s * dt_s
                    r_d = (3.5 + 3.2 * hash1d(d_i * 53)) * (1.0 - dt_s / 0.85)
                    ctx.arc(dx, dy, max(0.8, r_d), 0, math.tau)
                    ctx.set_source_rgba(0.88, 0.97, 0.98, 0.82 * (1.0 - dt_s / 0.78))
                    ctx.fill()

        _draw_pollen_and_insects(ctx, t, count=28, y_min=260.0, y_max=760.0, warm_alpha=0.50, seed=333)
        layer.composite_over(frame_pond, opacity=1.0)
        _draw_foreground_dof_blades(frame_pond, cam, t, out_w, out_h, blade_count=10, seed=390)
        apply_cinematic_post(frame_pond, t, sun_norm=cam.sun_screen, sun_strength=cam.sun_strength, bloom_strength=0.24)

    # Render hyper-detailed Frog Eye Close-Up insert when active!
    frame_eye: Optional[np.ndarray] = None
    if eye_mix > 0.005:
        frame_eye = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        elayer = CairoLayer(out_w, out_h)
        ectx = elayer.ctx
        u_eye = clamp((lt - 9.2) / 2.5)
        scale_r = out_w / float(WIDTH)
        # Subtle dramatic push-in on the frog's gleaming eye + single alert blink + tight focus!
        er = lerp(225.0, 260.0, smootherstep(u_eye)) * scale_r
        eblink = pulse_bell(lt, 9.65, 9.86, sharp=1.2)
        efocus = smootherstep(inv_lerp(10.0, 11.2, lt))
        draw_frog_eye_closeup(ectx, out_w * 0.50, out_h * 0.50, radius=er, blink=eblink, focus_u=efocus, t=t)
        elayer.composite_over(frame_eye, opacity=1.0)
        apply_cinematic_post(frame_eye, t, sun_norm=(0.68, 0.25), sun_strength=0.0, bloom_strength=0.14, vignette_strength=1.25)

    if frame_pond is None:
        return frame_eye
    if frame_eye is None:
        return frame_pond
    return cv2.addWeighted(frame_pond, 1.0 - eye_mix, frame_eye, eye_mix, 0.0)


# =============================================================================
# SCENE 4: THE SNAKE (01:02 - 01:22)
# =============================================================================

def render_scene4(
    t: float,
    lt: float,
    env: Dict[str, np.ndarray],
    out_w: int = WIDTH,
    out_h: int = HEIGHT,
) -> np.ndarray:
    """
    SCENE 4: THE SNAKE (01:02 - 01:22, lt in [0..20])
      - 0.0 -> 6.5s: Sun-dappled rocky meadow; scaled snake emerges smoothly from left boulders
        using realistic lateral undulation, independent head tracking, and flicking forked tongue.
      - 6.5 -> 12.1s: Frog sits calmly near weathered right rock; snake approaches stealthily,
        tension builds through camera framing as snake gathers into a poised S-coil.
      - 12.1 -> 14.0s: Frog notices snake and leaps desperately to escape — snake strikes swiftly
        and catches the frog cleanly (educational, non-graphic!).
      - 14.0 -> 20.0s: Natural non-graphic swallowing bulge glides softly down the snake's neck
        as it slithers out into the bright open sunny rocky clearing.
    """
    # Frog resting beside the right boulder (rocky_meadow world origin=(-240, -135) -> boulder at world 1450, 680)
    frog_rest = (1395.0, 775.0)

    # Snake choreography (head position, heading, slither_phase, coil_u, strike_u, tongue_flick, mouth_open, swallow)
    mouth_open = 0.0
    swallow_s = None
    swallow_amp = 0.0

    # Periodic natural forked-tongue flicks while exploring & stalking!
    tongue_flick = 0.0
    for tf_start in (1.6, 4.1, 6.8, 9.6, 17.5):
        tongue_flick = max(tongue_flick, pulse_bell(lt, tf_start, tf_start + 0.62, sharp=0.85))

    if lt < 9.2:
        # Smooth lateral-undulation slithering out from beneath left rock shelter across dry leaves!
        u = lt / 9.2
        hx = lerp(240.0, 1045.0, ease_out_quad(u))
        hy = lerp(855.0, 815.0, u) + math.sin(u * math.pi * 2.0) * 14.0
        slither_phase = lt * 4.6
        coil_u = 0.0
        strike_u = 0.0
        heading = -0.05
        head_ang = math.atan2(frog_rest[1] - hy, frog_rest[0] - hx) if lt > 5.8 else -0.04
    elif lt < 12.65:
        # Tense stealthy S-coil gathering as predator locks onto prey!
        u = smootherstep((lt - 9.2) / 3.45)
        hx = lerp(1045.0, 1145.0, u)
        hy = lerp(815.0, 808.0, u)
        slither_phase = 9.2 * 4.6 + u * 2.2
        coil_u = lerp(0.0, 0.94, u)
        strike_u = 0.0
        heading = -0.06
        head_ang = math.atan2(frog_rest[1] - hy, frog_rest[0] - hx)
    elif lt < 13.55:
        # LIGHTNING STRIKE LUNGE to catch the leaping frog cleanly!
        su = ease_out_cubic(inv_lerp(12.65, 13.02, lt))
        hx = lerp(1145.0, 1445.0, su)
        hy = lerp(808.0, 725.0, su)
        slither_phase = 9.2 * 4.6 + 2.2
        coil_u = lerp(0.94, 0.35, su)
        strike_u = su
        heading = -0.26
        head_ang = -0.26
        mouth_open = pulse_bell(lt, 12.68, 13.38, sharp=0.55)
    else:
        # Calm non-graphic educational swallow + gliding out into sunny open clearing!
        u = smootherstep((lt - 13.55) / 6.45)
        hx = lerp(1445.0, 1615.0, u)
        hy = lerp(725.0, 820.0, ease_in_out_cubic(u))
        slither_phase = 9.2 * 4.6 + 2.2 + (lt - 13.55) * 2.6
        coil_u = lerp(0.35, 0.0, smootherstep(inv_lerp(13.55, 16.0, lt)))
        strike_u = lerp(1.0, 0.0, smootherstep(inv_lerp(13.55, 15.2, lt)))
        heading = 0.10
        head_ang = 0.08
        # Smooth gentle swallowing wave traveling down upper neck
        if lt < 18.8:
            su_w = inv_lerp(13.6, 18.4, lt)
            swallow_s = lerp(0.06, 0.36, su_w)
            swallow_amp = math.sin(clamp(su_w) * math.pi) ** 0.55

    spine = compute_snake_spine(
        head_xy=(hx, hy),
        heading_rad=heading,
        total_len=560.0,
        n_segs=58,
        slither_phase=slither_phase,
        wave_amp=36.0,
        coil_u=coil_u,
        strike_u=strike_u,
    )

    # Frog state: peaceful resting -> alert panic crouch -> desperate escape leap -> caught cleanly!
    frog_visible = True
    frog_crouch = 0.0
    frog_leap = 0.0
    frog_pitch = -0.05
    frog_facing = -1.0  # facing left toward approaching threat
    if lt < 11.4:
        fx, fy = frog_rest
    elif lt < 12.48:
        # Notices snake — freezes & coils urgently to escape!
        fx, fy = frog_rest
        frog_crouch = smootherstep((lt - 11.4) / 1.08)
        frog_facing = 1.0 if lt > 12.0 else -1.0  # spins right to jump away!
    elif lt < 13.05:
        # Desperate escape jump right as snake strikes!
        ju = (lt - 12.48) / 0.65
        fx, fy = parabolic_arc(frog_rest, (1580.0, 745.0), apex_height=135.0, u=ju)
        frog_leap = clamp(ju * 1.6)
        frog_facing = 1.0
        frog_pitch = -0.35
    elif lt < 13.45:
        # Caught smoothly (non-graphic — quickly tucked as snake jaws close)
        fx, fy = (hx + 16.0, hy + 2.0)
        frog_leap = 0.3
        frog_facing = 1.0
    else:
        frog_visible = False
        fx, fy = (hx, hy)

    # Tension-building camera choreography
    if lt < 6.4:
        u = smootherstep(lt / 6.4)
        cam = CameraState(
            cx=lerp(720.0, 1020.0, u),
            cy=lerp(665.0, 645.0, u),
            zoom=lerp(1.24, 1.12, u),
            sun_screen=(0.74, 0.16),
            sun_strength=0.26,
            dof_fg=0.46,
        )
    elif lt < 12.5:
        # Tightening predator-prey tension two-shot framing both coiled snake & unsuspecting frog!
        u = smootherstep((lt - 6.4) / 6.1)
        cam = CameraState(
            cx=lerp(1020.0, 1185.0, u),
            cy=lerp(645.0, 675.0, u),
            zoom=lerp(1.12, 1.32, u),
            sun_screen=(0.74, 0.16),
            sun_strength=0.24,
            dof_fg=0.52,
        )
    else:
        u = smootherstep((lt - 12.5) / 7.5)
        cam = CameraState(
            cx=lerp(1185.0, 1340.0, u),
            cy=lerp(675.0, 610.0, u),
            zoom=lerp(1.32, 1.06, u),
            shake_amp=pulse_bell(lt, 12.75, 13.25) * 4.5,
            sun_screen=(0.74, 0.16),
            sun_strength=lerp(0.24, 0.36, u),
            dof_fg=0.44,
        )
    cam = cam.with_organic_motion(t, breath_amp=1.3)

    frame = cam.sample_layer(env["rocky_meadow"], parallax=0.95, out_w=out_w, out_h=out_h, world_origin=(-240.0, -135.0))

    layer = CairoLayer(out_w, out_h)
    ctx = layer.ctx
    cam.apply_to_cairo(ctx, parallax=1.0, out_w=out_w, out_h=out_h)

    # Swaying background dry grass tufts around clearing edges
    for i in range(18):
        gx_b = 140.0 + i * 105.0
        if 480.0 < gx_b < 1560.0:
            continue
        gy_b = 760.0 + (i % 3) * 48.0
        wv = wind_field(gx_b, t, strength=0.75)
        draw_grass_blade(ctx, gx_b, gy_b, 155.0 + (i % 4) * 32.0, 14.0, (i % 5 - 2) * 24.0, wv, color_mix=0.82)

    # Frog drawn before snake during catch so snake's jaw overlaps cleanly!
    if frog_visible:
        f_alpha = 1.0 if lt < 13.05 else 1.0 - inv_lerp(13.05, 13.42, lt)
        f_sc = 1.12 if lt < 13.05 else lerp(1.12, 0.52, inv_lerp(13.05, 13.42, lt))
        draw_frog(
            ctx,
            x=fx,
            y=fy,
            scale=f_sc,
            facing=frog_facing,
            body_pitch=frog_pitch,
            t=t,
            crouch=frog_crouch,
            leap_stretch=frog_leap,
            blink=blink_envelope(t, period=3.1, offset=1.2),
            draw_shadow=(frog_leap < 0.3 and lt < 13.05),
            alpha=f_alpha,
        )

    # HERO SCALED UNDULATING SNAKE!
    draw_snake(
        ctx,
        spine_pts=spine,
        max_radius=21.0,
        head_angle=head_ang,
        mouth_open=mouth_open,
        tongue_flick=tongue_flick,
        swallow_bulge_s=swallow_s,
        swallow_bulge_amp=swallow_amp,
        draw_shadow=True,
        t=t,
    )

    # Subtle rustled dry leaves lifting gently near strike moment
    if 12.65 <= lt <= 14.4:
        dt_l = lt - 12.65
        for k in range(6):
            lx = 1220.0 + k * 42.0 + dt_l * (35.0 + k * 14.0)
            ly = 825.0 - math.sin(clamp(dt_l / 1.4) * math.pi) * (35.0 + (k % 3) * 18.0)
            ctx.save()
            ctx.translate(lx, ly)
            ctx.rotate(dt_l * 3.5 + k)
            ctx.scale(14.0, 5.5)
            ctx.arc(0, 0, 1.0, 0, math.tau)
            ctx.set_source_rgba(0.62, 0.38, 0.16, 0.85 * (1.0 - dt_l / 1.75))
            ctx.fill()
            ctx.restore()

    _draw_pollen_and_insects(ctx, t, count=22, y_min=260.0, y_max=880.0, warm_alpha=0.45, seed=444)
    layer.composite_over(frame, opacity=1.0)
    _draw_foreground_dof_blades(frame, cam, t, out_w, out_h, blade_count=10, seed=490, color_tint="dry")
    apply_cinematic_post(frame, t, sun_norm=cam.sun_screen, sun_strength=cam.sun_strength, bloom_strength=0.18)
    return frame


# =============================================================================
# SCENE 5: THE HAWK (01:22 - 01:47)
# =============================================================================

def render_scene5(
    t: float,
    lt: float,
    env: Dict[str, np.ndarray],
    out_w: int = WIDTH,
    out_h: int = HEIGHT,
) -> np.ndarray:
    """
    SCENE 5: THE HAWK (01:22 - 01:47, lt in [0..25])
      - 0.0 -> 6.8s: Camera tilts up into vast panoramic sunlit sky & grassland valley;
        Red-tailed Hawk wheels gracefully across high thermals with articulated feathered wings!
      - 6.8 -> 9.0s: High-angle aerial glimpse of the snake gliding across the sunlit meadow below.
      - 9.0 -> 11.4s: Dramatic macro close-up of the hawk's fierce piercing golden eye locking on!
      - 11.4 -> 17.2s: Hawk banks, folds wings into an aerodynamic stoop & dives steeply toward
        the grassland with dynamic tracking camera & rushing wind streaks!
      - 17.2 -> 20.8s: Low-level meadow swoop: hawk flares broad wings, extends scaly yellow
        talons, cleanly catches the snake (non-graphic!), and powers upward!
      - 20.8 -> 25.0s: Sweeping wide cinematic vista of the majestic hawk soaring across the
        sunlit valley landscape clutching its prey!
    """
    # Shot 5B: Aerial view of snake below (6.8 -> 9.0s)
    aerial_mix = smootherstep(inv_lerp(6.65, 6.95, lt)) * (1.0 - smootherstep(inv_lerp(8.85, 9.15, lt)))
    # Shot 5C: Dramatic Hawk Eye Macro Close-up (9.0 -> 11.4s)
    eye_mix = smootherstep(inv_lerp(8.95, 9.22, lt)) * (1.0 - smootherstep(inv_lerp(11.20, 11.50, lt)))
    # Shot 5E: Low-level meadow snatch (17.2 -> 21.2s)
    ground_mix = smootherstep(inv_lerp(17.0, 17.55, lt)) * (1.0 - smootherstep(inv_lerp(20.65, 21.35, lt)))

    # 1. PRIMARY SKY PANORAMA SHOT (Soaring 0..6.9s, Diving 11.2..17.5s, Victorious Soar 20.7..25.0s)
    sky_needed = (1.0 - max(aerial_mix, eye_mix, ground_mix)) > 0.005
    frame_sky: Optional[np.ndarray] = None
    if sky_needed:
        # sky_panorama is 2600x1600 placed at world_origin=(-340, -260)
        if lt < 7.2:
            # Tilting up from valley horizon to majestic circling hawk in sunlit clouds!
            u = smootherstep(lt / 7.2)
            cam_s = CameraState(
                cx=lerp(880.0, 1080.0, u),
                cy=lerp(720.0, 395.0, ease_out_cubic(u)),
                zoom=lerp(1.06, 0.96, u),
                sun_screen=(0.76, 0.16),
                sun_strength=0.40,
                dof_fg=0.0,
            )
            # Sweeping circular soar across the wide sunlit sky
            hu = lt / 7.2
            hx = lerp(220.0, 1480.0, ease_in_out_cubic(hu))
            hy = 345.0 - math.sin(hu * math.pi) * 95.0
            h_scale = lerp(0.82, 1.24, math.sin(hu * math.pi * 0.85))
            h_pitch = -0.10 + math.cos(hu * math.pi) * 0.10
            flap_ph = lt * 3.8
            flap_amp = 0.24 + 0.22 * pulse_bell(lt, 1.4, 3.8)
            dive_fold = 0.0
            tail_spread = 0.88
            talon_ext = 0.0
            talon_grip = 0.0
            carry = False
        elif lt < 17.6:
            # THRILLING HIGH-SPEED STOOP DIVE!
            du = clamp((lt - 11.2) / 6.3)
            de = ease_in_quad(du)
            cam_s = CameraState(
                cx=lerp(920.0, 1240.0, de),
                cy=lerp(360.0, 760.0, de),
                zoom=lerp(1.00, 1.24, du),
                roll_deg=lerp(0.0, 4.5, math.sin(du * math.pi)),
                shake_amp=1.8 * de,
                sun_screen=(0.76, 0.16),
                sun_strength=0.38,
                dof_fg=0.0,
            )
            hx = lerp(480.0, 1360.0, de)
            hy = lerp(260.0, 825.0, de)
            h_scale = lerp(1.02, 1.42, du)
            h_pitch = lerp(0.18, 0.68, ease_out_cubic(du))
            flap_ph = lt * 5.0
            flap_amp = lerp(0.30, 0.06, smootherstep(inv_lerp(0.0, 0.35, du)))
            dive_fold = smootherstep(inv_lerp(0.05, 0.42, du)) * 0.88
            tail_spread = lerp(0.75, 0.15, du)
            talon_ext = smootherstep(inv_lerp(0.76, 1.0, du)) * 0.85
            talon_grip = 0.0
            carry = False
        else:
            # WIDE CINEMATIC VICTORIOUS ASCENT WITH SNAKE CLUTCHED IN TALONS!
            vu = clamp((lt - 20.6) / 4.4)
            cam_s = CameraState(
                cx=lerp(1160.0, 960.0, smootherstep(vu)),
                cy=lerp(660.0, 460.0, smootherstep(vu)),
                zoom=lerp(1.12, 0.94, smootherstep(vu)),
                sun_screen=(0.76, 0.16),
                sun_strength=0.45,
                dof_fg=0.0,
            )
            hx = lerp(640.0, 1540.0, ease_out_quad(vu))
            hy = lerp(670.0, 315.0, ease_out_cubic(vu))
            h_scale = lerp(1.36, 0.92, vu)
            h_pitch = lerp(-0.26, -0.08, vu)
            flap_ph = lt * 6.2
            flap_amp = lerp(0.68, 0.28, vu)
            dive_fold = 0.0
            tail_spread = 0.82
            talon_ext = 0.92
            talon_grip = 1.0
            carry = True

        cam_s = cam_s.with_organic_motion(t, breath_amp=1.4)
        frame_sky = cam_s.sample_layer(env["sky_panorama"], parallax=0.85, out_w=out_w, out_h=out_h, world_origin=(-340.0, -260.0))

        slayer = CairoLayer(out_w, out_h)
        sctx = slayer.ctx
        cam_s.apply_to_cairo(sctx, parallax=1.0, out_w=out_w, out_h=out_h)

        # Distant flock of tiny birds soaring far below in valley
        _draw_distant_bird_flock(sctx, lt * 0.6, base_x=280.0, base_y=640.0, count=5)

        # Dynamic aerodynamic wind-streaks rushing past during the steep stoop dive!
        if 11.8 <= lt <= 17.4:
            dive_int = pulse_bell(lt, 11.8, 17.4, sharp=0.6)
            sctx.set_line_width(2.2)
            for k in range(22):
                prog = (lt * 2.8 + hash1d(k * 37 + 11)) % 1.0
                sx_w = lerp(1950.0, -150.0, prog) + (hash1d(k * 53) - 0.5) * 400.0
                sy_w = lerp(1080.0, -80.0, prog) + (hash1d(k * 61) - 0.5) * 500.0
                slen = 110.0 + 130.0 * hash1d(k * 71)
                sctx.move_to(sx_w, sy_w)
                sctx.line_to(sx_w - slen * 0.78, sy_w - slen * 0.62)
                sctx.set_source_rgba(1.0, 0.99, 0.95, 0.26 * dive_int)
                sctx.stroke()

        draw_hawk(
            sctx,
            x=hx,
            y=hy,
            scale=h_scale,
            facing=1.0,
            body_pitch=h_pitch,
            t=t,
            wing_flap_phase=flap_ph,
            flap_amp=flap_amp,
            dive_fold=dive_fold,
            tail_spread=tail_spread,
            talon_extend=talon_ext,
            talon_grip=talon_grip,
            head_pitch=0.14 if lt < 17.6 else -0.04,
            carrying_snake=carry,
        )

        slayer.composite_over(frame_sky, opacity=1.0)
        apply_cinematic_post(frame_sky, t, sun_norm=cam_s.sun_screen, sun_strength=cam_s.sun_strength * 0.55, bloom_strength=0.14)

    # 2. AERIAL HIGH-ANGLE VIEW OF SNAKE IN OPEN MEADOW (6.65 -> 9.15s)
    frame_aerial: Optional[np.ndarray] = None
    if aerial_mix > 0.005:
        u_a = inv_lerp(6.7, 9.1, lt)
        cam_a = CameraState(
            cx=lerp(1050.0, 1140.0, u_a),
            cy=lerp(670.0, 710.0, u_a),
            zoom=lerp(1.06, 1.24, u_a),
            sun_screen=(0.74, 0.14),
            sun_strength=0.22,
            dof_fg=0.0,
        ).with_organic_motion(t, breath_amp=1.0)
        frame_aerial = cam_a.sample_layer(env["rocky_meadow"], parallax=0.95, out_w=out_w, out_h=out_h, world_origin=(-240.0, -135.0))
        alayer = CairoLayer(out_w, out_h)
        actx = alayer.ctx
        cam_a.apply_to_cairo(actx, parallax=1.0, out_w=out_w, out_h=out_h)

        # Snake moving openly across the sunny meadow clearing — clearly exposed from above!
        sn_hx = lerp(960.0, 1210.0, u_a)
        sn_hy = 815.0 + math.sin(u_a * 3.0) * 14.0
        sp_a = compute_snake_spine((sn_hx, sn_hy), heading_rad=-0.04, total_len=490.0, slither_phase=lt * 4.5, wave_amp=34.0)
        draw_snake(actx, sp_a, max_radius=17.5, tongue_flick=pulse_bell(lt, 7.6, 8.2), t=t)

        # Subtle passing raptor wing-shadow sweeping across the meadow grass!
        sh_x = lerp(520.0, 1520.0, u_a)
        draw_soft_shadow(actx, sh_x, 820.0, 190.0, 52.0, angle=-0.18, alpha=0.20, color=(0.04, 0.07, 0.05))
        alayer.composite_over(frame_aerial, opacity=1.0)
        apply_cinematic_post(frame_aerial, t, sun_norm=cam_a.sun_screen, sun_strength=0.18, bloom_strength=0.16)

    # 3. FIERCE HAWK EYE MACRO CLOSE-UP (8.95 -> 11.5s)
    frame_eye: Optional[np.ndarray] = None
    if eye_mix > 0.005:
        frame_eye = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        elayer = CairoLayer(out_w, out_h)
        ectx = elayer.ctx
        ue = clamp((lt - 9.0) / 2.4)
        sc_r = out_w / float(WIDTH)
        draw_hawk_eye_closeup(
            ectx,
            out_w * 0.46,
            out_h * 0.50,
            radius=lerp(205.0, 245.0, smootherstep(ue)) * sc_r,
            focus_u=smootherstep(inv_lerp(0.18, 0.78, ue)),
            t=t,
        )
        elayer.composite_over(frame_eye, opacity=1.0)
        apply_cinematic_post(frame_eye, t, sun_norm=(0.72, 0.20), sun_strength=0.0, bloom_strength=0.12, vignette_strength=1.30)

    # 4. LOW-LEVEL MEADOW SWOOP, AIR-BRAKE FLARE & CLEAN SNAKE CATCH (17.0 -> 21.35s)
    frame_gnd: Optional[np.ndarray] = None
    if ground_mix > 0.005:
        gu = clamp((lt - 17.0) / 4.2)
        catch_t = 19.15  # Exact split-second talons lock onto the snake!
        caught = lt >= catch_t

        cam_g = CameraState(
            cx=lerp(960.0, 1210.0, smootherstep(gu)),
            cy=lerp(615.0, 560.0, smootherstep(gu)),
            zoom=lerp(1.10, 1.02, gu),
            shake_amp=pulse_bell(lt, catch_t - 0.04, catch_t + 0.32) * 5.2,
            sun_screen=(0.76, 0.16),
            sun_strength=0.38,
            dof_fg=0.45,
        ).with_organic_motion(t, breath_amp=1.5)

        frame_gnd = cam_g.sample_layer(env["rocky_meadow"], parallax=0.95, out_w=out_w, out_h=out_h, world_origin=(-240.0, -135.0))
        glayer = CairoLayer(out_w, out_h)
        gctx = glayer.ctx
        cam_g.apply_to_cairo(gctx, parallax=1.0, out_w=out_w, out_h=out_h)

        # Snake gliding in sunny clearing until exact catch_t!
        sn_x = lerp(1040.0, 1185.0, inv_lerp(17.0, catch_t, lt))
        sn_y = 842.0
        if not caught:
            sp_g = compute_snake_spine((sn_x, sn_y), heading_rad=0.02, total_len=480.0, slither_phase=lt * 5.0, wave_amp=34.0)
            draw_snake(gctx, sp_g, max_radius=18.5, t=t)

        # Hawk swoops smoothly in from upper-left, flares wings open to brake, grabs snake, powers skyward!
        if lt < catch_t:
            su = smootherstep(inv_lerp(17.0, catch_t, lt))
            hx_g = lerp(290.0, sn_x - 75.0, su)
            hy_g = lerp(215.0, sn_y - 72.0, ease_out_quad(su))
            h_pitch_g = lerp(0.54, -0.18, ease_in_quad(su))  # pitches nose-up into classic raptor air-brake flare!
            dive_fold_g = lerp(0.75, 0.0, smootherstep(inv_lerp(17.5, 18.7, lt)))
            flap_amp_g = lerp(0.10, 0.58, smootherstep(inv_lerp(18.2, catch_t, lt)))
            talon_ext_g = smootherstep(inv_lerp(17.9, 18.9, lt))
            talon_grip_g = smootherstep(inv_lerp(19.02, catch_t, lt))
            tail_sp_g = lerp(0.25, 1.0, su)
        else:
            lu = ease_in_out_cubic(inv_lerp(catch_t, 21.35, lt))
            hx_g = lerp(sn_x - 75.0, 1620.0, lu)
            hy_g = lerp(sn_y - 72.0, 320.0, lu ** 0.85)
            h_pitch_g = lerp(-0.22, -0.34, lu)
            dive_fold_g = 0.0
            flap_amp_g = 0.78
            talon_ext_g = 0.95
            talon_grip_g = 1.0
            tail_sp_g = 0.86

        # Rapidly shrinking/expanding ground hawk shadow under swoop!
        dist_gnd = clamp((sn_y - hy_g) / 650.0)
        draw_soft_shadow(gctx, hx_g, sn_y + 18.0, lerp(165.0, 85.0, dist_gnd), lerp(36.0, 16.0, dist_gnd), alpha=lerp(0.36, 0.12, dist_gnd))

        # Wing downwash kicking up golden dust motes & dry leaves right at snatch!
        if catch_t - 0.1 <= lt <= catch_t + 1.4:
            dt_d = lt - (catch_t - 0.1)
            for k in range(14):
                ang_d = (k / 14.0) * math.tau
                rad_d = dt_d * (140.0 + 95.0 * hash1d(k * 41))
                dx = sn_x - 60 + math.cos(ang_d) * rad_d * 1.4
                dy = sn_y + 10 - abs(math.sin(ang_d)) * rad_d * 0.55
                a_d = (1.0 - dt_d / 1.5) * 0.55
                gctx.arc(dx, dy, 4.5 + (k % 3) * 2.5, 0, math.tau)
                gctx.set_source_rgba(0.76, 0.68, 0.46, a_d)
                gctx.fill()

        draw_hawk(
            gctx,
            x=hx_g,
            y=hy_g,
            scale=1.42,
            facing=1.0,
            body_pitch=h_pitch_g,
            t=t,
            wing_flap_phase=lt * 7.8,
            flap_amp=flap_amp_g,
            dive_fold=dive_fold_g,
            tail_spread=tail_sp_g,
            talon_extend=talon_ext_g,
            talon_grip=talon_grip_g,
            head_pitch=0.18 if not caught else -0.05,
            carrying_snake=caught,
        )

        glayer.composite_over(frame_gnd, opacity=1.0)
        _draw_foreground_dof_blades(frame_gnd, cam_g, t, out_w, out_h, blade_count=10, seed=590, color_tint="dry")
        apply_cinematic_post(frame_gnd, t, sun_norm=cam_g.sun_screen, sun_strength=cam_g.sun_strength, bloom_strength=0.24)

    # Blend active shot cleanly
    out = frame_sky if frame_sky is not None else np.zeros((out_h, out_w, 3), dtype=np.uint8)
    if frame_aerial is not None and aerial_mix > 0.005:
        out = cv2.addWeighted(out, 1.0 - aerial_mix, frame_aerial, aerial_mix, 0.0)
    if frame_eye is not None and eye_mix > 0.005:
        out = cv2.addWeighted(out, 1.0 - eye_mix, frame_eye, eye_mix, 0.0)
    if frame_gnd is not None and ground_mix > 0.005:
        out = cv2.addWeighted(out, 1.0 - ground_mix, frame_gnd, ground_mix, 0.0)
    return out


# =============================================================================
# SCENE 6: THE COMPLETE FOOD CHAIN (01:47 - 02:04)
# =============================================================================

def _draw_energy_arrow(
    ctx: cairo.Context,
    x0: float,
    x1: float,
    y: float,
    draw_u: float,
    t: float,
    seed: int = 0,
) -> None:
    """
    Draw a sleek, glowing golden energy-transfer arrow connecting two organisms
    with luminous solar energy particles flowing continuously left -> right!
    """
    du = clamp(draw_u)
    if du <= 0.01:
        return
    xe = lerp(x0, x1, ease_out_cubic(du))

    # Soft golden halo behind arrow shaft
    ctx.new_path()
    ctx.move_to(x0, y)
    ctx.line_to(xe, y)
    ctx.set_source_rgba(1.0, 0.84, 0.34, 0.24 * du)
    ctx.set_line_width(14.0)
    ctx.stroke()

    # Crisp golden core shaft
    pat = cairo.LinearGradient(x0, y, x1, y)
    pat.add_color_stop_rgba(0.0, 0.88, 0.72, 0.28, 0.90 * du)
    pat.add_color_stop_rgba(0.65, 1.00, 0.88, 0.44, 0.98 * du)
    pat.add_color_stop_rgba(1.0, 1.00, 0.96, 0.72, 1.00 * du)
    ctx.new_path()
    ctx.move_to(x0, y)
    ctx.line_to(xe, y)
    ctx.set_source(pat)
    ctx.set_line_width(4.2)
    ctx.stroke()

    # Elegant sculpted arrowhead once shaft completes > 70%
    head_u = smootherstep(inv_lerp(0.65, 1.0, du))
    if head_u > 0.01:
        hs = 14.0 * head_u
        ctx.new_path()
        ctx.move_to(xe + 6.0 * head_u, y)
        ctx.line_to(xe - hs * 1.15, y - hs * 0.78)
        ctx.line_to(xe - hs * 0.72, y)
        ctx.line_to(xe - hs * 1.15, y + hs * 0.78)
        ctx.close_path()
        ctx.set_source_rgba(1.0, 0.92, 0.56, 0.98 * head_u)
        ctx.fill()

    # Continuous flowing golden energy-transfer particles moving left -> right!
    if du > 0.45:
        p_alpha = smootherstep(inv_lerp(0.45, 1.0, du))
        for k in range(5):
            pu = (t * 0.75 + k / 5.0 + seed * 0.17) % 1.0
            px = lerp(x0 + 4.0, xe - 4.0, pu)
            py = y + math.sin(pu * math.tau * 2.0 + t * 4.0 + k) * 5.5
            env_p = math.sin(pu * math.pi) * p_alpha
            r = 4.2 + 2.2 * math.sin(t * 5.0 + k)
            g = cairo.RadialGradient(px, py, 0.5, px, py, r * 2.4)
            g.add_color_stop_rgba(0.0, 1.0, 0.99, 0.86, 0.95 * env_p)
            g.add_color_stop_rgba(0.45, 1.0, 0.84, 0.32, 0.55 * env_p)
            g.add_color_stop_rgba(1.0, 1.0, 0.76, 0.20, 0.0)
            ctx.set_source(g)
            ctx.arc(px, py, r * 2.4, 0, math.tau)
            ctx.fill()


def render_scene6(
    t: float,
    lt: float,
    env: Dict[str, np.ndarray],
    out_w: int = WIDTH,
    out_h: int = HEIGHT,
) -> np.ndarray:
    """
    SCENE 6: THE COMPLETE FOOD CHAIN (01:47 - 02:04, lt in [0..17])
      - Clean, luminous natural sunlit meadow backdrop with delicate botanical border
      - All 5 LIVE animated organisms enter sequentially in order:
          GRASS -> GRASSHOPPER -> FROG -> SNAKE -> HAWK
      - Luminous golden energy-transfer arrows with flowing energy particles connect them
      - Clear, polished documentary typography (Species + Ecological Role)
      - Culminates with glowing "EVERY LIVING THING IS CONNECTED." and peaceful fade-out!
    """
    bg_src = env["finale_backdrop"]
    if bg_src.shape[1] != out_w or bg_src.shape[0] != out_h:
        frame = cv2.resize(bg_src, (out_w, out_h), interpolation=cv2.INTER_AREA)
    else:
        frame = bg_src.copy()

    layer = CairoLayer(out_w, out_h)
    ctx = layer.ctx
    scale = out_w / float(WIDTH)
    ctx.scale(scale, scale)

    # Gentle floating golden pollen across the luminous finale stage
    _draw_pollen_and_insects(ctx, t, count=30, y_min=100.0, y_max=980.0, warm_alpha=0.52, seed=666)

    # 5 balanced horizontal column centers
    col_xs = [225.0, 592.0, 960.0, 1328.0, 1695.0]
    stage_y = 505.0

    # Sequential entry timings synced smoothly with Scene 6 narration!
    entry_times = [0.8, 2.3, 3.8, 5.3, 6.8]
    arrow_times = [2.0, 3.5, 5.0, 6.5]
    entries = [smootherstep(inv_lerp(et, et + 1.05, lt)) for et in entry_times]
    arrows = [smootherstep(inv_lerp(at, at + 1.0, lt)) for at in arrow_times]

    # Soft atmospheric deep slate-sage cushion along top sky so title typography reads crisply!
    hdr_a = smootherstep(inv_lerp(0.25, 1.35, lt))
    if hdr_a > 0.01:
        top_veil = cairo.LinearGradient(0, 0, 0, 290)
        top_veil.add_color_stop_rgba(0.0, 0.08, 0.14, 0.12, 0.56 * hdr_a)
        top_veil.add_color_stop_rgba(0.65, 0.10, 0.16, 0.14, 0.34 * hdr_a)
        top_veil.add_color_stop_rgba(1.0, 0.10, 0.16, 0.14, 0.0)
        ctx.set_source(top_veil)
        ctx.rectangle(0, 0, WIDTH, 290)
        ctx.fill()

        hw_r = 280.0 * ease_out_cubic(hdr_a)
        ctx.move_to(WIDTH * 0.5 - hw_r, 198.0)
        ctx.line_to(WIDTH * 0.5 + hw_r, 198.0)
        ctx.set_source_rgba(1.0, 0.90, 0.56, 0.72 * hdr_a)
        ctx.set_line_width(2.2)
        ctx.stroke()

    # Soft luminous circular botanical showcase medallions behind each organism
    climax_glow = smootherstep(inv_lerp(11.6, 13.4, lt))
    for i, (cx, eu) in enumerate(zip(col_xs, entries)):
        if eu <= 0.01:
            continue
        r_med = 128.0 * ease_out_back(eu, overshoot=0.6)
        pulse = 1.0 + 0.03 * math.sin(t * 2.2 - i * 0.8) + 0.05 * climax_glow
        pat = cairo.RadialGradient(cx, stage_y - 12, 16, cx, stage_y, r_med * pulse)
        pat.add_color_stop_rgba(0.0, 0.98, 0.95, 0.82, (0.28 + 0.14 * climax_glow) * eu)
        pat.add_color_stop_rgba(0.68, 0.88, 0.86, 0.62, 0.14 * eu)
        pat.add_color_stop_rgba(1.0, 0.80, 0.84, 0.56, 0.0)
        ctx.set_source(pat)
        ctx.arc(cx, stage_y, r_med * pulse, 0, math.tau)
        ctx.fill()

        # Delicate thin golden halo ring around each organism pedestal
        ctx.arc(cx, stage_y, r_med * 0.92, 0, math.tau)
        ctx.set_source_rgba(0.98, 0.90, 0.60, (0.32 + 0.25 * climax_glow) * eu)
        ctx.set_line_width(1.6)
        ctx.stroke()

    # Draw the 4 glowing energy-transfer arrows between the 5 organisms!
    for i, au in enumerate(arrows):
        x0 = col_xs[i] + 132.0
        x1 = col_xs[i + 1] - 132.0
        _draw_energy_arrow(ctx, x0, x1, stage_y, au, t=t, seed=i)

    # 1. LIVE ANIMATED GRASS TUFT (Producer)
    if entries[0] > 0.01:
        eu = entries[0]
        gy_off = lerp(34.0, 0.0, ease_out_cubic(eu))
        cx = col_xs[0]
        base_y = stage_y + 74.0 + gy_off
        draw_soft_shadow(ctx, cx, base_y + 4, 72, 14, alpha=0.30 * eu)
        tuft = [
            (-42, 148, 14, -36, 0.20, None),
            (-24, 178, 16, -22, 0.45, 0.70),
            ( -6, 198, 17,  -4, 0.30, None),
            ( 12, 186, 16,  18, 0.60, 0.74),
            ( 30, 158, 14,  34, 0.35, None),
            ( 44, 130, 12,  44, 0.72, None),
        ]
        for dx, bh, bw, lean, cmix, dew in tuft:
            wv = wind_field(cx + dx, t, strength=0.75)
            draw_grass_blade(ctx, cx + dx, base_y, bh, bw, lean, wv, grow_u=ease_out_cubic(eu), color_mix=cmix, dew_s=dew, alpha=eu)

    # 2. LIVE ANIMATED GRASSHOPPER (Primary Consumer)
    if entries[1] > 0.01:
        eu = entries[1]
        sc = 1.02 * ease_out_back(eu, overshoot=0.45)
        draw_grasshopper(
            ctx,
            x=col_xs[1] + 14.0,
            y=stage_y + 6.0 + lerp(28.0, 0.0, ease_out_cubic(eu)),
            scale=sc,
            facing=1.0,
            t=t,
            chew_phase=0.0,
            alpha=eu,
        )

    # 3. LIVE ANIMATED EMERALD FROG (Secondary Consumer)
    if entries[2] > 0.01:
        eu = entries[2]
        sc = 1.15 * ease_out_back(eu, overshoot=0.45)
        draw_frog(
            ctx,
            x=col_xs[2] - 4.0,
            y=stage_y + 10.0 + lerp(28.0, 0.0, ease_out_cubic(eu)),
            scale=sc,
            facing=1.0,
            t=t,
            blink=blink_envelope(t, period=3.0, offset=0.4),
            alpha=eu,
        )

    # 4. LIVE ANIMATED SCALED SNAKE (Tertiary Consumer)
    if entries[3] > 0.01:
        eu = entries[3]
        sc_u = ease_out_cubic(eu)
        cx = col_xs[3]
        sp6 = compute_snake_spine(
            head_xy=(cx + 82.0, stage_y + 12.0 + lerp(26.0, 0.0, sc_u)),
            heading_rad=-0.10,
            total_len=215.0 * sc_u,
            n_segs=42,
            slither_phase=t * 2.8,
            wave_amp=26.0,
            coil_u=0.28,
        )
        draw_snake(
            ctx,
            sp6,
            max_radius=12.2 * sc_u,
            head_angle=-0.10,
            tongue_flick=pulse_bell((t % 3.2), 1.0, 1.55),
            draw_shadow=True,
            alpha=eu,
            t=t,
        )

    # 5. LIVE ANIMATED RED-TAILED HAWK (Top Predator)
    if entries[4] > 0.01:
        eu = entries[4]
        sc = 0.80 * ease_out_back(eu, overshoot=0.4)
        bob = math.sin(t * 2.3) * 6.0
        draw_hawk(
            ctx,
            x=col_xs[4] - 6.0,
            y=stage_y + 12.0 + bob + lerp(-32.0, 0.0, ease_out_cubic(eu)),
            scale=sc,
            facing=1.0,
            body_pitch=-0.10,
            t=t,
            wing_flap_phase=t * 3.4,
            flap_amp=0.22,
            dive_fold=0.0,
            tail_spread=0.82,
            talon_extend=0.15,
            carrying_snake=False,
            alpha=eu,
        )

    layer.composite_over(frame, opacity=1.0)

    # Crisp high-DPI documentary typography overlays!
    labels = [
        ("GRASS",       "Producer"),
        ("GRASSHOPPER", "Primary Consumer"),
        ("FROG",        "Secondary Consumer"),
        ("SNAKE",       "Tertiary Consumer"),
        ("HAWK",        "Top Predator"),
    ]
    texts: List[Dict[str, object]] = [
        {
            "text": "THE FOOD CHAIN",
            "x": WIDTH * 0.5,
            "y": 140.0,
            "size": 62,
            "style": "serif_bold",
            "color_rgb": (255, 250, 232),
            "alpha": hdr_a,
            "shadow": True,
            "glow": True,
        },
        {
            "text": "Energy Flow Through a Grassland Ecosystem",
            "x": WIDTH * 0.5,
            "y": 232.0,
            "size": 26,
            "style": "serif",
            "color_rgb": (240, 236, 212),
            "alpha": hdr_a * 0.92,
            "shadow": True,
        },
    ]

    for i, ((name, role), cx, eu) in enumerate(zip(labels, col_xs, entries)):
        lbl_a = smootherstep(inv_lerp(0.25, 1.0, eu))
        if lbl_a <= 0.01:
            continue
        texts.append({
            "text": name,
            "x": cx,
            "y": 698.0,
            "size": 31,
            "style": "serif_bold",
            "color_rgb": (255, 249, 228),
            "alpha": lbl_a,
            "shadow": True,
            "glow": climax_glow > 0.3,
        })
        texts.append({
            "text": role,
            "x": cx,
            "y": 742.0,
            "size": 23,
            "style": "sans",
            "color_rgb": (244, 224, 152),
            "alpha": lbl_a * 0.95,
            "shadow": True,
        })

    # Unforgettable closing thesis line synced to final narration!
    close_a = smootherstep(inv_lerp(11.6, 12.8, lt))
    if close_a > 0.01:
        texts.append({
            "text": "EVERY LIVING THING IS CONNECTED.",
            "x": WIDTH * 0.5,
            "y": 905.0,
            "size": 46,
            "style": "serif_bold",
            "color_rgb": (255, 246, 206),
            "alpha": close_a,
            "shadow": True,
            "glow": True,
            "glow_rgb": (255, 218, 108),
        })

    draw_text_overlay(frame, texts)
    apply_cinematic_post(frame, t, sun_norm=(0.50, -0.10), sun_strength=0.14 + 0.06 * climax_glow, bloom_strength=0.16, vignette_strength=0.85)

    # Peaceful final fade-out over the last 1.3 seconds of the film
    scene_dur = float(SCENE_SPECS[-1]["end"]) - float(SCENE_SPECS[-1]["start"])
    if lt > scene_dur - 1.3:
        fade = 1.0 - smootherstep(inv_lerp(scene_dur - 1.3, scene_dur, lt))
        frame = np.clip(frame.astype(np.float32) * fade, 0, 255).astype(np.uint8)

    return frame


# =============================================================================
# Master Continuous Film Frame Router (with Seamless Atmospheric Transitions)
# =============================================================================

SCENE_RENDERERS = {
    1: render_scene1,
    2: render_scene2,
    3: render_scene3,
    4: render_scene4,
    5: render_scene5,
    6: render_scene6,
}


def render_single_scene_frame(
    scene_id: int,
    local_t: float,
    env: Dict[str, np.ndarray],
    out_w: int = WIDTH,
    out_h: int = HEIGHT,
) -> np.ndarray:
    spec = SCENE_SPECS[scene_id - 1]
    global_t = float(spec["start"]) + local_t
    return SCENE_RENDERERS[scene_id](global_t, local_t, env, out_w=out_w, out_h=out_h)


def render_film_frame(
    t: float,
    env: Dict[str, np.ndarray],
    out_w: int = WIDTH,
    out_h: int = HEIGHT,
) -> np.ndarray:
    """
    Render global timestamp `t` of the complete continuous film, seamlessly
    dissolving across consecutive scene boundaries using `TRANSITION_OVERLAP`.
    """
    # Find primary active scene
    active_idx = len(SCENE_SPECS) - 1
    for i, sp in enumerate(SCENE_SPECS):
        if float(sp["start"]) <= t < float(sp["end"]):
            active_idx = i
            break

    cur = SCENE_SPECS[active_idx]
    sid = int(cur["id"])
    lt = t - float(cur["start"])
    frame = SCENE_RENDERERS[sid](t, lt, env, out_w=out_w, out_h=out_h)

    # Check outgoing overlap transition into next scene (centered on boundary!)
    if active_idx + 1 < len(SCENE_SPECS):
        nxt = SCENE_SPECS[active_idx + 1]
        nsid = int(nxt["id"])
        overlap = TRANSITION_OVERLAP.get((sid, nsid), 1.0)
        boundary = float(cur["end"])
        if t >= boundary - overlap * 0.5:
            u = smootherstep(inv_lerp(boundary - overlap * 0.5, boundary + overlap * 0.5, t))
            nxt_lt = max(0.0, t - boundary)
            nxt_frame = SCENE_RENDERERS[nsid](t, nxt_lt, env, out_w=out_w, out_h=out_h)
            frame = cv2.addWeighted(frame, 1.0 - u, nxt_frame, u, 0.0)

    # Check incoming overlap tail from previous scene
    if active_idx > 0:
        prv = SCENE_SPECS[active_idx - 1]
        psid = int(prv["id"])
        overlap = TRANSITION_OVERLAP.get((psid, sid), 1.0)
        boundary = float(cur["start"])
        if t < boundary + overlap * 0.5:
            u = smootherstep(inv_lerp(boundary - overlap * 0.5, boundary + overlap * 0.5, t))
            prv_lt = min(float(prv["end"]) - float(prv["start"]), t - float(prv["start"]))
            prv_frame = SCENE_RENDERERS[psid](t, prv_lt, env, out_w=out_w, out_h=out_h)
            frame = cv2.addWeighted(prv_frame, 1.0 - u, frame, u, 0.0)

    return frame


# =============================================================================
# Visual QA Storyboard Contact Sheet Generator (4 Key Moments per Scene!)
# =============================================================================

if __name__ == "__main__":
    import os
    from environments import load_environment_assets

    base_dir = os.path.dirname(__file__)
    env = load_environment_assets(os.path.join(base_dir, "assets"))
    prev_dir = os.path.join(base_dir, "previews")
    os.makedirs(prev_dir, exist_ok=True)

    # 4 defining dramatic timestamps inside each of the 6 scenes!
    qa_timestamps = {
        1: [2.2, 6.2, 9.0, 17.5],
        2: [3.0, 7.2, 13.8, 18.8],
        3: [3.2, 10.6, 13.88, 16.0],
        4: [4.2, 11.4, 12.95, 16.5],
        5: [3.5, 10.2, 14.8, 19.3],
        6: [3.2, 7.5, 10.5, 13.8],
    }

    for sid, lts in qa_timestamps.items():
        tiles = []
        for lt in lts:
            im = render_single_scene_frame(sid, lt, env, out_w=960, out_h=540)
            tiles.append(im)
        sheet = np.concatenate([
            np.concatenate(tiles[:2], axis=1),
            np.concatenate(tiles[2:], axis=1),
        ], axis=0)
        out_p = os.path.join(prev_dir, f"qa_scene{sid}.png")
        cv2.imwrite(out_p, sheet)
        print(f"Saved visual QA sheet: {out_p}")

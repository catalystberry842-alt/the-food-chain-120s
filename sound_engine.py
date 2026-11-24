"""
sound_engine.py — Complete Wildlife Soundscape, Foley Synthesis & Mastering
===========================================================================

Procedurally synthesizes an immersive stereo wildlife sound environment synced
frame-accurately to every scene's visual beats:
  - Scene 1: Morning birdsong, soft dawn breeze, subterranean soil movement,
             water percolation droplets, tender sprouting grass rustle.
  - Scene 2: Meadow breeze, warm insect/cicada hum, grass stalk rustle,
             rhythmic crisp grasshopper leaf-chewing foley synced to bites.
  - Scene 3: Freshwater pond lapping ambience, resonant formant pond frog croaks,
             dragonfly/insect buzz, leap launch rustle, rapid wet tongue flick,
             water splash droplets & ripples.
  - Scene 4: Crisp dry autumn leaves, panned scaly snake slithering friction,
             subtle breathy sibilant hiss, sharp lunge strike swoosh.
  - Scene 5: High-altitude sweeping sky wind, authentic piercing descending
             Red-tailed Hawk cry, deep rhythmic wingbeats/flapping, intense
             rushing dive wind, crisp talon-snatch & wing-flare whoosh.
  - Scene 6: Serene meadow breeze, luminous golden energy-arrow chimes & final
             harmonic shimmer.

Combines narration (`narration.py`), original score (`music.py`), and layered
wildlife foley with automatic broadcast sidechain ducking into `master_soundtrack.wav`.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile

from animation import SAMPLE_RATE, SCENE_SPECS, TOTAL_DURATION
from music import generate_original_score, midi_hz
from narration import generate_all_narration


def _pink_noise(n: int, seed: int = 1) -> np.ndarray:
    """Fast spectral 1/f pink noise normalized to unit std."""
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    scale = 1.0 / np.sqrt(np.maximum(freqs, 18.0))
    scale[0] = 0.0
    pink = np.fft.irfft(fft * scale, n=n)
    std = float(np.std(pink))
    return (pink / max(std, 1e-8)).astype(np.float32)


def _add_panned(
    mix: np.ndarray,
    mono: np.ndarray,
    start_s: float,
    pan: float = 0.0,
    gain: float = 1.0,
    sr: int = SAMPLE_RATE,
) -> None:
    s0 = int(round(start_s * sr))
    if s0 >= mix.shape[0] or s0 + len(mono) <= 0:
        return
    src0 = max(0, -s0)
    dst0 = max(0, s0)
    cnt = min(len(mono) - src0, mix.shape[0] - dst0)
    if cnt <= 0:
        return
    ang = (pan * 0.5 + 0.5) * math.pi * 0.5
    seg = mono[src0 : src0 + cnt]
    mix[dst0 : dst0 + cnt, 0] += seg * (math.cos(ang) * gain)
    mix[dst0 : dst0 + cnt, 1] += seg * (math.sin(ang) * gain)


def _envelope_trapezoid(n: int, fi: float, fo: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    a = min(int(fi * sr), n // 2)
    b = min(int(fo * sr), n - a)
    if a > 1:
        env[:a] = np.sin(np.linspace(0.0, math.pi * 0.5, a, dtype=np.float32))
    if b > 1:
        env[-b:] = np.cos(np.linspace(0.0, math.pi * 0.5, b, dtype=np.float32))
    return env


# =============================================================================
# Organic Wildlife & Nature Sound Synthesizers
# =============================================================================

def sfx_wind_bed(dur: float, fc_lo: float = 160.0, fc_hi: float = 650.0, amp: float = 0.12, seed: int = 11) -> np.ndarray:
    """Breathing natural wind ambience (stereo-ready via distinct seeds)."""
    n = int(round(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    p = _pink_noise(n, seed=seed)
    sos = signal.butter(3, [fc_lo, fc_hi], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    w = signal.sosfilt(sos, p).astype(np.float32)
    # Slow breathing wind gusts
    gust = 0.55 + 0.30 * np.sin(t * 0.48 + seed) + 0.15 * np.sin(t * 1.15 + seed * 1.7)
    w *= gust * _envelope_trapezoid(n, 0.8, 1.0)
    peak = float(np.max(np.abs(w)))
    return (w * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_water_droplet(base_hz: float = 1350.0, amp: float = 0.24) -> np.ndarray:
    """Crystalline water droplet plop with classic upward bubble-resonance pitch chirp."""
    dur = 0.11
    n = int(round(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    f_t = base_hz * (1.0 + 0.85 * (1.0 - np.exp(-t * 38.0)))
    ph = 2.0 * math.pi * np.cumsum(f_t) / SAMPLE_RATE
    y = np.sin(ph) * np.exp(-t * 46.0)
    att = min(int(0.003 * SAMPLE_RATE), n // 4)
    y[:att] *= np.linspace(0.0, 1.0, att)
    return (y * amp).astype(np.float32)


def sfx_soil_rustle(dur: float = 1.6, amp: float = 0.12, seed: int = 21) -> np.ndarray:
    """Soft subterranean earth settling & delicate root-creak granularity."""
    n = int(round(dur * SAMPLE_RATE))
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, n)
    # Sparse granular clod clicks + earthy low-mid rumble
    grain_env = (rng.random(n) > 0.9965).astype(np.float64)
    grain_env = signal.convolve(grain_env, np.hanning(180), mode="same")
    sos_low = signal.butter(2, [90.0, 420.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    sos_grit = signal.butter(2, [900.0, 3200.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    y = 0.55 * signal.sosfilt(sos_low, raw) + 0.95 * signal.sosfilt(sos_grit, raw) * grain_env
    y *= _envelope_trapezoid(n, 0.25, 0.45)
    peak = float(np.max(np.abs(y)))
    return (y * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_birdsong_phrase(seed: int = 101, amp: float = 0.16) -> np.ndarray:
    """Melodic natural morning songbird phrase (multi-note FM chirp & trill warble)."""
    rng = np.random.default_rng(seed)
    n_notes = int(rng.integers(3, 6))
    total_n = int(1.6 * SAMPLE_RATE)
    out = np.zeros(total_n, dtype=np.float64)
    pos = 0
    base_f = float(rng.uniform(2550.0, 3850.0))

    for k in range(n_notes):
        ndur = float(rng.uniform(0.09, 0.20))
        nn = int(round(ndur * SAMPLE_RATE))
        if pos + nn >= total_n:
            break
        t = np.arange(nn, dtype=np.float64) / SAMPLE_RATE
        u = t / ndur
        f_start = base_f * float(rng.uniform(0.85, 1.25))
        f_end = base_f * float(rng.uniform(0.80, 1.32))
        trill = float(rng.uniform(0.0, 0.08)) * np.sin(2.0 * math.pi * float(rng.uniform(22.0, 42.0)) * t)
        f_t = (f_start + (f_end - f_start) * (u ** 1.2)) * (1.0 + trill)
        ph = 2.0 * math.pi * np.cumsum(f_t) / SAMPLE_RATE
        env = np.sin(math.pi * u) ** 1.15
        note = (0.85 * np.sin(ph) + 0.15 * np.sin(2.0 * ph)) * env
        out[pos : pos + nn] += note
        pos += nn + int(rng.uniform(0.05, 0.14) * SAMPLE_RATE)

    peak = float(np.max(np.abs(out)))
    return (out * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_grass_rustle(dur: float = 1.4, amp: float = 0.11, seed: int = 33) -> np.ndarray:
    """Delicate soft-leaf swish of wind or insect moving across grass blades."""
    n = int(round(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    raw = _pink_noise(n, seed=seed)
    sos = signal.butter(3, [1400.0, 5600.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    y = signal.sosfilt(sos, raw).astype(np.float32)
    swish = 0.55 + 0.45 * np.sin(t * 4.2 + seed)
    y *= swish * _envelope_trapezoid(n, 0.2, 0.35)
    peak = float(np.max(np.abs(y)))
    return (y * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_insect_buzz(dur: float = 2.4, freq_hz: float = 185.0, amp: float = 0.075, seed: int = 41) -> np.ndarray:
    """Warm natural hovering meadow insect / dragonfly wing buzz."""
    n = int(round(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    f_t = freq_hz * (1.0 + 0.08 * np.sin(2.0 * math.pi * 1.4 * t + seed))
    ph = 2.0 * math.pi * np.cumsum(f_t) / SAMPLE_RATE
    # Harmonic buzz filtered softly so it sounds organic & distant (never annoying!)
    y = 0.50 * np.sin(ph) + 0.35 * np.sin(2.0 * ph) + 0.25 * np.sin(3.0 * ph) + 0.15 * np.sin(5.0 * ph)
    sos = signal.butter(2, [300.0, 2600.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    y = signal.sosfilt(sos, y) * _envelope_trapezoid(n, 0.45, 0.65)
    peak = float(np.max(np.abs(y)))
    return (y * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_leaf_chew_bite(seed: int = 51, amp: float = 0.14) -> np.ndarray:
    """Crisp tiny organic crunch of grasshopper mandibles clipping a fresh green grass edge."""
    dur = 0.13
    n = int(round(dur * SAMPLE_RATE))
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    raw = rng.normal(0.0, 1.0, n)
    # 3 micro-fiber snaps inside one crisp bite
    burst = (
        np.exp(-((t - 0.022) ** 2) / 0.00009)
        + 0.85 * np.exp(-((t - 0.052) ** 2) / 0.00011)
        + 0.60 * np.exp(-((t - 0.084) ** 2) / 0.00014)
    )
    sos = signal.butter(3, [1800.0, 7200.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    y = signal.sosfilt(sos, raw) * burst
    peak = float(np.max(np.abs(y)))
    return (y * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_frog_croak(pitch_hz: float = 112.0, dur: float = 0.48, amp: float = 0.24) -> np.ndarray:
    """Realistic resonant green pond frog croak using glottal pulses + dual vocal-sac formants."""
    n = int(round(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    u = t / dur
    # Slightly rising then settling croak pulse rate (~42 -> 56 -> 38 pulses/sec)
    p_rate = 40.0 + 18.0 * np.sin(math.pi * u)
    ph_p = 2.0 * math.pi * np.cumsum(p_rate) / SAMPLE_RATE
    pulses = np.maximum(0.0, np.sin(ph_p)) ** 3.2

    # Fundamental tone + vocal sac harmonics
    f_t = pitch_hz * (1.0 + 0.18 * np.sin(math.pi * u))
    ph = 2.0 * math.pi * np.cumsum(f_t) / SAMPLE_RATE
    exc = (np.sin(ph) + 0.65 * np.sin(2 * ph) + 0.45 * np.sin(3 * ph) + 0.30 * np.sin(5 * ph)) * pulses

    # Resonant amphibian vocal-sac formants at ~340 Hz and ~1180 Hz
    sos1 = signal.butter(3, [240.0, 480.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    sos2 = signal.butter(3, [920.0, 1520.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    y = 0.65 * signal.sosfilt(sos1, exc) + 0.55 * signal.sosfilt(sos2, exc)
    y *= (np.sin(math.pi * u) ** 0.75)
    peak = float(np.max(np.abs(y)))
    return (y * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_pond_ambience(dur: float, amp: float = 0.09, seed: int = 61) -> np.ndarray:
    """Gentle sparkling shoreline water lapping & tranquil pond bed."""
    n = int(round(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    p = _pink_noise(n, seed=seed)
    sos_w = signal.butter(3, [260.0, 1450.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    bed = signal.sosfilt(sos_w, p).astype(np.float32) * (0.55 + 0.45 * np.sin(t * 1.15) * np.cos(t * 0.63))
    # Add occasional soft random shoreline bubble droplets
    rng = np.random.default_rng(seed + 5)
    out = bed.copy()
    for _ in range(int(dur * 1.4)):
        d_t = float(rng.uniform(0.3, dur - 0.3))
        drop = sfx_water_droplet(base_hz=float(rng.uniform(750.0, 1650.0)), amp=0.16)
        s0 = int(d_t * SAMPLE_RATE)
        cnt = min(len(drop), n - s0)
        if cnt > 0:
            out[s0 : s0 + cnt] += drop[:cnt]
    out *= _envelope_trapezoid(n, 0.8, 1.0)
    peak = float(np.max(np.abs(out)))
    return (out * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_tongue_flick(amp: float = 0.22) -> np.ndarray:
    """Crisp wet fleshy frog tongue lash & snap catch."""
    dur = 0.32
    n = int(round(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    # Fast elastic upward sweep + sticky wet pop at impact (0.18s)
    f_sw = 240.0 + 1350.0 * np.exp(-((t - 0.08) ** 2) / 0.0025)
    ph = 2.0 * math.pi * np.cumsum(f_sw) / SAMPLE_RATE
    whip = np.sin(ph) * np.exp(-t * 12.0) * 0.55
    rng = np.random.default_rng(88)
    air = signal.sosfilt(signal.butter(2, [900.0, 4800.0], btype="bandpass", fs=SAMPLE_RATE, output="sos"), rng.normal(0, 1, n))
    air *= np.exp(-((t - 0.11) ** 2) / 0.003)
    pop = sfx_water_droplet(base_hz=920.0, amp=0.85)
    y = whip + air * 0.65
    s_p = int(0.17 * SAMPLE_RATE)
    cnt = min(len(pop), n - s_p)
    y[s_p : s_p + cnt] += pop[:cnt]
    peak = float(np.max(np.abs(y)))
    return (y * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_water_splash(dur: float = 0.65, amp: float = 0.24, seed: int = 71) -> np.ndarray:
    """Natural localized frog splash & settling water droplets."""
    n = int(round(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, n)
    sos = signal.butter(3, [480.0, 4200.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    spray = signal.sosfilt(sos, raw) * np.exp(-t * 5.8) * np.sin(np.minimum(1.0, t / 0.04) * math.pi * 0.5)
    # Resonant low plunk + 4 trailing droplets
    plunk = np.sin(2.0 * math.pi * (165.0 + 180.0 * t) * t) * np.exp(-t * 16.0) * 0.45
    y = spray + plunk
    for k in range(4):
        dt = 0.08 + k * 0.09
        dr = sfx_water_droplet(base_hz=880.0 + k * 240.0, amp=0.45 * (0.8 ** k))
        s0 = int(dt * SAMPLE_RATE)
        cnt = min(len(dr), n - s0)
        if cnt > 0:
            y[s0 : s0 + cnt] += dr[:cnt]
    peak = float(np.max(np.abs(y)))
    return (y * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_dry_leaves_slither(dur: float = 5.5, amp: float = 0.14, seed: int = 81) -> np.ndarray:
    """Crisp dry leaf crackles + rhythmic scaly slithering swish across rocky ground."""
    n = int(round(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, n)
    # Soft rhythmic scaly undulation swish
    sos_s = signal.butter(3, [1200.0, 4800.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    swish = signal.sosfilt(sos_s, raw) * (0.45 + 0.55 * (np.sin(t * 4.5) ** 2))
    # Crisp sporadic dry leaf crackles
    crackle_imp = (rng.random(n) > 0.9985).astype(np.float64)
    crackle_env = signal.convolve(crackle_imp, np.exp(-np.linspace(0, 6, 320)), mode="same")
    sos_c = signal.butter(3, [2600.0, 8800.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    crackle = signal.sosfilt(sos_c, raw) * crackle_env * 1.4
    y = (0.55 * swish + 0.75 * crackle) * _envelope_trapezoid(n, 0.4, 0.6)
    peak = float(np.max(np.abs(y)))
    return (y * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_snake_hiss(dur: float = 1.25, amp: float = 0.16, seed: int = 89) -> np.ndarray:
    """Subtle breathy sibilant snake warning/coiling hiss."""
    n = int(round(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    u = t / dur
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, n)
    sos = signal.butter(4, [3400.0, 9200.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    y = signal.sosfilt(sos, raw) * (np.sin(math.pi * u) ** 1.15) * (0.85 + 0.15 * np.sin(t * 28.0))
    peak = float(np.max(np.abs(y)))
    return (y * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_fast_whoosh(dur: float = 0.42, fc_start: float = 350.0, fc_peak: float = 2400.0, amp: float = 0.28, seed: int = 93) -> np.ndarray:
    """Crisp fast strike lunge / wing flare / talon snatch air-swoosh."""
    n = int(round(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    u = t / dur
    raw = _pink_noise(n, seed=seed).astype(np.float64)
    # Time-varying resonant band-pass approximation via crossfaded low + high bands
    sos_lo = signal.butter(3, [fc_start, fc_start * 3.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    sos_hi = signal.butter(3, [fc_peak * 0.55, min(14000.0, fc_peak * 1.8)], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    lo = signal.sosfilt(sos_lo, raw)
    hi = signal.sosfilt(sos_hi, raw)
    bell = np.sin(math.pi * u) ** 1.35
    y = (lo * (1.0 - bell * 0.65) + hi * bell) * bell
    peak = float(np.max(np.abs(y)))
    return (y * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_hawk_call(dur: float = 1.45, amp: float = 0.28, seed: int = 109) -> np.ndarray:
    """
    Iconic piercing descending Red-tailed Hawk scream ('keeee-eeee-arrr!'):
    Descending harmonic whistle with authentic raspy avian vocal-syrinx roughness.
    """
    n = int(round(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    u = t / dur
    # Characteristic pitch contour: sharp rise -> sustained soaring 2.7 kHz -> dramatic raspy drop to 1.25 kHz!
    f_base = 2780.0 - 1480.0 * (u ** 1.65) + 180.0 * np.exp(-((u - 0.08) ** 2) / 0.008)
    # Syrinx rasp FM modulation (~115 Hz rough vibrato!)
    rasp = 1.0 + 0.048 * np.sin(2.0 * math.pi * 112.0 * t)
    ph = 2.0 * math.pi * np.cumsum(f_base * rasp) / SAMPLE_RATE

    tone = 0.72 * np.sin(ph) + 0.42 * np.sin(2.0 * ph) + 0.20 * np.sin(3.0 * ph)
    # Raspy breath formant noise fused into the cry
    rng = np.random.default_rng(seed)
    air = signal.sosfilt(signal.butter(3, [1600.0, 5200.0], btype="bandpass", fs=SAMPLE_RATE, output="sos"), rng.normal(0, 1, n))
    y = (tone + 0.36 * air) * (np.sin(math.pi * u) ** 0.75) * np.exp(-u * 0.55)
    # Gentle distance echo for open valley acoustics
    d = int(0.19 * SAMPLE_RATE)
    if d < n:
        y[d:] += y[:-d] * 0.26
    peak = float(np.max(np.abs(y)))
    return (y * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_wing_flap(amp: float = 0.24, seed: int = 121) -> np.ndarray:
    """Deep rhythmic heavy raptor wingbeat ('whoomp-fwsh!')."""
    dur = 0.34
    n = int(round(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    u = t / dur
    raw = _pink_noise(n, seed=seed).astype(np.float64)
    sos_low = signal.butter(3, [65.0, 290.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    sos_feath = signal.butter(3, [480.0, 1900.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    env = (np.sin(math.pi * u) ** 1.45) * np.exp(-u * 1.2)
    y = (1.15 * signal.sosfilt(sos_low, raw) + 0.45 * signal.sosfilt(sos_feath, raw)) * env
    peak = float(np.max(np.abs(y)))
    return (y * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_dive_wind(dur: float = 6.5, amp: float = 0.28, seed: int = 131) -> np.ndarray:
    """Thrilling intensifying aerodynamic rushing wind during hawk stoop dive."""
    n = int(round(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    u = t / dur
    raw = _pink_noise(n, seed=seed).astype(np.float64)
    sos_lo = signal.butter(3, [140.0, 620.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    sos_hi = signal.butter(3, [750.0, 3200.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    # Intensity & pitch rush steadily higher as hawk accelerates toward Earth!
    rush = (u ** 1.4) * np.sin(np.minimum(1.0, (1.0 - u) / 0.14) * math.pi * 0.5)
    y = (signal.sosfilt(sos_lo, raw) * (0.45 + 0.55 * rush) + signal.sosfilt(sos_hi, raw) * (rush ** 1.3)) * (0.2 + 0.8 * rush)
    peak = float(np.max(np.abs(y)))
    return (y * (amp / max(peak, 1e-6))).astype(np.float32)


def sfx_energy_chime(midi_note: float, dur: float = 1.6, amp: float = 0.16) -> np.ndarray:
    """Delicate luminous harmonic chime synced to golden energy arrows in Scene 6."""
    n = int(round(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    f = midi_hz(midi_note)
    y = (
        0.70 * np.sin(2.0 * math.pi * f * t) * np.exp(-t * 2.6)
        + 0.28 * np.sin(2.0 * math.pi * f * 2.0 * t) * np.exp(-t * 4.2)
        + 0.12 * np.sin(2.0 * math.pi * f * 3.0 * t) * np.exp(-t * 6.5)
    )
    att = min(int(0.012 * SAMPLE_RATE), n // 8)
    y[:att] *= np.linspace(0.0, 1.0, att)
    peak = float(np.max(np.abs(y)))
    return (y * (amp / max(peak, 1e-6))).astype(np.float32)


# =============================================================================
# Full Film Soundscape Builder & Final Stereo Broadcast Masterer
# =============================================================================

def generate_sfx_mix(
    audio_dir: str,
    total_duration: float = TOTAL_DURATION,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Synthesize all 6 scenes' synchronized wildlife SFX & ambience into stereo stems."""
    os.makedirs(audio_dir, exist_ok=True)
    n_samples = int(round(total_duration * sr))
    sfx = np.zeros((n_samples, 2), dtype=np.float32)

    # --- SCENE 1 (00:00 - 00:22): LIFE BEGINS ---
    _add_panned(sfx, sfx_wind_bed(23.5, 140, 580, amp=0.11, seed=101), 0.0, pan=-0.35)
    _add_panned(sfx, sfx_wind_bed(23.5, 160, 640, amp=0.11, seed=102), 0.0, pan=+0.35)
    # Water percolation droplets soaking seed (0.5 -> 3.8s)
    for idx, dt in enumerate([0.7, 1.35, 1.95, 2.55, 3.15, 3.65]):
        _add_panned(sfx, sfx_water_droplet(base_hz=1150.0 + idx * 140.0, amp=0.17), dt, pan=-0.15 + idx * 0.06)
    # Gentle subterranean soil & root movement (2.8 -> 8.5s)
    _add_panned(sfx, sfx_soil_rustle(3.2, amp=0.14, seed=21), 2.8, pan=0.0)
    _add_panned(sfx, sfx_soil_rustle(2.8, amp=0.15, seed=22), 6.2, pan=0.05)
    # Sprouting & swaying grass rustle (7.2 -> 22.0s)
    _add_panned(sfx, sfx_grass_rustle(4.5, amp=0.11, seed=31), 7.2, pan=-0.2)
    _add_panned(sfx, sfx_grass_rustle(6.5, amp=0.12, seed=32), 14.5, pan=+0.25)
    # Morning birdsong across sunlit valley (9.5 -> 21.0s)
    for b_i, (bt, pan) in enumerate([(2.2, -0.55), (10.2, 0.45), (13.4, -0.40), (16.8, 0.55), (19.6, -0.25)]):
        _add_panned(sfx, sfx_birdsong_phrase(seed=100 + b_i * 7, amp=0.15), bt, pan=pan)

    # --- SCENE 2 (00:22 - 00:42): THE GRASSHOPPER ---
    _add_panned(sfx, sfx_wind_bed(21.5, 160, 620, amp=0.09, seed=201), 21.5, pan=-0.3)
    _add_panned(sfx, sfx_wind_bed(21.5, 175, 680, amp=0.09, seed=202), 21.5, pan=+0.3)
    # Meadow insect buzzing & hop wing-flutter (24.5s = Scene 2 lt 2.5s hop!)
    _add_panned(sfx, sfx_insect_buzz(4.2, freq_hz=178.0, amp=0.08, seed=211), 22.8, pan=-0.35)
    _add_panned(sfx, sfx_insect_buzz(1.1, freq_hz=230.0, amp=0.15, seed=212), 24.55, pan=-0.10)  # hop wing buzz!
    _add_panned(sfx, sfx_grass_rustle(3.6, amp=0.14, seed=215), 26.2, pan=0.15)  # climbing the hero blade!
    # Crisp rhythmic leaf-chewing bites synced directly to Scene 2 feeding (lt 9.5 -> 15.6s = global 31.5 -> 37.6s)!
    bite_t = 31.45
    b_k = 0
    while bite_t < 37.65:
        _add_panned(sfx, sfx_leaf_chew_bite(seed=240 + b_k, amp=0.17), bite_t, pan=0.12)
        bite_t += 0.34 + (b_k % 3) * 0.06
        b_k += 1
    _add_panned(sfx, sfx_birdsong_phrase(seed=260, amp=0.10), 38.8, pan=-0.50)

    # --- SCENE 3 (00:42 - 01:02): THE FROG HUNTS ---
    _add_panned(sfx, sfx_pond_ambience(21.5, amp=0.12, seed=301), 41.5, pan=-0.25)
    _add_panned(sfx, sfx_pond_ambience(21.5, amp=0.12, seed=302), 41.5, pan=+0.25)
    # Resonant pond frog croaks synced to throat swells (44.6s, 46.1s, 49.4s)!
    _add_panned(sfx, sfx_frog_croak(110.0, dur=0.52, amp=0.25), 44.65, pan=-0.28)
    _add_panned(sfx, sfx_frog_croak(116.0, dur=0.45, amp=0.22), 46.15, pan=-0.25)
    _add_panned(sfx, sfx_frog_croak(105.0, dur=0.42, amp=0.16), 60.20, pan=+0.22)
    # Dragonfly / pond insect buzzes
    _add_panned(sfx, sfx_insect_buzz(2.8, freq_hz=195.0, amp=0.07, seed=315), 43.2, pan=0.40)
    _add_panned(sfx, sfx_insect_buzz(1.6, freq_hz=215.0, amp=0.09, seed=316), 47.2, pan=0.35)  # grasshopper arriving!
    # Frog explosive jump takeoff (lt 13.25s = global 55.25s), wet tongue flick (55.68s) & landing splash (56.35s)!
    _add_panned(sfx, sfx_fast_whoosh(0.32, 220, 1450, amp=0.22, seed=321), 55.22, pan=-0.20)
    _add_panned(sfx, sfx_water_splash(0.45, amp=0.16, seed=322), 55.28, pan=-0.22)
    _add_panned(sfx, sfx_tongue_flick(amp=0.26), 55.65, pan=+0.08)
    _add_panned(sfx, sfx_water_splash(0.68, amp=0.24, seed=325), 56.35, pan=+0.24)

    # --- SCENE 4 (01:02 - 01:22): THE SNAKE ---
    _add_panned(sfx, sfx_wind_bed(21.5, 130, 520, amp=0.08, seed=401), 61.5, pan=0.0)
    # Panned dry leaves & scaly slithering tracking left -> right (62.5 -> 74.2s)!
    _add_panned(sfx, sfx_dry_leaves_slither(4.2, amp=0.15, seed=411), 62.5, pan=-0.45)
    _add_panned(sfx, sfx_dry_leaves_slither(4.2, amp=0.16, seed=412), 66.2, pan=-0.12)
    _add_panned(sfx, sfx_dry_leaves_slither(3.8, amp=0.15, seed=413), 70.0, pan=+0.10)
    # Subtle tense sibilant snake hiss during S-coil (lt 10.6 -> 12.2s = global 72.6 -> 74.2s)!
    _add_panned(sfx, sfx_snake_hiss(1.35, amp=0.17, seed=421), 72.55, pan=+0.05)
    # Frog panicked escape rustle & rapid snake strike swoosh (lt 12.65s = global 74.65s)!
    _add_panned(sfx, sfx_fast_whoosh(0.38, 280, 2650, amp=0.30, seed=425), 74.62, pan=+0.18)
    _add_panned(sfx, sfx_dry_leaves_slither(2.6, amp=0.14, seed=428), 75.2, pan=+0.25)

    # --- SCENE 5 (01:22 - 01:47): THE HAWK ---
    # Sweeping panoramic high-altitude wind across the valley
    _add_panned(sfx, sfx_wind_bed(26.0, 120, 740, amp=0.15, seed=501), 81.5, pan=-0.40)
    _add_panned(sfx, sfx_wind_bed(26.0, 135, 810, amp=0.15, seed=502), 81.5, pan=+0.40)
    # Authentic piercing Red-tailed Hawk screech calls (soaring entrance 84.0s & eye-lock dive start 91.5s)!
    _add_panned(sfx, sfx_hawk_call(1.50, amp=0.30, seed=511), 83.9, pan=-0.25)
    _add_panned(sfx, sfx_hawk_call(1.35, amp=0.28, seed=512), 91.8, pan=+0.12)
    # Deep rhythmic soaring wingbeats (83.2 -> 86.5s)
    for k, wt in enumerate([83.3, 84.1, 84.9, 85.7]):
        _add_panned(sfx, sfx_wing_flap(amp=0.18, seed=520 + k), wt, pan=-0.30 + k * 0.12)
    # Thrilling aerodynamic rushing dive wind (lt 11.4 -> 18.2s = global 93.4 -> 100.2s)!
    _add_panned(sfx, sfx_dive_wind(6.8, amp=0.31, seed=531), 93.4, pan=-0.12)
    _add_panned(sfx, sfx_dive_wind(6.8, amp=0.28, seed=532), 93.45, pan=+0.12)
    # Dramatic wing-brake flare & talon snatch swoosh (lt 18.5 -> 19.4s = global 100.5 -> 101.4s)!
    _add_panned(sfx, sfx_fast_whoosh(0.55, 190, 2200, amp=0.30, seed=541), 100.55, pan=0.05)
    _add_panned(sfx, sfx_dry_leaves_slither(1.1, amp=0.16, seed=542), 101.05, pan=0.10)
    # Powerful lifting ascent wingbeats carrying snake across golden sky (101.4 -> 106.2s)!
    for k, wt in enumerate([101.45, 102.20, 102.95, 103.75, 104.55, 105.40]):
        _add_panned(sfx, sfx_wing_flap(amp=0.26 * (0.92 ** k), seed=550 + k), wt, pan=-0.10 + k * 0.08)

    # --- SCENE 6 (01:47 - 02:04): THE COMPLETE FOOD CHAIN ---
    _add_panned(sfx, sfx_wind_bed(16.5, 150, 550, amp=0.07, seed=601), 107.0, pan=0.0)
    # Luminous crystalline energy-arrow chimes synced to each arrow illuminating!
    arrow_chimes = [(109.0, 74), (110.5, 76), (112.0, 78), (113.5, 81)]
    for idx, (ct, mn) in enumerate(arrow_chimes):
        _add_panned(sfx, sfx_energy_chime(mn, dur=1.8, amp=0.16), ct, pan=-0.45 + idx * 0.30)
    # Warm resonant concluding shimmer chord on "EVERY LIVING THING IS CONNECTED."
    for mn, pan in [(74, -0.35), (78, 0.0), (81, 0.35), (86, 0.0)]:
        _add_panned(sfx, sfx_energy_chime(mn, dur=3.2, amp=0.11), 118.6, pan=pan)

    # Subtle painterly air-brush transition swells at scene boundaries
    for bt in (22.0, 42.0, 62.0, 82.0, 107.0):
        _add_panned(sfx, sfx_fast_whoosh(0.85, 240, 1100, amp=0.10, seed=int(bt * 10)), bt - 0.42, pan=0.0)

    # Graceful ending fade-out
    fo = int(2.2 * sr)
    sfx[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)[:, None] ** 1.5

    # Save full SFX mix + individual scene SFX stems
    sfx_wav = os.path.join(audio_dir, "sound_effects_mix.wav")
    wavfile.write(sfx_wav, sr, np.clip(sfx * 32767.0, -32767, 32767).astype(np.int16))

    for sp in SCENE_SPECS:
        sid = int(sp["id"])
        s0 = int(round(float(sp["start"]) * sr))
        s1 = min(n_samples, int(round(float(sp["end"]) * sr)))
        stem_path = os.path.join(audio_dir, f"sfx_scene{sid}.wav")
        wavfile.write(stem_path, sr, np.clip(sfx[s0:s1] * 32767.0, -32767, 32767).astype(np.int16))

    return sfx


def build_master_soundtrack(
    audio_dir: str,
    total_duration: float = TOTAL_DURATION,
    force_tts: bool = False,
) -> str:
    """
    Build narration, original score, and wildlife SFX stems, then combine them
    with smooth automatic sidechain ducking & soft broadcast limiting into
    `audio/master_soundtrack.wav` (44.1 kHz stereo).
    """
    os.makedirs(audio_dir, exist_ok=True)

    print("[audio] Building documentary narration stem...")
    _, narr_stereo, duck_env = generate_all_narration(audio_dir, total_duration=total_duration, force=force_tts)

    music_path = os.path.join(audio_dir, "background_music.wav")
    print("[audio] Composing original 2-minute cinematic orchestral/acoustic score...")
    music_stereo = generate_original_score(music_path, total_duration=total_duration)

    print("[audio] Synthesizing layered wildlife foley & environmental soundscapes...")
    sfx_stereo = generate_sfx_mix(audio_dir, total_duration=total_duration)

    # Ensure identical exact sample count across all three stems
    n = int(round(total_duration * SAMPLE_RATE))

    def _fit(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == n:
            return arr
        if arr.shape[0] > n:
            return arr[:n]
        pad = np.zeros((n - arr.shape[0],) + arr.shape[1:], dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)

    narr_stereo = _fit(narr_stereo)
    music_stereo = _fit(music_stereo)
    sfx_stereo = _fit(sfx_stereo)
    duck_env = _fit(duck_env)

    # Smooth broadcast sidechain ducking: music dips ~5.5 dB & SFX dips ~3.2 dB when narrator speaks!
    music_gain = (1.0 - 0.46 * duck_env)[:, None] * 0.56
    sfx_gain = (1.0 - 0.30 * duck_env)[:, None] * 1.18
    narr_gain = 1.00

    master = music_stereo * music_gain + sfx_stereo * sfx_gain + narr_stereo * narr_gain

    # Soft-knee warm mastering limiter & final normalization to -1.0 dBFS (zero digital clipping!)
    master = np.tanh(master * 1.12) / 1.12
    peak = float(np.max(np.abs(master)))
    if peak > 1e-6:
        master = master * (0.90 / peak)

    master_wav = os.path.join(audio_dir, "master_soundtrack.wav")
    wavfile.write(
        master_wav,
        SAMPLE_RATE,
        np.clip(master * 32767.0, -32767, 32767).astype(np.int16),
    )

    rms_db = 20.0 * math.log10(max(float(np.sqrt(np.mean(master ** 2))), 1e-8))
    peak_db = 20.0 * math.log10(max(float(np.max(np.abs(master))), 1e-8))
    print(f"[audio] Mastered soundtrack saved: {master_wav} ({n/SAMPLE_RATE:.2f}s | RMS={rms_db:.1f} dBFS | Peak={peak_db:.1f} dBFS)")
    return master_wav


if __name__ == "__main__":
    audio_out = os.path.join(os.path.dirname(__file__), "audio")
    build_master_soundtrack(audio_out, total_duration=TOTAL_DURATION, force_tts=False)

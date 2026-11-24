"""
music.py — Original 124-Second Cinematic Instrumental Nature Film Score
=======================================================================

Synthesizes a complete, royalty-free, multi-movement acoustic & orchestral score
entirely in Python (NumPy + SciPy DSP) synchronized to the 6-scene food-chain
timeline:
  - Scene 1 (00:00-00:22): Pastoral D-major dawn — acoustic harp arpeggios,
                           warm string pad bloom, tender airy wooden flute melody.
  - Scene 2 (00:22-00:42): Bright sunlit meadow — playful plucked harp, light
                           pizzicato strings, gentle dancing flute motifs.
  - Scene 3 (00:42-01:02): Tranquil pond -> tense stillness during frog eye lock
                           -> brisk joyful woodwind & harp spring on catch.
  - Scene 4 (01:02-01:22): B-minor creeping tension — deep brooding cello pedal,
                           hushed rhythmic low-string ostinato & muted strike swell.
  - Scene 5 (01:22-01:47): Sweeping panoramic rise — soaring strings, heroic warm
                           French horn theme, building timpani roll & triumphant apex.
  - Scene 6 (01:47-02:04): Radiant educational resolution — shimmering harp, lush
                           full strings, warm felt piano resolving to Dmaj9 peace.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile

from animation import SAMPLE_RATE, TOTAL_DURATION

# Musical pitch helper (MIDI note -> Hz, A4 = 69 = 440 Hz)
def midi_hz(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))


# =============================================================================
# Organic Acoustic & Orchestral Instrument Synthesizers
# =============================================================================

def _synth_harp_note(freq: float, dur: float, amp: float = 0.25, sr: int = SAMPLE_RATE, bright: float = 0.55) -> np.ndarray:
    """Karplus-Strong plucked concert harp waveguide with warm body filtering."""
    n = int(round(dur * sr))
    if n <= 8 or freq < 35.0:
        return np.zeros(max(1, n), dtype=np.float32)

    period = max(2, int(round(sr / freq)))
    rng = np.random.default_rng(int(freq * 100) % 100000)
    # Softly filtered noise pluck excitation (resembles warm finger-plucked nylon harp string)
    exc = rng.uniform(-1.0, 1.0, period).astype(np.float64)
    exc -= exc.mean()
    # Smooth initial burst for warm harp tone
    b_lp, a_lp = signal.butter(1, min(0.45, (freq * (3.5 + 4.0 * bright)) / (sr * 0.5)), btype="low")
    exc = signal.lfilter(b_lp, a_lp, exc)

    out = np.zeros(n, dtype=np.float64)
    buf = exc.copy()
    decay = 0.9965 if freq < 350 else 0.9942
    # Vectorized block-by-block Karplus-Strong propagation (10x faster than Python sample loop!)
    pos = 0
    while pos < n:
        take = min(period, n - pos)
        out[pos : pos + take] = buf[:take]
        # Low-pass feedback averaging across one full period
        nxt = np.empty_like(buf)
        nxt[0] = 0.5 * (buf[0] + buf[-1]) * decay
        nxt[1:] = 0.5 * (buf[1:] + buf[:-1]) * decay
        buf = nxt
        pos += take

    # Gentle attack smoothing & natural exponential release
    att = min(int(0.008 * sr), n // 4)
    if att > 1:
        out[:att] *= np.linspace(0.0, 1.0, att)
    rel = min(int(0.18 * sr), n // 2)
    if rel > 1:
        out[-rel:] *= np.linspace(1.0, 0.0, rel) ** 1.5

    peak = np.max(np.abs(out))
    if peak > 1e-8:
        out = out * (amp / peak)
    return out.astype(np.float32)


def _synth_felt_piano(freq: float, dur: float, amp: float = 0.22, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Warm intimate felt-piano tone with natural harmonic decay & soft hammer bloom."""
    n = int(round(dur * sr))
    if n <= 8:
        return np.zeros(max(1, n), dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / sr
    # Warm harmonic series with faster high-partial damping (felt hammer character)
    partials = [(1.0, 1.0, 1.05), (2.0, 0.44, 1.75), (3.0, 0.20, 2.7), (4.0, 0.08, 4.1), (5.0, 0.03, 5.8)]
    y = np.zeros(n, dtype=np.float64)
    for mult, w, dec in partials:
        if freq * mult < sr * 0.42:
            env = np.exp(-t * dec * (0.65 + 0.0008 * freq))
            y += w * env * np.sin(2.0 * math.pi * freq * mult * t)
    # Soft 18ms attack + smooth release
    att = min(int(0.018 * sr), n // 4)
    if att > 1:
        y[:att] *= np.sin(np.linspace(0.0, math.pi * 0.5, att))
    rel = min(int(0.25 * sr), n // 2)
    if rel > 1:
        y[-rel:] *= np.linspace(1.0, 0.0, rel) ** 1.6
    peak = np.max(np.abs(y))
    if peak > 1e-8:
        y *= amp / peak
    return y.astype(np.float32)


def _synth_string_voice(
    freq: float,
    dur: float,
    amp: float = 0.14,
    sr: int = SAMPLE_RATE,
    attack: float = 0.45,
    release: float = 0.65,
    vibrato_hz: float = 5.1,
    vibrato_depth: float = 0.0045,
    warmth: float = 1.0,
) -> np.ndarray:
    """Lush bowed string ensemble note (3 detuned chorused voices with warm harmonic rolloff)."""
    n = int(round(dur * sr))
    if n <= 16:
        return np.zeros(max(1, n), dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / sr
    y = np.zeros(n, dtype=np.float64)

    # 3 slightly detuned ensemble players for natural orchestral warmth
    detunes = [-0.0022, 0.0, +0.0025]
    for idx, det in enumerate(detunes):
        f0 = freq * (1.0 + det)
        vib = vibrato_depth * (1.0 - np.exp(-t * 2.2)) * np.sin(2.0 * math.pi * (vibrato_hz + idx * 0.23) * t + idx)
        phase = 2.0 * math.pi * np.cumsum(f0 * (1.0 + vib)) / sr
        voice = np.zeros(n, dtype=np.float64)
        # Bowed saw-like harmonic rolloff shaped softly for silky strings
        for h in range(1, 9):
            if f0 * h >= sr * 0.38:
                break
            weight = (1.0 / (h ** (1.28 + 0.25 * warmth))) * (1.0 if h % 2 == 1 else 0.72)
            voice += weight * np.sin(h * phase + idx * 0.7)
        y += voice / 3.0

    # Smooth low-pass silk filter
    cutoff = min(sr * 0.35, max(850.0, freq * (5.5 - 1.5 * warmth)))
    sos = signal.butter(2, cutoff, btype="low", fs=sr, output="sos")
    y = signal.sosfilt(sos, y)

    # Expressive bow swell envelope
    env = np.ones(n, dtype=np.float64)
    a_n = min(int(attack * sr), n // 2)
    r_n = min(int(release * sr), n - a_n)
    if a_n > 1:
        env[:a_n] = np.sin(np.linspace(0.0, math.pi * 0.5, a_n)) ** 1.3
    if r_n > 1:
        env[-r_n:] = np.cos(np.linspace(0.0, math.pi * 0.5, r_n)) ** 1.4
    y *= env
    peak = np.max(np.abs(y))
    if peak > 1e-8:
        y *= amp / peak
    return y.astype(np.float32)


def _synth_wooden_flute(freq: float, dur: float, amp: float = 0.16, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Airy pastoral wooden documentary flute with expressive breath chiff & warm vibrato."""
    n = int(round(dur * sr))
    if n <= 16:
        return np.zeros(max(1, n), dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / sr
    vib = 0.0055 * (1.0 - np.exp(-t * 3.0)) * np.sin(2.0 * math.pi * 5.4 * t)
    phase = 2.0 * math.pi * np.cumsum(freq * (1.0 + vib)) / sr

    y = 0.80 * np.sin(phase) + 0.16 * np.sin(2.0 * phase) + 0.05 * np.sin(3.0 * phase)
    # Delicate breathy air chiff band-passed around the note frequency
    rng = np.random.default_rng(int(freq * 37) % 99991)
    breath = rng.normal(0.0, 1.0, n)
    lo = max(120.0, freq * 0.85)
    hi = min(sr * 0.42, freq * 2.2)
    sos = signal.butter(2, [lo, hi], btype="bandpass", fs=sr, output="sos")
    breath = signal.sosfilt(sos, breath) * (0.06 + 0.10 * np.exp(-t * 9.0))
    y += breath

    # Smooth lyric breath envelope
    env = np.ones(n, dtype=np.float64)
    a_n = min(int(0.11 * sr), n // 3)
    r_n = min(int(0.22 * sr), n // 2)
    if a_n > 1:
        env[:a_n] = np.sin(np.linspace(0.0, math.pi * 0.5, a_n)) ** 1.2
    if r_n > 1:
        env[-r_n:] = np.cos(np.linspace(0.0, math.pi * 0.5, r_n)) ** 1.5
    y *= env
    peak = np.max(np.abs(y))
    if peak > 1e-8:
        y *= amp / peak
    return y.astype(np.float32)


def _synth_french_horn(freq: float, dur: float, amp: float = 0.22, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Warm heroic cinema French horn with dynamic brassy bloom."""
    n = int(round(dur * sr))
    if n <= 16:
        return np.zeros(max(1, n), dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / sr
    vib = 0.003 * (1.0 - np.exp(-t * 2.5)) * np.sin(2.0 * math.pi * 4.8 * t)
    phase = 2.0 * math.pi * np.cumsum(freq * (1.0 + vib)) / sr

    bloom = np.sin(np.clip(t / max(dur, 0.2), 0.0, 1.0) * math.pi) ** 0.7
    y = np.zeros(n, dtype=np.float64)
    weights = [1.0, 0.82, 0.58, 0.36, 0.22, 0.12]
    for h_i, w in enumerate(weights, start=1):
        if freq * h_i < sr * 0.38:
            # Higher brass harmonics swell naturally during the horn's crescendo
            h_env = 1.0 if h_i <= 2 else (0.4 + 0.6 * bloom)
            y += w * h_env * np.sin(h_i * phase)

    sos = signal.butter(2, min(sr * 0.35, freq * 4.2), btype="low", fs=sr, output="sos")
    y = signal.sosfilt(sos, y)

    env = np.ones(n, dtype=np.float64)
    a_n = min(int(0.22 * sr), n // 3)
    r_n = min(int(0.35 * sr), n // 2)
    if a_n > 1:
        env[:a_n] = np.sin(np.linspace(0.0, math.pi * 0.5, a_n)) ** 1.3
    if r_n > 1:
        env[-r_n:] = np.cos(np.linspace(0.0, math.pi * 0.5, r_n)) ** 1.4
    y *= env
    peak = np.max(np.abs(y))
    if peak > 1e-8:
        y *= amp / peak
    return y.astype(np.float32)


def _synth_timpani_hit(freq: float = 73.4, dur: float = 1.4, amp: float = 0.26, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resonant orchestral timpani drum strike with slight membrane pitch drop."""
    n = int(round(dur * sr))
    t = np.arange(n, dtype=np.float64) / sr
    # Slight initial membrane tension pitch drop (+12% -> settled pitch)
    f_inst = freq * (1.0 + 0.12 * np.exp(-t * 28.0))
    ph = 2.0 * math.pi * np.cumsum(f_inst) / sr
    body = (
        np.sin(ph) * np.exp(-t * 2.4)
        + 0.45 * np.sin(1.50 * ph) * np.exp(-t * 4.5)
        + 0.20 * np.sin(2.00 * ph) * np.exp(-t * 7.0)
    )
    # Soft felt-mallet thump
    rng = np.random.default_rng(777)
    thump = rng.normal(0.0, 1.0, n) * np.exp(-t * 65.0)
    sos = signal.butter(2, 260.0, btype="low", fs=sr, output="sos")
    thump = signal.sosfilt(sos, thump) * 0.55
    y = body + thump
    att = min(int(0.005 * sr), n // 8)
    if att > 1:
        y[:att] *= np.linspace(0.0, 1.0, att)
    peak = np.max(np.abs(y))
    if peak > 1e-8:
        y *= amp / peak
    return y.astype(np.float32)


# =============================================================================
# Stereo Concert Hall Algorithmic Reverb
# =============================================================================

def _apply_concert_reverb(stereo: np.ndarray, wet: float = 0.28, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Warm stereo multi-tap algorithmic concert-hall reverb (fast NumPy delay taps)."""
    n = stereo.shape[0]
    rev_l = np.zeros(n, dtype=np.float32)
    rev_r = np.zeros(n, dtype=np.float32)

    # Prime-spaced stereo reflections with exponential concert-hall decay
    taps_l = [(0.031, 0.40), (0.053, 0.34), (0.089, 0.28), (0.137, 0.23), (0.197, 0.19), (0.281, 0.14), (0.389, 0.10), (0.521, 0.07)]
    taps_r = [(0.037, 0.40), (0.061, 0.34), (0.097, 0.28), (0.149, 0.23), (0.211, 0.19), (0.293, 0.14), (0.409, 0.10), (0.547, 0.07)]

    mono_in = 0.5 * (stereo[:, 0] + stereo[:, 1])
    # Warm high-frequency hall absorption on reverb send
    sos = signal.butter(2, 3400.0, btype="low", fs=sr, output="sos")
    send = signal.sosfilt(sos, mono_in).astype(np.float32)

    for d_s, g in taps_l:
        d = int(round(d_s * sr))
        if d < n:
            rev_l[d:] += send[:-d] * g
    for d_s, g in taps_r:
        d = int(round(d_s * sr))
        if d < n:
            rev_r[d:] += send[:-d] * g

    # Second diffusion pass for silky smooth tail
    for d_s, g in [(0.173, 0.38), (0.311, 0.28), (0.467, 0.18)]:
        d = int(round(d_s * sr))
        if d < n:
            rev_l[d:] += rev_r[:-d] * g * 0.6
            rev_r[d:] += rev_l[:-d] * g * 0.6

    out = stereo.copy()
    out[:, 0] = stereo[:, 0] * (1.0 - wet * 0.35) + rev_l * wet
    out[:, 1] = stereo[:, 1] * (1.0 - wet * 0.35) + rev_r * wet
    return out


def _add_stereo(mix: np.ndarray, mono: np.ndarray, start_s: float, pan: float = 0.0, gain: float = 1.0, sr: int = SAMPLE_RATE) -> None:
    """Add a mono audio event into stereo `mix` at `start_s` with equal-power pan in [-1, +1]."""
    s0 = int(round(start_s * sr))
    if s0 >= mix.shape[0] or s0 + len(mono) <= 0:
        return
    src_start = max(0, -s0)
    dst_start = max(0, s0)
    count = min(len(mono) - src_start, mix.shape[0] - dst_start)
    if count <= 0:
        return
    ang = (pan * 0.5 + 0.5) * math.pi * 0.5
    gl = math.cos(ang) * gain
    gr = math.sin(ang) * gain
    seg = mono[src_start : src_start + count]
    mix[dst_start : dst_start + count, 0] += seg * gl
    mix[dst_start : dst_start + count, 1] += seg * gr


# =============================================================================
# Master 6-Movement Score Composer
# =============================================================================

def generate_original_score(
    out_wav_path: str,
    total_duration: float = TOTAL_DURATION,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Compose and render the complete original 124-second stereo nature-film score."""
    n_samples = int(round(total_duration * sr))
    mix = np.zeros((n_samples, 2), dtype=np.float32)

    # -------------------------------------------------------------------------
    # MOVEMENT 1 (00:00 - 00:22): LIFE BEGINS — Pastoral D Major Dawn
    # Chords: Dmaj7 (0-5.5), Gmaj7 (5.5-11), Bm7 (11-16.5), Aadd9 (16.5-22)
    # -------------------------------------------------------------------------
    m1_chords = [
        (0.2,  5.6, [50, 57, 62, 66, 69]),   # Dmaj (warm open)
        (5.6,  5.6, [43, 55, 59, 62, 66]),   # Gmaj7
        (11.0, 5.6, [47, 54, 59, 62, 66]),   # Bm7
        (16.4, 5.8, [45, 52, 57, 61, 64]),   # Aadd9
    ]
    for st, dur, notes in m1_chords:
        # Soft blooming string pads (start gently after seed begins waking)
        pad_gain = 0.55 if st < 5.0 else 0.92
        for idx, mn in enumerate(notes[:4]):
            pan = -0.45 + idx * 0.30
            s_v = _synth_string_voice(midi_hz(mn), dur + 0.8, amp=0.075 * pad_gain, attack=1.1, release=1.2)
            _add_stereo(mix, s_v, st, pan=pan)
        # Sparkling morning harp arpeggio
        arp = notes + notes[1:-1][::-1]
        step = 0.46
        t_cur = st
        k = 0
        while t_cur < st + dur - 0.15:
            mn = arp[k % len(arp)] + (12 if (k // len(arp)) % 2 == 1 else 0)
            h_n = _synth_harp_note(midi_hz(mn), 2.1, amp=0.145)
            _add_stereo(mix, h_n, t_cur, pan=-0.35 + 0.70 * ((k % 5) / 4.0))
            t_cur += step
            k += 1

    # Tender morning flute melody entering as the grass breaks through into sunlight (8.5s - 21.5s)!
    flute_m1 = [
        (8.6, 1.3, 74), (9.9, 0.9, 76), (10.8, 1.8, 78),
        (13.0, 1.0, 81), (14.0, 0.9, 78), (14.9, 1.6, 74),
        (17.0, 1.2, 76), (18.2, 1.0, 73), (19.2, 2.3, 74),
    ]
    for st, dur, mn in flute_m1:
        _add_stereo(mix, _synth_wooden_flute(midi_hz(mn), dur + 0.15, amp=0.135), st, pan=0.18)

    # -------------------------------------------------------------------------
    # MOVEMENT 2 (00:22 - 00:42): THE GRASSHOPPER — Lively Sunlit Meadow
    # -------------------------------------------------------------------------
    m2_chords = [
        (22.0, 5.0, [50, 57, 62, 66]),       # D
        (27.0, 5.0, [43, 55, 59, 62]),       # G
        (32.0, 5.0, [45, 52, 57, 64]),       # A
        (37.0, 5.2, [50, 57, 62, 66]),       # D
    ]
    for st, dur, notes in m2_chords:
        for idx, mn in enumerate(notes[:3]):
            s_v = _synth_string_voice(midi_hz(mn), dur + 0.6, amp=0.068, attack=0.8, release=0.9)
            _add_stereo(mix, s_v, st, pan=-0.35 + idx * 0.35)
        # Bouncy pastoral harp & soft felt piano pulse
        t_cur = st
        k = 0
        while t_cur < st + dur - 0.1:
            mn = notes[k % len(notes)] + (12 if k % 3 == 2 else 0)
            _add_stereo(mix, _synth_harp_note(midi_hz(mn), 1.4, amp=0.135, bright=0.7), t_cur, pan=-0.30 + (k % 4) * 0.20)
            if k % 4 == 0:
                _add_stereo(mix, _synth_felt_piano(midi_hz(notes[0] + 12), 1.9, amp=0.12), t_cur, pan=-0.20)
            t_cur += 0.38
            k += 1

    flute_m2 = [
        (23.5, 0.7, 78), (24.2, 0.6, 81), (24.8, 1.2, 78),
        (27.8, 0.7, 74), (28.5, 0.6, 76), (29.1, 1.5, 79),
        (34.2, 0.8, 76), (35.0, 0.7, 74), (35.7, 1.4, 73), (37.2, 2.2, 74),
    ]
    for st, dur, mn in flute_m2:
        _add_stereo(mix, _synth_wooden_flute(midi_hz(mn), dur + 0.12, amp=0.12), st, pan=0.22)

    # -------------------------------------------------------------------------
    # MOVEMENT 3 (00:42 - 01:02): THE FROG HUNTS — Shimmering Pond -> Stillness -> Leap
    # -------------------------------------------------------------------------
    # Tranquil shimmering water harp & piano (42.0 -> 51.0s)
    m3_calm = [
        (42.0, 4.6, [43, 50, 59, 62, 66]),   # Gmaj7
        (46.6, 4.6, [45, 52, 57, 64, 67]),   # A7sus4
    ]
    for st, dur, notes in m3_calm:
        for idx, mn in enumerate(notes[:3]):
            _add_stereo(mix, _synth_string_voice(midi_hz(mn), dur + 0.8, amp=0.068, attack=0.9, release=1.0), st, pan=-0.4 + idx * 0.4)
        t_cur = st
        k = 0
        while t_cur < st + dur - 0.1:
            mn = notes[k % len(notes)] + 12
            _add_stereo(mix, _synth_harp_note(midi_hz(mn), 1.8, amp=0.12), t_cur, pan=-0.4 + (k % 5) * 0.2)
            t_cur += 0.44
            k += 1

    # Hushed expectant stillness during frog eye close-up & crouch (51.0 -> 55.2s)!
    for mn, pan in [(47, -0.35), (54, 0.0), (62, 0.35)]:
        _add_stereo(mix, _synth_string_voice(midi_hz(mn), 4.6, amp=0.072, attack=0.9, release=0.7), 51.0, pan=pan)
    # Soft single felt piano pulse ticks building anticipation
    for tp in (51.8, 53.0, 54.1):
        _add_stereo(mix, _synth_felt_piano(midi_hz(66), 1.2, amp=0.11), tp, pan=0.1)

    # Crisp springy leap flourish & warm pond resolution (55.2 -> 62.0s)!
    _add_stereo(mix, _synth_harp_note(midi_hz(62), 1.4, amp=0.18, bright=0.8), 55.25, pan=-0.2)
    _add_stereo(mix, _synth_harp_note(midi_hz(69), 1.4, amp=0.19, bright=0.8), 55.42, pan=0.0)
    _add_stereo(mix, _synth_harp_note(midi_hz(74), 1.8, amp=0.20, bright=0.85), 55.58, pan=0.2)
    _add_stereo(mix, _synth_wooden_flute(midi_hz(78), 1.5, amp=0.14), 56.3, pan=0.15)
    _add_stereo(mix, _synth_wooden_flute(midi_hz(74), 2.1, amp=0.12), 57.8, pan=0.15)
    for mn, pan in [(43, -0.4), (50, 0.0), (59, 0.4)]:
        _add_stereo(mix, _synth_string_voice(midi_hz(mn), 5.8, amp=0.072, attack=0.7, release=1.2), 56.2, pan=pan)

    # -------------------------------------------------------------------------
    # MOVEMENT 4 (01:02 - 01:22): THE SNAKE — Tense B Minor Stealth & Strike
    # -------------------------------------------------------------------------
    # Deep sustained cello/bass pedal (B1 -> F#1 -> G1 -> F#1) creating authentic nature-doc tension!
    m4_pedals = [
        (62.0, 5.5, [35, 47, 54]),   # Bm dark root
        (67.5, 5.0, [37, 49, 55]),   # C# dim tension
        (72.5, 2.8, [42, 49, 58]),   # F#7 dominant suspense right into strike!
        (75.2, 6.8, [35, 47, 54, 57]),  # Bm7 solemn aftermath
    ]
    for st, dur, notes in m4_pedals:
        for idx, mn in enumerate(notes):
            amp = 0.115 if idx == 0 else 0.070
            _add_stereo(mix, _synth_string_voice(midi_hz(mn), dur + 0.7, amp=amp, attack=0.8, release=0.9, warmth=1.3), st, pan=-0.35 + idx * 0.28)

    # Hushed creeping rhythmic low plucked ostinato accelerating subtly toward strike (63.5 -> 74.6s)
    t_cur = 63.5
    step_i = 0
    ost_notes = [47, 54, 50, 54, 47, 55, 54, 50]
    while t_cur < 74.5:
        mn = ost_notes[step_i % len(ost_notes)]
        urg = (t_cur - 63.5) / 11.0
        _add_stereo(mix, _synth_harp_note(midi_hz(mn), 0.75, amp=0.11 + 0.06 * urg, bright=0.25), t_cur, pan=-0.25 if step_i % 2 == 0 else 0.25)
        if t_cur > 71.0 and step_i % 2 == 0:
            _add_stereo(mix, _synth_timpani_hit(freq=midi_hz(35), dur=0.9, amp=0.14 + 0.08 * urg), t_cur, pan=0.0)
        t_cur += max(0.32, 0.56 - 0.22 * urg)
        step_i += 1

    # Sharp dramatic strike downbeat accent at 74.7s (Scene 4 lt = 12.7s)!
    _add_stereo(mix, _synth_timpani_hit(freq=midi_hz(35), dur=1.8, amp=0.30), 74.68, pan=0.0)
    _add_stereo(mix, _synth_string_voice(midi_hz(59), 2.2, amp=0.12, attack=0.06, release=1.4), 74.70, pan=-0.3)
    _add_stereo(mix, _synth_string_voice(midi_hz(66), 2.2, amp=0.11, attack=0.06, release=1.4), 74.70, pan=0.3)

    # -------------------------------------------------------------------------
    # MOVEMENT 5 (01:22 - 01:47): THE HAWK — Panoramic Rise, Stoop & Triumph!
    # -------------------------------------------------------------------------
    # Sweeping open D-major/G-major mountain-sky strings & French Horn heroic theme!
    m5_chords = [
        (82.0, 6.2, [38, 50, 57, 62, 66]),   # Wide soaring D major
        (88.2, 5.2, [43, 55, 59, 62, 67]),   # G major panoramic expanse
        (93.4, 4.4, [45, 52, 57, 64, 69]),   # A sus4 -> A (diving buildup!)
        (97.8, 3.6, [47, 54, 59, 66, 71]),   # Bm thrilling dive climax!
        (101.2, 6.0, [38, 50, 57, 62, 69, 74]),  # GLORIOUS TRIUMPHANT D MAJOR APEX!
    ]
    for st, dur, notes in m5_chords:
        is_apex = st >= 100.5
        sc = 1.32 if is_apex else 1.0
        for idx, mn in enumerate(notes):
            pan = -0.55 + (idx / float(max(1, len(notes) - 1))) * 1.10
            _add_stereo(mix, _synth_string_voice(midi_hz(mn), dur + 1.0, amp=0.088 * sc, attack=0.75, release=1.2, warmth=0.8), st, pan=pan)

    # Noble French horn soaring theme (84.0 -> 91.5s) & heroic ascent theme (101.2 -> 106.5s)!
    horn_phrases = [
        (83.5, 2.2, 57), (85.7, 1.5, 62), (87.2, 2.6, 66), (89.8, 2.4, 64),
        # Stoop urgency horn call
        (93.8, 1.4, 62), (95.2, 1.4, 64), (96.6, 1.6, 66), (98.2, 2.4, 69),
        # Triumphant catch & soaring skyward brass fanfare!
        (101.2, 1.6, 62), (102.8, 1.4, 66), (104.2, 2.8, 69),
    ]
    for st, dur, mn in horn_phrases:
        _add_stereo(mix, _synth_french_horn(midi_hz(mn), dur + 0.2, amp=0.175), st, pan=-0.12)

    # Building orchestral timpani roll driving the thrilling hawk stoop dive (93.6 -> 101.2s)!
    t_roll = 93.6
    r_idx = 0
    while t_roll < 101.1:
        prog = (t_roll - 93.6) / 7.5
        hit_amp = 0.08 + 0.22 * (prog ** 1.35)
        f_timp = midi_hz(38) if r_idx % 2 == 0 else midi_hz(45)
        _add_stereo(mix, _synth_timpani_hit(freq=f_timp, dur=0.75, amp=hit_amp), t_roll, pan=-0.18 if r_idx % 2 == 0 else 0.18)
        t_roll += max(0.14, 0.36 - 0.20 * prog)
        r_idx += 1
    # Thunderous triumphant impact timpani at exact snatch (101.15s = Scene 5 lt 19.15s)!
    _add_stereo(mix, _synth_timpani_hit(freq=midi_hz(38), dur=2.6, amp=0.38), 101.15, pan=0.0)

    # -------------------------------------------------------------------------
    # MOVEMENT 6 (01:47 - 02:04): THE COMPLETE FOOD CHAIN — Radiant Resolution
    # -------------------------------------------------------------------------
    m6_chords = [
        (107.0, 4.2, [43, 50, 59, 62, 66]),      # Gmaj7
        (111.2, 4.2, [45, 52, 57, 61, 64]),      # Aadd9
        (115.4, 7.6, [38, 50, 57, 62, 66, 69, 76]),  # Radiant Dmaj9 final resolution!
    ]
    for st, dur, notes in m6_chords:
        for idx, mn in enumerate(notes[:5]):
            pan = -0.50 + idx * 0.25
            _add_stereo(mix, _synth_string_voice(midi_hz(mn), dur + 1.0, amp=0.082, attack=0.95, release=2.2), st, pan=pan)
        # Shimmering concert harp & warm felt piano resolving peacefully
        t_cur = st
        k = 0
        while t_cur < min(st + dur - 0.1, 121.2):
            mn = notes[k % len(notes)] + (12 if k % 3 == 1 else 0)
            _add_stereo(mix, _synth_harp_note(midi_hz(mn), 2.4, amp=0.14, bright=0.65), t_cur, pan=-0.40 + (k % 5) * 0.20)
            if k % 4 == 0:
                _add_stereo(mix, _synth_felt_piano(midi_hz(notes[0] + 12), 2.8, amp=0.13), t_cur, pan=0.0)
            t_cur += 0.48
            k += 1

    # Gentle resolving wooden flute echoes
    for st, dur, mn in [(108.5, 1.6, 74), (110.2, 1.4, 76), (111.8, 2.2, 78), (115.6, 3.6, 74)]:
        _add_stereo(mix, _synth_wooden_flute(midi_hz(mn), dur + 0.3, amp=0.115), st, pan=0.15)

    # Apply silky stereo concert hall reverb + graceful fade-in/out + soft saturation mastering
    mix = _apply_concert_reverb(mix, wet=0.28, sr=sr)

    # Gentle opening fade-in (1.2s) & serene ending fade-out (2.5s)
    fi = int(1.2 * sr)
    fo = int(2.6 * sr)
    mix[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None] ** 1.3
    mix[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)[:, None] ** 1.6

    # Gentle warm limiter & normalize to -5 dBFS headroom for mixing with SFX & narration
    peak = float(np.max(np.abs(mix)))
    if peak > 1e-6:
        mix = mix * (0.62 / peak)
    mix = np.tanh(mix * 1.15) / 1.15

    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
    wavfile.write(
        out_wav_path,
        sr,
        np.clip(mix * 32767.0, -32767, 32767).astype(np.int16),
    )
    return mix.astype(np.float32)


if __name__ == "__main__":
    audio_dir = os.path.join(os.path.dirname(__file__), "audio")
    out_p = os.path.join(audio_dir, "background_music.wav")
    score = generate_original_score(out_p, total_duration=TOTAL_DURATION)
    print(f"Generated original film score: {out_p} ({len(score)/SAMPLE_RATE:.2f}s, peak={np.max(np.abs(score)):.3f})")

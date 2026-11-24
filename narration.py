"""
narration.py — Documentary Narration Synthesis & Timeline Synchronization
=========================================================================

Generates warm, natural wildlife-documentary narration for all 6 scenes using
Microsoft Neural TTS via `edge-tts` (with automatic fallback to `espeak-ng` if
offline), converts every clip to 44.1 kHz stereo WAV, applies subtle broadcast
warmth EQ + gentle compression, and builds a full timeline-synchronized
narration stem plus JSON timing metadata shared across video and audio modules.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile

SAMPLE_RATE = 44100
DEFAULT_VOICE = "en-GB-RyanNeural"
DEFAULT_RATE = "-8%"
DEFAULT_PITCH = "-3Hz"


@dataclass
class NarrationCue:
    scene_id: int
    key: str
    start_time: float
    text: str
    rate: str = DEFAULT_RATE
    pitch: str = DEFAULT_PITCH
    duration: float = 0.0
    end_time: float = 0.0
    wav_path: str = ""


# Exact narration script specified by the director, placed cleanly inside each
# scene window with breathing room at scene openings and action peaks.
NARRATION_SCRIPT: List[NarrationCue] = [
    NarrationCue(
        scene_id=1,
        key="scene1",
        start_time=4.8,
        text=(
            "Every food chain begins with a producer. "
            "In this grassland, the producer is grass. "
            "Through photosynthesis, grass uses sunlight to make its own food."
        ),
        rate="-8%",
        pitch="-3Hz",
    ),
    NarrationCue(
        scene_id=2,
        key="scene2",
        start_time=25.8,
        text=(
            "The grasshopper feeds on the grass. "
            "It depends on the producer for energy. "
            "This makes it a primary consumer."
        ),
        rate="-8%",
        pitch="-3Hz",
    ),
    NarrationCue(
        scene_id=3,
        key="scene3",
        start_time=44.8,
        text=(
            "Next comes the frog. "
            "It hunts the grasshopper for food. "
            "The frog is a secondary consumer because it eats the primary consumer."
        ),
        rate="-8%",
        pitch="-3Hz",
    ),
    NarrationCue(
        scene_id=4,
        key="scene4",
        start_time=65.0,
        text=(
            "The snake feeds on the frog. "
            "This makes it a tertiary consumer. "
            "Energy continues to move from one organism to another."
        ),
        rate="-9%",
        pitch="-4Hz",
    ),
    NarrationCue(
        scene_id=5,
        key="scene5",
        start_time=86.0,
        text=(
            "Finally, the hawk catches the snake. "
            "It is the top predator in this food chain. "
            "It receives energy from the organisms that came before it."
        ),
        rate="-8%",
        pitch="-3Hz",
    ),
    NarrationCue(
        scene_id=6,
        key="scene6_main",
        start_time=108.0,
        text=(
            "This is a food chain. "
            "It shows how energy moves from one living organism to another. "
            "Every organism plays an important role in the balance of nature."
        ),
        rate="-9%",
        pitch="-3Hz",
    ),
    NarrationCue(
        scene_id=6,
        key="scene6_closing",
        start_time=119.2,
        text="Every living thing is connected.",
        rate="-13%",
        pitch="-4Hz",
    ),
]


async def _synthesize_edge_tts(
    text: str, out_mp3: str, voice: str, rate: str, pitch: str
) -> None:
    import edge_tts

    comm = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    await comm.save(out_mp3)


def _mp3_to_wav(mp3_path: str, wav_path: str, sr: int = SAMPLE_RATE) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        mp3_path,
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-sample_fmt",
        "s16",
        wav_path,
    ]
    subprocess.run(cmd, check=True)


def _fallback_espeak(text: str, wav_path: str, sr: int = SAMPLE_RATE) -> None:
    raw_wav = wav_path + ".raw.wav"
    subprocess.run(
        ["espeak-ng", "-v", "en-gb", "-s", "142", "-p", "44", "-w", raw_wav, text],
        check=True,
    )
    _mp3_to_wav(raw_wav, wav_path, sr=sr)
    if os.path.exists(raw_wav):
        os.remove(raw_wav)


def _polish_voice(mono: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply subtle documentary voice EQ, high-pass rumble cut, warm body, and soft compression."""
    x = mono.astype(np.float64)
    # Trim leading/trailing near-silence gently (keep tiny breath padding)
    thresh = 0.004 * max(np.max(np.abs(x)), 1e-6)
    nz = np.where(np.abs(x) > thresh)[0]
    if len(nz) > 0:
        pad = int(0.06 * sr)
        start = max(0, nz[0] - pad)
        end = min(len(x), nz[-1] + pad)
        x = x[start:end]

    # High-pass at 85 Hz to remove sub-rumble
    sos_hp = signal.butter(2, 85.0, btype="highpass", fs=sr, output="sos")
    x = signal.sosfilt(sos_hp, x)

    # Gentle warm chest boost around 180 Hz (+1.8 dB) and presence clarity around 3.2 kHz (+1.5 dB)
    sos_warm = signal.butter(2, [120.0, 280.0], btype="bandpass", fs=sr, output="sos")
    sos_pres = signal.butter(2, [2400.0, 4800.0], btype="bandpass", fs=sr, output="sos")
    x = x + 0.14 * signal.sosfilt(sos_warm, x) + 0.12 * signal.sosfilt(sos_pres, x)

    # Smooth RMS envelope compressor for broadcast documentary consistency
    env_sos = signal.butter(1, 12.0, btype="lowpass", fs=sr, output="sos")
    env = np.sqrt(np.maximum(signal.sosfilt(env_sos, x * x), 1e-10))
    target_rms = 0.16
    gain = np.minimum(2.0, target_rms / (env + 0.045))
    x = x * gain

    # Soft peak limit & normalize to -2.5 dBFS
    x = np.tanh(x * 1.15) / 1.15
    peak = np.max(np.abs(x))
    if peak > 1e-6:
        x = x * (0.76 / peak)

    # Micro fade in/out to prevent clicks
    fade = min(int(0.025 * sr), len(x) // 4)
    if fade > 1:
        x[:fade] *= np.linspace(0.0, 1.0, fade)
        x[-fade:] *= np.linspace(1.0, 0.0, fade)

    return x.astype(np.float32)


def _add_subtle_studio_room(mono: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Convert dry mono narration to lush intimate documentary stereo with tiny room early reflections."""
    n = len(mono)
    left = mono.copy()
    right = mono.copy()
    # Very subtle stereo studio warmth (early reflections 9ms / 14ms, -24 dB)
    d1 = int(0.009 * sr)
    d2 = int(0.014 * sr)
    if n > d2 + 10:
        left[d1:] += 0.032 * mono[:-d1]
        right[d2:] += 0.032 * mono[:-d2]
        left[d2:] += 0.016 * mono[:-d2]
        right[d1:] += 0.016 * mono[:-d1]
    return np.stack([left, right], axis=1)


def generate_all_narration(
    audio_dir: str,
    total_duration: float = 123.0,
    voice: str = DEFAULT_VOICE,
    force: bool = False,
) -> Tuple[List[NarrationCue], np.ndarray, np.ndarray]:
    """
    Generate individual narration WAV files + full synchronized stereo narration stem
    and voice-activity envelope (for smooth music/SFX sidechain ducking).

    Returns:
        (cues, stereo_narration [N, 2], duck_envelope [N])
    """
    os.makedirs(audio_dir, exist_ok=True)
    meta_path = os.path.join(audio_dir, "narration_timing.json")

    cues: List[NarrationCue] = [
        NarrationCue(**asdict(c)) for c in NARRATION_SCRIPT
    ]

    for cue in cues:
        wav_path = os.path.join(audio_dir, f"narration_{cue.key}.wav")
        mp3_path = os.path.join(audio_dir, f"narration_{cue.key}.mp3")
        cue.wav_path = wav_path

        if force or not os.path.exists(wav_path):
            try:
                asyncio.run(
                    _synthesize_edge_tts(
                        cue.text, mp3_path, voice=voice, rate=cue.rate, pitch=cue.pitch
                    )
                )
                raw_wav = wav_path + ".raw.wav"
                _mp3_to_wav(mp3_path, raw_wav, sr=SAMPLE_RATE)
                _, data = wavfile.read(raw_wav)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                mono = data.astype(np.float32) / 32768.0
                polished = _polish_voice(mono, sr=SAMPLE_RATE)
                wavfile.write(
                    wav_path, SAMPLE_RATE, np.clip(polished * 32767.0, -32767, 32767).astype(np.int16)
                )
                for tmp in (raw_wav, mp3_path):
                    if os.path.exists(tmp):
                        os.remove(tmp)
            except Exception as exc:
                print(f"[narration] edge-tts failed for {cue.key} ({exc}); falling back to espeak-ng")
                _fallback_espeak(cue.text, wav_path, sr=SAMPLE_RATE)
                _, data = wavfile.read(wav_path)
                mono = data.astype(np.float32) / 32768.0
                polished = _polish_voice(mono, sr=SAMPLE_RATE)
                wavfile.write(
                    wav_path, SAMPLE_RATE, np.clip(polished * 32767.0, -32767, 32767).astype(np.int16)
                )

        sr_read, data_read = wavfile.read(wav_path)
        cue.duration = len(data_read) / float(sr_read)
        cue.end_time = cue.start_time + cue.duration

    # Ensure closing tagline follows scene6_main with a dignified 1.0s pause
    main6 = next(c for c in cues if c.key == "scene6_main")
    close6 = next(c for c in cues if c.key == "scene6_closing")
    desired_close_start = max(close6.start_time, main6.end_time + 0.85)
    close6.start_time = round(desired_close_start, 2)
    close6.end_time = close6.start_time + close6.duration

    # Verify total duration encompasses closing narration + graceful hold
    min_needed = close6.end_time + 2.5
    final_duration = max(total_duration, min_needed)

    total_samples = int(round(final_duration * SAMPLE_RATE))
    full_mono = np.zeros(total_samples, dtype=np.float32)
    voice_mask = np.zeros(total_samples, dtype=np.float32)

    for cue in cues:
        _, data = wavfile.read(cue.wav_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        clip = data.astype(np.float32) / 32768.0
        s0 = int(round(cue.start_time * SAMPLE_RATE))
        s1 = min(total_samples, s0 + len(clip))
        if s1 > s0:
            full_mono[s0:s1] += clip[: s1 - s0]
            # Mark active speech window with soft padding for music ducking
            p0 = max(0, s0 - int(0.35 * SAMPLE_RATE))
            p1 = min(total_samples, s1 + int(0.45 * SAMPLE_RATE))
            voice_mask[p0:p1] = 1.0

    # Smooth the ducking envelope (~250ms attack/release)
    kernel_len = int(0.35 * SAMPLE_RATE) | 1
    window = np.hanning(kernel_len).astype(np.float32)
    window /= window.sum()
    duck_env = signal.fftconvolve(voice_mask, window, mode="same")
    duck_env = np.clip(duck_env, 0.0, 1.0).astype(np.float32)

    full_stereo = _add_subtle_studio_room(full_mono, sr=SAMPLE_RATE)
    full_wav = os.path.join(audio_dir, "narration_full.wav")
    wavfile.write(
        full_wav,
        SAMPLE_RATE,
        np.clip(full_stereo * 32767.0, -32767, 32767).astype(np.int16),
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "voice": voice,
                "total_duration": round(final_duration, 3),
                "cues": [asdict(c) for c in cues],
            },
            f,
            indent=2,
        )

    return cues, full_stereo, duck_env


if __name__ == "__main__":
    audio_out = os.path.join(os.path.dirname(__file__), "audio")
    cues, stereo, _ = generate_all_narration(audio_out, total_duration=123.0, force=True)
    print("Generated narration cues:")
    for c in cues:
        print(
            f"  Scene {c.scene_id} [{c.key:14s}]: "
            f"{c.start_time:6.2f}s -> {c.end_time:6.2f}s  (dur={c.duration:5.2f}s)"
        )
    print(f"Total narration stem length: {len(stereo) / SAMPLE_RATE:.2f}s")

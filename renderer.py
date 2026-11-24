"""
renderer.py — Parallel Multiprocessing 1080p@30fps Video Renderer & FFmpeg Encoder
==================================================================================

Features a memory-bounded, resumable segmented render-farm architecture:
  - Renders individual scene preview MP4s (`render_scene_preview`)
  - Renders the complete continuous 1920x1080 @ 30 FPS Full-HD film in 6 exact
    frame-contiguous segments (`output/segments/part_0.mp4` .. `part_5.mp4`),
    preserving every seamless cross-scene transition down to the exact frame.
  - Concatenates all 6 segments losslessly (`-c:v copy`) with the mastered
    44.1 kHz stereo soundtrack (`master_soundtrack.wav`) into
    `output/food_chain_animation.mp4` and `food_chain_animation.mp4`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import wavfile

from animation import FPS, HEIGHT, SAMPLE_RATE, SCENE_SPECS, TOTAL_DURATION, WIDTH
from environments import ensure_environment_assets, load_environment_assets
from scenes import render_film_frame, render_single_scene_frame

# Worker-local cached painted environment assets
_WORKER_ENV: Optional[Dict[str, np.ndarray]] = None


def _init_worker(assets_dir: str) -> None:
    global _WORKER_ENV
    _WORKER_ENV = load_environment_assets(assets_dir)


def _render_chunk_film(args: Tuple[int, int, int, int, int]) -> List[bytes]:
    start_f, end_f, out_w, out_h, fps = args
    assert _WORKER_ENV is not None
    frames_bytes: List[bytes] = []
    for f_idx in range(start_f, end_f):
        t = f_idx / float(fps)
        bgr = render_film_frame(t, _WORKER_ENV, out_w=out_w, out_h=out_h)
        frames_bytes.append(bgr.tobytes())
    return frames_bytes


def _render_chunk_scene(args: Tuple[int, int, int, int, int, int]) -> List[bytes]:
    scene_id, start_f, end_f, out_w, out_h, fps = args
    assert _WORKER_ENV is not None
    frames_bytes: List[bytes] = []
    for f_idx in range(start_f, end_f):
        lt = f_idx / float(fps)
        bgr = render_single_scene_frame(scene_id, lt, _WORKER_ENV, out_w=out_w, out_h=out_h)
        frames_bytes.append(bgr.tobytes())
    return frames_bytes


def _slice_audio_wav(master_wav: str, start_s: float, end_s: float, out_wav: str) -> str:
    sr, data = wavfile.read(master_wav)
    s0 = max(0, int(round(start_s * sr)))
    s1 = min(len(data), int(round(end_s * sr)))
    seg = data[s0:s1].copy()
    fade = min(int(0.05 * sr), len(seg) // 6)
    if fade > 1:
        f_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
        f_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
        seg[:fade] = np.clip(seg[:fade].astype(np.float32) * f_in, -32767, 32767).astype(np.int16)
        seg[-fade:] = np.clip(seg[-fade:].astype(np.float32) * f_out, -32767, 32767).astype(np.int16)
    wavfile.write(out_wav, sr, seg)
    return out_wav


def _start_ffmpeg_video_pipe(
    out_mp4: str,
    width: int,
    height: int,
    fps: int,
    audio_wav: Optional[str] = None,
    crf: int = 16,
    preset: str = "fast",
) -> subprocess.Popen:
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
    ]
    if audio_wav is not None:
        cmd += ["-i", audio_wav, "-map", "0:v:0", "-map", "1:a:0"]
    else:
        cmd += ["-an"]

    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-g",
        str(fps),
        "-pix_fmt",
        "yuv420p",
        "-video_track_timescale",
        "30000",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
    ]
    if audio_wav is not None:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE), "-ac", "2", "-shortest"]

    cmd += ["-movflags", "+faststart", out_mp4]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _stream_chunks_bounded(
    pool: ProcessPoolExecutor,
    fn,
    chunks: Sequence[Tuple],
    stdin_pipe,
    max_in_flight: int = 4,
    label: str = "render",
    total_frames: int = 0,
) -> None:
    """
    Submit work in a bounded sliding window so memory stays strictly tiny
    even across thousands of 1080p frames (zero OOM risk!).
    """
    from collections import deque

    pending = deque()
    it = iter(chunks)

    def _submit_next() -> bool:
        try:
            c = next(it)
        except StopIteration:
            return False
        pending.append(pool.submit(fn, c))
        return True

    for _ in range(max_in_flight):
        if not _submit_next():
            break

    t0 = time.time()
    last_log = t0
    done_f = 0

    while pending:
        fut = pending.popleft()
        chunk_bytes = fut.result()
        _submit_next()
        for fb in chunk_bytes:
            stdin_pipe.write(fb)
        done_f += len(chunk_bytes)
        now = time.time()
        if total_frames > 0 and (now - last_log >= 15.0 or done_f == total_frames):
            pct = 100.0 * done_f / float(total_frames)
            fps_eff = done_f / max(now - t0, 0.01)
            eta = (total_frames - done_f) / max(fps_eff, 0.01)
            print(f"  [{label}] {done_f:4d}/{total_frames} ({pct:5.1f}%) | {fps_eff:5.2f} fps | ETA {eta:5.1f}s", flush=True)
            last_log = now


def render_scene_preview(
    scene_id: int,
    project_dir: str,
    width: int = 1280,
    height: int = 720,
    fps: int = FPS,
    workers: Optional[int] = None,
) -> str:
    """Render an individual scene preview MP4 with its exact synchronized audio slice."""
    assets_dir = os.path.join(project_dir, "assets")
    audio_dir = os.path.join(project_dir, "audio")
    prev_dir = os.path.join(project_dir, "previews")
    os.makedirs(prev_dir, exist_ok=True)
    ensure_environment_assets(assets_dir, force=False)

    spec = SCENE_SPECS[scene_id - 1]
    start_s = float(spec["start"])
    end_s = float(spec["end"])
    dur = end_s - start_s
    slug = str(spec["slug"])
    total_f = int(round(dur * fps))

    master_wav = os.path.join(audio_dir, "master_soundtrack.wav")
    slice_wav = os.path.join(prev_dir, f"_{slug}_audio.wav")
    _slice_audio_wav(master_wav, start_s, end_s, slice_wav)

    out_mp4 = os.path.join(prev_dir, f"{slug}_preview.mp4")
    n_workers = workers or max(1, os.cpu_count() or 2)
    chunk_size = 12
    chunks = [
        (scene_id, f0, min(total_f, f0 + chunk_size), width, height, fps)
        for f0 in range(0, total_f, chunk_size)
    ]

    t0 = time.time()
    proc = _start_ffmpeg_video_pipe(out_mp4, width, height, fps, audio_wav=slice_wav, crf=20, preset="fast")
    assert proc.stdin is not None
    try:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(assets_dir,)) as pool:
            _stream_chunks_bounded(pool, _render_chunk_scene, chunks, proc.stdin, max_in_flight=n_workers * 2)
    finally:
        proc.stdin.close()
        ret = proc.wait()
        if os.path.exists(slice_wav):
            os.remove(slice_wav)

    if ret != 0:
        raise RuntimeError(f"FFmpeg failed while rendering preview for Scene {scene_id} (code {ret})")

    elapsed = time.time() - t0
    print(f"[renderer] Scene {scene_id} preview saved: {out_mp4} ({total_f} frames @ {width}x{height} in {elapsed:.1f}s)", flush=True)
    return out_mp4


def get_film_segment_ranges(fps: int = FPS) -> List[Tuple[int, int]]:
    """Return 6 contiguous frame ranges covering [0 .. TOTAL_FRAMES) aligned with the 6 scenes."""
    ranges: List[Tuple[int, int]] = []
    for sp in SCENE_SPECS:
        f0 = int(round(float(sp["start"]) * fps))
        f1 = int(round(float(sp["end"]) * fps))
        ranges.append((f0, f1))
    return ranges


def render_film_segments(
    project_dir: str,
    part_indices: Optional[Sequence[int]] = None,
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
    workers: Optional[int] = None,
    crf: int = 16,
    preset: str = "fast",
    force: bool = False,
) -> List[str]:
    """
    Render selected Full-HD continuous film segments (`part_0.mp4` .. `part_5.mp4`).
    Because every frame calls `render_film_frame(global_t)`, all cross-scene
    dissolves are baked seamlessly across segment boundaries with zero seams!
    """
    assets_dir = os.path.join(project_dir, "assets")
    seg_dir = os.path.join(project_dir, "output", "segments")
    os.makedirs(seg_dir, exist_ok=True)
    ensure_environment_assets(assets_dir, force=False)

    ranges = get_film_segment_ranges(fps=fps)
    selected = list(part_indices) if part_indices is not None else list(range(len(ranges)))
    n_workers = workers or max(1, os.cpu_count() or 2)
    chunk_size = 10
    out_paths: List[str] = []

    for p_idx in selected:
        f_start, f_end = ranges[p_idx]
        n_f = f_end - f_start
        seg_path = os.path.join(seg_dir, f"part_{p_idx}.mp4")
        out_paths.append(seg_path)
        if not force and os.path.exists(seg_path) and os.path.getsize(seg_path) > 100_000:
            print(f"[renderer] Segment part_{p_idx}.mp4 already exists — skipping.", flush=True)
            continue

        print(f"[renderer] Rendering Full-HD segment part_{p_idx} (frames {f_start}..{f_end}, {n_f} frames @ {width}x{height})...", flush=True)
        t0 = time.time()
        chunks = [
            (f0, min(f_end, f0 + chunk_size), width, height, fps)
            for f0 in range(f_start, f_end, chunk_size)
        ]
        tmp_path = seg_path + ".tmp.mp4"
        proc = _start_ffmpeg_video_pipe(tmp_path, width, height, fps, audio_wav=None, crf=crf, preset=preset)
        assert proc.stdin is not None
        try:
            with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(assets_dir,)) as pool:
                _stream_chunks_bounded(
                    pool, _render_chunk_film, chunks, proc.stdin,
                    max_in_flight=n_workers * 2, label=f"part_{p_idx}", total_frames=n_f,
                )
        finally:
            proc.stdin.close()
            ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"FFmpeg failed on segment part_{p_idx} (code {ret})")
        os.replace(tmp_path, seg_path)
        print(f"[renderer] Saved {seg_path} in {time.time() - t0:.1f}s", flush=True)

    return out_paths


def assemble_final_film(project_dir: str, fps: int = FPS) -> str:
    """Losslessly concatenate all 6 Full-HD video segments and mux with mastered stereo soundtrack."""
    seg_dir = os.path.join(project_dir, "output", "segments")
    audio_dir = os.path.join(project_dir, "audio")
    out_dir = os.path.join(project_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    master_wav = os.path.join(audio_dir, "master_soundtrack.wav")
    if not os.path.exists(master_wav):
        raise FileNotFoundError(f"Master soundtrack missing: {master_wav}")

    ranges = get_film_segment_ranges(fps=fps)
    list_txt = os.path.join(seg_dir, "concat_list.txt")
    with open(list_txt, "w", encoding="utf-8") as f:
        for p_idx in range(len(ranges)):
            seg_path = os.path.join(seg_dir, f"part_{p_idx}.mp4")
            if not os.path.exists(seg_path):
                raise FileNotFoundError(f"Missing rendered Full-HD segment: {seg_path}")
            f.write(f"file '{seg_path}'\n")

    out_mp4 = os.path.join(out_dir, "food_chain_animation.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_txt,
        "-i",
        master_wav,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-shortest",
        out_mp4,
    ]
    subprocess.run(cmd, check=True)

    root_copy = os.path.join(project_dir, "food_chain_animation.mp4")
    shutil.copy2(out_mp4, root_copy)
    size_mb = os.path.getsize(out_mp4) / (1024.0 * 1024.0)
    print(f"[renderer] ASSEMBLED FINAL MASTER: {out_mp4} ({size_mb:.1f} MB)", flush=True)
    return out_mp4


def render_full_film(
    project_dir: str,
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
    workers: Optional[int] = None,
    crf: int = 16,
    preset: str = "fast",
    part_indices: Optional[Sequence[int]] = None,
    assemble: bool = True,
) -> Optional[str]:
    """Render all (or selected) Full-HD segments and optionally assemble the final master MP4."""
    render_film_segments(
        project_dir,
        part_indices=part_indices,
        width=width,
        height=height,
        fps=fps,
        workers=workers,
        crf=crf,
        preset=preset,
    )
    if assemble:
        return assemble_final_film(project_dir, fps=fps)
    return None

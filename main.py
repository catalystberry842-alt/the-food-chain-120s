#!/usr/bin/env python3
"""
main.py — Project: The Food Chain (Cinematic 2-Minute Animated Nature Film)
===========================================================================

Master production orchestrator for the entire Python animation studio pipeline:
  Stage 1     : Environment & character design asset generation
  Stages 2-7  : Visual QA contact sheets across all 6 choreographed scenes
  Stage 8     : Documentary narration, original orchestral score & wildlife SFX
  Stage 9     : Individual synchronized scene preview MP4 rendering
  Stage 10    : Final 1920x1080 @ 30 FPS Full-HD MP4 production & verification

Usage:
  python3 main.py --stage all
  python3 main.py --stage assets | qa | audio | previews | final
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from animation import FPS, HEIGHT, SCENE_SPECS, TOTAL_DURATION, WIDTH
from environments import ensure_environment_assets
from renderer import render_full_film, render_scene_preview
from sound_engine import build_master_soundtrack


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_assets_stage(force: bool = False) -> None:
    assets_dir = os.path.join(PROJECT_DIR, "assets")
    print("=== STAGE 1: Generating Painterly Environment Plates & Character Design Sheet ===")
    ensure_environment_assets(assets_dir, force=force)
    subprocess.run([sys.executable, os.path.join(PROJECT_DIR, "characters.py")], check=True)


def run_qa_stage() -> None:
    print("=== STAGES 2-7: Rendering 6-Scene Keyframe Visual QA Contact Sheets ===")
    subprocess.run([sys.executable, os.path.join(PROJECT_DIR, "scenes.py")], check=True)


def run_audio_stage(force_tts: bool = False) -> None:
    print("=== STAGE 8: Generating Narration, Original Film Score & Wildlife Soundscape ===")
    audio_dir = os.path.join(PROJECT_DIR, "audio")
    build_master_soundtrack(audio_dir, total_duration=TOTAL_DURATION, force_tts=force_tts)


def run_previews_stage(scenes: list[int] | None = None) -> None:
    print("=== STAGE 9: Rendering Individual Synchronized Scene Previews (720p @ 30 FPS) ===")
    target_ids = scenes if scenes else [int(s["id"]) for s in SCENE_SPECS]
    for sid in target_ids:
        render_scene_preview(sid, PROJECT_DIR, width=1280, height=720, fps=FPS)


def run_final_stage(parts: list[int] | None = None, assemble: bool = True) -> str | None:
    print("=== STAGE 10: Rendering Final 1920x1080 @ 30 FPS Full-HD Master MP4 ===")
    out_mp4 = render_full_film(
        PROJECT_DIR,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        crf=16,
        preset="fast",
        part_indices=parts,
        assemble=assemble,
    )
    if out_mp4:
        print("\n--- Final Deliverable Technical Verification (ffprobe) ---")
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,size,bit_rate:stream=codec_name,width,height,r_frame_rate,channels,sample_rate",
                "-of",
                "default=noprint_wrappers=1",
                out_mp4,
            ],
            check=False,
        )
    return out_mp4


def main() -> None:
    parser = argparse.ArgumentParser(description="THE FOOD CHAIN — Cinematic 2-Minute Python Nature Film Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "assets", "qa", "audio", "previews", "final"],
        default="all",
        help="Production stage to execute",
    )
    parser.add_argument("--force-assets", action="store_true", help="Regenerate cached painterly environment plates")
    parser.add_argument("--force-tts", action="store_true", help="Re-synthesize neural TTS narration clips")
    parser.add_argument("--scenes", type=int, nargs="*", help="Specific scene IDs (1..6) for --stage previews")
    parser.add_argument("--parts", type=int, nargs="*", help="Specific Full-HD segment indices (0..5) for --stage final")
    parser.add_argument("--no-assemble", action="store_true", help="Render selected --parts without assembling final MP4 yet")
    args = parser.parse_args()

    if args.stage in ("all", "assets"):
        run_assets_stage(force=args.force_assets)
    if args.stage in ("all", "qa"):
        run_qa_stage()
    if args.stage in ("all", "audio"):
        run_audio_stage(force_tts=args.force_tts)
    if args.stage in ("all", "previews"):
        run_previews_stage(scenes=args.scenes)
    if args.stage in ("all", "final"):
        run_final_stage(parts=args.parts, assemble=not args.no_assemble)


if __name__ == "__main__":
    main()

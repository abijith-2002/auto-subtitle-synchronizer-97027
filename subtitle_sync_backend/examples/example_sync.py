#!/usr/bin/env python3
"""
Example: Sync subtitles with video using the processing module.

This script demonstrates how to use the PUBLIC_INTERFACE functions from:
  subtitle_sync_backend/src/processing/processing.py

It:
- Creates a working directory
- Invokes the core synchronization function
- Prints the path to the synced subtitle file

Usage:
  python examples/example_sync.py /path/to/video.mp4 /path/to/subtitles.srt [--prefer-whisper] [--whisper-model base] [--output-dir /tmp]

Notes:
- By default, it uses a stub transcriber (no heavy dependencies).
- If you pass --prefer-whisper, it will attempt to use Whisper (requires 'openai-whisper' package and ffmpeg);
  if Whisper is not available, it will gracefully fall back to the stub transcriber.
"""

import argparse
import os
import sys

# Import public interfaces from processing module
# The project structure expects running this from repository root or the backend folder.
# If you run this from the repository root, ensure Python path can resolve the src package.
# One way is to execute: python -m subtitle_sync_backend.examples.example_sync ...
# or run from within the subtitle_sync_backend directory.
try:
    from src.processing.processing import (
        process_and_sync_subtitles,
        create_workdir,
        SubtitleSyncError,
    )
except ModuleNotFoundError:
    # If executed directly from repository root, adjust sys.path so `src` is importable.
    # This allows: python auto-subtitle-synchronizer-97027/subtitle_sync_backend/examples/example_sync.py ...
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.abspath(os.path.join(current_dir, ".."))
    src_dir = os.path.join(backend_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, backend_root)  # make "src" importable as a top-level package
    from src.processing.processing import (
        process_and_sync_subtitles,
        create_workdir,
        SubtitleSyncError,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync subtitles with video audio using the processing module.")
    parser.add_argument("--video", help="Path to the video file (e.g., .mp4, .mkv, etc.).")
    parser.add_argument("--subtitle", help="Path to the subtitle file (.srt, .vtt, .ass, .ssa).")
    parser.add_argument(
        "--prefer-whisper",
        action="store_true",
        default=False,
        help="Prefer Whisper for transcription if available (requires openai-whisper and ffmpeg).",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model name (e.g., tiny, base, small, medium, large). Default: base",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the synced subtitle. Defaults to a temporary working directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    video_path = os.path.abspath(args.video)
    subtitle_path = os.path.abspath(args.subtitle)

    if not os.path.isfile(video_path):
        print(f"Error: video file not found: {video_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(subtitle_path):
        print(f"Error: subtitle file not found: {subtitle_path}", file=sys.stderr)
        return 1

    # Create or use provided output directory
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        # PUBLIC_INTERFACE: create a work directory for this job
        output_dir = create_workdir()

    print("Starting subtitle synchronization...")
    print(f"  Video:    {video_path}")
    print(f"  Subtitle: {subtitle_path}")
    print(f"  Output:   {output_dir}")
    print(f"  Prefer Whisper: {args.prefer_whisper} (model: {args.whisper_model})")

    try:
        # PUBLIC_INTERFACE: Run the core processing and synchronization logic
        result = process_and_sync_subtitles(
            video_path=video_path,
            subtitle_path=subtitle_path,
            output_dir=output_dir,
            prefer_whisper=args.prefer_whisper,
            whisper_model=args.whisper_model,
        )
    except SubtitleSyncError as e:
        print(f"Sync failed: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 3

    # Result dict contains: synced_subtitle_path, format, shift_seconds
    synced_path = result.get("synced_subtitle_path", "")
    fmt = result.get("format", "unknown")
    shift_seconds = result.get("shift_seconds", "0.000")

    print("\nSync completed successfully!")
    print(f"  Synced subtitle path: {synced_path}")
    print(f"  Format:               {fmt}")
    print(f"  Applied shift (secs): {shift_seconds}")

    # Exit status 0 indicates success
    return 0


if __name__ == "__main__":
    sys.exit(main())

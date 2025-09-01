#!/usr/bin/env python3
"""
Example: Minimal subtitle sync using processing.py

This script demonstrates how to import and use the PUBLIC_INTERFACE function(s)
from src/processing/processing.py to sync a subtitle file with a video's audio.

It:
- Imports process_and_sync_subtitles and create_workdir from processing.py
- Defines realistic demo paths for a video and a subtitle (you can replace them)
- Calls the sync function
- Prints the resulting synced subtitle path and basic metadata

Usage options:
  1) Run as module (recommended from repo root or backend folder):
     python -m subtitle_sync_backend.examples.example_sync_minimal

  2) Run directly (ensure PYTHONPATH includes the backend src):
     PYTHONPATH=auto-subtitle-synchronizer-97027/subtitle_sync_backend \
       python auto-subtitle-synchronizer-97027/subtitle_sync_backend/examples/example_sync_minimal.py

Notes:
- If you want Whisper transcription, pass prefer_whisper=True and ensure openai-whisper/ffmpeg are available.
- By default, a stub transcriber is used so the script can run offline.
"""

from __future__ import annotations

import os
import sys


def _setup_path_for_direct_run() -> None:
    """
    If executed directly from repository root, adjust sys.path so `src` is importable.
    This allows running the file via absolute path without -m.
    """
    try:
        # Try normal import first; if it fails, patch sys.path
        import src  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.abspath(os.path.join(current_dir, ".."))
    if backend_root not in sys.path:
        # Make "src" importable as a top-level package
        sys.path.insert(0, backend_root)


# Ensure imports resolve whether run as module or directly
_setup_path_for_direct_run()

# Import the public interface from processing.py
try:
    # PUBLIC_INTERFACE imports
    from src.processing.processing import (
        process_and_sync_subtitles,
        create_workdir,
        SubtitleSyncError,
    )
except Exception as exc:
    print(f"Failed to import processing interfaces: {exc}", file=sys.stderr)
    sys.exit(2)


# PUBLIC_INTERFACE
def main() -> int:
    """
    PUBLIC_INTERFACE
    Minimal demo that syncs a subtitle with a video's audio using processing.py.

    Parameters: None (uses built-in demo paths; edit as needed)
    Returns:
        0 on success; non-zero on failure.
    """
    # Realistic dummy file paths (replace with your own)
    # You can place demo assets under 'attachments/' in repo root if available.
    video_path = os.path.abspath("attachments/2mins.mp4")
    subtitle_path = os.path.abspath("attachments/2mins_inaccurate.srt")

    # If demo assets are not present, show an instruction but continue to demonstrate code flow.
    if not os.path.isfile(video_path):
        print(f"Warning: demo video not found at {video_path}")
        print("Provide a valid video path by editing this script or placing a file at the path above.")
    if not os.path.isfile(subtitle_path):
        print(f"Warning: demo subtitle not found at {subtitle_path}")
        print("Provide a valid subtitle path by editing this script or placing a file at the path above.")

    # Create a working directory for outputs
    out_dir = create_workdir()
    print("=== Minimal Subtitle Sync Demo ===")
    print(f"Video:      {video_path}")
    print(f"Subtitle:   {subtitle_path}")
    print(f"Output dir: {out_dir}")

    # Configure transcription choice
    prefer_whisper = False   # Set True to prefer Whisper if installed (optional)
    whisper_model = "base"   # Whisper model name if using Whisper

    try:
        # PUBLIC_INTERFACE: core sync call from processing.py
        result = process_and_sync_subtitles(
            video_path=video_path,
            subtitle_path=subtitle_path,
            output_dir=out_dir,
            prefer_whisper=prefer_whisper,
            whisper_model=whisper_model,
        )
    except SubtitleSyncError as e:
        print(f"Sync failed: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"Unexpected error during sync: {e}", file=sys.stderr)
        return 4

    # Unpack result and report
    synced_path = result.get("synced_subtitle_path")
    fmt = result.get("format")
    shift_seconds = result.get("shift_seconds")
    llm_variant = result.get("llm_corrected_subtitle_path")

    print("\nSync completed.")
    print(f"- Synced subtitle path: {synced_path}")
    print(f"- Format:               {fmt}")
    print(f"- Applied shift (secs): {shift_seconds}")
    if llm_variant:
        print(f"- LLM-corrected variant: {llm_variant}")

    if synced_path and os.path.isfile(synced_path):
        print("\nSuccess: Synced subtitle file was written.")
    else:
        print("\nNote: The synced subtitle path was returned but the file may not exist if the input files were missing.")
        print("      Provide real files and rerun to produce an actual synced subtitle.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

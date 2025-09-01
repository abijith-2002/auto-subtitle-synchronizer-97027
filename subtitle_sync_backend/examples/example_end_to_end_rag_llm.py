#!/usr/bin/env python3
"""
Example: End-to-end subtitle correction with transcription, RAG, and LLM.

This example demonstrates how to:
  1) Load a video and a subtitle file
  2) Run full-video transcription
  3) Retrieve top-10 relevant transcript segments per subtitle cue (RAG)
  4) Run LLM-based correction for each cue using the RAG context
  5) Save a corrected SRT file next to the synced/shifted subtitle

It uses only PUBLIC_INTERFACE functions from src/processing.

Usage (from repository root or backend folder):
  python -m subtitle_sync_backend.examples.example_end_to_end_rag_llm \
     --video attachments/2mins.mp4 \
     --subtitle attachments/2mins_inaccurate.srt \
     [--prefer-whisper --whisper-model base] \
     [--output-dir /tmp/out_dir] \
     [--dry-run]

Notes:
- If Whisper is not available, a stub transcriber will be used automatically.
- If OPENAI_API_KEY is not provided, an internal stub LLM client is used that performs a light cleanup.
- The pipeline writes:
    - A synced subtitle file with "-synced" suffix (baseline time alignment).
    - An optional LLM-corrected variant with "-synced-llm" suffix.

Environment variables (do NOT hardcode secrets in code; set them via .env):
- OPENAI_API_KEY         : API key for OpenAI-compatible endpoints (optional)
- OPENAI_API_BASE        : Optional override base URL for OpenAI-compatible services
- OPENAI_MODEL           : Optional model name (default: gpt-4o-mini)
- OPENAI_REQUEST_TIMEOUT : Optional request timeout seconds (default: 30)

Example with provided demo assets (from repo root):
  python -m subtitle_sync_backend.examples.example_end_to_end_rag_llm \
      --video attachments/2mins.mp4 \
      --subtitle attachments/2mins_inaccurate.srt \
      --output-dir /tmp/subtitle_demo

If you prefer to run directly (not as a module), ensure PYTHONPATH includes the backend src:
  PYTHONPATH=auto-subtitle-synchronizer-97027/subtitle_sync_backend python \
      auto-subtitle-synchronizer-97027/subtitle_sync_backend/examples/example_end_to_end_rag_llm.py \
      --video attachments/2mins.mp4 \
      --subtitle attachments/2mins_inaccurate.srt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Dict, Any

# Attempt normal imports; adjust sys.path if executed directly from repo root.
try:
    from src.processing.processing import (
        process_and_sync_subtitles,
        create_workdir,
        SubtitleSyncError,
        detect_subtitle_format,
        parse_srt,
        format_srt,
        parse_vtt,
        format_vtt,
        parse_ass_ssa,
        format_ass_ssa,
        SubtitleEntry,
    )
    from src.processing import (
        rag_top_k_for_subtitles,
        correct_subtitles_with_llm,
        apply_corrected_texts,
    )
except ModuleNotFoundError:
    # If executed directly from repository root, adjust sys.path so `src` is importable.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.abspath(os.path.join(current_dir, ".."))
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)
    from src.processing.processing import (
        process_and_sync_subtitles,
        create_workdir,
        SubtitleSyncError,
        detect_subtitle_format,
        parse_srt,
        format_srt,
        parse_vtt,
        format_vtt,
        parse_ass_ssa,
        format_ass_ssa,
        SubtitleEntry,
    )
    from src.processing import (
        rag_top_k_for_subtitles,
        correct_subtitles_with_llm,
        apply_corrected_texts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end subtitle correction with transcription, RAG, and LLM."
    )
    parser.add_argument(
        "--video",
        required=False,
        default="attachments/2mins.mp4",
        help="Path to the video file (default uses bundled demo if present).",
    )
    parser.add_argument(
        "--subtitle",
        required=False,
        default="attachments/2mins_inaccurate.srt",
        help="Path to the subtitle file (default uses bundled demo if present).",
    )
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
        help="Directory to write the outputs. Defaults to a temporary working directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write extra corrected files; just print what would be done.",
    )
    return parser.parse_args()


def _read_subtitle(fmt: str, path: str) -> Dict[str, Any]:
    """
    Read and parse subtitle file into entries and optional header for ASS/SSA.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    header = ""
    if fmt == "srt":
        entries = parse_srt(content)
    elif fmt == "vtt":
        entries = parse_vtt(content)
    else:
        header, entries = parse_ass_ssa(content)
    return {"header": header, "entries": entries}


def _write_subtitle(fmt: str, path: str, header: str, entries: List[SubtitleEntry]) -> None:
    """
    Serialize entries (and optional header) back to subtitle file.
    """
    if fmt == "srt":
        out = format_srt(entries)
    elif fmt == "vtt":
        out = format_vtt(entries)
    else:
        out = format_ass_ssa(header, entries)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)


def main() -> int:
    args = parse_args()

    video_path = os.path.abspath(args.video)
    subtitle_path = os.path.abspath(args.subtitle)
    if not os.path.isfile(video_path):
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(subtitle_path):
        print(f"Error: subtitle not found: {subtitle_path}", file=sys.stderr)
        return 1

    # Output directory
    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = create_workdir()

    print("=== End-to-end Subtitle Correction Demo ===")
    print(f"Video:         {video_path}")
    print(f"Subtitle:      {subtitle_path}")
    print(f"Output dir:    {out_dir}")
    print(f"Prefer Whisper: {args.prefer_whisper} (model: {args.whisper_model})")
    print("")
    print("Step 1) Transcribing and syncing subtitle timestamps...")

    try:
        # PUBLIC_INTERFACE: This runs full transcription, naive sync, writes -synced file,
        # and attempts an LLM correction pass internally (optional).
        sync_result = process_and_sync_subtitles(
            video_path=video_path,
            subtitle_path=subtitle_path,
            output_dir=out_dir,
            prefer_whisper=args.prefer_whisper,
            whisper_model=args.whisper_model,
        )
    except SubtitleSyncError as e:
        print(f"Sync failed: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Unexpected error during sync: {e}", file=sys.stderr)
        return 3

    synced_path = sync_result.get("synced_subtitle_path")
    fmt = sync_result.get("format", "srt")
    shift_seconds = sync_result.get("shift_seconds", "0.000")
    entries = sync_result.get("entries") or []
    entries_path = sync_result.get("entries_path") or ""
    llm_variant = sync_result.get("llm_corrected_subtitle_path") or ""

    print("Sync complete.")
    print(f"  Synced subtitle:  {synced_path}")
    print(f"  Format:           {fmt}")
    print(f"  Shift (seconds):  {shift_seconds}")
    print(f"  Transcript items: {len(entries)} (persisted: {bool(entries_path)})")
    if entries_path and os.path.isfile(entries_path):
        print(f"  Entries JSON:     {entries_path}")
    if llm_variant and os.path.isfile(llm_variant):
        print(f"  LLM variant:      {llm_variant}")
    print("")

    # The call above already attempted LLM correction. The following steps explicitly
    # show how to do RAG + LLM correction "manually" for demonstration or custom flows.

    print("Step 2) Parsing the synced subtitle back into entries...")
    if not synced_path or not os.path.isfile(synced_path):
        print("Synced subtitle file not found; cannot proceed with manual RAG/LLM demo.", file=sys.stderr)
        return 4

    synced_fmt = detect_subtitle_format(synced_path)
    parsed = _read_subtitle(synced_fmt, synced_path)
    header = parsed["header"]
    synced_entries: List[SubtitleEntry] = parsed["entries"]
    print(f"  Parsed synced cues: {len(synced_entries)}")
    if not synced_entries:
        print("No cues found after parsing synced file; aborting.", file=sys.stderr)
        return 5

    # Use transcript entries from the earlier result for RAG source.
    print("Step 3) Computing top-10 transcript segments per cue (RAG)...")
    topk_per_cue: List[List[Dict[str, Any]]] = rag_top_k_for_subtitles(
        synced_entries, entries, k=10
    )
    print("  RAG retrieval done.")

    print("Step 4) Running LLM-based correction using RAG context...")
    # PUBLIC_INTERFACE: This function runs batch LLM calls with robust output parsing and returns corrected text list.
    corrected_texts, rag_debug = correct_subtitles_with_llm(
        subtitle_entries=synced_entries,
        transcript_entries=entries,
        fmt=synced_fmt,
        k=10,
    )
    # RAG debug is the same shape as topk_per_cue. We won't print it fully to avoid noise.
    print(f"  Corrected {len(corrected_texts)} cues.")

    print("Step 5) Applying corrected texts to produce a new subtitle file...")
    corrected_entries = apply_corrected_texts(synced_entries, corrected_texts)

    # Derive corrected file path. We'll place it next to the synced file with "-synced-llm" suffix.
    base, ext = os.path.splitext(os.path.basename(synced_path))
    corrected_name = f"{base}-llm{ext}"
    corrected_path = os.path.join(out_dir, corrected_name)
    if args.dry_run:
        print("Dry-run enabled: skipping write of corrected subtitle file.")
    else:
        _write_subtitle(synced_fmt, corrected_path, header, corrected_entries)
        print(f"  Wrote corrected subtitle: {corrected_path}")

    # Optional: Save a small JSON preview of first cues to help inspect changes.
    preview_json = os.path.join(out_dir, "llm_corrections_preview.json")
    try:
        preview = []
        for i in range(min(5, len(synced_entries))):
            preview.append({
                "index": synced_entries[i].index,
                "original": synced_entries[i].text,
                "corrected": corrected_entries[i].text if i < len(corrected_entries) else corrected_texts[i],
                "topk_sample": [
                    {"start": r.get("start"), "end": r.get("end"), "score": r.get("score"), "text": r.get("text")}
                    for r in (topk_per_cue[i] if i < len(topk_per_cue) else [])[:3]
                ]
            })
        if not args.dry_run:
            with open(preview_json, "w", encoding="utf-8") as jf:
                json.dump(preview, jf, ensure_ascii=False, indent=2)
            print(f"  Wrote preview JSON (first 5 cues): {preview_json}")
    except Exception as e:
        print(f"  Warning: failed to write preview JSON: {e}")

    print("")
    print("All steps completed.")
    print("Reminder:")
    print("- If OPENAI_API_KEY is not set, a stub LLM is used (works offline).")
    print("- If sentence-transformers is unavailable, TF-IDF RAG is used as fallback.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

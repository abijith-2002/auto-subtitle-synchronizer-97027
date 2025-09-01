#!/usr/bin/env python3
"""
Example: Explicit RAG loop for subtitle syncing with detailed steps.

This script demonstrates the full Retrieval-Augmented Generation (RAG) loop for subtitle
correction. The pipeline is:

  1) Transcribe the entire video into transcript entries (start/end/text)
  2) Parse the provided subtitle file into entries (index/start/end/text)
  3) For each subtitle cue, retrieve the top-10 most relevant transcript segments (RAG)
      using retrieve_top_k_for_cues
  4) For each cue, call the LLM (e.g., OpenAI or Gemini) with:
      - The original cue text
      - Original start/end timestamps
      - The top-10 transcript segments
     Prompt it to fix/clean the subtitle, and return the revised line
  5) Parse the LLM response to extract a clean, single-line corrected subtitle
  6) Build and write the final corrected subtitle file next to the synced one

This example intentionally shows each step "manually" in code, even though the backend
provides higher-level helpers that do much of the work. It uses PUBLIC_INTERFACE
imports from the backend processing package and can run with dummy files.

Usage (from repository root or backend folder):
  python -m subtitle_sync_backend.examples.example_full_rag_loop \
      --video attachments/2mins.mp4 \
      --subtitle attachments/2mins_inaccurate.srt \
      [--prefer-whisper --whisper-model base] \
      [--output-dir /tmp/out_dir] \
      [--model openai|gemini] \
      [--k 10] \
      [--dry-run]

Notes:
- If OPENAI_API_KEY is not set, a deterministic stub LLM is used for OpenAI path.
- If GEMINI_API_KEY is not set, a minimal stub correction is used for Gemini path.
- If sentence-transformers is unavailable, TF-IDF RAG fallback is used automatically.

Environment variables (request these from user; do not hardcode secrets):
- OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL, OPENAI_REQUEST_TIMEOUT
- GEMINI_API_KEY

The example writes:
- A synced subtitle file with "-synced" suffix (baseline time alignment)
- A corrected file from the explicit RAG+LLM loop with "-synced-rag-llm" suffix
- A small JSON preview of the first cues to inspect RAG results and corrections
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

# Attempt direct imports; if run from repo root directly, patch sys.path.
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
    from src.processing.rag import retrieve_top_k_for_cues
    from src.processing.llm import (
        build_cue_prompt,
    )
    from src.processing.parser import extract_corrected_line_from_response
except ModuleNotFoundError:
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
    from src.processing.rag import retrieve_top_k_for_cues
    from src.processing.llm import (
        build_cue_prompt,
    )
    from src.processing.parser import extract_corrected_line_from_response


def _read_subtitle(fmt: str, path: str) -> Tuple[str, List[SubtitleEntry]]:
    """
    Read and parse subtitle file into entries and optional header (for ASS/SSA).
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
    return header, entries


def _write_subtitle(fmt: str, path: str, header: str, entries: List[SubtitleEntry]) -> None:
    """
    Serialize entries and optional header back to the correct subtitle format.
    """
    if fmt == "srt":
        out = format_srt(entries)
    elif fmt == "vtt":
        out = format_vtt(entries)
    else:
        out = format_ass_ssa(header, entries)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)


def _clean_single_line(s: str) -> str:
    """
    Utility to ensure a single-line subtitle with normalized spacing.
    """
    return " ".join((s or "").strip().split())


def _openai_complete(prompt: str) -> str:
    """
    Call OpenAI-compatible Chat Completions.
    If OPENAI_API_KEY is missing or client fails, return a stub correction derived from the prompt.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Deterministic stub: extract original cue line from the prompt; ensure punctuation.
        text = prompt.split("Original cue:", 1)[-1].strip()
        if "\n" in text:
            text = text.splitlines()[0].strip()
        if text and text[-1] not in ".?!":
            text += "."
        return _clean_single_line(text)

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        # Fallback stub if dependency not installed
        text = prompt.split("Original cue:", 1)[-1].strip()
        if "\n" in text:
            text = text.splitlines()[0].strip()
        if text and text[-1] not in ".?!":
            text += "."
        return _clean_single_line(text)

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_API_BASE") or None,
    )
    model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You correct and improve subtitle text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=96,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _clean_single_line(extract_corrected_line_from_response(raw))
    except Exception:
        # On error, minimal fallback cleanup
        text = prompt.split("Original cue:", 1)[-1].strip()
        if "\n" in text:
            text = text.splitlines()[0].strip()
        if text and text[-1] not in ".?!":
            text += "."
        return _clean_single_line(text)


def _gemini_complete(prompt: str) -> str:
    """
    Call Google Gemini via google.genai SDK (no direct REST).
    If GEMINI_API_KEY is missing or SDK fails, return a stub correction derived from the prompt.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        text = prompt.split("Original cue:", 1)[-1].strip()
        if "\n" in text:
            text = text.splitlines()[0].strip()
        if text and text[-1] not in ".?!":
            text += "."
        return _clean_single_line(text)

    try:
        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore
    except Exception:
        text = prompt.split("Original cue:", 1)[-1].strip()
        if "\n" in text:
            text = text.splitlines()[0].strip()
        if text and text[-1] not in ".?!":
            text += "."
        return _clean_single_line(text)

    client = genai.Client(api_key=api_key)
    try:
        result = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[
                genai_types.Content(role="user", parts=[genai_types.Part.from_text(prompt)])
            ],
        )
    except Exception:
        text = prompt.split("Original cue:", 1)[-1].strip()
        if "\n" in text:
            text = text.splitlines()[0].strip()
        if text and text[-1] not in ".?!":
            text += "."
        return _clean_single_line(text)

    # Extract text parts and parse a single corrected line
    raw = ""
    try:
        for cand in getattr(result, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                if hasattr(part, "text") and part.text:
                    raw += part.text + "\n"
    except Exception:
        pass

    return _clean_single_line(extract_corrected_line_from_response(raw))


def _llm_fix_subtitle_line(
    model_choice: str,
    cue_text: str,
    start: float,
    end: float,
    topk_segments: List[Dict[str, Any]],
) -> str:
    """
    Helper that builds the instruction prompt with timestamps and RAG context,
    sends it to the chosen LLM (openai or gemini), and returns the cleaned single-line text.

    - model_choice: "openai" or "gemini"
    - cue_text/start/end: the original cue and its timing
    - topk_segments: list of similar transcript segments; each item has start/end/text/score
    """
    # Build a rich instruction prompt that:
    # - Preserves meaning
    # - Fixes typos/casing/punctuation
    # - Keeps concise for subtitles
    # - Returns ONLY the corrected text (no quotes or prefix)
    prompt = build_cue_prompt(
        ctx=type("Ctx", (), {"cue_text": cue_text, "cue_start": start, "cue_end": end, "topk_segments": topk_segments})
    )

    if model_choice.lower() == "gemini":
        return _gemini_complete(prompt)
    # Default to OpenAI client behavior
    return _openai_complete(prompt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explicit RAG loop for subtitle syncing and LLM correction."
    )
    parser.add_argument(
        "--video",
        required=False,
        default=r"../../../attachments/2mins.mp4",
        help="Path to the video file (dummy path acceptable).",
    )
    parser.add_argument(
        "--subtitle",
        required=False,
        default=r"../../../attachments/2mins_inaccurate.srt",
        help="Path to the subtitle file (.srt/.vtt/.ass/.ssa).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write outputs (defaults to temp workdir).",
    )
    parser.add_argument(
        "--prefer-whisper",
        action="store_true",
        default=False,
        help="Prefer Whisper for transcription if available.",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model name if using Whisper (e.g., tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--model",
        choices=["openai", "gemini"],
        default="openai",
        help="Select which LLM stack to use for correction.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-K transcript segments to retrieve per cue.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not write the final corrected subtitle file; print paths and stats only.",
    )
    return parser.parse_args()


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

    # Prepare output directory (use a unique workdir by default)
    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = create_workdir()

    print("=== Explicit RAG Loop for Subtitle Syncing ===")
    print(f"Video:        {video_path}")
    print(f"Subtitle:     {subtitle_path}")
    print(f"Output dir:   {out_dir}")
    print(f"LLM model:    {args.model}")
    print(f"Top-K (RAG):  {args.k}")
    print(f"Prefer Whisper: {args.prefer_whisper} (model: {args.whisper_model})")
    print("")
    print("Step 1) Transcribe and perform baseline sync...")

    # Step 1: Transcribe full video and baseline align the subtitle timestamps.
    # We use the existing PUBLIC_INTERFACE pipeline to produce:
    # - A synced subtitle file ("-synced" suffix)
    # - The transcript entries (used for RAG)
    try:
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

    synced_path = sync_result.get("synced_subtitle_path") or ""
    fmt = sync_result.get("format", "srt")
    shift_seconds = sync_result.get("shift_seconds", "0.000")
    transcript_entries: List[Dict[str, Any]] = sync_result.get("entries") or []
    entries_json_path = sync_result.get("entries_path") or ""

    print("Baseline sync completed.")
    print(f"  Synced subtitle:  {synced_path}")
    print(f"  Format:           {fmt}")
    print(f"  Shift (seconds):  {shift_seconds}")
    print(f"  Transcript items: {len(transcript_entries)}")
    if entries_json_path and os.path.isfile(entries_json_path):
        print(f"  Entries JSON:     {entries_json_path}")
    print("")

    # Step 2: Parse the synced subtitle file into entries we can work with
    print("Step 2) Parse synced subtitle into entries...")
    if not synced_path or not os.path.isfile(synced_path):
        print("Error: Synced subtitle file not found; aborting.", file=sys.stderr)
        return 4

    synced_fmt = detect_subtitle_format(synced_path)
    header, synced_entries = _read_subtitle(synced_fmt, synced_path)
    if not synced_entries:
        print("Error: No cues parsed from the synced file; aborting.", file=sys.stderr)
        return 5
    print(f"  Parsed synced cues: {len(synced_entries)}")

    # Step 3: RAG retrieval - for each cue, select top-K relevant transcript segments
    print("Step 3) Compute top-K relevant transcript segments per cue (RAG)...")
    cues_payload = [{"text": s.text, "start": s.start, "end": s.end} for s in synced_entries]
    topk_per_cue: List[List[Dict[str, Any]]] = retrieve_top_k_for_cues(
        cues=cues_payload,
        entries=transcript_entries,
        k=int(args.k),
    )
    print("  RAG retrieval done.")

    # Step 4: For each cue, build the LLM prompt and send it to the selected model (OpenAI or Gemini).
    print("Step 4) LLM correction for each cue using RAG context...")
    corrected_texts: List[str] = []
    for i, cue in enumerate(synced_entries):
        topk = topk_per_cue[i] if i < len(topk_per_cue) else []
        fixed = _llm_fix_subtitle_line(
            model_choice=args.model,
            cue_text=cue.text,
            start=cue.start,
            end=cue.end,
            topk_segments=topk,
        )
        corrected_texts.append(fixed)
    print(f"  Corrected {len(corrected_texts)} cues via LLM.")

    # Step 5: Apply corrected texts and write final corrected subtitle file
    print("Step 5) Build and write the final corrected subtitle...")
    corrected_entries: List[SubtitleEntry] = []
    for i, e in enumerate(synced_entries):
        new_text = corrected_texts[i] if i < len(corrected_texts) else e.text
        corrected_entries.append(type(e)(index=e.index, start=e.start, end=e.end, text=new_text))

    # Construct output filename next to the synced file
    base, ext = os.path.splitext(os.path.basename(synced_path))
    corrected_name = f"{base}-rag-llm{ext}"
    corrected_path = os.path.join(out_dir, corrected_name)

    if args.dry_run:
        print("Dry-run enabled: skipping write of corrected subtitle file.")
    else:
        _write_subtitle(synced_fmt, corrected_path, header, corrected_entries)
        print(f"  Wrote corrected subtitle: {corrected_path}")

    # Optional: save a small RAG+LLM preview JSON for the first few cues
    preview_json = os.path.join(out_dir, "explicit_rag_llm_preview.json")
    try:
        preview: List[Dict[str, Any]] = []
        for i in range(min(5, len(synced_entries))):
            preview.append(
                {
                    "index": synced_entries[i].index,
                    "original": synced_entries[i].text,
                    "corrected": corrected_entries[i].text if i < len(corrected_entries) else corrected_texts[i],
                    "topk_sample": [
                        {
                            "start": r.get("start"),
                            "end": r.get("end"),
                            "score": r.get("score"),
                            "text": r.get("text"),
                        }
                        for r in (topk_per_cue[i] if i < len(topk_per_cue) else [])[:3]
                    ],
                }
            )
        if not args.dry_run:
            with open(preview_json, "w", encoding="utf-8") as jf:
                json.dump(preview, jf, ensure_ascii=False, indent=2)
            print(f"  Wrote RAG+LLM preview JSON (first 5 cues): {preview_json}")
    except Exception as e:
        print(f"  Warning: failed to write preview JSON: {e}")

    print("")
    print("All steps completed.")
    print("Hints:")
    print("- For OpenAI: set OPENAI_API_KEY (and optionally OPENAI_API_BASE, OPENAI_MODEL).")
    print("- For Gemini: set GEMINI_API_KEY and ensure google-genai is installed.")
    print("- The RAG stage will use sentence-transformers if available, otherwise a TF-IDF fallback.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

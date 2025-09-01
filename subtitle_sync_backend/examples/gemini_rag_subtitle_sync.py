#!/usr/bin/env python3
"""
Example: End-to-end subtitle syncing using Gemini AI API (transcription) with simple RAG alignment.

This example demonstrates how to:
  1) Load a video and a subtitle file
  2) Transcribe the video's audio using Gemini AI API
  3) Compute a naive time shift and retrieve top-K transcript segments per cue (RAG)
  4) Produce a synced subtitle file and a simple RAG preview JSON

It is self-contained for demonstration and does not modify the backend code.
For production-grade usage, prefer the built-in pipeline:
  python -m subtitle_sync_backend.examples.example_end_to_end_rag_llm ...

Authentication and configuration:
- Reads Gemini API key from environment variable: GEMINI_API_KEY
- If no native Gemini SDK is available, this script uses HTTP requests.
- You can optionally install the google-generativeai SDK and set it up, but this
  example sticks to HTTP for portability.

Environment variables (do NOT hardcode secrets in code; set them via .env):
- GEMINI_API_KEY: API key for Gemini AI

Dependencies:
- Standard library only for core logic.
- For HTTP invocation:
  - Either ensure 'httpx' is installed (already included in backend requirements),
    or alternatively you can replace httpx with Python's urllib if needed.
  The backend already includes httpx in requirements.txt, so importing httpx should work.

Inputs:
- Video file path (e.g., attachments/2mins.mp4)
- Subtitle file path (.srt, .vtt, .ass, .ssa)

Outputs:
- A synced subtitle file written to the specified output directory with "-synced" suffix
- A JSON RAG preview for the first few cues

Usage (from repository root or backend folder):
  python -m subtitle_sync_backend.examples.gemini_rag_subtitle_sync \
      --video attachments/2mins.mp4 \
      --subtitle attachments/2mins_inaccurate.srt \
      --output-dir /tmp/gemini_demo

If you prefer to run directly, ensure PYTHONPATH includes the backend src:
  PYTHONPATH=auto-subtitle-synchronizer-97027/subtitle_sync_backend python \
    auto-subtitle-synchronizer-97027/subtitle_sync_backend/examples/gemini_rag_subtitle_sync.py \
    --video attachments/2mins.mp4 \
    --subtitle attachments/2mins_inaccurate.srt \
    --output-dir /tmp/gemini_demo

Notes:
- This script uses a simple, transparent flow and mirrors some logic from the backend processing
  in a minimal way for demonstration.
- The RAG retrieval uses TF-IDF-like scoring if sentence-transformers is not available.
- Place sample files (video and subtitle) anywhere and pass their paths via CLI arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

# Try to import httpx which is already part of backend requirements
try:
    import httpx
except Exception:
    httpx = None  # We'll error with instructions if missing.

# --- Minimal subtitle parsing/formatting utilities (adapted for this demo) ---

import re

_TIME_RE_SRT = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})")
_TIME_RE_VTT = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<ms>\d{3})")
SUPPORTED_SUB_EXTENSIONS = {".srt", ".vtt", ".ass", ".ssa"}


@dataclass
class SubtitleEntry:
    index: int
    start: float
    end: float
    text: str


def _time_to_seconds(h: int, m: int, s: int, ms: int) -> float:
    return h * 3600 + m * 60 + s + ms / 1000.0


def _seconds_to_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    ms = int((seconds - int(seconds)) * 1000)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _seconds_to_vtt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    ms = int((seconds - int(seconds)) * 1000)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def detect_subtitle_format(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".srt"):
        return "srt"
    if lower.endswith(".vtt"):
        return "vtt"
    if lower.endswith(".ass") or lower.endswith(".ssa"):
        return "ass"
    return "unknown"


def parse_srt(content: str) -> List[SubtitleEntry]:
    entries: List[SubtitleEntry] = []
    blocks = re.split(r"\n\s*\n", content.strip(), flags=re.MULTILINE)
    idx_counter = 1
    for block in blocks:
        lines = [l.strip("\ufeff") for l in block.strip().splitlines()]
        if not lines:
            continue
        line_idx = 0
        index = idx_counter
        if re.fullmatch(r"\d+", lines[0].strip()):
            try:
                index = int(lines[0].strip())
                line_idx = 1
            except ValueError:
                index = idx_counter

        if line_idx >= len(lines):
            continue

        times_line = lines[line_idx]
        line_idx += 1
        if "-->" not in times_line:
            continue
        start_str, end_str = [t.strip() for t in times_line.split("-->")]
        m1 = _TIME_RE_SRT.search(start_str)
        m2 = _TIME_RE_SRT.search(end_str)
        if not m1 or not m2:
            continue
        start = _time_to_seconds(int(m1["h"]), int(m1["m"]), int(m1["s"]), int(m1["ms"]))
        end = _time_to_seconds(int(m2["h"]), int(m2["m"]), int(m2["s"]), int(m2["ms"]))
        text = "\n".join(lines[line_idx:])
        entries.append(SubtitleEntry(index=index, start=start, end=end, text=text))
        idx_counter += 1
    return entries


def format_srt(entries: List[SubtitleEntry]) -> str:
    out_lines: List[str] = []
    for i, e in enumerate(entries, start=1):
        out_lines.append(str(i))
        out_lines.append(f"{_seconds_to_srt_time(e.start)} --> {_seconds_to_srt_time(e.end)}")
        out_lines.extend(e.text.splitlines())
        out_lines.append("")
    return "\n".join(out_lines).strip() + "\n"


def parse_vtt(content: str) -> List[SubtitleEntry]:
    lines = content.strip().splitlines()
    idx = 0
    if lines and lines[0].strip().upper().startswith("WEBVTT"):
        idx = 1
    buf = "\n".join(lines[idx:])
    entries: List[SubtitleEntry] = []
    blocks = re.split(r"\n\s*\n", buf.strip(), flags=re.MULTILINE)
    idx_counter = 1
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        times_line = lines[0]
        if "-->" not in times_line:
            continue
        start_str, end_str = [t.strip() for t in times_line.split("-->")]
        m1 = _TIME_RE_VTT.search(start_str)
        m2 = _TIME_RE_VTT.search(end_str)
        if not m1 or not m2:
            continue
        start = _time_to_seconds(int(m1["h"]), int(m1["m"]), int(m1["s"]), int(m1["ms"]))
        end = _time_to_seconds(int(m2["h"]), int(m2["m"]), int(m2["s"]), int(m2["ms"]))
        text = "\n".join(lines[1:])
        entries.append(SubtitleEntry(index=idx_counter, start=start, end=end, text=text))
        idx_counter += 1
    return entries


def format_vtt(entries: List[SubtitleEntry]) -> str:
    out_lines: List[str] = ["WEBVTT", ""]
    for e in entries:
        out_lines.append(f"{_seconds_to_vtt_time(e.start)} --> {_seconds_to_vtt_time(e.end)}")
        out_lines.extend(e.text.splitlines())
        out_lines.append("")
    return "\n".join(out_lines).strip() + "\n"


def parse_ass_ssa(content: str) -> Tuple[str, List[SubtitleEntry]]:
    header_lines: List[str] = []
    entries: List[SubtitleEntry] = []
    in_events = False
    index = 1
    for line in content.splitlines():
        if line.strip().startswith("[Events]"):
            in_events = True
            header_lines.append(line)
            continue
        if not in_events:
            header_lines.append(line)
            continue
        if line.startswith("Dialogue:"):
            try:
                _, rest = line.split(":", 1)
                parts = rest.split(",", 9)
                if len(parts) >= 10:
                    start = _parse_ass_time(parts[1].strip())
                    end = _parse_ass_time(parts[2].strip())
                    text = parts[9]
                    entries.append(SubtitleEntry(index=index, start=start, end=end, text=text))
                    index += 1
            except Exception:
                pass
        else:
            header_lines.append(line)
    header = "\n".join(header_lines)
    return header, entries


def format_ass_ssa(header: str, entries: List[SubtitleEntry]) -> str:
    out_lines = header.splitlines()
    if not any(l.strip().startswith("[Events]") for l in out_lines):
        out_lines.append("[Events]")
    for e in entries:
        start = _seconds_to_ass_time(e.start)
        end = _seconds_to_ass_time(e.end)
        dialogue = f"Dialogue: 0,{start},{end},Default,,0,0,0,,{e.text}"
        out_lines.append(dialogue)
    return "\n".join(out_lines) + "\n"


def _parse_ass_time(ts: str) -> float:
    m = re.match(r"(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<cs>\d{2})", ts)
    if not m:
        return 0.0
    h = int(m["h"])
    m_ = int(m["m"])
    s = int(m["s"])
    cs = int(m["cs"])
    return h * 3600 + m_ * 60 + s + cs / 100.0


def _seconds_to_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total_cs = int(round(seconds * 100))
    h = total_cs // (3600 * 100)
    rem = total_cs % (3600 * 100)
    m = rem // (60 * 100)
    rem = rem % (60 * 100)
    s = rem // 100
    cs = rem % 100
    return f"{h}:{m:02}:{s:02}.{cs:02}"


# --- Simple TF-IDF style vectorizer for RAG (no external dependency) ---

class SimpleVectorizer:
    def __init__(self) -> None:
        self._idf: Dict[str, float] = {}
        self._tokenize = self._simple_tokenize

    def _simple_tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return [t for t in text.split() if t]

    def build_idf(self, corpus: List[str]) -> None:
        df: Dict[str, int] = {}
        N = len(corpus) or 1
        for doc in corpus:
            for t in set(self._tokenize(doc)):
                df[t] = df.get(t, 0) + 1
        self._idf = {t: 1.0 + (0.0 if df[t] == 0 else (float(__import__("math").log(N / (1 + df[t]))))) for t in df}

    def vectorize(self, text: str) -> Dict[str, float]:
        toks = self._tokenize(text)
        if not toks:
            return {}
        tf: Dict[str, float] = {}
        for t in toks:
            tf[t] = tf.get(t, 0.0) + 1.0
        L = float(len(toks))
        for t in list(tf.keys()):
            tf[t] = tf[t] / L
        out: Dict[str, float] = {}
        for t, v in tf.items():
            idf = self._idf.get(t, 1.0)
            out[t] = v * idf
        return out

    @staticmethod
    def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        if len(a) < len(b):
            small, large = a, b
        else:
            small, large = b, a
        dot = 0.0
        for k, v in small.items():
            bv = large.get(k)
            if bv is not None:
                dot += v * bv
        import math
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return (dot / (na * nb)) if na > 0 and nb > 0 else 0.0

    def top_k(self, query: str, corpus: List[str], k: int = 10) -> List[Tuple[int, float]]:
        if not corpus:
            return []
        self.build_idf(corpus)
        corpus_vecs = [self.vectorize(c) for c in corpus]
        qv = self.vectorize(query)
        scores = [(i, self._cosine_sparse(qv, v)) for i, v in enumerate(corpus_vecs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[: max(1, k)]


# --- Gemini transcription via HTTP ---

class GeminiTranscriptionError(Exception):
    pass


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


# PUBLIC_INTERFACE
def transcribe_with_gemini(video_path: str, api_key_env: str = "GEMINI_API_KEY") -> List[TranscriptSegment]:
    """
    PUBLIC_INTERFACE
    Transcribe the provided video file using Gemini AI API (HTTP).

    Parameters:
        video_path: Absolute path to the video file (audio track will be used by Gemini's server-side processing).
        api_key_env: Environment variable name holding the Gemini API key (default: GEMINI_API_KEY).

    Returns:
        List[TranscriptSegment] with approximate start/end times (if model returns timestamps);
        if timestamps are not returned, segments will be naive splits.

    Notes:
        - This demo uses a plausible HTTP flow for Gemini; actual endpoints and request formats
          may differ depending on the model and API version. Adjust accordingly to your Gemini access.
        - If httpx is unavailable, raises an informative error.
    """
    if httpx is None:
        raise GeminiTranscriptionError(
            "httpx is required for HTTP calls. Please ensure it is installed (already in backend requirements)."
        )
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise GeminiTranscriptionError(
            f"Missing API key. Please set the environment variable {api_key_env}."
        )

    # Placeholder endpoint and headers for illustrative purposes.
    # Replace 'model' with a transcription-capable Gemini model you have access to.
    # For example, "models/gemini-1.5-pro" with audio input support.
    GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    # For demo purposes, send a "task description" and attach file as base64 bytes via "inline_data".
    # In reality, you might upload the media to GCS and reference it, or use a multipart endpoint if supported.
    import base64
    try:
        with open(video_path, "rb") as vf:
            video_bytes = vf.read()
    except Exception as e:
        raise GeminiTranscriptionError(f"Failed to read video file: {e}") from e

    b64 = base64.b64encode(video_bytes).decode("utf-8")

    # The prompt asks for a transcript broken into segments with start/end times if possible.
    # Some Gemini multimodal models can infer timestamps; if not, we'll split text afterward.
    system_instruction = (
        "You are a transcription assistant. Transcribe the following video/audio input. "
        "If possible, provide JSON with segments including start, end (in seconds), and text. "
        "If precise timestamps are unavailable, still return a coherent transcript text."
    )

    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": system_instruction},
                    {
                        "inline_data": {
                            "mime_type": "video/mp4",
                            "data": b64,
                        }
                    },
                ],
            }
        ]
    }

    try:
        resp = httpx.post(GEMINI_ENDPOINT, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise GeminiTranscriptionError(f"Gemini HTTP request failed: {e}") from e

    # Attempt to extract structured segments from Gemini response.
    # Adjust this parsing logic to match your actual Gemini response schema.
    text_response = ""
    try:
        # Typical structure: candidates[0].content.parts[].text
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for p in parts:
                if "text" in p:
                    text_response += p["text"] + "\n"
    except Exception:
        pass

    text_response = (text_response or "").strip()

    # Try to parse JSON segment structure if present.
    segments: List[TranscriptSegment] = []
    parsed_json = None
    if text_response:
        # Heuristic: search for a JSON-like array of segments in the output
        try:
            # Basic extraction: look for the last JSON-looking block
            start_idx = text_response.find("[")
            end_idx = text_response.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                block = text_response[start_idx : end_idx + 1]
                parsed_json = json.loads(block)
        except Exception:
            parsed_json = None

    if isinstance(parsed_json, list):
        for item in parsed_json:
            try:
                start = float(item.get("start", 0.0))
                end = float(item.get("end", max(start + 1.0, 1.0)))
                text = str(item.get("text", "")).strip()
                if text:
                    segments.append(TranscriptSegment(start=start, end=end, text=text))
            except Exception:
                continue

    # Fallback: if no structured segments, split plain transcript into rough chunks.
    if not segments:
        if not text_response:
            raise GeminiTranscriptionError("No transcription text returned by Gemini.")
        # Split into sentences/lines; assign naive timings (e.g., 3s per line).
        lines = [ln.strip() for ln in re.split(r"[.\n]+", text_response) if ln.strip()]
        start = 0.0
        approx_dur = 3.5
        for i, ln in enumerate(lines):
            seg = TranscriptSegment(start=start, end=start + approx_dur, text=ln + ".")
            segments.append(seg)
            start += approx_dur

    return segments


# --- Core demo pipeline: estimate shift, RAG retrieval, and write outputs ---

def _estimate_shift_from_transcript(transcript: List[TranscriptSegment], subs: List[SubtitleEntry]) -> float:
    first_sub = next((s for s in subs if s.text.strip()), None)
    first_seg = next((t for t in transcript if t.text.strip()), None)
    if not first_sub or not first_seg:
        return 0.0
    return max(0.0, first_seg.start) - max(0.0, first_sub.start)


def _apply_shift(entries: List[SubtitleEntry], shift: float) -> List[SubtitleEntry]:
    out: List[SubtitleEntry] = []
    for e in entries:
        out.append(SubtitleEntry(index=e.index, start=e.start + shift, end=e.end + shift, text=e.text))
    return out


def _clamp_nonnegative(entries: List[SubtitleEntry]) -> List[SubtitleEntry]:
    out: List[SubtitleEntry] = []
    for e in entries:
        out.append(SubtitleEntry(index=e.index, start=max(0.0, e.start), end=max(0.0, e.end), text=e.text))
    return out


def _read_subtitle(fmt: str, path: str) -> Tuple[str, List[SubtitleEntry]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    header = ""
    if fmt == "srt":
        entries = parse_srt(content)
    elif fmt == "vtt":
        entries = parse_vtt(content)
    elif fmt in ("ass",):
        header, entries = parse_ass_ssa(content)
    else:
        raise ValueError("Unsupported subtitle format for this demo.")
    return header, entries


def _write_subtitle(fmt: str, path: str, header: str, entries: List[SubtitleEntry]) -> None:
    if fmt == "srt":
        out = format_srt(entries)
    elif fmt == "vtt":
        out = format_vtt(entries)
    else:
        out = format_ass_ssa(header, entries)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)


def _rag_top_k_for_subtitles(subs: List[SubtitleEntry], transcript_entries: List[Dict[str, Any]], k: int = 10) -> List[List[Dict[str, Any]]]:
    vec = SimpleVectorizer()
    corpus = [str(e.get("text", "")) for e in transcript_entries]
    results: List[List[Dict[str, Any]]] = []
    for cue in subs:
        scores = vec.top_k(cue.text, corpus, k=k)
        top: List[Dict[str, Any]] = []
        for idx, sc in scores:
            item = dict(transcript_entries[idx])
            item["score"] = float(sc)
            top.append(item)
        results.append(top)
    return results


def _ensure_out_dir(out_dir: Optional[str]) -> str:
    if out_dir:
        od = os.path.abspath(out_dir)
        os.makedirs(od, exist_ok=True)
        return od
    # default temp dir
    od = tempfile.mkdtemp(prefix="gemini_rag_demo_")
    return od


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end subtitle syncing using Gemini AI transcription and simple RAG.")
    parser.add_argument("--video", required=False, default="attachments/2mins.mp4", help="Path to the video file.")
    parser.add_argument("--subtitle", required=False, default="attachments/2mins_inaccurate.srt", help="Path to the subtitle file (.srt/.vtt/.ass/.ssa).")
    parser.add_argument("--output-dir", default=None, help="Directory to write outputs. Defaults to a temporary directory.")
    parser.add_argument("--k", type=int, default=10, help="Top-K transcript segments to retrieve per cue for RAG.")
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

    out_dir = _ensure_out_dir(args.output_dir)

    fmt = detect_subtitle_format(subtitle_path)
    if fmt == "unknown":
        print("Unsupported subtitle format. Use .srt, .vtt, .ass, or .ssa", file=sys.stderr)
        return 2

    print("=== Gemini RAG Subtitle Sync Demo ===")
    print(f"Video:      {video_path}")
    print(f"Subtitle:   {subtitle_path}")
    print(f"Output dir: {out_dir}")
    print("Step 1) Transcribing with Gemini...")

    try:
        segments = transcribe_with_gemini(video_path=video_path, api_key_env="GEMINI_API_KEY")
    except GeminiTranscriptionError as e:
        print(f"Transcription failed: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"Unexpected error during transcription: {e}", file=sys.stderr)
        return 4

    # Normalize transcript entries for RAG and persistence
    entries: List[Dict[str, Any]] = [{"start": float(s.start), "end": float(s.end), "text": s.text} for s in segments]

    entries_json = os.path.join(out_dir, "gemini_transcript_entries.json")
    try:
        with open(entries_json, "w", encoding="utf-8") as jf:
            json.dump(entries, jf, ensure_ascii=False, indent=2)
        print(f"  Wrote transcript entries JSON: {entries_json}")
    except Exception as e:
        print(f"  Warning: failed to write transcript entries JSON: {e}")

    print("Step 2) Parsing subtitle and estimating time shift...")
    try:
        header, sub_entries = _read_subtitle(fmt, subtitle_path)
    except Exception as e:
        print(f"Failed to parse subtitle: {e}", file=sys.stderr)
        return 5
    if not sub_entries:
        print("No subtitle entries parsed; aborting.", file=sys.stderr)
        return 6

    shift = _estimate_shift_from_transcript(segments, sub_entries)
    shifted = _apply_shift(sub_entries, shift)
    shifted = _clamp_nonnegative(shifted)

    base_name, ext = os.path.splitext(os.path.basename(subtitle_path))
    synced_name = f"{base_name}-synced{ext}"
    synced_path = os.path.join(out_dir, synced_name)
    try:
        _write_subtitle(fmt, synced_path, header, shifted)
        print(f"  Wrote synced subtitle: {synced_path}")
        print(f"  Applied shift (secs): {shift:.3f}")
    except Exception as e:
        print(f"Failed to write synced subtitle: {e}", file=sys.stderr)
        return 7

    print("Step 3) Running simple RAG (top-K retrieval per cue)...")
    topk_per_cue = _rag_top_k_for_subtitles(shifted, entries, k=int(args.k))
    # Save a small preview JSON for the first few cues
    preview_json = os.path.join(out_dir, "rag_preview.json")
    try:
        preview = []
        for i in range(min(5, len(shifted))):
            preview.append({
                "index": shifted[i].index,
                "text": shifted[i].text,
                "topk_sample": [
                    {"start": r.get("start"), "end": r.get("end"), "score": r.get("score"), "text": r.get("text")}
                    for r in (topk_per_cue[i] if i < len(topk_per_cue) else [])[:3]
                ]
            })
        with open(preview_json, "w", encoding="utf-8") as jf:
            json.dump(preview, jf, ensure_ascii=False, indent=2)
        print(f"  Wrote RAG preview JSON: {preview_json}")
    except Exception as e:
        print(f"  Warning: failed to write RAG preview JSON: {e}")

    print("")
    print("All steps completed.")
    print("Reminder:")
    print("- Set GEMINI_API_KEY in your environment before running this script.")
    print("- Place your sample files anywhere and point to them with --video and --subtitle.")
    print("- This script focuses on Gemini transcription and basic RAG retrieval;")
    print("  for LLM-based correction using OpenAI-compatible models, see example_end_to_end_rag_llm.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Processing module for video audio transcription and subtitle synchronization.

This module encapsulates the core business logic for:
- Extracting audio (optional)
- Transcribing audio using a pluggable transcriber (Whisper-compatible interface)
- Parsing and adjusting subtitle timestamps (SRT/VTT/ASS/SSA)
- Producing a synchronized subtitle file

Design notes:
- PUBLIC_INTERFACE methods are provided for use by API layer without exposing internal details.
- No heavy ML dependencies are mandated here; a stub and an interface are provided. If Whisper is available,
  wire up WhisperTranscriber to use it.
- File operations write to a working directory and return paths for the API to serve.
"""
from __future__ import annotations

import os
import re
import uuid
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Tuple, Dict, Any

# RAG retrieval utilities
from .rag import retrieve_top_k_for_cues

# Supported subtitle extensions
SUPPORTED_SUB_EXTENSIONS = {".srt", ".vtt", ".ass", ".ssa"}

# ---------------------------------------------
# Transcription interfaces and implementations
# ---------------------------------------------

class TranscriptionError(Exception):
    """Raised when transcription fails."""


class SubtitleSyncError(Exception):
    """Raised when subtitle synchronization fails."""


@dataclass
class TranscriptSegment:
    """Represents a single transcript segment with timing and text."""
    start: float  # seconds
    end: float    # seconds
    text: str


class BaseTranscriber:
    """Base class interface for transcribers."""

    # PUBLIC_INTERFACE
    def transcribe(self, audio_or_video_path: str) -> List[TranscriptSegment]:
        """Transcribe the given audio or video file and return segments.

        Returns:
            List[TranscriptSegment]: Ordered, contiguous or non-contiguous segments that cover the full audio where possible.
        """
        raise NotImplementedError


class StubTranscriber(BaseTranscriber):
    """
    A stub transcriber that returns a few fake segments.
    Useful for development without installing Whisper.
    """

    # PUBLIC_INTERFACE
    def transcribe(self, audio_or_video_path: str) -> List[TranscriptSegment]:
        # Simulate a short 30-second clip split into 3 segments
        return [
            TranscriptSegment(0.0, 10.0, "Hello and welcome."),
            TranscriptSegment(10.0, 20.0, "This is a sample transcription."),
            TranscriptSegment(20.0, 30.0, "We will sync subtitles accordingly."),
        ]


class WhisperTranscriber(BaseTranscriber):
    """
    Whisper-based transcriber (optional).
    Requires 'openai-whisper' or 'whisper' package and ffmpeg availability.
    This class is instantiated only if the environment has the dependency.
    """
    def __init__(self, model_name: str = "base"):
        try:
            import whisper  # type: ignore
        except Exception as exc:
            raise TranscriptionError(
                "Whisper dependency not installed. Please add 'openai-whisper' to requirements and system ffmpeg."
            ) from exc
        self._whisper = whisper
        self._model = whisper.load_model(model_name)

    # PUBLIC_INTERFACE
    def transcribe(self, audio_or_video_path: str) -> List[TranscriptSegment]:
        try:
            result = self._model.transcribe(audio_or_video_path)
            segments: List[TranscriptSegment] = []
            for seg in result.get("segments", []):
                segments.append(
                    TranscriptSegment(
                        start=float(seg.get("start", 0.0)),
                        end=float(seg.get("end", 0.0)),
                        text=str(seg.get("text", "")).strip(),
                    )
                )
            if not segments and "text" in result:
                # Single blob fallback
                text = str(result["text"]).strip()
                segments.append(TranscriptSegment(0.0, 0.0, text))
            return segments
        except Exception as exc:
            raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc


# ---------------------------------------------
# Subtitle parsing and formatting utilities
# ---------------------------------------------

_TIME_RE_SRT = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})"
)
_TIME_RE_VTT = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<ms>\d{3})"
)

@dataclass
class SubtitleEntry:
    index: int
    start: float  # seconds
    end: float    # seconds
    text: str


def _time_to_seconds(h: int, m: int, s: int, ms: int) -> float:
    return h * 3600 + m * 60 + s + ms / 1000.0


def _seconds_to_srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    ms = int((seconds - int(seconds)) * 1000)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _seconds_to_vtt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    ms = int((seconds - int(seconds)) * 1000)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def parse_srt(content: str) -> List[SubtitleEntry]:
    entries: List[SubtitleEntry] = []
    blocks = re.split(r"\n\s*\n", content.strip(), flags=re.MULTILINE)
    idx_counter = 1
    for block in blocks:
        lines = [l.strip("\ufeff") for l in block.strip().splitlines()]
        if not lines:
            continue
        # Some SRTs include index on first line
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
            # malformed
            continue
        start_str, end_str = [t.strip() for t in times_line.split("-->")]
        ms_match_start = _TIME_RE_SRT.search(start_str)
        ms_match_end = _TIME_RE_SRT.search(end_str)
        if not ms_match_start or not ms_match_end:
            continue
        start = _time_to_seconds(
            int(ms_match_start["h"]),
            int(ms_match_start["m"]),
            int(ms_match_start["s"]),
            int(ms_match_start["ms"]),
        )
        end = _time_to_seconds(
            int(ms_match_end["h"]),
            int(ms_match_end["m"]),
            int(ms_match_end["s"]),
            int(ms_match_end["ms"]),
        )
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
        out_lines.append("")  # blank line
    return "\n".join(out_lines).strip() + "\n"


def parse_vtt(content: str) -> List[SubtitleEntry]:
    lines = content.strip().splitlines()
    # Skip WEBVTT header if present
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
        ms_match_start = _TIME_RE_VTT.search(start_str)
        ms_match_end = _TIME_RE_VTT.search(end_str)
        if not ms_match_start or not ms_match_end:
            continue
        start = _time_to_seconds(
            int(ms_match_start["h"]),
            int(ms_match_start["m"]),
            int(ms_match_start["s"]),
            int(ms_match_start["ms"]),
        )
        end = _time_to_seconds(
            int(ms_match_end["h"]),
            int(ms_match_end["m"]),
            int(ms_match_end["s"]),
            int(ms_match_end["ms"]),
        )
        text = "\n".join(lines[1:])
        entries.append(SubtitleEntry(index=idx_counter, start=start, end=end, text=text))
        idx_counter += 1
    return entries


def format_vtt(entries: List[SubtitleEntry]) -> str:
    out_lines: List[str] = ["WEBVTT", ""]
    for e in entries:
        out_lines.append(f"{_seconds_to_vtt_time(e.start)} --> {_seconds_to_vtt_time(e.end)}")
        out_lines.extend(e.text.splitlines())
        out_lines.append("")  # blank line
    return "\n".join(out_lines).strip() + "\n"


def detect_subtitle_format(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".srt"):
        return "srt"
    if lower.endswith(".vtt"):
        return "vtt"
    if lower.endswith(".ass") or lower.endswith(".ssa"):
        return "ass"
    return "unknown"


# Minimal ASS/SSA passthrough: We don't parse deeply; we time-shift uniformly
def parse_ass_ssa(content: str) -> Tuple[str, List[SubtitleEntry]]:
    """
    Naive parser: retain header lines, parse Dialogue lines to extract start/end,
    but keep original line texts for later reconstruction.
    Returns header_text, entries.
    """
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
        # Event lines look like:
        # Dialogue: 0,0:00:03.45,0:00:05.23,Default,,0,0,0,,Subtitle text
        if line.startswith("Dialogue:"):
            try:
                prefix, rest = line.split(":", 1)
                parts = rest.split(",", 9)  # split into 10 parts max
                if len(parts) >= 10:
                    # parts[0] is 'layer' which we don't use
                    start = _parse_ass_time(parts[1].strip())
                    end = _parse_ass_time(parts[2].strip())
                    text = parts[9]  # remaining after 9 commas
                    entries.append(SubtitleEntry(index=index, start=start, end=end, text=text))
                    index += 1
            except Exception:
                # ignore malformed line
                pass
        else:
            header_lines.append(line)
    header = "\n".join(header_lines)
    return header, entries


def format_ass_ssa(header: str, entries: List[SubtitleEntry]) -> str:
    """
    Reconstructs a basic ASS/SSA file. Dialogue fields except times/text are left simple.
    This keeps the original header as much as possible.
    """
    out_lines = header.splitlines()
    # Ensure Events header present
    if not any(l.strip().startswith("[Events]") for l in out_lines):
        out_lines.append("[Events]")
    # We will append Dialogue lines after header
    for e in entries:
        start = _seconds_to_ass_time(e.start)
        end = _seconds_to_ass_time(e.end)
        # Very simple default Dialogue structure
        dialogue = f"Dialogue: 0,{start},{end},Default,,0,0,0,,{e.text}"
        out_lines.append(dialogue)
    return "\n".join(out_lines) + "\n"


def _parse_ass_time(ts: str) -> float:
    # H:MM:SS.cs (centiseconds)
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


# ---------------------------------------------
# Core synchronization logic
# ---------------------------------------------

def _estimate_shift_from_transcript(transcript: List[TranscriptSegment], subs: List[SubtitleEntry]) -> float:
    """
    Naive shift estimation:
    - If both have content, align the first non-empty subtitle to first transcript start.
    - Compute shift so that sub.start matches transcript.start.
    Returns shift in seconds (add this to subtitle times).
    """
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


# PUBLIC_INTERFACE
def process_and_sync_subtitles(
    video_path: str,
    subtitle_path: str,
    output_dir: str,
    prefer_whisper: bool = False,
    whisper_model: str = "base",
) -> Dict[str, Any]:
    """
    Process video and subtitle files:
    - Transcribe audio using StubTranscriber by default, or Whisper if prefer_whisper=True and available.
    - Parse subtitles
    - Estimate shift using naive approach and apply
    - Write adjusted subtitle to output_dir with suffix '-synced'
    - Persist full transcription segments as 'entries' JSON for later RAG/LLM usage

    Returns:
    {
      "synced_subtitle_path": "<path>",
      "format": "srt|vtt|ass",
      "shift_seconds": "<float as str>",
      "entries": [ { "start": float, "end": float, "text": str }, ... ],
      "entries_path": "<absolute path to JSON file with entries>"
    }
    """
    os.makedirs(output_dir, exist_ok=True)
    # Choose transcriber
    transcriber: BaseTranscriber = StubTranscriber()
    if prefer_whisper:
        try:
            transcriber = WhisperTranscriber(model_name=whisper_model)
        except TranscriptionError:
            # Fallback to stub if whisper unavailable
            transcriber = StubTranscriber()

    # Transcribe entire video
    transcript = transcriber.transcribe(video_path)

    # Serialize transcript segments into dicts for portability
    entries: List[Dict[str, Any]] = [
        {"start": float(seg.start), "end": float(seg.end), "text": seg.text}
        for seg in transcript
    ]

    # Also persist to JSON file for downstream RAG steps within same workdir
    entries_path = os.path.join(output_dir, "transcript_entries.json")
    try:
        import json

        with open(entries_path, "w", encoding="utf-8") as jf:
            json.dump(entries, jf, ensure_ascii=False, indent=2)
    except Exception:
        # Do not fail main processing if persistence fails; entries are still returned in-memory.
        entries_path = ""

    # Parse subtitle
    fmt = detect_subtitle_format(subtitle_path)
    if fmt == "unknown":
        raise SubtitleSyncError("Unsupported subtitle format. Use .srt, .vtt, .ass, or .ssa")

    with open(subtitle_path, "r", encoding="utf-8", errors="ignore") as f:
        sub_content = f.read()

    header = ""
    sub_entries: List[SubtitleEntry] = []
    if fmt == "srt":
        sub_entries = parse_srt(sub_content)
    elif fmt == "vtt":
        sub_entries = parse_vtt(sub_content)
    else:
        header, sub_entries = parse_ass_ssa(sub_content)

    if not sub_entries:
        raise SubtitleSyncError("No subtitle entries parsed.")

    # Estimate shift and apply
    shift = _estimate_shift_from_transcript(transcript, sub_entries)
    shifted = _apply_shift(sub_entries, shift)
    shifted = _clamp_nonnegative(shifted)

    # Format output
    base_name, ext = os.path.splitext(os.path.basename(subtitle_path))
    out_name = f"{base_name}-synced{ext}"
    out_path = os.path.join(output_dir, out_name)

    if fmt == "srt":
        content_out = format_srt(shifted)
    elif fmt == "vtt":
        content_out = format_vtt(shifted)
    else:
        content_out = format_ass_ssa(header, shifted)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content_out)

    # Attempt LLM correction pass with RAG context; write a sibling file if successful.
    corrected_variant_path = ""
    try:
        from .correction import correct_subtitles_with_llm, apply_corrected_texts  # local import to avoid cycles
        # Compute RAG and corrected texts (top-10 segments per cue)
        corrected_texts, _rag = correct_subtitles_with_llm(shifted, entries, fmt=fmt, k=10)
        corrected_entries = apply_corrected_texts(shifted, corrected_texts)
        # Write corrected file alongside synced one
        base_name_noext, ext = os.path.splitext(os.path.basename(out_path))
        corrected_name = f"{base_name_noext}-llm{ext}"
        corrected_variant_path = os.path.join(output_dir, corrected_name)
        if fmt == "srt":
            corrected_content = format_srt(corrected_entries)
        elif fmt == "vtt":
            corrected_content = format_vtt(corrected_entries)
        else:
            corrected_content = format_ass_ssa(header, corrected_entries)
        with open(corrected_variant_path, "w", encoding="utf-8") as cf:
            cf.write(corrected_content)
    except Exception:
        # LLM correction is optional; ignore failures to keep base pipeline robust.
        corrected_variant_path = ""

    return {
        "synced_subtitle_path": out_path,
        "format": fmt,
        "shift_seconds": f"{shift:.3f}",
        "entries": entries,           # full transcription segments for RAG/LLM
        "entries_path": entries_path, # persisted JSON location (may be empty if persistence failed)
        # Expose optional corrected variant path for clients that want it
        "llm_corrected_subtitle_path": corrected_variant_path or None,
    }


# PUBLIC_INTERFACE
def create_workdir(root_dir: Optional[str] = None) -> str:
    """Create and return a new unique working directory for a job."""
    base = root_dir or tempfile.gettempdir()
    job_id = uuid.uuid4().hex
    workdir = os.path.join(base, f"subtitle_sync_{job_id}")
    os.makedirs(workdir, exist_ok=True)
    return workdir


# PUBLIC_INTERFACE
def save_upload(file_bytes: bytes, filename: str, dest_dir: str) -> str:
    """
    Save uploaded file bytes into dest_dir and return the absolute file path.
    """
    os.makedirs(dest_dir, exist_ok=True)
    safe_name = filename.replace("/", "_").replace("\\", "_")
    out_path = os.path.join(dest_dir, safe_name)
    with open(out_path, "wb") as f:
        f.write(file_bytes)
    return out_path


# PUBLIC_INTERFACE
def rag_top_k_for_subtitles(
    subtitle_entries: List[SubtitleEntry],
    transcript_entries: List[Dict[str, Any]],
    k: int = 10,
) -> List[List[Dict[str, Any]]]:
    """
    Compute top-k relevant transcript segments for each subtitle cue.

    Parameters:
        subtitle_entries: Parsed subtitle cues as SubtitleEntry objects.
        transcript_entries: Full-video transcription 'entries' as returned by process_and_sync_subtitles.
        k: Number of segments to retrieve for each cue (default 10).

    Returns:
        List[List[Dict[str, Any]]]: For each subtitle cue, a list of up to k transcript entries
        augmented with a 'score' field, sorted by descending similarity.
    """
    cues = [{"text": s.text, "start": s.start, "end": s.end} for s in subtitle_entries]
    return retrieve_top_k_for_cues(cues, transcript_entries, k=k)

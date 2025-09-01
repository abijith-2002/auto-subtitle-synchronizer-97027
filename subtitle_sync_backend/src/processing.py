import hashlib
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# Note: We import whisper lazily where used to avoid import overhead at module import time.

# Directory structure configuration (relative to project container)
BASE_DIR = Path(__file__).resolve().parent.parent  # subtitle_sync_backend/src
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
OUTPUTS_DIR = STORAGE_DIR / "outputs"
HISTORY_FILE = STORAGE_DIR / "history.json"

# Ensure directories exist
for d in [STORAGE_DIR, UPLOADS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class TaskProgress:
    """Represents task progress and status."""
    task_id: str
    status: str  # queued|processing|completed|error
    progress: float  # 0..1
    message: str = ""
    created_at: float = time.time()
    updated_at: float = time.time()
    video_filename: Optional[str] = None
    subtitle_filename: Optional[str] = None
    output_filename: Optional[str] = None
    model: Optional[str] = "base"

    def to_dict(self) -> Dict:
        return asdict(self)


# In-memory task registry for simplicity; replace with persistent store if needed
_TASKS: Dict[str, TaskProgress] = {}
_HISTORY: List[Dict] = []  # simple in-memory cache; saved to HISTORY_FILE


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _now_ts() -> float:
    return time.time()


def _write_history():
    """Persist in-memory history to disk."""
    try:
        import json
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(_HISTORY, f, indent=2)
    except Exception:
        # Best-effort; ignore history write failures
        pass


def _load_history():
    """Load history from disk on module import."""
    global _HISTORY
    try:
        import json
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                _HISTORY = json.load(f)
    except Exception:
        _HISTORY = []


_load_history()


# PUBLIC_INTERFACE
def create_task(video_bytes: bytes, video_filename: str, subtitle_bytes: bytes, subtitle_filename: str, model: str = "base") -> str:
    """Create a processing task by saving uploads to disk and returning a task_id.

    Stores the uploaded video and subtitle files, registers a new task with queued status,
    and returns the generated task_id to be used for sync initiation and querying.

    Args:
        video_bytes: Raw bytes of the uploaded video file.
        video_filename: Original filename of the video.
        subtitle_bytes: Raw bytes of the uploaded subtitle file.
        subtitle_filename: Original filename of the subtitle.
        model: Whisper model name to use for transcription (default "base").

    Returns:
        A unique task_id string.
    """
    upload_id = _hash_bytes(video_bytes + subtitle_bytes + str(_now_ts()).encode())
    task_id = upload_id

    # Create a per-task directory
    task_dir = UPLOADS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Normalize filenames
    safe_video_name = _safe_filename(video_filename)
    safe_sub_name = _safe_filename(subtitle_filename)

    video_path = task_dir / safe_video_name
    sub_path = task_dir / safe_sub_name

    with open(video_path, "wb") as f:
        f.write(video_bytes)
    with open(sub_path, "wb") as f:
        f.write(subtitle_bytes)

    progress = TaskProgress(
        task_id=task_id,
        status="queued",
        progress=0.0,
        message="Task created, awaiting sync trigger",
        created_at=_now_ts(),
        updated_at=_now_ts(),
        video_filename=safe_video_name,
        subtitle_filename=safe_sub_name,
        model=model,
    )
    _TASKS[task_id] = progress
    return task_id


# PUBLIC_INTERFACE
def get_progress(task_id: str) -> Optional[Dict]:
    """Get progress for a task.

    Args:
        task_id: ID of the task.

    Returns:
        Dict with progress info, or None if not found.
    """
    p = _TASKS.get(task_id)
    return p.to_dict() if p else None


# PUBLIC_INTERFACE
def get_history(limit: int = 50) -> List[Dict]:
    """Return recent processing history.

    Args:
        limit: Maximum number of entries to return.
    """
    return list(reversed(_HISTORY[-limit:]))


# PUBLIC_INTERFACE
def get_output_file_path(task_id: str) -> Optional[Path]:
    """Return the path to the output synced subtitle file for a given task if available."""
    prog = _TASKS.get(task_id)
    if not prog or not prog.output_filename:
        return None
    out_path = OUTPUTS_DIR / task_id / prog.output_filename
    return out_path if out_path.exists() else None


# PUBLIC_INTERFACE
async def start_sync(task_id: str) -> None:
    """Start subtitle syncing for a task asynchronously.

    This transcribes audio from the uploaded video using Whisper and aligns the subtitle timestamps
    to best match the transcription text. Stores the resulting synced subtitle in outputs.
    """
    progress = _TASKS.get(task_id)
    if not progress:
        return

    if progress.status in ("processing", "completed"):
        return

    progress.status = "processing"
    progress.progress = 0.01
    progress.message = "Initializing transcription"
    progress.updated_at = _now_ts()

    try:
        task_dir = UPLOADS_DIR / task_id
        video_path = task_dir / (progress.video_filename or "")
        sub_path = task_dir / (progress.subtitle_filename or "")

        if not video_path.exists() or not sub_path.exists():
            raise FileNotFoundError("Uploaded files not found")

        # Step 1: Transcribe with Whisper
        progress.message = "Loading Whisper model"
        progress.progress = 0.05
        progress.updated_at = _now_ts()

        # Lazy import to keep module import fast
        import whisper  # type: ignore

        model_name = progress.model or "base"
        model = whisper.load_model(model_name)

        progress.message = "Transcribing audio"
        progress.progress = 0.10
        progress.updated_at = _now_ts()

        # Whisper transcribe; to keep it simple for CPU environments
        # Note: In real deployments, consider specifying fp16=False when no GPU is available.
        result = model.transcribe(str(video_path), fp16=False)
        segments = result.get("segments", [])

        # Build list of transcript cues: (start, end, text)
        transcript: List[Tuple[float, float, str]] = []
        for seg in segments:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", s))
            txt = str(seg.get("text", "")).strip()
            if txt:
                transcript.append((s, e, txt))

        if not transcript:
            raise RuntimeError("No transcription produced by Whisper")

        progress.message = "Parsing subtitle file"
        progress.progress = 0.40
        progress.updated_at = _now_ts()

        # Step 2: Parse subtitles
        sub_format = _detect_subtitle_format(sub_path.name)
        cues = _parse_subtitle(sub_path, sub_format)

        progress.message = "Computing time offset"
        progress.progress = 0.55
        progress.updated_at = _now_ts()

        # Step 3: Compute alignment offset (simple text-similarity heuristic)
        offset = _estimate_time_offset(cues, transcript)

        progress.message = f"Applying offset ({offset:+.2f}s) and writing output"
        progress.progress = 0.75
        progress.updated_at = _now_ts()

        # Step 4: Apply offset and save
        adjusted_cues = _apply_offset(cues, offset)
        out_task_dir = OUTPUTS_DIR / task_id
        out_task_dir.mkdir(parents=True, exist_ok=True)
        out_name = _replace_ext(progress.subtitle_filename or "synced.srt", "srt")
        out_path = out_task_dir / out_name

        _write_srt(adjusted_cues, out_path)

        # Update history
        history_entry = {
            "task_id": task_id,
            "video_filename": progress.video_filename,
            "subtitle_filename": progress.subtitle_filename,
            "output_filename": out_name,
            "model": model_name,
            "offset_seconds": offset,
            "created_at": progress.created_at,
            "completed_at": _now_ts(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        _HISTORY.append(history_entry)
        _write_history()

        # Finalize progress
        progress.status = "completed"
        progress.progress = 1.0
        progress.message = "Sync completed"
        progress.output_filename = out_name
        progress.updated_at = _now_ts()

    except Exception as e:
        progress.status = "error"
        progress.message = f"Error: {e}"
        progress.updated_at = _now_ts()
        progress.progress = 1.0


# --------------------------
# Subtitle utilities
# --------------------------

@dataclass
class SubtitleCue:
    index: int
    start: float  # seconds
    end: float    # seconds
    text: str


def _safe_filename(name: str) -> str:
    keep = ("-", "_", ".", " ")
    return "".join(c for c in name if c.isalnum() or c in keep).strip().replace(" ", "_") or "file"


def _replace_ext(name: str, new_ext: str) -> str:
    base = os.path.splitext(name)[0]
    return f"{base}.{new_ext}"


def _detect_subtitle_format(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".srt",):
        return "srt"
    elif ext in (".vtt",):
        return "vtt"
    elif ext in (".ass", ".ssa"):
        return "ass"
    else:
        # Default to srt for writing, but parsing might fail for unknown
        return "srt"


def _parse_time_srt(time_str: str) -> float:
    # Format: HH:MM:SS,mmm
    hms, ms = time_str.split(",")
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s) + int(ms) / 1000.0


def _format_time_srt(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    ms = ms_total % 1000
    s_total = ms_total // 1000
    s = s_total % 60
    m_total = s_total // 60
    m = m_total % 60
    h = m_total // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_srt(path: Path) -> List[SubtitleCue]:
    cues: List[SubtitleCue] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    blocks = [b.strip() for b in content.replace("\r\n", "\n").split("\n\n") if b.strip()]
    idx = 0
    for block in blocks:
        lines = block.split("\n")
        if len(lines) < 2:
            continue
        # first line maybe index
        time_line_idx = 0
        if lines[0].strip().isdigit():
            time_line_idx = 1
        try:
            times = lines[time_line_idx]
            if "-->" not in times:
                continue
            start_str, end_str = [t.strip() for t in times.split("-->")]
            start = _parse_time_srt(start_str.replace(".", ","))  # normalize potential dots
            end = _parse_time_srt(end_str.replace(".", ","))
            text_lines = lines[time_line_idx + 1 :]
            text = "\n".join(text_lines).strip()
            idx += 1
            cues.append(SubtitleCue(index=idx, start=start, end=end, text=text))
        except Exception:
            continue
    return cues


def _write_srt(cues: List[SubtitleCue], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, cue in enumerate(cues, start=1):
            f.write(f"{i}\n")
            f.write(f"{_format_time_srt(cue.start)} --> {_format_time_srt(cue.end)}\n")
            f.write(f"{cue.text}\n\n")


def _parse_vtt(path: Path) -> List[SubtitleCue]:
    # Simple VTT parsing; ignores header and NOTE blocks
    cues: List[SubtitleCue] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f]

    # Skip WEBVTT header if present
    i = 0
    if i < len(lines) and lines[i].strip().upper().startswith("WEBVTT"):
        i += 1
    idx = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # Optional identifier line (non-timing)
        if "-->" not in line and (i + 1) < len(lines) and "-->" in lines[i + 1]:
            i += 1
            line = lines[i].strip()

        if "-->" in line:
            try:
                start_str, end_str = [t.strip().split(" ")[0] for t in line.split("-->")]
                # VTT uses '.' for milliseconds
                start = _parse_time_vtt(start_str)
                end = _parse_time_vtt(end_str)
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i])
                    i += 1
                text = "\n".join(text_lines).strip()
                idx += 1
                cues.append(SubtitleCue(index=idx, start=start, end=end, text=text))
            except Exception:
                pass
        i += 1
    return cues


def _parse_time_vtt(ts: str) -> float:
    # "HH:MM:SS.mmm" or "MM:SS.mmm"
    parts = ts.split(":")
    if len(parts) == 3:
        h = int(parts[0])
        m = int(parts[1])
        s_ms = float(parts[2])
        return h * 3600 + m * 60 + s_ms
    elif len(parts) == 2:
        m = int(parts[0])
        s_ms = float(parts[1])
        return m * 60 + s_ms
    else:
        return float(ts)


def _parse_ass(path: Path) -> List[SubtitleCue]:
    # Minimal ASS/SSA parsing: reads Dialogue lines, uses start/end and raw text (without styling)
    cues: List[SubtitleCue] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f]

    events = False
    format_fields: List[str] = []
    idx = 0
    for line in lines:
        if line.startswith("[Events]"):
            events = True
            continue
        if events and line.startswith("Format:"):
            # Example: Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
            fields = line[len("Format:"):].strip().split(",")
            format_fields = [fld.strip().lower() for fld in fields]
        if events and line.startswith("Dialogue:"):
            payload = line[len("Dialogue:"):].strip()
            parts = payload.split(",", len(format_fields) - 1) if format_fields else payload.split(",", 9)
            if not format_fields:
                # Fallback to typical positions
                # 0-Layer,1-Start,2-End,3-Style,4-Name,5-MarginL,6-MarginR,7-MarginV,8-Effect,9-Text
                if len(parts) < 10:
                    continue
                start_str, end_str, text = parts[1], parts[2], parts[9]
            else:
                try:
                    start_idx = format_fields.index("start")
                    end_idx = format_fields.index("end")
                    text_idx = format_fields.index("text")
                    start_str = parts[start_idx]
                    end_str = parts[end_idx]
                    text = parts[text_idx]
                except Exception:
                    continue
            start = _parse_time_ass(start_str)
            end = _parse_time_ass(end_str)
            text_clean = _strip_ass_tags(text)
            idx += 1
            cues.append(SubtitleCue(index=idx, start=start, end=end, text=text_clean))
    return cues


def _parse_time_ass(ts: str) -> float:
    # "H:MM:SS.cs" (centiseconds)
    # Example: 0:01:23.45
    try:
        hms, cs = ts.split(".")
        h, m, s = hms.split(":")
        total = int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100.0
        return total
    except Exception:
        return 0.0


def _strip_ass_tags(text: str) -> str:
    # Remove basic ASS style overrides like {\i1} etc.
    out = []
    skip = False
    depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
            skip = True
        elif ch == "}":
            depth = max(0, depth - 1)
            if depth == 0:
                skip = False
        elif not skip:
            out.append(ch)
    return "".join(out).replace("\\N", "\n")


def _parse_subtitle(path: Path, fmt: str) -> List[SubtitleCue]:
    if fmt == "srt":
        return _parse_srt(path)
    elif fmt == "vtt":
        return _parse_vtt(path)
    elif fmt == "ass":
        return _parse_ass(path)
    else:
        # Default try srt
        return _parse_srt(path)


def _estimate_time_offset(cues: List[SubtitleCue], transcript: List[Tuple[float, float, str]]) -> float:
    """Estimate global time offset between subtitles and transcript using text similarity.

    Strategy:
    - Build a sliding window over transcript texts and compare to early subtitle cues' texts.
    - For simplicity and speed, we use a very naive similarity: overlapping token count.
    - Choose offset that best aligns text windows; fallback to 0 if low confidence.

    Returns:
        offset in seconds (positive means shift subtitles forward).
    """
    def tokenize(s: str) -> List[str]:
        return [w.lower() for w in "".join(ch if ch.isalnum() else " " for ch in s).split() if w]

    # Use first N cues as reference
    N = min(20, len(cues))
    ref_cues = cues[:N]
    ref_tokens = [tokenize(c.text) for c in ref_cues]

    # Build transcript windows roughly around duration of cues; use simple window of 1 segment
    best_score = -1
    best_offset = 0.0

    for t_start, t_end, t_text in transcript:
        t_tokens = tokenize(t_text)
        if not t_tokens:
            continue

        # Score vs each reference cue
        for cue, cue_tokens in zip(ref_cues, ref_tokens):
            if not cue_tokens:
                continue
            overlap = len(set(cue_tokens) & set(t_tokens))
            if overlap <= 0:
                continue
            # Candidate offset to align cue.start with t_start
            cand_offset = t_start - cue.start
            # Weight by overlap; prefer small absolute offsets if tie
            score = overlap - 0.001 * abs(cand_offset)
            if score > best_score:
                best_score = score
                best_offset = cand_offset

    # Confidence threshold: require at least some overlap
    if best_score < 1:
        return 0.0
    # Clamp offset to reasonable range to avoid absurd shifts
    return float(max(-1200.0, min(1200.0, best_offset)))


def _apply_offset(cues: List[SubtitleCue], offset: float) -> List[SubtitleCue]:
    out: List[SubtitleCue] = []
    for cue in cues:
        start = max(0.0, cue.start + offset)
        end = max(start, cue.end + offset)
        out.append(SubtitleCue(index=cue.index, start=start, end=end, text=cue.text))
    return out

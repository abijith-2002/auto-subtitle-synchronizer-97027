"""
LLM integration utilities for correcting or upgrading subtitle cues using
Retrieval-Augmented-Generation (RAG) context from the full transcript.

This module isolates all interactions with the LLM provider and contains:
- Provider-agnostic prompt builder
- Environment-based client selection (OpenAI-compatible or stub)
- PUBLIC_INTERFACE helpers for per-cue and batch processing

Environment variables required (not read directly here; loaded via os.getenv):
- OPENAI_API_KEY: API key for OpenAI-compatible endpoints (if using OpenAI)
- OPENAI_API_BASE: Optional, override base URL for OpenAI-compatible servers
- OPENAI_MODEL: Optional, model name (default: 'gpt-4o-mini' or any compatible)
- OPENAI_REQUEST_TIMEOUT: Optional, request timeout in seconds (default 30)
- OPENAI_CONCURRENCY: Optional, max concurrent requests for batching (default 4)
Notes:
- Do NOT hardcode secrets in code. The orchestrator will populate .env.

The functions here are designed to be optional: if API key is not present,
a deterministic stub is used so pipelines can still function.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# --------------------------
# Exceptions and Data Models
# --------------------------
class LLMClientError(Exception):
    """Raised when LLM client fails or is misconfigured."""


@dataclass
class LLMPromptContext:
    """Holds the information used to build the LLM prompt for a cue."""
    cue_text: str
    cue_start: float
    cue_end: float
    topk_segments: List[Dict[str, Any]]  # each has start, end, text, score


# --------------------------
# Prompt builder
# --------------------------
def _format_time(seconds: float) -> str:
    """Format seconds into H:MM:SS.mmm for readability in prompts."""
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h}:{m:02}:{s:02}.{ms:03}"


def build_cue_prompt(ctx: LLMPromptContext) -> str:
    """
    Build an instruction-style prompt for the LLM given a subtitle cue context.

    The instruction asks the model to:
    - Preserve meaning based on transcript evidence
    - Fix minor imperfections / casing / punctuation
    - Keep it concise and suitable for subtitles
    - Return ONLY the corrected subtitle text (no quotes or extra text)

    The prompt contains:
    - Cue text and timestamps
    - Top-10 retrieved transcript segments with similarity scores
    """
    lines: List[str] = []
    lines.append("You are a helpful assistant that corrects subtitle cues using the most relevant transcript context.")
    lines.append("Goals:")
    lines.append("1) Fix wording, typos, casing, punctuation; keep speaker intent.")
    lines.append("2) Keep the subtitle concise and readable.")
    lines.append("3) Use the retrieved transcript segments as ground truth where applicable.")
    lines.append("4) Return ONLY the corrected subtitle line without extra commentary or quotes.")
    lines.append("")
    lines.append(f"Cue time: { _format_time(ctx.cue_start) } --> { _format_time(ctx.cue_end) }")
    lines.append(f"Original cue: {ctx.cue_text.strip()}")
    lines.append("")
    lines.append("Top relevant transcript segments:")
    for i, seg in enumerate(ctx.topk_segments[:10], start=1):
        sstart = _format_time(float(seg.get('start', 0.0)))
        send = _format_time(float(seg.get('end', 0.0)))
        score = float(seg.get("score", 0.0))
        text = str(seg.get("text", "")).strip()
        lines.append(f"{i}. [{sstart} - {send}] (score={score:.3f}) {text}")
    lines.append("")
    lines.append("Return only the corrected subtitle text:")
    return "\n".join(lines)


# --------------------------
# Client selection and calls
# --------------------------
class _StubLLMClient:
    """Deterministic stub client used when no API key is provided."""

    def complete(self, prompt: str, timeout: Optional[float] = None) -> str:
        # A naive 'correction': just strip and ensure it ends with punctuation.
        text = prompt.strip().split("Original cue:", 1)[-1].strip()
        if "\n" in text:
            text = text.splitlines()[0].strip()
        if not text:
            return ""
        terminal = text[-1]
        if terminal not in ".?!":
            text = text + "."
        return " ".join(text.split())


class _OpenAIClient:
    """OpenAI-compatible client using chat completions."""

    def __init__(self, api_key: str, base_url: Optional[str] = None, model_name: Optional[str] = None, timeout: Optional[float] = None) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise LLMClientError(
                "OpenAI client not installed. Please add 'openai' to requirements.txt."
            ) from exc
        self._model = model_name or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self._client = OpenAI(api_key=api_key, base_url=base_url or os.getenv("OPENAI_API_BASE") or None)
        self._timeout = timeout or float(os.getenv("OPENAI_REQUEST_TIMEOUT") or 30)

    def complete(self, prompt: str, timeout: Optional[float] = None) -> str:
        try:
            # Allow per-call timeout override (kept for API parity; client may not use it directly)
            _ = timeout or self._timeout  # no-op to avoid linter error if unused
            # The official client does not take timeout directly in create(); rely on global/httpx config if needed.
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You correct and improve subtitle text."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=96,
            )
            content = resp.choices[0].message.content or ""
            return content.strip().strip('"').strip()
        except Exception as exc:
            raise LLMClientError(f"OpenAI completion failed: {exc}") from exc


def _get_llm_client() -> Any:
    """
    Return a usable LLM client. If no API key is present, return a stub so the
    pipeline can still run deterministically.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return _OpenAIClient(
            api_key=api_key,
            base_url=os.getenv("OPENAI_API_BASE"),
            model_name=os.getenv("OPENAI_MODEL"),
            timeout=float(os.getenv("OPENAI_REQUEST_TIMEOUT") or 30),
        )
    return _StubLLMClient()


def _clean_llm_text(s: str) -> str:
    """
    Normalize LLM outputs by using a robust parser that extracts a single corrected line.
    Falls back to a minimal cleanup if parser import fails.
    """
    try:
        from .parser import extract_corrected_line_from_response  # local import
        cleaned = extract_corrected_line_from_response(s or "")
        return cleaned
    except Exception:
        # Minimal fallback
        s = (s or "").strip()
        if s.startswith("```") and s.endswith("```"):
            s = s.strip("`").strip()
        s = s.strip('"').strip("'").strip()
        return " ".join(s.split())


def _retry_with_backoff(callable_fn, max_retries: int = 2, base_delay: float = 0.5) -> Tuple[bool, str]:
    """
    Execute a callable that returns a string, retrying on failure with exponential backoff.
    Returns (ok, value_or_errorstr)
    """
    attempt = 0
    while True:
        try:
            value = callable_fn()
            return True, value
        except Exception as e:
            if attempt >= max_retries:
                return False, str(e)
            time.sleep(base_delay * (2 ** attempt))
            attempt += 1


# --------------------------
# PUBLIC interface helpers
# --------------------------

# PUBLIC_INTERFACE
def correct_cue_with_rag(cue_text: str, start: float, end: float, topk_segments: List[Dict[str, Any]]) -> str:
    """
    Correct a single subtitle cue using top-K retrieved transcript segments.

    Parameters:
        cue_text: Original subtitle text for the cue.
        start: Cue start time (seconds).
        end: Cue end time (seconds).
        topk_segments: List of dicts representing similar transcript segments with keys:
                       {start: float, end: float, text: str, score: float}

    Returns:
        Corrected subtitle string (best-effort). If LLM is unavailable, a stub correction is used.
    """
    ctx = LLMPromptContext(
        cue_text=str(cue_text or ""),
        cue_start=float(start or 0.0),
        cue_end=float(end or 0.0),
        topk_segments=topk_segments or [],
    )
    prompt = build_cue_prompt(ctx)
    client = _get_llm_client()
    ok, out = _retry_with_backoff(lambda: client.complete(prompt), max_retries=2, base_delay=0.4)
    if not ok:
        # Fallback to original cue if repeated failures
        return str(cue_text or "").strip()
    return _clean_llm_text(out)


# PUBLIC_INTERFACE
def batch_correct_cues_with_rag(
    cues: List[Dict[str, Any]],
    rag_results: List[List[Dict[str, Any]]],
) -> List[str]:
    """
    Batch-correct multiple subtitle cues.

    Parameters:
        cues: List of cues, each with {'text': str, 'start': float, 'end': float}
        rag_results: For each cue, a list of top transcript segments (same ordering as cues)

    Returns:
        List of corrected strings aligned with cues order.
    """
    outputs: List[str] = []
    client = _get_llm_client()
    # Concurrency control via simple windowing to avoid overwhelming API
    # Concurrency value reserved for future parallel implementation; currently processing sequentially for robustness.
    timeout = float(os.getenv("OPENAI_REQUEST_TIMEOUT") or 30)

    # Process in small batches sequentially (simple and robust).
    for i, cue in enumerate(cues):
        topk = rag_results[i] if i < len(rag_results) else []
        ctx = LLMPromptContext(
            cue_text=str(cue.get("text", "")),
            cue_start=float(cue.get("start", 0.0)),
            cue_end=float(cue.get("end", 0.0)),
            topk_segments=topk,
        )
        prompt = build_cue_prompt(ctx)

        def _do():
            return client.complete(prompt, timeout=timeout)

        ok, out = _retry_with_backoff(_do, max_retries=2, base_delay=0.4)
        if not ok:
            outputs.append(str(cue.get("text", "")))
        else:
            outputs.append(_clean_llm_text(out))
    return outputs

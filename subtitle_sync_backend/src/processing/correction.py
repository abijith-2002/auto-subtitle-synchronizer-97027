"""
Subtitle correction pipeline utilities that combine RAG retrieval with LLM calls.

This module provides PUBLIC_INTERFACE helpers that:
- Given subtitle entries and full transcript entries, compute top-k RAG results
- Call the LLM for each cue with its context
- Return a new list of corrected texts, preserving original timing
- Optionally write an additional corrected subtitle file alongside the synced one
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .processing import SubtitleEntry, rag_top_k_for_subtitles
from .llm import batch_correct_cues_with_rag


# PUBLIC_INTERFACE
def correct_subtitles_with_llm(
    subtitle_entries: List[SubtitleEntry],
    transcript_entries: List[Dict[str, Any]],
    fmt: str = "srt",
    k: int = 10,
) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
    """
    Correct each subtitle cue using an LLM with RAG context.

    Parameters:
        subtitle_entries: Parsed subtitle entries (with start/end/text)
        transcript_entries: Full transcript entries [{start,end,text}]
        fmt: Subtitle format ('srt' | 'vtt' | 'ass') - used by callers for later formatting
        k: Number of transcript segments to retrieve per cue

    Returns:
        (corrected_texts, rag_results)
        corrected_texts: list of LLM-improved text strings aligned with subtitle_entries
        rag_results: list of top-k results per cue used as context
    """
    # Prepare cues in dict form
    cues = [{"text": s.text, "start": s.start, "end": s.end} for s in subtitle_entries]
    rag_results = rag_top_k_for_subtitles(subtitle_entries, transcript_entries, k=k)
    corrected = batch_correct_cues_with_rag(cues, rag_results)
    return corrected, rag_results


# PUBLIC_INTERFACE
def apply_corrected_texts(
    subtitle_entries: List[SubtitleEntry],
    corrected_texts: List[str],
) -> List[SubtitleEntry]:
    """
    Create a new list of SubtitleEntry objects with corrected text but identical timing.

    Parameters:
        subtitle_entries: Original entries with timing.
        corrected_texts: Per-cue corrected text.

    Returns:
        New list of SubtitleEntry with corrected text.
    """
    out: List[SubtitleEntry] = []
    for i, e in enumerate(subtitle_entries):
        text = corrected_texts[i] if i < len(corrected_texts) else e.text
        out.append(type(e)(index=e.index, start=e.start, end=e.end, text=str(text or "").strip()))
    return out

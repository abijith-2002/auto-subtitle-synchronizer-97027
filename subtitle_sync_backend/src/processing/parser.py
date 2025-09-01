"""
Utilities to robustly parse corrected subtitle lines from LLM responses.

LLM outputs can vary:
- Plain text lines (ideal)
- Wrapped in quotes or code fences
- JSON or YAML-like structures
- Prefixed with labels like "Corrected:", "Fixed:", numbers, bullets
- Surrounded by extra commentary

This module provides helper functions to extract a single corrected subtitle line
safely for each cue.

Public interfaces:
- extract_corrected_line_from_response(response_text: str) -> str
"""
from __future__ import annotations

import re
from typing import Optional, List


_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\s*([\s\S]*?)\s*```$", re.MULTILINE)
_QUOTE_WRAP_RE = re.compile(r'^[\s\'"]*(.*?)[\s\'"]*$')
_JSON_TEXT_FIELD_RE = re.compile(
    r'"(?:text|subtitle|line|corrected|corrected_text)"\s*:\s*"([^"]+)"', re.IGNORECASE
)
_SINGLE_FIELD_JSON_RE = re.compile(
    r'^\s*\{\s*"(?:text|subtitle|line|corrected|corrected_text)"\s*:\s*"(.*?)"\s*\}\s*$',
    re.IGNORECASE | re.DOTALL,
)
_LABEL_PREFIXES = [
    r"corrected(?:\s*subtitle)?\s*[:\-]\s*",
    r"fixed\s*[:\-]\s*",
    r"rewrite\s*[:\-]\s*",
    r"answer\s*[:\-]\s*",
    r"output\s*[:\-]\s*",
    r"result\s*[:\-]\s*",
]


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    m = _CODE_FENCE_RE.match(s)
    if m:
        return m.group(1).strip()
    return s


def _strip_markers_and_quotes(s: str) -> str:
    s = s.strip()
    # Remove leading/trailing quotes and common markers
    s = s.strip('\'"')
    # Remove surrounding backticks (single-line)
    if s.startswith("`") and s.endswith("`"):
        s = s[1:-1].strip()
    # Collapse internal whitespace
    s = " ".join(s.split())
    return s


def _try_json_field(s: str) -> Optional[str]:
    # Quick path: object with single "text" or similar field
    m_obj = _SINGLE_FIELD_JSON_RE.match(s.strip())
    if m_obj:
        return m_obj.group(1).strip()
    # Fallback: search anywhere for a "text"/"corrected" field
    m = _JSON_TEXT_FIELD_RE.search(s)
    if m:
        return m.group(1).strip()
    return None


def _remove_common_prefix_labels(s: str) -> str:
    s_stripped = s.strip()
    # Remove enumeration or bullets
    s_stripped = re.sub(r"^\s*[\-\*\u2022]\s*", "", s_stripped)  # -, *, •
    s_stripped = re.sub(r"^\s*\(?\d+\)?[.)]\s*", "", s_stripped)  # 1) 1. (1)
    # Remove common "Corrected:"-style labels
    for lbl in _LABEL_PREFIXES:
        s_stripped = re.sub(r"^(?i)" + lbl, "", s_stripped, flags=re.IGNORECASE)
    return s_stripped.strip()


def _pick_best_line(lines: List[str]) -> str:
    """
    From multiple candidate lines, pick the most likely corrected subtitle:
    - Prefer the line with reasonable length (3..120 chars)
    - Avoid lines that look like instructions or metadata
    - If multiple candidates, choose the one with most letters
    """
    def score(line: str) -> int:
        letters = sum(c.isalpha() for c in line)
        return letters

    candidates = []
    for ln in lines:
        t = ln.strip()
        if not t:
            continue
        # Skip obviously meta lines
        if re.search(r"^\s*(return|answer|corrected|output|fixed)\b[:\-]", t, flags=re.IGNORECASE):
            continue
        if 3 <= len(t) <= 200:
            candidates.append(t)

    if not candidates:
        return (lines[0].strip() if lines else "").strip()

    candidates.sort(key=score, reverse=True)
    return candidates[0]


# PUBLIC_INTERFACE
def extract_corrected_line_from_response(response_text: str) -> str:
    """
    Extract a single corrected subtitle line from a raw LLM response.

    Heuristics applied in order:
    1) Strip code fences if present.
    2) Attempt to parse a JSON-like "text" field.
    3) If multi-line, consider the best line candidate by simple heuristics.
    4) Remove common labels/prefixes and surrounding quotes/markers.
    5) Collapse whitespace.

    Returns:
        A cleaned, single-line corrected subtitle string. May be empty if no content found.
    """
    s = (response_text or "").strip()
    if not s:
        return ""

    # 1) Code fences
    s = _strip_code_fences(s)

    # 2) JSON-like field
    json_field = _try_json_field(s)
    if json_field is not None:
        return _strip_markers_and_quotes(_remove_common_prefix_labels(json_field))

    # 3) Multi-line handling: choose best candidate line
    # First, attempt to remove instructional suffix/prefix noise by focusing on lines after empty separator
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""

    # If there is "Return only..." style instruction, focus on the last non-empty line
    instruction_idx = None
    for i, ln in enumerate(lines):
        if re.search(r"return\s+only", ln, re.IGNORECASE):
            instruction_idx = i
    if instruction_idx is not None:
        post = [ln for ln in lines[instruction_idx + 1 :] if ln]
        if post:
            lines = post

    # Attempt to strip labels from each line and choose
    candidates = [_remove_common_prefix_labels(ln) for ln in lines]
    chosen = _pick_best_line(candidates)

    # 4) Final cleanup of quotes/markers
    chosen = _strip_markers_and_quotes(chosen)
    return chosen

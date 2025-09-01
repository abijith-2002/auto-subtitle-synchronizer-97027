#!/usr/bin/env python3
"""
Example: Quick connectivity test for Google Gemini (google.genai SDK)

This script:
- Loads the GEMINI_API_KEY from the environment (do NOT hardcode secrets)
- Uses the official google.genai SDK to send a simple prompt ("Hello, Gemini!")
- Prints the LLM response or a clear error if credentials are missing/invalid

Usage (from repo root):
  python -m subtitle_sync_backend.examples.test_gemini_connection

Or run directly (ensure PYTHONPATH includes backend src if needed):
  PYTHONPATH=auto-subtitle-synchronizer-97027/subtitle_sync_backend \
    python auto-subtitle-synchronizer-97027/subtitle_sync_backend/examples/test_gemini_connection.py

Environment variables required:
- GEMINI_API_KEY: Your Gemini API key (request this from the user; set via .env in deployment)
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def _print_header() -> None:
    print("=== Google Gemini Connectivity Test (google.genai) ===")


def _check_sdk() -> Optional[str]:
    try:
        # Import within function to keep script lightweight if SDK missing.
        # Use __import__ to avoid binding unused names that trigger linters.
        __import__("google.genai")
        return None
    except Exception as exc:
        return (
            "The google-genai SDK is not installed or not importable.\n"
            "Install it (non-interactively in CI) with:\n"
            "  pip install google-genai\n"
            f"Details: {exc}"
        )


def _get_api_key(env_var: str = "GEMINI_API_KEY") -> Optional[str]:
    return os.getenv(env_var)


def _safe_one_liner(text: str) -> str:
    text = (text or "").strip()
    # Collapse to single line to avoid printing large multi-line generations
    return " ".join(text.split())


# PUBLIC_INTERFACE
def run_test(model: str = "gemini-1.5-pro") -> int:
    """
    PUBLIC_INTERFACE
    Run a minimal prompt against Gemini using google.genai and print the result.

    Parameters:
        model: The Gemini model to use (default: "gemini-1.5-pro").

    Returns:
        0 on success, non-zero on failure.
    """
    _print_header()

    sdk_issue = _check_sdk()
    if sdk_issue:
        print(f"Error: {sdk_issue}", file=sys.stderr)
        return 2

    try:
        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore
    except Exception as exc:
        print(f"Error: Failed to import google.genai after check: {exc}", file=sys.stderr)
        return 3

    api_key = _get_api_key("GEMINI_API_KEY")
    if not api_key:
        print(
            "Error: Missing GEMINI_API_KEY environment variable.\n"
            "- Request this secret from the user and set it via .env (do not hardcode).\n"
            "- Example (bash): export GEMINI_API_KEY='your_key_here'\n",
            file=sys.stderr,
        )
        return 1

    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        print(f"Error: Failed to create google.genai Client: {exc}", file=sys.stderr)
        return 4

    try:
        # Simple "Hello, Gemini!" prompt
        result = client.models.generate_content(
            model=model,
            contents=[genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text("Hello, Gemini!")]
            )],
        )
    except Exception as exc:
        print(
            "Error: Gemini request failed. This might indicate invalid credentials or network issues.\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 5

    # Extract the text response robustly
    text_response = ""
    try:
        for cand in getattr(result, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                if hasattr(part, "text") and part.text:
                    text_response += part.text + "\n"
    except Exception:
        pass

    text_response = _safe_one_liner(text_response)
    if not text_response:
        print("Warning: No text returned by Gemini; check model availability and key permissions.")
    else:
        print("Gemini responded:")
        print(f"  {text_response}")

    print("\nConnectivity test completed.")
    return 0


if __name__ == "__main__":
    sys.exit(run_test())

# Processing package initializer.
# Expose key interfaces for easier imports if needed.

from .processing import (
    BaseTranscriber,
    StubTranscriber,
    WhisperTranscriber,
    TranscriptSegment,
    process_and_sync_subtitles,
    create_workdir,
    save_upload,
    SubtitleSyncError,
    rag_top_k_for_subtitles,
)
from .rag import retrieve_top_k_for_cue, retrieve_top_k_for_cues

__all__ = [
    "BaseTranscriber",
    "StubTranscriber",
    "WhisperTranscriber",
    "TranscriptSegment",
    "process_and_sync_subtitles",
    "create_workdir",
    "save_upload",
    "SubtitleSyncError",
    "rag_top_k_for_subtitles",
    "retrieve_top_k_for_cue",
    "retrieve_top_k_for_cues",
]

from __future__ import annotations

import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

from src.processing.processing import (
    create_workdir,
    save_upload,
    process_and_sync_subtitles,
    SubtitleSyncError,
    SUPPORTED_SUB_EXTENSIONS,
)

router = APIRouter(prefix="/api", tags=["Subtitle Sync"])

class SyncResponse(BaseModel):
    synced_subtitle_path: str = Field(..., description="Absolute path to the synced subtitle file.")
    format: str = Field(..., description="Subtitle format: srt | vtt | ass")
    shift_seconds: str = Field(..., description="Applied shift to align subtitles with transcription.")

class SyncJobResult(BaseModel):
    job_id: str = Field(..., description="Job identifier for this processing run.")
    result: SyncResponse = Field(..., description="Processing result details.")

# PUBLIC_INTERFACE
@router.post(
    "/sync",
    response_model=SyncJobResult,
    summary="Sync subtitle with video audio using Whisper transcription",
    description="Uploads a video file and a subtitle file, performs transcription and adjusts subtitle timestamps to match the audio.",
)
async def sync_subtitles(
    video: UploadFile = File(..., description="Video file containing the audio."),
    subtitle: UploadFile = File(..., description="Subtitle file (.srt, .vtt, .ass, .ssa)."),
    prefer_whisper: bool = Form(False, description="Use Whisper if available, otherwise stub."),
    whisper_model: str = Form("base", description="Whisper model name if prefer_whisper is true."),
) -> SyncJobResult:
    """
    PUBLIC_INTERFACE
    Sync the provided subtitle file to the audio track of the given video.

    Parameters:
    - video: The uploaded video file.
    - subtitle: The uploaded subtitle file (srt, vtt, ass, or ssa).
    - prefer_whisper: If true, attempts to use Whisper; falls back to stub on failure.
    - whisper_model: The Whisper model name (e.g., 'base', 'small', etc.).

    Returns:
    - SyncJobResult containing job_id and details including path to the synced subtitle.
    """
    # Validate subtitle extension
    sub_name = subtitle.filename or ""
    if not any(sub_name.lower().endswith(ext) for ext in SUPPORTED_SUB_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Unsupported subtitle format. Use .srt, .vtt, .ass, or .ssa")

    workdir = create_workdir()
    try:
        video_bytes = await video.read()
        subtitle_bytes = await subtitle.read()
        video_path = save_upload(video_bytes, video.filename or "video.mp4", workdir)
        subtitle_path = save_upload(subtitle_bytes, subtitle.filename or "subtitle.srt", workdir)

        result_dict = process_and_sync_subtitles(
            video_path=video_path,
            subtitle_path=subtitle_path,
            output_dir=workdir,
            prefer_whisper=prefer_whisper,
            whisper_model=whisper_model,
        )
        job_id = os.path.basename(workdir)
        return SyncJobResult(job_id=job_id, result=SyncResponse(**result_dict))
    except SubtitleSyncError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}") from e

class DownloadResponse(BaseModel):
    job_id: str = Field(..., description="Job identifier.")
    path: str = Field(..., description="Absolute path to the generated synced subtitle.")
    exists: bool = Field(..., description="Whether the file exists and is ready for download.")

# PUBLIC_INTERFACE
@router.get(
    "/download-info/{job_id}",
    response_model=DownloadResponse,
    summary="Get info for a synced subtitle by job ID",
    description="Returns metadata to locate the synced subtitle. The actual file can be served via a static server or file response.",
)
async def get_download_info(job_id: str) -> DownloadResponse:
    """
    PUBLIC_INTERFACE
    Retrieve the path and existence of the synced subtitle file corresponding to a job.

    Parameters:
    - job_id: The working directory name returned from the sync call.

    Returns:
    - DownloadResponse containing path and a boolean flag indicating existence.
    """
    candidates = []

    # Scan likely directories
    for base in ["/tmp", os.getenv("TMPDIR") or "/tmp"]:
        cand_dir = os.path.join(base, job_id)
        if os.path.isdir(cand_dir):
            for fn in os.listdir(cand_dir):
                if fn.lower().endswith((".srt", ".vtt", ".ass", ".ssa")) and "-synced" in fn.lower():
                    candidates.append(os.path.join(cand_dir, fn))

    path = candidates[0] if candidates else ""
    return DownloadResponse(job_id=job_id, path=path, exists=os.path.isfile(path))

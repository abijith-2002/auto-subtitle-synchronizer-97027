from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.processing import (
    create_task,
    get_progress,
    get_output_file_path,
    get_history,
    start_sync,
)

openapi_tags = [
    {"name": "health", "description": "Service health and info"},
    {"name": "files", "description": "File upload and download endpoints"},
    {"name": "processing", "description": "Trigger and track subtitle synchronization"},
    {"name": "history", "description": "History of processed items"},
]

app = FastAPI(
    title="Subtitle Sync Backend",
    description="API for uploading videos and subtitle files, transcribing audio with Whisper, and aligning subtitle timestamps.",
    version="1.0.0",
    openapi_tags=openapi_tags,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting in production via ENV
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# PUBLIC_INTERFACE
@app.get("/", tags=["health"], summary="Health Check")
def health_check():
    """Health Check endpoint.

    Returns:
        JSON object with service status.
    """
    return {"message": "Healthy"}


class UploadResponse(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the created task")


# PUBLIC_INTERFACE
@app.post(
    "/upload",
    tags=["files"],
    summary="Upload video and subtitle files",
    description="Uploads a video file and a subtitle file, registers a processing task, and returns task_id.",
    response_model=UploadResponse,
)
async def upload_files(
    video: UploadFile = File(..., description="Video file containing audio to transcribe"),
    subtitle: UploadFile = File(..., description="Subtitle file (.srt, .vtt, .ass/.ssa) to be synced"),
    model: Optional[str] = Form("base", description="Whisper model to use (e.g., tiny, base, small, medium)"),
):
    try:
        video_bytes = await video.read()
        subtitle_bytes = await subtitle.read()
        task_id = create_task(video_bytes, video.filename or "video", subtitle_bytes, subtitle.filename or "subtitle", model=model or "base")
        return UploadResponse(task_id=task_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {e}")


class SyncResponse(BaseModel):
    task_id: str = Field(..., description="Task being processed")
    status: str = Field(..., description="Current status immediately after trigger")


# PUBLIC_INTERFACE
@app.post(
    "/sync",
    tags=["processing"],
    summary="Initiate subtitle synchronization",
    description="Begins processing the uploaded files for the given task_id. Returns immediately while background processing continues.",
    response_model=SyncResponse,
)
async def trigger_sync(task_id: str = Form(..., description="Task ID returned from /upload")):
    prog = get_progress(task_id)
    if not prog:
        raise HTTPException(status_code=404, detail="Task not found")
    # Start the processing asynchronously
    # Note: We don't await; we schedule the coroutine in the event loop.
    try:
        import asyncio

        asyncio.create_task(start_sync(task_id))
    except RuntimeError:
        # In case of different loop context, fallback to await (not ideal)
        await start_sync(task_id)
    return SyncResponse(task_id=task_id, status="started")


class ProgressResponse(BaseModel):
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="queued|processing|completed|error")
    progress: float = Field(..., description="Progress ratio 0..1")
    message: str = Field(..., description="Human-readable status message")
    video_filename: Optional[str] = Field(None, description="Original video filename")
    subtitle_filename: Optional[str] = Field(None, description="Original subtitle filename")
    output_filename: Optional[str] = Field(None, description="Output subtitle filename when done")
    model: Optional[str] = Field(None, description="Whisper model used")


# PUBLIC_INTERFACE
@app.get(
    "/progress/{task_id}",
    tags=["processing"],
    summary="Get task progress",
    description="Returns progress information for a given task.",
    response_model=ProgressResponse,
)
async def progress(task_id: str):
    prog = get_progress(task_id)
    if not prog:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(content=prog)


# PUBLIC_INTERFACE
@app.get(
    "/download/{task_id}",
    tags=["files"],
    summary="Download synced subtitle",
    description="Downloads the synced subtitle file for the completed task.",
)
async def download(task_id: str):
    path = get_output_file_path(task_id)
    if not path:
        prog = get_progress(task_id)
        if not prog:
            raise HTTPException(status_code=404, detail="Task not found")
        if prog.get("status") != "completed":
            raise HTTPException(status_code=400, detail=f"Task not completed. Current status: {prog.get('status')}")
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(
        path=path,
        media_type="text/plain",
        filename=path.name,
    )


class HistoryItem(BaseModel):
    task_id: str = Field(..., description="Task ID")
    video_filename: Optional[str] = Field(None, description="Original video filename")
    subtitle_filename: Optional[str] = Field(None, description="Original subtitle filename")
    output_filename: Optional[str] = Field(None, description="Generated output filename")
    model: Optional[str] = Field(None, description="Whisper model used")
    offset_seconds: Optional[float] = Field(None, description="Estimated global offset applied")
    created_at: Optional[float] = Field(None, description="Creation timestamp")
    completed_at: Optional[float] = Field(None, description="Completion timestamp")
    timestamp: Optional[str] = Field(None, description="Completion ISO timestamp")


# PUBLIC_INTERFACE
@app.get(
    "/history",
    tags=["history"],
    summary="Get processing history",
    description="Returns recent processing history entries.",
    response_model=list[HistoryItem],
)
async def history():
    items = get_history(limit=100)
    return items

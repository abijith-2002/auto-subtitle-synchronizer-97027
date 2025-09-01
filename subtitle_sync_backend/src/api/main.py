from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router as api_router

openapi_tags = [
    {"name": "Health", "description": "Service health and metadata."},
    {"name": "Subtitle Sync", "description": "Upload video and subtitles, process and download synced subtitles."},
]

# Initialize FastAPI with metadata
app = FastAPI(
    title="Subtitle Sync Backend",
    description="Transcribe audio with Whisper and align subtitle timestamps automatically.",
    version="0.1.0",
    openapi_tags=openapi_tags,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root health check
@app.get("/", tags=["Health"], summary="Health Check")
def health_check():
    """PUBLIC_INTERFACE
    Health check endpoint to verify service availability.

    Returns:
        JSON object with message and status.
    """
    return {"message": "Healthy", "status": "ok"}

# API router
app.include_router(api_router)

# API docs help for (future) websocket usage info
@app.get(
    "/docs/websocket",
    tags=["Health"],
    summary="WebSocket usage help",
    description="This project currently does not expose websockets. If added in the future, usage details will appear here.",
)
def websocket_help():
    """PUBLIC_INTERFACE
    Provide general information about potential WebSocket usage.
    """
    return JSONResponse({"websocket": False, "note": "No WebSocket endpoints are currently available."})

# Project Repository

Subtitle Sync Backend

How to run:
- Install dependencies: pip install -r subtitle_sync_backend/requirements.txt
- Start server: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 3001 (from subtitle_sync_backend directory)

Key endpoints:
- GET / : Health check
- POST /upload : multipart form with fields 'video' (file), 'subtitle' (file), 'model' (optional)
- POST /sync : form field 'task_id'
- GET /progress/{task_id} : get processing status
- GET /download/{task_id} : download synced subtitle when complete
- GET /history : recent processing history

Design:
- All processing logic (transcription, parsing, syncing, file I/O) is in src/processing.py
- All FastAPI endpoints are in src/api/main.py and import functions from processing.py
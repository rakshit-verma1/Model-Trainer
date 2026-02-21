"""
views.py — Endpoints:
  GET  /             → SPA
  POST /upload/      → stream ZIP to disk, extract, return upload_id
  POST /train/       → clean previous session files, start training, return session_id
  GET  /stream/<s>/  → SSE progress stream
  GET  /download/<s>/→ serve model_<s>.pt, then delete
  GET  /metrics/<s>/ → serve metrics_<s>.csv, then delete
"""
import json
import shutil
import uuid
import zipfile
from pathlib import Path

from django.conf import settings
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .training import clear_session, get_queue, start_training


def index(request):
    return render(request, "index.html")


# ── Upload ─────────────────────────────────────────────────────────────────────
@csrf_exempt
def upload_dataset(request):
    if request.method != "POST":
        return HttpResponse(status=405)

    f = request.FILES.get("file")
    if not f:
        return JsonResponse({"error": "No file field in request"}, status=400)

    upload_id = str(uuid.uuid4())
    upload_dir = Path(settings.MEDIA_ROOT) / "uploads" / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    zip_path = upload_dir / "dataset.zip"
    with open(zip_path, "wb") as out:
        for chunk in f.chunks(chunk_size=8 * 1024 * 1024):
            out.write(chunk)

    try:
        with zipfile.ZipFile(zip_path) as z:
            safe = [m for m in z.namelist() if not m.startswith(("/", "..")) and ".." not in m]
            z.extractall(upload_dir, members=safe)
        zip_path.unlink()
    except zipfile.BadZipFile:
        shutil.rmtree(upload_dir, ignore_errors=True)
        return JsonResponse({"error": "Invalid or corrupt ZIP file"}, status=400)

    return JsonResponse({"upload_id": upload_id})


def _clean_previous_sessions():
    """
    Delete all model .pt and metrics .csv files from previous training runs.
    Called at the start of every new training so only the latest session's
    files ever exist on disk.
    """
    for subdir in ("models", "metrics"):
        d = Path(settings.MEDIA_ROOT) / subdir
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


# ── Train ──────────────────────────────────────────────────────────────────────
@csrf_exempt
def start_train(request):
    if request.method != "POST":
        return HttpResponse(status=405)

    try:
        config = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Wipe any files left from the previous session before starting a new one
    _clean_previous_sessions()

    session_id = str(uuid.uuid4())
    start_training(session_id, config)
    return JsonResponse({"session_id": session_id})


# ── SSE stream ─────────────────────────────────────────────────────────────────
def stream_progress(request, session_id: str):
    q = get_queue(session_id)

    def event_generator():
        while True:
            try:
                item = q.get(timeout=30)
                event = item["event"]
                data  = json.dumps(item["data"])
                yield f"event: {event}\ndata: {data}\n\n"
                if event in ("done", "error"):
                    clear_session(session_id)
                    break
            except Exception:
                yield ": keep-alive\n\n"

    response = StreamingHttpResponse(event_generator(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


# ── Downloads ──────────────────────────────────────────────────────────────────
def _serve_and_delete(path: Path, filename: str, content_type: str):
    """Read a file into memory, delete it, return it as an attachment."""
    if not path.exists():
        return HttpResponse("File not found or already downloaded.", status=404)
    with open(path, "rb") as f:
        data = f.read()
    path.unlink(missing_ok=True)
    response = HttpResponse(data, content_type=content_type)
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


def download_model(request, session_id: str):
    path = Path(settings.MEDIA_ROOT) / "models" / f"{session_id}.pt"
    return _serve_and_delete(path, f"model_{session_id[:8]}.pt", "application/octet-stream")


def download_metrics(request, session_id: str):
    path = Path(settings.MEDIA_ROOT) / "metrics" / f"{session_id}.csv"
    return _serve_and_delete(path, f"metrics_{session_id[:8]}.csv", "text/csv")

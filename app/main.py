from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from app.pest_assistant import PestAssistant

app = FastAPI(title="Organic Farm Pest Assistant API")
assistant = PestAssistant()


@app.get("/")
def read_root() -> dict[str, object]:
    """Liveness plus an honest description of the model actually loaded."""
    return {
        "message": "Organic Farm Pest Assistant is running",
        "model_loaded": assistant.model is not None,
        "model_path": str(assistant.loaded.path) if assistant.loaded.path else None,
        "num_classes": len(assistant.class_names),
        "under_trained": assistant.loaded.under_trained,
        "status": assistant.status_message,
    }


@app.post("/analyze")
def analyze_image(file: UploadFile = File(...)) -> JSONResponse:
    """Identify the pest in an uploaded photo. Returns the top-3 candidates."""
    suffix = Path(file.filename or "").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(file.file.read())
        temp_path = handle.name
    try:
        result = assistant.analyze_image(temp_path)
    finally:
        # The upload is only needed for the duration of the prediction, and a
        # long-running server would otherwise leak one file per request.
        Path(temp_path).unlink(missing_ok=True)
    return JSONResponse(content=result)

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from app.pest_assistant import PestAssistant

app = FastAPI(title="Organic Farm Pest Assistant API")
assistant = PestAssistant()


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Organic Farm Pest Assistant is running"}


@app.post("/analyze")
def analyze_image(file: UploadFile = File(...)) -> JSONResponse:
    suffix = Path(file.filename or "").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(file.file.read())
        temp_path = handle.name
    result = assistant.analyze_image(temp_path)
    return JSONResponse(content=result)

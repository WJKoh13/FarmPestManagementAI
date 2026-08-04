"""The HTTP face of the assistant.

The Streamlit app does not call this -- it holds a `PestAssistant` in process,
which keeps the offline demo to one command. This exists for anything that is
not Streamlit, and carries the same three capabilities: identify a photo, hold a
conversation with history, and say honestly what it is running.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.conversation import Conversation, Message, PestContext
from app.pest_assistant import PestAssistant

app = FastAPI(title="Organic Farm Pest Assistant API")
assistant = PestAssistant()


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    # Prior turns, oldest first. The client owns the history; the server holds
    # no session, so two clients can never be served each other's context.
    history: list[ChatTurn] = []
    # The pest a previous /analyze identified, so follow-ups resolve "it".
    pest_name: str | None = None


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


@app.get("/health")
def health() -> dict[str, object]:
    """Both halves of the app: the classifier and the local language model."""
    llm_ready = assistant.llm.available()
    return {
        "ok": assistant.model is not None,
        "classifier": {
            "loaded": assistant.model is not None,
            "path": str(assistant.loaded.path) if assistant.loaded.path else None,
            "under_trained": assistant.loaded.under_trained,
            "reason": assistant.loaded.reason,
        },
        "llm": {
            "ready": llm_ready,
            "model": assistant.llm.model,
            "base_url": assistant.llm.base_url,
            # Not an error: without it, replies come from the written guides.
            "note": "" if llm_ready else "Ollama unreachable — serving the offline guides.",
        },
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


@app.post("/chat")
def chat(request: ChatRequest) -> JSONResponse:
    """A text turn with history, grounded in the organic treatment guides."""
    conversation = Conversation(
        messages=[Message(role=turn.role, content=turn.content) for turn in request.history]
    )
    if request.pest_name:
        conversation.pest = PestContext(
            slug=request.pest_name,
            display_name=assistant.display_name_for(request.pest_name),
            confidence=0.0,
        )
    result = assistant.chat_reply(request.message, conversation=conversation)
    return JSONResponse(content=result)

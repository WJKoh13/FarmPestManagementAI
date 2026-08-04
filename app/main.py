"""The backend. Everything that thinks lives behind here.

The three components the app is built from are a front end (`streamlit_app.py`),
this backend, and the AI engine it owns -- whichever pest CNN in `runs/` scores
highest (ProPestNet, custom_cnn, VGG19 -- see `docs/`), the local language
model, and the reference library. The front end holds none of them: it sends a
message to `/agent` and renders what comes back, so it imports no torch, loads no
weights and makes no decisions.

The server keeps **no session**. Every request carries its own history, so two
browsers can never be served each other's conversation, and restarting the
backend loses nothing that was not already saved to disk by the front end.

Run it with::

    uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.agent import PestAgent
from app.cnn_model import describe_runs
from app.conversation import Conversation, Message, PestContext, store_image
from app.pest_assistant import PestAssistant

app = FastAPI(title="Organic Farm Pest Assistant API")

# One assistant and one agent for the process. Loading the checkpoint takes
# seconds, so it happens at import and is reused; `/models/select` is the only
# thing that replaces them.
assistant = PestAssistant()
agent = PestAgent(assistant=assistant)


class ChatTurn(BaseModel):
    role: str
    content: str
    # Lets the backend rebuild the set of photos this conversation may look at,
    # without keeping any state of its own.
    image_path: str | None = None


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []
    pest_name: str | None = None


class AgentRequest(BaseModel):
    """One turn. The client owns the history; the server owns the thinking."""

    message: str = ""
    image_path: str | None = None
    history: list[ChatTurn] = []
    # The pest a previous turn identified, so "how often do I spray it?" resolves.
    pest_name: str | None = None
    pest_uncertain: bool = False


class ModelChoice(BaseModel):
    path: str


def _rebuild(request: AgentRequest | ChatRequest) -> Conversation:
    """Turn a request's history back into a Conversation the agent understands."""
    conversation = Conversation(messages=[
        Message(role=turn.role, content=turn.content, image_path=turn.image_path)
        for turn in request.history
    ])
    if request.pest_name:
        conversation.pest = PestContext(
            slug=request.pest_name,
            display_name=assistant.display_name_for(request.pest_name),
            confidence=0.0,
            uncertain=getattr(request, "pest_uncertain", False),
        )
    return conversation


# --------------------------------------------------------------------- status
@app.get("/")
def read_root() -> dict[str, object]:
    """Liveness plus an honest description of the model actually loaded."""
    return {
        "message": "Organic Farm Pest Assistant is running",
        "model_loaded": assistant.model is not None,
        # Which model, in the form a person uses -- `custom_cnn (Zi Yang)`. The
        # path answers where, not whose, and several people's runs live here.
        "model": assistant.loaded.display_name if assistant.model else None,
        "model_path": str(assistant.loaded.path) if assistant.loaded.path else None,
        "num_classes": len(assistant.class_names),
        "under_trained": assistant.loaded.under_trained,
        "status": assistant.status_message,
    }


@app.get("/health")
def health() -> dict[str, object]:
    """Every part the front end needs to describe, in one call."""
    llm_ready = assistant.llm.available()
    knowledge = agent.knowledge
    return {
        "ok": assistant.model is not None,
        "classifier": {
            "loaded": assistant.model is not None,
            "model": assistant.loaded.display_name if assistant.model else None,
            "author": assistant.loaded.author or None,
            "path": str(assistant.loaded.path) if assistant.loaded.path else None,
            "under_trained": assistant.loaded.under_trained,
            "reason": assistant.loaded.reason,
            "status": assistant.status_message,
        },
        "llm": {
            "ready": llm_ready,
            "model": assistant.llm.model,
            "resolved_model": assistant.llm.resolve_model(),
            "status_line": assistant.llm.status_line,
            "base_url": assistant.llm.base_url,
            # Not an error: without it, replies come from the written guides.
            "note": "" if llm_ready else "Ollama unreachable — serving the offline guides.",
        },
        "knowledge": {
            "passages": len(knowledge),
            "embedded": knowledge.embedded,
        },
        # The front end needs these to label the confidence bars, and it no
        # longer has an assistant of its own to ask.
        "display_names": assistant.display_names,
    }


@app.get("/models")
def models() -> dict[str, object]:
    """Every checkpoint in runs/, for the picker."""
    described = describe_runs(num_classes=len(assistant.class_names))
    return {
        "current": str(assistant.loaded.path) if assistant.loaded.path else None,
        "runs": [{**run, "path": str(run["path"])} for run in described],
    }


@app.post("/models/select")
def select_model(choice: ModelChoice) -> dict[str, object]:
    """Swap the served checkpoint.

    This mutates process-wide state, which is acceptable here and nowhere near
    general: the app is one farmer on one laptop, and the alternative -- a model
    per session -- would mean several hundred megabytes per browser tab.
    """
    global assistant, agent
    assistant = PestAssistant(model_path=choice.path)
    agent = PestAgent(assistant=assistant)
    return {"ok": assistant.model is not None,
            "model": assistant.loaded.display_name if assistant.model else None,
            "path": str(assistant.loaded.path) if assistant.loaded.path else None,
            "reason": assistant.loaded.reason}


# ---------------------------------------------------------------------- files
@app.post("/images")
def upload_image(file: UploadFile = File(...)) -> dict[str, str]:
    """Store an upload and return where it went.

    The store belongs to the backend because the tools read from it and the
    allowlist is defined in terms of it. Front end and backend share a
    filesystem by design -- this is a single-machine offline app -- and that is
    the one assumption to revisit if the backend ever moves to another host.
    """
    data = file.file.read()
    path = store_image(data, file.filename or "upload.jpg")
    return {"image_path": str(path)}


# ------------------------------------------------------------------ the agent
@app.post("/agent")
def agent_turn(request: AgentRequest) -> JSONResponse:
    """One turn, with the language model choosing which tools to run.

    Returns everything the front end needs to draw the turn: what the tools did,
    what the classifier saw, and the finished answer. Not streamed -- a single
    request and a single response is enough here, and it keeps the front end to
    one function call.
    """
    conversation = _rebuild(request)
    text = request.message or ("Please identify this pest." if request.image_path else "")

    turn = None
    if assistant.llm.available():
        turn = agent.plan(text, image_path=request.image_path, conversation=conversation)
        if not turn.ok:
            turn = None

    if turn is not None:
        view = turn.view or {}
        body = "".join(assistant.llm.stream_chat(turn.messages)).strip()
        if body:
            pest = conversation.pest
            return JSONResponse(content={
                "answer": body,
                "heading": view.get("heading_plain", ""),
                "note": view.get("note", ""),
                "candidates": [[slug, float(score)] for slug, score in turn.candidates],
                "trace": turn.trace,
                "pest_name": pest.slug if pest else None,
                "confidence": pest.confidence if pest else None,
                "uncertain": bool(pest.uncertain) if pest else False,
                "fallback": False,
            })

    # Every failure lands here: no Ollama, a transport error, or a model that
    # produced nothing. The deterministic path still answers in full.
    deterministic = assistant.prepare_turn(text, image_path=request.image_path,
                                           conversation=conversation)
    pest = conversation.pest
    return JSONResponse(content={
        "answer": deterministic["fallback_body"],
        "heading": deterministic["heading_plain"],
        "note": deterministic["note"],
        "candidates": [[slug, float(score)] for slug, score in deterministic["candidates"]],
        "trace": [],
        "pest_name": pest.slug if pest else None,
        "confidence": pest.confidence if pest else None,
        "uncertain": bool(pest.uncertain) if pest else False,
        "fallback": True,
    })


# ------------------------------------------------------------ older endpoints
@app.post("/analyze")
def analyze_image(file: UploadFile = File(...)) -> JSONResponse:
    """Identify the pest in an uploaded photo. The classifier alone, no agent."""
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
    """A text turn without the agent, kept for anything already calling it."""
    result = assistant.chat_reply(request.message, conversation=_rebuild(request))
    return JSONResponse(content=result)

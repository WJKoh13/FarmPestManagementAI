"""How the front end talks to the backend.

Every function here degrades rather than raises. A farmer -- or a projector --
must never be shown a traceback because a server was slow to start, so an
unreachable backend produces an empty result and a sentence in the sidebar.

`requests` is already a dependency (it is how the app reaches Ollama), so this
adds nothing to install.
"""

from __future__ import annotations

import os
from typing import Any

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover
    requests = None

API_BASE = os.getenv("PEST_API_URL", "http://127.0.0.1:8000").rstrip("/")

# The agent makes two calls to a local language model, and a slow first token on
# a cold model is normal rather than a hang.
TURN_TIMEOUT = float(os.getenv("PEST_API_TIMEOUT", "300"))
QUICK_TIMEOUT = 5.0


def _get(path: str, timeout: float = QUICK_TIMEOUT) -> dict[str, Any]:
    if requests is None:
        return {}
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception:  # noqa: BLE001 - "the backend is not there" is not an error here
        return {}


def health() -> dict[str, Any]:
    """Everything about the backend's state, or {} when it is not running."""
    return _get("/health")


def models() -> dict[str, Any]:
    return _get("/models")


def select_model(path: str) -> bool:
    if requests is None:
        return False
    try:
        response = requests.post(f"{API_BASE}/models/select", json={"path": path},
                                 timeout=120)
        response.raise_for_status()
        return bool(response.json().get("ok"))
    except Exception:  # noqa: BLE001
        return False


def upload_image(data: bytes, filename: str) -> str | None:
    """Send a photo to the backend's store and get back where it landed."""
    if requests is None:
        return None
    try:
        response = requests.post(f"{API_BASE}/images",
                                 files={"file": (filename, data)}, timeout=60)
        response.raise_for_status()
        return response.json().get("image_path")
    except Exception:  # noqa: BLE001
        return None


def agent_turn(message: str, image_path: str | None, history: list[dict[str, Any]],
               pest_name: str | None, pest_uncertain: bool = False) -> dict[str, Any]:
    """One conversational turn.

    On failure returns ``{"error": ...}`` rather than raising, so the caller has
    exactly one shape to render and the front end never has a try/except in the
    middle of drawing a chat message.
    """
    if requests is None:
        return {"error": "The requests library is not installed."}
    try:
        response = requests.post(
            f"{API_BASE}/agent",
            json={"message": message, "image_path": image_path, "history": history,
                  "pest_name": pest_name, "pest_uncertain": pest_uncertain},
            timeout=TURN_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except Exception as error:  # noqa: BLE001
        return {"error": f"Could not reach the backend at {API_BASE} ({type(error).__name__})."}

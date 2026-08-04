"""A thin client for a local Ollama server.

The app is offline-first: Ollama is an enhancement that rephrases the vetted
guidance in `treatment_guides.py` around what the farmer actually asked. When it
is not running, every caller must still produce a useful answer, so failure is
reported as ``LLMReply.ok is False`` rather than raised.

Start it with::

    brew install ollama && ollama serve
    ollama pull phi3        # or llama3.2:3b / qwen2.5:3b, then set OLLAMA_MODEL
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Iterator

try:
    import requests
except ModuleNotFoundError:  # The CNN demo can run without a local LLM client.
    requests = None

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "phi3"

# A small model generating a few hundred tokens on CPU is a slow first token,
# not a hung server. The old 10s ceiling made every real reply look like an
# outage. Connection failures still fail fast -- that is the connect timeout.
CONNECT_TIMEOUT = 3.0
READ_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

# How long an `available()` result is trusted. Streamlit reruns the whole script
# on every interaction, so an uncached check would probe the server several
# times per keystroke.
HEALTH_TTL_SECONDS = 15.0

UNAVAILABLE_MESSAGE = (
    "I'm running in offline fallback mode. Upload a pest photo for a quick identification "
    "and treatment suggestion."
)
NO_CLIENT_MESSAGE = "Local language model is unavailable."


@dataclass
class LLMReply:
    """A reply plus whether it actually came from the language model.

    Callers branch on ``ok``, never on the text. The previous code sniffed for
    substrings of the fallback message, which silently broke the moment that
    wording changed.
    """

    text: str
    ok: bool = True

    def __bool__(self) -> bool:
        return self.ok


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 timeout: float | None = None) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        # `model` defaulted to "phi3" here, so `model or os.getenv(...)` could
        # never reach the environment variable. It is None-by-default now.
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self.timeout = (CONNECT_TIMEOUT, timeout or READ_TIMEOUT)
        self._health: tuple[float, bool] | None = None

    # ------------------------------------------------------------------ health
    def available(self, force: bool = False) -> bool:
        """Whether the server answers and has the configured model pulled."""
        if requests is None:
            return False
        now = time.monotonic()
        if not force and self._health and now - self._health[0] < HEALTH_TTL_SECONDS:
            return self._health[1]

        ok = False
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=CONNECT_TIMEOUT)
            response.raise_for_status()
            names = [model.get("name", "") for model in response.json().get("models", [])]
            # Ollama reports "phi3:latest" for a model pulled as "phi3".
            ok = any(name == self.model or name.split(":")[0] == self.model.split(":")[0]
                     for name in names)
        except Exception:  # noqa: BLE001 - any failure means "not usable", not a crash
            ok = False

        self._health = (now, ok)
        return ok

    @property
    def status_line(self) -> str:
        """One short line for the sidebar."""
        if requests is None:
            return "requests not installed"
        if self.available():
            return f"{self.model} · ready"
        return f"{self.model} · not running"

    # -------------------------------------------------------------------- chat
    def chat(self, messages: list[dict[str, str]]) -> LLMReply:
        """A multi-turn completion. ``messages`` is Ollama's role/content list."""
        if requests is None:
            return LLMReply(NO_CLIENT_MESSAGE, ok=False)
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
            text = (response.json().get("message") or {}).get("content", "").strip()
            return LLMReply(text, ok=True) if text else LLMReply(UNAVAILABLE_MESSAGE, ok=False)
        except Exception:  # noqa: BLE001
            return LLMReply(UNAVAILABLE_MESSAGE, ok=False)

    def stream_chat(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield reply chunks as they arrive, or nothing at all if the call fails.

        Yielding nothing is the signal to fall back: a caller that collects an
        empty string knows the model never spoke, without inspecting wording.
        """
        if requests is None:
            return
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": True},
                timeout=self.timeout,
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except ValueError:
                        continue
                    if chunk := (payload.get("message") or {}).get("content"):
                        yield chunk
                    if payload.get("done"):
                        break
        except Exception:  # noqa: BLE001
            return

    # ----------------------------------------------------------- back-compat
    def generate(self, prompt: str) -> str:
        """Single-prompt completion, kept for callers that have no history."""
        return self.chat([{"role": "user", "content": prompt}]).text

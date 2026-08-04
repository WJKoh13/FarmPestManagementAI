"""A thin client for a local Ollama server.

The app is offline-first: Ollama is an enhancement that rephrases the vetted
guidance in `treatment_guides.py` around what the farmer actually asked. When it
is not running, every caller must still produce a useful answer, so failure is
reported as ``LLMReply.ok is False`` rather than raised.

Three capabilities, because the agent needs all three from one server:
``chat``/``stream_chat`` for prose, ``chat_with_tools`` for the function-calling
round-trips in `agent.py`, and ``embed`` for the knowledge index.

Start it with::

    brew install ollama && ollama serve
    ollama pull qwen2.5:3b        # tool-capable, instruction-tuned, sub-3B
    ollama pull qwen2.5:1.5b      # the faster fallback
    ollama pull nomic-embed-text  # embeddings for the knowledge base
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

try:
    import requests
except ModuleNotFoundError:  # The CNN demo can run without a local LLM client.
    requests = None

DEFAULT_BASE_URL = "http://localhost:11434"

# Tool calling is not optional for this app any more -- `agent.py` is the primary
# path and it needs a model that can emit tool_calls. phi3, the previous default,
# cannot, so it would have silently demoted every turn to the offline guides.
DEFAULT_MODEL = "qwen2.5:3b"
FALLBACK_MODEL = "qwen2.5:1.5b"
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Ollama unloads a model after five minutes by default. A turn makes two calls --
# the tool phase, then the streamed answer -- and a farmer's follow-up comes
# minutes later, so the default would pay the load cost several times per
# conversation. This is the single biggest lever on how fast the app feels.
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# How many times the agent may go round the call-a-tool loop. Enough to classify,
# look up a guide, and recover from one malformed call; low enough that a model
# stuck in a loop fails fast instead of hanging.
MAX_TOOL_ROUNDS = 3

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


@dataclass
class ToolCall:
    """One request from the model to run a tool, already parsed and normalised."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallReply:
    """A tool-phase reply: prose, a request to run tools, or a failure.

    ``unsupported`` is separate from ``ok`` on purpose. A model that cannot call
    tools is not a broken server -- the caller can retry the same messages
    without a ``tools`` array and still get a grounded answer, which is a much
    better outcome than dropping straight to the written guide.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Echoed back into the transcript verbatim. Ollama expects the assistant turn
    # that *asked* for the tools to precede their results, and rebuilding it by
    # hand drops fields some models require to make sense of their own request.
    raw_message: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    unsupported: bool = False

    def __bool__(self) -> bool:
        return self.ok


def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """The model's ``tool_calls`` as a clean list, tolerating how small models emit them.

    A 3B model is loose about the envelope in three specific ways, all seen in
    practice and all recoverable: ``arguments`` arrives as a JSON *string*
    rather than an object; the name comes back with different case or
    surrounding whitespace; or a malformed entry sits beside two good ones.
    None of those is worth failing a turn over, so each is repaired here and a
    hopeless entry is dropped rather than raised.
    """
    parsed: list[ToolCall] = []
    for entry in message.get("tool_calls") or []:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        name = str(function.get("name") or "").strip().lower()
        if not name:
            continue

        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                # A model that emits an unparseable argument blob still told us
                # which tool it wants. Run it with defaults and let the tool's
                # own leniency (or its error payload) handle the rest.
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        parsed.append(ToolCall(name=name, arguments=arguments))
    return parsed


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
    def pulled_models(self) -> list[str]:
        """Every model tag the server has locally, or [] if it cannot be asked."""
        if requests is None:
            return []
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=CONNECT_TIMEOUT)
            response.raise_for_status()
            return [model.get("name", "") for model in response.json().get("models", [])]
        except Exception:  # noqa: BLE001 - any failure means "nothing usable", not a crash
            return []

    @staticmethod
    def _has(names: list[str], wanted: str) -> bool:
        # Ollama reports "qwen2.5:3b" for a model pulled under that tag, but
        # "phi3:latest" for one pulled as bare "phi3" -- so compare the stem too.
        return any(name == wanted or name.split(":")[0] == wanted.split(":")[0] for name in names)

    def available(self, force: bool = False) -> bool:
        """Whether the server answers and has the configured model pulled."""
        if requests is None:
            return False
        now = time.monotonic()
        if not force and self._health and now - self._health[0] < HEALTH_TTL_SECONDS:
            return self._health[1]

        ok = self._has(self.pulled_models(), self.model)
        self._health = (now, ok)
        return ok

    def resolve_model(self) -> str:
        """The model to actually call: the configured one, or the smaller fallback.

        Falling back rather than failing matters on a demo machine where only the
        1.5B got pulled in time. Both speak the same tool protocol, so the only
        thing that changes is answer quality -- and a slightly worse answer beats
        no language model at all.
        """
        names = self.pulled_models()
        if not names or self._has(names, self.model):
            return self.model
        if self._has(names, FALLBACK_MODEL):
            return FALLBACK_MODEL
        return self.model

    @property
    def status_line(self) -> str:
        """One short line for the sidebar."""
        if requests is None:
            return "requests not installed"
        if self.available():
            return f"{self.model} · ready"
        resolved = self.resolve_model()
        if resolved != self.model:
            return f"{resolved} · ready (fallback)"
        return f"{self.model} · not running"

    # -------------------------------------------------------------------- chat
    def chat(self, messages: list[dict[str, str]]) -> LLMReply:
        """A multi-turn completion. ``messages`` is Ollama's role/content list."""
        if requests is None:
            return LLMReply(NO_CLIENT_MESSAGE, ok=False)
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False,
                      "keep_alive": KEEP_ALIVE},
                timeout=self.timeout,
            )
            response.raise_for_status()
            text = (response.json().get("message") or {}).get("content", "").strip()
            return LLMReply(text, ok=True) if text else LLMReply(UNAVAILABLE_MESSAGE, ok=False)
        except Exception:  # noqa: BLE001
            return LLMReply(UNAVAILABLE_MESSAGE, ok=False)

    def chat_with_tools(self, messages: list[dict[str, Any]],
                        tools: list[dict[str, Any]]) -> ToolCallReply:
        """One tool-phase turn: offer the tools and see what the model asks for.

        Deliberately not streamed. Ollama only fills ``message.tool_calls`` on a
        non-streaming response, and a half-parsed tool call is worse than a
        slightly later first token -- the answer phase streams instead, once the
        tools have run.
        """
        if requests is None:
            return ToolCallReply(text=NO_CLIENT_MESSAGE, ok=False)
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "tools": tools,
                      "stream": False, "keep_alive": KEEP_ALIVE},
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                # A model without tool support is a recoverable condition, not an
                # outage: the caller retries the same messages without `tools`.
                body = (response.text or "").lower()
                if "tool" in body and ("support" in body or "not supported" in body):
                    return ToolCallReply(ok=False, unsupported=True)
                response.raise_for_status()

            message = response.json().get("message") or {}
            return ToolCallReply(
                text=str(message.get("content") or "").strip(),
                tool_calls=_parse_tool_calls(message),
                raw_message=message,
                ok=True,
            )
        except Exception:  # noqa: BLE001
            return ToolCallReply(text=UNAVAILABLE_MESSAGE, ok=False)

    def stream_chat(self, messages: list[dict[str, Any]]) -> Iterator[str]:
        """Yield reply chunks as they arrive, or nothing at all if the call fails.

        Yielding nothing is the signal to fall back: a caller that collects an
        empty string knows the model never spoke, without inspecting wording.
        """
        if requests is None:
            return
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": True,
                      "keep_alive": KEEP_ALIVE},
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

    # -------------------------------------------------------------- embeddings
    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Embed a batch, or None if the embedding model cannot be reached.

        None rather than an exception or a zero vector: `knowledge.py` reads it
        as "fall back to keyword search". A zero vector would instead score every
        passage identically and quietly return nonsense.
        """
        if requests is None or not texts:
            return None
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": EMBED_MODEL, "input": texts, "keep_alive": KEEP_ALIVE},
                timeout=self.timeout,
            )
            response.raise_for_status()
            vectors = response.json().get("embeddings")
            if not isinstance(vectors, list) or len(vectors) != len(texts):
                return None
            return [[float(value) for value in vector] for vector in vectors]
        except Exception:  # noqa: BLE001
            return None

    def embeddings_available(self) -> bool:
        """Whether the embedding model is pulled. Cheap enough to call per request."""
        return self._has(self.pulled_models(), EMBED_MODEL)

    # ----------------------------------------------------------- back-compat
    def generate(self, prompt: str) -> str:
        """Single-prompt completion, kept for callers that have no history."""
        return self.chat([{"role": "user", "content": prompt}]).text

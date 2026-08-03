from __future__ import annotations

import json
import os
from typing import Any

try:
    import requests
except ModuleNotFoundError:  # The CNN demo can run without a local LLM client.
    requests = None


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str = "phi3") -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "phi3")

    def generate(self, prompt: str) -> str:
        if requests is None:
            return "Local language model is unavailable."
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("response", "")
        except Exception:
            return (
                "I’m running in offline fallback mode. Upload a pest photo for a quick identification "
                "and treatment suggestion."
            )

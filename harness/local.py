"""Local model client: talks to any OpenAI-compatible server on your machine.

Works with llama.cpp's `llama-server`, Ollama (`/v1`), LM Studio, etc.
Same chat() interface as OpenRouterClient, so agents work unchanged —
no API key, no rate limits, no pacing needed.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Dict, List, Optional

import requests


class LocalError(Exception):
    pass


class LocalClient:
    """Minimal OpenAI-compatible chat client for local servers."""

    def __init__(self, base_url: str = "http://localhost:8080/v1", timeout: int = 300):
        self.base_url = (base_url or "").rstrip("/") or "http://localhost:8080/v1"
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
        on_wait: Optional[Callable[[str], None]] = None,
        retries: Optional[int] = None,
    ) -> str:
        """Run a chat completion against the local server."""
        payload = {
            "model": model or "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        max_retries = 2 if retries is None else retries
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                    stream=stream,
                )
            except requests.RequestException as e:
                last_err = LocalError(
                    f"Cannot reach local server at {self.base_url} ({e}). "
                    f"Is llama-server / Ollama / LM Studio running?"
                )
                time.sleep(1 + attempt)
                continue
            if resp.status_code != 200:
                raise LocalError(f"Local server error ({resp.status_code}): {resp.text[:300]}")
            if not stream:
                try:
                    return resp.json()["choices"][0]["message"]["content"] or ""
                except Exception as e:
                    raise LocalError(f"Unexpected local server response: {e}")
            full: List[str] = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    piece = json.loads(chunk)["choices"][0]["delta"].get("content") or ""
                except Exception:
                    continue
                if piece:
                    full.append(piece)
                    if on_token:
                        on_token(piece)
            return "".join(full)
        raise last_err or LocalError("Local server request failed")

    def list_models(self) -> List[str]:
        """Best-effort model list from the local server; may be empty."""
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=10)
            resp.raise_for_status()
            return [m.get("id", "") for m in resp.json().get("data", []) if m.get("id")]
        except Exception:
            return []

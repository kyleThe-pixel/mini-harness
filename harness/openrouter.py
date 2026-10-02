"""Minimal OpenRouter client (chat completions + model listing).

Uses the OpenAI-compatible /chat/completions endpoint so any OpenRouter
model — including the free :free tier — works through one code path.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Dict, List, Optional, Tuple

import requests

from .config import FALLBACK_FREE_MODELS, OPENROUTER_BASE_URL


class OpenRouterError(Exception):
    """Raised for API-level failures with a human-readable message."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class OpenRouterClient:
    """Thin wrapper over OpenRouter's chat API with retry + streaming."""

    def __init__(
        self,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        app_name: str = "mini-harness",
        timeout: int = 120,
        min_interval: float = 3.2,
        max_retries: int = 8,
    ):
        if not api_key:
            raise OpenRouterError(
                "No OpenRouter API key. Add one in the web UI (stored only in your "
                "browser) or set OPENROUTER_API_KEY in a .env file. Free keys at "
                "https://openrouter.ai/keys"
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.app_name = app_name
        self.timeout = timeout
        # Pacing + retries keep missions alive under the free-tier ~20 req/min limit.
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last_call_ts = 0.0

    def _pace(self):
        """Space requests out so we stay under the per-minute rate limit."""
        wait = self.min_interval - (time.time() - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.time()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": self.app_name,
        }

    def _friendly_error(self, status: int, body: str) -> str:
        if status == 401:
            return "OpenRouter rejected the API key (401). Check the key at https://openrouter.ai/keys"
        if status == 402:
            return "OpenRouter reports insufficient credits (402). Free :free models should not need credits."
        if status == 429:
            return ("OpenRouter rate limit hit (429). Free tier is roughly 20 requests/min "
                    "and resets each minute — the client paces requests and retries "
                    "patiently, so just wait it out.")
        try:
            msg = json.loads(body).get("error", {}).get("message", "")
            if msg:
                return f"OpenRouter error ({status}): {msg[:300]}"
        except Exception:
            pass
        return f"OpenRouter error ({status}): {body[:300]}"

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
        """Run a chat completion. Returns the full assistant text.

        With stream=True, calls on_token(chunk) for each streamed piece.
        Requests are paced (~3s apart) and 429s are retried with backoff,
        calling on_wait(status_message) while waiting.
        """
        max_retries = self.max_retries if retries is None else retries
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        last_err: Optional[OpenRouterError] = None
        for attempt in range(max_retries + 1):
            self._pace()
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                    stream=stream,
                )
            except requests.RequestException as e:
                last_err = OpenRouterError(f"Network error talking to OpenRouter: {e}")
                time.sleep(1 + attempt)
                continue
            if resp.status_code == 429 and attempt < max_retries:
                try:
                    wait = int(resp.headers.get("Retry-After", 0)) or min(60, 5 * (2 ** attempt))
                except (TypeError, ValueError):
                    wait = min(60, 5 * (2 ** attempt))
                last_err = OpenRouterError(self._friendly_error(429, resp.text), 429)
                if on_wait:
                    on_wait(f"Rate limited (429) — waiting {wait}s before retry "
                            f"{attempt + 1}/{max_retries}…")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise OpenRouterError(self._friendly_error(resp.status_code, resp.text), resp.status_code)
            if not stream:
                data = resp.json()
                return data["choices"][0]["message"]["content"] or ""
            # Streaming path: parse SSE chunks.
            full: List[str] = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"]
                except Exception:
                    continue
                piece = delta.get("content") or ""
                # Some providers put tool calls here; we use a text protocol instead.
                if piece:
                    full.append(piece)
                    if on_token:
                        on_token(piece)
            return "".join(full)
        raise last_err or OpenRouterError("OpenRouter request failed after retries")

    def list_free_models(self) -> List[Tuple[str, str]]:
        """Return [(model_id, label)] for currently-free models, live from OpenRouter."""
        try:
            resp = requests.get(
                f"{self.base_url}/models", headers=self._headers(), timeout=30
            )
            resp.raise_for_status()
            out: List[Tuple[str, str]] = []
            for m in resp.json().get("data", []):
                pricing = m.get("pricing", {}) or {}
                try:
                    free = float(pricing.get("prompt", "1")) == 0 and float(
                        pricing.get("completion", "1")
                    ) == 0
                except (TypeError, ValueError):
                    free = False
                if free:
                    name = m.get("name") or m["id"]
                    out.append((m["id"], f"{name} (free)"))
            out.sort(key=lambda x: x[0])
            if out:
                return out
        except Exception:
            pass
        return list(FALLBACK_FREE_MODELS)

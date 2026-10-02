"""OpenRouter client with multi-key routing.

Give it one key or many. With several keys (e.g. a group pooling each
person's own account quota), requests rotate round-robin across keys, each
paced independently. A 429 on one key just reroutes to the next while the
throttled key cools down — callers only wait when every key is throttled.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Dict, List, Optional

import requests

from .config import FALLBACK_FREE_MODELS, OPENROUTER_BASE_URL
from .keypool import KeyPool, fingerprint


class OpenRouterError(Exception):
    """Raised for API-level failures with a human-readable message."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


_NO_KEY_MSG = (
    "No OpenRouter API key. Add one in the web UI (stored only in your "
    "browser) or set OPENROUTER_API_KEY / OPENROUTER_API_KEYS in a .env file. "
    "Free keys at https://openrouter.ai/keys"
)


class OpenRouterClient:
    """Thin wrapper over OpenRouter's chat API with key rotation + streaming."""

    def __init__(
        self,
        api_key: str = "",
        api_keys: Optional[List[str]] = None,
        pool: Optional[KeyPool] = None,
        base_url: str = OPENROUTER_BASE_URL,
        app_name: str = "mini-harness",
        timeout: int = 120,
        min_interval: float = 3.2,
        max_retries: int = 8,
    ):
        if pool is not None:
            self.pool = pool
        else:
            keys = list(api_keys) if api_keys else []
            if api_key:
                keys.append(api_key)
            keys = [k.strip() for k in keys if k and k.strip()]
            if not keys:
                raise OpenRouterError(_NO_KEY_MSG)
            self.pool = KeyPool(keys, min_interval=min_interval)
        self.base_url = base_url.rstrip("/")
        self.app_name = app_name
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self, key: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {key}",
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
        Requests rotate across the key pool; a 429 reroutes to the next key
        and on_wait(status_message) fires while waiting.
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
            key, wait = self.pool.acquire()
            if wait > 0:
                if on_wait:
                    on_wait(f"All {self.pool.size} keys throttled — waiting {wait:.0f}s…")
                time.sleep(wait)
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(key),
                    json=payload,
                    timeout=self.timeout,
                    stream=stream,
                )
            except requests.RequestException as e:
                last_err = OpenRouterError(f"Network error talking to OpenRouter: {e}")
                time.sleep(1 + attempt)
                continue
            if resp.status_code == 429:
                try:
                    wait_s = int(resp.headers.get("Retry-After", 0)) or min(60, 5 * (2 ** attempt))
                except (TypeError, ValueError):
                    wait_s = min(60, 5 * (2 ** attempt))
                self.pool.report_throttled(key, wait_s)
                last_err = OpenRouterError(self._friendly_error(429, resp.text), 429)
                if on_wait:
                    on_wait(f"Key {fingerprint(key)} rate limited — rotating to next key…")
                continue
            self.pool.report_ok(key)
            if resp.status_code != 200:
                raise OpenRouterError(self._friendly_error(resp.status_code, resp.text), resp.status_code)
            if not stream:
                return resp.json()["choices"][0]["message"]["content"] or ""
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
        raise last_err or OpenRouterError("OpenRouter request failed after retries")

    def list_free_models(self) -> List[tuple]:
        """Return [(model_id, label)] for currently-free models, live from OpenRouter."""
        key, _ = self.pool.acquire()
        try:
            resp = requests.get(
                f"{self.base_url}/models", headers=self._headers(key), timeout=30
            )
            resp.raise_for_status()
            out: List[tuple] = []
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

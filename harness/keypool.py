"""KeyPool: round-robin routing across multiple OpenRouter API keys.

Each key is rate-limited independently by OpenRouter (limits are per
account), so a group pooling each person's own key multiplies the shared
quota. Throttled keys are skipped until their cool-down expires; each key
is paced independently to stay under ~20 req/min.

Thread-safe: the server shares one pool across all request threads.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Tuple


def fingerprint(key: str) -> str:
    """Last-4 fingerprint for logs/UI — never expose full keys."""
    return "••••" + key[-4:] if len(key) >= 4 else "••••"


class KeyPool:
    def __init__(self, keys: List[str], min_interval: float = 3.2):
        keys = [k.strip() for k in keys if k and k.strip()]
        if not keys:
            raise ValueError("KeyPool needs at least one API key")
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._entries: List[Dict] = [
            {"key": k, "fp": fingerprint(k), "throttled_until": 0.0, "last_used": 0.0}
            for k in keys
        ]

    @property
    def size(self) -> int:
        return len(self._entries)

    def acquire(self) -> Tuple[str, float]:
        """Pick the least-recently-used available key.

        Returns (key, wait_seconds). If every key is throttled, returns the
        key whose cool-down ends soonest plus how long to wait for it.
        """
        with self._lock:
            now = time.time()
            avail = [e for e in self._entries if e["throttled_until"] <= now]
            if not avail:
                soonest = min(self._entries, key=lambda e: e["throttled_until"])
                return soonest["key"], max(0.0, soonest["throttled_until"] - now)
            avail.sort(key=lambda e: e["last_used"])
            entry = avail[0]
            wait = max(0.0, self.min_interval - (now - entry["last_used"]))
            entry["last_used"] = now + wait
            return entry["key"], wait

    def report_ok(self, key: str) -> None:
        with self._lock:
            for e in self._entries:
                if e["key"] == key:
                    e["throttled_until"] = 0.0

    def report_throttled(self, key: str, wait: float) -> None:
        with self._lock:
            for e in self._entries:
                if e["key"] == key:
                    e["throttled_until"] = time.time() + max(0.0, wait)

    def status(self) -> Dict:
        with self._lock:
            now = time.time()
            return {
                "keys": len(self._entries),
                "throttled": sum(1 for e in self._entries if e["throttled_until"] > now),
                "fingerprints": [e["fp"] for e in self._entries],
            }

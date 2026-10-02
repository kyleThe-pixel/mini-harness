"""Configuration: settings and the OpenRouter free-model catalog."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Offline fallback list of free OpenRouter models (id -> human label).
# The app prefers the live list from OpenRouter's /models endpoint; this is
# only used when that request fails. Free-model availability changes, so
# refresh from the UI when in doubt.
FALLBACK_FREE_MODELS = [
    ("qwen/qwen3-coder:free", "Qwen3 Coder (free) — best for code"),
    ("deepseek/deepseek-chat-v3-0324:free", "DeepSeek V3 0324 (free) — strong all-round"),
    ("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B (free)"),
    ("google/gemma-3-27b-it:free", "Gemma 3 27B (free)"),
    ("mistralai/mistral-small-3.1-24b-instruct:free", "Mistral Small 3.1 24B (free)"),
    ("qwen/qwen3-235b-a22b:free", "Qwen3 235B A22B (free)"),
]


@dataclass
class Settings:
    """Runtime settings. API key can come from the request, env, or .env file."""

    api_key: str = ""
    base_url: str = OPENROUTER_BASE_URL
    workspace: str = "workspace"
    app_name: str = "mini-harness"
    default_model: str = "qwen/qwen3-coder:free"
    request_timeout: int = 120
    max_inner_iters: int = 8
    max_rounds: int = 4
    # Local provider (Ollama / llama.cpp / LM Studio). "openrouter" or "local".
    provider: str = "openrouter"
    local_url: str = "http://localhost:8080/v1"
    local_model: str = ""

    @classmethod
    def from_env(cls, api_key_override: str = "") -> "Settings":
        # Tiny .env loader so we don't need python-dotenv.
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env_file):
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return cls(
            api_key=api_key_override or os.environ.get("OPENROUTER_API_KEY", ""),
            base_url=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            workspace=os.environ.get("HARNESS_WORKSPACE", "workspace"),
            app_name=os.environ.get("HARNESS_APP_NAME", "mini-harness"),
            default_model=os.environ.get("HARNESS_DEFAULT_MODEL", "qwen/qwen3-coder:free"),
            provider=os.environ.get("HARNESS_PROVIDER", "openrouter"),
            local_url=os.environ.get("HARNESS_LOCAL_URL", "http://localhost:8080/v1"),
            local_model=os.environ.get("HARNESS_LOCAL_MODEL", ""),
        )

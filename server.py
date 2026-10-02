"""Flask server: serves the web UI and streams agent/team runs over SSE."""
from __future__ import annotations

import json
import os
import queue
import threading

from flask import Flask, Response, jsonify, request, send_from_directory

from harness.agent import Agent
from harness.config import Settings
from harness.openrouter import OpenRouterClient, OpenRouterError
from harness.team import Team
from harness.tools import ToolRegistry

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "web"), static_url_path="")


def _settings(api_key_override: str = "") -> Settings:
    s = Settings.from_env(api_key_override)
    # workspace is relative to the project root
    if not os.path.isabs(s.workspace):
        s.workspace = os.path.join(BASE_DIR, s.workspace)
    return s


def _client(api_key: str, settings: Settings) -> OpenRouterClient:
    return OpenRouterClient(api_key or settings.api_key,
                            base_url=settings.base_url,
                            app_name=settings.app_name,
                            timeout=settings.request_timeout)


# One KeyPool shared by every request thread, so pacing and throttle state
# are global. Built once from the server's env keys.
_pool = None
_pool_lock = threading.Lock()


def _shared_pool(settings: Settings):
    global _pool
    with _pool_lock:
        if _pool is None and settings.api_keys:
            from harness.keypool import KeyPool
            _pool = KeyPool(settings.api_keys)
        return _pool


def _openrouter_client(key: str, settings: Settings) -> OpenRouterClient:
    """Client backed by the shared pooled keys, or a single request key."""
    pool = _shared_pool(settings)
    if pool is None:
        single = (key or settings.api_key or "").strip()
        if not single:
            raise OpenRouterError(
                "No OpenRouter API key. Add one in the web UI (stored only in your "
                "browser) or set OPENROUTER_API_KEY / OPENROUTER_API_KEYS in a .env "
                "file. Free keys at https://openrouter.ai/keys")
        from harness.keypool import KeyPool
        pool = KeyPool([single])
    return OpenRouterClient(pool=pool, base_url=settings.base_url,
                            app_name=settings.app_name,
                            timeout=settings.request_timeout)


def _resolve_client(key: str, settings: Settings, body: dict):
    """Pick OpenRouter or a local server based on the request (falls back to env)."""
    provider = (body.get("provider") or settings.provider or "openrouter").lower()
    if provider == "local":
        from harness.local import LocalClient
        return LocalClient(body.get("local_url") or settings.local_url,
                           timeout=settings.request_timeout), "local"
    return _openrouter_client(key, settings), "openrouter"


def _resolve_model(settings: Settings, body: dict, provider: str) -> str:
    if provider == "local":
        return body.get("model") or settings.local_model or "default"
    return body.get("model") or settings.default_model


def _sse(responses: "queue.Queue[str]"):
    """Yield queued JSON payloads as server-sent events."""
    while True:
        payload = responses.get()
        if payload is None:  # sentinel: end of stream
            break
        yield f"data: {payload}\n\n"


def _run_streamed(target):
    """Run target(emit) in a thread; stream its events as SSE."""
    responses: queue.Queue = queue.Queue()

    def emit(ev: dict):
        responses.put(json.dumps(ev))

    def worker():
        try:
            target(emit)
        except OpenRouterError as e:
            emit({"type": "error", "text": str(e)})
        except Exception as e:  # noqa: BLE001 - surface it to the UI
            emit({"type": "error", "text": f"{type(e).__name__}: {e}"})
        finally:
            emit({"type": "done"})
            responses.put(None)

    threading.Thread(target=worker, daemon=True).start()
    return Response(_sse(responses), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# -- pages -----------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# -- models ----------------------------------------------------------------
@app.get("/api/models")
def models():
    key = request.args.get("key", "")
    settings = _settings(key)
    provider = (request.args.get("provider") or settings.provider or "openrouter").lower()
    if provider == "local":
        from harness.local import LocalClient
        local_url = request.args.get("local_url") or settings.local_url
        names = LocalClient(local_url).list_models()
        if settings.local_model and settings.local_model not in names:
            names.insert(0, settings.local_model)
        if not names:
            names = [settings.local_model or "default"]
        return jsonify({
            "provider": "local",
            "models": [{"id": n, "label": f"{n} (local)"} for n in names],
            "default": names[0],
            "key_configured": True,
            "local_url": local_url,
        })
    try:
        client = _openrouter_client(key, settings)
        free = client.list_free_models()
    except OpenRouterError:
        from harness.config import FALLBACK_FREE_MODELS
        free = list(FALLBACK_FREE_MODELS)
    return jsonify({
        "provider": "openrouter",
        "models": [{"id": mid, "label": label} for mid, label in free],
        "default": settings.default_model,
        "key_configured": bool(key or settings.api_key or settings.api_keys),
    })


# -- key pool status ---------------------------------------------------------
@app.get("/api/pool")
def pool_info():
    """How many pooled keys the gateway holds (fingerprints only, never keys)."""
    settings = _settings(request.args.get("key", ""))
    pool = _shared_pool(settings)
    if pool is None:
        return jsonify({"keys": 0, "throttled": 0, "fingerprints": []})
    return jsonify(pool.status())


# -- single-agent chat ------------------------------------------------------
@app.post("/api/chat")
def chat():
    body = request.get_json(force=True) or {}
    message = body.get("message", "").strip()
    key = body.get("key", "")
    if not message:
        return jsonify({"error": "empty message"}), 400
    settings = _settings(key)

    def target(emit):
        client, provider = _resolve_client(key, settings, body)
        model = _resolve_model(settings, body, provider)
        registry = ToolRegistry(settings.workspace)
        agent = Agent("solo", "helpful assistant", model, client, registry,
                      max_inner_iters=settings.max_inner_iters)
        emit({"type": "agent_start", "agent": "solo"})
        out = agent.run_turn(message, on_event=emit)
        emit({"type": "agent_done", "agent": "solo", "final": out["final"]})

    return _run_streamed(target)


# -- multi-agent mission ----------------------------------------------------
@app.post("/api/mission")
def mission():
    body = request.get_json(force=True) or {}
    goal = body.get("goal", "").strip()
    specs = body.get("agents") or []
    key = body.get("key", "")
    if not goal:
        return jsonify({"error": "empty goal"}), 400
    if not specs:
        return jsonify({"error": "no agents specified"}), 400
    settings = _settings(key)

    def target(emit):
        client, provider = _resolve_client(key, settings, body)
        default_model = _resolve_model(settings, body, provider)
        for s in specs:
            s["model"] = s.get("model") or default_model
        registry = ToolRegistry(settings.workspace)
        team = Team(client, registry,
                    max_inner_iters=settings.max_inner_iters,
                    max_rounds=int(body.get("max_rounds", settings.max_rounds)))
        team.run_mission(goal, specs, on_event=emit)

    return _run_streamed(target)


# -- file browsing (read-only view of the agent workspace) -------------------
@app.get("/api/files")
def files():
    settings = _settings(request.args.get("key", ""))
    root = settings.workspace
    os.makedirs(root, exist_ok=True)

    def tree(path, rel=""):
        nodes = []
        for e in sorted(os.listdir(path)):
            if e.startswith("."):
                continue
            full = os.path.join(path, e)
            r = os.path.join(rel, e)
            if os.path.isdir(full):
                nodes.append({"name": e, "path": r, "dir": True,
                              "children": tree(full, r)})
            else:
                nodes.append({"name": e, "path": r, "dir": False})
        return nodes

    return jsonify({"root": root, "tree": tree(root)})


@app.get("/api/file")
def file_content():
    settings = _settings(request.args.get("key", ""))
    rel = request.args.get("path", "")
    full = os.path.abspath(os.path.join(settings.workspace, rel))
    if not full.startswith(os.path.abspath(settings.workspace) + os.sep):
        return jsonify({"error": "path escapes workspace"}), 400
    if not os.path.isfile(full):
        return jsonify({"error": "not a file"}), 404
    with open(full, encoding="utf-8", errors="replace") as f:
        content = f.read(200000)
    return jsonify({"path": rel, "content": content})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"mini-harness on http://localhost:{port}  (workspace: {_settings().workspace})")
    app.run(host="0.0.0.0", port=port, threaded=True)

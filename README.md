# ⚡ mini-harness

A hyper-lightweight multi-agent AI harness. A tiny "you" that runs in a browser tab:
multiple AI agents that talk to each other, use file/shell tools, and complete tasks —
powered by **free models on [OpenRouter](https://openrouter.ai)**.

No build step. No database. One Python server + one static web page.

## What it does

- **Agent loop** — each agent thinks, calls tools (`read_file`, `write_file`, `edit_file`,
  `list_dir`, `delete_file`, `run_shell`, `web_fetch`), reads results, and repeats until done.
- **Multi-agent teams** — a lead agent breaks your goal into subtasks, assigns them to
  teammates, and the agents message each other (`💬 → coder: ...`) while they work.
  The lead synthesizes a final summary.
- **Free models** — uses OpenRouter's `:free` tier (Qwen3 Coder, DeepSeek V3, Llama 3.3…).
  The UI pulls the live free-model list from OpenRouter; pick a different model per agent.
- **Web UI** — mission control (team view with live per-agent feeds), solo chat, and a
  read-only workspace file browser. Streams everything live over SSE.
- **Sandboxed tools** — agents can only touch files inside `workspace/`. Path escapes are rejected.

## Quickstart

1. **Get a free OpenRouter key** → https://openrouter.ai/keys (sign up, create a key —
   the `:free` models cost nothing).

2. **Install & run:**
   macOS / Linux:
   ```bash
   cd harness
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   .venv/bin/python server.py
   ```
   Windows (cmd):
   ```bat
   cd harness
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   .venv\Scripts\python server.py
   ```
   Open http://localhost:5000

3. **Paste your key** in the top bar (saved in your browser only, never on disk
   unless you put it in `.env`), pick models, and launch a mission like:

   > *Build a tiny Python CLI that converts CSV to JSON, with tests.*

   Watch the lead split it up, the coder write files, the reviewer check them —
   all live.

Or set the key via `.env` instead:
```bash
cp .env.example .env   # then edit OPENROUTER_API_KEY
```

## How it works

```
web/ (static UI) ──SSE──▶ server.py (Flask)
                              ├─ POST /api/mission → Team.run_mission()
                              │     lead plans → rounds: agents run turns,
                              │     message each other, share a bulletin
                              │     → lead synthesizes summary
                              ├─ POST /api/chat    → single Agent.run_turn()
                              ├─ GET  /api/models  → live free-model list
                              └─ GET  /api/files   → workspace browser
harness/
  agent.py      the tool-calling loop (text protocol, model-agnostic)
  team.py       multi-agent orchestration + message routing
  tools.py      sandboxed file/shell/web tools + protocol parser
  openrouter.py OpenRouter client (streaming, retries, free-model listing)
  config.py     settings + fallback model catalog
```

**Why a text tool protocol?** Free-tier models have spotty native function-calling.
Agents emit fenced blocks instead — ```` ```tool {"name": "read_file", ...} ```` —
which the loop parses and executes. Works with any chat model.

## Project layout

```
harness/
  server.py            # run this
  harness/             # the agent framework
  web/                 # the UI (index.html, app.js, style.css)
  tests/               # python3 tests/test_harness.py — 7 tests, no key needed
  workspace/           # where agents work (created on first run, git-ignored)
  requirements.txt     # flask, requests
```

## Pooling API keys (group gateway)

Run one server for the whole group. Put each person's **own** OpenRouter key in the
server env (comma-separated):

```bash
OPENROUTER_API_KEYS=sk-or-v1-aaa,sk-or-v1-bbb,sk-or-v1-ccc
```

The gateway routes requests round-robin across keys, each paced independently
(~3s apart, so N keys ≈ N×18 req/min). A 429 on one key just rotates to the next —
throttled keys cool down in the background and rejoin automatically. Limits are per
OpenRouter account, so each person's key contributes their own quota.

- `GET /api/pool` shows key count and throttled state (key fingerprints only —
  full keys never leave the server).
- The header shows `🔑 ×N pooled` when the server holds pooled keys.
- No per-user caps, no auth — it's a trusted-group tool. Don't expose it to the internet.

## Notes & limits

- Free-tier rate limits are roughly 20 req/min per model — the client paces requests (~3s
  apart) and retries 429s with exponential backoff (up to 8 tries), so missions slow
  down instead of dying. You'll see `⏳ Rate limited…` status lines in the UI while it waits.
- `run_shell` is powerful by design (agents need to run code). Only run the server on
  machines you trust, and keep the workspace separate from anything precious.
- The UI has no login; don't expose port 5000 to the internet as-is.

## Local models (no rate limits)

Flip the provider dropdown in the UI header from **OpenRouter** to **Local** and the
harness talks to any OpenAI-compatible server on your machine — no API key, no quotas.
Set the server URL and model name in the header; every agent uses that model.

**llama.cpp** (recommended for raw `.gguf` files) — grab `llama-server` from the
[llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases), then:

```bat
llama-server -m "C:\models\Bonsai-27B-GGUF\model.gguf" --port 8080 --ctx-size 16384 -ngl 99
```

**Ollama** — `ollama serve`, then `ollama run <model>`; the OpenAI endpoint is
`http://localhost:11434/v1`. Custom GGUFs can be imported with a Modelfile
(`ollama create mymodel -f Modelfile`).

**LM Studio** — start its built-in server (`http://localhost:1234/v1`).

Tips:
- Give agents plenty of context: `--ctx-size 16384` or higher if you have the RAM.
- Bigger models follow the tool protocol far better. A 27B-class model is the sweet
  spot for missions; 1–3B models are fine for simple chat but get flaky on
  multi-step tool loops.
- Same knobs via env: `HARNESS_PROVIDER=local`, `HARNESS_LOCAL_URL`, `HARNESS_LOCAL_MODEL`.

## Roadmap

See [ROADMAP.md](ROADMAP.md) — local Qwen/Ollama as a backup provider is the big one.

# Roadmap — wanted / backup options

## Done
- [x] **Local models provider** (2026-10-02): `harness/local.py` + provider switch in the
  UI. Works with llama.cpp `llama-server`, Ollama, LM Studio — anything OpenAI-compatible.
  No key, no rate limits.
- [x] **Pooled multi-key routing** (2026-10-02): `harness/keypool.py` — one server-wide
  KeyPool, round-robin across each person's own OpenRouter key with per-key pacing and
  automatic throttle cool-down. `GET /api/pool` status, header indicator. No per-user
  caps, no auth (trusted group).

## Near-term
- [ ] Per-agent system-prompt editing in the UI
- [ ] Mission history (save transcripts to `workspace/.missions/`)
- [ ] Pause / stop button for a running mission (cancel the SSE + thread)
- [ ] Token usage + cost estimate per mission (OpenRouter returns usage)
- [ ] File diff view: show what agents changed, not just the file tree
- [ ] More tools: `grep`/`search_files`, `apply_patch`, image viewing

## Later / maybe
- [ ] Basic auth for the UI (currently localhost-only by assumption)
- [ ] Persistent agent memory across missions (vector store or just notes files)
- [ ] Scheduled missions (cron-style "run this every morning")
- [ ] One-click deploy (Dockerfile + compose)

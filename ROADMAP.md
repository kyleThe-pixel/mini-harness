# Roadmap — wanted / backup options

Things deliberately held for later. Highest value first.

## Backup: local models (Qwen via Ollama)
The original idea included local Qwen coder models as an alternative to API keys.
Held off for now; OpenRouter free tier is the primary path.

When wanted, the shape is already prepared:
- `OpenRouterClient` is a small class with one method the agents need: `chat(...)`.
- Add `harness/ollama.py` with an `OllamaClient` exposing the same `chat()` signature
  against `http://localhost:11434/api/chat` (Ollama's OpenAI-compatible endpoint
  is `/v1/chat/completions`, so it's nearly a drop-in).
- Add a provider toggle in `server.py` (`HARNESS_PROVIDER=openrouter|ollama`) and a
  model textbox in the UI for local model names (e.g. `qwen3-coder:30b`).
- The text tool protocol in `agent.py` already works with local chat models —
  no agent changes needed.

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

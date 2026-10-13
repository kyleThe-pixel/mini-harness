# 3050 Benchmark Harness v1

One runner that benchmarks every local model on **fixed** prompts / seeds /
settings, so runs are comparable over time. Metrics per run: wall time,
tok/s (LLM) or seconds per output (image/video), **peak VRAM**, **peak temp**,
**avg power** (all via `nvidia-smi` sampling), **peak RAM** (psutil), plus the
actual output file. Results append to `results.json`; `leaderboard.html` is
regenerated every run with a sortable table and best-per-model summary.

## Quick start

1. `setup_bench_v1.bat` — one-time, creates an isolated `bench-venv`
   (psutil + requests + gradio_client). It does **not** touch Forge or
   wan2gp envs.
2. **Edit `bench_config.json` paths first** (see below).
3. `run_bench_v1.bat` — runs everything.
   - Quick run without the 30-min video model:
     `run_bench_v1.bat --skip Wan-2.2-5B-TI2V`
   - Just the LLMs: `run_bench_v1.bat --models Bonsai-27B-Q1_0,MiMo-V2.6-Distill-Qwen-9B`
   - List models: `run_bench_v1.bat --list`

## What runs (v1 lineup)

| Model | Adapter | Primary metric |
|---|---|---|
| Bonsai-27B-Q1_0 | llama.cpp (`llama-cli`) | tok/s |
| MiMo-V2.6-Distill-Qwen-9B | llama.cpp (`llama-cli`) | tok/s |
| Z-Image-Turbo | diffusers (in `wan2gp` conda env) | s/image (gen) + load s |
| FLUX.1-Kontext | Forge API img2img | s/edit |
| Wan-2.2-5B-TI2V | WanGP Gradio API | s/video (~30 min) |

## Before the first run — check these

- **GGUF paths** in `bench_config.json` for the two llama.cpp models. They
  default to the llama.cpp folder root — fix them if your `.gguf` files live
  elsewhere. A wrong path fails fast with a clear message.
- **`n_gpu_layers`** for the LLM models is a starting guess (Bonsai 28,
  MiMo 24). Tune it to whatever your working llama-server/run.bat uses;
  lower it if you OOM.
- **Forge must be running** for the Kontext test (it uses
  `http://127.0.0.1:7860` by default — change `paths.forge_url` if yours is
  elsewhere). The Kontext checkpoint should already be loaded, or set
  `checkpoint` in its config entry to switch it first.
- **WanGP must be running** for the video test (default
  `http://127.0.0.1:7860`). Forge and WanGP share port 7860 in the
  defaults — run them one at a time.
- **Z-Image**: first run downloads ~14GB of weights into the HF cache.
  `cpu_offload: true` streams weights for your 4GB VRAM (slower, but it fits).
- **Kontext input**: defaults to `from_zimage`, i.e. the newest Z-Image
  output under `runs/`. Run Z-Image first, or set `input_image` to a path.

## Adding a model later

Add one JSON object to `models` in `bench_config.json` — that's it. Copy an
existing entry, change `name`, `adapter`, and the adapter's fields. The
runner picks it up automatically. Example (new llama.cpp model):

```json
{
  "name": "Qwen3-14B-Q4",
  "adapter": "llamacpp",
  "gguf": "C:\\models\\Qwen3-14B-Q4_K_M.gguf",
  "tokens": 200,
  "seed": 42,
  "temp": 0.0,
  "threads": 8,
  "ctx": 4096,
  "n_gpu_layers": 24
}
```

## Adding a new adapter later

1. Create `adapters/myadapter.py` with a `run(model, run_dir, sampler, cfg)`
   function returning a dict with `primary_label`, `primary_value`, and
   optionally `output_file` (filename inside `run_dir`). Use
   `sampler.track(proc)` on any subprocess you spawn so its RAM is tracked.
2. Import it in `bench.py` and add it to the `ADAPTERS` dict.
3. Reference it from a model entry: `"adapter": "myadapter"`.

## Notes / known limitations

- One failing adapter doesn't kill the run — it logs `error` in
  `results.json` and moves on.
- Kontext/WanGP are API adapters: RAM is not attributed (the server owns
  it), and their VRAM numbers reflect the server's GPU, which is the same
  card.
- WanGP has no official REST API, so its adapter auto-discovers the
  text-to-video Gradio endpoint. If discovery fails, it writes
  `wangp_api_dump.txt` in the run folder — set `endpoint` in the model
  config and put the endpoint's arguments in `extra_args`.
- WanGP runs take ~30 min; use `--skip` for fast iteration.
- `results.json` is append-only history. Delete it (and `runs/`) to start
  fresh. `leaderboard.html` is regenerated from it every run.

## Files

- `bench.py` — runner, sampler, dispatch
- `bench_config.json` — models, fixed test conditions, paths
- `adapters/` — one module per backend + `zimage_worker.py`
- `report.py` — results.json -> leaderboard.html
- `setup_bench_v1.bat` / `run_bench_v1.bat`

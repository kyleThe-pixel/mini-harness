"""llama.cpp adapter: runs llama-cli on a fixed prompt, parses tok/s.

Config fields (per model):
  gguf          - path to the .gguf file (REQUIRED, edit in bench_config.json)
  prompt        - overrides defaults.llm_prompt
  tokens        - tokens to generate (default 200)
  seed / temp / threads / ctx / n_gpu_layers
"""
import re
import subprocess
import time
from pathlib import Path


def run(model, run_dir, sampler, cfg):
    cli = cfg["paths"]["llama_cli"]
    if not Path(cli).exists():
        raise FileNotFoundError(f"llama-cli not found at {cli} (edit paths.llama_cli)")
    gguf = model.get("gguf", "")
    if not gguf or not Path(gguf).exists():
        raise FileNotFoundError(
            f"GGUF not found: {gguf!r}. Edit the gguf path for "
            f"{model['name']} in bench_config.json"
        )
    prompt = model.get("prompt", cfg.get("defaults", {}).get("llm_prompt", "Hello"))
    cmd = [
        cli,
        "-m", gguf,
        "-p", prompt,
        "-n", str(model.get("tokens", 200)),
        "--seed", str(model.get("seed", 42)),
        "-t", str(model.get("threads", 8)),
        "-c", str(model.get("ctx", 4096)),
        "--temp", str(model.get("temp", 0.0)),
        "--no-display-prompt",
    ]
    ngl = model.get("n_gpu_layers")
    if ngl is not None:
        cmd += ["-ngl", str(ngl)]
    print("  $ " + " ".join(cmd))

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    sampler.track(proc)
    out, _ = proc.communicate()
    wall = time.perf_counter() - t0
    (run_dir / "raw_output.txt").write_text(out, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"llama-cli exited {proc.returncode}; see raw_output.txt")

    # llama-cli prints: "... (1234.56 tokens per second)"
    hits = re.findall(r"([\d.]+)\s+tokens per second", out)
    if hits:
        tok_s = float(hits[-1])
    else:
        tok_s = model.get("tokens", 200) / wall if wall else 0
    return {
        "primary_label": "tok/s",
        "primary_value": round(tok_s, 2),
        "wall_seconds": round(wall, 2),
        "output_file": "raw_output.txt",
    }

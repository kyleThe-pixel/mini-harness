"""FLUX.1-Kontext adapter: POSTs a fixed img2img edit to the Forge API.

Forge must be running with the Kontext checkpoint loaded (or set
`checkpoint` in the model config to switch it via /sdapi/v1/options first).

Config fields (per model):
  prompt / steps / cfg_scale / distilled_cfg_scale / denoising / sampler /
  seed / width / height / checkpoint (optional)
  input_image: "from_zimage" (default: newest output.png under runs/)
             or an explicit image path
"""
import base64
import time
from pathlib import Path

import requests


def _latest_zimage(runs_dir):
    pngs = sorted(Path(runs_dir).glob("*/output.png"),
                  key=lambda p: p.stat().st_mtime)
    if not pngs:
        raise FileNotFoundError(
            "No Z-Image output found under runs/. Run the Z-Image Turbo "
            "benchmark first, or set input_image to an explicit path."
        )
    return pngs[-1]


def run(model, run_dir, sampler, cfg):
    url = cfg["paths"]["forge_url"].rstrip("/")
    inp = model.get("input_image", "from_zimage")
    if inp == "from_zimage":
        inp = _latest_zimage(cfg["paths"]["runs_dir"])
    if not Path(inp).exists():
        raise FileNotFoundError(f"Kontext input image not found: {inp}")

    if model.get("checkpoint"):
        requests.post(f"{url}/sdapi/v1/options",
                      json={"sd_model_checkpoint": model["checkpoint"]},
                      timeout=30).raise_for_status()

    payload = {
        "prompt": model["prompt"],
        "init_images": [base64.b64encode(Path(inp).read_bytes()).decode()],
        "denoising_strength": model.get("denoising", 1.0),
        "cfg_scale": model.get("cfg_scale", 1),
        "distilled_cfg_scale": model.get("distilled_cfg_scale", 3),
        "steps": model.get("steps", 20),
        "sampler_name": model.get("sampler", "Euler"),
        "seed": model.get("seed", 42),
        "width": model.get("width", 1024),
        "height": model.get("height", 1024),
    }
    t0 = time.perf_counter()
    r = requests.post(f"{url}/sdapi/v1/img2img", json=payload, timeout=3600)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    (run_dir / "output.png").write_bytes(base64.b64decode(data["images"][0]))
    (run_dir / "raw_output.txt").write_text(
        f"input: {inp}\ninfo: {data.get('info', '')}", encoding="utf-8")
    return {
        "primary_label": "s/edit",
        "primary_value": round(wall, 2),
        "wall_seconds": round(wall, 2),
        "output_file": "output.png",
    }

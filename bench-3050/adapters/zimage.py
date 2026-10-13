"""Z-Image Turbo adapter: spawns zimage_worker.py inside the wan2gp conda env.

Config fields (per model):
  prompt / width / height / steps / guidance / seed / model_id / cpu_offload
"""
import json
import subprocess
import time
from pathlib import Path


def run(model, run_dir, sampler, cfg):
    conda = cfg["paths"].get("conda_exe", "conda")
    env = cfg["paths"].get("conda_env", "wan2gp")
    worker = Path(__file__).resolve().parent / "zimage_worker.py"
    out_path = run_dir / "output.png"
    cmd = [
        conda, "run", "-n", env, "--no-banner", "python", str(worker),
        "--model-id", model.get("model_id", "Tongyi-MAI/Z-Image-Turbo"),
        "--prompt", model["prompt"],
        "--width", str(model.get("width", 1024)),
        "--height", str(model.get("height", 1024)),
        "--steps", str(model.get("steps", 9)),
        "--guidance", str(model.get("guidance", 0.0)),
        "--seed", str(model.get("seed", 42)),
        "--out", str(out_path),
    ]
    if model.get("cpu_offload", True):
        cmd.append("--cpu-offload")
    print("  $ " + " ".join(cmd))

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    sampler.track(proc)
    out, _ = proc.communicate()
    wall = time.perf_counter() - t0
    (run_dir / "raw_output.txt").write_text(out, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"zimage worker exited {proc.returncode}; see raw_output.txt")

    data = None
    for line in out.splitlines():
        if line.startswith("BENCH_JSON:"):
            data = json.loads(line[len("BENCH_JSON:"):])
    if not data:
        raise RuntimeError("zimage worker produced no metrics; see raw_output.txt")
    return {
        "primary_label": "s/image (gen)",
        "primary_value": data["gen_seconds"],
        "load_seconds": data["load_seconds"],
        "wall_seconds": round(wall, 2),
        "output_file": "output.png",
    }

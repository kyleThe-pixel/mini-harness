"""3050 benchmark harness - core runner (Windows).

Runs every model in bench_config.json on FIXED prompts/seeds/settings,
samples nvidia-smi in the background (peak VRAM / peak temp / avg power),
tracks peak RAM of the child process (psutil, if installed), saves the
output of every run, appends to results.json, and regenerates
leaderboard.html.

Usage:
    python bench.py [--config bench_config.json]
                    [--models Bonsai-27B-Q1_0,MiMo-V2.6-Distill-Qwen-9B]
                    [--skip Wan-2.2-5B-TI2V]
                    [--list]
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from adapters import llamacpp, zimage, kontext, wangp  # noqa: E402
import report  # noqa: E402

ADAPTERS = {
    "llamacpp": llamacpp,
    "zimage": zimage,
    "kontext": kontext,
    "wangp": wangp,
}

try:
    import psutil

    HAVE_PSUTIL = True
except ImportError:
    psutil = None
    HAVE_PSUTIL = False


class Sampler(threading.Thread):
    """Background sampler: GPU via nvidia-smi, RAM via psutil."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.peak_vram = 0
        self.peak_temp = 0
        self.power_samples = []
        self.peak_ram = 0
        self._procs = []

    def track(self, proc):
        """proc: subprocess.Popen whose process tree gets RAM-tracked."""
        self._procs.append(proc)

    def run(self):
        while not self.stop_flag.wait(0.5):
            self._sample_gpu()
            self._sample_ram()

    def _sample_gpu(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            mem, temp, power = [x.strip() for x in out.split(",")]
            self.peak_vram = max(self.peak_vram, int(float(mem)))
            self.peak_temp = max(self.peak_temp, int(float(temp)))
            self.power_samples.append(float(power))
        except Exception:
            pass  # no GPU / nvidia-smi unavailable: leave metrics empty

    def _sample_ram(self):
        if not HAVE_PSUTIL:
            return
        for p in list(self._procs):
            try:
                root = psutil.Process(p.pid)
                total = sum(
                    c.memory_info().rss
                    for c in [root] + root.children(recursive=True)
                )
                self.peak_ram = max(self.peak_ram, total // (1024 * 1024))
            except Exception:
                pass

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=5)
        return {
            "peak_vram_mb": self.peak_vram or None,
            "peak_temp_c": self.peak_temp or None,
            "avg_power_w": (
                round(sum(self.power_samples) / len(self.power_samples), 1)
                if self.power_samples
                else None
            ),
            "peak_ram_mb": self.peak_ram or None,
        }


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("runs_dir", "runs")
    cfg["paths"].setdefault("results_file", "results.json")
    return cfg


def run_one(model, cfg):
    name = model["name"]
    adapter = ADAPTERS[model["adapter"]]
    runs_dir = Path(cfg["paths"]["runs_dir"])
    run_dir = runs_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {name}  [{model['adapter']}] ===")
    sampler = Sampler()
    sampler.start()
    t0 = time.perf_counter()
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": name,
        "adapter": model["adapter"],
        "params": {k: v for k, v in model.items() if k not in ("name", "adapter")},
        "status": "ok",
        "output": None,
    }
    try:
        metrics = adapter.run(model, run_dir, sampler, cfg)
        record["metrics"] = metrics
        out = metrics.get("output_file")
        if out:
            record["output"] = str(run_dir / out)
        print(f"  -> {metrics.get('primary_label')}: {metrics.get('primary_value')}")
    except Exception as exc:  # one failing adapter must not kill the whole run
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["metrics"] = {}
        print(f"  !! FAILED: {exc}")
    record["wall_seconds"] = round(time.perf_counter() - t0, 2)
    record["metrics"].update(sampler.stop())
    return record


def main():
    ap = argparse.ArgumentParser(description="RTX 3050 benchmark harness")
    ap.add_argument("--config", default="bench_config.json")
    ap.add_argument("--models", default="", help="comma-separated model names to run")
    ap.add_argument("--skip", default="", help="comma-separated model names to skip")
    ap.add_argument("--list", action="store_true", help="list configured models and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = cfg.get("models", [])
    if args.list:
        for m in models:
            print(f"{m['name']}  [{m['adapter']}]")
        return
    if args.models:
        wanted = {x.strip() for x in args.models.split(",")}
        models = [m for m in models if m["name"] in wanted]
    if args.skip:
        banned = {x.strip() for x in args.skip.split(",")}
        models = [m for m in models if m["name"] not in banned]
    if not models:
        print("No models selected. Check --models/--skip or bench_config.json.")
        return

    results_path = Path(cfg["paths"]["results_file"])
    results = []
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            results = []

    for model in models:
        record = run_one(model, cfg)
        results.append(record)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        report.generate(results_path, ROOT / "leaderboard.html")

    print(f"\nDone. {len(models)} model(s). Results in {results_path}, "
          f"leaderboard at {ROOT / 'leaderboard.html'}")


if __name__ == "__main__":
    main()

"""WanGP adapter: drives the running WanGP Gradio app via gradio_client.

Upstream WanGP has no official REST API, so this adapter DISCOVERS the
text-to-video endpoint from the live Gradio API instead of guessing it:
  1. connect to paths.wangp_url (WanGP must already be running)
  2. if models[].endpoint is set, use it
  3. else pick the first endpoint whose name mentions t2v / text-to-video
  4. else dump the API listing to wangp_api_dump.txt and fail with
     instructions (then set endpoint + extra_args in bench_config.json)

Config fields (per model):
  prompt / endpoint (optional) / extra_args (dict, passed through)
  seconds / width / height / steps / seed are NOT passed automatically --
  WanGP's endpoint args are UI-defined and vary by version. Put whatever
  the endpoint needs in extra_args, keyed by the parameter names from
  the API dump.
"""
import shutil
import time
from pathlib import Path


def _discover_endpoints(client):
    try:
        api = client.view_api(return_format="dict")
        return api.get("named_endpoints", {}) or {}
    except Exception:
        return {}


def _find_video_file(result, run_dir):
    candidates = []

    def walk(x):
        if isinstance(x, dict):
            p = x.get("path") or x.get("name")
            if p:
                walk(p)
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, str) and x.lower().endswith((".mp4", ".webm", ".mov")):
            candidates.append(x)

    walk(result)
    for c in candidates:
        src = Path(c)
        if src.exists():
            dst = run_dir / "output.mp4"
            shutil.copyfile(src, dst)
            return "output.mp4"
    return None


def run(model, run_dir, sampler, cfg):
    from gradio_client import Client

    url = cfg["paths"]["wangp_url"]
    print(f"  connecting to WanGP at {url} ...")
    client = Client(url)
    endpoint = model.get("endpoint")
    if not endpoint:
        eps = _discover_endpoints(client)
        hits = [n for n in eps
                if any(k in n.lower() for k in
                       ("t2v", "text_to_video", "text2video", "generate_video"))]
        if not hits:
            (run_dir / "wangp_api_dump.txt").write_text(
                "WanGP Gradio endpoints (no T2V match found):\n"
                + "\n".join(sorted(eps)) + "\n\nSet models[].endpoint in "
                "bench_config.json to the right one, and put its arguments "
                "in models[].extra_args keyed by parameter name.",
                encoding="utf-8")
            raise RuntimeError(
                "No T2V endpoint auto-detected; see wangp_api_dump.txt")
        endpoint = hits[0]
        print(f"  auto-detected endpoint: {endpoint}")

    kwargs = {"prompt": model["prompt"]}
    kwargs.update(model.get("extra_args", {}))
    t0 = time.perf_counter()
    result = client.predict(api_name=endpoint, **kwargs)
    wall = time.perf_counter() - t0
    out_file = _find_video_file(result, run_dir)
    (run_dir / "raw_output.txt").write_text(repr(result)[:4000], encoding="utf-8")
    return {
        "primary_label": "s/video",
        "primary_value": round(wall, 2),
        "wall_seconds": round(wall, 2),
        "endpoint": endpoint,
        "output_file": out_file,
    }

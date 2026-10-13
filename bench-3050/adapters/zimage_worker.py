"""Z-Image Turbo worker. Runs INSIDE the wan2gp conda env (diffusers+torch).

Prints one line at the end:  BENCH_JSON:{"gen_seconds": ..., "load_seconds": ...}
The wrapper (zimage.py) parses that line. Everything else on stdout is noise.
"""
import argparse
import json
import time

import torch
from diffusers import ZImagePipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Tongyi-MAI/Z-Image-Turbo")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=9)  # 9 steps = 8 DiT forwards
    ap.add_argument("--guidance", type=float, default=0.0)  # Turbo wants 0.0
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu-offload", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    pipe = ZImagePipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()  # streams weights for 4GB VRAM
    else:
        pipe.to("cuda")
    load_s = time.perf_counter() - t0

    gen = torch.Generator("cuda").manual_seed(args.seed)
    t1 = time.perf_counter()
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=gen,
    ).images[0]
    gen_s = time.perf_counter() - t1
    image.save(args.out)
    print("BENCH_JSON:" + json.dumps({
        "gen_seconds": round(gen_s, 2),
        "load_seconds": round(load_s, 2),
    }), flush=True)


if __name__ == "__main__":
    main()

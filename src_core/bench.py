"""Throughput, latency and memory benchmark of the runtime pipeline on this machine.

    python src_core/bench.py --frames 300 [--source bdd100k-10k/val] [--no-dino] [--no-enhance]

Reports per-stage latency, p50/p95/p99 end-to-end latency, sustained FPS, and process RSS plus GPU memory at the
start and the end (a growing RSS over a sustained run would indicate a leak).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RESULTS, save_json  # noqa: E402


def rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--source", default="bdd100k-10k/val", help="folder of images used as the frame stream")
    p.add_argument("--size", default="1280x720", help="resize frames to WxH before processing (camera-like)")
    p.add_argument("--detector", default=None)
    p.add_argument("--no-dino", action="store_true")
    p.add_argument("--no-enhance", action="store_true")
    p.add_argument("--weather-every", type=int, default=5)
    p.add_argument("--flip-every", type=int, default=2)
    p.add_argument("--out", type=Path, default=RESULTS / "benchmark.json")
    a = p.parse_args(argv)
    import torch
    from pipeline import Pipeline
    w, h = (int(x) for x in a.size.split("x"))
    paths = sorted(glob.glob(str(HERE.parent / a.source / "*.jpg")) + glob.glob(str(HERE.parent / a.source / "*.png")))
    if not paths:
        sys.exit(f"no images under {a.source}")
    frames = [cv2.resize(cv2.imread(p), (w, h)) for p in paths[:64]]
    pipe = Pipeline(a.detector, enhancement=not a.no_enhance, weather_every=a.weather_every, flip_every=a.flip_every, use_dino=not a.no_dino)
    for f in frames[:5]:
        pipe.process(f)  # warm-up
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rss0 = rss_mb()
    lat, stages = [], {}
    t0 = time.perf_counter()
    for i in range(a.frames):
        f = frames[i % len(frames)]
        t = time.perf_counter()
        r = pipe.process(f)
        lat.append((time.perf_counter() - t) * 1000)
        for k, v in r["timings_ms"].items():
            stages.setdefault(k, []).append(v)
    el = time.perf_counter() - t0
    rss1 = rss_mb()
    lat = np.array(lat)
    res = {"frames": a.frames, "resolution": a.size, "detector": pipe.det.name, "enhancement": not a.no_enhance, "dino": pipe.dino is not None,
           "trust_model": pipe.trust is not None, "weather_every": a.weather_every, "flip_every": a.flip_every,
           "fps_sustained": a.frames / el, "latency_ms": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
                                                           "p95": float(np.percentile(lat, 95)), "p99": float(np.percentile(lat, 99)), "max": float(lat.max())},
           "stage_mean_ms": {k: float(np.mean(v)) for k, v in stages.items()},
           "rss_mb_start": rss0, "rss_mb_end": rss1, "rss_growth_mb": rss1 - rss0,
           "gpu_peak_mb": float(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
           "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    save_json(res, a.out)
    print(json.dumps(res, indent=2))
    print(f"[bench] wrote {a.out}")


if __name__ == "__main__":
    main()

"""Run the whole experiment chain in order, resumable (steps whose outputs exist are skipped unless --force).

    python src_core/run_experiments.py                # everything with defaults
    python src_core/run_experiments.py --models yolo26m --skip-synthetic   # a shorter run

Chain:  strata -> detect(off) per model -> compare (choose detector) -> detect(on) + synthetic for the chosen
        detector -> trust train -> analysis + figures -> summary.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RESULTS, save_json  # noqa: E402

PY = sys.executable


def sh(args: list[str]) -> None:
    print(f"\n>>> {' '.join(args)}", flush=True)
    t = time.perf_counter()
    r = subprocess.run([PY, *args], cwd=str(HERE.parent))
    if r.returncode != 0:
        sys.exit(f"[run] step failed with exit code {r.returncode}: {' '.join(args)}")
    print(f"<<< done in {(time.perf_counter() - t) / 60:.1f} min", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--models", nargs="+", default=["yolov8m", "yolo11m", "yolo26m"])
    p.add_argument("--force", action="store_true", help="redo steps whose outputs already exist")
    p.add_argument("--skip-synthetic", action="store_true")
    p.add_argument("--skip-enhancement", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="frames per stratum (smoke run)")
    p.add_argument("--cap-bdd-train", type=int, default=3000)
    p.add_argument("--cap-bdd-val", type=int, default=1500)
    p.add_argument("--cap-cs-scenes", type=int, default=800)
    a = p.parse_args(argv)
    lim = ["--limit", str(a.limit)] if a.limit else []
    t0 = time.perf_counter()
    if a.force or not (RESULTS / "strata.json").exists():
        sh(["src_core/sweep.py", "strata", "--cap-bdd-train", str(a.cap_bdd_train), "--cap-bdd-val", str(a.cap_bdd_val), "--cap-cs-scenes", str(a.cap_cs_scenes)])
    for m in a.models:
        if a.force or not (RESULTS / f"metrics_{m}_off.csv").exists():
            sh(["src_core/sweep.py", "detect", "--model", m, "--enhancement", "off", *lim])
    sh(["src_core/sweep.py", "compare", "--models", *a.models])
    chosen = json.load(open(RESULTS / "detector_choice.json"))["detector"]
    print(f"[run] detector chosen by the adverse-weather rule: {chosen}")
    if not a.skip_synthetic and (a.force or not (RESULTS / f"metrics_{chosen}_off.csv").exists() or
                                 "syn_fog_1" not in open(RESULTS / f"metrics_{chosen}_off.csv").read()):
        sh(["src_core/sweep.py", "detect", "--model", chosen, "--enhancement", "off", "--synthetic", *lim])
    if not a.skip_enhancement and (a.force or not (RESULTS / f"metrics_{chosen}_on.csv").exists()):
        sh(["src_core/sweep.py", "detect", "--model", chosen, "--enhancement", "on", *lim])
    rec_off = RESULTS / "records" / f"{chosen}_off.jsonl"
    rec_on = RESULTS / "records" / f"{chosen}_on.jsonl"
    sh(["src_core/trust.py", "train", "--records", str(rec_off)])
    sh(["src_core/analysis.py", "--records", str(rec_off)] + (["--records-on", str(rec_on)] if rec_on.exists() else []))
    save_json({"detector": chosen, "models_compared": a.models, "records_off": str(rec_off), "records_on": str(rec_on) if rec_on.exists() else None,
               "minutes": round((time.perf_counter() - t0) / 60, 1), "limit": a.limit}, RESULTS / "summary.json")
    print(f"[run] complete in {(time.perf_counter() - t0) / 60:.1f} min -> {RESULTS}")


if __name__ == "__main__":
    main()

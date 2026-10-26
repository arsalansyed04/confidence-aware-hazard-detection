r"""Runtime application. Two modes:

    python src_core/app.py video  --input path\to\clip.mp4 [--output out.mp4] [--csv out.csv] [--show]
    python src_core/app.py camera [--device 0] [--output out.mp4] [--csv out.csv]

Each frame gets: weather + fog level, adaptive enhancement (ON/OFF), frozen YOLO detections, trust score,
danger score and the trust x danger quadrant, drawn as an overlay. Press q (or Esc) in the window to stop.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RESULTS, append_jsonl, start_log  # noqa: E402

COLORS = {"person": (0, 0, 255), "bike": (0, 128, 255), "motor": (0, 128, 255), "car": (0, 200, 0), "truck": (200, 160, 0),
          "bus": (200, 160, 0), "traffic light": (200, 200, 200)}


def parse_args(argv=None):
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--detector", default=None, help="yolov8m | yolo11m | yolo26m | path (default: results/core/detector_choice.json or yolo26m)")
    c.add_argument("--trust-model", type=Path, default=None)
    c.add_argument("--no-enhance", action="store_true", help="enhancement OFF")
    c.add_argument("--conf", type=float, default=0.25)
    c.add_argument("--imgsz", type=int, default=640)
    c.add_argument("--weather-every", type=int, default=5, help="run the weather model every N frames")
    c.add_argument("--flip-every", type=int, default=2, help="run the mirrored pass every N frames")
    c.add_argument("--no-dino", action="store_true", help="skip the DINOv2 distance feature (faster)")
    c.add_argument("--device", default="cuda")
    c.add_argument("--output", type=Path, default=None, help="annotated video (mp4)")
    c.add_argument("--csv", type=Path, default=None, help="per-frame CSV")
    c.add_argument("--jsonl", type=Path, default=None, help="per-frame JSON lines (detections included)")
    c.add_argument("--show", action="store_true", help="open a window (camera mode always shows)")
    c.add_argument("--max-frames", type=int, default=None)
    c.add_argument("--stop-file", type=Path, default=None, help="stop gracefully as soon as this file exists (used by the GUI)")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    v = sub.add_parser("video", parents=[c])
    v.add_argument("--input", type=Path, required=True)
    v.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    w = sub.add_parser("camera", parents=[c])
    w.add_argument("--device-index", type=int, default=0, dest="camera")
    w.add_argument("--width", type=int, default=1280)
    w.add_argument("--height", type=int, default=720)
    w.add_argument("--no-show", action="store_true", help="headless (tests)")
    return p.parse_args(argv)


def draw_overlay(frame: np.ndarray, r: dict, fps: float | None = None) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    for d in r["detections"]:
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        col = COLORS.get(d["cls"], (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{d['cls']} {d['conf']:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    trust, dz = r["trust"], r["danger"]
    panel_h = 118
    cv2.rectangle(out, (0, 0), (min(w, 430), panel_h), (0, 0, 0), -1)
    weather = r["weather"] + (f" ({r['fog_level']})" if r["weather"] == "fog" else "")
    lines = [f"weather: {weather}   p={max(r['p_weather'].values()):.2f}",
             f"enhancement: {'ON' if r['enhancement'] else 'off'}   objects: {len(r['detections'])}",
             f"trust: {trust:.2f}" if trust == trust else "trust: n/a (no trust model)",
             f"danger: {dz:.2f}   {r['quadrant']}"]
    if fps is not None:
        lines.append(f"{fps:.1f} fps")
    for i, s in enumerate(lines):
        cv2.putText(out, s, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    # bars
    def bar(y, value, label, good_high):
        v = 0.0 if value != value else float(np.clip(value, 0, 1))
        x0, x1 = 440, min(w - 10, 640)
        cv2.rectangle(out, (x0, y), (x1, y + 14), (60, 60, 60), -1)
        g = int(255 * (v if good_high else 1 - v))
        cv2.rectangle(out, (x0, y), (x0 + int((x1 - x0) * v), y + 14), (0, g, 255 - g), -1)
        cv2.putText(out, label, (x0, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    if w > 660:
        bar(22, trust, "trust", True)
        bar(62, dz, "danger", False)
    if trust == trust and trust < 0.5 and dz < 0.4:
        cv2.putText(out, "LOW TRUST / LOW APPARENT DANGER: scene may be unsafe", (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return out


CSV_COLS = ["frame", "time_s", "weather", "fog_level", "p_clear", "p_fog", "p_night", "p_rain", "p_snow", "enhancement", "n_detections",
            "n_person", "trust", "danger", "quadrant", "flip_dice", "ms_weather", "ms_enhance", "ms_detect", "ms_trust_danger"]


def csv_row(i: int, t: float, r: dict) -> list:
    p = r["p_weather"]
    return [i, round(t, 3), r["weather"], r["fog_level"], *[round(p.get(c, 0.0), 4) for c in ("clear", "fog", "night", "rain", "snow")],
            int(bool(r["enhancement"])), len(r["detections"]), sum(1 for d in r["detections"] if d["cls"] == "person"),
            round(r["trust"], 4) if r["trust"] == r["trust"] else "", round(r["danger"], 4), r["quadrant"], round(r["consistency"]["f_dice"], 3),
            *[round(r["timings_ms"].get(k, 0.0), 1) for k in ("weather", "enhance", "detect", "trust_danger")]]


def run(a, cap: cv2.VideoCapture, label: str, show: bool, stride: int = 1, fps_in: float | None = None):
    from pipeline import Pipeline
    pipe = Pipeline(a.detector, a.trust_model, not a.no_enhance, a.device, a.imgsz, a.conf, a.weather_every, a.flip_every, not a.no_dino)
    print(f"[app] {label} | detector {pipe.det.name} | trust model {'loaded' if pipe.trust else 'MISSING (trust shown as n/a)'} | "
          f"enhancement {'ON' if not a.no_enhance else 'off'} | dino {pipe.dino is not None}")
    writer = csv_f = csv_w = None
    writer_size = None
    if a.csv:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_f = open(a.csv, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(CSV_COLS)
    if a.jsonl:
        a.jsonl.parent.mkdir(parents=True, exist_ok=True)
        open(a.jsonl, "w").close()
    n = done = 0
    t_start = time.perf_counter()
    last = t_start
    fps_est = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            n += 1
            if (n - 1) % stride:
                continue
            r = pipe.process(frame)
            now = time.perf_counter()
            dt = now - last
            last = now
            fps_est = (1 / dt) if fps_est is None else 0.9 * fps_est + 0.1 / max(dt, 1e-6)
            t_s = (n - 1) / fps_in if fps_in else now - t_start
            vis = draw_overlay(frame, r, fps_est)
            if writer is None and a.output:
                a.output.parent.mkdir(parents=True, exist_ok=True)
                writer_size = (vis.shape[1], vis.shape[0])
                writer = cv2.VideoWriter(str(a.output), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, (fps_in or fps_est or 10) / stride), writer_size)
                if not writer.isOpened():
                    raise RuntimeError(f"cannot open video writer for {a.output}")
            if writer is not None:
                if (vis.shape[1], vis.shape[0]) != writer_size:  # a camera can change resolution mid-stream
                    vis = cv2.resize(vis, writer_size)
                writer.write(vis)
            if csv_w is not None:
                csv_w.writerow(csv_row(n - 1, t_s, r))
            if a.jsonl:
                append_jsonl(a.jsonl, {"frame": n - 1, "time_s": t_s, "weather": r["weather"], "fog_level": r["fog_level"], "p_weather": r["p_weather"],
                                       "detections": r["detections"], "trust": r["trust"], "danger": r["danger"], "quadrant": r["quadrant"],
                                       "meta": r["meta"], "timings_ms": r["timings_ms"]})
            done += 1
            if show:
                cv2.imshow("weather-adaptive hazard detection", vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    print("[app] stopped by user")
                    break
            if done % 50 == 0:
                print(f"[app] {done} frames | {fps_est:.1f} fps | weather {r['weather']} | trust {r['trust']:.2f} | danger {r['danger']:.2f}")
            if a.max_frames and done >= a.max_frames:
                break
            if a.stop_file is not None and a.stop_file.exists():
                print("[app] stop requested")
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_f is not None:
            csv_f.close()
        if show:
            cv2.destroyAllWindows()
    el = time.perf_counter() - t_start
    print(f"[app] processed {done} frames in {el:.1f}s ({done / max(el, 1e-6):.1f} fps)"
          + (f" | video -> {a.output}" if a.output else "") + (f" | csv -> {a.csv}" if a.csv else ""))
    return done


def main(argv=None):
    a = parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    start_log(RESULTS / "log_app.txt")
    if a.mode == "video":
        if not a.input.exists():
            sys.exit(f"[app] input video not found: {a.input}")
        cap = cv2.VideoCapture(str(a.input))
        if not cap.isOpened():
            sys.exit(f"[app] cannot open video: {a.input}")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or None
        print(f"[app] video {a.input} | {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} | "
              f"{fps_in or '?'} fps | {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames | stride {a.stride}")
        run(a, cap, f"video {a.input.name}", a.show, a.stride, fps_in)
    else:
        cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
        if not cap.isOpened():
            sys.exit(f"[app] cannot open camera {a.camera}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        print(f"[app] camera {a.camera} | {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} | press q to stop")
        run(a, cap, f"camera {a.camera}", not a.no_show)


if __name__ == "__main__":
    main()

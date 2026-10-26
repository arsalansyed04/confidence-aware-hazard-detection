"""The runtime app: overlay, CSV, video and camera modes. Fast tests use a fake pipeline; --run-slow uses the real models."""
import csv
import json
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import app as A
import pipeline as P


class FakePipeline:
    """Same output contract as pipeline.Pipeline, no models."""

    def __init__(self, *args, **kwargs):
        self.det = SimpleNamespace(name="fake")
        self.trust = object()
        self.dino = None
        self.n = 0

    def process(self, frame):
        self.n += 1
        h, w = frame.shape[:2]
        dets = [{"cls": "person", "conf": 0.9, "xyxy": [w * 0.1, h * 0.3, w * 0.2, h * 0.9]}] if self.n % 2 else []
        return {"frame_index": self.n - 1, "weather": "fog", "fog_level": "dense", "p_weather": {"clear": 0.1, "fog": 0.9, "night": 0, "rain": 0, "snow": 0},
                "detections": dets, "consistency": {"f_dice": 1.0}, "meta": {}, "trust": 0.3, "danger": 0.2,
                "danger_components": {}, "quadrant": "low trust, low apparent danger", "enhanced": frame, "enhancement": True,
                "timings_ms": {"weather": 1.0, "enhance": 1.0, "detect": 2.0, "trust_danger": 0.5}}


def make_video(path, n=12, size=(320, 240)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    assert w.isOpened()
    for i in range(n):
        f = np.full((size[1], size[0], 3), (40 + 5 * i) % 256, np.uint8)
        w.write(f)
    w.release()


def test_overlay_draws_and_keeps_input(street_image):
    r = FakePipeline().process(street_image)
    vis = A.draw_overlay(street_image, r, 12.5)
    assert vis.shape == street_image.shape and not np.array_equal(vis, street_image)
    r2 = dict(r, trust=float("nan"), quadrant="no trust model", detections=[])
    assert A.draw_overlay(np.zeros((100, 100, 3), np.uint8), r2).shape == (100, 100, 3)  # tiny frame, no bars


def test_csv_row_matches_columns(street_image):
    r = FakePipeline().process(street_image)
    row = A.csv_row(3, 0.3, r)
    assert len(row) == len(A.CSV_COLS) and row[2] == "fog" and row[3] == "dense" and row[A.CSV_COLS.index("n_person")] == 1
    r["trust"] = float("nan")
    assert A.csv_row(0, 0.0, r)[A.CSV_COLS.index("trust")] == ""


def test_video_mode_end_to_end_with_fake_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "Pipeline", FakePipeline)
    monkeypatch.setattr(A, "RESULTS", tmp_path)
    src = tmp_path / "in.mp4"
    make_video(src, 12)
    out, csvp, jl = tmp_path / "out.mp4", tmp_path / "out.csv", tmp_path / "out.jsonl"
    A.main(["video", "--input", str(src), "--output", str(out), "--csv", str(csvp), "--jsonl", str(jl), "--stride", "2"])
    assert out.exists() and out.stat().st_size > 0
    rows = list(csv.DictReader(open(csvp)))
    assert len(rows) == 6 and rows[0]["weather"] == "fog" and [int(r["frame"]) for r in rows] == [0, 2, 4, 6, 8, 10]
    lines = [json.loads(l) for l in open(jl)]
    assert len(lines) == 6 and "detections" in lines[0] and lines[1]["detections"] == []
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    cap.release()


def test_video_mode_rejects_missing_input(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "RESULTS", tmp_path)
    with pytest.raises(SystemExit):
        A.main(["video", "--input", str(tmp_path / "nope.mp4")])


def test_camera_mode_headless_with_fake_capture(tmp_path, monkeypatch):
    class FakeCap:
        def __init__(self, *a, **k):
            self.i = 0

        def isOpened(self):
            return True

        def set(self, *a):
            return True

        def get(self, prop):
            return 320.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else 240.0

        def read(self):
            self.i += 1
            return (True, np.zeros((240, 320, 3), np.uint8)) if self.i <= 7 else (False, None)

        def release(self):
            pass

    monkeypatch.setattr(P, "Pipeline", FakePipeline)
    monkeypatch.setattr(A, "RESULTS", tmp_path)
    monkeypatch.setattr(A.cv2, "VideoCapture", FakeCap)
    csvp = tmp_path / "cam.csv"
    A.main(["camera", "--no-show", "--csv", str(csvp), "--max-frames", "5"])
    assert len(list(csv.DictReader(open(csvp)))) == 5


@pytest.mark.slow
def test_real_pipeline_on_real_frames():
    """Loads v4, YOLO and (if trained) the trust model; needs the GPU and the datasets."""
    import glob
    paths = sorted(glob.glob("rgb_anon_trainvaltest_ACDC/rgb_anon/night/val/*/*.png"))[:3]
    if not paths:
        pytest.skip("ACDC not on this machine")
    pipe = P.Pipeline(enhancement=True, weather_every=1, flip_every=1)
    for p in paths:
        r = pipe.process(cv2.imread(p))
        assert r["weather"] in ("clear", "fog", "night", "rain", "snow") and 0 <= r["danger"] <= 1
        if pipe.trust is not None:
            assert 0 <= r["trust"] <= 1 and not math.isnan(r["trust"])
        assert all(d["conf"] >= 0.25 for d in r["detections"])

"""Runtime pipeline: one frame in -> weather, enhancement, detections, trust, danger, quadrant out.

The same code path the experiments used, so runtime numbers match the evaluated ones:
  raw frame -> v4 weather (+ fog level) -> enhancement (ON/OFF) -> frozen YOLO -> flip-consistency
            -> trust features (quality on the RAW frame) -> XGBoost trust -> danger from detections -> quadrant
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from common import DEFAULT_CONF, RESULTS, filter_boxes
import danger as DG
import features as F
from enhance import enhance


class Pipeline:
    def __init__(self, detector: str | None = None, trust_model: Path | None = None, enhancement: bool = True,
                 device: str = "cuda", imgsz: int = 640, conf: float = DEFAULT_CONF, weather_every: int = 5,
                 flip_every: int = 1, use_dino: bool = True, tta: bool = False):
        from detect import Detector
        from weather import WeatherModel
        from trust import TrustModel
        if detector is None:
            choice = RESULTS / "detector_choice.json"
            detector = __import__("json").load(open(choice))["detector"] if choice.exists() else "yolo26m"
        self.det = Detector(detector, device, imgsz)
        self.weather = WeatherModel(device=device, tta=tta)
        tm = Path(trust_model) if trust_model else RESULTS / "trust" / "trust_model.json"
        self.trust = TrustModel.load(tm) if tm.exists() else None
        self.knn = None
        self.dino = None
        if use_dino and self.trust is not None and any(f in self.trust.features for f in F.DINO_FEATURES):
            ref = RESULTS / "knn_reference_bdd_train_clear.npy"
            if ref.exists():
                self.dino = F.DinoEmbedder(device)
                self.knn = F.KnnReference(np.load(ref), k=5)
        self.enhancement, self.conf = enhancement, conf
        self.weather_every, self.flip_every = max(1, weather_every), max(1, flip_every)
        self._n = 0
        self._last_weather = None
        self._last_flip = None
        self.timings = {}

    def process(self, frame_bgr: np.ndarray) -> dict:
        t = {}
        t0 = time.perf_counter()
        hw = frame_bgr.shape[:2]
        if self._last_weather is None or self._n % self.weather_every == 0:
            self._last_weather = self.weather.predict([frame_bgr])[0]
        wx = self._last_weather
        t["weather"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        enh = enhance(frame_bgr, wx["weather"], wx["fog_level"], self.enhancement)
        t["enhance"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        if self._n % self.flip_every == 0:
            dets, flips = self.det.detect_with_flip([enh])
            dets, flip = dets[0], flips[0]
            self._last_flip = F.flip_consistency(dets, flip)
        else:
            dets = self.det.detect([enh])[0]
        consistency = self._last_flip or F.flip_consistency(dets, dets)
        t["detect"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        meta = {}
        meta.update(F.quality_features(frame_bgr))
        meta.update(F.conf_features(dets, self.conf, hw))
        meta.update(consistency)
        meta.update(F.weather_features(wx["p_weather"], wx["fog_level"]))
        if self.dino is not None:
            meta.update(self.knn.features(self.dino.embed([frame_bgr]))[0])
        trust = float(self.trust.trust([meta])[0]) if self.trust is not None else float("nan")
        dz = DG.danger(dets, hw, wx["weather"], wx["fog_level"], self.conf)
        t["trust_danger"] = time.perf_counter() - t3
        self._n += 1
        self.timings = t
        return {"frame_index": self._n - 1, "weather": wx["weather"], "fog_level": wx["fog_level"], "p_weather": wx["p_weather"],
                "detections": filter_boxes(dets, self.conf), "consistency": consistency, "meta": meta, "trust": trust,
                "danger": dz["danger"], "danger_components": dz, "quadrant": DG.quadrant(trust, dz["danger"]) if trust == trust else "no trust model",
                "enhanced": enh, "enhancement": self.enhancement and wx["weather"] != "clear", "timings_ms": {k: v * 1000 for k, v in t.items()}}

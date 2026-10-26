"""Shared definitions: paths, class vocabulary, box matching, failure labels, per-frame record I/O."""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RESULTS = PROJECT_ROOT / "results" / "core"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# --------------------------------------------------------------------------- class vocabulary
PROJECT_CLASSES = ["person", "car", "truck", "bus", "bike", "motor", "traffic light"]
COCO_TO_PROJECT = {0: "person", 1: "bike", 2: "car", 3: "motor", 5: "bus", 7: "truck", 9: "traffic light"}
BDD_TO_PROJECT = {"person": "person", "rider": "person", "car": "car", "truck": "truck", "bus": "bus",
                  "bike": "bike", "motor": "motor", "traffic light": "traffic light"}  # train, traffic sign, lanes ignored
CITYSCAPES_ID_TO_PROJECT = {24: "person", 25: "person", 26: "car", 27: "truck", 28: "bus", 32: "motor", 33: "bike"}
ACDC_NAME_TO_PROJECT = {"person": "person", "rider": "person", "car": "car", "truck": "truck", "bus": "bus",
                        "motorcycle": "motor", "bicycle": "bike"}  # train ignored
WEATHER_CONDITIONS = ["clear", "fog", "night", "rain", "snow"]
VULNERABLE = {"person", "bike", "motor"}

DEFAULT_CONF = 0.25      # operating threshold for metrics and the runtime
RECORD_CONF = 0.05       # detections stored in records (lower, so thresholds can be re-applied later)
IOU_MATCH = 0.5
MIN_BOX_PX = 12          # GT and detections shorter than this (height) are ignored for failure labels


# --------------------------------------------------------------------------- boxes
def box_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every row of a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)).astype(np.float32)


def filter_boxes(boxes: list, conf: float | None = None, min_px: float = 0.0) -> list:
    out = []
    for b in boxes:
        if conf is not None and b.get("conf", 1.0) < conf:
            continue
        if (b["xyxy"][3] - b["xyxy"][1]) < min_px:
            continue
        out.append(b)
    return out


def greedy_match(dets: list, gts: list, iou_thr: float = IOU_MATCH, class_aware: bool = True) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by descending IoU. dets sorted by confidence first. Returns (det_i, gt_j, iou)."""
    if not dets or not gts:
        return []
    order = sorted(range(len(dets)), key=lambda i: -dets[i].get("conf", 1.0))
    ious = iou_matrix(np.array([dets[i]["xyxy"] for i in order]), np.array([g["xyxy"] for g in gts]))
    if class_aware:
        same = np.array([[dets[i]["cls"] == g["cls"] for g in gts] for i in order])
        ious = np.where(same, ious, 0.0)
    used = set()
    pairs = []
    for r, i in enumerate(order):
        j = int(np.argmax(ious[r])) if ious.shape[1] else -1
        if j < 0 or ious[r, j] < iou_thr:
            continue
        # pick the best unused gt for this det
        cands = np.argsort(-ious[r])
        for j in cands:
            if ious[r, j] < iou_thr:
                break
            if int(j) not in used:
                used.add(int(j))
                pairs.append((i, int(j), float(ious[r, j])))
                break
    return pairs


def frame_metrics(dets: list, gts: list, conf: float = DEFAULT_CONF, min_px: float = MIN_BOX_PX) -> dict:
    """Per-frame TP/FP/FN, precision, recall, F1, missed persons and the failure labels."""
    d = filter_boxes(dets, conf, min_px)
    g = filter_boxes(gts, None, min_px)
    pairs = greedy_match(d, g)
    tp = len(pairs)
    fp, fn = len(d) - tp, len(g) - tp
    prec = tp / max(len(d), 1)
    rec = tp / max(len(g), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9) if (len(d) or len(g)) else 1.0
    matched_gt = {j for _, j, _ in pairs}
    n_person = sum(1 for x in g if x["cls"] == "person")
    missed_person = sum(1 for j, x in enumerate(g) if x["cls"] == "person" and j not in matched_gt)
    missed_vulnerable = sum(1 for j, x in enumerate(g) if x["cls"] in VULNERABLE and j not in matched_gt)
    return {"n_det": len(d), "n_gt": len(g), "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "n_gt_person": n_person, "missed_person": missed_person, "missed_vulnerable": missed_vulnerable,
            "fail_f1": bool(len(g) > 0 and f1 < 0.5), "fail_person": bool(missed_person > 0),
            "fail": bool((len(g) > 0 and f1 < 0.5) or missed_person > 0)}


def average_precision(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """VOC-style AP (all-points interpolation) from per-detection scores and TP flags."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(~is_tp[order])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def stratum_ap50(frames: list[tuple[list, list]], min_px: float = MIN_BOX_PX, classes=PROJECT_CLASSES) -> dict:
    """mAP50 over a stratum. frames = [(dets, gts), ...]. Per-class AP, then the mean over classes with GT."""
    per_class = {}
    for c in classes:
        scores, flags, n_gt = [], [], 0
        for dets, gts in frames:
            d = [x for x in filter_boxes(dets, RECORD_CONF, min_px) if x["cls"] == c]
            g = [x for x in filter_boxes(gts, None, min_px) if x["cls"] == c]
            n_gt += len(g)
            pairs = greedy_match(d, g, class_aware=False)
            matched = {i for i, _, _ in pairs}
            for i, x in enumerate(d):
                scores.append(x["conf"])
                flags.append(i in matched)
        ap = average_precision(np.array(scores, float), np.array(flags, bool), n_gt)
        if n_gt > 0:
            per_class[c] = ap
    return {"map50": float(np.mean(list(per_class.values()))) if per_class else float("nan"), "per_class": per_class}


# --------------------------------------------------------------------------- records
@dataclass
class FrameGT:
    frame_id: str
    path: str
    source: str          # bdd | acdc | cityscapes
    split: str
    weather_gt: str      # clear | fog | night | rain | snow   (dataset label, never a prediction)
    timeofday_gt: str
    stratum: str
    boxes: list = field(default_factory=list)      # [{cls, xyxy}]
    extra: dict = field(default_factory=dict)      # scene id, fog beta, ...

    def to_dict(self) -> dict:
        return asdict(self)


def record_template(fg: FrameGT, detector: str, enhancement: str, corruption: dict | None = None) -> dict:
    """The frozen per-frame record (handoff section 7) plus detector / fog-level fields."""
    return {"frame_id": fg.frame_id, "path": fg.path, "source": fg.source, "split": fg.split, "stratum": fg.stratum,
            "weather_gt": fg.weather_gt, "timeofday_gt": fg.timeofday_gt, "corruption": corruption,
            "enhancement": enhancement, "detector": detector, "detections": [], "gt_boxes": fg.boxes,
            "consistency": {}, "meta": {}, "scores": {"trust": None, "danger_est": None, "danger_true": None},
            "weather_pred": None, "fog_level_pred": None, "image_hw": None, "extra": fg.extra}


def write_jsonl(path: Path, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


# --------------------------------------------------------------------------- logging
class _Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_log(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", encoding="utf-8")
    fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} | {' '.join(sys.argv)}\n")
    sys.stdout = _Tee(sys.stdout, fh)
    print(f"[log] console output is also appended to {path}")


def imread(path) -> np.ndarray | None:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None and os.path.exists(path):
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

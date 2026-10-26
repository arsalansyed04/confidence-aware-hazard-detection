"""Danger score: a normalised, weighted, hand-set heuristic computed TWICE per frame, on detections (R_est) and on
ground-truth annotations (R_true). Only the gap R_true - R_est is claimed to be meaningful: the weights largely
cancel, and what is measured is the distortion that perception introduces."""
from __future__ import annotations

import numpy as np

from common import DEFAULT_CONF, filter_boxes

CLASS_WEIGHTS = {"person": 1.0, "bike": 0.9, "motor": 0.9, "car": 0.5, "truck": 0.6, "bus": 0.6, "traffic light": 0.1}
WEATHER_SEVERITY = {"clear": 0.0, "rain": 0.4, "snow": 0.5, "night": 0.5, "fog": 0.7}
FOG_LEVEL_SEVERITY = {"light": 0.5, "medium": 0.7, "dense": 0.9, "unrated": 0.7, "none": 0.7}
OBJECT_SCALE = 3.0       # saturation rate of the object term: 1 - exp(-k * sum)
W_OBJECT, W_WEATHER = 0.7, 0.3


def object_term(box: list, hw: tuple, class_weights: dict = CLASS_WEIGHTS) -> float:
    """Relevance of one box: larger, lower in the frame and more central = closer and more relevant."""
    h, w = hw
    x1, y1, x2, y2 = box["xyxy"]
    size = float(np.sqrt(max(0.0, (x2 - x1) * (y2 - y1)) / max(h * w, 1)))
    lower = float(np.clip(y2 / max(h, 1), 0, 1))
    central = 1.0 - float(min(1.0, abs((x1 + x2) / 2 / max(w, 1) - 0.5) * 2))
    return class_weights.get(box["cls"], 0.0) * np.sqrt(size) * (0.4 + 0.6 * lower) * (0.5 + 0.5 * central)


def severity(weather: str, fog_level: str = "none") -> float:
    s = WEATHER_SEVERITY.get(weather, 0.0)
    return FOG_LEVEL_SEVERITY.get(fog_level, s) if weather == "fog" else s


def danger(boxes: list, hw: tuple, weather: str, fog_level: str = "none", conf: float | None = DEFAULT_CONF,
           class_weights: dict = CLASS_WEIGHTS, k: float = OBJECT_SCALE) -> dict:
    """Danger in [0,1] with its components. Pass conf=None for ground-truth boxes."""
    b = filter_boxes(boxes, conf) if conf is not None else boxes
    s = float(sum(object_term(x, hw, class_weights) for x in b))
    d_obj = 1.0 - float(np.exp(-k * s))
    sev = severity(weather, fog_level)
    return {"danger": float(np.clip(W_OBJECT * d_obj + W_WEATHER * sev, 0.0, 1.0)), "object_term": d_obj, "object_sum": s,
            "weather_term": sev, "n_objects": len(b)}


def gap(dets: list, gts: list, hw: tuple, weather_gt: str, fog_level: str = "none", conf: float = DEFAULT_CONF) -> dict:
    """R_true - R_est with the SAME weather term on both sides, so the gap measures detection distortion only."""
    est = danger(dets, hw, weather_gt, fog_level, conf)
    true = danger(gts, hw, weather_gt, fog_level, None)
    return {"danger_est": est["danger"], "danger_true": true["danger"], "gap": true["danger"] - est["danger"],
            "object_est": est["object_term"], "object_true": true["object_term"]}


def quadrant(trust: float, danger_value: float, trust_thr: float = 0.5, danger_thr: float = 0.4) -> str:
    """The two-axis readout. 'low trust, low apparent danger' is the most dangerous corner: the scene looks safe
    precisely because perception may be failing."""
    t = "high trust" if trust >= trust_thr else "low trust"
    d = "high danger" if danger_value >= danger_thr else "low apparent danger"
    return f"{t}, {d}"


def weight_sensitivity(frames: list[dict], perturb: float = 0.5) -> dict:
    """Perturb each class weight by +-perturb and report how stable the high/medium/low ranking of frames is.
    frames: [{boxes, hw, weather, fog_level}] evaluated with the ground-truth boxes."""
    from scipy.stats import spearmanr

    def scores(weights):
        return np.array([danger(f["boxes"], f["hw"], f["weather"], f.get("fog_level", "none"), None, weights)["danger"] for f in frames])

    base = scores(CLASS_WEIGHTS)
    terc = lambda s: np.digitize(s, np.quantile(s, [1 / 3, 2 / 3]))  # noqa: E731
    base_t = terc(base)
    out = {}
    for c in CLASS_WEIGHTS:
        for sign in (-1, 1):
            w = dict(CLASS_WEIGHTS)
            w[c] = w[c] * (1 + sign * perturb)
            s = scores(w)
            rho = float(spearmanr(base, s).correlation) if len(s) > 2 else float("nan")
            out[f"{c}{'+' if sign > 0 else '-'}{int(perturb * 100)}%"] = {"spearman": rho, "tercile_agreement": float((terc(s) == base_t).mean())}
    return out

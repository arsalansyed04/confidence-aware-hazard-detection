"""Trust meta-features: (a) confidence statistics, (b) flip-consistency, (c) feature-space distance (DINOv2 kNN),
(d) RAW-frame image quality computed BEFORE enhancement, (e) the weather model's probabilities (runtime signal)."""
from __future__ import annotations

import math

import cv2
import numpy as np

from common import DEFAULT_CONF, PROJECT_CLASSES, greedy_match, filter_boxes

QUALITY_FEATURES = ["q_laplacian_var", "q_rms_contrast", "q_mean_brightness", "q_dark_channel", "q_saturation", "q_edge_density"]
CONF_FEATURES = ["c_n_det", "c_n_person", "c_mean", "c_max", "c_min", "c_std", "c_frac_low", "c_frac_high", "c_mean_area",
                 "c_class_entropy", "c_n_small"]
FLIP_FEATURES = ["f_dice", "f_conf_agreement", "f_count_diff", "f_person_dice"]
DINO_FEATURES = ["d_knn_dist", "d_knn_dist_min", "d_norm_to_mean"]
WEATHER_FEATURES = ["w_p_clear", "w_p_fog", "w_p_night", "w_p_rain", "w_p_snow", "w_fog_level"]
FAMILIES = {"quality": QUALITY_FEATURES, "confidence": CONF_FEATURES, "flip": FLIP_FEATURES, "dino": DINO_FEATURES,
            "weather": WEATHER_FEATURES}
ALL_FEATURES = [f for fam in FAMILIES.values() for f in fam]
FOG_LEVEL_ORD = {"none": 0.0, "unrated": 1.0, "light": 1.0, "medium": 2.0, "dense": 3.0}


# --------------------------------------------------------------------------- (d) raw-frame quality
def quality_features(bgr_raw: np.ndarray) -> dict:
    """No-reference image quality on the RAW frame (never the enhanced one): blur, contrast, exposure, haze."""
    h, w = bgr_raw.shape[:2]
    scale = 640.0 / max(h, w)
    small = cv2.resize(bgr_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr_raw
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    dark = cv2.erode(small.min(axis=2), cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 50, 150)
    return {"q_laplacian_var": float(lap), "q_rms_contrast": float(g.std()), "q_mean_brightness": float(g.mean()),
            "q_dark_channel": float(dark.mean() / 255.0), "q_saturation": float(hsv[:, :, 1].mean() / 255.0),
            "q_edge_density": float((edges > 0).mean())}


# --------------------------------------------------------------------------- (a) confidence statistics
def conf_features(dets: list, conf: float = DEFAULT_CONF, image_hw: tuple | None = None) -> dict:
    d = filter_boxes(dets, conf)
    n = len(d)
    if n == 0:
        return {"c_n_det": 0, "c_n_person": 0, "c_mean": 0.0, "c_max": 0.0, "c_min": 0.0, "c_std": 0.0, "c_frac_low": 0.0,
                "c_frac_high": 0.0, "c_mean_area": 0.0, "c_class_entropy": 0.0, "c_n_small": 0}
    c = np.array([b["conf"] for b in d], float)
    area = 1.0
    if image_hw:
        area = float(image_hw[0] * image_hw[1])
    areas = np.array([(b["xyxy"][2] - b["xyxy"][0]) * (b["xyxy"][3] - b["xyxy"][1]) for b in d]) / area
    heights = np.array([b["xyxy"][3] - b["xyxy"][1] for b in d])
    counts = np.array([sum(1 for b in d if b["cls"] == k) for k in PROJECT_CLASSES], float)
    p = counts / counts.sum()
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {"c_n_det": n, "c_n_person": int(sum(1 for b in d if b["cls"] == "person")), "c_mean": float(c.mean()),
            "c_max": float(c.max()), "c_min": float(c.min()), "c_std": float(c.std()), "c_frac_low": float((c < 0.5).mean()),
            "c_frac_high": float((c >= 0.8).mean()), "c_mean_area": float(areas.mean()), "c_class_entropy": ent,
            "c_n_small": int((heights < 32).sum())}


# --------------------------------------------------------------------------- (b) flip-consistency
def unflip_boxes(dets: list, width: int) -> list:
    out = []
    for b in dets:
        x1, y1, x2, y2 = b["xyxy"]
        out.append(dict(b, xyxy=[width - x2, y1, width - x1, y2]))
    return out


def flip_consistency(dets: list, dets_flip_unflipped: list, conf: float = DEFAULT_CONF) -> dict:
    """Stability of the detector under a left-right mirror: Dice overlap of the two detection sets."""
    a = filter_boxes(dets, conf)
    b = filter_boxes(dets_flip_unflipped, conf)
    if not a and not b:
        return {"f_dice": 1.0, "f_conf_agreement": 1.0, "f_count_diff": 0, "f_person_dice": 1.0}
    pairs = greedy_match(a, b)
    dice = 2 * len(pairs) / max(len(a) + len(b), 1)
    agree = 1.0 - float(np.mean([abs(a[i]["conf"] - b[j]["conf"]) for i, j, _ in pairs])) if pairs else 0.0
    pa = [x for x in a if x["cls"] == "person"]
    pb = [x for x in b if x["cls"] == "person"]
    pdice = 1.0 if not pa and not pb else 2 * len(greedy_match(pa, pb)) / max(len(pa) + len(pb), 1)
    return {"f_dice": float(dice), "f_conf_agreement": float(agree), "f_count_diff": int(abs(len(a) - len(b))), "f_person_dice": float(pdice)}


# --------------------------------------------------------------------------- (c) feature-space distance
class DinoEmbedder:
    """DINOv2 ViT-S/14 image embedding (class token + mean patch token), L2-normalised. Reuses the cached timm weights."""

    def __init__(self, device="cuda", img_size: int = 224):
        import timm
        import torch
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=img_size)
        self.model.to(self.device).eval()
        self.size = img_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _prep(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        s = self.size / min(h, w)
        img = cv2.resize(bgr, (max(self.size, round(w * s)), max(self.size, round(h * s))), interpolation=cv2.INTER_AREA)
        y, x = (img.shape[0] - self.size) // 2, (img.shape[1] - self.size) // 2
        return np.ascontiguousarray(img[y:y + self.size, x:x + self.size, ::-1])

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), 64):
                x = torch.from_numpy(np.stack([self._prep(im) for im in images[i:i + 64]])).to(self.device)
                x = x.permute(0, 3, 1, 2).float().div_(255.0).sub_(self.mean).div_(self.std)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                    t = self.model.forward_features(x)
                k = self.model.num_prefix_tokens
                f = torch.cat([t[:, 0], t[:, k:].mean(1)], 1).float()
                out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


class KnnReference:
    """Distance of an embedding to the clear-day training distribution (cosine, k nearest)."""

    def __init__(self, ref: np.ndarray, k: int = 5):
        from sklearn.neighbors import NearestNeighbors
        self.ref = np.asarray(ref, np.float32)
        self.k = min(k, len(self.ref))
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="cosine").fit(self.ref)
        self.mean = self.ref.mean(0)
        self.mean /= max(np.linalg.norm(self.mean), 1e-9)

    def features(self, emb: np.ndarray) -> list[dict]:
        emb = np.asarray(emb, np.float32)
        if len(emb) == 0:
            return []
        d, _ = self.nn.kneighbors(emb)
        to_mean = 1.0 - emb @ self.mean
        return [{"d_knn_dist": float(d[i].mean()), "d_knn_dist_min": float(d[i].min()), "d_norm_to_mean": float(to_mean[i])}
                for i in range(len(emb))]


# --------------------------------------------------------------------------- (e) weather model output
def weather_features(p_weather: dict | None, fog_level: str | None) -> dict:
    p = p_weather or {}
    return {"w_p_clear": float(p.get("clear", 0.0)), "w_p_fog": float(p.get("fog", 0.0)), "w_p_night": float(p.get("night", 0.0)),
            "w_p_rain": float(p.get("rain", 0.0)), "w_p_snow": float(p.get("snow", 0.0)),
            "w_fog_level": FOG_LEVEL_ORD.get(fog_level or "none", 0.0)}


def feature_vector(meta: dict, names: list[str] = ALL_FEATURES) -> list[float]:
    return [float(meta.get(n, 0.0) if meta.get(n) is not None and not (isinstance(meta.get(n), float) and math.isnan(meta[n])) else 0.0)
            for n in names]

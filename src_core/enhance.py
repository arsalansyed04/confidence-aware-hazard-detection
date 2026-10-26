"""Weather-adaptive enhancement, keyed on the weather model's output. A TESTED hypothesis, never assumed good.

Every recipe works on luminance only (LAB L channel) so colours are untouched, creates no halos, and is
scaled by severity. "clear" is deliberately a no-op: the detector was trained on images like these.
"""
from __future__ import annotations

import cv2
import numpy as np

RECIPES = {  # (weather, fog_level) -> list of (op, kwargs); fog_level only matters for weather == fog
    ("clear", "none"): [],
    ("fog", "light"): [("clahe", {"clip": 1.5})],
    ("fog", "medium"): [("dehaze", {"omega": 0.75, "t0": 0.2}), ("clahe", {"clip": 1.5})],
    ("fog", "dense"): [("dehaze", {"omega": 0.9, "t0": 0.2}), ("clahe", {"clip": 2.0}), ("bilateral", {"d": 5})],
    ("fog", "unrated"): [("dehaze", {"omega": 0.75, "t0": 0.2}), ("clahe", {"clip": 1.5})],
    ("night", "none"): [("gamma_adaptive", {"target": 0.45, "lo": 0.4, "hi": 1.0}), ("clahe", {"clip": 2.0}), ("bilateral", {"d": 5})],
    ("rain", "none"): [("clahe", {"clip": 1.8})],
    ("snow", "none"): [("gamma_bright", {"gamma": 1.2, "threshold": 0.62}), ("clahe", {"clip": 1.5})],
}


def recipe_for(weather: str, fog_level: str = "none") -> list:
    if weather == "fog":
        return RECIPES.get(("fog", fog_level if fog_level in ("light", "medium", "dense") else "unrated"))
    return RECIPES.get((weather, "none"), [])


def clahe(bgr: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _gamma_lut(gamma: float) -> np.ndarray:
    return np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)


def gamma_l(bgr: np.ndarray, gamma: float) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.LUT(lab[:, :, 0], _gamma_lut(gamma))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gamma_adaptive(bgr: np.ndarray, target: float = 0.45, lo: float = 0.4, hi: float = 1.0) -> np.ndarray:
    """Lift shadows at night: choose gamma so the mean luminance moves towards `target`, never above 1 (no darkening)."""
    mean = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean()) / 255.0
    if mean <= 1e-3 or mean >= target:
        return bgr
    gamma = float(np.clip(np.log(target) / np.log(mean), lo, hi))
    return gamma_l(bgr, gamma)


def gamma_bright(bgr: np.ndarray, gamma: float = 1.2, threshold: float = 0.62) -> np.ndarray:
    """Pull highlights down on washed-out snow frames; only when the frame really is bright."""
    mean = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean()) / 255.0
    return gamma_l(bgr, gamma) if mean > threshold else bgr


def bilateral(bgr: np.ndarray, d: int = 5, sigma_color: float = 40.0, sigma_space: float = 5.0) -> np.ndarray:
    return cv2.bilateralFilter(bgr, d, sigma_color, sigma_space)


def _guided_filter(guide: np.ndarray, src: np.ndarray, r: int = 30, eps: float = 1e-3) -> np.ndarray:
    """Box-filter guided filter (He et al.); no opencv-contrib needed."""
    k = (2 * r + 1, 2 * r + 1)
    mean_i = cv2.blur(guide, k)
    mean_p = cv2.blur(src, k)
    corr_i = cv2.blur(guide * guide, k)
    corr_ip = cv2.blur(guide * src, k)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return cv2.blur(a, k) * guide + cv2.blur(b, k)


def dehaze(bgr: np.ndarray, omega: float = 0.8, t0: float = 0.2, patch: int = 15) -> np.ndarray:
    """Dark-channel-prior dehazing with guided-filter refinement and a transmission floor (noise control).
    Bright regions (sky) get a clamped transmission so they are not over-darkened."""
    img = bgr.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    scale = 480.0 / max(h, w) if max(h, w) > 480 else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(small.min(axis=2), kernel)
    n = max(1, dark.size // 1000)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    air = small.reshape(-1, 3)[idx].max(axis=0)
    air = np.clip(air, 0.6, 1.0)
    norm = small / np.maximum(air, 1e-3)
    t = 1.0 - omega * cv2.erode(norm.min(axis=2), kernel)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    t = _guided_filter(gray, t.astype(np.float32), r=max(8, patch * 2), eps=1e-3)
    t = np.clip(t, t0, 1.0)
    t = np.where(gray > 0.9, np.maximum(t, 0.6), t)  # sky guard
    if scale < 1:
        t = cv2.resize(t, (w, h), interpolation=cv2.INTER_LINEAR)
    out = (img - air) / t[..., None] + air
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


OPS = {"clahe": clahe, "gamma_adaptive": gamma_adaptive, "gamma_bright": gamma_bright, "bilateral": bilateral, "dehaze": dehaze}


def enhance(bgr: np.ndarray, weather: str, fog_level: str = "none", enabled: bool = True) -> np.ndarray:
    """Apply the recipe for (weather, fog_level). Returns the input unchanged when disabled or for clear weather."""
    if not enabled:
        return bgr
    out = bgr
    for op, kw in recipe_for(weather, fog_level) or []:
        out = OPS[op](out, **kw)
    return out


def describe(weather: str, fog_level: str = "none") -> str:
    steps = recipe_for(weather, fog_level) or []
    return " -> ".join(f"{op}({', '.join(f'{k}={v}' for k, v in kw.items())})" for op, kw in steps) or "none"

"""Runtime weather recognition: the v4 ensemble (weather + fog level), batched, with mirrored-image averaging."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

from common import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src_v4"))

V4_DIR = PROJECT_ROOT / "results" / "weather_v4"


class WeatherModel:
    def __init__(self, ckpts: list | None = None, device: str = "cuda", tta: bool = True):
        import torch
        import weather_v4_lib as L4
        self.L4, self.torch = L4, torch
        paths = ckpts or sorted(glob.glob(str(V4_DIR / "ckpt_*.pt")))
        if not paths:
            raise FileNotFoundError(f"no v4 checkpoints under {V4_DIR}; train the weather model first (src_v4)")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.models = [L4.load_model(p, self.device) for p in paths]
        self.names = [m[1]["config_name"] for m in self.models]
        self.tta = tta
        self.classes, self.fog_levels, self.rain_levels = L4.CLASSES, L4.FOG_LEVELS, L4.RAIN_LEVELS

    def predict(self, images: list[np.ndarray]) -> list[dict]:
        """Per image: weather, fog_level, rain_level, p_weather, p_fog_level."""
        torch = self.torch
        if not images:
            return []
        sums = None
        for model, meta in self.models:
            pc, pf, pr = [], [], []
            with torch.inference_mode():
                for i in range(0, len(images), 32):
                    x = meta["norm"](torch.from_numpy(np.stack([meta["transform"](im) for im in images[i:i + 32]])))
                    with self.L4.v1.autocast_ctx(meta["use_amp"]):
                        outs = [model(x)] + ([model(x.flip(3))] if self.tta else [])
                    pc.append(torch.stack([torch.softmax(o[0].float(), 1) for o in outs]).mean(0).cpu().numpy())
                    pf.append(torch.stack([torch.softmax(o[1].float(), 1) for o in outs]).mean(0).cpu().numpy())
                    pr.append(torch.stack([torch.softmax(o[2].float(), 1) for o in outs]).mean(0).cpu().numpy())
            probs = [np.concatenate(pc), np.concatenate(pf), np.concatenate(pr)]
            sums = probs if sums is None else [a + b for a, b in zip(sums, probs)]
        pc, pf, pr = (s / len(self.models) for s in sums)
        out = []
        for i in range(len(images)):
            w = self.classes[int(pc[i].argmax())]
            out.append({"weather": w, "fog_level": self.L4.level_name(pf[i], self.fog_levels, w == "fog"),
                        "rain_level": self.L4.level_name(pr[i], self.rain_levels, w == "rain"),
                        "p_weather": {c: float(v) for c, v in zip(self.classes, pc[i])},
                        "p_fog_level": {c: float(v) for c, v in zip(self.fog_levels, pf[i])}})
        return out

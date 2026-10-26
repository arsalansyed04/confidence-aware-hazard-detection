"""Frozen YOLO detector wrapper (Ultralytics). The model is a flag: yolov8m, yolo11m, yolo26m. Never trained."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import COCO_TO_PROJECT, RECORD_CONF, WEIGHTS_DIR

KNOWN = {"yolov8m": "yolov8m.pt", "yolo11m": "yolo11m.pt", "yolo26m": "yolo26m.pt",
         "yolov8n": "yolov8n.pt", "yolo11n": "yolo11n.pt", "yolo26n": "yolo26n.pt"}


def resolve_weights(name: str) -> str:
    """A short name, a file name, or a path. Downloads go to weights/ on first use."""
    p = Path(name)
    if p.exists():
        return str(p)
    fn = KNOWN.get(name, name if name.endswith(".pt") else name + ".pt")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    local = WEIGHTS_DIR / fn
    return str(local) if local.exists() else fn  # ultralytics downloads a bare file name into the cwd


class Detector:
    def __init__(self, name: str = "yolo26m", device: str = "cuda", imgsz: int = 640, conf: float = RECORD_CONF):
        import os
        import torch
        from ultralytics import YOLO
        self.name = Path(name).stem
        weights = resolve_weights(name)
        cwd = os.getcwd()
        try:
            if not Path(weights).exists():  # let ultralytics download into weights/
                os.chdir(WEIGHTS_DIR)
            self.model = YOLO(weights)
        finally:
            os.chdir(cwd)
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.imgsz, self.conf = imgsz, conf
        self.nms_free = "26" in self.name
        self.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), verbose=False, device=self.device, imgsz=imgsz, conf=0.5)  # warm-up at the real size

    def detect(self, images: list[np.ndarray], conf: float | None = None) -> list[list[dict]]:
        """Batch detection. Returns per image a list of {cls, conf, xyxy} in PROJECT classes (COCO classes mapped, others dropped)."""
        if not images:
            return []
        out = []
        for i in range(0, len(images), 16):
            res = self.model.predict(images[i:i + 16], verbose=False, device=self.device, imgsz=self.imgsz,
                                     conf=self.conf if conf is None else conf, classes=list(COCO_TO_PROJECT))
            for r in res:
                dets = []
                if r.boxes is not None and len(r.boxes):
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    cf = r.boxes.conf.cpu().numpy()
                    cl = r.boxes.cls.cpu().numpy().astype(int)
                    for b, c, k in zip(xyxy, cf, cl):
                        name = COCO_TO_PROJECT.get(int(k))
                        if name:
                            dets.append({"cls": name, "conf": float(c), "xyxy": [float(v) for v in b]})
                out.append(dets)
        return out

    def detect_with_flip(self, images: list[np.ndarray], conf: float | None = None) -> tuple[list[list[dict]], list[list[dict]]]:
        """Detections on the images and on their mirrors, the latter un-mirrored back into image coordinates."""
        from features import unflip_boxes
        plain = self.detect(images, conf)
        flipped = self.detect([np.ascontiguousarray(im[:, ::-1]) for im in images], conf)
        return plain, [unflip_boxes(d, im.shape[1]) for d, im in zip(flipped, images)]

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_core"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads the real models (GPU, datasets); enable with --run-slow")


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run tests that load the real models (GPU)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow (loads real models)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def street_image(rng):
    """A synthetic 'road scene': sky gradient, dark road, two bright rectangles. Deterministic."""
    h, w = 360, 640
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        img[y, :] = (200 - y // 3, 180 - y // 4, 160 - y // 4) if y < h // 2 else (60, 60, 60)
    img[200:300, 100:180] = (30, 30, 200)
    img[220:320, 400:520] = (0, 160, 0)
    img = np.clip(img.astype(np.int16) + rng.integers(-8, 9, img.shape), 0, 255).astype(np.uint8)
    return img

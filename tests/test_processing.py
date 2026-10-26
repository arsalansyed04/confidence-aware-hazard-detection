import math

import numpy as np
import pytest

import danger as DG
import features as F
from enhance import OPS, RECIPES, dehaze, describe, enhance, recipe_for


# --------------------------------------------------------------------------- enhancement
def test_clear_is_untouched_and_off_is_identity(street_image):
    assert np.array_equal(enhance(street_image, "clear", "none"), street_image)
    assert np.array_equal(enhance(street_image, "fog", "dense", enabled=False), street_image)


@pytest.mark.parametrize("weather,level", [("fog", "light"), ("fog", "medium"), ("fog", "dense"), ("fog", "unrated"),
                                           ("night", "none"), ("rain", "none"), ("snow", "none")])
def test_recipes_preserve_shape_dtype_and_change_pixels(street_image, weather, level):
    out = enhance(street_image, weather, level)
    assert out.shape == street_image.shape and out.dtype == np.uint8
    assert describe(weather, level) != "none"
    if weather != "snow":  # snow gamma only fires on bright frames; CLAHE still changes it
        assert not np.array_equal(out, street_image)


def test_night_recipe_brightens_dark_frames_and_never_darkens(street_image):
    dark = (street_image * 0.15).astype(np.uint8)
    out = enhance(dark, "night")
    assert out.mean() > dark.mean()
    bright = np.full_like(street_image, 200)
    assert OPS["gamma_adaptive"](bright).mean() >= bright.mean() - 1


def test_dehaze_reduces_haze_on_a_hazed_image(street_image):
    hazed = (street_image.astype(np.float32) * 0.5 + 255 * 0.5).astype(np.uint8)
    out = dehaze(hazed, omega=0.9, t0=0.2)
    assert out.std() > hazed.std(), "dehazing should restore contrast"
    assert out.shape == hazed.shape and out.dtype == np.uint8


def test_every_recipe_op_exists():
    for steps in RECIPES.values():
        for op, _ in steps:
            assert op in OPS
    assert recipe_for("fog", "weird") == RECIPES[("fog", "unrated")]
    assert recipe_for("unknown_weather") == []


# --------------------------------------------------------------------------- features
def test_quality_features_track_blur_and_haze(street_image):
    import cv2
    q = F.quality_features(street_image)
    assert set(q) == set(F.QUALITY_FEATURES) and all(math.isfinite(v) for v in q.values())
    blurred = cv2.GaussianBlur(street_image, (0, 0), 6)
    assert F.quality_features(blurred)["q_laplacian_var"] < q["q_laplacian_var"]
    hazed = (street_image.astype(np.float32) * 0.4 + 255 * 0.6).astype(np.uint8)
    hq = F.quality_features(hazed)
    assert hq["q_dark_channel"] > q["q_dark_channel"] and hq["q_rms_contrast"] < q["q_rms_contrast"]


def test_conf_features_empty_and_filled():
    e = F.conf_features([], 0.25, (100, 100))
    assert e["c_n_det"] == 0 and set(e) == set(F.CONF_FEATURES)
    dets = [{"cls": "person", "conf": 0.9, "xyxy": [0, 0, 10, 20]}, {"cls": "car", "conf": 0.3, "xyxy": [0, 0, 50, 50]},
            {"cls": "car", "conf": 0.1, "xyxy": [0, 0, 50, 50]}]
    f = F.conf_features(dets, 0.25, (100, 100))
    assert f["c_n_det"] == 2 and f["c_n_person"] == 1 and math.isclose(f["c_max"], 0.9) and f["c_n_small"] == 1
    assert 0 < f["c_class_entropy"] <= math.log(2) + 1e-9


def test_flip_consistency_and_unflip():
    dets = [{"cls": "person", "conf": 0.9, "xyxy": [10, 10, 30, 60]}]
    un = F.unflip_boxes(F.unflip_boxes(dets, 100), 100)
    assert un[0]["xyxy"] == dets[0]["xyxy"]
    assert F.flip_consistency(dets, dets)["f_dice"] == 1.0
    assert F.flip_consistency(dets, [])["f_dice"] == 0.0
    assert F.flip_consistency([], [])["f_dice"] == 1.0
    moved = [{"cls": "person", "conf": 0.5, "xyxy": [60, 10, 80, 60]}]
    r = F.flip_consistency(dets, moved)
    assert r["f_dice"] == 0.0 and r["f_person_dice"] == 0.0


def test_weather_features_and_vector():
    w = F.weather_features({"fog": 0.7, "clear": 0.3}, "dense")
    assert w["w_fog_level"] == 3.0 and w["w_p_fog"] == 0.7 and w["w_p_snow"] == 0.0
    v = F.feature_vector({"q_laplacian_var": float("nan"), "c_mean": 0.5}, ["q_laplacian_var", "c_mean", "missing"])
    assert v == [0.0, 0.5, 0.0]


def test_knn_reference_distances(rng):
    ref = rng.normal(size=(50, 8)).astype(np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    knn = F.KnnReference(ref, k=5)
    near = knn.features(ref[:1])[0]
    far = knn.features(-ref[:1])[0]
    assert near["d_knn_dist_min"] < 1e-5 and far["d_knn_dist"] > near["d_knn_dist"]
    assert knn.features(np.zeros((0, 8))) == []


# --------------------------------------------------------------------------- danger
def test_danger_bounds_and_monotonicity():
    hw = (720, 1280)
    assert DG.danger([], hw, "clear")["danger"] == 0.0
    d_clear = DG.danger([], hw, "fog", "dense")["danger"]
    assert 0 < d_clear <= DG.W_WEATHER + 1e-9
    near = [{"cls": "person", "conf": 0.9, "xyxy": [600, 300, 700, 700]}]
    far = [{"cls": "person", "conf": 0.9, "xyxy": [600, 300, 610, 330]}]
    assert DG.danger(near, hw, "clear")["danger"] > DG.danger(far, hw, "clear")["danger"]
    car = [{"cls": "car", "conf": 0.9, "xyxy": [600, 300, 700, 700]}]
    assert DG.danger(near, hw, "clear")["danger"] > DG.danger(car, hw, "clear")["danger"], "pedestrians weigh more than cars"
    many = near * 20
    assert DG.danger(many, hw, "clear")["danger"] <= 1.0


def test_gap_is_zero_when_detections_equal_ground_truth():
    hw = (720, 1280)
    gts = [{"cls": "person", "xyxy": [600, 300, 700, 700]}, {"cls": "car", "xyxy": [0, 400, 300, 700]}]
    dets = [dict(g, conf=0.9) for g in gts]
    g = DG.gap(dets, gts, hw, "fog", "dense")
    assert math.isclose(g["gap"], 0.0, abs_tol=1e-9)
    missed = DG.gap(dets[1:], gts, hw, "fog", "dense")
    assert missed["gap"] > 0, "a missed pedestrian makes perception underestimate danger"


def test_quadrant_and_sensitivity():
    assert DG.quadrant(0.2, 0.1) == "low trust, low apparent danger"
    assert DG.quadrant(0.9, 0.9) == "high trust, high danger"
    rng = np.random.default_rng(1)
    frames = []
    for _ in range(60):
        n = int(rng.integers(0, 6))
        boxes = [{"cls": str(rng.choice(["person", "car", "truck", "bike"])), "xyxy": sorted(rng.uniform(0, 1280, 2)).__add__([])}
                 for _ in range(n)]
        boxes = [{"cls": b["cls"], "xyxy": [b["xyxy"][0], 200, b["xyxy"][1], 200 + rng.uniform(20, 400)]} for b in boxes]
        frames.append({"boxes": boxes, "hw": (720, 1280), "weather": "clear"})
    s = DG.weight_sensitivity(frames)
    assert len(s) == 2 * len(DG.CLASS_WEIGHTS)
    assert all(0.0 <= v["tercile_agreement"] <= 1.0 for v in s.values())

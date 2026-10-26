import json
import math

import numpy as np
import pandas as pd
import pytest

import features as F
from common import FrameGT, write_jsonl, record_template
from gt_loaders import cityscapes_boxes, bdd_condition
from trust import TrustModel, auroc, fit_xgb, risk_coverage, table_from_records, evaluate


# --------------------------------------------------------------------------- loaders
def test_cityscapes_boxes_from_instance_mask(tmp_path):
    import cv2
    m = np.zeros((100, 200), np.uint16)
    m[10:50, 20:60] = 24000 + 1      # person instance
    m[60:90, 100:180] = 26000 + 3    # car instance
    m[0:5, 0:5] = 31000              # train: not a project class
    m[70:75, 10:15] = 24             # semantic-only pixel (< 1000): ignored
    p = tmp_path / "x_000000_000019_gtFine_instanceIds.png"
    cv2.imwrite(str(p), m)
    boxes = sorted(cityscapes_boxes(p), key=lambda b: b["cls"])
    assert [b["cls"] for b in boxes] == ["car", "person"]
    assert boxes[1]["xyxy"] == [20.0, 10.0, 60.0, 50.0]
    assert boxes[0]["xyxy"] == [100.0, 60.0, 180.0, 90.0]


def test_bdd_condition_mapping():
    assert bdd_condition({"weather": "clear", "timeofday": "daytime"}) == "clear"
    assert bdd_condition({"weather": "overcast", "timeofday": "daytime"}) == "clear"
    assert bdd_condition({"weather": "rainy", "timeofday": "night"}) == "night"
    assert bdd_condition({"weather": "rainy", "timeofday": "daytime"}) == "rain"
    assert bdd_condition({"weather": "clear", "timeofday": "dawn/dusk"}) is None
    assert bdd_condition({"weather": "undefined", "timeofday": "daytime"}) is None


def test_acdc_loader_filters_the_mispackaged_rain_file():
    from gt_loaders import ACDC_DET, load_acdc
    if not (ACDC_DET / "rain" / "instancesonly_rain_train_gt_detection.json").exists():
        pytest.skip("ACDC detection package not on this machine")
    frames = load_acdc("rain", "train")
    assert len(frames) == 400 and all(f.path.replace("\\", "/").count("/rain/train/") == 1 for f in frames)
    assert all(b["cls"] in ("person", "car", "truck", "bus", "bike", "motor") for f in frames for b in f.boxes)


# --------------------------------------------------------------------------- trust
def _synthetic_records(tmp_path, n=400, seed=0):
    """Records where failure is driven by low confidence and high haze, with a clear-day training stratum."""
    rng = np.random.default_rng(seed)
    rows = []
    for stratum in ("bdd_train_clear", "acdc_fog", "bdd_val_night"):
        for i in range(n):
            haze = rng.uniform(0.0, 0.3 if stratum == "bdd_train_clear" else 0.8)
            conf = float(np.clip(rng.normal(0.8 - haze, 0.1), 0.05, 0.99))
            fail = rng.random() < 1 / (1 + math.exp(-(10 * haze - 5 * conf + 1)))
            fg = FrameGT(f"{stratum}_{i}", "x.jpg", "bdd", "val", "fog" if "fog" in stratum else "clear", "daytime", stratum, [])
            rec = record_template(fg, "fake", "off")
            rec["stratum"] = stratum
            rec["meta"] = {"q_dark_channel": haze, "c_mean": conf, "c_n_det": int(rng.integers(0, 8)), "f_dice": float(rng.uniform(0.5, 1))}
            rec["eval"] = {"fail": bool(fail), "fail_person": bool(fail), "fail_f1": False}
            rec["scores"] = {"danger_est": 0.3, "danger_true": 0.5, "danger_gap": 0.2}
            rows.append(rec)
    p = tmp_path / "records.jsonl"
    write_jsonl(p, rows)
    return p


def test_trust_model_learns_transfers_and_roundtrips(tmp_path):
    p = _synthetic_records(tmp_path)
    df = table_from_records(p)
    assert set(F.ALL_FEATURES) <= set(df.columns) and df["q_edge_density"].eq(0).all(), "missing features default to 0"
    train = df[df["stratum"] == "bdd_train_clear"]
    feats = ["q_dark_channel", "c_mean", "c_n_det", "f_dice"]
    tm = TrustModel(fit_xgb(train[feats].to_numpy(np.float32), train["fail"].to_numpy(bool)), feats, "fail")
    res = evaluate(tm, df, "fail").set_index("stratum")
    assert res.loc["acdc_fog", "auroc"] > 0.75, "must transfer to the unseen fog stratum on this synthetic data"
    assert res.loc["acdc_fog", "fail_at_50"] <= res.loc["acdc_fog", "fail_at_100"], "trusted half must fail less often"
    tm.save(tmp_path / "m.json")
    back = TrustModel.load(tmp_path / "m.json")
    assert back.features == feats and back.label == "fail"
    t1, t2 = tm.trust(df.head(5)), back.trust([dict(r) for r in df.head(5)[feats].to_dict("records")])
    assert np.allclose(t1, t2, atol=1e-6)
    assert np.all((t1 >= 0) & (t1 <= 1))


def test_auroc_and_risk_coverage_edge_cases():
    assert math.isnan(auroc(np.zeros(5, bool), np.arange(5)))
    assert auroc(np.array([0, 0, 1, 1], bool), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    rc = risk_coverage(np.array([0.9, 0.8, 0.1, 0.0]), np.array([False, False, True, True]))
    assert rc["fail_at_50"] == 0.0 and rc["fail_at_100"] == 0.5


def test_trust_cli_train_writes_outputs(tmp_path, monkeypatch):
    import trust as T
    p = _synthetic_records(tmp_path)
    out = tmp_path / "trust"
    monkeypatch.setattr(T, "RESULTS", tmp_path)
    T.main(["train", "--records", str(p), "--out", str(out), "--families", "quality", "confidence", "flip"])
    for f in ("trust_model.json", "trust_model.meta.json", "trust_results.csv", "trust_ablation.csv", "trust_scores.csv", "risk_coverage.png", "trust_auroc.png"):
        assert (out / f).exists(), f
    res = pd.read_csv(out / "trust_results.csv")
    assert any("in-sample" in s for s in res["stratum"]) and res["auroc"].notna().sum() >= 2

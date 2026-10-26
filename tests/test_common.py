import json
import math

import numpy as np
import pytest

from common import (FrameGT, average_precision, filter_boxes, frame_metrics, greedy_match, iou_matrix, read_jsonl,
                    record_template, stratum_ap50, write_jsonl)


def box(cls, x1, y1, x2, y2, conf=None):
    b = {"cls": cls, "xyxy": [x1, y1, x2, y2]}
    if conf is not None:
        b["conf"] = conf
    return b


def test_iou_matrix_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
    m = iou_matrix(a, b)
    assert m.shape == (1, 3)
    assert math.isclose(m[0, 0], 1.0)
    assert math.isclose(m[0, 1], 25 / 175, rel_tol=1e-5)
    assert m[0, 2] == 0.0
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_greedy_match_is_one_to_one_and_class_aware():
    gts = [box("person", 0, 0, 10, 10), box("car", 0, 0, 10, 10)]
    dets = [box("person", 0, 0, 10, 10, 0.9), box("person", 1, 1, 10, 10, 0.8), box("car", 0, 0, 10, 10, 0.7)]
    pairs = greedy_match(dets, gts)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 0), (2, 1)]  # second person det has no free gt
    assert len(greedy_match(dets, gts, class_aware=False)) == 2
    assert greedy_match([], gts) == [] and greedy_match(dets, []) == []


def test_frame_metrics_failure_labels():
    gts = [box("person", 0, 0, 20, 60), box("car", 100, 100, 200, 160)]
    perfect = frame_metrics([box("person", 0, 0, 20, 60, 0.9), box("car", 100, 100, 200, 160, 0.9)], gts)
    assert perfect["f1"] == 1.0 and not perfect["fail"] and perfect["missed_person"] == 0
    missed = frame_metrics([box("car", 100, 100, 200, 160, 0.9)], gts)
    assert missed["missed_person"] == 1 and missed["fail_person"] and missed["fail"]
    low_conf = frame_metrics([box("person", 0, 0, 20, 60, 0.1), box("car", 100, 100, 200, 160, 0.9)], gts, conf=0.25)
    assert low_conf["missed_person"] == 1, "detections below the operating threshold must not count"
    empty = frame_metrics([], [])
    assert empty["f1"] == 1.0 and not empty["fail"]
    tiny = frame_metrics([], [box("person", 0, 0, 5, 8)], min_px=12)
    assert tiny["n_gt"] == 0 and not tiny["fail"], "boxes shorter than min_px are ignored"


def test_filter_boxes():
    dets = [box("car", 0, 0, 10, 5, 0.9), box("car", 0, 0, 10, 50, 0.1)]
    assert len(filter_boxes(dets, conf=0.25)) == 1
    assert len(filter_boxes(dets, min_px=12)) == 1
    assert len(filter_boxes(dets, conf=0.25, min_px=12)) == 0


def test_average_precision_and_map50():
    assert average_precision(np.array([0.9, 0.8]), np.array([True, True]), 2) == 1.0
    assert average_precision(np.array([0.9, 0.8]), np.array([False, False]), 2) == 0.0
    assert math.isnan(average_precision(np.array([]), np.array([], bool), 0))
    ap = average_precision(np.array([0.9, 0.8, 0.7]), np.array([True, False, True]), 2)
    assert 0.5 < ap < 1.0
    gts = [box("person", 0, 0, 20, 60), box("car", 100, 100, 200, 160)]
    dets = [box("person", 0, 0, 20, 60, 0.9), box("car", 300, 300, 400, 360, 0.95)]
    r = stratum_ap50([(dets, gts)])
    assert r["per_class"]["person"] == 1.0 and r["per_class"]["car"] == 0.0 and r["map50"] == 0.5


def test_record_roundtrip(tmp_path):
    fg = FrameGT("f1", "x.jpg", "bdd", "val", "fog", "daytime", "acdc_fog", [box("person", 1, 2, 3, 4)], {"beta": "0.01"})
    rec = record_template(fg, "yolo26m", "off")
    rec["detections"] = [box("car", 0, 0, 1, 1, np.float32(0.5))]
    rec["meta"] = {"x": np.float64(1.5), "n": np.int64(3)}
    p = tmp_path / "r.jsonl"
    assert write_jsonl(p, [rec, rec]) == 2
    back = list(read_jsonl(p))
    assert len(back) == 2 and back[0]["gt_boxes"][0]["cls"] == "person" and back[0]["detector"] == "yolo26m"
    assert back[0]["meta"] == {"x": 1.5, "n": 3}
    json.dumps(back[0])  # plain JSON types only

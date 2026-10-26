"""The desktop interface: command building (pure) and the app's graceful stop file."""
import csv

import pytest

import gui as G


def base_opts(**over):
    o = {"mode": "video", "input": "clip.mp4", "stride": 2, "camera": 1, "width": 640, "height": 480, "detector": "yolo11m",
         "conf": 0.3, "imgsz": 960, "weather_every": 5, "flip_every": 2, "device": "cuda", "enhance": True, "dino": True, "show": True,
         "save_video": True, "output": "o.mp4", "save_csv": True, "csv": "o.csv", "save_jsonl": False, "jsonl": "o.jsonl",
         "max_frames": 0, "stop_file": "stop.txt"}
    o.update(over)
    return o


def test_build_command_video():
    cmd = G.build_command(base_opts())
    assert cmd[0].endswith("app.py") and cmd[1] == "video"
    assert cmd[cmd.index("--input") + 1] == "clip.mp4" and cmd[cmd.index("--stride") + 1] == "2"
    assert cmd[cmd.index("--detector") + 1] == "yolo11m" and cmd[cmd.index("--conf") + 1] == "0.30" and cmd[cmd.index("--imgsz") + 1] == "960"
    assert "--show" in cmd and "--no-enhance" not in cmd and "--no-dino" not in cmd and "--jsonl" not in cmd
    assert cmd[cmd.index("--output") + 1] == "o.mp4" and cmd[cmd.index("--csv") + 1] == "o.csv" and cmd[cmd.index("--stop-file") + 1] == "stop.txt"
    assert "--max-frames" not in cmd and "--device-index" not in cmd


def test_build_command_camera_and_flags():
    cmd = G.build_command(base_opts(mode="camera", enhance=False, dino=False, show=False, save_video=False, save_csv=False, save_jsonl=True, max_frames=120))
    assert cmd[1] == "camera" and cmd[cmd.index("--device-index") + 1] == "1" and cmd[cmd.index("--width") + 1] == "640"
    assert "--no-enhance" in cmd and "--no-dino" in cmd and "--no-show" in cmd and "--input" not in cmd and "--stride" not in cmd
    assert "--output" not in cmd and "--csv" not in cmd and cmd[cmd.index("--jsonl") + 1] == "o.jsonl" and cmd[cmd.index("--max-frames") + 1] == "120"


def test_app_stops_on_stop_file(tmp_path, monkeypatch):
    import app as A
    import pipeline as P
    from test_app import FakePipeline, make_video
    monkeypatch.setattr(P, "Pipeline", FakePipeline)
    monkeypatch.setattr(A, "RESULTS", tmp_path)
    src = tmp_path / "in.mp4"
    make_video(src, 40)
    stop = tmp_path / "stop"
    stop.write_text("stop")  # already present: the app must process one frame, then stop
    csvp = tmp_path / "out.csv"
    A.main(["video", "--input", str(src), "--csv", str(csvp), "--stop-file", str(stop)])
    assert len(list(csv.DictReader(open(csvp)))) == 1


@pytest.mark.slow
def test_gui_constructs():
    from tkinter import Tk
    root = Tk()
    root.withdraw()
    app = G.App(root)
    app.v["mode"].set("camera")
    app.refresh()
    assert "camera" in app.cmd_box.get("1.0", "end")
    root.destroy()

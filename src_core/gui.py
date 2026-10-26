"""Desktop interface for the hazard-detection app (Tkinter, no extra dependency).

    python src_core/gui.py

Choose Camera or Video file, set the options, press Start. The interface builds the exact command line for
src_core/app.py, runs it as a subprocess, streams its log, and stops it gracefully through a stop file.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results" / "core"
DETECTORS = ["yolo26m", "yolo11m", "yolov8m"]


def build_command(o: dict) -> list[str]:
    """Translate the option dictionary into the app's argv. Pure function (unit-tested)."""
    cmd = [str(HERE / "app.py"), o["mode"]]
    if o["mode"] == "video":
        cmd += ["--input", o["input"], "--stride", str(int(o["stride"]))]
    else:
        cmd += ["--device-index", str(int(o["camera"])), "--width", str(int(o["width"])), "--height", str(int(o["height"]))]
    cmd += ["--detector", o["detector"], "--conf", f"{float(o['conf']):.2f}", "--imgsz", str(int(o["imgsz"])),
            "--weather-every", str(int(o["weather_every"])), "--flip-every", str(int(o["flip_every"])), "--device", o["device"]]
    if not o["enhance"]:
        cmd.append("--no-enhance")
    if not o["dino"]:
        cmd.append("--no-dino")
    if o["mode"] == "video" and o["show"]:
        cmd.append("--show")
    if o["mode"] == "camera" and not o["show"]:
        cmd.append("--no-show")
    if o["save_video"]:
        cmd += ["--output", o["output"]]
    if o["save_csv"]:
        cmd += ["--csv", o["csv"]]
    if o["save_jsonl"]:
        cmd += ["--jsonl", o["jsonl"]]
    if int(o.get("max_frames") or 0) > 0:
        cmd += ["--max-frames", str(int(o["max_frames"]))]
    if o.get("stop_file"):
        cmd += ["--stop-file", o["stop_file"]]
    return cmd


class App:
    def __init__(self, root: Tk):
        self.root = root
        root.title("Weather-Adaptive Hazard Detection")
        root.minsize(760, 640)
        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue = queue.Queue()
        self.stop_file = RESULTS / ".stop_app"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        v = self.v = {
            "mode": StringVar(value="video"), "input": StringVar(value=str(RESULTS / "sample_input.mp4")), "stride": IntVar(value=1),
            "camera": IntVar(value=0), "width": IntVar(value=1280), "height": IntVar(value=720),
            "detector": StringVar(value="yolo26m"), "conf": DoubleVar(value=0.25), "imgsz": IntVar(value=640),
            "weather_every": IntVar(value=5), "flip_every": IntVar(value=2), "device": StringVar(value="cuda"),
            "enhance": BooleanVar(value=True), "dino": BooleanVar(value=True), "show": BooleanVar(value=True),
            "save_video": BooleanVar(value=True), "output": StringVar(value=str(RESULTS / f"run_{stamp}.mp4")),
            "save_csv": BooleanVar(value=True), "csv": StringVar(value=str(RESULTS / f"run_{stamp}.csv")),
            "save_jsonl": BooleanVar(value=False), "jsonl": StringVar(value=str(RESULTS / f"run_{stamp}.jsonl")),
            "max_frames": IntVar(value=0),
        }
        pad = {"padx": 6, "pady": 3}
        frm = ttk.Frame(root, padding=10)
        frm.pack(fill="both", expand=True)

        # --- source
        src = ttk.LabelFrame(frm, text="1. Source", padding=8)
        src.pack(fill="x", **pad)
        ttk.Radiobutton(src, text="Real-time camera (webcam)", variable=v["mode"], value="camera", command=self.refresh).grid(row=0, column=0, sticky="w", columnspan=2)
        ttk.Radiobutton(src, text="Existing video file", variable=v["mode"], value="video", command=self.refresh).grid(row=0, column=2, sticky="w", columnspan=2)
        ttk.Label(src, text="Video file").grid(row=1, column=0, sticky="e")
        self.e_input = ttk.Entry(src, textvariable=v["input"], width=58)
        self.e_input.grid(row=1, column=1, columnspan=2, sticky="we")
        self.b_browse = ttk.Button(src, text="Browse...", command=self.browse_input)
        self.b_browse.grid(row=1, column=3)
        ttk.Label(src, text="Process every Nth frame (stride)").grid(row=2, column=0, sticky="e")
        self.s_stride = ttk.Spinbox(src, from_=1, to=30, textvariable=v["stride"], width=6)
        self.s_stride.grid(row=2, column=1, sticky="w")
        ttk.Label(src, text="Camera index").grid(row=3, column=0, sticky="e")
        self.s_cam = ttk.Spinbox(src, from_=0, to=9, textvariable=v["camera"], width=6)
        self.s_cam.grid(row=3, column=1, sticky="w")
        ttk.Label(src, text="Camera size (w x h)").grid(row=3, column=2, sticky="e")
        camsz = ttk.Frame(src)
        camsz.grid(row=3, column=3, sticky="w")
        self.s_w = ttk.Spinbox(camsz, from_=320, to=3840, increment=160, textvariable=v["width"], width=6)
        self.s_w.pack(side="left")
        self.s_h = ttk.Spinbox(camsz, from_=240, to=2160, increment=120, textvariable=v["height"], width=6)
        self.s_h.pack(side="left")
        src.columnconfigure(1, weight=1)

        # --- pipeline options
        opt = ttk.LabelFrame(frm, text="2. Pipeline options", padding=8)
        opt.pack(fill="x", **pad)
        ttk.Label(opt, text="Detector").grid(row=0, column=0, sticky="e")
        ttk.Combobox(opt, textvariable=v["detector"], values=DETECTORS, state="readonly", width=10).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(opt, text="Weather-adaptive enhancement (ON)", variable=v["enhance"]).grid(row=0, column=2, sticky="w", columnspan=2)
        ttk.Checkbutton(opt, text="DINOv2 distance feature (slower, better trust)", variable=v["dino"]).grid(row=1, column=2, sticky="w", columnspan=2)
        ttk.Label(opt, text="Confidence threshold").grid(row=1, column=0, sticky="e")
        ttk.Spinbox(opt, from_=0.05, to=0.95, increment=0.05, textvariable=v["conf"], width=6, format="%.2f").grid(row=1, column=1, sticky="w")
        ttk.Label(opt, text="Weather model every N frames").grid(row=2, column=0, sticky="e")
        ttk.Spinbox(opt, from_=1, to=60, textvariable=v["weather_every"], width=6).grid(row=2, column=1, sticky="w")
        ttk.Label(opt, text="Mirror (flip) pass every N frames").grid(row=2, column=2, sticky="e")
        ttk.Spinbox(opt, from_=1, to=60, textvariable=v["flip_every"], width=6).grid(row=2, column=3, sticky="w")
        ttk.Label(opt, text="Detector input size").grid(row=3, column=0, sticky="e")
        ttk.Combobox(opt, textvariable=v["imgsz"], values=[480, 640, 960, 1280], state="readonly", width=6).grid(row=3, column=1, sticky="w")
        ttk.Label(opt, text="Device").grid(row=3, column=2, sticky="e")
        ttk.Combobox(opt, textvariable=v["device"], values=["cuda", "cpu"], state="readonly", width=6).grid(row=3, column=3, sticky="w")
        ttk.Label(opt, text="Max frames (0 = all)").grid(row=4, column=0, sticky="e")
        ttk.Spinbox(opt, from_=0, to=100000, increment=50, textvariable=v["max_frames"], width=8).grid(row=4, column=1, sticky="w")
        ttk.Checkbutton(opt, text="Show live window", variable=v["show"]).grid(row=4, column=2, sticky="w", columnspan=2)
        ttk.Button(opt, text="Fast preset", command=self.preset_fast).grid(row=5, column=1, sticky="w", pady=4)
        ttk.Button(opt, text="Accurate preset", command=self.preset_accurate).grid(row=5, column=2, sticky="w", pady=4)

        # --- outputs
        out = ttk.LabelFrame(frm, text="3. Outputs", padding=8)
        out.pack(fill="x", **pad)
        for r, (flag, var, label, ext) in enumerate((("save_video", "output", "Annotated video (.mp4)", ".mp4"), ("save_csv", "csv", "Per-frame CSV", ".csv"),
                                                   ("save_jsonl", "jsonl", "Per-frame JSONL with detections", ".jsonl"))):
            ttk.Checkbutton(out, text=label, variable=v[flag]).grid(row=r, column=0, sticky="w")
            ttk.Entry(out, textvariable=v[var], width=52).grid(row=r, column=1, sticky="we")
            ttk.Button(out, text="...", width=3, command=lambda var=var, ext=ext: self.browse_save(var, ext)).grid(row=r, column=2)
        out.columnconfigure(1, weight=1)

        # --- run
        run = ttk.Frame(frm)
        run.pack(fill="x", **pad)
        self.b_start = ttk.Button(run, text="Start", command=self.start)
        self.b_start.pack(side="left")
        self.b_stop = ttk.Button(run, text="Stop", command=self.stop, state="disabled")
        self.b_stop.pack(side="left", padx=6)
        ttk.Button(run, text="Open results folder", command=self.open_results).pack(side="left", padx=6)
        self.status = StringVar(value="idle")
        ttk.Label(run, textvariable=self.status).pack(side="right")
        ttk.Label(frm, text="Command that will run:").pack(anchor="w", padx=6)
        self.cmd_box = ScrolledText(frm, height=3, wrap="word")
        self.cmd_box.pack(fill="x", padx=6)
        ttk.Label(frm, text="Log:").pack(anchor="w", padx=6)
        self.log = ScrolledText(frm, height=12, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        for var in v.values():
            var.trace_add("write", lambda *_: self.refresh())
        self.refresh()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(200, self.pump)

    # ----------------------------------------------------------------- helpers
    def options(self) -> dict:
        o = {k: var.get() for k, var in self.v.items()}
        o["stop_file"] = str(self.stop_file)
        return o

    def refresh(self):
        video = self.v["mode"].get() == "video"
        for w in (self.e_input, self.b_browse, self.s_stride):
            w.configure(state="normal" if video else "disabled")
        for w in (self.s_cam, self.s_w, self.s_h):
            w.configure(state="disabled" if video else "normal")
        try:
            cmd = build_command(self.options())
            self.cmd_box.configure(state="normal")
            self.cmd_box.delete("1.0", "end")
            self.cmd_box.insert("end", "python " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
            self.cmd_box.configure(state="disabled")
        except Exception as e:  # noqa: BLE001  (half-typed numbers while editing)
            self.cmd_box.configure(state="normal")
            self.cmd_box.delete("1.0", "end")
            self.cmd_box.insert("end", f"(incomplete options: {e})")
            self.cmd_box.configure(state="disabled")

    def preset_fast(self):
        self.v["dino"].set(False)
        self.v["flip_every"].set(5)
        self.v["weather_every"].set(10)
        self.v["imgsz"].set(640)

    def preset_accurate(self):
        self.v["dino"].set(True)
        self.v["flip_every"].set(1)
        self.v["weather_every"].set(1)
        self.v["imgsz"].set(960)

    def browse_input(self):
        p = filedialog.askopenfilename(title="Choose a video", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")])
        if p:
            self.v["input"].set(p)

    def browse_save(self, var: str, ext: str):
        p = filedialog.asksaveasfilename(defaultextension=ext, initialfile=Path(self.v[var].get()).name)
        if p:
            self.v[var].set(p)

    def open_results(self):
        RESULTS.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(RESULTS)  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(RESULTS)])

    def append_log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    # ----------------------------------------------------------------- run control
    def start(self):
        if self.proc is not None:
            return
        o = self.options()
        if o["mode"] == "video" and not Path(o["input"]).is_file():
            messagebox.showerror("Input", f"Video file not found:\n{o['input']}")
            return
        RESULTS.mkdir(parents=True, exist_ok=True)
        if self.stop_file.exists():
            self.stop_file.unlink()
        cmd = [sys.executable, *build_command(o)]
        self.append_log("\n> " + " ".join(cmd) + "\n")
        try:
            self.proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:
            messagebox.showerror("Start failed", str(e))
            self.proc = None
            return
        threading.Thread(target=self.reader, daemon=True).start()
        self.b_start.configure(state="disabled")
        self.b_stop.configure(state="normal")
        self.status.set("running")

    def reader(self):
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            if "UserWarning" in line or "warnings.warn" in line:
                continue
            self.q.put(line)
        code = self.proc.wait()
        self.q.put(f"[gui] process exited with code {code}\n")
        self.q.put(None)

    def pump(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item is None:
                    self.proc = None
                    self.b_start.configure(state="normal")
                    self.b_stop.configure(state="disabled")
                    self.status.set("idle")
                else:
                    self.append_log(item)
        except queue.Empty:
            pass
        self.root.after(200, self.pump)

    def stop(self):
        if self.proc is None:
            return
        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        self.stop_file.write_text("stop")
        self.status.set("stopping (finishing the output file)...")
        self.root.after(8000, self.force_stop)

    def force_stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self.append_log("[gui] process did not stop in 8 s; terminated\n")

    def on_close(self):
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno("Quit", "A run is in progress. Stop it and quit?"):
                return
            self.stop_file.write_text("stop")
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
        self.root.destroy()


def main():
    root = Tk()
    try:
        ttk.Style().theme_use("vista" if sys.platform.startswith("win") else "clam")
    except Exception:  # noqa: BLE001
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

"""Render one MP4 per event: video on top, the four traces below with a moving cursor (matplotlib + ffmpeg).

Read-only on data and videos; output goes to results/clips/ (gitignored).
"""
import subprocess

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import io
from .config import FS, RESULTS, VIDEO_FILES
from .drops import DropParams
from .video import container_fps
from .viewer import _session_arrays

CLIP_DIR = RESULTS / "clips"
CLASS_COLOR = {("ALL_THREE", False): "#c0392b", ("ALL_THREE", True): "#7d3c98", ("OTHER", False): "#e08e0b", ("OTHER", True): "#566573"}
TRACE_COLOR = {"lfp": "#1baf7a", "thermistor": "#2a78d6", "airflow": "#eb6834", "emg": "#7f8c8d"}


def _read_frames(path, t_video, n, width=640, height=512):
    """n consecutive grey frames starting at video time t_video, decoded by ffmpeg (accurate seek)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t_video:.4f}", "-i", str(path), "-frames:v", str(n),
           "-vf", f"scale={width}:{height}", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width)


def export_session_clips(events, session, out_dir=CLIP_DIR, half=5.0, camera="left", out_fps=30, limit=None, p=DropParams()):
    """Write <event_id>.mp4 for each event of `session` that has video coverage. Returns the list of paths."""
    path = VIDEO_FILES.get(session, {}).get(camera)
    if path is None:
        raise ValueError(f"no {camera} video for {session}")
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = _session_arrays(session, p)
    cam_t = io.load_cam_times(session, camera)
    fps = container_fps(path)
    ev = events[(events.session_id == session) & events.frame_start.notna()]
    written = []
    for r in list(ev.itertuples())[:limit]:
        c = (r.t_start_rec + r.t_end_rec) / 2
        f0, f1 = np.searchsorted(cam_t, [c - half, c + half])
        f0 = max(f0, 0); f1 = min(f1, len(cam_t))
        frames_idx = np.arange(f0, f1, max(1, round(60 / out_fps)))
        vid = _read_frames(path, f0 / fps, f1 - f0)
        sel = frames_idx - f0
        sel = sel[sel < len(vid)]
        t_frames = cam_t[frames_idx[:len(sel)]]
        t0, t1 = c - half, c + half
        fig = plt.figure(figsize=(8, 8.4), dpi=80)
        gs = fig.add_gridspec(5, 1, height_ratios=[4.2, 1, 1, 1, 1.3], hspace=0.12, left=0.09, right=0.98, top=0.98, bottom=0.06)
        axv = fig.add_subplot(gs[0]); axv.axis("off")
        im = axv.imshow(vid[sel[0]], cmap="gray", vmin=0, vmax=255)
        txt = axv.text(0.01, 0.97, "", transform=axv.transAxes, color="w", va="top", fontsize=9)
        col = CLASS_COLOR[("ALL_THREE" if r.type == "ALL_THREE" else "OTHER", bool(r.emg_burst))]
        axes, cursors = [], []
        for k, (name, key) in enumerate([("LFP ch0", "lfp"), ("thermistor", "thermistor"), ("airflow", "airflow"), ("EMG", "emg")]):
            ax = fig.add_subplot(gs[k + 1]); axes.append(ax)
            i0, i1 = int(t0 * FS), int(t1 * FS)
            d = arr[key][i0:i1]; tt = np.arange(i0, i1) / FS
            ax.plot(tt, d, color=TRACE_COLOR[key], lw=0.6)
            if key == "emg":
                ax.plot(tt, arr["rms"][i0:i1], color="#c0392b", lw=1.4)
                ax.axhline(2.5 * arr["emg_median"], color="k", ls=":", lw=0.8)
            ax.axvspan(r.t_start_rec, r.t_end_rec, color=col, alpha=0.25, ls="--" if r.emg_burst else "-")
            ax.set_xlim(t0, t1); ax.set_ylabel(name, fontsize=8); ax.tick_params(labelsize=7)
            if k < 3:
                ax.set_xticklabels([])
            cursors.append(ax.axvline(t_frames[0], color="k", lw=1.5))
        axes[-1].set_xlabel("recording time (s)", fontsize=8)
        fig.suptitle("", fontsize=1)
        out = out_dir / f"{r.event_id}.mp4"
        w, h = fig.canvas.get_width_height()
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(out_fps),
                                "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)], stdin=subprocess.PIPE)
        for s, t in zip(sel, t_frames):
            im.set_data(vid[s])
            for cur in cursors:
                cur.set_xdata([t, t])
            txt.set_text(f"{r.type}  {r.sensors_dropped}  t={t:.2f} s" + ("  EMG burst" if r.emg_burst else ""))
            fig.canvas.draw()
            enc.stdin.write(bytes(fig.canvas.buffer_rgba()))
        enc.stdin.close(); enc.wait(); plt.close(fig)
        written.append(out)
    return written

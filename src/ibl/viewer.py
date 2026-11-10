"""Precompute per-event signal chunks (250 Hz JSON) and the event index for viewer/index.html.

Only windows around events are written, never whole sessions. All data is read-only; output goes to
results/viewer_chunks/ (gitignored) and viewer/video/ symlinks (gitignored).
"""
import json
import os

import numpy as np

from . import io
from .config import CHUNK_DIR, FS, ROOT, VIDEO_FILES
from .drops import DropParams, emg_filter, load_sensor, rms_envelope
from .signal import sos_bandpass
from .video import container_fps, measure_fps

DEC = 4                 # 1000 Hz -> 250 Hz
RMS_DEC = 20            # RMS is smooth (0.25 s window): 50 Hz is enough
VIEWER_DIR = ROOT / "viewer"


def link_videos():
    """Symlink the (read-only) video files into viewer/video/ so a static server can reach them."""
    d = VIEWER_DIR / "video"
    d.mkdir(parents=True, exist_ok=True)
    for s, cams in VIDEO_FILES.items():
        for cam, path in cams.items():
            link = d / f"{s}_{cam}.mp4"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(path, link)


def _quantize(x):
    """float -> int16-range ints plus scale (keeps the JSON small)."""
    scale = float(np.abs(x).max()) / 32000 or 1.0
    return np.round(x / scale).astype(int).tolist(), scale


def _session_arrays(session, p):
    out = {"lfp": sos_bandpass(load_sensor(session, "lfp"), *p.band)}
    for s in io.available_sensors(session):
        if s != "chest":
            out[s] = sos_bandpass(load_sensor(session, s), *p.band)
    out["emg"] = emg_filter(io.load_emg(session, 0), p)          # same notch + band-pass as the detector
    out["rms"] = rms_envelope(out["emg"], p)
    out["emg_median"] = float(np.median(out["rms"]))
    return out


def export_event_chunks(events, out_dir=CHUNK_DIR, half=8.0, p=DropParams(), sessions=None):
    """Write one JSON per event (+- `half` s around its centre) and index.json. Returns the index list.

    Chunk fields: t0 (recording time of sample 0), fs / rms_fs, traces as int arrays with *_scale,
    cam_frame0 + cam_t (camera frame times inside the chunk, for the video <-> recording time mapping).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for sid in sessions or list(dict.fromkeys(events.session_id)):
        ev = events[events.session_id == sid]
        arr = _session_arrays(sid, p)
        cam_t = io.load_cam_times(sid, "left")
        n = len(arr["lfp"])
        has_video = bool(VIDEO_FILES.get(sid))
        fps = container_fps(VIDEO_FILES[sid]["left"]) if has_video else measure_fps(cam_t)
        (out_dir / sid).mkdir(exist_ok=True)
        for r in ev.itertuples():
            c = (r.t_start_rec + r.t_end_rec) / 2
            i0 = max(0, int((c - half) * FS)); i0 -= i0 % DEC
            i1 = min(n, int((c + half) * FS)); i1 -= i1 % DEC
            chunk = {"event_id": r.event_id, "session_id": sid, "type": r.type, "sensors": r.sensors_dropped,
                     "emg_burst": bool(r.emg_burst), "emg_peak_ratio": float(r.emg_peak_ratio),
                     "t_start": float(r.t_start_rec), "t_end": float(r.t_end_rec), "t0": i0 / FS,
                     "fs": FS // DEC, "rms_fs": FS // RMS_DEC, "emg_median": arr["emg_median"]}
            for k in ("lfp", "thermistor", "airflow"):
                if k in arr:
                    chunk[k], chunk[k + "_scale"] = _quantize(arr[k][i0:i1:DEC])
            e = arr["emg"][i0:i1].reshape(-1, DEC)
            chunk["emg"], chunk["emg_scale"] = _quantize(e[np.arange(len(e)), np.abs(e).argmax(1)])   # peak-preserving
            chunk["rms"], chunk["rms_scale"] = _quantize(arr["rms"][i0:i1:RMS_DEC])
            a, b = np.searchsorted(cam_t, [i0 / FS, i1 / FS])
            chunk["cam_frame0"], chunk["cam_t"] = int(a), np.round(cam_t[a:b], 5).tolist()
            path = out_dir / sid / f"{r.event_id}.json"
            path.write_text(json.dumps(chunk, separators=(",", ":")))
            covered = bool(np.isfinite(r.frame_start))
            index.append({"id": r.event_id, "session": sid, "type": r.type, "sensors": r.sensors_dropped,
                          "emg": bool(r.emg_burst), "peak": float(r.emg_peak_ratio), "ts": float(r.t_start_rec),
                          "te": float(r.t_end_rec), "chunk": f"{sid}/{r.event_id}.json",
                          "video": covered and has_video, "fps": round(float(fps), 4)})
        del arr
    (out_dir / "index.json").write_text(json.dumps({"half": half, "events": index}, separators=(",", ":")))
    return index

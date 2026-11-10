"""Recording time <-> video frame mapping, frame-count check, and video columns for the events table."""
import struct

import numpy as np
import pandas as pd

from . import io
from .config import VIDEO_FILES


def measure_fps(cam_t):
    return (len(cam_t) - 1) / (cam_t[-1] - cam_t[0])


def rec_to_video_time(t, cam_t, fps=None):
    """Map recording time(s) t to (frame, video_time_s). frame = searchsorted(cam_t, t); video time = frame / fps.

    NaN where t lies outside the camera coverage [cam_t[0], cam_t[-1]].
    """
    t = np.asarray(t, dtype=float)
    fps = measure_fps(cam_t) if fps is None else fps
    frame = np.searchsorted(cam_t, t).astype(float)
    outside = (t < cam_t[0]) | (t > cam_t[-1])
    frame[outside] = np.nan
    return frame, frame / fps


def container_fps(path):
    """Frame rate of the MP4 timeline (frames / duration): what a player uses when it seeks, e.g. exactly 60.0."""
    i = mp4_info(path)
    return i["frames"] / i["duration_s"]


def format_timestamp(sec):
    """HH:MM:SS.mmm (empty string for NaN)."""
    if sec is None or not np.isfinite(sec):
        return ""
    ms = int(round(sec * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


# --------------------------------------------------------------------- mp4 header probe (no decoder needed)
def _atoms(f, start, end):
    pos = start
    while pos < end:
        f.seek(pos)
        h = f.read(8)
        if len(h) < 8:
            return
        size, typ = struct.unpack(">I4s", h)
        hl = 8
        if size == 1:
            size, hl = struct.unpack(">Q", f.read(8))[0], 16
        elif size == 0:
            size = end - pos
        yield typ.decode("latin1"), pos + hl, pos + size
        pos += size


def mp4_info(path):
    """{'frames', 'duration_s', 'width', 'height', 'codec'} read from the moov atom only (read-only)."""
    out = {}
    with open(path, "rb") as f:
        f.seek(0, 2)
        end = f.tell()
        for t, s, e in _atoms(f, 0, end):
            if t != "moov":
                continue
            for t2, s2, e2 in _atoms(f, s, e):
                if t2 == "mvhd":
                    f.seek(s2)
                    v = f.read(1)[0]
                    f.seek(s2 + 4 + (16 if v else 8))
                    ts, dur = struct.unpack(">IQ" if v else ">II", f.read(12 if v else 8))
                    out["duration_s"] = dur / ts
                if t2 == "trak":
                    for t3, s3, e3 in _atoms(f, s2, e2):
                        if t3 != "mdia":
                            continue
                        for t4, s4, e4 in _atoms(f, s3, e3):
                            if t4 != "minf":
                                continue
                            for t5, s5, e5 in _atoms(f, s4, e4):
                                if t5 != "stbl":
                                    continue
                                for t6, s6, e6 in _atoms(f, s5, e5):
                                    if t6 == "stts":
                                        f.seek(s6 + 4)
                                        n = struct.unpack(">I", f.read(4))[0]
                                        out["frames"] = sum(struct.unpack(">II", f.read(8))[0] for _ in range(n))
                                    if t6 == "stsd":
                                        f.seek(s6 + 8 + 4)
                                        out["codec"] = f.read(4).decode("latin1")
                                        f.seek(s6 + 8 + 4 + 4 + 24)
                                        out["width"], out["height"] = struct.unpack(">HH", f.read(4))
    return out


def check_frame_counts(session):
    """Real video frame count must equal len(cam_t) (camera_times_curated = False). Raises if not."""
    res = {}
    for cam, path in VIDEO_FILES.get(session, {}).items():
        n_vid, n_t = mp4_info(path)["frames"], len(io.load_cam_times(session, cam))
        res[cam] = (n_vid, n_t)
        if n_vid != n_t:
            raise RuntimeError(f"{session} {cam}: video has {n_vid} frames but cam_t has {n_t}. Stop and check the sync.")
    return res


def add_video_columns(df, camera="left"):
    """Add camera, frame_start/end, video_file, t_start/end_video, video_timestamp, video_fps.

    Frame numbers always come from searchsorted(cam_t, t). Video time = frame / fps where fps is the MP4
    container rate when the video file exists (that is the timeline players seek on; the measured camera
    rate 60.0047 Hz would drift ~0.8 s by the end of a 3 h session) and the measured cam_t rate otherwise.
    Rows outside the camera coverage stay NaN / empty.
    """
    df = df.copy()
    cols = ["frame_start", "frame_end", "t_start_video", "t_end_video"]
    for c in cols:
        df[c] = np.nan
    df["camera"], df["video_file"], df["video_timestamp"], df["video_fps"] = camera, "", "", np.nan
    for sid, idx in df.groupby("session_id").groups.items():
        cam_t = io.load_cam_times(sid, camera)
        path = VIDEO_FILES.get(sid, {}).get(camera)
        fps = container_fps(path) if path is not None else measure_fps(cam_t)
        df.loc[idx, "video_fps"] = round(fps, 4)
        f0, v0 = rec_to_video_time(df.loc[idx, "t_start_rec"].values, cam_t, fps)
        f1, v1 = rec_to_video_time(df.loc[idx, "t_end_rec"].values, cam_t, fps)
        ok = np.isfinite(f0) & np.isfinite(f1)
        for c, a in zip(cols, (f0, f1, v0, v1)):
            df.loc[idx, c] = np.where(ok, a, np.nan)
        if path is not None:
            df.loc[idx, "video_file"] = np.where(ok, path.name, "")
        df.loc[idx, "video_timestamp"] = [format_timestamp(v) if o else "" for v, o in zip(v0, ok)]
    for c in ("frame_start", "frame_end"):
        df[c] = df[c].astype("Int64")
    order = ["session_id", "mouse", "event_id", "type", "sensors_dropped", "emg_burst", "emg_peak_ratio",
             "t_start_rec", "t_end_rec", "duration_s", "camera", "frame_start", "frame_end", "video_file",
             "t_start_video", "t_end_video", "video_timestamp", "video_fps"]
    return df[order]

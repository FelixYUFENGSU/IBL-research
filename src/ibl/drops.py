"""Signal-drop detection: thermistor / airflow / LFP ch0 amplitude drops, EMG-burst flag, event classes.

Pipeline per sensor (one channel at a time, float32):
  1-15 Hz band-pass -> Hilbert amplitude -> 0.25 s smoothing -> compare with 60 s rolling median.
Drops are merged / filtered by duration, then grouped across sensors into events (classify_events).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import fft as sfft
from scipy.ndimage import median_filter, uniform_filter1d

from . import io
from .config import FS
from .signal import mains_notch, sos_bandpass

SENSOR_ORDER = ("thermistor", "airflow", "lfp")


@dataclass
class DropParams:
    band: tuple = (1.0, 15.0)      # Hz, 3rd-order Butterworth, zero-phase
    smooth_s: float = 0.25         # envelope smoothing
    median_win_s: float = 60.0     # rolling-median reference window
    frac: float = 0.35             # drop = amplitude < frac * rolling median
    min_dur_s: float = 0.3         # minimum drop duration
    merge_s: float = 0.2           # merge drops closer than this
    overlap: float = 0.5           # cross-sensor overlap (fraction of the shorter drop) to count as together
    emg_band: tuple = (30.0, 300.0)
    emg_win_s: float = 0.25
    notch: bool = True             # remove mains hum before the EMG band-pass (see docs/decisions.md, 2026-11-10)
    notch_f0: float = 50.0         # Hz, mains fundamental
    notch_harmonics: int = 7       # 50 ... 350 Hz (07-21 needs 350; 400/450 add nothing)
    notch_q: float = 30.0          # bandwidth = f / Q: 1.7 Hz at 50 Hz ... 11.7 Hz at 350 Hz
    emg_ratio: float = 2.5         # burst = RMS > emg_ratio * session median
    emg_pad_s: float = 1.0         # look in [start - pad, end + pad]


# ---------------------------------------------------------------------------------------- amplitude
def hilbert_amplitude(y):
    """|analytic signal| of an already band-passed float32 y, kept in float32 / complex64."""
    n = len(y)
    N = sfft.next_fast_len(n)
    X = sfft.fft(y, N)                                   # complex64 for float32 input
    h = np.zeros(N, dtype=np.float32)
    h[0] = 1
    if N % 2 == 0:
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[1:(N + 1) // 2] = 2
    X *= h
    return np.abs(sfft.ifft(X)[:n]).astype(np.float32, copy=False)


def amplitude_envelope(x, p=DropParams(), fs=FS):
    """1-15 Hz band-pass -> Hilbert amplitude -> moving-average smoothing. float32 in/out."""
    y = sos_bandpass(x, *p.band, fs=fs)
    env = hilbert_amplitude(y)
    del y
    return uniform_filter1d(env, max(1, int(round(p.smooth_s * fs))), mode="nearest")


def rolling_median(env, win_s=60.0, fs=FS, block=250):
    """Rolling median of a smooth envelope, computed on block means (4 Hz) and interpolated back."""
    n = len(env)
    m = n // block
    if m < 3:
        return np.full(n, np.median(env), dtype=np.float32)
    blocks = env[:m * block].reshape(m, block).mean(axis=1)
    size = max(3, int(round(win_s * fs / block)) | 1)    # odd window length
    med = median_filter(blocks, size=size, mode="nearest")
    centres = (np.arange(m) + 0.5) * block - 0.5
    return np.interp(np.arange(n), centres, med).astype(np.float32)


# -------------------------------------------------------------------------------------------- drops
def find_runs(mask):
    """[start, end) sample indices of True runs in a boolean array."""
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def detect_drops(ratio, p=DropParams(), fs=FS):
    """Drop intervals (n, 2) in seconds from ratio = amplitude / rolling median.

    ratio < p.frac, gaps shorter than p.merge_s merged, then runs shorter than p.min_dur_s removed.
    """
    s, e = find_runs(ratio < p.frac)
    if len(s) == 0:
        return np.empty((0, 2))
    new_run = np.concatenate([[True], (s[1:] - e[:-1]) >= p.merge_s * fs])   # False = merge into previous run
    first = np.flatnonzero(new_run)
    starts = s[first]
    ends = e[np.append(first[1:] - 1, len(e) - 1)]
    out = np.column_stack([starts, ends]) / fs
    return out[(out[:, 1] - out[:, 0]) >= p.min_dur_s]


def classify_events(drops, overlap=0.5):
    """Group per-sensor drops into events.

    drops: {sensor: (n,2) seconds}. Two drops of different sensors are linked if their overlap is at
    least `overlap` of the shorter one; linked drops (transitively) form one event. Returns a list of
    dicts {t_start, t_end, sensors, type} sorted by start; type is ALL_THREE / PAIR / SINGLE.
    """
    nodes = sorted((s, e, k) for k, arr in drops.items() for s, e in arr)
    parent = list(range(len(nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (s1, e1, k1) in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            s2, e2, k2 = nodes[j]
            if s2 >= e1:
                break
            if k1 != k2 and (min(e1, e2) - s2) >= overlap * min(e1 - s1, e2 - s2):
                parent[find(j)] = find(i)
    groups = {}
    for i, nd in enumerate(nodes):
        groups.setdefault(find(i), []).append(nd)
    events = []
    for g in groups.values():
        sens = [k for k in SENSOR_ORDER if k in {n[2] for n in g}]
        events.append(dict(t_start=min(n[0] for n in g), t_end=max(n[1] for n in g), sensors=sens,
                           type={3: "ALL_THREE", 2: "PAIR", 1: "SINGLE"}[len(sens)]))
    return sorted(events, key=lambda ev: ev["t_start"])


# ---------------------------------------------------------------------------------------------- EMG
def emg_filter(emg, p=DropParams(), fs=FS):
    """EMG as used everywhere (detection, viewer, clips): mains notches (optional) then the 30-300 Hz band-pass. float32."""
    if p.notch:
        emg = mains_notch(emg, p.notch_f0, p.notch_harmonics, p.notch_q, fs)
    return sos_bandpass(emg, *p.emg_band, fs=fs)


def rms_envelope(y, p=DropParams(), fs=FS):
    """RMS of an already filtered EMG over p.emg_win_s (moving window). float32."""
    y = np.square(y)
    r = uniform_filter1d(y, max(1, int(round(p.emg_win_s * fs))), mode="nearest")
    return np.sqrt(r, out=r)


def emg_rms(emg, p=DropParams(), fs=FS):
    """Notch (if enabled) -> 30-300 Hz band-pass -> RMS over 0.25 s. float32."""
    return rms_envelope(emg_filter(emg, p, fs), p, fs)


def emg_burst_flag(rms, t_start, t_end, p=DropParams(), fs=FS, median=None):
    """For each event: (burst bool, peak ratio) with ratio = max RMS in [start-pad, end+pad] / session median."""
    med = float(np.median(rms)) if median is None else median
    t_start, t_end = np.atleast_1d(t_start), np.atleast_1d(t_end)
    peaks = np.empty(len(t_start))
    for i, (a, b) in enumerate(zip(t_start, t_end)):
        i0 = max(0, int((a - p.emg_pad_s) * fs))
        i1 = min(len(rms), int(np.ceil((b + p.emg_pad_s) * fs)) + 1)
        peaks[i] = rms[i0:i1].max() / med
    return peaks > p.emg_ratio, peaks


# ------------------------------------------------------------------------------------------ session
def load_sensor(session, sensor):
    return io.load_lfp(session, 0) if sensor == "lfp" else io.load_resp(session, sensor)


def session_sensors(session):
    avail = io.available_sensors(session)
    return [s for s in SENSOR_ORDER if s == "lfp" or s in avail]


def sensor_ratios(session, p=DropParams(), keep=False):
    """{sensor: amplitude/rolling-median (float32)} computed one sensor at a time.

    keep=False returns only drop intervals, never holding more than one sensor in memory.
    """
    out = {}
    for s in session_sensors(session):
        env = amplitude_envelope(load_sensor(session, s), p)
        r = env / rolling_median(env, p.median_win_s)
        out[s] = r if keep else detect_drops(r, p)
        del env, r
    return out


def detect_session_events(session, p=DropParams()):
    """Events DataFrame for one session (video columns left empty; see video.add_video_columns)."""
    drops = sensor_ratios(session, p)
    events = classify_events(drops, p.overlap)
    rms = emg_rms(io.load_emg(session, 0), p)
    ts = np.array([e["t_start"] for e in events])
    te = np.array([e["t_end"] for e in events])
    burst, peak = emg_burst_flag(rms, ts, te, p) if events else (np.array([], bool), np.array([]))
    return pd.DataFrame({
        "session_id": session, "mouse": io.load_info(session).get("subject", ""),
        "event_id": [f"{session}_{i:04d}" for i in range(len(events))],
        "type": [e["type"] for e in events], "sensors_dropped": ["+".join(e["sensors"]) for e in events],
        "emg_burst": burst, "emg_peak_ratio": np.round(peak, 3),
        "t_start_rec": np.round(ts, 3), "t_end_rec": np.round(te, 3), "duration_s": np.round(te - ts, 3),
    })


def summarize(df):
    """Counts and median duration by type x EMG burst, per session."""
    out = {}
    for sid, g in df.groupby("session_id"):
        out[sid] = {}
        for (typ, burst), h in g.groupby(["type", "emg_burst"]):
            out[sid][f"{typ}|emg_burst={bool(burst)}"] = {"n": int(len(h)), "median_duration_s": float(h.duration_s.median())}
    return out

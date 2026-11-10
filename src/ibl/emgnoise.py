"""Measurements and candidate filters for mains-hum (50 Hz + harmonics) in the EMG.

Analysis helpers only. The filter used by the pipeline lives in ibl.drops (emg_filter / emg_rms) and is
configured through DropParams; the candidates below exist so the choice can be justified in the notebook.
"""
import numpy as np
from scipy import signal as sg
from scipy.ndimage import median_filter

from .config import FS
from .signal import notch_sos  # noqa: F401  (re-exported for the notebook)

LINES = tuple(50.0 * k for k in range(1, 7))        # 50 ... 300 Hz


def welch(x, fs=FS, nperseg=2 ** 15):
    return sg.welch(np.asarray(x, np.float32), fs, nperseg=nperseg, noverlap=nperseg // 2)


def line_mask(f, lines=LINES, halfwidth=1.0):
    m = np.zeros(len(f), bool)
    for k in lines:
        m |= np.abs(f - k) <= halfwidth
    return m


def hum_share(f, P, lines=LINES, halfwidth=1.0, band=(30.0, 300.0)):
    """Fraction of 30-300 Hz power that lies within +-halfwidth Hz of the mains lines."""
    b = (f >= band[0]) & (f <= band[1])
    return float(P[b & line_mask(f, lines, halfwidth)].sum() / P[b].sum())


def hum_share_by_window(x, fs=FS, win_s=60, nperseg=8192, **kw):
    """(window start times in s, hum share) for consecutive windows."""
    n = int(win_s * fs)
    starts = np.arange(0, len(x) - n + 1, n)
    out = []
    for a in starts:
        f, P = welch(x[a:a + n], fs, nperseg)
        out.append(hum_share(f, P, **kw))
    return starts / fs, np.array(out)


def fundamental_by_window(x, fs=FS, win_s=60, harmonic=3, pad=2 ** 18):
    """Mains fundamental (Hz) per window, from the power-weighted peak of one harmonic (finer than the 50 Hz line itself)."""
    n = int(win_s * fs)
    out = []
    w = np.hanning(n).astype(np.float32)
    for a in range(0, len(x) - n + 1, n):
        seg = np.asarray(x[a:a + n], np.float32)
        X = np.abs(np.fft.rfft((seg - seg.mean()) * w, pad)) ** 2
        f = np.fft.rfftfreq(pad, 1 / fs)
        m = np.abs(f - 50.0 * harmonic) <= 1.0
        out.append(float((f[m] * X[m]).sum() / X[m].sum()) / harmonic)
    return np.array(out)


def local_floor(f, P, width_hz=6.0):
    """Running-median spectrum (the broadband floor), used to find lines and to measure collateral loss."""
    return median_filter(P, size=max(3, int(width_hz / (f[1] - f[0]))))


# ------------------------------------------------------------------------------------ candidate filters
def comb_sos(f0=50.0, bw_hz=1.0, fs=FS):
    """IIR comb filter with notches at every multiple of f0 up to Nyquist (scipy.signal.iircomb)."""
    b, a = sg.iircomb(f0, f0 / bw_hz, ftype="notch", fs=fs)
    return sg.tf2sos(b, a)


def spectral_interp(x, lines, halfwidth=1.0, fs=FS, nperseg=8192):
    """STFT filter: within +-halfwidth of each line replace the magnitude by the neighbouring floor (phase kept)."""
    f, _, Z = sg.stft(np.asarray(x, np.float32), fs, nperseg=nperseg)
    mag, ph = np.abs(Z), np.angle(Z)
    df = f[1] - f[0]
    for k in lines:
        i0, i1 = int((k - halfwidth) / df), int((k + halfwidth) / df) + 1
        w = int(4 / df)
        floor = np.median(np.concatenate([mag[i0 - w:i0], mag[i1:i1 + w]], axis=0), axis=0)
        mag[i0:i1] = np.minimum(mag[i0:i1], floor)
    return sg.istft(mag * np.exp(1j * ph), fs, nperseg=nperseg)[1][:len(x)].astype(np.float32)


# ------------------------------------------------------------------------------------------ evaluation
from scipy.ndimage import uniform_filter1d  # noqa: E402
from .signal import sos_bandpass  # noqa: E402

ALL_LINES = tuple(50.0 * k for k in range(1, 10))   # 50 ... 450 Hz


def filt_notch(lines, q):
    sos = notch_sos(lines, q=q)
    return lambda x: sg.sosfiltfilt(sos, np.asarray(x, np.float64)).astype(np.float32)


def filt_comb(q, f0=50.0):
    sos = comb_sos(f0, f0 / q)
    return lambda x: sg.sosfiltfilt(sos, np.asarray(x, np.float64)).astype(np.float32)


def filt_interp(lines, q):
    """Spectral interpolation with the same -3 dB half-bandwidth as a constant-Q notch (f0 / 2q) at each line."""
    def f(x):
        y = np.asarray(x, np.float32)
        for k in lines:                                    # one line at a time keeps the bandwidth per line
            y = spectral_interp(y, [k], halfwidth=max(1.0, k / (2 * q)))
        return y
    return f


def evaluate(x, filt=None, fs=FS, lines=ALL_LINES, band=(30.0, 300.0), rms_win_s=0.25):
    """Run `filt` (None = no hum removal) then the 30-300 Hz band-pass and 0.25 s RMS; return the numbers that matter.

    hum_share: fraction of the final signal's power within +-1 Hz of any 50k line (k=1..9)
    floor_kept: broadband floor (running median of the spectrum, lines excluded) after / before, 30-300 Hz
    """
    y0 = np.asarray(x, np.float32)
    y = y0 if filt is None else filt(y0)
    f, P0 = welch(y0, fs)
    yb = sos_bandpass(y, *band, fs=fs)
    f, P1 = welch(yb, fs)
    pre = sos_bandpass(y0, *band, fs=fs)
    _, Pp = welch(pre, fs)
    b = (f >= band[0]) & (f <= band[1])
    rms = np.sqrt(uniform_filter1d(yb.astype(np.float32) ** 2, int(rms_win_s * fs), mode="nearest"))
    return {"hum_share": hum_share(f, P1, lines, 1.0, (0, fs / 2)),
            "floor_kept": float(np.mean(local_floor(f, P1)[b] / local_floor(f, Pp)[b])),
            "median_rms": float(np.median(rms)), "max_over_median_rms": float(rms.max() / np.median(rms))}


def white_noise_kept(filt, fs=FS, n=600_000, band=(30.0, 300.0), seed=0):
    """Fraction of white-noise power in 30-300 Hz that survives `filt` followed by the band-pass: the cost on broadband muscle activity."""
    w = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    f, P0 = welch(sos_bandpass(w, *band, fs=fs), fs, 4096)
    f, P1 = welch(sos_bandpass(filt(w), *band, fs=fs), fs, 4096)
    b = (f >= band[0]) & (f <= band[1])
    return float(P1[b].sum() / P0[b].sum())


def burst_fidelity(filt, fs=FS, seed=1, gain=6.0, dur_s=0.3):
    """Synthetic muscle burst (noise with `gain` x amplitude for dur_s) in white noise + large hum: peak RMS ratio and half-width of the RMS bump."""
    rng = np.random.default_rng(seed)
    n = 120 * fs
    t = np.arange(n) / fs
    x = rng.standard_normal(n).astype(np.float32) * 5
    sl = slice(60 * fs, int((60 + dur_s) * fs))
    x[sl] *= gain
    x += sum(2000 * np.sin(2 * np.pi * k * 50.0 * t) for k in (1, 3, 5, 7)).astype(np.float32)
    y = sos_bandpass(x if filt is None else filt(x), 30, 300, fs=fs)
    rms = np.sqrt(uniform_filter1d(y ** 2, int(0.25 * fs), mode="nearest"))
    ratio = rms / np.median(rms)
    pk = int(np.argmax(ratio[55 * fs:66 * fs])) + 55 * fs
    half = np.flatnonzero(ratio[55 * fs:66 * fs] > ratio[pk] / 2)
    return {"peak_ratio": float(ratio[pk]), "peak_time_s": pk / fs, "half_width_s": float((half[-1] - half[0]) / fs)}


# ---------------------------------------------------------------------- bursts, threshold, video evidence
import json  # noqa: E402

import pandas as pd  # noqa: E402

from . import io  # noqa: E402
from .config import RESULTS, VIDEO_FILES  # noqa: E402

CACHE = RESULTS / "emg_notch"


def burst_episodes(rms, median, thr=2.5, merge_s=0.5, min_dur_s=0.2, fs=FS):
    """(start, end) in s of stretches where RMS > thr * median; stretches closer than merge_s are joined, shorter than min_dur_s dropped."""
    from .drops import find_runs
    st, en = find_runs(rms > thr * median)
    out = []
    for a, b in zip(st, en):
        if out and a - out[-1][1] < merge_s * fs:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a / fs, b / fs) for a, b in out if (b - a) >= min_dur_s * fs]


def episodes_vs_threshold(rms, ks, fs=FS):
    """Number of merged burst episodes and fraction of time above k * median, for each k."""
    m = np.median(rms)
    return np.array([len(burst_episodes(rms, m, k, fs=fs)) for k in ks]), np.array([(rms > k * m).mean() for k in ks])


def burst_quiet_spectrum(x_notched, rms, fs=FS, nperseg=256, quiet_ratio=1.3, thr=2.5, max_quiet=400):
    """Mean PSD (Welch, 3.9 Hz bins) inside burst episodes and inside quiet stretches; returns f, P_burst, P_quiet."""
    from .drops import find_runs
    m = np.median(rms)
    segs_b = [x_notched[int(a * fs):int(b * fs)] for a, b in burst_episodes(rms, m, thr, fs=fs) if b - a >= 0.3]
    qs, qe = find_runs(rms < quiet_ratio * m)
    segs_q = [x_notched[a:b] for a, b in zip(qs, qe) if b - a >= 2 * fs][:max_quiet]

    def avg(segs):
        out = 0
        for g in segs:
            f, p = sg.welch(g, fs, nperseg=nperseg)
            out = out + p
        return f, out / len(segs)
    f, Pb = avg(segs_b)
    return f, Pb, avg(segs_q)[1], len(segs_b), len(segs_q)


def window_motion(session, t0, t1, camera="left", lag_s=0.3, pre_s=0.5):
    """Mean frame-to-frame pixel change (160x128 grey, decoded by ffmpeg) for frames with recording time in [t0, t1 + lag_s]."""
    from .clips import _read_frames
    from .video import container_fps
    path = VIDEO_FILES[session][camera]
    ct = io.load_cam_times(session, camera)
    f0, f1 = np.searchsorted(ct, [t0 - pre_s, t1 + lag_s + 0.3])
    fr = _read_frames(path, f0 / container_fps(path), f1 - f0, 160, 128).astype(np.float32)
    d = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
    tt = ct[f0 + 1:f0 + 1 + len(d)]
    sel = (tt >= t0) & (tt <= t1 + lag_s)
    return float(d[sel].mean()) if sel.any() else np.nan


def motion_vs_ratio(session, rms, n_random=120, n_high=120, win_s=1.0, seed=0, camera="left"):
    """Video motion and EMG ratio for 1 s windows: n_random uniform in time plus n_high where the ratio exceeds 1.5 (to cover the range)."""
    rng = np.random.default_rng(seed)
    ct = io.load_cam_times(session, camera)
    m = np.median(rms)
    step = int(win_s * FS)
    lo, hi = int((ct[0] + 5) * FS), int((ct[-1] - 5) * FS)
    starts = np.arange(lo, hi - step, step)
    peak = np.array([rms[a:a + step].max() / m for a in starts])
    pick = np.concatenate([rng.choice(len(starts), n_random, replace=False),
                           rng.choice(np.flatnonzero(peak > 1.5), n_high, replace=False)])
    pick = np.unique(pick)
    rows = [{"t0": starts[i] / FS, "ratio": float(peak[i]), "motion": window_motion(session, starts[i] / FS, starts[i] / FS + win_s - 0.3, camera)}
            for i in pick]
    return pd.DataFrame(rows)


def cached(name, fn):
    """Small JSON/CSV cache in results/emg_notch/ for the slow video measurements (delete the file to recompute)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if path.exists():
        return pd.read_csv(path)
    df = fn()
    df.to_csv(path, index=False)
    return df


def line_centroids(f, P, lines=ALL_LINES, halfwidth=0.5):
    """Power-weighted centre frequency of each mains line (+-halfwidth Hz window): exactly k*50.0 if the line does not drift."""
    out = []
    for k in lines:
        m = np.abs(f - k) <= halfwidth
        out.append(float((f[m] * P[m]).sum() / P[m].sum()))
    return np.array(out)


def option_table(sessions, options, excerpt_s=1800, camera=None):
    """One row per (option, session): the numbers used to choose the filter, on the first excerpt_s seconds of EMG ch0."""
    rows = []
    for name, filt in options.items():
        wn = white_noise_kept(filt) if filt else 1.0
        bf = burst_fidelity(filt)
        for s in sessions:
            r = evaluate(io.load_emg(s, 0)[:int(excerpt_s * FS)], filt)
            rows.append({"option": name, "session": s, "white_noise_kept": wn, "synthetic_burst_ratio": bf["peak_ratio"],
                         "synthetic_burst_halfwidth_s": bf["half_width_s"], **r})
    return pd.DataFrame(rows)


def minute_hum_stats(session, filt, win_s=60):
    """Hum share of the final (filtered + band-passed) EMG: whole session, median minute, fraction of minutes above 5 %."""
    x = io.load_emg(session, 0)[:]
    y = sos_bandpass(filt(x) if filt else x, 30, 300)
    f, P = welch(y)
    _, sh = hum_share_by_window(y, lines=ALL_LINES, band=(0, FS / 2), win_s=win_s)
    return {"whole": hum_share(f, P, ALL_LINES, 1.0, (0, FS / 2)), "median_minute": float(np.median(sh)), "frac_minutes_gt5pct": float((sh > 0.05).mean())}


def burst_vs_quiet_motion(session, rms, n=14, seed=0, camera="left"):
    """Video motion during n spread-out EMG burst episodes vs the same number of matched-length quiet control windows."""
    ct = io.load_cam_times(session, camera)
    m = np.median(rms)
    cand = [e for e in burst_episodes(rms, m) if e[1] - e[0] >= 0.3 and ct[0] + 3 < e[0] < ct[-1] - 3]
    pick = [cand[i] for i in np.linspace(0, len(cand) - 1, min(n, len(cand))).astype(int)]
    quiet = rms < 1.3 * m
    ok = np.flatnonzero(quiet) / FS
    ok = ok[(ok > ct[0] + 10) & (ok < ct[-1] - 10)]
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in pick:
        rows.append({"kind": "burst", "t0": a, "dur": b - a, "motion": window_motion(session, a, b, camera)})
        for _ in range(100):
            t = rng.choice(ok)
            if quiet[int((t - 1) * FS):int((t + (b - a) + 1.3) * FS)].all():
                rows.append({"kind": "quiet", "t0": t, "dur": b - a, "motion": window_motion(session, t, t + (b - a), camera)})
                break
    return pd.DataFrame(rows)

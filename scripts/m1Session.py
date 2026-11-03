"""Run sections 3-7 of PipelineStruct_marimo.py on every M1 session.

  3 whole-session overview     4 clean / messy example windows
  5 Welch spectra + coherence  6 phase comparison (clean / messy)
  7 sliding PLV + EMG, session-wide phase lag
LFP channel 0 only. Clean / messy windows are picked per session: the 5-s windows
with the highest / lowest agreement of the weakest sensor pair (the notebook's
CLEAN=295 s and MESSY=90 s were hand-picked for one session).

Usage (from repo root):
  uv run python scripts/m1Session.py                          # all sessions
  uv run python scripts/m1Session.py --sessions 2026-06-18_16-25-09_000 --workers 1
  uv run python scripts/m1Session.py --max-seconds 600        # smoke test
Output: results/m1Session/<session>/{summary.json, *.png}, results/m1Session/across_sessions.png
"""
import os

# Cap CPU use: MAX_WORKERS processes x 1 thread each. Must be set before numpy imports.
MAX_WORKERS = 4
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse, json, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft as sfft, signal
from scipy.ndimage import uniform_filter1d

ROOT = next(p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
            if (p / "pyproject.toml").exists())
DATA = ROOT / "data" / "M1"
FS = 1000
PAIRS = [("Thermistor", "Airflow"), ("Thermistor", "LFP"), ("Airflow", "LFP")]
OUT = ROOT / "results" / "m1Session"
COLORS = {"Thermistor": "#2a78d6", "Airflow": "#eb6834", "LFP": "#1baf7a"}
PAIR_COLORS = ["#eb6834", "#1baf7a", "#6250d6"]
plt.rcParams.update({"figure.figsize": (11, 3.5), "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})

zscore = lambda x: (x - x.mean()) / x.std()
smooth = lambda x, win_s: uniform_filter1d(x, int(win_s * FS), mode="constant")


def bandpass(x, lo, hi, order=3):
    b, a = signal.butter(order, [lo, hi], btype="band", fs=FS)
    return signal.filtfilt(b, a, x)                       # forward-backward: no time shift


def get_phase(x, lo=1, hi=15):
    y = bandpass(x, lo, hi)
    n = len(y)
    return np.angle(signal.hilbert(y, N=sfft.next_fast_len(n))[:n])   # fast FFT length


def sliding_plv(p1, p2, win_s=5, step_s=1):
    """Window-centre times and PLV values (5-s windows, 1-s step, as in the notebook)."""
    W, S = int(win_s * FS), int(step_s * FS)
    c = np.concatenate([[0], np.cumsum(np.exp(1j * (p1 - p2)))])      # O(n) windowed means
    starts = np.arange(0, len(p1) - W, S)
    return (starts + W / 2) / FS, np.abs(c[starts + W] - c[starts]) / W


def pi_ticks(ax):
    ax.set_yticks([-np.pi, 0, np.pi]); ax.set_yticklabels(["-π", "0", "π"])


def plot_window(sensors, t, start, length, title):
    i0, i1 = int(start * FS), int((start + length) * FS)
    fig = plt.figure()
    for k, (name, x) in enumerate(sensors.items()):
        plt.plot(t[i0:i1], zscore(x)[i0:i1] - 5 * k, color=COLORS[name], lw=1, label=name)
    plt.yticks([]); plt.xlabel("time (s)"); plt.title(title, loc="left"); plt.legend(loc="upper right", ncol=3)
    return fig


def compare_phases(phase, t, start, length, title):
    i0, i1 = int(start * FS), int((start + length) * FS)
    tt = t[i0:i1]
    th_starts = np.where(np.diff(phase["Thermistor"][i0:i1]) < -np.pi)[0]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for ax, name in zip(axes[:3], ["Thermistor", "LFP", "Airflow"]):
        p = phase[name][i0:i1]
        n_cycles = round((np.unwrap(p)[-1] - np.unwrap(p)[0]) / (2 * np.pi))
        ax.plot(tt, p, color=COLORS[name], lw=1.2)
        ax.set_title(f"{name} phase: {n_cycles} cycles", loc="left"); pi_ticks(ax)
        for i in th_starts:
            ax.axvline(tt[i], color="gray", ls=":", lw=0.8)
    for name in ["LFP", "Airflow"]:
        diff = np.angle(np.exp(1j * (phase["Thermistor"][i0:i1] - phase[name][i0:i1])))
        axes[3].plot(tt, diff, color=COLORS[name], lw=1.1, label=f"Thermistor - {name}")
    axes[3].set_ylim(-np.pi - 0.2, np.pi + 0.2); pi_ticks(axes[3])
    axes[3].set_title("Phase difference (flat = in step)", loc="left")
    axes[3].legend(loc="lower right"); axes[3].set_xlabel("time (s)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12); fig.tight_layout()
    return fig


def run_session(args):
    """args = (session, max_seconds[, figs]). With a `figs` dict, figures are returned in it
    (for notebooks) instead of being written to disk."""
    name, max_seconds, *rest = args
    figs = rest[0] if rest else None
    t0 = time.perf_counter()
    d = DATA / name / "alf"
    out = OUT / (f"{name}.smoke{int(max_seconds)}" if max_seconds else name)
    if figs is None:
        out.mkdir(parents=True, exist_ok=True)
    n = int(np.load(d / "ephys.nSamples.npy"))
    if max_seconds:
        n = min(n, int(max_seconds * FS))
    load = lambda f: np.asarray(np.load(d / f, mmap_mode="r")[:n], float)
    sensors = {"Thermistor": load("_cavelli_thermistor.signal.npy"),
               "Airflow": load("_cavelli_airflow.signal.npy"),
               "LFP": np.asarray(np.load(d / "_cavelli_lfp.signal.npy", mmap_mode="r")[:n, 0], float)}
    emg = np.asarray(np.load(d / "_cavelli_emg.signal.npy", mmap_mode="r")[:n, 0], float)
    cam = np.load(d / "leftCamera.times.npy")
    t = np.arange(n) / FS
    res = {"session": name, "duration_s": n / FS, "video_s": [float(cam[0]), float(cam[-1])]}

    def save(fig, fname):
        if figs is not None:
            figs[fname] = fig
            return
        fig.savefig(out / fname, dpi=100, bbox_inches="tight"); plt.close(fig)

    # ---- phase + PLV first: needed to pick the clean / messy windows ----
    phase = {k: get_phase(x) for k, x in sensors.items()}
    plv = {}
    for a, b in PAIRS:
        centers, plv[(a, b)] = sliding_plv(phase[a], phase[b])
    score = np.min([v for v in plv.values()], axis=0)                 # weakest pair per window
    CLEAN = float(centers[np.argmax(score)] - 2.5)
    MESSY = float(centers[np.argmin(score)] - 2.5)
    res["clean_start_s"], res["messy_start_s"] = CLEAN, MESSY

    # ---- 3. whole session ----
    step = max(1, n // 1_000_000)                                     # decimate for plotting only
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    for ax, (k, x) in zip(axes, sensors.items()):
        ax.plot(t[::step], zscore(x)[::step], color=COLORS[k], lw=0.3); ax.set_ylabel(k); ax.set_ylim(-6, 6)
    axes[3].plot(t[::step], zscore(emg)[::step], color="0.3", lw=0.3)
    axes[3].set_ylabel("EMG"); axes[3].set_xlabel("time (s)")
    for ax in axes:
        ax.axvline(cam[0], color="gray", ls="--", lw=0.8); ax.axvline(cam[-1], color="gray", ls="--", lw=0.8)
        ax.axvspan(CLEAN, CLEAN + 5, color="gold", alpha=0.4); ax.axvspan(MESSY, MESSY + 5, color="0.6", alpha=0.4)
    axes[0].set_title(f"{name}: whole session (dashed = video start/end)", loc="left")
    fig.tight_layout(); save(fig, "3_whole_session.png")

    # ---- 4. clean / messy windows ----
    c0, m0 = max(0.0, CLEAN - 2.5), MESSY
    save(plot_window(sensors, t, c0, 10, f"Clean: {c0:.1f}-{c0 + 10:.1f} s"), "4_clean_window.png")
    save(plot_window(sensors, t, m0, 10, f"Messy: {m0:.1f}-{m0 + 10:.1f} s"), "4_messy_window.png")

    # ---- 5. Welch spectra + coherence ----
    fig = plt.figure(); res["welch_peak_hz"] = {}
    for k, x in sensors.items():
        f, p = signal.welch(x, FS, nperseg=8192); keep = f < 25
        plt.semilogy(f[keep], p[keep] / p[keep].max(), color=COLORS[k], label=k)
        res["welch_peak_hz"][k] = float(f[keep][np.argmax(p[keep])])
    plt.axvspan(1, 15, color="gray", alpha=0.1, label="1-15 Hz")
    plt.xlabel("frequency (Hz)"); plt.ylabel("power (normalized, log)"); plt.legend()
    plt.title(f"{name}: Welch spectra", loc="left"); save(fig, "5_welch.png")

    fig = plt.figure(figsize=(11, 4)); res["coherence"] = {}
    for (a, b), c in zip(PAIRS, PAIR_COLORS):
        f, C = signal.coherence(sensors[a], sensors[b], FS, nperseg=4096)
        _, C0 = signal.coherence(sensors[a], np.roll(sensors[b], 60 * FS), FS, nperseg=4096)   # chance level
        k = f <= 20
        plt.plot(f[k], C[k], color=c, lw=1.6, label=f"{a} - {b}"); plt.plot(f[k], C0[k], color=c, lw=0.8, ls="--")
        band = lambda lo, hi: float(C[(f >= lo) & (f < hi)].mean())
        sel = (f >= 0.5) & (f <= 15)
        res["coherence"][f"{a}-{b}"] = {"1-3Hz": band(1, 3), "3-6Hz": band(3, 6), "6-15Hz": band(6, 15),
                                        "peak_hz": float(f[sel][np.argmax(C[sel])]),
                                        "chance_1-15Hz": float(C0[(f >= 1) & (f < 15)].mean())}
    plt.axvspan(1, 15, color="gray", alpha=0.08, label="filter band (1-15 Hz)"); plt.ylim(0, 1)
    plt.xlabel("frequency (Hz)"); plt.ylabel("coherence")
    plt.title(f"{name}: coherence (dashed = chance, one sensor shifted 60 s)", loc="left")
    plt.legend(loc="upper right", fontsize=9); save(fig, "5_coherence.png")

    # ---- 6. phase comparison ----
    save(compare_phases(phase, t, CLEAN, 5, f"Clean: {CLEAN:.1f}-{CLEAN + 5:.1f} s"), "6_phase_clean.png")
    save(compare_phases(phase, t, MESSY, 5, f"Messy: {MESSY:.1f}-{MESSY + 5:.1f} s"), "6_phase_messy.png")

    # ---- 7. PLV over time + EMG, phase lag ----
    emg_power = np.sqrt(smooth(bandpass(emg, 30, 300) ** 2, 5))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    res["plv"], res["lag"] = {}, {}
    for (a, b), c in zip(PAIRS, PAIR_COLORS):
        v = plv[(a, b)]
        axes[0].plot(centers, v, color=c, label=f"{a} - {b} (median {np.median(v):.2f})")
        lag = float(np.angle(np.mean(np.exp(1j * (phase[a] - phase[b])))))
        res["plv"][f"{a}-{b}"] = {"median": float(np.median(v)), "frac_above_0.7": float(np.mean(v > 0.7))}
        res["lag"][f"{a}-{b}"] = {"rad": lag, "deg": float(np.degrees(lag)), "cycle": lag / (2 * np.pi)}
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("PLV")
    axes[0].set_title(f"{name}: sensor agreement over time (5-s windows)", loc="left")
    axes[1].plot(t[::50], emg_power[::50], color="0.3"); axes[1].set_ylabel("EMG power")
    axes[1].set_xlabel("time (s)"); axes[1].set_title("Muscle activity (EMG)", loc="left")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1]); save(fig, "7_plv_emg.png")

    res["elapsed_s"] = time.perf_counter() - t0
    if figs is None:
        (out / "summary.json").write_text(json.dumps(res, indent=2))
    return res


def summarise(results):
    print(f"\n{'session':26s} {'dur_s':>6s}  PLV T-A  T-L  A-L | lag T-A  T-L  A-L (rad) | Welch peak T/A/L (Hz) | CLEAN  MESSY")
    for r in results:
        p = [r["plv"][f"{a}-{b}"]["median"] for a, b in PAIRS]
        l = [r["lag"][f"{a}-{b}"]["rad"] for a, b in PAIRS]
        w = r["welch_peak_hz"]
        print(f"{r['session']:26s} {r['duration_s']:6.0f}  {p[0]:7.2f} {p[1]:4.2f} {p[2]:4.2f} | "
              f"{l[0]:+6.2f} {l[1]:+5.2f} {l[2]:+5.2f}      | {w['Thermistor']:4.1f}/{w['Airflow']:4.1f}/{w['LFP']:4.1f}"
              f"            | {r['clean_start_s']:5.0f} {r['messy_start_s']:6.0f}")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(results))
    for k, ((a, b), c) in enumerate(zip(PAIRS, PAIR_COLORS)):
        ax.bar(x + (k - 1) * 0.27, [r["plv"][f"{a}-{b}"]["median"] for r in results], 0.27, color=c, label=f"{a} - {b}")
    ax.set_xticks(x); ax.set_xticklabels([r["session"][:10] + r["session"][-4:] for r in results], fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("median PLV"); ax.legend(fontsize=8)
    ax.set_title("Sensor agreement across sessions (1-15 Hz)", loc="left")
    fig.savefig(OUT / "across_sessions.png", dpi=100, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", help="default: all under data/M1")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--max-seconds", type=float, default=0, help="crop each session (smoke test)")
    a = ap.parse_args()
    names = a.sessions or sorted(p.name for p in DATA.iterdir() if (p / "alf").is_dir())
    workers = max(1, min(a.workers, MAX_WORKERS, len(names)))
    names.sort(key=lambda s: -int(np.load(DATA / s / "alf" / "ephys.nSamples.npy")))   # longest first
    print(f"{len(names)} sessions, {workers} worker(s), 1 thread each", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(run_session, [(s, a.max_seconds) for s in names]))
    print(f"wall time {time.perf_counter() - t0:.1f}s")
    summarise(sorted(results, key=lambda r: r["session"]))


if __name__ == "__main__":
    main()

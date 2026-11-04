"""Run sections 3-10 of PipelineStruct_marimo.py on every M1 session.

  3 whole-session overview     4 clean / messy example windows
  5 Welch spectra + coherence  6 phase comparison (clean / messy)
  7 sliding PLV + EMG, phase lag          8 possible breathing pause
  9 LFP channel choice (PLV bars + correlation matrix)
  10 filter-band sweep
Sensors use LFP channel 0; section 9 looks at all channels. Clean / messy windows are picked
per session: the 5-s windows with the highest / lowest agreement of the weakest sensor pair
(the notebook's CLEAN=295 s and MESSY=90 s were hand-picked for one session). The pause window
is the deepest joint amplitude drop of all three sensors while EMG stays near typical.

Structure:  analyze_session() -> (D, R)   all computation: arrays D for plotting, JSON-able summary R
            plot_*(D, R) -> Figure         one function per figure, listed in PLOTS (12 per session)
            run_session()                  analyze, then render every entry of PLOTS

Usage (from repo root):
  uv run python scripts/m1Session.py                          # all sessions
  uv run python scripts/m1Session.py --sessions 2026-06-18_16-25-09_000 --workers 1
  uv run python scripts/m1Session.py --max-seconds 600        # smoke test
Output: results/m1Session/<session>/{summary.json, 12 *.png}, results/m1Session/across_sessions.png
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


def analytic(y):
    """Hilbert analytic signal of an already band-passed y (fast FFT length)."""
    n = len(y)
    return signal.hilbert(y, N=sfft.next_fast_len(n))[:n]


def get_phase(x, lo=1, hi=15):
    return np.angle(analytic(bandpass(x, lo, hi)))


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


BANDS = [(1, 15), (1, 10), (1, 8), (1, 6), (0.8, 5)]
PAUSE_LEN_S = 15


def analyze_session(name, max_seconds=0):
    """All computation. Returns (D, R): D = arrays used by the plot functions, R = JSON-able summary."""
    d = DATA / name / "alf"
    n = int(np.load(d / "ephys.nSamples.npy"))
    if max_seconds:
        n = min(n, int(max_seconds * FS))
    lfp_mm = np.load(d / "_cavelli_lfp.signal.npy", mmap_mode="r")
    load = lambda f: np.asarray(np.load(d / f, mmap_mode="r")[:n], float)
    sensors = {"Thermistor": load("_cavelli_thermistor.signal.npy"),
               "Airflow": load("_cavelli_airflow.signal.npy"),
               "LFP": np.asarray(lfp_mm[:n, 0], float)}                 # channel 0
    emg = np.asarray(np.load(d / "_cavelli_emg.signal.npy", mmap_mode="r")[:n, 0], float)
    cam = np.load(d / "leftCamera.times.npy")
    t = np.arange(n) / FS
    D = {"name": name, "n": n, "t": t, "sensors": sensors, "emg": emg, "cam": cam}
    R = {"session": name, "duration_s": n / FS, "video_s": [float(cam[0]), float(cam[-1])]}

    # ---- phase + PLV (1-15 Hz); also picks the clean / messy windows ----
    z = {k: analytic(bandpass(x, 1, 15)) for k, x in sensors.items()}
    phase = {k: np.angle(v) for k, v in z.items()}
    plv = {}
    for a, b in PAIRS:
        centers, plv[(a, b)] = sliding_plv(phase[a], phase[b])
    score = np.min(list(plv.values()), axis=0)                          # weakest pair per window
    CLEAN = float(centers[np.argmax(score)] - 2.5)
    MESSY = float(centers[np.argmin(score)] - 2.5)
    R["clean_start_s"], R["messy_start_s"] = CLEAN, MESSY
    D.update(phase=phase, plv=plv, centers=centers, CLEAN=CLEAN, MESSY=MESSY)

    # ---- 5. Welch + coherence ----
    D["welch"], R["welch_peak_hz"] = {}, {}
    for k, x in sensors.items():
        f, p = signal.welch(x, FS, nperseg=8192); keep = f < 25
        D["welch"][k] = (f[keep], p[keep] / p[keep].max())
        R["welch_peak_hz"][k] = float(f[keep][np.argmax(p[keep])])
    D["coh"], R["coherence"] = {}, {}
    for a, b in PAIRS:
        f, C = signal.coherence(sensors[a], sensors[b], FS, nperseg=4096)
        _, C0 = signal.coherence(sensors[a], np.roll(sensors[b], 60 * FS), FS, nperseg=4096)   # chance level
        D["coh"][(a, b)] = (f, C, C0)
        band = lambda lo, hi: float(C[(f >= lo) & (f < hi)].mean())
        sel = (f >= 0.5) & (f <= 15)
        R["coherence"][f"{a}-{b}"] = {"1-3Hz": band(1, 3), "3-6Hz": band(3, 6), "6-15Hz": band(6, 15),
                                      "peak_hz": float(f[sel][np.argmax(C[sel])]),
                                      "chance_1-15Hz": float(C0[(f >= 1) & (f < 15)].mean())}

    # ---- 7. PLV summary, lag, EMG power ----
    R["plv"], R["lag"] = {}, {}
    for a, b in PAIRS:
        v = plv[(a, b)]
        lag = float(np.angle(np.mean(np.exp(1j * (phase[a] - phase[b])))))
        R["plv"][f"{a}-{b}"] = {"median": float(np.median(v)), "frac_above_0.7": float(np.mean(v > 0.7))}
        R["lag"][f"{a}-{b}"] = {"rad": lag, "deg": float(np.degrees(lag)), "cycle": lag / (2 * np.pi)}
    emg_hf = bandpass(emg, 30, 300)
    D["emg_power"] = np.sqrt(smooth(emg_hf ** 2, 5))

    # ---- 8. possible breathing pause: all three amplitudes drop together while EMG stays typical ----
    emg_fine = np.sqrt(smooth(emg_hf ** 2, 0.25))
    D["emg_fine"], D["emg_coarse"] = emg_fine, D["emg_power"]
    D["amp"] = {k: smooth(np.abs(v), 0.5) for k, v in z.items()}
    joint = np.max([D["amp"][k] / np.median(D["amp"][k]) for k in sensors], axis=0)   # all three low <=> max low
    joint[emg_fine / np.median(emg_fine) > 2] = np.inf                  # skip movement bursts
    edge = 10 * FS
    joint[:edge], joint[-edge:] = np.inf, np.inf
    pc = int(np.argmin(joint)) / FS if np.isfinite(joint).any() else n / FS / 2
    D["pause_start"] = float(np.clip(pc - PAUSE_LEN_S / 2, 0, max(0, n / FS - PAUSE_LEN_S)))
    R["pause_center_s"] = float(pc)
    del z

    # ---- 9. LFP channel choice: one channel at a time to bound memory ----
    nch = lfp_mm.shape[1]
    bp = np.empty((n, nch), np.float32)
    vs_th, vs_af = [], []
    for c in range(nch):
        y = bandpass(np.asarray(lfp_mm[:n, c], float), 1, 15)
        bp[:, c] = y
        ph = np.angle(analytic(y))
        vs_th.append(float(np.median(sliding_plv(ph, phase["Thermistor"])[1])))
        vs_af.append(float(np.median(sliding_plv(ph, phase["Airflow"])[1])))
    D["lfp_vs_th"], D["lfp_vs_af"] = vs_th, vs_af
    D["lfp_corr"] = np.corrcoef(bp.T)
    R["lfp_channels"] = {"plv_vs_thermistor": vs_th, "plv_vs_airflow": vs_af,
                         "best_channel": int(np.argmax(np.add(vs_th, vs_af)))}
    del bp

    # ---- 10. filter-band sweep ----
    D["bands"] = {f"{lo}-{hi}": {} for lo, hi in BANDS}
    for lo, hi in BANDS:
        ph = phase if (lo, hi) == (1, 15) else {k: get_phase(x, lo, hi) for k, x in sensors.items()}
        for a, b in PAIRS:
            D["bands"][f"{lo}-{hi}"][(a, b)] = float(np.median(sliding_plv(ph[a], ph[b])[1]))
    R["bands"] = {bk: {f"{a}-{b}": v for (a, b), v in row.items()} for bk, row in D["bands"].items()}
    return D, R


# ======================================================================
# Plot functions: one per figure, each takes (D, R) from analyze_session
# ======================================================================
def plot_whole_session(D, R):
    t, n, cam = D["t"], D["n"], D["cam"]
    step = max(1, n // 1_000_000)                                       # decimate for plotting only
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    for ax, (k, x) in zip(axes, D["sensors"].items()):
        ax.plot(t[::step], zscore(x)[::step], color=COLORS[k], lw=0.3); ax.set_ylabel(k); ax.set_ylim(-6, 6)
    axes[3].plot(t[::step], zscore(D["emg"])[::step], color="0.3", lw=0.3)
    axes[3].set_ylabel("EMG"); axes[3].set_xlabel("time (s)")
    for ax in axes:
        ax.axvline(cam[0], color="gray", ls="--", lw=0.8); ax.axvline(cam[-1], color="gray", ls="--", lw=0.8)
        ax.axvspan(D["CLEAN"], D["CLEAN"] + 5, color="gold", alpha=0.4)
        ax.axvspan(D["MESSY"], D["MESSY"] + 5, color="0.6", alpha=0.4)
    axes[0].set_title(f"{D['name']}: whole session (dashed = video start/end)", loc="left")
    fig.tight_layout()
    return fig


def plot_clean_window(D, R):
    c0 = max(0.0, D["CLEAN"] - 2.5)
    return plot_window(D["sensors"], D["t"], c0, 10, f"Clean: {c0:.1f}-{c0 + 10:.1f} s")


def plot_messy_window(D, R):
    m0 = D["MESSY"]
    return plot_window(D["sensors"], D["t"], m0, 10, f"Messy: {m0:.1f}-{m0 + 10:.1f} s")


def plot_welch(D, R):
    fig = plt.figure()
    for k, (f, p) in D["welch"].items():
        plt.semilogy(f, p, color=COLORS[k], label=k)
    plt.axvspan(1, 15, color="gray", alpha=0.1, label="1-15 Hz")
    plt.xlabel("frequency (Hz)"); plt.ylabel("power (normalized, log)"); plt.legend()
    plt.title(f"{D['name']}: Welch spectra", loc="left")
    return fig


def plot_coherence(D, R):
    fig = plt.figure(figsize=(11, 4))
    for (a, b), c in zip(PAIRS, PAIR_COLORS):
        f, C, C0 = D["coh"][(a, b)]
        k = f <= 20
        plt.plot(f[k], C[k], color=c, lw=1.6, label=f"{a} - {b}"); plt.plot(f[k], C0[k], color=c, lw=0.8, ls="--")
    plt.axvspan(1, 15, color="gray", alpha=0.08, label="filter band (1-15 Hz)"); plt.ylim(0, 1)
    plt.xlabel("frequency (Hz)"); plt.ylabel("coherence")
    plt.title(f"{D['name']}: coherence (dashed = chance, one sensor shifted 60 s)", loc="left")
    plt.legend(loc="upper right", fontsize=9)
    return fig


def plot_phase_clean(D, R):
    c = D["CLEAN"]
    return compare_phases(D["phase"], D["t"], c, 5, f"Clean: {c:.1f}-{c + 5:.1f} s")


def plot_phase_messy(D, R):
    m = D["MESSY"]
    return compare_phases(D["phase"], D["t"], m, 5, f"Messy: {m:.1f}-{m + 5:.1f} s")


def plot_plv_emg(D, R):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for (a, b), c in zip(PAIRS, PAIR_COLORS):
        v = D["plv"][(a, b)]
        axes[0].plot(D["centers"], v, color=c, label=f"{a} - {b} (median {np.median(v):.2f})")
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("PLV")
    axes[0].set_title(f"{D['name']}: sensor agreement over time (5-s windows)", loc="left")
    axes[1].plot(D["t"][::50], D["emg_power"][::50], color="0.3"); axes[1].set_ylabel("EMG power")
    axes[1].set_xlabel("time (s)"); axes[1].set_title("Muscle activity (EMG)", loc="left")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def plot_pause(D, R):
    t = D["t"]
    i0 = int(D["pause_start"] * FS); i1 = min(D["n"], i0 + PAUSE_LEN_S * FS)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for k, a in D["amp"].items():
        axes[0].plot(t[i0:i1], a[i0:i1] / np.median(a), color=COLORS[k], label=k)
    axes[0].axhline(1, color="0.5", lw=0.8, ls=":"); axes[0].set_yscale("log")
    axes[0].set_ylabel("amplitude / typical"); axes[0].legend(ncol=3)
    axes[0].set_title(f"Breathing-signal strength (1 = typical), possible pause near {R['pause_center_s']:.1f} s", loc="left")
    ef, ec = D["emg_fine"], D["emg_coarse"]
    axes[1].plot(t[i0:i1], ef[i0:i1] / np.median(ef), color="0.2", lw=1, label="0.25 s smoothing")
    axes[1].plot(t[i0:i1], ec[i0:i1] / np.median(ec), color="#e34948", lw=2, label="5 s smoothing")
    axes[1].axhline(1, color="0.5", lw=0.8, ls=":"); axes[1].set_ylabel("EMG / typical")
    axes[1].set_xlabel("time (s)"); axes[1].legend(); axes[1].set_title("Muscle activity", loc="left")
    fig.tight_layout()
    return fig


def plot_lfp_channels(D, R):
    x = np.arange(len(D["lfp_vs_th"]))
    fig = plt.figure(figsize=(9, 3.8))
    plt.bar(x - 0.2, D["lfp_vs_th"], 0.4, color=COLORS["Thermistor"], label="vs thermistor")
    plt.bar(x + 0.2, D["lfp_vs_af"], 0.4, color=COLORS["Airflow"], label="vs airflow")
    plt.xticks(x); plt.xlabel("LFP channel"); plt.ylabel("median PLV"); plt.legend()
    plt.title("Breathing agreement per LFP channel (1-15 Hz)", loc="left")
    return fig


def plot_lfp_corr(D, R):
    fig = plt.figure(figsize=(5, 4))
    plt.imshow(D["lfp_corr"], cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(label="correlation"); plt.xlabel("channel"); plt.ylabel("channel"); plt.grid(False)
    plt.title("LFP channel correlation, 1-15 Hz", loc="left")
    return fig


def plot_filter_band(D, R):
    labels = list(D["bands"])
    fig = plt.figure(figsize=(8, 3.8))
    for (a, b), c in zip(PAIRS, PAIR_COLORS):
        plt.plot(labels, [D["bands"][bk][(a, b)] for bk in labels], "o-", color=c, label=f"{a}-{b}")
    plt.xlabel("filter band (Hz)"); plt.ylabel("median PLV"); plt.ylim(0, 1); plt.legend()
    plt.title("Agreement vs filter band", loc="left")
    return fig


PLOTS = [("3_whole_session.png", plot_whole_session),
         ("4_clean_window.png", plot_clean_window), ("4_messy_window.png", plot_messy_window),
         ("5_welch.png", plot_welch), ("5_coherence.png", plot_coherence),
         ("6_phase_clean.png", plot_phase_clean), ("6_phase_messy.png", plot_phase_messy),
         ("7_plv_emg.png", plot_plv_emg), ("8_pause.png", plot_pause),
         ("9_lfp_channels.png", plot_lfp_channels), ("9_lfp_corr.png", plot_lfp_corr),
         ("10_filter_band.png", plot_filter_band)]
assert len(PLOTS) < 20


def run_session(args):
    """args = (session, max_seconds[, figs]). With a `figs` dict, figures are returned in it
    (for notebooks) instead of being written to disk."""
    name, max_seconds, *rest = args
    figs = rest[0] if rest else None
    t0 = time.perf_counter()
    D, R = analyze_session(name, max_seconds)
    out = OUT / (f"{name}.smoke{int(max_seconds)}" if max_seconds else name)
    if figs is None:
        out.mkdir(parents=True, exist_ok=True)
    for fname, fn in PLOTS:
        fig = fn(D, R)
        if figs is not None:
            figs[fname] = fig
        else:
            fig.savefig(out / fname, dpi=100, bbox_inches="tight"); plt.close(fig)
    R["elapsed_s"] = time.perf_counter() - t0
    if figs is None:
        (out / "summary.json").write_text(json.dumps(R, indent=2))
    return R


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

"""Step 5: do sensor-agreement findings from session _000 hold across sessions?

For each recording with local signals: band-pass 1-15 Hz, Hilbert phase,
PLV over 5-s windows (1-s step) for each sensor pair.

Chance level: airflow circularly shifted by 20 random amounts (>= 30 s) and
compared with thermistor; the 95th percentile of those PLVs is the threshold.
Negative control: EKG-airflow PLV (is airflow dominated by the heartbeat?).
Read-only: never writes into data/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import fft, signal

FS = 1000
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def bandpass(x, lo=1, hi=15, order=3):
    b, a = signal.butter(order, [lo, hi], btype="band", fs=FS) # pyright: ignore
    return signal.filtfilt(b, a, x)


def get_phase(x):
    n = len(x)
    # Pad the FFT to a "fast" length: hilbert on an awkward length (e.g. a
    # large prime) can be very slow. Then cut back to the original length.
    analytic = signal.hilbert(bandpass(x), N=fft.next_fast_len(n))[:n]
    return np.angle(analytic) # pyright: ignore


def sliding_plv(p1, p2, win_s=5, step_s=1):
    # Same as before, but fast: a running sum gives every window's mean at once.
    z = np.exp(1j * (p1 - p2))
    cs = np.concatenate([[0], np.cumsum(z)])
    W, S = win_s * FS, step_s * FS
    starts = np.arange(0, len(z) - W, S)
    return np.abs(cs[starts + W] - cs[starts]) / W


def load(alf, name, channel=None):
    x = np.load(alf / f"_cavelli_{name}.signal.npy", mmap_mode="r")
    if channel is not None:
        x = x[:, channel]
    return np.asarray(x, dtype=np.float64)


rows = []
for alf in sorted(DATA.glob("M*/*/alf")):
    subject, session = alf.parent.parent.name, alf.parent.name
    meta = json.loads((DATA / "all_metadata" / f"{subject}_{session}.json").read_text())
    reref = bool(meta["processing"]["rereferencing"].get("channels_rereferenced"))

    th = get_phase(load(alf, "thermistor"))
    af = get_phase(load(alf, "airflow"))
    lfp = get_phase(load(alf, "lfp", channel=0))   # channel 0 = CH9, most anterior
    ekg = get_phase(load(alf, "ekg", channel=0))

    # Chance level: 20 random circular shifts of airflow (at least 30 s), fixed seed
    rng = np.random.default_rng(0)
    shifts = rng.integers(30 * FS, len(af) - 30 * FS, size=20)
    chance95 = np.percentile(
        np.concatenate([sliding_plv(th, np.roll(af, s)) for s in shifts]), 95)

    plv_th_af = sliding_plv(th, af)
    rows.append({
        "subject": subject,
        "session": session,
        "hours": round(len(th) / FS / 3600, 2),
        "lfp_reref": reref,
        "th_af": np.median(plv_th_af),
        "th_lfp": np.median(sliding_plv(th, lfp)),
        "af_lfp": np.median(sliding_plv(af, lfp)),
        "chance95": chance95,
        "frac_th_af": np.mean(plv_th_af > chance95),   # fraction of windows above chance
        "ekg_af": np.median(sliding_plv(ekg, af)),     # negative control: should be low
    })
    print(f"done {subject} {session}")

df = pd.DataFrame(rows).round(2)
print()
print(df.to_string(index=False))

pairs = ["th_af", "th_lfp", "af_lfp"]
df["all_above_chance"] = df[pairs].min(axis=1) > df["chance95"]
df["ranking_holds"] = (df["th_af"] > df["th_lfp"]) & (df["th_lfp"] > df["af_lfp"])
print(f"\nPrediction 1 (all pairs above chance): {df['all_above_chance'].sum()}/{len(df)} sessions")
print(f"Prediction 2 (ranking holds):          {df['ranking_holds'].sum()}/{len(df)} sessions")
print("Prediction 3 (median PLV by LFP processing):")
print(df.groupby("lfp_reref")[["th_lfp", "af_lfp"]].median().round(2).to_string())
print(f"Prediction 4 (EKG-airflow near chance): "
      f"{(df['ekg_af'] <= df['chance95']).sum()}/{len(df)} sessions at or below chance")

out = ROOT / "results" / "agreement_sessions.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"\nSaved {out}")
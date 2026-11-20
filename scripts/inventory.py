"""Inventory of respiratory sensors for all 33 sessions.

Reads metadata from data/all_metadata/ (one session_info.json per session,
named <subject>_<session_id>.json). Where the session's signals exist locally
(data/<subject>/<session_id>/alf/), also checks metadata against the signal.
Read-only: never writes into data/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SENSORS = ["airflow", "thermistor", "chest"]
SHORT = {"airflow": "A", "thermistor": "T", "chest": "C"}

rows = []
for meta_path in sorted((DATA / "all_metadata").glob("*.json")):
    meta = json.loads(meta_path.read_text())
    subject, session = meta.get("subject"), meta.get("session_id")
    alf = DATA / subject / session / "alf"           # only exists for sessions we downloaded
    has_signals = alf.exists()

    avail = meta.get("channels", {}).get("respiration_available", {})
    n = meta.get("processing", {}).get("processed_n_samples")

    row = {
        "subject": subject,
        "session": session,
        "hours": round(n / 1000 / 3600, 2) if n else None,
        "sensors": "+".join(SHORT[s] for s in SENSORS if avail.get(s)) or "none",
        "n_videos": len(meta.get("derived_files", {}).get("raw_video_data", []) or []),
        "signals_local": has_signals,
                "n_lfp": len(meta.get("channels", {}).get("lfp_channels", []) or []),
        "lfp_reref_to": meta.get("processing", {}).get("rereferencing", {}).get("reference_channel")
                        if meta.get("processing", {}).get("rereferencing", {}).get("channels_rereferenced") else None,
    }

    for s in SENSORS:
        claimed = bool(avail.get(s, False))
        row[f"{s}_meta"] = claimed
        f = alf / f"_cavelli_{s}.signal.npy"
        if has_signals and f.exists():
            x = np.load(f, mmap_mode="r")
            flat = float(np.max(np.abs(x))) == 0.0
            row[f"{s}_MISMATCH"] = (claimed == flat)
        else:
            row[f"{s}_MISMATCH"] = None                # not checked: no local signal
    rows.append(row)

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)
print(df[["subject", "session", "hours", "sensors", "n_videos", "signals_local", "n_lfp", "lfp_reref_to"]].to_string(index=False))

# --- Summaries -----------------------------------------------------------
usable = df["airflow_meta"] & df["thermistor_meta"]   # what fusion needs (+ LFP)
print(f"\nSessions: {len(df)}, total hours: {df['hours'].sum():.2f}")
print(f"Usable for fusion (airflow + thermistor): {usable.sum()} sessions, "
      f"{df.loc[usable, 'hours'].sum():.2f} h")

print("\nPer mouse (usable sessions only):")
per_mouse = df[usable].groupby("subject").agg(sessions=("session", "count"),
                                              hours=("hours", "sum"))
print(per_mouse.round(2).to_string())

print("\nSensor combinations:")
print(df["sensors"].value_counts().to_string())

mismatch_cols = [f"{s}_MISMATCH" for s in SENSORS]
checked = df[mismatch_cols].notna().any(axis=1)
print(f"\nSignal check: {checked.sum()} sessions checked, "
      f"mismatches: {df.loc[checked, mismatch_cols].fillna(False).any(axis=None)}")

out = ROOT / "results" / "inventory_all.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"\nSaved {out}")
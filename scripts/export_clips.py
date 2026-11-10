"""Render MP4 clips (video + traces with cursor) into results/clips/ to share examples.

Usage: uv run python scripts/export_clips.py --session 2026-06-18_16-25-09_000 [--type ALL_THREE] [--emg 0|1] [--limit 5]
"""
import argparse

import pandas as pd

from ibl.clips import export_session_clips
from ibl.config import DROP_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True)
ap.add_argument("--type")
ap.add_argument("--emg", type=int, choices=[0, 1])
ap.add_argument("--limit", type=int, default=5)
ap.add_argument("--half", type=float, default=5.0)
a = ap.parse_args()
ev = pd.read_csv(DROP_DIR / "drop_events.csv")
if a.type:
    ev = ev[ev.type == a.type]
if a.emg is not None:
    ev = ev[ev.emg_burst == bool(a.emg)]
for p in export_session_clips(ev, a.session, half=a.half, limit=a.limit):
    print(p)

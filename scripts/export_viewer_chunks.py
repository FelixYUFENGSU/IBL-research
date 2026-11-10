"""Write results/viewer_chunks/ (per-event 250 Hz JSON + index.json) and the viewer/video/ symlinks.

Usage: uv run python scripts/export_viewer_chunks.py [--sessions S ...]   (run run_drop_events.py first)
"""
import argparse
import time

import pandas as pd

from ibl.config import DROP_DIR
from ibl.viewer import export_event_chunks, link_videos

ap = argparse.ArgumentParser()
ap.add_argument("--sessions", nargs="*")
a = ap.parse_args()
ev = pd.read_csv(DROP_DIR / "drop_events.csv")
link_videos()
t = time.time()
idx = export_event_chunks(ev, sessions=a.sessions)
print(f"{len(idx)} events, {time.time() - t:.0f} s")

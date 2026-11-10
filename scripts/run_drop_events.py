"""Detect signal drops in every session and write results/drop_events/{drop_events.csv, summary.json}.

Usage (repo root):  uv run python scripts/run_drop_events.py [--sessions S1 S2 ...]
Read-only on data/ and videos. Video columns are filled only for sessions whose video frame count equals
len(cam_t); a mismatch stops the run.
"""
import argparse
import json
import resource
import time

import pandas as pd

from ibl import io
from ibl.config import DROP_DIR, SESSIONS, VIDEO_FILES
from ibl.drops import DropParams, detect_session_events, summarize
from ibl.video import add_video_columns, check_frame_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=SESSIONS)
    args = ap.parse_args()
    p = DropParams()
    DROP_DIR.mkdir(parents=True, exist_ok=True)
    frames, checks, parts = [], {}, []
    for s in args.sessions:
        t0 = time.time()
        df = detect_session_events(s, p)
        checks[s] = {cam: {"video_frames": a, "len_cam_t": b} for cam, (a, b) in check_frame_counts(s).items()}  # raises on mismatch
        parts.append(df)
        print(f"{s}: {len(df)} events, {time.time() - t0:.0f} s, video check: {checks[s] or 'no video'}", flush=True)
    df = add_video_columns(pd.concat(parts, ignore_index=True), "left")
    df.to_csv(DROP_DIR / "drop_events.csv", index=False)
    summary = {"params": p.__dict__, "counts": summarize(df), "frame_check": checks,
               "sessions_without_video": [s for s in args.sessions if not VIDEO_FILES.get(s)],
               "n_events": df.groupby("session_id").size().to_dict()}
    json.dump(summary, open(DROP_DIR / "summary.json", "w"), indent=2, default=str)
    print("peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)


if __name__ == "__main__":
    main()

"""Project paths and per-session settings. Edit VIDEO_DIRS when the videos are located."""
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
DATA = ROOT / "data" / "M1"
RESULTS = ROOT / "results"
DROP_DIR = RESULTS / "drop_events"
CHUNK_DIR = RESULTS / "viewer_chunks"
FS = 1000  # Hz, all ALF signals

SESSIONS = [
    "2026-06-18_16-25-09_000",
    "2026-06-18_16-25-09_001",
    "2026-07-09_14-23-53_002",
    "2026-07-21_11-48-31_000",
]

# Video files per session and camera (read-only). Matched to sessions by frame count == len(cam_t):
# 659515 -> 07-21, 625257 -> 07-09, 22217 -> 06-18 _000. NOTE the loose files' numbering is crossed
# (leftCamera.raw-001 pairs with rightCamera.raw-002). Session 2026-06-18_16-25-09_001 (436897 frames):
# no video found yet.
_ZIPS = DATA / "_video_zips"
_EXTRACTED = Path.home() / "IBL_videos"
VIDEO_FILES: dict[str, dict[str, Path]] = {
    "2026-06-18_16-25-09_000": {"left": _EXTRACTED / "2026-06-18_16-25-09_000" / "leftCamera.raw.mp4",
                                "right": _EXTRACTED / "2026-06-18_16-25-09_000" / "rightCamera.raw.mp4"},
    "2026-06-18_16-25-09_001": {},
    "2026-07-09_14-23-53_002": {"left": _ZIPS / "leftCamera.raw-002.mp4", "right": _ZIPS / "rightCamera.raw-001.mp4"},
    "2026-07-21_11-48-31_000": {"left": _ZIPS / "leftCamera.raw-001.mp4", "right": _ZIPS / "rightCamera.raw-002.mp4"},
}

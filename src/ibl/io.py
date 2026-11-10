"""Read-only session loaders. Only np.load / json.load; nothing is ever written under data/."""
import json

import numpy as np

from .config import DATA, FS, SESSIONS  # noqa: F401  (SESSIONS re-exported for convenience)

SENSORS = ("thermistor", "airflow", "chest")


def session_dir(session):
    return DATA / session


def load_info(session):
    with open(session_dir(session) / "metadata" / "session_info.json") as f:
        return json.load(f)


def available_sensors(session):
    """Respiration sensors that really exist (missing ones are saved as all-zero arrays)."""
    avail = load_info(session)["channels"]["respiration_available"]
    return [s for s in SENSORS if avail.get(s)]


def _load(session, name, mmap=True):
    return np.load(session_dir(session) / "alf" / name, mmap_mode="r" if mmap else None)


def n_samples(session):
    return _load(session, "_cavelli_thermistor.signal.npy").shape[0]


def load_resp(session, sensor):
    """One respiration channel as an in-memory float32 array (N,)."""
    assert sensor in available_sensors(session), f"{sensor} not available in {session}"
    return np.array(_load(session, f"_cavelli_{sensor}.signal.npy"), dtype=np.float32)


def load_lfp(session, ch=0):
    """One LFP channel (N,) as contiguous float32; the (N,10) array is memory-mapped, never fully loaded."""
    return np.ascontiguousarray(_load(session, "_cavelli_lfp.signal.npy")[:, ch], dtype=np.float32)


def load_emg(session, ch=0):
    return np.ascontiguousarray(_load(session, "_cavelli_emg.signal.npy")[:, ch], dtype=np.float32)


def load_cam_times(session, camera="left"):
    """Frame times in s on the recording clock (float64, length = number of frames)."""
    return np.array(_load(session, f"{camera}Camera.times.npy"))


def time_axis(n, fs=FS):
    return np.arange(n, dtype=np.float64) / fs

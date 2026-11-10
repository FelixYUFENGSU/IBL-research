"""Signal helpers. The first group is moved unchanged from scripts/m1Session.py; the rest is new."""
import numpy as np
from scipy import fft as sfft, signal
from scipy.ndimage import uniform_filter1d

from .config import FS

# ---- moved unchanged from scripts/m1Session.py -------------------------------------------------
zscore = lambda x: (x - x.mean()) / x.std()
smooth = lambda x, win_s: uniform_filter1d(x, int(win_s * FS), mode="constant")


def bandpass(x, lo, hi, order=3):
    b, a = signal.butter(order, [lo, hi], btype="band", fs=FS)
    return signal.filtfilt(b, a, x)                       # forward-backward: no time shift


def analytic(y):
    """Hilbert analytic signal of an already band-passed y (fast FFT length)."""
    n = len(y)
    return np.asarray(signal.hilbert(y, N=sfft.next_fast_len(n)))[:n]    # asarray: scipy's stubs type hilbert as a tuple


def get_phase(x, lo=1, hi=15):
    return np.angle(analytic(bandpass(x, lo, hi)))


def sliding_plv(p1, p2, win_s=5, step_s=1):
    """Window-centre times and PLV values (5-s windows, 1-s step, as in the notebook)."""
    W, S = int(win_s * FS), int(step_s * FS)
    c = np.concatenate([[0], np.cumsum(np.exp(1j * (p1 - p2)))])      # O(n) windowed means
    starts = np.arange(0, len(p1) - W, S)
    return (starts + W / 2) / FS, np.abs(c[starts + W] - c[starts]) / W


# ---- new -----------------------------------------------------------------------------------------
def sos_bandpass(x, lo, hi, order=3, fs=FS):
    """Zero-phase Butterworth band-pass, float32 in -> float32 out.

    Same filter as `bandpass` (butter + forward-backward) but in second-order sections, which is
    numerically stable for a narrow low band such as 1-15 Hz at 1 kHz. One channel at a time.
    """
    sos = signal.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float32)).astype(np.float32, copy=False)


def notch_sos(lines, q=None, bw_hz=None, fs=FS):
    """Cascade of second-order IIR notches, one per line (Hz).

    q: constant quality factor (bandwidth = f0 / q, so it widens with the harmonic, as the line drift does);
    or bw_hz: the same -3 dB bandwidth in Hz for every line.
    """
    return np.vstack([np.array(signal.tf2sos(*signal.iirnotch(f0, q if q else f0 / bw_hz, fs))) for f0 in lines])


def mains_notch(x, f0=50.0, harmonics=7, q=30.0, fs=FS):
    """Zero-phase removal of mains hum: notches at f0 * 1..harmonics (all below Nyquist). float32 in -> float32 out.

    Runs in float64 internally (the hum can be 1e5 x the signal) and filters one channel at a time.
    """
    lines = [f0 * k for k in range(1, harmonics + 1) if f0 * k < fs / 2]
    y = signal.sosfiltfilt(notch_sos(lines, q=q, fs=fs), np.asarray(x, dtype=np.float64))
    return y.astype(np.float32)

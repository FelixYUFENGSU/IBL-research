import numpy as np
import pytest

from ibl.drops import DropParams, amplitude_envelope, classify_events, detect_drops, emg_burst_flag, emg_rms, rolling_median
from ibl.video import format_timestamp, rec_to_video_time

FS = 1000


def _ratio(x):
    env = amplitude_envelope(x)
    return env / rolling_median(env)


def test_sine_gap_is_detected():
    t = np.arange(200 * FS) / FS
    x = np.sin(2 * np.pi * 5 * t).astype(np.float32)
    x[100 * FS:int(101.0 * FS)] = 0                      # 1-s gap
    d = detect_drops(_ratio(x))
    assert len(d) == 1
    assert 99.8 < d[0, 0] < 100.6 and 100.4 < d[0, 1] < 101.3


def test_short_dip_ignored_and_close_dips_merged():
    r = np.ones(20 * FS, dtype=np.float32)
    r[5000:5200] = 0                                     # 0.2 s: shorter than min duration
    r[8000:8200] = 0; r[8300:8500] = 0                   # two 0.2-s dips 0.1 s apart -> merged to 0.5 s
    d = detect_drops(r)
    assert len(d) == 1 and d[0, 0] == pytest.approx(8.0) and d[0, 1] == pytest.approx(8.5)


def test_classify_all_three_pair_single():
    drops = {"thermistor": np.array([[10, 11.0], [30, 31.0], [50, 51.0]]),
             "airflow": np.array([[10.1, 11.1], [30.2, 31.2]]),
             "lfp": np.array([[10.0, 10.9]])}
    ev = classify_events(drops)
    assert [e["type"] for e in ev] == ["ALL_THREE", "PAIR", "SINGLE"]
    assert ev[1]["sensors"] == ["thermistor", "airflow"]


def test_low_overlap_not_linked():
    ev = classify_events({"thermistor": np.array([[10, 12.0]]), "airflow": np.array([[11.8, 14.0]])})
    assert [e["type"] for e in ev] == ["SINGLE", "SINGLE"]


def test_emg_burst_flag():
    rng = np.random.default_rng(0)
    emg = (rng.standard_normal(60 * FS) * 5).astype(np.float32)
    emg[30 * FS:int(30.2 * FS)] *= 20                    # burst at 30 s
    rms = emg_rms(emg)
    burst, peak = emg_burst_flag(rms, [10, 29.5, 33.5], [10.5, 29.8, 34])
    assert list(burst) == [False, True, False]           # third event's +-1 s window ends before the burst... 
    assert peak[1] > 2.5 > peak[0]


def test_rec_to_video_time():
    cam_t = 17.74 + np.arange(1000) / 60.0
    f, v = rec_to_video_time([5.0, 17.74, 17.74 + 10 / 60 + 1e-6, 1000.0], cam_t)
    assert np.isnan(f[0]) and np.isnan(f[3])
    assert f[1] == 0 and f[2] == 11
    assert v[2] == pytest.approx(11 / 60.0, rel=1e-3)


def test_timestamp_format():
    assert format_timestamp(3725.5) == "01:02:05.500"
    assert format_timestamp(float("nan")) == ""


def _hum_and_burst(seed=0):
    """Broadband noise + 50/150/250 Hz hum (1e2-1e3 x the noise power) + a 0.3 s muscle-like burst (x6) at 60 s."""
    rng = np.random.default_rng(seed)
    t = np.arange(120 * FS) / FS
    x = rng.standard_normal(len(t)) * 5
    x[60 * FS:int(60.3 * FS)] *= 6
    x += sum(2000 * np.sin(2 * np.pi * f * t + 0.3 * f) for f in (50, 150, 250))
    return x.astype(np.float32)


def _line_power(y, lines=(50, 150, 250)):
    from scipy import signal as sg
    f, P = sg.welch(y, FS, nperseg=2 ** 14)
    return sum(P[np.abs(f - k) <= 1].sum() for k in lines)


def test_mains_notch_suppresses_lines_and_keeps_burst():
    from ibl.drops import emg_filter
    from ibl.signal import sos_bandpass
    x = _hum_and_burst()
    off = DropParams(notch=False)
    y_off, y_on = emg_filter(x, off), emg_filter(x, DropParams())
    assert y_off.dtype == y_on.dtype == np.float32
    suppression_db = 10 * np.log10(_line_power(y_off) / _line_power(y_on))
    assert suppression_db > 40, suppression_db            # lines at least 40 dB down (about 80 dB for pure tones)
    # without the notch the hum swamps the RMS, so the burst is not detectable; with it the burst is
    for p, expect in ((off, False), (DropParams(), True)):
        rms = emg_rms(x, p)
        burst, peak = emg_burst_flag(rms, [59.9, 20.0], [60.4, 21.0], p)
        assert bool(burst[0]) is expect and not burst[1], (p.notch, peak)
    assert peak[0] > 4                                    # close to the true x6 amplitude step (RMS window smears it)

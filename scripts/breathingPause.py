import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import dataclasses
    import json

    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from ibl import io
    from ibl.config import DROP_DIR, FS, VIDEO_FILES
    from ibl.drops import (DropParams, amplitude_envelope, classify_events, detect_drops, emg_burst_flag, emg_rms,
                           find_runs, load_sensor, rolling_median, session_sensors)
    from ibl.signal import sos_bandpass
    from ibl.video import measure_fps, mp4_info, rec_to_video_time

    matplotlib.use("Agg")
    plt.rcParams.update({"figure.figsize": (11, 3.6), "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})
    COLORS = {"thermistor": "#2a78d6", "airflow": "#eb6834", "lfp": "#1baf7a"}
    TYPE_COLORS = {"ALL_THREE": "#c0392b", "PAIR": "#e08e0b", "SINGLE": "#7f8c8d"}
    return (COLORS, DROP_DIR, DropParams, FS, TYPE_COLORS, VIDEO_FILES, amplitude_envelope, classify_events,
            dataclasses, detect_drops, emg_burst_flag, emg_rms, find_runs, io, json, load_sensor, measure_fps, mo,
            mp4_info, np, pd, plt, rec_to_video_time, rolling_median, session_sensors, sos_bandpass)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Breathing pauses vs. sensor failures: detecting signal drops

    **Question.** Three sensors record breathing at the same time: the **thermistor**, the **airflow** sensor and the **LFP** (channel 0). Sometimes their signals shrink for a moment. Two very different things can cause that:

    - **A real pause in breathing** (for example a swallow). Then *all three* sensors should lose their rhythm *together*.
    - **A sensor problem** (the thermistor shifts, the airflow tube is blocked, the LFP is noisy). Then only *one or two* sensors drop.

    A third thing complicates both: **movement**. The mice are head-fixed, but they can still move their jaw and face, and that is visible in the EMG (muscle) channel. Movement can disturb a sensor without any change in breathing, so every event is also tagged with whether EMG activity is high around it.

    **How this notebook is organised.** Each step of the detector is explained first, then plotted, and **every threshold has a justification plot placed right before it is used**. The development session (6.6 min) is used for tuning and validation. The code lives in `src/ibl/drops.py`; the notebook only calls it.

    **Parameters to justify** (starting values from the task): amplitude threshold = 0.35 × rolling median, minimum duration = 0.3 s, merge gap = 0.2 s, cross-sensor overlap = 50 %, EMG burst = 2.5 × median within ±1 s.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The data

    All signals are on one clock at 1000 Hz (`t = arange(N) / 1000`), saved as float32. A sensor that was not recorded is stored as an all-zero array, so the notebook asks `session_info.json -> channels.respiration_available` which sensors really exist instead of looking at the data. Here that is thermistor and airflow (the chest sensor was not recorded). LFP and EMG are multi-channel arrays; we only read **LFP channel 0** and **EMG channel 0**, one channel at a time, to keep memory low (some sessions have 11 million samples).

    The table below should show 397,714 samples (about 398 s), float32, and two respiration sensors.
    """)
    return


@app.cell
def _(DropParams, FS, io, load_sensor, mo, np, session_sensors):
    SID = "2026-06-18_16-25-09_000"
    P = DropParams()
    sensors = session_sensors(SID)                      # thermistor, airflow, lfp
    raw = {s: load_sensor(SID, s) for s in sensors}
    emg = io.load_emg(SID, 0)
    cam_t = io.load_cam_times(SID, "left")
    t = np.arange(len(emg)) / FS
    mo.md(f"""
    | | value |
    |---|---|
    | session | `{SID}` |
    | samples / duration | {len(emg):,} / {len(emg) / FS:.1f} s |
    | respiration sensors available | {io.available_sensors(SID)} (+ LFP ch0) |
    | dtype | {emg.dtype} |
    | camera frames / first-last frame time | {len(cam_t):,} / {cam_t[0]:.2f} – {cam_t[-1]:.2f} s |
    """)
    return P, SID, cam_t, emg, raw, sensors, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Breathing strength over time ("amplitude")

    Breathing is a rhythm of roughly 2–10 Hz, so to ask "is the sensor still showing a rhythm?" we measure the **strength of the oscillation**, not the raw value:

    1. **Band-pass 1–15 Hz** (3rd-order Butterworth, run forward and backward so nothing is shifted in time). This removes slow drift and fast noise but keeps the breathing rhythm.
    2. **Hilbert amplitude** `|hilbert(x)|`: the instantaneous height of the oscillation, whatever its phase.
    3. **Smooth over 0.25 s** so individual cycles do not make the amplitude flicker.

    **What to look for in the plot below.** Each row is one sensor, 85–100 s. The grey trace is the filtered signal; the coloured line is its amplitude (the envelope). The envelope should hug the peaks of the wave and fall towards zero when the rhythm fades. All three should visibly shrink around 87.5–88.5 s and 92–93 s.
    """)
    return


@app.cell
def _(COLORS, FS, P, amplitude_envelope, plt, raw, sensors, sos_bandpass, t):
    env = {s: amplitude_envelope(raw[s], P) for s in sensors}
    _i0, _i1 = 85 * FS, 100 * FS
    _fig, _ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for _a, _s in zip(_ax, sensors):
        _a.plot(t[_i0:_i1], sos_bandpass(raw[_s], *P.band)[_i0:_i1], color="0.6", lw=0.8, label="1–15 Hz signal")
        _a.plot(t[_i0:_i1], env[_s][_i0:_i1], color=COLORS[_s], lw=2, label="amplitude (Hilbert, 0.25 s)")
        _a.set_ylabel(_s); _a.legend(loc="upper right", ncol=2, fontsize=8)
    _ax[-1].set_xlabel("time (s)")
    _fig.tight_layout()
    _fig
    return (env,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. What counts as "low"? A rolling reference

    Absolute amplitude is not comparable between sensors (different units), and each sensor slowly drifts over a session. So every sensor is compared with **its own 60-s rolling median**: `ratio = amplitude / rolling median`. A ratio of 1 means "typical right now", 0.35 means "about a third of typical".

    A median (not a mean) is used so that the drops themselves do not pull the reference down. 60 s is long compared with a drop (about 1 s) but short compared with a session, so it follows slow changes in signal quality. (Technically it is computed on 4 Hz block averages and interpolated, which is much faster and gives the same smooth line.)

    **The plot below** shows the amplitude (colour), the 60-s median (black) and the 0.35 × median line (dashed) for the whole session, on a log scale. The amplitude should wander around the black line and only occasionally dive below the dashed one.
    """)
    return


@app.cell
def _(COLORS, P, env, plt, rolling_median, sensors, t):
    med = {s: rolling_median(env[s], P.median_win_s) for s in sensors}
    ratio = {s: env[s] / med[s] for s in sensors}
    _fig, _ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for _a, _s in zip(_ax, sensors):
        _a.plot(t, env[_s], color=COLORS[_s], lw=0.6)
        _a.plot(t, med[_s], color="k", lw=1.2, label="60 s rolling median")
        _a.plot(t, P.frac * med[_s], color="k", lw=1, ls="--", label=f"{P.frac} × median")
        _a.set_yscale("log"); _a.set_ylabel(_s); _a.legend(loc="upper right", ncol=2, fontsize=8)
    _ax[-1].set_xlabel("time (s)")
    _fig.tight_layout()
    _fig
    return med, ratio


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Threshold 1: how low is "a drop"? (0.35 × median)

    **Justification plot.** The histogram shows how often each ratio value occurs in this session (log scale on x, so 0.5 and 2 are equally far from 1). Normal breathing forms a hump near 1. A real drop should appear as a **separate shoulder or tail on the left**, not just the left edge of the hump. A good threshold sits in the trough between the hump and the tail, so that ordinary variation is not counted.

    The dashed line is 0.35. Read the legend for the share of time each sensor spends below it; it should be small (a few percent at most). If it were tens of percent, 0.35 would be cutting into normal breathing.
    """)
    return


@app.cell
def _(COLORS, P, np, plt, ratio, sensors):
    _fig, _ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=False)
    _bins = np.linspace(-2, 1, 120)
    for _a, _s in zip(_ax, sensors):
        _a.hist(np.log10(np.clip(ratio[_s], 1e-3, None)), bins=_bins, color=COLORS[_s], alpha=0.8)
        _a.axvline(np.log10(P.frac), color="k", ls="--", label=f"{P.frac}: {100 * (ratio[_s] < P.frac).mean():.1f}% of time below")
        _a.axvline(0, color="0.5", ls=":")
        _a.set_yscale("log"); _a.set_title(_s, loc="left"); _a.set_xlabel("log10(amplitude / median)")
        _a.legend(fontsize=8, loc="upper left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Threshold 2: merging nearby drops (0.2 s)

    The ratio can briefly bounce back above the threshold in the middle of what is really one drop, because the amplitude of an oscillation passes through zero between breaths. If we did not merge, one pause would be cut into several short pieces and then thrown away by the minimum-duration rule.

    **Justification plot.** For every pair of consecutive below-threshold runs, the histogram shows the gap between them. Gaps from the bounce-back effect should pile up at **very short** values; gaps between truly separate events should be much longer. 0.2 s (dashed) should lie in the valley between the two groups, or at least cut off the short pile.
    """)
    return


@app.cell
def _(FS, P, find_runs, np, plt, ratio, sensors):
    _gaps = np.concatenate([(lambda r: (r[0][1:] - r[1][:-1]) / FS)(find_runs(ratio[_s] < P.frac)) for _s in sensors])
    _fig, _ax = plt.subplots(figsize=(8, 3.4))
    _ax.hist(_gaps, bins=np.logspace(-3, 2, 60), color="0.4")
    _ax.axvline(P.merge_s, color="k", ls="--", label=f"{P.merge_s} s merge gap ({100 * (_gaps < P.merge_s).mean():.0f}% of gaps are shorter)")
    _ax.set_xscale("log"); _ax.set_xlabel("gap between consecutive low runs (s)"); _ax.set_ylabel("count"); _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Threshold 3: minimum duration (0.3 s)

    A drop must last at least `min_duration` to count. Too short and we report single flickers of noise; too long and we miss swallow-length pauses. A single breath at the rate seen here lasts roughly 0.1–0.3 s, so a gap shorter than that can occur by chance in the middle of normal breathing, while a pause is longer than a breath.

    **Justification plot, left.** Distribution of the (merged) low-run durations, all sensors together. There should be a large number of very short runs (noise) falling off quickly, and a tail of longer drops. 0.3 s (dashed) should be where the steep part ends.
    **Right.** How many *events* (after combining sensors) we get as the minimum duration changes, and how many of them involve all three sensors. A good value sits where the all-three count is already stable (a plateau) while the total keeps falling: that means we only lose noise.
    """)
    return


@app.cell
def _(DropParams, P, classify_events, dataclasses, detect_drops, np, plt, ratio, sensors):
    def n_events(frac, min_dur):
        _p = dataclasses.replace(P, frac=frac, min_dur_s=min_dur)
        _ev = classify_events({_s: detect_drops(ratio[_s], _p) for _s in sensors}, _p.overlap)
        return len(_ev), sum(e["type"] == "ALL_THREE" for e in _ev)

    _p0 = dataclasses.replace(P, min_dur_s=0.0)
    _dur = np.concatenate([(lambda d: d[:, 1] - d[:, 0])(detect_drops(ratio[_s], _p0)) for _s in sensors])
    _mds = np.arange(0.1, 0.81, 0.05)
    _res = np.array([n_events(P.frac, m) for m in _mds])
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.6))
    _ax[0].hist(_dur, bins=np.linspace(0, 2, 80), color="0.4")
    _ax[0].axvline(P.min_dur_s, color="k", ls="--", label=f"{P.min_dur_s} s")
    _ax[0].set_yscale("log"); _ax[0].set_xlabel("run duration (s)"); _ax[0].set_ylabel("count"); _ax[0].legend()
    _ax[1].plot(_mds, _res[:, 0], "o-", color="0.4", label="all events"); _ax[1].plot(_mds, _res[:, 1], "o-", color="#c0392b", label="ALL_THREE")
    _ax[1].axvline(P.min_dur_s, color="k", ls="--"); _ax[1].set_xlabel("minimum duration (s)"); _ax[1].set_ylabel("events"); _ax[1].legend()
    _fig.tight_layout()
    _fig
    return (n_events,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Tuning both thresholds together

    The two thresholds interact (a lower amplitude threshold makes drops shorter, so more of them fail the duration rule). The grid below varies both. Read it as "how many **ALL_THREE** events" (candidate breathing pauses) and "how many events in total" for each pair of values.

    **What we want.** (a) The two events we know are real (~88 s and ~92.5–93 s, all three sensors dropping) must stay detected. (b) The number of ALL_THREE events should not change much when we nudge the thresholds (a stable region), while the total number of events should not explode (that would be noise being admitted). The chosen cell (0.35, 0.3 s) is outlined in the left panel.
    """)
    return


@app.cell
def _(P, n_events, np, plt):
    _fr = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    _md = [0.2, 0.3, 0.4, 0.5]
    _g = np.array([[n_events(f, m) for m in _md] for f in _fr])    # (frac, min_dur, [total, all3])
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for _a, _k, _name in zip(_ax, (1, 0), ("ALL_THREE events", "all events")):
        _a.imshow(_g[:, :, _k], cmap="Blues", aspect="auto")
        for _i in range(len(_fr)):
            for _j in range(len(_md)):
                _a.text(_j, _i, int(_g[_i, _j, _k]), ha="center", va="center")
        _a.set_xticks(range(len(_md))); _a.set_xticklabels(_md); _a.set_yticks(range(len(_fr))); _a.set_yticklabels(_fr)
        _a.set_xlabel("minimum duration (s)"); _a.set_ylabel("amplitude fraction"); _a.set_title(_name, loc="left"); _a.grid(False)
    _ax[0].add_patch(plt.Rectangle((_md.index(P.min_dur_s) - 0.5, _fr.index(P.frac) - 0.5), 1, 1, fill=False, ec="red", lw=2))
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading the grid.** At fraction 0.35 the ALL_THREE count is **5 for every minimum duration from 0.3 to 0.5 s**; only at 0.2 s does it jump to 7 (extra short fragments). Looser fractions (0.40, 0.45, 0.50 at 0.3 s) give 6, 7 and 10 ALL_THREE events while the total goes from 35 to 45, 69 and 94: the extra events are borderline ones, mostly single-sensor and pair wobbles. Tighter fractions lose real events: at 0.25 the ALL_THREE count falls to 3 (0.3 s) and 1 (0.4 s), and at 0.4 s the ~88 s event disappears. So **0.35 and 0.3 s are kept**. They sit at the start of the plateau, which favours not missing a real pause, and SINGLE / PAIR are labelled *candidates* anyway.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Threshold 4: when do drops in two sensors count as "together"? (50 % overlap)

    Each sensor gives its own list of drop intervals. Two drops from different sensors are treated as the same event if they **overlap by at least 50 % of the shorter one**. Chains are followed: if A overlaps B and B overlaps C, all three belong to one event. Events are then labelled **ALL_THREE**, **PAIR** (which two are recorded) or **SINGLE**.

    **Justification plot.** For every pair of drops from different sensors that touch at all, the histogram shows the overlap as a fraction of the shorter drop. If there are two groups, "barely touching" near 0 and "almost the same interval" near 1, then any cut-off in between gives the same answer and 0.5 is a safe middle. A big smear around 0.5 would mean the choice matters.
    """)
    return


@app.cell
def _(P, detect_drops, np, plt, ratio, sensors):
    _d = {_s: detect_drops(ratio[_s], P) for _s in sensors}
    _fr = []
    for _i, _a in enumerate(sensors):
        for _b in sensors[_i + 1:]:
            for _s1, _e1 in _d[_a]:
                for _s2, _e2 in _d[_b]:
                    _ov = min(_e1, _e2) - max(_s1, _s2)
                    if _ov > 0:
                        _fr.append(_ov / min(_e1 - _s1, _e2 - _s2))
    _fig, _ax = plt.subplots(figsize=(8, 3.4))
    _ax.hist(_fr, bins=np.linspace(0, 1, 21), color="0.4")
    _ax.axvline(P.overlap, color="k", ls="--", label=f"{P.overlap}: {sum(f < P.overlap for f in _fr)} of {len(_fr)} touching pairs fall below")
    _ax.set_xlabel("overlap / shorter drop"); _ax.set_ylabel("pairs of drops"); _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. EMG line-noise removal (mains hum)

    **Why this section exists.** When the EMG pipeline was first built, the 30–300 Hz band-pass was assumed to leave "muscle". It does not: the raw EMG also contains **mains hum**, a 50 Hz tone and its harmonics picked up from the building's electricity, and the band-pass lets those through. A hum is steady, so it adds a large *constant* amount to the RMS. The burst rule below compares the RMS with its session median, so a big constant makes a muscle burst look small by comparison and the rule stops working.

    **What was observed and is checked here, step by step:** (9.1) how big the hum is in each session, (9.2) how it changes over time, (9.3) whether it sits exactly at 50.0 Hz, (9.4) which harmonics matter, (9.5) which filter, (9.6) how narrow it can be, (9.7) global or per-session setting, (9.8) that the filter does not damage real bursts, (9.9) that the 2.5× threshold still holds. Everything is computed on **EMG channel 0**; the code is in `src/ibl/emgnoise.py` (measurements) and `src/ibl/drops.py` (`emg_filter`, the filter actually used).

    All four sessions are loaded once below (one channel each, float32; the three long ones are 2–3 hours).
    """)
    return


@app.cell
def _(io):
    from ibl import emgnoise as en
    from ibl.drops import emg_filter, rms_envelope
    from scipy import signal as sg_signal
    SESS = list(io.SESSIONS)
    SHORT = {s: s[5:10] + "…" + s[-3:] for s in SESS}
    emg_all = {s: io.load_emg(s, 0)[:] for s in SESS}
    return SESS, SHORT, emg_all, emg_filter, en, rms_envelope, sg_signal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.1 How big is the hum? (power spectrum of the raw EMG)

    A **power spectrum** shows how much signal power sits at each frequency (vertical axis, log scale, so each gridline is 10× more power). Muscle activity is broadband: a smooth, slowly falling curve. Mains hum is a set of **narrow vertical spikes** at 50, 100, 150, … Hz.

    **What to look for.** In every panel the spikes (dashed grey lines mark 50·k Hz) stand far above the smooth floor. In the first three sessions the strongest lines are about 3 orders of magnitude above it; in 07-21 (bottom right) they are about **5 orders** above, and the vertical scale of that panel is correspondingly larger. The *hum share* in each title is the fraction of the 30–300 Hz power that lies within ±1 Hz of a 50·k Hz line (k = 1…6).
    """)
    return


@app.cell
def _(SESS, SHORT, emg_all, en, np, plt):
    _fig, _ax = plt.subplots(2, 2, figsize=(12, 6.5))
    spec = {}
    for _a, _s in zip(_ax.ravel(), SESS):
        _f, _P = en.welch(emg_all[_s], nperseg=2 ** 15)
        spec[_s] = (_f, _P)
        _a.semilogy(_f, _P, color="0.25", lw=0.6)
        for _k in range(1, 10):
            _a.axvline(50 * _k, color="#c0392b", lw=0.6, ls="--", alpha=0.5)
        _a.set_xlim(0, 500); _a.set_xlabel("frequency (Hz)"); _a.set_ylabel("power density")
        _a.set_title(f"{_s}  ·  hum share {100 * en.hum_share(_f, _P):.0f}%", loc="left", fontsize=9)
    _fig.tight_layout()
    _fig
    return (spec,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.2 How does the hum change over time, and what does it do to the RMS?

    The same hum share, computed for every **one-minute window** (left), and the consequence for the burst rule (right): the **highest RMS in each minute divided by that minute's median RMS**. A real muscle burst should make this ratio clearly larger than 1; with hum dominating, the RMS is nearly constant and the ratio stays close to 1.

    **What to look for.** Left: 07-21 sits at about 100% for the whole 3 hours, the others wander between roughly 20% and 90%. Right (hum **not** removed): in 07-21 the ratio is stuck near 1.0–1.1, so a "2.5× the median" rule can almost never fire there; the other sessions still reach 2–10 in some minutes.
    """)
    return


@app.cell
def _(DropParams, SESS, SHORT, emg_all, emg_filter, en, np, plt, rms_envelope):
    _off = DropParams(notch=False)
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.6))
    minute_stats = {}
    for _s in SESS:
        _t, _sh = en.hum_share_by_window(emg_all[_s])
        _r = rms_envelope(emg_filter(emg_all[_s], _off), _off)
        _n = 60000 * (len(_r) // 60000)
        _m = _r[:_n].reshape(-1, 60000)
        _ratio = _m.max(1) / np.median(_m, axis=1)
        minute_stats[_s] = (_t / 60, _sh, _ratio)
        _ax[0].plot(_t / 60, 100 * _sh, lw=1, label=SHORT[_s]); _ax[1].plot(_t / 60, _ratio, lw=1, label=SHORT[_s])
    _ax[0].set_xlabel("minute of session"); _ax[0].set_ylabel("hum share of 30–300 Hz power (%)"); _ax[0].legend(fontsize=8)
    _ax[1].set_xlabel("minute of session"); _ax[1].set_ylabel("max / median RMS within the minute"); _ax[1].set_yscale("log"); _ax[1].axhline(2.5, color="k", ls=":")
    _fig.tight_layout()
    _fig
    return (minute_stats,)


@app.cell(hide_code=True)
def _(mo, minute_stats, np):
    _rows = "\n".join(f"| {s[5:10]}…{s[-3:]} | {100 * np.median(v[1]):.0f}% | {100 * v[1].min():.0f}–{100 * v[1].max():.0f}% | {np.median(v[2]):.2f} | {v[2].min():.2f}–{v[2].max():.2f} |" for s, v in minute_stats.items())
    mo.md(f"""
    **Numbers behind the plot** (hum *not* removed):

    | session | hum share, median minute | hum share, range | max/median RMS, median minute | range |
    |---|---|---|---|---|
    {_rows}

    **Finding.** The 07-21 EMG is essentially pure hum (about 100% of the band power in every minute, max/median ≈ 1.0–1.1). The other sessions are also contaminated (their median minute is roughly 40–75% hum), which hides part of every burst but leaves the rule usable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.3 Is the hum exactly 50.0 Hz? Does it drift?

    A filter has to be placed at the right frequency. If the hum frequency wanders, a very narrow filter would miss it. Mains frequency varies slightly with grid load, and every harmonic wanders **k times as much** as the fundamental (a 0.1 Hz change at 50 Hz is 0.5 Hz at 250 Hz).

    **Method.** For each minute the fundamental is estimated from its 3rd harmonic (150 Hz), a spike with 3× finer relative precision than the 50 Hz line itself, then divided by 3. The table gives each line's centre frequency over the whole session.

    **What to look for.** Lines that sit within a few hundredths of a Hz of 50·k mean "fixed frequency"; a spread of ±0.1 Hz or more means drift.
    """)
    return


@app.cell
def _(SESS, SHORT, emg_all, en, mo, np, plt, spec):
    _fig, _ax = plt.subplots(figsize=(10, 3.4))
    fund = {}
    for _s in SESS:
        fund[_s] = en.fundamental_by_window(emg_all[_s])
        _ax.plot(np.arange(len(fund[_s])), fund[_s], lw=1, label=SHORT[_s])
    _ax.axhline(50, color="k", ls=":"); _ax.set_xlabel("minute of session"); _ax.set_ylabel("mains fundamental (Hz)"); _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = []
    for _s in SESS:
        _f, _P = spec[_s]
        _c = en.line_centroids(_f, _P, en.ALL_LINES[:7])
        _rows.append("| " + SHORT[_s] + " | " + " | ".join(f"{_x - 50 * (i + 1):+.3f}" for i, _x in enumerate(_c)) + f" | {fund[_s].min():.3f}–{fund[_s].max():.3f} |")
    mo.vstack([_fig, mo.md("**Line centre minus 50·k Hz (Hz), whole session, and range of the per-minute fundamental:**\n\n| session | 50 | 100 | 150 | 200 | 250 | 300 | 350 | fundamental range |\n|---|---|---|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return (fund,)


@app.cell(hide_code=True)
def _(fund, mo, np):
    _span = {s[5:10] + "…" + s[-3:]: (v.max() - v.min()) for s, v in fund.items()}
    mo.md(f"""
    **Finding.** Lines are at **50·k Hz** in all four sessions (centres within about ±0.2 Hz even at 350 Hz). Three sessions are very stable (fundamental range {min(_span.values()):.2f}–{sorted(_span.values())[2]:.2f} Hz over the whole recording). **Session 07-09 drifts more**: its minute-to-minute fundamental spans {max(_span.values()):.2f} Hz, which is about **{5 * max(_span.values()):.1f} Hz at the 250 Hz harmonic**. So the notch bandwidth must **grow with the harmonic number** (constant Q), not stay constant in Hz. The next two steps choose the bandwidth from data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.4 Which harmonics have to be removed?

    The 30–300 Hz band-pass is only a 3rd-order filter, so hum **above** 300 Hz is attenuated gently (about −20 dB at 450 Hz) rather than removed. When the hum is 10⁵ times the muscle signal, even a −20 dB leak matters. Rather than guess, notch filters were placed at 50·1 … 50·n Hz with n = 0 (no notch), 5 (50–250 Hz), 6, 7, 9, and the hum that remains in the *final* signal (after the 30–300 Hz band-pass) was measured. All notches use Q = 30 here; 9.6 explains why. Measured on the first 30 minutes of each session (the dev session: all 6.6 min).

    **What to look for.** The bars are the share of the final signal's power that is still within ±1 Hz of a 50·k Hz line (k = 1…9). A good filter brings every session below the 5% line.
    """)
    return


@app.cell
def _(SESS, SHORT, en, mo, np, plt):
    _L = en.ALL_LINES
    harm_opts = {"no notch": None}
    for _n in (5, 6, 7, 9):
        harm_opts[f"{_n} harmonics (50–{50 * _n} Hz)"] = en.filt_notch(_L[:_n], 30)
    harm = en.cached("harmonics_excerpt30.csv", lambda: en.option_table(SESS, harm_opts))
    _fig, _ax = plt.subplots(figsize=(10, 3.6))
    _names = list(harm_opts)
    for _i, _s in enumerate(SESS):
        _v = [harm[(harm.option == _o) & (harm.session == _s)].hum_share.iloc[0] for _o in _names]
        _ax.bar(np.arange(len(_names)) + 0.2 * (_i - 1.5), 100 * np.array(_v), 0.2, label=SHORT[_s])
    _ax.axhline(5, color="k", ls="--", label="5%"); _ax.set_yscale("log"); _ax.set_xticks(range(len(_names))); _ax.set_xticklabels(_names, fontsize=8)
    _ax.set_ylabel("hum left in final signal (% of power)"); _ax.legend(fontsize=8, ncol=5)
    _fig.tight_layout()
    _fig
    return harm, harm_opts


@app.cell(hide_code=True)
def _(harm, mo):
    _g = lambda o: 100 * harm[(harm.option == o) & harm.session.str.startswith("2026-07-21")].hum_share.iloc[0]
    _h5, _h7, _h9 = _g("5 harmonics (50–250 Hz)"), _g("7 harmonics (50–350 Hz)"), _g("9 harmonics (50–450 Hz)")
    mo.md(f"""
    **Decision: notch 50, 100, …, 350 Hz (7 harmonics).** In session 07-21, with only 50–250 Hz notched, **{_h5:.0f}%** of the final power is still hum (the 300 and 350 Hz lines leak through the band-pass); with 50–350 Hz it is **{_h7:.1f}%**. Adding 400 and 450 Hz (9 harmonics) changes nothing ({_h9:.1f}%), so the list stops at 350 Hz. The other sessions are already below 5% with 5 harmonics and improve a little with 7. (A quick first test with only 50–250 Hz therefore recovered bursts in 07-21 but was not enough: it left about a third of the power as hum.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.5 Which kind of filter?

    Three ways of removing narrow lines were compared, all applied before the 30–300 Hz band-pass and all **zero-phase** (run forward and backward, so bursts are not shifted in time):

    1. **Cascade of IIR notches** at 50·k Hz, each with bandwidth = frequency / Q (so wider at higher harmonics, as the drift is).
    2. **IIR comb filter** (`scipy.signal.iircomb`): one filter with teeth at every multiple of 50 Hz up to Nyquist, all with the *same* bandwidth in Hz.
    3. **Spectral interpolation**: short-time Fourier transform, and in a band around each line the magnitude is replaced by the neighbouring noise floor (phase kept). It is the only option that does not remove broadband power near the line.

    **Left plot: frequency response** of the two IIR options (how much each frequency is let through; 0 dB = unchanged; the response is for the forward-backward pass). The notches get wider with frequency; the comb's teeth stay the same width. **Right plot:** the average spectrum of white noise before and after each option, i.e., what each option does to genuine broadband activity. A flat curve at 0 dB is ideal.
    """)
    return


@app.cell
def _(en, np, plt, sg_signal):
    _fs = 1000
    _L = en.ALL_LINES[:7]
    _w = np.linspace(20, 400, 8000)
    _resp = {"notch ×7, Q=30": en.notch_sos(_L, q=30), "comb, bw 1.7 Hz": en.comb_sos(50.0, 1.7)}
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for _n, _sos in _resp.items():
        _, _h = sg_signal.sosfreqz(_sos, worN=_w, fs=_fs)
        _ax[0].plot(_w, 40 * np.log10(np.maximum(np.abs(_h), 1e-6)), label=_n)         # forward-backward pass: gain = |H|^2, i.e. 40 log10|H| dB
    _ax[0].set_ylim(-80, 3); _ax[0].set_xlabel("frequency (Hz)"); _ax[0].set_ylabel("gain (dB, zero-phase)"); _ax[0].legend(fontsize=8)
    _rng = np.random.default_rng(0)
    _wn = _rng.standard_normal(600_000).astype(np.float32)
    for _n, _f in (("notch ×7, Q=30", en.filt_notch(_L, 30)), ("comb, bw 1.7 Hz", en.filt_comb(30)), ("interpolation", en.filt_interp(_L, 30))):
        _f0, _p0 = en.welch(_wn, nperseg=4096); _, _p1 = en.welch(_f(_wn), nperseg=4096)
        _ax[1].plot(_f0, 10 * np.log10(_p1 / _p0), lw=0.8, label=_n)
    _ax[1].set_xlim(20, 400); _ax[1].set_ylim(-30, 3); _ax[1].set_xlabel("frequency (Hz)"); _ax[1].set_ylabel("white-noise power after / before (dB)"); _ax[1].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Numbers for the options** (each applied to the first 30 minutes of every session, then the usual 30–300 Hz band-pass and 0.25 s RMS). Columns, per session: **hum left** (% of the final power within ±1 Hz of a 50·k Hz line; below about 3% is the level of the plain noise floor); **white noise kept** (share of broadband power surviving; the cost of the option); **synthetic burst**: a 0.3 s burst 6× stronger than the noise, added to noise plus a big hum, should come out near 6× the median RMS with the same width; and **max/median RMS** in each session.

    **What to look for.** A good option has low "hum left" in *every* session including 07-21, keeps the synthetic burst near 6, and costs as little white noise as possible.
    """)
    return


@app.cell
def _(SESS, en, mo, pd):
    _L = en.ALL_LINES
    opts = {"none": None,
            "notch ×7 Q=15": en.filt_notch(_L[:7], 15), "notch ×7 Q=30": en.filt_notch(_L[:7], 30),
            "notch ×7 Q=50": en.filt_notch(_L[:7], 50), "notch ×7 Q=100": en.filt_notch(_L[:7], 100),
            "comb bw 1.7 Hz": en.filt_comb(30), "comb bw 0.8 Hz": en.filt_comb(60),
            "interpolation (Q=30 bandwidth)": en.filt_interp(_L[:7], 30)}
    tab = en.cached("options_excerpt30.csv", lambda: en.option_table(SESS, opts))
    _p = tab.pivot(index="option", columns="session", values="hum_share").loc[list(opts)]
    _mx = tab.pivot(index="option", columns="session", values="max_over_median_rms").loc[list(opts)]
    _o = tab.groupby("option")[["white_noise_kept", "synthetic_burst_ratio"]].first().loc[list(opts)]
    option_view = pd.DataFrame({"white noise kept": _o.white_noise_kept.round(2), "synthetic burst (true 6)": _o.synthetic_burst_ratio.round(1)})
    for _s in SESS:
        option_view[f"hum left {_s[5:10]}…{_s[-3:]} (%)"] = (100 * _p[_s]).round(1)
    for _s in SESS:
        option_view[f"max/med RMS {_s[5:10]}…{_s[-3:]}"] = _mx[_s].round(1)
    mo.ui.table(option_view.reset_index(), selection=None, page_size=12)
    return opts, option_view, tab


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Decision: IIR notch cascade at 50–350 Hz.** (a) **Comb**: its fixed bandwidth is too narrow for the broad skirts of the high harmonics, so a lot of hum is left in 07-21 (about 20% with a 1.7 Hz tooth, about 70% with 0.8 Hz). It is cheap on broadband power (keeps about 93–96%) but does not suppress the lines where it matters. (b) **Spectral interpolation** does suppress the lines (about 3% left) and keeps most broadband power (about 94%), but it needs an 8 s short-time Fourier transform with a non-linear magnitude replacement (heavier on memory for 11 million samples, and each sample then depends on the neighbouring spectrum). It gives essentially the same burst ratios as the notches (the last four columns), so its extra broadband power does not change what is detected. (c) The **notch cascade** is linear, stable, cheap enough for 11 million samples, and its effect on bursts is exactly known from the synthetic burst (peak 6.2 against the true 6, same width and no time shift).

    **The cost to be aware of.** At Q = 30 the notches remove about **24%** of white-noise power from the 30–300 Hz band (kept 0.76 in the table). Because the burst rule compares the RMS with its own median, a uniform loss of this kind scales both and does not change the ratio. Absolute RMS values after this step are lower than before (mostly because the hum is gone, plus about 13% in amplitude for genuine broadband activity) and should not be compared with the old numbers; ratios can be.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.6 How narrow can the notch be? (choosing Q)

    A **higher Q means a narrower notch**: it throws away less of the real muscle signal but leaves more of the hum's skirts (hum is not perfectly sharp: it drifts a little and is slightly modulated by movement). So the right Q is the **narrowest that still suppresses the lines**. "Suppresses the lines" is defined *before* looking at the answer as: over the whole session, **less than 2.5% of the final signal's power is within ±1 Hz of a mains line**, and **at most 5% of the one-minute windows exceed 5%**. A single worst minute is not used as a criterion because minutes with real movement show more leakage.

    **What to look for.** The two panels show these two numbers against Q (whole sessions, 7 notches). The chosen Q is the largest one below both dashed lines in every session.
    """)
    return


@app.cell
def _(SESS, SHORT, en, mo, np, pd, plt):
    _L = en.ALL_LINES[:7]
    _Q = (15, 20, 30, 40, 50, 75)
    def _build():
        rows = []
        for q in _Q:
            for s in SESS:
                rows.append({"Q": q, "session": s, **en.minute_hum_stats(s, en.filt_notch(_L, q))})
        return pd.DataFrame(rows)
    qtab = en.cached("q_sweep_whole_sessions.csv", _build)
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for _s in SESS:
        _d = qtab[qtab.session == _s]
        _ax[0].plot(_d.Q, 100 * _d.whole, "o-", label=SHORT[_s]); _ax[1].plot(_d.Q, 100 * _d.frac_minutes_gt5pct, "o-", label=SHORT[_s])
    _ax[0].axhline(2.5, color="k", ls="--"); _ax[1].axhline(5, color="k", ls="--")
    _ax[0].set_ylabel("hum left, whole session (%)"); _ax[1].set_ylabel("minutes with >5% hum left (%)")
    for _a in _ax:
        _a.set_xlabel("Q"); _a.set_xscale("log"); _a.set_xticks(_Q); _a.set_xticklabels(_Q); _a.legend(fontsize=8)
    _fig.tight_layout()
    ok_q = [q for q in _Q if (qtab[qtab.Q == q].whole < 0.025).all() and (qtab[qtab.Q == q].frac_minutes_gt5pct <= 0.05).all()]
    mo.vstack([_fig, mo.md(f"**Q values meeting both criteria in all four sessions: {ok_q}.** The narrowest (largest) of them is **Q = {max(ok_q)}**, which is the one used: bandwidth {50 / max(ok_q):.1f} Hz at 50 Hz up to {350 / max(ok_q):.1f} Hz at 350 Hz. Q = 40 and above already leave more than 2.5% hum in 07-21 or more than 5% of its minutes above 5%.")])
    return ok_q, qtab


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.7 One setting for all sessions, or measured per session?

    Because each session's line frequency differs a little (9.3), one could place the notches at that session's *own* measured fundamental instead of 50.000 Hz. The table compares the two on every session with the chosen Q: the hum left in the worst one-minute window and in the median window. If they agree, the simpler global setting is preferable (no estimation step that could fail on a bad recording).
    """)
    return


@app.cell
def _(SESS, SHORT, emg_all, en, fund, mo, np, pd, sos_bandpass):
    def _build():
        rows = []
        for s in SESS:
            f0 = float(np.median(fund[s]))
            for name, lines in (("global 50.000 Hz", [50.0 * k for k in range(1, 8)]), (f"per-session {f0:.3f} Hz", [f0 * k for k in range(1, 8)])):
                y = sos_bandpass(en.filt_notch(lines, 30)(emg_all[s]), 30, 300)
                _, sh = en.hum_share_by_window(y, lines=en.ALL_LINES, band=(0, 500))
                rows.append({"session": SHORT[s], "centre": name, "median minute (%)": round(100 * float(np.median(sh)), 2), "worst minute (%)": round(100 * float(sh.max()), 1)})
        return pd.DataFrame(rows)
    centring = en.cached("global_vs_session_centre.csv", _build)
    mo.ui.table(centring, selection=None)
    return (centring,)


@app.cell(hide_code=True)
def _(centring, mo):
    mo.md(f"""
    **Decision: one global setting (50.000 Hz fundamental, 7 harmonics, Q = 30).** The median-minute hum is {centring['median minute (%)'].max():.2f}% or less with either approach in every session, so per-session centring does not improve the typical case. It lowers the worst minute only in session 07-09, the one whose frequency drifts, and the worst minute is dominated by minutes with real movement. The notches are wide enough (0.8 Hz either side of 50 Hz, scaling up with the harmonic) to cover the drift measured in 9.3, so a measured centre adds a failure mode (a wrong estimate on a noisy recording) for no gain in the median.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.8 What the filter does to the spectrum (before and after)

    The raw EMG spectrum (grey) and the spectrum after the chosen notch cascade (red) for each session. The red curve should follow the grey one **everywhere except at the lines**, i.e., the smooth muscle floor must not change, and the spikes should be gone, leaving only narrow dips.

    The title shows the **floor kept**: the median of the spectrum away from the lines (a running median, so spikes do not count) after / before, over 30–300 Hz. A value of 1.0 means the muscle floor is untouched; Q = 30 is expected to reduce it to about 0.75 because the notch skirts take some floor with them.
    """)
    return


@app.cell
def _(SESS, emg_all, en, np, plt, spec):
    _fig, _ax = plt.subplots(2, 2, figsize=(12, 6.5))
    for _a, _s in zip(_ax.ravel(), SESS):
        _f, _P = spec[_s]
        _y = en.filt_notch(en.ALL_LINES[:7], 30)(emg_all[_s])
        _, _P1 = en.welch(_y, nperseg=2 ** 15)
        _b = (_f >= 30) & (_f <= 300)
        _kept = float(np.mean(en.local_floor(_f, _P1)[_b] / en.local_floor(_f, _P)[_b]))
        _a.semilogy(_f, _P, color="0.6", lw=0.6, label="raw"); _a.semilogy(_f, _P1, color="#c0392b", lw=0.6, label="after notches")
        _a.set_xlim(0, 500); _a.set_xlabel("frequency (Hz)"); _a.set_title(f"{_s}  ·  floor kept {_kept:.2f}", loc="left", fontsize=9); _a.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.9 Does it keep real muscle bursts? (development session)

    Session 2026-06-18_16-25-09_000 has bursts that were identified by eye before any filtering (about 17 s, 95.5 s, 111–112 s, 320–322 s). The plot overlays the RMS **before** the notch (grey) and **after** (red), each divided by its own session median so they are on the same scale, for 85–100 s, which contains the 95.5 s burst, and for the whole session at the right.

    **What to look for.** The bursts must be at the same times with the same shape, only (if anything) taller, because removing the hum lowers the median they are compared with. The text under the plot gives the peak ratio and its timing before and after.
    """)
    return


@app.cell
def _(DropParams, SID, emg, emg_filter, en, io, mo, np, plt, rms_envelope):
    _off, _on = DropParams(notch=False), DropParams()
    _r0, _r1 = rms_envelope(emg_filter(emg, _off), _off), rms_envelope(emg_filter(emg, _on), _on)
    _m0, _m1 = np.median(_r0), np.median(_r1)
    _tt = np.arange(len(emg)) / 1000
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 2]})
    _a, _b = 85000, 100000
    _ax[0].plot(_tt[_a:_b], _r0[_a:_b] / _m0, color="0.55", label="before notch"); _ax[0].plot(_tt[_a:_b], _r1[_a:_b] / _m1, color="#c0392b", label="after notch")
    _ax[0].axhline(2.5, color="k", ls=":"); _ax[0].set_xlabel("time (s)"); _ax[0].set_ylabel("RMS / session median"); _ax[0].legend(fontsize=8)
    _ax[1].plot(_tt, _r0 / _m0, color="0.55", lw=0.6); _ax[1].plot(_tt, _r1 / _m1, color="#c0392b", lw=0.6)
    _ax[1].axhline(2.5, color="k", ls=":"); _ax[1].set_yscale("log"); _ax[1].set_xlabel("time (s)")
    _fig.tight_layout()
    _p0, _p1 = _r0[_a:_b].argmax(), _r1[_a:_b].argmax()
    _e0, _e1 = len(en.burst_episodes(_r0, _m0)), len(en.burst_episodes(_r1, _m1))
    mo.vstack([_fig, mo.md(f"""
    **Result.** The 95.5 s burst peaks at {(_a + _p0) / 1000:.3f} s before and {(_a + _p1) / 1000:.3f} s after the notch (a shift of {abs(_p1 - _p0)} ms), and its peak ratio goes from **{_r0[_a:_b].max() / _m0:.1f}×** to **{_r1[_a:_b].max() / _m1:.1f}×**. The two RMS curves are strongly correlated over the whole session (r = {np.corrcoef(_r0, _r1)[0, 1]:.3f}). The number of merged burst episodes above 2.5× goes from {_e0} to {_e1}: the same bursts (about 17 s, 95 s, 111 s, 320 s) plus a few weaker ones that were previously masked. **The known bursts are preserved.**
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.10 Do bursts appear in 07-21, and are they muscle?

    Before the notch, 07-21 has essentially no bursts (max/median RMS 1.0–1.1 per minute). The plot shows three bursts after the notch, picked as the strongest, a middle one and a weak one (all ≥ 0.3 s above 2.5× the median), in ±5 s windows: the RMS *before* the notch (grey, flat) against *after* (red). Three checks that these are muscle activity and not an artefact of filtering follow.

    **What to look for here.** Grey flat and red clearly rising above the dotted 2.5× line at the burst.
    """)
    return


@app.cell
def _(DropParams, emg_all, emg_filter, en, np, plt, rms_envelope):
    _s = "2026-07-21_11-48-31_000"
    _off, _on = DropParams(notch=False), DropParams()
    r07_off = rms_envelope(emg_filter(emg_all[_s], _off), _off)
    r07_on = rms_envelope(emg_filter(emg_all[_s], _on), _on)
    _m0, _m1 = np.median(r07_off), np.median(r07_on)
    _ep = [e for e in en.burst_episodes(r07_on, _m1) if e[1] - e[0] >= 0.3 and 10 < e[0] < 10990]
    _peaks = np.array([r07_on[int(a * 1000):int(b * 1000) + 1].max() / _m1 for a, b in _ep])
    _order = np.argsort(_peaks)
    ex = [_ep[_order[-1]], _ep[_order[len(_order) // 2]], _ep[_order[len(_order) // 10]]]
    _fig, _ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for _a, (_t0, _t1) in zip(_ax, ex):
        _c = (_t0 + _t1) / 2; _i0, _i1 = int((_c - 5) * 1000), int((_c + 5) * 1000)
        _x = np.arange(_i0, _i1) / 1000
        _a.plot(_x, r07_off[_i0:_i1] / _m0, color="0.55", label="before notch"); _a.plot(_x, r07_on[_i0:_i1] / _m1, color="#c0392b", label="after notch")
        _a.axhline(2.5, color="k", ls=":"); _a.set_xlabel("time (s)"); _a.set_title(f"burst at {_t0:.1f} s, peak {r07_on[int(_t0 * 1000):int(_t1 * 1000) + 1].max() / _m1:.1f}×", loc="left", fontsize=9)
    _ax[0].set_ylabel("RMS / session median"); _ax[0].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return ex, r07_off, r07_on


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the examples show.** The strongest burst (35×) and the middle one (4.3×) are clear, isolated rises. The weakest example (2.8×, just over the threshold) is different: its RMS rises and falls slowly with a period of about 3.5 s and only reaches the threshold at the peaks. That could be real rhythmic activity, or residual hum slowly beating, and the data here cannot tell which. Bursts close to 2.5× in 07-21 should therefore be treated with more caution than strong ones; this is one more reason to look at the video for them.

    **Check 1: what do the bursts look like in frequency?** Real muscle activity raises power across the **whole 30–300 Hz band**. An artefact of the filter, or hum that is modulated when the animal moves, would add power only **right next to the 50·k Hz lines**. The plot shows the ratio of the average spectrum *inside* bursts to the average *in quiet stretches* (RMS below 1.3× median), after the notch, for 07-21. Dashed lines mark 50·k Hz.

    **What to look for.** A ratio well above 1 at all frequencies between the dips (the dips at the lines are the notches themselves, where there is nothing left to compare). If the broadband ratio is much larger than the one near the lines, the extra power is not coming from the hum.
    """)
    return


@app.cell
def _(SESS, emg_all, en, mo, np, plt, r07_on):
    _s = "2026-07-21_11-48-31_000"
    _y = en.filt_notch(en.ALL_LINES[:7], 30)(emg_all[_s])
    _f, _Pb, _Pq, _nb, _nq = en.burst_quiet_spectrum(_y, r07_on)
    _ratio = _Pb / _Pq
    _band = (_f >= 30) & (_f <= 300)
    _near = np.zeros_like(_f, bool)
    for _k in range(1, 8):
        _near |= np.abs(_f - 50 * _k) <= 5
    _fig, _ax = plt.subplots(figsize=(10, 3.4))
    _ax.semilogy(_f, _ratio, color="0.2")
    for _k in range(1, 8):
        _ax.axvline(50 * _k, color="#c0392b", ls="--", lw=0.6)
    _ax.axhline(1, color="k", ls=":"); _ax.set_xlim(20, 320); _ax.set_xlabel("frequency (Hz)"); _ax.set_ylabel("PSD in bursts / PSD in quiet")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"""
    **Result.** ({_nb} bursts, {_nq} quiet stretches.) Away from the lines the burst power is **{np.median(_ratio[_band & ~_near]):.0f}×** the quiet power (and every bin is above 2×: {np.mean(_ratio[_band & ~_near] > 2):.0%}); within ±5 Hz of the lines it is {np.median(_ratio[_band & _near]):.0f}×. The increase is broadband, as for muscle, and not concentrated around the hum.
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Check 2: does the video show movement during the bursts?** For up to 14 bursts per session (spread over the session, ≥ 0.3 s) the video was decoded and the **mean absolute change between consecutive frames** (a simple movement measure; 160×128 grey, left camera; each window runs from burst start to end plus 0.3 s, since movement can follow the muscle by a fraction of a second) was compared with the same measure in the same number of **quiet control windows of equal length** (EMG below 1.3× median for 3+ s). Frames are located through the camera frame times (`cam_t`).

    **What to look for.** Burst windows (red) clearly higher than quiet windows (grey). `AUC` is the chance that a random burst window shows more motion than a random control: 0.5 = no relation, 1 = perfect.
    """)
    return


@app.cell
def _(SESS, SHORT, en, mo, np, pd, plt, DropParams, emg_all, emg_filter, rms_envelope, r07_on):
    from scipy import stats as _st
    _vs = [s for s in SESS if s != "2026-06-18_16-25-09_001"]
    _p = DropParams()
    bq = {}
    for _s in _vs:
        _r = r07_on if _s.startswith("2026-07-21") else rms_envelope(emg_filter(emg_all[_s], _p), _p)
        bq[_s] = en.cached(f"burst_vs_quiet_motion_{_s}.csv", lambda s=_s, r=_r: en.burst_vs_quiet_motion(s, r))
    _fig, _ax = plt.subplots(figsize=(8, 3.6))
    _rows = []
    for _i, _s in enumerate(_vs):
        _d = bq[_s]; _b, _q = _d[_d.kind == "burst"].motion.values, _d[_d.kind == "quiet"].motion.values
        _u = _st.mannwhitneyu(_b, _q, alternative="greater")
        _ax.scatter(np.full(len(_q), _i - 0.15) + np.random.default_rng(1).uniform(-.06, .06, len(_q)), _q, color="0.6", s=18, label="quiet" if _i == 0 else None)
        _ax.scatter(np.full(len(_b), _i + 0.15) + np.random.default_rng(2).uniform(-.06, .06, len(_b)), _b, color="#c0392b", s=18, label="burst" if _i == 0 else None)
        _rows.append(f"| {SHORT[_s]} | {len(_b)} | {np.median(_b):.2f} | {np.median(_q):.2f} | {_u.statistic / (len(_b) * len(_q)):.2f} | {_u.pvalue:.4f} |")
    _ax.set_xticks(range(len(_vs))); _ax.set_xticklabels([SHORT[s] for s in _vs]); _ax.set_ylabel("mean |frame change|"); _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| session | bursts | median motion, bursts | median motion, quiet | AUC | p (one-sided) |\n|---|---|---|---|---|---|\n" + "\n".join(_rows) + "\n\n**Result.** In both long sessions bursts coincide with clearly more movement than quiet periods (about 3–4× the median motion), including **07-21, where there were no detectable bursts before the notch**. The development session has only 3 bursts per group, so its p-value cannot be small.")])
    return (bq,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.11 Does the 2.5× burst threshold still hold?

    The 2.5 × median rule was chosen on hum-contaminated RMS (section 10). It is re-examined here on the **notched** RMS in two ways.

    **Left: how many bursts does each threshold give?** Number of merged burst episodes against the threshold (dashed = before notch, solid = after). A **plateau** around 2.5 means the exact value is not critical; a steep slope means it is arbitrary.

    **Right: where does movement start?** For ~240 one-second windows per session (half uniformly random, half chosen where the EMG ratio exceeds 1.5 so that the high range is covered) the highest EMG ratio in the window is compared with the video motion in the same window. The line is the median motion per ratio bin. If 2.5 is a good threshold, motion should stay at its baseline below 2.5 and rise above it.

    **What to look for.** Left: slope near 2.5. Right: where the median leaves the baseline.
    """)
    return


@app.cell
def _(DropParams, SESS, SHORT, emg_all, emg_filter, en, np, plt, r07_off, r07_on, rms_envelope):
    _ks = np.arange(1.5, 8.01, 0.25)
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for _i, _s in enumerate(SESS):
        _p0, _p1 = DropParams(notch=False), DropParams()
        _r0 = r07_off if _s.startswith("2026-07-21") else rms_envelope(emg_filter(emg_all[_s], _p0), _p0)
        _r1 = r07_on if _s.startswith("2026-07-21") else rms_envelope(emg_filter(emg_all[_s], _p1), _p1)
        _n0, _ = en.episodes_vs_threshold(_r0, _ks); _n1, _ = en.episodes_vs_threshold(_r1, _ks)
        _dur_h = len(_r1) / 1000 / 3600
        _ax[0].plot(_ks, np.maximum(_n0 / _dur_h, 0.3), ls="--", color=f"C{_i}", lw=1); _ax[0].plot(_ks, np.maximum(_n1 / _dur_h, 0.3), color=f"C{_i}", label=SHORT[_s])
    _ax[0].axvline(2.5, color="k", ls=":"); _ax[0].set_yscale("log"); _ax[0].set_title("dashed = before notch, solid = after (zero shown at 0.3)", loc="left", fontsize=8); _ax[0].set_xlabel("threshold (× session median)"); _ax[0].set_ylabel("burst episodes per hour"); _ax[0].legend(fontsize=8)
    mv = {}
    for _i, _s in enumerate(("2026-07-09_14-23-53_002", "2026-07-21_11-48-31_000")):
        _r1 = r07_on if _s.startswith("2026-07-21") else rms_envelope(emg_filter(emg_all[_s], DropParams()), DropParams())
        mv[_s] = en.cached(f"motion_vs_ratio_{_s}.csv", lambda s=_s, r=_r1: en.motion_vs_ratio(s, r))
        _d = mv[_s]; _bins = [0, 1.2, 1.5, 2, 2.5, 3, 4, 6, 100]
        _g = _d.groupby(__import__("pandas").cut(_d.ratio, _bins), observed=True).motion.median()
        _ax[1].plot(range(len(_g)), _g.values, "o-", label=SHORT[_s], color=f"C{[2, 3][_i]}")
    _ax[1].set_xticks(range(8)); _ax[1].set_xticklabels(["<1.2", "1.2–1.5", "1.5–2", "2–2.5", "2.5–3", "3–4", "4–6", ">6"], fontsize=8)
    _ax[1].axvline(3.5, color="k", ls=":"); _ax[1].set_xlabel("highest EMG ratio in the 1 s window"); _ax[1].set_ylabel("median video motion"); _ax[1].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return (mv,)


@app.cell(hide_code=True)
def _(mo, mv):
    _t = []
    for _s, _d in mv.items():
        _lo = _d[_d.ratio < 2.5].motion.median(); _hi = _d[_d.ratio >= 3].motion.median(); _mid = _d[(_d.ratio >= 2.5) & (_d.ratio < 3)].motion.median()
        _t.append(f"| {_s[5:10]}…{_s[-3:]} | {_lo:.2f} | {_mid:.2f} | {_hi:.2f} |")
    mo.md("| session | motion, ratio < 2.5 | motion, 2.5–3 | motion, ≥ 3 |\n|---|---|---|---|\n" + "\n".join(_t) + """

    **Decision: keep 2.5.** The curves on the left have no sharp feature at 2.5 (the episode count falls smoothly with the threshold, before and after the notch), so the count alone neither supports nor contradicts the value. The right-hand plot gives the real evidence: motion stays at its quiet baseline (about 0.3–0.4) below 2.5 and is clearly above it for windows with ratio ≥ 3; in 07-09 the 2.5–3 bin has already risen, in 07-21 it has not yet. 2.5 therefore sits at the start of the rise in one session and slightly below it in the other. The new plot does **not** clearly support a different value, so the rule stays at 2.5 × median. (If a stricter rule is wanted for 07-21, 3 would match its video better.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9.12 What changed in the results

    The drop detection (sections 2–8) does not use EMG, so the **list of events is identical** before and after (the table below checks this). Only the two EMG columns change: `emg_burst` and `emg_peak_ratio`. The results from before the notch were kept in `results/drop_events_pre_notch/`.

    **How to read the table.** Per session and class: number of events, how many were flagged with an EMG burst before and after, and the median `emg_peak_ratio` (highest EMG RMS within ±1 s of the event divided by the session median). The last column is a **chance level**: the share of a session's time that lies within 1 s of an EMG burst, i.e., the fraction of events one would expect to be flagged if events and bursts were unrelated.
    """)
    return


@app.cell
def _(DROP_DIR, DropParams, SESS, emg_all, emg_filter, en, mo, np, pd, r07_on, rms_envelope):
    _a = pd.read_csv(DROP_DIR.parent / "drop_events_pre_notch" / "drop_events.csv"); _b = pd.read_csv(DROP_DIR / "drop_events.csv")
    _same = _a[["event_id", "type", "t_start_rec", "t_end_rec"]].equals(_b[["event_id", "type", "t_start_rec", "t_end_rec"]])
    _chance = {}
    for _s in SESS:
        _r = r07_on if _s.startswith("2026-07-21") else rms_envelope(emg_filter(emg_all[_s], DropParams()), DropParams())
        _m = np.zeros(len(_r), bool)
        for _x, _y in en.burst_episodes(_r, np.median(_r)):
            _m[max(0, int((_x - 1) * 1000)):int((_y + 1) * 1000)] = True
        _chance[_s] = _m.mean()
    _g = lambda d: d.groupby(["session_id", "type"]).agg(events=("event_id", "size"), burst=("emg_burst", "sum"), median_ratio=("emg_peak_ratio", "median"))
    _t = _g(_a).add_suffix("_before").join(_g(_b).add_suffix("_after")).reset_index()
    _t["flagged before"] = (100 * _t.burst_before / _t.events_before).round(1)
    _t["flagged after"] = (100 * _t.burst_after / _t.events_after).round(1)
    _t["chance level (%)"] = (100 * _t.session_id.map(_chance)).round(1)
    _t["session"] = _t.session_id.str[5:10] + "…" + _t.session_id.str[-3:]
    _t[["median_ratio_before", "median_ratio_after"]] = _t[["median_ratio_before", "median_ratio_after"]].round(2)
    _v = _t[["session", "type", "events_before", "burst_before", "burst_after", "flagged before", "flagged after", "median_ratio_before", "median_ratio_after", "chance level (%)"]].rename(columns={"events_before": "events", "burst_before": "flagged before (n)", "burst_after": "flagged after (n)"})
    mo.vstack([mo.md(f"**Event list identical before and after: {_same}**"), mo.ui.table(_v, selection=None, page_size=15)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading the changes.** 07-21 goes from **0** flagged events (every ratio ≈ 1) to **315** (14%), with median ratios of 1.4–1.65 against 1.02 before: the rule now works there. 07-09 goes from 47 to 103 flagged and 001 from 29 to 48, because the median they are compared with fell by roughly 35–55% once the hum was removed (the same bursts now stand out more). In the dev session one event is newly flagged: a SINGLE drop at 214.8–215.1 s with peak ratio 2.52, just over the threshold.

    **A pattern worth knowing about.** Compare each class with its chance level. PAIR and SINGLE events are flagged at about the chance level in every session (for example 07-21: 13% and 12% against 13%). **ALL_THREE events are flagged 2–4× above chance in the three sessions where this can be tested** (07-21: 24% against 13%; 07-09: 12% against 6%; 001: 17% against 4%). So candidate breathing pauses accompany EMG bursts (movement) more often than random events, while candidate sensor failures do not. That is compatible with some of the "pauses" being movement-related (for example swallowing or a head movement) rather than pure breathing pauses, and it is a reason to look at the EMG-flagged ALL_THREE events in the viewer before interpreting them. In every session most ALL_THREE events (75–100%) are not accompanied by a burst.

    **Caveats.** (1) Absolute EMG RMS values are lower than before (hum removed, plus about 13% of the amplitude of genuine broadband activity), so RMS magnitudes are not comparable with earlier numbers; ratios are. (2) In 07-21 a small amount of hum remains (median minute under 1%, a few minutes between 5% and 11%, mostly minutes with movement). (3) The cause of the unusually strong hum in 07-21 (grounding or shielding in the recording setup) is not known from the data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. EMG: was the animal moving?

    EMG channel 0 is first cleaned of **mains hum** (notches at 50, 100, …, 350 Hz, Q = 30; section 9), then band-passed at **30–300 Hz** (muscle activity), squared and averaged over **0.25 s** (this is the RMS, the muscle "loudness"). It is compared with the **median of the whole session**. An event gets `emg_burst = True` if the RMS exceeds **2.5 × that median** at any time between 1 s before the event starts and 1 s after it ends. The highest ratio in that window is stored as `emg_peak_ratio`.

    **Justification plots.** Left: the EMG ratio over time (log scale); calm periods sit around 1 and bursts rise clearly above, with the 2.5 line dashed. Right: the number of separate bursts as the threshold is raised (for all four sessions, with and without the notch, see section 9.11). If the number of bursts is stable around 2.5 (a plateau), the exact choice does not matter much; if it were falling steeply there, 2.5 would be arbitrary.
    """)
    return


@app.cell
def _(emg, emg_rms, find_runs, np, plt, t):
    rms = emg_rms(emg)
    emg_med = float(np.median(rms))
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    _ax[0].plot(t, rms / emg_med, color="0.2", lw=0.6)
    _ax[0].axhline(2.5, color="k", ls="--", label="2.5 × median"); _ax[0].set_yscale("log"); _ax[0].set_xlabel("time (s)"); _ax[0].set_ylabel("EMG RMS / session median"); _ax[0].legend()
    _ks = np.arange(1.5, 6.1, 0.25)
    _n = [len(find_runs(rms > k * emg_med)[0]) for k in _ks]
    _ax[1].plot(_ks, _n, "o-", color="0.3"); _ax[1].axvline(2.5, color="k", ls="--"); _ax[1].set_xlabel("threshold (× median)"); _ax[1].set_ylabel("separate bursts")
    _fig.tight_layout()
    _fig
    return emg_med, rms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Putting it together: the events in the development session

    The table lists every event with its class, which sensors dropped, whether EMG was high around it, and how high. The timeline underneath places every event on the session: **red = ALL_THREE** (candidate breathing pause), **orange = PAIR**, **grey = SINGLE** (candidate sensor failures). Markers with a black outline are events *with* an EMG burst nearby (only a few in this short session, see the validation below).
    """)
    return


@app.cell
def _(P, SID, TYPE_COLORS, classify_events, detect_drops, emg_burst_flag, io, mo, np, pd, plt, ratio, rms, sensors):
    ev = classify_events({_s: detect_drops(ratio[_s], P) for _s in sensors}, P.overlap)
    _ts = np.array([e["t_start"] for e in ev]); _te = np.array([e["t_end"] for e in ev])
    _burst, _peak = emg_burst_flag(rms, _ts, _te, P)
    ev_df = pd.DataFrame({"type": [e["type"] for e in ev], "sensors": ["+".join(e["sensors"]) for e in ev],
                          "t_start": _ts.round(2), "t_end": _te.round(2), "duration_s": (_te - _ts).round(2),
                          "emg_burst": _burst, "emg_peak_ratio": _peak.round(2)})
    _fig, _ax = plt.subplots(figsize=(11, 1.8))
    for _typ, _y in (("ALL_THREE", 2), ("PAIR", 1), ("SINGLE", 0)):
        _m = ev_df.type == _typ
        _ax.scatter(ev_df.t_start[_m], np.full(_m.sum(), _y), c=[TYPE_COLORS[_typ]] * _m.sum(), s=40,
                    edgecolors=np.where(ev_df.emg_burst[_m], "k", "none"))
    _ax.set_yticks([0, 1, 2]); _ax.set_yticklabels(["SINGLE", "PAIR", "ALL_THREE"]); _ax.set_xlabel("time (s)")
    _fig.tight_layout()
    mo.vstack([mo.ui.table(ev_df, selection=None, page_size=40), _fig])
    return ev, ev_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Validation on this session

    What we know from looking at the raw data (see `PipelineStruct_marimo.py`):

    1. **~88 s**: all three sensors drop together, EMG normal -> expect an `ALL_THREE` event with `emg_burst = False`.
    2. **~92.5–93 s**: the same -> another `ALL_THREE`, EMG normal.
    3. **~95.5 s**: a sharp EMG burst -> any event near it must have `emg_burst = True`.

    The checks below compute each of these from the detector output. Then three ±5 s windows are plotted: top to bottom LFP ch0 (1–15 Hz), thermistor (1–15 Hz), airflow (1–15 Hz) and EMG (30–300 Hz) with its RMS in red. The shaded band is the detected event (red = ALL_THREE); a dashed vertical line marks the 95.5 s burst.
    """)
    return


@app.cell
def _(ev_df, mo, np, pd):
    def _overlapping(a, b):
        return ev_df[(ev_df.t_start < b) & (ev_df.t_end > a)]

    _c1 = _overlapping(86.5, 89.0)
    _c2 = _overlapping(91.5, 93.5)
    _near = ev_df[(ev_df.t_end > 95.5 - 1) & (ev_df.t_start < 95.5 + 1)]          # an event within ±1 s of the burst
    _nearest = ev_df.iloc[np.argmin(np.minimum(np.abs(ev_df.t_start - 95.5), np.abs(ev_df.t_end - 95.5)))]
    checks = pd.DataFrame([
        {"check": "ALL_THREE near 88 s, EMG normal", "pass": bool(((_c1.type == "ALL_THREE") & ~_c1.emg_burst).any()),
         "detail": _c1[["type", "t_start", "t_end", "emg_peak_ratio"]].to_dict("records")},
        {"check": "ALL_THREE near 92.5-93 s, EMG normal", "pass": bool(((_c2.type == "ALL_THREE") & ~_c2.emg_burst).any()),
         "detail": _c2[["type", "t_start", "t_end", "emg_peak_ratio"]].to_dict("records")},
        {"check": "event(s) near the 95.5 s EMG burst flagged emg_burst=True",
         "pass": bool(len(_near) > 0 and _near.emg_burst.all()),
         "detail": f"events within ±1 s of 95.5 s: {len(_near)}; nearest event is {_nearest.type} at {_nearest.t_start}-{_nearest.t_end} s (emg_peak_ratio {_nearest.emg_peak_ratio})"},
    ])
    mo.ui.table(checks, selection=None)
    return (checks,)


@app.cell
def _(COLORS, FS, P, TYPE_COLORS, emg, emg_filter, ev_df, io, np, plt, raw, rms, emg_med, sensors, sos_bandpass, t):
    filt = {s: sos_bandpass(raw[s], *P.band) for s in sensors}
    emg_f = emg_filter(emg, P)

    def event_window(centre, title, half=5.0, burst_at=None):
        i0, i1 = int((centre - half) * FS), int((centre + half) * FS)
        fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
        for a, s, lab in zip(ax[:3], ("lfp", "thermistor", "airflow"), ("LFP ch0", "thermistor", "airflow")):
            a.plot(t[i0:i1], filt[s][i0:i1], color=COLORS[s], lw=0.9); a.set_ylabel(lab)
        ax[3].plot(t[i0:i1], emg_f[i0:i1], color="0.6", lw=0.4); ax[3].plot(t[i0:i1], rms[i0:i1], color="#c0392b", lw=1.5, label="RMS")
        ax[3].axhline(2.5 * emg_med, color="k", ls=":", lw=1, label="2.5 × median"); ax[3].set_ylabel("EMG"); ax[3].legend(loc="upper right", fontsize=8)
        for _, r in ev_df[(ev_df.t_end > centre - half) & (ev_df.t_start < centre + half)].iterrows():
            for a in ax:
                a.axvspan(r.t_start, r.t_end, color=TYPE_COLORS[r.type], alpha=0.25)
        if burst_at:
            for a in ax:
                a.axvline(burst_at, color="k", ls="--", lw=1)
        ax[0].set_title(title, loc="left"); ax[3].set_xlabel("time (s)"); fig.tight_layout()
        return fig

    return (event_window,)


@app.cell
def _(event_window):
    event_window(88.0, "±5 s around the ~88 s event (shaded = detected event)")
    return


@app.cell
def _(event_window):
    event_window(92.6, "±5 s around the ~92.5–93 s event (dashed = EMG burst at 95.5 s)", burst_at=95.5)
    return


@app.cell
def _(event_window):
    event_window(95.5, "±5 s around the 95.5 s EMG burst: the sensors do NOT drop here", burst_at=95.5)
    return


@app.cell(hide_code=True)
def _(checks, mo):
    _lines = "\n".join(f"- **{'PASS' if r['pass'] else 'FAIL'}**: {r['check']}" for _, r in checks.iterrows())
    mo.md(f"""
    ### Validation result

    {_lines}

    **Interpretation.** The two known pauses are found as ALL_THREE with normal EMG. The third check fails *by construction*, not because the flag is broken: in the third plot the EMG RMS shoots up at 95.5 s (ratio about 14 after hum removal, 9 before), but the three breathing amplitudes **do not fall below the threshold** there. In 94–98 s the lowest ratios are 0.41 (thermistor), 0.63 (airflow) and 0.70 (LFP); the existing notebook describes this burst as coinciding with an LFP artifact burst, so there is **no event to flag**. The nearest event (92.0–93.2 s) ends 2.3 s before the burst, beyond the ±1 s window; widening the window enough to catch the burst would also flag that event, which must stay `emg_burst = False`.

    Sweeping the amplitude fraction up to 0.55 and the minimum duration down to 0.2 s never created an event at 95.5 s either. The EMG flag does work: with looser thresholds it flags short drops near EMG bursts at 17.8 s, 113 s and 321 s, and in three of the four sessions it flags events in every class (see section 13). The unit test `test_emg_burst_flag` checks it on a synthetic burst.

    **Decision for you.** Either (a) accept that the 95.5 s burst is an artifact burst rather than a breathing drop, so the expectation should be "no event there", or (b) add a separate "artifact" detector for amplitude *rises*. I have not changed anything to force this check to pass.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. All four sessions

    `scripts/run_drop_events.py` runs the same detector (same parameters, no per-session tuning) on every session and writes `results/drop_events/drop_events.csv` and `summary.json`. The table below reads that file. One row per session, class and EMG flag: number of events and median duration.

    **Things to expect.** The long sessions have far more events simply because they are 30 times longer. SINGLE events are the most common and the shortest (a single sensor wobbling); ALL_THREE events are the longest. Events with an EMG burst are a small minority.
    """)
    return


@app.cell
def _(DROP_DIR, mo, pd):
    events = pd.read_csv(DROP_DIR / "drop_events.csv")
    _tab = (events.groupby(["session_id", "type", "emg_burst"]).duration_s.agg(n="size", median_duration_s="median")
            .round(2).reset_index())
    mo.ui.table(_tab, selection=None, page_size=30)
    return (events,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. Mapping events to video frames

    The camera frame times (`leftCamera.times.npy`) say when each video frame was exposed on the recording clock. For an event time `t`, the frame is `searchsorted(cam_t, t)` (the first frame at or after `t`), and its position in the video is `frame / fps`, with fps measured from `cam_t` itself (about 60.005, not exactly 60). **No fixed offset is used**: the video starts ~7–21 s after the recording, and that is already inside `cam_t`. Events before the first frame or after the last one have no video fields (empty in the CSV).

    **One deviation from the task text, on purpose.** The task says to use the fps measured from `cam_t` (60.0047). The MP4 files, however, run on an exact 60.000 fps timeline, and that is the timeline a video player seeks on. Dividing the frame number by 60.0047 instead would put the video position up to **0.8 s too early by the end of a 3-hour session** (0.03 s in the 6-minute one). So when a video file exists the CSV uses the container rate, and falls back to the measured rate only when there is no video; the rate used is written to the `video_fps` column. Frame numbers are the same either way.

    Before trusting the mapping, the **real number of frames in each video** (read from the file header, no decoding) must equal `len(cam_t)`. The metadata says the camera times were not curated, so this is the check that the two agree. Expect `match = True` for every video that exists.
    """)
    return


@app.cell
def _(VIDEO_FILES, io, measure_fps, mo, mp4_info, pd):
    _rows = []
    for _s, _v in VIDEO_FILES.items():
        _ct = io.load_cam_times(_s, "left")
        if not _v:
            _rows.append({"session": _s, "camera": "-", "video file": "NOT FOUND", "video frames": None, "len(cam_t)": len(_ct),
                          "match": None, "fps": round(measure_fps(_ct), 4)})
        for _cam, _path in _v.items():
            _i = mp4_info(_path)
            _rows.append({"session": _s, "camera": _cam, "video file": _path.name, "video frames": _i["frames"], "len(cam_t)": len(_ct),
                          "match": _i["frames"] == len(_ct), "fps": round(measure_fps(_ct), 4), "codec": _i["codec"],
                          "size": f'{_i["width"]}x{_i["height"]}'})
    mo.ui.table(pd.DataFrame(_rows), selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15. Sync check: does the video line up with the EMG?

    **Method.** The viewer page was driven in Chrome (no opencv or ffmpeg needed: the browser decodes the video). The left-camera video of the development session was played through once; for every presented frame the mean absolute pixel change from the previous one was taken as a **motion energy** (whole frame, 160×128 grey). Because the browser only presented every 8th frame at 4× speed, that gives 2,322 motion values 0.13 s apart. The EMG RMS was averaged over exactly the same time span of each one (found through `cam_t`), then the EMG window was shifted by -1 to +1 s to see where the two agree best.

    **Result.** The correlation is r = 0.56 with **no shift**, peaks at 0 to +0.2 s (movement following the muscle burst, which is the physiological order), and falls steadily towards 0.39 at -1 s and 0.49 at +0.8 s. So there is **no offset or drift larger than about 0.1–0.2 s**, the resolution of this test. A single burst (95.5 s) was too noisy to judge on its own: the motion trace has several similar spikes in that window.

    **Limits.** (1) Whole-frame motion is a coarse proxy; a region around the jaw or face would be sharper. (2) Only the 6-minute session was tested. The long sessions rely on the container-fps rule in section 14, which was checked arithmetically but not with a late-session EMG burst. (3) The proper check is your own eye: open an EMG-burst event in the viewer, play at 0.25× and see whether the cursor crosses the burst when the animal moves.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16. What is still open

    - **Session 2026-06-18_16-25-09_001 has no video** among the files provided (nothing with 436,897 frames). Its events have frame numbers but no `video_file`, and the viewer shows them in signals-only mode.
    - **The 95.5 s validation check fails** as explained in section 12 (no drop event exists there); this needs your decision.
    - **MP4 clips** (video on top, traces with a moving cursor below) are available now that ffmpeg is installed: `uv run python scripts/export_clips.py --session <id> [--type ALL_THREE] [--emg 0|1] [--limit N]` writes them to `results/clips/` (gitignored). Two example clips of the development session are there already.
    - **Viewer:** `uv run python viewer/serve.py`, then open `http://localhost:8000/viewer/index.html` (the server is needed because the browser must seek in multi-GB files with HTTP Range requests, which `python -m http.server` does not support).
    """)
    return


if __name__ == "__main__":
    app.run()

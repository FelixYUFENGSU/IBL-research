# Decision log

Newest first. Each entry: what was decided, the evidence, the alternatives, what changed, and what is still open.
The evidence plots for the EMG entry are in `scripts/breathingPause.py`, section 9.

---

## 2026-11-10: Remove mains hum from the EMG before the 30–300 Hz band-pass

**Problem.** The `emg_burst` flag (RMS above 2.5 × the session median within ±1 s of a drop event) did not work in session 2026-07-21_11-48-31_000: 0 of 2189 events were ever flagged. Other sessions were affected less.

**Evidence** (all measured on EMG channel 0; sections 9.1–9.3 of the notebook).
- EMG ch0 contains mains hum: lines at 50·k Hz, k = 1…9, in every session. Share of the 30–300 Hz power within ±1 Hz of 50·k Hz (whole session): 0.43 (06-18 _000), 0.65 (_001), 0.69 (07-09), **1.00 (07-21)**. In 07-21 the lines are about 10⁵ times the muscle floor, and the band-passed RMS is about 2150 against about 42 in the dev session.
- Because the hum is steady, the RMS is nearly constant and the session median is dominated by it. Max/median RMS within a minute is 1.0–1.1 in 07-21 (burst rule cannot fire); the other sessions still reach 2–10 in some minutes.
- Lines sit at 50·k Hz in all sessions. The fundamental is stable in three sessions (range over the whole recording 0.05, 0.07, 0.09 Hz) and drifts in 07-09 (49.95–50.27 Hz, i.e. up to ±0.8 Hz at the 250 Hz harmonic). EMG ch1 has the same hum (hum share 0.41, 0.64, 0.69, 1.00); only ch0 is used by the pipeline.

**Options considered** (9.4–9.6; evaluated on the first 30 min of each session, the dev session in full).
- *Which harmonics:* notches at 50–250 Hz (the first quick test) leave 33% of the power in 07-21 as hum, because lines at 300 and 350 Hz leak through the 3rd-order band-pass. 50–350 Hz leaves 2.4%. Adding 400 and 450 Hz changes nothing (2.5%). Chosen: 50–350 Hz (7 harmonics).
- *Which filter:* (a) cascade of IIR notches with constant Q, (b) IIR comb filter with constant bandwidth, (c) spectral interpolation (STFT, magnitude in the line band replaced by the neighbouring floor). Hum left in 07-21 / white noise power kept in 30–300 Hz: notch Q=30 2.3% / 0.76; comb 1.7 Hz tooth 19.5% / 0.93; comb 0.8 Hz tooth 69% / 0.96; interpolation 3.1% / 0.94. All three keep a synthetic 6× burst at 6.0–6.2×, without time shift. The comb fails where it matters (its fixed bandwidth is too narrow for the broad skirts of the high harmonics). Interpolation works and costs the least broadband power but is non-linear, block-based and heavier for 11 million samples, and gave the same burst ratios as the notches. Chosen: notch cascade.
- *Q:* criterion fixed beforehand: whole-session hum left < 2.5% and at most 5% of the one-minute windows above 5%. Q = 15, 20 and 30 pass in all four sessions; Q = 40 and above fail in 07-21 (3.1% whole session, 13.7% of minutes above 5%). Chosen: **Q = 30** (bandwidth f/30: 1.7 Hz at 50 Hz, 11.7 Hz at 350 Hz), the narrowest that passes. Cost: 24% of white-noise power in 30–300 Hz (13% in amplitude).
- *Global or per session:* the median-minute hum is 0.16–0.60% for either; per-session centring only improves the worst minute in 07-09 (4.6% to 2.3%). Chosen: global 50.000 Hz (no estimation step that could fail).

**Does it keep real bursts?** (9.9–9.10)
- Dev session: the 95.5 s burst peaks 25 ms earlier after notching and its ratio goes from 9.2 to 13.8; RMS before/after correlate at r = 0.967; the same bursts (about 17, 95, 111, 320 s) remain.
- 07-21: 493 burst episodes now exist. Their spectrum is broadband (burst/quiet power 55× away from the lines against 22× within ±5 Hz of them), and video motion is higher during bursts (median 1.37 against 0.35 in quiet controls, AUC 0.79, p = 0.005, 14 + 14 windows). Same test in 07-09: 1.13 against 0.32, AUC 0.90, p = 0.0002.

**Threshold re-check** (9.11). Kept 2.5 × median. Video motion stays at its quiet level (0.3–0.4) for EMG ratios below 2.5 and rises above it (07-09: 0.89 in the 2.5–3 bin; 07-21: still 0.38 there and 1.18 at 3–4). The new evidence does not clearly support a different value. For 07-21 alone, 3 would fit the video slightly better.

**What changed in the code.**
- `src/ibl/signal.py`: `notch_sos`, `mains_notch` (zero-phase, float64 inside, float32 out).
- `src/ibl/drops.py`: `DropParams` gains `notch` (True), `notch_f0` (50), `notch_harmonics` (7), `notch_q` (30). New `emg_filter` (notch then 30–300 Hz band-pass) and `rms_envelope`; `emg_rms` now uses them.
- `src/ibl/viewer.py` (and so `src/ibl/clips.py`) now use `emg_filter` / `rms_envelope`, so the viewer traces, the RMS overlay and the clips show the same signal as the detector.
- `src/ibl/emgnoise.py`: measurements and candidate filters used for the evidence (new).
- `tests/test_drops.py`: `test_mains_notch_suppresses_lines_and_keeps_burst` (synthetic 50/150/250 Hz hum 1e3 times the noise power plus a 0.3 s burst: lines suppressed by more than 40 dB, burst undetectable without the notch and detected with it).
- `scripts/breathingPause.py`: new section 9; the EMG section is now 10 and later sections are renumbered.

**Effect on results.** The drop events are identical (same 4652 events, same times); only `emg_burst` and `emg_peak_ratio` change. Previous results are in `results/drop_events_pre_notch/`.

| session | events | flagged before | flagged after | median `emg_peak_ratio` | 90th percentile |
|---|---|---|---|---|---|
| 06-18 _000 | 35 | 0 | 1 | 1.17 to 1.23 | 1.40 to 1.65 |
| 06-18 _001 | 823 | 29 | 48 | 1.20 to 1.29 | 1.50 to 1.97 |
| 07-09 _002 | 1605 | 47 | 103 | 1.18 to 1.33 | 1.82 to 2.08 |
| 07-21 _000 | 2189 | 0 | 315 | 1.02 to 1.47 | 1.07 to 2.83 |

Compared with the chance level (share of session time within 1 s of a burst), PAIR and SINGLE events are flagged at chance in every session, while ALL_THREE events are flagged 2–4× above chance in 001, 07-09 and 07-21 (17% / 12% / 24% against 4% / 6% / 13%). Most ALL_THREE events (75% or more) are still not accompanied by a burst.

**Open questions.**
1. **Grounding / shielding on 07-21.** The hum there is about 10⁵ × the muscle floor, much larger than in the other sessions. The cause (electrode, ground, cable or room) is not visible in the data and should be raised with the Cavelli lab. Residual hum remains after the notch (median minute under 1%, a few minutes between 5% and 11%), mostly in minutes with movement.
2. **Weak bursts in 07-21** (ratio close to 2.5) include slow rhythmic fluctuations (period about 3.5 s) that may be residual hum beating rather than muscle. They should be viewed before being trusted.
3. **ALL_THREE events co-occur with EMG bursts more than chance.** Some candidate breathing pauses may therefore be movement-related. This needs to be looked at in the viewer.
4. **Absolute EMG RMS values are not comparable** with numbers from before this change; ratios are.
5. **Session 06-18 _001** has no video, so its EMG flags cannot be checked against the video.

---

## 2026-11-09: Video time uses the MP4 container rate, not the measured camera rate

The task text said to compute video time as `frame / fps` with fps measured from `cam_t` (60.0047 Hz). The MP4 files run on an exact 60.000 fps timeline (for example 625257 frames over 10420.95 s), which is what a player seeks on; using 60.0047 would put the position about 0.8 s early at the end of a 3-hour session (0.03 s in the 6.6-minute one). `add_video_columns` therefore uses the container rate when the video file exists and the measured rate otherwise, and writes the rate used into the `video_fps` column. Frame numbers (`searchsorted(cam_t, t)`) are unaffected. A whole-session cross-correlation of video motion against EMG in the dev session found no offset above about 0.1–0.2 s.

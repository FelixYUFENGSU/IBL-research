# Methods: breathing-pause detection, video mapping and export tools

Mouse M1, IBL Phase 1 respiration + video project (Cavelli Lab).

This document describes what was done to the data, why, and what came out. Code is background: each
step names the file and function that does it. Every number comes from the code, the notebook text
(`scripts/breathingPause.py`, cited as "NB §n"), `docs/decisions.md`, or `results/`. Counts in section 9
were computed with a short read-only script from `results/drop_events/drop_events.csv`. Where a value is
recorded nowhere, it says "not recorded". Disagreements between sources are listed in section 16.

Nothing under `data/` or the video folders was modified at any point (loaders use `np.load` only).

---

## 1. Overview

**Question.** Three sensors record breathing at the same time: the thermistor, the airflow sensor and the
LFP (channel 0). Sometimes all of them shrink briefly. A *real pause in breathing* should make all three
lose their rhythm together; a *sensor problem* should affect only one or two. Movement (jaw, face) can
disturb a sensor without any change in breathing, so every event is also tagged with whether the EMG
(muscle signal, channel 0) was high around it.

**Sessions** (all 1000 Hz, one shared clock; sensors from `session_info.json`):

| Session | Samples | Duration | Respiration sensors | LFP / EMG used | Camera frames | Video |
|---|---|---|---|---|---|---|
| 2026-06-18_16-25-09_000 (development) | 397,714 | 397.7 s (6.6 min) | thermistor, airflow | LFP ch0, EMG ch0 | 22,217 | yes |
| 2026-06-18_16-25-09_001 | 7,320,918 | 7320.9 s (122.0 min) | thermistor, airflow | LFP ch0, EMG ch0 | 436,897 | **none found** |
| 2026-07-09_14-23-53_002 | 10,452,566 | 10452.6 s (174.2 min) | thermistor, airflow | LFP ch0, EMG ch0 | 625,257 | yes |
| 2026-07-21_11-48-31_000 | 11,016,611 | 11016.6 s (183.6 min) | thermistor, airflow | LFP ch0, EMG ch0 | 659,515 | yes |

The chest sensor was not recorded in any session. Durations are samples / 1000 (computed from the files).

**Pipeline in one paragraph.** For each sensor, the 1-15 Hz breathing rhythm is isolated, its amplitude
(Hilbert envelope) is computed and divided by its own 60 s rolling median. Stretches where this ratio is
below 0.35 for at least 0.3 s (nearby stretches merged) are "drops". Drops of different sensors that
overlap by at least half of the shorter one form one event, labelled ALL_THREE, PAIR or SINGLE. EMG
(mains hum removed, 30-300 Hz band-passed, 0.25 s RMS) flags events that have a burst within ±1 s. Each
event gets a video frame and timestamp. A viewer and an MP4 clip exporter show video and traces together.

**Main findings.**
- 4652 events over four sessions: 35 (development), 823, 1605, 2189.
- On the development session, the two known breathing pauses (about 88 s and 92.5-93 s) are found as
  ALL_THREE with normal EMG (2 of 3 validation checks pass); the third check (an event next to the 95.5 s
  EMG burst) fails because no drop exists there (section 8).
- EMG ch0 carries strong 50 Hz mains hum in every session; in 2026-07-21 it is essentially all of the
  EMG. A notch filter was added (section 7.2) and the burst flag now works there (0 to 315 flagged events).
- ALL_THREE events are flagged with an EMG burst about 2-4 times more often than expected by chance in
  three sessions; PAIR and SINGLE are at chance. Some candidate pauses may therefore be movement-related.
- Video frame counts match the camera times exactly for all three videos (section 10).

---

## 2. Data

- **Sampling and clock.** All ALF signals are 1000 Hz float32, `t = arange(N) / 1000` (`config.FS`,
  `io.time_axis`). Respiration, LFP, EMG and camera frame times share the recording clock.
- **Missing sensors.** A sensor that was not recorded is saved as an all-zero array. The loaders therefore
  read `session_info.json -> channels.respiration_available` instead of inspecting the data
  (`io.available_sensors`). Here that is thermistor and airflow.
- **Channels used.** LFP channel 0 and EMG channel 0 only, read one channel at a time from the
  memory-mapped arrays (`io.load_lfp`, `io.load_emg`), to limit memory (up to 11 million samples).
  EMG ch1 is not processed.
- **Camera times.** `leftCamera.times.npy`: recording time of each frame (`io.load_cam_times`). Left and
  right frame times were identical. The first frame comes at 17.74 s (000), 20.76 s (001), 14.69 s (002)
  and 6.79 s (07-21) of the recording, so no fixed video offset is used: it is already inside the times.

Implemented in: `src/ibl/io.py`, `src/ibl/config.py`.

---

## 3. Breathing-amplitude estimation

**Purpose.** Ask "is the sensor still showing a breathing rhythm?" by measuring the *strength of the
oscillation* rather than the raw value.

**Procedure.** Per sensor, one at a time in float32:
1. **Band-pass** 1-15 Hz, 3rd-order Butterworth, applied forward and backward (zero-phase, no time shift),
   in second-order sections for numerical stability.
2. **Hilbert amplitude**: the instantaneous height of the oscillation regardless of phase (magnitude of
   the analytic signal).
3. **Smoothing**: moving average over 0.25 s so single cycles do not make the amplitude flicker.

**Justification.** Breathing is a roughly 2-10 Hz rhythm (NB §2), so 1-15 Hz keeps it while removing drift
and fast noise. 0.25 s smoothing is a judgement; no plot varies it. NB §2 shows the envelope hugging the
peaks of the filtered signal for 85-100 s and shrinking in all three sensors near 87.5-88.5 s and 92-93 s.

**Result.** The envelope behaves as described in NB §2. (The notebook states this as an expected
appearance of the plot; no number is recorded.)

**Implemented in:** `signal.sos_bandpass`, `drops.hilbert_amplitude` (an FFT-based float32 version),
`drops.amplitude_envelope`.

---

## 4. Relative-amplitude reference (rolling median)

**Purpose.** Absolute amplitude is not comparable across sensors (different units) and drifts within a
session, so each sensor is compared with its own typical amplitude.

**Procedure.** `ratio = amplitude / rolling median`, with a 60 s rolling median. For speed the median is
computed on 4 Hz block means (blocks of 250 samples; window of an odd number of blocks) and interpolated
back to 1000 Hz. A ratio of 1 means "typical right now"; 0.35 means about a third of typical.

**Justification.** A median (not a mean) is used so the drops themselves do not pull the reference down.
60 s is long compared with a drop (about 1 s) and short compared with a session (NB §3). NB §3 plots the
amplitude, the median and 0.35 × median on a log scale for the development session: the amplitude wanders
around the median and occasionally dives below the 0.35 line. The block-mean shortcut is claimed
in NB §3 to give "the same smooth line"; no comparison number is recorded.

**Result.** Qualitative (NB §3 plot).

**Implemented in:** `drops.rolling_median`; ratio formed in `drops.sensor_ratios`.

---

## 5. Drop detection

A **drop** is a run of samples with `ratio < 0.35`, with nearby runs merged and short runs discarded.
Three parameters, each tuned on the development session (NB §4-7).

### 5.1 Amplitude threshold: 0.35 × median
- **Purpose.** Decide how low is "a drop".
- **Procedure.** `ratio < frac`, `frac = 0.35`.
- **Justification (NB §4).** Histogram of log10(ratio) per sensor. Normal breathing forms a hump near 1; a
  real drop should be a separate left shoulder. A good threshold sits in the trough. The legend prints
  the percentage of time below 0.35 per sensor (a few percent expected); **those percentages are not
  recorded in the notebook text.** The final choice rests on the grid in section 5.4.

### 5.2 Merge gap: 0.2 s
- **Purpose.** The amplitude of an oscillation passes through zero between breaths, so one pause can show
  brief bounce-backs above the threshold. Without merging, one pause would be cut into short pieces and
  then removed by the minimum-duration rule.
- **Procedure.** Gaps shorter than 0.2 s between consecutive low runs are joined.
- **Justification (NB §5).** Histogram of the gaps (log x). Short bounce-back gaps pile up at very short
  values, separate events are much longer. 0.2 s should cut off the short pile; the share of gaps shorter
  than 0.2 s is printed in the legend and is **not recorded in text**.

### 5.3 Minimum duration: 0.3 s
- **Purpose.** Ignore flickers of noise without losing swallow-length pauses. NB §6 argues a single breath
  lasts roughly 0.1-0.3 s (stated in the notebook; no measurement is recorded).
- **Procedure.** Merged runs shorter than 0.3 s are removed.
- **Justification (NB §6).** Left: histogram of run durations (many short, a tail of longer ones). Right:
  number of events and of ALL_THREE events against the minimum duration (0.1-0.8 s).

### 5.4 Joint tuning (NB §7)
Grid of amplitude fraction (0.25-0.50) × minimum duration (0.2-0.5 s) on the development session. The
requirements: the two known pauses (about 88 s and 92.5-93 s) stay detected, and the ALL_THREE count is
stable. **Result, from NB §7 text:**
- At fraction 0.35 the ALL_THREE count is 5 for every minimum duration from 0.3 to 0.5 s; at 0.2 s it is 7
  (extra short fragments).
- Looser fractions at 0.3 s (0.40, 0.45, 0.50) give 6, 7 and 10 ALL_THREE events, while the total goes
  from 35 to 45, 69 and 94.
- Tighter fractions lose real events: at 0.25 the ALL_THREE count is 3 (0.3 s) and 1 (0.4 s); at 0.4 s the
  88 s event disappears.
- **0.35 and 0.3 s were kept**, at the start of the plateau.

**Implemented in:** `drops.detect_drops` (`DropParams.frac`, `merge_s`, `min_dur_s`), `drops.find_runs`.

---

## 6. Cross-sensor event classification

**Purpose.** Separate drops seen by all sensors (candidate breathing pause) from drops seen by fewer
(candidate sensor problem).

**Procedure.** Drops from different sensors are linked when the overlap is at least **50 % of the shorter
drop**. Links are followed through chains (A overlaps B, B overlaps C: one event), using union-find. The
event spans from the earliest start to the latest end. Class by number of sensors: ALL_THREE (3), PAIR (2),
SINGLE (1). The list of sensors is kept (`sensors_dropped`, e.g. `airflow+lfp`).

**Justification (NB §8).** For every pair of overlapping drops from different sensors, a histogram of
overlap / shorter-drop. If the values fall into two groups (barely touching near 0, near-identical near 1),
any cut-off between them gives the same answer and 0.5 is a safe middle. The plot prints how many touching
pairs fall below 0.5; **that count is not recorded in the notebook text**, so whether the two groups are
cleanly separated cannot be confirmed from the text alone.

**Result.** Event counts per class are in section 9.

**Implemented in:** `drops.classify_events` (`DropParams.overlap`).

---

## 7. EMG movement flag

### 7.1 The flag
- **Purpose.** Mark events that coincide with movement (muscle activity).
- **Procedure.** EMG ch0 → mains-hum notch (7.2) → 30-300 Hz band-pass (3rd-order Butterworth,
  zero-phase) → square → moving average over 0.25 s → square root (**RMS**, the muscle "loudness"). The
  RMS is divided by its **median over the whole session**. An event gets `emg_burst = True` if the ratio
  exceeds **2.5** at any time in [start − 1 s, end + 1 s]; the maximum ratio in that window is stored as
  `emg_peak_ratio`.
- **Justification (NB §10, §9.11).** Left plot: EMG ratio over time, calm periods near 1, bursts above the
  2.5 line. Right plot: number of separate bursts against the threshold. The 2.5 value was a starting
  value from the task; the evidence for keeping it is in 7.3.

**Implemented in:** `drops.emg_filter`, `rms_envelope`, `emg_rms`, `emg_burst_flag`.

### 7.2 Mains-hum (line-noise) removal
Full evidence is in NB §9 and `docs/decisions.md` (entry 2026-11-10).

- **Purpose.** EMG ch0 contains 50 Hz mains hum and harmonics. Hum is steady, so it adds a large constant
  to the RMS, and a burst compared with the session median looks small. The original rule (no notch) never
  fired in 2026-07-21 (0 of 2189 events flagged).
- **Procedure.** Before the 30-300 Hz band-pass, zero-phase IIR notch filters at 50, 100, …, 350 Hz
  (7 lines), each with quality factor **Q = 30** (bandwidth = f/Q: 1.7 Hz at 50 Hz, 11.7 Hz at 350 Hz),
  fundamental fixed at 50.000 Hz for all sessions. Computed in float64, one channel at a time, cast to
  float32.
- **Justification, step by step (NB §9.1-9.8).**

  | Question | Evidence | Decision |
  |---|---|---|
  | How big is the hum? (9.1, 9.2) | Share of 30-300 Hz power within ±1 Hz of the lines: 0.43 / 0.65 / 0.69 in the first three sessions (order 000, 001, 07-09 per `decisions.md`) and 1.00 in 07-21; 07-21 lines are about 10^5 × the muscle floor. Without notch, max/median RMS per minute is 1.0-1.1 in 07-21. | Hum removal needed |
  | Does it drift? (9.3) | Fundamental stable in three sessions (range over the recording 0.05, 0.07, 0.09 Hz), drifts in 07-09 (49.95-50.27 Hz, up to ±0.8 Hz at 250 Hz). | Notch width must grow with harmonic: constant Q |
  | Which harmonics? (9.4) | Hum left in 07-21 after notching 50-250 Hz: 33 %; 50-350 Hz: 2.4 %; 50-450 Hz: 2.5 %. | 50-350 Hz (7 lines) |
  | Which filter? (9.5) | Notch cascade, IIR comb and STFT spectral interpolation compared on hum left, white noise kept and a synthetic 6× burst. Comb leaves 19.5 % (1.7 Hz tooth) or 69 % (0.8 Hz) in 07-21. Interpolation leaves 3.1 % but is non-linear and heavier. The notch leaves 2.3 %, keeps 0.76 of white-noise power and returns the synthetic burst at 6.2 (true 6). | Notch cascade |
  | Which Q? (9.6) | Criterion fixed beforehand: whole-session hum left < 2.5 % and at most 5 % of one-minute windows above 5 % hum. Q = 15, 20, 30 pass in all four sessions; Q ≥ 40 fails in 07-21 (3.1 % whole session, 13.7 % of minutes above 5 %). | Q = 30 (narrowest that passes) |
  | Global or per session? (9.7) | Median-minute hum 0.16-0.60 % either way; per-session centring only improves the worst minute in 07-09 (4.6 % to 2.3 %). | Global 50.000 Hz |

- **Result (NB §9.9-9.12, `decisions.md`).**
  - Development session: the known 95.5 s burst keeps its time (peak shifts by 25 ms) and its ratio goes
    from 9.2 to 13.8; RMS before and after are correlated at r = 0.967; the same bursts (about 17, 95, 111,
    320 s) remain.
  - 2026-07-21: 493 burst episodes now exist. Burst power is 55× the quiet power away from the lines and
    22× within ±5 Hz of them (broadband, as for muscle). Video motion is higher during bursts (median
    1.37 versus 0.35 in quiet controls; AUC 0.79; one-sided p = 0.005; 14 + 14 windows). Same test in
    07-09: 1.13 versus 0.32, AUC 0.90, p = 0.0002.
  - Cost: the notches remove about 24 % of white-noise power in 30-300 Hz (about 13 % in amplitude).
    Absolute RMS values are therefore not comparable with pre-notch values; ratios are.
  - Drop events are unchanged (same 4652 events); only `emg_burst` and `emg_peak_ratio` change:

    | Session | Flagged before | Flagged after |
    |---|---|---|
    | 000 | 0 | 1 |
    | 001 | 29 | 48 |
    | 07-09 | 47 | 103 |
    | 07-21 | 0 | 315 |

    The pre-notch results are kept in `results/drop_events_pre_notch/` (untracked).

**Implemented in:** `signal.notch_sos`, `signal.mains_notch`, `drops.emg_filter`; parameters in
`DropParams` (`notch`, `notch_f0`, `notch_harmonics`, `notch_q`). The measurements are in
`src/ibl/emgnoise.py` (analysis only). Unit test: `tests/test_drops.py::test_mains_notch_suppresses_lines_and_keeps_burst`.

### 7.3 Re-check of the 2.5 × median threshold (NB §9.11)
- **Method.** (Left) burst episodes per hour against the threshold, before and after the notch; (right)
  for about 240 one-second windows per session (half uniform in time, half where the EMG ratio exceeds
  1.5), the highest ratio against the mean absolute frame-to-frame video change in that window.
- **Result.** The episode count falls smoothly through 2.5 (no plateau), so it neither supports nor
  contradicts the value. Video motion stays at its quiet baseline (about 0.3-0.4) below 2.5 and is clearly
  higher for ratio ≥ 3. In 07-09 the 2.5-3 bin has already risen (0.89); in 07-21 it has not (0.38; 1.18
  for 3-4). **2.5 was kept.** For 07-21 alone, 3 would fit the video slightly better.

---

## 8. Validation on the development session (2026-06-18_16-25-09_000)

The expectations come from inspecting the raw data earlier (NB §12): (1) about 88 s, all three sensors
drop, EMG normal; (2) about 92.5-93 s, the same; (3) about 95.5 s, a sharp EMG burst, so any event near it
must have `emg_burst = True`. Events in the development session, from the CSV:

| Event | Type | Start–end (s) | emg_burst | emg_peak_ratio |
|---|---|---|---|---|
| near 88 s | ALL_THREE (thermistor+airflow+lfp) | 87.571–88.426 | False | 0.787 |
| after it | PAIR (airflow+lfp) | 88.658–89.276 | False | 0.813 |
| near 92.5-93 s | ALL_THREE | 92.020–93.152 | False | 1.569 |

| Check | Result |
|---|---|
| 1. ALL_THREE near 88 s with normal EMG (window 86.5-89.0 s) | **PASS** |
| 2. ALL_THREE near 92.5-93 s with normal EMG (window 91.5-93.5 s) | **PASS** |
| 3. Event(s) within ±1 s of 95.5 s have `emg_burst = True` | **FAIL** |

**Interpretation of the failure (NB §12).** It fails by construction, not because the flag is broken. At
95.5 s the EMG RMS rises to about 14× (9.2× before the notch), but the three breathing amplitudes do not
fall below the threshold: in 94-98 s the lowest ratios are 0.41 (thermistor), 0.63 (airflow), 0.70 (LFP).
The notebook says an existing description attributes this burst to an LFP artifact burst, so there is no
drop to flag. The nearest event (92.0-93.2 s) ends 2.3 s before the burst, outside the ±1 s window;
widening the window enough would also flag that event, which must stay `emg_burst = False`. Sweeping the
amplitude fraction up to 0.55 and the minimum duration down to 0.2 s never created an event at 95.5 s.
The flag does work elsewhere: with looser thresholds it flags short drops next to EMG bursts at 17.8,
113 and 321 s (NB §12), and the unit test `test_emg_burst_flag` checks it on a synthetic burst.

**Open decision (not made):** either (a) accept that 95.5 s is an artifact burst, not a breathing drop,
and change the expectation to "no event there", or (b) add a separate "artifact" detector for amplitude
*rises*. Nothing was changed to force the check to pass.

With the current notch, the only development-session event flagged is a SINGLE drop at 214.814-215.145 s
with `emg_peak_ratio` 2.522, just over the threshold.

---

## 9. Results across all sessions

Same parameters for all sessions (no per-session tuning). Counts and median durations (s) per class and
EMG flag, computed from `results/drop_events/drop_events.csv`:

| Session | Class | EMG burst = False: n (median s) | EMG burst = True: n (median s) |
|---|---|---|---|
| 000 (dev) | ALL_THREE | 5 (1.13) | 0 |
| | PAIR | 12 (0.51) | 0 |
| | SINGLE | 17 (0.38) | 1 (0.33) |
| 001 | ALL_THREE | 59 (0.91) | 12 (1.24) |
| | PAIR | 164 (0.64) | 8 (0.67) |
| | SINGLE | 552 (0.38) | 28 (0.41) |
| 07-09 | ALL_THREE | 215 (1.06) | 30 (1.48) |
| | PAIR | 351 (0.67) | 24 (0.75) |
| | SINGLE | 936 (0.40) | 49 (0.43) |
| 07-21 | ALL_THREE | 261 (1.06) | 83 (1.23) |
| | PAIR | 379 (0.66) | 58 (0.71) |
| | SINGLE | 1234 (0.41) | 174 (0.40) |

Totals: 35 + 823 + 1605 + 2189 = 4652 events; 467 (10.0 %) flagged with an EMG burst. SINGLE events are the
most common and shortest; ALL_THREE are the longest.

**Link between ALL_THREE and EMG (NB §9.12).** The chance level is the share of session time within 1 s of
an EMG burst. PAIR and SINGLE are flagged at about chance in every session (07-21: 13 % and 12 % against
13 %). ALL_THREE events are flagged 2-4× above chance in the three sessions where it can be tested (07-21:
24 % against 13 %; 07-09: 12 % against 6 %; 001: 17 % against 4 %). Most ALL_THREE events (75-100 %) have
no burst nearby. This is compatible with some candidate pauses being movement-related (for example
swallowing or a head movement) and is a reason to inspect the EMG-flagged ALL_THREE events in the viewer
before interpreting them.

---

## 10. Event-to-video mapping

- **Purpose.** Locate each event in the video file so it can be inspected.
- **Procedure.** For event time `t` (recording clock): `frame = searchsorted(cam_t, t)` (the first frame at
  or after t); `video time = frame / fps`, written as `HH:MM:SS.mmm`. Events before the first or after the
  last frame have empty video fields (in the CSV: 31 of 35 in 000, 1598 of 1605 in 07-09, 2182 of 2189 in
  07-21 have a video file; session 001 has none). Columns: `camera`, `frame_start`, `frame_end`,
  `video_file`, `t_start_video`, `t_end_video`, `video_timestamp`, `video_fps`. Left camera is used.
- **Measured versus container fps (deviation from the original task text).** The camera times give
  60.0047 Hz. The MP4 files run on a 60 fps timeline, which is what a player seeks on. Dividing by 60.0047
  would put the position up to about 0.8 s early at the end of a 3-hour session (0.03 s in the 6-minute
  one; `decisions.md` 2026-11-09). So the container rate is used when a video exists
  (`video_fps` in the CSV: 59.9999 for 000, 60.0000 for 07-09 and 07-21) and the measured rate otherwise
  (60.0047 for 001). Frame numbers are identical either way.
- **Frame-count check.** The number of frames in each video (read from the MP4 header, no decoding)
  must equal `len(cam_t)`; a mismatch stops the run. Camera times are not curated, so this is the proof
  that they agree. From `summary.json`:

  | Session | Left video frames | Right video frames | len(cam_t) | Match |
  |---|---|---|---|---|
  | 000 | 22,217 | 22,217 | 22,217 | yes |
  | 07-09 | 625,257 | 625,257 | 625,257 | yes |
  | 07-21 | 659,515 | 659,515 | 659,515 | yes |
  | 001 | no video (436,897 camera frames) | | | n/a |

- **File matching.** Videos were matched to sessions by frame count; the file numbering is crossed (07-09:
  `leftCamera.raw-002.mp4` + `rightCamera.raw-001.mp4`; 07-21: `leftCamera.raw-001.mp4` +
  `rightCamera.raw-002.mp4`; 000: `leftCamera.raw.mp4`/`rightCamera.raw.mp4` in `~/IBL_videos`; see
  `config.VIDEO_FILES`).

**Implemented in:** `video.rec_to_video_time`, `video.container_fps`, `video.mp4_info`,
`video.check_frame_counts`, `video.add_video_columns`; run by `scripts/run_drop_events.py`.

---

## 11. Audio-visual (video-to-signal) sync check

The task is to check that video and EMG lie on the same time axis (no offset or drift).

- **Method (NB §15).** The viewer page was driven in Chrome; the browser decoded the left-camera video of
  the development session. For every presented frame the mean absolute pixel change from the previous one
  (160×128 grey, whole frame) was the *motion energy*. At 4× speed the browser presented every 8th frame,
  giving 2,322 values about 0.13 s apart. EMG RMS was averaged over the same time spans (found through
  `cam_t`), then shifted by −1 s to +1 s to find where the two agree best.
- **Result.** Correlation r = 0.56 with no shift; the maximum is at 0 to +0.2 s (movement following muscle
  activity, the physiological order); it falls to 0.39 at −1 s and 0.49 at +0.8 s. So no offset or drift
  larger than about 0.1-0.2 s (the resolution of the test). The single 95.5 s burst was too noisy to judge
  alone.
- **Limits.** Whole-frame motion is a coarse proxy; only the 6-minute session was tested, so the long
  sessions rest on the container-fps rule (section 10) checked arithmetically, not against a late-session
  burst. The Chrome driver script is not in the repository, so this exact measurement cannot be
  re-run from the code here (the later ffmpeg-based measurements in 7.2-7.3 can). The proper final check is
  by eye in the viewer.

---

## 12. Export tools

### 12.1 MP4 clips (`scripts/export_clips.py` → `ibl.clips.export_session_clips`)
- **Shows.** One MP4 per event: the left-camera frame on top (grey) and four traces below with a moving
  cursor: LFP ch0 (1-15 Hz), thermistor (1-15 Hz), airflow (1-15 Hz) and EMG (30-300 Hz, notched) with its
  RMS in red and a dotted line at 2.5 × session median. The event is shaded (red ALL_THREE, orange other;
  purple/grey if EMG burst). A text overlay gives type, sensors and time.
- **Window.** ±5 s around the event centre (`--half`, default 5.0).
- **Frame rate.** Every 2nd video frame (60 / 30), written at 30 fps, i.e. real time. Frames come from
  ffmpeg with accurate seeking at `frame / container fps`; image size is 640 × 672.
- **Options.** `--session` (required), `--type` (ALL_THREE / PAIR / SINGLE), `--emg 0|1`, `--limit N`
  (default 5), `--half`.
- **Output.** `results/clips/<event_id>.mp4` (gitignored). Needs ffmpeg. Only events with video coverage.
- **Filters.** Signals use the same band-passes and notch as the detector (`viewer._session_arrays`).

### 12.2 Viewer chunks (`scripts/export_viewer_chunks.py` → `ibl.viewer.export_event_chunks`) and viewer
- **What is precomputed and why.** The browser cannot load hours of 1000 Hz data, so only a window around
  each event is exported (never whole sessions). One JSON per event under
  `results/viewer_chunks/<session>/<event_id>.json`, plus `index.json` (`{"half": 8.0, "events": [...]}`;
  one entry per event: id, session, type, sensors, emg flag, peak, start, end, chunk path, `video`
  true/false, fps). The current index has 4652 events, 3811 with video; the folder is 465 MB.
- **Window.** ±8 s around the event centre (viewer shows ±5 s by default, adjustable up to 8 s).
- **Downsampling.** LFP, thermistor and airflow (1-15 Hz filtered) are decimated 1000 → 250 Hz (every 4th
  sample; the band-limit makes this safe). EMG (30-300 Hz) is also 250 Hz but **peak-preserving**: from
  each block of 4 samples the one with the largest absolute value is kept, so short spikes survive. The
  RMS envelope is decimated 1000 → 50 Hz (it is smooth over 0.25 s).
- **Quantization.** Each trace is scaled to the int16 range (`max|x| / 32000`) and stored as integers plus
  a scale factor, to keep the JSON small.
- **Camera times per chunk.** Each chunk contains `cam_frame0` and the `cam_t` slice for its window, so the
  browser maps recording time to video time through the real frame times, no fixed offset.
- **Video access.** `viewer/video/` holds symlinks to the read-only videos (`viewer.link_videos`), so a
  static server can reach them.
- **Why a Range-capable server.** Browsers must seek inside multi-GB MP4s with HTTP Range requests;
  `python -m http.server` ignores them. `viewer/serve.py` adds Range support (partial content, 206).
- **Viewer (`viewer/index.html`).** Filters (session, type, EMG, has video); a list of events; video plus
  four traces on a canvas with a cursor tied to the video time through `cam_t`; play/pause, ±events,
  speed 0.25/0.5/1×, window ±1-8 s, left/right camera, click to seek. Sessions or events without video
  open in signals-only mode.

---

## 13. Known issues and open questions

1. **Session 2026-06-18_16-25-09_001 has no video** among the supplied files (nothing with 436,897 frames).
   Its events have frame numbers but no `video_file`; the viewer shows them signals-only.
2. **95.5 s validation check fails** (section 8); a decision between (a) and (b) is needed.
3. **EMG mains hum.** EMG ch0 contains strong 50 Hz hum and harmonics. In 2026-07-21 it makes up essentially
   all of the 30-300 Hz power (about 10^5 × the muscle floor) and, without a filter, the `emg_burst` flag
   cannot fire there. **Status: fixed in the code** (notch 50-350 Hz, Q = 30, section 7.2); the grounding or
   shielding cause on 07-21 is unknown from the data and worth raising with the lab. Residual hum
   remains in some minutes (median minute under 1 %, a few minutes 5-11 %, mostly with movement).
4. **Weak bursts in 07-21** (ratio near 2.5) can show slow rhythmic fluctuations (period about 3.5 s) that
   may be residual hum rather than muscle (NB §9.10); treat them with care.
5. **ALL_THREE and movement** (section 9): possible contamination of candidate pauses by movement.
6. **Absolute EMG RMS values** changed with the notch; ratios are comparable.
7. **EMG ch1** is not processed.
8. The sync check (section 11) was done on one 6-minute session only.

---

## 14. How to reproduce

From the repository root (read-only on `data/` and videos; ffmpeg is required for clips and video-motion
measurements):

```
uv run pytest                                                        # 8 unit tests
uv run python scripts/run_drop_events.py [--sessions S ...]          # results/drop_events/{drop_events.csv, summary.json}
uv run python scripts/export_viewer_chunks.py [--sessions S ...]     # results/viewer_chunks/ + viewer/video/ symlinks
uv run python scripts/export_clips.py --session 2026-06-18_16-25-09_000 --limit 5 [--type ALL_THREE] [--emg 0|1]
uv run python viewer/serve.py [port]                                 # then open http://localhost:8000/viewer/index.html
uv run marimo edit scripts/breathingPause.py                         # notebook with all justification plots
```

Order: `run_drop_events.py` first (it writes the CSV that the others read), then
`export_viewer_chunks.py` and/or `export_clips.py`, then `serve.py`. `run_drop_events.py` stops if a
video's frame count differs from `len(cam_t)`.

---

## 15. Parameter table

| Parameter | Value | Unit | Set in | Justification |
|---|---|---|---|---|
| Sampling rate `FS` | 1000 | Hz | `config.FS` | Data property |
| Respiration/LFP band | 1-15, 3rd-order Butterworth, zero-phase | Hz | `DropParams.band` | Breathing about 2-10 Hz (NB §2) |
| Envelope smoothing `smooth_s` | 0.25 | s | `DropParams` | Judgement; not varied in NB |
| Median window `median_win_s` | 60 | s | `DropParams` | Long versus a drop, short versus session (NB §3) |
| Median block size | 250 samples (4 Hz) | samples | `drops.rolling_median` | Speed; no comparison recorded |
| Drop threshold `frac` | 0.35 | × median | `DropParams` | NB §4 histogram, NB §7 grid |
| Merge gap `merge_s` | 0.2 | s | `DropParams` | NB §5 gap histogram |
| Minimum duration `min_dur_s` | 0.3 | s | `DropParams` | NB §6, §7 |
| Cross-sensor overlap `overlap` | 0.5 | fraction of shorter drop | `DropParams` | NB §8 histogram |
| EMG band `emg_band` | 30-300, 3rd-order Butterworth, zero-phase | Hz | `DropParams` | Muscle band (task default); no variation in NB |
| EMG RMS window `emg_win_s` | 0.25 | s | `DropParams` | Task default |
| Notch on/off `notch` | True | - | `DropParams` | NB §9 |
| Notch fundamental `notch_f0` | 50.0 | Hz | `DropParams` | NB §9.3, §9.7 |
| Notch harmonics `notch_harmonics` | 7 (50-350) | count | `DropParams` | NB §9.4 |
| Notch quality `notch_q` | 30 (bandwidth f/30) | - | `DropParams` | NB §9.6 |
| EMG burst threshold `emg_ratio` | 2.5 | × session median RMS | `DropParams` | NB §10, §9.11 |
| EMG window pad `emg_pad_s` | 1.0 | s | `DropParams` | Task default; not varied |
| Chunk half-width `half` | 8.0 | s | `viewer.export_event_chunks` | Covers viewer window up to ±8 s |
| Chunk decimation `DEC` | 4 (250 Hz) | - | `viewer.DEC` | Size; signals band-limited |
| RMS decimation `RMS_DEC` | 20 (50 Hz) | - | `viewer.RMS_DEC` | RMS smooth over 0.25 s |
| Quantization | int16 range (scale = max/32000) | - | `viewer._quantize` | JSON size |
| Clip half-width | 5.0 | s | `clips.export_session_clips`, `--half` | Default |
| Clip frame rate | 30 (every 2nd frame) | fps | `clips.export_session_clips` | Real-time playback |
| Clip size | 640 × 672 | px | `clips.py` (8 × 8.4 in at 80 dpi) | - |
| Video fps for time | container (60.000) or measured (60.0047) | Hz | `video.add_video_columns` | `decisions.md` 2026-11-09 |

---

## 16. Inconsistencies found

1. **Frame to video time in the viewer.** `video.rec_to_video_time` (CSV, clips) uses `frame / fps`;
   `viewer/index.html` (`recToVideo`) uses `(cam_frame0 + lowerBound + 0.5) / fps`, i.e. half a frame
   (about 8 ms at 60 fps) later. Not recorded as a deliberate choice anywhere; harmless for the 0.1-0.2 s
   resolution of the sync test but the two differ.
2. **fps wording.** NB §14 first says the position is `frame / fps` with fps "measured from cam_t (about
   60.005)", then states the deviation; the CSV uses the container rate. `decisions.md` says the MP4 files
   are "exactly 60.000"; the recorded `video_fps` for the development session is 59.9999.
3. **Number of example clips.** NB §16 says two example clips of the development session exist; `results/clips/`
   currently holds five (events 0005, 0007, 0008, 0016, 0018).
4. **Hum-share lines.** The code default (`emgnoise.LINES`) is 50-300 Hz (k = 1-6), and NB §9.1 says k = 1…6;
   `decisions.md` mentions lines for k = 1…9 for the same hum-share figures. The hum-share numbers in
   section 7.2 are quoted as recorded.
5. **Hard-coded threshold in clips.** `clips.py` draws the dotted EMG line at a literal `2.5` rather than
   `DropParams.emg_ratio`; identical today, not if the parameter changes.
6. **Hilbert amplitude.** NB §2 writes `|hilbert(x)|`; the pipeline uses its own FFT-based float32
   implementation (`drops.hilbert_amplitude`). The scipy version (`signal.analytic`) is used only by the
   older helper functions. No numeric comparison of the two is recorded.
7. **Quantities from plots only (not a disagreement, but not recorded).** The percent of time below 0.35, share of gaps below 0.2 s and the
   overlap histogram count (5.1, 5.2, 6) are printed on plots but not recorded in text.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import m1Session as m1
    import matplotlib
    matplotlib.use("Agg")   # figures are rendered by marimo, not a GUI window

    return m1, mo


@app.cell(hide_code=True)
def _(m1, mo):
    sessions = sorted(p.name for p in m1.DATA.iterdir() if (p / "alf").is_dir())
    picker = mo.ui.dropdown(sessions, value=sessions[0], label="Session")
    mo.md(f"""
    # M1 sessions: PipelineStruct sections 3-10
    Pick a session; all plots below are recomputed for it (LFP channel 0). The clean / messy windows are the
    best / worst 5-s windows of the weakest sensor pair; the pause window is the deepest joint amplitude drop.

    {picker}
    """)
    return (picker,)


@app.cell
def _(m1, mo, picker):
    figs = {}
    with mo.status.spinner(title=f"Running {picker.value}"):
        res = m1.run_session((picker.value, 0, figs))
    return figs, res


@app.cell(hide_code=True)
def _(m1, mo, res):
    rows = [{"pair": k, "median PLV": round(res["plv"][k]["median"], 2),
             "windows > 0.7": f'{res["plv"][k]["frac_above_0.7"]:.0%}',
             "lag (rad)": round(res["lag"][k]["rad"], 2)} for k in res["plv"]]
    mo.md(f"""
    **{res["session"]}** — {res["duration_s"]:.0f} s, video {res["video_s"][0]:.0f}-{res["video_s"][1]:.0f} s.
    Clean window starts {res["clean_start_s"]:.0f} s, messy {res["messy_start_s"]:.0f} s.
    Possible pause near {res["pause_center_s"]:.1f} s. Best LFP channel (summed PLV): {res["lfp_channels"]["best_channel"]}.
    Welch peaks (Hz): {", ".join(f"{k} {v:.1f}" for k, v in res["welch_peak_hz"].items())}.

    {mo.ui.table(rows, selection=None)}
    """)
    return


@app.cell(hide_code=True)
def _(figs, mo):
    show = lambda *names: mo.vstack([figs[n] for n in names])
    mo.ui.tabs({
        "3 Whole session": show("3_whole_session.png"),
        "4 Windows": show("4_clean_window.png", "4_messy_window.png"),
        "5 Spectra": show("5_welch.png", "5_coherence.png"),
        "6 Phase": show("6_phase_clean.png", "6_phase_messy.png"),
        "7 Agreement": show("7_plv_emg.png"),
        "8 Pause": show("8_pause.png"),
        "9 LFP channel": show("9_lfp_channels.png", "9_lfp_corr.png"),
        "10 Filter band": show("10_filter_band.png"),
    })
    return


if __name__ == "__main__":
    app.run()

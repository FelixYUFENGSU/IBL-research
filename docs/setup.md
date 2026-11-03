# Setup

## One-time setup (Terminal)

```
cd ~/Documents/IBL
uv sync
```

This creates `.venv` with everything the notebooks and scripts need (numpy, scipy, matplotlib, pandas, marimo), per `pyproject.toml`.

## Running a notebook in VS Code

1. Open VS Code -> File -> Open Folder -> `IBL`.
2. Open a notebook under `scripts/` (e.g. `scripts/PipelineStruct.ipynb`).
3. Top right, click **Select Kernel** -> **Python Environments** -> `.venv`.
4. Press Shift+Enter to run each cell.

## Folder layout

```
IBL/
  pyproject.toml, uv.lock   Python environment (uv)
  src/ibl/                  reusable Python modules
  scripts/                  notebooks and standalone scripts
  data/                     raw session data (not synced to source control)
    M1/<session>/alf/       processed, analysis-ready signals (.npy)
    M1/<session>/metadata/  session_info.json per session
  results/figures/          generated plots (not synced to source control)
  docs/                     setup notes and research notes
  tests/                    tests for src/ibl
```

To add more sessions, put each session's `alf/` and `metadata/` folders under `data/M1/<session name>/`.

## Legacy conda env

Earlier versions of this project used `conda` (see git history for `environment.yml`). The project now uses `uv` + `pyproject.toml` as the single source of truth for dependencies.

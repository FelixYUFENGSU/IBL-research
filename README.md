# Mouse breathing: sensor fusion

## One-time setup (Terminal)

```
cd ~/Documents/mouse-breathing
conda env create -f environment.yml
```

This creates a Python environment called `mouse-breathing` with everything the notebooks need.
(If `conda` isn't found, install Miniconda from https://docs.conda.io/en/latest/miniconda.html, then reopen Terminal.)

## Running a notebook in VS Code

1. Open VS Code -> File -> Open Folder -> `mouse-breathing`.
2. Open `01_first_look.ipynb`.
3. Top right, click **Select Kernel** -> **Python Environments** -> `mouse-breathing`.
4. Press Shift+Enter to run each cell.

## Folder layout

```
mouse-breathing/
  environment.yml        Python packages
  01_first_look.ipynb    load, plot, phase, sensor agreement
  data/
    2026-06-18_16-25-09_000/alf/   one session's .npy files
```

To add more sessions, put each session's `alf/` folder under `data/<session name>/` and change `DATA` at the top of the notebook.

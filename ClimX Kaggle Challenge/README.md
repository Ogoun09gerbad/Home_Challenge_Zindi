# ClimX: forced-analogue climate emulator

Solution to the Kaggle competition [ClimX: Extreme-aware climate model emulation](https://www.kaggle.com/competitions/climx-main-track) (Sep–Dec 2026).

The task is to emulate the NorESM2-MM climate model under the SSP2-4.5 scenario from forcings only (greenhouse gases and aerosols), and to predict 15 annual climate-extreme indices (TXx, TNn, SU, TR, FD, ID, WSDI, CSDI, GSL, Rx5day, CDD, CWD, R95p, R10mm, SDII) on the 192x288 grid for 2015–2100. The metric is a region-weighted normalised NSE (`rnNSE`).

## Approach

The emulator produces daily fields from forcings only, then computes the 15 indices with the official code (`repo_ClimX/src/indices_xclim.py`) and averages them over realisations.

1. CO2, CH4, N2O and SO2 give a forced global temperature (energy-balance regression with two response time scales).
2. Per cell, month and variable, a regression gives the forced mean as a function of that state.
3. A realisation of a target year copies a training year with a close forced state (48 candidates), shifted by the difference between the two forced means. Daily weather, spells, tails and consistency across cells and variables come from the copied year.

Averaging indices over realisations estimates their expected value given the forcing, which is what minimises the squared error in the metric. Indices are never computed on averaged fields.

## Layout

| Path | Content |
|---|---|
| `climx_emulator/model.py` | Forced-analogue emulator |
| `climx_emulator/common.py` | Data access and wrappers around the official index and scoring code |
| `climx_emulator/validate.py` | Leave-one-scenario-out validation on the lite grid |
| `climx_emulator/make_submission.py` | Trains on all scenarios, emulates SSP2-4.5, writes and checks the submission |
| `climx_emulator/native.py`, `run_native.py` | Full-resolution (192x288) pipeline, tile by tile, resumable |
| `climx_emulator/test_native_grid.py` | Tests of the native-grid paths on synthetic data |
| `climx_emulator/build_kaggle_notebook.py` | Builds `kaggle/climx_native.ipynb`, a self-contained notebook embedding the code |
| `outputs/` | Validation scores (`validation_*.json`) and logs of the ablations |

## Results

Leave-one-scenario-out validation on the lite grid, 48 realisations and 48 analogues (`outputs/validation_*_k48.json`):

| Held-out scenario | Overall rnNSE (lite) |
|---|---|
| SSP3-7.0 | 0.234 |
| SSP1-2.6 | -0.007 |

Lite scores are computed on 16x16 cell blocks: they compare variants with each other but do not estimate the Kaggle score, which is computed against the full-resolution truth. The lite submission upscaled by nearest neighbour scored -0.44571 on the public leaderboard; the native-resolution pipeline (`run_native.py`, `kaggle/climx_native.ipynb`) targets that gap.

Ablations (`k6`/`k24`/`k48`/`k96` analogues, `noaero`, `noquad`, `noshift`) are in `outputs/`.

## Reproduce

Python 3.11, from this folder:

    git clone https://github.com/IPL-UV/ClimX.git repo_ClimX
    python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
    # Lite data in data/lite (set HF_HUB_DISABLE_XET=1 if the Xet transfer fails)
    .venv/Scripts/python -c "from huggingface_hub import snapshot_download as s; s('isp-uv-es/ClimX', repo_type='dataset', local_dir='data', allow_patterns=['lite/*'])"
    .venv/Scripts/python climx_emulator/validate.py --holdout ssp370 --realizations 48 --n-analogues 48
    .venv/Scripts/python climx_emulator/make_submission.py

The submission is written to `outputs/submission_lite_forced_analogue.parquet`.

Full resolution:

    python climx_emulator/run_native.py fetch
    python climx_emulator/run_native.py validate --holdout ssp370 --realizations 12
    python climx_emulator/run_native.py submit --realizations 48

Paths can be overridden with the `CLIMX_REPO`, `CLIMX_LITE` and `CLIMX_OUT` environment variables.

The competition data, the official ClimX repository (`repo_ClimX/`), generated daily fields (`*.nc`) and submission files (`*.parquet`) are not included.

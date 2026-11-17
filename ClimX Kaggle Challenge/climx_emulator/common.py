"""Data access and official-code wrappers shared by validation and submission scripts."""
from __future__ import annotations

import glob
import os
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import cftime
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("CLIMX_REPO", ROOT / "repo_ClimX"))
DATA = Path(os.environ.get("CLIMX_LITE", ROOT / "data" / "lite"))
OUT = Path(os.environ.get("CLIMX_OUT", ROOT / "outputs"))
sys.path.insert(0, str(REPO))

import rnNSE  # noqa: E402  (official scorer)
from src import indices_xclim as I  # noqa: E402  (official index code)
from src.utils import kaggle_submission as ks  # noqa: E402
from src.utils.indices_utils import format_indices  # noqa: E402
from src.utils.original_data_values import LAT, LON  # noqa: E402

TARGETS = ["huss", "pr", "psl", "sfcWind", "tas", "tasmax", "tasmin"]
TARGET_UNITS = {"huss": "1", "pr": "kg m-2 s-1", "psl": "Pa", "sfcWind": "m s-1", "tas": "K", "tasmax": "K", "tasmin": "K"}
GHG = ["CO2_LBC", "CH4_LBC", "N2O_LBC", "CFC11eq_LBC", "CF2CL2_LBC"]
AEROSOLS = ["BC_AX", "BC_N", "SO2", "SO4_PR", "OM_NI"]
TRAIN_SCENARIOS = ["ssp126", "ssp370", "ssp585"]
DAYS = 365


@dataclass
class Run:
    """One continuous simulation: daily targets by year plus annual forcing series."""

    name: str
    years: np.ndarray
    fields: dict[str, np.ndarray] | None  # var -> (n_years, 365, lat, lon)
    ghg: dict[str, np.ndarray]  # var -> (n_years,)
    aerosols: dict[str, np.ndarray]  # var -> (n_years,) global area-weighted mean


def _files(pattern: str) -> list[str]:
    found = sorted(glob.glob(str(DATA / pattern)))
    if not found:
        raise FileNotFoundError(f"No file matching {pattern} under {DATA}")
    return found


def _squeeze_scenario(ds: xr.Dataset) -> xr.Dataset:
    if "forcing_scenario" in ds.dims:
        ds = ds.isel(forcing_scenario=0, drop=True)
    return ds.drop_vars("forcing_scenario", errors="ignore")


def open_hist() -> xr.Dataset:
    """Historical run, loaded in memory, with CF units kept for the official index code."""
    parts = [_squeeze_scenario(xr.open_dataset(f)) for f in _files("*hist_train_*.nc")]
    daily = xr.concat([p[TARGETS] for p in parts], dim="time").load()
    # Each file already holds the full 1850-2014 annual and monthly forcing series.
    forcings = parts[0][GHG + AEROSOLS].load()
    for p in parts:
        p.close()
    return xr.merge([daily, forcings])


def open_ssp(scenario: str) -> xr.Dataset:
    pattern = f"*test_forcings_{scenario}.nc" if scenario == "ssp245" else f"*train_{scenario}.nc"
    with xr.open_dataset(_files(pattern)[0]) as ds:
        return _squeeze_scenario(ds).load()


def _area_weights(lat: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(lat))
    return w / w.sum()


def _to_year_blocks(da: xr.DataArray, years: np.ndarray) -> np.ndarray:
    """Reshape a noleap daily array to (year, 365, lat, lon). A truncated final day is repeated."""
    values = da.transpose("time", "lat", "lon").values.astype(np.float32)
    stamp = da["time"].dt.year.values
    out = np.empty((len(years), DAYS) + values.shape[1:], dtype=np.float32)
    for i, year in enumerate(years):
        block = values[stamp == year]
        if len(block) < DAYS:
            pad = np.repeat(block[-1:], DAYS - len(block), axis=0)
            block = np.concatenate([block, pad])
        out[i] = block
    return out


def to_run(name: str, ds: xr.Dataset) -> Run:
    years = np.unique(ds["time_year"].dt.year.values)
    w = _area_weights(ds["lat"].values)
    ghg = {v: ds[v].values.astype(np.float64) for v in GHG}
    aerosols = {}
    for v in AEROSOLS:
        # Global, annual mean emission flux: the forcing's spatial pattern is not used yet.
        spatial = (ds[v].values * w[None, :, None]).sum(axis=1).mean(axis=1)
        aerosols[v] = pd.Series(spatial).groupby(ds["time_month"].dt.year.values).mean().values
    fields = None
    if "tas" in ds:
        fields = {v: _to_year_blocks(ds[v], years) for v in TARGETS}
    return Run(name, years, fields, ghg, aerosols)


@lru_cache(maxsize=4)
def time_axis(first_year: int, last_year: int) -> xr.CFTimeIndex:
    """Noleap daily axis at 12:00, ending on Dec 30 of the last year like the dataset."""
    stamps = [
        cftime.DatetimeNoLeap(y, 1, 1, 12) + pd.Timedelta(days=d)
        for y in range(first_year, last_year + 1)
        for d in range(DAYS)
    ]
    return xr.CFTimeIndex(stamps[:-1])


def blocks_to_dataset(fields: dict[str, np.ndarray], first_year: int, lat, lon, units: dict) -> xr.Dataset:
    """Wrap (year, 365, lat, lon) arrays as a daily dataset with the dataset's calendar and units."""
    n_years = next(iter(fields.values())).shape[0]
    time = time_axis(first_year, first_year + n_years - 1)
    data = {}
    for var, arr in fields.items():
        flat = arr.reshape((-1,) + arr.shape[2:])[: len(time)]
        data[var] = xr.DataArray(flat, dims=("time", "lat", "lon"), attrs={"units": units[var]})
    return xr.Dataset(data, coords={"time": time, "lat": lat, "lon": lon})


def compute_indices(daily: xr.Dataset, hist: xr.Dataset) -> xr.Dataset:
    """Compute the 15 indices with the official functions, then the official annualisation."""
    needed = daily[[v for v in ("tas", "tasmax", "tasmin", "pr") if v in daily]]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, func in I.SELECTED_INDICES.items():
            kwargs = {"historical_data_path": hist} if "historical_data_path" in func.__code__.co_varnames else {}
            out[name] = func(needed, **kwargs).compute()
        out = format_indices(out, "emulator", "lite", save_indices=False)
    return xr.Dataset({k: v.drop_vars("quantile", errors="ignore") for k, v in out.items()})


def upscale_nearest(indices: xr.Dataset) -> xr.Dataset:
    """Nearest-neighbour upscaling to the competition grid, as in the starter notebook."""
    lat = np.asarray(LAT, dtype=float)
    lon = np.asarray(LON, dtype=float)
    return indices.sortby("lat").sortby("lon").interp(
        lat=lat, lon=lon, method="nearest", kwargs={"fill_value": "extrapolate"}
    )


def kaggle_frame(indices: xr.Dataset) -> pd.DataFrame:
    """Build the row_id + 15 columns frame through the official conversion."""
    return ks.indices_xr_to_kaggle_dataframe(indices, index_columns=ks.CLIMX_INDEX_COLUMNS)


def official_scores(truth: xr.Dataset, pred: xr.Dataset) -> dict[str, float]:
    """Official scorer overall and per index, on two lite-grid index sets upscaled the same way.

    Truth is the lite-grid truth, so this ranks models but does not estimate the Kaggle score,
    which compares against native-resolution truth.
    """
    sol = kaggle_frame(upscale_nearest(truth))
    sub = kaggle_frame(upscale_nearest(pred))
    scores = {"overall": rnNSE.score(sol.copy(), sub.copy(), "row_id")}
    for col in ks.CLIMX_INDEX_COLUMNS:
        scores[col] = rnNSE.score(sol[["row_id", col]].copy(), sub[["row_id", col]].copy(), "row_id")
    return scores

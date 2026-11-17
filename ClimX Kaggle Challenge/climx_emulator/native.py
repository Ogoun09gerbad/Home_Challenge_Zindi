"""Tile store and emulation of land cells at full resolution.

The native files hold one gzip chunk per global day, so everything scored is extracted once into
per-tile arrays: land cells only, and the four variables the official indices read. The forced
state and the analogue choice come from the lite data and are identical for all tiles, so tiles
of a realisation stay coherent with each other.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cftime
import numpy as np
import xarray as xr

from common import DAYS, LAT, LON, TARGET_UNITS, compute_indices, rnNSE, time_axis
from model import MONTH_CENTERS, MONTH_OF_DAY, MULTIPLICATIVE, PR_FLOOR

INDEX_VARS = ("tas", "tasmax", "tasmin", "pr")
TEMP_OFFSET, TEMP_SCALE = 150.0, 0.005  # K: uint16 covers 150-477 K at 0.005 K, half the disk of float32
TILE_CELLS = 1500
BASELINE = (1961, 1990)
MAX_LOG_SHIFT = np.log(2.0)  # guards arid cells, whose monthly means are mostly noise


@dataclass(frozen=True)
class Grid:
    name: str
    lat: np.ndarray
    lon: np.ndarray
    cells: np.ndarray  # flat indices (lat, lon order) of the cells the emulator handles

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.lat), len(self.lon)

    def tiles(self, tile_cells: int = TILE_CELLS) -> list[np.ndarray]:
        return np.array_split(np.arange(len(self.cells)), int(np.ceil(len(self.cells) / tile_cells)))


def native_grid() -> Grid:
    """Competition grid, restricted to the land cells of the official scorer's region mask."""
    mask, _ = rnNSE._decode_mask()
    return Grid("native", np.asarray(LAT, float), np.asarray(LON, float), np.flatnonzero(mask.ravel() >= 0))


def lite_grid(lat: np.ndarray, lon: np.ndarray) -> Grid:
    """All lite cells: runs the same code on the small dataset, to test the pipeline."""
    return Grid("lite", np.asarray(lat, float), np.asarray(lon, float), np.arange(len(lat) * len(lon)))


def _encode(var: str, values: np.ndarray) -> np.ndarray:
    if var == "pr":
        return values.astype(np.float32)
    return np.rint((values - TEMP_OFFSET) / TEMP_SCALE).astype(np.uint16)


def _decode(var: str, stored: np.ndarray) -> np.ndarray:
    if var == "pr":
        return np.asarray(stored, dtype=np.float32)
    return TEMP_OFFSET + TEMP_SCALE * np.asarray(stored, dtype=np.float32)


class TileStore:
    """(year, day, cell) arrays on disk, one file per run, variable and tile."""

    def __init__(self, root: Path, grid: Grid, tile_cells: int = TILE_CELLS):
        self.root = Path(root)
        self.grid = grid
        self.tile_ids = grid.tiles(tile_cells)

    def _path(self, run: str, var: str, tile: int) -> Path:
        return self.root / run / f"{var}_{tile:03d}.bin"

    def _meta_path(self, run: str) -> Path:
        return self.root / run / "meta.json"

    def years(self, run: str) -> np.ndarray | None:
        path = self._meta_path(run)
        if not path.exists():
            return None
        meta = json.loads(path.read_text())
        return np.arange(meta["first_year"], meta["first_year"] + meta["done_years"])

    def _open(self, run: str, var: str, tile: int, n_years: int, mode: str) -> np.memmap:
        shape = (n_years, DAYS, len(self.tile_ids[tile]))
        dtype = np.float32 if var == "pr" else np.uint16
        return np.memmap(self._path(run, var, tile), dtype=dtype, mode=mode, shape=shape)

    def extract(self, run: str, files: list[Path], first_year: int, last_year: int) -> None:
        """Copy land cells of the given NetCDF files, year by year. Resumes from the last complete year."""
        n_years = last_year - first_year + 1
        (self.root / run).mkdir(parents=True, exist_ok=True)
        meta_path = self._meta_path(run)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if meta is None:
            meta = {"first_year": first_year, "n_years": n_years, "done_years": 0}
            for var in INDEX_VARS:
                for tile in range(len(self.tile_ids)):
                    self._open(run, var, tile, n_years, "w+").flush()
            meta_path.write_text(json.dumps(meta))
        for file in files:
            with xr.open_dataset(file, chunks={}) as ds:
                ds = ds.isel(forcing_scenario=0, drop=True) if "forcing_scenario" in ds.dims else ds
                assert np.allclose(ds["lat"].values, self.grid.lat, atol=1e-3), "Unexpected latitude order"
                assert np.allclose(ds["lon"].values, self.grid.lon, atol=1e-3), "Unexpected longitude order"
                stamps = ds["time"].dt.year.values
                for year in np.unique(stamps):
                    y = int(year) - first_year
                    if y < meta["done_years"] or y >= n_years:
                        continue
                    picked = np.flatnonzero(stamps == year)
                    for var in INDEX_VARS:
                        block = ds[var].isel(time=slice(picked[0], picked[-1] + 1)).transpose("time", "lat", "lon").values
                        flat = block.reshape(block.shape[0], -1)[:, self.grid.cells]
                        if len(flat) < DAYS:  # the final year stops on Dec 30
                            flat = np.concatenate([flat, np.repeat(flat[-1:], DAYS - len(flat), axis=0)])
                        coded = _encode(var, flat)
                        for tile, ids in enumerate(self.tile_ids):
                            out = self._open(run, var, tile, n_years, "r+")
                            out[y] = coded[:, ids]
                            out.flush()
                    meta["done_years"] = y + 1
                    meta_path.write_text(json.dumps(meta))
                    print(f"{run} {year} extracted", flush=True)

    def read_year(self, run: str, tile: int, year_index: int, variables=INDEX_VARS) -> dict[str, np.ndarray]:
        n_years = json.loads(self._meta_path(run).read_text())["n_years"]
        return {v: _decode(v, self._open(run, v, tile, n_years, "r")[year_index]) for v in variables}

    def monthly(self, run: str, tile: int) -> dict[str, np.ndarray]:
        """Monthly means per year in regression space, cached next to the tile files."""
        cache = self.root / run / f"monthly_{tile:03d}.npz"
        if cache.exists():
            with np.load(cache) as data:
                return {v: data[v] for v in INDEX_VARS}
        n_years = len(self.years(run))
        out = {}
        for var in INDEX_VARS:
            months = np.empty((n_years, 12, len(self.tile_ids[tile])), dtype=np.float32)
            for y in range(n_years):
                daily = self.read_year(run, tile, y, (var,))[var]
                months[y] = np.stack([daily[MONTH_OF_DAY == m].mean(axis=0) for m in range(12)])
            out[var] = np.log(np.maximum(months, PR_FLOOR)) if var in MULTIPLICATIVE else months
        np.savez(cache, **out)
        return out


@lru_cache(maxsize=1)
def daily_weights() -> np.ndarray:
    """(365, 12) periodic linear interpolation weights from monthly to daily values."""
    centers = np.concatenate([[MONTH_CENTERS[-1] - DAYS], MONTH_CENTERS, [MONTH_CENTERS[0] + DAYS]])
    weights = np.zeros((DAYS, 12))
    for d in range(DAYS):
        pos = np.clip(np.searchsorted(centers, d + 0.5) - 1, 0, len(centers) - 2)
        frac = (d + 0.5 - centers[pos]) / (centers[pos + 1] - centers[pos])
        weights[d, (pos - 1) % 12] += 1 - frac
        weights[d, pos % 12] += frac
    return weights


def tile_dataset(fields: dict[str, np.ndarray], first_year: int, n_years: int) -> xr.Dataset:
    """Wrap (year, 365, cell) arrays with the dataset's calendar, including the missing last day."""
    time = time_axis(first_year, first_year + n_years - 1)
    data = {
        v: xr.DataArray(a.reshape((-1, a.shape[2]))[: len(time)], dims=("time", "cell"), attrs={"units": TARGET_UNITS[v]})
        for v, a in fields.items()
    }
    return xr.Dataset(data, coords={"time": time})


def baseline_dataset(store: TileStore, tile: int) -> xr.Dataset:
    """1961-1990 of the historical run: the base period of the official percentile thresholds."""
    years = store.years("hist")
    picked = np.arange(BASELINE[0], BASELINE[1] + 1) - years[0]
    assert picked[0] >= 0 and picked[-1] < len(years), "The baseline years have not been extracted"
    blocks = [store.read_year("hist", tile, int(y), ("tasmax", "tasmin", "pr")) for y in picked]
    fields = {v: np.stack([b[v] for b in blocks]) for v in blocks[0]}
    n = fields["pr"].shape[0] * DAYS
    time = xr.cftime_range(cftime.DatetimeNoLeap(BASELINE[0], 1, 1, 12), periods=n, freq="D", calendar="noleap")
    return xr.Dataset(
        {v: xr.DataArray(a.reshape((n, -1)), dims=("time", "cell"), attrs={"units": TARGET_UNITS[v]}) for v, a in fields.items()},
        coords={"time": time},
    )


class TileEmulator:
    """Forced-analogue emulation of one tile, from a plan saved by run_native.make_plan."""

    def __init__(self, store: TileStore, plan: dict, tile: int):
        self.store = store
        self.plan = plan
        self.tile = tile
        monthly = {}
        runs = list(dict.fromkeys(plan["entry_run"].tolist()))
        by_run = {run: store.monthly(run, tile) for run in runs}
        first = {run: int(store.years(run)[0]) for run in runs}
        for var in INDEX_VARS:
            monthly[var] = np.stack(
                [by_run[r][var][int(y) - first[r]] for r, y in zip(plan["entry_run"], plan["entry_year"])]
            )
        pinv = np.linalg.pinv(plan["design_entries"])
        self.coef = {v: np.tensordot(pinv, monthly[v], axes=(1, 0)) for v in INDEX_VARS}
        self.first = first

    def realization(self, r: int) -> dict[str, np.ndarray]:
        plan = self.plan
        weights = daily_weights()
        n_years = len(plan["target_years"])
        out = {
            v: np.empty((n_years, DAYS, len(self.store.tile_ids[self.tile])), dtype=np.float32) for v in INDEX_VARS
        }
        for y in range(n_years):
            candidates = plan["candidates"][y]
            order = np.random.default_rng([int(plan["seed"]), y]).permutation(len(candidates))
            entry = int(candidates[order[r % len(candidates)]])
            run = str(plan["entry_run"][entry])
            year = int(plan["entry_year"][entry]) - self.first[run]
            source = self.store.read_year(run, self.tile, year)
            gap = plan["design_target"][y] - plan["design_entries"][entry]
            for var in INDEX_VARS:
                daily = source[var]
                if plan["apply_shift"]:
                    delta = weights @ np.tensordot(gap, self.coef[var], axes=(0, 0))
                    if var in MULTIPLICATIVE:
                        daily = daily * np.exp(np.clip(delta, -MAX_LOG_SHIFT, MAX_LOG_SHIFT))
                    else:
                        daily = daily + delta
                out[var][y] = daily
        out["tasmax"] = np.maximum(out["tasmax"], out["tas"])
        out["tasmin"] = np.minimum(out["tasmin"], out["tas"])
        return out

    def indices(self, r: int, baseline: xr.Dataset) -> xr.Dataset:
        ds = tile_dataset(self.realization(r), int(self.plan["target_years"][0]), len(self.plan["target_years"]))
        return compute_indices(ds, baseline)


def truth_indices(store: TileStore, run: str, tile: int, baseline: xr.Dataset) -> xr.Dataset:
    """Official indices of a held-out scenario, for validation."""
    years = store.years(run)
    blocks = [store.read_year(run, tile, y) for y in range(len(years))]
    fields = {v: np.stack([b[v] for b in blocks]) for v in INDEX_VARS}
    return compute_indices(tile_dataset(fields, int(years[0]), len(years)), baseline)

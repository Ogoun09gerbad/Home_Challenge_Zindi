"""Tests of the native-grid paths the lite run does not reach, on synthetic data (no download).

1. Extraction: a 192x288 NetCDF stopping on Dec 30 comes back with the right values in the right cells.
2. Submission: known tile results go through the real `submit` command, and each land row of the
   parquet must hold the value of its own cell and year. Unscored rows come from the lite emulator.

Usage: CLIMX_WORK=<empty dir> CLIMX_OUT=<empty dir> python test_native_grid.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import xarray as xr

from common import OUT, LAT, LON, TARGET_UNITS, ks, rnNSE
from native import DAYS, INDEX_VARS, TileStore, native_grid

WORK = Path(os.environ["CLIMX_WORK"])
HERE = Path(__file__).resolve().parent
COLUMNS = list(ks.CLIMX_INDEX_COLUMNS)


def test_extract(grid) -> None:
    lat, lon = np.asarray(LAT, float), np.asarray(LON, float)
    # Two years of a run ending on Dec 30, as the real files do. Values encode cell and day.
    times = [cftime.DatetimeNoLeap(y, 1, 1, 12) + pd.Timedelta(days=d) for y in (2099, 2100) for d in range(DAYS)][:-1]
    flat = np.arange(lat.size * lon.size, dtype=np.float32).reshape(lat.size, lon.size)
    day = np.arange(len(times), dtype=np.float32)[:, None, None]
    fields = {
        "tas": 250 + flat / 1000 + day / 100,
        "tasmax": 255 + flat / 1000 + day / 100,
        "tasmin": 245 + flat / 1000 + day / 100,
        "pr": (1 + flat) * 1e-9 * (1 + day),
    }
    ds = xr.Dataset(
        {v: (("forcing_scenario", "time", "lat", "lon"), a[None], {"units": TARGET_UNITS[v]}) for v, a in fields.items()},
        coords={"forcing_scenario": ["ssp370"], "time": xr.CFTimeIndex(times), "lat": lat, "lon": lon},
    )
    source = WORK / "synthetic.nc"
    ds.to_netcdf(source)
    store = TileStore(WORK / "store_test", grid)
    store.extract("ssp370", [source], 2099, 2100)
    assert len(store.years("ssp370")) == 2
    # Interrupted runs: extracting again must change nothing and must not fail.
    store.extract("ssp370", [source], 2099, 2100)

    for tile in (0, len(store.tile_ids) // 2, len(store.tile_ids) - 1):
        cells = grid.cells[store.tile_ids[tile]]
        for y in (0, 1):
            year = store.read_year("ssp370", tile, y)
            for d in (0, 200, DAYS - 1):
                # The missing Dec 31 of the final year repeats Dec 30.
                absolute = min(y * DAYS + d, len(times) - 1)
                for var, base in (("tas", 250), ("tasmax", 255), ("tasmin", 245)):
                    expected = base + cells / 1000 + absolute / 100
                    assert np.abs(year[var][d] - expected).max() < 0.004, (var, tile, y, d)
                expected = (1 + cells) * 1e-9 * (1 + absolute)
                assert np.allclose(year["pr"][d], expected, rtol=1e-5), ("pr", tile, y, d)
    source.unlink()
    print("extraction: OK")


def test_submit(grid) -> None:
    store = TileStore(WORK / "store_native", grid)
    folder = WORK / "runs_native" / "ssp245_k48"
    folder.mkdir(parents=True, exist_ok=True)
    years = np.arange(2015, 2101)
    # `submit` only needs the target years here: every tile already reports its realisations.
    np.savez(folder / "plan.npz", target_years=years)
    realizations = 12
    for tile, ids in enumerate(store.tile_ids):
        cells = grid.cells[ids]
        values = np.empty((len(COLUMNS), len(years), len(ids)))
        for i in range(len(COLUMNS)):
            for t in range(len(years)):
                values[i, t] = i * 1e7 + t * 1e5 + cells
        np.savez(folder / f"sum_{tile:03d}.npz", total=values * realizations, count=realizations)
    env = dict(os.environ)
    subprocess.run(
        [sys.executable, str(HERE / "run_native.py"), "submit", "--realizations", str(realizations), "--workers", "2"],
        check=True, env=env, cwd=WORK,
    )
    path = OUT / f"submission_native_k48_m{realizations}.parquet"
    df = pd.read_parquet(path)
    assert list(df.columns) == ["row_id", *COLUMNS]
    assert len(df) == 86 * 192 * 288 and (df["row_id"].values == np.arange(len(df))).all()
    for i, c in enumerate(COLUMNS):
        values = df[c].values.reshape(86, 192, 288)
        expected = (i * 1e7 + np.arange(86)[:, None] * 1e5 + grid.cells[None, :])
        assert np.abs(values.reshape(86, -1)[:, grid.cells] - expected).max() < 1e-6, f"Land rows of {c}"
        land = np.zeros(192 * 288, dtype=bool)
        land[grid.cells] = True
        ocean = values.reshape(86, -1)[:, ~land]
        assert np.isfinite(ocean).all() and np.abs(ocean).max() < 1e6, f"Unscored rows of {c} look wrong"
    # A perfect prediction must score 1 under the official metric on this same file.
    assert abs(rnNSE.score(df.copy(), df.copy(), "row_id") - 1.0) < 1e-12
    print(f"submission: OK ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    grid = native_grid()
    print(f"{len(grid.cells)} land cells in {len(TileStore(WORK / 'x', grid).tile_ids)} tiles")
    test_extract(grid)
    test_submit(grid)

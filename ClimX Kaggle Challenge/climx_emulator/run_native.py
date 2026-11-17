"""Full-resolution pipeline: extract, emulate tile by tile, score or write the submission.

Every step can resume: extraction by year, emulation by tile and realisation.

    python run_native.py fetch    # download, verify, extract, delete each source file
    python run_native.py validate --holdout ssp370 --realizations 12
    python run_native.py submit --realizations 48
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

from common import (
    DATA, I, OUT, TRAIN_SCENARIOS, TARGET_UNITS, blocks_to_dataset, compute_indices, ks, official_scores, open_hist,
    open_ssp, rnNSE, to_run, upscale_nearest,
)
from model import Config, ForcedAnalogueEmulator
from native import INDEX_VARS, Grid, TileEmulator, TileStore, baseline_dataset, lite_grid, native_grid, truth_indices

WORK = Path(os.environ.get("CLIMX_WORK", OUT.parent / "work"))
REPO_ID = "isp-uv-es/ClimX"
FULL_FILES = {
    # Earlier history is far colder than SSP2-4.5, so it would never be chosen as an analogue.
    "hist": ("NorESM2-MM_r1i1p1f1_with_forcing_hist_train_1932_2014.nc", 1932, 2014),
    "ssp370": ("NorESM2-MM_r1i1p1f1_with_forcing_train_ssp370.nc", 2015, 2100),
    "ssp126": ("NorESM2-MM_r1i1p1f1_with_forcing_train_ssp126.nc", 2015, 2100),
    "ssp585": ("NorESM2-MM_r1i1p1f1_with_forcing_train_ssp585.nc", 2015, 2100),
}
INDEX_COLUMNS = list(ks.CLIMX_INDEX_COLUMNS)


def get_grid(name: str) -> Grid:
    if name == "native":
        return native_grid()
    with xr.open_dataset(next(DATA.glob("*test_forcings_ssp245.nc"))) as ds:
        return lite_grid(ds["lat"].values, ds["lon"].values)


def store_for(grid: Grid) -> TileStore:
    return TileStore(WORK / f"store_{grid.name}", grid)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(args) -> None:
    """Download each file, check it, extract land cells and delete it, keeping peak disk use low."""
    from huggingface_hub import HfApi, hf_hub_download

    grid = get_grid(args.grid)
    store = store_for(grid)
    if args.grid == "lite":
        hist_files = sorted(DATA.glob("*hist_train_*.nc"))
        store.extract("hist", hist_files, 1850, 2014)
        for scenario in TRAIN_SCENARIOS:
            store.extract(scenario, list(DATA.glob(f"*train_{scenario}.nc")), 2015, 2100)
        return

    info = HfApi().dataset_info(REPO_ID, files_metadata=True)
    sizes = {s.rfilename: s for s in info.siblings}
    downloads = WORK / "downloads"
    for run in args.runs:
        name, first, last = FULL_FILES[run]
        years = store.years(run)
        if years is not None and len(years) == last - first + 1:
            print(f"{run}: already extracted", flush=True)
            continue
        remote = sizes[f"full/{name}"]
        free = shutil.disk_usage(WORK).free
        need = remote.size + 16e9
        if free < need:
            assert run != "hist", f"{free / 1e9:.0f} GB free: the historical run is required for the baselines"
            print(f"{run}: SKIPPED, {free / 1e9:.0f} GB free and {need / 1e9:.0f} GB needed", flush=True)
            continue
        started = time.time()
        for attempt in range(50):
            try:
                path = Path(hf_hub_download(REPO_ID, f"full/{name}", repo_type="dataset", local_dir=downloads))
                break
            except Exception as error:  # flaky connections: the next attempt resumes the partial file
                print(f"{run}: attempt {attempt + 1} failed ({error}); retrying", flush=True)
                time.sleep(30)
        else:
            raise RuntimeError(f"Could not download {name}")
        print(f"{run}: downloaded {remote.size / 1e9:.1f} GB in {time.time() - started:.0f}s", flush=True)
        assert _sha256(path) == remote.lfs.sha256, f"Checksum mismatch for {name}"
        store.extract(run, [path], first, last)
        path.unlink()


def make_plan(args, grid: Grid, store: TileStore, holdout: str | None) -> Path:
    """Forced state and analogue choices from the lite data, common to all tiles."""
    hist_ds = open_hist()
    hist = to_run("hist", hist_ds)
    train = {s: to_run(s, open_ssp(s)) for s in TRAIN_SCENARIOS}
    target = train.pop(holdout) if holdout else to_run("ssp245", open_ssp("ssp245"))
    config = Config(n_analogues=args.n_analogues, apply_shift=not args.no_shift)
    model = ForcedAnalogueEmulator(config).fit(hist, list(train.values()), hist_ds["lat"].values)

    names = np.array([model.library[i].name for i in model.lib_runs])
    calendar = np.array([model.library[i].years[y] for i, y in zip(model.lib_runs, model.lib_years)])
    available = np.zeros(len(names), dtype=bool)
    for run in set(names.tolist()):
        years = store.years(run)
        if years is None:
            print(f"Run {run} has not been extracted: not used as a library", flush=True)
            continue
        available |= (names == run) & np.isin(calendar, years)
    assert available.sum() >= config.n_analogues, f"Only {available.sum()} library years available"
    forced, aero = model.forced_state(target)
    chosen = model.choose_analogues(forced, aero, available)
    position = np.cumsum(available) - 1  # library entry -> position among extracted entries
    plan = {
        "entry_run": names[available],
        "entry_year": calendar[available],
        "design_entries": model._design_means(model.lib_forced, model.lib_aero)[available],
        "design_target": model._design_means(forced, aero),
        "candidates": np.array([position[c] for c in chosen]),
        "target_years": target.years,
        "seed": config.seed,
        "apply_shift": config.apply_shift,
    }
    path = _run_dir(args, grid, holdout) / "plan.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **plan)
    print(f"Plan: {available.sum()} library years; forced GMST {forced[0]:.2f} -> {forced[-1]:.2f} K", flush=True)
    return path


def _run_dir(args, grid: Grid, holdout: str | None) -> Path:
    label = f"{holdout or 'ssp245'}_k{args.n_analogues}{args.tag}"
    return WORK / f"runs_{grid.name}" / label


def _as_array(indices: xr.Dataset) -> np.ndarray:
    return np.stack([indices[c].transpose("time", "cell").values for c in INDEX_COLUMNS]).astype(np.float64)


def run_tile(grid_name: str, plan_path: str, tile: int, realizations: int, holdout: str | None) -> tuple[int, float]:
    """Add realisations to the running sum of a tile, saving after each one."""
    grid = get_grid(grid_name)
    store = store_for(grid)
    with np.load(plan_path) as data:
        plan = {k: data[k] for k in data.files}
    folder = Path(plan_path).parent
    state_path = folder / f"sum_{tile:03d}.npz"
    total, done = None, 0
    if state_path.exists():
        with np.load(state_path) as state:
            total, done = state["total"], int(state["count"])
    started = time.time()
    if done >= realizations and (holdout is None or (folder / f"truth_{tile:03d}.npy").exists()):
        return tile, 0.0
    # The official percentile cache is keyed by id(dataset): a later tile could reuse the id.
    I._PERCENTILE_CACHE.clear()
    baseline = baseline_dataset(store, tile)
    if holdout and not (folder / f"truth_{tile:03d}.npy").exists():
        np.save(folder / f"truth_{tile:03d}.npy", _as_array(truth_indices(store, holdout, tile, baseline)))
    emulator = TileEmulator(store, plan, tile)
    for r in range(done, realizations):
        values = _as_array(emulator.indices(r, baseline))
        assert np.isfinite(values).all(), f"Non-finite index in tile {tile}, realisation {r}"
        total = values if total is None else total + values
        # One file renamed into place, so an interruption cannot separate the sum from its count.
        temp = folder / f"sum_{tile:03d}.tmp.npz"
        np.savez(temp, total=total, count=r + 1)
        os.replace(temp, state_path)
    return tile, time.time() - started


def _run_all(args, grid: Grid, plan_path: Path, holdout: str | None) -> None:
    store = store_for(grid)
    tiles = range(len(store.tile_ids)) if args.max_tiles is None else range(args.max_tiles)
    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(run_tile, grid.name, str(plan_path), t, args.realizations, holdout) for t in tiles]
        for n, job in enumerate(as_completed(jobs), 1):
            tile, seconds = job.result()
            print(f"tile {tile}: {seconds:.0f}s ({n}/{len(jobs)} tiles, {time.time() - started:.0f}s total)", flush=True)


def assemble(folder: Path, grid: Grid, store: TileStore, prefix: str, years: np.ndarray, fill: xr.Dataset | None, n: int) -> xr.Dataset:
    """Tile files to a dataset of index maps on the grid; cells outside the tiles come from fill."""
    nlat, nlon = grid.shape
    maps = {c: np.zeros((len(years), nlat * nlon)) for c in INDEX_COLUMNS}
    if fill is not None:
        for c in INDEX_COLUMNS:
            # float64: land results are copied into this array and some indices are stored as float32.
            maps[c] = fill[c].transpose("time", "lat", "lon").values.reshape(len(years), -1).astype(np.float64)
    for tile, ids in enumerate(store.tile_ids):
        if prefix == "sum":
            path = folder / f"sum_{tile:03d}.npz"
            if not path.exists():
                continue
            with np.load(path) as state:
                assert int(state["count"]) == n, f"Tile {tile} has {int(state['count'])} realisations, expected {n}"
                values = state["total"] / n
        else:
            path = folder / f"{prefix}_{tile:03d}.npy"
            if not path.exists():
                continue
            values = np.load(path)
        for i, c in enumerate(INDEX_COLUMNS):
            maps[c][:, grid.cells[ids]] = values[i]
    time_values = np.array([np.datetime64(f"{int(y)}-01-01", "ns") for y in years])
    return xr.Dataset(
        {c: (("time", "lat", "lon"), m.reshape(len(years), nlat, nlon)) for c, m in maps.items()},
        coords={"time": time_values, "lat": grid.lat, "lon": grid.lon},
    )


def validate(args) -> None:
    grid = get_grid(args.grid)
    store = store_for(grid)
    plan_path = _run_dir(args, grid, args.holdout) / "plan.npz"
    if not plan_path.exists():
        make_plan(args, grid, store, args.holdout)
    _run_all(args, grid, plan_path, args.holdout)
    years = np.load(plan_path)["target_years"]
    pred = assemble(plan_path.parent, grid, store, "sum", years, None, args.realizations)
    truth = assemble(plan_path.parent, grid, store, "truth", years, None, 1)
    if args.max_tiles is not None:
        print(f"Partial run: {args.max_tiles} tiles. Unprocessed cells are zero in both maps and score as invalid.")
    if grid.name == "lite":
        scores = official_scores(truth, pred)
    else:
        sol, sub = ks.indices_xr_to_kaggle_dataframe(truth, index_columns=ks.CLIMX_INDEX_COLUMNS), ks.indices_xr_to_kaggle_dataframe(pred, index_columns=ks.CLIMX_INDEX_COLUMNS)
        scores = {"overall": rnNSE.score(sol.copy(), sub.copy(), "row_id")}
        for c in INDEX_COLUMNS:
            scores[c] = rnNSE.score(sol[["row_id", c]].copy(), sub[["row_id", c]].copy(), "row_id")
    print(f"Validation {args.holdout}, grid {grid.name}: overall {scores['overall']:+.4f}")
    print("  " + "  ".join(f"{k} {v:+.2f}" for k, v in scores.items() if k != "overall"))
    OUT.mkdir(exist_ok=True)
    path = OUT / f"native_validation_{grid.name}_{args.holdout}_k{args.n_analogues}_m{args.realizations}{args.tag}.json"
    path.write_text(json.dumps({"config": {k: v for k, v in vars(args).items() if k != "func"}, "scores": scores}, indent=2))


def ocean_fill(realizations: int = 8) -> xr.Dataset:
    """Index maps for unscored cells, from the seven-variable lite emulator and the official code."""
    hist_ds = open_hist()
    lat, lon = hist_ds["lat"].values, hist_ds["lon"].values
    hist = to_run("hist", hist_ds)
    library = [to_run(s, open_ssp(s)) for s in TRAIN_SCENARIOS]
    target = to_run("ssp245", open_ssp("ssp245"))
    model = ForcedAnalogueEmulator(Config()).fit(hist, library, lat)
    total = None
    for r in range(realizations):
        daily = model.realization(target, r, variables=list(INDEX_VARS))
        indices = compute_indices(blocks_to_dataset(daily, int(target.years[0]), lat, lon, TARGET_UNITS), hist_ds)
        total = indices if total is None else total + indices
    return upscale_nearest(total / realizations)


def submit(args) -> None:
    from make_submission import check_submission

    grid = get_grid(args.grid)
    assert grid.name == "native", "Only the native grid can be submitted through this script"
    store = store_for(grid)
    plan_path = _run_dir(args, grid, None) / "plan.npz"
    if not plan_path.exists():
        make_plan(args, grid, store, None)
    _run_all(args, grid, plan_path, None)
    folder = plan_path.parent
    missing_tiles = [t for t in range(len(store.tile_ids)) if not (folder / f"sum_{t:03d}.npz").exists()]
    assert not missing_tiles, f"Tiles without results: {missing_tiles}"
    years = np.load(plan_path)["target_years"]
    fill_path = folder / "ocean_fill.nc"
    if not fill_path.exists():
        ocean_fill().transpose("time", "lat", "lon").to_netcdf(fill_path)
    with xr.open_dataset(fill_path) as fill:
        maps = assemble(folder, grid, store, "sum", years, fill.load(), args.realizations)
    missing = {c: int(maps[c].isnull().sum()) for c in INDEX_COLUMNS if maps[c].isnull().any()}
    assert not missing, f"NaN in computed indices: {missing}"
    OUT.mkdir(exist_ok=True)
    path = OUT / f"submission_native_k{args.n_analogues}_m{args.realizations}{args.tag}.parquet"
    ks.write_kaggle_submission_parquet(maps, path, cast_integer_columns=False)
    check_submission(path)
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fetch_p = sub.add_parser("fetch")
    fetch_p.add_argument("--runs", nargs="+", default=list(FULL_FILES))
    fetch_p.set_defaults(func=fetch)
    for name, func in (("validate", validate), ("submit", submit)):
        p = sub.add_parser(name)
        p.add_argument("--realizations", type=int, default=48)
        p.add_argument("--n-analogues", type=int, default=48)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--no-shift", action="store_true")
        p.add_argument("--max-tiles", type=int, default=None, help="process only the first tiles (benchmark)")
        p.add_argument("--tag", default="")
        p.set_defaults(func=func)
        if name == "validate":
            p.add_argument("--holdout", choices=TRAIN_SCENARIOS, default="ssp370")
    for p in (fetch_p, *sub.choices.values()):
        if not any(a.dest == "grid" for a in p._actions):
            p.add_argument("--grid", choices=["native", "lite"], default="native")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()

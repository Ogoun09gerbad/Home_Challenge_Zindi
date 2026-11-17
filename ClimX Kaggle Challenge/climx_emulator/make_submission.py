"""Train on all scenarios, emulate SSP2-4.5 and write the Kaggle submission file.

Example: python climx_emulator/make_submission.py --realizations 48
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from common import OUT, TARGET_UNITS, TARGETS, TRAIN_SCENARIOS, blocks_to_dataset, compute_indices, ks, open_hist, open_ssp, rnNSE, to_run, upscale_nearest
from model import Config, ForcedAnalogueEmulator

N_ROWS = 86 * 192 * 288


def check_submission(path) -> None:
    """Reread the written file and check it against the documented format and the official scorer."""
    df = pd.read_parquet(path)
    expected = ["row_id", *ks.CLIMX_INDEX_COLUMNS]
    assert list(df.columns) == expected, f"Unexpected columns: {list(df.columns)}"
    assert len(df) == N_ROWS, f"Expected {N_ROWS} rows, found {len(df)}"
    assert (df["row_id"].values == np.arange(N_ROWS)).all(), "row_id is not 0..N-1 in order"
    values = df[list(ks.CLIMX_INDEX_COLUMNS)].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all(), "Submission contains NaN or infinite values"
    ks.validate_kaggle_submission_dataframe(df)
    self_score = rnNSE.score(df.copy(), df.copy(), "row_id")
    assert abs(self_score - 1.0) < 1e-12, f"Official scorer self-check returned {self_score}"
    print(f"Format checks passed; official scorer accepts the file (self-score {self_score:.6f}).")
    print(df[list(ks.CLIMX_INDEX_COLUMNS)].describe().loc[["min", "mean", "max"]].T.round(2).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--realizations", type=int, default=48)
    parser.add_argument("--n-analogues", type=int, default=Config.n_analogues)
    parser.add_argument("--no-aerosols", action="store_true")
    parser.add_argument("--no-quadratic", action="store_true")
    parser.add_argument("--name", default="submission_lite_forced_analogue")
    args = parser.parse_args()

    start = time.time()
    hist_ds = open_hist()
    lat, lon = hist_ds["lat"].values, hist_ds["lon"].values
    hist = to_run("hist", hist_ds)
    library = [to_run(s, open_ssp(s)) for s in TRAIN_SCENARIOS]
    test_ds = open_ssp("ssp245")
    test_time = test_ds["time"].values
    target = to_run("ssp245", test_ds)

    config = Config(
        n_analogues=args.n_analogues,
        use_aerosols=not args.no_aerosols,
        quadratic=not args.no_quadratic,
    )
    emulator = ForcedAnalogueEmulator(config).fit(hist, library, lat)
    forced, _ = emulator.forced_state(target)
    print(f"Forced GMST SSP2-4.5: {forced[0]:.2f} K (2015) -> {forced[-1]:.2f} K (2100)", flush=True)

    OUT.mkdir(exist_ok=True)
    total = None
    for r in range(args.realizations):
        # The emulator always produces all seven daily target variables; indices read four of them.
        daily = emulator.realization(target, r, variables=TARGETS)
        ds = blocks_to_dataset(daily, int(target.years[0]), lat, lon, TARGET_UNITS)
        if r == 0:
            assert (ds["time"].values == test_time).all(), "Emulated time axis differs from the test file"
            encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
            ds.to_netcdf(OUT / "daily_fields_ssp245_realisation0.nc", encoding=encoding)
        indices = compute_indices(ds, hist_ds)
        total = indices if total is None else total + indices
        print(f"realisation {r + 1}/{args.realizations} ({time.time() - start:.0f}s)", flush=True)
    mean_indices = total / args.realizations
    # The official writer silently turns NaN into 0, so reject them before it runs.
    missing = {k: int(v.isnull().sum()) for k, v in mean_indices.items() if v.isnull().any()}
    assert not missing, f"NaN in computed indices: {missing}"
    mean_indices.to_netcdf(OUT / "indices_ssp245_lite.nc")

    path = OUT / f"{args.name}.parquet"
    ks.write_kaggle_submission_parquet(upscale_nearest(mean_indices), path, cast_integer_columns=False)
    check_submission(path)
    (OUT / f"{args.name}.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.0f} MB) in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()

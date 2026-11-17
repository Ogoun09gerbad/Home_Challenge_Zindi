"""Leave-one-scenario-out validation with the official index code and scorer.

Example: python climx_emulator/validate.py --holdout ssp370 --realizations 16
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import xarray as xr

from common import OUT, TARGET_UNITS, TRAIN_SCENARIOS, blocks_to_dataset, compute_indices, official_scores, open_hist, open_ssp, to_run
from model import Config, ForcedAnalogueEmulator


def ensemble_indices(emulator, target, n, hist_ds, lat, lon) -> tuple[xr.Dataset, xr.Dataset]:
    """Unweighted mean of indices over realisations; also returns the first realisation's indices."""
    total = first = None
    for r in range(n):
        daily = emulator.realization(target, r, variables=["tas", "tasmax", "tasmin", "pr"])
        ds = blocks_to_dataset(daily, int(target.years[0]), lat, lon, TARGET_UNITS)
        indices = compute_indices(ds, hist_ds)
        first = indices if first is None else first
        total = indices if total is None else total + indices
        print(f"  realisation {r + 1}/{n}", flush=True)
    return total / n, first


def climatology_indices(hist, target, hist_ds, lat, lon) -> xr.Dataset:
    """Reference: repeat the 1985-2014 mean annual cycle (the smooth-field failure mode)."""
    keep = hist.years >= 1985
    daily = {
        v: np.repeat(hist.fields[v][keep].mean(axis=0, keepdims=True), len(target.years), axis=0)
        for v in ("tas", "tasmax", "tasmin", "pr")
    }
    return compute_indices(blocks_to_dataset(daily, int(target.years[0]), lat, lon, TARGET_UNITS), hist_ds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", choices=TRAIN_SCENARIOS, default="ssp370")
    parser.add_argument("--realizations", type=int, default=16)
    parser.add_argument("--n-analogues", type=int, default=Config.n_analogues)
    parser.add_argument("--no-shift", action="store_true")
    parser.add_argument("--no-aerosols", action="store_true")
    parser.add_argument("--no-quadratic", action="store_true")
    parser.add_argument("--baselines", action="store_true", help="also score climatology and one realisation")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    start = time.time()
    hist_ds = open_hist()
    lat, lon = hist_ds["lat"].values, hist_ds["lon"].values
    hist = to_run("hist", hist_ds)
    train = {s: to_run(s, open_ssp(s)) for s in TRAIN_SCENARIOS}
    target = train.pop(args.holdout)
    print(f"Loaded data in {time.time() - start:.0f}s; holdout={args.holdout}", flush=True)

    config = Config(
        n_analogues=args.n_analogues,
        apply_shift=not args.no_shift,
        use_aerosols=not args.no_aerosols,
        quadratic=not args.no_quadratic,
    )
    emulator = ForcedAnalogueEmulator(config).fit(hist, list(train.values()), lat)
    forced, _ = emulator.forced_state(target)
    actual = emulator._gmst(target)
    print(
        f"Forced GMST on holdout: RMSE {np.sqrt(np.mean((forced - actual) ** 2)):.3f} K, "
        f"bias {np.mean(forced - actual):+.3f} K (train fit RMSE {emulator.ebm_rmse:.3f} K)",
        flush=True,
    )

    truth_daily = blocks_to_dataset(
        {v: target.fields[v] for v in ("tas", "tasmax", "tasmin", "pr")}, int(target.years[0]), lat, lon, TARGET_UNITS
    )
    t0 = time.time()
    truth = compute_indices(truth_daily, hist_ds)
    print(f"Truth indices in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    mean_indices, first = ensemble_indices(emulator, target, args.realizations, hist_ds, lat, lon)
    print(f"{args.realizations} realisations in {time.time() - t0:.0f}s", flush=True)

    results = {f"ensemble_{args.realizations}": official_scores(truth, mean_indices)}
    if args.baselines:
        results["single_realisation"] = official_scores(truth, first)
        results["climatology_1985_2014"] = official_scores(
            truth, climatology_indices(hist, target, hist_ds, lat, lon)
        )
    for name, scores in results.items():
        print(f"\n{name}: overall {scores['overall']:+.4f}")
        print("  " + "  ".join(f"{k} {v:+.2f}" for k, v in scores.items() if k != "overall"))

    OUT.mkdir(exist_ok=True)
    path = OUT / f"validation_{args.holdout}{args.tag}.json"
    path.write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    print(f"\nSaved {path} ({time.time() - start:.0f}s total)")


if __name__ == "__main__":
    main()

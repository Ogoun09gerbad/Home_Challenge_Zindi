"""Generates the Zindi submission file (columns ID, Target).

The score used to label the data is additive and its noise is Gaussian
(experiments.md): three additive models (one function per feature, summed) on the
FINAL_N_FEATURES top-ranked features, trained with the probit likelihood
where possible.
  - Probit LightGBM (single-split trees, monotonicity, fine binning)
  - Probit P-spline GAM (gam.py), same monotonicity constraints
  - Probit CatBoost (single-split trees)
Each one is bagged over N_FOLDS folds x BAGGING_SEEDS seeds, then blended on the probit
scale (weighted mean of Φ⁻¹(p), weights maximising the Zindi score on the OOF). No
final recalibration: the blend is already calibrated in every risk bucket.

Usage:
    python src/predict.py
"""

import json
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (
    BLEND_WEIGHT_PATH,
    FINAL_CATBOOST_EARLY_STOPPING,
    FINAL_CATBOOST_PARAMS,
    FINAL_LGBM_EARLY_STOPPING,
    FINAL_N_FEATURES,
    ProbitCatBoostClassifier,
    ProbitLGBMClassifier,
    best_probit_blend_weights,
    final_gam_params,
    final_lgbm_params,
    load_train_features,
    probit_blend,
    run_cv,
    run_cv_catboost,
    run_cv_gam,
    select_features,
    zindi_score,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PREDS_DIR = PROCESSED_DIR / "predictions"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
ID_COL = "ID"
SUBMISSION_TARGET_COL = "Target"  # name expected by Zindi (see SampleSubmission.csv)
BAGGING_SEEDS = [42, 123, 2024, 7, 99, 777, 2025]
N_FOLDS = 10

# Pseudo-labeling (experiments.md, 2026-10-15): the test set (this challenge's Zindi rules
# verified as allowing its use during training) is added to each training
# fold with the soft label of the stage 1 blend (the "teacher", not
# pseudo-labelled) and this weight. Validated in diagnostics (weight 0.3 to 20, three CV
# seeds): monotone but decelerating gain. Confirmed on the public leaderboard with a single
# model at weight=20 (0.73902 → 0.74346 local, 0.741727 → 0.741914 public). Weight=1.0
# chosen for production (not 3 or 20): CatBoost slows down sharply with the
# weight (28s/fold without pseudo -> 83s at 0.6 -> 125s at 1.0 -> 262s at 1.3, 5-fold
# single-seed diagnostic); at weight=3 over 10 folds x 7 seeds, the extrapolation exceeded a
# plausible runtime on a machine shared with other jobs. Stage 2 bagging reduced to
# PSEUDO_BAGGING_SEEDS (3 seeds instead of 7) for the same cost reason.
PSEUDO_WEIGHT = 1.0
PSEUDO_BAGGING_SEEDS = [42, 123, 2024]


def load_test_features():
    df = pd.read_csv(PROCESSED_DIR / "test_features.csv", index_col=0, float_precision="round_trip")
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df


def bagged_run(run_fn, X, y, cat_cols, params, seed_key, esr, X_test, seeds=BAGGING_SEEDS, pseudo=None):
    """Trains `run_fn` (run_cv, run_cv_catboost or run_cv_gam) with several seeds and
    averages the OOF and test predictions over all seeds. The seed sets the
    fold split and, if `seed_key` is given, the model's randomness (the GAM has none).
    `pseudo=(X_p, y_p, weight)`: see train.run_cv."""
    oof_sum = np.zeros(len(y))
    test_sum = np.zeros(len(X_test))

    for seed in seeds:
        seeded_params = {**params, seed_key: seed} if seed_key else params
        models, oof_pred, _ = run_fn(
            X, y, cat_cols, params=seeded_params, n_splits=N_FOLDS, seed=seed, verbose=False, esr=esr,
            pseudo=pseudo,
        )
        oof_sum += oof_pred
        test_sum += np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)

    oof_avg = oof_sum / len(seeds)
    test_avg = test_sum / len(seeds)
    metrics = {"logloss": log_loss(y, oof_avg), "auc": roc_auc_score(y, oof_avg)}
    print(f"{N_FOLDS}-fold x {len(seeds)} seeds : logloss={metrics['logloss']:.5f}  auc={metrics['auc']:.5f}",
          flush=True)
    return oof_avg, test_avg, metrics


def report(label, y, p):
    ll, auc = log_loss(y, p), roc_auc_score(y, p)
    print(f"{label:48s} logloss={ll:.5f}  auc={auc:.5f}  estimated Zindi score={zindi_score(ll, auc):.5f}")


def main():
    X_train, y_train, cat_cols = load_train_features()
    X_test = load_test_features()
    y = y_train.to_numpy()

    # Categorical columns must have the same categories in train/test.
    for c in cat_cols:
        categories = X_train[c].cat.categories
        X_test[c] = X_test[c].astype(pd.CategoricalDtype(categories=categories))

    selected = select_features(X_train, y_train, cat_cols, FINAL_N_FEATURES)
    selected_cats = [c for c in cat_cols if c in selected]
    print(f"{len(selected)} features selected out of {X_train.shape[1]} | {N_FOLDS} folds | "
          f"bagging seeds: {BAGGING_SEEDS}")
    print("  " + ", ".join(selected) + "\n")

    X_sel, X_test_sel = X_train[selected], X_test[selected]

    def train_three(pseudo=None, seeds=BAGGING_SEEDS):
        oof_lgbm, test_lgbm, _ = bagged_run(
            partial(run_cv, model_cls=ProbitLGBMClassifier), X_sel, y_train, selected_cats,
            final_lgbm_params(selected), "random_state", FINAL_LGBM_EARLY_STOPPING, X_test_sel,
            seeds=seeds, pseudo=pseudo,
        )
        oof_gam, test_gam, _ = bagged_run(
            run_cv_gam, X_sel, y_train, selected_cats, final_gam_params(selected), None, None, X_test_sel,
            seeds=seeds, pseudo=pseudo,
        )
        oof_cb, test_cb, _ = bagged_run(
            partial(run_cv_catboost, model_cls=ProbitCatBoostClassifier), X_sel, y_train, selected_cats,
            FINAL_CATBOOST_PARAMS, "random_seed", FINAL_CATBOOST_EARLY_STOPPING, X_test_sel,
            seeds=seeds, pseudo=pseudo,
        )
        return (
            {"lightgbm": oof_lgbm, "gam": oof_gam, "catboost": oof_cb},
            {"lightgbm": test_lgbm, "gam": test_gam, "catboost": test_cb},
        )

    print("=== Stage 1/2: base models (no pseudo-labeling), to build the teacher ===")
    print("--- Additive probit LightGBM (single-split trees, monotonicity) ---")
    print("--- Probit P-spline GAM (monotonicity) / additive probit CatBoost ---", flush=True)
    oof1, test1 = train_three()
    weights1, _ = best_probit_blend_weights(y, oof1)
    weights1 = {name: float(w) for name, w in weights1.items()}
    oof_blend1 = probit_blend(oof1, weights1)
    test_pred1 = probit_blend(test1, weights1)
    print()
    report("Stage 1 — Probit LightGBM alone", y, oof1["lightgbm"])
    report("Stage 1 — Probit GAM alone", y, oof1["gam"])
    report("Stage 1 — Probit CatBoost alone", y, oof1["catboost"])
    report("Stage 1 — Blend (teacher) " + " + ".join(f"{w:.2f}*{n}" for n, w in weights1.items()),
           y, oof_blend1)

    print(f"\n=== Stage 2/2: retraining with pseudo-labels (full test set, weight={PSEUDO_WEIGHT}, "
          f"{len(PSEUDO_BAGGING_SEEDS)} seeds) ===", flush=True)
    pseudo = (X_test_sel, test_pred1, PSEUDO_WEIGHT)
    oof, test = train_three(pseudo=pseudo, seeds=PSEUDO_BAGGING_SEEDS)

    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    for name in oof:
        np.save(PREDS_DIR / f"oof_{name}.npy", oof[name])
        np.save(PREDS_DIR / f"test_{name}.npy", test[name])

    weights, _ = best_probit_blend_weights(y, oof)
    weights = {name: float(w) for name, w in weights.items()}
    with open(BLEND_WEIGHT_PATH, "w") as f:
        json.dump({"probit_blend_weights": weights}, f, indent=2)
    oof_blend = probit_blend(oof, weights)
    test_pred = probit_blend(test, weights)

    print()
    report("Stage 2 — Probit LightGBM alone", y, oof["lightgbm"])
    report("Stage 2 — Probit GAM alone", y, oof["gam"])
    report("Stage 2 — Probit CatBoost alone", y, oof["catboost"])
    report("Stage 2 — Final blend " + " + ".join(f"{w:.2f}*{n}" for n, w in weights.items()), y, oof_blend)

    submission = pd.DataFrame({ID_COL: X_test.index, SUBMISSION_TARGET_COL: test_pred})
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SUBMISSIONS_DIR / f"submission_{timestamp}.csv"
    submission.to_csv(out_path, index=False)

    print(f"\nSubmission saved: {out_path}")
    print(
        "Reminder: check the number of remaining submissions (max 10/day, 200 in "
        "total) before uploading this file to Zindi."
    )


if __name__ == "__main__":
    main()

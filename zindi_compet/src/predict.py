"""Génère le fichier de soumission Zindi (colonnes ID, Target).

Le score qui a servi à étiqueter les données est additif et son bruit est gaussien
(experiments.md) : trois modèles additifs (une fonction par feature, sommées) sur les
FINAL_N_FEATURES features les mieux classées, entraînés avec la vraisemblance probit
quand c'est possible.
  - LightGBM probit (arbres à une division, monotonie, discrétisation fine)
  - GAM probit à P-splines (gam.py), mêmes contraintes de monotonie
  - CatBoost probit (arbres à une division)
Chacun est baggé sur N_FOLDS plis x BAGGING_SEEDS seeds, puis blendés sur l'échelle
probit (moyenne pondérée de Φ⁻¹(p), poids maximisant le score Zindi sur les OOF). Pas
de recalibration finale : le blend est déjà calibré dans toutes les tranches de risque.

Usage :
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
SUBMISSION_TARGET_COL = "Target"  # nom attendu par Zindi (voir SampleSubmission.csv)
BAGGING_SEEDS = [42, 123, 2024, 7, 99, 777, 2025]
N_FOLDS = 10

# Pseudo-labeling (experiments.md, 2026-10-15) : le test set (règlement Zindi de ce défi
# vérifié comme autorisant son usage pendant l'entraînement) est ajouté à chaque pli
# d'entraînement avec l'étiquette douce du blend de l'étape 1 (le « professeur », non
# pseudo-labellisé) et ce poids. Validé en diagnostic (poids 0.3 à 20, trois seeds de
# CV) : gain monotone mais décélérant. Confirmé sur le classement public avec un seul
# modèle à poids=20 (0.73902 → 0.74346 local, 0.741727 → 0.741914 public). Poids=1.0
# retenu pour la production (pas 3 ou 20) : CatBoost ralentit très fortement avec le
# poids (28s/pli sans pseudo -> 83s à 0.6 -> 125s à 1.0 -> 262s à 1.3, diagnostic 5-fold
# seed unique) ; à poids=3 sur 10 plis x 7 seeds, l'extrapolation dépassait l'heure
# plausible sur une machine partagée avec d'autres jobs. Bagging de l'étape 2 réduit à
# PSEUDO_BAGGING_SEEDS (3 seeds au lieu de 7) pour la même raison de coût.
PSEUDO_WEIGHT = 1.0
PSEUDO_BAGGING_SEEDS = [42, 123, 2024]


def load_test_features():
    df = pd.read_csv(PROCESSED_DIR / "test_features.csv", index_col=0, float_precision="round_trip")
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df


def bagged_run(run_fn, X, y, cat_cols, params, seed_key, esr, X_test, seeds=BAGGING_SEEDS, pseudo=None):
    """Entraîne `run_fn` (run_cv, run_cv_catboost ou run_cv_gam) avec plusieurs seeds et
    moyenne les prédictions OOF et test sur l'ensemble des seeds. La seed fixe le
    découpage en plis et, si `seed_key` est donné, l'aléa du modèle (le GAM n'en a pas).
    `pseudo=(X_p, y_p, poids)` : voir train.run_cv."""
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
    print(f"{label:48s} logloss={ll:.5f}  auc={auc:.5f}  score Zindi estimé={zindi_score(ll, auc):.5f}")


def main():
    X_train, y_train, cat_cols = load_train_features()
    X_test = load_test_features()
    y = y_train.to_numpy()

    # Les colonnes catégorielles doivent avoir les mêmes catégories en train/test.
    for c in cat_cols:
        categories = X_train[c].cat.categories
        X_test[c] = X_test[c].astype(pd.CategoricalDtype(categories=categories))

    selected = select_features(X_train, y_train, cat_cols, FINAL_N_FEATURES)
    selected_cats = [c for c in cat_cols if c in selected]
    print(f"{len(selected)} features sélectionnées sur {X_train.shape[1]} | {N_FOLDS} plis | "
          f"seeds de bagging : {BAGGING_SEEDS}")
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

    print("=== Étape 1/2 : modèles de base (sans pseudo-labeling), pour fabriquer le professeur ===")
    print("--- LightGBM additif probit (arbres à une division, monotonie) ---")
    print("--- GAM probit à P-splines (monotonie) / CatBoost additif probit ---", flush=True)
    oof1, test1 = train_three()
    weights1, _ = best_probit_blend_weights(y, oof1)
    weights1 = {name: float(w) for name, w in weights1.items()}
    oof_blend1 = probit_blend(oof1, weights1)
    test_pred1 = probit_blend(test1, weights1)
    print()
    report("Étape 1 — LightGBM probit seul", y, oof1["lightgbm"])
    report("Étape 1 — GAM probit seul", y, oof1["gam"])
    report("Étape 1 — CatBoost probit seul", y, oof1["catboost"])
    report("Étape 1 — Blend (professeur) " + " + ".join(f"{w:.2f}*{n}" for n, w in weights1.items()),
           y, oof_blend1)

    print(f"\n=== Étape 2/2 : ré-entraînement avec pseudo-labels (test entier, poids={PSEUDO_WEIGHT}, "
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
    report("Étape 2 — LightGBM probit seul", y, oof["lightgbm"])
    report("Étape 2 — GAM probit seul", y, oof["gam"])
    report("Étape 2 — CatBoost probit seul", y, oof["catboost"])
    report("Étape 2 — Blend final " + " + ".join(f"{w:.2f}*{n}" for n, w in weights.items()), y, oof_blend)

    submission = pd.DataFrame({ID_COL: X_test.index, SUBMISSION_TARGET_COL: test_pred})
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SUBMISSIONS_DIR / f"submission_{timestamp}.csv"
    submission.to_csv(out_path, index=False)

    print(f"\nSoumission sauvegardée : {out_path}")
    print(
        "Rappel : vérifie le nombre de soumissions restantes (max 10/jour, 200 au "
        "total) avant d'envoyer ce fichier sur Zindi."
    )


if __name__ == "__main__":
    main()

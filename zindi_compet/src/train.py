"""Modèles, hyperparamètres finaux et outils de blend du pipeline (voir predict.py).

Métrique du défi, retrouvée à partir de deux scores publics (experiments.md,
2026-10-10) :
    score = 0.4 * AUC + 0.6 * (1 - logloss / 0.595)
Un millième de logloss vaut donc ~1 millième de score, contre 0.4 pour l'AUC.
"""

import itertools
import sys
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import special
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gam import ProbitGAM
from validation import N_SPLITS, SEED, get_folds

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TARGET = "liquidity_stress_next_30d"
BLEND_WEIGHT_PATH = PROCESSED_DIR / "blend_weight.json"

LOGLOSS_WEIGHT = 0.6
AUC_WEIGHT = 0.4
ZINDI_LOGLOSS_REF = 0.595


def zindi_score(logloss: float, auc: float) -> float:
    return AUC_WEIGHT * auc + LOGLOSS_WEIGHT * (1 - logloss / ZINDI_LOGLOSS_REF)


# Hyperparamètres figés dans le code (pas lus depuis data/processed/, exclu de git) pour
# que le pipeline se reproduise à l'identique depuis un clone du dépôt. Avec les features
# v3, le score caché est additif : des arbres à une seule division (une fonction par
# feature, sommées) font mieux que des arbres plus profonds (experiments.md, 2026-10-10 :
# LightGBM profondeur 1/2/3 = 0.23101/0.23155/0.23237, CatBoost 1/2/4 = 0.23059/0.23099/0.23096).
FINAL_LGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 20000,
    "learning_rate": 0.05,
    "max_depth": 1,
    "num_leaves": 2,
    "min_child_samples": 64,
    "subsample": 0.89,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "reg_lambda": 0.013,
    "random_state": SEED,
    "verbosity": -1,
}
FINAL_LGBM_EARLY_STOPPING = 500

FINAL_CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 20000,
    "learning_rate": 0.08,
    "depth": 1,
    "l2_leaf_reg": 11.73,
    "random_strength": 0.0625,
    "bagging_temperature": 2.42,
    "random_seed": SEED,
    "verbose": False,
}
FINAL_CATBOOST_EARLY_STOPPING = 500

# 15 features suffisent, sélectionnées sans biais dans chaque pli (rank_features) :
# au-delà on ajoute de la variance sans signal (0.22983 à 15 features contre 0.23045
# à 137, experiments.md 2026-10-10).
FINAL_N_FEATURES = 15

# Sens de variation imposés à LightGBM (1 : le risque augmente avec la feature, -1 : il
# diminue). Lus sur les fonctions de forme du modèle sans contrainte (corrélation de
# rang > 0.8) et conformes à la logique financière : dépenser plus que ses entrées, voir
# ses entrées ou son solde baisser, de nouveaux paiements, un solde volatil ou une
# faible activité font tous monter le risque. Contraintes utiles à LightGBM (0.22950
# contre 0.22978 sans) mais dégradent CatBoost (0.23217 contre 0.22976), qui reste libre.
MONOTONE_DIRECTIONS = {
    "outflow_over_inflow_rec": 1,
    "withdraw_over_inflow_rec": 1,
    "payments_value_ratio": 1,
    "daily_avg_bal_cv_6m": 1,
    "inflow_value_ratio": -1,
    "daily_avg_bal_momentum_recent_vs_old": -1,
    "bal_ratio": -1,
    "net_rec_over_inflow_old": -1,
    "x_90_d_activity_rate": -1,
}
# Discrétisation plus fine que le défaut (255) : les ratios sont très asymétriques.
FINAL_LGBM_MAX_BIN = 1023


def final_lgbm_params(columns):
    """FINAL_LGBM_PARAMS + discrétisation fine + monotonie alignée sur `columns`."""
    return {
        **FINAL_LGBM_PARAMS,
        "max_bin": FINAL_LGBM_MAX_BIN,
        "monotone_constraints": [MONOTONE_DIRECTIONS.get(c, 0) for c in columns],
    }


# GAM probit à P-splines (gam.py), second modèle du blend : 20 nœuds, λ=5 (grille
# testée en 5-fold, experiments.md 2026-10-10 : λ=10 -> 0.23050, λ=20 -> 0.23073,
# 15 nœuds -> 0.23068 ; retenu : 0.23043).
FINAL_GAM_PARAMS = {"n_knots": 20, "lam": 5.0}


def final_gam_params(columns):
    """FINAL_GAM_PARAMS + monotonie (MONOTONE_DIRECTIONS) alignée sur `columns`."""
    return {**FINAL_GAM_PARAMS, "monotone": [MONOTONE_DIRECTIONS.get(c, 0) for c in columns]}


# Lien probit (experiments.md, 2026-10-10) : la cible vaut 1 pour les 15% de lignes dont
# « score caché + bruit » est le plus élevé, et ce bruit est gaussien. Donc
# P(y=1|x) = Φ(s(x) - seuil) : le bon objectif est la vraisemblance probit, pas logit
# (qui surestime les faibles risques de 3 σ et perd 0.0007 de logloss).
def probit_grad_hess(eta, y):
    """Gradient et hessienne de la logloss probit −[y·logΦ(η) + (1−y)·log(1−Φ(η))].

    En log (log_ndtr) pour rester stable quand Φ(η) est proche de 0 ou de 1. La NLL est
    linéaire en y, donc cette formule reste valide pour y continu dans [0,1] (étiquette
    douce de pseudo-labeling), pas seulement y binaire : elle en est l'interpolation
    linéaire exacte (identique au cas dur quand y∈{0,1}).
    """
    log_cdf, log_sf = special.log_ndtr(eta), special.log_ndtr(-eta)
    log_pdf = -0.5 * eta**2 - 0.5 * np.log(2 * np.pi)
    r1, r0 = np.exp(log_pdf - log_cdf), np.exp(log_pdf - log_sf)  # φ/Φ et φ/(1−Φ)
    grad = -y * r1 + (1 - y) * r0
    hess = y * (r1 * (r1 + eta)) + (1 - y) * (r0 * (r0 - eta))
    return grad, hess


def probit_logloss(y, eta):
    log_cdf, log_sf = special.log_ndtr(eta), special.log_ndtr(-eta)
    return -np.mean(y * log_cdf + (1 - y) * log_sf)


def _lgb_probit_objective(preds, data):
    # preds inclut déjà l'init_score du Dataset : pas à rajouter.
    grad, hess = probit_grad_hess(preds, data.get_label())
    # LightGBM renvoie weight=None quand tous les poids valent 1 (optimisation interne) :
    # il faut alors le traiter comme 1.0, sinon TypeError (pseudo-labeling, train.py 2026-10-15).
    weight = data.get_weight()
    if weight is None:
        weight = 1.0
    return grad * weight, hess * weight


def _lgb_probit_metric(preds, data):
    return "probit_logloss", probit_logloss(data.get_label(), preds), False


class ProbitLGBMClassifier:
    """LightGBM entraîné avec la vraisemblance probit : P(y=1|x) = Φ(η(x)).

    Même interface que LGBMClassifier (fit avec jeu de validation pour l'arrêt
    précoce, predict_proba) pour s'intégrer dans run_cv. Le score part de
    Φ⁻¹(taux de positifs), l'équivalent probit de boost_from_average.
    """

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y, eval_X, eval_y, categorical_feature="auto", callbacks=None, sample_weight=None):
        y, eval_y = np.asarray(y, dtype=float), np.asarray(eval_y, dtype=float)
        self.init_score_ = special.ndtri(np.average(y, weights=sample_weight))
        params = {k: v for k, v in self.params.items() if k not in ("objective", "n_estimators")}
        dataset_params = {"max_bin": params.get("max_bin", 255)}
        train_set = lgb.Dataset(X, y, weight=sample_weight, init_score=np.full(len(y), self.init_score_),
                                categorical_feature=categorical_feature, params=dataset_params)
        valid_set = lgb.Dataset(eval_X, eval_y, init_score=np.full(len(eval_y), self.init_score_),
                                reference=train_set)
        self.booster_ = lgb.train(
            {**params, "objective": _lgb_probit_objective, "metric": "None"},
            train_set,
            num_boost_round=self.params.get("n_estimators", 20000),
            valid_sets=[valid_set],
            feval=_lgb_probit_metric,
            callbacks=callbacks,
        )
        return self

    def decision_function(self, X):
        """Score η sur l'échelle probit."""
        return self.booster_.predict(X, num_iteration=self.booster_.best_iteration) + self.init_score_

    def predict_proba(self, X):
        p = special.ndtr(self.decision_function(X))
        return np.column_stack([1 - p, p])


class _CatBoostProbitObjective:
    """Dérivées de la logloss probit pour CatBoost (convention der1/der2 = gradient et
    hessienne changés de signe, comme pour l'exemple Logloss natif de la doc CatBoost)."""

    def calc_ders_range(self, approxes, targets, weights):
        grad, hess = probit_grad_hess(np.asarray(approxes, dtype=float), np.asarray(targets, dtype=float))
        w = np.asarray(weights, dtype=float) if weights is not None else 1.0
        return list(zip((-grad * w).tolist(), (-hess * w).tolist()))


class _CatBoostProbitMetric:
    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        eta, y = np.asarray(approxes[0], dtype=float), np.asarray(target, dtype=float)
        return probit_logloss(y, eta) * len(y), float(len(y))

    def get_final_error(self, error, weight):
        return error / weight if weight else 0.0


class ProbitCatBoostClassifier:
    """CatBoost entraîné avec la vraisemblance probit (même raison que ProbitLGBMClassifier).

    Même interface native que CatBoostClassifier (fit avec eval_set, cat_features,
    early_stopping_rounds) pour s'intégrer dans run_cv_catboost sans rien y changer.
    """

    def __init__(self, **params):
        self.params = {k: v for k, v in params.items() if k not in ("loss_function", "eval_metric")}

    def fit(self, X, y, eval_set, cat_features=None, early_stopping_rounds=500, verbose=False, sample_weight=None):
        y = np.asarray(y, dtype=float)
        X_va, y_va = eval_set
        y_va = np.asarray(y_va, dtype=float)
        self.init_score_ = special.ndtri(np.average(y, weights=sample_weight))
        pool_tr = cb.Pool(X, y, cat_features=cat_features, weight=sample_weight,
                          baseline=np.full(len(y), self.init_score_))
        pool_va = cb.Pool(X_va, y_va, cat_features=cat_features, baseline=np.full(len(y_va), self.init_score_))
        self.booster_ = cb.CatBoost(
            {**self.params, "loss_function": _CatBoostProbitObjective(), "eval_metric": _CatBoostProbitMetric()}
        )
        self.booster_.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        return self

    def decision_function(self, X):
        """Score η sur l'échelle probit."""
        return self.booster_.predict(X, prediction_type="RawFormulaVal") + self.init_score_

    def predict_proba(self, X):
        p = special.ndtr(self.decision_function(X))
        return np.column_stack([1 - p, p])


def load_train_features():
    # round_trip : relit les flottants exactement (le lecteur rapide par défaut arrondit
    # à 13 chiffres significatifs, assez pour déplacer une valeur d'un côté d'un seuil).
    df = pd.read_csv(PROCESSED_DIR / "train_features.csv", index_col=0, float_precision="round_trip")
    y = df.pop(TARGET)
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df, y, cat_cols


def _augment_with_pseudo(X_tr, y_tr, pseudo):
    """Ajoute les lignes pseudo-labellisées (X_p, y_p, poids) à l'entraînement d'un pli ;
    `None` si pas de pseudo-labeling. Jamais ajoutées à la validation (OOF honnête sur
    les seules vraies lignes train). `y_p` peut être une étiquette douce (probabilité
    continue), `probit_grad_hess` généralisé le gère nativement."""
    weight_tr = np.ones(len(y_tr))
    if pseudo is None:
        return X_tr, y_tr, weight_tr
    X_p, y_p, weight_p = pseudo
    X_tr = pd.concat([X_tr, X_p], axis=0, ignore_index=True)
    y_tr = pd.concat([pd.Series(np.asarray(y_tr)), pd.Series(np.asarray(y_p))], ignore_index=True)
    weight_tr = np.r_[weight_tr, np.full(len(y_p), weight_p)]
    return X_tr, y_tr, weight_tr


def _cv_loop(X, y, n_splits, seed, fit_fold, verbose):
    """Boucle de validation croisée commune à run_cv / run_cv_catboost / run_cv_gam :
    `fit_fold(X_tr, y_tr, X_va, y_va)` entraîne un modèle et renvoie (modèle, proba)."""
    oof_pred = np.zeros(len(y))
    models = []
    for fold, (tr_idx, va_idx) in enumerate(get_folds(y, n_splits=n_splits, seed=seed)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model, proba = fit_fold(X_tr, y_tr, X_va, y_va)
        oof_pred[va_idx] = proba
        models.append(model)
        if verbose:
            print(f"Fold {fold}: logloss={log_loss(y_va, proba):.5f}  auc={roc_auc_score(y_va, proba):.5f}")

    metrics = {"logloss": log_loss(y, oof_pred), "auc": roc_auc_score(y, oof_pred)}
    if verbose:
        print(f"\nOOF global : logloss={metrics['logloss']:.5f}  auc={metrics['auc']:.5f}")
    return models, oof_pred, metrics


def run_cv(X, y, cat_cols, params=None, n_splits=N_SPLITS, seed=SEED, verbose=True, esr=50,
           model_cls=lgb.LGBMClassifier, pseudo=None):
    """Validation croisée LightGBM ; `model_cls=ProbitLGBMClassifier` pour le lien probit.
    `pseudo=(X_p, y_p, poids)` : lignes de test pseudo-labellisées ajoutées à chaque pli
    d'entraînement (jamais à la validation)."""
    params = params or FINAL_LGBM_PARAMS

    def fit_fold(X_tr, y_tr, X_va, y_va):
        X_tr, y_tr, weight_tr = _augment_with_pseudo(X_tr, y_tr, pseudo)
        model = model_cls(**params)
        model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va, categorical_feature=cat_cols,
                  callbacks=[lgb.early_stopping(esr, verbose=False)], sample_weight=weight_tr)
        return model, model.predict_proba(X_va)[:, 1]

    return _cv_loop(X, y, n_splits, seed, fit_fold, verbose)


def rank_features(X, y, cat_cols, n_splits=N_SPLITS, seed=SEED):
    """Classe les features par gain cumulé du LightGBM final (additif) sur les plis.

    Déterministe (seed et plis fixés) : redonne le même classement à chaque exécution,
    sans dépendre d'un fichier généré hors du dépôt.
    """
    gain = np.zeros(X.shape[1])
    for tr_idx, va_idx in get_folds(y, n_splits=n_splits, seed=seed):
        model = lgb.LGBMClassifier(**FINAL_LGBM_PARAMS)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx], eval_X=X.iloc[va_idx], eval_y=y.iloc[va_idx],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(FINAL_LGBM_EARLY_STOPPING, verbose=False)],
        )
        gain += model.booster_.feature_importance("gain")
    return pd.Series(gain, index=X.columns).sort_values(ascending=False)


def select_features(X, y, cat_cols, k):
    """Les k features au plus fort gain (rank_features)."""
    return list(rank_features(X, y, cat_cols).index[:k])


def run_cv_catboost(X, y, cat_cols, params=None, n_splits=N_SPLITS, seed=SEED, verbose=True, esr=50,
                     model_cls=cb.CatBoostClassifier, pseudo=None):
    """Même logique que run_cv, avec CatBoost (`model_cls=ProbitCatBoostClassifier` pour
    le lien probit) : son target encoding ordonné des catégorielles diversifie le blend
    par rapport aux splits de LightGBM. `pseudo` : voir run_cv."""
    params = params or FINAL_CATBOOST_PARAMS
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    def fit_fold(X_tr, y_tr, X_va, y_va):
        X_tr, y_tr, weight_tr = _augment_with_pseudo(X_tr, y_tr, pseudo)
        model = model_cls(**params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_idx,
                  early_stopping_rounds=esr, verbose=False, sample_weight=weight_tr)
        return model, model.predict_proba(X_va)[:, 1]

    return _cv_loop(X, y, n_splits, seed, fit_fold, verbose)


def run_cv_gam(X, y, cat_cols, params=None, n_splits=N_SPLITS, seed=SEED, verbose=True, esr=None,
               pseudo=None):
    """Même logique que run_cv, avec le GAM probit (gam.py). Pas d'arrêt précoce : la
    pénalité λ règle déjà la régularisation ; cat_cols et esr sont ignorés (features
    numériques uniquement). `pseudo` : voir run_cv."""
    params = params or final_gam_params(X.columns)

    def fit_fold(X_tr, y_tr, X_va, y_va):
        X_tr, y_tr, weight_tr = _augment_with_pseudo(X_tr, y_tr, pseudo)
        model = ProbitGAM(**params).fit(X_tr, y_tr, sample_weight=weight_tr)
        return model, model.predict_proba(X_va)[:, 1]

    return _cv_loop(X, y, n_splits, seed, fit_fold, verbose)


def probit_blend(preds, weights):
    """Moyenne pondérée des scores probit Φ⁻¹(p) des modèles, ramenée en probabilité.

    Moyenner les scores plutôt que les probabilités garde la structure additive du
    modèle (sur les OOF, les deux moyennes ne diffèrent que de 0.00001 en score).
    """
    eta = sum(w * special.ndtri(np.clip(preds[name], 1e-15, 1 - 1e-15)) for name, w in weights.items())
    return special.ndtr(eta)


def best_probit_blend_weights(y, oof_dict, step: float = 0.05):
    """Poids (positifs, de somme 1, grille de pas `step`) qui maximisent le score Zindi
    du blend probit des prédictions OOF. Trois modèles : 231 combinaisons testées."""
    names = list(oof_dict)
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
    best_score, best_weights, best_metrics = -np.inf, None, None
    for head in itertools.product(grid, repeat=len(names) - 1):
        if sum(head) > 1.0 + 1e-9:
            continue
        weights = dict(zip(names, [*head, max(0.0, round(1.0 - sum(head), 10))]))
        blend = probit_blend(oof_dict, weights)
        logloss, auc = log_loss(y, blend), roc_auc_score(y, blend)
        score = zindi_score(logloss, auc)
        if score > best_score:
            best_score, best_weights = score, weights
            best_metrics = {"logloss": logloss, "auc": auc}
    return best_weights, best_metrics

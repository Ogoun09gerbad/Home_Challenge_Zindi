"""Models, final hyperparameters and blending tools for the pipeline (see predict.py).

Challenge metric, recovered from two public scores (experiments.md,
2026-10-10):
    score = 0.4 * AUC + 0.6 * (1 - logloss / 0.595)
One thousandth of logloss is therefore worth ~1 thousandth of score, vs 0.4 for AUC.
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


# Hyperparameters pinned in the code (not read from data/processed/, excluded from git) so
# that the pipeline reproduces identically from a clone of the repository. With the v3
# features, the hidden score is additive: single-split trees (one function per
# feature, summed) do better than deeper trees (experiments.md, 2026-10-10:
# LightGBM depth 1/2/3 = 0.23101/0.23155/0.23237, CatBoost 1/2/4 = 0.23059/0.23099/0.23096).
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

# 15 features are enough, selected without bias within each fold (rank_features):
# beyond that we add variance without signal (0.22983 with 15 features vs 0.23045
# with 137, experiments.md 2026-10-10).
FINAL_N_FEATURES = 15

# Monotonic directions imposed on LightGBM (1: risk increases with the feature, -1: it
# decreases). Read from the shape functions of the unconstrained model (rank
# correlation > 0.8) and consistent with financial logic: spending more than one's inflows,
# seeing inflows or balance drop, new payments, a volatile balance or
# low activity all raise the risk. Constraints help LightGBM (0.22950
# vs 0.22978 without) but hurt CatBoost (0.23217 vs 0.22976), which stays unconstrained.
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
# Finer binning than the default (255): the ratios are highly skewed.
FINAL_LGBM_MAX_BIN = 1023


def final_lgbm_params(columns):
    """FINAL_LGBM_PARAMS + fine binning + monotonicity aligned with `columns`."""
    return {
        **FINAL_LGBM_PARAMS,
        "max_bin": FINAL_LGBM_MAX_BIN,
        "monotone_constraints": [MONOTONE_DIRECTIONS.get(c, 0) for c in columns],
    }


# Probit P-spline GAM (gam.py), second model of the blend: 20 knots, λ=5 (grid
# tested in 5-fold, experiments.md 2026-10-10: λ=10 -> 0.23050, λ=20 -> 0.23073,
# 15 knots -> 0.23068; chosen: 0.23043).
FINAL_GAM_PARAMS = {"n_knots": 20, "lam": 5.0}


def final_gam_params(columns):
    """FINAL_GAM_PARAMS + monotonicity (MONOTONE_DIRECTIONS) aligned with `columns`."""
    return {**FINAL_GAM_PARAMS, "monotone": [MONOTONE_DIRECTIONS.get(c, 0) for c in columns]}


# Probit link (experiments.md, 2026-10-10): the target is 1 for the 15% of rows whose
# "hidden score + noise" is highest, and this noise is Gaussian. Hence
# P(y=1|x) = Φ(s(x) - threshold): the right objective is the probit likelihood, not logit
# (which overestimates low risks by 3 σ and loses 0.0007 of logloss).
def probit_grad_hess(eta, y):
    """Gradient and Hessian of the probit logloss −[y·logΦ(η) + (1−y)·log(1−Φ(η))].

    Computed in log space (log_ndtr) to stay stable when Φ(η) is close to 0 or 1. The NLL is
    linear in y, so this formula remains valid for continuous y in [0,1] (pseudo-labeling
    soft label), not just binary y: it is its exact linear
    interpolation (identical to the hard case when y∈{0,1}).
    """
    log_cdf, log_sf = special.log_ndtr(eta), special.log_ndtr(-eta)
    log_pdf = -0.5 * eta**2 - 0.5 * np.log(2 * np.pi)
    r1, r0 = np.exp(log_pdf - log_cdf), np.exp(log_pdf - log_sf)  # φ/Φ and φ/(1−Φ)
    grad = -y * r1 + (1 - y) * r0
    hess = y * (r1 * (r1 + eta)) + (1 - y) * (r0 * (r0 - eta))
    return grad, hess


def probit_logloss(y, eta):
    log_cdf, log_sf = special.log_ndtr(eta), special.log_ndtr(-eta)
    return -np.mean(y * log_cdf + (1 - y) * log_sf)


def _lgb_probit_objective(preds, data):
    # preds already includes the Dataset's init_score: no need to add it.
    grad, hess = probit_grad_hess(preds, data.get_label())
    # LightGBM returns weight=None when all weights are 1 (internal optimisation):
    # it must then be treated as 1.0, otherwise TypeError (pseudo-labeling, train.py 2026-10-15).
    weight = data.get_weight()
    if weight is None:
        weight = 1.0
    return grad * weight, hess * weight


def _lgb_probit_metric(preds, data):
    return "probit_logloss", probit_logloss(data.get_label(), preds), False


class ProbitLGBMClassifier:
    """LightGBM trained with the probit likelihood: P(y=1|x) = Φ(η(x)).

    Same interface as LGBMClassifier (fit with a validation set for early
    stopping, predict_proba) to plug into run_cv. The score starts from
    Φ⁻¹(positive rate), the probit equivalent of boost_from_average.
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
        """Score η on the probit scale."""
        return self.booster_.predict(X, num_iteration=self.booster_.best_iteration) + self.init_score_

    def predict_proba(self, X):
        p = special.ndtr(self.decision_function(X))
        return np.column_stack([1 - p, p])


class _CatBoostProbitObjective:
    """Derivatives of the probit logloss for CatBoost (der1/der2 convention = gradient and
    Hessian with flipped sign, as in the native Logloss example of the CatBoost docs)."""

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
    """CatBoost trained with the probit likelihood (same reason as ProbitLGBMClassifier).

    Same native interface as CatBoostClassifier (fit with eval_set, cat_features,
    early_stopping_rounds) to plug into run_cv_catboost without changing anything there.
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
        """Score η on the probit scale."""
        return self.booster_.predict(X, prediction_type="RawFormulaVal") + self.init_score_

    def predict_proba(self, X):
        p = special.ndtr(self.decision_function(X))
        return np.column_stack([1 - p, p])


def load_train_features():
    # round_trip: reads floats back exactly (the default fast reader rounds
    # to 13 significant digits, enough to move a value to the other side of a threshold).
    df = pd.read_csv(PROCESSED_DIR / "train_features.csv", index_col=0, float_precision="round_trip")
    y = df.pop(TARGET)
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df, y, cat_cols


def _augment_with_pseudo(X_tr, y_tr, pseudo):
    """Adds the pseudo-labelled rows (X_p, y_p, weight) to a fold's training set;
    `None` if no pseudo-labeling. Never added to validation (honest OOF on
    the real train rows only). `y_p` can be a soft label (continuous
    probability); the generalised `probit_grad_hess` handles it natively."""
    weight_tr = np.ones(len(y_tr))
    if pseudo is None:
        return X_tr, y_tr, weight_tr
    X_p, y_p, weight_p = pseudo
    X_tr = pd.concat([X_tr, X_p], axis=0, ignore_index=True)
    y_tr = pd.concat([pd.Series(np.asarray(y_tr)), pd.Series(np.asarray(y_p))], ignore_index=True)
    weight_tr = np.r_[weight_tr, np.full(len(y_p), weight_p)]
    return X_tr, y_tr, weight_tr


def _cv_loop(X, y, n_splits, seed, fit_fold, verbose):
    """Cross-validation loop shared by run_cv / run_cv_catboost / run_cv_gam:
    `fit_fold(X_tr, y_tr, X_va, y_va)` trains a model and returns (model, proba)."""
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
        print(f"\nOverall OOF: logloss={metrics['logloss']:.5f}  auc={metrics['auc']:.5f}")
    return models, oof_pred, metrics


def run_cv(X, y, cat_cols, params=None, n_splits=N_SPLITS, seed=SEED, verbose=True, esr=50,
           model_cls=lgb.LGBMClassifier, pseudo=None):
    """LightGBM cross-validation; `model_cls=ProbitLGBMClassifier` for the probit link.
    `pseudo=(X_p, y_p, weight)`: pseudo-labelled test rows added to each training
    fold (never to validation)."""
    params = params or FINAL_LGBM_PARAMS

    def fit_fold(X_tr, y_tr, X_va, y_va):
        X_tr, y_tr, weight_tr = _augment_with_pseudo(X_tr, y_tr, pseudo)
        model = model_cls(**params)
        model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va, categorical_feature=cat_cols,
                  callbacks=[lgb.early_stopping(esr, verbose=False)], sample_weight=weight_tr)
        return model, model.predict_proba(X_va)[:, 1]

    return _cv_loop(X, y, n_splits, seed, fit_fold, verbose)


def rank_features(X, y, cat_cols, n_splits=N_SPLITS, seed=SEED):
    """Ranks features by cumulative gain of the final (additive) LightGBM across folds.

    Deterministic (fixed seed and folds): returns the same ranking on every run,
    without depending on a file generated outside the repository.
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
    """The k features with the highest gain (rank_features)."""
    return list(rank_features(X, y, cat_cols).index[:k])


def run_cv_catboost(X, y, cat_cols, params=None, n_splits=N_SPLITS, seed=SEED, verbose=True, esr=50,
                     model_cls=cb.CatBoostClassifier, pseudo=None):
    """Same logic as run_cv, with CatBoost (`model_cls=ProbitCatBoostClassifier` for
    the probit link): its ordered target encoding of categoricals diversifies the blend
    relative to LightGBM's splits. `pseudo`: see run_cv."""
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
    """Same logic as run_cv, with the probit GAM (gam.py). No early stopping: the
    penalty λ already controls regularisation; cat_cols and esr are ignored (numeric
    features only). `pseudo`: see run_cv."""
    params = params or final_gam_params(X.columns)

    def fit_fold(X_tr, y_tr, X_va, y_va):
        X_tr, y_tr, weight_tr = _augment_with_pseudo(X_tr, y_tr, pseudo)
        model = ProbitGAM(**params).fit(X_tr, y_tr, sample_weight=weight_tr)
        return model, model.predict_proba(X_va)[:, 1]

    return _cv_loop(X, y, n_splits, seed, fit_fold, verbose)


def probit_blend(preds, weights):
    """Weighted mean of the models' probit scores Φ⁻¹(p), mapped back to a probability.

    Averaging scores rather than probabilities preserves the additive structure of the
    model (on the OOF, the two averages differ by only 0.00001 in score).
    """
    eta = sum(w * special.ndtri(np.clip(preds[name], 1e-15, 1 - 1e-15)) for name, w in weights.items())
    return special.ndtr(eta)


def best_probit_blend_weights(y, oof_dict, step: float = 0.05):
    """Weights (non-negative, summing to 1, grid with step `step`) that maximise the Zindi
    score of the probit blend of the OOF predictions. Three models: 231 combinations tested."""
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

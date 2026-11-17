"""Feature engineering par client, à partir des agrégats mensuels M1 (récent) à M6 (ancien).

Les données brutes couvrent 7 types de transactions mobile money (volume, valeur
totale, montant max, nb de contreparties) plus le solde journalier moyen, par mois.
Seules la valeur totale et le solde portent du signal : contreparties et montant max
sont du bruit (tirages aléatoires sans lien avec la cible), et le volume n'ajoute rien
à la valeur totale (experiments.md, 2026-10-10).

Pour chaque série de 6 points mensuels : niveau, volatilité, comparaison bloc récent
(M1-M3) / bloc ancien (M4-M6), ratios entre flux, indicateurs d'apparition et d'arrêt.
"""

import numpy as np
import pandas as pd

MONTHS = [1, 2, 3, 4, 5, 6]

# Seule métrique utilisée par type de transaction (voir docstring du module).
TRANSACTION_TYPES = {
    "paybill": ["total_value"],
    "merchantpay": ["total_value"],
    "transfer_from_bank": ["total_value"],
    "mm_send": ["total_value"],
    "received": ["total_value"],
    "deposit": ["total_value"],
    "withdraw": ["total_value"],
}

OUTFLOW_TYPES = ["paybill", "merchantpay", "mm_send", "withdraw"]
INFLOW_TYPES = ["deposit", "received", "transfer_from_bank"]

EPS = 1e-6


def _monthly_columns(prefix_fn):
    return [prefix_fn(m) for m in MONTHS]


def _series_features(df: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    """Niveau, volatilité et momentum (bloc récent vs bloc ancien) d'une série de 6 mois.

    Seuls les totaux par bloc de 3 mois et la volatilité sur 6 mois portent du signal
    (experiments.md, 2026-10-10) : la tendance, le dernier mois seul, le ratio M1/M2
    et les min/max par bloc ont été retirés sans perte (LightGBM 0.23146 sans eux
    contre 0.23155 avec).
    """
    values = df[cols].to_numpy(dtype=float)
    recent = values[:, :3]  # M1, M2, M3
    older = values[:, 3:]  # M4, M5, M6

    mean = values.mean(axis=1)
    std = values.std(axis=1)

    out = pd.DataFrame(index=df.index)
    out[f"{name}_sum_6m"] = values.sum(axis=1)
    out[f"{name}_mean_6m"] = mean
    out[f"{name}_std_6m"] = std
    out[f"{name}_cv_6m"] = std / (np.abs(mean) + EPS)
    out[f"{name}_momentum_recent_vs_old"] = (recent.mean(axis=1) + EPS) / (
        older.mean(axis=1) + EPS
    )
    return out


def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de niveau/volatilité/momentum pour chaque (type de transaction, métrique)."""
    blocks = []
    for ttype, metrics in TRANSACTION_TYPES.items():
        for metric in metrics:
            cols = _monthly_columns(lambda m, t=ttype, me=metric: f"m{m}_{t}_{me}")
            blocks.append(_series_features(df, cols, f"{ttype}_{metric}"))
    return pd.concat(blocks, axis=1)


def build_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de niveau/volatilité/momentum pour le solde journalier moyen."""
    cols = _monthly_columns(lambda m: f"m{m}_daily_avg_bal")
    return _series_features(df, cols, "daily_avg_bal")


def _type_total_value_6m(df: pd.DataFrame, ttype: str) -> pd.Series:
    cols = _monthly_columns(lambda m, t=ttype: f"m{m}_{t}_total_value")
    return df[cols].sum(axis=1)


def build_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios inter-types de transaction (flux sortants/entrants, retraits/dépôts)."""
    out = pd.DataFrame(index=df.index)

    outflow_value = sum(_type_total_value_6m(df, t) for t in OUTFLOW_TYPES)
    inflow_value = sum(_type_total_value_6m(df, t) for t in INFLOW_TYPES)
    out["outflow_to_inflow_value_ratio"] = (outflow_value + EPS) / (inflow_value + EPS)

    withdraw_value = _type_total_value_6m(df, "withdraw")
    deposit_value = _type_total_value_6m(df, "deposit")
    out["withdraw_to_deposit_value_ratio"] = (withdraw_value + EPS) / (
        deposit_value + EPS
    )
    return out


ALL_TYPES = list(TRANSACTION_TYPES.keys())


def build_overall_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Niveau/volatilité/momentum de l'activité tous types confondus : un client peut
    basculer d'un type à l'autre (paybill -> mm_send) sans que son activité globale
    ne bouge, signal invisible aux features par type."""
    value_cols = [
        _monthly_columns(lambda m, t=ttype: f"m{m}_{ttype}_total_value")
        for ttype in ALL_TYPES
    ]
    total_value_per_month = sum(df[cols].to_numpy(dtype=float) for cols in value_cols)
    tmp_value = pd.DataFrame(
        total_value_per_month, columns=_monthly_columns(lambda m: f"m{m}"), index=df.index
    )
    return _series_features(tmp_value, tmp_value.columns.tolist(), "overall_value")


def build_share_features(df: pd.DataFrame) -> pd.DataFrame:
    """Part de chaque type dans le total des sorties/entrées sur 6 mois : capture la
    composition du comportement (cash vs factures), pas seulement son niveau."""
    out = pd.DataFrame(index=df.index)

    outflow_value = sum(_type_total_value_6m(df, t) for t in OUTFLOW_TYPES)
    for t in OUTFLOW_TYPES:
        out[f"{t}_share_of_outflow"] = (_type_total_value_6m(df, t) + EPS) / (
            outflow_value + EPS
        )

    inflow_value = sum(_type_total_value_6m(df, t) for t in INFLOW_TYPES)
    for t in INFLOW_TYPES:
        out[f"{t}_share_of_inflow"] = (_type_total_value_6m(df, t) + EPS) / (
            inflow_value + EPS
        )

    return out


def build_balance_to_income_features(df: pd.DataFrame) -> pd.DataFrame:
    """Solde rapporté au revenu (ARPU) : un solde bas n'inquiète pas pareil selon
    le revenu du client, ce qu'aucune des deux variables seule ne capture."""
    bal_cols = _monthly_columns(lambda m: f"m{m}_daily_avg_bal")
    bal_mean = df[bal_cols].mean(axis=1)
    out = pd.DataFrame(index=df.index)
    out["balance_to_arpu_ratio"] = bal_mean / (df["arpu"] + EPS)
    return out


FLOW_SMOOTHING = 1.0  # lissage additif des ratios : évite qu'un mois quasi nul fasse exploser un ratio


def _block_features(values: np.ndarray, name: str, out: dict) -> None:
    """Compare le bloc récent (M1-M3) au bloc ancien (M4-M6) d'une série mensuelle.

    L'analyse (experiments.md, 2026-10-10) montre que la cible se lit au niveau des
    blocs de 3 mois : chez les instantanés stressés, l'écart avec les non-stressés
    est quasi constant à l'intérieur de chaque bloc et bascule entre M4 et M3.
    """
    rec, old = values[:, :3], values[:, 3:]
    out[f"{name}_rec_mean"] = rec.mean(axis=1)
    out[f"{name}_old_mean"] = old.mean(axis=1)
    out[f"{name}_ratio"] = (rec.mean(axis=1) + FLOW_SMOOTHING) / (old.mean(axis=1) + FLOW_SMOOTHING)
    out[f"{name}_logratio"] = np.log1p(rec).mean(axis=1) - np.log1p(old).mean(axis=1)


def _monthly_array(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[_monthly_columns(lambda m: f"m{m}_{name}")].to_numpy(dtype=float)


def build_flow_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrégats des flux entrants / sortants et événements d'arrêt / apparition.

    Un type pris seul est bruité ; sommer entrées (dépôts + reçus + virements) et
    sorties (retraits + paybill + paiements marchands + envois) fait ressortir le
    signal (AUC univarié du ratio récent/ancien : 0.72, contre 0.66 pour le meilleur
    type isolé ; entrées sous 5% du niveau ancien -> 95% de stress).

    Les indicateurs d'arrêt / apparition captent ce que les ratios lissent : dépôts
    arrêtés -> 37% de stress, retraits qui apparaissent -> 44%.
    """
    out = {}
    inflow_value = sum(_monthly_array(df, f"{t}_total_value") for t in INFLOW_TYPES)
    outflow_value = sum(_monthly_array(df, f"{t}_total_value") for t in OUTFLOW_TYPES)
    balance = _monthly_array(df, "daily_avg_bal")

    _block_features(inflow_value, "inflow_value", out)
    _block_features(outflow_value, "outflow_value", out)
    _block_features(balance, "bal", out)

    net = inflow_value - outflow_value
    out["net_rec_sum"] = net[:, :3].sum(axis=1)
    out["net_old_sum"] = net[:, 3:].sum(axis=1)
    out["net_rec_over_bal_old"] = net[:, :3].sum(axis=1) / (balance[:, 3:].mean(axis=1) + FLOW_SMOOTHING)
    out["net_rec_over_inflow_old"] = net[:, :3].sum(axis=1) / (inflow_value[:, 3:].sum(axis=1) + FLOW_SMOOTHING)
    out["outflow_over_inflow_rec"] = (outflow_value[:, :3].sum(axis=1) + FLOW_SMOOTHING) / (
        inflow_value[:, :3].sum(axis=1) + FLOW_SMOOTHING
    )
    out["outflow_over_inflow_old"] = (outflow_value[:, 3:].sum(axis=1) + FLOW_SMOOTHING) / (
        inflow_value[:, 3:].sum(axis=1) + FLOW_SMOOTHING
    )

    n_stopped = np.zeros(len(df))
    n_started = np.zeros(len(df))
    for t in TRANSACTION_TYPES:
        value = _monthly_array(df, f"{t}_total_value")
        rec_zero = (value[:, :3] == 0).all(axis=1)
        old_zero = (value[:, 3:] == 0).all(axis=1)
        stopped = (rec_zero & ~old_zero).astype(float)
        started = (~rec_zero & old_zero).astype(float)
        out[f"{t}_stopped"] = stopped
        out[f"{t}_started"] = started
        out[f"{t}_rec_zero_months"] = (value[:, :3] == 0).sum(axis=1)
        out[f"{t}_old_zero_months"] = (value[:, 3:] == 0).sum(axis=1)
        n_stopped += stopped
        n_started += started
    out["n_types_stopped"] = n_stopped
    out["n_types_started"] = n_started
    out["inflow_types_stopped"] = sum(out[f"{t}_stopped"] for t in INFLOW_TYPES)
    out["outflow_types_started"] = sum(out[f"{t}_started"] for t in OUTFLOW_TYPES)
    return pd.DataFrame(out, index=df.index)


# Paybill et paiements marchands se comportent comme une seule catégorie "paiements" :
# une apparition de paybill quand les paiements marchands sont déjà actifs laisse le
# stress au taux de base (14%), contre 34% quand les deux apparaissent depuis rien
# (40% si ensemble). Ajouter retraits ou envois à ce groupe n'apporte rien (testé).
CATEGORY_GROUPS = {
    "payments": ["paybill", "merchantpay"],
}


def build_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """Comparaisons bloc récent / bloc ancien par catégorie regroupée, et ratios lissés.

    - Par catégorie (CATEGORY_GROUPS) : comparaisons de blocs (_block_features),
      apparition/arrêt, mois actifs par bloc.
    - Par type : ratio des moyennes de bloc lissé à 1 unité monétaire (FLOW_SMOOTHING),
      contre 1e-6 pour les ratios `_momentum_recent_vs_old` — avec un lissage de 1, une
      apparition vaut son montant récent, comparable à une hausse de même ampleur
      (AUC 0.82 pour le montant des retraits qui apparaissent).
    - Paiements et retraits rapportés aux entrées de chaque bloc, sur le même principe
      que `outflow_over_inflow_rec`, la feature la plus importante du modèle.
    - Arrêt simultané des dépôts et réceptions (79% de stress, contre 37% et 28% pour
      chacun pris seul).
    """
    out = {}
    for name, types in CATEGORY_GROUPS.items():
        values = sum(_monthly_array(df, f"{t}_total_value") for t in types)
        _block_features(values, f"{name}_value", out)
        rec_sum, old_sum = values[:, :3].sum(axis=1), values[:, 3:].sum(axis=1)
        out[f"{name}_started"] = ((old_sum == 0) & (rec_sum > 0)).astype(float)
        out[f"{name}_stopped"] = ((rec_sum == 0) & (old_sum > 0)).astype(float)
        out[f"{name}_rec_active_months"] = (values[:, :3] > 0).sum(axis=1)
        out[f"{name}_old_active_months"] = (values[:, 3:] > 0).sum(axis=1)

    for t in TRANSACTION_TYPES:
        values = _monthly_array(df, f"{t}_total_value")
        out[f"{t}_block_ratio"] = (values[:, :3].mean(axis=1) + FLOW_SMOOTHING) / (
            values[:, 3:].mean(axis=1) + FLOW_SMOOTHING
        )

    inflow = sum(_monthly_array(df, f"{t}_total_value") for t in INFLOW_TYPES)
    payments = sum(_monthly_array(df, f"{t}_total_value") for t in CATEGORY_GROUPS["payments"])
    withdraw = _monthly_array(df, "withdraw_total_value")
    for block, cols in [("rec", slice(0, 3)), ("old", slice(3, 6))]:
        inflow_sum = inflow[:, cols].sum(axis=1) + FLOW_SMOOTHING
        out[f"payments_over_inflow_{block}"] = (payments[:, cols].sum(axis=1) + FLOW_SMOOTHING) / inflow_sum
        out[f"withdraw_over_inflow_{block}"] = (withdraw[:, cols].sum(axis=1) + FLOW_SMOOTHING) / inflow_sum

    dep_rec = _monthly_array(df, "deposit_total_value") + _monthly_array(df, "received_total_value")
    out["deposit_received_stopped"] = (
        (dep_rec[:, :3].sum(axis=1) == 0) & (dep_rec[:, 3:].sum(axis=1) > 0)
    ).astype(float)
    return pd.DataFrame(out, index=df.index)


PASSTHROUGH_COLUMNS = [
    "arpu",
    "age",
    "x_90_d_activity_rate",
    "gender",
    "region",
    "smartphone",
    "segment",
    "earning_pattern",
]


GROUP_RELATIVE_COLS = [
    "daily_avg_bal_mean_6m",
    "outflow_to_inflow_value_ratio",
    "balance_to_arpu_ratio",
]


def fit_group_stats(features_df: pd.DataFrame, group_col: str = "segment", cols=None):
    """Calcule moyenne/écart-type par groupe (segment) sur le train UNIQUEMENT,
    comme un scaler : fit sur train, puis appliqué tel quel au test (apply_group_relative_features)
    pour ne jamais faire dépendre une feature de train de la distribution du test."""
    cols = cols or GROUP_RELATIVE_COLS
    return features_df.groupby(group_col)[cols].agg(["mean", "std"])


def apply_group_relative_features(
    features_df: pd.DataFrame, group_stats, group_col: str = "segment", cols=None
) -> pd.DataFrame:
    """Z-score de quelques features clés par rapport à leur groupe (segment) : un
    solde ou un ratio bas n'a pas le même sens pour un client HVC que pour un LVC."""
    cols = cols or GROUP_RELATIVE_COLS
    out = pd.DataFrame(index=features_df.index)
    for col in cols:
        means = features_df[group_col].map(group_stats[(col, "mean")])
        stds = features_df[group_col].map(group_stats[(col, "std")])
        z = (features_df[col] - means) / (stds.replace(0, np.nan) + EPS)
        out[f"{col}_z_by_segment"] = z.fillna(0.0)
    return out


def build_segment_cross_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Croisement segment x earning_pattern : les arbres ne retrouvent une
    interaction entre deux catégorielles que si la profondeur le permet. On la
    fournit directement pour ne pas dépendre de la profondeur choisie."""
    out = pd.DataFrame(index=df.index)
    out["segment_x_earning_pattern"] = (
        df["segment"].astype(str) + "_" + df["earning_pattern"].astype(str)
    )
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit le dataset de features par client à partir des données brutes Zindi."""
    blocks = [
        df[PASSTHROUGH_COLUMNS],
        build_transaction_features(df),
        build_balance_features(df),
        build_ratio_features(df),
        build_overall_activity_features(df),
        build_share_features(df),
        build_balance_to_income_features(df),
        build_segment_cross_feature(df),
        build_flow_aggregate_features(df),
        build_category_features(df),
    ]
    features = pd.concat(blocks, axis=1)
    features.index = df["ID"] if "ID" in df.columns else df.index
    return features

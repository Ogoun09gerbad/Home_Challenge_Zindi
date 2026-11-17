"""Per-customer feature engineering, from the monthly aggregates M1 (recent) to M6 (oldest).

The raw data covers 7 mobile money transaction types (volume, total
value, max amount, number of counterparties) plus the average daily balance, per month.
Only the total value and the balance carry signal: counterparties and max amount
are noise (random draws unrelated to the target), and volume adds nothing
over total value (experiments.md, 2026-10-10).

For each series of 6 monthly points: level, volatility, recent block
(M1-M3) / older block (M4-M6) comparison, ratios between flows, start and stop indicators.
"""

import numpy as np
import pandas as pd

MONTHS = [1, 2, 3, 4, 5, 6]

# Only metric used per transaction type (see module docstring).
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
    """Level, volatility and momentum (recent block vs older block) of a 6-month series.

    Only the 3-month block totals and the 6-month volatility carry signal
    (experiments.md, 2026-10-10): the trend, the last month alone, the M1/M2 ratio
    and the per-block min/max were removed with no loss (LightGBM 0.23146 without them
    vs 0.23155 with).
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
    """Level/volatility/momentum features for each (transaction type, metric)."""
    blocks = []
    for ttype, metrics in TRANSACTION_TYPES.items():
        for metric in metrics:
            cols = _monthly_columns(lambda m, t=ttype, me=metric: f"m{m}_{t}_{me}")
            blocks.append(_series_features(df, cols, f"{ttype}_{metric}"))
    return pd.concat(blocks, axis=1)


def build_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Level/volatility/momentum features for the average daily balance."""
    cols = _monthly_columns(lambda m: f"m{m}_daily_avg_bal")
    return _series_features(df, cols, "daily_avg_bal")


def _type_total_value_6m(df: pd.DataFrame, ttype: str) -> pd.Series:
    cols = _monthly_columns(lambda m, t=ttype: f"m{m}_{t}_total_value")
    return df[cols].sum(axis=1)


def build_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-type transaction ratios (outflows/inflows, withdrawals/deposits)."""
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
    """Level/volatility/momentum of activity across all types combined: a customer can
    switch from one type to another (paybill -> mm_send) without their overall activity
    changing, a signal invisible to per-type features."""
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
    """Share of each type in total outflows/inflows over 6 months: captures the
    composition of behaviour (cash vs bills), not just its level."""
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
    """Balance relative to income (ARPU): a low balance is not equally worrying depending
    on the customer's income, which neither variable captures on its own."""
    bal_cols = _monthly_columns(lambda m: f"m{m}_daily_avg_bal")
    bal_mean = df[bal_cols].mean(axis=1)
    out = pd.DataFrame(index=df.index)
    out["balance_to_arpu_ratio"] = bal_mean / (df["arpu"] + EPS)
    return out


FLOW_SMOOTHING = 1.0  # additive smoothing of ratios: prevents a near-zero month from blowing up a ratio


def _block_features(values: np.ndarray, name: str, out: dict) -> None:
    """Compares the recent block (M1-M3) to the older block (M4-M6) of a monthly series.

    The analysis (experiments.md, 2026-10-10) shows that the target is visible at the level
    of 3-month blocks: for stressed snapshots, the gap with non-stressed ones
    is nearly constant within each block and flips between M4 and M3.
    """
    rec, old = values[:, :3], values[:, 3:]
    out[f"{name}_rec_mean"] = rec.mean(axis=1)
    out[f"{name}_old_mean"] = old.mean(axis=1)
    out[f"{name}_ratio"] = (rec.mean(axis=1) + FLOW_SMOOTHING) / (old.mean(axis=1) + FLOW_SMOOTHING)
    out[f"{name}_logratio"] = np.log1p(rec).mean(axis=1) - np.log1p(old).mean(axis=1)


def _monthly_array(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[_monthly_columns(lambda m: f"m{m}_{name}")].to_numpy(dtype=float)


def build_flow_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inflow / outflow aggregates and stop / start events.

    A single type on its own is noisy; summing inflows (deposits + received + transfers) and
    outflows (withdrawals + paybill + merchant payments + sends) brings out the
    signal (univariate AUC of the recent/older ratio: 0.72, vs 0.66 for the best
    single type; inflows below 5% of the older level -> 95% stress).

    Stop / start indicators capture what ratios smooth out: deposits
    stopped -> 37% stress, withdrawals appearing -> 44%.
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


# Paybill and merchant payments behave as a single "payments" category:
# paybill appearing when merchant payments are already active leaves
# stress at the base rate (14%), vs 34% when payments appear from nothing
# (40% if both appear together). Adding withdrawals or sends to this group adds nothing (tested).
CATEGORY_GROUPS = {
    "payments": ["paybill", "merchantpay"],
}


def build_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recent block / older block comparisons per grouped category, and smoothed ratios.

    - Per category (CATEGORY_GROUPS): block comparisons (_block_features),
      start/stop, active months per block.
    - Per type: ratio of block means smoothed by 1 monetary unit (FLOW_SMOOTHING),
      vs 1e-6 for the `_momentum_recent_vs_old` ratios — with a smoothing of 1, an
      appearance is worth its recent amount, comparable to an increase of the same size
      (AUC 0.82 for the amount of withdrawals that appear).
    - Payments and withdrawals relative to the inflows of each block, on the same principle
      as `outflow_over_inflow_rec`, the most important feature of the model.
    - Simultaneous stop of deposits and receipts (79% stress, vs 37% and 28% for
      each on its own).
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
    """Computes mean/std per group (segment) on the train set ONLY,
    like a scaler: fit on train, then applied as-is to test (apply_group_relative_features)
    so that a train feature never depends on the test distribution."""
    cols = cols or GROUP_RELATIVE_COLS
    return features_df.groupby(group_col)[cols].agg(["mean", "std"])


def apply_group_relative_features(
    features_df: pd.DataFrame, group_stats, group_col: str = "segment", cols=None
) -> pd.DataFrame:
    """Z-score of a few key features relative to their group (segment): a low
    balance or ratio does not mean the same thing for an HVC customer as for an LVC one."""
    cols = cols or GROUP_RELATIVE_COLS
    out = pd.DataFrame(index=features_df.index)
    for col in cols:
        means = features_df[group_col].map(group_stats[(col, "mean")])
        stds = features_df[group_col].map(group_stats[(col, "std")])
        z = (features_df[col] - means) / (stds.replace(0, np.nan) + EPS)
        out[f"{col}_z_by_segment"] = z.fillna(0.0)
    return out


def build_segment_cross_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Segment x earning_pattern cross: trees only recover an
    interaction between two categoricals if the depth allows it. We
    provide it directly so as not to depend on the chosen depth."""
    out = pd.DataFrame(index=df.index)
    out["segment_x_earning_pattern"] = (
        df["segment"].astype(str) + "_" + df["earning_pattern"].astype(str)
    )
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds the per-customer feature dataset from the raw Zindi data."""
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

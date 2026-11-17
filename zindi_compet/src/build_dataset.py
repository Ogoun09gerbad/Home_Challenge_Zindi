"""Builds data/processed/{train,test}_features.csv from the raw data.

Group-relative features (z-score per segment) are computed like a
scaler: the statistics (mean/std per segment) are fitted on the
train set only, then applied as-is to the test set, so that a train feature never
depends on the test distribution.

Usage:
    python src/build_dataset.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import apply_group_relative_features, build_features, fit_group_stats

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TARGET = "liquidity_stress_next_30d"


def main():
    train = pd.read_csv(RAW_DIR / "Train.csv")
    test = pd.read_csv(RAW_DIR / "Test.csv")

    feats_train = build_features(train)
    feats_test = build_features(test)

    group_stats = fit_group_stats(feats_train)
    feats_train = pd.concat(
        [feats_train, apply_group_relative_features(feats_train, group_stats)], axis=1
    )
    feats_test = pd.concat(
        [feats_test, apply_group_relative_features(feats_test, group_stats)], axis=1
    )

    feats_train[TARGET] = train[TARGET].values

    assert feats_train.isna().sum().sum() == 0, "NaN detected in train features"
    assert feats_test.isna().sum().sum() == 0, "NaN detected in test features"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feats_train.to_csv(PROCESSED_DIR / "train_features.csv")
    feats_test.to_csv(PROCESSED_DIR / "test_features.csv")

    print(f"Train features: {feats_train.shape}")
    print(f"Test features:  {feats_test.shape}")


if __name__ == "__main__":
    main()

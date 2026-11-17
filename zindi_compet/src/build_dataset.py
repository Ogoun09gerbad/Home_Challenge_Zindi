"""Construit data/processed/{train,test}_features.csv à partir des données brutes.

Les features relatives au groupe (z-score par segment) sont calculées comme un
scaler : les statistiques (moyenne/écart-type par segment) sont ajustées sur le
train uniquement, puis appliquées telles quelles au test, pour ne jamais faire
dépendre une feature de train de la distribution du test.

Usage :
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

    assert feats_train.isna().sum().sum() == 0, "NaN détectés dans les features train"
    assert feats_test.isna().sum().sum() == 0, "NaN détectés dans les features test"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feats_train.to_csv(PROCESSED_DIR / "train_features.csv")
    feats_test.to_csv(PROCESSED_DIR / "test_features.csv")

    print(f"Features train : {feats_train.shape}")
    print(f"Features test  : {feats_test.shape}")


if __name__ == "__main__":
    main()

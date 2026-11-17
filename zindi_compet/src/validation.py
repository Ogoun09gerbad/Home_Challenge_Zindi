"""Validation strategy: K-fold stratified on the target.

Rationale (see experiments.md for the details of the checks):
- no date column in the data (each row is a customer snapshot aggregated
  over M1-M6, with no chronology between rows);
- no ID overlap between train and test (disjoint customer populations);
- train-vs-test adversarial validation: AUC ~0.50 (no distribution shift);
- univariate AUC of each column against the target: max ~0.57 (no leakage detected).

A temporal split would therefore have nothing to order and would provide no
additional guarantee against leakage. We use a standard stratified K-fold, which
preserves the target imbalance (85/15) in each fold.
"""

from sklearn.model_selection import StratifiedKFold

SEED = 42
N_SPLITS = 5


def get_folds(y, n_splits: int = N_SPLITS, seed: int = SEED):
    """Returns the (train_idx, val_idx) indices of a StratifiedKFold on y."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(y, y))

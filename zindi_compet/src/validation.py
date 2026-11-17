"""Stratégie de validation : K-fold stratifié sur la cible.

Justification (voir experiments.md pour le détail des vérifications) :
- pas de colonne de date dans les données (chaque ligne est un snapshot client agrégé
  sur M1-M6, sans chronologie entre les lignes) ;
- aucun chevauchement d'ID entre train et test (populations de clients disjointes) ;
- validation adversariale train-vs-test : AUC ~0.50 (pas de décalage de distribution) ;
- AUC univarié de chaque colonne contre la cible : max ~0.57 (pas de fuite détectée).

Un split temporel n'aurait donc rien à ordonner et n'apporterait aucune garantie
supplémentaire contre la fuite. On utilise un K-fold stratifié classique, qui
respecte le déséquilibre de la cible (85/15) dans chaque fold.
"""

from sklearn.model_selection import StratifiedKFold

SEED = 42
N_SPLITS = 5


def get_folds(y, n_splits: int = N_SPLITS, seed: int = SEED):
    """Retourne les indices (train_idx, val_idx) d'un StratifiedKFold sur y."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(y, y))

# Défi de prédiction du stress financier (Zindi, édition de septembre)

Prédire la probabilité qu'un client connaisse des difficultés financières dans les 30
prochains jours, à partir de 6 mois d'historique de transactions de monnaie mobile.

- **Métrique** : `score = 0.4 × AUC + 0.6 × (1 − logloss / 0.595)` (formule retrouvée
  à partir de deux scores publics, voir `experiments.md`)
- **Sortie attendue** : probabilités brutes (pas de labels, pas de seuillage)
- **Date limite** : 2026-10-25
- **Contraintes** : outils open source uniquement, pas d'AutoML, seed fixée, max
  10 soumissions/jour et 200 au total.

## Reproduire la soumission

Python 3.14, versions figées dans `requirements.txt`.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -X utf8 src\download_data.py   # ou déposer Train.csv / Test.csv à la main dans data\raw\
python src\build_dataset.py   # features -> data\processed\{train,test}_features.csv
python src\predict.py         # modèles + blend -> submissions\submission_<horodatage>.csv
```

`predict.py` dure une vingtaine de minutes. Tous les hyperparamètres sont
figés dans `src/train.py` (`FINAL_LGBM_PARAMS`, `FINAL_GAM_PARAMS`, `FINAL_CATBOOST_PARAMS`) et la seed
vaut 42 partout : relancer le pipeline redonne la même soumission.

Pour télécharger les données via l'API, copier `.env.example` en `.env` et renseigner
`ZINDI_USERNAME` / `ZINDI_PASSWORD`.

## Approche

1. **Analyse** : 10 000 clients ayant chacun 7 instantanés (4 en train, 3 en test),
   sans chevauchement temporel entre instantanés. La cible se lit dans la rupture
   entre le bloc récent (M1-M3) et le bloc ancien (M4-M6).
2. **Rétro-ingénierie du générateur** : les données sont synthétiques et stationnaires ;
   la cible est calculée à partir des features (score + bruit, 15% des lignes de chaque
   position étiquetées). Les contreparties, montants max et volumes sont du bruit. Le
   score est une **somme** de quelques facteurs : ratio sorties/entrées récent, baisse
   des entrées, baisse du solde, apparition de retraits, apparition de paiements
   (paybill + paiements marchands, qui forment une seule catégorie), volatilité du
   solde, taux d'activité. Le bruit est **gaussien** : P(stress | x) = Φ(score), les
   modèles sont donc additifs sur l'échelle probit plutôt que logit.
3. **Features** (`src/features.py`, 137 au total, calculées sur les montants et le
   solde uniquement) : niveau, volatilité et momentum par série ; comparaisons bloc
   récent / bloc ancien ; agrégats des entrées et des sorties ; catégorie paiements ;
   indicateurs d'arrêt ou d'apparition ; z-scores par segment.
4. **Validation** : K-fold stratifié. Il n'y a pas de date, la validation
   adversariale train/test donne une AUC de 0.50, et aucune colonne n'est suspecte
   de fuite (ni l'ID, ni l'ordre des lignes).
5. **Modèles** (`src/predict.py`) : trois modèles **additifs** (une fonction par
   feature, sommées) sur les 15 features les mieux classées — LightGBM probit à arbres
   d'une division avec contraintes de monotonie, GAM probit à P-splines avec les mêmes
   contraintes (`src/gam.py`), et CatBoost probit à arbres d'une division — 10 plis × 3 seeds,
   puis blend sur l'échelle probit avec des poids choisis sur les prédictions OOF.

Le raisonnement complet, expérience par expérience, est dans `experiments.md`. Le
notebook `notebooks/02_full_pipeline.ipynb` reproduit le pipeline de façon autonome
(Google Colab).

## Structure du repo

```
.
├── data/
│   ├── raw/                # fichiers originaux de Zindi, jamais modifiés
│   └── processed/          # features et prédictions générées par les scripts
├── notebooks/
│   ├── 01_baseline_starter.ipynb  # reproduction du starter Zindi (point de départ)
│   └── 02_full_pipeline.ipynb     # pipeline complet autonome (Colab)
├── src/
│   ├── download_data.py    # téléchargement des données via l'API Zindi
│   ├── features.py         # feature engineering
│   ├── build_dataset.py    # construit les jeux de features train / test
│   ├── validation.py       # stratégie de validation (K-fold stratifié)
│   ├── train.py            # modèles, hyperparamètres finaux, blend
│   ├── gam.py              # GAM probit à P-splines (second modèle du blend)
│   └── predict.py          # pipeline final et fichier de soumission
├── submissions/            # fichiers de soumission horodatés
├── experiments.md          # journal des expériences
└── requirements.txt
```

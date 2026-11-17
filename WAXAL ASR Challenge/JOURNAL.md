# JOURNAL

## 2026-10-26
- Env : venv `.venv` (Python 3.12.10, uv), torch 2.11+cu128 (CUDA OK, RTX 4070 Laptop 8 Go), transformers 5.20, datasets 3.6, numpy 2.5. requirements.txt écrit.
- kenlm : build impossible sous Windows (pas de MSVC). LM n-gramme à faire sous WSL/Linux ou après installation des Build Tools VS.
- WaxalNLP : schéma parquet = id, speaker_id, transcription, language, gender, audio{bytes,path (mp3)}. Splits train/validation/test (+ unlabeled ~78 Go, exclu). Langues : lin, sna, lug.
- Téléchargement `prepare_data.py` lancé en arrière-plan (~12,6 Go labellisés, lent : ~1 h), log dans logs/prepare_data.log.
- `common.py` : seeds, normalisation, métrique 0.5*WER+0.5*CER, chargement parquet + décodage audio — testé sur lin-test.
- BLOQUANT : ./zindi/ vide (Train/Test/Test_Phase2/SampleSubmission + starter notebook). Format de soumission et affichage du score (brut ou 1-erreur) non vérifiés.
- ATTENTION règle Zindi : le split `test` de WaxalNLP a des transcriptions publiques ; ne pas l'utiliser pour entraîner (s'il correspond au test Phase 1). À confirmer avec Test.csv.
- Info challenge (2026-10-26) : Phase 1 test = split `test` de WaxalNLP (transcriptions publiques => INTERDIT pour entraîner/tuner). Entraînement sur `train`, suivi sur `validation`. Phase 2 = audio seul (newaudios.zip), sans langue/locuteur => LID ou modèle multilingue unique. Clôture 2026-11-01 (≈6 jours).

### 2026-10-26 (après-midi) — recherche + premières mesures
- Deep research : reports/Stratégie ASR WAXAL lin sna lug.md. Points clés : score = 1 - 0.5*(WER+CER) (plus haut = mieux) ; WER sur minuscules AVEC ponctuation, CER sur texte brut => sortie avec casse+ponctuation (vocab "raw"). Phase 2 originale : lin + sna seulement.
- Décision utilisateur : partir du checkpoint public anyantudre/waxal-w2vbert-linsnalug-raw (Apache-2.0). Sa fiche : validation WAXAL retirée de l'entraînement => notre validation est propre pour lui.
- Conformité : transcriptions du split test retirées du cache et de load_split (jamais exposées au code).
- kenlm compilé sous Windows (MSVC 2026, DISTUTILS_USE_SDK=1) ; ARPA produits par build_lm.py (KN interpolé, Python pur, normalisation vérifiée : somme des probas = 1).
- Phase 2 : 1785 wav (data/phase2/newaudios/ID_XXXXXX.wav).

| Exp | Modèle | Décodage | lin score | sna score | lug score |
|---|---|---|---|---|---|
| E1 | anyraw (linsnalug-raw) | glouton | 0.7421 (WER .364 CER .152) | 0.8369 (WER .258 CER .068) | 0.8974 |
| E2 | anyraw | beam 48 sans LM | 0.7446 | 0.8382 | — |
| E3 | anyraw | beam + LM 4-gram train (meilleur alpha) | 0.7435 (α .2) ↓ | ≤0.8382 (LM nuit) | 0.9039 (α .2) |
- Conclusion : le LM n-gramme sur le texte du train n'aide pas (lin/sna) — le modèle a déjà absorbé ce texte. Abandonné sauf lug.
- LID textuelle (NB n-grammes de caractères sur l'hypothèse) : 100 % sur 900 clips val (lin/sna/lug).
- Prochaine étape : ensemble MBR de checkpoints publics Apache-2.0 du même auteur (seeds/soups/specialists), évalués sur val lin+sna.

### 2026-10-26 (soir) — ensemble MBR + première soumission Phase 2
- 10 checkpoints publics Apache-2.0 (anyantudre/waxal-w2vbert-*) évalués sur validation lin+sna (300 clips/langue) :
  meilleurs seuls : lin-specialist 0.8025, soup6 0.8003, soup5 0.8000, distilled 0.7988, afrivoicemix 0.7914.
- Ensemble MBR (sélection par minimisation du risque 0.5*WER+0.5*CER pairwise) sur les 6 meilleurs
  (lin-specialist, afrivoicemix, distilled, soup5, soup6, sna-specialist) : **0.8039** sur validation (lin 0.7650, sna 0.8425).
- LM n-gramme (build_lm.py, KN order 4) : aide seulement lug (0.8974 -> 0.9039, alpha=0.2) ; nuit en lin/sna (déjà vu par les checkpoints publics). Non utilisé lin/sna.
- Inférence Phase 2 (892 clips, 5h, data/phase2/newaudios) par les 6 membres (~8-15 min/membre selon téléchargement), ensemble MBR -> logs/ens_phase2.csv.
- BLOQUANT persistant : zindi/ toujours vide. submissions/phase2_ens_top6mbr.csv produit au format par défaut (id, transcription) — FORMAT NON VÉRIFIÉ contre SampleSubmission.csv. Pas encore soumis sur Zindi (score public inconnu).
- À faire : vérifier colonnes exactes attendues par Zindi (ID vs id, Target vs transcription) avant toute soumission réelle ; LID textuelle + LM lug sur Phase 2 si des clips lug s'y trouvent (rapport dit lin+sna seulement pour l'édition originale, à reconfirmer).

### 2026-10-26 (soir, suite) — correction format de soumission
- Premier envoi réel refusé par Zindi : "Submission is missing the following columns: ID, Target" -> le format exact est confirmé : colonnes `ID` et `Target` (majuscules), pas `id`/`transcription`.
- Corrigé dans make_submission.py et predict.py (fallback sans SampleSubmission.csv). Soumission régénérée : submissions/phase2_ens_top6mbr.csv (892 lignes, colonnes ID,Target, 0 vide).
- ATTENTION métrique : la page Zindi dit seulement "le score final est la moyenne pondérée de WER et CER" (0.5/0.5), sans préciser si c'est l'erreur brute (plus bas = mieux) ou 1-erreur (plus haut = mieux, comme supposé jusqu'ici d'après la fiche d'un participant). Reste à confirmer empiriquement avec le score public retourné par ce dépôt. Si c'est l'erreur brute, la cible "~0.77" donnée au départ serait un très mauvais score (77% d'erreur) — peu plausible — donc l'hypothèse "1-erreur" reste la plus probable, à vérifier dès que le score public tombe.

### 2026-10-26 (nuit) — calibrage ensemble (rang 7, 0.7469 public)
- Score public soumission 1 : 0.746912 => score = 1 − 0.5(WER+CER) confirmé. #4 = 0.7601.
- Moyenne des log-probs par trame (6 membres, vocab identique 121 tokens) : NUIT (0.7919 au mieux, 0.728 sans pénalité de blank) — pics CTC désalignés entre fine-tunes indépendants (cf. Kurata 2019). Abandonné.
- Pénalité de blank par membre (grid −0.5..1) : gains ≤ +0.003 sauf afrivoicemix sna (+0.024 à 0.75). Utilisée comme générateur de candidats (re-décodages) pour le MBR.
- MBR (val lin+sna, 300/langue) : base6 0.8039 ; 12 cand. 0.8041 ; 18 cand. 0.8019 ; routé LID 11 cand. 0.8057 ; **routé LID petit (E) 0.8093** (lin 0.7761).
- LID textuelle sur l'hypothèse : 99 % sur val ; Phase 2 : 447 lin / 445 sna.
- Post-traitement (majuscule initiale, point final, suppression des virgules) : effet nul (±0.0005) — les modèles émettent déjà casse/points.
- Soumission 2 prête : submissions/phase2_mbrE_routed.csv.

### 2026-10-27 — CONFORMITÉ : membres contaminés exclus
- Fiches HF lues (logs/provenance*.txt) : `linsna-distilled` = entraîné sur les 892 clips Phase 2 (pseudo-transcriptions d'ensemble) ; `linsna-soup6` = moyenne incluant distilled et table de données listant le split test WaxalNLP (3 615 lignes, labels Phase 1 interdits) ; seeds 43/44 (et 46) entraînés sur le split test Phase 1 (gold). => EXCLUS définitivement : distilled, soup6, seed43/44/46, whisper-turbo, tous yehoshua01/*.
- Membres propres (train+val WAXAL hors holdout/val, AfriVoice, pseudo-labels sur le pool *unlabeled*) : lin-specialist, sna-specialist, afrivoicemix, soup5, linsnalug-raw.
- IMPACT : la soumission 1 (0.746912, rang 7) et la config E utilisaient distilled/soup6 => à NE PAS retenir comme soumission finale ; E non soumise.
- Ensemble propre F (lin : lin-spec bp0/.5 + afrivoicemix bp0/.5 + soup5 ; sna : soup5 + sna-spec bp0/.5 + afrivoicemix bp1 + lin-spec) : val 0.8107 (lin 0.7762, sna 0.8451) ; base 5 membres sans variantes : 0.8039.

### 2026-10-27 — pistes propres (membres externes, post-traitement, médiane)
- Soumission propre prête : submissions/phase2_clean_F.csv (val 0.8107).
- Boucle-collapse (anyantudre) : 2 clips val / 3 clips Phase 2 modifiés, effet nul (-0.0001) => non utilisé (chez eux utile car membres blank-penalty 2.0 bouclent).
- Frontières de mots par fréquences train (wordfix.py, split/merge, ratios 2-10) : effet nul (±0.0003) ; les règles Pasketti ont leur sens choisi par sondage du leaderboard => à tester seulement par A/B réels si budget de soumissions.
- Membre externe propre `keystats/w2v-bert-2.0-multilingual-waxal-shona-lingala-luganda-best` (WAXAL train seul, Apache-2.0) : val lin 0.7333 / sna 0.8304 seul ; ajouté au MBR F : 0.8105-0.8109 (aucun gain).
- Médiane de chaînes niveau caractère (median_ens.py, façon Pasketti `median_extended`) : 0.809 (F) / 0.8007 (base5) < MBR (0.8107 / 0.8039). Membres trop corrélés.
- whisper-large-v3 LoRA (mrwalker123) : fiche dit train+validation WAXAL => validation non fiable pour lui ; non évalué.
- omniASR (DariusTheGeek) : nécessite fairseq2 (WSL2, pas de sudo ici) ; non tenté.
- En cours : TTA vitesse 0.9/1.1 sur soup5 et lin-specialist (run_tta.sh) ; keystats Phase 2.
- TTA vitesse (0.9/1.1) soup5 seul : 0.7972 / 0.7958 (base 0.8000) ; ajouté au MBR F : 0.8098 < 0.8107 => aucun gain, abandonné (runs lin-specialist arrêtés). keystats Phase 2 terminé mais non utilisé (pas de gain en val).
- Bilan : l'ensemble de membres anyantudre propres est saturé autour de 0.810 en val. Meilleure soumission propre : submissions/phase2_clean_F.csv.
- Whisper large-v3 + LoRA swa (mrwalker123, beam 3, langue via LID texte) : val lin 0.6596 / sna 0.7616 / ALL 0.7122 (val GONFLÉE : il a vu train+val) — nettement sous les CTC (0.776 / 0.845). Ajouté au MBR F : 0.8095 (poids 1) / 0.8102 (poids 0.5) vs 0.8107 => aucun gain, même avec une val optimiste. Inférence Phase 2 (≈50 min) NON lancée.
- Leçon : les familles externes testées (keystats, Whisper) sont trop faibles pour corriger les erreurs des membres anyantudre ; le MBR ne les choisit que sur ~3 % des clips.

### 2026-10-27 — score public de F, test des délétions shona
- phase2_clean_F.csv : **0.751052** public (CER 0.1135, WER 0.3844) vs val 0.8107 (écart −0.060). Rang ≈ 7 (#6 = 0.7526, #4 = 0.7601).
- Débits (mots/s) : réf train lin 1.527 / sna 1.152 ; val réf lin 1.469 / sna 1.145 ; sorties à bp=0.5 collent à la réf en val. Phase 2 avec bp=0 : lin 1.47-1.50 (OK), sna 1.09-1.10 (< 1.15) => délétions probables en shona sur nouveaux locuteurs.
- Candidats A/B pour le leaderboard : N (sna bp 1/1.5, rate 1.125) val 0.8071 ; M (sna bp 1.5/2, rate 1.132) val 0.8051. Si le public monte au-dessus de 0.7511 => le déficit vient bien des délétions et la val est trompeuse.
- Test leaderboard (fichier N ou M, à confirmer par l'utilisateur) : 0.752102 (CER 0.1139, WER 0.3819) vs F 0.751052 : WER −0.0025, CER +0.0004 => +0.001, dans le bruit (±0.003).
- Moyenne uniforme de poids de 5 checkpoints propres (make_soup.py -> models/soupC : raw, lin-spec, sna-spec, afrivoicemix, soup5) : val lin 0.7615 / sna 0.8432 / ALL 0.8024 (≈ meilleur membre seul). Dans le MBR F : 0.8094 (poids 1), 0.8072 (poids 2) < 0.8107 => aucun gain. Phase 2 non lancée.

### 2026-10-27 — fine-tuning LoRA propre (ft1)
- Base soup5 + LoRA r32 (22.7 M params entraînables), lin+sna WAXAL train seul (27 619 clips, 143.7 h dispo ; ~3 500 clips vus), augmentation vitesse/gain/bruit + SpecAugment, lr 1e-4, 220 pas (eff. batch 16, ~12 s/pas, ≈45 min). Bug corrigé en route : tête CTC à 123 sorties (2 tokens ajoutés) vs vocab.json à 121 -> réinitialisation aléatoire ; on reprend désormais le processeur du checkpoint.
- Val (300/langue) : lin 0.7571 / sna 0.8409 / ALL 0.7991 vs soup5 0.7573 / 0.8425 / 0.8000 => aucun gain. Critère de passage (ALL ≥ 0.802) non atteint ; Phase 2 non lancée.
- Limite : la val (locuteurs déjà vus en train) ne mesure pas le gain de robustesse visé par l'augmentation ; seul un envoi leaderboard pourrait le montrer.
- ft1 Phase 2 (526 s, RTF 0.029) ; variantes bp 0-2 générées (make_variants inclut m_ft1). MBR R (F + 5 candidats ft1) val 0.8073 (dilution) ; Q (F + ft1 greedy) val 0.8093 => soumission submissions/phase2_clean_FT.csv (config Q).
- Score public phase2_clean_FT.csv : **0.749777** (CER 0.1136, WER 0.3869) < F 0.751052 < fichier sna-bp (N ou M) 0.752102. Le fine-tuning LoRA court n'améliore pas la robustesse aux nouveaux locuteurs ; piste abandonnée.
- Classement des soumissions propres au public : sna-bp (0.7521) > F (0.7511) > FT (0.7498). Écart au rang 4 (0.7601) : ≈ 0.008.

## 2026-10-30
- Deep research : reports/ASR low-resource WAXAL techniques.md. Leviers restants identifiés : ROVER caractère à ancre (anyantudre) et règles de frontières de mots (Pasketti). KenLM/LID/pseudo-labels/TTA/soupes déjà fermés.
- `rover.py` (ROVER caractère à ancre, réimplémentation de l'algorithme MIT d'anyantudre + option marge). Validation (lin+sna, 300/langue) :
  - 5 membres propres bp0, ancre soup5 : 0.8008 ; ×{0,0.5} : 0.8020 ; ×{0,1,1.5,2} (20 membres, seuil 2) : 0.7902.
  - Ancre = sortie F, marge 3, membres bp0 : 0.8108 (= F 0.8107) ; + membres bp 1–2 : 0.7939–0.8047.
  - Routé LID (ancres lin-spec / soup5) : ≤ 0.8071.
  - => le vote ne bat pas le MBR F en validation : membres trop corrélés (même lignée linsnalug-raw). Critère du rapport (≥ 0.8137) non atteint.
  - Variante shona seule (ancre F + 16 membres bp 0–2, marge 4) : val −0.0021 (sna −0.0045) ; Phase 2 sna +1.9 % de mots (1.10 → ≈1.12 mot/s). Test A/B possible seulement au LB.
- `pasketti_rules.py` (règles MIT, étapes confirmed+baverb, sans tables corpus/pool) : lin namoni/nazomona/tozomona → split, ba+{zali,lati,komi,fandi,vandi,sali,telemi,lakisi} → join ; sna `^(\w*ri)(ku\w{3,})$` → split, 8 splits ne-. Val : 0.8107 → 0.8012 (attendu : la val suit le train, où namoni est soudé à 73 %). Déclenchements Phase 2 : namoni 223, nazomona 128, tozomona 17, ba+verbe 1, sna progressif 153, ne- 15 (Pasketti : 208/126/118 sur le même test => même audio).
- Point final : F reproduit déjà le taux du train (lin 67 %, sna 95–100 %) ; cap_after_stop neutre en val (−0.00003). Non ajoutés.
- Fichiers prêts : submissions/phase2_clean_F_pklin.csv (216 lignes ≠ F), phase2_clean_F_pk.csv (330), phase2_clean_F_snarover_pk.csv (504).
- **phase2_clean_F_pk.csv : 0.760535 public (rang 5)**, +0.0095 vs F. Cible rang 3 : 0.766456 (≈ +0.006). La val est inversée pour l'orthographe : seul le LB juge ces règles.
- Nouvelles sondes (chacune isolée vs F_pk) : `--final-stop` (146 lignes), `--ba-noun` (ba+nom, liste manuelle : racine fréquente seule dans le train ; verbes exclus car le test soude ba+verbe), `--bazo-split --na-split` (bazo+verbe depuis la table train, analogue à nazo/tozo mona). Val (aveugle) : 0.8011 / 0.8002 / 0.7991 vs 0.8012.
- Écarté : coupure shona -ri + mot non-ku (surtout rwegoridhe = concordant+couleur, −0.0085 chez Pasketti ; et vrais mots comme chirimwa).

## 2026-10-31 — résultat final et revue de code Zindi
- Classement privé : **rang 6** avec phase2_clean_F_pk_banoun.csv (public 0.763719, privé 0.771129). Code + doc à téléverser avant le 2026-11-02 00h15 GMT.
- Préparation de la revue (articles Zindi « documentation » et « code review ») : `reproduce.sh` (pipeline complet, 4 membres propres à commits figés), `requirements.txt` à versions figées (pipeline seul) + `requirements-experiments.txt`, LID textuelle déplacée dans `lid.py` (plus de dépendance kenlm/pyctcdecode), imports en tête et docstrings anglaises dans les scripts du pipeline, `prepare_data.py --splits`, `postproc.py --mode cap`, README anglais en 13 sections (structure de l'article Zindi), DATASETS.md en anglais (sources AfriVoice des checkpoints, déclaration sur les règles Pasketti).
- Vérification : ré-exécution complète de `reproduce.sh` (1425 s) → logits des 4 modèles identiques bit à bit (892/892), 15 fichiers intermédiaires identiques, CSV final identique à l'octet près.
- Archive : OGB2000_WAXAL_solution.zip.

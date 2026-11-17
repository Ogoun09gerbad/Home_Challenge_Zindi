# Décodage LM + post-traitement pour CTC low-resource (Lingala / Shona / Luganda) — WAXAL Zindi

Score cible : `1 - 0.5*(WER + CER)`, WER en minuscules avec ponctuation conservée, CER sur chaîne brute. Contrainte : GPU 8GB, quelques jours, pas de réentraînement acoustique lourd. Toutes les techniques ci-dessous sont **inference/post-processing only**.

## 0. Ce qui existe déjà pour ce challenge
Il existe un dataset HF **`keystats/waxal-kenlm-models`** contenant des modèles KenLM 5-gram déjà construits pour Lingala, Shona et Luganda, pensés pour le shallow fusion avec `pyctcdecode`, avec les splits validation/test WaxalNLP explicitement exclus du corpus d'entraînement (pour éviter toute fuite). À vérifier/réutiliser en premier lieu avant de reconstruire son propre LM — cela peut faire gagner un jour entier.

## 1. Décodage beam-search CTC + LM n-gram (KenLM)

### Pipeline standard (pyctcdecode)
1. Récupérer le vocabulaire/alphabet du tokenizer CTC du modèle wav2vec2/w2v-BERT (`processor.tokenizer.get_vocab()`).
2. Construire un LM KenLM :
   - `lmplz --discount_fallback -o 5 -S 4G < corpus.lower.txt > lm.arpa` (ordre 3 à 5 ; 5-gram si le corpus est assez gros, sinon 3-gram pour éviter l'overfit sur un petit corpus africain).
   - `build_binary lm.arpa lm.binary` (binaire = chargement rapide, important avec peu de RAM).
3. `pyctcdecode.build_ctcdecoder(labels, kenlm_model_path="lm.binary", alpha=0.5, beta=1.0)` — `alpha` = poids LM, `beta` = bonus longueur mot. **Tuner alpha/beta sur un petit dev set** (grid search grossier, ex. alpha∈{0.2,0.5,0.8,1.2}, beta∈{0,0.5,1.5}) car l'optimum dépend fortement de la taille/qualité du LM.
4. Décoder les logits CTC (pas besoin de réentraîner le modèle acoustique) : `decoder.decode(logits, beam_width=100-500)`.
5. Alternative : `flashlight-text` (bindings Python du décodeur flashlight/wav2letter) donne un décodage CTC+KenLM plus rapide en C++ si pyctcdecode est trop lent en CPU — API similaire (`LexiconFreeDecoder` ou `LexiconDecoder` + `KenLM`).

### Cas LM pauvre en données (langues africaines)
- Avec peu de texte cible, préférer un **LM d'ordre bas (3-gram)** avec Kneser-Ney + `--discount_fallback` (évite les erreurs KenLM quand les comptes de n-grammes rares sont insuffisants).
- Interpoler/mélanger plusieurs sources (voir §2) en pondérant par similarité de domaine (corpus WaxalNLP train >> corpus web générique).
- Un petit LM mal réglé peut **dégrader** le CER (le LM biaise vers des mots fréquents et peut casser des mots rares/propres) : toujours comparer contre le greedy CTC baseline sur le dev set, feature par feature (alpha=0 doit retrouver le greedy).
- Le score final mélange WER (texte normalisé minuscule+ponctuation) et CER (brut) : un LM trop agressif qui "corrige" la ponctuation ou la casse peut améliorer WER mais nuire au CER si le texte prédit s'éloigne trop en caractères de la référence — à valider empiriquement, pas supposé.

### Rescoring alternatif (sans intégrer le LM dans le beam search)
- Générer N-best hypothèses CTC (greedy + quelques beams) puis rescorer avec un LM externe (KenLM ou même un petit modèle causal de type GPT-2 africain/mBERT fine-tuné) — moins risqué que le shallow fusion intégral, et permet de combiner facilement plusieurs signaux (longueur, LM, score acoustique) via une petite régression logistique entraînée sur le dev set.

## 2. Sources de texte pour les LM n-gram : Lingala / Shona / Luganda

| Source | Couverture connue | Notes |
|---|---|---|
| **CC-100 (Facebook/StatMT)** | Luganda (`lg`, ~7.3M tokens), Lingala (`ln`, ~2.3M tokens) | Shona non confirmé dans cette release ; construit depuis snapshots CommonCrawl 2018 via CC-Net. |
| **MADLAD-400** | Shona (`sn`), Luganda (`lg`), Lingala (`ln`) tous présents dans le mapping de langues | Dataset CommonCrawl audité manuellement, 3T tokens, 419 langues ; filtrer/dédupliquer avant usage car bruité pour les langues rares. |
| **Wikipedia (dumps + wikilangs)** | Lingala a un wiki actif (modèles "wikilangs/ln" dérivés des dumps mensuels) ; Shona et Luganda ont des Wikipédias plus petites mais exploitables (dumps officiels wikipedia.org) | Utile comme complément propre et bien formaté, mais volume faible seul. |
| **OPUS** | Non confirmé pour ces 3 langues dans les recherches effectuées — à vérifier directement sur opus.nlpl.eu (filtrer par code ISO `lin`/`sna`/`lug`) ; JW300/bible-uedin sur OPUS couvrent historiquement de nombreuses langues africaines y compris potentiellement ces trois | **Action concrète : aller sur opus.nlpl.eu et chercher les corpus JW300, bible-uedin, Tatoeba, CC-Aligned pour ces codes langues avant de les écarter.** |
| **Corpus bibliques / JW.org** | Souvent disponibles langue par langue (traductions complètes de la Bible et de littérature JW existent en Shona, Lingala, Luganda) | Texte religieux au style formel, différent du style conversationnel ASR mais utile pour la couverture lexicale et la morphologie ; mélanger avec des sources plus "orales" si possible. |
| **Corpus Makerere / Sunbird AI / SALT (Luganda)** | SALT = corpus parallèle MT pour 5 langues ougandaises dont Luganda ; `sagepond/lug_web` (HF) = corpus Luganda pour LM (oral literature, social media, news, religieux) ; papier "Building Text and Speech Benchmark Datasets... East African Languages" (Applied AI Letters 2024, équipe Makerere) décrit aussi des données ASR/texte Luganda | Le plus riche en ressources pour Luganda spécifiquement — prioriser. |
| **Corpus académiques Shona / Lingala spécifiques** | Non localisés précisément dans cette recherche (à creuser : universités zimbabwéennes pour Shona, RDC/Congo pour Lingala, projets NLP africains comme Masakhane) | **Masakhane** (collectif NLP africain) est une piste forte à explorer pour des ressources Shona/Lingala organisées par la communauté. |
| **Texte brut WaxalNLP (train split)** | Déjà dans le dataset challenge | **Priorité n°1** : c'est le texte le plus proche du domaine/style cible (déjà utilisé pour construire `keystats/waxal-kenlm-models`). |

**Stratégie recommandée** : LM principal = texte WaxalNLP train (in-domain) interpolé/augmenté avec CC-100 + MADLAD-400 (filtré pour qualité, dédupliqué) + Wikipedia + Bible/JW pour la couverture lexicale, en pondérant fortement le corpus in-domain (ex. upsampling x5-10 du texte WaxalNLP dans le corpus final avant `lmplz`, ou interpolation de deux LM séparés via les outils KenLM d'interpolation).

## 3. Normalisation / post-traitement texte

Objectif : réduire WER (texte normalisé) et CER (texte brut) sans introduire d'erreurs.

- **Casse** : le WER est calculé en minuscules → lowercasing systématique en sortie n'a aucun coût pour le WER mais attention : le CER est calculé sur la chaîne brute, donc si la référence a de la casse native, ne PAS lowercaser la sortie finale soumise sans vérifier le format attendu exact de la soumission (le prompt dit "WER sur texte en minuscules" = c'est la métrique qui lowercase, pas forcément vous — vérifier le script d'évaluation officiel Zindi avant de modifier la casse de la sortie).
- **Ponctuation** : conservée pour le WER → un modèle CTC classique ne produit souvent pas de ponctuation. Options sans réentraînement :
  - Modèle de **restauration de ponctuation** léger pré-entraîné (ex. modèles multilingues `punctuation_restoration` sur HF, ou un modèle de ponctuation basé sur mBERT) appliqué en post-traitement sur la sortie ASR.
  - Règles heuristiques simples (point final en fin de phrase, pas de virgule hallucinée) — risque de dégrader le WER si mal calibré ; à valider sur dev set uniquement, jamais supposé.
- **Nombres** : normalisation inverse text (chiffres → mots ou l'inverse) selon ce que la référence utilise — vérifier empiriquement le format des références WaxalNLP (chiffres en toutes lettres ou numériques) et aligner la sortie CTC en conséquence.
- **Répétitions / hallucinations CTC** : les CTC peuvent produire des répétitions de mots/syllabes en audio bruité — un post-filtre simple de déduplication de n-grammes consécutifs identiques (fenêtre glissante) peut réduire le CER sans risque si appliqué prudemment (ne supprimer que les répétitions exactes consécutives suspectes, pas les répétitions légitimes de la langue).
- **Mots composés / segmentation** : les langues bantoues (Shona, Luganda) sont fortement agglutinantes — attention à la segmentation en mots du LM, un LM mal segmenté peut forcer des découpages de mots incorrects. Vérifier la cohérence de segmentation entre le LM et la référence annotée WaxalNLP.
- **Spécificités orthographiques par langue (à valider contre les références réelles, pas de règle universelle supposée)** :
  - **Luganda** : voyelles longues souvent notées par doublement de la voyelle (ex. *-aa-*) ; tons (3 tons) généralement non marqués en orthographe standard — ne pas essayer de restaurer de tons non présents dans les références.
  - **Shona** : orthographe construite sur le principe "un son, un symbole" (Doke), révisée en 1955/1967 ; problèmes historiques documentés de découpage des mots (word division) — source potentielle de variance WER si le decoder segmente différemment de la convention de référence.
  - **Lingala** : 2 tons (haut/bas), souvent non marqués en texte écrit courant ; accents/diacritiques (ex. voyelles ouvertes ɔ/ɛ) fréquemment omis dans le texte web informel (ex. "likɔngá" → "likonga") — si le LM est entraîné sur du texte web non accentué alors que les références WaxalNLP utilisent les diacritiques (ou l'inverse), il y a un risque de mismatch systématique à vérifier en premier sur un échantillon du dev set avant tout gros travail de LM.

**Recommandation pratique** : avant toute règle de normalisation, extraire 50-100 exemples du dev set, comparer précisément la sortie brute CTC vs référence caractère par caractère, et ne coder que les corrections qui répondent à des erreurs réellement observées (éviter les règles "théoriques" non vérifiées).

## 4. Identification de langue (LID) pour routage audio → modèle/décodeur

- **MMS-LID (Meta)** : checkpoints publics `facebook/mms-lid-256` (et variantes 126/1024 langues), fine-tunés depuis `mms-1b`. Couvre Shona et Lingala avec de bons scores (MMS-LID-1024 ≈ 96 F1 macro sur Lingala, ≈ 87-91 sur Shona selon le test set, d'après le papier "Voice of a Continent", arXiv 2505.18436). Luganda non confirmé dans les tables trouvées — à tester directement avec le checkpoint 1024-langues sur un échantillon Luganda avant de l'écarter.
- **VoxLingua107 / ECAPA-TDNN (SpeechBrain, TalTech)** : dataset contient Lingala (90h) et Shona (30h) ; modèle ESPnet `lid_voxlingua107_mms_ecapa` combine encodeur MMS-1B + embeddings ECAPA-TDNN. Luganda absent de VoxLingua107 — limite connue.
- **Approche légère recommandée pour 3 langues connues à l'avance** (plutôt qu'un LID 1000+ langues généraliste) :
  1. Extraire des embeddings audio courts (x-vector/ECAPA ou embeddings MMS) sur des segments de quelques secondes.
  2. Entraîner un **classifieur simple 3 classes** (logistic regression / petit MLP / SVM) sur ces embeddings avec un petit jeu labellisé (train WaxalNLP, qui est déjà labellisé par langue) — beaucoup plus précis et rapide qu'un LID 1000-langues, car le problème est fermé à 3 classes.
  3. Alternative texte : si un modèle acoustique multilingue produit déjà une transcription brute, un **LID textuel léger** (fastText langid, ou un simple classifieur n-gram de caractères) sur la sortie CTC peut aussi servir de deuxième signal de routage/vote, utile pour arbitrer en cas de doute du LID audio.
  4. Attention licence : MMS checkpoints = CC-BY-NC pour certaines variantes (vérifier le model card exact) — peut poser un problème pour un usage "commercial"/compétitif selon les règles Zindi ; les embeddings speechbrain VoxLingua107 (TalTech) ont une licence plus permissive à vérifier aussi.

## 5. Ensembling de checkpoints publics (ROVER, model soups, combinaison logits)

- **ROVER (Recognizer Output Voting Error Reduction)** : technique classique (Fiscus 1997) qui aligne les N-best hypothèses de plusieurs systèmes ASR via alignement multiple (type edit-distance/WTN - word transition network) puis fait un vote mot par mot (majorité, pondérée par confiance). Implémentations : `sclite`/`asclite` (NIST SCTK toolkit) ont un module ROVER historique ; des réimplémentations Python existent (ex. `rover-nist`, ou scripts custom basés sur `jiwer`/alignement Levenshtein multiple). Avantage : fonctionne sur la sortie texte finale, donc combine facilement des checkpoints de licences/architectures différentes (wav2vec2, w2v-BERT, Whisper, MMS...) sans toucher aux poids.
- **Combinaison de logits CTC** (avant décodage) : moyenner les logits/probabilités frame-level de 2-3 checkpoints CTC compatibles (même vocabulaire/alphabet, idéalement même architecture ou même tokenizer) avant le beam search avec LM — plus puissant que ROVER si les modèles partagent l'alphabet, mais nécessite alignement temporel compatible (mêmes hyperparamètres de frame rate) et le même espace d'étiquettes CTC.
- **Model soups / weight averaging** : moyenner directement les poids de plusieurs checkpoints fine-tunés à partir du **même modèle de base** (ex. plusieurs fine-tunes de wav2vec2-XLSR sur des données/seeds différents) — documenté efficace dans le papier "Model soups" (Wortsman et al. 2022) pour la vision, et appliqué à des modèles speech dans des travaux suivants (greedy soup : ajouter les checkpoints un par un si ça améliore le dev score). Limite stricte : **ne fonctionne que si les checkpoints partagent la même architecture/initialisation** — on ne peut pas "souper" un wav2vec2 avec un w2v-BERT.
- **Limites légales/pratiques à vérifier avant de combiner des checkpoints publics** :
  - Vérifier individuellement la licence de chaque checkpoint HF (Apache-2.0, MIT, CC-BY-NC, modèles Meta/MMS souvent CC-BY-NC ou licences research-only) — combiner des poids ou des sorties de modèles sous licences incompatibles (ex. une licence non-commerciale) peut rendre le système final non conforme aux règles du challenge si le règlement Zindi interdit l'usage de checkpoints non-commerciaux ou exige l'attribution ; **relire précisément les règles de compliance WAXAL (cf. memory waxal-challenge-decisions.md)** avant de choisir les checkpoints à ensembler.
  - Le weight averaging multiplie le coût mémoire/inférence (plusieurs forward passes) — avec un GPU 8GB, privilégier 2 checkpoints petits (ex. wav2vec2-base-size) plutôt que des combinaisons lourdes, ou faire l'ensembling au niveau texte (ROVER) qui ne coûte presque rien en GPU.

## 6. Pseudo-labeling / self-training sur données non étiquetées

- **Principe** : utiliser le(s) modèle(s) ASR disponibles (éventuellement déjà améliorés par LM+decoding) pour transcrire de l'audio non étiqueté du même domaine/langue, filtrer les transcriptions à haute confiance (score CTC, longueur plausible, cohérence LID), puis les utiliser comme signal supplémentaire — mais le prompt précise "sans réentraînement du modèle acoustique" donc le vrai self-training (fine-tuning sur pseudo-labels) est hors scope direct de cette tâche. Ce qui reste utilisable sans réentraînement acoustique :
  - Utiliser les pseudo-labels texte pour **enrichir le corpus du LM n-gram** (texte généré automatiquement, filtré par confiance) — c'est une forme de self-training qui améliore seulement le LM, pas l'acoustique, et reste donc dans les contraintes de la tâche.
  - Risque de contamination : si les pseudo-labels sont de mauvaise qualité (haut WER), le LM va apprendre des patterns erronés et biaiser le décodage vers ces erreurs (boucle de rétroaction). Toujours filtrer agressivement (seuil de confiance élevé, ou ne garder que les segments où plusieurs systèmes de l'ensemble sont d'accord — "agreement filtering", lui-même lié au ROVER du §5).
  - Risque de biais : le pseudo-labeling sur un domaine différent (ex. audio lu vs audio spontané) peut introduire un LM trop adapté à un style non représentatif du test set — vérifier la distribution avant d'injecter massivement du texte pseudo-labellisé.

## 7. Plan d'action concret (ordre de priorité, GPU 8GB, quelques jours)

1. **Jour 1** : récupérer/valider le LM existant `keystats/waxal-kenlm-models` ; brancher `pyctcdecode` sur le modèle acoustique CTC public choisi ; tuner alpha/beta sur dev set par langue. Mesurer le gain WER/CER vs greedy.
2. **Jour 1-2** : construire un LM amélioré par langue en combinant texte WaxalNLP train (pondéré fort) + CC-100/MADLAD-400 filtré + Wikipedia + Bible/JW si disponible pour chaque langue (chercher explicitement sur opus.nlpl.eu, Masakhane, et `sagepond/lug_web` pour Luganda).
3. **Jour 2** : LID 3-classes léger (classifieur sur embeddings MMS/ECAPA, entraîné sur train WaxalNLP) pour router chaque audio vers le LM/décodeur de la bonne langue si le test set est multilingue non étiqueté.
4. **Jour 2-3** : post-traitement texte ciblé — seulement après avoir diagnostiqué les erreurs réelles sur dev set (casse, ponctuation, répétitions, segmentation spécifique bantou).
5. **Jour 3** (si temps restant) : ROVER entre 2-3 checkpoints ASR publics distincts (ex. MMS-1B-all + wav2vec2-XLSR fine-tuné + w2v-BERT) — vérifier licences avant de figer le choix final ; sinon, model soup si checkpoints de même architecture disponibles.
6. En continu : revalider chaque étape sur le dev set avec la formule exacte du score Zindi (`1 - 0.5*(WER+CER)`), car certaines techniques qui améliorent le WER (ex. restauration de ponctuation agressive) peuvent dégrader le CER et donc le score composite.

## Sources principales consultées
- [AfriqueLLM: Data Mixing and Model Architecture for African Languages](https://arxiv.org/pdf/2601.06395)
- [MADLAD-400 paper](https://arxiv.org/pdf/2309.04662)
- [keystats/waxal-kenlm-models (HF dataset)](https://huggingface.co/datasets/keystats/waxal-kenlm-models)
- [CC-100 dataset card](https://huggingface.co/datasets/statmt/cc100)
- [wikilangs Lingala model](https://huggingface.co/wikilangs/ln/blob/main/README.md)
- [Voice of a Continent: Mapping Africa's Speech Technology Frontier (arXiv 2505.18436)](https://arxiv.org/pdf/2505.18436)
- [OpenLID-v3 report](https://arxiv.org/pdf/2602.13139)
- [facebook/mms-lid-256 model card](https://huggingface.co/facebook/mms-lid-256)
- [espnet/lid_voxlingua107_mms_ecapa](https://huggingface.co/espnet/lid_voxlingua107_mms_ecapa)
- [VoxLingua107 project page (TalTech)](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/)
- [BibleTTS: multilingual African speech corpus (arXiv 2207.03546)](https://arxiv.org/pdf/2207.03546)
- [Shona writing system analysis (MSU library)](https://d.lib.msu.edu/juz/746?q=2)
- [Lingala Diacritic Correction (aflat.org)](https://www.aflat.org/node/329)
- [sagepond/lug_web Luganda LM corpus (HF)](https://huggingface.co/datasets/sagepond/lug_web)
- [Building Text and Speech Benchmark Datasets for Low-Resourced East African Languages (NeurIPS 2025 listing)](https://neurips.cc/virtual/2025/133714)
- [Building a Parallel Corpus and Training Translation Models Between Luganda and English (arXiv)](https://arxiv.org/abs/2301.02773v1)

**Note méthodologique** : certaines informations (couverture exacte d'OPUS pour ces 3 langues, corpus Shona news/Lingala Bible précis, support Luganda dans MMS-LID/VoxLingua107) n'ont pas pu être confirmées directement dans les recherches effectuées — elles sont signalées comme "à vérifier" dans le texte ci-dessus plutôt que supposées.

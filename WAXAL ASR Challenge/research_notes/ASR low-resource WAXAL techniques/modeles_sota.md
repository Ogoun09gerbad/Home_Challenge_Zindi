# Architectures SOTA pour ASR low-resource — Lingala, Shona, Luganda (challenge WAXAL)

> Portée corrigée : le challenge Zindi WAXAL couvre **Lingala (lin), Shona (sna), Luganda (lug)** — pas le wolof/sereer. Checkpoint de référence cité par l'utilisateur : `anyantudre/waxal-w2vbert-linsnalug-raw`.
> Contrainte : GPU 8GB → privilégier fine-tuning léger (CTC head, LoRA) plutôt que réentraînement lourd d'un decoder LLM 7B.

## 1. w2v-BERT 2.0 (Meta) — architecture et pré-entraînement

- **Papier source** : SeamlessM4T (arXiv 2308.11596) section 3.2.1 ; checkpoint public `facebook/w2v-bert-2.0`.
- **Architecture** : encodeur Conformer, 24 couches, ~600M paramètres (0.6B). Objectif de pré-entraînement combine apprentissage contrastif + masked prediction (comme w2v-BERT original, Chung et al. 2021), amélioré avec **deux codebooks GVQ** (product quantization) au lieu d'un seul. Perte totale = pondération 1.0 (contrastive) + 0.5 + 0.5 (deux termes masked-prediction).
- **Taille des données de pré-entraînement** : sources divergentes — papier SeamlessM4T v1 dit 1M heures / 143 langues ; page Meta SeamlessM4T v2 dit 4.5M heures. La page HF `facebook/w2v-bert-2.0` indique aussi 4.5M heures / >143 langues (metadata liste 96 langues taguées).
- **Licence : MIT** (facebook/w2v-bert-2.0 sur HF). C'est un checkpoint **"bare"** — pas de tête de modélisation, nécessite fine-tuning (ex. tête CTC) pour faire de l'ASR. Pas de taille alternative plus petite publiée par Meta pour ce modèle précis (contrairement à MMS qui a des variantes 300M/1B).
- **Comparaison avec MMS** : d'après le benchmark AfricaNLP 2026 (Nahabwe et al., arXiv 2512.10968) — MMS et w2v-BERT sont tous deux **plus data-efficients en régime très low-resource** (quelques heures à quelques dizaines d'heures), alors que XLS-R scale mieux quand plus de données sont disponibles, et Whisper est meilleur en régime mid-resource. w2v-BERT 2.0 (via wav2vec2-bert + tête CTC) est le choix de base du checkpoint `anyantudre/waxal-w2vbert-linsnalug-raw`.

## 2. MMS (Massively Multilingual Speech, Meta)

- Famille de checkpoints `facebook/mms-300m`, `facebook/mms-1b-all` (fine-tuné CTC sur 1162 langues), `facebook/mms-1b-fl102`. Architecture wav2vec2 standard avec têtes CTC par langue (adapters légers par langue — utile pour fine-tuning low-resource car seule une petite tête/adapter est entraînée).
- **Checkpoint spécifique Lingala WAXAL** : `waxal-benchmarking/mms-300m-waxal-lin` — WER **42.6%**, CER **18.9%** sur le test WAXAL filtré (base `facebook/mms-300m`).
- D'après le papier Omnilingual ASR de Meta (arXiv 2511.09690, nov. 2025), MMS a été en partie superseded par **Omnilingual ASR** (voir §5), qui couvre 1600+ langues avec des variantes CTC et decoder-LLM.
- Gains vs taille de données : sur 13 langues africaines (arXiv 2512.10968), la plupart des gains sont obtenus avant 50h d'entraînement ; au-delà les courbes plafonnent et varient fortement par langue (ex. Lingala : WER ~51.3% à 1h → ~23.96% à 100h — chiffre cité pour Lingala dans ce papier, à vérifier contre la version complète du PDF non entièrement extractible).
- **Licence** : checkpoints MMS généralement CC-BY-NC 4.0 pour certains, mais `facebook/mms-1b-all`/`mms-300m` sur HF sont publiés sous licence permettant le fine-tuning (vérifier la fiche modèle exacte avant usage commercial — Meta a historiquement mis certains assets MMS sous licence research-only CC-BY-NC alors que les poids wav2vec2-only sont souvent Apache/MIT-like via transformers).

## 3. Whisper (OpenAI) fine-tuné bas-ressource — pièges connus

- **Hallucinations en zero-shot** : sur le Bambara, *tous* les variants Whisper testés dépassent 100% WER (plus de tokens produits que de référence), signe d'hallucination massive (arXiv 2602.09785).
- **Insertions / répétitions** : étude Kikuyu (arXiv 2510.07221, fine-tuning sur ~270h) tague les échantillons à WER≥40% avec un tag "spurious repetition" quand les répétitions dépassent 10% du texte — problème persistant même après fine-tuning partiel.
- **Le fine-tuning corrige largement le problème** : sur Swivuriso (Afrique du Sud, arXiv 2512.02201), Whisper zero-shot donne WER 223% (Setswana) et 190% (Xitsonga) ; après fine-tuning, ces chiffres tombent à 13% et 12%. Les auteurs lient les hallucinations low-resource à des erreurs d'insertion.
- **Résultats spécifiques aux 3 langues du challenge** :
  - **Luganda** — `waxal-benchmarking/whisper-tiny-lug-victor` (Whisper Tiny, fine-tuné sur WAXAL) : WER **33.8%**, CER **12.8%** (test WAXAL spontané). `BuzzASR/luganda` (Whisper Large v3 fine-tuné) : WER 50.0% (FLEURS) / 42.92% (Common Voice 25) / combiné 44.8%, CER combiné 24.97%.
  - **Shona** — `Code-Dev/whisper-small-shona-v3(-a)` (fine-tuning progressif sur FLEURS + Bible Shona) : WER descend de 37.15% (baseline FLEURS) à 25.52–26.42% après plusieurs étapes. `CasperMuz/whisper-base-sna` (fine-tuné sur WAXAL Shona) : WER 41.48%.
  - **Lingala** — fine-tune Whisper-base générique (deepdml, carte mal étiquetée comme igbo) : WER ~54.9% sur FLEURS, CER ~21.4%. Pas de chiffre Whisper-Lingala propre au WAXAL trouvé dans les résultats de recherche, contrairement à MMS/w2v-BERT.
- **Sunbird AI "Whisper 51 African Languages"** : adaptation multilingue de Whisper Large v3 couvrant 49-51 langues africaines dont le lingala (`lin`), listant explicitement hallucination et erreurs de substitution comme limitations connues, surtout pour les langues les moins dotées.
- **Conclusion pratique** : Whisper zero-shot est **inutilisable tel quel** sur ces 3 langues (hallucinations lourdes) ; un fine-tuning, même léger, est indispensable. CTC (w2v-BERT/MMS) tend à être plus robuste et plus data-efficient en très low-resource, sans le risque d'hallucination texte-libre propre aux architectures seq2seq/decoder.

## 4. CTC vs seq2seq (encoder-decoder/attention) — low-resource

- Aucune étude trouvée ne compare directement CTC vs attention *sur WER + robustesse au bruit* avec un protocole strictement identique en low-resource africain, mais le consensus de la littérature (Myanmar ~26h, Babel turc, hybrides CTC/attention MERL sur WSJ0 ~15h) va dans le même sens :
  - **CTC** : moins cher à entraîner, plus facile à intégrer (décodage WFST/n-gram LM), **meilleur候 en très faibles données** (quelques dizaines d'heures) — c'est l'approche de tous les checkpoints WAXAL mentionnés ci-dessus basés sur w2v-BERT/MMS/XLS-R.
  - **Attention/seq2seq pur** (type Whisper) : peut dépasser CTC quand les données sont abondantes, mais est **plus cher**, plus difficile à paralléliser, et plus sujet aux hallucinations/répétitions en low-resource/zero-shot (voir §3).
  - **Hybride CTC/attention** : la perte CTC auxiliaire améliore la robustesse au bruit en forçant un alignement monotone ; gains rapportés de 5.4–14.6% CER relatif par rapport à l'attention seule (comparaison MERL, hors contexte africain spécifique).
  - Le papier **Omnilingual ASR** de Meta (voir §5) propose justement les deux têtes (CTC et decoder-LLM) sur le même encodeur, ce qui permet une comparaison directe ; les variantes CTC-300M sont nettement moins coûteuses à fine-tuner sur GPU 8GB qu'un decoder-LLM 7B.
- **Pour un GPU 8GB** : fine-tuner une tête CTC sur un encodeur gelé/partiellement gelé (w2v-BERT 2.0 ou MMS-300M) est largement praticable ; fine-tuner un decoder-LLM 7B (type `facebook/omniASR-LLM-7B` utilisé par la solution Zindi `ravitejac/waxal-asr-solution`) nécessite LoRA/quantization agressive et reste à la limite voire hors de portée sans offloading.

## 5. Omnilingual ASR (Meta, nov. 2025) — nouveau foundation model pertinent

- **Papier** : arXiv 2511.09690, "Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages". Blog Meta AI correspondant.
- Couvre **1600+ langues**, dont 500+ jamais servies par un système ASR auparavant ; permet d'ajouter une langue non couverte avec seulement quelques échantillons (few-shot/in-context, via le decoder-LLM).
- **Deux variantes de tête** sur le même encodeur self-supervised scalé jusqu'à 7B : (a) tête **CTC**, (b) **decoder type LLM** (wav2vec2 encoder + decoder transformer façon LLM). Tailles : de 300M (device-friendly) à 7B.
- **Licence** : poids Apache-2.0 (selon couverture secondaire) ; le **Omnilingual ASR Corpus** (350 langues sous-dotées, données commissionnées) est publié en CC-BY. À vérifier précisément sur le repo `facebookresearch/omnilingual-asr` avant usage.
- **Pertinence directe pour le challenge** : l'organisation HF `waxal-benchmarking` a publié des collections **"WAXALNet"** fine-tunées par langue dérivées de ce foundation model :
  - **WAXALNet — Omnilingual CTC-300M** (19 langues) : tête CTC, 300M params — probablement le point de départ le plus praticable sur GPU 8GB.
  - **WAXALNet — Omnilingual LLM-300M** (19 langues) : decoder wav2vec2→LLM-style, 300M params.
  - **WAXALNet — Whisper Tiny** fine-tuné WAXAL.
  - Des checkpoints d'ablation nommés `omniasr-llm-300m-waxal-loo-{ach,ewe,ful,lin}` existent mais sont explicitement tagués "Not intended for use" (archives d'ablation leave-one-out).
  - La solution gagnante/top Zindi `ravitejac/waxal-asr-solution` combine : fine-tunes CTC par langue de `facebook/w2v-bert-2.0` (Lingala/Shona séparés), un decoder 7B `facebook/omniASR-LLM-7B` fine-tuné avec objectif **MWER** (minimum word error rate, expected-risk), KenLM 5-gram pour le décodage, routing de langue via `facebook/mms-lid-256`, et un correcteur d'erreurs LoRA r16 sur `google/gemma-4-E4B-it`. Score public 0.757, privé 0.766 (métrique = 1 − (WER+CER)/2).

## 6. Checkpoints publics HF — tableau récapitulatif (Lingala / Shona / Luganda)

| Langue | Modèle | Base | Métrique | Licence (si connue) | Notes |
|---|---|---|---|---|---|
| lin+sna+lug | `anyantudre/waxal-w2vbert-linsnalug-raw` | w2v-BERT 2.0 (600M) + CTC, vocab 97 tokens | leaderboard public : score 0.742, CER 0.116, WER 0.400 | Apache-2.0 (modèle), CC-BY-4.0 (données WaxalNLP) | Multi-langue, moins bon que les spécialistes séparés par langue ; pas de mesure fiable sur Luganda (absent du test set Phase 2) |
| lin | `waxal-benchmarking/mms-300m-waxal-lin` | MMS-300M + CTC | WER 42.6%, CER 18.9% | — | Test WAXAL filtré |
| lin | deepdml whisper-base (carte étiquetée ig) | Whisper-base | WER ~54.9%, CER ~21.4% | — | Fiche mal étiquetée (igbo/lingala confus), prudence |
| sna | `Code-Dev/whisper-small-shona-v3-a` / `-v3` | Whisper-small fine-tuné progressif | WER 25.5–26.4% | — | Test FLEURS + Bible Shona |
| sna | `CasperMuz/whisper-base-sna` | Whisper-base | WER 41.48% | — | Fine-tuné sur WAXAL Shona |
| sna | `KhayaAI/w2v-bert-sna` | w2v-BERT 2.0 + CTC | WER 3.02% | — | Test in-domain étroit — ne pas généraliser |
| lug | `waxal-benchmarking/whisper-tiny-lug-victor` | Whisper-tiny | WER 33.8%, CER 12.8% | — | Test WAXAL spontané |
| lug | `BuzzASR/luganda` | Whisper Large v3 | WER combiné 44.8%, CER 24.97% | — | FLEURS + Common Voice 25 |
| lug | `dmusingu/XLS-R-LUGANDA-ASR-CV14` | XLS-R 300M + CTC | WER 24.1%, CER 5.4% | — | Common Voice 14 |
| lug | `indonesian-nlp/wav2vec2-luganda` | XLSR-53 + CTC | WER 15.38% (sans LM) | — | Common Voice 7 (ancien) |
| lug | `VinyVan/xlsr-luganda-waxal` | XLSR Swahili 400h + CTC | WER 21.44%, CER 4.46% | — | Dataset d'éval non précisé |
| lin+sna+lug | `waxal-benchmarking/WAXALNet-Omnilingual-CTC-300M` (collection, 19 langues) | Omnilingual ASR CTC-300M | non publié dans les pages consultées | Apache-2.0 (poids, à confirmer) | Candidat le plus praticable sur GPU 8GB pour fine-tuning complémentaire |

**Attention** : ces chiffres proviennent de fiches modèles différentes, sur des jeux de test différents (WAXAL filtré, FLEURS, Common Voice versions variées, Bible Shona) — **non directement comparables**. Pour une comparaison valide, réévaluer les checkpoints candidats sur un même split WAXAL hold-out.

## 7. Travaux récents (2024–2026) pertinents

- **Omnilingual ASR** (Meta, nov. 2025, arXiv 2511.09690) — few-shot/in-context extension à des langues non vues, pertinent si le challenge WAXAL élargit à d'autres langues bantoues.
- **Benchmarking ASR for African Languages** (Nahabwe et al. 2026, arXiv 2512.10968) — comparaison MMS/XLS-R/W2v-BERT/Whisper sur 13 langues africaines, courbe de gains vs heures de données, plafonnement après ~50h.
- **Voice of a Continent: Mapping Africa's Speech Technology Frontier** (arXiv 2505.18436) — panorama des ressources et modèles pour les langues africaines.
- **WAXAL corpus (Google, 2026)** — papier source du challenge ; fine-tuning rapporté réduisant le WER macro-moyen jusqu'à 17% pour MMS-1B-all, jusqu'à 58% pour Whisper-Large-v3, jusqu'à 56% pour Gemma 3n 2B.
- **Critique de la métrique WER** (AfricaNLP 2026) — argumente que le WER mélange erreurs phonologiques/tonales/lexicales pour les langues africaines et recommande CER + mesures au niveau des traits phonétiques — pertinent car Shona et Luganda ont des phénomènes tonaux/morphologiques non triviaux pour le WER standard.
- **Where Are We At with ASR for Bambara?** (arXiv 2602.09785) et **Swivuriso** (arXiv 2512.02201) — documentent systématiquement les hallucinations Whisper zero-shot sur langues bantoues/mandées apparentées et leur correction par fine-tuning.
- **Fine-tuning Whisper for Low-Resource Real-World Applications** (arXiv 2412.15726) — méthodologie générique de fine-tuning Whisper bas-ressource (utile comme recette pratique).

## 8. Recommandations pratiques (contrainte GPU 8GB)

1. **Partir d'un encodeur CTC** (w2v-BERT 2.0 600M ou MMS-300M, voire Omnilingual-CTC-300M) plutôt que d'un decoder-LLM 7B — bien plus réalisable en fine-tuning sur 8GB (geler une partie de l'encodeur, batch réduit, bf16/fp16, gradient checkpointing — cf. recette `anyantudre/waxal-w2vbert-linsnalug-raw` : 1 GPU, bf16, gradient checkpointing, batch effectif 36).
2. **Spécialiser par langue** plutôt qu'un modèle unique lin+sna+lug : les fiches modèles indiquent systématiquement que les checkpoints spécialistes par langue surpassent le multi-langue combiné.
3. **Éviter Whisper zero-shot** pour ces 3 langues (risque d'hallucination/WER>100%) ; si Whisper est utilisé, fine-tuning obligatoire + idéalement contrainte de longueur/répétition en décodage.
4. **Décodage avec LM externe (KenLM n-gram)** apporte un gain mesurable (ex. solution Zindi top avec KenLM 5-gram) et reste peu coûteux en mémoire par rapport à un correcteur LLM.
5. **Augmentation audio** (bruit MUSAN, RIR, gain aléatoire) utilisée par le checkpoint de référence — technique standard et peu coûteuse à ajouter.
6. **Vérifier les licences avant usage** : w2v-BERT 2.0 et checkpoints `anyantudre` = MIT/Apache-2.0 (permissifs) ; données WaxalNLP = CC-BY-4.0 ; MMS a historiquement des variantes sous licence plus restrictive (CC-BY-NC) — vérifier la fiche modèle exacte choisie avant tout usage en dehors du cadre challenge.

## Sources principales
- SeamlessM4T paper: https://arxiv.org/pdf/2308.11596
- facebook/w2v-bert-2.0: https://huggingface.co/facebook/w2v-bert-2.0
- anyantudre/waxal-w2vbert-linsnalug-raw: https://huggingface.co/anyantudre/waxal-w2vbert-linsnalug-raw
- Benchmarking ASR for African Languages: https://arxiv.org/pdf/2512.10968
- Bambara ASR: https://arxiv.org/pdf/2602.09785
- Swivuriso dataset: https://arxiv.org/pdf/2512.02201
- Kinyarwanda/Kikuyu data scaling: https://arxiv.org/pdf/2510.07221
- Omnilingual ASR: https://arxiv.org/pdf/2511.09690 ; https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/
- waxal-benchmarking org (HF): https://huggingface.co/waxal-benchmarking
- waxal-benchmarking/mms-300m-waxal-lin: https://huggingface.co/waxal-benchmarking/mms-300m-waxal-lin
- waxal-benchmarking/whisper-tiny-lug-victor: https://huggingface.co/waxal-benchmarking/whisper-tiny-lug-victor
- Code-Dev/whisper-small-shona-v3: https://huggingface.co/Code-Dev/whisper-small-shona-v3
- BuzzASR/luganda: https://huggingface.co/BuzzASR/luganda
- dmusingu/XLS-R-LUGANDA-ASR-CV14: https://huggingface.co/dmusingu/XLS-R-LUGANDA-ASR-CV14
- ravitejac/waxal-asr-solution: https://huggingface.co/ravitejac/waxal-asr-solution
- Sunbird Whisper 51 African Languages: https://salt.sunbird.ai/models/asr-whisper-51-african-languages/
- WAXAL corpus (Google Research): https://research.google/pubs/waxal-a-large-scale-multilingual-african-language-speech-corpus/

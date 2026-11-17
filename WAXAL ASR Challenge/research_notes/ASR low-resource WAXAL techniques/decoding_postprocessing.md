# LM decoding + post-processing for low-resource CTC (Lingala / Shona / Luganda) — WAXAL Zindi

Target score: `1 - 0.5*(WER + CER)`, WER on lowercased text with punctuation kept, CER on the raw string. Constraints: 8GB GPU, a few days, no heavy acoustic retraining. All techniques below are **inference/post-processing only**.

## 0. What already exists for this challenge
There is an HF dataset **`keystats/waxal-kenlm-models`** containing pre-built 5-gram KenLM models for Lingala, Shona and Luganda, designed for shallow fusion with `pyctcdecode`, with the WaxalNLP validation/test splits explicitly excluded from the training corpus (to avoid any leakage). Check/reuse it first before building your own LM — this can save a full day.

## 1. CTC beam-search decoding + n-gram LM (KenLM)

### Standard pipeline (pyctcdecode)
1. Get the vocabulary/alphabet of the wav2vec2/w2v-BERT model's CTC tokenizer (`processor.tokenizer.get_vocab()`).
2. Build a KenLM LM:
   - `lmplz --discount_fallback -o 5 -S 4G < corpus.lower.txt > lm.arpa` (order 3 to 5; 5-gram if the corpus is large enough, otherwise 3-gram to avoid overfitting on a small African-language corpus).
   - `build_binary lm.arpa lm.binary` (binary = fast loading, important with little RAM).
3. `pyctcdecode.build_ctcdecoder(labels, kenlm_model_path="lm.binary", alpha=0.5, beta=1.0)` — `alpha` = LM weight, `beta` = word-length bonus. **Tune alpha/beta on a small dev set** (coarse grid search, e.g. alpha∈{0.2,0.5,0.8,1.2}, beta∈{0,0.5,1.5}) since the optimum depends heavily on the size/quality of the LM.
4. Decode the CTC logits (no need to retrain the acoustic model): `decoder.decode(logits, beam_width=100-500)`.
5. Alternative: `flashlight-text` (Python bindings of the flashlight/wav2letter decoder) gives faster C++ CTC+KenLM decoding if pyctcdecode is too slow on CPU — similar API (`LexiconFreeDecoder` or `LexiconDecoder` + `KenLM`).

### Data-poor LM case (African languages)
- With little target text, prefer a **low-order LM (3-gram)** with Kneser-Ney + `--discount_fallback` (avoids KenLM errors when rare n-gram counts are insufficient).
- Interpolate/mix several sources (see §2), weighting by domain similarity (WaxalNLP train corpus >> generic web corpus).
- A small, poorly tuned LM can **degrade** the CER (the LM biases toward frequent words and can break rare words/proper nouns): always compare against the greedy CTC baseline on the dev set, feature by feature (alpha=0 must reproduce greedy).
- The final score mixes WER (normalized lowercase+punctuation text) and CER (raw): an overly aggressive LM that "corrects" punctuation or casing may improve WER but hurt CER if the predicted text drifts too far from the reference at the character level — validate empirically, do not assume.

### Alternative rescoring (without integrating the LM into the beam search)
- Generate N-best CTC hypotheses (greedy + a few beams), then rescore with an external LM (KenLM or even a small causal model such as an African GPT-2/fine-tuned mBERT) — less risky than full shallow fusion, and makes it easy to combine several signals (length, LM, acoustic score) via a small logistic regression trained on the dev set.

## 2. Text sources for n-gram LMs: Lingala / Shona / Luganda

| Source | Known coverage | Notes |
|---|---|---|
| **CC-100 (Facebook/StatMT)** | Luganda (`lg`, ~7.3M tokens), Lingala (`ln`, ~2.3M tokens) | Shona not confirmed in this release; built from 2018 CommonCrawl snapshots via CC-Net. |
| **MADLAD-400** | Shona (`sn`), Luganda (`lg`), Lingala (`ln`) all present in the language mapping | Manually audited CommonCrawl dataset, 3T tokens, 419 languages; filter/deduplicate before use since it is noisy for rare languages. |
| **Wikipedia (dumps + wikilangs)** | Lingala has an active wiki ("wikilangs/ln" models derived from monthly dumps); Shona and Luganda have smaller but usable Wikipedias (official wikipedia.org dumps) | Useful as a clean, well-formatted complement, but low volume on its own. |
| **OPUS** | Not confirmed for these 3 languages in the searches carried out — check directly on opus.nlpl.eu (filter by ISO code `lin`/`sna`/`lug`); JW300/bible-uedin on OPUS have historically covered many African languages, potentially including these three | **Concrete action: go to opus.nlpl.eu and search the JW300, bible-uedin, Tatoeba, CC-Aligned corpora for these language codes before ruling them out.** |
| **Bible / JW.org corpora** | Often available language by language (complete translations of the Bible and JW literature exist in Shona, Lingala, Luganda) | Religious text in a formal style, different from conversational ASR style but useful for lexical coverage and morphology; mix with more "oral" sources if possible. |
| **Makerere / Sunbird AI / SALT corpora (Luganda)** | SALT = MT parallel corpus for 5 Ugandan languages including Luganda; `sagepond/lug_web` (HF) = Luganda LM corpus (oral literature, social media, news, religious); the paper "Building Text and Speech Benchmark Datasets... East African Languages" (Applied AI Letters 2024, Makerere team) also describes Luganda ASR/text data | The richest resources for Luganda specifically — prioritize. |
| **Shona / Lingala-specific academic corpora** | Not precisely located in this search (to dig into: Zimbabwean universities for Shona, DRC/Congo for Lingala, African NLP projects such as Masakhane) | **Masakhane** (African NLP collective) is a strong lead to explore for community-organized Shona/Lingala resources. |
| **Raw WaxalNLP text (train split)** | Already in the challenge dataset | **Priority #1**: this is the text closest to the target domain/style (already used to build `keystats/waxal-kenlm-models`). |

**Recommended strategy**: main LM = WaxalNLP train text (in-domain) interpolated/augmented with CC-100 + MADLAD-400 (quality-filtered, deduplicated) + Wikipedia + Bible/JW for lexical coverage, weighting the in-domain corpus heavily (e.g. x5-10 upsampling of the WaxalNLP text in the final corpus before `lmplz`, or interpolation of two separate LMs via the KenLM interpolation tools).

## 3. Text normalization / post-processing

Goal: reduce WER (normalized text) and CER (raw text) without introducing errors.

- **Casing**: WER is computed in lowercase → systematic lowercasing of the output costs nothing for WER, but beware: CER is computed on the raw string, so if the reference has native casing, do NOT lowercase the final submitted output without checking the exact expected submission format (the prompt says "WER on lowercased text" = it is the metric that lowercases, not necessarily you — check the official Zindi evaluation script before changing the output casing).
- **Punctuation**: kept for WER → a classic CTC model often produces no punctuation. Options without retraining:
  - A lightweight pre-trained **punctuation restoration** model (e.g. multilingual `punctuation_restoration` models on HF, or an mBERT-based punctuation model) applied as post-processing on the ASR output.
  - Simple heuristic rules (final period at end of sentence, no hallucinated comma) — risk of degrading WER if poorly calibrated; validate on the dev set only, never assume.
- **Numbers**: inverse text normalization (digits → words or the reverse) depending on what the reference uses — empirically check the format of the WaxalNLP references (numbers spelled out or numeric) and align the CTC output accordingly.
- **CTC repetitions / hallucinations**: CTC models can produce repeated words/syllables on noisy audio — a simple post-filter deduplicating identical consecutive n-grams (sliding window) can reduce CER safely if applied carefully (only remove suspicious exact consecutive repetitions, not legitimate repetitions in the language).
- **Compound words / segmentation**: Bantu languages (Shona, Luganda) are highly agglutinative — watch the word segmentation of the LM; a badly segmented LM can force incorrect word splits. Check segmentation consistency between the LM and the annotated WaxalNLP reference.
- **Per-language orthographic specifics (to validate against the actual references, no universal rule assumed)**:
  - **Luganda**: long vowels often written by doubling the vowel (e.g. *-aa-*); tones (3 tones) generally not marked in standard orthography — do not try to restore tones that are not present in the references.
  - **Shona**: orthography built on the "one sound, one symbol" principle (Doke), revised in 1955/1967; documented historical word-division problems — a potential source of WER variance if the decoder segments differently from the reference convention.
  - **Lingala**: 2 tones (high/low), often unmarked in everyday written text; accents/diacritics (e.g. open vowels ɔ/ɛ) frequently omitted in informal web text (e.g. "likɔngá" → "likonga") — if the LM is trained on unaccented web text while the WaxalNLP references use diacritics (or the reverse), there is a risk of systematic mismatch, to check first on a dev-set sample before any major LM work.

**Practical recommendation**: before any normalization rule, extract 50-100 examples from the dev set, compare the raw CTC output vs reference character by character, and only code corrections that address actually observed errors (avoid unverified "theoretical" rules).

## 4. Language identification (LID) for audio → model/decoder routing

- **MMS-LID (Meta)**: public checkpoints `facebook/mms-lid-256` (and 126/1024-language variants), fine-tuned from `mms-1b`. Covers Shona and Lingala with good scores (MMS-LID-1024 ≈ 96 macro F1 on Lingala, ≈ 87-91 on Shona depending on the test set, according to the "Voice of a Continent" paper, arXiv 2505.18436). Luganda not confirmed in the tables found — test directly with the 1024-language checkpoint on a Luganda sample before ruling it out.
- **VoxLingua107 / ECAPA-TDNN (SpeechBrain, TalTech)**: the dataset contains Lingala (90h) and Shona (30h); the ESPnet model `lid_voxlingua107_mms_ecapa` combines an MMS-1B encoder + ECAPA-TDNN embeddings. Luganda is absent from VoxLingua107 — a known limitation.
- **Recommended lightweight approach for 3 languages known in advance** (rather than a general 1000+-language LID):
  1. Extract short audio embeddings (x-vector/ECAPA or MMS embeddings) on segments of a few seconds.
  2. Train a **simple 3-class classifier** (logistic regression / small MLP / SVM) on these embeddings with a small labeled set (WaxalNLP train, which is already labeled by language) — much more accurate and faster than a 1000-language LID, since the problem is closed to 3 classes.
  3. Text alternative: if a multilingual acoustic model already produces a raw transcription, a **lightweight text LID** (fastText langid, or a simple character n-gram classifier) on the CTC output can also serve as a second routing/voting signal, useful to arbitrate when the audio LID is uncertain.
  4. License caution: MMS checkpoints = CC-BY-NC for some variants (check the exact model card) — may be a problem for "commercial"/competitive use depending on the Zindi rules; the speechbrain VoxLingua107 (TalTech) embeddings have a more permissive license, also to be checked.

## 5. Ensembling public checkpoints (ROVER, model soups, logit combination)

- **ROVER (Recognizer Output Voting Error Reduction)**: classic technique (Fiscus 1997) that aligns the N-best hypotheses of several ASR systems via multiple alignment (edit-distance/WTN - word transition network style), then votes word by word (majority, weighted by confidence). Implementations: `sclite`/`asclite` (NIST SCTK toolkit) have a historical ROVER module; Python reimplementations exist (e.g. `rover-nist`, or custom scripts based on `jiwer`/multiple Levenshtein alignment). Advantage: works on the final text output, so it easily combines checkpoints with different licenses/architectures (wav2vec2, w2v-BERT, Whisper, MMS...) without touching the weights.
- **CTC logit combination** (before decoding): average the frame-level logits/probabilities of 2-3 compatible CTC checkpoints (same vocabulary/alphabet, ideally same architecture or same tokenizer) before the LM beam search — more powerful than ROVER if the models share the alphabet, but requires compatible time alignment (same frame-rate hyperparameters) and the same CTC label space.
- **Model soups / weight averaging**: directly average the weights of several checkpoints fine-tuned from the **same base model** (e.g. several wav2vec2-XLSR fine-tunes on different data/seeds) — documented as effective in the "Model soups" paper (Wortsman et al. 2022) for vision, and applied to speech models in later work (greedy soup: add checkpoints one by one if it improves the dev score). Strict limitation: **only works if the checkpoints share the same architecture/initialization** — you cannot "soup" a wav2vec2 with a w2v-BERT.
- **Legal/practical limits to check before combining public checkpoints**:
  - Check the license of each HF checkpoint individually (Apache-2.0, MIT, CC-BY-NC; Meta/MMS models are often CC-BY-NC or research-only licenses) — combining weights or outputs of models under incompatible licenses (e.g. a non-commercial license) can make the final system non-compliant with the challenge rules if the Zindi rules forbid non-commercial checkpoints or require attribution; **carefully re-read the WAXAL compliance rules (cf. memory waxal-challenge-decisions.md)** before choosing the checkpoints to ensemble.
  - Ensembling multiplies memory/inference cost (several forward passes) — with an 8GB GPU, favor 2 small checkpoints (e.g. wav2vec2-base size) over heavy combinations, or ensemble at the text level (ROVER), which costs almost nothing on the GPU.

## 6. Pseudo-labeling / self-training on unlabeled data

- **Principle**: use the available ASR model(s) (possibly already improved by LM+decoding) to transcribe unlabeled audio from the same domain/language, filter high-confidence transcriptions (CTC score, plausible length, LID consistency), then use them as an extra signal — but the prompt specifies "without retraining the acoustic model", so true self-training (fine-tuning on pseudo-labels) is out of the direct scope of this task. What remains usable without acoustic retraining:
  - Use the text pseudo-labels to **enrich the n-gram LM corpus** (automatically generated text, filtered by confidence) — a form of self-training that only improves the LM, not the acoustics, and therefore stays within the task's constraints.
  - Contamination risk: if the pseudo-labels are of poor quality (high WER), the LM will learn erroneous patterns and bias decoding toward these errors (feedback loop). Always filter aggressively (high confidence threshold, or only keep segments where several systems of the ensemble agree — "agreement filtering", itself related to the ROVER of §5).
  - Bias risk: pseudo-labeling on a different domain (e.g. read vs spontaneous audio) can introduce an LM overly adapted to a style not representative of the test set — check the distribution before injecting large amounts of pseudo-labeled text.

## 7. Concrete action plan (priority order, 8GB GPU, a few days)

1. **Day 1**: fetch/validate the existing LM `keystats/waxal-kenlm-models`; plug `pyctcdecode` into the chosen public CTC acoustic model; tune alpha/beta on the dev set per language. Measure the WER/CER gain vs greedy.
2. **Days 1-2**: build an improved LM per language by combining WaxalNLP train text (heavily weighted) + filtered CC-100/MADLAD-400 + Wikipedia + Bible/JW if available for each language (search explicitly on opus.nlpl.eu, Masakhane, and `sagepond/lug_web` for Luganda).
3. **Day 2**: lightweight 3-class LID (classifier on MMS/ECAPA embeddings, trained on WaxalNLP train) to route each audio to the right language's LM/decoder if the test set is multilingual and unlabeled.
4. **Days 2-3**: targeted text post-processing — only after diagnosing the actual errors on the dev set (casing, punctuation, repetitions, Bantu-specific segmentation).
5. **Day 3** (if time remains): ROVER across 2-3 distinct public ASR checkpoints (e.g. MMS-1B-all + fine-tuned wav2vec2-XLSR + w2v-BERT) — check licenses before freezing the final choice; otherwise, a model soup if checkpoints of the same architecture are available.
6. Continuously: re-validate each step on the dev set with the exact Zindi score formula (`1 - 0.5*(WER+CER)`), since some techniques that improve WER (e.g. aggressive punctuation restoration) can degrade CER and therefore the composite score.

## Main sources consulted
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

**Methodological note**: some information (exact OPUS coverage for these 3 languages, specific Shona news/Lingala Bible corpora, Luganda support in MMS-LID/VoxLingua107) could not be confirmed directly in the searches carried out — it is flagged as "to be checked" in the text above rather than assumed.

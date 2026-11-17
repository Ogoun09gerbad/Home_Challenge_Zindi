# WAXAL ASR data, lin/sna/lug baselines, and the Zindi WAXAL ASR challenge(s)

Research date: 2026-10-26. Most sources are from 2025-2026; older ones are flagged. "UNVERIFIED" marks claims from secondary or third-party sources, or my own inferences.

IMPORTANT FRAMING NOTE: I found **no public page for an "October Study Jam: Google WAXAL ASR Challenge"**. I tried several web searches ("Study Jam" Zindi WAXAL / 2026 ASR) and the zindi.world competitions listing. Everything below about the Zindi challenge comes from the **original "Google WAXAL ASR Challenge"** (26 Jun to 9 Aug 2026, $10k, winners announced 22 Sep 2026). The October Study Jam is probably a re-run or "knowledge" version on the same data. The organisers' own page should be checked for differences in metric, test set and languages.

---

## Q1. What is the WAXAL ASR data and how was it built (lin / sna / lug)?

### Takeaway
WAXAL ASR is **image-prompted, spontaneous, descriptive speech**, not read speech. Speakers described Open Images pictures (50+ topics, at least 15 s each) in their own environment. Each recording was transcribed by one paid local annotator. Audio is 16 kHz, and splits are 80/10/10. In **v1, speakers overlap between train and eval for lin/sna/lug** (up to 100% overlap). A speaker-disjoint **ASR_v2** re-split exists. Lingala and Shona come from Digital Umuganda (AfriVoice, CC-BY-SA-4.0). Luganda comes from Makerere (Yogera, CC-BY-SA-4.0). No public transcription-convention document was found (casing, punctuation, tags, numerals).

### Cited Findings
**Identity and versions**
- HF dataset `google/WaxalNLP`, version 2.0.0, updated 03/2026. Paper is arXiv 2602.02734, "WAXAL: A Large-Scale Multilingual African Language Speech Corpus". Viewer shows ~3,378,718 rows and 821 GB in Parquet. — [HF dataset card](https://huggingface.co/datasets/google/WaxalNLP)
- Funded by Google and the Gates Foundation. Partners: Makerere University, University of Ghana, Digital Umuganda, Media Trust, Loud and Clear, AIMS Senegal. — [HF card](https://huggingface.co/datasets/google/WaxalNLP)
- The card describes ASR as "~1,250 hours of transcribed natural speech" across 19 languages. — [HF card](https://huggingface.co/datasets/google/WaxalNLP)
- Headline figures conflict between sources:
  - arXiv v3 (Mar 2026): ~1,250 h ASR (paper text says 14 languages, 224,767 instances) and ~235 h TTS. — [arXiv v3 HTML](https://arxiv.org/html/2602.02734v3)
  - NeurIPS 2026 poster: ~2,340 h of ASR out of more than 14,100 h collected. — [NeurIPS poster](https://neurips.cc/virtual/2026/poster/139649) (via search snippet, UNVERIFIED in full)
  - Google Research blog (6 Mar 2026): 27 languages, ~1,846 h ASR, more than 565 h TTS. — [Google Research blog](https://research.google/blog/waxal-a-large-scale-open-resource-for-african-language-speech-technology/)
  - Winners blog (Sep 2026): "32 African languages". — [Google Africa blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-winners-of-the-waxal-speech-recognition-challenge/)

**Per-language (lin / sna / lug)**
- Viewer row counts for the ASR configs:
  - `lin_asr` (Lingala): 89k rows
  - `sna_asr` (Shona): 103k rows
  - `lug_asr` (Luganda): 98.5k rows
  - A Luganda TTS config also exists (`lug` TTS: 2.02k rows).

  These counts include the large **unlabeled** split, so the labelled data is much smaller. For example, `ach_asr` has 109k unlabeled rows and only train 4.11k / val 519 / test 528. — [HF card](https://huggingface.co/datasets/google/WaxalNLP)
- Per-language hours and speaker counts appear only in charts or images in the paper and blog. Neither the arXiv HTML text nor the HF card text gives numbers. — [arXiv v3](https://arxiv.org/html/2602.02734v3); [Google blog](https://research.google/blog/waxal-a-large-scale-open-resource-for-african-language-speech-technology/)
- Sunbird lists training hours of WAXAL plus other sources combined: Lingala 183.1 h, Shona 86.2 h, Luganda 180.7 h. These are not WAXAL-only figures. — [Sunbird model card](https://huggingface.co/Sunbird/asr-whisper-51-african-languages)

**Providers and licences**
- Lingala and Shona: Digital Umuganda ("AfriVoice"), CC-BY-SA-4.0. Luganda: Makerere University ("Yogera Dataset", doi 10.7910/DVN/BEROE0), CC-BY-SA-4.0. — [HF card](https://huggingface.co/datasets/google/WaxalNLP)
- The Google Research blog says the release is CC-BY-4.0. The per-provider table on the HF card says CC-BY-SA-4.0 for these three languages. The card is more specific. — [Google blog](https://research.google/blog/waxal-a-large-scale-open-resource-for-african-language-speech-technology/) vs [HF card](https://huggingface.co/datasets/google/WaxalNLP)

**Recording protocol**
- Participants were shown images on "at least 50 topics" and described them in their native language. Minimum duration is 15 s per recording, captured in the speaker's natural environment. — [arXiv v3](https://arxiv.org/html/2602.02734v3)
- Prompts are images from Google's Open Images. Participants described them rather than reading scripts, which yields natural speech with tonal nuances and **code-switching**. — [Google Research blog](https://research.google/blog/waxal-a-large-scale-open-resource-for-african-language-speech-technology/)
- In the viewer preview, clip durations range from ~3 s to ~66.6 s. — [HF card viewer](https://huggingface.co/datasets/google/WaxalNLP)
- TTS is different: single speakers reading phonetically balanced scripts. — [HF card](https://huggingface.co/datasets/google/WaxalNLP)

**Transcription**
- "About 10%" of the audio was transcribed by paid local experts, one annotator per recording. The rest is the unlabeled split. — [arXiv v3](https://arxiv.org/html/2602.02734v3)
- "Transcriptions were created using the local script where available", otherwise transliterated to the English alphabet. QC checked clarity, language accuracy, prompt relevance and content, and PII was removed. — [arXiv v3](https://arxiv.org/html/2602.02734v3)
- No public guideline was found on casing, punctuation, numerals, or noise/disfluency tags such as `[noise]`. The HF card gives no normalization rules. — [HF card](https://huggingface.co/datasets/google/WaxalNLP)

**Format, fields and splits**
- ASR fields: `id` (e.g. `sna_0`), `speaker_id`, `audio` (example shows `sample_rate: 16_000`), `transcription`, `language` (ISO 639-2), `gender`. — [HF card](https://huggingface.co/datasets/google/WaxalNLP)
- Labelled ASR split is train 80% / validation 10% / test 10%, plus a separate `unlabeled` split. — [HF card](https://huggingface.co/datasets/google/WaxalNLP)
- **Speaker overlap:**
  - In v1, "eval speakers also in train" is **up to 100%**, and 1.09% of eval utterances have an exact transcript twin in train.
  - ASR_v2 (`data/ASR_v2/`) has 0% speaker overlap and 0.09% transcript twins. Its split by duration is 84.5/7.7/7.7, over 438,230 utterances and 2,242 h, with gender matched across splits.
  - Languages that were already speaker-disjoint in v1: Akan, Amharic, Oromo, Sidama, Tigrinya, Wolaytta. **Lingala, Shona and Luganda are not on that list, so their v1 splits overlap in speakers.**
  - Excluded rows include `lin_9193` (empty transcript). — [HF card](https://huggingface.co/datasets/google/WaxalNLP)

### Inferences
- In v1, lin/sna/lug validation sets share speakers with train, so local validation will be **optimistic** compared with a speaker-disjoint hidden test. The Zindi Phase 2 test was new audio (see Q3). The v2 split, or a custom speaker-grouped split, should be used for model selection.
- The speech is spontaneous image description, so expect disfluencies, code-switching (French in Lingala, English in Shona and Luganda) and variable spelling from single annotators. Spelling-convention noise likely inflates WER much more than CER.
- The 1.09% transcript-twin rate shows that prompts or images repeat. An n-gram LM built on train text will partly fit topic vocabulary, which helps on in-domain test data.

### Gaps
- Exact per-language hours, utterance counts in the labelled splits, and speaker counts for lin/sna/lug. These are only in paper figures and would need someone to read the PDF figures or count the dataset locally.
- An official transcription guideline (casing, punctuation, numerals as digits or words, noise tags, code-switch spelling). None was found.

---

## Q2. Published baseline WER/CER for Lingala, Shona, Luganda

### Takeaway
The WAXAL paper text and the Google blog give **no baseline WER/CER**. The best public numbers come from two sources:
- The Makerere benchmark (arXiv 2512.10968, Dec 2025 / DLI 2025). It covers XLS-R, W2v-BERT, MMS and Whisper, with and without a 5-gram LM, on AfriVoice, FLEURS and Common Voice data.
- Sunbird's SunflowerASR (Whisper-large-v3 fine-tuned on 51 African languages including WAXAL).

The official starter notebook (Gemma-3n LoRA) gives Shona WER 52.4% / CER 14.4% on 200 test clips.

### Cited Findings
- The WAXAL paper (arXiv v3 HTML) contains **no ASR baseline results**. — [arXiv v3](https://arxiv.org/html/2602.02734v3)
- The Google blog cites a benchmark of Whisper, XLS-R, MMS and W2v-BERT on 13 African languages but gives no numbers in its text. — [Google Research blog](https://research.google/blog/waxal-a-large-scale-open-resource-for-african-language-speech-technology/)

**Makerere benchmark** ("Benchmarking ASR Models for African Languages", Nahabwe et al., arXiv 2512.10968 / PMLR v302). Values are WER/CER in %. Data was AfriVoice and others, not the WAXAL HF splits. — [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1); [PMLR](https://proceedings.mlr.press/v302/nahabwe26a.html)
- **Lingala** (FLEURS + AfriVoice + LRSC), best at 100 h:
  - XLS-R 18.82/5.69
  - W2v-BERT-LM 22.74/8.52
  - MMS 23.96/8.32
  - W2v-BERT 24.19/9.59
  - Whisper 25.25/10.21
  - XLS-R-LM 25.39/9.25 (the LM *hurts* XLS-R from 5 h upward)
- **Shona** (AfriVoice only, the same source as WAXAL Shona), at 50 h / 100 h:
  - W2v-BERT-LM 8.61/2 and 9.28/2
  - W2v-BERT 23.54/4 and 22.56/4
  - MMS 24.60/4 and 23.70/4
  - Whisper 26.75/7 and 30.92/7
  - XLS-R 30.71/5 and 25.83/4

  The LM cuts W2v-BERT WER by roughly 60% relative from 10 h onward. The authors do not explain the jump at 10 h.
- **Luganda** (FLEURS + Common Voice, read speech), at 200 h:
  - W2v-BERT-LM 16.30/3.72
  - XLS-R-LM 17.43/4.09
  - Whisper 20.45/5.54
  - W2v-BERT 39.75/7.41
  - XLS-R 40.51/7.78
  - MMS 42.69/8.56

  The LM helps every model at every size.
- Setup details from the same paper — [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1):
  - Normalization: lowercase, punctuation removed except apostrophes, non-native diacritics stripped.
  - Audio: clips of 2-30 s kept, resampled to 16 kHz.
  - LM: 5-gram KenLM (modified Kneser-Ney) via pyctcdecode, trained on about 9.03M words (lin), 5.37M (sna) and 9.17M (lug).
  - Training: AdamW, LR 1e-5 to 1e-3, 10% warm-up, batch 64, early stopping with patience 10.
  - Conclusion: MMS and W2v-BERT are best at the lowest data sizes, XLS-R scales best, Whisper is strong at mid sizes, and LM gains depend on how well text and audio match.

**SunflowerASR-51** (Sunbird; Whisper-large-v3, full SFT on 7,412 h, then GRPO RL optimising CER; trained on FLEURS, **WAXAL**, Common Voice and others; Apache-2.0, gated) — [Sunbird model card](https://huggingface.co/Sunbird/asr-whisper-51-african-languages)
- Lingala: WER 0.215 / CER 0.087 (767 test clips)
- Shona: WER 0.194 / CER 0.037 (594 clips)
- Luganda: WER 0.303 / CER 0.077 (950 clips)
- Macro average over 51 languages: WER 0.293 / CER 0.091.
- The language token must be forced at decode time (`lin`, `sna`, `lug`).

**Official starter notebook** (`Waxal_Challenge_Starter_Code.ipynb`, as mirrored in a participant repo) — [notebook](https://raw.githubusercontent.com/Lord-LLM/Google-WAXAL-ASR-Challenge/main/Waxal_Challenge_Starter_Code.ipynb)
- Model: `google/gemma-3n-E2B-it` with LoRA (r=8, alpha=16, target `v_proj`/`o_proj`), LR 1e-3, 500 steps, effective batch 16.
- Shona test (first 200 clips): **WER 52.36%, CER 14.38%**.

**Community checkpoints** — [keystats/waxal-kenlm-models](https://huggingface.co/datasets/keystats/waxal-kenlm-models)
- W2v-BERT-2.0 CTC models: `keystats/w2v-bert-2.0-{lingala,shona,luganda}-main` and `-main-best`.
- Paired 5-gram KenLMs.
- No scores published.

### Inferences
- For Shona, a W2v-BERT-2.0 CTC model plus an in-domain 5-gram LM is the strongest published recipe on AfriVoice-style data. The LM gain there is very large (WER roughly 23% to roughly 9%).
- For Lingala, the LM sometimes hurt in the Makerere study. This may be because of spelling or word-segmentation inconsistency, which matches the winners' notes about Lingala word boundaries (Q4). LM weights need tuning per language.
- The Makerere Luganda numbers are on read speech, so expect much higher WER on WAXAL Luganda (spontaneous speech).
- SunflowerASR already saw WAXAL in training. It is a strong, allowed starting point because it is openly available, though gated. The first-place team used it (Q4).

### Gaps
- No baseline trained and evaluated on the WAXAL HF splits for lin/sna/lug has been officially published by Google.
- Whether Sunbird's test clips overlap WAXAL test is not stated.

---

## Q3. Zindi challenge: metric, submission format, starter notebook, normalization, leaderboard, rules

### Takeaway
The score is a weighted mean, 0.5×WER + 0.5×CER. The Zindi page does not say whether the leaderboard shows raw error (lower is better) or 1 − error. One participant README states **Score = 1 − (WER + CER)/2 (higher is better)** (UNVERIFIED).

The submission CSV has columns `ID,Target`. In the original challenge, the final ranking used a **Phase 2 hidden test (892 new wav files, no metadata)**. According to the first-place team, that test contained only **Lingala and Shona**.

External public data and openly available pretrained models were allowed (with disclosure). Using Phase 1 test labels (public on HF) was forbidden.

### Cited Findings
**Dates and prizes**
- Original challenge: start 26 Jun 2026, close 9 Aug 2026, reveal 10 Aug 2026. Prizes: $4,000 / $2,500 / $2,000, plus $1,500 for the best African participant. — [Zindi page](https://zindi.world/competitions/google-waxal-asr-challenge)
- Zindi moved from zindi.africa to zindi.world (301 redirect). — [Zindi page](https://zindi.world/competitions/google-waxal-asr-challenge)
- Search snippets showed other close dates (2 or 3 Aug), so the page has been edited. — [WebSearch snippets](https://zindi.africa/competitions/google-waxal-asr-challenge)

**Metric**
- "Word Error Rate (WER) and Character Error Rate (CER)… weights 0.5 each… The final score on the leaderboard is the weighted mean of the two evaluation metrics." The page does not say whether higher or lower is better, and does not define normalization. — [Zindi page](https://zindi.world/competitions/google-waxal-asr-challenge)
- A participant repo README states "Score = 1 − (WER + CER) / 2 (higher is better)" (UNVERIFIED; third party). — [Lord-LLM repo](https://github.com/Lord-LLM/Google-WAXAL-ASR-Challenge)
- Starter notebook evaluation — [notebook](https://raw.githubusercontent.com/Lord-LLM/Google-WAXAL-ASR-Challenge/main/Waxal_Challenge_Starter_Code.ipynb):
  - Uses `jiwer.wer` / `jiwer.cer` with default settings.
  - Lowercases both reference and hypothesis and strips whitespace.
  - Does **not** remove punctuation.

  This is the notebook's evaluation, not necessarily Zindi's scorer.

**Public/private split**
- The page contradicts itself: the summary says ~30% public / 70% private, and the full rules say 20% / 80%. — [Zindi page](https://zindi.world/competitions/google-waxal-asr-challenge)

**Submissions**
- 5 per day, 200 in total. Choose 2 for final scoring; otherwise the 2 best public submissions are used. Teams of up to 4. Top 10 must provide code. — [Zindi page](https://zindi.world/competitions/google-waxal-asr-challenge)

**Files**
- `Train.csv` (7.5 MB), `Test.csv` (41.1 KB, Phase 1), `Test_Phase2.csv` (27 KB), `SampleSubmission.csv` (57.7 KB), `Waxal_Challenge_Starter_Code.ipynb` (282.5 KB). — [Zindi data page](https://zindi.world/competitions/google-waxal-asr-challenge/data)
- Phase 1 used the WAXAL HF train/validation/test splits, with the test split scored on the leaderboard.
- Phase 2 was "a new, unseen test set" whose audio was not in any WAXAL split. Audio only, released about 1 week before close, with **no language, speaker or gender metadata**, at `storage.googleapis.com/waxalphase2/newaudios.zip`. **Final rankings were based on Phase 2.** — [Zindi data page](https://zindi.world/competitions/google-waxal-asr-challenge/data)

**Submission format**
- The first-place postprocessor fails if "the schema isn't `ID,Target`", if IDs differ or are duplicated, or if any Target is empty. — [Pasketti postprocess README](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/tree/main/postprocess)
- The Phase 2 test was 892 wav files plus `Test_Phase2.csv`. — [Pasketti README](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place)
- The exact ID string format was not seen.

**Languages actually in the final test**
- The first-place team trained only on lin and sna "since only these two languages were in the final test set". — [Pasketti README](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place)
- Google's winners post also frames the results as Lingala and Shona only. — [Google Africa blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-winners-of-the-waxal-speech-recognition-challenge/)

**External data and model rules**
- "Pretrained models are allowed if openly available to everyone." Open-source tools only, no AutoML, no paid services. — [Zindi page](https://zindi.world/competitions/google-waxal-asr-challenge)
- The data page allows public, legally licensed external speech and text datasets, with disclosure. Using the public ground truth for the Phase 1 test is a breach. — [Zindi data page](https://zindi.world/competitions/google-waxal-asr-challenge/data)

**Participation**
- 1,462 participants from 100 countries, 8,837 submissions. "Top three scores were very close". No public numeric scores. — [Google Africa blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-winners-of-the-waxal-speech-recognition-challenge/); [tech.africa](https://tech.africa/waxal-african-speech-challenge/)

### Inferences
- **Interpreting a ~0.77 score (UNVERIFIED):**
  - If the leaderboard is 1 − mean(WER, CER), then 0.77 means mean error ≈ 0.23, for example WER ≈ 0.33 and CER ≈ 0.13. That is roughly Sunflower-level quality: Sunflower lin gives 1 − (0.215+0.087)/2 = 0.849, and sna gives 0.885 on Sunbird's own test.
  - The starter Gemma Shona result (WER 0.524, CER 0.144) would score about 0.666 under this formula.
  - If the leaderboard were raw error (lower is better), 0.77 would be worse than the starter notebook. A score near 0.77 is therefore most consistent with the **higher-is-better 1 − error** reading.
  - This should be confirmed by submitting `SampleSubmission.csv` or an empty/trivial submission and checking the direction.
- **Casing and punctuation matter (UNVERIFIED):**
  - The first-place postprocessor deliberately adds **initial capitalisation and sentence-final punctuation**.
  - The keystats KenLM repo ships a separate "-best" set trained on **raw, cased, punctuated text**, paired with `-main-best` acoustic models.
  - Both suggest the Zindi scorer is case- and punctuation-sensitive, and that references keep WAXAL's raw casing and punctuation. Outputting lowercase text with no punctuation would then be penalised on both WER and CER.
- Phase 2 gave no language ID, so the pipeline needs language identification (lin vs sna). The first-place team routed clips with a "language-predictions file".
- If the October Study Jam reuses the Phase 1 HF test split, its labels are public on HF and their use is forbidden. Leaderboard scores could be contaminated by others.

### Gaps
- No official Zindi statement on scorer normalization (lowercasing, punctuation) or on score direction was found.
- Discussion threads could not be retrieved: the discussions page rendered no thread content.
- No public leaderboard numbers were found (winning scores were not disclosed).
- No page or rules were found for the "October Study Jam" variant (closing 2026-11-01). Its test set, languages and metric are unverified.

---

## Q4. Winning approaches and tips: WAXAL challenge and comparable African ASR competitions

### Takeaway
WAXAL winners relied on **heavy ensembling of diverse ASR systems**. First place combined 8 model outputs: MMS, IBM Granite, Parakeet, Qwen3-ASR, wav2vec, Whisper, a Whisper SALT noisy-student, and Sunbird's Whisper-51. They used character-level median/weighted ensembling, then orthography-aware postprocessing for Lingala and Shona word-boundary conventions, then capitalisation and punctuation.

Other winners used multi-view decoding with confidence-based selection, and language-specific context lengths. In earlier African ASR competitions, the common winning ingredients were simple ensembles of CTC models (XLS-R, NeMo), noise augmentation, and n-gram LM decoding. Most of those write-ups are older.

### Cited Findings
**WAXAL 1st place, Team Pasketti** (Roman Solovyev "ZFTurbo" and Enes Mahmut Kulak) — [GitHub README](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place)
- Inference models: facebook MMS, IBM Granite, Parakeet, Qwen3-ASR, wav2vec, Whisper, and a Whisper SALT noisy-student. Six were trained, some in two stages (frozen encoder, then full fine-tuning). They also used `Sunbird/asr-whisper-51-african-languages`.
- Trained on lin and sna only.
- Audio was cleaned with BS-RoFormer (MSST) and converted to 16 kHz mono. The README says cleaning was "probably unnecessary because the final test data wasn't noisy".
- Hardware: a single A6000 Blackwell 96 GB.
- Pipeline: 8 jsonl outputs → CSV → `r03_ensemble_subms.py` → postprocess.

**Pasketti postprocessing** — [postprocess README](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/tree/main/postprocess)
- **Weighted character ensemble.** The character median is kept by default. It gives "excellent CER" but can create non-words.
- **Orthography rules:**
  - Lingala function-word splits: `namoni` → `na moni`, `naye` → `na ye`.
  - Lingala `ba/bazo/baza` prefixes are split, but a fixed list of conjugated verbs is rejoined (`bazali`, `batie`, `balati`).
  - Shona auxiliary/progressive splits: `varikufamba` → `vari kufamba`, using a frequency table and a rare `ri + ku…` pattern.
  - Initial capitalisation and sentence-final punctuation are applied.
- **OOV "ghost token" repair.** A one-edit, same-clip alternative is accepted only with enough votes (≥5, or ≥4 with a margin of 2). It is then verified by 6 architecturally diverse systems: 41 token repairs on 39 rows in the winning run.
- No LM generates text. No ablation numbers are reported.

**WAXAL 2nd place** (Alban Nyantudre, Burkina Faso): runs several analyses of the same clip and selects the transcription it trusts most. **3rd place** (Abdourahamane Ide Salifou, Niger): for Lingala, where words blend together, the model sees a longer stretch of speech before committing; for Shona, with clearer word boundaries, it leans more on surrounding context. — [Google Africa blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-winners-of-the-waxal-speech-recognition-challenge/)

**Community LM resources for WAXAL** — [keystats/waxal-kenlm-models](https://huggingface.co/datasets/keystats/waxal-kenlm-models)
- 5-gram KenLMs for lin/sna/lug for pyctcdecode shallow fusion. Text was lowercased, NFKC-normalised, restricted to a fixed alphabet, with no punctuation.
- Built with `lmplz -o 5 -S 40% --discount_fallback`, then `build_binary trie`. A `</s>` unigram was added because pyctcdecode requires it.
- Text sources:
  - lin: WaxalNLP train, KasuleTrevor/lingala_10hr and Lingala_100hrs, Svngoku/lingala-asr-dataset, shunyalabs, Regineforte/asr_lingala
  - sna: WaxalNLP train, badrex/shona-speech, Beijuka/DigitalUmuganda_AfriVoice_shona, shunyalabs, realtime-speech/shona2
  - lug: WaxalNLP train, keystats/luganda_asr_dataset, FarmerlineML/luganda_dataset_2.0, Bateesa/luganda-tts-toby
- WAXAL validation and test were excluded. A separate "-best" set is built on raw cased and punctuated text.

**Comparable competitions** (mostly older; flagged)
- AI4D Tanzania Swahili ASR (Zindi, **2023**, older): the winner's post describes "a simple ensemble of two notebooks": XLSR inference plus an NVIDIA NeMo notebook. — [Zindi discussion (via search snippet)](https://zindi.africa/competitions/ai4d-africas-anglophone-research-lab-tanzania-automatic-speech-recognition-challenge/discussions/18923) (UNVERIFIED, snippet only)
- "Your Voice, Your Device, Your Language" (Swahili, Zindi, **2025**): the winning write-up reportedly used noise augmentation and lightweight fine-tuning. — [Zindi discussion (search snippet)](https://zindi.africa/competitions/your-voice-your-device-your-language-challenge/discussions/29417) (UNVERIFIED, not fetched)
- Other related Zindi ASR competitions that exist (no write-ups retrieved):
  - Mozilla Luganda ASR (older, ~2021)
  - Google ASR Hack Series West Africa / Africa ASR Data challenges (2022, older)
  - Intron AfriSpeech-200 ASR challenge (2023, older)

  — [Zindi Luganda](https://zindi.africa/competitions/mozilla-luganda-automatic-speech-recognition); [AfriSpeech-200](https://zindi.africa/competitions/intron-afrispeech-200-automatic-speech-recognition-challenge); [West Africa ASR](https://zindi.africa/competitions/google-asr-hack-series-west-africa-asr-challenge)
- Makerere benchmark finding: an external LM "helps in some cases, while in others it levels off or adds errors, depending on how well the acoustic and text data line up". — [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1)

### Inferences
- These are evidence-based levers ranked by likely gain for a 0.5·WER + 0.5·CER metric on lin/sna:
  1. Strong pretrained African checkpoints (SunflowerASR Whisper-51, MMS-1B-all adapters, W2v-BERT-2.0 CTC) fine-tuned on WAXAL lin/sna.
  2. KenLM shallow fusion for CTC models, especially Shona.
  3. Multi-system ensembling. A character-level ensemble gives large CER gains; word-level voting (ROVER-style) gives WER gains.
  4. Language-specific word-segmentation postprocessing (Lingala `na`, `ba-` and verb forms; Shona `vari ku-`). Word-boundary disagreements cost WER but almost no CER.
  5. Matching reference casing and punctuation conventions.
- Language ID is required when the test lacks metadata. A mis-routed clip (wrong LM or adapter) can be catastrophic for that clip.
- The winning README suggests audio denoising was not a big lever on the final test.

### Gaps
- No public scores or ablations from any WAXAL winner, so the size of each lever's gain is unknown.
- 2nd- and 3rd-place code repos were not found.
- Swahili 2025 and AfriSpeech winner write-ups were not fetched in full (snippets only).
- No Interspeech/ICASSP 2025-2026 African ASR shared-task results were retrieved within the tool budget.

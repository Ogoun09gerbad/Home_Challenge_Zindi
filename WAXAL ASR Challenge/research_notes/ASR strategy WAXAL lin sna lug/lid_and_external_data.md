# Spoken LID for Lingala / Shona / Luganda and external open datasets to supplement WAXAL

Research date: 2026-10-26. Items marked **[UNVERIFIED]** come from prior knowledge or uploader self-reports and were not confirmed against a primary source in this session.

## 0. Challenge context that drives the LID question

### Takeaway
The Zindi "Google WAXAL ASR Challenge" allows external data if it is public, legally licensed for research or development, and disclosed. The Phase 2 test set comes as **audio only, with no language metadata**. LID, or a language-agnostic model, is therefore required for the final ranking.

### Cited Findings
- External data rule: "Participants may supplement the provided challenge data with other publicly available open-source speech or language datasets. Any external datasets used must be publicly accessible, legally licensed for research or development, and disclosed in the final solution documentation." — [Zindi data page](https://zindi.world/competitions/google-waxal-asr-challenge/data) (zindi.africa now redirects to zindi.world)
- Phase 2 is a new, unseen test set released as audio only about one week before close. "Metadata and auxiliary information such as language, speaker identity, gender" are not provided. Final ranking uses Phase 2 only. Using the public ground-truth labels of the Phase 1 test set breaks the rules and can lead to disqualification. — [Zindi data page](https://zindi.world/competitions/google-waxal-asr-challenge/data)
- Dates shown: start 26 Jun 2026, close 09 Aug 2026, prize USD 10,000. The page shows it as completed. — [Zindi data page](https://zindi.world/competitions/google-waxal-asr-challenge/data)
- Press coverage says the challenge focused on **Lingala and Shona**. It drew 1,462 participants from 100 countries and 8,837 submissions, and results were announced 22 Sep 2026. The winner, Team Pasketti, used an ensemble of eight models. — [af.net / tech.africa](https://af.net/realtime/google-waxal-challenge-tests-lingala-and-shona-speech-ai/); [Google blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-winners-of-the-waxal-speech-recognition-challenge/)
- WAXAL licensing on HF: lin and sna come from Digital Umuganda under CC-BY-SA-4.0. lug comes from Makerere University under CC-BY-SA-4.0, and Luganda TTS data also comes from Loud and Clear. Viewer row counts: `lin_asr` 89k, `sna_asr` 103k, `lug_asr` 98.5k. The card says "Please check the license for the specific languages you are using, as they may differ between providers." It is version 2.0.0, last updated 03/2026. — [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP)

### Inferences
- The press reports name only Lingala and Shona as competition languages. Luganda may be part of the user's own scope (WAXAL has `lug_asr`) rather than an official track. Confirm against the official Zindi overview page.
- The Phase 1 test labels are public on HF (`google/WaxalNLP` test split). Do not train on them; that would be a rule breach.

### Gaps
- I did not find the full Zindi rules page with the metric (likely WER) or a rule on pretrained models. The data page says neither.

## (A1) facebook/mms-lid-* coverage, accuracy, VRAM, and restricting to 3 classes

### Takeaway
`facebook/mms-lid-126` (1B parameters) already contains `lin`, `sna` and `lug`, and so do all larger MMS-LID variants (each larger set includes the smaller ones). Reported accuracy is about 97% over 102 FLEURS languages. Restricting the argmax to the three logits, or fine-tuning a 3-way head, should give higher accuracy in-domain.

### Cited Findings
- The mms-lid-126 label list includes `lin`, `sna`, `lug`. It also includes confusable neighbours such as `nya` (Chichewa), `swh`, `luo`, `kam`, `xho`, `zul`, `nso`, `umb`. Model size is 1.0B parameters in F32, and it needs transformers >= 4.30. Usage: `Wav2Vec2ForSequenceClassification` + `AutoFeatureExtractor`, 16 kHz input. — [HF model card mms-lid-126](https://huggingface.co/facebook/mms-lid-126)
- The 126 languages are "the union of the languages in both FLEURS and VoxLingua-107". Larger models add languages, and "smaller language subsets are included in the larger subsets". — [MMS paper, arXiv 2305.13516, §6.3 and Table 7](https://arxiv.org/abs/2305.13516)
- MMS paper Table 7, test accuracy (%), for the MMS-lab-U+unlab+FL+VL models:

| #lang | FLEURS (102) | VoxLingua (33) | BABEL (23, OOD) | VoxPopuli (25, OOD) |
|---|---|---|---|---|
| 126 | 97.5 | 93.9 | 84.1 | 87.3 |
| 256 | 97.2 | 93.4 | 80.1 | 87.6 |
| 512 | 96.8 | 92.9 | 81.6 | 85.6 |
| 1,024 | 97.0 | 92.8 | 80.5 | 86.2 |
| 2,048 | 97.3 | 92.8 | 81.5 | 86.6 |
| 4,017 | 97.2 | 93.9 | 80.5 | 87.1 |

  — [MMS paper Table 7](https://arxiv.org/abs/2305.13516)
- The same table gives prior-work FLEURS-102 LID accuracy: Whisper 64.5% (82 languages), mSLAM 77.7%, and an MMS baseline trained on FLEURS only 96.2%. On VoxLingua-107, SpeechBrain gets 93.3% and XLS-R 94.3%. — [MMS paper Table 7](https://arxiv.org/abs/2305.13516)
- On out-of-domain conversational telephone speech (BABEL, which includes African languages), accuracy drops to about 80–84%. — [MMS paper Table 7](https://arxiv.org/abs/2305.13516)
- MMS LID training: a linear classifier on the MMS 1B pretrained model, with all parameters fine-tuned, using Adam, LR 1e-5 to 7e-6, 10K–50K updates, on 16 GPUs. — [MMS paper §6.1](https://arxiv.org/abs/2305.13516)

### Inferences
- **Restricting to 3 classes:** load mms-lid-126, find the indices of `lin`, `sna`, `lug` in `model.config.label2id`, and take the argmax over `logits[:, [i_lin, i_sna, i_lug]]`. This removes errors where a clip is mislabelled as another language (e.g. `nya`, `swh`). Expect accuracy above the 97.5% 102-way figure on read speech, but lower on noisy, spontaneous WAXAL-style audio (compare BABEL at 84% over 23 languages). This is my estimate, not a measured number.
- **Better:** fine-tune mms-lid-126 or MMS-300M/1B with a fresh 3-way head on the WAXAL train sets (roughly 90–180 h per language of labelled audio). With this much in-domain data and only 3 classes, clip-level accuracy of ≥99% is plausible for clips of a few seconds or more. Validate on the WAXAL validation splits.
- **VRAM/speed:** 1B parameters is about 4 GB of weights in fp32 and about 2 GB in fp16. Inference on a single consumer GPU (8–16 GB) is fine. Clips can be truncated to the first 10–20 s for LID. No published throughput figure was found.
- The 256 to 4017 variants give no benefit here; they only add confusable classes. Use 126.

### Gaps
- No published per-language LID accuracy for lin/sna/lug specifically. The FLEURS paper's per-language confusion was not retrieved.
- No measured VRAM or RTF figures for mms-lid-126.

## (A2) Whisper language detection

### Takeaway
Whisper's tokenizer has `sn` (Shona) and `ln` (Lingala) but **not `lg` (Luganda)**. Its LID accuracy on FLEURS is poor (64.5% over 82 languages), so it is not a good fit for a 3-way task that includes Luganda.

### Cited Findings
- Whisper's `LANGUAGES` dict contains `"sn": "shona"` and `"ln": "lingala"`, and has no `"lg"` entry. — [openai/whisper tokenizer.py](https://raw.githubusercontent.com/openai/whisper/main/whisper/tokenizer.py)
- Whisper's FLEURS LID accuracy is 64.5% over 82 languages, against 97.5% for MMS-LID-126. — [MMS paper Table 7](https://arxiv.org/abs/2305.13516)
- On FLEURS-54 average WER, Whisper large-v2 scores 44.3 and MMS (LSAH, 1,107 languages) 18.7. — [MMS paper Table 3](https://arxiv.org/abs/2305.13516)

### Inferences
- Luganda clips would be forced into some other language token, likely `sw` or `sn`. You could restrict to {sn, ln} and treat "low probability for both" as Luganda, but that is fragile. Not recommended.

### Gaps
- No per-language Whisper LID accuracy for sn/ln was found.

## (A3) SpeechBrain VoxLingua107 ECAPA, and an SSL-embedding classifier

### Takeaway
The SpeechBrain ECAPA VoxLingua107 model covers Lingala and Shona but **not Luganda**. Its embeddings (or pooled MMS/XLS-R hidden states) can still feed a small 3-way classifier trained on WAXAL. That is cheap, and very high accuracy is expected.

### Cited Findings
- `speechbrain/lang-id-voxlingua107-ecapa`: Lingala and Shona are in the list and Luganda is not. It reports a 6.7% error rate on the VoxLingua107 dev set. `encode_batch` returns utterance embeddings "for building your own language ID model", and the model expects 16 kHz audio. — [HF model card](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
- The MMS paper lists SpeechBrain at 93.3% and XLS-R LID at 94.3% on VoxLingua-107. — [MMS paper Table 7](https://arxiv.org/abs/2305.13516)

### Inferences
- **Recipe:** extract ECAPA embeddings (256-d, about 20M parameters [UNVERIFIED size]) or mean-pooled middle-layer hidden states from MMS-300M / mms-1b / w2v-bert-2.0. Then train logistic regression or a 1-hidden-layer MLP on WAXAL train (lin/sna/lug). Mean-pooled SSL features are known to carry strong language identity. With 3 languages and tens of thousands of training clips, ≥98–99% clip accuracy is a reasonable expectation. This is an estimate, not a measured result; validate on WAXAL validation.
- **Cost:** embedding extraction is one forward pass. A CPU-trainable classifier finishes in minutes. ECAPA runs on CPU in real time.
- Reusing the encoder of your fine-tuned ASR model (pooled encoder states) avoids a separate model entirely.

### Gaps
- No published 3-way lin/sna/lug result was found. Speaker and channel confounds in WAXAL (each language comes from a different provider and recording setup) could inflate apparent accuracy. Check on the held-out validation set and on out-of-provider data such as FLEURS.

## (A4) Text-based LID on first-pass transcripts

### Takeaway
Running a shared multilingual CTC model first and then classifying the transcript with a character n-gram / fastText LID (GlotLID) is a viable alternative. The three orthographies differ clearly. A classifier trained on your own WAXAL transcripts (3 classes) will beat a generic 2000-language model.

### Cited Findings
- GlotLID (`cis-lmu/glotlid`) is a fastText model with more than 2000 labels (V3: 2102), under Apache-2.0. Usage is `fasttext.load_model(hf_hub_download("cis-lmu/glotlid","model.bin"))` then `.predict(text)`. — [HF model card](https://huggingface.co/cis-lmu/glotlid); paper [arXiv 2310.16248](https://arxiv.org/abs/2310.16248)

### Inferences
- **[UNVERIFIED]** GlotLID labels are ISO 639-3 + script (`lin_Latn`, `sna_Latn`, `lug_Latn` expected). Check `languages-v3.md` in the GlotLID GitHub repo.
- Better: train a 3-class fastText or char-n-gram logistic regression on WAXAL transcripts plus the text corpora in section B5. Orthographic cues are distinctive: Shona has digraphs and trigraphs such as zv, sv, dzv, mh, bh; Luganda has doubled vowels/consonants (aa, ee, kk), ky, gw and ŋ/ng'; Lingala has frequent function words (na, ya, ko, ezali, mpe) and ɛ/ɔ in some orthographies. Accuracy on transcripts with ≥5 words should be near-perfect, but short or noisy CTC outputs are the risk. This is an inference.
- **Hybrid:** combine the acoustic LID posterior with the text LID posterior, or pick the language whose per-language model + LM gives the best length-normalised score.

### Gaps
- No published accuracy for text LID on noisy ASR hypotheses for these languages.

## (A5) Single multilingual CTC model vs LID + per-language models

### Takeaway
MMS evidence shows that dense (fully shared) multilingual CTC models lose accuracy through **language confusion**, and language-specific adapters/heads (LSAH) recover it. The effect grows with the number of languages and is small at 61 languages. With only 3 orthographically distinct languages, a shared model is likely competitive. LID + per-language LM/adapter is still usually a little better, especially for n-gram LM decoding, where the LM must match the language.

### Cited Findings
- "Performance degrades quickly for dense models which have no language-specific parameters: for FLEURS-61, CER increases by 5.1 when moving from 61 to 1,107 languages ... This is mainly due to languages being confused with each other ... Language-specific parameters (LSAH) alleviate this issue and show only very little degradation (0.4 CER for FLEURS-61 ...)." — [MMS paper §5.2, Fig. 12](https://arxiv.org/abs/2305.13516)
- FLEURS-54 test WER with n-gram LM: MMS dense 61 languages 20.7 vs MMS LSAH 61 languages 19.1. MMS dense 1,107 languages 24.8 vs LSAH 1,107 languages 18.7. — [MMS paper Table 3](https://arxiv.org/abs/2305.13516)
- MMS adapters add about 2M parameters per language (about 2% of the model), plus a language-specific output layer. — [MMS paper §5.1](https://arxiv.org/abs/2305.13516)

### Inferences
- At 61 languages, LSAH beats dense by about 1.6 WER absolute (20.7 to 19.1). At 3 languages the gap is likely smaller. Note that LSAH results assume **oracle language** at test time, so LID errors would subtract from the gain.
- **Practical design for Phase 2:**
  1. Fine-tune one shared-vocab CTC model (lin+sna+lug) as a robust fallback.
  2. Run LID (fine-tuned MMS 3-way head or an embedding classifier).
  3. Decode with the language-specific KenLM, or with per-language fine-tuned models/adapters.
  4. Optionally decode with all 3 LMs and pick the best normalised score as an implicit LID.
- The 2026 winner used an 8-model ensemble ([af.net](https://af.net/realtime/google-waxal-challenge-tests-lingala-and-shona-speech-ai/)). No public detail was found on whether it used LID.

### Gaps
- No published head-to-head shared vs LID-routed comparison for exactly lin/sna/lug.

## (B) External speech datasets (openly licensed) for ASR

### Takeaway
There is substantial extra open speech data. Lingala has about 105 h AfriVoice (CC-BY-4.0 mirror), plus LRSC, FLEURS, Shunya Labs and BibleTTS. Shona has about 91 h AfriVoice mirror, plus FLEURS and the ASR-Africa benchmark. Luganda is the best served: Common Voice (CC0, large), Sunbird SALT (CC-BY-SA-4.0), FLEURS and BibleTTS. Watch out for duplicates of WAXAL: Digital Umuganda's AfriVoice is also the provider of WAXAL lin/sna, so de-duplicate against the WAXAL test/validation sets.

### Cited Findings (table)

| Dataset | Lang | Size | Domain | License | Location | Notes / source |
|---|---|---|---|---|---|---|
| WAXAL (baseline) | lin, sna, lug | rows: lin 89k, sna 103k, lug 98.5k (may include unlabeled); lin+sna train+val about 33,002 rows / about 179 h per one third-party build | natural/descriptive speech | CC-BY-SA-4.0 | [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP) | hours per [anyantudre/waxal-linsna](https://huggingface.co/datasets/anyantudre/waxal-linsna) (self-reported) |
| AfriVoice Lingala (aggregated) | lin | 110.7 h total (train 100.0 / val 5.1 / test 5.6), 23,539 rows; sources AfriVoice 17,544 rows + LRSC 2,937 + FLEURS ln_cd 2,995 | read/descriptive | CC-BY-4.0 (aggregation); AfriVoice upstream is gated with access terms | [KasuleTrevor/Lingala_100hrs](https://huggingface.co/datasets/KasuleTrevor/Lingala_100hrs) | [card](https://huggingface.co/datasets/KasuleTrevor/Lingala_100hrs) |
| AfriVoice Shona mirror | sna | 15,923 rows, 91.0 h | read/descriptive | CC-BY-4.0 (per mirror) | [realtime-speech/shona1](https://huggingface.co/datasets/realtime-speech/shona1) | figures from [anyantudre/waxal-linsna](https://huggingface.co/datasets/anyantudre/waxal-linsna), uploader self-reported |
| Other Shona HF sets | sna | not verified | various | not verified | `badrex/shona-speech`, `Beijuka/DigitalUmuganda_AfriVoice_shona`, `shunyalabs/shona-speech-dataset`, `realtime-speech/shona2` | used to train [keystats/w2v-bert-2.0-shona-main-best](https://huggingface.co/keystats/w2v-bert-2.0-shona-main-best); check each license |
| Shunya Labs Lingala | lin | 4,341 rows, 21.5 h | ? | "upstream terms" (not verified) | [shunyalabs/lingala-speech-dataset](https://huggingface.co/datasets/shunyalabs/lingala-speech-dataset) | [anyantudre/waxal-linsna](https://huggingface.co/datasets/anyantudre/waxal-linsna) |
| FLEURS | lin (`ln_cd`), sna (`sn_zw`), lug (`lg_ug`) | lin 2,847 rows / 15.0 h; sna 3,704 rows / 15.0 h (per third-party build); lug [UNVERIFIED] about 12–15 h | read Wikipedia sentences (FLoRes) | CC-BY-4.0 [UNVERIFIED this session; widely documented] | [google/fleurs](https://huggingface.co/datasets/google/fleurs) | hours from [anyantudre/waxal-linsna](https://huggingface.co/datasets/anyantudre/waxal-linsna). FLEURS dev/test sentences overlap FLORES-200; keep them out of LM text if evaluating on FLEURS |
| ASR-Africa Data Efficiency Benchmark | sna (config `Shona`) | 1,055 rows, 6.0 h | ? | upstream terms (not verified) | [asr-africa/ASRAfricaDataEfficiencyBenchmark](https://huggingface.co/datasets/asr-africa/ASRAfricaDataEfficiencyBenchmark) | [anyantudre/waxal-linsna](https://huggingface.co/datasets/anyantudre/waxal-linsna) |
| Lingala Read Speech Corpus (LRSC) | lin | 2,937 rows included in the aggregation above | read | not verified | via [KasuleTrevor/Lingala_100hrs](https://huggingface.co/datasets/KasuleTrevor/Lingala_100hrs) | original source/licence not checked |
| Common Voice Scripted Speech (Luganda) | lug | v25.0 about 11.06 GB MP3; grew from 8 h to 80 h in 2021; current hours not retrieved | read prompts, crowd | **CC0-1.0** | [Mozilla Data Collective](https://datacollective.mozillafoundation.org/datasets?page=2) (CV moved to Data Collective; v26.0 latest release) | [Data Collective listing](https://datacollective.mozillafoundation.org/datasets?page=2); [Mozilla blog 2021](https://www.mozillafoundation.org/ca/blog/mozilla-common-voice-adds-16-new-languages-and-4600-new-hours-of-speech/); [v26 release](https://discourse.mozilla.org/t/release-live-mcv-scripted-speech-v26-0-and-spontaneous-speech-v4-0/148687). A curated female-only subset of CV v12 Luganda was 7,842 utterances / about 19.2 h ([PMC12337013](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12337013/)) |
| Common Voice Spontaneous Speech (Shona) | sna | v2.0 sample tiny ("6 clips … 1 hours" per datasheet; v3.0 also exists) | spontaneous | CC0-1.0, but the Data Collective terms forbid re-hosting | [MDC Shona SPS 2.0](https://datacollective.mozillafoundation.org/datasets/cmj8u48ga005pnxzpmsz76k9o) | small; check v3/v4 sizes |
| Sunbird SALT | lug (+ach, lgg, teo, nyn, swa, eng) | 25,000 sentences, about 5,000 with speech per language; 1.74 GB total; no hours given | multispeaker natural (`multispeaker-lug`) + studio TTS (`studio-lug`) + parallel text (`text-all`) | **CC-BY-SA-4.0**; HF gated (agree to share contact info) | [Sunbird/salt](https://huggingface.co/datasets/Sunbird/salt) | [card](https://huggingface.co/datasets/Sunbird/salt) |
| BibleTTS | lin, lug (+8 others) | "up to 86 hours" single-speaker 48 kHz per language; per-language hours not retrieved | Bible reading, studio, single speaker | **CC-BY-SA** (derivative of Open.Bible / Biblica) | paper [arXiv 2207.03546](https://arxiv.org/abs/2207.03546) (download page likely OpenSLR [UNVERIFIED]) | [Interspeech 2022 PDF](https://www.lsv.uni-saarland.de/wp-content/uploads/2022/07/Interspeech2022_BibleTTS.pdf). Single speaker, so limited acoustic diversity; good for LM-domain text too |
| Makerere Radio Speech Corpus | lug | 155 h radio audio, of which 20 h human-transcribed; 16 kHz mono | radio (spontaneous) | **CC BY-NC-ND 4.0**; Zenodo record marked "Restricted" | [Zenodo 5855017](https://zenodo.org/record/5855017); [LREC 2022](https://aclanthology.org/2022.lrec-1.208) | NC-ND and restricted access: likely fails "publicly accessible" and is risky. Avoid or ask organisers |
| Lingala-TTS-Dataset (IADH) | lin | about 4 h 25 min | TTS | NOODL-1.0, research only, no re-hosting | via [MDC search](https://datacollective.mozillafoundation.org/datasets/cmm23jslb003znq07vdska54l) | research-only. Likely allowed under "research or development" but disclose; avoid if in doubt |
| DigitalUmuganda AfriVoice_V2 | Kirundi, Ndau, Ndebele, Oshiwambo | 1,392.76 h | image-prompt | gated | [DigitalUmuganda/Afrivoice_V2](https://huggingface.co/datasets/DigitalUmuganda/Afrivoice_V2) | **not** lin/sna/lug. Ndau is closely related to Shona and could serve as auxiliary pretraining data [inference] |

Not relevant / not covering these languages ([UNVERIFIED], from prior knowledge): AfriSpeech-200 is accented English. ZambeziVoice covers Bemba, Lozi, Nyanja and Tonga (no Shona). NaijaVoices covers Hausa, Igbo and Yoruba. ALFFA covers Amharic, Swahili, Wolof, Fongbe and Hausa. CMU Wilderness is derived from Bible.is recordings, which are not openly licensed, so avoid it. Kencorpus covers Swahili, Dholuo and Luhya.

### Inferences
- **Duplication risk:** AfriVoice (Digital Umuganda) is the same provider as WAXAL lin/sna, and HF mirrors may contain WAXAL clips including test/validation. De-duplicate by transcript text and audio hash against WAXAL val/test before training. Keep anything resembling the Phase 1 test labels out of training (rule breach).
- The `anyantudre/waxal-linsna` repo is a Phase 2 participant's corpus. Its hours are "corpus build statistics from our own build logs", so treat them as self-reported.
- **Licensing for Zindi:** CC0, CC-BY-4.0 and CC-BY-SA-4.0 all qualify. CC-BY-SA requires share-alike if you redistribute derived *data*; model weights are generally treated separately, but disclose anyway. Gated HF repos (SALT, AfriVoice) are still publicly accessible after click-through. NC-ND and "Restricted" Zenodo (Makerere Radio) is doubtful. Research-only (NOODL) is arguably "licensed for research or development" but riskier.

### Gaps
- Current Common Voice Luganda validated hours and speaker count (v25/v26). The Data Collective page was not retrieved and the old GitHub stats JSON returned 404. From prior knowledge [UNVERIFIED], Luganda is among the larger African CV locales (hundreds of hours).
- Per-language BibleTTS hours for Lingala/Luganda (the paper table was not retrieved).
- FLEURS licence and Luganda hours not confirmed on the card (the fetch was truncated).
- Licences of Shunya Labs, ASR-Africa, LRSC and badrex/shona-speech not checked.

## (B5) Text corpora for n-gram LM

### Takeaway
MADLAD-400 (CC-BY-4.0) is the largest verified open source for all three languages. Shona is huge (31.6M clean tokens), while Luganda (6.1M) and Lingala (3.4M) are moderate. Add Wikipedia (CC-BY-SA), eBible text (Lingala CC-BY-SA-4.0), SALT text (CC-BY-SA-4.0), and WAXAL/FLEURS transcripts.

### Cited Findings
- MADLAD-400 per-language sizes (whitespace tokens):
  - Lingala (ln): docs 94.7K noisy / 3.3K clean; sentences 718.7K / 139K; tokens 42.4M / 3.4M
  - Shona (sn): docs 3.1M / 60.2K; sentences 3.1M / 1.2M; tokens 1.3B / 31.6M
  - Luganda (lg): docs 61.1K / 13K; sentences 510.9K / 166.1K; tokens 21.4M / 6.1M

  The license section says CC-BY-4.0 while the metadata tag says odc-by. — [allenai/MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400)
- Sunbird SALT `text-all`: 25,000 parallel sentences including Luganda, CC-BY-SA-4.0. — [Sunbird/salt](https://huggingface.co/datasets/Sunbird/salt)
- Lingala Bible text (Biblica "Mokanda na Bomoi") on eBible is CC BY-SA 4.0 with a share-alike condition. — [ebible.org lin copyright](https://ebible.org/lin/copyright.htm)

### Inferences
- MADLAD "noisy" Shona (1.3B tokens) is mostly noise. Use the clean split, filter with GlotLID or your 3-class text LID, and normalise to the WAXAL transcript conventions (lowercase, punctuation stripped, numbers) before building KenLM.
- [UNVERIFIED, prior knowledge] Also usable: Wikipedia ln/sn/lg dumps (CC-BY-SA-4.0; small, thousands of articles each); CC-100 (includes `lg`, `ln`, and possibly `sn`; no explicit licence beyond Common Crawl terms); Leipzig Corpora (CC-BY for downloads; check lin/sna/lug availability); FLORES-200 (CC-BY-SA-4.0; but FLEURS reads FLORES sentences, so avoid if evaluating on FLEURS); MAFAND-MT (has Luganda, Shona; check Lingala; CC-BY-NC for some news portions); OSCAR (minimal coverage of these languages). Verify each before use.
- Exclude WAXAL Phase 1 test transcripts from LM text (rule breach).

### Gaps
- No verified figures retrieved for Wikipedia, CC-100, Leipzig, OSCAR or MAFAND for these languages.
- Shona news corpora (e.g. Kwayedza) are copyrighted and not openly licensed [UNVERIFIED].

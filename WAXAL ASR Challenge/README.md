# WAXAL ASR Study Jam (October 2026) — solution of OGB2000

| | |
|---|---|
| Final submission | `phase2_clean_F_pk_banoun.csv` (Zindi ID `zvHmM5JA`, 30 Oct 2026) |
| Public / private score | **0.763719** / **0.771129** — private rank **6** |
| Reproduce | `bash reproduce.sh` (one command, 24 min + downloads on one 8 GB laptop GPU) |
| Reproducibility check | full `reproduce.sh` re-run on 31 Oct 2026: log-probs of the 4 models bit-identical (892/892 clips each) and final CSV **byte-identical** to the submitted file |

---

## 1. Overview and objectives
**Problem.** Transcribe Lingala (lin) and Shona (sna) speech from the Phase 2 test audio: 892 clips (~5 h) delivered
**without any metadata** (no language, no speaker). Score = `1 − 0.5·(WER + CER)` (WER on lower-cased text with
punctuation, CER on the raw string), so casing and punctuation matter.

**Objectives.**
1. Generalise to *unseen speakers*: WAXAL validation speakers are almost all in train (EDA: 72/73 lin, 97/100 sna),
   so validation scores are optimistic. Phase 2 is a harder, new-speaker set.
2. Stay strictly within the rules: only openly-available pretrained models, no Phase 1 test labels, no model trained on Phase 2 audio.
3. Remain cheap: inference only, a single 8 GB laptop GPU, 24 minutes end to end.

**Approach in one sentence.** Four public w2v-BERT 2.0 CTC models transcribe each clip; a text language-ID routes the
clip to a language-specific candidate pool; Minimum-Bayes-Risk selection keeps the consensus transcription; deterministic
rules then fix capitalisation and Lingala/Shona word boundaries.

**Expected outcome.** A transcription per clip in WAXAL style (cased, punctuated). Measured: 0.7637 public / 0.7711 private.

## 2. Architecture diagram
```
 EXTRACT                       TRANSFORM / LOAD                 MODELLING (inference only)                     OUTPUT
┌──────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────────────────────┐
│ Phase 2 zip (GCS)    │──►│ decode wav → 16 kHz mono │──►│ 4 × w2v-BERT 2.0 CTC (fp16, GPU)        │
│ newaudios.zip, 892   │   │ int16 memmap + parquet   │   │  lin-specialist  sna-specialist         │
│ wav                  │   │ index (cache_audio.py)   │   │  linsna-afrivoicemix  linsna-soup5      │
└──────────────────────┘   └─────────────────────────┘   │ → log-probs saved (logits/*.npz)        │
                                                         └──────────────────┬──────────────────────┘
                                                                            ▼
                                                         ┌─────────────────────────────────────────┐
                                                         │ greedy re-decodes, blank penalty 0/.5/1 │
                                                         │ → 12 candidate transcriptions per clip  │
                                                         └──────────────────┬──────────────────────┘
┌──────────────────────┐   ┌─────────────────────────┐                      ▼
│ WaxalNLP (HF)        │──►│ train transcriptions     │──►┌─────────────────────────────────────────┐
│ train parquet        │   │ char 1-4-gram Naive Bayes│   │ language ID on soup5 transcription      │
└──────────────────────┘   │ (lid.py)                 │   │ → lin pool (5 cand.) | sna pool (5)     │
                           └─────────────────────────┘   │ MBR: min Σ 0.5·WER+0.5·CER vs. others    │
                                                         └──────────────────┬──────────────────────┘
                                                                            ▼
                                                         ┌─────────────────────────────────────────┐   ┌───────────────┐
                                                         │ capitalise first letter                 │──►│ ID,Target CSV │
                                                         │ lin/sna word-boundary rules             │   │ 892 rows      │
                                                         └─────────────────────────────────────────┘   └───────────────┘
```

## 3. ETL process
### Extract (`prepare_data.py`)
| Source | Format | Volume | Method |
|---|---|---|---|
| Phase 2 audio `https://storage.googleapis.com/waxalphase2/newaudios.zip` | zip of 892 `ID_XXXXXX.wav` | ~1 GB, 5.0 h | HTTP with `Range` resume, 5 retries with back-off, extracted to `data/phase2/` |
| `google/WaxalNLP`, `data/ASR/{lin,sna,lug}/*-train-*.parquet` | parquet (id, speaker_id, transcription, language, gender, mp3 audio) | ~9 GB (only the transcription column is read) | `huggingface_hub.snapshot_download` with file patterns; resumable |
| 4 model checkpoints (Hugging Face) | safetensors + processor | ~2.4 GB each | `snapshot_download` at pinned commits (`dl_members.py`) |

Extraction is a one-off (static test set). The `unlabeled` WaxalNLP files (~78 GB) and Phase 1 test labels are never downloaded/used.

### Transform
- **Audio** (`common.decode_audio`, `cache_audio.py`): decode with libsndfile (fallback librosa), down-mix to mono,
  resample to 16 kHz, store as int16. The w2v-BERT processor then computes 80-dim log-mel filter-banks (stacked to 160-dim
  features); no other feature engineering.
- **Text**: model outputs are cleaned by `common.clean_raw` (Unicode NFC, unified quotes/apostrophes, no space before
  punctuation) — casing and punctuation are kept because the metric counts them. For language ID only, text is
  normalised (`common.normalize`: lower-case, no punctuation).

### Load
- Audio is concatenated into one flat memory-mapped file `data/cache/phase2_p2.int16` with a parquet index
  (`id, offset, length`): constant-time random access, no re-decoding between models, low RAM.
- Model outputs are stored as compressed float16 log-prob matrices (`logits/{model}_phase2.npz`, ~84 MB per model), so
  every downstream step (re-decoding, ensembling, rules) runs on CPU in seconds without re-running the GPU.
- Intermediate transcriptions are plain CSVs in `logs/` (one per candidate), which makes each step inspectable.

## 4. Data modelling
### Models
All four members are **w2v-BERT 2.0** (600 M-parameter Conformer encoder pre-trained on 4.5 M h of unlabelled speech,
`facebook/w2v-bert-2.0`, MIT) fine-tuned with a **CTC** head over a 121-token *raw* character vocabulary (cased letters +
punctuation), published under Apache-2.0 by `anyantudre`. According to their model cards:

| Member | Fine-tuning data (per model card) | WAXAL val score (lin / sna / all) |
|---|---|---|
| `lin-specialist` | from `linsnalug-raw`; AfriVoice Lingala (104.5 h, CC-BY-4.0) + WAXAL Lingala train, then WAXAL only | 0.7673 / 0.8376 / 0.8025 |
| `sna-specialist` | from `linsnalug-raw`; AfriVoice Shona (91 h, CC-BY-4.0) + WAXAL Shona train, then WAXAL only | 0.7352 / 0.8405 / 0.7878 |
| `linsna-afrivoicemix` | from `linsnalug-raw`; WAXAL lin+sna train + AfriVoice Lingala (104.5 h) and Shona (91 h), mixed, 2 epochs | 0.7693 / 0.8123 / 0.7914 |
| `linsna-soup5` | uniform weight average of 5 lin/sna checkpoints (incl. pseudo-label rounds on the WaxalNLP *unlabeled* pool, not test data) | 0.7573 / 0.8425 / 0.8000 |

**We did not train any model for this submission.** Rationale: one 8 GB laptop GPU cannot reproduce the 32 GB-GPU, multi-epoch
recipes of these checkpoints. Our own LoRA fine-tune of soup5 (`train.py`, WAXAL train only, augmentation) did not beat
soup5 on validation (0.7991 vs 0.8000) nor on the leaderboard (0.7498 vs 0.7511), so it was left out (see `JOURNAL.md`).

### Candidate generation (`make_variants.py`)
Each member's log-probs are greedily decoded after subtracting a **blank penalty** `bp ∈ {0, 0.5, 1}` from the CTC blank
log-prob. A larger `bp` makes the model emit more characters; this counters deletions on new speakers and diversifies candidates.

### Language identification (`lid.py`)
Multinomial Naive Bayes on character 1–4-grams of normalised text, add-0.5 smoothing, priors from class frequencies;
trained on all WaxalNLP train transcriptions (lin, sna, lug). Applied to the `soup5, bp=0` transcription of each clip.
Accuracy on WAXAL validation: **99.0 %** (600 clips). Phase 2 prediction: 447 lin / 445 sna (any `lug` would be mapped to `lin`).

### MBR ensemble (`mbr_routed.py`, `ensemble.py`, `configs/F_clean_small.json`)
For each clip, among the candidates of its language, keep the one minimising the sum over the other candidates of
`0.5·WER + 0.5·CER` (i.e. the metric itself used as the MBR loss; equal weights).

| Pool | Candidates (model, blank penalty) |
|---|---|
| Lingala | lin-specialist 0 and 0.5, afrivoicemix 0 and 0.5, soup5 0 |
| Shona | soup5 0, sna-specialist 0 and 0.5, afrivoicemix 1, lin-specialist 0 |

The pools were chosen on WAXAL validation (greedy search over members × blank penalties, see `JOURNAL.md`).

### Post-processing (`postproc.py`, `pasketti_rules.py`)
1. Upper-case the first character.
2. Word-boundary rules (WAXAL train and the Phase 2 references segment some words differently; the model follows train):
   - Lingala: split `namoni → na moni`, `nazomona → nazo mona`, `tozomona → tozo mona`; split `ba` + 43 listed nouns
     (`bandeko → ba ndeko`); join `ba` + 8 listed verbs (`ba zali → bazali`).
   - Shona: split progressive `…ri` + `ku…` (`varikufamba → vari kufamba`); split 8 listed `ne`-forms (`newaya → ne waya`).

### Validation
- Local: 300 random clips per language from WAXAL `validation` (`random_state=42`), scored with `common.score` (same formula as Zindi).
- Public leaderboard: used to decide the word-boundary rules, which validation cannot judge (it follows the train convention, so the rules *lower* the validation score while raising the leaderboard score).

## 5. Inference
- **Infrastructure:** a single machine with one CUDA GPU (≥ 8 GB) — no service, no API. Batches are length-sorted and capped
  at 60 s of audio (`--max-batch-sec`), so memory stays bounded whatever the clip lengths.
- **Input:** a folder of audio files (any format libsndfile/librosa reads; any sampling rate). The file name without extension is the ID.
- **Output:** `submissions/phase2_clean_F_pk_banoun.csv`, columns `ID,Target`; `Target` is the cased, punctuated transcription.
  `logs/mbr_F_phase2.csv` additionally gives, per clip, the predicted language (`lid`) and the selected candidate (`member`).
- **New data:** `python cache_audio.py --audio-dir <folder> --name <name>`, then steps 3–8 of `reproduce.sh` with `--split <name>`.
- **Versioning:** checkpoints are pinned by commit hash (`dl_members.py`, `reproduce.sh`); package versions in `requirements.txt`;
  the ensemble is fully described by `configs/F_clean_small.json`.
- **Updates / retraining:** a new checkpoint is added by running `eval_model.py` on it, `make_variants.py`, and adding its
  candidates to the JSON config after checking the validation score. A single-model fallback is
  `python predict.py --model anyantudre/waxal-w2vbert-linsna-soup5 --audio-dir <folder> --out out.csv`.

## 6. Run time
Measured on the reference machine (section 10) during the full `reproduce.sh` re-run of 31 Oct 2026:
**1425 s (24 min) end to end**, data and checkpoints already downloaded.

| Step | Script | Run time |
|---|---|---|
| 1 | `prepare_data.py --splits train` / `cache_audio.py` | download network-bound (~9 GB); audio caching ~1 min |
| 2 | `dl_members.py` | download network-bound (~9.6 GB) |
| 3 | `eval_model.py` × 4 (GPU, fp16) | lin-specialist 319 s, sna-specialist 378 s, afrivoicemix 283 s, soup5 289 s of inference (RTF 0.016–0.021) + ~20 s model loading each |
| 4 | `make_variants.py` | ~2 min |
| 5 | `mbr_routed.py` (LID training + MBR) | ~1.5 min (CPU) |
| 6–8 | `postproc.py`, `pasketti_rules.py`, `make_submission.py` | a few seconds each |

## 7. Performance metrics
| Submission | WAXAL val (lin / sna / all) | Public | Private |
|---|---|---|---|
| MBR ensemble F | 0.7762 / 0.8451 / 0.8107 | 0.7511 | — |
| F + lin/sna word-boundary rules | 0.7615 / 0.8417 / 0.8013 | 0.7605 | — |
| **F + rules + `ba`+noun split (final)** | 0.7596 / 0.8417 / 0.8003 | **0.763719** (CER 0.1111, WER 0.3615) | **0.771129** (CER 0.1144, WER 0.3433) |

Other metrics used during development: WER and CER separately (the ensemble mainly lowers WER), language-ID accuracy (99 %),
real-time factor (0.018 per model on the GPU), output words per second (used to detect Shona deletions on new speakers).
Validation is ~0.05 above the leaderboard because validation speakers are seen in training.

## 8. Error handling and logging
- Downloads: 5 retries with increasing back-off; the Phase 2 zip and Hub files resume after interruption.
- Audio: an undecodable file becomes an empty clip instead of crashing (`cache_audio._to_int16`).
- Every step prints progress and counts (clips processed, real-time factor, language distribution, selected members,
  rows changed by the rules); `reproduce.sh` runs with `set -euo pipefail`, so any failing step stops the pipeline.
- `make_submission.py` asserts no duplicate ID and no NaN and reports empty transcriptions (0 in the final file);
  with `zindi/SampleSubmission.csv` present it also checks columns, row count and the ID set.

## 9. Maintenance and monitoring
- **Monitoring:** watch the language distribution and the selected-member distribution printed by `mbr_routed.py`,
  and the words-per-second rate of the output; a shift in any of them signals a domain change (new language, new recording conditions).
- **Maintenance:** the word-boundary lists are plain Python dictionaries at the top of `pasketti_rules.py`; they encode
  the orthographic convention of the Phase 2 references and should be revisited if the reference convention changes.
- **Scaling:** inference is embarrassingly parallel per clip and per model — run the four `eval_model.py` calls on four GPUs
  or split the audio folder; MBR already uses a process pool (16 workers). A single-model mode (`predict.py`) costs
  ~0.01 of validation score (soup5 alone 0.8000 vs 0.8107) for 4× less GPU compute.
- **Lifecycle:** adding or removing a member only means editing `configs/F_clean_small.json`; the cached log-probs make
  re-tuning the ensemble a CPU-only task.

## 10. Environment and installation
Reference machine: Windows 11 Pro, Intel Core i9-13900H (14 cores / 20 threads), 64 GB RAM, NVIDIA RTX 4070 Laptop GPU 8 GB,
Python 3.12.10, CUDA 12.8; scripts run from Git Bash. Nothing is Windows-specific: the same commands work on Linux (e.g. Colab/Kaggle GPU).

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
bash reproduce.sh                                         # or: PY=.venv/Scripts/python.exe bash reproduce.sh
```
`requirements.txt` lists only the pinned packages of the final pipeline; `requirements-experiments.txt` adds those of the
experiment scripts. Seeds: `set_seed(42)` (Python, NumPy, PyTorch); decoding is argmax (no sampling); post-processing is rule-based.
A full re-run gave log-probs bit-identical to the original run for all 4 models and a byte-identical submission file.

## 11. Data, pretrained models and compliance
Links and licences: **[DATASETS.md](DATASETS.md)**.

| Rule | How this solution complies |
|---|---|
| Only the specified datasets | Only `google/WaxalNLP` (train transcriptions, for language ID) and the Phase 2 audio are read. |
| Pretrained models openly available | 4 public Hugging Face checkpoints (Apache-2.0), pinned commits; base model `facebook/w2v-bert-2.0` (MIT). |
| Open-source tools, no paid service, no AutoML | Python, PyTorch, Transformers, pandas, jiwer, librosa. |
| No data leak | Phase 1 test transcriptions are blanked at load time (`common.load_split`) and never read. Checkpoints trained on Phase 1 test labels or on Phase 2 audio (`linsna-distilled`, `linsna-soup6`, seeds 43/44/46) were excluded after reading their model cards. |
| All processing in code | Every step from download to CSV is a script called by `reproduce.sh`. |

**Disclosure.** The word-boundary lists re-implement the explicit lists of Team Pasketti's MIT-licensed public solution to
the original WAXAL challenge, whose direction (split or join) that team chose by comparing leaderboard scores on the same
Phase 2 audio; we likewise kept or dropped rule variants from our own public-leaderboard scores. No test transcription is
used to build the rules. Removing step 7 of `reproduce.sh` gives the rule-free "MBR ensemble F" submission (public 0.7511).

## 12. Data insights
`eda.py` produces [reports/EDA.md](reports/EDA.md): ~72 h lin / 81 h sna / 36 h lug of training audio; clips of 18–24 s
on average (max ~45 s); validation speakers almost all present in train (hence optimistic validation); lin/sna recorded at
48 kHz, lug at 16 kHz; heavy punctuation use in Shona (≈ 2.7 full stops per clip) — which is why we keep a cased,
punctuated vocabulary rather than normalising text.

## 13. Repository files
| File | Role |
|---|---|
| `reproduce.sh` | end-to-end pipeline of the final submission |
| `common.py` | paths, seeds, text normalisation, Zindi metric, WaxalNLP loader (test labels blanked), audio decoding |
| `prepare_data.py`, `cache_audio.py`, `dl_members.py` | extraction and audio cache |
| `predict.py`, `eval_model.py` | model loading, batched CTC inference, greedy decoding |
| `logit_ens.py`, `make_variants.py` | blank-penalty re-decoding |
| `lid.py`, `ensemble.py`, `mbr_routed.py` | language ID and MBR selection |
| `postproc.py`, `pasketti_rules.py`, `make_submission.py` | post-processing and CSV writer |
| `configs/F_clean_small.json` | final ensemble configuration (other configs: experiments) |
| `eda.py`, `reports/EDA.md` | data exploration |
| `train.py`, `rover.py`, `build_lm.py`, `decode_lm.py`, `median_ens.py`, `whisper_infer.py`, … | experiments **not** in the final submission (results in `JOURNAL.md` / `SUBMISSIONS.md`, in French) |

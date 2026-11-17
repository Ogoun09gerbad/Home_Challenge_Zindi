# JOURNAL

## 2026-10-26
- Env: venv `.venv` (Python 3.12.10, uv), torch 2.11+cu128 (CUDA OK, RTX 4070 Laptop 8 GB), transformers 5.20, datasets 3.6, numpy 2.5. requirements.txt written.
- kenlm: build impossible on Windows (no MSVC). N-gram LM to be done under WSL/Linux or after installing VS Build Tools.
- WaxalNLP: parquet schema = id, speaker_id, transcription, language, gender, audio{bytes,path (mp3)}. Splits train/validation/test (+ unlabeled ~78 GB, excluded). Languages: lin, sna, lug.
- `prepare_data.py` download started in the background (~12.6 GB labeled, slow: ~1 h), log in logs/prepare_data.log.
- `common.py`: seeds, normalization, metric 0.5*WER+0.5*CER, parquet loading + audio decoding — tested on lin-test.
- BLOCKER: ./zindi/ empty (Train/Test/Test_Phase2/SampleSubmission + starter notebook). Submission format and score display (raw or 1-error) not verified.
- WARNING Zindi rule: the WaxalNLP `test` split has public transcriptions; do not use it for training (if it matches the Phase 1 test). To be confirmed with Test.csv.
- Challenge info (2026-10-26): Phase 1 test = WaxalNLP `test` split (public transcriptions => FORBIDDEN for training/tuning). Train on `train`, monitor on `validation`. Phase 2 = audio only (newaudios.zip), without language/speaker => LID or a single multilingual model. Closes 2026-11-01 (≈6 days).

### 2026-10-26 (afternoon) — research + first measurements
- Deep research: reports/ASR strategy WAXAL lin sna lug.md. Key points: score = 1 - 0.5*(WER+CER) (higher = better); WER on lowercase WITH punctuation, CER on raw text => output with case+punctuation ("raw" vocab). Original Phase 2: lin + sna only.
- User decision: start from the public checkpoint anyantudre/waxal-w2vbert-linsnalug-raw (Apache-2.0). Its model card: WAXAL validation removed from training => our validation is clean for it.
- Compliance: test split transcriptions removed from the cache and from load_split (never exposed to the code).
- kenlm compiled on Windows (MSVC 2026, DISTUTILS_USE_SDK=1); ARPA files produced by build_lm.py (interpolated KN, pure Python, normalization checked: probabilities sum to 1).
- Phase 2: 1785 wav (data/phase2/newaudios/ID_XXXXXX.wav).

| Exp | Model | Decoding | lin score | sna score | lug score |
|---|---|---|---|---|---|
| E1 | anyraw (linsnalug-raw) | greedy | 0.7421 (WER .364 CER .152) | 0.8369 (WER .258 CER .068) | 0.8974 |
| E2 | anyraw | beam 48 without LM | 0.7446 | 0.8382 | — |
| E3 | anyraw | beam + 4-gram train LM (best alpha) | 0.7435 (α .2) ↓ | ≤0.8382 (LM hurts) | 0.9039 (α .2) |
- Conclusion: the n-gram LM on train text does not help (lin/sna) — the model has already absorbed that text. Dropped except for lug.
- Text LID (character n-gram NB on the hypothesis): 100 % on 900 val clips (lin/sna/lug).
- Next step: MBR ensemble of public Apache-2.0 checkpoints from the same author (seeds/soups/specialists), evaluated on val lin+sna.

### 2026-10-26 (evening) — MBR ensemble + first Phase 2 submission
- 10 public Apache-2.0 checkpoints (anyantudre/waxal-w2vbert-*) evaluated on validation lin+sna (300 clips/language):
  best single: lin-specialist 0.8025, soup6 0.8003, soup5 0.8000, distilled 0.7988, afrivoicemix 0.7914.
- MBR ensemble (selection by minimizing pairwise 0.5*WER+0.5*CER risk) over the 6 best
  (lin-specialist, afrivoicemix, distilled, soup5, soup6, sna-specialist): **0.8039** on validation (lin 0.7650, sna 0.8425).
- N-gram LM (build_lm.py, KN order 4): only helps lug (0.8974 -> 0.9039, alpha=0.2); hurts lin/sna (already seen by the public checkpoints). Not used for lin/sna.
- Phase 2 inference (892 clips, 5h, data/phase2/newaudios) by the 6 members (~8-15 min/member depending on download), MBR ensemble -> logs/ens_phase2.csv.
- Persistent BLOCKER: zindi/ still empty. submissions/phase2_ens_top6mbr.csv produced in the default format (id, transcription) — FORMAT NOT VERIFIED against SampleSubmission.csv. Not yet submitted on Zindi (public score unknown).
- To do: check the exact columns expected by Zindi (ID vs id, Target vs transcription) before any real submission; text LID + lug LM on Phase 2 if it contains lug clips (report says lin+sna only for the original edition, to be reconfirmed).

### 2026-10-26 (evening, continued) — submission format fix
- First real upload rejected by Zindi: "Submission is missing the following columns: ID, Target" -> exact format confirmed: columns `ID` and `Target` (uppercase), not `id`/`transcription`.
- Fixed in make_submission.py and predict.py (fallback without SampleSubmission.csv). Submission regenerated: submissions/phase2_ens_top6mbr.csv (892 rows, columns ID,Target, 0 empty).
- WARNING metric: the Zindi page only says "the final score is the weighted average of WER and CER" (0.5/0.5), without specifying whether it is the raw error (lower = better) or 1-error (higher = better, as assumed so far from a participant's model card). Still to be confirmed empirically with the public score returned for this upload. If it is the raw error, the "~0.77" target given at the start would be a very bad score (77% error) — implausible — so the "1-error" hypothesis remains the most likely, to be checked as soon as the public score comes in.

### 2026-10-26 (night) — ensemble calibration (rank 7, 0.7469 public)
- Public score of submission 1: 0.746912 => score = 1 − 0.5(WER+CER) confirmed. #4 = 0.7601.
- Frame-level log-prob averaging (6 members, identical 121-token vocab): HURTS (0.7919 at best, 0.728 without blank penalty) — CTC spikes misaligned between independent fine-tunes (cf. Kurata 2019). Dropped.
- Per-member blank penalty (grid −0.5..1): gains ≤ +0.003 except afrivoicemix sna (+0.024 at 0.75). Used as a candidate generator (re-decodings) for the MBR.
- MBR (val lin+sna, 300/language): base6 0.8039; 12 cand. 0.8041; 18 cand. 0.8019; LID-routed 11 cand. 0.8057; **LID-routed small (E) 0.8093** (lin 0.7761).
- Text LID on the hypothesis: 99 % on val; Phase 2: 447 lin / 445 sna.
- Post-processing (initial capital, final period, comma removal): no effect (±0.0005) — the models already emit case/periods.
- Submission 2 ready: submissions/phase2_mbrE_routed.csv.

### 2026-10-27 — COMPLIANCE: contaminated members excluded
- HF model cards read (logs/provenance*.txt): `linsna-distilled` = trained on the 892 Phase 2 clips (ensemble pseudo-transcriptions); `linsna-soup6` = average including distilled, with a data table listing the WaxalNLP test split (3,615 rows, forbidden Phase 1 labels); seeds 43/44 (and 46) trained on the Phase 1 test split (gold). => PERMANENTLY EXCLUDED: distilled, soup6, seed43/44/46, whisper-turbo, all yehoshua01/*.
- Clean members (WAXAL train+val excluding holdout/val, AfriVoice, pseudo-labels on the *unlabeled* pool): lin-specialist, sna-specialist, afrivoicemix, soup5, linsnalug-raw.
- IMPACT: submission 1 (0.746912, rank 7) and config E used distilled/soup6 => NOT to be kept as final submission; E not submitted.
- Clean ensemble F (lin: lin-spec bp0/.5 + afrivoicemix bp0/.5 + soup5; sna: soup5 + sna-spec bp0/.5 + afrivoicemix bp1 + lin-spec): val 0.8107 (lin 0.7762, sna 0.8451); base 5 members without variants: 0.8039.

### 2026-10-27 — clean leads (external members, post-processing, median)
- Clean submission ready: submissions/phase2_clean_F.csv (val 0.8107).
- Loop-collapse (anyantudre): 2 val clips / 3 Phase 2 clips modified, no effect (-0.0001) => not used (useful for them because blank-penalty 2.0 members loop).
- Word boundaries from train frequencies (wordfix.py, split/merge, ratios 2-10): no effect (±0.0003); the Pasketti rules had their direction chosen by probing the leaderboard => only to be tested via real A/B if the submission budget allows.
- Clean external member `keystats/w2v-bert-2.0-multilingual-waxal-shona-lingala-luganda-best` (WAXAL train only, Apache-2.0): val lin 0.7333 / sna 0.8304 alone; added to MBR F: 0.8105-0.8109 (no gain).
- Character-level string median (median_ens.py, in the style of Pasketti `median_extended`): 0.809 (F) / 0.8007 (base5) < MBR (0.8107 / 0.8039). Members too correlated.
- whisper-large-v3 LoRA (mrwalker123): model card says WAXAL train+validation => validation unreliable for it; not evaluated.
- omniASR (DariusTheGeek): requires fairseq2 (WSL2, no sudo here); not attempted.
- In progress: speed TTA 0.9/1.1 on soup5 and lin-specialist (run_tta.sh); keystats Phase 2.
- Speed TTA (0.9/1.1) soup5 alone: 0.7972 / 0.7958 (base 0.8000); added to MBR F: 0.8098 < 0.8107 => no gain, dropped (lin-specialist runs stopped). keystats Phase 2 finished but not used (no val gain).
- Summary: the ensemble of clean anyantudre members saturates around 0.810 on val. Best clean submission: submissions/phase2_clean_F.csv.
- Whisper large-v3 + LoRA swa (mrwalker123, beam 3, language via text LID): val lin 0.6596 / sna 0.7616 / ALL 0.7122 (val INFLATED: it saw train+val) — well below the CTC models (0.776 / 0.845). Added to MBR F: 0.8095 (weight 1) / 0.8102 (weight 0.5) vs 0.8107 => no gain, even with an optimistic val. Phase 2 inference (≈50 min) NOT run.
- Lesson: the external families tested (keystats, Whisper) are too weak to correct the errors of the anyantudre members; the MBR only picks them on ~3 % of clips.

### 2026-10-27 — public score of F, Shona deletion test
- phase2_clean_F.csv: **0.751052** public (CER 0.1135, WER 0.3844) vs val 0.8107 (gap −0.060). Rank ≈ 7 (#6 = 0.7526, #4 = 0.7601).
- Speech rates (words/s): train ref lin 1.527 / sna 1.152; val ref lin 1.469 / sna 1.145; outputs at bp=0.5 match the ref on val. Phase 2 with bp=0: lin 1.47-1.50 (OK), sna 1.09-1.10 (< 1.15) => likely Shona deletions on new speakers.
- A/B candidates for the leaderboard: N (sna bp 1/1.5, rate 1.125) val 0.8071; M (sna bp 1.5/2, rate 1.132) val 0.8051. If public goes above 0.7511 => the deficit does come from deletions and val is misleading.
- Leaderboard test (file N or M, to be confirmed by the user): 0.752102 (CER 0.1139, WER 0.3819) vs F 0.751052: WER −0.0025, CER +0.0004 => +0.001, within noise (±0.003).
- Uniform weight averaging of 5 clean checkpoints (make_soup.py -> models/soupC: raw, lin-spec, sna-spec, afrivoicemix, soup5): val lin 0.7615 / sna 0.8432 / ALL 0.8024 (≈ best single member). In MBR F: 0.8094 (weight 1), 0.8072 (weight 2) < 0.8107 => no gain. Phase 2 not run.

### 2026-10-27 — clean LoRA fine-tuning (ft1)
- Base soup5 + LoRA r32 (22.7 M trainable params), lin+sna WAXAL train only (27,619 clips, 143.7 h available; ~3,500 clips seen), speed/gain/noise augmentation + SpecAugment, lr 1e-4, 220 steps (eff. batch 16, ~12 s/step, ≈45 min). Bug fixed along the way: CTC head with 123 outputs (2 tokens added) vs vocab.json with 121 -> random re-initialization; we now reuse the checkpoint's processor.
- Val (300/language): lin 0.7571 / sna 0.8409 / ALL 0.7991 vs soup5 0.7573 / 0.8425 / 0.8000 => no gain. Pass criterion (ALL ≥ 0.802) not met; Phase 2 not run.
- Limitation: val (speakers already seen in train) does not measure the robustness gain targeted by augmentation; only a leaderboard upload could show it.
- ft1 Phase 2 (526 s, RTF 0.029); bp 0-2 variants generated (make_variants includes m_ft1). MBR R (F + 5 ft1 candidates) val 0.8073 (dilution); Q (F + ft1 greedy) val 0.8093 => submission submissions/phase2_clean_FT.csv (config Q).
- Public score phase2_clean_FT.csv: **0.749777** (CER 0.1136, WER 0.3869) < F 0.751052 < sna-bp file (N or M) 0.752102. Short LoRA fine-tuning does not improve robustness to new speakers; lead dropped.
- Public ranking of clean submissions: sna-bp (0.7521) > F (0.7511) > FT (0.7498). Gap to rank 4 (0.7601): ≈ 0.008.

## 2026-10-30
- Deep research: reports/ASR low-resource WAXAL techniques.md. Remaining levers identified: anchored character ROVER (anyantudre) and word-boundary rules (Pasketti). KenLM/LID/pseudo-labels/TTA/soups already closed.
- `rover.py` (anchored character ROVER, reimplementation of anyantudre's MIT algorithm + margin option). Validation (lin+sna, 300/language):
  - 5 clean members bp0, anchor soup5: 0.8008; ×{0,0.5}: 0.8020; ×{0,1,1.5,2} (20 members, threshold 2): 0.7902.
  - Anchor = output F, margin 3, members bp0: 0.8108 (= F 0.8107); + members bp 1–2: 0.7939–0.8047.
  - LID-routed (anchors lin-spec / soup5): ≤ 0.8071.
  - => voting does not beat MBR F on validation: members too correlated (same linsnalug-raw lineage). Report criterion (≥ 0.8137) not met.
  - Shona-only variant (anchor F + 16 members bp 0–2, margin 4): val −0.0021 (sna −0.0045); Phase 2 sna +1.9 % words (1.10 → ≈1.12 words/s). A/B test only possible on the LB.
- `pasketti_rules.py` (MIT rules, confirmed+baverb stages, without corpus/pool tables): lin namoni/nazomona/tozomona → split, ba+{zali,lati,komi,fandi,vandi,sali,telemi,lakisi} → join; sna `^(\w*ri)(ku\w{3,})$` → split, 8 ne- splits. Val: 0.8107 → 0.8012 (expected: val follows train, where namoni is joined 73 % of the time). Phase 2 triggers: namoni 223, nazomona 128, tozomona 17, ba+verb 1, sna progressive 153, ne- 15 (Pasketti: 208/126/118 on the same test => same audio).
- Final period: F already reproduces the train rate (lin 67 %, sna 95–100 %); cap_after_stop neutral on val (−0.00003). Not added.
- Files ready: submissions/phase2_clean_F_pklin.csv (216 lines ≠ F), phase2_clean_F_pk.csv (330), phase2_clean_F_snarover_pk.csv (504).
- **phase2_clean_F_pk.csv: 0.760535 public (rank 5)**, +0.0095 vs F. Rank 3 target: 0.766456 (≈ +0.006). Val is inverted for orthography: only the LB can judge these rules.
- New probes (each isolated vs F_pk): `--final-stop` (146 lines), `--ba-noun` (ba+noun, manual list: frequent root standalone in train; verbs excluded because the test joins ba+verb), `--bazo-split --na-split` (bazo+verb from the train table, analogous to nazo/tozo mona). Val (blind): 0.8011 / 0.8002 / 0.7991 vs 0.8012.
- Rejected: Shona -ri + non-ku word split (mostly rwegoridhe = concord+colour, −0.0085 for Pasketti; and real words like chirimwa).

## 2026-10-31 — final result and Zindi code review
- Private leaderboard: **rank 6** with phase2_clean_F_pk_banoun.csv (public 0.763719, private 0.771129). Code + docs to upload before 2026-11-02 00:15 GMT.
- Review preparation (Zindi "documentation" and "code review" articles): `reproduce.sh` (full pipeline, 4 clean members at pinned commits), `requirements.txt` with pinned versions (pipeline only) + `requirements-experiments.txt`, text LID moved into `lid.py` (no more kenlm/pyctcdecode dependency), imports at the top and English docstrings in the pipeline scripts, `prepare_data.py --splits`, `postproc.py --mode cap`, English README in 13 sections (Zindi article structure), DATASETS.md in English (AfriVoice sources of the checkpoints, statement on the Pasketti rules).
- Verification: full re-run of `reproduce.sh` (1425 s) → logits of the 4 models bit-identical (892/892), 15 intermediate files identical, final CSV byte-identical.
- Archive: OGB2000_WAXAL_solution.zip.

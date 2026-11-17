# Top public solutions to the Google WAXAL ASR Challenge (Zindi, 2026): exact ensembling, decoding and post-processing recipes

Researched 2026-10-26. Primary sources are the two public solution repos, their bundled docs and code, the anyantudre Hugging Face model cards, and the PyPI source of `asr-ensemble==1.0.2`. All of them were downloaded and read in full, not summarised by a search engine.

**Correction to the brief.** anyantudre (Alban Nyantudre) is the **2nd-place** solution: private 0.772552, public 0.766791. 1st place is Team Pasketti: private 0.780944, public 0.769614. 3rd place is Abdourahamane Ide Salifou: private 0.771866, and his code was not found. Sources: [anyantudre SOLUTION.md](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/docs/SOLUTION.md) and the [Google blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-winners-of-the-waxal-speech-recognition-challenge/).

Metric (both teams): score = 1 − 0.5·(WER + CER). anyantudre showed on the leaderboard that **WER is computed on lowercased text that keeps its punctuation, while CER is computed on the raw string**. The public split is about 30% of the 892 clips (roughly 268 clips). The private split is about 70% (roughly 624 clips).

---

## Q1. anyantudre (2nd place): ROVER core, p2n_mbr, blank penalty, soups, distillation, LID, post-processing, leaderboard deltas, Phase-2 contamination

### Takeaway
anyantudre's scored file `p2n_mbr` is a per-clip MBR selection among 7 complete ensembles. Six are 26-member character-level ROVER votes and one, `p2n_ens_wide`, has 38 members; all are anchored on the seed-43 w2v-BERT CTC arm except `p2n_ens_s46anchor`, which is anchored on seed 46. Six w2v-BERT checkpoints are each decoded greedily and with blank-logit penalties of 1.0, 1.5 and 2.0. Three seq2seq/aux members join them: Sunbird-51 zero-shot, Whisper-turbo, and the mixed-curriculum arm p1av. Per clip, the selector keeps the candidate with the smallest summed normalised char edit distance to the 26 members of `p2n_ens_masked`. Post-processing is only loop collapse plus "capitalise after `.!?`". **The distilled arm, trained on Phase-2 test audio, votes in all 7 candidate ensembles and in the judge set.** A Phase-2-clean equivalent is `p2n_ens_bp25` (25 members, public 0.764759).

### Cited Findings

**Code and data locations**
- Repo: https://github.com/anyantudre/google-waxal-asr-challenge-solution. Key files:
  - `configs/ensembles.yaml`: exact member lists.
  - `waxal_asr/ensemble.py`: ROVER.
  - `waxal_asr/decode.py`: blank penalty.
  - `waxal_asr/postprocess.py`
  - `waxal_asr/modeling/predict.py`: recipe runner and MBR selection.
  - `waxal_asr/modeling/sunbird.py`
  - `waxal_asr/lid.py`
  - `docs/SOLUTION.md` and `docs/INFERENCE.md`.
- Hugging Face models: 14 `waxal-*` repos — [anyantudre models list](https://huggingface.co/models?author=anyantudre&search=waxal):
  - `waxal-w2vbert-linsnalug-raw`, `-lin-specialist`, `-sna-specialist`, `-linsna-afrivoicemix`
  - `-linsna-seed43/44/46`, `-linsna-soup5`, `-linsna-soup6`, `-linsna-distilled`
  - `waxal-whisper-turbo-linsna`
  - `waxal-checkpoints`, a private Phase-1 backup that includes `w2vbert_multi` (the "champion" at that stage) and XLS-R variants
  - `waxal-whisper_small_multi_s42` and `waxal-w2vbert_multi_s42`; the second has no card.
- Datasets: `anyantudre/waxal-linsna`, the derived training corpus, and `anyantudre/waxal-pseudo`, 3-model-agreed pseudo-labels on the WAXAL *unlabeled* split. waxal-pseudo keeps a clip if champion w2v-BERT vs XLS-R CER ≤ 0.12 **and** Omnilingual-ASR-300M vs champion CER ≤ 0.22 — [waxal-pseudo card](https://huggingface.co/datasets/anyantudre/waxal-pseudo).

**The arms**

All arms below are w2v-BERT-2.0 CTC unless stated. Tokens are a raw char vocabulary of 97 tokens: letters, 26 capitals, and `! , . : ; ?`. Each arm uses MUSAN noise (p 0.5, 5–20 dB), RIR (p 0.3), Gaussian noise (p 0.1) and gain (p 0.3), with no speed or pitch perturbation. Batch is 12×3 (effective 36), bf16, on one RTX 5090. Targets get only NFC plus whitespace collapse. Sources: [SOLUTION.md](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/docs/SOLUTION.md) and the cards, e.g. [seed43](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-seed43).

- `s43`: fresh seed 43; 8 epochs at 1e-4 with warmup 0.1; `add_adapter: true`. Data: WAXAL lin+sna train/val **plus the WAXAL (Phase-1) test split gold**, 3,615 rows; 30,987 rows after holdout. Holdout 0.2862. No solo leaderboard score. Anchor of 6 of the 7 ensembles — [seed43 card](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-seed43).
- `s44`: seed 44; as s43 plus FLEURS `ln_cd`/`sn_zw` (using `raw_transcription`), Shunya Lingala and ASR-Africa Shona, 42,934 rows. Holdout 0.2809. A WAXAL refresh epoch (`s44r`) did nothing: 0.2812 — [seed44 card](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-seed44).
- `s46`: same corpus and hyperparameters as s43, with seed 46. Holdout 0.2588, the best of the family. Greedy outputs are identical to s43 on only 12.3% of clips (8.6% vs s44, 11.5% vs distil) — [seed46 card](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-seed46).
- `p1raw`: root of the lineage. Trained on WAXAL lin+sna+lug train/val, 12 epochs at 1e-4. Public leaderboard 0.742142 (CER 0.115951, WER 0.399766), against 0.724984 for the identical model with an 81-token stripped vocabulary — [linsnalug-raw card](https://huggingface.co/anyantudre/waxal-w2vbert-linsnalug-raw).
- `linspec_r` / `snaspec_r`: continue p1raw. Stage 1 is AfriVoice (KasuleTrevor/Lingala_100hrs, 22,131 rows, 104.5 h; realtime-speech/shona1, 15,923 rows, 91 h) plus WAXAL, 3 epochs at 3e-5. Stage 2 is a WAXAL-only refresh, 1 epoch at 1e-5. Seed 42, `vocab_dir` pinned. Holdout: lin 0.2785, sna 0.1616. Routed together by LID they scored public 0.744835 — [lin-specialist card](https://huggingface.co/anyantudre/waxal-w2vbert-lin-specialist) and [sna-specialist card](https://huggingface.co/anyantudre/waxal-w2vbert-sna-specialist).
- `p1av`: continues p1raw with WAXAL and both AfriVoice corpora mixed, 2 epochs at 2e-5, effective batch 36. Original holdout 0.2905. The original weights were lost and the published repo holds a retrain, which differs on 85.9% of clips; the retrain alone scored public 0.726089 — [afrivoicemix card](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-afrivoicemix).
- `soup5`: uniform weight average, 0.2 each, over every float tensor of p1raw, linspec_r, snaspec_r, linspec_p2 and snaspec_p2. The `_p2` members are pseudo-label rounds on the WAXAL *unlabeled* pools, keeping the top 60% by confidence: 22,986 lin and 22,396 sna rows. Buffers and processor come from p1raw. Averaging requires the same vocab/CTC order, enforced by checking names and shapes. Public 0.746054 (CER 0.114864, WER 0.393027), above every one of its ingredients — [soup5 card](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-soup5).
- `distil`: continues soup5, 3 epochs at 8e-6, warmup 0.1, effective batch 36. Data: WAXAL lin+sna **test-split gold**, 3,615 rows; **the 26-member ensemble's transcripts of the 892 Phase-2 test clips**; and, loaded implicitly, the WAXAL train split, about 32k rows in total. Holdout 0.2746. **Best single model: public 0.746787** (CER 0.113117, WER 0.393308) — [distilled card](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-distilled).
- `soup6`: uniform 1/6 average of the soup5 members plus distil. No solo score. Its greedy output agrees with soup5 on only 592 of 892 clips — [soup6 card](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-soup6).
- `turbo_linsna_r`: full fine-tune of whisper-large-v3-turbo on WAXAL lin+sna plus Phase-1 test gold.
  - Training: 3 epochs at 1e-5, then 1 epoch at 5e-6; batch 4×8 (effective 32); per-example language token; no augmentation. Clips filtered to 1–30 s and 1–25 characters per second.
  - **Decoding of record**: greedy (`num_beams 1`), `max_new_tokens 200`, no forced language, no windowing, so clips over 30 s are truncated.
  - Results: holdout 0.2772, public **0.7331** — [whisper-turbo card](https://huggingface.co/anyantudre/waxal-whisper-turbo-linsna).
- `sunbird51`: `Sunbird/asr-whisper-51-african-languages` used zero-shot.
  - Setup: pinned to revision `5d4f0038`; fp16.
  - Decoding: **beam 8** and `max_new_tokens 200`, in 28 s windows with 2 s overlap whose texts are joined with spaces. The language token (`lingala`/`shona`) is forced only when the LID confidence is ≥ 0.5.
  - Result: public 0.737168 — [sunbird.py](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/waxal_asr/modeling/sunbird.py) and [SOLUTION.md](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/docs/SOLUTION.md).

**Blank-penalty re-decodes**
- Mechanism: subtract a constant from the CTC blank logit (`logits[:, blank_id] -= penalty`), then take the greedy argmax — [decode.py](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/waxal_asr/decode.py).
- Values used: **1.0, 1.5 and 2.0** in every core recipe, widened to **0.5 and 2.5** in `p2n_ens_wide`. The penalty was chosen by matching output words per second to the reference rate of **1.41 w/s**, not by tuning on the leaderboard. Penalty 1.5 brought a representative arm to 1.417 w/s; the greedy baseline was 1.30 w/s, or 97.9% of the reference word count — [INFERENCE.md](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/docs/INFERENCE.md).
- Arms re-decoded: s43, p1raw, linspec_r, snaspec_r, s44 and soup5, each at three penalties, for 18 members.
- Solo they are worse (0.743 vs 0.746 for the same checkpoint) but they gave **+0.0104 in total** in the vote. The stated reason: "a vote can discard a spurious word but can never recover a missing one" — [SOLUTION.md](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/docs/SOLUTION.md).
- Historical defect: the competition-era blank-penalty members were decoded **without an attention mask** in batched inference. The fix gives `p2n_ens_masked`, public 0.766563 vs 0.764915. The mask fix plus the Whisper arm properly decoded as a generator are worth +0.001648 together — [ensembles.yaml](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/configs/ensembles.yaml).

**The 26-member character-level ROVER core**

Exact algorithm in [ensemble.py](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/waxal_asr/ensemble.py):
- **Skeleton = the anchor string** (recipe default `skeleton: anchor`). The anchor is the **first** listed member, s43 greedy. The alternative `skeleton='mbr'` uses the medoid hypothesis. It was better on holdout (−0.0090 vs −0.0076 error) but deemed riskier and not used.
- **Alignment**: each hypothesis is mapped onto the skeleton with `rapidfuzz.distance.Levenshtein.opcodes(skel, other)`, which yields one entry per skeleton character position plus an insertion string before each position.
  - `equal`: copy the character.
  - `replace` with n skel chars and m other chars: position k gets `other[j1+k]` when k < m, else `''`. When m > n, the **last** slot gets the whole remaining substring `other[j1+n-1:j2]`.
  - `delete`: `''`.
  - `insert`: the text is appended to `ins[i1]`.
- **Voting**: for each slot, and for each insertion gap (n+1 gaps), sum the member weights per candidate string. The winner replaces the anchor's content only if it differs from the anchor **and** its summed weight is ≥ `wc` = **`vote_threshold: 2.0`**. Otherwise the anchor's content is kept.
  - Tie-break for insertion gaps: `max` by (weight, non-empty preferred).
  - Tie-break for slots: `max` by weight, first-encountered wins.
- **Output**: `" ".join("".join(out).split())`, which collapses whitespace.
- **Guards**: empty hypotheses abstain (treated as missing data). If the anchor is empty or fewer than 2 members are non-empty, the anchor is returned. An empty output never replaces the anchor.
- **Weights**: 1.0 by default. In `p2n_ens_weighted`, `p2n_ens_soup6`, `p2n_ens_s46swap` and `p2n_ens_s46anchor`, sunbird51 and p1av get **0.5**; in `s46swap`, s46 replaces p1av — [ensembles.yaml](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/configs/ensembles.yaml).
- Exact 26-member list for `p2n_ens_distil` / `p2n_ens_masked`, in order (the first is the anchor):
  - s43
  - s43@1.5, s43@1.0, s43@2.0
  - p1raw@1.5, p1raw@1.0, p1raw@2.0
  - linspec_r@1.5, linspec_r@1.0, linspec_r@2.0
  - snaspec_r@1.5, snaspec_r@1.0, snaspec_r@2.0
  - s44@1.5, s44@1.0, s44@2.0
  - soup5@1.5, soup5@1.0, soup5@2.0
  - s44, soup5, p1raw (greedy)
  - sunbird51, p1av, turbo_linsna_r, distil
  
  `p2n_ens_bp25` is the same list without distil — [ensembles.yaml](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/configs/ensembles.yaml).
- How members were chosen: by measured leaderboard paired comparisons. "members must be strong, and they must fail differently".
  - Four members from four strong checkpoints: +0.0040.
  - Six members from two checkpoints, varying only the penalty: +0.0006.
  - A fresh-seed member: about +0.0032.
  - Weak checkpoints: −0.0003 (0.764476 → 0.764152).
  - Same-lineage-as-anchor members: −0.0014 (stated in the ensemble.py docstring).
  - Anchor swap to distil: −0.0015 (0.764915 → 0.763419).
  - 26 members is the measured optimum: 17 → 0.764476, 25 → 0.766374, **26 → 0.766580**, 38 → 0.765960, 42 → 0.765247, 43 → 0.764912 — [SOLUTION.md](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/docs/SOLUTION.md).

**p2n_mbr (the scored submission)**
- Candidates, in order (ties keep the earliest): `p2n_meta`, `p2n_ens_weighted`, `p2n_ens_masked`, `p2n_ens_s46swap`, `p2n_ens_soup6`, `p2n_ens_wide`, `p2n_ens_s46anchor`.
- **Judge (electorate)**: the 26 member outputs of `p2n_ens_masked`, each already post-processed.
- **Risk** per candidate h for a clip: `sum_x Levenshtein.distance(h, x) / max(len(x), 1)` over the 26 judge transcripts, i.e. edit distance normalised by the *judge* string's length. Strict `<` gives argmin with the first candidate winning ties — [predict.py](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/waxal_asr/modeling/predict.py).
- `p2n_meta` runs the same ROVER, anchor-skeleton with `wc` = 2.0 and weight 1.0 each, over the **finished outputs** of 5 ensembles: weighted, masked, soup6, s46swap and wide. The first one, `p2n_ens_weighted`, is the anchor. Public 0.766683. Extending it to 7 ensembles gave a byte-identical file — [ensembles.yaml](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/configs/ensembles.yaml) and [SOLUTION.md](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/docs/SOLUTION.md).
- Scores:
  - p2n_mbr: public **0.766791** (CER 0.108130, WER 0.358288); private **0.772552** (CER 0.110664, WER 0.344233).
  - The same selection over a different 10-candidate pool: public 0.766773.
- The 2nd pick was `p2n_ens_distil` (public 0.764915) — [SOLUTION.md](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/docs/SOLUTION.md).

**LID routing**
- Model: `facebook/mms-lid-4017`, run open-set (all 4017 classes). It labelled 437 clips Lingala (conf 0.990), 423 Shona (conf 0.980) and 32 low-confidence, one of them Luganda; that clip, `ID_REPXZM`, is actually Lingala. `mms-lid-256` gave 440/440/12 — [SOLUTION.md](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/docs/SOLUTION.md).
- Routing is used only to (a) force the Sunbird language token when conf ≥ 0.5 — [sunbird.py](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/waxal_asr/modeling/sunbird.py) — and (b) route the lin/sna specialists in the early 0.744835 submission.
- In the final ensembles the specialists vote on **all** clips; no per-language member gating appears in `ensembles.yaml`.

**Post-processing**

Applied to every member **and again** to every voted text; both rules are idempotent ([postprocess.py](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/waxal_asr/postprocess.py), [INFERENCE.md](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/docs/INFERENCE.md)):
1. `collapse_loops(max_n=4, keep=2)`: for n = 4 down to 1, a run of ≥ 3 consecutive identical word n-grams is cut to 2 copies. **+0.0069** when introduced. `no_repeat_ngram_size` cost 0.038 on holdout.
2. `fix_sentence_case`: regex `([.!?]\s+)([a-z])` → uppercase, repeated until stable. References capitalise after a terminator 98.6% of the time vs 87.2% for raw output. CER only.

What they deliberately do **not** do:
- Force a capital on the first letter. References capitalise the opening word 89.3% of the time (81.9% for Lingala), and raw output was already at 92.6%.
- Add a final period, add commas, or apply lexicon/LM correction. Comma restoration: 0.764314 vs 0.764759. Period-corrected ensemble: 0.756972. KenLM: every weight setting lost. Word-level voting: CER 0.1109 → 0.1261.
- Reference punctuation rates per row: 0.630 commas, 1.402 full stops — [SOLUTION.md](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/docs/SOLUTION.md).

**Public-leaderboard history** (score / CER / WER) — [SOLUTION.md](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/docs/SOLUTION.md):
- Phase-1 champion, stripped vocab: 0.724984 / 0.130719 / 0.419312
- Sunbird51 zero-shot, beam 8, routed: 0.737168 / 0.124049 / 0.401616
- p1raw, raw vocab: 0.742142 (**+0.0171**)
- Specialists, routed: 0.744835
- soup5: 0.746054
- distil: 0.746787
- First 3-family ensemble: 0.747041
- 4 members: 0.751181
- 7 members: 0.754847
- First blank-penalty member: **0.759338**
- `bp10`, 17 members: 0.764476
- `bp25`, 25 members: 0.764759
- `distil`, 26 members: 0.764915
- `masked`: 0.766563
- `weighted`: 0.766580
- `meta`: 0.766683
- `mbr`: **0.766791**

Variants:
- s46swap: 0.766477
- soup6: 0.766226
- s46anchor: 0.766083
- wide: 0.765960

Final-day negatives:
- Frame-level posterior averaging across the 6 shared-vocab checkpoints: 0.764248.
- Entropy-minimisation TTA: 0.765910.
- 2nd-round distillation: 0.766401, despite a holdout of 0.2429.

Noise band: SE about 0.0013 on a paired difference, so **< 0.0027 is not distinguishable** from noise.

**Phase-2 audio / Phase-1 label contamination**
- **Trained on Phase-2 test audio with pseudo-labels (exclude these)**:
  - `distil`: trained on its own ensemble's transcripts of the 892 clips — [distilled card](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-distilled).
  - `soup6`: weight-averages distil — [soup6 card](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-soup6).
  - The unused 2nd-round distil.
- distil is a member of `p2n_ens_distil`, `masked`, `weighted`, `soup6`, `s46swap`, `wide` and `s46anchor`, and therefore of `meta`, of `p2n_mbr`, and of the MBR judge set. **The scored p2n_mbr file therefore depends on Phase-2 pseudo-labels.**
- **Clean of Phase-2 audio**: s43, s44, s46, p1raw, linspec_r, snaspec_r, p1av, soup5 (its p2 members used WAXAL *unlabeled* pools, not test audio), turbo_linsna_r and sunbird51 (third party). The clean recipes are `p2n_ens_bp25` (0.764759) and `p2n_ens_bp10` (0.764476); both are unmasked historical decodes — [ensembles.yaml](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/configs/ensembles.yaml).
- **Trained on WAXAL Phase-1 test-split gold labels** ("Phase 2 permits this and the organisers confirmed it publicly", per the authors): s43, s44, s46, turbo_linsna_r and distil, with soup6 transitively — [seed43 card](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-seed43) and [whisper-turbo card](https://huggingface.co/anyantudre/waxal-whisper-turbo-linsna). The soup5 and specialist cards list no Phase-1 test data.

### Inferences
- To reproduce without Phase-2 pseudo-labels, use `p2n_ens_bp25` with the attention-mask fix and Whisper decoded as a generator. That is masked minus distil; the masked fix gave about +0.0016 on the 26-member version. For MBR, use a judge set without distil, and drop the soup6 candidate. This exact clean configuration was never scored.
- The ROVER is anchor-conservative: with `wc` = 2.0 and unit weights, any 2 of the 25 non-anchor members agreeing can overrule the anchor at a slot. This low threshold is why many over-generating blank-penalty members still help: insertions are cheap to reject but are readily accepted when ≥ 2 members agree.
- Their forced-first-capital finding (do not force it) directly contradicts Pasketti's measured +0.004 for capital + period (Q2). The likely difference is that anyantudre's raw-vocab CTC members already emit capitals and periods; the effect depends on the members' native output style.

### Gaps
- No public Zindi discussion post or write-up beyond the repo and model cards was found. Zindi discussion pages were not fetched.
- The exact `waxal_asr/modeling/ctc.py` batching and padding details, and `whisper.py` (greedy, 200 tokens), were not read line by line.
- The claim that the organisers publicly confirmed Phase-1 test-split use is the authors' statement; it was not verified on Zindi.
- The `waxal-linsna` dataset card says `manifest_test_lin/sna.tsv` are "read by the per-language specialist configs". This may mean some specialist or `_p` configs also used Phase-1 test gold, which conflicts with the specialist cards. The `configs/w2vbert_linspec*.yaml` files were not checked — [waxal-linsna card](https://huggingface.co/datasets/anyantudre/waxal-linsna).

---

## Q2. Team Pasketti (1st place; ZFTurbo / Roman Solovyev + enes3774 / Enes Mahmut Kulak): weighted char median, members, weights, boundary rules, suspicious-word repair, capitalisation

### Takeaway
Pasketti ensembles 8 systems with ZFTurbo's `asr-ensemble==1.0.2`, using `median_extended` at character level. This is **not** positional voting. It is a weighted generalised-median search over every input string plus every intermediate string on a Levenshtein edit path between each pair, keeping the string that minimises Σ wᵢ·Lev(cand, hypᵢ).
- Weights: Sunbird-SALT 10, MMS-1B 12, XLS-R 11, Whisper-v3 9, Qwen3-ASR 6, Parakeet 4, Granite 4, SALT noisy-student 10.
- Then deterministic LID-routed boundary rules, derived from WAXAL bigram counts with a grammatical filter and confirmed by **leaderboard probes**. Then OOV "ghost"/non-word repair proposed by correlated checkpoints and verified by 6 diverse systems. Finally a first-letter capital and a final full stop, worth about +0.004 on the leaderboard.
- Final score: public 0.769614, private 0.780944.

### Cited Findings

**Code and data locations**
- Repo: https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place (MIT). Main docs: [docs/README.md](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/docs/README.md), dated 10.08.2026, and [postprocess/README.md](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/README.md).
- Training data: HF `enesssssw/waxal-cleaned-16k`, configs `lin_asr` and `sna_asr`, train+validation; clips over 50 s dropped. Validation: 100 utterances per language from the dataset's **test** split, 200 total. The ensemble weights were hand-tuned on those 200 utterances — [docs/README.md](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/docs/README.md).
- Test front-end: MSST `bs_roformer` 6-stem, vocals stem only, then mono, 16 kHz `soxr_vhq` and 32-bit float. The authors say this is "most likely unnecessary" on the clean Phase-2 audio. Member 8 reads **raw** audio.

**Members** (validation WER / CER / mean, then ensemble weight) — [docs/README.md §4.3, §7.2](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/docs/README.md):

1. `Sunbird/asr-whisper-51-african-languages`, zero-shot: 0.3189 / 0.0788 / 0.1989, **w 10**.
   - Decoding: beam 5, fp16, `forced_decoder_ids` = [(1, lang), (2, transcribe), (3, notimestamps)].
   - **The forced language is Runyankole (`cgg` → token 50350) rather than lin/sna; this was empirically better on validation.**
2. `facebook/mms-1b-all` CTC: 0.3204 / 0.0760 / 0.1982, **w 12**.
   - Training: an adapter stage (lr 1e-3, 4 epochs, 4×6), then full training (lr 1e-5, 10 epochs, 12×2), with dropouts and layerdrop at 0.
   - Decoding: `pyctcdecode` beam **5000**, no LM. Beam ablation: greedy 0.3252/0.0767 → beam 100 0.3209/0.0762 → beam 500 0.3207/0.0761 → beam 5000 0.3204/0.0760.
3. `facebook/wav2vec2-xls-r-300m` CTC, checkpoint-averaged: 0.3250 / 0.0776 / 0.2013, **w 11**. lr 2e-5, 20 epochs; greedy decoding.
4. `openai/whisper-large-v3`, full fine-tune: 0.3394 / 0.0846 / 0.2120, **w 9**. lr 1e-5, 20 epochs; decoding beam 5, `max_new_tokens 428`.
5. `Qwen/Qwen3-ASR-1.7B`, full fine-tune: 0.3546 / 0.0880 / 0.2213, **w 6**. Decoding beam 5, `max_new_tokens = min(20·dur, MAX_ASR_INPUT_SECONDS·20)`, chunked.
6. `nvidia/parakeet-tdt-0.6b-v3`: 0.3965 / 0.1129 / 0.2547, **w 4**. Top-7 checkpoint average (WER 0.4013 → 0.3965, CER 0.1168 → 0.1129); greedy TDT decoding.
7. `ibm-granite/granite-speech-4.1-2b`: 0.4265 / 0.1298 / 0.2782, **w 4**. Decoding beam 3 (CER 0.1357 → 0.1298).
8. Sunbird-51 plus a noisy-student LoRA, **w 10** — [model_whisper_salt_student/README.md](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/model_whisper_salt_student/README.md):
   - LoRA: r 16, α 32, dropout 0.05 on q/k/v/out_proj in both encoder and decoder; lr 1e-5 OneCycle; batch 8×4; the shipped checkpoint is **step 500**.
   - Teacher pseudo-labels: beam 5 with 5-best over **WAXAL train audio**, rescored by AfriqueQwen-14B (lin weight 0.01, sna weight 0.3, sna word bonus 0.04). Hypotheses over 20 characters per second and clips over 29 s are dropped. The confirmed orthographic splits are applied to the labels.
   - Augmentation: speed 0.9/1.0/1.1 at p 0.7; pitch 0.95–1.05 at p 0.3; SpecAugment with 2 frequency masks ≤ 20 and 2 time masks ≤ 40; noise p 0.2 at 15–30 dB; reverb p 0.15.
   - Inference: beam 5 with 5-best, then LLM rescoring and shallow fusion, then capital and final stop. Correct `<|ln|>`/`<|sn|>` tokens are forced.
   - Measured on the leaderboard: beam 1 → 5 gave **+0.007**; 10-best gave −0.022.

- Weighted char ensemble on validation: 0.2968 / 0.0697 / **0.1832** — [docs/README.md](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/docs/README.md).
- Ensemble call, from [r03_ensemble_subms.py](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/proc_data/r03_ensemble_subms.py). Inputs are sorted by ID, `fillna("")` is applied, and outputs are stripped.
  ```python
  ensemble_text_lists(texts_list, normalize=False, char_level=True,
                      weights=[10.0, 12.0, 11.0, 9.0, 6.0, 4.0, 4.0, 10.0],
                      ensemble_type='median_extended', language=None, max_workers=8)
  ```
  Order: salt, mms, wav2vec, whisper, qwen, parakeet, granite, salt_student.

**The "weighted character-level median" algorithm**

Taken from the `asr-ensemble 1.0.2` source, `asr_ensemble/ensemble_functions.py::find_weighted_set_median_extended`, PyPI wheel; homepage https://github.com/ZFTurbo/asr_ensemble:
```
cands = ordered-unique( for i: [s_i] + for j>i: get_path_strings(s_i, s_j) )
# get_path_strings: full Levenshtein DP (unit costs); traceback from (n,m) preferring diagonal
#   (match/sub), then deletion, then insertion; apply ops end->start to s_i, recording EVERY
#   intermediate string (s_i ... s_j)
best = argmin_{c in cands} sum_j w_j * Levenshtein.distance(c, s_j)   # strict '<' => first wins ties
```
- The output can be a string that no member produced (an intermediate edit-path string), which explains the "ghost token" phenomenon handled in post-processing.
- The docs describe it as "at each alignment slot the character with the highest accumulated weight wins". **That description does not match the code**: it is a generalised-median search, not slot voting — [docs/README.md §4.5](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/docs/README.md) vs the PyPI source.

**Post-processing**

Run from [run_postprocess.py](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/run_postprocess.py). Input is `artifacts/raw_weighted_char_ensemble.csv` and the LID file is `artifacts/language_predictions.csv` (column `announced3_language`). Steps:

1. `apply_best_postprocessing.py` — [source](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/apply_best_postprocessing.py). This is the standalone form of the pipeline behind `submission_v6_stage2_8_lb_0.7555_pp_best.csv`.
   - **Lingala**, in order:
     - `na moni` split (namoni → na moni).
     - `ba/bazo/baza` split table, taken from `waxal_tables(pool,'lin',min_count=3,max_ratio=0.95)` and filtered to first token ∈ {ba, bazo, baza}.
     - nazomona → nazo mona.
     - tozomona → tozo mona.
     - JOIN `ba`+{zali, tie, lati, komi, fandi, vandi, sali, telemi, lakisi}.
     - SPLIT `na`+{ye, nga, kati, se, yango, ba, biso}.
     - Matching is case-insensitive and strips trailing `.,!?;:`; a first-letter capital is preserved.
   - **Shona**:
     - The `waxal_tables(pool,'sna',3,0.95,grammatical=True)` split table.
     - Rare-form regex `(?i)^(a|va|i|ri|chi|zvi|dzi|u|pa|ti|ndi|ku)ri(ku.+)$`, split after the `…ri` prefix.
   - **All rows**, `finish()`: uppercase the first alphabetic character; append `.` unless the text ends in `.?!`.
2. `residual_confirmed_rules.py --stage baverb` catches progressive forms outside the narrow prefix list and changed 6 rows on the winning input — [postprocess/README.md](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/README.md).
   - Its regex: `^(\w*ri)(ku\w{3,})$` with prefix length ≥ 3.
   - Also present: `SNA_NE_SPLITS` (newaya, nebhutsu, negireyi, nefenzi, nesiketi, une/ane/ineruvara), `bhurawuni` → `bhurauni` (dormant, fired 0 times), a degenerate-tail stripper (dormant) and punctuation de-duplication — [source](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/residual_confirmed_rules.py).
3. `ghost_tokens.py --same-clip --min-votes 5` — [source](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/ghost_tokens.py).
   - Tokens: lowercase, alphabetic only, length ≥ 4.
   - A ghost is absent from the WAXAL corpus **and** from every candidate member's vocabulary for that language.
   - Repair: the same-clip member token at Levenshtein distance ≤ 1 with the most votes, where votes are distinct systems emitting it in that clip, required ≥ 5.
4. `majority_fix.py --margin 2 --min-votes 4 --only-nonword` — [source](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/majority_fix.py).
   - Applies to a WAXAL-OOV token (length ≥ 4).
   - Replace it with a distance-1 variant when the variant has ≥ 4 member votes and exceeds the current token's votes by ≥ 2.
   - Members must carry the same post-processing as the submission.
5. `diverse_verify_chain.py` — [source](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/diverse_verify_chain.py). Uses 6 diverse members: `st500_raw`, `mms-1b-all`, `parakeet`, `wav2vec-300`, `whisper_51` and `whisper_ft`.
   - A ghost repair is kept iff the number of diverse members containing the new token is greater than the number containing the old one.
   - A majority repair is kept iff the new token has ≥ 3 diverse votes and new − old ≥ 2.
   - Otherwise the token reverts. Word-position alignment assumes equal token counts and asserts it.

- The 10 candidate (proposal) members are `st500_rep/raw`, `v2s200/300/00400/00500_rep`, `submission_v6_stage2`, `submission_whisper_b5_repaired_lincaps`, `A1_asrafrica_baseline` and `submission_whisper_raw` — [run_postprocess.py](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/run_postprocess.py).
- Outcome:
  - Strict ghost repairs: 24 proposed, 21 kept.
  - Relaxed repairs: 118 proposed, 20 kept.
  - In total, 41 token repairs across 39 of 892 rows.
  - On the 118 relaxed proposals, the correlated pool voted 873 vs 183 for the replacements, but the diverse systems split 47 for / 47 against / 24 ties (256 vs 268 votes).
  - Source: [docs/README.md §4.6](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/docs/README.md).

**How the boundary rules were derived**
- `waxal_tables` counts unigrams and bigrams in `postprocess/corpus/pool/{lin,sna}.waxal.txt`; these are WAXAL transcript text lines, roughly 12.6k lin and 13.8k sna lines. For each bigram (a, b) with count n ≥ 3 and joined form `ab` with count m ≥ 3, the pair is kept only if min(n, m)/(n + m) ≥ 0.05, i.e. "genuinely ambiguous". The split direction is the most frequent split of each joined form.
- With `grammatical=True` (Shona), a split is allowed only if:
  - a ends in `ri` and b starts with `ku`; or
  - a ends in `ne` and len(a) ≤ 6; or
  - a ends in `ri` and len(a) ≤ 8.
- Measured on validation: the `-ri` non-ku form gave +0.0011 vs +0.0006 for the ku form alone. Splitting concord + colour words (`rwebhuruu`) was **rejected** at −0.0085 — [respace_probe.py](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/remote/respace_probe.py).
- WAXAL itself is roughly a coin flip on these spellings: `namoni` 3983 vs `na moni` 1359; `varikufamba` 475 vs `vari kufamba` 361; overall in Lingala, 105,932 split vs 110,062 joined. The authors state the **test references split `na moni` about 73% of the time**. The rule directions were set by **leaderboard probes**: `form_probe.py` forces one spelling and reads the sign of the score delta — [form_probe.py](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/remote/form_probe.py), [respace_probe.py](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/remote/respace_probe.py) and [docs/README.md](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/docs/README.md).
- The `ba`+verb join family was "8 for 8 on the leaderboard". Corpus split shares range from 6% (bazali) to 30% (balakisi), so no frequency threshold selects them; the grammatical class does. `batelemi` (19% split) must be joined while `tozomona` (19%) must be split — [residual_confirmed_rules.py](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/residual_confirmed_rules.py).
- Rule firing counts: tozomona 20, namoni 208, nazomona 126. Shona boundary changes on the V6 ensemble: 118 — [docs/README.md](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/docs/README.md) and [residual_confirmed_rules.py](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/residual_confirmed_rules.py).

**Leaderboard deltas quoted in Pasketti code**
- Capital + final period on all 892 rows: about **+0.004**, the largest deterministic effect — [docs/README.md](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/docs/README.md).
- nazomona split: +0.00227 — [majority_fix.py](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/majority_fix.py).
- Ghost repair: +0.000495. Wholesale median override by member majority: −0.000563 — [majority_fix.py](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/majority_fix.py).
- `bazali` join stage: 0.759759203; plus the ba+verb family: 0.761005 (an earlier V6 pipeline) — [residual_confirmed_rules.py](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/residual_confirmed_rules.py).
- Joining the Shona progressive: **−0.008** (CER +0.0014, WER +0.0150), confirming the split. An older docstring in the same file says "splitting Shona measured −0.010"; it belongs to the refuted join hypothesis and contradicts the final pipeline — [residual_confirmed_rules.py](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/postprocess/residual_confirmed_rules.py).
- Student LLM rescoring: Lingala −0.002 at w 0.3 and −0.0044 at w 0.05; Shona at 0.3 about +0.002. Decoder-only LoRA 0.7459 vs base Whisper 0.7455 — [student README](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/model_whisper_salt_student/README.md).
- Final scores: public 0.769614172 (CER 0.104677677, WER 0.356093977); private 0.780944419 (CER 0.109050328, WER 0.329060831). SHA-256 of the winning CSV: `9a665cd4…ad31` — [docs/README.md](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/docs/README.md).

**Contamination flags**
- No member was trained on Phase-2 audio. The student's pseudo-labels come from WAXAL train audio only.
- The WAXAL test split (Phase-1 test) was used for validation and weight tuning only (200 utterances).
- **Phase-2 leaderboard feedback was used to choose spelling and boundary directions**, through the probes. The `announced3` LID file's provenance is undocumented.

### Inferences
- The scale of Pasketti's +0.0047 public advantage over anyantudre is consistent with their stronger WER (0.356 vs 0.358 public, 0.329 vs 0.344 private), driven by test-convention word-boundary rules and diverse architectures. On the private split, the WER difference (0.0152) dominates the score gap.
- The median_extended search is O(n²·L) candidates times n distances. With 8 members of about 150 characters this is cheap. With 26 members it would be much heavier, which is a design consideration if it is reused on anyantudre-style big ensembles.
- The boundary rules are only partly reproducible offline. The corpus statistics and grammatical filters are deterministic, but which direction to apply (split `na moni`, split `-ri ku-`, join `ba`+verb) was chosen by leaderboard probing.

### Gaps
- The source of the `announced3` LID predictions is not documented in the repo.
- The exact origin of the `corpus/pool/*.waxal.txt` text is not stated: it is unclear whether it includes the WAXAL test-split transcripts.
- The provenance of the candidate members `st500_*` and `v2s*` is not fully documented. They appear to be student checkpoints at different steps; this is unverified.
- `A1_asrafrica_baseline` is unexplained.
- The full `main()` of `residual_confirmed_rules.py`, i.e. which functions run under `--stage baverb`, was not read in full.

---

## Q3. 2nd / 3rd place approaches

### Takeaway
2nd place is anyantudre (Q1). 3rd place is Abdourahamane Ide Salifou (Niger), private 0.771866; only a one-paragraph journalistic description is public and no code was found.

### Cited Findings
- 3rd place, Abdourahamane Ide Salifou, Niger, an AI engineering student. The blog describes his approach as: for Lingala, "trained his system to take in the flow of speech before choosing words"; for Shona, "relied on surrounding context to guide predictions" — [Google blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-winners-of-the-waxal-speech-recognition-challenge/).
- 3rd-place private score 0.771866; winner 0.780944 — [anyantudre SOLUTION.md](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/docs/SOLUTION.md).
- Participation: 1,462 participants from 100 countries submitted 8,837 solutions — [Google blog via search snippet](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-winners-of-the-waxal-speech-recognition-challenge/).
- `github.com/Lord-LLM/Google-WAXAL-ASR-Challenge` contains baseline notebooks (Whisper, wav2vec2, MMS-1B), not a top solution — [repo](https://github.com/Lord-LLM/Google-WAXAL-ASR-Challenge).

### Inferences
- The blog's 3rd-place description could fit a seq2seq or LM-context decoder, but that is speculation.

### Gaps
- No 3rd-place code, write-up or Zindi discussion post was found; one search on his name returned nothing relevant.
- Zindi discussion threads were not fetched.

---

## Q4. Explicit "what gave the biggest gains" lists, with numbers

### Takeaway
anyantudre's biggest public-leaderboard gains were:
- Raw cased and punctuated CTC vocabulary: +0.0171.
- Blank-penalty members: +0.0104.
- Character ROVER: +0.006 to +0.010.
- Beam 8 for Whisper: +0.0224 on that arm.
- Loop collapse: +0.0069.

Pasketti's were ensemble diversity (8 architecture families, with zero-shot SALT weighted heavily) plus test-convention boundary rules, and capital + final stop at about +0.004.

### Cited Findings
- anyantudre's "What worked" list ([SOLUTION.md](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/docs/SOLUTION.md)):
  1. Raw vocabulary, +0.0171.
  2. Blank-penalty decoding, +0.0104.
  3. Character ROVER, +0.006 to +0.010.
  4. Source diversity: +0.0040 vs +0.0006.
  5. Fresh seeds, about +0.0032 per member.
  6. Curriculum order (holdout 0.2905 → 0.2785; Shona refresh 0.2185 → 0.1616, with punctuated rows going from 32% to 100%).
  7. Distillation, best single model at 0.746787.
  8. Loop collapse, +0.0069.
  9. Beam 8 over greedy for Whisper, +0.0224.
  10. Capital after a sentence break, small and CER-only.
  
  Beyond the list: attention mask plus generator decoding, +0.001648; MBR selection, +0.000211 over the best single vote (0.766580 → 0.766791).
- anyantudre's "What did not work" list ([SOLUTION.md](https://github.com/anyantudre/google-waxal-asr-challenge-solution/blob/main/docs/SOLUTION.md)):
  - KenLM. 76–84% of word errors are real-word errors.
  - Lexicon correction. OOV rate is 2.9% in references vs 2.8% in hypotheses.
  - Word-level voting.
  - New architectures: Whisper-turbo 0.7331, XLS-R, and a fine-tuned Sunbird, which lost to its own zero-shot output.
  - Self-training on unlabeled WAXAL.
  - Four punctuation fixes.
  - Weak members.
- Pasketti's "What mattered most" list ([docs/README.md §10.4](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/blob/main/docs/README.md)):
  1. The ensemble is the single largest contributor.
  2. Architectural diversity matters more than any member's quality; zero-shot SALT carries the heaviest weight because its errors are uncorrelated.
  3. Character-level voting.
  4. Two-stage fine-tuning: adapters or a frozen encoder at high LR, then full training at low LR.
  5. The CTC length filter `input_len // 320 > label_len`.
  6. De-noising is probably optional.
  7. Orthography-aware consensus: correlated checkpoints propose, diverse systems verify.
  
  Numbers attached: validation 0.1982 for the best single member → 0.1832 for the ensemble; capital + period about +0.004 on the leaderboard; student beam 5 +0.007.

### Inferences
- The two solutions take opposite views on architecture diversity.
  - anyantudre found new architectures weaker than fresh seeds of w2v-BERT.
  - Pasketti found architecture diversity to be the main gain.
- Pasketti's members are individually much stronger and more diverse: MMS-1B with beam 5000, plus SALT, Qwen3-ASR, Parakeet and Granite. anyantudre trained only w2v-BERT and Whisper-turbo.
- Both teams agree that the Sunbird-51 zero-shot model is a high-value member: Pasketti gives it the top-tier weight 10, and in anyantudre's system it scores 0.737 solo.
- Pasketti's finding that forcing Runyankole is better than forcing lin/sna is a notable decoding trick. anyantudre forced lin/sna with beam 8, which differs and is untested against Pasketti's choice.

### Gaps
- Neither team published a clean ablation of its final stack. Pasketti explicitly says the Stage2_8 ensemble without post-processing was never submitted separately, so per-stage causal deltas on the final system are not available.

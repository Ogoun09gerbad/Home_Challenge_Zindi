# Post-processing, external-text LMs and cheap adaptation for Lingala and Shona WAXAL ASR, under score = 1 − 0.5×(WER_lowercased_with_punct + CER_raw)

Research date: 2026-10-26.

Tags used below:
- **[VERIFIED]**: fetched and read this session.
- **[LOCAL]**: computed this session on the project's own WAXAL **train + validation** transcripts in `C:\Users\DATALAB\Documents\WAXAL ASR Challenge\data\cache\{train,validation}_{lin,sna}.parquet`. No test or Phase 2 labels were used.
- **[UNVERIFIED]**: prior knowledge or a search snippet that was not opened.

The scorer I assumed is the one already in `common.py::score`: corpus-level `jiwer.wer` on `.lower()` strings and `jiwer.cer` on raw strings, using jiwer 4.0.0 from the project venv. In jiwer's defaults:
- `wer_default` = RemoveMultipleSpaces → Strip → split on whitespace. Punctuation is **not** removed, so a token such as `ndako.` is a different word from `ndako`.
- `cer_default` = Strip → characters. Internal double spaces **are** counted. Source: `.venv\Lib\site-packages\jiwer\transformations.py` lines 40-46 and 106-111 [LOCAL, read].

Scripts used for the [LOCAL] numbers (scratchpad, session-specific):
- `C:\Users\DATALAB\AppData\Local\Temp\claude\c--Users-DATALAB-Documents-WAXAL-ASR-Challenge\94f20fa4-953c-4d20-9736-b9a4c20f1ac2\scratchpad\textstats.py`
- `...\scratchpad\sim.py`
- `...\scratchpad\seg.py`
- `...\scratchpad\oov.py`

All of their outputs are reproduced below.

---

## Q1. Post-processing under this metric: capitalisation, final period, commas, spaces before punctuation, apostrophes, hyphens, accents, numerals

### Takeaway
- **Always capitalise the first letter and end with a period.** This matches 75% (lin) and 99.6% (sna) of references for the capital, and 67% (lin) and 95% (sna) for the final period.
- **Attach punctuation to the preceding word, with no space before it.**
- **Never lowercase the whole hypothesis.** Doing so costs about 0.002 to 0.005 of score in CER alone.
- **Shona is a special case.** 83% of Shona references contain **sentence-internal periods**. Stripping all punctuation from a *perfect* Shona transcript costs about 0.08 of score (WER +0.14). The Shona system must therefore emit sentence periods and the capital letter after each one, either from an ASR model trained on raw punctuated and cased text or from a punctuation-restoration step.
- **Commas** are worth at most about 0.010 to 0.015 of score even when predicted perfectly. Keep them only if comma precision on validation is above 50%; otherwise strip them.

### Cited Findings
**Reference conventions, lin n=16,243 and sna n=15,836 (train + validation) [LOCAL, textstats.py]**

| Statistic | lin | sna |
|---|---|---|
| Starts with an uppercase letter | 75.4% | 99.6% |
| Last character is `.` | 67.0% | 95.1% |
| Ends with a bare letter (no punctuation) | about 31% | about 4.6% |
| Ends with `?` / `!` | 0.07% / 1.1% anywhere | 0% / 0.01% |
| Contains a comma | 38.2% (0.81 commas/utt) | 32.9% (0.51/utt) |
| Contains a sentence-internal period | 8.5% | **82.8%** |
| Space before punctuation (e.g. `telemi .`) | 9.0% | 7.0% |
| No space after a comma | 14.5% | 3.1% |
| Double spaces somewhere | 15.3% | 24.0% |
| ASCII apostrophe `'` | 2.1% | 1.4% |
| Curly `’` | 0% | 0% |
| Backtick or `‘` | 0.01% | 0.03% |
| Hyphen `-` | 3.2% | 3.3% |
| Digits | 0.9% | 0.18% |
| Brackets `(` | 0% | 0.5% |
| `"` | 0.09% | 0.8% |
| Tokens carrying attached trailing punctuation | 5.1% | 13.0% |
| Capitalised tokens | 3.6% | 11.5% (sentence starts) |
| Strings not in NFC | 0% | 0% |

Further [LOCAL, textstats.py] detail:
- **lin non-ASCII characters:** é 1,292, è 93, î 86, ô 57, â 24, à 24, ñ 21, ê 15, ï 14, ç 11, É 8. These come from French loanwords and code-switching (e.g. `ba soldats`), plus a little noise (⭐, `«»`, `\xa0`).
- **sna non-ASCII characters:** almost none (í 60, ú 17, à 14).
- **Punctuation inventory:**
  - lin: `,` 13,203; `.` 13,036; `-` 582; `'` 412; `!` 272; `;` 53; `"` 26; `?` 12.
  - sna: `.` 42,439; `,` 8,044; `-` 588; `"` 316; `'` 314; `(` / `)` 112 each.

**Oracle simulation [LOCAL, sim.py].** Each validation reference was transformed and then scored as a hypothesis against the untransformed reference. This gives the irreducible cost of each normalisation choice, assuming a perfect recogniser.

| Hypothesis = transformed reference | lin WER | lin CER | lin score | sna WER | sna CER | sna score |
|---|---|---|---|---|---|---|
| identity | 0 | 0 | 1.0000 | 0 | 0 | 1.0000 |
| collapse whitespace only | .0006 | .0018 | .9988 | .0001 | .0026 | .9986 |
| + remove space before punctuation ("clean") | .0094 | .0027 | .9940 | .0067 | .0030 | .9952 |
| clean + **finish** (capitalise first letter, add `.` if the last char is not `.?!`) | .0217 | .0066 | .9858 | .0088 | .0033 | .9939 |
| clean, strip final period | .0299 | .0071 | .9815 | .0471 | .0080 | .9724 |
| clean, lowercase first char | .0094 | .0076 | .9915 | .0067 | .0083 | .9925 |
| clean, fully lowercased + finish | .0217 | .0085 | .9849 | .0088 | .0130 | .9891 |
| clean + remove all commas | .0408 | .0079 | .9757 | .0272 | .0056 | .9836 |
| keep periods only (no commas), sentence caps, finish | .0534 | .0121 | .9672 | .0304 | .0061 | .9817 |
| remove all punctuation, keep case, + finish | .0582 | .0129 | .9644 | .1043 | .0149 | .9404 |
| remove all punctuation, lowercase | .0661 | .0202 | .9568 | **.1425** | .0345 | **.9115** |
| clean, strip accents (é→e etc.) | .0125 | .0034 | .9921 | .0068 | .0030 | .9951 |

**Winning team's finishing rule** (1st place, Team Pasketti, `apply_best_postprocessing.py::finish()`) — [GitHub, raw file](https://raw.githubusercontent.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/main/postprocess/apply_best_postprocessing.py) [VERIFIED]:
- Strip the string.
- Uppercase the first alphabetic character.
- Append `.` if the last character is not one of `.?!`. A trailing comma or colon does not block the period.
- Collapse repeated whitespace with `' '.join(text.split())`.
- Commas are never inserted, removed or normalised. `TRAIL='.,!?;:'` is only used to detach punctuation for word matching and is reattached afterwards.
- Rows that are neither `lin` nor `sna` receive only `finish()`.

**Other sources:**
- The postprocess README lists first-pass "Initial capitalization and sentence-final punctuation", and reports no ablation numbers — [Pasketti postprocess README](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/tree/main/postprocess) [VERIFIED]
- In the Makerere benchmark, the normalisation lowercased text and removed punctuation except apostrophes. Its WER numbers are therefore not comparable to this cased and punctuated metric — [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1) (from the earlier notes in `research_notes/ASR strategy WAXAL lin sna lug/waxal_dataset_and_challenge.md`)

### Inferences
1. **Decision rule for any punctuation mark.** Because WER sees `ndako.` ≠ `ndako`, each predicted mark is a ±1-word bet:
   - Correct placement saves one substitution.
   - Placement where the reference has none adds one substitution.
   - Omission where the reference has one costs one substitution.
   
   Emit a mark only where P(reference has it there) > 0.5. CER adds about ±1 character per mark, which is minor.
   - Final period: P = 0.67 (lin) and 0.95 (sna), so **always add it**. Oracle gain: lin WER −0.008 / score +0.004; sna WER −0.038 / score +0.021 (".9815 → .9858" and ".9724 → .9939" rows, relative to stripping it).
   - Capital first letter: P = 0.75 and 0.996, so **always capitalise**. It is CER-only: +0.0024 (lin) and +0.0027 (sna) score.
2. **Commas: test, don't assume.**
   - The oracle value of perfect commas is lin WER 0.031 and sna WER 0.020, i.e. +0.018 / +0.012 of score (rows "clean" vs "remove all commas").
   - Commas in spontaneous descriptions are transcriber-dependent. A model's comma precision may well be below 50%, in which case stripping helps.
   - **Test:** on validation hypotheses, compare `score(hyp)` with `score(hyp.replace(',', ' ') collapsed)`, per language. Keep the variant that wins on a **speaker-disjoint** split (v1 validation shares speakers with train; see the earlier notes).
   - **Expected:** stripping commas helps when the model is weak or hallucinates commas, and hurts slightly when it is trained on raw text and has a good comma F1.
3. **Shona sentence periods are the biggest single text lever.**
   - A system that outputs no internal periods loses about 0.04 to 0.06 of score even with perfect words (row "remove all punct + finish" at .9404 vs "keep periods only" at .9817).
   - **Options, cheapest first:**
     - (a) Train or fine-tune the acoustic model on **raw** (cased, punctuated) Shona transcripts, so that `.` and uppercase are CTC tokens.
     - (b) Restore punctuation and casing with a small token-classification model (e.g. AfroXLMR/mBERT fine-tuned on the train transcripts to predict "period after this token / comma after / capitalise").
     - (c) Use a heuristic: insert `. ` + capital before frequent Shona sentence starters (`Pane`, `Mufananidzo`, `Vanhu`, `Uye`...), learned from train bigrams, with a high-precision threshold.
   - Variant (c) is speculative and needs validation testing.
   - After any period, capitalise the next word. CER only; sna sentence capitals matter because 11.5% of tokens are capitalised.
4. **Spaces before punctuation.** 91–93% of references attach punctuation, so always output `word.` / `word,`. The 7–9% that do not are an irreducible cost of about 0.003 of score.
5. **Double spaces.** These are a CER-only, irreducible cost of about 0.001 of score. Do not try to predict them. Always collapse whitespace, because spurious double spaces in the hypothesis would add CER.
6. **Apostrophes.** References only contain ASCII `'`. Map `’ ‘ \`` → `'`. This is safe, though its effect is tiny.
7. **Accents (lin).** Keep é/è/ê/à/ç in the vocabulary. Stripping them costs about 0.003 WER (oracle). Do not "correct" French loanwords to French spelling: references mix `bus`/`bisi`, `stade`, `soldats`, and Lingala-ised spellings such as `kiacuafur`.
8. **Hyphens.** About 3% of utterances contain them (e.g. `Mwana-mwasi`). Leave them as the model outputs them. No rule is recommended without validation evidence.
9. **Numerals.** Only 0.9% (lin) and 0.18% (sna) of references contain digits, so the convention is words. Do **not** convert number words to digits. If a model emits digits, consider converting them to words. Gains are negligible, so this is low priority.
10. **Leading `Oyo`/`Awa` capitalisation in lin.** 25% of lin references start lowercase, but capitalising is still the better bet (P = 0.75).

### Gaps
- The exact Zindi scorer code is not public. I assumed the `common.py` interpretation (lowercase for WER, raw for CER, jiwer defaults); per the in-file comment, a participant verified it on the public leaderboard, but I did not see that evidence.
- It is unknown whether Phase 2 or the Study Jam test references follow the same punctuation and casing distribution as train/validation.
- The rate of comma and period hallucination in actual model output was not measured. The table is oracle-only.

---

## Q2. Word-boundary conventions (Lingala particles, Shona auxiliaries) and lexicon-based split/merge repair

### Takeaway
WAXAL train transcripts are **internally inconsistent**:
- Shona `vari kufamba` vs `varikufamba` is split 44/56.
- Lingala `na moni` vs `namoni` is split 27/73.

So any single global convention leaves an irreducible WER floor of about 3–4% (oracle).

The winner's rules do **not** always follow the train majority. They split `namoni` → `na moni` and `varikufamba` → `vari kufamba`, which matches standard orthography, even though the joined forms are more frequent in train. This suggests, without proving it, that the final-test references or the winners' leaderboard probing favoured standard split forms.

**Recommended approach:**
- Build per-pair rules from train frequencies, using majority with a margin.
- Treat the handful of contested pairs (namoni, nazomona, Xri+ku) as **A/B leaderboard tests**, not as settled rules.

### Cited Findings
**Pasketti 1st-place rules** — [apply_best_postprocessing.py](https://raw.githubusercontent.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/main/postprocess/apply_best_postprocessing.py) [VERIFIED]; [postprocess README](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/tree/main/postprocess) [VERIFIED]

*Lingala, in this order:*
1. Split `namoni` → `na moni`.
2. Apply a `ba`/`bazo`/`baza` lookup table built by `waxal_tables(pool,'lin',3,.95)`, keeping the capital of the original token.
3. Split `nazomona` → `nazo mona` and `tozomona` → `tozo mona`.
4. `JOIN_RULES`: always join `ba` + {zali, tie, lati, komi, fandi, vandi, sali, telemi, lakisi}.
5. `NA_RULES`: split `na` + {ye, nga, kati, se, yango, ba, biso}.

There is no `ya` rule.

*Shona:*
- `apply_split` with `waxal_tables(pool,'sna',3,.95,grammatical=True)`, a frequent-form table. The arguments probably mean minimum count 3 and a share of 0.95, but this is [UNVERIFIED]: the defining module `respace_probe.py` was not found at the guessed raw URL (HTTP 404).
- A rare-form regex: `RARE_RIKU = r'(?i)^(a|va|i|ri|chi|zvi|dzi|u|pa|ti|ndi|ku)ri(ku.+)$'`. On a match, a space is inserted after `…ri`.
- `residual_confirmed_rules.py` catches the remaining progressive forms (6 rows changed in the winning run).

*Lexicon source:* `corpus/pool/{lin,sna}.waxal.txt`, i.e. WAXAL transcripts.

**Split vs joined counts in train + validation (lowercased, punctuation stripped) [LOCAL, seg.py]**

*Lingala:*

| Pair | Split | Joined | Split share | Pasketti action |
|---|---|---|---|---|
| na moni | 1,943 | 5,124 | 0.27 | **split** (against majority) |
| nazo mona | 610 | 2,742 | 0.18 | **split** (against majority) |
| tozo mona | 124 | 540 | 0.19 | **split** (against majority) |
| na ye | 1,393 | 574 | 0.71 | split (majority) |
| na kati | 3,403 | 550 | 0.86 | split |
| na yango | 880 | 91 | 0.91 | split |
| na ba | 1,861 | 322 | 0.85 | split |
| na se | 1,041 | 198 | 0.84 | split |
| na nga | 517 | 242 | 0.68 | split |
| na biso | 200 | 59 | 0.77 | split |
| ba zali | 54 | 793 | 0.06 | join |
| ba lati | 107 | 1,099 | 0.09 | join |
| ba telemi | 160 | 699 | 0.19 | join |
| ba lakisi | 355 | 878 | 0.29 | join |
| ba tie | 43 | 679 | 0.06 | join |
| ba komi | 59 | 461 | 0.11 | join |
| ba fandi | 55 | 371 | 0.13 | join |

These counts were not in Pasketti's rule lists but are useful for the same purpose:
- `eza na` 7,883 / 32 (always split)
- `na langi` 5,611 / 77
- `mpo na` 13 / 594 (written `mpona`)
- `po na` 69 / 196
- `ko mona` 183 / 2,277 (`komona`)
- `to moni` 155 / 1,430
- `bala bala` 221 / 1,537 (`balabala`)
- `ba langi` 483 / 497 (50/50)
- `ba nzete` 461 / 360
- `ba ndako` 149 / 179
- `na ngai` 837 / 194
- `tozali komona` 877 / 40

There are 110 Lingala pairs where both forms occur at least 30 times.

*Shona:*

| Pair | Split | Joined | Split share |
|---|---|---|---|
| vari kufamba | 528 | 667 | 0.44 (Pasketti splits) |
| ari kufamba | 199 | 281 | 0.41 |
| dziri kufamba | 259 | 233 | 0.53 |
| iri kufamba | 93 | 91 | 0.51 |
| vari kuita | 119 | 174 | 0.41 |
| vari kuratidza | 132 | 176 | 0.43 |
| tiri kuona | 101 | 172 | 0.37 |
| ine ruvara | 1,374 | 525 | 0.72 |
| ane ruvara | 964 | 476 | 0.67 |
| rine ruvara | 756 | 184 | 0.80 |
| pane imwe | 205 | 31 | 0.87 |
| ine mashizha | 380 | 56 | 0.87 |
| rwe girini | 63 | 696 | 0.08 (written `rwegirini`) |
| rwe bhuruu | 81 | 481 | 0.14 |
| pamwe chete | 152 | 173 | 0.47 |
| imwe chete | 194 | 142 | 0.58 |

- `pakati`, `mukati`, `pamusoro` and `mashizha` are always written joined (0 split occurrences).
- Aggregate regex counts: `vari ku…` 1,419 vs `variku…` 1,803; `ari ku…` 982 vs `ariku…` 1,132; `iri ku…` 731 vs `iriku…` 641; `zviri ku…` 357 vs `zviriku…` 311 [LOCAL, textstats.py].
- There are 46 Shona pairs where both forms occur at least 30 times.

**Oracle cost of applying one global convention to the validation references [LOCAL, sim.py]** (starting from the "clean" baseline WER, lin .0094 / sna .0067):
- **lin:**
  - Split all `na`+{moni, kati, ye, biso, yango}: WER .0383
  - Join them all: .0421
  - Join every `ba X`: .0561
  - Split `ba(zo|za)X`: .0176
- **sna:**
  - Split every `Xri+ku…` with Pasketti's regex: WER .0340
  - Join every `Xri ku…`: .0343

So the Shona progressive convention is a near coin-flip in train/validation, and it costs about 2.7% WER whichever way you choose.

**Shona orthography background:**
- [UNVERIFIED / prior knowledge] Standard Shona orthography (the 1967 conjunctive standard, as used in the *Duramazwi*) writes subject concord + verb as one word.
- Auxiliary constructions such as `vari kufamba` ("they are walking", from *-ri* "be" + infinitive) are conventionally written as two words, as are copulative/associative forms like `ine ruvara`.
- The train majority for `ine ruvara` (split) agrees with this. The near 50/50 for `vari kufamba` reflects annotator variation.
- I did not open a primary orthography source this session.

### Inferences
1. **Lexicon repair algorithm (testable).**
   - Build a unigram table U and a bigram table B from train transcripts, lowercased with punctuation detached (optionally add validation for the final model).
   - **Split rule.** For each hypothesis token w = a+b (try every cut with |a|, |b| ≥ 2): if B[a,b] ≥ k·U[w] and B[a,b] ≥ m, replace w by `a b`.
   - **Merge rule.** For each adjacent pair (a, b): if U[a+b] ≥ k·B[a,b] and U[a+b] ≥ m, merge them.
   - Start with k = 2 and m = 20. Sweep k ∈ {1.5, 2, 3, 5} on a speaker-disjoint validation set.
   - Re-attach punctuation and the case of the first character, as Pasketti does.
   - Compute the expected gain on validation per rule and keep only rules with a positive gain. Contested pairs with a split share between 0.4 and 0.6 have near-zero expected gain under the train distribution, so leave them to the model or to an A/B test.
2. **Why the winners' "against-majority" splits might be right on the test.**
   - Possible reasons: (a) the Phase 2 references were transcribed by a different team or with stricter QC that follows standard orthography; (b) the winners tuned these choices on the Phase 2 public leaderboard.
   - Neither can be confirmed.
   - **Cheap test:** two submissions identical except for (i) `namoni/nazomona/tozomona` split vs left as is, and (ii) Shona `Xri+ku` split vs joined. With about 30% public LB, a difference of about 0.002 or more is meaningful only if it is consistent.
3. **Expected effect size.**
   - Segmentation fixes affect **WER only**. CER changes by about 1 character per fix.
   - Oracle bounds suggest that correctly standardising the contested pairs is worth up to about 0.01 to 0.02 of score in lin and about 0.01 to 0.015 in sna, if the test references are consistent with the rule.
   - If the test references are as inconsistent as train, the expected gain from majority rules is smaller, at about 0.002 to 0.005. This is the researcher's estimate from the tables above.
4. **Avoid over-splitting `ba`+noun.** `ba langi`, `ba nzete` and `ba ndako` are about 50/50, while verbs (`bazali`, `balati`...) are about 90% joined. That is exactly why Pasketti uses a noun table plus a verb rejoin list.
5. **The lin 3rd-place remark** that "words blend together" in Lingala is consistent with these segmentation-dominated errors — [Google Africa blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-winners-of-the-waxal-speech-recognition-challenge/) (via the earlier notes).

### Gaps
- `respace_probe.py` (Pasketti's table builder) could not be fetched, so the exact meaning of the thresholds `3, .95, grammatical=True` is unverified.
- No public Digital Umuganda / AfriVoice transcription guideline was found that states the intended word-boundary convention.
- There is no primary Lingala orthography reference. Lingala has no single enforced standard; `na`/`ya` connective spelling varies.

---

## Q3. External text for LMs (MADLAD-400, Wikipedia, CC-100, Leipzig, Bible): licences, download, and whether LM fusion or rescoring helps CTC on African languages; spelling correction risk

### Takeaway
External text is **limited for Lingala and moderate for Shona**:
- MADLAD-400 clean: sn 31.6M tokens, ln 3.4M (ln is partly Bible/JW).
- CC-100: ln only, about 2.3M size units; no Shona.

The best LM text is the **in-domain WAXAL train transcripts**, kept **cased and punctuated** to match this metric. External text should be interpolated with a small weight.

Published evidence (Makerere benchmark, AfriVoice data):
- A 5-gram LM cut Shona w2v-BERT WER by about 60% relative.
- For Lingala it gave only about 6% relative with w2v-BERT, and **hurt** XLS-R by +34% relative.

So shallow fusion is high-value for Shona and must be tuned, or switched off, for Lingala.

SymSpell-style correction toward a train lexicon is risky. Validation OOV rates against the train vocabulary are 3.5% (lin) and **7.8% (sna)**, so valid unseen Shona words would be "corrected" into wrong ones.

### Cited Findings
**MADLAD-400** — [HF card allenai/MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400) [VERIFIED]
- Licence is inconsistent: the card text says "released with the CC-BY-4.0 license", while the header field says `odc-by`. Both are permissive with attribution.
- Loading per the card: `load_dataset("allenai/madlad-400", languages=["sn"], split="clean")`. The repo file layout (`data/sn/sn_clean_*.jsonl.gz`) was **not verified** on the card.
- Counts:

| Lang | Docs noisy | Docs clean | Sentences clean | Tokens noisy | Tokens clean | Size clean |
|---|---|---|---|---|---|---|
| sn | 3.1M | 60.2K | 1.2M | 1.3B | **31.6M** | 92.5 MB |
| ln | 94.7K | 3.3K | 139K | 42.4M | **3.4M** | 6.8 MB |

  The column mapping was interpreted by the fetch tool from the table layout. Tokens are whitespace tokens.
- Audit notes: sn "ok"; ln "ok bible jw", meaning it is partly Bible and jw.org text. The clean split drops documents with more than 20% questionable sentences or fewer than 5 sentences. The audit sampled 20 documents per language, by non-speakers.

**CC-100** — [HF card statmt/cc100](https://huggingface.co/datasets/statmt/cc100) [VERIFIED]
- Lingala is listed as "Lingala (2.3M)". The unit is not stated; the card's sizes are probably file sizes [UNVERIFIED].
- **Shona is not present.**
- Licence field: "unknown". Users are bound by the Common Crawl terms of use. Homepage: data.statmt.org/cc-100/. The HF loader is a script, so `trust_remote_code` is needed.
- [UNVERIFIED] Direct file pattern: `https://data.statmt.org/cc-100/ln.txt.xz`.

**Leipzig Corpora Collection** — [Wortschatz terms of usage](https://wortschatz.uni-leipzig.de/en/usage) (via search snippet; the page is bot-protected) [UNVERIFIED in full]
- Downloadable corpora are stated to be under CC BY, but the same page mentions CC BY-NC and forbids commercial use without permission. The terms are therefore ambiguous; for a non-commercial competition this is probably acceptable, but check.
- Download portal: https://wortschatz.uni-leipzig.de/en/download. An Anubis anti-bot page blocked the fetch, so the availability of Shona and Lingala corpora is **unverified**.

**Community WAXAL KenLMs** — [keystats/waxal-kenlm-models](https://huggingface.co/datasets/keystats/waxal-kenlm-models) (from the earlier notes, VERIFIED there)
- 5-gram models: `lmplz -o 5 -S 40% --discount_fallback` then `build_binary trie`, with a `</s>` unigram added.
- Text: WAXAL train plus other open lin/sna ASR transcript sets (e.g. `Beijuka/DigitalUmuganda_AfriVoice_shona`, `badrex/shona-speech`, `KasuleTrevor/Lingala_100hrs`).
- WAXAL validation and test were excluded.
- A separate "-best" set is built on **raw cased and punctuated** text.

**LM effect for CTC on African languages** — [arXiv 2512.10968 (Makerere benchmark)](https://arxiv.org/html/2512.10968) (VERIFIED in the earlier notes)
- 5-gram modified Kneser-Ney (KenLM) with pyctcdecode. LM text came from news, health, education, agriculture and religious sources: about 9.03M words (lin) and 5.37M (sna).
- w2v-BERT 2.0 at 50 h:
  - Shona: 23.54 → 8.61 WER
  - Lingala: 24.65 → 23.13
- XLS-R at 50 h:
  - Lingala: 19.85 → 26.55 (worse)
  - Shona: 30.71 → 34.73 (worse)
- The authors say LM gains depend on "how well the acoustic and text data line up".

**N-best / rescoring:** I found no source this session that measures N-best rescoring with an external LM on WAXAL or AfriVoice. **The 1st-place system used no LM to generate text.** It used character-level ensembling plus vote-verified OOV repair — [Pasketti postprocess README](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/tree/main/postprocess) [VERIFIED].

**Pasketti OOV repair (a safer alternative to SymSpell)** — [Pasketti postprocess README](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place/tree/main/postprocess) [VERIFIED]
- A one-edit replacement of an OOV ensemble token is proposed only if it comes from the same clip's member outputs with ≥5 votes, or ≥4 votes with a margin of 2.
- It is then verified by 6 architecturally diverse systems.
- 21 of 24 strict and 20 of 118 relaxed proposals survived, giving 41 token repairs over 39 rows; 101 proposals were rejected.
- That is a tiny number of edits, and very conservative by design.

**Vocabulary openness [LOCAL, oov.py]**

| | lin | sna |
|---|---|---|
| Train tokens | 394,123 | 329,644 |
| Train types | 18,812 | 38,563 |
| Hapax share of types | 62.9% | 63.3% |
| Validation token OOV rate vs train vocabulary | 3.47% | **7.76%** |
| Validation tokens with train count ≤ 2 | 5.4% | 12.5% |

Validation shares speakers with train in v1, so the true test OOV rate is probably higher.

**Other licences and sources:**
- Wikipedia text is CC BY-SA 4.0 [UNVERIFIED / prior knowledge]. Dumps follow the pattern `https://dumps.wikimedia.org/lnwiki/latest/lnwiki-latest-pages-articles.xml.bz2` and `https://dumps.wikimedia.org/snwiki/latest/snwiki-latest-pages-articles.xml.bz2` [UNVERIFIED URL pattern]. Article counts and sizes were not retrieved; both wikis are small (prior knowledge, unverified).
- Bible text: jw.org content is copyrighted under restrictive terms of use [UNVERIFIED]. Some Bible translations on eBible.org are public domain or CC-licensed [UNVERIFIED]. MADLAD ln already contains Bible/JW text. Religious register is a domain mismatch for image descriptions, which is consistent with the Makerere "domain match" caveat.
- SymSpell (wolfgarbe/SymSpell) and its Python port symspellpy are MIT-licensed [UNVERIFIED / prior knowledge]. KenLM is LGPL [UNVERIFIED / prior knowledge].

### Inferences
1. **The LM must match the metric's text form.**
   - If the acoustic CTC vocabulary contains uppercase and punctuation, a KenLM trained on lowercased, unpunctuated text will mis-score tokens such as `ndako.` and `Pane`.
   - **Options:**
     - (a) Train KenLM on **raw** WAXAL train text with punctuation **separated as its own token** (`ndako .`), decode, then re-attach. The keystats "-best" approach uses raw text.
     - (b) Decode with a lowercased, unpunctuated LM on a lowercased CTC vocabulary, then run punctuation and casing restoration (Q1) afterwards.
   - For Shona, where 83% of references have internal periods, (b) needs a good restoration step. (a) is simpler if the acoustic model was trained on raw text.
2. **Recommended LM recipe (testable).**
   - *Shona:* 4- or 5-gram, in-domain WAXAL train text (~330k tokens) **interpolated** with MADLAD sn clean (31.6M tokens). Either use KenLM-built separate ARPAs mixed by weight on validation perplexity (needs SRILM `ngram -mix-lm` or a custom mix; SRILM has a non-OSI licence), or, simplest, oversample the WAXAL text about 20–50× in the concatenated training file.
   - Grid alpha ∈ [0.2, 1.5] and beta ∈ [0, 3].
   - **Expected:** a large WER gain for a weak Shona acoustic model, smaller for a strong fine-tuned one. The Makerere result (−60% relative) used weak, unpunctuated baselines, so expect much less on a strong model (unverified size).
   - *Lingala:* in-domain-only LM first, and keep "no LM" as a candidate. Add MADLAD ln (3.4M, Bible-heavy) and CC-100 ln only if validation improves.
   - Choose per language and per acoustic model. The LM hurt XLS-R in both languages in Makerere.
3. **Ensembles.** When combining several systems (character-median / ROVER), apply the LM inside each CTC member's beam search, not after ensembling. Post-ensemble word-level rescoring needs N-best lists, and no evidence was found for it on WAXAL.
4. **Spelling correction risk.**
   - With 7.8% validation OOV in Shona, a SymSpell edit-distance-1 correction toward the train lexicon would rewrite many correct agglutinated words (e.g. a new `ndiri kuona` form) into frequent neighbours, raising WER.
   - **Safer variants:**
     - (a) Pasketti-style: only correct OOV tokens when other ensemble members propose the in-vocabulary alternative for the same clip.
     - (b) Restrict corrections to tokens whose CTC confidence is low **and** whose top-1 correction has count ≥ 50 and edit distance 1.
     - (c) Use an MADLAD-expanded lexicon (sn) to reduce false OOVs.
   - Measure each on validation: number of tokens changed, WER delta, CER delta.
   - **Expected:** a slight gain at best (about +0.001 to 0.003 score), and a loss if the thresholds are loose. This is an estimate.
5. **Do not include validation or test transcripts in the final LM used for model selection.** The final Phase 2 / Study Jam LM may include validation, but **never Phase 1 test labels** (forbidden).

### Gaps
- Per-language Wikipedia sizes, Leipzig Shona and Lingala availability, and exact CC-100 units were not verified (bot protection, budget).
- There are no published numbers for N-best rescoring or for SymSpell on Bantu-language ASR output.
- The LM gain on a **cased and punctuated** CTC model for these languages is unknown: no source evaluated that setup.

---

## Q4. Cheap adaptation under 8 GB VRAM in about 2 days: pseudo-labels / noisy student on WAXAL unlabeled audio, LoRA/adapters on w2v-BERT 2.0, freezing lower layers

### Takeaway
Evidence for big gains from self-training in this setting is **weak**:
- A WAXAL competitor (2nd-place author "anyantudre") built a high-precision 3-model agreement pseudo-label set from the WAXAL `unlabeled` split, about 15k clips per language, but published no gain.
- The user's note that his self-training was "flat" could not be verified from public pages.
- In a Swiss-German wav2vec2 study, noisy student training was **worse** than the baseline.
- A Swahili w2v-BERT "pseudo-label continued pretraining" paper reports large gains but confounds data size.

With about 2 days on 8 GB, pseudo-labelling is a lower-priority bet than post-processing (Q1–Q2), LM tuning (Q3) and ensembling. If tried, use strict multi-model agreement filtering, mix about 1:1 with real labels, and fine-tune only the top layers or LoRA.

### Cited Findings
**anyantudre/waxal-pseudo** — [HF dataset (gated)](https://huggingface.co/datasets/anyantudre/waxal-pseudo) [VERIFIED card]
- Pseudo-labels for the WAXAL `unlabeled` split of `google/WaxalNLP` (lin/lug/sna), from a 3-model agreement cascade:
  - fine-tuned w2v-BERT-2.0 CTC "champion" vs XLS-R-300m, kept if CER ≤ 0.12;
  - then confirmed by a fine-tuned Omnilingual-ASR-300M at CER ≤ 0.22.
  The label is the champion's transcript.
- 44,948 rows in total, about 15k per language, 27.4 GB. Durations are about 1 to 25 s.
- "Contains no Phase-1 test transcriptions".
- "PRIVATE working artifact … not for redistribution". Licence "other".
- **No WER/CER results reported.**
- Identification of this author as the 2nd-place winner (Alban Nyantudre) comes from the name match with the Google winners blog. That is an inference, [UNVERIFIED].
- The user's claim that "self-training was flat" was **not found in any public source** this session; the GitHub profile [github.com/anyantudre](https://github.com/anyantudre) shows no WAXAL repo [VERIFIED].

**Swahili pseudo-labelled continued pretraining** (arXiv 2603.11378, 2026) — [arXiv HTML](https://arxiv.org/html/2603.11378v1) [VERIFIED]
- Setup: facebook/w2v-bert-2.0 with a CTC head. The labeller was fine-tuned on Common Voice Swahili and produced greedy CTC labels; segments with confidence > 75% were kept.
- Results: 5K labelled + CPT 10.89% WER and 20K + CPT 3.24%, vs a 50K-labelled no-CPT baseline at 17.71%.
- There is **no matched ablation**: the 5K/20K runs have no non-CPT counterpart, so attribution to pseudo-labelling is unproven. The amount of unlabelled audio and the compute are not reported.

**Swiss German** (Interspeech 2022, Schraner et al.) — [ISCA archive](https://www.isca-archive.org/interspeech_2022/schraner22_interspeech.html) (search snippet) [UNVERIFIED in full]
- Supervised training on noisy labels gave 26.4% WER, wav2vec pretraining 27.8%, and noisy student training 30.3% (worse).

**Iterative / continuous pseudo-labelling literature** [search results, not opened, UNVERIFIED in detail]:
- [Self-training for E2E ASR, arXiv 1909.09116](https://arxiv.org/pdf/1909.09116)
- [Semi-supervised ASR by E2E self-training, arXiv 2001.09128](https://arxiv.org/pdf/2001.09128): reports 14.4% relative WER reduction on WSJ (snippet)
- [Continuous pseudo-labeling from the start, arXiv 2210.08711](https://arxiv.org/pdf/2210.08711): warns that pretraining before pseudo-labelling can overfit in low-resource settings (snippet)

**Pasketti 1st place:** used a "Whisper SALT noisy-student" member among 8 systems. Training was two-stage for some models (frozen encoder, then full fine-tune) on an A6000 96 GB. No ablation was published — [Pasketti README](https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place) (VERIFIED in the earlier notes).

**w2v-BERT 2.0 memory:** 600M params. A full AdamW fine-tune is about 9.6 GB of weights, gradients and optimizer state, so it does not fit in 8 GB without 8-bit Adam, layer freezing or LoRA. This is the researcher's estimate from `research_notes/ASR strategy WAXAL lin sna lug/finetuning_and_lm.md` §2.

**LoRA / adapters on w2v-BERT 2.0 CTC:** no source with measured WER was found this session. PEFT LoRA on `Wav2Vec2BertForCTC` linear layers (`linear_q`, `linear_k`, `linear_v`, `linear_out`, `intermediate_dense`, `output_dense`) is technically possible [UNVERIFIED: module names from prior knowledge; check `model.named_modules()`].

### Inferences
1. **Priority under about 2 days and 8 GB:** (1) Q1 text rules; (2) Q2 segmentation A/B; (3) LM alpha/beta per language; (4) ensembling. These cost minutes to hours and have quantified oracle upper bounds. Pseudo-labelling costs about 1 day of GPU time for an uncertain, possibly zero, gain. Do it last, if at all.
2. **If you do pseudo-labelling (testable recipe):**
   - Source only the WAXAL `unlabeled` lin/sna split. Never Phase 2 audio, never Phase 1 test labels.
   - Label with your best ensemble.
   - Keep clips where ≥2 diverse systems agree at CER ≤ 0.10–0.12 (the anyantudre thresholds) and the length ratio is sane. This keeps "easy" clips, so the expected benefit is mostly robustness and domain coverage.
   - Fine-tune the current best checkpoint for 1–2 epochs on about 1:1 real : pseudo data, at a low LR (about 1e-5 for w2v-BERT top layers) with SpecAugment on (the "noisy" in noisy student).
   - Freeze the CNN, the feature projection and the bottom 50% of conformer layers, or use LoRA r=16–32 on attention and FFN linears plus a trainable CTC head.
   - Accept the change only if it beats the baseline on a **speaker-disjoint** validation set by more than about 0.003 of score.
   - **Expected:** −0 to −5% relative WER, by analogy with the literature cited above (unverified for this data). Agreement-filtered labels add little information the teacher did not already have.
3. **Pseudo-labels must carry the metric's text form.** If the teacher outputs lowercase unpunctuated text, training the student on it will **destroy** Shona punctuation and casing (Q1, a cost of about 0.04 to 0.08 of score). Either keep punctuation and case in the pseudo-labels, or train only a lowercase model whose output is restored afterwards.
4. **Cheaper "adaptation" alternatives:**
   - Fine-tune the acoustic model for 1–2 epochs on raw-text transcripts (cased and punctuated) if it was trained on normalised text. This directly targets the Q1 losses.
   - Average the last 3–5 checkpoints.
   - Tune the language-routing threshold for lin vs sna, since Phase 2 had no language metadata.

### Gaps
- No public measurement of self-training gains on WAXAL lin/sna, and no confirmation of the "flat" result.
- No measured LoRA vs partial-freeze vs full fine-tune comparison for w2v-BERT 2.0 on African ASR within 8 GB.
- The anyantudre pseudo-label dataset is gated, and redistribution is disallowed by its author, so it should not be used without permission. Its licence is "other".

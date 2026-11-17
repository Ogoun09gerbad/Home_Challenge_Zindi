# Extra public ASR checkpoints for Lingala and Shona trained on Google WAXAL: candidate ensemble members

Research date: 2026-10-26. Method: I queried the HF Hub API (`/api/models?search=waxal`, which returned **457 repos**, not ~200) and pulled README, file tree and `vocab.json` for about 80 lin/sna ASR repos. I compared every downloadable CTC `vocab.json` against `anyantudre/waxal-w2vbert-linsnalug-raw`. I also read three participant GitHub READMEs: yehoshua0 (4th), DariusTheGeek (9th) and the omnilingual-asr/fairseq2 docs. "UNVERIFIED" marks anything not confirmed from a primary source. VRAM figures are my estimates from weight-file size unless they cite the omnilingual table. Scores are self-reported by model authors.

## Q1. Candidate inventory: which public checkpoints exist, and what are they?

### Takeaway
Besides the anyantudre lineage, few lin/sna checkpoints are both clean and documented. The most useful ones come from a **different root**:
- keystats w2v-BERT: an independent run with a cased+punct vocabulary.
- ayymen omniASR-CTC-300M-v2 HF port and ayymen w2v-BERT.
- DariusTheGeek omniASR CTC-1B / LLM-1B (Apache-2.0, 9th place).
- mrwalker123 Whisper-large-v3 QLoRA.
- Paulownia omniASR CTC-1B.
- waxal-benchmarking omniASR/Whisper/MMS (lowercased output).

Most yehoshua01 checkpoints, and anyantudre's distilled, soup6 and whisper-turbo, were trained on Phase 2 test audio or on Phase 1 test gold.

### Cited Findings

**Master table.** Status key: ✅ = no test data per the card, ⚠️ = unverified or risky, ❌ = trained on Phase 1 test gold or Phase 2 test audio (exclude).

| Repo ID | Arch | Size (weights on Hub) | Vocab / output | Langs | License | Reported scores | VRAM 8 GB inference (est.) | Test-data status |
|---|---|---|---|---|---|---|---|---|
| `keystats/w2v-bert-2.0-multilingual-waxal-shona-lingala-luganda-best` | w2v-BERT 2.0 CTC | 2.42 GB fp32 (~600M) | char, **cased+punct**, 121 entries. Same symbol set as anyantudre except it lacks `ë`, `ᵑ` and adds `“ ”`. Built independently, so the **token order differs** | sna+lin+lug | Apache-2.0 | "Evaluation numbers … aren't included" | ~2 GB fp16 | ✅ WAXAL train only (val used for early stopping) |
| `keystats/...-luganda-lusoga` | same | 2.42 GB | same style | +Lusoga | Apache-2.0 | none ("weren't retained") | ~2 GB | ✅ |
| `ayymen/w2v-bert-2.0-waxal_lin_sna-1epoch` | w2v-BERT 2.0 CTC | 2.42 GB model (9.69 GB repo incl. last-checkpoint) | char, **cased+punct** (`! " ' , - . ; ?`, no `:`), 97 entries, **no digits**. Not the anyantudre vocab | lin+sna | MIT | eval WER 0.3383 / CER 0.1054 (eval split "./WAXAL_LIN_SNA", unstated) | ~2 GB | ⚠️ data split not documented |
| `ayymen/w2v-bert-2.0-waxal_lin_sna-2epoch`, `..._bench_fleurs_lin_sna-2epoch` | – | **empty repos** (only .gitattributes) | – | – | – | – | – | n/a |
| `ayymen/omniASR-CTC-300M-v2-waxal_lin_sna-1epoch` / `-2epoch` | omniASR CTC-300M v2, **HF-transformers port** (base `aadel4/omniASR-CTC-300M-v2`) | ~1.3 GB model (5.22 GB repo) | omni "written" tokenizer exported as vocab.json, **10,288 entries**, cased+punct | lin+sna | no tag (omni base is Apache-2.0) | 1-epoch: WER 0.4072 / CER 0.1181; 2-epoch: no card | <2 GB | ⚠️ split undocumented |
| `waxal-benchmarking/omniasr-ctc-300m-waxal-lin` / `-sna` | omniASR CTC-300M (fairseq2) | 1.30 GB | **lowercased, punct removed** | 1 each | CC-BY-4.0 | lin WER 31.2% (card replaced Oct 2026: the first upload was a step-1,000 checkpoint) | ~2 GiB | ✅ train split; WAXAL test used for eval only |
| `waxal-benchmarking/omniasr-llm-300m-waxal-all19` / `-all19-20k` / `-bantu` | omniASR LLM-300M (fairseq2) | 3.26 GB | lowercased, no punct | 19 / Bantu group | CC-BY-4.0 | all19: lin 29.9/12.5, sna 21.6/3.9 (prior note); all19-20k macro WER 32.5; bantu macro 31.6 | ~5 GiB | ✅ train splits only |
| `waxal-benchmarking/whisper-small-waxal-lin` / `-sna` | Whisper-small | ~0.97 GB | Whisper BPE; normalisation UNVERIFIED | 1 each | CC-BY-4.0 | lin 42.7/21.7; sna 26.9/5.1 (filtered WAXAL test) | ~1.5 GB | ✅ |
| `waxal-benchmarking/mms-300m-waxal-lin` / `-sna` | MMS-300M CTC | ~1.2 GB | normalised (lowercase) | 1 each | **CC-BY-NC-4.0** | lin 42.6/18.9; sna 25.0/4.3 | ~1.5 GB | ✅ |
| `DariusTheGeek/waxal-lin-omniasr-ctc-1b` | omniASR CTC-1B (fairseq2 `model.pt`) | 3.90 GB | omni written-v2 SentencePiece (can emit case and punct; whether the fine-tune keeps it is UNVERIFIED) | lin | Apache-2.0 | part of 9th-place system: public 0.758111 / private 0.767488 | ~3 GiB | ✅ per README: gold 32,328 rows; the unlabelled pool was used only for a voice gallery |
| `DariusTheGeek/waxal-lin-omniasr-llm-1b`, `waxal-sna-omniasr-llm-1b` | omniASR LLM-1B v2 | 9.12 GB file | same tokenizer | lin / sna | Apache-2.0 | no per-model score in README | ~6 GiB (tight) | ✅ (same) |
| `DariusTheGeek/waxal-lin-omniasr-ctc-3b` | omniASR CTC-3B v2 | 12.33 GB | same | lin | Apache-2.0 | "3B scores highest of the Lingala models on validation" | ~8 GiB (**does not fit reliably**) | ✅ |
| `DariusTheGeek/waxal-lin-omniasr-llm-3b`, `-sna-omniasr-llm-3b`, `-sna-omniasr-ctc-7b` | LLM-3B / CTC-7B | 17.5 GB / 78 GB | same | – | Apache-2.0 | – | ~10 / ~15 GiB (**no**) | ✅ |
| `DariusTheGeek/waxal-joint-ctc-1b-lid` | omniASR CTC-1B, joint lin+sna + LID | 3.90 GB | same | lin+sna | Apache-2.0 | routing lane | ~3 GiB | ✅ |
| `DariusTheGeek/waxal-lin-mms-1b`, `waxal-sna-mms-1b` | MMS-1B-all CTC | 3.86 GB | **lowercase, apostrophe only**, 44 / 39 tokens | lin / sna | **CC-BY-NC-4.0** | – | ~2.5 GB | ✅ |
| `Paulownia/waxal-omniasr-ctc-1b-multilingual` | omniASR CTC-1B (fairseq2) | best/final `model.pt` (repo 66 GB with resume state) | omni tokenizer | lin+sna+lug | Apache-2.0 | speaker-held-out internal WER 24.64% (29,308 train utts / 164.28 h; dev 4,655 utts) | ~3 GiB | ⚠️ the source splits of the 164 h are not stated |
| `mrwalker123/whisper-large-v3-waxal-lin-sna` | Whisper-large-v3 + QLoRA r64 adapters (`swa/`, `best-checkpoint-4000/`) | 1.01 GB adapters + base | Whisper BPE (cased+punct native) | lin+sna | MIT | eval WER 0.3234 (step 4000) | ~4–5 GB fp16 merged | ✅ "WAXAL Lingala + Shona train/validation" |
| `Abdoul27/waxal-af51-linsna-ck1200` | Sunbird-51 (Whisper-large-v3) full FT | 3.09 GB (fp16) | **raw cased** labels | lin+sna | MIT | speaker-disjoint holdout: zero-shot 0.8854 → ck1200 lower (0.8787 at step 600) | ~4–5 GB | ⚠️ base Sunbird already trained on WAXAL |
| `Abdoul27/waxal-mms1b-lin-v4-ck3000` | MMS-1B-all + lin adapter | 3.86 GB | Meta lin adapter vocab (81 symbols, lowercase + punct) | lin | no tag (base **CC-BY-NC**) | zero-shot holdout 0.8438; v3 + KenLM probe 0.289448 | ~2.5 GB | ✅ train (8 speakers held out) |
| `Abdoul27/w2vbert2-lin-waxal` | w2v-BERT CTC | 2.42 GB per ckpt | **lowercase, punct stripped**, 43 symbols | lin | no tag | Zindi Lingala-half probe 0.280922 greedy / 0.287731 +KenLM | ~2 GB | ✅ 16,320 WAXAL lin clips |
| `Sunbird/SunflowerASR-51-african-languages` (old ID `Sunbird/asr-whisper-51-african-languages` now redirects) | Whisper-large-v3 FT | ~3.1 GB fp16 | cased+punct | 51 incl. lin, sna | Apache-2.0, **gated (auto)** | lin 24.3/10.4, sna 19.6/5.0 (own dev); as yehoshua's Lingala half zero-shot: 0.747908 public | ~4–5 GB | ⚠️ trained on WAXAL lin/sna (which splits is unknown) |
| `Okwija/sunbird-whisper-lin-sna-V2` | Sunbird-51 FT | ~6.2 GB fp32 | Whisper BPE | lin+sna | no tag | eval WER 0.2508 / CER 0.0425 | ~4–5 GB | ⚠️ data unstated; base already saw WAXAL |
| `GodwillN/w2vbert-waxal` | w2v-BERT CTC | 2.42 GB | lowercase, 70 entries | ? | MIT | WER 0.4034 / CER 0.2183 | ~2 GB | ⚠️ |
| `sulaimank/w2vbert-waxal-r5` | w2v-BERT CTC | 2.42 GB | cased (24 capitals) + `! ' , . ; ?`, 68 entries | multi | no license | none | ~2 GB | ⚠️ siblings `-multi7-pseudo`, `-p2`, `-p2b` suggest pseudo-label rounds; source pool unknown |
| `sulaimank/w2vbert-{lingala,shona}-waxal-punct(-v2)`, `whisper-small-{lingala,shona}-cased` | w2v-BERT / Whisper-small | 2.42 / 0.97 GB | cased+punct by name | 1 each | Apache/MIT | – | small | ⚠️ **gated (manual)**, card unreadable |
| `douyeszn/w2vbert-{lin,sna}-waxal-aug(-ft)` | w2v-BERT CTC | 2.42 GB | – | 1 each | Apache-2.0 | – | ~2 GB | ⚠️ gated manual; `-pseudo` siblings exist |
| `emmilly/waxal-fair-{mms,whisper,omni}-{native,restore}-*` (Oct 2026) | MMS / Whisper / omni | 1.93 / 3.09 / 4.56 GB | `mms-native-lin` vocab 102 entries, cased+punct; `restore` = lowercase (66 entries) | lin, sna, multi | no card | – | fits | ⚠️ no documentation |
| `emmilina/waxal-mms-adapters` (`lin_full_casepunct/` …), `waxal-omni-ctc-300m-lug-sna-lin-casepunct`, `waxal-whisper-large-v3-v6` | MMS / omni / Whisper | various | casepunct by name | lin/sna/lug | no card | – | fits (omni-300M, MMS); large-v3 ~5 GB | ⚠️ |
| `ravitejac/waxal-asr-solution` | omniASR-LLM-7B FT + `w2vbert-{lin,sna}-v1` CTC + Gemma-4 LoRA corrector | 7B not usable on 8 GB; w2vbert ~2.4 GB | – | lin+sna | Apache-2.0 (repo) | submission public 0.757397 / private 0.766450 | w2vbert only | ⚠️ "TTA specialist" checkpoint implies test-time adaptation |
| `anyantudre/waxal-w2vbert-linsna-afrivoicemix` | w2v-BERT CTC | 2.42 GB | **identical vocab.json** to the reference | lin+sna | Apache-2.0 | public 0.726089 solo | ~2 GB | ✅ train+val + AfriVoice. Near-duplicate of the reference |
| `anyantudre/waxal-w2vbert-{lin,sna}-specialist` | w2v-BERT CTC | 2.42 GB | identical vocab | 1 each | Apache-2.0 | specialist pair public 0.744835 | ~2 GB | ✅ train+val + AfriVoice; near-duplicate root |
| `anyantudre/waxal-whisper-turbo-linsna` | Whisper-large-v3-turbo FT | 3.24 GB | Whisper BPE, raw cased+punct | lin+sna | Apache-2.0 | public 0.7331 solo | ~3 GB | ❌ **trained on the WAXAL test split gold** (3,615 rows) |
| `anyantudre/waxal-w2vbert-linsna-distilled` | w2v-BERT CTC | 2.42 GB | identical vocab | lin+sna | Apache-2.0 | public 0.746787 | ~2 GB | ❌ test split gold **plus the 892 Phase 2 test clips** with ensemble labels |
| `anyantudre/waxal-w2vbert-linsna-soup6` (and `soup5` lineage) | weight soup | 2.42 GB | identical vocab | lin+sna | Apache-2.0 | – | ~2 GB | ❌ includes test split and the distilled member |
| `anyantudre/waxal-w2vbert_multi_s42`, `waxal-whisper_small_multi_s42` | w2v-BERT / Whisper-small | – | lowercase 79 entries / BPE | multi | –/Apache | whisper-small WER 0.7159 | small | weak |
| `yehoshua01/waxal-whisper-turbo-lin-r1` / `-r2` | Whisper-large-v3-turbo FT (decode with `<|sw|>`) | 3.24 GB | cased BPE | lin | Apache-2.0 | r1 as solo Lingala half: 0.752910 | ~3 GB | ❌ card: pseudo-labels on Phase 2 test audio |
| `yehoshua01/waxal-mms-1b-lin-pl2-spk` | MMS-1B-all full FT | 3.86 GB | lin adapter vocab (81) | lin | CC-BY-NC-4.0 | 4th-place primary (private 0.771848) | ~2.5 GB | ❌ PL + speaker adaptation on test audio |
| `yehoshua01/waxal-sunbird51-sna-pl2-spk` | Sunbird-51 FT | 6.17 GB | cased BPE | sna | CC-BY-SA-4.0 | Shona primary | ~4–5 GB | ❌ |
| `yehoshua01/waxal-mms-1b-lin-full`, `-fullmeta` | MMS-1B-all FT | 3.86 GB | 81-symbol lowercase + punct | lin | CC-BY-NC-4.0 | `-full` = "no pseudo-labels" ablation (its chain reached 0.747925); fullmeta 0.694839 vs 0.684586 on dev | ~2.5 GB | ⚠️ the title says no PL, but the boilerplate says "partly adapted to that specific set" |
| `yehoshua01/waxal-omni-ctc1b-lin` / `-sna`, `waxal-qlora-largev3-lin`, `waxal-sunbird51-lin-ft-{r2,light}`, `waxal-ssa-hubert-lin` | omni CTC-1B / QLoRA large-v3 / Sunbird / HuBERT-large | 3.9 / 0.46 / 6.17 / 3.85 GB | omni cased; BPE; HuBERT vocab 112 cased+punct | lin/sna | CC-BY-NC (omni, hubert), Apache (qlora, sunbird-lin) | omni-sna voter +0.000253; qlora as Lingala half 0.742006 | fit | ❌ all carry the "pseudo-labels were computed on the phase-2 test audio" clause |
| `Mubarak127/waxal-whisper-large-v3-sna_asr` | Whisper-large-v3 | 6.17 GB fp32 | BPE | sna | Apache-2.0 | none | ~4–5 GB | ⚠️ "unknown dataset" |
| `rwamux/whisper-large-v3-lin-waxal` (actually a **whisper-small LoRA** despite the name), `ronnybscales/whisper-small-shona-waxal`, `sandilecwili007/waxal-multilingual-small`, `Mavela/*`, `Offei-lad/*`, `patrick2kayemba/whisper-medium3-multilingual-waxal`, `VelkroLM/*` (empty) | Whisper small/medium/large | small | BPE | – | Apache / none | rwamux WER 0.4711 / CER 0.2272; ronnybscales WER 52.0; sandilecwili007 WER 59.9 | small | ⚠️ weak or undocumented |
| `Okwija/mms-adapter-waxal-{lin,sna}`, `mms-asr-waxal-pooled`, `whis-waxal`, `gemma3n-asr-{lin,sna}-waxal(-merged)` | MMS adapters / Whisper / Gemma 3n | 1.27 / 3.09 / 10.8 GB | MMS vocab lowercase, **no punct** (61 / 58) | lin/sna | none (Gemma terms for gemma3n) | none | fit (gemma3n ~11 GB: no) | ⚠️ |
| `KasuleTrevor/*` | XLS-R / w2v-BERT / Whisper on **AfriVoice Lingala** (e.g. `w2v-bert-2.0-ln-afrivoice-10hr-v4`, `wav2vec2-xlsr-ln-50hr-v2`, `whisper-ln-afrivoice-20hr-v1r`) | – | not WAXAL-style | lin | – | – | fit | ✅ no WAXAL. Out-of-domain orthography |

Sources: HF API search [huggingface.co/models?search=waxal](https://huggingface.co/models?search=waxal) (457 hits on 2026-10-26); model cards at `https://huggingface.co/<repo-id>`, including [keystats best](https://huggingface.co/keystats/w2v-bert-2.0-multilingual-waxal-shona-lingala-luganda-best), [ayymen w2v-bert](https://huggingface.co/ayymen/w2v-bert-2.0-waxal_lin_sna-1epoch), [ayymen omni](https://huggingface.co/ayymen/omniASR-CTC-300M-v2-waxal_lin_sna-1epoch), [wb omni-ctc lin](https://huggingface.co/waxal-benchmarking/omniasr-ctc-300m-waxal-lin), [wb bantu](https://huggingface.co/waxal-benchmarking/omniasr-llm-300m-waxal-bantu), [wb all19-20k](https://huggingface.co/waxal-benchmarking/omniasr-llm-300m-waxal-all19-20k), [wb whisper-small-lin](https://huggingface.co/waxal-benchmarking/whisper-small-waxal-lin), [Darius ctc-1b](https://huggingface.co/DariusTheGeek/waxal-lin-omniasr-ctc-1b), [Darius mms](https://huggingface.co/DariusTheGeek/waxal-lin-mms-1b), [Paulownia](https://huggingface.co/Paulownia/waxal-omniasr-ctc-1b-multilingual), [mrwalker123](https://huggingface.co/mrwalker123/whisper-large-v3-waxal-lin-sna), [Abdoul27 af51](https://huggingface.co/Abdoul27/waxal-af51-linsna-ck1200), [Abdoul27 mms](https://huggingface.co/Abdoul27/waxal-mms1b-lin-v4-ck3000), [Abdoul27 w2vbert](https://huggingface.co/Abdoul27/w2vbert2-lin-waxal), [Okwija sunbird V2](https://huggingface.co/Okwija/sunbird-whisper-lin-sna-V2), [GodwillN](https://huggingface.co/GodwillN/w2vbert-waxal), [ravitejac](https://huggingface.co/ravitejac/waxal-asr-solution), [anyantudre afrivoicemix](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-afrivoicemix), [anyantudre lin-specialist](https://huggingface.co/anyantudre/waxal-w2vbert-lin-specialist), [anyantudre whisper-turbo](https://huggingface.co/anyantudre/waxal-whisper-turbo-linsna), [anyantudre distilled](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-distilled), [anyantudre soup6](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-soup6), [yehoshua01 turbo r1](https://huggingface.co/yehoshua01/waxal-whisper-turbo-lin-r1), [yehoshua01 mms full](https://huggingface.co/yehoshua01/waxal-mms-1b-lin-full), [yehoshua01 omni-ctc1b-lin](https://huggingface.co/yehoshua01/waxal-omni-ctc1b-lin), [rwamux](https://huggingface.co/rwamux/whisper-large-v3-lin-waxal), [Sunbird API redirect](https://huggingface.co/api/models/Sunbird/asr-whisper-51-african-languages) → [Sunbird/SunflowerASR-51-african-languages](https://huggingface.co/Sunbird/SunflowerASR-51-african-languages), [KasuleTrevor](https://huggingface.co/KasuleTrevor); 9th-place README [DariusTheGeek/waxal-asr-solution](https://github.com/DariusTheGeek/waxal-asr-solution); 4th-place README [yehoshua0/waxal-asr-phase2](https://github.com/yehoshua0/waxal-asr-phase2).

Other facts:
- Leaderboard context: yehoshua0 finished 4th, private 0.771848284 / public 0.762400298 — [yehoshua0 README](https://github.com/yehoshua0/waxal-asr-phase2). DariusTheGeek finished 9th, private 0.767487837 / public 0.758111283 — [Darius README](https://github.com/DariusTheGeek/waxal-asr-solution). anyantudre's final submission scored 0.766791 public / 0.772552 private — [anyantudre card](https://huggingface.co/anyantudre/waxal-w2vbert-linsnalug-raw).
- **Reference vocab size discrepancy:** the anyantudre card says "97 tokens". The `vocab.json` on the Hub has **121 entries**: 31 single uppercase chars incl. accented ones, `[UNK]`, `[PAD]`, `|`, digits, `« »`, rare symbols such as `⭐`, `ᵑ`, `Ŋ`. The siblings afrivoicemix and lin-specialist carry a byte-identical vocab.json, and no non-anyantudre checkpoint has the identical vocab — [anyantudre card](https://huggingface.co/anyantudre/waxal-w2vbert-linsnalug-raw); vocab.json downloaded via `https://huggingface.co/<repo>/resolve/main/vocab.json`.

### Inferences
- No checkpoint outside the anyantudre family shares its CTC token order. Weight-averaging or logit-averaging with the reference is therefore only possible inside that family. Every external member has to be combined at the text level (ROVER/MBR) or by CTC posterior fusion after a vocab mapping. keystats is the closest, since it has the same symbol set except for 4 symbols.
- Usable on 8 GB and clean per their cards: keystats (w2v-BERT), waxal-benchmarking omni-CTC/LLM-300M, Whisper-small and MMS-300M, Darius omni CTC-1B / LLM-1B / joint CTC-1B, Paulownia CTC-1B, mrwalker123 large-v3 QLoRA, Abdoul27 mms1b-lin and w2vbert2-lin. ayymen (two roots) is usable if its data split is acceptable.

### Gaps
- ~370 of the 457 "waxal" hits are other languages or TTS. I did not open cards for Luganda-only repos (Phase 2 has no Luganda per the anyantudre card).
- Gated repos (sulaimank `*-punct`, `*-cased`; douyeszn) could not be read without auth.
- No numbers exist for keystats, emmilly, emmilina, Okwija or Mubarak127. Every self-reported eval WER uses a different split and normalisation, so the numbers are not comparable.
- Zindi's official top-3 solutions were not found publicly ([Zindi page](https://zindi.africa/competitions/google-waxal-asr-challenge) shows the competition closed). `ravitejac/waxal-asr-solution` does not state its rank.

## Q2. Which candidates output cased + punctuated text (CER on raw string; WER on lowercased text with punctuation kept)?

### Takeaway
Cased+punct output: all Whisper fine-tunes trained on raw WAXAL text, keystats, ayymen (both), sulaimank r5 / "-punct" / "-cased", emmilly `mms-native`, yehoshua `ssa-hubert`, and the omniASR models that kept the written tokenizer (ayymen; probably Darius and Paulownia).

Lowercase/normalised output that needs truecase + punctuation restoration before fusion: waxal-benchmarking (all), Darius MMS, Okwija MMS, Abdoul27 w2vbert, GodwillN, anyantudre `_multi_s42`, emmilly `restore`.

### Cited Findings
- The metric's behaviour was established empirically. Changing the capitalisation of 50 rows left WER identical to 9 decimals while CER moved. Adding 87 commas changed WER. So CER is on the raw string and WER on lowercased but un-stripped text. The raw vocab was worth +0.0171 on the LB — [anyantudre card](https://huggingface.co/anyantudre/waxal-w2vbert-linsnalug-raw)
- keystats: vocab built with "NFKC-only normalization"; case and punctuation preserved; outputs "cased, punctuated text" — [keystats card](https://huggingface.co/keystats/w2v-bert-2.0-multilingual-waxal-shona-lingala-luganda-best)
- waxal-benchmarking omniASR LLM: "transcripts NFC-normalized, lower-cased, punctuation removed, diacritics kept" — [all19-20k card](https://huggingface.co/waxal-benchmarking/omniasr-llm-300m-waxal-all19-20k); the same for bantu — [bantu card](https://huggingface.co/waxal-benchmarking/omniasr-llm-300m-waxal-bantu)
- Abdoul27 w2v-BERT: "Training text is lowercased with punctuation stripped" (43 symbols) — [card](https://huggingface.co/Abdoul27/w2vbert2-lin-waxal). Abdoul27 AF51: "RAW cased labels; normalisation only at scoring" — [card](https://huggingface.co/Abdoul27/waxal-af51-linsna-ck1200)
- anyantudre whisper-turbo: Whisper's own tokenizer unchanged; targets NFC only, no lowercasing, no punctuation stripping — [card](https://huggingface.co/anyantudre/waxal-whisper-turbo-linsna)
- Vocab inspection (this session, from each repo's `vocab.json`):
  - Cased+punct: ayymen w2v-BERT (97 entries, 29 capitals), ayymen omni (10,288 entries), emmilly mms-native-lin (102), yehoshua ssa-hubert (112), sulaimank r5 (68, 24 capitals).
  - Lowercase: Darius MMS (44/39, `'` only), Okwija MMS (61/58, no punct), GodwillN (70), anyantudre multi_s42 (79), emmilly mms-restore (66).
  - The MMS-1B `lin` adapter vocab used by yehoshua01 and Abdoul27 has 81 lowercase symbols incl. `! , . : ; ?`.
  - Sources: [Darius mms](https://huggingface.co/DariusTheGeek/waxal-lin-mms-1b), [Okwija](https://huggingface.co/Okwija/mms-adapter-waxal-lin), [emmilly](https://huggingface.co/emmilly/waxal-fair-mms-native-lin-lr3e-05-s1), [yehoshua01 mms full](https://huggingface.co/yehoshua01/waxal-mms-1b-lin-full)
- Restoration tools exist on the Hub: `sulaimank/truecaser-lin-sna` (afro-xlmr-large token classifier, accuracy 0.9932, cap recall 0.9319), `Professor/waxal-punct-restore-lin-sna` (no card), `sulaimank/waxal-punct-restorer` — [truecaser card](https://huggingface.co/sulaimank/truecaser-lin-sna); [sulaimank author list](https://huggingface.co/sulaimank)
- yehoshua's shipped CSVs "carry a formatter (leading capital, terminal period) that a raw decode does not". Raw MMS decodes therefore matched only 1 row of 892 exactly while reproducing 22,733 of 22,749 words — [yehoshua0 README](https://github.com/yehoshua0/waxal-asr-phase2)
- "Every external Lingala candidate failed with the same signature: WER down, CER up, net negative — the deficit is orthography" — [yehoshua01 ssa-hubert card](https://huggingface.co/yehoshua01/waxal-ssa-hubert-lin)

### Inferences
- Before ROVER or MBR, map lowercase members to the reference's casing and punctuation. One option is to vote on lowercased tokens and re-project case and punctuation from the pivot (anyantudre or keystats). Otherwise a lowercase member wins votes it should not, and CER rises.

### Gaps
- Whether Darius and Paulownia omniASR fine-tunes still emit capitals and punctuation was not verified (I did not inspect the training configs).
- The normalisation of waxal-benchmarking Whisper-small outputs was not verified.

## Q3. Whisper-based members: output style, cost on 8 GB fp16, hallucination risk on 20–35 s clips

### Takeaway
Whisper fine-tunes give the strongest family diversity against w2v-BERT CTC and natively output cased+punct BPE text. Inference fits 8 GB:
- large-v3 / Sunbird ≈ 1.55B params, ~3.1 GB fp16 weights.
- turbo ≈ 809M, ~1.6 GB.

They carry three specific risks:
- The 30 s encoder window truncates longer clips.
- The language token is learned during fine-tuning, so the wrong token silently degrades output.
- Repetition controls corrupt Bantu reduplication.

Fine-tunes that are clean per their cards: mrwalker123 (QLoRA large-v3), Abdoul27 AF51 (base contamination caveat), waxal-benchmarking whisper-small, and Sunbird zero-shot.

### Cited Findings
- 4th-place solution findings — [yehoshua0 README](https://github.com/yehoshua0/waxal-asr-phase2):
  - It needed a "tail merge for clips longer than whisper's 30 s encoder window", re-decoding 13 clips past 30 s with Sunbird-51.
  - "`no_repeat_ngram_size=3` would corrupt 29.2% of lin transcripts — reduplication is grammatical in Bantu."
  - "Silero VAD: −0.0035."
  - Beam 3 gained +0.003369 on the turbo FT and lost 0.000374 on the Sunbird per-speaker model.
  - fp16 decoding is not reproducible run to run.
- The turbo FT "trained under the placeholder `<|sw|>` and must be decoded under it"; its usage code slices audio to `[:30 * 16_000]` and uses `max_new_tokens=220` — [yehoshua01 turbo r1](https://huggingface.co/yehoshua01/waxal-whisper-turbo-lin-r1)
- AF51 / Sunbird language tokens: `ln` = 50353, `sn` = 50324; force them through `forced_decoder_ids` — [Abdoul27 af51](https://huggingface.co/Abdoul27/waxal-af51-linsna-ck1200)
- mrwalker123 loads `openai/whisper-large-v3` in fp16, merges the LoRA, and picks the higher-`sequences_scores` hypothesis of the two adapters (SWA vs best-4000) per utterance — [mrwalker123 card](https://huggingface.co/mrwalker123/whisper-large-v3-waxal-lin-sna). yehoshua's pipeline reads "mrwalker `preds_swa.json`" as a voter — [REPRODUCTION.md](https://github.com/yehoshua0/waxal-asr-phase2/blob/main/REPRODUCTION.md)
- Solo LB as the Lingala half: turbo FT r2 beam 3 = 0.752910; Sunbird-51 zero-shot = 0.747908; QLoRA large-v3 = 0.742006. On Shona, Sunbird + PL was the best half; MMS-1B FT on sna cost −0.017824 — [yehoshua0 README](https://github.com/yehoshua0/waxal-asr-phase2)
- anyantudre's turbo arm scored 0.7331 solo on the public LB, below the w2v-BERT arm (0.742142), and was kept as a weak ensemble component — [anyantudre whisper-turbo card](https://huggingface.co/anyantudre/waxal-whisper-turbo-linsna)
- Weight sizes from the Hub trees: turbo FTs 3.24 GB (fp32), Sunbird-based FTs 6.17 GB (fp32) or 3.09 GB (fp16), mrwalker adapters 1.01 GB, waxal-benchmarking whisper-small ~0.97 GB — [HF API trees](https://huggingface.co/api/models/yehoshua01/waxal-whisper-turbo-lin-r1/tree/main)

### Inferences
- VRAM (estimate, UNVERIFIED): large-v3-class fp16 at batch 1, beam ≤3 needs roughly 4–5 GB; turbo roughly 2.5–3 GB; whisper-small < 1.5 GB. All fit an 8 GB RTX 4070 Laptop.
- For 20–35 s clips, chunk anything over 30 s with overlap, or decode the tail separately with a long-form strategy. Keep `no_repeat_ngram_size` off. Use mild beam (≤3), a length cap, and compression-ratio / repetition checks to catch hallucination loops instead of n-gram bans.

### Gaps
- No measured hallucination rate for WAXAL lin/sna Whisper fine-tunes was found.
- The only clean, documented, large Whisper member is mrwalker123. Its score (WER 0.3234) uses an unstated eval split.

## Q4. omniASR (Meta Omnilingual, fairseq2): native Windows or WSL2/Linux only? CTC-300M/1B VRAM

### Takeaway
fairseq2 has **no native Windows support and no plans for it**; Meta's own guidance is WSL2 with CUDA. The omnilingual-asr README is looser ("reference inference pipeline works across platforms; Windows may need additional setup"), but the dependency points to WSL2/Linux.

Official inference VRAM: CTC-300M ~2 GiB, CTC-1B ~3 GiB, CTC-3B ~8 GiB, LLM-300M ~5 GiB, LLM-1B ~6 GiB. Audio must be shorter than 40 s.

A Windows-native escape hatch exists: the HF-transformers port of CTC-300M-v2 used by ayymen (`aadel4/omniASR-CTC-300M-v2` base, with `vocab.json` and `preprocessor_config.json`).

### Cited Findings
- "fairseq2 does not have native support for Windows and there are no plans to [add it] … use Windows Subsystem for Linux (a.k.a. WSL) along with full CUDA support introduced in WSL 2" — [fairseq2 README](https://github.com/facebookresearch/fairseq2#installing-on-windows)
- "While we provide a reference inference pipeline that works across platforms, audio support requires libsndfile (… Windows may need an additional setup)"; "Currently only audio files shorter than 40 seconds are accepted for inference on CTC and LLM model suites"; code and models under Apache 2.0 — [omnilingual-asr README](https://github.com/facebookresearch/omnilingual-asr)
- Official VRAM / real-time factor: CTC_300M(_v2) ~2 GiB, 0.001 RTF; CTC_1B(_v2) ~3 GiB, 0.002; CTC_3B ~8 GiB; CTC_7B ~15 GiB; LLM_300M ~5 GiB, 0.090; LLM_1B ~6 GiB; LLM_3B ~10 GiB; LLM_7B ~17 GiB — [omnilingual-asr README](https://github.com/facebookresearch/omnilingual-asr)
- Loading a community fine-tune: yehoshua01's shard is loaded over `omniASR_CTC_1B_v2` with `strict=False` through `fairseq2.models.hub.load_model` and `ASRInferencePipeline` — [yehoshua01 omni-ctc1b-lin](https://huggingface.co/yehoshua01/waxal-omni-ctc1b-lin). Darius ships `model.pt` plus `card.yaml` and decodes with batch size 1 in a pinned `.venvs/omni` env through `install.sh` (Linux shell) — [Darius ctc-1b card](https://huggingface.co/DariusTheGeek/waxal-lin-omniasr-ctc-1b)
- ayymen's omni fine-tune ships HF-style files (`model.safetensors`, `vocab.json`, `preprocessor_config.json`, `tokenizer_config.json`) and names `aadel4/omniASR-CTC-300M-v2` as its base — [ayymen card](https://huggingface.co/ayymen/omniASR-CTC-300M-v2-waxal_lin_sna-1epoch); [HF tree](https://huggingface.co/api/models/ayymen/omniASR-CTC-300M-v2-waxal_lin_sna-1epoch/tree/main)

### Inferences
- On the Windows 11 laptop, use WSL2 + CUDA for the Darius, Paulownia, waxal-benchmarking and yehoshua omni checkpoints. The ayymen port can likely run in native-Windows transformers (the exact model class is UNVERIFIED; check `config.json`).
- Of the omni candidates, CTC-300M, CTC-1B and LLM-300M fit comfortably on 8 GB. LLM-1B (~6 GiB) is tight. CTC-3B (~8 GiB) and above do not fit reliably.

### Gaps
- No measured VRAM on an 8 GB laptop GPU for any omni checkpoint; the figures above are Meta's A100 numbers.
- Whether `aadel4/omniASR-CTC-300M-v2` loads as `Wav2Vec2ForCTC` was not checked.

## Q5. Licenses: which are non-commercial or non-OSI and must be flagged under an "open source" constraint?

### Takeaway
Flag or exclude everything derived from MMS: waxal-benchmarking mms-300m-*, Darius mms-1b-*, yehoshua mms-1b-*, and Abdoul27 mms1b-lin (no tag, but the base is CC-BY-NC-4.0). Also flag yehoshua's ssa-hubert and omni-ctc1b, which are tagged CC-BY-NC-4.0. Gemma-3n fine-tunes (Okwija) fall under Gemma terms.

Open-licensed options: Apache-2.0 (keystats, Darius omni, Paulownia, Sunbird, Mubarak127, anyantudre), MIT (ayymen w2v-BERT, mrwalker123, Abdoul27 AF51, GodwillN), CC-BY-4.0 (waxal-benchmarking omni/whisper). Several repos have **no license** (ayymen omni, Okwija, emmilly, emmilina, sulaimank r5) and should be treated as all-rights-reserved until clarified.

### Cited Findings
- Darius MMS models: "Non-commercial. This licence is inherited from `facebook/mms-1b-all` … The OmniASR models in this solution are `apache-2.0`" — [Darius mms card](https://huggingface.co/DariusTheGeek/waxal-lin-mms-1b)
- yehoshua lists `facebook/omnilingual-asr` (CTC-1B, CTC-7B) as CC-BY-NC-4.0 and WaxalNLP as CC-BY-SA-4.0. It concludes that its lin checkpoint is "CC-BY-NC-4.0 ∧ CC-BY-SA-4.0", "not commercialisable" — [yehoshua0 README](https://github.com/yehoshua0/waxal-asr-phase2). **Conflicts:** Meta states omnilingual code and models are Apache 2.0 — [omnilingual-asr README](https://github.com/facebookresearch/omnilingual-asr) — and anyantudre lists WaxalNLP as CC-BY-4.0 — [anyantudre card](https://huggingface.co/anyantudre/waxal-w2vbert-linsnalug-raw).
- waxal-benchmarking: omni and whisper repos CC-BY-4.0, MMS repos CC-BY-NC-4.0 (HF metadata) — [HF search](https://huggingface.co/models?search=waxal-benchmarking)
- Sunbird 51 is now `Sunbird/SunflowerASR-51-african-languages`: Apache-2.0, gated "auto", base `openai/whisper-large-v3` + `huwenjie333/whisper-v3-ft-af51-0903`, last modified 2026-10-08 — [HF API](https://huggingface.co/api/models/Sunbird/SunflowerASR-51-african-languages)
- `sandilecwili007`, `ronnybscales`, `Mubarak127` and `rwamux` are tagged Apache-2.0; `mrwalker123` and `Abdoul27/waxal-af51-linsna-ck1200` are MIT (HF metadata) — [HF search](https://huggingface.co/models?search=waxal)

### Inferences
- If "open source" means OSI-compatible, avoid MMS and Seamless derivatives and repos with no license. Treat the WaxalNLP data license (CC-BY vs CC-BY-SA, a conflict between participants) as a separate check on the dataset card.

### Gaps
- I did not open the google/WaxalNLP dataset card in this session to settle CC-BY-4.0 vs CC-BY-SA-4.0.

## Q6. Contamination: which checkpoints were trained on Phase 1 test labels or on Phase 2 audio (pseudo-labels) and must be excluded?

### Takeaway
Exclude:
- `anyantudre/waxal-whisper-turbo-linsna` (Phase 1 test gold).
- `anyantudre/waxal-w2vbert-linsna-distilled` (test gold + the 892 Phase 2 clips).
- `anyantudre/waxal-w2vbert-linsna-soup6` (contains both).
- Every `yehoshua01/*` checkpoint. All cards carry "pseudo-labels were computed on the phase-2 test audio … partly adapted to that specific set". `-full` and `-fullmeta` are described as no-PL ablations but keep the same clause.

Treat as risky:
- `ravitejac/waxal-asr-solution` ("TTA specialist").
- sulaimank `*-pseudo`, `-p2*` and douyeszn `*-pseudo` (pseudo-label source unknown).
- Sunbird-51 and its fine-tunes (trained on WAXAL; which splits is unknown).

Clean per their cards: keystats, waxal-benchmarking, DariusTheGeek, mrwalker123, Abdoul27, anyantudre raw / afrivoicemix / specialists.

### Cited Findings
- anyantudre whisper-turbo trained on "google/WaxalNLP, Lingala and Shona test split, gold | 3,615 (1,866 lin, 1,749 sna) | 19.2 h". The card says use of the Phase 1 test split for training "is permitted in Phase 2 and the organisers confirmed this publicly" — [card](https://huggingface.co/anyantudre/waxal-whisper-turbo-linsna)
- distilled: "**This model is trained on the 892 Phase 2 test clips**", using ensemble transcripts, plus test-split gold. The authors fingerprinted the clips and found no public reference transcripts — [card](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-distilled)
- soup6 includes the WAXAL test split, pseudo-labels over the WAXAL *unlabeled* lin/sna pools (22,986 / 22,396 rows, top 60% kept) and the distilled member — [card](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-soup6)
- yehoshua primaries: "pseudo-labelling on the test audio, then per-speaker adaptation". Every yehoshua01 card ends with "Pseudo-labels were computed on the phase-2 test audio (transductive self-training, permitted for phase-2 training by the host), so the checkpoint is partly adapted to that specific set" — [yehoshua01 pl2-spk](https://huggingface.co/yehoshua01/waxal-mms-1b-lin-pl2-spk); [omni-ctc1b-lin](https://huggingface.co/yehoshua01/waxal-omni-ctc1b-lin); [turbo r1](https://huggingface.co/yehoshua01/waxal-whisper-turbo-lin-r1). `-full` is titled "(no pseudo-labels)" and described as "the primary without the pseudo-label and per-speaker stages" — [card](https://huggingface.co/yehoshua01/waxal-mms-1b-lin-full)
- Darius: training uses "`google/WaxalNLP` gold transcripts | 32,328 rows"; the "unlabelled release | 5,531 clips" serves only as "pool audio for the TTIA gallery" — [Darius README](https://github.com/DariusTheGeek/waxal-asr-solution)
- keystats: "`train` splits from all three language configs pooled … `validation` splits … used to [monitor]"; "Only WAXAL data was used" — [keystats card](https://huggingface.co/keystats/w2v-bert-2.0-multilingual-waxal-shona-lingala-luganda-best)
- mrwalker123: "WAXAL corpus (Lingala + Shona train/validation splits)" — [card](https://huggingface.co/mrwalker123/whisper-large-v3-waxal-lin-sna)
- waxal-benchmarking: trained on train splits and scored on the held-out test split (≥1.5 s) — [bantu card](https://huggingface.co/waxal-benchmarking/omniasr-llm-300m-waxal-bantu)
- Sunbird-51 training data includes Google WAXAL lin/lug/sna and FLEURS — [salt.sunbird.ai](https://salt.sunbird.ai/models/asr-whisper-51-african-languages/) (from prior notes). Abdoul27 found that AF51's validation "declines from step 0" (zero-shot 0.8854) and asks "whether WAXAL-derived validation is unreliable for a base model that may already have seen the corpus" — [Abdoul27 af51](https://huggingface.co/Abdoul27/waxal-af51-linsna-ck1200)
- ravitejac: `checkpoints/ft-7b-tta-spec/` "7B decoder fine-tune (TTA specialist)". It states it contains "no test-set reference transcriptions" — [card](https://huggingface.co/ravitejac/waxal-asr-solution)

### Inferences
- If WAXAL test (Phase 1) is used as the calibration or dev set for ensemble weights, do not count members trained on it: anyantudre whisper-turbo, distilled and soup6. Sunbird-51 is also suspect, and its calibration numbers will be inflated, as the AF51 zero-shot 0.885 suggests.
- Phase 2 pseudo-label members (yehoshua01 \*, anyantudre distilled) would bias any Phase 2-based calibration and violate a no-test-audio rule. Exclude them.
- Members trained on the WAXAL *unlabeled* pool (anyantudre `_p2`, sulaimank pseudo, douyeszn pseudo) are acceptable only if that pool is disjoint from Phase 2 test audio. That is unverified; anyantudre's fingerprinting only checked labelled public data.

### Gaps
- ayymen, Paulownia, GodwillN, Okwija, emmilly and emmilina do not document their training splits.
- Whether the WaxalNLP unlabeled pool overlaps the Phase 2 892 clips was not established.

## Q7. Diversity value: which members are most likely to help ROVER/MBR over anyantudre near-duplicates?

### Takeaway
Two top solutions found that **family distinctness beats solo quality** for voters, and that agreement among models trained on the same targets carries little independent evidence. The most diverse clean additions, in order:
1. An omniASR CTC-1B / LLM-1B (Darius or Paulownia; WSL2), or the ayymen omni-300M HF port.
2. A Whisper-large-v3 member (mrwalker123 QLoRA, Sunbird zero-shot or AF51, with contamination caveats).
3. keystats w2v-BERT: same architecture, but an independent vocab and seed with no augmentation.
4. MMS-1B lin adapter models (Abdoul27; but CC-BY-NC).

anyantudre siblings (afrivoicemix, specialists) are near-duplicates.

### Cited Findings
- omni-ctc1b-sna was "the worst solo model of its day and still the only voter that moved the leaderboard that night (+0.000253) — voter value tracks family distinctness, not solo quality" — [yehoshua01 omni-ctc1b-sna](https://huggingface.co/yehoshua01/waxal-omni-ctc1b-sna)
- Negative results — [yehoshua0 README](https://github.com/yehoshua0/waxal-asr-phase2):
  - "the agreement of N models from one corpus has an effective sample size of 1"
  - "Cross-family fusion on this test: −0.0085. When one parent dominates by 0.0186 … Fusion pays only when the parents are near parity"
  - "Word-level voters are saturated: four doses, one positive (+0.000216)"
- Darius chose OmniASR LLM for "decoder-side language modelling — a different error profile from the CTC models, which is what the fusion trades on", and MMS as "architecturally distant". The lin lane fuses four omni members by conservative word ROVER, which "needs at least three sources" — [Darius README](https://github.com/DariusTheGeek/waxal-asr-solution)
- anyantudre: the afrivoicemix member changed the 26-member ensemble on only 50 of 892 rows (≈+0.0013). The 26-member char-level ROVER core scored 0.764915 against a best single model of about 0.746. "Fresh random seeds on the strongest architecture were worth more" than other architectures as solo models — [afrivoicemix card](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-afrivoicemix); [whisper-turbo card](https://huggingface.co/anyantudre/waxal-whisper-turbo-linsna); [distilled card](https://huggingface.co/anyantudre/waxal-w2vbert-linsna-distilled)

### Inferences
- With a strong anyantudre pivot (~0.742 solo), weak cross-family members should act as gated or conservative voters, changing a word only on consensus plus an LM margin. Plain majority votes are risky when the external member is far weaker, as yehoshua's −0.0085 cross-family result shows.
- Practical shortlist for 8 GB, clean, OSI-friendly:
  1. `keystats/...-best` (Apache, w2v-BERT, cased+punct).
  2. `ayymen/omniASR-CTC-300M-v2-waxal_lin_sna-1epoch` (cased+punct, omni root, HF format; license tag missing).
  3. `DariusTheGeek/waxal-lin-omniasr-ctc-1b` + `waxal-joint-ctc-1b-lid` (Apache, WSL2).
  4. `Paulownia/waxal-omniasr-ctc-1b-multilingual` (Apache, WSL2).
  5. `mrwalker123/whisper-large-v3-waxal-lin-sna` (MIT).
  6. `waxal-benchmarking/omniasr-llm-300m-waxal-all19` (CC-BY-4.0; lowercased, so needs re-casing).
  7. `ayymen/w2v-bert-2.0-waxal_lin_sna-1epoch` (MIT).

### Gaps
- No public measurement of the marginal ROVER gain of any of these specific external members on top of the anyantudre ensemble exists. It has to be measured locally on a clean dev set.

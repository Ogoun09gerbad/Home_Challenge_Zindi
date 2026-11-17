# Pretrained open-source ASR models for Lingala (lin), Shona (sna), Luganda (lug): candidates, reported WER/CER, hardware fit

Research date: 2026-10-26. All numbers are as reported by the cited source. Metrics differ between sources (normalisation, test filtering, LM use), so numbers from different rows are not strictly comparable. "UNVERIFIED" marks claims I could not confirm from a primary source.

## Q0. Master comparison table (zero-shot and fine-tuned candidates)

### Takeaway
On WAXAL's own conversational test set, the best published numbers come from small models fine-tuned on WAXAL: Omnilingual-LLM-300M fine-tuned (lin 29.9 / sna 21.6 / lug 12.1 WER) and MMS-300M CTC fine-tuned (42.6 / 25.0 / 16.9 WER). Zero-shot models (Omnilingual 1B, MMS-1B-all, Whisper large-v3) are much worse on WAXAL, even when they do well on FLEURS. The Zindi WAXAL challenge's top published solution used w2v-BERT 2.0 + CTC.

### Cited Findings

**Table A – Base / zero-shot models**

| Model (HF ID) | Params | Arch | License | lin / sna / lug covered? | Reported results (WER / CER %, benchmark) | Inference VRAM |
|---|---|---|---|---|---|---|
| `facebook/omniASR-CTC-300M` / `-1B` / `-3B` / `-7B` (+ `_v2` checkpoints, Dec 2025) | 325M / 975M / 3.08B / 6.5B | CTC (wav2vec2 encoder + linear head) | Apache 2.0 | yes (`lin_Latn`, `sna_Latn`, `lug_Latn`) | WAXAL-NET, "Omnilingual 1B" zero-shot (paper doesn't say CTC or LLM): WAXAL test lin 45.3/25.1, sna 35.6/8.5, lug 42.9/8.5; FLEURS lin 18.4/5.6, sna 24.0/5.1, lug 50.1/9.2 | ~2 / ~3 / ~8 / ~15 GiB |
| `facebook/omniASR-LLM-300M` / `-1B` / `-3B` / `-7B` (+ v2, "Unlimited" v2, `-7B-ZS`) | 1.63B / 2.28B / 4.38B / 7.8B | encoder + LLM-style autoregressive decoder | Apache 2.0 | yes | 7B LLM-ASR per-language CER (official CSV; the eval set isn't named in the CSV): **lin 3.9, lug 4.2, sna 1.9** (training hours 168.2 / 178.2 / 220.2) | ~5 / ~6 / ~10 / ~17 GiB (7B-ZS ~20 GiB) |
| `facebook/mms-1b-all` | ~1B | CTC + per-language adapters | CC-BY-NC-4.0 | yes: `adapter.lin`, `adapter.sna`, `adapter.lug` present | WAXAL-NET zero-shot WAXAL: lin 44.7/24.8, sna 36.9/8.9, lug 32.1/6.7; FLEURS lin 101.2/76.9, sna 100.3/85.7, lug 100.5/85.6 (suspect, see Q1) | ~4-6 GB fp32, less in fp16 (UNVERIFIED estimate) |
| `facebook/mms-1b-fl102` | ~1B | CTC + adapters (fine-tuned on FLEURS-102) | CC-BY-NC-4.0 | yes (adapters lin/sna/lug present) | MMS paper Table A3 (MMS fine-tuned on FLEURS with adapters + n-gram LM), FLEURS CER dev/test: **lin 4.0/4.3, sna 3.8/4.1, lug (Ganda) 8.0/8.4** | as above |
| `facebook/mms-300m` | 300M | SSL encoder only (needs CTC fine-tuning) | CC-BY-NC-4.0 | pretraining on 1,400 languages | no zero-shot ASR; see fine-tunes in Table B | ~1-2 GB |
| `facebook/w2v-bert-2.0` | ~600M | SSL Conformer encoder (Seamless v2), needs CTC head | MIT (UNVERIFIED, not checked this session) | pretrained on 4.5M h, 143+ languages (UNVERIFIED) | no zero-shot ASR; see Q2 | ~2-3 GB |
| `facebook/seamless-m4t-v2-large` | 2.3B | encoder-decoder (multitask) | CC-BY-NC-4.0 | speech input: **lug yes, sna yes, lin NO** | no per-language WER found | >8 GB fp32 (UNVERIFIED) |
| `facebook/wav2vec2-xls-r-300m` / `-1b` | 300M / 1B | SSL encoder, needs CTC | Apache 2.0 (UNVERIFIED) | 128 pretraining languages | see Q2/Q3 (fine-tuned results) | ~1-2 / ~4 GB |
| `ajesujoba/AfriHuBERT` | ~95M (mHuBERT-147 base size, UNVERIFIED) | SSL HuBERT encoder | not checked | continued pretraining on African speech | Interspeech 2025: -2.1% avg WER vs mHuBERT-147 on FLEURS | <2 GB |
| `openai/whisper-large-v3` / `-large-v3-turbo` | 1.55B / 809M | encoder-decoder | MIT | native tokens for **sn (Shona) and ln (Lingala)**; **no Luganda token** | Whisper paper (large-v2, FLEURS WER): lin 75.6, sna 121.0. WAXAL-NET large-v3 zero-shot: WAXAL lin 93.0/64.0, sna 112.4/29.9; FLEURS lin 77.0/20.9, sna 120.3/29.3; lug N/A | ~4-6 GB fp16 (UNVERIFIED) |
| `nvidia/canary-1b-v2`, `nvidia/parakeet-tdt-0.6b-v3` | 1B / 0.6B | AED / TDT | CC-BY-4.0 | **No**: only 25 European languages | – | – |
| `espnet/owsm_v4_*`, `espnet/owsm_ctc_v4_1B` | 102M–1B | AED / CTC | not checked | coverage of lin/sna/lug not verified | none found | – |
| Google USM | – | – | **not public** | – | – | – |

**Table B – Checkpoints fine-tuned on WAXAL or on these languages (HF)**

| Model (HF ID) | Base | Langs | License | Reported WER / CER (%) | Benchmark / notes |
|---|---|---|---|---|---|
| `waxal-benchmarking/omniasr-llm-300m-waxal-all19` | omniASR-LLM-300M | 19 WAXAL languages | CC-BY-4.0 | **lin 29.9/12.5, sna 21.6/3.9, lug 12.1/2.8** (monolingual baselines WER: lin 30.4, sna 21.6, lug 10.9) | WAXAL test (≥1.5 s), NFC + lowercased; trained on 2× H200 |
| `waxal-benchmarking/omniasr-llm-300m-waxal-lin` (and per-language siblings) | omniASR-LLM-300M | lin | – | lin 30.4 WER (per the all19 card) | WAXAL test |
| `waxal-benchmarking/mms-300m-waxal-lin` / `-sna` / `-lug` | facebook/mms-300m | 1 each | CC-BY-NC-4.0 | lin 42.6/18.9, sna 25.0/4.3, lug 16.9/3.4 | filtered WAXAL test (≥1.5 s, ≥4 WPS); FLEURS: lin 38.3/12.8, sna 39.3/10.9, lug 57.8/15.1 |
| `waxal-benchmarking/whisper-small-waxal-{lin,sna,lug}` | whisper-small (244M) | 1 each | – | lin 42.7/21.7, sna 26.9/5.1, lug 21.6/5.5 | WAXAL test |
| `waxal-benchmarking/whisper-tiny-waxal-*` | whisper-tiny (39M) | 1 each | – | lin 49.0/30.9, sna 31.4/8.9, lug 33.8/12.8 | WAXAL test |
| `anyantudre/waxal-w2vbert-linsnalug-raw` (+ `waxal-w2vbert-lin-specialist`, `-sna-specialist`, seeds, soups) | facebook/w2v-bert-2.0 | lin+sna+lug | Apache 2.0 | Zindi public LB single model 0.742142 (CER 0.1160, WER 0.3998); 26-member ROVER ensemble 0.764915; final 0.766791 public / 0.772552 private | Zindi Google WAXAL ASR Challenge; score = 1 − 0.5·(WER+CER); trained on 1× RTX 5090 32 GB |
| `keystats/w2v-bert-2.0-multilingual-waxal-shona-lingala-luganda(-best)` | w2v-bert-2.0 | sna+lin+lug | Apache 2.0 | no numbers retrieved | WAXAL only; LR 3e-5, eff. batch 32 (4×8), fp16 + grad checkpointing |
| `ayymen/omniASR-CTC-300M-v2-waxal_lin_sna-{1,2}epoch` | omniASR-CTC-300M-v2 | lin+sna | – | no card numbers | – |
| `sulaimank/xlsr-luganda-waxal` | sulaimank/wav2vec2-xlsr-swahili-400hr | lug | Apache 2.0 | WER 21.44, CER 4.46 (eval split not stated) | WAXAL |
| `Sunbird/asr-whisper-51-african-languages` | whisper-large-v3 (1.54B) | 51 incl. lin, sna, lug | Apache 2.0 | lin 24.3/10.4 (n=193), sna 19.6/5.0 (n=147), lug 10.9/2.2 (n=103, SALT multispeaker dev) | **trained on WAXAL lin/lug/sna + FLEURS**, so WAXAL/FLEURS numbers are contaminated |
| `jq/whisper-large-v3-salt-plus-xog-myx-kin-swa` (Sunbird "whisper-large-v3-salt") | whisper-large-v3 | Ugandan langs incl. lug | – | no lug figure found | – |
| `Sunbird/asr-mms-salt` | mms-1b-all | 6 Ugandan langs incl. lug | CC-BY-NC-4.0 (third-party listing) | no figure found | – |
| `badrex/w2v-bert-2.0-shona-asr` | w2v-bert-2.0 | sna | CC-BY-4.0 | "in-domain WER below 23%" | 72 h Shona |
| Other HF hits (no metrics checked) | | | | | `Okwija/mms-adapter-waxal-{lin,sna,lug}`, `Okwija/mms-asr-waxal-pooled`, `DariusTheGeek/waxal-{lin,sna}-mms-1b`, `yehoshua01/waxal-whisper-turbo-lin-r2`, `yehoshua01/waxal-mms-1b-lin-*`, `Mubarak127/waxal-whisper-large-v3-sna_asr`, `ElizabethMwangi/whisper-large-v3-luganda-waxal`, `sulaimank/w2vbert-lingala-waxal`, `sulaimank/mms-lingala-waxal`, `dhasmana/WAXAL-*-w2v-bert-2.0`, `noirlab/whisper-large-v3-{lingala,shona}-asr`, `CraneAILabs/crane-nemo-asr-1.1/1.2`, `asr-africa/W2V2-Bert_*_Shona_*hr` |

Sources:
- Omnilingual checkpoint sizes, VRAM, license, CSV link — [GitHub facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr); per-language CER rows `lin_Latn,168.2,3.9`, `lug_Latn,178.2,4.2`, `sna_Latn,220.2,1.9` — [per_language_results_table_7B_llm_asr.csv](https://github.com/facebookresearch/omnilingual-asr/blob/main/per_language_results_table_7B_llm_asr.csv); HF repos `facebook/omniASR-{CTC,LLM,W2V}-*` exist — [HF omniASR-LLM-7B](https://huggingface.co/facebook/omniASR-LLM-7B)
- WAXAL-NET zero-shot and fine-tuned tables, dataset hours — [WAXAL-NET, arXiv 2606.02375](https://arxiv.org/html/2606.02375v1)
- MMS adapters for lin/sna/lug and CC-BY-NC-4.0 license — HF API file listing of [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) and [facebook/mms-1b-fl102](https://huggingface.co/facebook/mms-1b-fl102)
- MMS FLEURS-102 CER (Table A3) and Shona Whisper comparison (Table A1) — [MMS paper, arXiv 2305.13516](https://arxiv.org/pdf/2305.13516)
- Whisper FLEURS WER (Appendix D.2.4) — [Whisper paper, arXiv 2212.04356](https://arxiv.org/pdf/2212.04356)
- Seamless v2 language table (lug and sna listed for speech input, lin not listed; CC-BY-NC-4.0) — [facebook/seamless-m4t-v2-large card](https://huggingface.co/facebook/seamless-m4t-v2-large)
- Canary/Parakeet languages and licenses — [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2), [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- WAXAL fine-tunes — [omniasr-llm-300m-waxal-all19](https://huggingface.co/waxal-benchmarking/omniasr-llm-300m-waxal-all19), [mms-300m-waxal-lin](https://huggingface.co/waxal-benchmarking/mms-300m-waxal-lin), [mms-300m-waxal-sna](https://huggingface.co/waxal-benchmarking/mms-300m-waxal-sna), [mms-300m-waxal-lug](https://huggingface.co/waxal-benchmarking/mms-300m-waxal-lug), [whisper-small-waxal-lug](https://huggingface.co/waxal-benchmarking/whisper-small-waxal-lug), [anyantudre/waxal-w2vbert-linsnalug-raw](https://huggingface.co/anyantudre/waxal-w2vbert-linsnalug-raw), [keystats multilingual](https://huggingface.co/keystats/w2v-bert-2.0-multilingual-waxal-shona-lingala-luganda-best), [sulaimank/xlsr-luganda-waxal](https://huggingface.co/sulaimank/xlsr-luganda-waxal), [badrex/w2v-bert-2.0-shona-asr](https://huggingface.co/badrex/w2v-bert-2.0-shona-asr)
- Sunbird 51-language Whisper: per-language table, training sets, license — [salt.sunbird.ai model page](https://salt.sunbird.ai/models/asr-whisper-51-african-languages/), [HF Sunbird/asr-whisper-51-african-languages](https://huggingface.co/Sunbird/asr-whisper-51-african-languages)
- HF Hub search "waxal" (~200 repos, including the IDs above) — [HF models search](https://huggingface.co/models?search=waxal)

### Inferences
- For the WAXAL domain (spontaneous image-description speech, average 20.2 s clips), fine-tuning on WAXAL matters more than model size. A fine-tuned 300M model beats every zero-shot model of 1–7B.
- Lingala is the hardest of the three on WAXAL (best ~30% WER) even though it is easy on FLEURS (Omnilingual zero-shot 18.4% WER). The reason may be domain mismatch or transcription conventions, not acoustics.

### Gaps
- No per-language FLEURS numbers for Whisper large-v3 / large-v3-turbo were found in a primary source; the Whisper paper only gives large-v2.
- No published numbers were found for OWSM v4 on these languages, and its language coverage for lin/sna/lug was not verified.

## Q1. Meta MMS (mms-1b-all, mms-1b-fl102, mms-300m): adapters and FLEURS CER/WER

### Takeaway
mms-1b-all and mms-1b-fl102 both ship adapters for lin, sna and lug. On FLEURS, MMS fine-tuned with adapters and an n-gram LM gets very low CER (lin 4.3, sna 4.1, lug 8.4 test). On WAXAL, zero-shot mms-1b-all is weak (32–45% WER), while mms-300m fine-tuned on WAXAL is a strong, cheap baseline. All MMS weights are CC-BY-NC-4.0 (non-commercial).

### Cited Findings
- `adapter.lin.safetensors`, `adapter.lug.safetensors` and `adapter.sna.safetensors` are present in both facebook/mms-1b-all and facebook/mms-1b-fl102. Both cards declare license cc-by-nc-4.0 — HF API listing of [mms-1b-all](https://huggingface.co/facebook/mms-1b-all) / [mms-1b-fl102](https://huggingface.co/facebook/mms-1b-fl102).
- MMS paper Table A3 (FLEURS-102, MMS fine-tuned on FLEURS labeled data with language-specific adapters and heads, decoded with the best of a Common Crawl word LM or a FLEURS char LM). CER dev/test: Lingala 4.0/4.3 (FL LM), Shona 3.8/4.1 (FL LM), Ganda 8.0/8.4 (FL LM). Average over 102 languages: 6.3/6.3 — [MMS paper](https://arxiv.org/pdf/2305.13516)
- MMS paper Table A1 (FLEURS test WER, Shona). Whisper medium 143.9, Whisper large-v2 121.0. MMS L-61: 38.1 (no LM) / 30.4 (CC LM); L-61 with LSAH: 37.7 / 30.1. MMS L-1107 (the 1,107-language model): 43.0 / 29.9; L-1107 with LSAH: 37.8 / 29.6. Lingala and Luganda are not in this 54-language Whisper comparison — [MMS paper](https://arxiv.org/pdf/2305.13516)
- WAXAL-NET zero-shot MMS-1B on the WAXAL test: lin 44.7/24.8, sna 36.9/8.9, lug 32.1/6.7 WER/CER. On FLEURS (300 utterances): lin 101.2/76.9, sna 100.3/85.7, lug 100.5/85.6 — [WAXAL-NET](https://arxiv.org/html/2606.02375v1)
- MMS-300M fine-tuned per language on WAXAL (encoder partly frozen, CTC greedy decoding, lr 1e-4, effective batch 32, up to 30 epochs, A100-80GB). WAXAL test: lin 42.6/18.9, sna 25.0/4.3, lug 16.9/3.4. MMS-300M had lower CER than Whisper-Small in 17 of 19 languages — [WAXAL-NET](https://arxiv.org/html/2606.02375v1); [mms-300m-waxal-lug card](https://huggingface.co/waxal-benchmarking/mms-300m-waxal-lug)
- Data-scaling study (Luganda: FLEURS+CV; Shona: AfriVoice; Lingala: FLEURS+AfriVoice+LRSC), mms-1b-all fine-tuned, WER/CER. Luganda 1h 58.99/15.86 → 200h 42.69/8.56. Shona 5h 30.28/5 → 100h 23.70/4. Lingala 1h 51.3/~21 → 100h 23.96/8.32 — [Nahabwe et al., arXiv 2512.10968](https://arxiv.org/html/2512.10968v1)
- A third-party listing describes Sunbird/asr-mms-salt as built on mms-1b-all with adapters, covering six Ugandan languages incl. Luganda, CC-BY-NC-4.0; no WER given — [PromptLayer listing](https://www.promptlayer.com/models/asr-mms-salt) (secondary source)

### Inferences
- The WAXAL-NET zero-shot MMS-1B FLEURS numbers (~100% WER, 76–86% CER) contradict both the MMS paper (FLEURS CER ~4–8%) and WAXAL-NET's own reasonable WAXAL-test numbers for the same model. A loading or decoding error (for example, the adapter not switched or the wrong target language) is likely. Treat those FLEURS cells as unreliable.
- The FLEURS numbers in the MMS paper come from mms-1b-fl102-style training on FLEURS itself with an n-gram LM. They are in-domain for FLEURS and say little about WAXAL performance.
- Because of the CC-BY-NC license, MMS-based submissions may be restricted if the challenge or later use requires commercial-compatible licensing. Check the challenge rules.

### Gaps
- The MMS paper also includes an mms-1b-all ("L-1107") per-language FLEURS CER table for lin and lug. I only extracted Table A1 (Shona, WER) and Table A3 (fl102).
- I found no figure for mms-1b-all + adapter fine-tuning on WAXAL. The `Okwija/mms-adapter-waxal-*` and `DariusTheGeek/waxal-*-mms-1b` cards had no metrics.

## Q2. facebook/w2v-bert-2.0 CTC fine-tuning; SeamlessM4T v2 ASR support

### Takeaway
w2v-BERT 2.0 + CTC is among the most data-efficient recipes for these languages, especially with a 5-gram KenLM. It was the backbone of the top published Zindi WAXAL solution (single model 0.742, ensemble 0.773 private). SeamlessM4T v2 supports Luganda and Shona speech input but not Lingala, and it is CC-BY-NC.

### Cited Findings
- Benchmark of Whisper-small, XLS-R-300M, w2v-BERT-2.0 (600M) and mms-1b-all across data sizes. WER/CER %:
  - **Luganda:** w2v-BERT 1h 55.69/11.63 → 200h 39.75/7.41; w2v-BERT + 5-gram LM 1h 38.16/8.32 → 200h **16.30/3.72**. XLS-R-LM 200h 17.43/4.09; Whisper-small 200h 20.45/5.54; MMS 200h 42.69/8.56.
  - **Shona:** w2v-BERT 100h 22.56/4; w2v-BERT-LM 50h **8.61/2** (100h 9.28/2); MMS 100h 23.70/4; Whisper 50h 26.75/7.
  - **Lingala:** w2v-BERT 100h 24.19/9.59; w2v-BERT-LM 100h 22.74/8.52; XLS-R 100h **18.82/5.69**; MMS 100h 23.96/8.32.
  - The authors conclude that "MMS and W2v-BERT are more data efficient in very low-resource regimes," while XLS-R scales better with more data. LM decoding can hurt when the text domain mismatches. Training ran on A40/L4/RTX A6000/A100 GPUs (24–80 GB).
  - Source: [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1)
- `anyantudre/waxal-w2vbert-linsnalug-raw`, built for the Zindi "Google WAXAL ASR Challenge" — [model card](https://huggingface.co/anyantudre/waxal-w2vbert-linsnalug-raw):
  - Fine-tunes w2v-bert-2.0 (600M) with a CTC head over a raw 97-token vocabulary that keeps capitals and punctuation `! , . : ; ?`.
  - Training: 12 epochs, lr 1e-4, batch 12 × grad-accum 3, bf16 + gradient checkpointing, MUSAN noise + RIR augmentation, 1× RTX 5090 32 GB.
  - Results: public LB 0.742142 (CER 0.115951, WER 0.399766), against 0.724984 for the same model with a stripped 81-token vocabulary. The final ensemble (ROVER + MBR selection) scored 0.766791 public / 0.772552 private.
  - The card's experiments indicate that the competition's WER is computed on lowercased but not punctuation-stripped text, while CER is computed on the raw string.
  - The card states that the "corrected Phase 2 test set contains no Luganda" (roughly half Lingala, half Shona, 892 clips).
- keystats multilingual w2v-BERT (sna+lin+lug, WAXAL only): lr 3e-5, effective batch 32 (4 × 8), fp16, gradient checkpointing, 4 epochs with early stopping; no metric extracted — [card](https://huggingface.co/keystats/w2v-bert-2.0-multilingual-waxal-shona-lingala-luganda-best)
- badrex/w2v-bert-2.0-shona-asr: 72 h of Shona, "in-domain WER is below 23%", ~600M params, CC-BY-4.0 — [card](https://huggingface.co/badrex/w2v-bert-2.0-shona-asr)
- The SeamlessM4T v2 card lists `lug` (Ganda) and `sna` (Shona) with "Sp, Tx" source modality. `lin` is not in the list. License cc-by-nc-4.0 — [facebook/seamless-m4t-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large)

### Inferences
- For the challenge metric, a char-level CTC model that emits case and punctuation (as in the anyantudre recipe) has a structural advantage on CER. Whisper-style models produce cased, punctuated text natively.
- Given the claim that Phase 2 has no Luganda, effort is better spent on lin and sna. This should be cross-checked against the official Zindi page, since it comes from a participant's model card.

### Gaps
- No zero-shot Seamless v2 ASR WER for lug/sna was found.
- The license of facebook/w2v-bert-2.0 was not re-verified in this session (believed MIT).

## Q3. XLS-R 300M/1B, AfriHuBERT and other African-pretrained SSL models

### Takeaway
XLS-R-300M is competitive when 50–100 h are available (best Lingala WER in one benchmark: 18.8%). AfriHuBERT (`ajesujoba/AfriHuBERT`) is a small HuBERT model with modest reported gains over mHuBERT-147, but there are no lin/sna/lug WAXAL numbers for it.

### Cited Findings
- XLS-R-300M fine-tuned: Lingala 100h 18.82/5.69 WER/CER (the best Lingala row in that study); Luganda 200h 40.51/7.78 (17.43/4.09 with LM); Shona 100h 25.83/4 — [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1)
- sulaimank/xlsr-luganda-waxal (continued from an XLS-R Swahili-400h model) reports WER 0.2144 / CER 0.0446; the eval split is not stated. Apache 2.0 — [card](https://huggingface.co/sulaimank/xlsr-luganda-waxal)
- AfriHuBERT is mHuBERT-147 extended by continued pretraining. v1 used 6,500+ h across 39 African languages. The Interspeech 2025 version reports 10K+ h, 1,226 languages, -2.1% average WER on FLEURS ASR vs mHuBERT-147, and claims competitiveness with MMS and XEUS — [arXiv 2409.20201](https://arxiv.org/html/2409.20201v1); [Interspeech 2025](https://www.isca-archive.org/interspeech_2025/alabi25_interspeech.html). HF checkpoints: `ajesujoba/AfriHuBERT` and `ajesujoba/AfriHuBERTo` — [HF search](https://huggingface.co/models?search=afrihubert)
- An Africa-centric SSL pretraining paper reports FLEURS test CER of Lingala 6.9, Shona 8.3 and Ganda 10.7 in its "fine-tuned-all" setting (from a search snippet, not opened) — [arXiv 2404.02000](https://arxiv.org/pdf/2404.02000) (UNVERIFIED detail)
- The CraneAI NeMo models reportedly cut WER for Luganda from 33.8% to 20.9% and for Shona from 37.7% to 27.3% after fine-tuning (search snippet only) — [CraneAILabs/crane-nemo-asr-1.1](https://huggingface.co/CraneAILabs/crane-nemo-asr-1.1) (UNVERIFIED)

### Inferences
- AfriHuBERT is small (base-size), so it is easy to fine-tune on 8 GB, but there is no evidence it beats w2v-BERT 2.0 or Omnilingual on these languages.

### Gaps
- No "Wav2Vec2 Africa" or Orange-released SSL checkpoint with lin/sna/lug results was found.
- I did not check the AfriHuBERT license or parameter count on its card.

## Q4. Whisper large-v3 / large-v3-turbo: language support, zero-shot WER, community fine-tunes

### Takeaway
Whisper has native tokens for Shona (`sn`) and Lingala (`ln`) but none for Luganda. Zero-shot quality is unusable (FLEURS WER lin 75.6, sna 121.0 for large-v2; WAXAL WER 93% / 112% for large-v3). Fine-tuned Whisper models are competitive: Sunbird's 51-language large-v3 reports lug 10.9 / sna 19.6 / lin 24.3 WER on its own dev sets, but it was trained on WAXAL.

### Cited Findings
- Whisper paper, FLEURS WER (tiny → large-v2): Lingala 105.4, 96.7, 91.3, 83.2, 76.8, **75.6**; Shona 128.1, 126.0, 140.0, 143.9, 130.0, **121.0**. Luganda is not evaluated — [Whisper paper App. D.2.4](https://arxiv.org/pdf/2212.04356)
- WAXAL-NET, Whisper large-v3 zero-shot: WAXAL lin 93.0/64.0, sna 112.4/29.9; FLEURS lin 77.0/20.9, sna 120.3/29.3; Luganda N/A (unsupported) — [WAXAL-NET](https://arxiv.org/html/2606.02375v1)
- Sunbird's inference code lists Whisper's native tokens `"sna": 50324` and `"lin": 50353`. Luganda (`"lug": 50332`) is mapped onto an "unused" overwritten token, which confirms Luganda is not native — [Sunbird model page](https://salt.sunbird.ai/models/asr-whisper-51-african-languages/)
- Sunbird/asr-whisper-51-african-languages (1.54B, Apache 2.0, fine-tuned from whisper-large-v3 and huwenjie333/whisper-v3-ft-af51). WER/CER: lin 0.243/0.104 (193 examples), sna 0.196/0.050 (147), lug 0.109/0.022 (103, SALT multispeaker dev). Training data includes Google WAXAL (lin, lug, sna among others) and FLEURS — [Sunbird model page](https://salt.sunbird.ai/models/asr-whisper-51-african-languages/)
- Sunbird whisper-large-v3-salt is a Whisper large-v3 adaptation with custom SALT language tokens; base listed as `jq/whisper-large-v3-salt-plus-xog-myx-kin-swa`; no Luganda WER found — [Sunbird page](https://salt.sunbird.ai/models/asr-whisper-large-v3-salt/)
- Whisper-small fine-tuned on WAXAL: lin 42.7/21.7, sna 26.9/5.1, lug 21.6/5.5 — [WAXAL-NET](https://arxiv.org/html/2606.02375v1)
- Whisper-small fine-tuned in the scaling study: Luganda 200h 20.45/5.54; Shona 50h 26.75/7; Lingala 100h 25.25/10.21 — [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1)
- Other community Whisper fine-tunes on WAXAL exist: `yehoshua01/waxal-whisper-turbo-lin-r2` (large-v3-turbo; its card says it used transductive pseudo-labels on phase-2 test audio, "not a general-purpose ASR model"), `Mubarak127/waxal-whisper-large-v3-sna_asr`, `ElizabethMwangi/whisper-large-v3-luganda-waxal`, `cdli/whisper-large-v3_finetuned_ugandan_luganda_*` — [HF search](https://huggingface.co/models?search=waxal); [yehoshua01 card](https://huggingface.co/yehoshua01/waxal-whisper-turbo-lin-r2)

### Inferences
- The Sunbird 51 model is a strong starting checkpoint (it already saw WAXAL lin/sna/lug), but its reported numbers are not an independent WAXAL benchmark. If you use it, hold out speakers carefully, because WAXAL test speakers may overlap with its training data.
- Whisper large-v3 (1.55B) cannot be fully fine-tuned on 8 GB. LoRA/QLoRA on large-v3-turbo (809M) or full fine-tuning of whisper-small is feasible (UNVERIFIED estimate).

### Gaps
- No OpenAI-published per-language FLEURS WER for large-v3 / large-v3-turbo was retrieved.

## Q5. OWSM v3.1/v4, NVIDIA Canary/Parakeet, Google USM, Meta Omnilingual ASR

### Takeaway
Omnilingual ASR (Nov 2025, Apache 2.0, v2 checkpoints Dec 2025) is the only one of these that covers all three languages with strong numbers. Canary/Parakeet are European-only. USM is not public. OWSM coverage of these languages is unverified.

### Cited Findings
- Omnilingual ASR covers 1,600+ languages, with CTC and LLM-ASR checkpoints at 300M/1B/3B/7B, v2 checkpoints with improved CER (Dec 2025), and an "Unlimited" v2 LLM for long audio. The CTC/LLM suites accept only audio under 40 s. Inference uses the fairseq2-based `omnilingual_asr` package. Fine-tuning recipes exist for CTC and LLM, at `workflows/recipes/wav2vec2/asr/`. License: Apache 2.0 (code and models) — [GitHub](https://github.com/facebookresearch/omnilingual-asr)
- The 7B LLM-ASR has CER < 10 for 78% of languages. Per-language CER: lin 3.9, lug 4.2, sna 1.9 — [GitHub](https://github.com/facebookresearch/omnilingual-asr); [CSV](https://github.com/facebookresearch/omnilingual-asr/blob/main/per_language_results_table_7B_llm_asr.csv)
- The paper's coverage appendix lists lin_Latn (Lingala) and lug_Latn (Ganda) in resource class "H" and sna_Latn (Shona) in class "M". The 7B LLM-ASR has an 80% win rate (65/81 languages) against Whisper large-v3 on FLEURS-81. The paper states USM and its variants "are not publicly accessible" — [Omnilingual ASR paper, arXiv 2511.09690](https://arxiv.org/pdf/2511.09690)
- The paper's low-resource fine-tuning recipe: CTC seeded from the OmniASR CTC checkpoint, lr 1e-5, 5K steps; "CTC fine-tuning takes ~1 hour of walltime on 32 GPUs for the 300M scale" (languages with 5–10 h) — [arXiv 2511.09690](https://arxiv.org/pdf/2511.09690)
- Omnilingual corpus dataset `facebook/omnilingual-asr-corpus` (348 under-served languages, CC-BY-4.0) — [HF dataset](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus)
- canary-1b-v2 and parakeet-tdt-0.6b-v3 card languages: bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk (no African languages). License cc-by-4.0 — [canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2), [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- OWSM checkpoints on HF include `espnet/owsm_v4_base_102M`, `owsm_v4_small_370M`, `owsm_v4_medium_1B`, `owsm_ctc_v4_1B`, `owsm_v3.1_ebf` — [HF espnet search](https://huggingface.co/models?author=espnet&search=owsm)

### Inferences
- Omnilingual CTC-300M/1B (inference VRAM ~2–3 GiB) and LLM-300M (~5 GiB) fit an 8 GB GPU for inference. CTC-300M (325M params) fine-tuning should also fit with bf16 and small batches (UNVERIFIED estimate). The fine-tuned LLM-300M results on WAXAL (lin 29.9, sna 21.6, lug 12.1 WER) are the best independent WAXAL numbers found.
- fairseq2 is the required runtime. Native Windows support is UNVERIFIED (likely Linux-only), so plan on WSL2/Linux for the RTX 4070 laptop.

### Gaps
- OWSM v4 language list and per-language results for lin/sna/lug were not verified.
- It is unclear which Omnilingual variant (CTC or LLM, v1 or v2) WAXAL-NET used as "Omnilingual 1B".
- The evaluation set behind the official per-language CSV is not documented in the CSV itself. The paper evaluates on MMS-Lab, FLEURS, CV22 and Omnilingual test sets.
- No Hugging Face `transformers` integration for Omnilingual was confirmed.

## Q6. Most promising option for an 8 GB laptop GPU (RTX 4070 Laptop) with ~5 days of compute

### Takeaway
Two options stand out. (1) Fine-tune w2v-BERT 2.0 + CTC with a raw cased/punctuated char vocabulary, which is the proven Zindi recipe. (2) Fine-tune Omnilingual CTC-300M-v2 or LLM-300M on WAXAL lin+sna (+lug). Both are around 300–600M params and trainable on 8 GB with bf16/fp16, gradient checkpointing and small batches. Add a KenLM 5-gram trained on WAXAL text for CTC models. Use Omnilingual-1B CTC / Sunbird-51 zero-shot only for pseudo-labelling or ensembling.

### Cited Findings
- Reference hardware used by others: the WAXAL-NET models were trained on A100 40/80 GB. The anyantudre w2v-BERT was trained on an RTX 5090 32 GB with batch 12 × 3. The keystats w2v-BERT used per-device batch 4 × grad-accum 8, fp16 and gradient checkpointing (GPU not stated). The omniASR-LLM-300M all19 model used 2× H200 for 5,000 steps — [WAXAL-NET](https://arxiv.org/html/2606.02375v1); [anyantudre card](https://huggingface.co/anyantudre/waxal-w2vbert-linsnalug-raw); [keystats card](https://huggingface.co/keystats/w2v-bert-2.0-multilingual-waxal-shona-lingala-luganda-best); [all19 card](https://huggingface.co/waxal-benchmarking/omniasr-llm-300m-waxal-all19)
- WAXAL train sizes: lin 71.9 h (14,400 utts), sna 79.7 h (14,109), lug 37.3 h (5,455). Test: lin 9.3 h, sna 9.9 h, lug 4.3 h — [WAXAL-NET](https://arxiv.org/html/2606.02375v1). WAXAL clips average 20.2 s — [anyantudre card](https://huggingface.co/anyantudre/waxal-w2vbert-linsnalug-raw)
- Omnilingual inference VRAM: CTC-300M ~2 GiB, CTC-1B ~3 GiB, CTC-3B ~8 GiB, LLM-300M ~5 GiB, LLM-1B ~6 GiB (BF16, A100) — [GitHub](https://github.com/facebookresearch/omnilingual-asr)
- An n-gram LM gives large gains for w2v-BERT/XLS-R CTC (e.g., Luganda 1h 55.69 → 38.16 WER; Shona 50h 23.54 → 8.61) — [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1)
- Raw vocabulary with case and punctuation gave +0.0171 on the Zindi LB, the "largest single modelling gain". Ensembling (ROVER) added about +0.02 — [anyantudre card](https://huggingface.co/anyantudre/waxal-w2vbert-linsnalug-raw)

### Inferences (UNVERIFIED engineering estimates, not from sources)
- **VRAM for fine-tuning on 8 GB.** Assumes 16 kHz audio, clips ≤20–30 s, mixed precision, gradient checkpointing, 8-bit AdamW where possible.
  - 300M CTC models (mms-300m, XLS-R-300M, omniASR-CTC-300M): about 5–7 GB at batch 4–8; fits.
  - w2v-BERT 2.0 (600M): about 7–8 GB at batch 2–4 with grad-accum; tight but feasible.
  - mms-1b-all, adapter-only training (~2M trainable params): fits.
  - Whisper-small, full fine-tuning: fits. Whisper large-v3-turbo: LoRA/QLoRA only.
  - Whisper large-v3, omniASR 1B+ CTC or LLM, full fine-tuning: does not fit.
- **Inference:** every model in Table A except the 7B (and possibly 3B LLM / Seamless v2 fp32) runs on 8 GB.
- **Compute budget:** ~190 h of lin+sna+lug audio means one epoch of a 600M encoder on a laptop RTX 4070 likely takes several hours. 5 days allows about 8–15 epochs of one model, or a few shorter runs. That is enough for one solid w2v-BERT or omniASR-300M model plus LM tuning, but not the 26-model ensemble of the top Zindi solution.
- **Suggested ranking for the challenge** (lin+sna emphasis, metric 1 − 0.5·(WER+CER)):
  1. Continue from the public `anyantudre/waxal-w2vbert-linsnalug-raw` (Apache 2.0) or `keystats` multilingual checkpoint, or fine-tune w2v-BERT 2.0 yourself with a raw vocabulary, then add a KenLM 5-gram.
  2. Fine-tune omniASR-CTC-300M-v2 (Apache 2.0, fairseq2/WSL2).
  3. Fine-tune mms-300m (CC-BY-NC) as a cheap baseline.
  4. Use Sunbird-51 Whisper (Apache 2.0) or omniASR-1B zero-shot for pseudo-labels or ensemble diversity.
- **Licensing:** prefer Apache/MIT bases (w2v-BERT 2.0, Omnilingual, Whisper, XLS-R) over CC-BY-NC (MMS, Seamless v2) if the challenge requires open or commercial-compatible licenses.

### Gaps
- No source gives measured VRAM or throughput for fine-tuning these models on an 8 GB laptop GPU. The figures above are estimates to validate with a short trial run.
- The current status, rules and deadlines of the Zindi "Google WAXAL ASR Challenge" were not checked. The anyantudre card reports final public/private scores, which suggests at least one phase has closed. Confirm on [Zindi](https://zindi.world/competitions/google-waxal-asr-challenge).

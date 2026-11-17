# SOTA architectures for low-resource ASR — Lingala, Shona, Luganda (WAXAL challenge)

> Corrected scope: the Zindi WAXAL challenge covers **Lingala (lin), Shona (sna), Luganda (lug)** — not Wolof/Serer. Reference checkpoint cited by the user: `anyantudre/waxal-w2vbert-linsnalug-raw`.
> Constraint: 8GB GPU → favor lightweight fine-tuning (CTC head, LoRA) over heavy retraining of a 7B LLM decoder.

## 1. w2v-BERT 2.0 (Meta) — architecture and pre-training

- **Source paper**: SeamlessM4T (arXiv 2308.11596) section 3.2.1; public checkpoint `facebook/w2v-bert-2.0`.
- **Architecture**: Conformer encoder, 24 layers, ~600M parameters (0.6B). The pre-training objective combines contrastive learning + masked prediction (like the original w2v-BERT, Chung et al. 2021), improved with **two GVQ codebooks** (product quantization) instead of one. Total loss = weighting 1.0 (contrastive) + 0.5 + 0.5 (two masked-prediction terms).
- **Pre-training data size**: sources disagree — the SeamlessM4T v1 paper says 1M hours / 143 languages; the Meta SeamlessM4T v2 page says 4.5M hours. The HF page `facebook/w2v-bert-2.0` also states 4.5M hours / >143 languages (metadata lists 96 tagged languages).
- **License: MIT** (facebook/w2v-bert-2.0 on HF). It is a **"bare"** checkpoint — no modeling head; it needs fine-tuning (e.g. a CTC head) to do ASR. Meta has published no smaller alternative size for this specific model (unlike MMS, which has 300M/1B variants).
- **Comparison with MMS**: according to the AfricaNLP 2026 benchmark (Nahabwe et al., arXiv 2512.10968), MMS and w2v-BERT are both **more data-efficient in the very low-resource regime** (a few hours to a few tens of hours), whereas XLS-R scales better when more data is available, and Whisper is better in the mid-resource regime. w2v-BERT 2.0 (via wav2vec2-bert + CTC head) is the base choice of the `anyantudre/waxal-w2vbert-linsnalug-raw` checkpoint.

## 2. MMS (Massively Multilingual Speech, Meta)

- Checkpoint family `facebook/mms-300m`, `facebook/mms-1b-all` (CTC fine-tuned on 1162 languages), `facebook/mms-1b-fl102`. Standard wav2vec2 architecture with per-language CTC heads (lightweight per-language adapters — useful for low-resource fine-tuning since only a small head/adapter is trained).
- **WAXAL Lingala-specific checkpoint**: `waxal-benchmarking/mms-300m-waxal-lin` — WER **42.6%**, CER **18.9%** on the filtered WAXAL test (base `facebook/mms-300m`).
- According to Meta's Omnilingual ASR paper (arXiv 2511.09690, Nov. 2025), MMS has been partly superseded by **Omnilingual ASR** (see §5), which covers 1600+ languages with CTC and LLM-decoder variants.
- Gains vs data size: on 13 African languages (arXiv 2512.10968), most gains are obtained before 50h of training; beyond that the curves plateau and vary strongly by language (e.g. Lingala: WER ~51.3% at 1h → ~23.96% at 100h — figure cited for Lingala in this paper, to be checked against the full version of the PDF, which could not be fully extracted).
- **License**: some MMS checkpoints are generally CC-BY-NC 4.0, but `facebook/mms-1b-all`/`mms-300m` on HF are released under a license that allows fine-tuning (check the exact model card before commercial use — Meta has historically put some MMS assets under a research-only CC-BY-NC license, while wav2vec2-only weights are often Apache/MIT-like via transformers).

## 3. Whisper (OpenAI) fine-tuned for low resource — known pitfalls

- **Zero-shot hallucinations**: on Bambara, *all* tested Whisper variants exceed 100% WER (more tokens produced than in the reference), a sign of massive hallucination (arXiv 2602.09785).
- **Insertions / repetitions**: a Kikuyu study (arXiv 2510.07221, fine-tuning on ~270h) tags samples with WER≥40% as "spurious repetition" when repetitions exceed 10% of the text — a problem that persists even after partial fine-tuning.
- **Fine-tuning largely fixes the problem**: on Swivuriso (South Africa, arXiv 2512.02201), zero-shot Whisper gives WER 223% (Setswana) and 190% (Xitsonga); after fine-tuning, these figures drop to 13% and 12%. The authors link low-resource hallucinations to insertion errors.
- **Results specific to the 3 challenge languages**:
  - **Luganda** — `waxal-benchmarking/whisper-tiny-lug-victor` (Whisper Tiny, fine-tuned on WAXAL): WER **33.8%**, CER **12.8%** (spontaneous WAXAL test). `BuzzASR/luganda` (fine-tuned Whisper Large v3): WER 50.0% (FLEURS) / 42.92% (Common Voice 25) / combined 44.8%, combined CER 24.97%.
  - **Shona** — `Code-Dev/whisper-small-shona-v3(-a)` (progressive fine-tuning on FLEURS + Shona Bible): WER drops from 37.15% (FLEURS baseline) to 25.52–26.42% after several stages. `CasperMuz/whisper-base-sna` (fine-tuned on WAXAL Shona): WER 41.48%.
  - **Lingala** — generic Whisper-base fine-tune (deepdml, card mislabeled as Igbo): WER ~54.9% on FLEURS, CER ~21.4%. No WAXAL-specific Whisper-Lingala figure was found in the search results, unlike MMS/w2v-BERT.
- **Sunbird AI "Whisper 51 African Languages"**: multilingual adaptation of Whisper Large v3 covering 49-51 African languages including Lingala (`lin`), explicitly listing hallucination and substitution errors as known limitations, especially for the least-resourced languages.
- **Practical conclusion**: zero-shot Whisper is **unusable as-is** on these 3 languages (heavy hallucinations); fine-tuning, even light, is essential. CTC (w2v-BERT/MMS) tends to be more robust and more data-efficient in very low-resource settings, without the free-text hallucination risk specific to seq2seq/decoder architectures.

## 4. CTC vs seq2seq (encoder-decoder/attention) — low-resource

- No study was found that directly compares CTC vs attention *on WER + noise robustness* with a strictly identical protocol in African low-resource settings, but the consensus in the literature (Myanmar ~26h, Babel Turkish, MERL hybrid CTC/attention on WSJ0 ~15h) points in the same direction:
  - **CTC**: cheaper to train, easier to integrate (WFST/n-gram LM decoding), **better with very little data** (a few tens of hours) — this is the approach of all the WAXAL checkpoints mentioned above based on w2v-BERT/MMS/XLS-R.
  - **Pure attention/seq2seq** (Whisper-like): can surpass CTC when data is abundant, but is **more expensive**, harder to parallelize, and more prone to hallucinations/repetitions in low-resource/zero-shot settings (see §3).
  - **Hybrid CTC/attention**: the auxiliary CTC loss improves noise robustness by enforcing monotonic alignment; reported gains of 5.4–14.6% relative CER over attention alone (MERL comparison, outside a specifically African context).
  - Meta's **Omnilingual ASR** paper (see §5) offers precisely both heads (CTC and LLM-decoder) on the same encoder, which allows a direct comparison; the CTC-300M variants are much cheaper to fine-tune on an 8GB GPU than a 7B LLM-decoder.
- **For an 8GB GPU**: fine-tuning a CTC head on a frozen/partially frozen encoder (w2v-BERT 2.0 or MMS-300M) is very feasible; fine-tuning a 7B LLM-decoder (such as `facebook/omniASR-LLM-7B`, used by the Zindi solution `ravitejac/waxal-asr-solution`) requires LoRA/aggressive quantization and remains borderline or even out of reach without offloading.

## 5. Omnilingual ASR (Meta, Nov. 2025) — relevant new foundation model

- **Paper**: arXiv 2511.09690, "Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages". Corresponding Meta AI blog post.
- Covers **1600+ languages**, including 500+ never before served by an ASR system; allows adding an uncovered language with only a few samples (few-shot/in-context, via the LLM-decoder).
- **Two head variants** on the same self-supervised encoder scaled up to 7B: (a) **CTC** head, (b) **LLM-style decoder** (wav2vec2 encoder + LLM-style transformer decoder). Sizes: from 300M (device-friendly) to 7B.
- **License**: Apache-2.0 weights (according to secondary coverage); the **Omnilingual ASR Corpus** (350 under-resourced languages, commissioned data) is released under CC-BY. To be verified precisely on the `facebookresearch/omnilingual-asr` repo before use.
- **Direct relevance for the challenge**: the HF organization `waxal-benchmarking` has published per-language fine-tuned **"WAXALNet"** collections derived from this foundation model:
  - **WAXALNet — Omnilingual CTC-300M** (19 languages): CTC head, 300M params — probably the most practical starting point on an 8GB GPU.
  - **WAXALNet — Omnilingual LLM-300M** (19 languages): wav2vec2→LLM-style decoder, 300M params.
  - **WAXALNet — Whisper Tiny** fine-tuned on WAXAL.
  - Ablation checkpoints named `omniasr-llm-300m-waxal-loo-{ach,ewe,ful,lin}` exist but are explicitly tagged "Not intended for use" (leave-one-out ablation archives).
  - The winning/top Zindi solution `ravitejac/waxal-asr-solution` combines: per-language CTC fine-tunes of `facebook/w2v-bert-2.0` (separate Lingala/Shona), a 7B decoder `facebook/omniASR-LLM-7B` fine-tuned with an **MWER** objective (minimum word error rate, expected-risk), a 5-gram KenLM for decoding, language routing via `facebook/mms-lid-256`, and a LoRA r16 error corrector on `google/gemma-4-E4B-it`. Public score 0.757, private 0.766 (metric = 1 − (WER+CER)/2).

## 6. Public HF checkpoints — summary table (Lingala / Shona / Luganda)

| Language | Model | Base | Metric | License (if known) | Notes |
|---|---|---|---|---|---|
| lin+sna+lug | `anyantudre/waxal-w2vbert-linsnalug-raw` | w2v-BERT 2.0 (600M) + CTC, 97-token vocab | public leaderboard: score 0.742, CER 0.116, WER 0.400 | Apache-2.0 (model), CC-BY-4.0 (WaxalNLP data) | Multilingual, weaker than separate per-language specialists; no reliable measurement on Luganda (absent from the Phase 2 test set) |
| lin | `waxal-benchmarking/mms-300m-waxal-lin` | MMS-300M + CTC | WER 42.6%, CER 18.9% | — | Filtered WAXAL test |
| lin | deepdml whisper-base (card labeled ig) | Whisper-base | WER ~54.9%, CER ~21.4% | — | Mislabeled card (Igbo/Lingala confused), use with caution |
| sna | `Code-Dev/whisper-small-shona-v3-a` / `-v3` | Progressively fine-tuned Whisper-small | WER 25.5–26.4% | — | FLEURS + Shona Bible test |
| sna | `CasperMuz/whisper-base-sna` | Whisper-base | WER 41.48% | — | Fine-tuned on WAXAL Shona |
| sna | `KhayaAI/w2v-bert-sna` | w2v-BERT 2.0 + CTC | WER 3.02% | — | Narrow in-domain test — do not generalize |
| lug | `waxal-benchmarking/whisper-tiny-lug-victor` | Whisper-tiny | WER 33.8%, CER 12.8% | — | Spontaneous WAXAL test |
| lug | `BuzzASR/luganda` | Whisper Large v3 | combined WER 44.8%, CER 24.97% | — | FLEURS + Common Voice 25 |
| lug | `dmusingu/XLS-R-LUGANDA-ASR-CV14` | XLS-R 300M + CTC | WER 24.1%, CER 5.4% | — | Common Voice 14 |
| lug | `indonesian-nlp/wav2vec2-luganda` | XLSR-53 + CTC | WER 15.38% (without LM) | — | Common Voice 7 (old) |
| lug | `VinyVan/xlsr-luganda-waxal` | XLSR Swahili 400h + CTC | WER 21.44%, CER 4.46% | — | Eval dataset not specified |
| lin+sna+lug | `waxal-benchmarking/WAXALNet-Omnilingual-CTC-300M` (collection, 19 languages) | Omnilingual ASR CTC-300M | not published on the pages consulted | Apache-2.0 (weights, to be confirmed) | Most practical candidate on an 8GB GPU for complementary fine-tuning |

**Caution**: these figures come from different model cards, on different test sets (filtered WAXAL, FLEURS, various Common Voice versions, Shona Bible) — **not directly comparable**. For a valid comparison, re-evaluate the candidate checkpoints on the same held-out WAXAL split.

## 7. Relevant recent work (2024–2026)

- **Omnilingual ASR** (Meta, Nov. 2025, arXiv 2511.09690) — few-shot/in-context extension to unseen languages, relevant if the WAXAL challenge expands to other Bantu languages.
- **Benchmarking ASR for African Languages** (Nahabwe et al. 2026, arXiv 2512.10968) — comparison of MMS/XLS-R/W2v-BERT/Whisper on 13 African languages, gains vs hours-of-data curve, plateau after ~50h.
- **Voice of a Continent: Mapping Africa's Speech Technology Frontier** (arXiv 2505.18436) — overview of resources and models for African languages.
- **WAXAL corpus (Google, 2026)** — source paper of the challenge; reported fine-tuning reduces macro-averaged WER by up to 17% for MMS-1B-all, up to 58% for Whisper-Large-v3, up to 56% for Gemma 3n 2B.
- **Critique of the WER metric** (AfricaNLP 2026) — argues that WER mixes phonological/tonal/lexical errors for African languages and recommends CER + phonetic-feature-level measures — relevant because Shona and Luganda have tonal/morphological phenomena that are non-trivial for standard WER.
- **Where Are We At with ASR for Bambara?** (arXiv 2602.09785) and **Swivuriso** (arXiv 2512.02201) — systematically document zero-shot Whisper hallucinations on related Bantu/Mande languages and their correction by fine-tuning.
- **Fine-tuning Whisper for Low-Resource Real-World Applications** (arXiv 2412.15726) — generic methodology for low-resource Whisper fine-tuning (useful as a practical recipe).

## 8. Practical recommendations (8GB GPU constraint)

1. **Start from a CTC encoder** (w2v-BERT 2.0 600M or MMS-300M, or even Omnilingual-CTC-300M) rather than a 7B LLM-decoder — far more feasible to fine-tune on 8GB (freeze part of the encoder, small batch, bf16/fp16, gradient checkpointing — cf. the `anyantudre/waxal-w2vbert-linsnalug-raw` recipe: 1 GPU, bf16, gradient checkpointing, effective batch 36).
2. **Specialize per language** rather than a single lin+sna+lug model: model cards consistently indicate that per-language specialist checkpoints outperform the combined multilingual one.
3. **Avoid zero-shot Whisper** for these 3 languages (risk of hallucination/WER>100%); if Whisper is used, fine-tuning is mandatory + ideally length/repetition constraints at decoding.
4. **Decoding with an external LM (KenLM n-gram)** brings a measurable gain (e.g. the top Zindi solution with 5-gram KenLM) and remains cheap in memory compared with an LLM corrector.
5. **Audio augmentation** (MUSAN noise, RIR, random gain) used by the reference checkpoint — a standard, cheap technique to add.
6. **Check licenses before use**: w2v-BERT 2.0 and `anyantudre` checkpoints = MIT/Apache-2.0 (permissive); WaxalNLP data = CC-BY-4.0; MMS has historically had variants under a more restrictive license (CC-BY-NC) — check the exact chosen model card before any use outside the challenge.

## Main sources
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

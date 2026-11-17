# Fine-tuning recipes for low-resource ASR on an 8 GB GPU, and n-gram LM decoding for CTC (Lingala / Shona / Luganda), as of 2026-10

Scope: Hugging Face transformers v5.x on Windows 11, RTX 4070 Laptop (8 GB), about tens of hours per language, about 4 days of compute.
Convention: **[VERIFIED]** means fetched and read this session. **[UNVERIFIED / prior knowledge]** means from the researcher's background knowledge or a search snippet that was not opened; check it before relying on it.

---

## 1. MMS-1B adapter fine-tuning (adapter + LM head only)

### Takeaway
The official HF recipe trains only the per-language adapters (about 2.5M weights) plus a new LM head on `facebook/mms-1b-all`. It uses LR 1e-3, 4 epochs, warmup 100, fp16 and gradient checkpointing, and reaches about 22% WER on Turkish from about 4 h of data in under 30 minutes. MMS adapters are per-language by design. A single model without language ID needs one shared adapter trained on the union vocabulary, or a full or partial fine-tune.

### Cited Findings
- Data: Turkish Common Voice 6.1, "ca. 4h of validated training data", train+validation used for training — [HF blog mms_adapters](https://huggingface.co/blog/mms_adapters) [VERIFIED]
- Model load: `Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, layerdrop=0.0, ctc_loss_reduction="mean", pad_token_id=..., vocab_size=len(processor.tokenizer), ignore_mismatched_sizes=True)`. `ignore_mismatched_sizes` is needed because the old head has 154 outputs — [HF blog mms_adapters](https://huggingface.co/blog/mms_adapters) [VERIFIED]
- Adapter-only training: `model.init_adapter_layers()`, `model.freeze_base_model()`, then `requires_grad=True` on `model._get_adapters()`. Each adapter is about 2.5M weights, and the base is about 99% of parameters — [HF blog mms_adapters](https://huggingface.co/blog/mms_adapters) [VERIFIED]
- The vocabulary is nested under a `target_lang` key (e.g. `"tur"`, ISO-639-3) so other languages can be added. Language switching uses `model.load_adapter("swe")` plus `processor.tokenizer.set_target_lang("swe")` — [HF blog mms_adapters](https://huggingface.co/blog/mms_adapters) [VERIFIED]
- TrainingArguments: `learning_rate=1e-3`, `per_device_train_batch_size=32`, `num_train_epochs=4`, `warmup_steps=100`, `group_by_length=True`, `gradient_checkpointing=True`, `fp16=True`, eval/logging every 100 steps, save every 200 steps, `save_total_limit=2` — [HF blog mms_adapters](https://huggingface.co/blog/mms_adapters) [VERIFIED]
- WER on the CV test set by step (the source table is garbled; this is the best reading): 100 → 0.280, 200 → 0.232, 300 → 0.229, 400 → 0.223 — [HF blog mms_adapters](https://huggingface.co/blog/mms_adapters) [VERIFIED]
- Training time: "less than 30 minutes depending on the GPU used". GPU and VRAM are not stated — [HF blog mms_adapters](https://huggingface.co/blog/mms_adapters) [VERIFIED]
- Adapter weights are saved separately: `safe_save_file(model._get_adapters(), "adapter.<lang>.safetensors", metadata={"format":"pt"})` — [HF blog mms_adapters](https://huggingface.co/blog/mms_adapters) [VERIFIED]
- In an African-language benchmark, `facebook/mms-1b-all` was fine-tuned on Lingala, Luganda, Shona and 10 other languages at 1–200 h. Setup: feature extractor frozen, linear head, BF16, activation checkpointing, AdamW, linear schedule with 10% warmup, LR tuned per language in 1e-5…1e-3, effective batch 64, dropout 0.1, early stopping (patience 10 epochs), on 24–80 GB GPUs. MMS was not evaluated with an LM in that paper — [Benchmarking ASR Models for African Languages, arXiv 2512.10968](https://arxiv.org/html/2512.10968) [VERIFIED]
- [UNVERIFIED / prior knowledge] `mms-1b-all` ships pretrained adapters for 1,162 languages, which very likely include `lin`, `sna` and `lug`. If so, a zero-shot baseline is free: `load_adapter("lug")` plus `set_target_lang("lug")`. Check the `adapter.lin/sna/lug.safetensors` files on the model repo.

### Inferences
- VRAM on 8 GB: the 1B base in fp16 is about 2 GB of weights. Because the base is frozen there are no optimizer states for it (Adam states only for about 2.5M params). Activations are the limit. Expect `per_device_train_batch_size` 4–8 with `gradient_accumulation_steps` 4–8 (effective 32) and audio capped at about 15–20 s. This is an estimate; profile it.
- A single model without language ID: per-language adapters need the language known at inference. For one model, train one shared adapter with `target_lang` set to a pseudo-code (e.g. `"mul"`) over the union vocabulary of the three Latin-script orthographies. The adapter's capacity (about 2.5M) may underfit three languages. An alternative is to also unfreeze the top N transformer layers.
- A cheap alternative is to train three adapters plus a small language-ID step (e.g. pick the adapter whose CTC output has the highest confidence, or a tiny LID classifier on pooled features). Only do this if the challenge allows per-utterance routing.
- Priority: adapter training is the fastest experiment (minutes to about 1 h per language). Run it first to get a strong baseline, especially if pretrained `lin/sna/lug` adapters exist to initialize from instead of calling `init_adapter_layers()`.

### Gaps
- No official VRAM numbers for MMS adapter training. No official multilingual-adapter recipe was found.
- I did not confirm that `lin`, `sna` and `lug` adapters exist in `mms-1b-all`.

---

## 2. w2v-BERT 2.0 CTC fine-tuning

### Takeaway
The HF recipe (Mongolian, about 14 h) uses LR 5e-5, batch 16 × grad-acc 2, 10 epochs, warmup 500, fp16, gradient checkpointing, `add_adapter=True` and all dropout/masking at 0. It reaches 32.4% WER, roughly equal to Whisper-large-v3 (33.3%). On African languages at 10–50 h, w2v-BERT 2.0 with a 5-gram LM is the strongest CTC setup for Luganda and Shona; for Lingala the picture is mixed.

### Cited Findings
- Hyperparameters: `per_device_train_batch_size=16`, `gradient_accumulation_steps=2`, `num_train_epochs=10`, `learning_rate=5e-5`, `warmup_steps=500`, `fp16=True`, `gradient_checkpointing=True`, `group_by_length=True`, eval/logging every 300 steps, save every 600 steps, keep 2 checkpoints — [HF blog fine-tune-w2v2-bert](https://huggingface.co/blog/fine-tune-w2v2-bert) [VERIFIED]
- Model config: `attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, mask_time_prob=0.0, layerdrop=0.0, ctc_loss_reduction="mean", add_adapter=True` — [HF blog fine-tune-w2v2-bert](https://huggingface.co/blog/fine-tune-w2v2-bert) [VERIFIED]
- Suggested post-hoc improvements: warmup at 5–15% of steps, more epochs, AdamW β2 of 0.95–0.98 — [HF blog fine-tune-w2v2-bert](https://huggingface.co/blog/fine-tune-w2v2-bert) [VERIFIED]
- Results (Mongolian CV16, about 14 h): WER 0.518 @300 → 0.442 @600 → 0.367 @900 → 0.324 @1200 steps. Whisper-large-v3 fine-tuned: 33.3%. The blog claims w2v-BERT is 10–30× faster and 2.5× more resource-efficient than Whisper-large-v3. The benchmark GPU was a 16 GB V100 — [HF blog fine-tune-w2v2-bert](https://huggingface.co/blog/fine-tune-w2v2-bert) [VERIFIED]
- African benchmark, w2v-BERT 2.0 (600M), WER without LM → with 5-gram LM — [arXiv 2512.10968](https://arxiv.org/html/2512.10968) [VERIFIED]

  | Lang | 10 h | 20 h | 50 h |
  |---|---|---|---|
  | Lingala | 29.33 → 26.94 | 28.78 → 25.36 | 24.65 → 23.13 |
  | Luganda | 49.09 → 28.88 | 47.25 → 23.79 | 44.26 → 21.93 |
  | Shona | 27.23 → 12.92 | 25.12 → 10.91 | 23.54 → 8.61 |

- Same benchmark, XLS-R 300M, WER without LM → with LM — [arXiv 2512.10968](https://arxiv.org/html/2512.10968) [VERIFIED]

  | Lang | 10 h | 20 h | 50 h |
  |---|---|---|---|
  | Lingala | 23.25 → 30.15 | 22.16 → 28.19 | 19.85 → 26.55 |
  | Luganda | 59.95 → 40.39 | 53.96 → 33.91 | 49.67 → 29.05 |
  | Shona | 53.07 → 43.87 | 36.77 → 32.51 | 30.71 → 34.73 |

- transformers v5.16.0 made SDPA attention dispatch work for wav2vec2-bert (`attn_implementation="sdpa"`). The release notes list this as a breaking change — [transformers v5.16.0 release notes](https://newreleases.io/project/github/huggingface/transformers/release/v5.16.0) [search snippet, not opened]

### Inferences
- In that benchmark, XLS-R without an LM is the best Lingala acoustic model (19.85–23.25 WER), and the LM hurts XLS-R on Lingala. w2v-BERT 2.0 plus LM wins clearly for Shona (8.6 vs 30.7) and Luganda (21.9 vs 29.1 at 50 h). So the LM effect depends on the language and the LM corpus (domain mismatch is likely), and alpha/beta must be tuned per language on a dev set.
- On 8 GB: w2v-BERT 2.0 has 600M params. A full fine-tune with AdamW in fp32 master weights needs about 600M × 16 B ≈ 9.6 GB for weights, gradients and optimizer, so it **will not fit** without memory tricks. Options: (a) 8-bit AdamW (`optim="adamw_bnb_8bit"`, which cuts optimizer state about 4×), together with gradient checkpointing, bf16 and batch 2–4 × grad-acc 8–16; (b) freeze the bottom 12 of 24 conformer layers; (c) LoRA on the conformer linear layers. This arithmetic is the researcher's estimate and is not measured.
- Use bf16 rather than fp16 on Ada (RTX 40xx) to avoid CTC loss overflow and NaNs.

### Gaps
- No official VRAM figure for w2v-BERT 2.0 fine-tuning. No published 8 GB recipe was found.
- The benchmark paper does not report the alpha/beta values or the LM corpora used per language.

---

## 3. XLS-R 300M / MMS-300M full CTC fine-tuning

### Takeaway
Standard recipe: LR about 3e-4 (range 1e-4 to 3e-4), about 500 warmup steps, freeze the CNN feature encoder, SpecAugment `mask_time_prob` about 0.05, layerdrop 0.0–0.1, gradient checkpointing, fp16/bf16, `group_by_length`, effective batch about 32 via accumulation, audio capped at about 15–20 s. A 300M model fits in 8 GB at batch about 4–8.

### Cited Findings
- The African benchmark freezes the feature extractor, adds a linear head, and uses BF16, activation checkpointing, AdamW with gradient clipping, a linear schedule with 10% warmup, LR 1e-5…1e-3 tuned per language, effective batch 64, dropout 0.1 and early stopping with patience 10 — [arXiv 2512.10968](https://arxiv.org/html/2512.10968) [VERIFIED]
- [UNVERIFIED / prior knowledge, not re-fetched] HF blog "Fine-tune XLS-R" ([fine-tune-xlsr-wav2vec2](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)) for Turkish CV: `attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, mask_time_prob=0.05, layerdrop=0.0, ctc_loss_reduction="mean"`, `model.freeze_feature_encoder()`, `per_device_train_batch_size=16`, `gradient_accumulation_steps=2`, `learning_rate=3e-4`, `warmup_steps=500`, `num_train_epochs=30`, `fp16=True`, `gradient_checkpointing=True`, `group_by_length=True`.
- In transformers v5, `warmup_ratio` is **removed**. Use `warmup_steps` with a float value (e.g. `warmup_steps=0.1`) — [MIGRATION_GUIDE_V5](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md) [VERIFIED]

### Inferences (recommended 8 GB config, researcher's synthesis)
- `facebook/wav2vec2-xls-r-300m` or `facebook/mms-300m`: `learning_rate=3e-4` (drop to 1e-4 if the loss diverges or goes NaN), `warmup_steps=0.1`, linear or tri-stage decay, `per_device_train_batch_size=8`, `gradient_accumulation_steps=4`, `bf16=True`, `gradient_checkpointing=True`, `group_by_length=True`, `freeze_feature_encoder()`, `mask_time_prob=0.05` (raise to 0.1–0.2 for larger data), `mask_time_length=10`, `mask_feature_prob=0.0–0.01`, `layerdrop=0.05`, dropouts 0.0–0.1. Filter utterances longer than about 16–20 s, which sets the VRAM peak. Training for 20–30 epochs on about 30–90 h is roughly 1 day on a 4070 Laptop (estimate).
- For a single multilingual model, train on the concatenated lin+sna+lug data with a union character vocabulary (lowercased, punctuation normalized), with temperature or uniform sampling across languages if the hours are imbalanced.

### Gaps
- No measured 8 GB VRAM profile for XLS-R 300M was found. The batch sizes above are estimates.

---

## 4. Whisper LoRA / QLoRA with peft + bitsandbytes on 8 GB (Windows)

### Takeaway
The official PEFT int8 recipe trains whisper-large-v2 with LoRA r=32, alpha=64 on q_proj/v_proj at LR 1e-3, batch 8, 3 epochs and fp16. It matches full fine-tuning (36.1 vs 35.6 WER on Marathi) on a 16 GB T4, with 1% trainable params. bitsandbytes officially supports Windows via pip since v0.43. On 8 GB, whisper-small/medium with LoRA (or large-v3-turbo in int8/4-bit) is feasible. Luganda is not a Whisper language, so a proxy language token is needed.

### Cited Findings
- PEFT notebook: `BitsAndBytesConfig(load_in_8bit=True)` on `openai/whisper-large-v2`; `prepare_model_for_kbit_training(model)` (freezes the base and casts non-int8 layers to fp32); `LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none")`; trainable 15.7M / 1.56B (about 1%) — [PEFT int8 Whisper notebook](https://raw.githubusercontent.com/huggingface/peft/refs/heads/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb) [VERIFIED]
- `Seq2SeqTrainingArguments`: `learning_rate=1e-3`, `per_device_train_batch_size=8`, `gradient_accumulation_steps=1`, `warmup_steps=50`, `num_train_epochs=3`, `fp16=True`, `generation_max_length=128`, `remove_unused_columns=False`, `label_names=["labels"]` (needed because the PeftModel forward signature differs) — [PEFT notebook](https://raw.githubusercontent.com/huggingface/peft/refs/heads/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb) [VERIFIED]
- Results (Marathi CV11, 3,927 train rows): LoRA int8 36.14 WER (14.02 normalized) vs full fine-tune 35.65 (13.64). The adapter is about 63 MB. Stated hardware: Colab T4 16 GB — [PEFT notebook](https://raw.githubusercontent.com/huggingface/peft/refs/heads/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb) [VERIFIED]
- Notebook caveats: `predict_with_generate` is avoided because int8 needs autocast and `generate` inside the Trainer does not use it, so evaluation is a manual loop with `torch.amp.autocast`. `forced_decoder_ids=None` and `suppress_tokens=[]` are set, and the language prompt is passed via `processor.get_decoder_prompt_ids(language, task)` at inference — [PEFT notebook](https://raw.githubusercontent.com/huggingface/peft/refs/heads/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb) [VERIFIED]
- transformers v5: `load_in_8bit` / `load_in_4bit` kwargs **removed**, so use `quantization_config=BitsAndBytesConfig(...)`. Generation parameters are no longer read from the model config, so set `model.generation_config.language / task / forced_decoder_ids`. `Trainer(tokenizer=...)` becomes `processing_class=`. `use_cache` defaults to `False` — [MIGRATION_GUIDE_V5](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md) [VERIFIED]
- bitsandbytes ≥0.43.0 officially supports Windows via `pip install bitsandbytes` — [bnb issue #1131](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1131) [search snippet]. Current wheels ship precompiled CUDA 11.8–13.0 binaries and need Python ≥3.10 and PyTorch ≥2.4 — [bnb installation docs](https://huggingface.co/docs/bitsandbytes/installation) [search snippet]. LLM.int8 needs compute capability ≥7.5; an RTX 4070 (Ada, 8.9) qualifies.
- On African languages, Whisper-small fine-tuned end-to-end was often worse than the CTC models at 10–50 h (e.g. Luganda and Swahili at 10 h) — [arXiv 2512.10968](https://arxiv.org/html/2512.10968) [VERIFIED; per the fetch summary, exact Whisper numbers not extracted]

### Inferences
- [UNVERIFIED / prior knowledge] Whisper's language list includes Shona (`sn`) and Lingala (`ln`) but **not** Luganda. Common workarounds: reuse a related Bantu token (e.g. `sw` Swahili or `sn`) for Luganda, or repurpose a rarely used token, and train with that prefix consistently. Without language ID at test time, either let Whisper auto-detect (unreliable for these languages) or force one shared token for all three languages.
- [UNVERIFIED / prior knowledge] Hallucination and repetition on silence or long audio is a known Whisper failure mode. Mitigations: train on segments of 30 s or less, `no_repeat_ngram_size`, `compression_ratio_threshold`, and a small `max_new_tokens`. This is a reason to prefer CTC models for a WER competition with a tight timeline.
- 8 GB feasibility (estimate): whisper-small (244M) full fine-tune in bf16 with 8-bit Adam fits. whisper-medium or large-v3-turbo (809M) need LoRA, plus int8 or 4-bit base weights for large models.

### Gaps
- No measured VRAM for Whisper LoRA on 8 GB was found. No Luganda-specific Whisper token study was found.

---

## 5. Data augmentation (speed perturbation, noise, MUSAN, audiomentations)

### Takeaway
Few sources were fetched this session. Speed perturbation (0.9/1.0/1.1) is the standard, cheap gain for low-resource ASR. Noise/RIR augmentation helps mainly when the test audio is noisy or mismatched. Gains for SSL fine-tuning are usually modest (a few % relative), and SpecAugment (`mask_time_prob`) already provides in-model augmentation.

### Cited Findings
- No source on augmentation gains was fetched this session.
- [UNVERIFIED / prior knowledge] Ko et al., "Audio Augmentation for Speech Recognition" (Interspeech 2015): 3-way speed perturbation (0.9, 1.0, 1.1) gave about 4.3% average relative WER reduction across 4 tasks (range about 0.3–6.7%).
- [UNVERIFIED / prior knowledge] audiomentations (pure Python, Windows-friendly) offers `AddGaussianNoise`, `AddBackgroundNoise` (e.g. with MUSAN noise/music), `TimeStretch`, `PitchShift`, `RoomSimulator` and `Gain`. Applying it on the fly in the data collator costs CPU time; precomputing speed-perturbed copies is cheaper on a laptop.

### Inferences
- With about 4 days: precompute speed-perturbed copies (0.9 and 1.1) for the smallest-resource language only, or for all languages if time allows. Keep `mask_time_prob` about 0.05. Add light background noise (SNR 10–30 dB, p≈0.3) only if the WAXAL audio is field-recorded and noisy.

### Gaps
- No measured gains for augmentation on Lingala, Shona or Luganda, or on SSL CTC fine-tuning in 2024–2026, were found this session.

---

## 6. N-gram LM decoding (pyctcdecode + KenLM) on Windows

### Takeaway
A 5-gram KenLM through pyctcdecode gives large gains for Luganda and Shona (often 30–60% relative WER reduction with w2v-BERT at 10–50 h). It can hurt (Lingala, Bemba, Bambara) when the LM text is out of domain or alpha/beta are not tuned. On Windows the blocker is building KenLM (`lmplz`/`build_binary` and the Python `kenlm` module): there is no official Windows wheel. WSL2 is the reliable route.

### Cited Findings
- HF recipe: build KenLM with CMake (`lmplz -o 5 < text.txt > 5gram.arpa`, Kneser-Ney by default). Then add the missing `</s>` unigram line (copy of `<s>`'s logprob) and increment the `ngram 1=` count, which transformers requires. Then `build_ctcdecoder(labels=sorted_vocab, kenlm_model_path=...)` → `Wav2Vec2ProcessorWithLM(feature_extractor, tokenizer, decoder)`. `build_binary` shrinks the LM (4.1 GB → 1.8 GB) and makes it load faster. The Python module is installed with `pip install https://github.com/kpu/kenlm/archive/master.zip pyctcdecode` — [HF blog wav2vec2-with-ngram](https://huggingface.co/blog/wav2vec2-with-ngram) [VERIFIED]
- The HF blog reports that the 5-gram gave a "roughly 30%" relative WER improvement for xls-r-300m-sv (final 18.85% on CV7 test) — [HF blog wav2vec2-with-ngram](https://huggingface.co/blog/wav2vec2-with-ngram) [VERIFIED]
- African benchmark: 5-gram modified Kneser-Ney with KenLM, decoded with pyctcdecode beam search. LM text came from news, health, education, agriculture and religious sources. LM gains are largest at 10–50 h, strongest for Luganda, Swahili and Shona, shrink above 100 h, and sometimes hurt (Bemba, Bambara, Lingala with XLS-R) — [arXiv 2512.10968](https://arxiv.org/html/2512.10968) [VERIFIED]
  - Luganda w2v-BERT 50 h: 44.26 → 21.93 WER (−50% rel). Shona w2v-BERT 50 h: 23.54 → 8.61 (−63% rel). Lingala w2v-BERT 50 h: 24.65 → 23.13 (−6% rel). Lingala XLS-R 50 h: 19.85 → 26.55 (+34% rel, worse) — computed from [arXiv 2512.10968](https://arxiv.org/html/2512.10968) [VERIFIED]
- pyctcdecode defaults: `DEFAULT_ALPHA=0.5`, `DEFAULT_BETA=1.5`, `DEFAULT_BEAM_WIDTH=100`, `DEFAULT_PRUNE_LOGP=-10.0`, `DEFAULT_UNK_LOGP_OFFSET=-10.0`, `DEFAULT_HOTWORD_WEIGHT=10.0`, `DEFAULT_MIN_TOKEN_LOGP=-5.0`, `DEFAULT_SCORE_LM_BOUNDARY=True` — [pyctcdecode constants.py](https://raw.githubusercontent.com/kensho-technologies/pyctcdecode/main/pyctcdecode/constants.py) [VERIFIED]
- The pyctcdecode README Quick Start uses `alpha=0.5, beta=1.0` ("tuned on a val set") and accepts `.arpa` or `.bin` LMs. Batch decoding uses `multiprocessing.get_context("fork")` in its example. `fork` does not exist on Windows, so use `pool=None` or a `spawn` pool — [pyctcdecode GitHub](https://github.com/kensho-technologies/pyctcdecode) [VERIFIED for README content; the Windows implication is the researcher's]
- KenLM Windows: no official Windows wheel was found. The kpu/kenlm README installs from a source zip. Guides list Windows build failures as a known pitfall and recommend WSL or Linux — [search results: kenlm Windows](https://adg.csdn.net/696f2c2a437a6b40336993a3.html), [piwheels pypi-kenlm (Linux ARM only)](https://www.piwheels.org/project/pypi-kenlm) [search snippets]. The only conda kenlm package found (bitextor channel) is linux-64 only — [anaconda.org bitextor/kenlm](https://anaconda.org/bitextor/kenlm) [search snippet]. I found no confirmation of a conda-forge win-64 kenlm.
- torchaudio entered maintenance mode at 2.9 and removed APIs deprecated in 2.8. The CUDA CTC decoder (`cuda_ctc_decoder`) was deprecated in 2.8 with removal planned for 2.9. The flashlight-text CPU decoder requires `pip install flashlight-text` separately (since torchaudio 2.1) — [torchaudio 2.9 cuda_ctc_decoder doc](https://docs.pytorch.org/audio/2.9.0/generated/torchaudio.models.decoder.cuda_ctc_decoder.html), [torchaudio 2.1 release notes](https://newreleases.io/project/github/pytorch/audio/release/v2.1.0) [search snippets]. Whether `torchaudio.models.decoder.ctc_decoder` survives in 2.9+ was **not confirmed**.

### Inferences (Windows-practical options, ranked)
1. **WSL2 (Ubuntu)**: `apt install build-essential cmake libboost-all-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev`, build KenLM, and run `lmplz -o 4/5 --discount_fallback` (fallback needed for small corpora) and `build_binary`. Either run pyctcdecode decoding in WSL as well (export the logits from Windows as .npy, so the GPU is not needed for decoding), or try `pip install https://github.com/kpu/kenlm/archive/master.zip` on native Windows with MSVC Build Tools plus CMake. That second path is [UNVERIFIED]; it may work because the Python module compiles only the query code, but expect friction.
2. **ARPA without lmplz**: ARPA is a plain-text format, so a pure-Python script can count n-grams and write a backoff ARPA (Kneser-Ney or Witten-Bell). For corpora of a few million words this is fast enough. [UNVERIFIED / prior knowledge] NLTK's `nltk.lm` has no ARPA export, and SRILM's `ngram-count` writes ARPA but its license is non-commercial/research, not OSI open-source. pyctcdecode still needs the `kenlm` Python module to *load* the ARPA.
3. **pyctcdecode without kenlm**: [UNVERIFIED / prior knowledge] pyctcdecode defines an `AbstractLanguageModel` interface (`language_model.py`), so a pure-Python n-gram scorer (reading an ARPA into dicts) can be plugged in. It is slower but fully native on Windows. Without any LM, pyctcdecode still supports beam search plus hotwords.
4. **NVIDIA NeMo n-gram tools** (`train_kenlm.py`, beam search via `pyctcdecode`/flashlight): these also rely on KenLM binaries and are Linux-oriented. Not investigated in depth.
- Alpha/beta tuning: grid alpha ∈ [0.2, 1.5], beta ∈ [0, 3] on a held-out dev set per language, with beam 50–100 during the grid search and 100–200 for the final run. Use `decoder.reset_params(alpha=..., beta=...)` [UNVERIFIED method name] to avoid reloading the LM. Given the Lingala result, keep "no-LM" as a candidate and choose per language.
- LM text: in-domain text (the training transcripts plus similar-domain text) matters more than size. For the single multilingual model, either build one LM on the combined lin+sna+lug text or three LMs selected per utterance. A combined LM is simpler with no language ID and is acceptable because the vocabularies barely overlap. **Exclude any test or dev transcripts from the LM.**

### Gaps
- I did not confirm whether flashlight-text publishes Windows wheels.
- I did not confirm whether conda-forge has a win-64 kenlm build. Run `conda search -c conda-forge kenlm` to check.
- I did not confirm whether transformers v5 `Wav2Vec2ProcessorWithLM.batch_decode` handles Windows' lack of `fork`.

---

## 7. Checkpoint averaging and ROVER / ensembling for CTC

### Takeaway
No primary source was fetched this session. Both techniques are standard and cheap. Averaging the weights of the last N checkpoints of one run (N≈3–5) typically gives a small, free gain. For ensembling CTC models that share a vocabulary and frame rate, averaging the per-frame log-posteriors before beam or LM decoding is simplest. ROVER (NIST SCTK `rover`) combines word-level outputs from heterogeneous systems.

### Cited Findings
- No fetched source. [UNVERIFIED / prior knowledge] Checkpoint averaging means loading the last N `checkpoint-*/model.safetensors` and averaging each tensor (as in fairseq's `average_checkpoints.py` and ESPnet). ROVER is implemented in NIST SCTK (`rover`, open-source), which takes CTM inputs.

### Inferences
- With `save_total_limit=2` in the HF recipes you cannot average later, so raise it to about 5 on the final runs.
- Frame-level logit averaging works for XLS-R 300M plus MMS-300M (same 20 ms frame rate and the same vocab if both are trained with the same tokenizer). For w2v-BERT 2.0 (also 20 ms frames, but with `add_adapter` downsampling) check the output lengths before combining. Otherwise use ROVER on the decoded words.

### Gaps
- No measured gains for checkpoint averaging or ROVER on African-language CTC were found this session.

---

## 8. transformers v5 breaking changes affecting Wav2Vec2ForCTC / Wav2Vec2BertForCTC / Whisper training scripts

### Takeaway
Most HF blog notebooks (2021–2024) need small edits for v5. The main ones: `evaluation_strategy` becomes `eval_strategy`, `Trainer(tokenizer=)` becomes `processing_class=`, `warmup_ratio` is removed (use a float `warmup_steps`), `load_in_8bit` becomes `quantization_config`, generation settings move to `generation_config`, tokenizer `batch_decode` merges into `decode`, the TF/Flax backends are gone, and `huggingface_hub>=1.0`.

### Cited Findings
- TrainingArguments: `logging_dir` removed (use the `TENSORBOARD_LOGGING_DIR` env var). `warmup_ratio` removed (`warmup_steps` accepts floats). `overwrite_output_dir` removed. `per_gpu_*` becomes `per_device_*`. `no_cuda` becomes `use_cpu`. `fp16_backend` / `half_precision_backend` removed (`torch.amp` only). `push_to_hub_token` becomes `hub_token`. `include_inputs_for_metrics` becomes `include_for_metrics` — [MIGRATION_GUIDE_V5](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md) [VERIFIED]
- Trainer: `tokenizer` becomes `processing_class`. `train(model_path=)` becomes `resume_from_checkpoint`. `use_cache` defaults to False. accelerate ≥1.1.0 — [MIGRATION_GUIDE_V5](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md) [VERIFIED]
- Tokenizers: `batch_decode` becomes `decode` (returns a list for batched input). `as_target_tokenizer()` removed (use `tokenizer(text_target=...)`), which matters for old Wav2Vec2 collators using `with processor.as_target_processor():`. `additional_special_tokens` becomes `extra_special_tokens`. `clean_up_tokenization()` moved to model-level classes (e.g. Wav2Vec2) — [MIGRATION_GUIDE_V5](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md) [VERIFIED]; [v5.0.0rc0 notes](https://newreleases.io/project/github/huggingface/transformers/release/v5.0.0rc0) [search snippet]
- Quantization: `load_in_4bit` / `load_in_8bit` removed, so use `quantization_config=BitsAndBytesConfig(...)`. Generation params are no longer read from the model config (Whisper: set `model.generation_config.language`, `.task`, `.forced_decoder_ids`) — [MIGRATION_GUIDE_V5](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md) [VERIFIED]
- Environment: TF/JAX removed. `TRANSFORMERS_CACHE` removed (use `HF_HOME`). `huggingface_hub>=1.0.0` (httpx). `hf_transfer` replaced by `hf_xet`. `safe_serialization` removed (always safetensors). `transformers-cli` becomes `transformers` — [MIGRATION_GUIDE_V5](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md) [VERIFIED]
- v5.16.0: SDPA dispatch now supported for wav2vec2-bert and wav2vec2_conformer (listed as breaking), plus Whisper tokenizer fixes — [v5.16.0 notes](https://newreleases.io/project/github/huggingface/transformers/release/v5.16.0) [search snippet]. An issue reports that lazy loading in Wav2Vec2/Hubert pulls in a torchvision dependency (breaks PyInstaller) — [issue #44963](https://github.com/huggingface/transformers/issues/44963) [search snippet]
- [UNVERIFIED / prior knowledge] `evaluation_strategy` was renamed `eval_strategy` (deprecated in v4.4x, removed by v5). The MMS adapter blog still uses `evaluation_strategy="steps"` — [HF blog mms_adapters](https://huggingface.co/blog/mms_adapters) [VERIFIED that the blog uses the old name].

### Inferences
- Porting checklist for the blog scripts: `evaluation_strategy`→`eval_strategy`; `tokenizer=processor.feature_extractor`→`processing_class=processor`; replace `as_target_processor()` with `processor(text=...)` / `processor.tokenizer(...)`; `processor.batch_decode`→`processor.decode` (check that the processor still exposes `batch_decode`); `warmup_ratio`→float `warmup_steps`; prefer `bf16=True` on an RTX 4070; on Windows set `dataloader_num_workers=0` or wrap the training in `if __name__ == "__main__":` (spawn semantics).

### Gaps
- The migration guide fetched is the RC0-era document. There may be audio-specific changes in v5.1–v5.20 (e.g. feature-extractor and processor saving to `processor_config.json`) that were not checked in detail.

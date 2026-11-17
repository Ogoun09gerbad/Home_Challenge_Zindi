"""Multilingual CTC fine-tuning (lin+sna+lug, shared character vocabulary, no language token).
Models: facebook/w2v-bert-2.0, facebook/mms-300m, facebook/wav2vec2-xls-r-300m, facebook/mms-1b-all (--adapter).
Prerequisite: python cache_audio.py --split train validation
Example:
  python train.py --model facebook/w2v-bert-2.0 --out models/w2vbert --epochs 10 --lr 5e-5 --bs 8 --accum 4
"""
import argparse, json, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from cache_audio import get_audio, load_cached, open_mmaps
from common import LANGS, ROOT, SR, clean_raw, normalize, score, set_seed


# ---- vocabulary ----------------------------------------------------------------
def build_vocab(texts, path):
    chars = sorted(set("".join(texts)) - {" "})
    vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, **{c: i + 3 for i, c in enumerate(chars)}}
    path.write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")
    return vocab


# ---- data -------------------------------------------------------------------
class CachedASR(Dataset):
    """Memmaps are opened lazily in each worker (a pickled memmap would be copied in full)."""
    def __init__(self, df, split, augment=False):
        self.df, self.split, self.augment, self.mms = df.reset_index(drop=True), split, augment, None

    def __getstate__(self):
        return {**self.__dict__, "mms": None}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        if self.mms is None:
            self.mms = open_mmaps(self.split)
        r = self.df.iloc[i]
        y = get_audio(r, self.mms)
        if self.augment:
            y = augment(y)
        return {"audio": y, "text": r.norm}


def augment(y):
    """Speed perturbation (0.9/1.0/1.1), random gain, light white noise."""
    if random.random() < 0.4 and len(y) > 2048:
        import librosa
        sp = random.choice([0.9, 1.1])
        y = librosa.resample(y, orig_sr=int(16000 * sp), target_sr=16000)
    if random.random() < 0.5:
        y = y * 10 ** (random.uniform(-6, 6) / 20)
    if random.random() < 0.3:
        snr = random.uniform(15, 35)
        p = np.mean(y ** 2) + 1e-9
        y = y + np.random.randn(len(y)).astype(np.float32) * np.sqrt(p / 10 ** (snr / 10))
    return np.clip(y, -1, 1).astype(np.float32)


class LengthBucketSampler(Sampler):
    """Shuffles in buckets of similar lengths (less padding), deterministic via the seed."""
    def __init__(self, lengths, bs, seed=42):
        self.lengths, self.bs, self.seed, self.epoch = np.asarray(lengths), bs, seed, 0

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch); self.epoch += 1
        idx = g.permutation(len(self.lengths))
        chunk = self.bs * 50
        out = []
        for k in range(0, len(idx), chunk):
            c = idx[k:k + chunk]
            c = c[np.argsort(self.lengths[c])]
            batches = [c[j:j + self.bs] for j in range(0, len(c), self.bs)]
            g.shuffle(batches)
            out += [int(i) for b in batches for i in b]
        return iter(out)


class Collator:
    def __init__(self, processor, is_bert):
        self.p, self.is_bert = processor, is_bert

    def __call__(self, feats):
        audios = [f["audio"] for f in feats]
        fe = self.p.feature_extractor(audios, sampling_rate=SR, return_tensors="pt", padding=True,
                                      return_attention_mask=True)
        lab = self.p.tokenizer([f["text"] for f in feats], padding=True, return_tensors="pt")
        labels = lab.input_ids.masked_fill(lab.attention_mask.ne(1), -100)
        key = "input_features" if self.is_bert else "input_values"
        return {key: fe[key], "attention_mask": fe["attention_mask"], "labels": labels}


# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/w2v-bert-2.0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", action="store_true", help="MMS-1B: train only a shared adapter + CTC head")
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--min-sec", type=float, default=0.5)
    ap.add_argument("--text", choices=["raw", "norm"], default="raw",
                    help="raw = case+punctuation kept (recommended, the scorer counts them); norm = lower-case without punctuation")
    ap.add_argument("--init-vocab", default=None, help="reuse a checkpoint's vocab.json (continued fine-tuning)")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--mask-time-prob", type=float, default=0.05)
    ap.add_argument("--layerdrop", type=float, default=0.0)
    ap.add_argument("--eval-n", type=int, default=200, help="validation clips per language for in-training evaluation")
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--train-frac", type=float, default=1.0, help="train subsample fraction (quick trials)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--langs", nargs="+", default=LANGS, help="training/eval languages (e.g. lin sna)")
    ap.add_argument("--lora", type=int, default=0, help="LoRA rank (0 = full fine-tuning); frozen base, CTC head + adapter trained")
    ap.add_argument("--max-minutes", type=float, default=0, help="safety stop after N minutes (0 = unlimited)")
    a = ap.parse_args()
    set_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from transformers import (AutoConfig, AutoFeatureExtractor, AutoModelForCTC, Trainer, TrainingArguments,
                              Wav2Vec2CTCTokenizer, Wav2Vec2Processor)

    tr, _ = load_cached("train", a.langs)
    va, _ = load_cached("validation", a.langs)
    fn = clean_raw if a.text == "raw" else normalize
    for d in (tr, va):
        d["norm"] = d.text.fillna("").map(fn)
    keep = (tr.length >= a.min_sec * SR) & (tr.length <= a.max_sec * SR) & (tr.norm.str.len() > 0)
    print(f"train: {keep.sum()}/{len(tr)} clips kept ({tr[keep].length.sum() / SR / 3600:.1f} h)")
    tr = tr[keep]
    if a.train_frac < 1:
        tr = tr.sample(frac=a.train_frac, random_state=a.seed)
    va_eval = pd.concat([g.sample(min(a.eval_n, len(g)), random_state=a.seed)
                         for _, g in va[va.norm.str.len() > 0].groupby("language")])
    print(tr.groupby("language").size().to_dict(), "| eval", va_eval.groupby("language").size().to_dict())

    if a.init_vocab:  # continuation: processor AND head size taken from the checkpoint as-is (added tokens included)
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(a.model)
        tok, fe = proc.tokenizer, proc.feature_extractor
        vocab = tok.get_vocab()
        n_out = AutoConfig.from_pretrained(a.model).vocab_size
        assert len(tok) == n_out, (len(tok), n_out)
        is_bert = "bert" in a.model.lower() or type(fe).__name__.startswith("SeamlessM4T")
    else:
        vocab = build_vocab(tr.norm.tolist(), out / "vocab.json")
        n_out = len(vocab)
        tok = Wav2Vec2CTCTokenizer(str(out / "vocab.json"), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        fe = AutoFeatureExtractor.from_pretrained(a.model)
        is_bert = "bert" in a.model.lower() or type(fe).__name__.startswith("SeamlessM4T")
        if is_bert:
            from transformers import Wav2Vec2BertProcessor
            proc = Wav2Vec2BertProcessor(feature_extractor=fe, tokenizer=tok)
        else:
            fe.return_attention_mask = True
            proc = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)
    proc.save_pretrained(out)

    kw = dict(vocab_size=n_out, pad_token_id=tok.pad_token_id, ctc_loss_reduction="mean",
              ctc_zero_infinity=True, mask_time_prob=a.mask_time_prob, layerdrop=a.layerdrop,
              attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, final_dropout=0.0)
    if is_bert:
        kw["add_adapter"] = True
    if a.adapter:
        kw["target_lang"] = None
        model = AutoModelForCTC.from_pretrained(a.model, ignore_mismatched_sizes=True, **kw)
        model.init_adapter_layers()
        model.freeze_base_model()
        for n, p in model.named_parameters():
            if "adapter" in n or "lm_head" in n:
                p.requires_grad = True
    else:
        model = AutoModelForCTC.from_pretrained(a.model, ignore_mismatched_sizes=True, **kw)
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
    if a.lora:
        from peft import LoraConfig, get_peft_model
        for p in model.parameters():
            p.requires_grad = False
        cfg = LoraConfig(r=a.lora, lora_alpha=2 * a.lora, lora_dropout=0.05,
                         target_modules=["linear_q", "linear_k", "linear_v", "linear_out", "intermediate_dense", "output_dense"],
                         modules_to_save=["lm_head"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {ntrain / 1e6:.1f} M")

    def compute_metrics(pred):
        ids = np.argmax(pred.predictions, axis=-1)
        hyps = proc.batch_decode(ids)
        lab = pred.label_ids.copy(); lab[lab == -100] = tok.pad_token_id
        refs = proc.batch_decode(lab, group_tokens=False)
        res = {"all_" + k: v for k, v in score(refs, hyps).items()}
        langs = va_eval.language.tolist()
        for l in a.langs:
            sel = [i for i, x in enumerate(langs) if x == l]
            if sel:
                res.update({f"{l}_{k}": v for k, v in score([refs[i] for i in sel], [hyps[i] for i in sel]).items()})
        return res

    args = TrainingArguments(
        output_dir=str(out), seed=a.seed, data_seed=a.seed,
        per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs, gradient_accumulation_steps=a.accum,
        num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr, warmup_steps=0.1,  # transformers v5: float = fraction
        lr_scheduler_type="linear", weight_decay=0.0,
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=not a.adapter, gradient_checkpointing_kwargs={"use_reentrant": False}, eval_strategy="no" if a.lora else "steps", eval_steps=a.eval_steps,
        save_strategy="no" if a.lora else "steps", save_steps=a.eval_steps, save_total_limit=2, logging_steps=10, load_best_model_at_end=not a.lora,
        metric_for_best_model="all_score", greater_is_better=True, report_to="none",
        dataloader_num_workers=a.workers, dataloader_persistent_workers=a.workers > 0,
        eval_accumulation_steps=16, remove_unused_columns=False,
    )
    train_ds = CachedASR(tr, "train", augment=a.augment)

    class BucketTrainer(Trainer):
        def _get_train_sampler(self, *args, **kwargs):
            return LengthBucketSampler(tr.length.values, a.bs, a.seed)

    import time
    from transformers import TrainerCallback

    class TimeLimit(TrainerCallback):
        def __init__(self, minutes):
            self.t0, self.limit = time.time(), minutes * 60

        def on_step_end(self, args, state, control, **kw):
            if self.limit and time.time() - self.t0 > self.limit:
                control.should_training_stop = True
            return control

    trainer = BucketTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=CachedASR(va_eval, "validation"),
                            data_collator=Collator(proc, is_bert), compute_metrics=compute_metrics,
                            processing_class=proc, callbacks=[TimeLimit(a.max_minutes)])
    trainer.train(resume_from_checkpoint=a.resume or None)
    if a.lora:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(out / "merged"); proc.save_pretrained(out / "merged")
        print("merged model ->", out / "merged")
    else:
        trainer.save_model(out / "best"); proc.save_pretrained(out / "best")
    if not a.lora:  # LoRA: evaluation outside the Trainer (eval_model.py on the merged model)
        json.dump(trainer.evaluate(), open(out / "best_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()

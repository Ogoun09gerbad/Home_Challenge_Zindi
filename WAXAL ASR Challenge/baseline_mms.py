"""Baseline zéro-shot MMS-1B-all (adapter par langue, langue oracle) sur la validation.
Usage : python baseline_mms.py [--n 300] -> logs/baseline_mms_val.csv + tableau WER/CER/score."""
import argparse, time

import pandas as pd
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

from common import LANGS, ROOT, SR, decode_audio, load_split, normalize, score_by_lang, set_seed


def transcribe(model, proc, audios, device, bs=8):
    out = []
    order = sorted(range(len(audios)), key=lambda i: len(audios[i]))
    res = [None] * len(audios)
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        inp = proc([audios[i] for i in idx], sampling_rate=SR, return_tensors="pt", padding=True)
        am = inp.get("attention_mask")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(inp.input_values.to(device, torch.float16),
                           attention_mask=None if am is None else am.to(device)).logits
        for i, t in zip(idx, proc.batch_decode(logits.argmax(-1))):
            res[i] = t
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="exemples de validation par langue (0 = tous)")
    ap.add_argument("--model", default="facebook/mms-1b-all")
    args = ap.parse_args()
    set_seed(42)
    device = "cuda"
    proc = AutoProcessor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
    val = load_split("validation", with_audio=True)
    parts = []
    for lang in LANGS:
        g = val[val.language == lang]
        if g.empty:
            print(f"{lang}: pas de validation téléchargée"); continue
        if args.n:
            g = g.sample(min(args.n, len(g)), random_state=42)
        proc.tokenizer.set_target_lang(lang); model.load_adapter(lang)
        t0 = time.time()
        g = g.assign(hyp=transcribe(model, proc, [decode_audio(b) for b in g.audio_bytes], device))
        print(f"{lang}: {len(g)} clips en {time.time() - t0:.0f}s")
        parts.append(g.drop(columns="audio_bytes"))
    df = pd.concat(parts)
    df.to_csv(ROOT / "logs" / "baseline_mms_val.csv", index=False)
    print("\n== brut (texte de référence non normalisé, hypothèse brute) ==")
    print(score_by_lang(df).round(4).to_markdown())
    dn = df.assign(text=df.text.fillna("").map(normalize), hyp=df.hyp.fillna("").map(normalize))
    print("\n== normalisé (normalize() des deux côtés) ==")
    print(score_by_lang(dn).round(4).to_markdown())


if __name__ == "__main__":
    main()

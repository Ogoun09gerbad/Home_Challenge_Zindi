"""Whisper large-v3 + adaptateur LoRA (mrwalker123/whisper-large-v3-waxal-lin-sna) -> hypothèses par clip.
La langue (ln/sn) est celle prédite par la LID textuelle sur l'ensemble CTC (--lid-from), jamais une métadonnée.
Les clips > 30 s sont coupés au point d'énergie minimale près du milieu, transcrits puis recollés.
Usage :
  python whisper_infer.py --split validation --n 300 --lid-from logs/mbr_F_clean_small_val.csv --tag w_swa
  python whisper_infer.py --split phase2 --lid-from logs/mbr_F_phase2.csv --tag w_swa
Sortie : logs/{tag}_{split}[_lin-sna].csv (id, language, ref, hyp)."""
import argparse, time

import numpy as np
import pandas as pd
import torch

from cache_audio import get_audio, load_cached, load_phase2
from common import ROOT, SR, clean_raw, score_by_lang, set_seed

REPO = "mrwalker123/whisper-large-v3-waxal-lin-sna"
LANG = {"lin": "ln", "sna": "sn"}


def split_long(y, max_s=29.5):
    if len(y) <= max_s * SR:
        return [y]
    mid = len(y) // 2
    w = int(3 * SR)
    seg = np.abs(y[mid - w:mid + w])
    k = np.convolve(seg, np.ones(int(0.1 * SR)) / int(0.1 * SR), mode="same").argmin()
    cut = mid - w + int(k)
    return split_long(y[:cut], max_s) + split_long(y[cut:], max_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--adapter", default="swa")
    ap.add_argument("--lid-from", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--beams", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    set_seed(42)
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if a.split == "phase2":
        df, mms = load_phase2()
    else:
        df, mms = load_cached(a.split, ["lin", "sna"])
        df = pd.concat([g.sample(min(a.n, len(g)), random_state=42) for _, g in df.groupby("language")])
    df = df.reset_index(drop=True)
    lid = pd.read_csv(a.lid_from, keep_default_na=False).set_index("id").lid
    df["lid"] = [lid.get(i, "lin") for i in df.id]

    proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16)
    from huggingface_hub import snapshot_download
    local = snapshot_download(REPO, allow_patterns=[f"{a.adapter}/adapter_*"])  # peft ne gère pas subfolder ici
    model = PeftModel.from_pretrained(model, f"{local}/{a.adapter}").merge_and_unload().to("cuda").eval()

    # une entrée = un segment ; on regroupe par langue pour fixer le jeton de langue
    segs = []  # (row, part, audio, lid)
    for r in df.itertuples():
        for p, y in enumerate(split_long(get_audio(r, mms))):
            segs.append((r.Index, p, y, r.lid))
    out = {}
    t0 = time.time()
    for lg in ("lin", "sna"):
        S = sorted([s for s in segs if s[3] == lg], key=lambda s: len(s[2]))
        for k in range(0, len(S), a.bs):
            B = S[k:k + a.bs]
            feats = proc.feature_extractor([b[2] for b in B], sampling_rate=SR, return_tensors="pt").input_features
            with torch.inference_mode():
                ids = model.generate(feats.to("cuda", torch.float16), language=LANG[lg], task="transcribe",
                                     num_beams=a.beams, max_new_tokens=400)
            for b, t in zip(B, proc.batch_decode(ids, skip_special_tokens=True)):
                out[(b[0], b[1])] = t.strip()
            if (k // a.bs) % 25 == 0:
                print(f"{lg} {k}/{len(S)} {time.time() - t0:.0f}s", flush=True)
    hyp = []
    for i in range(len(df)):
        parts = [out[key] for key in sorted(k for k in out if k[0] == i)]
        hyp.append(clean_raw(" ".join(parts)))
    df["hyp"] = hyp
    df["ref"] = "" if a.split == "phase2" else df.text.fillna("").map(clean_raw)
    suffix = "" if a.split == "phase2" else "_lin-sna"
    res = df[["id", "language", "speaker_id", "ref", "hyp"]].assign(lid=df.lid)
    res.to_csv(ROOT / "logs" / f"{a.tag}_{a.split}{suffix}.csv", index=False)
    if a.split != "phase2":
        print(score_by_lang(res, ref_col="ref").round(4).to_markdown())
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

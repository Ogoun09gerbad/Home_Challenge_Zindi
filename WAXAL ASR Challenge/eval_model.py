"""Run one CTC model on a cached split, save its log-probs and greedy transcriptions, and score it when labels exist.

Usage:
  python eval_model.py --model anyantudre/waxal-w2vbert-linsna-soup5 --revision <commit> --tag m_linsna-soup5 --split phase2
  python eval_model.py --model ... --tag ... --split validation --n 300 --langs lin sna
Outputs: logits/{tag}_{split}.npz (float16 log-probs per clip id), logs/{tag}_{split}.csv (id, ..., ref, hyp)
and, on labelled splits, a WER / CER / score table per language."""
import argparse, time

import librosa
import numpy as np
import pandas as pd

from cache_audio import get_audio, load_cached, load_phase2
from common import LANGS, ROOT, clean_raw, score_by_lang, set_seed
from predict import ctc_logits, greedy_decode, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True, help="name used for the output files")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n", type=int, default=300, help="clips per language on labelled splits (0 = all)")
    parser.add_argument("--max-batch-sec", type=float, default=120)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--langs", nargs="+", default=LANGS)
    parser.add_argument("--speed", type=float, default=1.0, help="test-time speed perturbation (0.9 = slower); not used in the final submission")
    parser.add_argument("--revision", default=None, help="Hub commit of the checkpoint (reproducibility)")
    args = parser.parse_args()
    set_seed(42)
    if args.split == "phase2":
        df, mmaps = load_phase2()
    else:
        df, mmaps = load_cached(args.split, args.langs)
    if args.n and args.split != "phase2":
        df = pd.concat([g.sample(min(args.n, len(g)), random_state=42) for _, g in df.groupby("language")])
    df = df.reset_index(drop=True)
    model, proc = load_model(args.model, "cuda", args.revision)
    t0, logps = time.time(), []
    for k in range(0, len(df), args.chunk):
        audios = [get_audio(r, mmaps) for r in df.iloc[k:k + args.chunk].itertuples()]
        if args.speed != 1.0:
            audios = [librosa.resample(y, orig_sr=int(16000 * args.speed), target_sr=16000).astype(np.float32) for y in audios]
        logps += ctc_logits(model, proc, audios, "cuda", args.max_batch_sec)
        print(f"{min(k + args.chunk, len(df))}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    audio_hours = df.length.sum() / 16000 / 3600
    print(f"{audio_hours:.2f} h of audio in {time.time() - t0:.0f}s (RTF {(time.time() - t0) / (audio_hours * 3600):.3f})")
    suffix = "" if args.split == "phase2" or set(args.langs) == set(LANGS) else "_" + "-".join(args.langs)
    (ROOT / "logits").mkdir(exist_ok=True)
    np.savez_compressed(ROOT / "logits" / f"{args.tag}_{args.split}{suffix}.npz",
                        **{i: lp.astype(np.float16) for i, lp in zip(df.id, logps)})
    df["hyp"] = [clean_raw(h) for h in greedy_decode(proc, logps)]
    df["ref"] = "" if args.split in ("test", "phase2") else df.text.fillna("").map(clean_raw)  # test labels never read
    df[["id", "language", "speaker_id", "ref", "hyp"]].to_csv(ROOT / "logs" / f"{args.tag}_{args.split}{suffix}.csv", index=False)
    if args.split not in ("test", "phase2"):
        print(score_by_lang(df, ref_col="ref").round(4).to_markdown())


if __name__ == "__main__":
    main()

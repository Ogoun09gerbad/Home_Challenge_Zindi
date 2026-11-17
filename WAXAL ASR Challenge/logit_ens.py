"""Frame-level helpers on saved CTC log-probs: loading, blank-penalty greedy decoding, and (experiment, not in the
final submission) weighted averaging of log-probs from members sharing the same vocabulary.
Usage (experiment):
  python logit_ens.py grid --split validation_lin-sna --members m_lin-specialist m_linsna-soup5 ...
  python logit_ens.py apply --split phase2 --members ... --config logs/logit_ens_best.json --out logs/lens_phase2.csv"""
import argparse, itertools, json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoProcessor

from common import ROOT, clean_raw, score_by_lang

MODEL_OF = lambda tag: "anyantudre/waxal-w2vbert-" + tag[2:]  # "m_linsna-soup5" -> Hub repo id


def load(tags, split):
    """Load saved log-probs written by eval_model.py.

    Parameters:
        tags (list[str]): model tags, e.g. ["m_linsna-soup5"].
        split (str): split name used in the file names, e.g. "phase2".

    Returns:
        tuple: (DataFrame of the first tag's CSV, list of NpzFile mapping clip id -> log-prob matrix).
    """
    npz_files = [np.load(ROOT / "logits" / f"{t}_{split}.npz") for t in tags]
    df = pd.read_csv(ROOT / "logs" / f"{tags[0]}_{split}.csv", keep_default_na=False)
    return df, npz_files


def get_tokenizer(tag):
    """Return the CTC tokenizer of the Hub model behind a tag (all members share the same 121-token vocabulary)."""
    return AutoProcessor.from_pretrained(MODEL_OF(tag)).tokenizer


def combine(lps, w, mode):
    """Weighted average of log-prob matrices (truncated to the shortest), geometric ("logp") or arithmetic ("prob")."""
    n_frames = min(lp.shape[0] for lp in lps)
    stacked = np.stack([lp[:n_frames].astype(np.float32) for lp in lps])  # members x frames x vocabulary
    w = np.asarray(w, np.float32)[:, None, None] / sum(w)
    if mode == "logp":
        return (stacked * w).sum(0)
    probs = (np.exp(stacked) * w).sum(0)
    return np.log(probs + 1e-12)


def decode(tok, lp, blank_id, bp):
    """Greedy CTC decoding after subtracting a blank penalty.

    A positive penalty makes the blank token less likely, so the model emits more characters; this counters
    deletions on unseen speakers and gives diverse candidates for the MBR ensemble.

    Parameters:
        tok: CTC tokenizer.
        lp (np.ndarray): (frames x vocabulary) log-probs.
        blank_id (int): index of the CTC blank (the tokenizer's pad token).
        bp (float): blank penalty subtracted from the blank log-prob.

    Returns:
        str: cleaned transcription.
    """
    penalised = lp.copy(); penalised[:, blank_id] -= bp
    return clean_raw(tok.decode(penalised.argmax(-1)))


def run(df, npz_files, tok, w, mode, bp):
    """Decode the frame-level average of several members for every clip of `df`."""
    blank = tok.pad_token_id
    return [decode(tok, combine([z[i] for z in npz_files], w, mode), blank, bp) for i in df.id]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["grid", "apply"])
    parser.add_argument("--split", default="validation_lin-sna")
    parser.add_argument("--members", nargs="+", required=True)
    parser.add_argument("--config", default=str(ROOT / "logs" / "logit_ens_best.json"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    df, npz_files = load(args.members, args.split)
    tok = get_tokenizer(args.members[0])

    if args.mode == "grid":
        rows = []
        for mode, bp in itertools.product(["logp", "prob"], [0.0, 0.5, 1.0, 1.5, 2.0]):
            hyp = run(df, npz_files, tok, [1.0] * len(npz_files), mode, bp)
            s = score_by_lang(df.assign(hyp=hyp), ref_col="ref")
            rows.append((mode, bp, s.loc["lin", "score"], s.loc["sna", "score"], s.loc["ALL", "score"]))
            print(f"{mode:5s} blank_pen={bp:.1f}  lin={rows[-1][2]:.4f} sna={rows[-1][3]:.4f} ALL={rows[-1][4]:.4f}", flush=True)
        best = max(rows, key=lambda r: r[4])
        cfg = {"members": args.members, "mode": best[0], "blank_penalty": best[1], "weights": [1.0] * len(npz_files), "val_ALL": best[4]}
        Path(args.config).write_text(json.dumps(cfg, indent=1)); print("best:", cfg)
    else:
        cfg = json.loads(Path(args.config).read_text())
        assert cfg["members"] == args.members, "members differ from the config"
        df["hyp"] = run(df, npz_files, tok, cfg["weights"], cfg["mode"], cfg["blank_penalty"])
        df.to_csv(args.out, index=False); print("->", args.out, len(df))


if __name__ == "__main__":
    main()

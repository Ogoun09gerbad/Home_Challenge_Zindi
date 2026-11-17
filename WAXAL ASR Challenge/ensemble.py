"""Minimum Bayes Risk (MBR) selection ensemble.

For each clip, keep the candidate transcription with the lowest weighted average Zindi error
(0.5*WER + 0.5*CER) against all the other candidates: the "consensus" hypothesis. Needs no metadata.
Usage (unrouted, experiment):
  python ensemble.py --inputs logs/a_validation.csv logs/b_validation.csv ... [--weights 1 1 ...] --out logs/ens_val.csv
Each CSV needs columns id, hyp (and ref, language on labelled splits for scoring)."""
import argparse

import jiwer
import numpy as np
import pandas as pd

from common import score_by_lang


def pair_err(ref, hyp):
    """Zindi error of `hyp` measured against `ref` (another candidate, not a true reference).

    Parameters:
        ref (str): candidate used as pseudo-reference.
        hyp (str): candidate being scored.

    Returns:
        float: 0.5 * WER(lower-cased) + 0.5 * CER(raw); 0 or 1 when the pseudo-reference is empty.
    """
    if not ref.strip():
        return 0.0 if not hyp.strip() else 1.0
    return 0.5 * jiwer.wer(ref.lower(), hyp.lower()) + 0.5 * jiwer.cer(ref, hyp)


def mbr_select(hyps, weights):
    """Pick the MBR candidate.

    Parameters:
        hyps (list[str]): one candidate transcription per ensemble member.
        weights (list[float]): weight of each member when it acts as pseudo-reference.

    Returns:
        int: index of the candidate with the lowest expected error (first one on ties).
    """
    n = len(hyps)
    if n == 1:
        return 0
    risk = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                risk[i] += weights[j] * pair_err(hyps[j], hyps[i])
    return int(np.argmin(risk))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    weights = args.weights or [1.0] * len(args.inputs)
    dfs = [pd.read_csv(p, keep_default_na=False).set_index("id") for p in args.inputs]
    ids = dfs[0].index
    for path, d in zip(args.inputs, dfs):
        assert set(d.index) == set(ids), f"different IDs: {path}"
    base = dfs[0].copy()
    chosen = []
    for clip_id in ids:
        candidates = [d.at[clip_id, "hyp"] for d in dfs]
        k = mbr_select(candidates, weights); chosen.append(k)
        base.at[clip_id, "hyp"] = candidates[k]
    base["member"] = chosen
    base.reset_index().to_csv(args.out, index=False)
    print("chosen member:", pd.Series(chosen).value_counts().sort_index().to_dict())
    if "ref" in base and (base.ref != "").any():
        for path, d in zip(args.inputs, dfs):
            print(path, score_by_lang(d.reset_index(), ref_col="ref").loc["ALL", "score"].round(4))
        print("\nMBR:\n", score_by_lang(base.reset_index(), ref_col="ref").round(4).to_markdown())


if __name__ == "__main__":
    main()

"""Médiane de chaînes (niveau caractère) entre hypothèses : candidats = hypothèses + chaînes intermédiaires sur les
chemins d'édition de Levenshtein entre paires ; on garde le candidat minimisant sum_j w_j * Lev(c, h_j).
Même principe que la solution 1ère place (median_extended). Routage par LID textuelle comme mbr_routed.py.
Usage : python median_ens.py --split validation_lin-sna --config configs/F_clean_small.json --out logs/med_F_val.csv"""
import argparse, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein as L

from common import ROOT, score_by_lang
from decode_lm import train_lid


def intermediates(a, b, fracs=(0.25, 0.5, 0.75)):
    ops = L.editops(a, b)
    out = set()
    n = len(ops)
    if n < 2:
        return out
    for f in fracs:
        k = max(1, min(n - 1, round(f * n)))
        out.add(L.editops(a, b)[:k].apply(a, b))
    return out


def median_string(hyps, w, rounds=2):
    cands = set(hyps)
    for i in range(len(hyps)):
        for j in range(i + 1, len(hyps)):
            if hyps[i] != hyps[j]:
                cands |= intermediates(hyps[i], hyps[j]) | intermediates(hyps[j], hyps[i])
    def cost(c):
        return sum(wj * L.distance(c, h) for wj, h in zip(w, hyps))
    best = min(cands, key=lambda c: (cost(c), c not in hyps))
    for _ in range(rounds - 1):  # raffinement : intermédiaires entre le meilleur et chaque hypothèse
        new = set()
        for h in hyps:
            if h != best:
                new |= intermediates(best, h)
        if not new:
            break
        b2 = min(new | {best}, key=cost)
        if cost(b2) >= cost(best):
            break
        best = b2
    return best


def _job(args):
    return median_string(*args)


def run(split, cfg, workers=16):
    names = sorted({n for l in ("lin", "sna") for n in cfg[l]})
    dfs = {n: pd.read_csv(ROOT / "logs" / f"{n}_{split}.csv", keep_default_na=False).set_index("id") for n in names}
    base = dfs[names[0]].copy()
    lid = train_lid()
    langs = [lid.predict(dfs[cfg.get("pivot", names[0])].at[i, "hyp"]) for i in base.index]
    langs = [l if l in ("lin", "sna") else "lin" for l in langs]
    W = cfg.get("weights", {})
    jobs = [([dfs[n].at[i, "hyp"] for n in cfg[l]], [W.get(n, 1.0) for n in cfg[l]]) for i, l in zip(base.index, langs)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_job, jobs, chunksize=8))
    base["hyp"], base["lid"] = res, langs
    return base.reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = run(a.split, json.loads(Path(a.config).read_text()))
    df.to_csv(a.out, index=False)
    print("LID :", df.lid.value_counts().to_dict())
    if (df.get("ref", pd.Series([""])) != "").any():
        print(score_by_lang(df, ref_col="ref").round(4).to_markdown())

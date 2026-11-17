"""MBR ensemble routed by text language ID.

The language of each clip is predicted from the pivot candidate's transcription (no metadata is used), then MBR
selection runs among the candidates configured for that language.
Usage:
  python mbr_routed.py --split phase2 --config configs/F_clean_small.json --out logs/mbr_F_phase2.csv
config: {"lin": ["v_m_lin-specialist_bp0", ...], "sna": [...], "pivot": "v_m_...", "weights": {"v_m_...": 1.5}}
Each name refers to logs/{name}_{split}.csv written by make_variants.py."""
import argparse, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from common import ROOT, score_by_lang
from ensemble import mbr_select
from lid import train_lid


def _select(job):
    """Worker wrapper around mbr_select for a (candidates, weights) tuple."""
    hyps, weights = job
    return mbr_select(hyps, weights)


def run(split, cfg, workers=16):
    """Run the routed MBR ensemble.

    Parameters:
        split (str): split name in the candidate file names, e.g. "phase2".
        cfg (dict): candidates per language ("lin", "sna"), optional "pivot" and "weights".
        workers (int): CPU processes for the pairwise WER/CER computations.

    Returns:
        pd.DataFrame: one row per clip with columns id, ..., hyp (selected), lid (predicted language),
        member (name of the selected candidate).
    """
    names = sorted({n for lang in ("lin", "sna") for n in cfg[lang]})
    dfs = {n: pd.read_csv(ROOT / "logs" / f"{n}_{split}.csv", keep_default_na=False).set_index("id") for n in names}
    base = dfs[names[0]].copy()
    lid = train_lid()
    pivot = cfg.get("pivot", names[0])
    langs = [lid.predict(dfs[pivot].at[i, "hyp"]) for i in base.index]
    langs = [lang if lang in ("lin", "sna") else "lin" for lang in langs]  # no Luganda in Phase 2
    weights = cfg.get("weights", {})
    jobs, candidates = [], []
    for clip_id, lang in zip(base.index, langs):
        names_for_lang = cfg[lang]
        candidates.append(names_for_lang)
        jobs.append(([dfs[n].at[clip_id, "hyp"] for n in names_for_lang], [weights.get(n, 1.0) for n in names_for_lang]))
    with ProcessPoolExecutor(workers) as ex:
        chosen = list(ex.map(_select, jobs, chunksize=8))
    base["hyp"] = [job[0][k] for job, k in zip(jobs, chosen)]
    base["lid"] = langs
    base["member"] = [c[k] for c, k in zip(candidates, chosen)]
    return base.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    df = run(args.split, cfg)
    df.to_csv(args.out, index=False)
    print("LID:", df.lid.value_counts().to_dict())
    print("selected members:", df.member.value_counts().to_dict())
    if (df.get("ref", pd.Series([""])) != "").any():
        if "language" in df and (df.language != "").any():
            print("LID accuracy:", round((df.lid == df.language).mean(), 4))
        print(score_by_lang(df, ref_col="ref").round(4).to_markdown())


if __name__ == "__main__":
    main()

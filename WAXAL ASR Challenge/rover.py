"""Vote ROVER au niveau caractère, ancré sur un squelette (algorithme d'anyantudre, 2e du WAXAL original,
dépôt MIT, réimplémenté) : chaque membre est aligné sur le squelette (Levenshtein.opcodes) ; pour chaque
caractère et chaque emplacement d'insertion, le contenu le plus pondéré remplace celui de l'ancre s'il en
diffère et totalise >= seuil. Contrairement au MBR, deux membres d'accord peuvent ajouter un mot manquant.
Usage :
  python rover.py --split validation_lin-sna --config configs/rover_G.json --out logs/rover_G_val.csv
  python rover.py --grid   (grille sur la validation)
config : {"members": [...], "anchor": "v_m_...", "threshold": 2.0, "skeleton": "anchor"|"mbr",
          "weights": {...}, "by_lang": {"lin": {"members": [...], "anchor": ...}, "sna": {...}}}"""
import argparse, itertools, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

from common import ROOT, score_by_lang
from postproc import cap_after_stop, collapse_loops, first_cap

CLEAN = ["m_linsna-soup5", "m_lin-specialist", "m_sna-specialist", "m_linsna-afrivoicemix", "m_linsnalug-raw"]


def _align(skel, other):
    n = len(skel)
    cols, ins = list(skel), [""] * (n + 1)
    for op, i1, i2, j1, j2 in Levenshtein.opcodes(skel, other):
        if op == "replace":
            k = min(i2 - i1, j2 - j1)
            for t in range(k):
                cols[i1 + t] = other[j1 + t]
            for t in range(i1 + k, i2):
                cols[t] = ""
            if j2 - j1 > k:
                cols[i1 + k - 1] = other[j1 + k - 1:j2]
        elif op == "delete":
            for t in range(i1, i2):
                cols[t] = ""
        elif op == "insert":
            ins[i1] += other[j1:j2]
    return cols, ins


def _win(tally, cand, cur, wc, margin):
    if cand == cur:
        return False
    if margin is not None:
        return tally[cand] - tally.get(cur, 0.0) >= margin
    return tally[cand] >= wc


def rover(hyps, weights, wc=2.0, skeleton="anchor", margin=None):
    """hyps[0] = ancre. Renvoie la chaîne votée. margin : remplace seulement si poids(candidat) - poids(contenu de l'ancre) >= margin."""
    keep = [(h, w) for h, w in zip(hyps, weights) if h.strip()]
    if not hyps[0].strip() or len(keep) < 2:
        return hyps[0]
    hs, ws = [h for h, _ in keep], [w for _, w in keep]
    if skeleton == "mbr":
        d = [sum(Levenshtein.distance(a, b) for b in hs) for a in hs]
        skel = hs[d.index(min(d))]
    else:
        skel = hs[0]
    al = [_align(skel, h) for h in hs]
    a_cols, a_ins = al[0]
    out = []
    for p in range(len(skel) + 1):
        tally = {}
        for (c, i), w in zip(al, ws):
            tally[i[p]] = tally.get(i[p], 0.0) + w
        cand = max(tally, key=lambda s: (tally[s], s != ""))
        out.append(cand if _win(tally, cand, a_ins[p], wc, margin) else a_ins[p])
        if p < len(skel):
            tally = {}
            for (c, i), w in zip(al, ws):
                tally[c[p]] = tally.get(c[p], 0.0) + w
            cand = max(tally, key=tally.get)
            out.append(cand if _win(tally, cand, a_cols[p], wc, margin) else a_cols[p])
    res = " ".join("".join(out).split())
    return res if res else hyps[0]


def _job(args):
    hyps, ws, wc, sk, mg = args
    return rover(hyps, ws, wc, sk, mg)


_CACHE = {}


def load(name, split):
    k = (name, split)
    if k not in _CACHE:
        df = pd.read_csv(ROOT / "logs" / f"{name}_{split}.csv", keep_default_na=False).set_index("id")
        df["hyp"] = df.hyp.map(collapse_loops)
        _CACHE[k] = df
    return _CACHE[k]


def lid_for(split):
    p = ROOT / "logs" / f"lid_{split}.csv"
    if p.exists():
        return pd.read_csv(p, keep_default_na=False).set_index("id").lid
    from decode_lm import train_lid
    lid = train_lid()
    base = load("v_m_linsna-soup5_bp0", split)
    s = pd.Series([lid.predict(h) for h in base.hyp], index=base.index, name="lid")
    s = s.where(s.isin(["lin", "sna"]), "lin")
    s.reset_index().to_csv(p, index=False)
    return s


def run(split, cfg, workers=16):
    W = cfg.get("weights", {})
    wc, sk, mg = cfg.get("threshold", 2.0), cfg.get("skeleton", "anchor"), cfg.get("margin")
    by_lang = cfg.get("by_lang")
    base = load(cfg.get("anchor") or by_lang["lin"]["anchor"], split).copy()
    langs = lid_for(split).reindex(base.index) if by_lang else None
    jobs = []
    for idx in base.index:
        c = by_lang[langs[idx]] if by_lang else cfg
        names = [c["anchor"]] + [m for m in c["members"] if m != c["anchor"]]
        jobs.append(([load(n, split).at[idx, "hyp"] for n in names], [W.get(n, 1.0) for n in names], wc, sk, mg))
    with ProcessPoolExecutor(workers) as ex:
        voted = list(ex.map(_job, jobs, chunksize=8))
    base["hyp"] = [first_cap(cap_after_stop(v)) for v in voted]
    if langs is not None:
        base["lid"] = langs
    return base.reset_index()


def members(tags, bps):
    return [f"v_{t}_bp{b:g}" for t in tags for b in bps]


def grid():
    split = "validation_lin-sna"
    A = "v_m_linsna-soup5_bp0"
    m4 = members(CLEAN, [0, 1, 1.5, 2])
    m5 = members(CLEAN, [0, 0.5, 1, 1.5, 2])
    m3 = members(CLEAN, [0, 1, 2])
    lin_t = [t for t in CLEAN if t != "m_sna-specialist"]
    sna_t = [t for t in CLEAN if t != "m_lin-specialist"]
    cfgs = {}
    for wc, sk in itertools.product([2, 3, 4, 5], ["anchor", "mbr"]):
        cfgs[f"all20_wc{wc}_{sk}"] = {"members": m4, "anchor": A, "threshold": wc, "skeleton": sk}
    for wc in [3, 4, 5, 6]:
        cfgs[f"all25_wc{wc}"] = {"members": m5, "anchor": A, "threshold": wc}
    for wc in [2, 3]:
        cfgs[f"all15_wc{wc}"] = {"members": m3, "anchor": A, "threshold": wc}
    for wc in [2, 3, 4]:
        cfgs[f"lid16_wc{wc}"] = {"threshold": wc, "by_lang": {
            "lin": {"members": members(lin_t, [0, 1, 1.5, 2]), "anchor": "v_m_lin-specialist_bp0"},
            "sna": {"members": members(sna_t, [0, 1, 1.5, 2]), "anchor": "v_m_linsna-soup5_bp0"}}}
    rows = []
    for k, c in cfgs.items():
        s = score_by_lang(run(split, c), ref_col="ref")
        rows.append({"cfg": k, "lin": s.loc["lin", "score"], "sna": s.loc["sna", "score"], "ALL": s.loc["ALL", "score"],
                     "wer": s.loc["ALL", "wer"], "cer": s.loc["ALL", "cer"]})
        print(f"{k:22s} lin={rows[-1]['lin']:.4f} sna={rows[-1]['sna']:.4f} ALL={rows[-1]['ALL']:.4f}", flush=True)
    pd.DataFrame(rows).sort_values("ALL", ascending=False).to_csv(ROOT / "logs" / "rover_grid.csv", index=False)
    Path(ROOT / "configs" / "rover_grid.json").write_text(json.dumps(cfgs, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--split")
    ap.add_argument("--config")
    ap.add_argument("--name", help="clé dans configs/rover_grid.json (au lieu de --config)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.grid:
        return grid()
    cfg = json.loads(Path(a.config).read_text()) if a.config else json.loads((ROOT / "configs" / "rover_grid.json").read_text())[a.name]
    df = run(a.split, cfg)
    df.to_csv(a.out, index=False)
    print("->", a.out)
    if (df.get("ref", pd.Series([""])) != "").any():
        print(score_by_lang(df, ref_col="ref").round(4).to_markdown())


if __name__ == "__main__":
    main()

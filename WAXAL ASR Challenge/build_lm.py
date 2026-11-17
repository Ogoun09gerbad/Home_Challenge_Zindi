"""Word n-gram LM (interpolated Kneser-Ney) written in ARPA format, in pure Python (no need for lmplz).
Text = normalised transcriptions of the train split (never validation/test). One LM per language + a shared one.
Usage: python build_lm.py --order 4 -> lm/{lin,sna,lug,all}.arpa (+ .unigrams.txt for pyctcdecode)"""
import argparse, math
from collections import Counter, defaultdict
from pathlib import Path

from common import LANGS, ROOT, load_split, normalize

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


def discount(counts):
    coc = Counter(v for v in counts.values() if v <= 2)
    n1, n2 = coc[1], coc[2]
    return min(max(n1 / (n1 + 2 * n2), 0.1), 0.9) if n1 + n2 else 0.5


def build_arpa(sentences, order, path):
    raw = [Counter() for _ in range(order + 1)]
    for s in sentences:
        toks = [BOS] + s.split() + [EOS]
        for k in range(1, order + 1):
            for i in range(len(toks) - k + 1):
                raw[k][tuple(toks[i:i + k])] += 1
    # adjusted counts: continuation (number of distinct left neighbours) except for max order or n-grams starting with <s>
    adj = [None] * (order + 1)
    adj[order] = raw[order]
    for k in range(order - 1, 0, -1):
        cont = Counter(g[1:] for g in raw[k + 1])
        adj[k] = Counter({g: (c if g[0] == BOS else cont.get(g, 0)) for g, c in raw[k].items()})
        adj[k] = Counter({g: c for g, c in adj[k].items() if c > 0 or g == (BOS,)})
    vocab = [g[0] for g in raw[1]]
    prob, gamma = [dict() for _ in range(order + 1)], [dict() for _ in range(order + 1)]
    # unigrams interpolated with the uniform distribution
    uni = {g: c for g, c in adj[1].items() if g != (BOS,)}  # <s> is never predicted
    D = discount(uni); tot = sum(uni.values()); V = len(uni) + 1
    g0 = D * len(uni) / tot
    for g, c in uni.items():
        prob[1][g] = max(c - D, 0) / tot + g0 / V
    prob[1][(UNK,)] = g0 / V
    for k in range(2, order + 1):
        D = discount(adj[k])
        ctx_tot, ctx_n = defaultdict(int), defaultdict(int)
        for g, c in adj[k].items():
            ctx_tot[g[:-1]] += c; ctx_n[g[:-1]] += 1
        for h in ctx_tot:
            gamma[k - 1][h] = D * ctx_n[h] / ctx_tot[h]
        for g, c in adj[k].items():
            h = g[:-1]
            lower = g[1:]
            pl = prob[k - 1].get(lower) or backoff_prob(lower, prob, gamma)
            prob[k][g] = max(c - D, 0) / ctx_tot[h] + gamma[k - 1][h] * pl
    prob[1][(BOS,)] = None
    lines = ["", "\\data\\"] + [f"ngram {k}={len(prob[k])}" for k in range(1, order + 1)] + [""]
    for k in range(1, order + 1):
        lines.append(f"\\{k}-grams:")
        for g, p in prob[k].items():
            lp = -99.0 if p is None else math.log10(p)
            bo = gamma[k].get(g) if k < order else None
            lines.append(f"{lp:.6f}\t{' '.join(g)}" + (f"\t{math.log10(bo):.6f}" if bo else ""))
        lines.append("")
    lines.append("\\end\\")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(str(path).replace(".arpa", ".unigrams.txt")).write_text(
        "\n".join(w for w in vocab if w not in (BOS, EOS)), encoding="utf-8")
    return {k: len(prob[k]) for k in range(1, order + 1)}


def backoff_prob(g, prob, gamma):
    """p(w|h) via standard ARPA backoff."""
    k = len(g)
    if k == 1:
        return prob[1].get(g) or prob[1][(UNK,)]
    if g in prob[k]:
        return prob[k][g]
    return gamma[k - 1].get(g[:-1], 1.0) * backoff_prob(g[1:], prob, gamma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "lm"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    tr = load_split("train")
    tr["norm"] = tr.text.fillna("").map(normalize)
    tr = tr[tr.norm.str.len() > 0]
    for lang in LANGS + ["all"]:
        sents = tr.norm.tolist() if lang == "all" else tr[tr.language == lang].norm.tolist()
        if not sents:
            continue
        n = build_arpa(sents, a.order, out / f"{lang}.arpa")
        print(lang, len(sents), "sentences ->", n)


if __name__ == "__main__":
    main()

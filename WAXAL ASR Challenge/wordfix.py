"""Réparation des frontières de mots par fréquences du train (aucun label de test/Phase 2).
- split : un token w (absent ou rare dans le train) devient "a b" si w = a+b et que le bigramme "a b"
  est >= ratio fois plus fréquent que w (les deux moitiés étant des mots connus du train).
- merge : deux tokens "a b" deviennent "ab" si "ab" est >= ratio fois plus fréquent que le bigramme "a b".
Comparaison en minuscules sans ponctuation ; la casse et la ponctuation finale du token sont conservées.
Usage : python wordfix.py --pred logs/mbr_E_routed_small_val.csv [--ratio 3 --min-count 5] [--out ...]"""
import argparse, re
from collections import Counter

import pandas as pd

from common import load_split, score_by_lang

_TOK = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)


def build_counts(lang):
    tr = load_split("train", langs=[lang])
    uni, bi = Counter(), Counter()
    for t in tr.text.fillna(""):
        ws = [re.sub(r"[^\w']", "", w.lower()) for w in t.split()]
        ws = [w for w in ws if w]
        uni.update(ws); bi.update(zip(ws, ws[1:]))
    return uni, bi


class WordFix:
    def __init__(self, lang, ratio=3.0, min_count=5, do_split=True, do_merge=True):
        self.uni, self.bi = build_counts(lang)
        self.ratio, self.min_count, self.do_split, self.do_merge = ratio, min_count, do_split, do_merge

    def _split(self, core):
        w = core.lower()
        best, bestc = None, 0
        for k in range(1, len(w)):
            a, b = w[:k], w[k:]
            c = self.bi.get((a, b), 0)
            if c > bestc and self.uni.get(a, 0) and self.uni.get(b, 0):
                best, bestc = k, c
        if best and bestc >= self.min_count and bestc >= self.ratio * max(self.uni.get(w, 0), 1):
            return core[:best] + " " + core[best:]
        return core

    def __call__(self, text):
        toks = text.split()
        if self.do_split:
            out = []
            for t in toks:
                pre, core, suf = _TOK.match(t).groups()
                out.append(pre + (self._split(core) if core else core) + suf)
            toks = " ".join(out).split()
        if self.do_merge:
            out, i = [], 0
            while i < len(toks):
                if i + 1 < len(toks):
                    p1, a, s1 = _TOK.match(toks[i]).groups()
                    p2, b, s2 = _TOK.match(toks[i + 1]).groups()
                    if a and b and not s1 and not p2:
                        j, cbi = (a + b).lower(), self.bi.get((a.lower(), b.lower()), 0)
                        if self.uni.get(j, 0) >= self.min_count and self.uni[j] >= self.ratio * max(cbi, 1):
                            out.append(p1 + a + b + s2); i += 2; continue
                out.append(toks[i]); i += 1
            toks = out
        return " ".join(toks)


def apply(df, lang_col, **kw):
    fixers = {l: WordFix(l, **kw) for l in ("lin", "sna")}
    return [fixers[l](h) if l in fixers else h for h, l in zip(df.hyp, df[lang_col])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--lang-col", default="lid")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ratio", type=float, default=None)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--mode", default="both", choices=["split", "merge", "both"])
    a = ap.parse_args()
    df = pd.read_csv(a.pred, keep_default_na=False)
    if a.ratio is None:  # grille sur la validation
        base = score_by_lang(df, ref_col="ref")
        print(f"base            lin={base.loc['lin','score']:.4f} sna={base.loc['sna','score']:.4f} ALL={base.loc['ALL','score']:.4f}")
        for mode in ["split", "merge", "both"]:
            for r in [2, 3, 5, 10]:
                for mc in [3, 10]:
                    h = apply(df, a.lang_col, ratio=r, min_count=mc, do_split=mode != "merge", do_merge=mode != "split")
                    s = score_by_lang(df.assign(hyp=h), ref_col="ref")
                    print(f"{mode:5s} r={r:<3} mc={mc:<3} lin={s.loc['lin','score']:.4f} sna={s.loc['sna','score']:.4f} ALL={s.loc['ALL','score']:.4f}", flush=True)
    else:
        df["hyp_raw"] = df.hyp
        df["hyp"] = apply(df, a.lang_col, ratio=a.ratio, min_count=a.min_count,
                          do_split=a.mode != "merge", do_merge=a.mode != "split")
        print("clips modifiés :", (df.hyp != df.hyp_raw).sum())
        df.to_csv(a.out, index=False); print("->", a.out)


if __name__ == "__main__":
    main()

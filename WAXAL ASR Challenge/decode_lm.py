"""CTC decoding with an n-gram LM (pyctcdecode + kenlm) and text LID, from saved log-probs.
- LID: Naive Bayes on character n-grams (1-4), trained on train transcriptions, applied to the greedy hypothesis.
- LM: one ARPA per language (build_lm.py), words normalised (lower-case, no punctuation) before the kenlm query;
  the acoustic model keeps case and punctuation.
Usage:
  python decode_lm.py grid --tag anyraw --model anyantudre/waxal-w2vbert-linsnalug-raw      # per-language alpha/beta tuning on validation
  python decode_lm.py decode --tag anyraw --split test --model ... --params lm/params_anyraw.json --out logs/anyraw_test_lm.csv
"""
import argparse, itertools, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common import LANGS, ROOT, clean_raw, normalize, score, score_by_lang
from lid import CharNB, train_lid  # noqa: F401  (text LID, moved to lid.py; re-exported for older scripts)

LM_DIR = ROOT / "lm"


# ---- decoder -----------------------------------------------------------------------
def labels_from_processor(model_path):
    from transformers import AutoProcessor
    tok = AutoProcessor.from_pretrained(model_path).tokenizer
    inv = {i: t for t, i in tok.get_vocab().items()}
    labels = []
    for i in range(len(inv)):
        t = inv.get(i)
        if t == tok.word_delimiter_token:
            labels.append(" ")
        elif t == tok.pad_token:
            labels.append("")  # blank CTC
        elif t is None or t == tok.unk_token or (t.startswith(("<", "[")) and len(t) > 1):
            labels.append(chr(0xE000 + i))  # special tokens: unique private-use character, stripped from the output
        else:
            labels.append(t)
    return labels


_PRIVATE = {c: None for c in range(0xE000, 0xE400)}


_DEC = {}


def _init_worker(labels, langs):
    import kenlm
    from pyctcdecode import BeamSearchDecoderCTC, Alphabet
    from pyctcdecode.language_model import LanguageModel

    class NormLM(LanguageModel):
        """The LM sees normalised words; a token reduced to punctuation only is not scored by the LM."""
        def score_partial_token(self, partial_token):
            return super().score_partial_token(normalize(partial_token))

        def score(self, prev_state, word, is_last_word=False):
            w = normalize(word)
            if not w:
                return self.beta, prev_state
            return super().score(prev_state, w.split()[0], is_last_word)

    alphabet = Alphabet.build_alphabet(labels)
    _DEC[None] = BeamSearchDecoderCTC(alphabet, None)
    for l in langs:
        p = LM_DIR / f"{l}.arpa"
        uni = (LM_DIR / f"{l}.unigrams.txt").read_text(encoding="utf-8").split("\n")
        _DEC[l] = BeamSearchDecoderCTC(alphabet, NormLM(kenlm.Model(str(p)), uni))


def _decode(job):
    lp, lang, alpha, beta, beam = job
    dec = _DEC[lang if alpha > 0 else None]
    if dec._language_model is not None:
        dec._language_model.alpha, dec._language_model.beta = alpha, beta
    return clean_raw(dec.decode(lp.astype(np.float32), beam_width=beam).translate(_PRIVATE))


def run_decode(logps, langs, params, labels, workers=16, beam=48):
    jobs = [(lp, l, *params.get(l, (0.0, 0.0)), beam) for lp, l in zip(logps, langs)]
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(labels, LANGS)) as ex:
        return list(ex.map(_decode, jobs, chunksize=4))


def load_logits(tag, split):
    z = np.load(ROOT / "logits" / f"{tag}_{split}.npz")
    df = pd.read_csv(ROOT / "logs" / f"{tag}_{split}.csv", keep_default_na=False)
    return df, [z[i] for i in df.id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["grid", "decode"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True, help="to retrieve the CTC vocabulary")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--params", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--beam", type=int, default=48)
    ap.add_argument("--langs", nargs="+", default=None, help="languages to tune (grid)")
    a = ap.parse_args()
    labels = labels_from_processor(a.model)
    df, logps = load_logits(a.tag, a.split)
    lid = train_lid()
    df["lid"] = [lid.predict(h) for h in df.hyp]
    if "language" in df and a.split != "test":
        acc = (df.lid == df.language).mean()
        print(f"Text LID: accuracy {acc:.4f}\n", pd.crosstab(df.language, df.lid))

    if a.mode == "grid":
        alphas, betas = [0.0, 0.2, 0.35, 0.5, 0.7], [0.0, 1.0, 2.0]
        best = {}
        for lang in (a.langs or sorted(df.language.unique())):
            sel = df.index[df.language == lang]
            rows = []
            for al, be in itertools.product(alphas, betas):
                if al == 0 and be > 0:
                    continue
                hyps = run_decode([logps[i] for i in sel], [lang] * len(sel), {lang: (al, be)}, labels, a.workers, a.beam)
                s = score(df.ref[sel].tolist(), hyps)
                rows.append((al, be, s["wer"], s["cer"], s["score"]))
                print(f"{lang} alpha={al} beta={be}  WER={s['wer']:.4f} CER={s['cer']:.4f} score={s['score']:.4f}", flush=True)
            r = max(rows, key=lambda x: x[4]); best[lang] = [r[0], r[1]]
            print(f"==> {lang}: best alpha={r[0]} beta={r[1]} score={r[4]:.4f}")
        out = LM_DIR / f"params_{a.tag}.json"
        if out.exists():
            best = {**json.loads(out.read_text()), **best}
        out.write_text(json.dumps(best, indent=1)); print("->", out, best)
        # Phase 2 simulation: language predicted by the LID, not the language column
        df["hyp_lm"] = run_decode(logps, df.lid.tolist(), best, labels, a.workers, a.beam)
        print("\nGreedy :\n", score_by_lang(df, hyp_col="hyp", ref_col="ref").round(4).to_markdown())
        print("\nLID + LM (no metadata):\n", score_by_lang(df, hyp_col="hyp_lm", ref_col="ref").round(4).to_markdown())
    else:
        params = json.loads(Path(a.params).read_text())
        df["hyp_lm"] = run_decode(logps, df.lid.tolist(), params, labels, a.workers, a.beam)
        df.to_csv(a.out, index=False); print("->", a.out)


if __name__ == "__main__":
    main()

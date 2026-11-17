"""Light text post-processing.
- first_cap: upper-case the first character (the only step used by the final submission: --mode cap).
- collapse_loops: a run of >= 3 repeats of a 1-4 word n-gram is cut back to 2 copies (CTC decoding loops).
- cap_after_stop: upper-case after . ! ?
Usage:
  python postproc.py --pred logs/mbr_F_phase2.csv --out logs/F_phase2.csv --mode cap
  python postproc.py --pred logs/mbr_F_val.csv       (no --out: scores every mode on the validation split)"""
import argparse, re

import pandas as pd

from common import score_by_lang


def collapse_loops(text, max_n=4, keep=2):
    """Shorten repeated word n-grams.

    Parameters:
        text (str): transcription.
        max_n (int): longest n-gram (in words) checked for repetition.
        keep (int): copies kept when an n-gram is repeated more than `keep` times in a row.

    Returns:
        str: transcription without decoding loops.
    """
    words = text.split()
    changed = True
    while changed:
        changed = False
        for n in range(max_n, 0, -1):
            i = 0
            while i + n * (keep + 1) <= len(words):
                unit = [x.lower().strip(".,!?;:") for x in words[i:i + n]]
                reps = 1
                while i + n * (reps + 1) <= len(words) and [x.lower().strip(".,!?;:") for x in words[i + n * reps:i + n * (reps + 1)]] == unit:
                    reps += 1
                if reps > keep:
                    words = words[:i + n * keep] + words[i + n * reps:]
                    changed = True
                i += 1
    return " ".join(words)


def cap_after_stop(text):
    """Upper-case the first letter after '.', '!' or '?'."""
    return re.sub(r"([.!?]\s+)([a-zà-ÿ])", lambda m: m.group(1) + m.group(2).upper(), text)


def first_cap(text):
    """Upper-case the first character of the transcription (WAXAL references start with a capital)."""
    return text[:1].upper() + text[1:] if text else text


PIPE = {
    "none": lambda t: t,
    "cap": first_cap,  # final submission
    "loops": collapse_loops,
    "loops+cap": lambda t: first_cap(collapse_loops(t)),
    "loops+cap+after": lambda t: first_cap(cap_after_stop(collapse_loops(t))),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--mode", default="loops+cap", choices=list(PIPE))
    args = parser.parse_args()
    df = pd.read_csv(args.pred, keep_default_na=False)
    if args.out:
        df["hyp"] = df.hyp.map(PIPE[args.mode]); df.to_csv(args.out, index=False); print("->", args.out)
    else:
        for name, fn in PIPE.items():
            s = score_by_lang(df.assign(hyp=df.hyp.map(fn)), ref_col="ref")
            print(f"{name:16s} lin={s.loc['lin','score']:.4f} sna={s.loc['sna','score']:.4f} ALL={s.loc['ALL','score']:.4f}")
        print("clips changed by collapse_loops:", (df.hyp.map(collapse_loops) != df.hyp).sum())

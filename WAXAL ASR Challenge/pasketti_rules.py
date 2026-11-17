"""Word-boundary rules for Lingala and Shona.

WAXAL train/validation transcriptions and the Phase 2 references segment some words differently (e.g. Lingala
"namoni" vs "na moni"). The model follows the train convention; these rules rewrite the output to the Phase 2
convention. Lists re-implemented from Team Pasketti's MIT-licensed public solution of the original WAXAL challenge
(residual_confirmed_rules.py, "confirmed" + "baverb" steps; their corpus/pool tables are NOT used), plus our own
`ba` + noun list (LIN_BA_NOUN). The direction of the rules was chosen from leaderboard feedback (see README).
Usage:
  python pasketti_rules.py --pred logs/F_phase2.csv --ba-noun --out logs/F_pk_banoun_phase2.csv     # final submission
  python pasketti_rules.py --pred logs/F_validation_lin-sna.csv                                      # score on validation"""
import argparse, re

import pandas as pd

from common import score_by_lang
from rover import lid_for

TRAIL = '.,!?;:"\''  # punctuation kept attached to the end of a rewritten word

# Lingala: words split in the Phase 2 references
LIN_SPLIT = {"namoni": ("na", "moni"), "nazomona": ("nazo", "mona"), "tozomona": ("tozo", "mona")}
# Lingala: "ba" + verb joined in the Phase 2 references
LIN_JOIN_BA = {"zali", "lati", "komi", "fandi", "vandi", "sali", "telemi", "lakisi"}
# Optional (not in the final submission): na + pronoun/word, from Pasketti's V6 pipeline
LIN_NA = {"na" + x: ("na", x) for x in ["ye", "nga", "kati", "se", "yango", "ba", "biso"]}
# Optional (not in the final submission): bazo + verb, forms whose split spelling exists in WAXAL train (>= 3, >= 5 %)
LIN_BAZO = {"bazo" + x: ("bazo", x) for x in ["beta", "kende", "kolakisa", "koma", "lakisa", "landa", "leka", "lia", "sakana",
                                              "sala", "seka", "sepela", "solola", "tala", "tambola", "tanga", "tekisa", "zela"]}
# Final submission (--ba-noun): plural prefix "ba" + noun split; hand-made list of nouns frequent on their own in WAXAL
# train (verbs excluded, since the references join "ba" + verb)
LIN_BA_NOUN = {"ba" + x: ("ba", x) for x in [
    "ndeko", "langi", "ndako", "ndaku", "kiti", "biloko", "nzete", "fololo", "buku", "tata", "mama", "mabanga", "mituka",
    "mutuka", "motuka", "mbuma", "kopo", "manga", "matiti", "ngomba", "bilamba", "elamba", "bilenge", "milangi", "misusu",
    "mashine", "machine", "mafuta", "mibali", "sapato", "mikate", "sahani", "salade", "zelo", "mosusu", "ndunda",
    "pilipili", "sabuni", "liputa", "mayi", "saki", "kwanga", "nyama"]}
# Shona: associative "ne" ("with/and") split before some nouns
SNA_NE = {"newaya": "ne waya", "nebhutsu": "ne bhutsu", "negireyi": "ne gireyi", "nefenzi": "ne fenzi",
          "nesiketi": "ne siketi", "uneruvara": "une ruvara", "aneruvara": "ane ruvara", "ineruvara": "ine ruvara"}
# Shona: progressive "-ri" + infinitive "ku..." split (e.g. "varikufamba" -> "vari kufamba")
SNA_PROG = re.compile(r"^(\w*ri)(ku\w{3,})$", re.IGNORECASE)


def _core(token):
    """Split a token into (word, trailing punctuation)."""
    core = token.rstrip(TRAIL)
    return core, token[len(core):]


def _cap_like(src, s):
    """Return `s`, capitalised if `src` starts with a capital letter."""
    return s[:1].upper() + s[1:] if src[:1].isupper() else s


def lin_rules(text, na_split=False, bazo_split=False, ba_noun=False):
    """Apply the Lingala word-boundary rules.

    Parameters:
        text (str): Lingala transcription.
        na_split, bazo_split (bool): optional rule sets (not used by the final submission).
        ba_noun (bool): split "ba" + noun (used by the final submission).

    Returns:
        str: rewritten transcription (casing and trailing punctuation preserved).
    """
    table = {**LIN_SPLIT, **(LIN_NA if na_split else {}), **(LIN_BAZO if bazo_split else {}), **(LIN_BA_NOUN if ba_noun else {})}
    split_tokens = []
    for token in text.split():
        core, trail = _core(token)
        parts = table.get(core.lower())
        split_tokens.append(f"{_cap_like(core, parts[0])} {parts[1]}{trail}" if parts else token)
    tokens, result, i = split_tokens, [], 0
    while i < len(tokens):
        word = tokens[i]
        if i + 1 < len(tokens) and word.lower() == "ba" and _core(tokens[i + 1])[0].lower() in LIN_JOIN_BA:
            core, trail = _core(tokens[i + 1])
            result.append(_cap_like(word, "ba" + core.lower()) + trail)
            i += 2
            continue
        result.append(word)
        i += 1
    return " ".join(result)


def sna_rules(text):
    """Apply the Shona word-boundary rules (progressive -ri + ku- split, ne- splits).

    Parameters:
        text (str): Shona transcription.

    Returns:
        str: rewritten transcription.
    """
    out = []
    for token in text.split():
        core, trail = _core(token)
        m = SNA_PROG.match(core)
        if m and len(m.group(1)) >= 3:
            out.append(f"{m.group(1)} {m.group(2)}{trail}")
            continue
        ne = SNA_NE.get(core.lower())
        out.append(_cap_like(core, ne) + trail if ne else token)
    return " ".join(out)


def collapse_punct(text):
    """Replace repeated punctuation ("..", ",,") by a single mark."""
    return re.sub(r"([.,!?;:])\1+", r"\1", text)


def final_stop(text):
    """Add a final full stop when the text does not end with . ? or ! (optional, not in the final submission)."""
    text = text.rstrip()
    return text if not text or text[-1] in ".?!" else text + "."


def apply(df, langs=("lin", "sna"), stop=False, lang_col="lid", na_split=False, bazo_split=False, ba_noun=False):
    """Apply the rules of each clip's predicted language.

    Parameters:
        df (pd.DataFrame): predictions with columns hyp and `lang_col`.
        langs (tuple[str]): languages whose rules are applied.
        stop (bool): also force a final full stop.
        lang_col (str): column holding the predicted language.
        na_split, bazo_split, ba_noun (bool): optional Lingala rule sets.

    Returns:
        pd.DataFrame: copy of `df` with the rewritten hyp column.
    """
    rules = {"lin": lambda t: lin_rules(t, na_split, bazo_split, ba_noun), "sna": sna_rules}
    hyps = []
    for hyp, lang in zip(df.hyp, df[lang_col]):
        hyp = collapse_punct(hyp)
        if lang in langs:
            hyp = rules[lang](hyp)
        hyps.append(final_stop(hyp) if stop else hyp)
    return df.assign(hyp=hyps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out")
    parser.add_argument("--langs", nargs="+", default=["lin", "sna"])
    parser.add_argument("--final-stop", action="store_true")
    parser.add_argument("--na-split", action="store_true", help="also split na+{ye,nga,kati,se,yango,ba,biso}")
    parser.add_argument("--bazo-split", action="store_true", help="split bazo+verb")
    parser.add_argument("--ba-noun", action="store_true", help="split ba+noun (LIN_BA_NOUN list)")
    args = parser.parse_args()
    df = pd.read_csv(args.pred, keep_default_na=False)
    if "lid" not in df or (df.lid == "").all():  # files not produced by mbr_routed.py: compute the language
        df["lid"] = df.id.map(lid_for("phase2" if df.id.str.startswith("ID_").all() else "validation_lin-sna"))
    new = apply(df, args.langs, args.final_stop, na_split=args.na_split, bazo_split=args.bazo_split, ba_noun=args.ba_noun)
    changed = new.hyp != df.hyp
    print("rows changed:", int(changed.sum()), "/", len(df), new[changed].groupby("lid").size().to_dict())
    if args.out:
        new.to_csv(args.out, index=False)
        print("->", args.out)
    if (df.get("ref", pd.Series([""])) != "").any():
        for name, d in [("before", df), ("after", new)]:
            s = score_by_lang(d, ref_col="ref")
            print(f"{name}: lin={s.loc['lin','score']:.4f} sna={s.loc['sna','score']:.4f} ALL={s.loc['ALL','score']:.4f}")


if __name__ == "__main__":
    main()

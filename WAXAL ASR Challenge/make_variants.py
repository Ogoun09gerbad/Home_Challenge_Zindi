"""Greedy re-decodes of saved log-probs with several blank penalties (extra candidates for the MBR ensemble).
Usage: python make_variants.py --split phase2 --bps 0 0.5 1 --tags m_lin-specialist ...
Output: logs/v_{tag}_bp{bp}_{split}.csv (same columns as the eval_model.py CSV, hyp replaced)."""
import argparse

from common import ROOT
from logit_ens import decode, get_tokenizer, load

# clean members only (distilled / soup6 / seeds excluded: Phase 1 test labels or Phase 2 audio in their training data)
TAGS = ["m_lin-specialist", "m_linsna-afrivoicemix", "m_linsna-soup5", "m_sna-specialist", "m_linsnalug-raw", "m_ft1"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--bps", nargs="+", type=float, default=[0.0, 0.5, 1.0], help="blank penalties")
    parser.add_argument("--tags", nargs="+", default=TAGS)
    args = parser.parse_args()
    tok = get_tokenizer(args.tags[0]); blank = tok.pad_token_id
    for tag in args.tags:
        df, (log_probs,) = load([tag], args.split)
        for bp in args.bps:
            out = ROOT / "logs" / f"v_{tag}_bp{bp:g}_{args.split}.csv"
            df.assign(hyp=[decode(tok, log_probs[i].astype("float32"), blank, bp) for i in df.id]).to_csv(out, index=False)
        print(tag, "ok", flush=True)

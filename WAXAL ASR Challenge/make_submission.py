"""Write the Zindi submission CSV from predictions (id, hyp).

If zindi/SampleSubmission.csv exists, its columns and ID order are used and checked; otherwise the file is written
with columns ID,Target (format confirmed by the Zindi submission checker).
Usage: python make_submission.py --pred logs/F_pk_banoun_phase2.csv --out submissions/phase2_clean_F_pk_banoun.csv"""
import argparse
from pathlib import Path

import pandas as pd

from common import ROOT
from predict import check_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="CSV with columns id, hyp")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample", default=str(ROOT / "zindi" / "SampleSubmission.csv"))
    args = parser.parse_args()
    pred = pd.read_csv(args.pred, keep_default_na=False)
    by_id = dict(zip(pred.id.astype(str), pred.hyp))

    sample_path = Path(args.sample)
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        id_col, text_col = sample.columns[0], sample.columns[1]
        missing = [i for i in sample[id_col].astype(str) if i not in by_id]
        if missing:
            print(f"WARNING: {len(missing)} SampleSubmission IDs without prediction (e.g. {missing[:3]}) -> empty")
        sub = sample[[id_col]].copy()
        sub[text_col] = [by_id.get(str(i), "") for i in sample[id_col]]
        check_submission(sub, sample)
    else:
        print(f"{sample_path} not found -> columns ID,Target")
        sub = pred[["id"]].rename(columns={"id": "ID"})
        sub["Target"] = pred.hyp
        assert not sub.ID.duplicated().any(), "duplicated IDs"
        assert not sub.isna().any().any(), "NaN in submission"
        print(f"{len(sub)} rows, {(sub.Target.str.strip() == '').sum()} empty transcriptions")
    sub = sub.fillna("")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print("->", args.out)


if __name__ == "__main__":
    main()

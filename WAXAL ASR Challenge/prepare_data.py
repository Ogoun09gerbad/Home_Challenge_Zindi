#!/usr/bin/env python
"""Download WaxalNLP (Hugging Face) and the Phase 2 audio, with automatic retry and resume.

Usage:
  python prepare_data.py                    # train + validation + test for lin, sna, lug (~12.6 GB)
  python prepare_data.py --splits train     # only what the final submission needs (text LID training data)
Exit code 2 means the optional Zindi CSVs are missing from ./zindi (not needed by the final pipeline)."""
import argparse, sys, time, zipfile
from pathlib import Path

import requests
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).parent
DATA, ZINDI = ROOT / "data", ROOT / "zindi"
REPO = "google/WaxalNLP"
PHASE2_URL = "https://storage.googleapis.com/waxalphase2/newaudios.zip"
LANGS = ["lin", "sna", "lug"]  # the ~78 GB "unlabeled" files are never downloaded
ZINDI_FILES = ["Train.csv", "Test.csv", "Test_Phase2.csv", "SampleSubmission.csv"]


def retry(fn, tries=5, wait=15):
    """Call `fn` until it succeeds, waiting 15 s, 30 s, ... between attempts.

    Parameters:
        fn (callable): function without arguments.
        tries (int): maximum number of attempts.
        wait (int): base waiting time in seconds.

    Returns:
        Whatever `fn` returns.
    """
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            print(f"[retry {i + 1}/{tries}] {e}", flush=True)
            time.sleep(wait * (i + 1))
    raise RuntimeError("Download failed after several attempts")


def check_zindi():
    """Return True if the Zindi CSVs are in ./zindi (informational only)."""
    missing = [f for f in ZINDI_FILES if not (ZINDI / f).exists()]
    if missing:
        print(f"Missing in {ZINDI}: {missing} (optional; download them from the Zindi Data tab if needed)")
        return False
    print("Zindi files OK")
    return True


def hf_dataset(splits):
    """Download the WaxalNLP ASR parquet files of the given splits for lin, sna and lug to data/waxal.

    Parameters:
        splits (list[str]): any of "train", "validation", "test".
    """
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    print(f"{len(files)} files in {REPO}")
    patterns = [f for f in files if any(f.startswith(f"data/ASR/{lang}/{lang}-{split}-") for lang in LANGS for split in splits)]
    patterns += [f for f in files if "/" not in f]  # README, licence, croissant metadata
    print(f"{len(patterns)} files selected ({LANGS}, splits {splits})")
    if len(patterns) <= 3:
        sys.exit("No ASR file found: the dataset layout has changed.")
    retry(lambda: snapshot_download(REPO, repo_type="dataset", local_dir=DATA / "waxal",
                                    allow_patterns=patterns, max_workers=8))


def phase2():
    """Download (resumable) and extract the Phase 2 audio zip to data/phase2."""
    out_dir, zip_path = DATA / "phase2", DATA / "newaudios.zip"
    if out_dir.exists() and any(out_dir.iterdir()):
        print("Phase 2 audio already extracted"); return

    def download():
        pos = zip_path.stat().st_size if zip_path.exists() else 0
        headers = {"Range": f"bytes={pos}-"} if pos else {}
        with requests.get(PHASE2_URL, headers=headers, stream=True, timeout=60) as r:
            if r.status_code == 416:  # already complete
                return
            r.raise_for_status()
            append = pos > 0 and r.status_code == 206  # 206 = resume accepted; 200 = server ignored Range
            with open(zip_path, "ab" if append else "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)

    retry(download)
    out_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    print("Phase 2 audio extracted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    zindi_ok = check_zindi()
    hf_dataset(args.splits)
    phase2()
    print("Data ready." if zindi_ok else "WaxalNLP and Phase 2 data ready; Zindi CSVs still missing.")
    sys.exit(0 if zindi_ok else 2)

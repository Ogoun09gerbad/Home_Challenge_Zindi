"""Uniform weight averaging of w2v-BERT checkpoints sharing the same CTC vocabulary/order (clean members only).
Usage: python make_soup.py --out models/soupC --members linsnalug-raw lin-specialist sna-specialist linsna-afrivoicemix linsna-soup5"""
import argparse, json, shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

CLEAN = {"linsnalug-raw", "lin-specialist", "sna-specialist", "linsna-afrivoicemix", "linsna-soup5"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    a = ap.parse_args()
    bad = set(a.members) - CLEAN
    assert not bad, f"members not allowed (provenance): {bad}"
    dirs = [Path(snapshot_download("anyantudre/waxal-w2vbert-" + m)) for m in a.members]
    vocabs = [json.loads((d / "vocab.json").read_text(encoding="utf-8")) for d in dirs]
    assert all(v == vocabs[0] for v in vocabs), "vocabularies differ"
    acc = None
    for d in dirs:
        sd = load_file(d / "model.safetensors")
        if acc is None:
            acc = {k: v.float().clone() if v.is_floating_point() else v for k, v in sd.items()}
        else:
            assert sd.keys() == acc.keys()
            for k, v in sd.items():
                if v.is_floating_point():
                    acc[k] += v.float()
    n = len(dirs)
    out = {k: (v / n).to(torch.float32) if v.is_floating_point() else v for k, v in acc.items()}
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, o / "model.safetensors", metadata={"format": "pt"})
    for f in dirs[0].iterdir():
        if f.name != "model.safetensors" and f.is_file():
            shutil.copy(f, o / f.name)
    print("soup of", n, "models ->", o)

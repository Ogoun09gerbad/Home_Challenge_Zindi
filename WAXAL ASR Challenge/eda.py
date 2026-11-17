"""EDA courte sur WaxalNLP (lin/sna/lug) -> reports/EDA.md. Usage : python eda.py [--audio-sample N]"""
import argparse, collections, sys, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from common import LANGS, ROOT, SR, decode_audio, load_split, normalize, set_seed


def silence_ratio(y, top_db=35, frame=512):
    import librosa
    if len(y) == 0:
        return 1.0
    iv = librosa.effects.split(y, top_db=top_db, frame_length=frame, hop_length=frame // 4)
    voiced = sum(e - s for s, e in iv)
    return 1 - voiced / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-sample", type=int, default=300, help="clips décodés par langue/split pour durée/silence/SR")
    args = ap.parse_args()
    set_seed(42)
    out = ["# EDA WaxalNLP (lin, sna, lug)\n"]
    char_counts = {}
    for split in ["train", "validation", "test"]:
        try:
            df = load_split(split, with_audio=True)
        except ValueError:
            out.append(f"\n## {split} : aucun fichier\n"); continue
        out.append(f"\n## {split}\n")
        rows = []
        for lang, g in df.groupby("language"):
            s = g.sample(min(args.audio_sample, len(g)), random_state=42)
            durs, sil, srs = [], [], collections.Counter()
            for b in s.audio_bytes:
                import io, soundfile as sf
                try:
                    srs[sf.info(io.BytesIO(b)).samplerate] += 1
                except Exception:
                    srs["?"] += 1
                y = decode_audio(b)
                durs.append(len(y) / SR); sil.append(silence_ratio(y))
            nw = g.text.fillna("").str.split().str.len()
            rows.append(dict(lang=lang, n=len(g), speakers=g.speaker_id.nunique(),
                             est_hours=round(np.mean(durs) * len(g) / 3600, 1),
                             dur_mean=round(np.mean(durs), 1), dur_p95=round(np.percentile(durs, 95), 1),
                             dur_max=round(max(durs), 1), silence=round(np.mean(sil), 2),
                             words_mean=round(nw.mean(), 1), empty_text=int((g.text.fillna("").str.strip() == "").sum()),
                             female=round((g.gender == "Female").mean(), 2), sr=dict(srs)))
            if split == "train":
                char_counts[lang] = collections.Counter("".join(g.text.fillna("")))
        out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")
        if split == "validation":
            tr = load_split("train")
            for lang in LANGS:
                a = set(tr[tr.language == lang].speaker_id); b = set(df[df.language == lang].speaker_id)
                out.append(f"- {lang}: locuteurs val aussi dans train = {len(a & b)}/{len(b)}\n")
    out.append("\n## Caractères (train)\n")
    for lang, c in char_counts.items():
        special = {ch: n for ch, n in c.most_common() if not (ch.isascii() and (ch.isalnum() or ch == " "))}
        out.append(f"\n**{lang}** : {len(c)} caractères distincts. Non ASCII-alphanum : "
                   + ", ".join(f"`{ch}` U+{ord(ch):04X} {unicodedata.name(ch, '?')} ({n})" for ch, n in list(special.items())[:40]) + "\n")
        normed = collections.Counter(normalize("".join(k * v for k, v in c.items() if k.strip() or k == " ")))
        out.append(f"  après normalize() : {''.join(sorted(normed))!r}\n")
    tr = load_split("train")
    out.append("\n## Exemples de transcriptions (train)\n")
    for lang in LANGS:
        for t in tr[tr.language == lang].text.sample(min(5, (tr.language == lang).sum()), random_state=1):
            out.append(f"- {lang}: {t}\n")
    Path(ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "EDA.md").write_text("".join(out), encoding="utf-8")
    print("".join(out))


if __name__ == "__main__":
    main()

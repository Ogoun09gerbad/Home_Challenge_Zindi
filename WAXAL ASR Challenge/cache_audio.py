"""Decode audio once to 16 kHz int16 samples stored in one flat memory-mapped file, plus a parquet index.

Usage:
  python cache_audio.py --split train validation [--langs lin sna lug] [--workers 8]     # WaxalNLP splits
  python cache_audio.py --audio-dir data/phase2 --name phase2                             # folder without metadata
Output: data/cache/{split}_{lang}.int16 (or {name}_p2.int16) and the matching .parquet index
(id, speaker_id, text, language, gender, offset, length). A clip is samples[offset : offset + length] / 32767."""
import argparse, glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import LANGS, ROOT, SR, WAXAL_DIR, decode_audio

CACHE = ROOT / "data" / "cache"
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}


def _to_int16(data):
    """Decode audio bytes to int16 at 16 kHz; an unreadable file becomes an empty clip instead of crashing."""
    try:
        audio = decode_audio(data)
    except Exception:
        audio = np.zeros(0, np.float32)
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


def cache(split, lang, workers):
    """Cache one WaxalNLP split / language (skipped if already cached).

    Parameters:
        split (str): "train", "validation" or "test" (test transcriptions are never stored).
        lang (str): language code.
        workers (int): decoding processes.
    """
    raw, index = CACHE / f"{split}_{lang}.int16", CACHE / f"{split}_{lang}.parquet"
    if index.exists():
        print(f"{index.name} already cached"); return
    files = sorted(glob.glob(str(WAXAL_DIR / lang / f"{lang}-{split}-*.parquet")))
    if not files:
        print(f"{split}/{lang}: no parquet file"); return
    rows, offset = [], 0
    with open(raw.with_suffix(".tmp"), "wb") as fout, ProcessPoolExecutor(workers) as ex:
        for f in files:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=["id", "speaker_id", "transcription", "language", "gender", "audio"]).to_pandas()
                for (_, r), samples in zip(table.iterrows(), ex.map(_to_int16, [a["bytes"] for a in table.audio], chunksize=8)):
                    fout.write(samples.tobytes())
                    text = "" if split == "test" else r.transcription  # Phase 1 test labels: never stored
                    rows.append((r.id, r.speaker_id, text, r.language, r.gender, offset, len(samples)))
                    offset += len(samples)
            print(f"{split}/{lang}: {Path(f).name} -> {len(rows)} clips, {offset / SR / 3600:.1f} h", flush=True)
    raw.with_suffix(".tmp").replace(raw)
    pd.DataFrame(rows, columns=["id", "speaker_id", "text", "language", "gender", "offset", "length"]).to_parquet(index)


def cache_dir(audio_dir, name, workers):
    """Cache every audio file of a folder (e.g. Phase 2) to data/cache/{name}_p2.*; id = file name without extension.

    Parameters:
        audio_dir (str): folder searched recursively (macOS "._*" / __MACOSX files of the zip are ignored).
        name (str): cache name.
        workers (int): decoding processes.
    """
    raw, index = CACHE / f"{name}_p2.int16", CACHE / f"{name}_p2.parquet"
    files = sorted(p for p in Path(audio_dir).rglob("*") if p.suffix.lower() in AUDIO_EXT
                   and not p.name.startswith("._") and "__MACOSX" not in p.parts)
    rows, offset = [], 0
    with open(raw, "wb") as fout, ProcessPoolExecutor(workers) as ex:
        for p, samples in zip(files, ex.map(_to_int16, [p.read_bytes() for p in files], chunksize=8)):
            fout.write(samples.tobytes())
            rows.append((p.stem, "", "", "", "", offset, len(samples)))
            offset += len(samples)
    pd.DataFrame(rows, columns=["id", "speaker_id", "text", "language", "gender", "offset", "length"]).to_parquet(index)
    print(f"{name}: {len(rows)} clips, {offset / SR / 3600:.2f} h, longest clip {max(r[6] for r in rows) / SR:.1f} s")


def open_mmaps(split, langs=LANGS):
    """Return {lang: read-only int16 memmap} for the cached languages of a split."""
    return {lang: np.memmap(CACHE / f"{split}_{lang}.int16", dtype=np.int16, mode="r")
            for lang in langs if (CACHE / f"{split}_{lang}.parquet").exists()}


def load_cached(split, langs=LANGS):
    """Return (index DataFrame with a `cache_lang` column = memmap key, {lang: memmap})."""
    dfs = [pd.read_parquet(CACHE / f"{split}_{lang}.parquet").assign(cache_lang=lang)
           for lang in langs if (CACHE / f"{split}_{lang}.parquet").exists()]
    return pd.concat(dfs, ignore_index=True), open_mmaps(split, langs)


def load_phase2(name="phase2"):
    """Return (index DataFrame, {"p2": memmap}) for a folder cached by cache_dir()."""
    df = pd.read_parquet(CACHE / f"{name}_p2.parquet").assign(cache_lang="p2")
    return df, {"p2": np.memmap(CACHE / f"{name}_p2.int16", dtype=np.int16, mode="r")}


def get_audio(row, mmaps):
    """Return the float32 waveform of one index row.

    Parameters:
        row: a row of the index (needs cache_lang, offset, length).
        mmaps (dict): memmaps returned by load_cached / load_phase2.

    Returns:
        np.ndarray: float32 waveform at 16 kHz.
    """
    return mmaps[row.cache_lang][row.offset:row.offset + row.length].astype(np.float32) / 32767


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--langs", nargs="+", default=LANGS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--audio-dir", default=None, help="folder without metadata, cached under --name")
    parser.add_argument("--name", default="phase2")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.audio_dir:
        cache_dir(args.audio_dir, args.name, args.workers); raise SystemExit
    for split in args.split:
        for lang in args.langs:
            cache(split, lang, args.workers)

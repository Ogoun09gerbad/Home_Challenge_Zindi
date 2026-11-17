"""Shared utilities: paths, random seeds, text normalisation, Zindi metric and WaxalNLP loading."""
import glob, io, random, re, unicodedata
from pathlib import Path

import jiwer
import librosa
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch

ROOT = Path(__file__).parent
WAXAL_DIR = ROOT / "data" / "waxal" / "data" / "ASR"
LANGS = ["lin", "sna", "lug"]
SR = 16000  # sampling rate expected by w2v-BERT 2.0


def set_seed(seed=42):
    """Seed the Python, NumPy and PyTorch (CPU and CUDA) random generators.

    Parameters:
        seed (int): seed value.
    """
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---- text normalisation --------------------------------------------------------
_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)


def normalize(text, lower=True, keep_apostrophe=True, strip_digits=False):
    """Normalise text for language ID and n-gram statistics (not for the submission).

    Parameters:
        text (str): input text.
        lower (bool): lower-case the text.
        keep_apostrophe (bool): keep apostrophes (word-internal in Luganda), drop other punctuation.
        strip_digits (bool): remove digits.

    Returns:
        str: NFC text without punctuation, single-spaced.
    """
    t = unicodedata.normalize("NFC", str(text))
    if lower:
        t = t.lower()
    t = t.replace("’", "'").replace("‘", "'").replace("`", "'")
    t = _PUNCT.sub(" ", t) if keep_apostrophe else re.sub(r"[^\w\s]", " ", t)
    if strip_digits:
        t = re.sub(r"\d+", " ", t)
    return re.sub(r"\s+", " ", t.replace("_", " ")).strip()


def clean_raw(text):
    """Light clean-up of a model output, keeping casing and punctuation (the Zindi scorer counts both).

    Parameters:
        text (str): raw decoded text.

    Returns:
        str: NFC text with unified quotes/apostrophes, no space before punctuation, single-spaced.
    """
    t = unicodedata.normalize("NFC", str(text))
    t = t.replace("’", "'").replace("‘", "'").replace("`", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


# ---- metric --------------------------------------------------------------------
def score(refs, hyps):
    """Zindi metric: WER on lower-cased text (punctuation kept), CER on the raw string; score = 1 - 0.5*(WER + CER).

    Parameters:
        refs (list[str]): reference transcriptions.
        hyps (list[str]): predicted transcriptions, same order.

    Returns:
        dict: {"wer", "cer", "error", "score"}; higher score is better.
    """
    refs = [r if r.strip() else " " for r in refs]
    hyps = [h if isinstance(h, str) else "" for h in hyps]
    wer = jiwer.wer([r.lower() for r in refs], [h.lower() for h in hyps])
    cer = jiwer.cer(refs, hyps)
    err = 0.5 * wer + 0.5 * cer
    return {"wer": wer, "cer": cer, "error": err, "score": 1 - err}


def score_by_lang(df, hyp_col="hyp", ref_col="text", lang_col="language"):
    """Zindi metric per language and overall.

    Parameters:
        df (pd.DataFrame): one row per clip with hypothesis, reference and language columns.
        hyp_col, ref_col, lang_col (str): column names.

    Returns:
        pd.DataFrame: one row per language plus "ALL", columns wer / cer / error / score.
    """
    rows = {lang: score(g[ref_col].tolist(), g[hyp_col].tolist()) for lang, g in df.groupby(lang_col)}
    rows["ALL"] = score(df[ref_col].tolist(), df[hyp_col].tolist())
    return pd.DataFrame(rows).T


# ---- WaxalNLP data -------------------------------------------------------------
def load_split(split, langs=LANGS, with_audio=False):
    """Load a WaxalNLP split from the downloaded parquet files.

    Phase 1 test transcriptions are public but forbidden by the challenge rules: they are blanked here and never
    reach any other code.

    Parameters:
        split (str): "train", "validation" or "test".
        langs (list[str]): language codes to load.
        with_audio (bool): also return the encoded audio bytes.

    Returns:
        pd.DataFrame: columns id, speaker_id, text, language, gender[, audio_bytes].
    """
    cols = ["id", "speaker_id", "transcription", "language", "gender"] + (["audio"] if with_audio else [])
    parts = []
    for lang in langs:
        for f in sorted(glob.glob(str(WAXAL_DIR / lang / f"{lang}-{split}-*.parquet"))):
            parts.append(pq.read_table(f, columns=cols).to_pandas())
    df = pd.concat(parts, ignore_index=True).rename(columns={"transcription": "text"})
    if split == "test":
        df["text"] = ""
    elif split not in ("train", "validation"):
        raise ValueError(split)
    if with_audio:
        df["audio_bytes"] = df.pop("audio").map(lambda a: a["bytes"])
    return df


def decode_audio(data, sr=SR):
    """Decode an encoded audio file to mono float32 at `sr` Hz.

    Parameters:
        data (bytes): file content (wav, mp3, flac, ...).
        sr (int): target sampling rate.

    Returns:
        np.ndarray: 1-D float32 waveform in [-1, 1].
    """
    try:
        audio, file_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(1)
        if file_sr != sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    except Exception:  # formats libsndfile cannot read (some mp3): fall back to librosa/audioread
        audio, _ = librosa.load(io.BytesIO(data), sr=sr, mono=True)
    return audio.astype(np.float32)

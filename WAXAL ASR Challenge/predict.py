"""CTC model loading, batched inference and greedy decoding; also a standalone single-model predictor.

Usage (single model, no ensemble):
  python predict.py --model anyantudre/waxal-w2vbert-linsna-soup5 --audio-dir data/phase2 --out submissions/single.csv
The CSV format (columns, IDs) follows zindi/SampleSubmission.csv when present, with automatic checks;
otherwise it is written as ID,Target (format confirmed by the Zindi submission checker)."""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCTC, AutoProcessor

from common import ROOT, SR, clean_raw, decode_audio, load_split, set_seed
from decode_lm import labels_from_processor, run_decode
from lid import train_lid

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}


def load_model(path, device, revision=None):
    """Load a CTC model in fp16 and its processor.

    Parameters:
        path (str): Hugging Face repo id or local folder.
        device (str): "cuda" or "cpu".
        revision (str | None): Hub commit hash, to pin the exact weights.

    Returns:
        tuple: (model in eval mode, processor).
    """
    proc = AutoProcessor.from_pretrained(path, revision=revision)
    model = AutoModelForCTC.from_pretrained(path, revision=revision, dtype=torch.float16).to(device).eval()
    return model, proc


@torch.inference_mode()
def ctc_logits(model, proc, audios, device, max_batch_sec=160):
    """Compute CTC log-probabilities for a list of waveforms.

    Clips are sorted by length and batched so that (batch size x longest clip) stays under `max_batch_sec`
    seconds of audio, which bounds GPU memory and minimises padding.

    Parameters:
        model, proc: from load_model().
        audios (list[np.ndarray]): float32 waveforms at 16 kHz.
        device (str): "cuda" or "cpu".
        max_batch_sec (float): audio budget per batch, in seconds.

    Returns:
        list[np.ndarray]: one (frames x vocabulary) float32 log-prob matrix per clip, in input order.
    """
    key = "input_features" if "Bert" in type(model).__name__ else "input_values"
    order = sorted(range(len(audios)), key=lambda i: len(audios[i]))
    results, start = [None] * len(audios), 0
    while start < len(order):
        n = 1  # ascending order: the last clip of the batch is the longest (budget = n x longest)
        while start + n < len(order) and (n + 1) * len(audios[order[start + n]]) <= max_batch_sec * SR:
            n += 1
        batch = order[start:start + n]; start += n
        features = proc.feature_extractor([audios[i] for i in batch], sampling_rate=SR, return_tensors="pt",
                                          padding=True, return_attention_mask=True)
        log_probs = torch.log_softmax(model(features[key].to(device, torch.float16),
                                            attention_mask=features["attention_mask"].to(device)).logits.float(), -1)
        ratio = log_probs.shape[1] / features[key].shape[1]  # effective down-sampling from input to CTC frames
        for j, i in enumerate(batch):
            n_frames = int(np.ceil(features["attention_mask"][j].sum().item() * ratio))
            results[i] = log_probs[j, : max(1, min(n_frames, log_probs.shape[1]))].cpu().numpy()
    return results


def greedy_decode(proc, logps):
    """Greedy (argmax) CTC decoding.

    Parameters:
        proc: processor whose tokenizer collapses repeats and removes blanks.
        logps (list[np.ndarray]): log-prob matrices from ctc_logits().

    Returns:
        list[str]: transcriptions.
    """
    return [proc.decode(np.argmax(lp, -1)) for lp in logps]


def check_submission(sub, sample):
    """Assert that a submission matches SampleSubmission.csv (columns, row count, IDs, no duplicates, no NaN)."""
    assert list(sub.columns) == list(sample.columns), (sub.columns, sample.columns)
    id_col = sample.columns[0]
    assert len(sub) == len(sample), f"{len(sub)} rows instead of {len(sample)}"
    assert set(sub[id_col]) == set(sample[id_col]), "IDs differ from SampleSubmission"
    assert not sub[id_col].duplicated().any(), "duplicated IDs"
    assert not sub.isna().any().any(), "NaN in submission"
    print(f"Checks OK: {len(sub)} rows, columns {list(sub.columns)}, "
          f"{(sub.iloc[:, 1].str.strip() == '').sum()} empty transcriptions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-dir")
    source.add_argument("--waxal-test", action="store_true", help="WaxalNLP test split audio (Phase 1); labels never read")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample", default=str(ROOT / "zindi" / "SampleSubmission.csv"))
    parser.add_argument("--chunk", type=int, default=256, help="clips decoded in memory at once")
    parser.add_argument("--lm-params", default=None, help="lm/params_{tag}.json (alpha, beta per language); default greedy")
    args = parser.parse_args()
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_model(args.model, device)

    if args.waxal_test:
        df = load_split("test", with_audio=True)
        ids, loaders = df.id.tolist(), [lambda b=b: decode_audio(b) for b in df.audio_bytes]
    else:
        files = sorted(p for p in Path(args.audio_dir).rglob("*") if p.suffix.lower() in AUDIO_EXT
                       and not p.name.startswith("._") and "__MACOSX" not in p.parts)  # macOS files of the zip
        ids, loaders = [p.stem for p in files], [lambda p=p: decode_audio(p.read_bytes()) for p in files]
    print(f"{len(ids)} clips")
    logps = []
    for k in range(0, len(ids), args.chunk):
        audios = [load() for load in loaders[k:k + args.chunk]]
        logps += ctc_logits(model, proc, audios, device)
        print(f"{min(k + args.chunk, len(ids))}/{len(ids)}", flush=True)
    del model; torch.cuda.empty_cache()
    hyps = [clean_raw(h) for h in greedy_decode(proc, logps)]
    if args.lm_params:  # text LID on the greedy output -> n-gram LM of the predicted language (no metadata)
        lid = train_lid()
        langs = [lid.predict(h) for h in hyps]
        print("LID:", pd.Series(langs).value_counts().to_dict())
        params = json.loads(Path(args.lm_params).read_text())
        hyps = run_decode(logps, langs, params, labels_from_processor(args.model))
        pd.DataFrame({"id": ids, "lid": langs}).to_csv(Path(args.out).with_suffix(".lid.csv"), index=False)
    pred = pd.DataFrame({"id": ids, "transcription": hyps})

    sample_path = Path(args.sample)
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        id_col, text_col = sample.columns[0], sample.columns[1]
        by_id = dict(zip(pred.id.astype(str), pred.transcription))
        missing = [i for i in sample[id_col].astype(str) if i not in by_id]
        if missing:
            print(f"WARNING: {len(missing)} SampleSubmission IDs without prediction (e.g. {missing[:3]}) -> empty string")
        sub = sample[[id_col]].copy()
        sub[text_col] = [by_id.get(str(i), "") for i in sample[id_col]]
        sub = sub[list(sample.columns)] if len(sample.columns) == 2 else sub
        check_submission(sub, sample)
    else:
        print(f"{sample_path} not found -> columns ID,Target")
        sub = pred.rename(columns={"id": "ID", "transcription": "Target"})
    sub = sub.fillna("")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

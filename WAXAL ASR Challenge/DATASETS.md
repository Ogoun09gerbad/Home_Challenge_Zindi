# Data and pretrained models

## Data read by the final pipeline
| Data | Source | Licence | Use |
|---|---|---|---|
| WAXAL ASR, `train` split (lin, sna, lug) | https://huggingface.co/datasets/google/WaxalNLP (`data/ASR/{lin,sna,lug}/`) — paper https://huggingface.co/papers/2602.02734 | CC-BY-SA-4.0 | Transcriptions only, to fit the text language-ID classifier (`lid.py`). |
| Phase 2 audio | https://storage.googleapis.com/waxalphase2/newaudios.zip | challenge data | Final predictions. |

Used during development only (not read by `reproduce.sh`): WAXAL `validation` (model / ensemble selection, 300 clips per
language) and the WAXAL `test` **audio** (Phase 1). The Phase 1 test transcriptions are public but forbidden: `common.load_split`
and `cache_audio.py` blank them, so they never reach any code. The ~78 GB `unlabeled` WaxalNLP files are not downloaded.

## Pretrained models (final submission)
| Model | Source (pinned commit) | Licence |
|---|---|---|
| w2v-BERT 2.0 (pre-training of every member) | https://huggingface.co/facebook/w2v-bert-2.0 | MIT |
| `waxal-w2vbert-lin-specialist` | https://huggingface.co/anyantudre/waxal-w2vbert-lin-specialist (`c5ed347…`) | Apache-2.0 |
| `waxal-w2vbert-sna-specialist` | https://huggingface.co/anyantudre/waxal-w2vbert-sna-specialist (`2cb8644…`) | Apache-2.0 |
| `waxal-w2vbert-linsna-afrivoicemix` | https://huggingface.co/anyantudre/waxal-w2vbert-linsna-afrivoicemix (`31de7fd…`) | Apache-2.0 |
| `waxal-w2vbert-linsna-soup5` | https://huggingface.co/anyantudre/waxal-w2vbert-linsna-soup5 (`16c13b7…`) | Apache-2.0 |

Full commit hashes: `dl_members.py`. According to their model cards, these checkpoints were fine-tuned (by their author, not
by us) on WAXAL lin/sna train, AfriVoice Lingala (https://huggingface.co/datasets/KasuleTrevor/Lingala_100hrs, CC-BY-4.0)
and AfriVoice Shona (https://huggingface.co/datasets/realtime-speech/shona1, CC-BY-4.0), and — for `soup5` — pseudo-labels
on the WaxalNLP *unlabeled* pool. None was trained on Phase 1 test labels or on Phase 2 audio. We use them as openly
available pretrained models only; we do not read those external datasets.

**Excluded after reading their model cards:** `linsna-distilled` (trained on the 892 Phase 2 clips with pseudo-labels),
`linsna-soup6` and seeds 43/44/46 (Phase 1 test transcriptions in their training data), `whisper-turbo-linsna`, all `yehoshua01/*`.

## Methods re-used from public solutions (no data, no weights)
| Method | Source | Licence | Note |
|---|---|---|---|
| Lingala / Shona word-boundary rules (`pasketti_rules.py`, re-implemented) | https://github.com/Team-Pasketti/Google-WAXAL-ASR-Challenge-1st-Place (`postprocess/residual_confirmed_rules.py`) | MIT | Explicit word lists only; their `corpus/pool` tables are not used. That team chose the direction of these rules from leaderboard scores of the original WAXAL challenge (same Phase 2 audio); no test transcription is used. |
| Character-level anchored ROVER (`rover.py`, re-implemented; experiment, not in the final submission) | https://github.com/anyantudre/google-waxal-asr-challenge-solution (`waxal_asr/ensemble.py`) | MIT | Algorithm only. |

No other external dataset is used.

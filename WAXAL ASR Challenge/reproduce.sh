#!/bin/bash
# Reproduces the final submission submissions/phase2_clean_F_pk_banoun.csv (Zindi ID zvHmM5JA, private rank 6).
# Usage: bash reproduce.sh          (set PY=path/to/python to choose the interpreter; a CUDA GPU is required)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=${PY:-python}
mkdir -p logs logits submissions

# 1. Data: WaxalNLP train transcriptions (to fit the text language ID) + Phase 2 audio (892 wav).
#    prepare_data.py exits with 2 when the optional Zindi CSVs are absent from ./zindi: not an error here.
$PY prepare_data.py --splits train || [ $? -eq 2 ]
$PY cache_audio.py --audio-dir data/phase2 --name phase2

# 2. Public Apache-2.0 checkpoints at pinned commits
$PY dl_members.py

# 3. CTC inference of each member on Phase 2 -> logits/m_{member}_phase2.npz (+ logs/m_{member}_phase2.csv)
declare -A REV=(
  [lin-specialist]=c5ed34742e6150c5169e70981f40393fd13cb7c6
  [sna-specialist]=2cb8644b065dd8690e166250df528d85927a084d
  [linsna-afrivoicemix]=31de7fd5fc16eb53f3ea3e717f2e02d4b5d8605d
  [linsna-soup5]=16c13b7381e666637cc291e25d57e27ff50616d6
)
for m in lin-specialist sna-specialist linsna-afrivoicemix linsna-soup5; do
  $PY eval_model.py --model "anyantudre/waxal-w2vbert-$m" --revision "${REV[$m]}" --tag "m_$m" --split phase2 --max-batch-sec 60
done

# 4. Greedy re-decodes with blank penalty 0 / 0.5 / 1 -> logs/v_m_{member}_bp{bp}_phase2.csv
$PY make_variants.py --split phase2 --bps 0 0.5 1 \
  --tags m_lin-specialist m_linsna-afrivoicemix m_linsna-soup5 m_sna-specialist

# 5. Text language ID + MBR selection routed by language (config F)
$PY mbr_routed.py --split phase2 --config configs/F_clean_small.json --out logs/mbr_F_phase2.csv

# 6. Upper-case first character
$PY postproc.py --pred logs/mbr_F_phase2.csv --out logs/F_phase2.csv --mode cap

# 7. Word-boundary rules (Lingala + Shona) + "ba" + noun split
$PY pasketti_rules.py --pred logs/F_phase2.csv --ba-noun --out logs/F_pk_banoun_phase2.csv

# 8. Zindi CSV (ID, Target)
$PY make_submission.py --pred logs/F_pk_banoun_phase2.csv --out submissions/phase2_clean_F_pk_banoun.csv

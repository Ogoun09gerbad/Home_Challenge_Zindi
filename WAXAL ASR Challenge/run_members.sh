#!/bin/bash
# Evaluates each selected public member (top-6 on lin+sna validation) on a split -> logs/m_{name}_{split}*.csv
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
SPLIT=${1:-validation}
MEMBERS="linsnalug-raw lin-specialist sna-specialist linsna-soup6 linsna-soup5 linsna-seed43 linsna-seed44 linsna-seed46 linsna-distilled linsna-afrivoicemix"
TOP6="lin-specialist linsna-afrivoicemix linsna-distilled linsna-soup5 linsna-soup6 sna-specialist"
LIST=$MEMBERS
[ "$2" = "top6" ] && LIST=$TOP6
for m in $LIST; do
  tag="m_${m}"
  if [ "$SPLIT" = "phase2" ]; then
    out="logs/${tag}_phase2.csv"
    [ -f "$out" ] && continue
    .venv/Scripts/python.exe eval_model.py --model "anyantudre/waxal-w2vbert-${m}" --tag "$tag" --split phase2 --max-batch-sec 60 > "logs/eval_${tag}_phase2.log" 2>&1
    echo "$tag phase2 : $(tail -3 logs/eval_${tag}_phase2.log | tr '\r' '\n' | tail -1)"
  else
    [ -f "logs/${tag}_${SPLIT}_lin-sna.csv" ] && continue
    .venv/Scripts/python.exe eval_model.py --model "anyantudre/waxal-w2vbert-${m}" --tag "$tag" --split "$SPLIT" --n 300 --langs lin sna --max-batch-sec 60 > "logs/eval_${tag}_${SPLIT}.log" 2>&1
    echo "$tag $(grep '| ALL' logs/eval_${tag}_${SPLIT}.log)"
  fi
done

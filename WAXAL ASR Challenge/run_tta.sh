#!/bin/bash
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
SPLIT=$1
for m in linsna-soup5 lin-specialist; do for sp in 0.9 1.1; do
  tag="t_${m}_s${sp}"
  if [ "$SPLIT" = "phase2" ]; then
    [ -f logs/${tag}_phase2.csv ] || .venv/Scripts/python.exe eval_model.py --model anyantudre/waxal-w2vbert-$m --tag $tag --split phase2 --speed $sp --max-batch-sec 60 > logs/eval_${tag}_phase2.log 2>&1
  else
    [ -f logs/${tag}_validation_lin-sna.csv ] || .venv/Scripts/python.exe eval_model.py --model anyantudre/waxal-w2vbert-$m --tag $tag --split validation --n 300 --langs lin sna --speed $sp --max-batch-sec 60 > logs/eval_${tag}_validation.log 2>&1
  fi
  echo "$tag $(tr '\r' '\n' < logs/eval_${tag}_${SPLIT}.log | grep '| ALL')"
done; done

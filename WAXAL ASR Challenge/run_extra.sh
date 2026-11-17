#!/bin/bash
# Clean external member (different vocab -> text-level vote only)
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
M="keystats/w2v-bert-2.0-multilingual-waxal-shona-lingala-luganda-best"; T="x_keystats"
.venv/Scripts/python.exe eval_model.py --model "$M" --tag $T --split validation --n 300 --langs lin sna --max-batch-sec 60 > logs/eval_${T}_validation.log 2>&1
.venv/Scripts/python.exe eval_model.py --model "$M" --tag $T --split phase2 --max-batch-sec 60 > logs/eval_${T}_phase2.log 2>&1
echo done

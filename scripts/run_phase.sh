#!/bin/bash
# run_phase.sh <model_tag> <phase> [seeds] [base_port]
# One runner per task, each talking to its own llama.cpp server (ports base..base+3).
cd "$(dirname "$0")/.."
source .venv/bin/activate
SEEDS=${3:-0,1}
P=${4:-18393}
for T in sst2 agnews arc squad; do
  python src/run_experiment.py --model-tag $1 --port $P --n 100 --tasks $T --phase $2 --seeds $SEEDS --workers 6 \
    > logs/run_$1_$2_$T.log 2>&1 &
  P=$((P+1))
done
wait

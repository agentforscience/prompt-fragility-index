#!/bin/bash
# chain.sh <wait_pid> <model_tag> <port> <job> [<job> ...]
# Waits for a running runner to exit, then runs jobs sequentially on one server.
# A job is "task:phase:seeds", e.g. "sst2:combos:0,1".
cd "$(dirname "$0")/.."
source .venv/bin/activate
WAIT=$1; MODEL=$2; PORT=$3; shift 3
while kill -0 "$WAIT" 2>/dev/null; do sleep 15; done
for JOB in "$@"; do
  IFS=: read -r T PH S <<< "$JOB"
  python src/run_experiment.py --model-tag "$MODEL" --port "$PORT" --n 100 --tasks "$T" --phase "$PH" --seeds "$S" --workers 6 \
    >> "logs/run_${MODEL}_${PH}_${T}.log" 2>&1
done

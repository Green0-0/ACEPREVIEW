#!/bin/bash
# Evaluate a predictions file with the Apptainer SWE-bench sandbox.
#
#   sbatch slurm/swebench_eval.sbatch PREDS.jsonl OUT.jsonl [extra swebench_eval.py args]
#   PREDS may be "gold" or "empty".
#
# Sharding one predictions file over several jobs (each writes its own file):
#   sbatch --array=0-3 slurm/swebench_eval.sbatch preds.jsonl out/results.jsonl
#   -> task i runs --shard i/4 into out/results.shard<i>.jsonl
#   cat out/results.shard*.jsonl > out/results.jsonl
#SBATCH --job-name=ace-swe-eval
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/usr/project/xtmp/kg349/ace/logs/eval/%x_%A_%a.out

set -euo pipefail
PREDS=$1; OUT=$2; shift 2
REPO=${ACE_REPO:-/home/users/kg349/ACEPREVIEW}
export ACE_SWEBENCH_SCRATCH=${ACE_SWEBENCH_SCRATCH:-/tmp}
export HF_HOME=/usr/project/xtmp/kg349/hf_cache
export APPTAINER_CACHEDIR=/usr/project/xtmp/kg349/ace/apptainer_cache
export PYTHONDONTWRITEBYTECODE=1
WORKERS=${WORKERS:-10}

if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "${SLURM_ARRAY_TASK_COUNT}" -gt 1 ]; then
    i=$SLURM_ARRAY_TASK_ID; n=${NSHARD:-$SLURM_ARRAY_TASK_COUNT}  # NSHARD: rerun a subset of shards
    OUT="${OUT%.jsonl}.shard${i}.jsonl"
    set -- --shard "$i/$n" "$@"
fi
echo "host=$(hostname) preds=$PREDS out=$OUT workers=$WORKERS args=$*"
exec /usr/project/xtmp/kg349/ace/venv/bin/python "$REPO/scripts/swebench_eval.py" \
    --predictions "$PREDS" --out "$OUT" --workers "$WORKERS" "$@"

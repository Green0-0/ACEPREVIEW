#!/bin/bash
#SBATCH --job-name=ace-vllm
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=compsci-cluster-fitz-01
#SBATCH --output=/usr/project/xtmp/kg349/ace/logs/%x_%j.out
#SBATCH --error=/usr/project/xtmp/kg349/ace/logs/%x_%j.out
#
# One vLLM OpenAI server for ACE. When healthy it writes
# $ACE_ROOT/endpoints/<jobid>.json ({"url", "model", ...}); clients discover servers by
# reading that directory (ace.serving.endpoints). The file is removed on exit.
#
# Env overrides: MODEL, SERVED (served model name), PP (pipeline-parallel size, = GPUs),
# MAXLEN, MAXSEQS, GMU, EXTRA (extra vllm args, '+'-separated), IDLE_MIN.
#
# Idle watchdog: GPUs must never sit idle for long. Every minute the job reads the
# server's generation-token counter from /metrics; if it has not moved for IDLE_MIN
# minutes (default 20, counted from when the server became healthy), the job exits and
# releases its GPUs. Clients wait for / fail over to other endpoints (ace.serving).
set -u
source /home/users/kg349/llm_speed_bench/sbatch/env_gpu.sh
ACE_ROOT=${ACE_ROOT:-/usr/project/xtmp/kg349/ace}
VENV=${VENV:-/usr/project/xtmp/kg349/venvs/vllm}
MODEL=${MODEL:-cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit}
SERVED=${SERVED:-gemma4-26b}
PP=${PP:-2}
MAXLEN=${MAXLEN:-65536}
MAXSEQS=${MAXSEQS:-64}
GMU=${GMU:-0.92}
EXTRA=${EXTRA:-}
IDLE_MIN=${IDLE_MIN:-20}
PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))
HOST=${SLURMD_NODENAME:-$(hostname -s)}
EP="$ACE_ROOT/endpoints/${SLURM_JOB_ID}.json"

echo "=== $SLURM_JOB_ID on $SLURMD_NODENAME $(date) model=$MODEL pp=$PP port=$PORT ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

cleanup() { rm -f "$EP"; kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; }
trap cleanup EXIT TERM INT

"$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  --pipeline-parallel-size "$PP" \
  --distributed-executor-backend mp \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAXSEQS" \
  --gpu-memory-utilization "$GMU" \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prompt-tokens-details \
  --disable-uvicorn-access-log \
  ${EXTRA//+/ } &
PID=$!

for i in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    printf '{"url": "http://%s:%s/v1", "model": "%s", "job": "%s", "node": "%s", "pp": %s, "started": "%s"}\n' \
      "$HOST" "$PORT" "$SERVED" "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$PP" "$(date -Is)" > "$EP.tmp" && mv "$EP.tmp" "$EP"
    echo "=== healthy after $((i*5))s: $(cat $EP)"
    break
  fi
  kill -0 "$PID" 2>/dev/null || { echo "vllm died during startup"; exit 1; }
  sleep 5
done

gen_tokens() {
  curl -sf --max-time 10 "http://127.0.0.1:$PORT/metrics" 2>/dev/null |
    awk '/^vllm:generation_tokens_total/ {s += $NF} /^vllm:num_requests_running/ {r += $NF}
         END {printf "%d %d\n", s, r}'
}
last=""; idle=0
while kill -0 "$PID" 2>/dev/null; do
  sleep 60
  read -r tok running <<< "$(gen_tokens)"
  if [ -n "$tok" ] && { [ "$tok" != "$last" ] || [ "${running:-0}" -gt 0 ]; }; then
    idle=0; last=$tok
  else
    idle=$((idle + 1))
  fi
  if [ "$idle" -ge "$IDLE_MIN" ]; then
    echo "=== idle for $idle min (gen tokens $tok); releasing GPUs $(date)"
    exit 0
  fi
done
echo "=== vllm exited $(date)"

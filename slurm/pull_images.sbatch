#!/bin/bash
# Pull SWE-bench Verified eval images and convert them to SIFs.
#
#   sbatch --array=0-49%20 slurm/pull_images.sbatch [ids_file] [per_task]
#
# Task i handles lines [i*per_task, (i+1)*per_task) of ids_file (default: all
# 500 Verified ids, 10 per task -> --array=0-49). Idempotent: existing SIFs that
# pass `apptainer sif list` are skipped; new SIFs are built to <id>.sif.tmp and
# renamed atomically. Source: Epoch AI's optimized images on GHCR
# (ghcr.io/epoch-research/swe-bench.eval.x86_64.<instance_id>:latest), with a
# fallback to docker.io/swebench/sweb.eval.x86_64.<id with __ -> _1776_>.
#SBATCH --job-name=ace-sif-pull
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/usr/project/xtmp/kg349/ace/logs/pull/%A_%a.out

set -uo pipefail
ACE=/usr/project/xtmp/kg349/ace
IDS=${1:-$ACE/data/verified_ids.txt}
PER=${2:-10}
IMG=$ACE/images
mkdir -p "$IMG" "$ACE/logs/pull"

TASK=${SLURM_ARRAY_TASK_ID:-0}
START=$((TASK * PER))
# Node-local /tmp (tmpfs on most nodes) unless it is small/full, else xtmp (NFS).
FREE_GB=$(df -BG --output=avail /tmp | tail -n 1 | tr -dc 0-9)
if [ "${FREE_GB:-0}" -gt 150 ]; then BASE=/tmp; else BASE=$ACE/apptainer_tmp; fi
LOCAL=$(mktemp -d "$BASE/ace-pull-XXXXXX")
trap 'rm -rf "$LOCAL"' EXIT  # (LOCAL may be reassigned below; trap reads it at exit)
export APPTAINER_TMPDIR=$LOCAL/tmp APPTAINER_CACHEDIR=$LOCAL/cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
echo "host=$(hostname) task=$TASK start=$START per=$PER tmp_free=${FREE_GB}G build_dir=$LOCAL"

pull_one() {  # $1=id
    local id=$1 out=$IMG/$1.sif tmp=$IMG/$1.sif.tmp.$$ src
    if [ -s "$out" ] && apptainer sif list "$out" >/dev/null 2>&1; then
        echo "SKIP $id"; return 0
    fi
    local dh="docker://docker.io/swebench/sweb.eval.x86_64.${id//__/_1776_}:latest"
    for attempt in 1 2 3 4; do
        for src in "docker://ghcr.io/epoch-research/swe-bench.eval.x86_64.$id:latest" "$dh"; do
            [ "$attempt" -lt 3 ] && [ "$src" = "$dh" ] && continue  # docker hub only as late fallback
            rm -f "$tmp"
            local t0=$(date +%s)
            # build on local disk, then copy to NFS and rename atomically
            if apptainer -s pull --disable-cache "$LOCAL/$id.sif" "$src" >"$LOCAL/pull.log" 2>&1 \
               && cp "$LOCAL/$id.sif" "$tmp" && apptainer sif list "$tmp" >/dev/null 2>&1; then
                mv -f "$tmp" "$out"; rm -f "$LOCAL/$id.sif"
                echo "OK $id src=$src secs=$(( $(date +%s) - t0 )) size=$(du -h "$out" | cut -f1)"
                return 0
            fi
            echo "FAIL $id attempt=$attempt src=$src: $(tail -n 3 "$LOCAL/pull.log" | tr '\n' ' ' | cut -c1-400)"
            if grep -q "no space left" "$LOCAL/pull.log" && [ "$BASE" = /tmp ]; then
                # node /tmp is per-user quota'd/full: move the build dir to xtmp
                rm -rf "$LOCAL"; BASE=$ACE/apptainer_tmp; LOCAL=$(mktemp -d "$BASE/ace-pull-XXXXXX")
                export APPTAINER_TMPDIR=$LOCAL/tmp APPTAINER_CACHEDIR=$LOCAL/cache
                mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"; echo "switching build dir to $LOCAL"
                continue
            fi
            rm -f "$tmp" "$LOCAL/$id.sif"; rm -rf "$APPTAINER_TMPDIR"/*
        done
        sleep $(( 30 * attempt * attempt ))
    done
    echo "GIVEUP $id"; return 1
}

rc=0
sed -n "$((START + 1)),$((START + PER))p" "$IDS" | while read -r id; do
    [ -n "$id" ] && { pull_one "$id" || rc=1; }
done
echo "done task=$TASK"

#!/bin/bash
#SBATCH --job-name=ace-driver
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/usr/project/xtmp/kg349/ace/logs/%x_%j.out
#SBATCH --error=/usr/project/xtmp/kg349/ace/logs/%x_%j.out
# CPU-side driver for ACE runs (NEAT evolution or SWE prediction runs). The command comes
# in as the sbatch arguments, e.g.
#   sbatch --job-name=ace-neat-a slurm/driver.sbatch scripts/run_neat_swe.py --config ...
set -u
cd /home/users/kg349/ACEPREVIEW
export HF_HOME=/usr/project/xtmp/kg349/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
echo "=== $SLURM_JOB_ID on $SLURMD_NODENAME $(date): $*"
exec /usr/project/xtmp/kg349/ace/venv/bin/python "$@"

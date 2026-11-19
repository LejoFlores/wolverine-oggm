#!/bin/bash
#SBATCH -J oggm-collect
#SBATCH -p short
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 00:30:00
#SBATCH -o oggm-collect-%j.log
# Merges the members' output into $OGGM_ROOT/results/<experiment>/ensemble.nc and summary.csv, and lists
# any member that has no output (rerun those by resubmitting the array; finished members are skipped).
#   sbatch borah/collect.slurm experiments/<name>.yml      (borah/submit.sh queues it after the array)
set -euo pipefail
CONFIG=${1:?usage: sbatch borah/collect.slurm CONFIG.yml}
source "${OGGM_ROOT:-/bsuscratch/$USER/oggm}/env.sh"
cd "$WOLVERINE_OGGM_REPO"
"$WOLVERINE_OGGM_PYTHON" -m wolverine_oggm.hpc collect "$CONFIG"

#!/bin/bash
#SBATCH -J oggm-members
#SBATCH -p bsudfq
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o oggm-members-%A_%a.log
# One array task runs MEMBERS_PER_TASK members (default: one per CPU, all at once). Each member works on a
# private copy of the glacier directories in node-local scratch, so tasks never share files; members that
# already have output are skipped, so resubmitting the same array only reruns what failed or timed out.
# borah/submit.sh computes the array size; by hand:
#   N=$($WOLVERINE_OGGM_PYTHON -m wolverine_oggm.hpc list CONFIG --quiet)
#   sbatch --array=0-$(( (N + 7) / 8 - 1 )) borah/run_members.slurm CONFIG
set -euo pipefail
CONFIG=${1:?usage: sbatch --array=0-K borah/run_members.slurm CONFIG.yml}
source "${OGGM_ROOT:-/bsuscratch/$USER/oggm}/env.sh"
PER=${MEMBERS_PER_TASK:-$SLURM_CPUS_PER_TASK}
START=$(( ${SLURM_ARRAY_TASK_ID:-0} * PER ))
END=$(( START + PER - 1 ))
export OGGM_LOCAL_SCRATCH=${OGGM_LOCAL_SCRATCH:-${TMPDIR:-/tmp}}
echo "node $(hostname), members $START-$END, $SLURM_CPUS_PER_TASK at a time, scratch $OGGM_LOCAL_SCRATCH"
cd "$WOLVERINE_OGGM_REPO"
"$WOLVERINE_OGGM_PYTHON" -m wolverine_oggm.hpc run "$CONFIG" --members "$START-$END" --procs "$SLURM_CPUS_PER_TASK"

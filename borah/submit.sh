#!/bin/bash
# Runs one experiment end to end. On the login node it only reads the config and calls sbatch:
#   bash borah/submit.sh experiments/<name>.yml
# 1. queues borah/prepare.slurm: downloads, preprocessing and spinup of the setup (skips what is done)
# 2. queues borah/run_members.slurm as an array, after 1 succeeds
# 3. queues borah/collect.slurm after the array ends (even if some members failed)
# Extra sbatch options: MEMBERS_SBATCH for the array (e.g. "-c 16 -t 12:00:00"), PREPARE_SBATCH for step 1.
# CPUs per array task default to 8 (one member per CPU).
set -euo pipefail
CONFIG=$(readlink -f "${1:?usage: bash borah/submit.sh CONFIG.yml}")
source "${OGGM_ROOT:-/bsuscratch/$USER/oggm}/env.sh"
PY=$WOLVERINE_OGGM_PYTHON
cd "$WOLVERINE_OGGM_REPO"
LOGS=$OGGM_ROOT/logs; mkdir -p "$LOGS"

N=$("$PY" -m wolverine_oggm.hpc list "$CONFIG" --quiet)
CPUS=${MEMBERS_CPUS:-8}
NT=$(( (N + CPUS - 1) / CPUS ))
echo "$N members -> $NT array task(s) of $CPUS"

J0=$(sbatch --parsable -o "$LOGS/prepare-%j.log" ${PREPARE_SBATCH:-} borah/prepare.slurm "$CONFIG")
echo "prepare job $J0"
DEP=(--dependency=afterok:$J0)
J1=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} -c "$CPUS" --array=0-$((NT - 1)) -o "$LOGS/members-%A_%a.log" \
     ${MEMBERS_SBATCH:-} borah/run_members.slurm "$CONFIG")
J2=$(sbatch --parsable --dependency=afterany:$J1 -o "$LOGS/collect-%j.log" borah/collect.slurm "$CONFIG")
echo "members array $J1, collect $J2. Logs in $LOGS; watch with: squeue -u ${USER:-$(id -un)}"
echo "Results will be in $("$PY" -c "import sys; from wolverine_oggm import hpc; print(hpc.results_dir(hpc.load_config(sys.argv[1])))" "$CONFIG")"

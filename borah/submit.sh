#!/bin/bash
# Runs one experiment end to end. On the LOGIN node:
#   bash borah/submit.sh experiments/<name>.yml
# 1. here, on the login node: `prepare --stage fetch` (downloads; skipped if the setup is already prepared)
# 2. queues borah/spinup.slurm (offline, once per setup; skipped if done)
# 3. queues borah/run_members.slurm as an array, after 2 succeeds
# 4. queues borah/collect.slurm after the array ends (even if some members failed)
# Extra sbatch options for the array go in MEMBERS_SBATCH, e.g. MEMBERS_SBATCH="-c 16 -t 12:00:00".
# CPUs per array task default to 8 (one member per CPU).
set -euo pipefail
CONFIG=$(readlink -f "${1:?usage: bash borah/submit.sh CONFIG.yml}")
source "${OGGM_ROOT:-/bsuscratch/$USER/oggm}/env.sh"
PY=$WOLVERINE_OGGM_PYTHON
cd "$WOLVERINE_OGGM_REPO"
LOGS=$OGGM_ROOT/logs; mkdir -p "$LOGS"

"$PY" -m wolverine_oggm.hpc prepare "$CONFIG" --stage fetch

N=$("$PY" -m wolverine_oggm.hpc list "$CONFIG" --quiet)
CPUS=${MEMBERS_CPUS:-8}
NT=$(( (N + CPUS - 1) / CPUS ))
echo "$N members -> $NT array task(s) of $CPUS"

DEP=()
if ! "$PY" -c "import sys; from wolverine_oggm import hpc; c=hpc.load_config(sys.argv[1]); sys.exit(0 if hpc._stage_done(hpc.setup_dir(c),'spinup',c['setup']) else 1)" "$CONFIG"; then
  J0=$(sbatch --parsable -o "$LOGS/spinup-%j.log" borah/spinup.slurm "$CONFIG")
  echo "spinup job $J0"
  DEP=(--dependency=afterok:$J0)
fi
J1=$(sbatch --parsable "${DEP[@]}" -c "$CPUS" --array=0-$((NT - 1)) -o "$LOGS/members-%A_%a.log" \
     ${MEMBERS_SBATCH:-} borah/run_members.slurm "$CONFIG")
J2=$(sbatch --parsable --dependency=afterany:$J1 -o "$LOGS/collect-%j.log" borah/collect.slurm "$CONFIG")
echo "members array $J1, collect $J2. Logs in $LOGS; watch with: squeue -u ${USER:-$(id -un)}"
echo "Results will be in $("$PY" -c "import sys; from wolverine_oggm import hpc; print(hpc.results_dir(hpc.load_config(sys.argv[1])))" "$CONFIG")"

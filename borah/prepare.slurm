#!/bin/bash
#SBATCH -J oggm-prepare
#SBATCH -p short
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -o oggm-prepare-%j.log
# Prepares the experiment's setup on a compute node: downloads (outlines, DEM tiles, climate, calibration data,
# ISIMIP3b files), DEM gridding, flowlines, MB calibration, inversion, then the historical run and dynamic
# spinup. Steps already done for this setup are skipped, so queueing it again is cheap. One process per
# glacier up to -c; a few hundred glaciers want more, e.g.  sbatch -p bsudfq -c 32 -t 6:00:00 ...
# Needs internet from the compute node (Borah's short-queue nodes reach it: schnaps-borah's
# build_domain.slurm downloads there too).
#   sbatch borah/prepare.slurm experiments/<name>.yml      (borah/submit.sh does this for you)
set -euo pipefail
CONFIG=${1:?usage: sbatch borah/prepare.slurm CONFIG.yml}
source "${OGGM_ROOT:-/bsuscratch/$USER/oggm}/env.sh"
curl -s -o /dev/null -m 30 https://cluster.klima.uni-bremen.de/~oggm/ || {
  echo "$(hostname) cannot reach cluster.klima.uni-bremen.de (OGGM's data server). Try another partition:"
  echo "  sbatch -p <partition> borah/prepare.slurm $CONFIG"; exit 1; }
cd "$WOLVERINE_OGGM_REPO"
"$WOLVERINE_OGGM_PYTHON" -m wolverine_oggm.hpc prepare "$CONFIG" --stage all

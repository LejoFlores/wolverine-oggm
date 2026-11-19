#!/bin/bash
#SBATCH -J oggm-spinup
#SBATCH -p short
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -o oggm-spinup-%j.log
# Offline half of `prepare`: historical run and dynamic spinup of the experiment's setup, with one
# process per glacier (OGGM's multiprocessing) up to -c. Wolverine alone at 25 m takes under a minute;
# a few hundred glaciers want more cores and time, e.g.  sbatch -p bsudfq -c 32 -t 4:00:00 ...
#   sbatch borah/spinup.slurm experiments/<name>.yml      (borah/submit.sh does this for you)
set -euo pipefail
CONFIG=${1:?usage: sbatch borah/spinup.slurm CONFIG.yml}
source "${OGGM_ROOT:-/bsuscratch/$USER/oggm}/env.sh"
cd "$WOLVERINE_OGGM_REPO"
"$WOLVERINE_OGGM_PYTHON" -m wolverine_oggm.hpc prepare "$CONFIG" --stage spinup

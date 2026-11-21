#!/bin/bash
#SBATCH -J oggm-install
#SBATCH -p short
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 01:30:00
#SBATCH -o oggm-install-%j.log
# The whole install, on a compute node (short queue). Nothing runs on the login node.
#   sbatch borah/install.slurm
# 1. Runs borah/fetch_env.sh here unless an earlier run finished it (FETCH=1 forces it): solves the env for
#    glibc 2.17 and downloads the packages, the OGGM wheel and the sample data into $OGGM_ROOT/install/.
#    This needs internet from the compute node (short-queue nodes have it).
# 2. Creates (or recreates) the env at $WOLVERINE_OGGM_PREFIX (default ~/envs/wolverine-oggm) from those files
#    with the same standalone micromamba (explicit list: no solver, no internet; your own conda is not used),
#    installs OGGM and this repo with pip from local files, unpacks the sample data into ~/.oggm and
#    ~/.salem_cache, writes $OGGM_ROOT/env.sh (sourced by every other script here) and tests the env.
set -euo pipefail
REPO=$(cd "${SLURM_SUBMIT_DIR:-.}" && git rev-parse --show-toplevel)
PREFIX=${WOLVERINE_OGGM_PREFIX:-$HOME/envs/wolverine-oggm}
OGGM_ROOT=${OGGM_ROOT:-/bsuscratch/$USER/oggm}
STAGE=$OGGM_ROOT/install
if [ ! -s "$STAGE/fetched.txt" ] || [ "${FETCH:-0}" = 1 ]; then
  curl -s -o /dev/null -m 30 https://conda.anaconda.org/conda-forge/ || {
    echo "$(hostname) cannot reach conda.anaconda.org. Run  bash borah/fetch_env.sh  where there is internet, then resubmit."
    exit 1; }
  bash "$REPO/borah/fetch_env.sh"
fi
echo "node $(hostname)  $(ldd --version | head -1)"

MM=$OGGM_ROOT/bin/micromamba
export MAMBA_ROOT_PREFIX=$OGGM_ROOT/micromamba
if [ -d "$PREFIX" ]; then
  echo "replacing the existing env at $PREFIX"
  rm -rf "$PREFIX"
fi
"$MM" create -y -p "$PREFIX" --offline -f "$STAGE/explicit.txt"
PY=$PREFIX/bin/python
"$PY" -m pip install -q --no-index --no-deps "$STAGE"/wheels/oggm-1.6.3-*.whl
"$PY" -m pip install -q --no-index --no-deps --no-build-isolation -e "$REPO"

# sample data where OGGM (~/.oggm) and salem (~/.salem_cache) look for it
mkdir -p "$HOME/.oggm" "$HOME/.salem_cache"
for z in "$STAGE"/sample/oggm-sample-data-*.zip; do "$PY" -m zipfile -e "$z" "$HOME/.oggm/"; done
for z in "$STAGE"/sample/salem-sample-data-*.zip; do "$PY" -m zipfile -e "$z" "$HOME/.salem_cache/"; done

mkdir -p "$OGGM_ROOT/download_cache"
cat > "$OGGM_ROOT/env.sh" <<EOF
# Written by borah/install.slurm on $(date +%F). Sourced by the scripts in borah/.
export WOLVERINE_OGGM_PYTHON=$PY
export WOLVERINE_OGGM_REPO=$REPO
export OGGM_ROOT=$OGGM_ROOT
export OGGM_DOWNLOAD_CACHE=$OGGM_ROOT/download_cache
# one thread per process: parallelism comes from running several members (or glaciers) at once
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
EOF
source "$OGGM_ROOT/env.sh"

"$PY" - <<EOF | tee "$OGGM_ROOT/install_manifest.txt"
import platform, sys
import salem, oggm, numpy, xarray, rasterio, geopandas, netCDF4
from oggm import cfg, utils
cfg.initialize(logging_level="CRITICAL")     # would download sample data if it were missing
cfg.PARAMS["has_internet"] = False
utils.get_demo_file("hef_srtm.tif")
print("date", "$(date '+%F %T')", "job", "${SLURM_JOB_ID:-none}")
print("host", platform.node(), "glibc", platform.libc_ver()[1])
print("python", sys.version.split()[0], "at", sys.executable)
for m in (oggm, numpy, xarray, rasterio, salem, geopandas, netCDF4):
    print(m.__name__, m.__version__)
EOF
git -C "$REPO" rev-parse --short HEAD | sed 's/^/wolverine-oggm commit /' >> "$OGGM_ROOT/install_manifest.txt"
cd "$REPO"
WOLVERINE_OGGM_TEST=1 "$PY" -m pytest -q -p no:warnings tests
"$MM" clean -y --all >/dev/null 2>&1 || true     # unpacked package cache
echo "install OK: $PREFIX"

#!/bin/bash
#SBATCH -J oggm-install
#SBATCH -p short
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:45:00
#SBATCH -o oggm-install-%j.log
# Step 2 of 2 of the install, on a compute node, offline. Run borah/fetch_env.sh on the login node first.
#   sbatch borah/install.slurm
# Creates (or recreates) the conda env "wolverine-oggm" from the downloaded packages (explicit list, so no
# solver and no internet), installs OGGM and this repo with pip from local files, unpacks the sample data
# into ~/.oggm and ~/.salem_cache, writes $OGGM_ROOT/env.sh (sourced by every other script here) and tests
# the env. Settings: OGGM_ROOT, WOLVERINE_OGGM_ENV (default wolverine-oggm), CONDA.
set -euo pipefail
REPO=$(cd "${SLURM_SUBMIT_DIR:-.}" && git rev-parse --show-toplevel)
ENV=${WOLVERINE_OGGM_ENV:-wolverine-oggm}
OGGM_ROOT=${OGGM_ROOT:-/bsuscratch/$USER/oggm}
STAGE=$OGGM_ROOT/install
[ -s "$STAGE/fetched.txt" ] || { echo "No complete download in $STAGE: run bash borah/fetch_env.sh on the login node first"; exit 1; }
echo "node $(hostname)  $(ldd --version | head -1)"

CONDA=${CONDA:-$(command -v conda || true)}
[ -n "$CONDA" ] || CONDA=$HOME/miniforge3/bin/conda
export CONDA_PKGS_DIRS=$STAGE/conda_pkgs_cache     # unpack here, not in ~ (quota)
if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV"; then
  echo "removing the existing env $ENV"
  "$CONDA" env remove -y -n "$ENV"
fi
"$CONDA" create -y -n "$ENV" --offline --file "$STAGE/explicit.txt"
PREFIX=$("$CONDA" env list | awk -v e="$ENV" '$1==e {print $NF}')
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
rm -rf "$CONDA_PKGS_DIRS"
echo "install OK: $PREFIX"

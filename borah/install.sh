#!/bin/bash
# Step 1 of the Borah workflow, on the LOGIN node (needs internet; compiles nothing).
#   bash borah/install.sh
# Creates (or updates) the conda env "wolverine-oggm", downloads the sample data that OGGM and salem fetch
# on first import (compute nodes have no internet, and OGGM's cfg.initialize() downloads it), and writes
# $OGGM_ROOT/env.sh, which every Slurm script here sources.
#
# Why the env is built this way:
#  - OGGM is pure Python, so nothing is compiled. The compiled pieces (numpy, GDAL/rasterio, HDF5/netCDF,
#    PROJ, scipy) are conda-forge binaries. Borah's compute nodes run glibc 2.17 and the login node a newer
#    one, so CONDA_OVERRIDE_GLIBC=2.17 makes the solver pick builds that also run on compute nodes.
#  - OGGM is not on conda-forge; it comes from PyPI (pip section of environment.yml).
#  - The Slurm scripts call the env's python by its path instead of "conda activate", so the base conda
#    installation never has to run on a compute node.
# Settings (environment variables): OGGM_ROOT (default /bsuscratch/$USER/oggm), WOLVERINE_OGGM_ENV
# (env name, default wolverine-oggm), CONDA (path to mamba or conda; default: found on PATH or ~/miniforge3).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV=${WOLVERINE_OGGM_ENV:-wolverine-oggm}
OGGM_ROOT=${OGGM_ROOT:-/bsuscratch/$USER/oggm}

CONDA=${CONDA:-$(command -v mamba || command -v conda || true)}
for c in "$HOME/miniforge3/bin/mamba" "$HOME/miniforge3/bin/conda"; do
  [ -n "$CONDA" ] || { [ -x "$c" ] && CONDA=$c; } || true
done
if [ -z "$CONDA" ]; then
  echo "No conda found: installing Miniforge in ~/miniforge3"
  curl -sSL -o /tmp/Miniforge3-$USER.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/Miniforge3-$USER.sh -b -p "$HOME/miniforge3"
  rm -f /tmp/Miniforge3-$USER.sh
  CONDA=$HOME/miniforge3/bin/mamba
fi
echo "Using $CONDA ($($CONDA --version 2>&1 | head -1))"

export CONDA_OVERRIDE_GLIBC=2.17
cd "$REPO"      # environment.yml installs this repo with "pip install -e ."
if $CONDA env list | awk '{print $1}' | grep -qx "$ENV"; then
  $CONDA env update -n "$ENV" -f environment.yml
else
  $CONDA env create -n "$ENV" -f environment.yml          # no -y: old conda-env rejects it
fi
PREFIX=$($CONDA env list | awk -v e="$ENV" '$1==e {print $NF}')
PY=$PREFIX/bin/python
[ -x "$PY" ] || { echo "env python not found at $PY"; exit 1; }

mkdir -p "$OGGM_ROOT/download_cache"
cat > "$OGGM_ROOT/env.sh" <<EOF
# Written by borah/install.sh on $(date +%F). Sourced by the Slurm scripts in borah/.
export WOLVERINE_OGGM_PYTHON=$PY
export WOLVERINE_OGGM_REPO=$REPO
export OGGM_ROOT=$OGGM_ROOT
export OGGM_DOWNLOAD_CACHE=$OGGM_ROOT/download_cache
# one thread per process: parallelism comes from running several members (or glaciers) at once
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
EOF
source "$OGGM_ROOT/env.sh"

# Sample data (OGGM's into ~/.oggm, salem's into ~/.salem_cache): fetched once here
"$PY" - <<'EOF'
import salem  # noqa: F401  (downloads salem-sample-data on first import)
from oggm import cfg, utils
cfg.initialize(logging_level="CRITICAL")       # downloads oggm-sample-data on first call
utils.get_demo_file("hef_srtm.tif")
print("sample data OK")
EOF

"$PY" - <<EOF > "$OGGM_ROOT/install_manifest.txt"
import platform, sys, oggm, numpy, xarray, rasterio, salem, geopandas, netCDF4
print("date", "$(date '+%F %T')")
print("host", platform.node(), "glibc", platform.libc_ver()[1])
print("python", sys.version.split()[0], "at", sys.executable)
for m in (oggm, numpy, xarray, rasterio, salem, geopandas, netCDF4):
    print(m.__name__, m.__version__)
EOF
cat "$OGGM_ROOT/install_manifest.txt"
git -C "$REPO" rev-parse --short HEAD | sed 's/^/wolverine-oggm commit /' >> "$OGGM_ROOT/install_manifest.txt"

echo
echo "Installed. Next, check it on a compute node (offline):  sbatch borah/check.slurm"

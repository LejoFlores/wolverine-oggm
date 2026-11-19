#!/bin/bash
#SBATCH -J oggm-check
#SBATCH -p short
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 00:20:00
#SBATCH -o oggm-check-%j.log
# Optional re-check (install.slurm already runs the tests): proves the env works on a compute node with no internet:
# glibc, imports, and the unit tests on OGGM's offline sample glacier.
#   sbatch borah/check.slurm
set -euo pipefail
source "${OGGM_ROOT:-/bsuscratch/$USER/oggm}/env.sh"
PY=$WOLVERINE_OGGM_PYTHON
echo "node $(hostname)  $(ldd --version | head -1)"
"$PY" -c "import oggm, numpy, rasterio, salem, netCDF4, geopandas, yaml; print('oggm', oggm.__version__, 'numpy', numpy.__version__, 'rasterio', rasterio.__version__)"
echo -n "internet from this node: "; curl -s -o /dev/null -m 10 -w '%{http_code}\n' https://cluster.klima.uni-bremen.de || echo "none"
cd "$WOLVERINE_OGGM_REPO"
WOLVERINE_OGGM_TEST=1 "$PY" -m pytest -q tests
echo "check OK"

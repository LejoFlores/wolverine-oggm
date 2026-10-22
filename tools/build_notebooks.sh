#!/usr/bin/env bash
# Notebooks are written as jupytext "percent" scripts (notebooks/*.py), which diff cleanly in git.
# This converts them to .ipynb, and with --execute runs each one top to bottom (in test mode
# unless WOLVERINE_OGGM_TEST is already set), failing on the first error.
#   bash tools/build_notebooks.sh            # .py -> .ipynb, no outputs
#   bash tools/build_notebooks.sh --execute  # also run them (offline test glacier by default)
set -euo pipefail
cd "$(dirname "$0")/../notebooks"
for py in [0-9][0-9]_*.py; do
  jupytext --quiet --to ipynb "$py"
done
if [[ "${1:-}" == "--execute" ]]; then
  export WOLVERINE_OGGM_TEST="${WOLVERINE_OGGM_TEST:-1}"
  for nb in [0-9][0-9]_*.ipynb; do
    echo "== $nb (WOLVERINE_OGGM_TEST=$WOLVERINE_OGGM_TEST)"
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 \
        --output "/tmp/executed_$nb" "$nb"
  done
fi

#!/bin/bash
# Download half of the install: installs and compiles nothing. borah/install.slurm runs it on a compute node
# for you; run it by hand only where a compute node has no internet.
#
# It does not use your conda. It downloads a standalone micromamba (one binary, pinned below) into
# $OGGM_ROOT/bin and solves the env with that, so an old conda installation doesn't matter and is not
# touched. Into $OGGM_ROOT/install/ (default /bsuscratch/$USER/oggm/install) it puts:
#   pkgs/          the conda-forge packages for the env, solved for the compute nodes' glibc 2.17
#                  (CONDA_OVERRIDE_GLIBC), so they run there even if solved on a newer machine
#   explicit.txt   the exact package list, which install.slurm installs with no solver and no internet
#   wheels/        OGGM 1.6.3 from PyPI (it is not on conda-forge; pure Python)
#   sample/        the sample-data archives OGGM and salem download on first use
# Settings: OGGM_ROOT.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OGGM_ROOT=${OGGM_ROOT:-/bsuscratch/$USER/oggm}
STAGE=$OGGM_ROOT/install
mkdir -p "$STAGE/pkgs" "$STAGE/wheels" "$STAGE/sample" "$OGGM_ROOT/bin"
PY3=$(command -v python3)      # the system python, only to read JSON and zip files

# 0. micromamba 2.10.0 from conda-forge (built for glibc 2.17; needs only system libraries)
MM=$OGGM_ROOT/bin/micromamba
MM_URL=https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.10.0-0.tar.bz2
MM_SHA=9a97976244929bbae28fb1fae504e74a3e7e6d1932870e3a03321f01b7fbd07b
if [ ! -x "$MM" ]; then
  curl -sSfL --retry 4 -o "$STAGE/micromamba.tar.bz2" "$MM_URL"
  echo "$MM_SHA  $STAGE/micromamba.tar.bz2" | sha256sum -c --quiet -
  tar -xjf "$STAGE/micromamba.tar.bz2" -C "$OGGM_ROOT" bin/micromamba
  rm -f "$STAGE/micromamba.tar.bz2"
fi
export MAMBA_ROOT_PREFIX=$OGGM_ROOT/micromamba     # micromamba's own cache, nothing in ~
echo "Using $MM $("$MM" --version)"

# 1. Solve (metadata only) for glibc 2.17 and list the packages
SPECS=$("$PY3" - "$REPO/environment.yml" <<'EOF'
import sys
out, on = [], False
for line in open(sys.argv[1]):
    s = line.strip()
    if s.startswith("dependencies:"):
        on = True; continue
    if on and s.startswith("- pip:"):
        break
    if on and s.startswith("- ") and not s.startswith("- pip"):
        out.append(s[2:].split("#")[0].strip())
print(" ".join(out + ["pip", "setuptools"]))
EOF
)
echo "Solving: $SPECS"
CONDA_OVERRIDE_GLIBC=2.17 "$MM" create -p "$STAGE/solve_prefix" --dry-run --json \
    --override-channels -c conda-forge $SPECS > "$STAGE/solve.json"
"$PY3" - "$STAGE" <<'EOF'
import json, sys, os
stage = sys.argv[1]
d = json.load(open(os.path.join(stage, "solve.json")))
if not d.get("success", True) or "actions" not in d:
    sys.exit("solve failed:\n" + json.dumps(d, indent=1)[:3000])
pk = sorted(d["actions"]["LINK"], key=lambda p: p["name"])
with open(os.path.join(stage, "urls.txt"), "w") as f:
    for p in pk:
        f.write("{} {}\n".format(p["url"], p.get("sha256", "")))
with open(os.path.join(stage, "explicit.txt"), "w") as f:
    f.write("# conda-forge packages for wolverine-oggm, solved for glibc 2.17\n@EXPLICIT\n")
    for p in pk:
        f.write("file://{}/pkgs/{}#{}\n".format(stage, p["fn"], p["md5"]))
print("{} packages, {:.0f} MB".format(len(pk), sum(p.get("size", 0) for p in pk) / 1e6))
EOF

# 2. Download the packages (skips ones already there with the right checksum)
n=0
while read -r url sha; do
  f=$STAGE/pkgs/$(basename "$url")
  if [ -s "$f" ] && [ -n "$sha" ] && echo "$sha  $f" | sha256sum -c --quiet - 2>/dev/null; then continue; fi
  curl -sSfL --retry 4 -o "$f" "$url"
  [ -z "$sha" ] || echo "$sha  $f" | sha256sum -c --quiet - || { echo "checksum mismatch: $url"; exit 1; }
  n=$((n + 1))
done < "$STAGE/urls.txt"
echo "downloaded $n package(s) into $STAGE/pkgs"

# 3. OGGM's wheel from PyPI, and the sample data OGGM and salem would otherwise fetch on first import
pypi_wheel() {   # name version outdir: the pure-Python wheel of that release
  local url
  url=$(curl -sSfL --retry 4 "https://pypi.org/pypi/$1/$2/json" | "$PY3" -c '
import json, sys
print([f["url"] for f in json.load(sys.stdin)["urls"] if f["filename"].endswith("-none-any.whl")][0])')
  curl -sSfL --retry 4 -o "$3/$(basename "$url")" "$url"
}
pypi_wheel oggm 1.6.3 "$STAGE/wheels"
SALEM_VER=$(grep -o 'conda-forge/noarch/salem-[0-9.]*-' "$STAGE/urls.txt" | sed 's/.*salem-//; s/-$//')
pypi_wheel salem "$SALEM_VER" "$STAGE/sample"
read -r OGGM_COMMIT SALEM_COMMIT < <("$PY3" - "$STAGE" <<'EOF'
import glob, re, sys, zipfile
st = sys.argv[1]
z = zipfile.ZipFile(glob.glob(st + "/wheels/oggm-1.6.3-*.whl")[0])
o = re.search(r"SAMPLE_DATA_COMMIT = '(\w+)'", z.read("oggm/utils/_downloads.py").decode()).group(1)
z = zipfile.ZipFile(glob.glob(st + "/sample/salem-*.whl")[0])
s = re.search(r"sample_data_gh_commit = '(\w+)'", z.read("salem/__init__.py").decode()).group(1)
print(o, s)
EOF
)
get_archive() {   # repo commit name: GitHub's zip, or a git clone if the zip download is refused
  local out=$STAGE/sample/$3-$2.zip tmp
  [ -s "$out" ] && return 0
  curl -sSfL --retry 4 -o "$out" "https://github.com/$1/archive/$2.zip" && return 0
  rm -f "$out"; tmp=$(mktemp -d)
  git clone -q "https://github.com/$1.git" "$tmp/r" && git -C "$tmp/r" archive --prefix="$3-$2/" -o "$out" "$2"
  rm -rf "$tmp"
}
get_archive OGGM/oggm-sample-data "$OGGM_COMMIT" oggm-sample-data
get_archive fmaussion/salem-sample-data "$SALEM_COMMIT" salem-sample-data
rm -f "$STAGE"/sample/salem-*.whl
echo "$(date '+%F %T') fetched on $(hostname): $(wc -l < "$STAGE/urls.txt") conda packages, oggm 1.6.3, sample data" > "$STAGE/fetched.txt"
du -sh "$STAGE"
echo "Downloads done."

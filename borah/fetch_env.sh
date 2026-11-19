#!/bin/bash
# Step 1 of 2 of the install, on the Borah LOGIN node: downloads only. It installs and compiles nothing.
#   bash borah/fetch_env.sh
#   sbatch borah/install.slurm        # step 2, on a compute node (short queue), offline
#
# Downloads into $OGGM_ROOT/install/ (default /bsuscratch/$USER/oggm/install):
#   pkgs/          the conda-forge packages for the env, solved for the compute nodes' glibc 2.17
#                  (CONDA_OVERRIDE_GLIBC), so they run there even though the login node is newer
#   explicit.txt   the exact package list, which install.slurm installs with no solver and no internet
#   wheels/        OGGM 1.6.3 from PyPI (it is not on conda-forge; pure Python)
#   sample/        the sample-data archives OGGM and salem download on first use
# Settings: OGGM_ROOT, CONDA (path to conda or mamba; default: on PATH or ~/miniforge3).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OGGM_ROOT=${OGGM_ROOT:-/bsuscratch/$USER/oggm}
STAGE=$OGGM_ROOT/install
mkdir -p "$STAGE/pkgs" "$STAGE/wheels" "$STAGE/sample"

CONDA=${CONDA:-$(command -v mamba || command -v conda || true)}
for c in "$HOME/miniforge3/bin/mamba" "$HOME/miniforge3/bin/conda"; do
  [ -n "$CONDA" ] || { [ -x "$c" ] && CONDA=$c; } || true
done
[ -n "$CONDA" ] || { echo "No conda found. Install Miniforge in ~/miniforge3 first (see borah/README.md)."; exit 1; }
BASEPY=$(dirname "$(readlink -f "$CONDA")")/python
[ -x "$BASEPY" ] || BASEPY=python3
echo "Using $CONDA"

# 1. Solve (metadata only) for glibc 2.17 and list the packages
SPECS=$("$BASEPY" - "$REPO/environment.yml" <<'EOF'
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
CONDA_OVERRIDE_GLIBC=2.17 "$CONDA" create -n __wolverine_oggm_solve --dry-run --json \
    --override-channels -c conda-forge $SPECS > "$STAGE/solve.json"
"$CONDA" info --json > "$STAGE/conda_info.json"
"$BASEPY" - "$STAGE" <<'EOF'
import json, sys, os
stage = sys.argv[1]
pkgs_dirs = json.load(open(os.path.join(stage, "conda_info.json"))).get("pkgs_dirs", [])
d = json.load(open(os.path.join(stage, "solve.json")))
if not d.get("success", True) or "actions" not in d:
    sys.exit("solve failed:\n" + json.dumps(d, indent=1)[:3000])
# mamba lists full records (url, md5) under LINK. conda lists them under FETCH for packages it would
# download, and only names under LINK; packages already in its cache have a repodata_record.json there.
recs = {p["name"]: p for p in d["actions"]["LINK"] if "url" in p}
recs.update({p["name"]: p for p in d["actions"].get("FETCH", [])})
for p in d["actions"]["LINK"]:
    if p["name"] in recs:
        continue
    for pd in pkgs_dirs:
        f = os.path.join(pd, p.get("dist_name", ""), "info", "repodata_record.json")
        if os.path.exists(f):
            recs[p["name"]] = json.load(open(f))
            break
missing = [p["name"] for p in d["actions"]["LINK"] if p["name"] not in recs]
if missing:
    sys.exit(f"no download record for: {missing}")
pk = [recs[p["name"]] for p in sorted(d["actions"]["LINK"], key=lambda p: p["name"])]
with open(os.path.join(stage, "urls.txt"), "w") as f:
    for p in pk:
        f.write(f"{p['url']} {p.get('sha256', '')}\n")
with open(os.path.join(stage, "explicit.txt"), "w") as f:
    f.write("# conda-forge packages for wolverine-oggm, solved for glibc 2.17\n@EXPLICIT\n")
    for p in pk:
        f.write(f"file://{stage}/pkgs/{p['fn']}#{p['md5']}\n")
print(f"{len(pk)} packages, {sum(p.get('size', 0) for p in pk) / 1e6:.0f} MB")
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

# 3. OGGM from PyPI, and the sample data OGGM and salem would otherwise fetch on first import
"$BASEPY" -m pip download -q --no-deps --only-binary=:all: -d "$STAGE/wheels" oggm==1.6.3
SALEM_VER=$(grep -o 'conda-forge/noarch/salem-[0-9.]*-' "$STAGE/urls.txt" | sed 's/.*salem-//; s/-$//')
"$BASEPY" -m pip download -q --no-deps --only-binary=:all: -d "$STAGE/sample" "salem==$SALEM_VER"
read -r OGGM_COMMIT SALEM_COMMIT < <("$BASEPY" - "$STAGE" <<'EOF'
import glob, re, sys, zipfile
st = sys.argv[1]
z = zipfile.ZipFile(glob.glob(f"{st}/wheels/oggm-1.6.3-*.whl")[0])
o = re.search(r"SAMPLE_DATA_COMMIT = '(\w+)'", z.read("oggm/utils/_downloads.py").decode()).group(1)
z = zipfile.ZipFile(glob.glob(f"{st}/sample/salem-*.whl")[0])
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
echo "Downloads done. Next, on a compute node:  sbatch borah/install.slurm"

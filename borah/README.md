# OGGM on Borah

Scripts to install OGGM on Boise State's Borah cluster and run large Wolverine experiments there:
finer grids, longer runs, many ensemble members, CMIP6 projections, and many glaciers at once.
The notebooks stay the place to learn the model and try ideas on one run; this is for running the
same experiments at scale.

## Quick start

```bash
git clone https://github.com/LejoFlores/wolverine-oggm.git ~/wolverine-oggm
cd ~/wolverine-oggm
sbatch borah/install.slurm                        # short queue: downloads, builds the env, tests it
bash borah/submit.sh experiments/smoke_test.yml   # only reads the config and calls sbatch
```

Nothing heavy runs on the login node. `install.slurm` and the `prepare` job that `submit.sh`
queues do their downloads from a compute node in the `short` queue, which has internet (as
schnaps-borah's `build_domain.slurm` found). `submit.sh` prints the job ids. When the jobs finish, the results are in
`/bsuscratch/$USER/oggm/results/smoke_test/`.

## How it is set up

**Nothing is compiled.** OGGM is pure Python. The compiled libraries it uses (numpy, GDAL and
rasterio, HDF5 and netCDF, PROJ, scipy) are conda-forge binaries. Two Borah facts shape the
install:

* The install doesn't use your conda. `install.slurm` calls `fetch_env.sh`, which downloads a
  standalone micromamba (one pinned binary) into `$OGGM_ROOT/bin`. That solves the package
  list and downloads the packages, the OGGM wheel and the sample data into
  `$OGGM_ROOT/install/`. Then the same micromamba builds the env at `~/envs/wolverine-oggm` from
  that exact list and those local files, and the job runs the tests. An old conda installation
  doesn't matter and is not touched. If a compute node ever lacks internet, run
  `bash borah/fetch_env.sh` somewhere that has it, then `sbatch borah/install.slurm`.
* Compute nodes run glibc 2.17. `fetch_env.sh` sets `CONDA_OVERRIDE_GLIBC=2.17`, so the
  solver only picks builds that run there, wherever the solve happens.
  A solve with that setting on 2026-11-19 gave Python 3.11, numpy 2.4, GDAL 3.12, rasterio 1.4,
  HDF5 2.2 and netCDF 4.10, all built for glibc 2.17.
* OGGM downloads on demand: sample data on its first `cfg.initialize()`, then DEMs, climate,
  calibration data and GCM files. All of that happens in two places, the install and the
  `prepare` job. The member runs work on local copies with OGGM's `has_internet = False`, so a
  missing file fails at once instead of a download starting in the middle of an array.

OGGM is not on conda-forge. It comes from PyPI (`oggm==1.6.3`, the pip section of
`environment.yml`). The `environment.yml` on main before this change could not be installed,
because it asked conda for OGGM.

The Slurm scripts call the env's Python by its full path. That path is written to
`$OGGM_ROOT/env.sh` by `install.slurm`. For Jupyter, run `~/envs/wolverine-oggm/bin/jupyter lab`.

Everything goes under `OGGM_ROOT` (default `/bsuscratch/$USER/oggm`):

| Path | What |
| --- | --- |
| `bin/micromamba` | Standalone micromamba 2.10.0 used by the install |
| `install/` | Downloads from `fetch_env.sh`: conda packages, `explicit.txt`, OGGM wheel, sample data |
| `env.sh` | Written by `install.slurm`. Python path, `OGGM_ROOT`, download cache, one thread per process |
| `install_manifest.txt` | Versions, host and glibc of the install |
| `download_cache/` | OGGM's download cache (`OGGM_DOWNLOAD_CACHE`): DEM tiles, W5E5, geodetic MB, ISIMIP3b... |
| `gdirs/<setup>/` | Prepared glacier directories, shared by every experiment that names this setup |
| `results/<experiment>/` | `members.csv`, `experiment.yml`, `members/`, `ensemble.nc`, `summary.csv` |
| `logs/` | Job logs from `submit.sh` |

## Experiments

An experiment is a YAML file in `experiments/`. It has two parts:

* A **setup**: which glaciers, the grid, the DEM, the climate and whether to spin up. It is
  prepared once, and every experiment that names it shares it.
* The **factors**: the phase scheme, warming, warm-precipitation offset, random-climate seed or
  GCM and SSP, and sliding coupling.

Each combination of factors is one **member**, which is one run for each glacier. The members
run as a Slurm array.

| File | Members | What |
| --- | --- | --- |
| `smoke_test.yml` | 4 | Default grid, 30 years. Run this one first |
| `h1_h3_phase_warming_25m.yml` | 75 | H1/H3 at 25 m: 5 phase schemes × 3 warmings × 5 seeds, 300 years |
| `h4_warm_precip_25m.yml` | 80 | H4: phase computed at T + 0 to 2 K, two schemes, two warmings |
| `h5_sliding_25m.yml` | 54 | H5: water-driven sliding, monthly output for the seasonal cycle |
| `projections_isimip3b_25m.yml` | 45 | 5 ISIMIP3b GCMs × 3 SSPs × 3 phase schemes, 2020 to 2100 |
| `regional_kenai_projections.yml` | 15 | Every glacier ≥ 0.1 km² in a Kenai Mountains box, 5 GCMs × 3 SSPs |

Setup keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | experiment name | Directory under `gdirs/`. Experiments that share it must use identical setup settings |
| `glaciers` | `[RGI60-01.09162]` | A list of RGI 6.2 ids, or `{bbox: [lon0, lat0, lon1, lat1], min_area_km2: 0.1}` (glacier centres in the box) |
| `dx` | `null` | Map grid spacing in m. `null` is OGGM's size-based default, 67 m for Wolverine |
| `flowline_dx` | `2` | Flowline spacing in grid cells. With `dx: 25, flowline_dx: 1` the model runs at 25 m, compared with 134 m by default |
| `dem` | OGGM's choice | `COPDEM30`, `COPDEM90`, `NASADEM`, ... or a GeoTIFF path, e.g. an IfSAR or lidar DTM. Without `dx` the default is the DEM OGGM chose for the glacier (COPDEM90 for Wolverine); with `dx` < 60 m it is COPDEM30 |
| `climate` | `GSWP3_W5E5` | OGGM baseline climate, 1901 to 2019. `ERA5` reaches later years |
| `spinup` | `true` | Dynamic melt_f calibration from 1979, as in OGGM's own level-5 directories |
| `shop` | `[]` | Observations binned on the flowline: `millan_velocity`, `itslive_velocity`, `millan_thickness`, `consensus_thickness`, `hugonnet_dhdt` |
| `params` | `{}` | Any other `cfg.PARAMS` value, e.g. `{glen_a: 3.0e-24}` |

Scenarios: `random` (years drawn from the 2000–2019 calibration period, with `nyears` and `seeds`),
`historical` (from the spun-up glacier in 2000 to the end of the climate data), or `gcm`
(ISIMIP3b `gcms` × `ssps` up to `ye`).

Factors:
* `phase`: `oggm`, `threshold0/1/2`, `jennings90`, `logistic:T50:width`, `threshold:T`,
  `ramp:Tsolid:Tliquid`, or a CSV with columns `t, frac`, such as a SCHNAPS-derived curve from
  notebook 05.
* `temp_std`: `file` (W5E5's daily spread), a number in K, or `null`.
* `recalibrate`: default true.
* `temperature_bias`, `precip_temp_offset`.
* `sliding`: `{fraction, coupling}` for `HydroSliding`.

With `recalibrate: true`, every scheme refits `melt_f` on fixed geometry to the same geodetic
MB. That includes OGGM's ramp, so all schemes are compared on equal terms (as in notebook 03).

## Steps, if you run them by hand

```bash
source /bsuscratch/$USER/oggm/env.sh; PY=$WOLVERINE_OGGM_PYTHON
$PY -m wolverine_oggm.hpc show    experiments/X.yml                   # parsed config, paths, member count
sbatch borah/prepare.slurm experiments/X.yml                          # downloads, preprocessing, spinup
$PY -m wolverine_oggm.hpc list    experiments/X.yml                   # writes members.csv
sbatch --array=0-9 borah/run_members.slurm experiments/X.yml          # 8 members per task by default
sbatch borah/collect.slurm experiments/X.yml                          # ensemble.nc, summary.csv
```

Each member copies the setup's glacier directories to node-local scratch and runs there. Then
it copies back these files:

* `m####.nc`: OGGM's compiled output, with volume, area and length per glacier.
* `m####_summary.csv`: one row per glacier and year, with mean and max surface velocity on
  the main flowline.
* `m####_<rgi_id>_fl_diagnostics.nc`: thickness and velocity along the flowline over time.
* `m####.json`: the factors, the calibration used, the run time and the job id.

Resubmitting an array skips members that already have output. So a timed-out or failed task
only reruns what is missing. `collect` lists any missing members.

## Reading the results

```python
import xarray as xr, pandas as pd
root = "/bsuscratch/USER/oggm/results/h1_h3_phase_warming_25m"
ens = xr.open_dataset(f"{root}/ensemble.nc")            # dims member, time, rgi_id; factors are member coords
vol = ens.volume.sum("rgi_id").groupby(ens.phase)       # e.g. by phase scheme
s = pd.read_csv(f"{root}/summary.csv")                  # member, factors, rgi_id, year, volume, velocity...
s[s.year == 300].groupby(["phase", "temperature_bias"]).velocity_mean_myr.agg(["mean", "std"])
```

The per-member `fl_diagnostics` files open with
`xr.open_dataset(path, group="fl_0")`, as in `wolverine_oggm.experiments.fl_diagnostics`.

## Tested so far (2026-11-19)

These were tested in a cloud container, not yet on Borah. The env was built with the same
`environment.yml` and the glibc 2.17 pin. Everything ran offline after the fetch step.

* `install.slurm` (with its `fetch_env.sh` step), using only the standalone micromamba:
  352 packages (500 MB) solved and fetched, the env built offline from the local files, and
  the sample data unpacked. The import check and the 9 tests passed.

* Default grid, Wolverine: the calibration matches the published level-5 directories
  (`prcp_fac` 1.01, `temp_bias` −3.07 K, `melt_f` 6.0, dynamic spinup succeeded). Random,
  historical and sliding ensembles ran, and the merge step worked.
* 25 m grid with 25 m flowline spacing (393 grid points, compared with 146 at 134 m), Millan and
  ITS_LIVE velocities on the flowline, dynamic spinup in 25 s, ISIMIP3b MRI-ESM2-0 SSP1-2.6 and
  SSP5-8.5 projections to 2100. COPDEM tiles were blocked there, so these runs used a
  user-supplied DEM (the 90 m Copernicus DEM resampled). That tests the pipeline but not real
  25 m topography.
* Cost of one Wolverine member at 25 m on one core: about 5 s for 300 years (annual steps), and
  about 9 s for 100 years of monthly HydroSliding. Wolverine alone is cheap even at 25 m. Borah
  pays off for big ensembles (sensitivity sweeps over many parameters and seeds), long runs,
  and many glaciers.

Not yet checked on Borah: the COPDEM30 tile download (OpenTopography's S3 bucket) from a
compute node, the `short` partition's time limit, and whether `$TMPDIR` on compute nodes is
node-local.

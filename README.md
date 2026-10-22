# wolverine-oggm

This repo runs the [Open Global Glacier Model (OGGM)](https://oggm.org) on Wolverine Glacier,
Alaska (USGS benchmark glacier, `RGI60-01.09162`). The experiments test how **precipitation
phase** (rain vs snow) affects mass balance, geometry and **ice velocity**. There is also a
pathway for forcing from the [SCHNAPS](https://codeberg.org/SCHNAPS-Model/SCHNAPS) runs in
[schnaps-borah](https://github.com/LejoFlores/schnaps-borah).

The notebooks are written to be read in order by someone new to OGGM. Each one explains the
model physics it relies on and says what standard OGGM can and cannot represent.

## Quick start

```bash
git clone https://github.com/LejoFlores/wolverine-oggm.git
cd wolverine-oggm
conda env create -f environment.yml
conda activate wolverine-oggm
jupyter lab notebooks/
```

The first notebook downloads Wolverine's preprocessed OGGM glacier directory (about 20 MB) from
`cluster.klima.uni-bremen.de` into `~/OGGM_runs/wolverine`. To put it elsewhere, set
`WOLVERINE_OGGM_WORKDIR`. Figures go to `$WOLVERINE_OGGM_WORKDIR/figures`, or to
`WOLVERINE_OGGM_FIGDIR` if set. On Borah, download on the login node, because compute nodes have
no internet.

## Notebooks

| # | Notebook | Content |
| --- | --- | --- |
| 00 | Setup and data | Glacier directory, outlines (RGI 6 and 7), flowline, W5E5 climate, calibrated parameters, WGMS series, thickness |
| 01 | OGGM primer | The equations: climate at height, temperature-index MB and the 0–2 °C rain/snow ramp, inversion, SIA flow and velocity. A first historical run |
| 02 | Mass balance and calibration | Model vs the USGS **winter/summer/annual** balances, `prcp_fac`–`melt_f` equifinality, recalibration under other phase schemes |
| 03 | Precipitation-phase experiments | H1–H4 below: phase schemes, sub-monthly temperature spread, warming with and without the snow-to-rain shift, warm-weather precipitation |
| 04 | Velocity and sliding | Model vs ITS_LIVE / Millan velocities by elevation, re-inversion with sliding at fixed volume, H5: water-driven sliding |
| 05 | SCHNAPS forcing | Phase curve, warm-precipitation offset and precipitation gradient from SCHNAPS microphysics on the glacier cells; splicing SCHNAPS months into the climate file |
| 06 | Publication figures | Multi-panel figures (PNG and editable PDF) from the runs in 03 and 04 |

### Hypotheses

| | Hypothesis | Where |
| --- | --- | --- |
| H1 | The rain/snow scheme matters little once the model is recalibrated | 03 |
| H2 | Sub-monthly temperature variability shifts accumulation where monthly means sit near 0 °C | 03 |
| H3 | Under warming, the loss of snowfall to rain is a first-order part of the glacier's response | 03, 06 |
| H4 | Precipitation in warmer-than-average weather (atmospheric rivers) reduces accumulation | 03, 05 |
| H5 | If sliding scales with liquid water input, rain/snow partitioning changes seasonal and long-term velocity | 04, 06 |

## What is added to OGGM (`wolverine_oggm/`)

* `massbalance.PhaseAwareTIModel` is OGGM's `MonthlyTIModel` with a configurable phase curve
  (ramp, threshold, logistic, Jennings et al. 2018, or tabulated from SCHNAPS). It averages that
  curve over the daily temperature spread (W5E5's `temp_std`), and can shift the temperature used
  for phase. With defaults it reproduces OGGM exactly, and a test checks this.
  `recalibrate` refits `melt_f` so every scheme matches the same geodetic MB.
* `massbalance.seasonal_balances` gives winter, summer and annual glacier-wide balances to
  compare with the USGS series.
* `dynamics.invert_with_sliding` re-inverts thickness with a sliding parameter that gives a
  chosen sliding fraction, keeping the volume. `dynamics.HydroSliding` is an evolution model
  whose sliding parameter follows monthly liquid water input (rain + melt). It is a conceptual
  sensitivity test, not a hydrology model.
* `schnaps` reads SCHNAPS output (raw case directories, or `post/extract_phase.py` files from
  schnaps-borah) and covers both forcing pathways.

## SCHNAPS → OGGM

The SCHNAPS Wolverine runs are **episodes** of days, at up to 50 m, hourly. OGGM needs monthly
climate over decades. So SCHNAPS enters in two ways:

1. **Now: calibrating the phase scheme.** From each run, fit solid fraction against 2 m
   temperature on the glacier cells, and measure the precipitation-weighted temperature offset
   and the precipitation–elevation gradient. These feed `PhaseAwareTIModel`, and the spread
   across runs (forecast vs analysis forcing, offshore domain, SNOWPACK vs Noah-MP, with and
   without HRRR w and hydrometeors) measures their uncertainty.
2. **Later: SCHNAPS months as climate.** For runs covering whole months, `write_oggm_climate`
   splices them into the glacier's baseline climate file (removing the calibrated precipitation
   factor and temperature bias, so the SCHNAPS values are used as they are). W5E5 ends in 2019,
   so recent periods need an ERA5 baseline.

Getting a compact file off Borah (in schnaps-borah):

```bash
python post/extract_phase.py $SCHNAPS_ROOT/cases/wolverine_oct2026_50m_noahmp_gpu/output \
    --domain d03 --out wolverine_oct2026_d03_phase.nc --site "Glacier station" 60.3869 -148.9371
```

Put it in `$WOLVERINE_OGGM_WORKDIR/schnaps/` and list it in `SCHNAPS_CASES` in notebook 05.
Until then, notebook 05 runs on a synthetic case, and its figures are stamped "SYNTHETIC".

## Status and testing

The notebooks are stored as `.ipynb` without outputs, paired with `.py` scripts through
[jupytext](https://jupytext.readthedocs.io). Edit either one, and jupytext keeps them in sync.

```bash
WOLVERINE_OGGM_TEST=1 pytest tests/                  # unit checks, about 10 s
bash tools/build_notebooks.sh --execute              # every notebook, top to bottom
```

`WOLVERINE_OGGM_TEST=1` swaps Wolverine for OGGM's offline sample glacier (Hintereisferner, with
HISTALP climate). It needs no data server, and it is what CI runs. Every notebook and test
passes in that mode. **The Wolverine-specific paths have not been run yet**, because the
environment they were written in could not reach OGGM's data server. Those paths are the
preprocessed directory download, the W5E5 `temp_std`, the WGMS seasonal series, the
ITS_LIVE/Millan velocities, the dynamical-spinup start state and the RGI 7 overlay. Run 00–02
first on Wolverine and report anything that breaks.

## Caveats

* OGGM's mass balance is monthly, with no refreezing, no rain-on-snow energy, and no
  precipitation gradient with elevation.
* Elevation-band flowlines are not real flow paths, so compare velocities by elevation, not
  by position.
* In standard OGGM, velocity responds to phase only through geometry. The sliding extension
  (H5) is a bounded "what if", not a process model.

## Key references

Maussion et al. (2019, GMD) for OGGM. Schuster et al. (2023) for the OGGM v1.6 mass balance and
calibration. Hugonnet et al. (2021) for the geodetic MB. Farinotti et al. (2019) for the
consensus thickness. Millan et al. (2022) for velocity and thickness. Jennings et al. (2018,
Nat. Commun.) for the rain/snow partitioning curve. O'Neel et al. (2019, J. Glaciol.) for the
USGS benchmark glacier mass balance.

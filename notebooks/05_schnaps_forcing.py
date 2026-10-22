# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 05 · SCHNAPS forcing: phase from the microphysics into OGGM
#
# [SCHNAPS](https://codeberg.org/SCHNAPS-Model/SCHNAPS) (the successor to HICAR) is an
# intermediate-complexity atmospheric model coupled to a land surface and snow model. Lejo runs it
# on Boise State's Borah cluster ([schnaps-borah](https://github.com/LejoFlores/schnaps-borah)).
# Over Wolverine it runs nested to **50 m** (d03), forced by HRRR-Alaska, for episodes such as
# the Oct 9–17 2026 atmospheric river. Its microphysics decide, hour by hour and cell by cell,
# whether precipitation reaches the glacier as rain, snow or graupel. That is the information
# OGGM's monthly temperature ramp tries to stand in for.
#
# There are two pathways into OGGM.
#
# 1. **Episodes (now).** SCHNAPS runs are days long, while OGGM works on monthly climate over
#    decades. So SCHNAPS cannot drive OGGM directly. Instead it **calibrates OGGM's phase
#    scheme**: fit solid fraction against 2 m temperature on the glacier cells, measure how much
#    warmer it is when it precipitates, and check OGGM's no-precipitation-gradient assumption.
#    Those numbers go into `PhaseAwareTIModel`.
# 2. **Long runs (later).** With seasons or years of SCHNAPS output, `monthly_climate` +
#    `write_oggm_climate` splice SCHNAPS months into the glacier's climate file, and OGGM runs on
#    them with `climate_input_filesuffix="_schnaps"`.
#
# ## Getting the data off Borah
#
# The raw 50 m case output is large. schnaps-borah's `post/extract_phase.py` writes a compact
# file with hourly precipitation, solid precipitation, rain, 2 m temperature, terrain and
# lat/lon:
#
# ```bash
# # on Borah, in ~/schnaps-borah (the Noah-MP runs write taix, which extract_phase.py prefers)
# python post/extract_phase.py $SCHNAPS_ROOT/cases/wolverine_oct2026_50m_noahmp_gpu/output \
#     --domain d03 --out wolverine_oct2026_d03_phase.nc --site "Glacier station" 60.3869 -148.9371
# scp borah:~/schnaps-borah/wolverine_oct2026_d03_phase.nc ~/OGGM_runs/wolverine/schnaps/
# ```
#
# Set `SCHNAPS_CASES` below to one or more of those files, or to raw output directories.
# **If none exists, the notebook makes a synthetic case** so that every cell runs. Synthetic
# figures are stamped "SYNTHETIC". Its numbers mean nothing, except that the fit should recover
# the curve it was built with (50% snow at 0.8 °C, width 0.6 K).

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from oggm import cfg

from wolverine_oggm import config, experiments, massbalance, phase, plotting, schnaps

warnings.filterwarnings("ignore")
plotting.set_style("paper")
gdir = config.get_glacier_directory()

SCHNAPS_DIR = os.path.join(config.WORKING_DIR, "schnaps")
SCHNAPS_CASES = {        # label: extract_phase.py file or raw output directory (domain d03)
    "50 m Noah-MP": os.path.join(SCHNAPS_DIR, "wolverine_oct2026_d03_phase.nc"),
}
DOMAIN = "d03"

cases = {k: v for k, v in SCHNAPS_CASES.items() if os.path.exists(v)}
SYNTHETIC = not cases or config.TEST_MODE
if SYNTHETIC:
    start = "1995-06-01" if config.TEST_MODE else "2026-10-09"     # test mode: inside the test climate
    hours = 24 * 61 if config.TEST_MODE else 24 * 8
    path = schnaps.make_synthetic_case(os.path.join(SCHNAPS_DIR, "synthetic"), gdir, start=start, hours=hours)
    cases = {"SYNTHETIC": path}
print("Cases:", cases)

# %% [markdown]
# ## Load a case and find the glacier cells

# %%
label, path = next(iter(cases.items()))
ds = schnaps.load_case(path, DOMAIN)
outline_kw = {"gdir": gdir} if (SYNTHETIC and config.TEST_MODE) else {}
mask = schnaps.glacier_mask(ds, **outline_kw)
print(f"{label}: {ds.sizes['time']} hours from {str(ds.time.values[0])[:13]} to {str(ds.time.values[-1])[:13]}, "
      f"{int(mask.sum())} glacier cells, terrain {float(ds.terrain.where(mask).min()):.0f}-{float(ds.terrain.where(mask).max()):.0f} m")

# %%
tot = ds["precip"].sum("time")
sfrac = ds["solid"].sum("time") / tot.where(tot > 0)
fig, axes = plt.subplots(1, 2, figsize=(plotting.WIDTH_2COL, 3), sharey=True)
im0 = axes[0].pcolormesh(ds.lon, ds.lat, tot, cmap="Blues", shading="auto")
plt.colorbar(im0, ax=axes[0], label="Total precipitation (mm)")
im1 = axes[1].pcolormesh(ds.lon, ds.lat, sfrac, cmap="RdPu", vmin=0, vmax=1, shading="auto")
plt.colorbar(im1, ax=axes[1], label="Solid fraction")
for ax in axes:
    ax.contour(ds.lon, ds.lat, mask.astype(int), levels=[0.5], colors="k", linewidths=0.8)
    ax.contour(ds.lon, ds.lat, ds.terrain, levels=np.arange(0, 4000, 200), colors=plotting.GREY, linewidths=0.3)
    ax.set_xlabel("Longitude")
    if SYNTHETIC:
        plotting.synthetic_stamp(ax)
axes[0].set_ylabel("Latitude")
plotting.label_panels(axes)
fig.tight_layout()

# %% [markdown]
# ## The phase curve SCHNAPS implies
#
# Every wet glacier cell-hour is one sample of (2 m temperature, solid fraction). Binning by
# temperature, weighted by precipitation, gives an empirical curve. A logistic fit gives two
# numbers to carry into OGGM: the 50% temperature and the width of the transition.
#
# Caveats: 2 m temperature over a melting snow/ice surface is pinned near 0 °C, which compresses
# the curve. The microphysics phase is what reaches the surface, so it includes melting in the
# lowest layers. And one storm samples a narrow range of conditions. Fit several runs and
# storms before trusting the numbers.

#
# The printed standard errors treat every cell-hour as independent. They are not: neighbouring
# cells and consecutive hours are strongly correlated, so the real uncertainty is much larger.
# The spread between runs and storms (below) is the better measure.

# %%
df = schnaps.phase_samples(ds, mask, min_precip=0.1)
emp, fit = schnaps.phase_curve_from_case(df)
cen, frac, ptot = phase.binned_fraction(df["t2"], df["solid"], df["precip"])
print(f"{len(df)} wet cell-hours; fit: 50% snow at {fit.t_half:.2f} ± {fit.stderr['t50']:.2f} °C, "
      f"width {fit.width:.2f} ± {fit.stderr['width']:.2f} K")
fig, ax = plt.subplots(figsize=(plotting.WIDTH_1COL * 1.3, 2.8))
plotting.phase_curves(ax, [phase.STANDARD_CURVES["oggm"], phase.STANDARD_CURVES["jennings90"], fit],
                      samples=(cen, frac, ptot))
ax.set_title(label)
if SYNTHETIC:
    plotting.synthetic_stamp(ax)

# %% [markdown]
# ## Precipitation in warm weather
#
# The offset between the precipitation-weighted and the plain mean temperature, over the same
# cells and hours, is `precip_temp_offset` in `PhaseAwareTIModel` (H4 in notebook 03). For an
# atmospheric river it should be positive.

# %%
offset = schnaps.precip_temperature_offset(ds, mask)
print(f"Precipitation-weighted minus mean 2 m temperature: {offset:+.2f} K")

# %% [markdown]
# ## Elevation bands: SCHNAPS vs what OGGM would assume
#
# For each 100 m band on the glacier:
#
# * `solid_frac_schnaps`: the microphysics
# * `solid_frac_oggm_hourly_T`: OGGM's ramp applied to the same hourly temperatures
# * `solid_frac_oggm_episode_mean_T`: OGGM's ramp applied to the band's mean temperature over the
#   episode, which is what a monthly model sees
# * `precip_mm_per_cell`: the precipitation gradient that OGGM assumes is zero

# %%
bands = schnaps.band_summary(df, ds, mask, band=100)
bands.round(2)

# %%
fig, axes = plt.subplots(1, 2, figsize=(plotting.WIDTH_2COL, 3), sharey=True)
z = bands.index
axes[0].plot(bands["solid_frac_schnaps"], z, "o-", color=plotting.MIXED, ms=3, label="SCHNAPS microphysics")
axes[0].plot(bands["solid_frac_oggm_hourly_T"], z, "s--", color="k", ms=3, label="OGGM ramp, hourly T")
axes[0].plot(bands["solid_frac_oggm_episode_mean_T"], z, "^:", color=plotting.GREY, ms=3, label="OGGM ramp, episode-mean T")
axes[0].set_xlabel("Solid fraction")
axes[0].set_ylabel("Elevation band (m)")
axes[0].legend(frameon=False, fontsize=6)
axes[1].plot(bands["precip_mm_per_cell"], z, "o-", color=plotting.ICE, ms=3)
axes[1].axvline(bands["precip_mm_per_cell"].mean(), color=plotting.GREY, ls="--", lw=0.7, label="no gradient (OGGM)")
axes[1].set_xlabel("Episode precipitation per cell (mm)")
axes[1].legend(frameon=False, fontsize=6)
plotting.label_panels(axes)
for ax in axes:
    if SYNTHETIC:
        plotting.synthetic_stamp(ax)
fig.tight_layout()

# %% [markdown]
# ## Pathway 1: OGGM with the SCHNAPS-derived phase scheme
#
# Build a phase model from the fitted curve, the measured offset and the daily temperature
# spread of the baseline climate. Recalibrate it, and compare it with the default in a
# stationary run, as in notebook 03. The fitted curve uses *instantaneous* (hourly) temperatures,
# so the daily spread is needed to apply it to monthly means.

# %%
spread = config.daily_temp_spread(gdir)
NYEARS = 60 if config.TEST_MODE else 150
default = massbalance.recalibrate(gdir, massbalance.phase_model_class(phase.STANDARD_CURVES["oggm"], temp_std=spread), "_ph4")
from_schnaps = massbalance.recalibrate(
    gdir, massbalance.phase_model_class(fit, temp_std=spread, precip_temp_offset=offset, name=f"SCHNAPS fit ({label})"),
    "_schnaps_phase")
print("melt_f default:", round(default.calib["melt_f"], 2), "| SCHNAPS phase:", round(from_schnaps.calib["melt_f"], 2))
for sfx, cls in (("_p1_default", default), ("_p1_schnaps", from_schnaps)):
    experiments.run_stationary(gdir, sfx, cls, NYEARS, seed=1)
    experiments.run_stationary(gdir, sfx + "_warm", cls, NYEARS, seed=1, temperature_bias=1.0)
experiments.compare(gdir, ["_p1_default", "_p1_schnaps", "_p1_default_warm", "_p1_schnaps_warm"],
                    labels=["default", "SCHNAPS phase", "default +1 K", "SCHNAPS phase +1 K"]).round(2)

# %% [markdown]
# ## Comparing several SCHNAPS runs
#
# The Wolverine runs in schnaps-borah differ in forcing (HRRR 6 h forecast vs analysis), outer
# domain (coastal vs offshore), snow model (SNOWPACK vs Noah-MP), resolution (250 m d02 vs 50 m
# d03), and whether HRRR's w and hydrometeors are included. Their spread in fitted phase
# parameters is a direct measure of how uncertain the "SCHNAPS phase" is. Add the extracted
# files to `SCHNAPS_CASES` and this table fills in.

# %%
rows = []
for lab, p in cases.items():
    d = schnaps.load_case(p, DOMAIN)
    m = schnaps.glacier_mask(d, **outline_kw)
    s = schnaps.phase_samples(d, m)
    _, f = schnaps.phase_curve_from_case(s)
    rows.append({"case": lab, "glacier_precip_mm": float(d["precip"].values[:, m.values].sum(0).mean()),
                 "solid_frac": s["solid"].sum() / s["precip"].sum(), "t50_C": f.t_half, "width_K": f.width,
                 "precip_T_offset_K": schnaps.precip_temperature_offset(d, m)})
pd.DataFrame(rows).set_index("case").round(2)

# %% [markdown]
# ## Pathway 2: SCHNAPS months in OGGM's climate file
#
# For runs long enough to cover whole months, aggregate the glacier cells to monthly climate at a
# reference height, and splice the covered months into a copy of the baseline climate. By default
# (`as_is=True`) SCHNAPS precipitation is divided by the calibrated `prcp_fac`, and the
# `temp_bias` is removed, so that the mass-balance model uses SCHNAPS values unchanged.
#
# The splice needs the baseline to cover the SCHNAPS months. W5E5 ends in 2019, so for 2026 runs
# set up the glacier with an ERA5 baseline (`cfg.PARAMS["baseline_climate"] = "ERA5"` and
# `tasks.process_climate_data`), and recalibrate.

# %%
mon = schnaps.monthly_climate(ds, mask)
print(f"Reference height {mon.attrs['ref_hgt']:.0f} m")
print(mon.round(2))
try:
    comp = schnaps.write_oggm_climate(gdir, mon, filesuffix="_schnaps", min_coverage=0.9)
    print(comp.round(1))
    clim_ok = True
except ValueError as err:
    print("Not spliced:", err)
    clim_ok = False

# %%
if clim_ok:
    from oggm import tasks
    run = config.historical_run(gdir)
    tasks.run_from_climate_data(gdir, climate_input_filesuffix="_schnaps", output_filesuffix="_hist_schnaps", **run)
    tasks.run_from_climate_data(gdir, output_filesuffix="_hist_base", **run)
    print(experiments.compare(gdir, ["_hist_base", "_hist_schnaps"], labels=["baseline", "with SCHNAPS months"], tail=1).round(3))

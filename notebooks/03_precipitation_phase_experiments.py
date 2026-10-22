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
# # 03 · Precipitation-phase experiments
#
# **Question.** How much does the partitioning of precipitation into rain and snow control
# Wolverine's mass balance, geometry and (through geometry) ice velocity?
#
# OGGM's answer depends on assumptions that are rarely tested, so we test them as alternative
# hypotheses.
#
# | | Hypothesis | Experiment |
# |---|---|---|
# | **H1** | The choice of rain/snow scheme matters little once the model is recalibrated | Same climate, five phase schemes, each recalibrated to the same geodetic MB; compare MB profiles and dynamics |
# | **H2** | Sub-monthly temperature variability shifts accumulation at Wolverine's monthly-mean temperatures near 0 °C | Ramp on monthly means vs the same ramp averaged over the daily temperature spread |
# | **H3** | Under warming, the loss of snowfall to rain is a first-order part of the glacier's response | +1 and +2 K: full response vs "phase frozen" (only melt warms) vs "phase only" (only the rain/snow split warms) |
# | **H4** | Precipitation that falls in warmer-than-average weather (atmospheric rivers) reduces accumulation | Shift the phase temperature by the precipitation-weighted offset, which notebook 05 derives from SCHNAPS |
#
# All runs are stationary-climate runs: years drawn at random from the calibration period, the
# same random sequence for every run, starting from the present glacier. This isolates each
# mechanism from the historical trajectory. For transient runs, see the end of the notebook.

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from oggm import cfg

from wolverine_oggm import config, experiments, massbalance, phase, plotting

warnings.filterwarnings("ignore")
plotting.set_style("paper")
gdir = config.get_glacier_directory()
NYEARS = 60 if config.TEST_MODE else 150       # long enough to approach a new equilibrium
SEED = 1
spread = config.daily_temp_spread(gdir)
y0, y1 = config.calibration_years()
print(f"Stationary climate drawn from {y0}-{y1}, {NYEARS} years; daily temperature spread: {spread}")

# %% [markdown]
# ## The schemes, recalibrated
#
# Each scheme gets its own melt factor, so the mean MB over the calibration period is the same.
# The calibration files are `mb_calib_ph*.json` (the same ones notebook 02 makes).

# %%
schemes = {
    "OGGM ramp 0–2 °C": massbalance.phase_model_class(),
    "Threshold 0 °C": massbalance.phase_model_class(phase.Threshold(0.0)),
    "Threshold 2 °C": massbalance.phase_model_class(phase.Threshold(2.0)),
    "Jennings RH90 + daily spread": massbalance.phase_model_class(phase.Jennings2018(90), temp_std=spread),
    "OGGM ramp + daily spread": massbalance.phase_model_class(phase.STANDARD_CURVES["oggm"], temp_std=spread),
}
cal = {label: massbalance.recalibrate(gdir, cls, f"_ph{k}") for k, (label, cls) in enumerate(schemes.items())}
pd.Series({k: c.calib["melt_f"] for k, c in cal.items()}, name="melt_f").round(2)

# %% [markdown]
# ## H1 and H2 on fixed geometry: mass-balance profiles
#
# Same mean MB by construction, but where on the glacier the mass is gained and lost differs.
# A steeper MB gradient means more mass turned over, so more ice flux and faster flow for the
# same glacier.

# %%
fl = gdir.read_pickle("model_flowlines")[-1]
hh = np.linspace(fl.surface_h[fl.thick > 0].min(), fl.surface_h.max(), 30)
fig, axes = plt.subplots(1, 2, figsize=(plotting.WIDTH_2COL, 3), sharey=True)
grad = {}
for label, cls in cal.items():
    mb = cls(gdir)
    smb = np.mean([mb.get_annual_mb(hh, y) for y in range(y0, y1 + 1)], axis=0) * cfg.SEC_IN_YEAR * mb.rho
    sol = np.mean([mb.get_annual_climate(hh, y)[3] for y in range(y0, y1 + 1)], axis=0)
    tot = np.mean([mb.get_annual_climate(hh, y)[2] for y in range(y0, y1 + 1)], axis=0)
    axes[0].plot(smb, hh, label=label)
    axes[1].plot(sol / tot, hh, label=label)
    lower = hh < np.percentile(hh, 50)
    grad[label] = np.polyfit(hh[lower], smb[lower], 1)[0] * 100       # kg m-2 per 100 m, lower half
axes[0].axvline(0, color=plotting.GREY, lw=0.5)
axes[0].set_xlabel("Mean SMB (kg m⁻² yr⁻¹)")
axes[0].set_ylabel("Elevation (m)")
axes[1].set_xlabel("Solid fraction of annual precipitation")
axes[0].legend(frameon=False, fontsize=6)
plotting.label_panels(axes)
fig.tight_layout()
plotting.test_mode_note(fig)
pd.Series(grad, name="SMB gradient, lower glacier (kg m-2 per 100 m)").round(0)

# %% [markdown]
# ## H1 and H2 dynamically

# %%
for k, (label, cls) in enumerate(cal.items()):
    experiments.run_stationary(gdir, f"_h1_{k}", cls, NYEARS, seed=SEED)
h1 = experiments.compare(gdir, [f"_h1_{k}" for k in range(len(cal))], labels=list(cal))
h1.round(2)

# %%
fig, axes = plt.subplots(1, 3, figsize=(plotting.WIDTH_2COL, 2.4))
for k, label in enumerate(cal):
    d = experiments.summary(gdir, f"_h1_{k}")
    axes[0].plot(d.index - d.index[0], d.volume_km3, label=label)
    axes[1].plot(d.index - d.index[0], d.velocity_mean_myr.rolling(5, center=True).mean())
    p = experiments.velocity_profile(gdir, f"_h1_{k}")
    axes[2].plot(p.index, p.values)
axes[0].set_ylabel("Volume (km³)")
axes[1].set_ylabel("Mean surface velocity (m yr⁻¹, 5-yr mean)")
axes[2].set_ylabel("Surface velocity, last 10 yr (m yr⁻¹)")
axes[0].set_xlabel("Years")
axes[1].set_xlabel("Years")
axes[2].set_xlabel("Distance along flowline (km)")
axes[0].legend(frameon=False, fontsize=5)
plotting.label_panels(axes)
fig.tight_layout()
plotting.test_mode_note(fig)

# %% [markdown]
# ## H3: warming with and without the phase change
#
# Three runs per warming level ΔT, all with the default scheme plus the daily spread:
#
# * **full**: temperature +ΔT everywhere (melt and phase)
# * **phase frozen**: melt sees +ΔT, phase sees the original temperature (`precip_temp_offset = −ΔT`)
# * **phase only**: phase sees +ΔT, melt does not (no temperature bias, `precip_temp_offset = +ΔT`)
#
# full − phase frozen ≈ the part of the warming response due to snow turning into rain. The
# decomposition is not exactly additive, because the glacier shrinks to different elevations, so
# report all three.

# %%
base = cal["OGGM ramp + daily spread"]
runs = {"control": (base, 0.0)}
for dT in (1.0, 2.0):
    runs[f"+{dT:g} K full"] = (base, dT)
    runs[f"+{dT:g} K phase frozen"] = (massbalance.with_settings(base, precip_temp_offset=-dT), dT)
    runs[f"+{dT:g} K phase only"] = (massbalance.with_settings(base, precip_temp_offset=+dT), 0.0)
for k, (label, (cls, dT)) in enumerate(runs.items()):
    experiments.run_stationary(gdir, f"_h3_{k}", cls, NYEARS, temperature_bias=dT, seed=SEED)
h3 = experiments.compare(gdir, [f"_h3_{k}" for k in range(len(runs))], labels=list(runs))
h3.round(2)

# %%
ctrl = h3.loc["control", "volume_end_km3"]
rows = []
for dT in (1.0, 2.0):
    full = ctrl - h3.loc[f"+{dT:g} K full", "volume_end_km3"]
    frozen = ctrl - h3.loc[f"+{dT:g} K phase frozen", "volume_end_km3"]
    only = ctrl - h3.loc[f"+{dT:g} K phase only", "volume_end_km3"]
    rows.append({"warming_K": dT, "volume_loss_full_km3": full, "loss_from_phase_km3 (full − frozen)": full - frozen,
                 "phase_share_pct": 100 * (full - frozen) / full if full else np.nan,
                 "loss_phase_only_km3": only})
pd.DataFrame(rows).set_index("warming_K").round(3)

# %%
colors = {"control": "k", "full": plotting.MIXED, "phase frozen": plotting.SNOW, "phase only": plotting.RAIN}
fig, axes = plt.subplots(1, 2, figsize=(plotting.WIDTH_2COL, 2.6), sharex=True)
for k, label in enumerate(runs):
    d = experiments.summary(gdir, f"_h3_{k}")
    kind = "control" if label == "control" else label.split(" K ")[1]
    ls = "-" if label.startswith("+2") or label == "control" else "--"
    axes[0].plot(d.index - d.index[0], d.volume_km3, color=colors[kind], ls=ls, label=label)
    axes[1].plot(d.index - d.index[0], d.velocity_mean_myr.rolling(5, center=True).mean(), color=colors[kind], ls=ls)
axes[0].set_ylabel("Volume (km³)")
axes[1].set_ylabel("Mean surface velocity (m yr⁻¹)")
for ax in axes:
    ax.set_xlabel("Years")
axes[0].legend(frameon=False, fontsize=5, ncol=2)
plotting.label_panels(axes)
fig.tight_layout()
plotting.test_mode_note(fig)

# %% [markdown]
# ## H4: precipitation in warm weather
#
# In an atmospheric river, the precipitation-weighted temperature is warmer than the month's
# mean. Notebook 05 measures this offset from SCHNAPS on the glacier cells. Here it is a
# parameter: 0, +1 and +2 K, each recalibrated, then the same stationary run.

# %%
h4 = []
for k, off in enumerate((0.0, 1.0, 2.0)):
    cls = massbalance.recalibrate(gdir, massbalance.with_settings(base, precip_temp_offset=off), f"_off{k}")
    experiments.run_stationary(gdir, f"_h4_{k}", cls, NYEARS, seed=SEED)
    h4.append(f"_h4_{k}")
h4t = experiments.compare(gdir, h4, labels=["offset 0 K", "offset +1 K", "offset +2 K"])
h4t.round(2)

# %% [markdown]
# ## Reading the results
#
# * Recalibration absorbs most scheme differences in the **mean** (H1), but not in the **MB
#   gradient**. The gradient sets the flux the glacier has to carry, and through that its
#   thickness and speed.
# * The phase share of the warming response (H3) is a headline number for the thesis question.
#   Compare it across schemes: rerun H3 with `base = cal["Threshold 0 °C"]` and the others.
# * Velocity in these runs follows geometry: as the glacier thins and retreats, speeds drop. A
#   scheme that loses more accumulation to rain slows the glacier more over decades.
#
# **Uncertainty.** Repeat with several seeds (`experiments.ensemble`) to separate the signal from
# the random year sequence. Also vary `NYEARS`: 150 years is not full equilibrium for a 16 km²
# glacier.
#
# **Transient version.** For the historical period use
# `tasks.run_from_climate_data(gdir, mb_model_class=cls, **config.historical_run(gdir))`. For
# projections, OGGM's `process_gcm_data`/`process_cmip_data` plus `run_from_climate_data` with
# `climate_filename="gcm_data"` take the same `mb_model_class`.

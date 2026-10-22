"""Figure helpers: one consistent look for talks and papers.

Colours follow schnaps-borah's bivariate rain/snow maps: teal for rain, pink for snow, violet for
mixed. `set_style("paper")` gives single-column-sized fonts, `set_style("talk")` larger ones.
`savefig` writes PNG (300 dpi) and PDF (vector) side by side in config.FIG_DIR.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import config

RAIN = "#1a9e9e"
SNOW = "#d64f9b"
MIXED = "#7b5cc4"
ICE = "#5b8fd6"
GREY = "#6b6b6b"
# for scenario lines, colour-blind-safe (Okabe-Ito)
CYCLE = ["#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]

WIDTH_1COL = 3.5     # inches, typical journal single column
WIDTH_2COL = 7.2


def set_style(context="paper"):
    base = {"paper": 8, "talk": 14, "poster": 18}[context]
    mpl.rcParams.update({
        "font.size": base, "axes.titlesize": base + 1, "axes.labelsize": base,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1, "legend.fontsize": base - 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "lines.linewidth": 1.2 if context == "paper" else 2.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text in Illustrator/Inkscape
    })


def savefig(fig, name, formats=("png", "pdf")):
    """Save to config.FIG_DIR/name.{png,pdf}; returns the paths."""
    os.makedirs(config.FIG_DIR, exist_ok=True)
    paths = []
    for f in formats:
        p = os.path.join(config.FIG_DIR, f"{name}.{f}")
        fig.savefig(p)
        paths.append(p)
    return paths


def label_panels(axes, x=-0.12, y=1.04):
    """(a), (b), ... in the upper left of each axis."""
    for k, ax in enumerate(np.ravel(axes)):
        ax.text(x, y, f"({chr(97 + k)})", transform=ax.transAxes, fontweight="bold", va="bottom")


def phase_curves(ax, curves, t=np.linspace(-6, 6, 241), samples=None, styles=None):
    """Solid fraction vs temperature for several curves; optional binned samples
    (centres, fraction, precipitation per bin) drawn as dots sized by precipitation."""
    for k, c in enumerate(curves):
        kw = (styles or {}).get(c.name, {})
        ax.plot(t, c(t), label=c.name, **kw)
    if samples is not None:
        cen, frac, ptot = samples
        s = 80 * ptot / np.nanmax(ptot)
        ax.scatter(cen, frac, s=s, color=MIXED, alpha=0.6, zorder=3, label="SCHNAPS bins (size: precip.)")
    ax.axvline(0, color=GREY, lw=0.5)
    ax.set_xlabel("Air temperature during precipitation (°C)")
    ax.set_ylabel("Solid fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)


def hovmoller(ax, da, cmap="viridis", vmin=None, vmax=None, label=None):
    """Time vs distance-along-flowline image of a fl_diagnostics variable (e.g. ice_velocity_myr)."""
    x = da["dis_along_flowline"].values / 1000
    im = ax.pcolormesh(x, da["time"].values, da.values, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_xlabel("Distance along flowline (km)")
    ax.set_ylabel("Year")
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    if label:
        cb.set_label(label)
    return im


def flowline_profile(ax, fl, label=None, bed=True, color=ICE):
    """Surface (and bed) along an OGGM flowline."""
    x = np.arange(fl.nx) * fl.dx_meter / 1000
    if bed:
        ax.plot(x, fl.bed_h, color="k", lw=0.8, label="Bed")
    ax.plot(x, fl.surface_h, color=color, label=label or "Surface")
    ax.fill_between(x, fl.bed_h, fl.surface_h, where=fl.thick > 0, color=color, alpha=0.25, lw=0)
    ax.set_xlabel("Distance along flowline (km)")
    ax.set_ylabel("Elevation (m)")


def synthetic_stamp(ax):
    """Large diagonal stamp for figures made from synthetic data."""
    ax.text(0.5, 0.5, "SYNTHETIC", transform=ax.transAxes, fontsize=28, color="red", alpha=0.18,
            ha="center", va="center", rotation=30, zorder=10)


def test_mode_note(fig):
    if config.TEST_MODE:
        fig.text(0.99, 0.005, "TEST MODE: offline sample glacier, not Wolverine", ha="right", va="bottom",
                 fontsize=6, color="red")

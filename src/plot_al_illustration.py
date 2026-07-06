"""
Publication figure: how the augmented Lagrangian resolves a constraint.

Test problem (chosen so every landmark is analytic):

    minimise f(x) = x^2   subject to   g(x) = 1 - x <= 0.

The feasible region is x >= 1, the constrained optimum is x* = 1, and the
KKT multiplier is gamma* = 2 (from 2x - gamma = 0 at x = 1). The augmented
Lagrangian (paper notation; mirrors pycma and Eq. (2) of the paper) is

    h(x, gamma, omega) = f(x) + / gamma g + (omega/2) g^2   if gamma + omega g >= 0
                                \ -gamma^2 / (2 omega)      otherwise.

On the penalised branch the minimiser is x_min = (gamma + omega)/(2 + omega),
so with omega = 8:

    gamma = 0.1  -> x_min = 0.81   (under-penalised: minimiser infeasible)
    gamma = 2    -> x_min = 1.00   (= gamma*: minimiser is exactly x*)
    gamma = 5    -> x_min = 1.30   (over-penalised: pushed into the interior)

The static quadratic penalty P(x, rho) = f + (rho/2) max(0, g)^2 minimises at
x = rho/(2 + rho) = 0.96 for rho = 50 — always slightly infeasible for any
finite rho, which is the visual argument for the AL's exactness at finite
omega via the multiplier.

Shaded regions: g(x) > 0 (infeasible, red) and the flat branch
gamma* + omega g(x) < 0 of the *bold* curve (green), i.e. x > 1 + gamma*/omega
= 1.25 — computed from the plotted (gamma*, omega) so the shading is
self-consistent with the drawn curves.
"""

from __future__ import annotations

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
})


# ---------------------------------------------------------------------------
# Test problem
# ---------------------------------------------------------------------------
def f(x):
    return x ** 2


def g(x):
    return 1.0 - x


GAMMA_STAR = 2.0          # KKT multiplier of min x^2 s.t. 1 - x <= 0
OMEGA = 8.0
GAMMAS = [0.1, GAMMA_STAR, 5.0]
RHO = 50.0                # static-penalty coefficient


def h_al(x, gamma, omega):
    """Augmented Lagrangian h = f + p, flat branch where gamma + omega g < 0."""
    gx = g(x)
    penalised = gamma * gx + 0.5 * omega * gx ** 2
    flat = np.full_like(np.asarray(x, dtype=float), -gamma ** 2 / (2.0 * omega))
    return f(x) + np.where(gamma + omega * gx >= 0.0, penalised, flat)


def p_static(x, rho):
    """Static quadratic penalty f + (rho/2) max(0, g)^2."""
    return f(x) + 0.5 * rho * np.maximum(0.0, g(x)) ** 2


def h_al_argmin(gamma, omega):
    """Penalised-branch stationary point (valid for all cases drawn here)."""
    return (gamma + omega) / (2.0 + omega)


# ---------------------------------------------------------------------------
# Colours (pre-blended: EPS has no alpha channel, so transparent artists
# would force pdftops to rasterise the region — see _blend)
# ---------------------------------------------------------------------------
def _blend(color, alpha, bg="white"):
    """Flatten a semi-transparent colour onto `bg`, returning an opaque hex."""
    c = np.asarray(mcolors.to_rgb(color))
    b = np.asarray(mcolors.to_rgb(bg))
    return mcolors.to_hex(alpha * c + (1.0 - alpha) * b)


COL_F = "#1f3fbf"
COL_G = "#c0392b"
COL_H = "#1e7a34"
COL_P = "#111111"
FILL_INFEAS = _blend("#e74c3c", 0.22)
FILL_FLAT = _blend("#2ecc71", 0.30)

XLIM = (-0.5, 2.6)
YLIM = (-2.0, 6.0)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.6))
xs = np.linspace(XLIM[0], XLIM[1], 800)

# Region shading. Red: infeasible g > 0 (x < 1). Green: flat branch of the
# bold gamma* curve, gamma* + omega g < 0  <=>  x > 1 + gamma*/omega.
x_flat = 1.0 + GAMMA_STAR / OMEGA
ax.axvspan(XLIM[0], 1.0, color=FILL_INFEAS, zorder=0.3, linewidth=0)
ax.axvspan(x_flat, XLIM[1], color=FILL_FLAT, zorder=0.3, linewidth=0)

ax.set_axisbelow(True)  # grid at zorder 0.5: above the fills, below curves
ax.grid(True, lw=0.4, color="#bbbbbb")

# Objective and constraint.
ax.plot(xs, f(xs), color=COL_F, lw=2.6, zorder=3, label=r"$f(x)$")
ax.plot(xs, g(xs), color=COL_G, lw=1.0, zorder=2, label=r"$g(x)$")

# Augmented Lagrangians: under-penalised (dashed), gamma* (bold solid),
# over-penalised (dash-dot).
H_STYLES = [
    dict(ls="--", lw=1.3),
    dict(ls="-", lw=2.4),
    dict(ls="-.", lw=1.3),
]
H_LABELS = [
    r"$h(x,\,0.1,\,8)$",
    r"$h(x,\,\gamma^*\!=\!2,\,8)$",
    r"$h(x,\,5,\,8)$",
]
for gamma, style, label in zip(GAMMAS, H_STYLES, H_LABELS):
    ax.plot(xs, h_al(xs, gamma, OMEGA), color=COL_H, zorder=4,
            label=label, **style)

# Static quadratic penalty for contrast.
ax.plot(xs, p_static(xs, RHO), color=COL_P, lw=1.0, zorder=5,
        label=r"$P(x,\,50)$")

# Minimisers: dots on each curve; star at the constrained optimum x* = 1,
# where only the gamma* curve's minimiser lands.
for gamma, style in zip(GAMMAS, H_STYLES):
    xm = h_al_argmin(gamma, OMEGA)
    ax.plot(xm, h_al(xm, gamma, OMEGA), "o", ms=5, color=COL_H,
            mec="white", mew=0.7, zorder=6)
xm_p = RHO / (2.0 + RHO)
ax.plot(xm_p, p_static(xm_p, RHO), "o", ms=5, color=COL_P,
        mec="white", mew=0.7, zorder=6)
ax.plot(1.0, f(1.0), "*", ms=13, color="#e6a817", mec="black", mew=0.6,
        zorder=7)
ax.annotate(r"$\mathbf{x}^*$", (1.0, 1.0), xytext=(6, -14),
            textcoords="offset points", fontsize=11)

ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_xlabel(r"$x_1$")

# Legend below the axes, region patches appended after the curve entries.
handles, labels = ax.get_legend_handles_labels()
handles += [Patch(facecolor=FILL_FLAT, label=r"$\gamma^* + \omega\,g(x) < 0$"),
            Patch(facecolor=FILL_INFEAS, label=r"$g(x) > 0$")]
labels += [r"$\gamma^* + \omega\,g(x) < 0$", r"$g(x) > 0$"]
# framealpha=1.0: the default (0.8) is a transparent artist, and a single
# one anywhere makes pdftops rasterise the whole page into a bitmap.
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.14),
          ncol=4, frameon=True, edgecolor="#888888", fontsize=9,
          columnspacing=1.2, handlelength=1.8, framealpha=1.0)

fig.tight_layout(rect=(0, 0.02, 1, 1))

os.makedirs("figures", exist_ok=True)
OUT_PNG = "figures/al_illustration.png"
OUT_PDF = "figures/al_illustration.pdf"
OUT_EPS = "figures/al_illustration.eps"
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
# EPS can't store transparency, so derive it from the PDF: every artist here
# is already an opaque vector fill, and pdftops keeps the .eps pure vector.
subprocess.run(["pdftops", "-eps", OUT_PDF, OUT_EPS], check=True)
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
print(f"wrote {OUT_EPS}")
for gamma in GAMMAS:
    print(f"h minimiser  gamma={gamma:<4}: x_min = {h_al_argmin(gamma, OMEGA):.4f}")
print(f"P minimiser  rho={RHO}:   x_min = {RHO / (2.0 + RHO):.4f}  (infeasible)")

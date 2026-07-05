"""
Publication figure: one CMA generation as three panels.

(a) Sampling      — two parents with isotropic C, principal axes, two offspring.
(b) Selection     — best two survive; bad offspring + replaced parent fade out.
(c) Rank-one      — surviving offspring's covariance is updated toward the
                    successful step direction.

The geometry is constructed analytically (not from a real run) so the
pedagogy is clean.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse, FancyArrowPatch
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
# Objective: tilted, elongated quadratic bowl
#   f(x) = (x - x*)^T Q (x - x*),  Q = R diag(λ_small, λ_large) R^T
# Smaller eigenvalue ⇒ direction of slow change ⇒ the valley.
# ---------------------------------------------------------------------------
X_STAR = np.array([2.0, 2.0])
THETA = np.radians(35.0)
R = np.array([[np.cos(THETA), -np.sin(THETA)],
              [np.sin(THETA),  np.cos(THETA)]])
LAMBDA = np.diag([0.25, 2.5])
Q = R @ LAMBDA @ R.T


def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx, dy = x - X_STAR[0], y - X_STAR[1]
    return Q[0, 0] * dx * dx + 2.0 * Q[0, 1] * dx * dy + Q[1, 1] * dy * dy


# ---------------------------------------------------------------------------
# Individuals
# ---------------------------------------------------------------------------
P1 = np.array([-1.6, -1.0])   # parent 1
P2 = np.array([ 0.6, -2.2])   # parent 2
SIGMA = 0.55                  # isotropic step size
C_ISO = SIGMA ** 2 * np.eye(2)

# Offspring chosen by hand to give clean geometry:
#   O_GOOD: P1 + step roughly along the valley → lower f
#   O_BAD : P2 + step away from valley         → higher f
O_GOOD = P1 + np.array([1.05, 0.75])
O_BAD  = P2 + np.array([0.8, -0.9])

# Rank-one update (exaggerated c1 for visibility)
C1 = 0.6
step_good = O_GOOD - P1
C_NEW = (1.0 - C1) * C_ISO + C1 * np.outer(step_good, step_good)

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
N_STD = 1.5  # iso-probability radius in σ-units

COL_P1 = "#1f5fbf"   # blue
COL_P2 = "#1f5fbf"
COL_OG = "#1f5fbf"
COL_OB = "#c0392b"   # red for the bad offspring
COL_NEW_ELLIPSE = "#1f5fbf"
COL_CONTOUR = "#cccccc"
FADE = 0.18         # alpha for de-emphasised individuals in (b) and (c)


def _blend(color, alpha, bg="white"):
    """Flatten a semi-transparent colour onto `bg`, returning an opaque hex.

    EPS has no alpha channel, so any transparent artist forces pdftops to
    rasterise that region into a bitmap (the source of the graininess).
    Pre-blending the colour here keeps every artist a solid vector fill, so
    the .eps stays pure vector and stays sharp at any zoom.
    """
    c = np.asarray(mcolors.to_rgb(color))
    b = np.asarray(mcolors.to_rgb(bg))
    return mcolors.to_hex(alpha * c + (1.0 - alpha) * b)


def _eig_sorted(cov: np.ndarray):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def draw_ellipse(ax, mean, cov, *, edgecolor, lw=1.2, ls=":", alpha=1.0,
                 facecolor="none", zorder=2):
    vals, vecs = _eig_sorted(cov)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2.0 * N_STD * np.sqrt(vals)
    ax.add_patch(Ellipse(mean, w, h, angle=angle,
                         edgecolor=edgecolor, facecolor=facecolor,
                         lw=lw, ls=ls, alpha=alpha, zorder=zorder))


AXIS_ALPHA = 0.5  # principal axes fainter than the iso-density ellipse


def draw_principal_axes(ax, mean, cov, *, color, lw=0.9, alpha=AXIS_ALPHA,
                        zorder=2):
    vals, vecs = _eig_sorted(cov)
    for v, vec in zip(vals, vecs.T):
        end = mean + N_STD * np.sqrt(v) * vec
        arrow = FancyArrowPatch(tuple(mean), tuple(end),
                                arrowstyle="->", mutation_scale=8,
                                color=color, lw=lw, ls=":",
                                alpha=alpha, zorder=zorder,
                                shrinkA=0, shrinkB=0)
        ax.add_patch(arrow)


def draw_point(ax, p, *, color, marker="o", size=55, label=None,
               edge="white", alpha=1.0, zorder=4):
    ax.scatter(p[0], p[1], s=size, c=color, marker=marker,
               edgecolors=edge, linewidths=1.0, alpha=alpha,
               zorder=zorder, label=label)


def draw_step_arrow(ax, src, dst, *, color, alpha=1.0, zorder=3):
    arrow = FancyArrowPatch(src, dst,
                            arrowstyle="->", mutation_scale=10,
                            color=color, lw=1.0, alpha=alpha,
                            zorder=zorder, shrinkA=6, shrinkB=6)
    ax.add_patch(arrow)


def setup_panel(ax, title=None):
    xg = np.linspace(-3.0, 3.5, 300)
    yg = np.linspace(-3.5, 3.0, 300)
    Xg, Yg = np.meshgrid(xg, yg)
    Z = f(Xg, Yg)
    levels = np.geomspace(0.4, 40.0, 9)
    ax.contour(Xg, Yg, Z, levels=levels, colors=COL_CONTOUR,
               linewidths=0.7, zorder=1)
    ax.scatter(*X_STAR, marker="*", s=80, color="#999999",
               edgecolors="white", linewidths=0.8, zorder=2)
    ax.set_xlim(-3.0, 3.5)
    ax.set_ylim(-3.5, 3.0)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    if title:
        ax.set_title(title)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

# --- (a) Sampling -----------------------------------------------------------
ax = axes[0]
setup_panel(ax, title="Sampling")
ax.set_ylabel(r"$x_2$")

for P, col in [(P1, COL_P1), (P2, COL_P2)]:
    draw_ellipse(ax, P, C_ISO, edgecolor=col, lw=1.1)
    draw_point(ax, P, color=col, marker="o")

# Sampling arrows parent → offspring
draw_step_arrow(ax, P1, O_GOOD, color="#888888")
draw_step_arrow(ax, P2, O_BAD,  color="#888888")

draw_point(ax, O_GOOD, color=COL_OG, marker="^")
draw_point(ax, O_BAD,  color=COL_OB, marker="^")

# Labels
ax.annotate(r"$p_1$", P1, xytext=(-14, -14), textcoords="offset points",
            fontsize=10, color=COL_P1)
ax.annotate(r"$p_2$", P2, xytext=(-14, -14), textcoords="offset points",
            fontsize=10, color=COL_P2)
ax.annotate(r"$o_1$ (good)", O_GOOD, xytext=(8, 4),
            textcoords="offset points", fontsize=9, color=COL_OG)
ax.annotate(r"$o_2$ (bad)",  O_BAD,  xytext=(8, -10),
            textcoords="offset points", fontsize=9, color=COL_OB)

# --- (b) Selection ----------------------------------------------------------
ax = axes[1]
setup_panel(ax, title="Selection")

# Survivors: P2 (untouched) and O_GOOD (replaces P1).
# Faded: P1 (replaced) and O_BAD (discarded).
draw_ellipse(ax, P1, C_ISO, edgecolor=_blend(COL_P1, FADE), lw=0.9)
draw_point(ax, P1, color=_blend(COL_P1, FADE), marker="o")
draw_point(ax, O_BAD, color=COL_OB, marker="^")

# Step that produced the surviving offspring (same style as in panel a).
draw_step_arrow(ax, P1, O_GOOD, color="#888888")

# Survivors emphasised — same marker size as panel (a); the bold ellipse
# outline and dark edge are what distinguish them.
draw_ellipse(ax, P2, C_ISO, edgecolor=COL_P2, lw=1.6)
draw_point(ax, P2, color=COL_P2, marker="o", edge="black", zorder=6)

draw_point(ax, O_GOOD, color=COL_OG, marker="^", edge="black", zorder=6)

ax.annotate(r"$p_2$", P2, xytext=(8, -14),
            textcoords="offset points", fontsize=10, color=COL_P2)
ax.annotate(r"$o_1 \rightarrow p_1'$", O_GOOD, xytext=(8, 4),
            textcoords="offset points", fontsize=10, color=COL_OG)

# --- (c) Rank-one update ----------------------------------------------------
ax = axes[2]
setup_panel(ax, title="Rank-one covariance update")

# Surviving parent p_2 keeps its isotropic C.
draw_ellipse(ax, P2, C_ISO, edgecolor=COL_P2, lw=1.2)
draw_point(ax, P2, color=COL_P2, marker="o", size=80, edge="black")

# Surviving offspring (now p_1') gets the updated covariance.
draw_ellipse(ax, O_GOOD, C_NEW, edgecolor=COL_NEW_ELLIPSE, lw=1.6)
draw_principal_axes(ax, O_GOOD, C_NEW, color=_blend(COL_NEW_ELLIPSE, AXIS_ALPHA),
                    lw=1.2, alpha=1.0)
draw_point(ax, O_GOOD, color=COL_OG, marker="^", edge="black")

ax.annotate(r"$p_2$", P2, xytext=(8, -14),
            textcoords="offset points", fontsize=10, color=COL_P2)
ax.annotate(r"$p_1'$", O_GOOD, xytext=(10, 6),
            textcoords="offset points", fontsize=10, color=COL_NEW_ELLIPSE)

# ---------------------------------------------------------------------------
# (a) (b) (c) panel labels below each axis
# ---------------------------------------------------------------------------
for ax, lab in zip(axes, ["(a)", "(b)", "(c)"]):
    ax.text(0.5, -0.18, lab, transform=ax.transAxes,
            ha="center", va="top", fontsize=12)

fig.tight_layout(rect=(0, 0.02, 1, 1))

OUT_PNG = "figures/cmaes_step_illustration.png"
OUT_PDF = "figures/cmaes_step_illustration.pdf"
OUT_EPS = "figures/cmaes_step_illustration.eps"
import os
import subprocess
os.makedirs("figures", exist_ok=True)
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
# EPS can't store transparency, so derive it from the PDF: pdftops flattens
# the alpha (faded) artists during conversion, keeping the .eps faithful.
subprocess.run(["pdftops", "-eps", OUT_PDF, OUT_EPS], check=True)
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
print(f"wrote {OUT_EPS}")

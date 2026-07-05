"""
Publication figure: one (1+1) MO-CMA generation with two objectives.

(a) Sampling      — three parents with isotropic C; each produces an offspring.
(b) Selection     — offspring that are non-dominated w.r.t. the parent
                    population replace their parent; the dominated offspring
                    is discarded and its parent is kept.
(c) Rank-one      — every surviving offspring updates its covariance toward
                    its successful step direction. The parent with no
                    successful offspring carries its isotropic C unchanged.

The geometry is constructed analytically (not from a real run); positions
are chosen so the dominance relationships are clear, and verified by an
assertion at the bottom of the configuration block.
"""

from __future__ import annotations

import os
import subprocess
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
# Two objectives — tilted, elongated quadratic bowls with distinct optima.
# Different valley orientations so the Pareto-optimal set in decision space
# is a curve connecting the two minima.
# ---------------------------------------------------------------------------
X_STAR_1 = np.array([ 1.0, -0.5])
X_STAR_2 = np.array([-0.5,  1.0])


def _quad_form(theta_deg: float, lam_small: float, lam_large: float):
    theta = np.radians(theta_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return R @ np.diag([lam_small, lam_large]) @ R.T


Q1 = _quad_form( 35.0, 0.4, 1.6)
Q2 = _quad_form(-35.0, 0.4, 1.6)


def f1(x, y):
    dx, dy = x - X_STAR_1[0], y - X_STAR_1[1]
    return Q1[0, 0] * dx * dx + 2 * Q1[0, 1] * dx * dy + Q1[1, 1] * dy * dy


def f2(x, y):
    dx, dy = x - X_STAR_2[0], y - X_STAR_2[1]
    return Q2[0, 0] * dx * dx + 2 * Q2[0, 1] * dx * dy + Q2[1, 1] * dy * dy


def F(p):
    return np.array([f1(p[0], p[1]), f2(p[0], p[1])])


def pareto_front_curve(n_alpha: int = 120):
    """Analytical Pareto-optimal set in decision space.

    For two quadratic objectives, every Pareto-optimal point minimises some
    convex combination alpha*f1 + (1-alpha)*f2. Setting the gradient to zero
    gives a closed-form linear system that traces the front as alpha sweeps
    from 0 (anchored at x_2*) to 1 (anchored at x_1*).
    """
    alphas = np.linspace(0.0, 1.0, n_alpha)
    pts = np.empty((n_alpha, 2))
    for i, a in enumerate(alphas):
        M = a * Q1 + (1.0 - a) * Q2
        rhs = a * (Q1 @ X_STAR_1) + (1.0 - a) * (Q2 @ X_STAR_2)
        pts[i] = np.linalg.solve(M, rhs)
    return pts


PARETO_CURVE = pareto_front_curve()


def dominates(a, b):
    """True iff F(a) Pareto-dominates F(b)."""
    fa, fb = F(a), F(b)
    return np.all(fa <= fb) and np.any(fa < fb)


# ---------------------------------------------------------------------------
# Population: three parents, each samples one offspring.
# Offspring positions are hand-chosen so:
#   o_1, o_2 are non-dominated w.r.t. the parent set
#   o_3      is dominated by every other individual
# ---------------------------------------------------------------------------
P1 = np.array([ 0.25,  1.50])    # above the front (f2 side)
P2 = np.array([-1.00, -0.50])    # below the front (low x, mid y)
P3 = np.array([ 1.30, -0.80])    # close to x_1* on the front

SIGMA = 0.35
C_ISO = SIGMA ** 2 * np.eye(2)

# Surviving offspring steps point toward the analytical Pareto front
# (lower-left from P1, upper-right from P2), which makes them
# Pareto-improving and therefore non-dominated. The third offspring
# steps further into the dominated region (lower-left from P3) so its
# HV contribution is zero and it is discarded.
O1 = P1 + np.array([-0.35, -0.35])
O2 = P2 + np.array([ 0.50,  0.30])
O3 = P3 + np.array([-0.50, -0.50])

# Rank-one update (exaggerated c1 for visibility)
C1 = 0.6
step_1 = O1 - P1
step_2 = O2 - P2
C_NEW_1 = (1.0 - C1) * C_ISO + C1 * np.outer(step_1, step_1)
C_NEW_2 = (1.0 - C1) * C_ISO + C1 * np.outer(step_2, step_2)

# Verify the discard scenario: O3 must be dominated by at least one other
# individual (so its HV contribution is zero), while O1 and O2 must be
# non-dominated.
_pop = [P1, P2, P3, O1, O2, O3]
_o3_dominated = any(
    dominates(other, O3) for other in _pop if other is not O3
)
_o1_non_dominated = not any(
    dominates(other, O1) for other in _pop if other is not O1
)
_o2_non_dominated = not any(
    dominates(other, O2) for other in _pop if other is not O2
)
assert _o3_dominated, "o_3 must be dominated by some other individual"
assert _o1_non_dominated, "o_1 must be non-dominated"
assert _o2_non_dominated, "o_2 must be non-dominated"


# ---------------------------------------------------------------------------
# Drawing helpers (mirrored from the single-objective figure for consistency)
# ---------------------------------------------------------------------------
N_STD = 1.5
AXIS_ALPHA = 0.5
FADE = 0.18

COL_IND = "#1f5fbf"        # all "live" individuals share a colour
COL_OG = "#1f5fbf"
COL_OB = "#c0392b"         # red for the dominated offspring
COL_F1 = "#bdbdbd"         # grey contours for f1
COL_F2 = "#d8a3a3"         # dusty salmon contours for f2
COL_X1 = "#666666"
COL_X2 = "#a85050"


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
    ax.add_patch(FancyArrowPatch(tuple(src), tuple(dst),
                                 arrowstyle="->", mutation_scale=10,
                                 color=color, lw=1.0, alpha=alpha,
                                 zorder=zorder, shrinkA=6, shrinkB=6))


def setup_panel(ax, title=None):
    xg = np.linspace(-1.8, 2.2, 300)
    yg = np.linspace(-2.2, 2.2, 300)
    Xg, Yg = np.meshgrid(xg, yg)
    Z1 = f1(Xg, Yg)
    Z2 = f2(Xg, Yg)
    levels = np.geomspace(0.15, 10.0, 8)
    ax.contour(Xg, Yg, Z1, levels=levels, colors=COL_F1,
               linewidths=0.7, zorder=1)
    ax.contour(Xg, Yg, Z2, levels=levels, colors=COL_F2,
               linewidths=0.7, zorder=1)
    ax.plot(PARETO_CURVE[:, 0], PARETO_CURVE[:, 1],
            color="#444444", lw=1.1, zorder=1.5,
            label="Pareto-optimal set")
    ax.scatter(*X_STAR_1, marker="*", s=90, color=COL_X1,
               edgecolors="white", linewidths=0.8, zorder=2)
    ax.scatter(*X_STAR_2, marker="*", s=90, color=COL_X2,
               edgecolors="white", linewidths=0.8, zorder=2)
    ax.set_xlim(-1.8, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    if title:
        ax.set_title(title)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))

# --- (a) Sampling -----------------------------------------------------------
ax = axes[0]
setup_panel(ax, title="Sampling")
ax.set_ylabel(r"$x_2$")

for P, name, dx, dy in [(P1, r"$p_1$",  8,  6),
                        (P2, r"$p_2$",   4, -8),
                        (P3, r"$p_3$",   4, -8)]:
    draw_ellipse(ax, P, C_ISO, edgecolor=COL_IND, lw=1.1)
    draw_point(ax, P, color=COL_IND, marker="o")
    ax.annotate(name, P, xytext=(dx, dy), textcoords="offset points",
                fontsize=10, color=COL_IND)

draw_step_arrow(ax, P1, O1, color="#888888")
draw_step_arrow(ax, P2, O2, color="#888888")
draw_step_arrow(ax, P3, O3, color="#888888")

draw_point(ax, O1, color=COL_OG, marker="^")
draw_point(ax, O2, color=COL_OG, marker="^")
draw_point(ax, O3, color=COL_OB, marker="^")

ax.annotate(r"$o_1$", O1, xytext=(4,  -2), textcoords="offset points",
            fontsize=9, color=COL_OG)
ax.annotate(r"$o_2$", O2, xytext=(4,  4), textcoords="offset points",
            fontsize=9, color=COL_OG)
ax.annotate(r"$o_3$ (dominated)", O3, xytext=(4, -9),
            textcoords="offset points", fontsize=9, color=COL_OB)

# --- (b) Selection ----------------------------------------------------------
ax = axes[1]
setup_panel(ax, title="Selection")

# Faded: replaced parents P1, P2 + discarded offspring O3.
for P in (P1, P2):
    draw_ellipse(ax, P, C_ISO, edgecolor=_blend(COL_IND, FADE), lw=0.9)
    draw_point(ax, P, color=_blend(COL_IND, FADE), marker="o")
draw_point(ax, O3, color=COL_OB, marker="^")

# Step arrows for the successful samples (same style as panel a).
draw_step_arrow(ax, P1, O1, color="#888888")
draw_step_arrow(ax, P2, O2, color="#888888")

# Surviving parent p_3 — keeps its isotropic C; emphasised.
draw_ellipse(ax, P3, C_ISO, edgecolor=COL_IND, lw=1.6)
draw_point(ax, P3, color=COL_IND, marker="o", edge="black", zorder=6)

# Surviving offspring — emphasised.
draw_point(ax, O1, color=COL_OG, marker="^", edge="black", zorder=6)
draw_point(ax, O2, color=COL_OG, marker="^", edge="black", zorder=6)

ax.annotate(r"$p_3$", P3, xytext=(8, -14),
            textcoords="offset points", fontsize=10, color=COL_IND)
ax.annotate(r"$o_1 \rightarrow p_1'$", O1, xytext=(13, 4),
            textcoords="offset points", fontsize=10, color=COL_OG)
ax.annotate(r"$o_2 \rightarrow p_2'$", O2, xytext=(8, 4),
            textcoords="offset points", fontsize=10, color=COL_OG)

# --- (c) Rank-one update ----------------------------------------------------
ax = axes[2]
setup_panel(ax, title="Rank-one covariance update")

# p_3 retains its isotropic C.
draw_ellipse(ax, P3, C_ISO, edgecolor=COL_IND, lw=1.2)
draw_point(ax, P3, color=COL_IND, marker="o", edge="black")

# p_1', p_2' get rank-one updated covariances.
for centre, C_new, label, dx, dy in [
    (O1, C_NEW_1, r"$p_1'$", 10,  6),
    (O2, C_NEW_2, r"$p_2'$", 10,  6),
]:
    draw_ellipse(ax, centre, C_new, edgecolor=COL_IND, lw=1.6)
    draw_principal_axes(ax, centre, C_new, color=_blend(COL_IND, AXIS_ALPHA),
                        lw=1.2, alpha=1.0)
    draw_point(ax, centre, color=COL_OG, marker="^", edge="black")
    ax.annotate(label, centre, xytext=(dx, dy),
                textcoords="offset points", fontsize=10, color=COL_IND)

ax.annotate(r"$p_3$", P3, xytext=(8, -14),
            textcoords="offset points", fontsize=10, color=COL_IND)

# ---------------------------------------------------------------------------
# (a) (b) (c) panel labels
# ---------------------------------------------------------------------------
for ax, lab in zip(axes, ["(a)", "(b)", "(c)"]):
    ax.text(0.5, -0.18, lab, transform=ax.transAxes,
            ha="center", va="top", fontsize=12)

fig.tight_layout(rect=(0, 0.02, 1, 1))

os.makedirs("figures", exist_ok=True)
OUT_PNG = "figures/mocmaes_step_illustration.png"
OUT_PDF = "figures/mocmaes_step_illustration.pdf"
OUT_EPS = "figures/mocmaes_step_illustration.eps"
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
# EPS can't store transparency, so derive it from the PDF: pdftops flattens
# the alpha (faded) artists during conversion, keeping the .eps faithful.
subprocess.run(["pdftops", "-eps", OUT_PDF, OUT_EPS], check=True)
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
print(f"wrote {OUT_EPS}")

"""
Publication figure: Arnold's CHT covariance squeeze at a feasibility corner.

Arnold's CHT is built for (1+1)-CMA-ES — each generation samples exactly
one offspring. The figure shows three distinct moments in the algorithm's
evolution, so the depicted state of v_j and C is internally consistent
at every panel.

(a) Generation 1   — parent with isotropic C samples one offspring; it
                     violates both constraints. The accumulators have not
                     yet been built up, so no v_j arrows are drawn — this
                     single Az kicks off the EMA.
(b) After many gens — v_1, v_2 have converged to their EMA state
                     (perpendicular to each constraint's interior normal),
                     and C has been progressively rank-2-squeezed by the
                     cumulative updates

                        A' = A - (beta/m_active) * sum_j (v_j w_j^T) / (w_j^T w_j)

                     producing the visibly anisotropic C'.
(c) Subsequent gen — parent samples one offspring from N(p, C'); the
                     squeeze makes it feasible.

Numerics use illustrative values (beta exaggerated for visibility); the
algorithmic form mirrors `_arnold_update_A` in src/algorithm/cmaes.py.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse, FancyArrowPatch


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
# Constraint setup
#   g_1(x) = -(x_2 + 0.4) <= 0   (horizontal floor; infeasible when x_2 < -0.4)
#   g_2(x) = 0.6 x_1 - x_2 - 0.6 <= 0
#                                 (sloped wall; infeasible when x_2 < 0.6 x_1 - 0.6)
# Outward normals (pointing INTO infeasible region) are the v_j directions
# the Arnold accumulators converge to after many violations.
# ---------------------------------------------------------------------------
def g1(x):
    return -(x[..., 1] + 0.4)


def g2(x):
    return 0.6 * x[..., 0] - x[..., 1] - 0.6


# Outward normals (unit vectors pointing into the infeasible side).
N1 = np.array([0.0, -1.0])
N2 = np.array([0.6, -1.0]) / np.linalg.norm([0.6, -1.0])

# Parent and initial covariance.
PARENT = np.array([0.0, 0.0])
SIGMA = 0.60
C_ISO = SIGMA ** 2 * np.eye(2)
A_ISO = SIGMA * np.eye(2)         # Cholesky factor of C_ISO

# Arnold parameters (illustrative; real values are smaller).
BETA = 0.50

# Per-constraint accumulators: at their converged direction (== outward
# normal) with a representative magnitude.
V_MAG = 0.55
V1 = V_MAG * N1
V2 = V_MAG * N2


def arnold_A_update(A: np.ndarray, v_list) -> np.ndarray:
    """Subtractive Cholesky-factor update from _arnold_update_A.

    A' = A - (beta / m_active) * sum_j (v_j w_j^T) / (w_j^T w_j),
    w_j = A^{-1} v_j.  Mirrors src/algorithm/cmaes.py:1178-1209.
    """
    m_active = len(v_list)
    if m_active == 0:
        return A.copy()
    A_inv = np.linalg.inv(A)
    delta = np.zeros_like(A)
    for v_j in v_list:
        w_j = A_inv @ v_j
        denom = float(w_j @ w_j)
        if denom < 1e-30:
            continue
        delta += np.outer(v_j, w_j) / denom
    return A - (BETA / m_active) * delta


A_NEW = arnold_A_update(A_ISO, [V1, V2])
C_NEW = A_NEW @ A_NEW.T


# ---------------------------------------------------------------------------
# (1+1): one offspring per generation. Positions are chosen deterministically
# so panel (a) shows a clear double-violation and panel (c) shows that the
# squeezed distribution produces a feasible sample at a comparable
# standardised step magnitude.
# ---------------------------------------------------------------------------
OFFSPRING_A = PARENT + np.array([-0.30, -1.00])   # generation g: infeasible
OFFSPRING_C = PARENT + np.array([-0.55,  0.30])   # generation g+1: feasible


def is_infeasible(pt: np.ndarray) -> bool:
    return bool((g1(pt) > 0) or (g2(pt) > 0))


assert is_infeasible(OFFSPRING_A), "panel (a) offspring must violate >= 1 constraint"
assert not is_infeasible(OFFSPRING_C), "panel (c) offspring must be feasible"


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
N_STD = 1.5
COL_FEAS = "#1f5fbf"
COL_INFEAS = "#c0392b"
COL_PARENT = "#1f5fbf"
COL_V = "#444444"
COL_BOUND = "#222222"
COL_INFEAS_FILL = "#f1d4d0"
COL_ELLIPSE_OLD = "#888888"
COL_ELLIPSE_NEW = "#1f5fbf"


def _eig_sorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def draw_ellipse(ax, mean, cov, *, edgecolor, lw=1.2, ls=":", alpha=1.0,
                 facecolor="none", zorder=3):
    vals, vecs = _eig_sorted(cov)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2.0 * N_STD * np.sqrt(vals)
    ax.add_patch(Ellipse(mean, w, h, angle=angle,
                         edgecolor=edgecolor, facecolor=facecolor,
                         lw=lw, ls=ls, alpha=alpha, zorder=zorder))


def draw_arrow(ax, src, dst, *, color, lw=1.4, mutation=12, ls="-",
               alpha=1.0, zorder=4, shrinkA=4, shrinkB=4):
    ax.add_patch(FancyArrowPatch(tuple(src), tuple(dst),
                                 arrowstyle="->", mutation_scale=mutation,
                                 color=color, lw=lw, ls=ls, alpha=alpha,
                                 zorder=zorder, shrinkA=shrinkA, shrinkB=shrinkB))


# ---------------------------------------------------------------------------
# Constraint geometry rendered on each panel
# ---------------------------------------------------------------------------
XLIM = (-2.0, 2.0)
YLIM = (-2.0, 1.6)


def draw_constraints(ax):
    xs = np.linspace(XLIM[0], XLIM[1], 200)

    # g_1: x_2 = -0.4  (floor)
    y1 = np.full_like(xs, -0.4)
    # g_2: x_2 = 0.6 x_1 - 0.6  (sloped wall)
    y2 = 0.6 * xs - 0.6

    # Infeasible fill for g_1 (everything below y=-0.4)
    ax.fill_between(xs, YLIM[0], y1, color=COL_INFEAS_FILL,
                    alpha=0.55, zorder=0.5, linewidth=0)
    # Infeasible fill for g_2 (everything below the line)
    ax.fill_between(xs, YLIM[0], y2, color=COL_INFEAS_FILL,
                    alpha=0.55, zorder=0.5, linewidth=0)

    # Boundary lines
    ax.plot(xs, y1, color=COL_BOUND, lw=1.1, zorder=1)
    ax.plot(xs, y2, color=COL_BOUND, lw=1.1, zorder=1)

    # Labels
    ax.text(-1.85, -0.32, r"$g_1(x)=0$", fontsize=9, color=COL_BOUND)
    ax.text(1.05, 0.10, r"$g_2(x)=0$", fontsize=9, color=COL_BOUND,
            rotation=np.degrees(np.arctan(0.6)))


def setup_panel(ax, title=None):
    draw_constraints(ax)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    if title:
        ax.set_title(title)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0))

# --- (a) Generation 1: first violation -------------------------------------
ax = axes[0]
setup_panel(ax, title=r"Generation 1: first violation")
ax.set_ylabel(r"$x_2$")

# Isotropic search distribution — no v_j yet.
draw_ellipse(ax, PARENT, C_ISO, edgecolor=COL_ELLIPSE_NEW, lw=1.2)

# Single (1+1) offspring — infeasible.
draw_arrow(ax, PARENT, OFFSPRING_A, color="#888888", lw=1.0,
           mutation=10, zorder=3, shrinkA=6, shrinkB=6)
ax.scatter(*OFFSPRING_A, s=55, color=COL_INFEAS, marker="^",
           edgecolors="white", linewidths=0.8, zorder=4)

# Parent.
ax.scatter(*PARENT, s=90, color=COL_PARENT, marker="o",
           edgecolors="black", linewidths=1.0, zorder=5)

ax.annotate(r"$p$", PARENT, xytext=(-14, 8),
            textcoords="offset points", fontsize=11, color=COL_PARENT)
ax.annotate(r"$o$", OFFSPRING_A, xytext=(8, -2),
            textcoords="offset points", fontsize=10, color=COL_INFEAS)

# --- (b) Rank-2 covariance squeeze -----------------------------------------
ax = axes[1]
setup_panel(ax, title=r"After many generations: $v_j$ EMA and squeezed $C'$")

# Faded original.
draw_ellipse(ax, PARENT, C_ISO, edgecolor=COL_ELLIPSE_OLD, lw=0.9, alpha=0.5)
# Updated.
draw_ellipse(ax, PARENT, C_NEW, edgecolor=COL_ELLIPSE_NEW, lw=1.6)

# Accumulators retained (they drive the update).
draw_arrow(ax, PARENT, PARENT + V1, color=COL_V, lw=1.6, mutation=14)
draw_arrow(ax, PARENT, PARENT + V2, color=COL_V, lw=1.6, mutation=14)

ax.scatter(*PARENT, s=90, color=COL_PARENT, marker="o",
           edgecolors="black", linewidths=1.0, zorder=5)
ax.annotate(r"$p$", PARENT, xytext=(8, 8),
            textcoords="offset points", fontsize=11, color=COL_PARENT)
ax.annotate(r"$v_1$", PARENT + V1, xytext=(-22, -2),
            textcoords="offset points", fontsize=10, color=COL_V)
ax.annotate(r"$v_2$", PARENT + V2, xytext=(8, -2),
            textcoords="offset points", fontsize=10, color=COL_V)
ax.text(XLIM[0] + 0.1, YLIM[1] - 0.18,
        r"$v_j$: EMA of past standardised steps that violated $g_j$",
        fontsize=9, color=COL_V)

# --- (c) Generation g+1: single feasible offspring -------------------------
ax = axes[2]
setup_panel(ax, title=r"Subsequent generation: feasible sample from $C'$")

draw_ellipse(ax, PARENT, C_NEW, edgecolor=COL_ELLIPSE_NEW, lw=1.4)

# Single (1+1) offspring — feasible under the squeezed distribution.
draw_arrow(ax, PARENT, OFFSPRING_C, color="#888888", lw=1.0,
           mutation=10, zorder=3, shrinkA=6, shrinkB=6)
ax.scatter(*OFFSPRING_C, s=55, color=COL_FEAS, marker="^",
           edgecolors="white", linewidths=0.8, zorder=4)

ax.scatter(*PARENT, s=90, color=COL_PARENT, marker="o",
           edgecolors="black", linewidths=1.0, zorder=5)
ax.annotate(r"$p$", PARENT, xytext=(8, 8),
            textcoords="offset points", fontsize=11, color=COL_PARENT)
ax.annotate(r"$o'$", OFFSPRING_C, xytext=(-18, 4),
            textcoords="offset points", fontsize=10, color=COL_FEAS)

# ---------------------------------------------------------------------------
# Panel labels
# ---------------------------------------------------------------------------
for ax, lab in zip(axes, ["(a)", "(b)", "(c)"]):
    ax.text(0.5, -0.18, lab, transform=ax.transAxes,
            ha="center", va="top", fontsize=12)

fig.tight_layout(rect=(0, 0.02, 1, 1))

os.makedirs("figures", exist_ok=True)
OUT_PNG = "figures/arnold_cht_illustration.png"
OUT_PDF = "figures/arnold_cht_illustration.pdf"
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
print(f"C original eigenvalues: {np.linalg.eigvalsh(C_ISO)}")
print(f"C' updated  eigenvalues: {np.linalg.eigvalsh(C_NEW)}")

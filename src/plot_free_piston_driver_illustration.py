"""
Publication figure: free piston driver tube + characteristics of piston motion.

Three stacked panels:

    A   cross-section schematic of the driver tube (reservoir | piston | driver
        gas | exit orifice)
    B   piston phase plane u(x) over the full stroke
    C   magnified end-of-stroke, classifying the three landing regimes

Model
-----
The three trajectories are not drawn by hand. They are one family, integrated
from the same equations, separated only by the driver fill pressure p_D0.

Piston dynamics (x measured from launch, driver gas ahead, reservoir behind):

    m_p du/dt = A_D (p_R - p_D)

Reservoir: isentropic expansion of a fixed charge as the piston uncovers volume

    p_R(x) = p_R0 [ V_R0 / (V_R0 + A_D x) ]^gamma_R

Driver: the gas ahead of the piston is compressed into a shrinking volume
V_D = A_D (x_wall - x). While the primary diaphragm holds, its mass is fixed.
Once p_D reaches p_rupt the diaphragm bursts and gas discharges through the
orifice A_d. The gas *remaining* in the driver expands isentropically, so

    p_D = p_D0 (rho_D / rho_D0)^gamma_D,     rho_D = m_D / V_D

and the vent is quasi-steady choked flow (the shock tube downstream is at ~10
kPa against MPa in the driver, so the orifice is choked throughout):

    dm_D/dt = -C_d A_d rho_D a_D [2/(gamma_D+1)]^[(gamma_D+1)/(2(gamma_D-1))]

Why venting is load-bearing
---------------------------
du/dt = 0 wherever p_R = p_D. That balance is crossed *twice*:

  1. early, while the driver is still being compressed -- this is u_max;
  2. again near the end wall, but only because rupture has bled the driver
     pressure back down below the reservoir pressure.

The second crossing is the point marked (a)/(b)/(c), and the sign of the
velocity there, u_m, is the entire classification:

    (a) u_m < 0   piston has already reversed  -> rebounds, is re-accelerated
                  over the recovered distance, and strikes the wall hardest
    (b) u_m = 0   velocity and acceleration vanish together -> soft landing
    (c) u_m > 0   piston never reverses -> direct impact

Without mass loss the system is conservative, the phase trajectory is symmetric
under u -> -u, and the rebound loop of case (a) cannot exist at all. The vent
is what opens the loop.

Calibration
-----------
Geometry and piston mass are X2-like (257 mm bore, 10.524 kg piston, 4.47 m
stroke to the diaphragm station, 85 mm orifice, 27.9 MPa diaphragm). The
soft-landing fill is *solved*, not chosen: bisection on p_D0 for min(u) = 0
converges to 87.408 kPa, giving u_m = 0 to ~1e-12 m/s. The other two cases are
that fill +/- 10%, so the figure shows one physical parameter being swept
through its critical value rather than three unrelated curves.

Usage
-----
    python src/plot_free_piston_driver_illustration.py
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch, ConnectionPatch
from scipy.integrate import solve_ivp


rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
})

INK = "black"
COL_A, COL_B, COL_C = "#0000cc", "black", "#cc0000"   # rebound / soft / direct


# ===========================================================================
# 1. Physical model
# ===========================================================================
BORE = 0.257                      # compression tube bore [m]
A_D = np.pi * BORE ** 2 / 4.0     # piston face area [m^2]
M_P = 10.524                      # piston mass [kg]
X_WALL = 4.47                     # diaphragm station / end wall [m]

GAM_R, GAM_D = 1.4, 5.0 / 3.0     # reservoir air, driver helium
R_D, T_D0 = 2077.0, 295.15        # helium gas constant [J/kg/K], fill temp [K]
P_R0, V_R0 = 4.3e6, 0.064         # reservoir charge [Pa], [m^3]

D_ORIF = 0.085                    # exit orifice (shock tube bore) [m]
A_ORIF = np.pi * D_ORIF ** 2 / 4.0
C_D = 0.85                        # orifice discharge coefficient
P_RUPT = 27.9e6                   # primary diaphragm burst pressure [Pa]

# Choked-flow constant [2/(g+1)]^[(g+1)/(2(g-1))]
G_FLOW = (2.0 / (GAM_D + 1.0)) ** ((GAM_D + 1.0) / (2.0 * (GAM_D - 1.0)))


def p_reservoir(x):
    """Isentropic expansion of the reservoir charge behind the piston."""
    return P_R0 * (V_R0 / (V_R0 + A_D * x)) ** GAM_R


def simulate(p_D0, t_max=0.6):
    """Integrate one piston trajectory for a given driver fill pressure."""
    rho_D0 = p_D0 / (R_D * T_D0)
    m_D0 = rho_D0 * A_D * X_WALL

    def driver(x, m_D):
        rho = m_D / (A_D * (X_WALL - x))
        return p_D0 * (rho / rho_D0) ** GAM_D, rho

    def rhs(t, y, venting):
        x, u, m_D = y
        p_D, rho = driver(x, m_D)
        du = A_D * (p_reservoir(x) - p_D) / M_P
        if venting and m_D > 1e-9:
            a_D = np.sqrt(GAM_D * p_D / rho)
            return [u, du, -C_D * A_ORIF * rho * a_D * G_FLOW]
        return [u, du, 0.0]

    def ev_rupture(t, y, venting):
        return driver(y[0], y[2])[0] - P_RUPT
    ev_rupture.terminal, ev_rupture.direction = True, 1.0

    def ev_wall(t, y, venting):
        return X_WALL - y[0] - 1e-4
    ev_wall.terminal, ev_wall.direction = True, -1.0

    kw = dict(rtol=1e-9, atol=[1e-9, 1e-7, 1e-12], max_step=2e-5)
    # Stage 1: diaphragm intact, driver mass fixed. Stops at burst.
    s1 = solve_ivp(rhs, (0.0, t_max), [0.0, 0.0, m_D0], args=(False,),
                   events=[ev_rupture, ev_wall], **kw)
    segs, x_rupt = [s1], None
    if s1.t_events[0].size:
        # Stage 2: diaphragm burst, driver vents through the orifice.
        x_rupt = s1.y[0, -1]
        segs.append(solve_ivp(rhs, (s1.t[-1], t_max), s1.y[:, -1],
                              args=(True,), events=[ev_wall], **kw))

    x = np.concatenate([s.y[0] for s in segs])
    u = np.concatenate([s.y[1] for s in segs])
    i_max = int(np.argmax(u))
    i_m = i_max + int(np.argmin(u[i_max:]))       # 2nd p_R = p_D crossing
    return dict(x=x, u=u, x_rupt=x_rupt, u_max=u[i_max], x_u_max=x[i_max],
                u_m=u[i_m], x_m=x[i_m], u_wall=u[-1])


def solve_soft_landing(lo=70e3, hi=140e3, iters=40):
    """Bisect the driver fill for which u and du/dt vanish together.

    min(u) after u_max decreases monotonically with fill pressure: a stiffer
    gas spring stops the piston sooner. The sign change of u_m therefore
    brackets the critical fill exactly.
    """
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if simulate(mid)["u_m"] > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ===========================================================================
# 2. Drawing helpers
# ===========================================================================
def hatched(ax, verts, spacing=0.16, lw=0.45, fc="white", zorder=2):
    """Filled outlined polygon overlaid with 45-degree hatch lines.

    Drawn explicitly rather than with matplotlib's `hatch=` so the result is
    guaranteed plain vector strokes in the EPS (pattern fills can be
    rasterised by the PDF->PS converter).
    """
    poly = Polygon(verts, closed=True, facecolor=fc, edgecolor=INK,
                   lw=0.9, zorder=zorder, joinstyle="miter")
    ax.add_patch(poly)
    xs, ys = np.array(verts).T
    step = spacing * np.sqrt(2.0)
    # Lines of unit slope: y = x - c. Sweep c across the polygon's bbox.
    c_lo, c_hi = xs.min() - ys.max(), xs.max() - ys.min()
    for c in np.arange(c_lo, c_hi + step, step):
        xa, xb = max(xs.min(), ys.min() + c), min(xs.max(), ys.max() + c)
        if xb <= xa:
            continue
        ln = Line2D([xa, xb], [xa - c, xb - c], lw=lw, color=INK,
                    zorder=zorder + 0.1)
        ln.set_clip_path(poly)
        ax.add_line(ln)
    return poly


def arrow(ax, tail, head, style="-|>", lw=0.8, scale=8, color=INK, **kw):
    ax.annotate("", xy=head, xytext=tail, zorder=6,
                arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                                shrinkA=0, shrinkB=0, mutation_scale=scale,
                                **kw))


def ticks_x(ax, y, positions, labels, dy=-3.0, size=8):
    """Tick marks + labels hung below a manually drawn horizontal axis."""
    for p, lab in zip(positions, labels):
        ax.annotate("", xy=(p, y), xytext=(0, dy), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=INK))
        ax.annotate(lab, xy=(p, y), xytext=(0, dy - 3.5), fontsize=size,
                    textcoords="offset points", ha="center", va="top")


def ticks_y(ax, x, positions, labels, dx=-3.0, size=8):
    """Tick marks + labels hung left of a manually drawn vertical axis."""
    for p, lab in zip(positions, labels):
        ax.annotate("", xy=(x, p), xytext=(dx, 0), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=INK))
        ax.annotate(lab, xy=(x, p), xytext=(dx - 2.0, 0), fontsize=size,
                    textcoords="offset points", ha="right", va="center")


def curve_head(ax, x, u, i, color, scale=8):
    """Arrowhead on a trajectory at sample i, pointing along the path."""
    j = max(0, i - 3)
    arrow(ax, (x[j], u[j]), (x[i], u[i]), color=color, lw=0.9, scale=scale)


# ===========================================================================
# 3. Panel A -- driver tube schematic
# ===========================================================================
H_IN, T_W = 1.0, 0.34            # inner half-height, wall thickness
H_OUT = H_IN + T_W
X_TUBE0, X_STEP, X_ENDF = 0.40, 13.90, 15.00
H_ORIF = 0.28                    # exit hole half-height
XP0, XP1 = 5.30, 7.45            # piston extent


def panel_schematic(ax):
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(-0.10, 16.30)
    ax.set_ylim(-1.85, 1.85)

    # Tube walls. The right end steps inward to leave the exit orifice, so the
    # wall and the end block are one polygon per side.
    top = [(X_TUBE0, H_IN), (X_STEP, H_IN), (X_STEP, H_ORIF),
           (X_ENDF, H_ORIF), (X_ENDF, H_OUT), (X_TUBE0, H_OUT)]
    hatched(ax, top, spacing=0.22)
    hatched(ax, [(x, -y) for x, y in top], spacing=0.22)

    # Break symbol: erase a slice of the walls, then draw two wavy edges.
    ax.add_patch(Rectangle((0.36, -H_OUT - 0.06), 0.30, 2 * H_OUT + 0.12,
                           facecolor="white", edgecolor="none", zorder=4))
    yy = np.linspace(-H_OUT - 0.04, H_OUT + 0.04, 200)
    for xw in (0.40, 0.62):
        ax.add_line(Line2D(xw + 0.055 * np.sin(2 * np.pi * yy / 0.62), yy,
                           lw=0.9, color=INK, zorder=5))

    # Piston: hollow cup opening rearward (light-alloy free piston).
    pist = [(XP0, -H_IN), (XP1, -H_IN), (XP1, H_IN), (XP0, H_IN),
            (XP0, 0.62), (XP0 + 1.35, 0.62), (XP0 + 1.35, -0.62),
            (XP0, -0.62)]
    ax.add_patch(Polygon(pist, closed=True, facecolor="#c9c9c9",
                         edgecolor=INK, lw=0.9, zorder=3))
    ax.add_patch(Rectangle((XP1 - 0.66, -0.40), 0.56, 0.80, facecolor="white",
                           edgecolor=INK, lw=0.8, zorder=4))
    ax.text(XP1 - 0.38, 0.0, r"$m_p$", ha="center", va="center", fontsize=8,
            zorder=5)

    # A_D across the bore, with the reservoir state below it.
    arrow(ax, (1.62, -H_IN), (1.62, H_IN), style="<|-|>", scale=7)
    ax.text(1.45, 0.45, r"$A_D$", ha="right", va="center")
    ax.text(3.05, -0.52, r"$p_R,\, a_R,\, \gamma_R$", ha="center", va="center")

    # Direction of piston travel, and the driver state ahead of it.
    arrow(ax, (8.15, 0.62), (9.55, 0.62), lw=1.3, scale=11)
    ax.text(9.75, 0.62, r"$u,\, x$", ha="left", va="center")
    ax.text(10.15, -0.52, r"$p_D,\, V_D,\, \rho_D,\, \gamma_D$",
            ha="center", va="center")

    # Driver gas discharging through the orifice after diaphragm rupture.
    ax.add_patch(FancyArrowPatch((12.15, 0.66), (14.30, 0.00),
                                 connectionstyle="arc3,rad=-0.30",
                                 arrowstyle="simple,head_length=8,"
                                            "head_width=7.5,tail_width=3.0",
                                 facecolor="white", edgecolor=INK, lw=0.8,
                                 zorder=5))
    arrow(ax, (15.28, -H_ORIF), (15.28, H_ORIF), style="<|-|>", scale=7)
    ax.text(15.45, 0.0, r"$A_d$", ha="left", va="center")

    ax.set_title("Illustration of a free piston driver tube.", fontsize=9,
                 pad=4)


# ===========================================================================
# 4. Panel B -- full-stroke characteristics
# ===========================================================================
ZOOM_X = (4.20, 4.50)
ZOOM_U = (-50.0, 100.0)


def panel_characteristics(ax, runs):
    ax.set_axis_off()
    ax.set_xlim(-0.10, 5.55)
    ax.set_ylim(-72.0, 308.0)

    for r, c in zip(runs, (COL_A, COL_B, COL_C)):
        ax.plot(r["x"], r["u"], color=c, lw=0.9, zorder=3)

    # Axes drawn by hand: a vertical u-axis, an x-axis arrow through u = 0,
    # and a separate ruled scale along the bottom.
    arrow(ax, (0.0, -50.0), (0.0, 278.0), scale=9)
    ax.text(0.10, 281.0, r"$u$ (m/s)", ha="left", va="bottom")
    ticks_y(ax, 0.0, range(-50, 300, 50), [str(v) for v in range(-50, 300, 50)])

    arrow(ax, (0.0, 0.0), (5.28, 0.0), scale=9)
    ax.text(5.36, 0.0, r"$x$ (m)", ha="left", va="center")

    xs = np.arange(0.0, 5.01, 0.5)
    ax.plot([0.0, 5.05], [-50.0, -50.0], color=INK, lw=0.8)
    ticks_x(ax, -50.0, xs, [f"{v:.1f}" for v in xs])

    # Region magnified in panel C.
    ax.add_patch(Rectangle((ZOOM_X[0], ZOOM_U[0]),
                           ZOOM_X[1] - ZOOM_X[0], ZOOM_U[1] - ZOOM_U[0],
                           facecolor="none", edgecolor=INK, lw=0.8,
                           ls=(0, (4, 3)), zorder=4))

    ax.set_title("Characteristics of piston motion.", fontsize=9, pad=2)


# ===========================================================================
# 5. Panel C -- magnified end of stroke
# ===========================================================================
def panel_zoom(ax, runs, x_m_soft):
    ax.set_axis_off()
    ax.set_xlim(4.176, 4.523)
    ax.set_ylim(-78.0, 150.0)

    box = Rectangle((ZOOM_X[0], ZOOM_U[0]), ZOOM_X[1] - ZOOM_X[0],
                    ZOOM_U[1] - ZOOM_U[0], facecolor="none", edgecolor=INK,
                    lw=0.8, ls=(0, (4, 3)), zorder=4)
    ax.add_patch(box)

    # End wall.
    ax.add_patch(Rectangle((X_WALL, ZOOM_U[0]), ZOOM_X[1] - X_WALL,
                           ZOOM_U[1] - ZOOM_U[0], facecolor="#b8b8b8",
                           edgecolor=INK, lw=0.7, zorder=2))
    ax.text(0.5 * (X_WALL + ZOOM_X[1]), 22.0, "End wall", rotation=90,
            ha="center", va="center", fontsize=7.5, zorder=3)

    for r, c in zip(runs, (COL_A, COL_B, COL_C)):
        m = (r["x"] >= ZOOM_X[0] - 0.01) & (r["u"] <= ZOOM_U[1] + 5)
        x, u = r["x"][m], r["u"][m]
        ln, = ax.plot(x, u, color=c, lw=0.9, zorder=3)
        ln.set_clip_path(box)
        curve_head(ax, x, u, len(x) - 1, c)          # impact on the end wall
        ax.plot([r["x_m"]], [r["u_m"]], "o", ms=3.4, color=c, zorder=5)

    # Axes.
    arrow(ax, (ZOOM_X[0], ZOOM_U[0]), (ZOOM_X[0], 140.0), scale=9)
    ax.text(ZOOM_X[0] + 0.006, 143.0, r"$u$ (m/s)", ha="left", va="bottom")
    ticks_y(ax, ZOOM_X[0], range(-50, 150, 50),
            [str(v) for v in range(-50, 150, 50)])
    xs = np.arange(4.20, 4.501, 0.05)
    ticks_x(ax, ZOOM_U[0], xs, [f"{v:.2f}" for v in xs])
    arrow(ax, (ZOOM_X[1], ZOOM_U[0]), (4.514, ZOOM_U[0]), scale=9)
    ax.text(4.517, ZOOM_U[0], r"$x$ (m)", ha="left", va="center", fontsize=8)

    # Regime names. Placed above the box: this driver decelerates hard enough
    # that all three curves sweep the full 0-100 m/s band inside it.
    for xt, txt, c in ((4.290, "Rebound\nimpact", COL_A),
                       (4.362, "Soft\nlanding", COL_B),
                       (4.432, "Direct\nimpact", COL_C)):
        ax.text(xt, 113.0, txt, ha="center", va="center", fontsize=8, color=c,
                linespacing=1.15, zorder=5)

    # The du/dt = 0 condition in each regime, anchored to its own marker.
    # Kept to one line each: the stacked fraction makes a two-line block taller
    # than the clear space under the rebound loop.
    cond = r"$\dfrac{du}{dt}=0$"
    lead = dict(arrowstyle="-|>", lw=0.7, shrinkA=3, shrinkB=5,
                mutation_scale=8)
    for run, c, tag, rel, xytext in (
            (runs[0], COL_A, "(a) ", r"$,\ u_m<0$", (4.272, -36.0)),
            (runs[1], COL_B, "(b) ", r"$,\ u_m=0$", (4.258, 62.0)),
            (runs[2], COL_C, "(c) ", r"$,\ u_m>0$", (4.424, 62.0))):
        ax.annotate(tag + cond + rel,
                    xy=(run["x_m"], run["u_m"]), xytext=xytext,
                    ha="center", va="center", fontsize=7.0, color=c,
                    zorder=5, arrowprops=dict(color=c, **lead))

    # L_m: the standoff the soft-landing piston still has to cover.
    ax.plot([x_m_soft, x_m_soft], [-48.0, 4.0], ls=(0, (1, 2)), lw=0.7,
            color=INK, zorder=3)
    arrow(ax, (x_m_soft, -42.0), (X_WALL, -42.0), style="<|-|>", scale=7)
    ax.text(0.5 * (x_m_soft + X_WALL), -39.0, r"$L_m$", ha="center",
            va="bottom", fontsize=8)


# ===========================================================================
# 6. Assemble
# ===========================================================================
def main() -> None:
    p_soft = solve_soft_landing()
    runs = [simulate(p_soft * f) for f in (1.10, 1.00, 0.90)]   # a, b, c

    fig = plt.figure(figsize=(6.4, 7.6))
    # Row 2 is an empty spacer: hspace is uniform, but the schematic wants a
    # tight gap below it while the zoom leaders need a wide one.
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1.25, 2.05, 0.45, 1.85],
                  hspace=0.20, left=0.085, right=0.975, top=0.955,
                  bottom=0.045)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[i]) for i in (0, 1, 3))

    panel_schematic(ax_a)
    panel_characteristics(ax_b, runs)
    panel_zoom(ax_c, runs, runs[1]["x_m"])

    # Leaders tying the magnified box to panel C.
    for xz, xc in ((ZOOM_X[0], 4.20), (ZOOM_X[1], 4.50)):
        fig.add_artist(ConnectionPatch(
            xyA=(xz, ZOOM_U[0]), coordsA=ax_b.transData,
            xyB=(xc, ZOOM_U[1]), coordsB=ax_c.transData,
            lw=0.8, ls=(0, (4, 3)), color=INK))

    os.makedirs("figures", exist_ok=True)
    stem = "figures/free_piston_driver_illustration"
    out_png, out_pdf, out_eps = f"{stem}.png", f"{stem}.pdf", f"{stem}.eps"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    try:
        subprocess.run(["pdftops", "-eps", out_pdf, out_eps], check=True)
    except (OSError, subprocess.CalledProcessError):
        fig.savefig(out_eps)
    for p in (out_png, out_pdf, out_eps):
        print(f"wrote {p}")

    print(f"\nsoft-landing fill (bisected): {p_soft:.1f} Pa "
          f"= {p_soft / 1e3:.3f} kPa")
    print(f"{'case':<9}{'fill kPa':>9}{'u_max':>8}{'x(u_max)':>10}"
          f"{'u_m':>9}{'x_m':>8}{'x_rupt':>8}{'u_wall':>8}")
    for lbl, f, r in zip(("(a) rebound", "(b) soft", "(c) direct"),
                         (1.10, 1.00, 0.90), runs):
        print(f"{lbl:<9}{p_soft * f / 1e3:9.2f}{r['u_max']:8.1f}"
              f"{r['x_u_max']:10.2f}{r['u_m']:9.2f}{r['x_m']:8.3f}"
              f"{r['x_rupt']:8.3f}{r['u_wall']:8.1f}")
    print(f"\nL_m (soft landing standoff) = {X_WALL - runs[1]['x_m']:.4f} m")


if __name__ == "__main__":
    main()

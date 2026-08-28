"""
Publication figure: effect of piston over-driving on driver pressure.

Reproduction of Fig. 6 ("Effect of piston over-driving on driver pressure.
Time is normalised") as a pure-vector schematic.

Construction
------------
The three curves are not drawn by hand: they come from one parametrisation in
which beta carries its physical meaning, so every landmark in the figure is
analytic.

Let u = t - t_rupt be normalised time measured from primary-diaphragm burst.
Working in log-pressure:

    u <= 0 :  ln(p / p_rupt) = k u                 (compression by the piston)
    u >  0 :  ln(p / p_rupt) = k_beta u - a u^2    (compression minus venting)

with

    k_beta = k (1 - 1/beta),      beta = (compression rate) / (venting rate).

Reading of the two terms:

  * k        piston compression rate. Shared by all three curves, because the
             pre-rupture stroke is identical -- same piston, same fill. This is
             why the rise is a single line that only splits at the top.
  * k_beta   net post-rupture slope. Pressure is continuous through burst but
             dp/dt is *not*: the diaphragm opens essentially instantaneously,
             so the venting term switches on as a step. The slope kink at
             rupture is physics, not a drawing artefact.
                 beta > 1  ->  k_beta > 0, p overshoots p_rupt  (over-driven)
                 beta = 1  ->  k_beta = 0, p is stationary at p_rupt (tailored)
                 beta < 1  ->  k_beta < 0, p falls away at once
  * -a u^2   piston deceleration / rebound, which eventually pulls every curve
             back down to the axis.

Landmarks that follow analytically (no fudging):

    peak of an over-driven curve   u* = k_beta / (2a),
                                   p*/p_rupt = exp(k_beta^2 / 4a).

so a is *derived* by inverting the second relation at beta = 1.2: the peak then
lands on p* = 1.10 p_rupt, i.e. exactly the +Delta_p line. The figure's own
"Delta_p / p_rupt ~= 10%" callout is therefore self-consistent with the drawn
curves rather than decorative, and the two cannot drift apart if k is retuned.

"Useful supply time" is then measured, not asserted: the interval over which a
curve stays inside the +/- Delta_p band around p_rupt, solved from the model
and printed at the end (over-driving roughly doubles it -- the point of the
figure).

Usage
-----
    python src/plot_piston_overdrive_illustration.py [--caption]

--caption burns the journal caption into the figure; omit it (the default) if
the caption will be supplied by the LaTeX float.
"""

from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# ---------------------------------------------------------------------------
# Publication style. Monochrome line art, to match the source figure.
# ---------------------------------------------------------------------------
rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
})

INK = "black"
LW_CURVE = 1.0
LW_THIN = 0.7


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
K = 13.0           # piston compression rate (shared pre-rupture rise)
BETAS = (1.2, 1.0, 0.88)
DELTA = 0.10       # Delta_p / p_rupt
P_RUPT = 1.0       # pressures are normalised by p_rupt


def k_beta(beta: float) -> float:
    """Net post-rupture log-slope: compression minus venting."""
    return K * (1.0 - 1.0 / beta)


# Piston deceleration. Derived, not tuned: this is the value for which the
# over-driven peak exp(k_beta^2 / 4a) sits exactly on the +Delta_p line, so the
# drawn curve and the "Delta_p / p_rupt ~= 10%" callout cannot drift apart.
A = k_beta(BETAS[0]) ** 2 / (4.0 * np.log(1.0 + DELTA))


def pressure(t, beta):
    """p(t) / p_rupt for a given over-drive factor, rupture at t = 0."""
    u = np.asarray(t, dtype=float)
    rise = K * u
    post = k_beta(beta) * u - A * u ** 2
    return np.exp(np.where(u <= 0.0, rise, post))


def band_exit(beta: float) -> float:
    """u at which the curve last leaves the -Delta_p level after rupture."""
    kb, target = k_beta(beta), np.log(1.0 - DELTA)
    # a u^2 - k_beta u + ln(1 - DELTA) = 0, take the later (descending) root.
    disc = kb ** 2 - 4.0 * A * target
    return (kb + np.sqrt(disc)) / (2.0 * A)


BAND_ENTRY = np.log(1.0 - DELTA) / K       # crossing -Delta_p on the shared rise
SUPPLY = {b: band_exit(b) - BAND_ENTRY for b in BETAS}


# ---------------------------------------------------------------------------
# Geometry of the schematic (data coordinates; p in units of p_rupt)
# ---------------------------------------------------------------------------
X0, Y0 = -0.62, 0.0        # origin of the drawn axes
X_AXIS_END, Y_AXIS_END = 1.35, 1.38
P_FLOOR = 4e-3             # below this the curves are visually on the axis

X_DASH_END = 0.72          # right end of the p_rupt dashed line
X_DOT_START, X_DOT_END = 0.02, 0.93   # extent of the +/- Delta_p dotted lines
X_DP_ARROW = 0.80          # the +/- Delta_p double arrow
Y_SUPPLY = 1.235           # height of the "useful supply time" span arrow


def arrow(ax, tail, head, style="-|>", lw=LW_THIN, scale=10, **kw):
    """Bare arrow between two data points (no text)."""
    ax.annotate("", xy=head, xytext=tail,
                arrowprops=dict(arrowstyle=style, lw=lw, color=INK,
                                shrinkA=0, shrinkB=0, mutation_scale=scale,
                                **kw))


def leader(ax, text, target, xytext, ha="left", va="center", fontsize=10):
    """Label with a thin leader arrow pointing at a curve."""
    ax.annotate(text, xy=target, xytext=xytext, ha=ha, va=va,
                fontsize=fontsize, color=INK, annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", lw=LW_THIN, color=INK,
                                shrinkA=3, shrinkB=1, mutation_scale=9))


def build(caption: bool = False):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.set_axis_off()

    # --- axes drawn by hand, as arrows ------------------------------------
    arrow(ax, (X0, Y0), (X_AXIS_END, Y0), lw=0.9, scale=12)
    arrow(ax, (X0, Y0), (X0, Y_AXIS_END), lw=0.9, scale=12)
    ax.text(X_AXIS_END + 0.05, Y0 - 0.005, r"$t$", ha="left", va="center")
    ax.text(X0 - 0.05, Y_AXIS_END - 0.06, r"$p$", ha="right", va="center")

    # --- reference levels: p_rupt (dashed) and p_rupt +/- Delta_p (dotted) --
    ax.plot([X0, X_DASH_END], [P_RUPT, P_RUPT], ls=(0, (5, 4)), lw=0.6,
            color=INK, zorder=1)
    ax.text(X0 - 0.05, P_RUPT, r"$p_{\mathrm{rupt}}$", ha="right", va="center")
    for lvl in (P_RUPT + DELTA, P_RUPT - DELTA):
        ax.plot([X_DOT_START, X_DOT_END], [lvl, lvl], ls=(0, (1, 2.2)),
                lw=0.6, color=INK, zorder=1)

    # --- the three pressure histories -------------------------------------
    t = np.linspace(X0, 0.85, 2000)
    for beta in BETAS:
        p = pressure(t, beta)
        m = p >= P_FLOOR
        ax.plot(t[m], p[m], color=INK, lw=LW_CURVE, solid_capstyle="round",
                zorder=3)

    # --- rupture ----------------------------------------------------------
    leader(ax, "rupture", target=(0.0, P_RUPT), xytext=(X0 + 0.03, 1.215))

    # --- useful supply time (for the over-driven curve) --------------------
    u_end = band_exit(BETAS[0])
    for x in (BAND_ENTRY, u_end):
        ax.plot([x, x], [P_RUPT - DELTA, Y_SUPPLY], ls=(0, (1, 2.2)), lw=0.6,
                color=INK, zorder=1)
    arrow(ax, (BAND_ENTRY, Y_SUPPLY), (u_end, Y_SUPPLY), style="<|-|>", scale=9)
    ax.text(0.10, 1.345, "useful supply time", ha="center", va="bottom")
    arrow(ax, (0.055, 1.335), (0.105, Y_SUPPLY + 0.018), style="-|>", scale=9)

    # --- +/- Delta_p about p_rupt, and the 10% callout ---------------------
    ax.plot([X_DP_ARROW - 0.022, X_DP_ARROW + 0.022], [P_RUPT, P_RUPT],
            lw=0.6, color=INK)
    arrow(ax, (X_DP_ARROW, P_RUPT), (X_DP_ARROW, P_RUPT + DELTA), scale=9)
    arrow(ax, (X_DP_ARROW, P_RUPT), (X_DP_ARROW, P_RUPT - DELTA), scale=9)
    ax.text(X_DP_ARROW + 0.035, P_RUPT + 0.055, r"$+\Delta p$",
            ha="left", va="center")
    ax.text(X_DP_ARROW + 0.035, P_RUPT - 0.055, r"$-\Delta p$",
            ha="left", va="center")
    ax.text(1.05, P_RUPT,
            r"$\left(\dfrac{\Delta p}{p_{\mathrm{rupt}}} \approx 10\%\right)$",
            ha="left", va="center", fontsize=11)

    # --- curve labels, each anchored to a solved point on its own limb -----
    #     target heights chosen so the three leaders fan out without crossing.
    labels = [
        (BETAS[0], 0.83, r"$\beta > 1$ (over-driven)", 0.790),
        (BETAS[1], 0.74, r"$\beta = 1$", 0.650),
        (BETAS[2], 0.62, r"$\beta < 1$", 0.515),
    ]
    for beta, p_at, text, y_text in labels:
        kb, target = k_beta(beta), np.log(p_at)
        u = (kb + np.sqrt(kb ** 2 - 4.0 * A * target)) / (2.0 * A)
        leader(ax, text, target=(u, p_at), xytext=(0.50, y_text))

    ax.set_xlim(-0.90, 1.62)
    ax.set_ylim(-0.13, 1.46)

    if caption:
        fig.text(0.5, 0.005,
                 "Fig. 6  Effect of piston over-driving on driver pressure. "
                 "Time is nor-\nmalised", ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=(0, 0.06 if caption else 0, 1, 1))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caption", action="store_true",
                    help="burn the journal caption into the figure")
    args = ap.parse_args()

    fig = build(caption=args.caption)

    os.makedirs("figures", exist_ok=True)
    stem = "figures/piston_overdrive_illustration"
    out_png, out_pdf, out_eps = f"{stem}.png", f"{stem}.pdf", f"{stem}.eps"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    # House pattern: derive the EPS from the PDF. Every artist here is opaque
    # line art, so pdftops keeps the .eps pure vector (no rasterisation).
    try:
        subprocess.run(["pdftops", "-eps", out_pdf, out_eps], check=True)
    except (OSError, subprocess.CalledProcessError):
        fig.savefig(out_eps, bbox_inches="tight")   # fallback
    for path in (out_png, out_pdf, out_eps):
        print(f"wrote {path}")

    print()
    print(f"rupture at p_rupt, band = p_rupt +/- {DELTA:.0%}"
          f"   (k = {K:g}, a = {A:.2f} derived)")
    for beta in BETAS:
        kb = k_beta(beta)
        peak = np.exp(kb ** 2 / (4.0 * A)) if kb > 0 else 1.0
        print(f"  beta = {beta:<5} k_beta = {kb:+6.3f}   "
              f"peak = {peak:.3f} p_rupt   "
              f"useful supply time = {SUPPLY[beta]:.3f}  "
              f"({SUPPLY[beta] / SUPPLY[1.0]:.2f}x the beta = 1 case)")


if __name__ == "__main__":
    main()

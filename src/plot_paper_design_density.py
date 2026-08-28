"""Design-variable occupancy of the non-dominated set, as six 1-D heat strips.

The Pareto figure (``plot_paper_pareto_summary.py``) shows *what* the runs
achieved.  This one shows *where in the design space* those solutions sit:
one horizontal strip per independent variable, spanning that variable's
full optimiser bounds, shaded by how often the non-dominated individuals
landed at each value.

    saturated  =  many non-dominated individuals at this value
    blank      =  the optimiser never settled here

Reading it — a strip that is saturated over a narrow band means every run
converged on the same value for that variable (the objective is sensitive
to it, or it is pinned against a bound).  A strip shaded across most of
its width means the variable is weakly constrained by the objectives: the
runs traded it freely along the front.

Colour — ``--style`` picks one of two bundles (see ``STYLES``):

* ``light`` (default) — white background, single-hue ``Blues`` ramp,
  empty bins masked to white.
* ``dark`` — ``magma`` over a lifted dark-grey floor, matching the
  project's other frequency heatmaps.

Both ramps are monotonic in lightness, which is what makes a sequential
scale readable — the property a rainbow map like jet lacks.  Each
bundle's parts can still be overridden individually with ``--cmap``,
``--floor`` and ``--mask-empty`` / ``--no-mask-empty``.

Each strip carries a **rug** of individual tick marks beneath the heat
band, so the underlying sample positions stay visible rather than being
hidden behind the binning.

Normalisation — each strip is scaled to its **own** maximum bin, so its
internal structure is readable regardless of how concentrated the other
variables are.  The colour bar is therefore "fraction of that variable's
busiest bin", not a count shared across strips.  ``--global-norm`` puts
all six on one absolute scale instead.

Data source and run selection follow the Pareto figure: the external
non-dominated archive of each run, with ``al_cht_0093`` excluded by
default (it ran a different strategy).  Design columns in
``archive.csv`` are stored in the optimiser's normalised [1, 2] box and
are mapped back through ``problem.config.BOUNDS``.

EPS notes — no alpha anywhere; the heat bands are vector ``pcolormesh``
quads with their edges painted to match their faces, which removes the
hairline seams PostScript otherwise leaves between adjacent cells.

Usage
-----
    python3 src/plot_paper_design_density.py
    python3 src/plot_paper_design_density.py --style dark
    python3 src/plot_paper_design_density.py --bins 60 --no-rug
    python3 src/plot_paper_design_density.py --global-norm
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem.config import BOUNDS, MIN_BOUND  # noqa: E402
from plot_paper_pareto_summary import (  # noqa: E402
    DEFAULT_EXCLUDE, RESULTS_DIR, FIGURES_DIR, non_dominated,
)

# Note: importing plot_paper_pareto_summary above runs *its* module-level
# rcParams.update, which turns on mirrored top/right ticks for that figure.
# Every key that differs here must therefore be set explicitly rather than
# left to the matplotlib default — otherwise it inherits the other
# figure's style (this is why xtick.top/ytick.right appear below).
rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.direction": "out",
    "xtick.top": False,
    "ytick.right": False,
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
})

# Display scaling for each design variable, in BOUNDS order.  The stored
# values are SI (Pa, m); these factors put each axis on a scale a reader
# can hold in their head, without changing the underlying data.
#   (label, unit, factor applied to the SI value)
VARIABLES = [
    ("Helium fraction, $X_\\mathrm{He}$",          "%",   1.0),
    ("Driver fill pressure, $p_\\mathrm{driver}$", "kPa", 1e-3),
    ("Burst pressure, $p_4$",                      "MPa", 1e-6),
    ("Throat diameter, $D_\\mathrm{throat}$",      "mm",  1e3),
    ("Reservoir pressure, $p_\\mathrm{res}$",      "MPa", 1e-6),
    ("Buffer length, $L_\\mathrm{buffer}$",        "mm",  1e3),
]

# The two looks, as coherent bundles.  These three settings interact — a
# grey floor under a light ramp gives muddy grey-blue empty bins, and
# masking empty bins white on a dark ramp punches holes in the strip — so
# they are chosen together rather than left as three independent flags.
#   style -> (cmap, mask empty bins white, grey floor under the dark end)
STYLES = {
    # light: single-hue ramp, empty bins white.  Zero has to be masked
    # because Blues bottoms out at a pale blue that reads as sparse
    # occupancy rather than none.
    "light": ("Blues", True, 0.0),
    # dark: matches the project's other frequency heatmaps
    # (arnold_diagnostics / resample_diagnostics use magma, vmin=0).
    "dark": ("magma", False, 0.18),
}


def lift_floor(cmap, floor: float):
    """Raise a colormap's dark end to a grey floor, keeping it monotonic.

    Applies the affine lift ``rgb' = floor + rgb * (1 - floor)`` to every
    entry in the lookup table.  Pure black maps to the grey ``floor``,
    fully-bright channels stay where they are, and everything between is
    moved by a shrinking amount — so lightness still increases all the way
    up the ramp.

    The alternative (recolouring only the zero-count bins) would break
    that: a mid-grey is *lighter* than magma's near-black low steps, so
    empty bins would read as busier than bins holding one or two
    individuals.
    """
    lut = cmap(np.linspace(0.0, 1.0, cmap.N))
    lut[:, :3] = floor + lut[:, :3] * (1.0 - floor)
    return ListedColormap(lut, name=f"{cmap.name}_floor{floor:g}")


def load_designs(run_dir: Path) -> np.ndarray | None:
    """Return the run's non-dominated designs as an (N, 6) array in SI units.

    Reads ``summary/archive.csv``: ``f_0``/``f_1`` are used only to
    re-verify non-dominance, ``design_0..5`` carry the design vector in the
    optimiser's normalised [1, 2] box.
    """
    path = run_dir / "summary" / "archive.csv"
    if not path.exists():
        return None

    f_rows, x_rows = [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                f_rows.append((float(row["f_0"]), float(row["f_1"])))
                x_rows.append([float(row[f"design_{i}"]) for i in range(len(BOUNDS))])
            except (KeyError, TypeError, ValueError):
                continue
    if not x_rows:
        return None

    f = np.asarray(f_rows, dtype=float)
    x = np.asarray(x_rows, dtype=float)
    x = x[non_dominated(f)]

    # normalised [1, 2] box -> physical SI, per problem/config.BOUNDS
    lo = np.array([b[0] for b in BOUNDS], dtype=float)
    hi = np.array([b[1] for b in BOUNDS], dtype=float)
    return lo + (x - MIN_BOUND) * (hi - lo)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out", type=Path,
                    default=FIGURES_DIR / "paper_design_variable_density.eps")
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE,
                    help=f"run directory names to omit "
                         f"(default: {' '.join(DEFAULT_EXCLUDE)})")
    ap.add_argument("--bins", type=int, default=100,
                    help="histogram bins across each variable's full bounds")
    ap.add_argument("--style", choices=tuple(STYLES), default="light",
                    help="'light' = white background, single-hue ramp "
                         "(default); 'dark' = magma on dark grey, matching "
                         "the project's other frequency heatmaps")
    ap.add_argument("--cmap", default=None,
                    help="override the style's colormap")
    ap.add_argument("--floor", type=float, default=None,
                    help="override the grey level (0-1) the colormap's dark "
                         "end is lifted to; 0 gives pure black")
    ap.add_argument("--mask-empty", dest="mask_empty", default=None,
                    action="store_true",
                    help="force zero-count bins to render white")
    ap.add_argument("--no-mask-empty", dest="mask_empty",
                    action="store_false",
                    help="force zero-count bins onto the colormap's bottom step")
    ap.add_argument("--global-norm", action="store_true",
                    help="one absolute colour scale across all six strips")
    ap.add_argument("--no-rug", action="store_true",
                    help="omit the per-individual tick marks")
    ap.add_argument("--also-png", action="store_true")
    args = ap.parse_args()

    # Resolve the style bundle, letting any explicit flag win over it.
    style_cmap, style_mask, style_floor = STYLES[args.style]
    cmap_name = args.cmap if args.cmap is not None else style_cmap
    mask_empty = args.mask_empty if args.mask_empty is not None else style_mask
    floor = args.floor if args.floor is not None else style_floor

    runs = sorted(d for d in args.results_dir.iterdir()
                  if d.is_dir() and d.name not in args.exclude)
    if not runs:
        print(f"No run directories found under {args.results_dir}", file=sys.stderr)
        return 1

    blocks, used = [], []
    for run_dir in runs:
        x = load_designs(run_dir)
        if x is None or len(x) == 0:
            print(f"  {run_dir.name}: no archive designs — skipped", file=sys.stderr)
            continue
        blocks.append(x)
        used.append((run_dir.name, len(x)))
    if not blocks:
        print("No design data loaded.", file=sys.stderr)
        return 1

    # Pool every run's non-dominated designs: the figure is about how often
    # a value is selected across the whole experiment set, not per run.
    X = np.vstack(blocks)

    nvar = len(BOUNDS)
    counts, edges = [], []
    for j in range(nvar):
        lo, hi = BOUNDS[j]
        c, e = np.histogram(X[:, j], bins=args.bins, range=(lo, hi))
        counts.append(c.astype(float))
        edges.append(e)

    if args.global_norm:
        vmax = max(c.max() for c in counts) or 1.0
        norm = [c / vmax for c in counts]
        cbar_label = "bin count / global busiest bin"
    else:
        norm = [c / (c.max() or 1.0) for c in counts]
        cbar_label = "bin count / that variable's busiest bin"

    # Empty bins are left to fall on the colormap's own bottom step rather
    # than being masked out: on a dark-background ramp like magma that step
    # is near-black, so "never visited" already reads as unlit.  (With a
    # light ramp the equivalent trick is masking them to white — see
    # --mask-empty.)
    cmap = plt.get_cmap(cmap_name).copy()
    if floor > 0.0:
        cmap = lift_floor(cmap, floor)
    if mask_empty:
        norm = [np.ma.masked_where(c == 0, n) for n, c in zip(norm, counts)]
        cmap.set_bad("white")

    fig, axes = plt.subplots(nvar, 1, figsize=(6.6, 5.9))
    mesh = None
    for j, ax in enumerate(axes):
        label, unit, factor = VARIABLES[j]
        lo, hi = BOUNDS[j]
        xe = edges[j] * factor

        # The heat band occupies the upper part of the axes; the rug of
        # individual values sits underneath it, inside the same frame.
        ytop, ybot = 1.0, (0.30 if not args.no_rug else 0.0)
        mesh = ax.pcolormesh(xe, np.array([ybot, ytop]), norm[j][None, :],
                             cmap=cmap, vmin=0.0, vmax=1.0,
                             shading="flat", linewidth=0.0)
        # PostScript leaves hairline gaps between adjacent quads unless the
        # cell edges are painted in the cell's own face colour.
        mesh.set_edgecolor("face")

        if not args.no_rug:
            ax.vlines(X[:, j] * factor, 0.03, 0.24, color="#1a1a1a",
                      lw=0.25)
            ax.axhline(ybot, color="0.55", lw=0.5)

        ax.set_xlim(lo * factor, hi * factor)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_ylabel(f"{label}\n[{unit}]", rotation=0, ha="right",
                      va="center", labelpad=8)
        ax.tick_params(axis="x", length=3, pad=2)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

        # The bounds themselves are part of the message — a variable pinned
        # against a bound reads very differently from one that settled in
        # the interior — so both endpoints are always labelled.  Auto ticks
        # landing near an endpoint are dropped first, otherwise the two
        # labels overprint each other.
        span = (hi - lo) * factor
        auto = [t for t in ax.get_xticks()
                if lo * factor < t < hi * factor
                and min(abs(t - lo * factor), abs(t - hi * factor)) > 0.07 * span]
        ax.set_xticks([lo * factor] + auto + [hi * factor])
        # Bounds are rarely round numbers once scaled (driver_p's upper
        # bound is 40/14.62 MPa); 4 significant figures keeps them honest
        # without printing a full float.
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4g}"))
        ax.set_xlim(lo * factor, hi * factor)

    fig.tight_layout(rect=(0, 0.05, 0.90, 0.97))
    cax = fig.add_axes((0.92, 0.18, 0.018, 0.64))
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.6)

    fig.suptitle(f"Design-variable occupancy of the non-dominated set "
                 f"($N$ = {len(X)} individuals, {len(used)} runs)",
                 fontsize=9.5, y=0.995)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="eps")
    if args.also_png:
        fig.savefig(args.out.with_suffix(".png"), dpi=400)
    plt.close(fig)

    print(f"\nWrote {args.out}")
    if args.exclude:
        print(f"Excluded: {', '.join(args.exclude)}")
    print(f"Pooled {len(X)} non-dominated individuals from {len(used)} runs: "
          + ", ".join(f"{n}({k})" for n, k in used))

    # Numbers behind the picture: how tightly each variable converged, as a
    # fraction of the range the optimiser was allowed to search.
    print(f"\n{'variable':<26}{'unit':>5}{'median':>12}{'p5':>12}{'p95':>12}"
          f"{'p5-p95 / bounds':>18}")
    for j in range(nvar):
        label, unit, factor = VARIABLES[j]
        lo, hi = BOUNDS[j]
        v = X[:, j] * factor
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        frac = (p95 - p5) / ((hi - lo) * factor)
        plain = label.split(",")[0]
        print(f"{plain:<26}{unit:>5}{p50:>12.4g}{p5:>12.4g}{p95:>12.4g}"
              f"{frac:>17.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

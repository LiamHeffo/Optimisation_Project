"""Cross-run summary figure: non-dominated fronts from every paper_copy run.

Reads ``Results/paper_copy/<run>/summary/archive.csv`` — the external
non-dominated archive each run persists at completion — and scatters all
runs onto one pair of physical axes:

    x = driver hold time      [ms]   (maximised — right is better)
    y = piston impact speed   [m/s]  (minimised — down is better)

so the desirable corner is **bottom-right** and the attainable frontier is
the lower-right envelope of each run's points.

Why the archive and not ``parents_gen_0500.csv``:  the archive is "every
point that was ever Pareto-optimal" and is a superset of the final parent
front (see al_plots.plot_archive_pareto).  It is the run's best-known
approximation set, which is what a cross-run comparison should show.  Pass
``--source parents`` to plot the final parent front instead.

Units — the archive stores *normalised* objectives ``f_0``/``f_1`` in
[0, 1] with 0 = best.  They are inverted back to physical units with the
same ideal/nadir reference points the optimiser used
(``problem.config.APPROX_IDEAL_2D`` / ``APPROX_NADIR_2D``), so the axes
carry no hard-coded constants:

    hold_time    = f_0 * (nadir_0 - ideal_0) + ideal_0     (ideal 5 ms,  nadir 0)
    impact_speed = f_1 * (nadir_1 - ideal_1) + ideal_1     (ideal 0 m/s, nadir 350)

Run selection — ``al_cht_0093`` is **excluded by default**: it ran
``Simulation Type = Resampling_AL`` while the other ten ran
``ArnoldCHT_AL``, so including it would mix two strategies.  The surviving
runs are labelled ``Run 1``…``Run 10`` in directory order; the summary
table printed to stdout keeps the mapping back to the ``al_cht_NNNN``
directories.  ``PALETTE`` holds exactly 10 verified slots, so putting
0093 back needs an 11th colour added to it — the script refuses to cycle
colours rather than emit a figure with two runs in the same hue.

EPS notes — the PostScript backend flattens transparency to opaque, so
this module uses **no alpha anywhere**: series are separated by hue *and*
marker shape, and the grid is a light solid grey rather than an alpha
grid.  ``ps.fonttype = 42`` embeds TrueType outlines so the text stays
selectable and does not fall back to bitmap Type-3 glyphs in the final
PDF.

Usage
-----
    python3 src/plot_paper_pareto_summary.py
    python3 src/plot_paper_pareto_summary.py --connect
    python3 src/plot_paper_pareto_summary.py --single-panel
    python3 src/plot_paper_pareto_summary.py --source parents

Note: do **not** invoke with ``PYTHONPATH=src`` — see the project's L1d
launch gotcha.  This script puts ``src`` on ``sys.path`` itself.
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
from matplotlib.ticker import MultipleLocator

# Import the optimiser's own reference points rather than restating them,
# so the axes cannot silently drift from the objective definition.  Done
# via sys.path (not the PYTHONPATH env var, which clobbers gdtk for any
# subprocess this shell later spawns).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem.config import APPROX_IDEAL_2D, APPROX_NADIR_2D  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "Results" / "paper_copy"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ── style ────────────────────────────────────────────────────────────────────
# Matches the serif/10pt convention used by the other paper figures
# (plot_arnold_cht_illustration.py et al.) so the plates look like a set.
rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
})

# 10 categorical hues found by a max-min search over the OKLCH in-band,
# in-chroma sRGB pool (capped at L <= 0.68 so no slot is too pale to read
# as a small marker on white), scored on the data-viz validator's own ΔE
# maths.  Verified all-pairs — the strict gate, since a scatter can put
# any two series side by side, unlike bars/lines which only need adjacent
# pairs to separate:  worst colour-blind ΔE 9.1 (target >= 8), worst
# normal-vision ΔE 17.7 (floor >= 15).
# Paired 1:1 with distinct marker shapes so identity never rests on hue
# alone (needed for greyscale printing and photocopies regardless of ΔE).
PALETTE = [
    "#C53DCC", "#50B200", "#0000FF", "#CC5C67", "#7400B2",
    "#1BABB2", "#99004C", "#7373FF", "#806000", "#176B99",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

# al_cht_0093 ran Simulation Type = Resampling_AL; the rest ran
# ArnoldCHT_AL.  It is excluded by default so the figure shows one
# strategy.  Pass --exclude '' to put it back.
DEFAULT_EXCLUDE = ["al_cht_0093"]

# Upper bound of the detail panel: the runs' fronts have a genuine gap
# between roughly 7 and 18 m/s, and ~31% of all archived points sit below
# 5 m/s (the soft-landing branch, where the gas cushion arrests the piston
# before it reaches the buffer).  On a single 0-62 m/s axis that whole
# branch collapses onto the abscissa, so it gets its own panel.
DETAIL_YMAX = 8.0


def load_front(run_dir: Path, source: str) -> np.ndarray:
    """Return the run's non-dominated set as an (N, 2) array of *normalised* f.

    ``source='archive'`` reads summary/archive.csv (columns f_0, f_1);
    ``source='parents'`` reads summary/parents_gen_*.csv (columns
    scaled_hold_time, scaled_impact_speed).
    """
    if source == "archive":
        path = run_dir / "summary" / "archive.csv"
        cols = ("f_0", "f_1")
    else:
        # Highest generation number = the final parent front.  These live
        # under parents/, not summary/ (summary/ holds only the archive and
        # the diversity/objective text reports).
        candidates = sorted((run_dir / "parents").glob("parents_gen_*.csv"))
        if not candidates:
            return np.empty((0, 2))
        path = candidates[-1]
        cols = ("scaled_hold_time", "scaled_impact_speed")

    if not path.exists():
        return np.empty((0, 2))

    pts = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                pts.append((float(row[cols[0]]), float(row[cols[1]])))
            except (KeyError, TypeError, ValueError):
                # Blank scaled_* cells occur for sentinel rows; skip them.
                continue
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def non_dominated(f: np.ndarray) -> np.ndarray:
    """Keep only mutually non-dominated rows.  Both columns are minimised.

    Applied in *normalised* space, where the minimise-both convention is
    unambiguous.  For these runs it is a no-op (the persisted archives are
    already fronts) — it is a guard against a partially-written archive,
    and the caller reports whenever it actually removes anything.
    """
    if len(f) == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(len(f), dtype=bool)
    for i, p in enumerate(f):
        # dominated iff some q is <= p in both and strictly < in at least one
        if np.any(np.all(f <= p, axis=1) & np.any(f < p, axis=1)):
            keep[i] = False
    return keep


def to_physical(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert the optimiser's normalisation → (hold_time [ms], impact [m/s])."""
    ideal = np.asarray(APPROX_IDEAL_2D, dtype=float)
    nadir = np.asarray(APPROX_NADIR_2D, dtype=float)
    phys = f * (nadir - ideal) + ideal
    return phys[:, 0] * 1e3, phys[:, 1]


def sim_type(run_dir: Path) -> str:
    """Read 'Simulation Type' out of the run's convergence header, or ''."""
    path = run_dir / "convergence" / "convergence_data.txt"
    if not path.exists():
        return ""
    with path.open() as fh:
        for line in fh:
            if line.startswith("Simulation Type"):
                return line.split("=", 1)[1].strip()
            if line.startswith("Hypervolume"):
                break
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out", type=Path,
                    default=FIGURES_DIR / "paper_pareto_fronts_all_runs.eps")
    ap.add_argument("--source", choices=("archive", "parents"), default="archive",
                    help="archive = external non-dominated archive (default); "
                         "parents = final parent population front")
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE,
                    help=f"run directory names to omit "
                         f"(default: {' '.join(DEFAULT_EXCLUDE)})")
    ap.add_argument("--connect", action="store_true",
                    help="join each run's front with a polyline "
                         "(default: markers only)")
    ap.add_argument("--single-panel", action="store_true",
                    help="omit the low-impact detail panel (full range only)")
    ap.add_argument("--also-png", action="store_true",
                    help="additionally write a 400 dpi PNG for quick viewing")
    args = ap.parse_args()

    runs = sorted(d for d in args.results_dir.iterdir()
                  if d.is_dir() and d.name not in args.exclude)
    if not runs:
        print(f"No run directories found under {args.results_dir}", file=sys.stderr)
        return 1
    if len(runs) > len(PALETTE):
        print(f"{len(runs)} runs exceeds the {len(PALETTE)}-slot palette; "
              "colours would have to be cycled. Reduce or facet.", file=sys.stderr)
        return 1

    npanel = 1 if args.single_panel else 2
    fig, axes = plt.subplots(1, npanel, figsize=(7.2, 3.9) if npanel == 2
                             else (5.0, 4.0), dpi=200)
    axes = np.atleast_1d(axes)
    for ax in axes:
        # Grid under the data. Solid light grey, not alpha — EPS flattens
        # transparency to opaque and would render an alpha grid as heavy black.
        ax.grid(color="0.88", lw=0.5, zorder=0)
        ax.set_axisbelow(True)

    summary, handles = [], []
    for i, run_dir in enumerate(runs):
        f = load_front(run_dir, args.source)
        if len(f) == 0:
            print(f"  {run_dir.name}: no {args.source} data — skipped", file=sys.stderr)
            continue
        keep = non_dominated(f)
        if (~keep).any():
            print(f"  {run_dir.name}: dropped {(~keep).sum()} dominated row(s)",
                  file=sys.stderr)
        f = f[keep]

        hold_ms, impact = to_physical(f)
        order = np.argsort(hold_ms)
        hold_ms, impact = hold_ms[order], impact[order]

        colour, marker = PALETTE[i], MARKERS[i]
        # Sequential 1..N labels rather than the internal al_cht_NNNN ids,
        # which carry no meaning for a reader.  The summary table printed
        # below keeps the mapping back to the source directories.
        label = f"Run {i + 1}"

        for k, ax in enumerate(axes):
            # The detail panel is clipped in y.  Subsetting the data (rather
            # than relying on axis limits) keeps the connector from being
            # drawn to an off-panel point, which would otherwise render as a
            # spurious full-height vertical line.  The front is monotone in
            # hold time, so the in-window subset is a contiguous prefix.
            if npanel == 2 and k == 1:
                sel = impact <= DETAIL_YMAX
                hx, ix = hold_ms[sel], impact[sel]
            else:
                hx, ix = hold_ms, impact
            if len(hx) == 0:
                continue
            if args.connect and len(hx) > 1:
                ax.plot(hx, ix, "-", color=colour, lw=0.6, zorder=2)
            line, = ax.plot(hx, ix, linestyle="none", marker=marker,
                            markersize=3.4, markerfacecolor=colour,
                            markeredgecolor="0.15", markeredgewidth=0.3,
                            zorder=3)
        line.set_label(f"{label} ($n$={len(hold_ms)})")
        handles.append(line)

        summary.append((label, run_dir.name, sim_type(run_dir), len(hold_ms),
                        hold_ms.max(), impact.min()))

    # Direction hints live in the axis labels rather than as a floating
    # "preferred" arrow: the arrow has to be placed in whitespace, and the
    # only whitespace here is the region the arrow needs to point into.
    xlabel = ("Driver hold time, $t_\\mathrm{hold}$  [ms]\n"
              "(maximised $\\rightarrow$)")
    ylabel = ("Piston impact speed, $v_\\mathrm{impact}$  [m s$^{-1}$]\n"
              "($\\leftarrow$ minimised)")
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        # Hold times span ~3 ms; the auto-locator drops to whole-ms ticks
        # once tight_layout narrows the panel, which is too coarse to read
        # a front against.
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    axes[0].set_ylabel(ylabel)

    if npanel == 2:
        axes[1].set_ylim(0, DETAIL_YMAX)
        axes[1].set_ylabel(ylabel)
        axes[0].set_title("(a) full objective range", pad=6)
        axes[1].set_title(f"(b) detail, $v_\\mathrm{{impact}} \\leq$ "
                          f"{DETAIL_YMAX:g} m s$^{{-1}}$", pad=6)
        # Mark the detail window on the overview so the two panels are
        # visibly the same data at two scales.
        axes[0].axhspan(0, DETAIL_YMAX, color="0.94", zorder=1, lw=0)
        axes[0].axhline(DETAIL_YMAX, color="0.55", lw=0.6, ls=(0, (4, 2)),
                        zorder=4)
        # Give the detail panel its own x-range — the soft-landing branch
        # stops well short of the longest holds, and stretching it to the
        # overview's limit would leave a third of the panel empty.
        xmax = max((ln.get_xdata().max() for ln in axes[1].lines
                    if len(ln.get_xdata())), default=None)
        if xmax is not None:
            axes[1].set_xlim(0, xmax * 1.06)

    # Legend below the panels: with 11 series an in-axes box either covers
    # data or crowds it, and a figure-level strip keeps both panels clean.
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    leg = fig.legend(handles=handles, loc="lower center", ncol=5,
                     frameon=True, framealpha=1.0, edgecolor="0.7",
                     fancybox=False, handletextpad=0.4, columnspacing=1.3,
                     labelspacing=0.3, borderpad=0.5,
                     bbox_to_anchor=(0.5, 0.005))
    leg.get_frame().set_linewidth(0.6)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="eps")
    if args.also_png:
        fig.savefig(args.out.with_suffix(".png"), dpi=400)
    plt.close(fig)

    print(f"\nWrote {args.out}")
    if args.exclude:
        print(f"Excluded: {', '.join(args.exclude)}")
    # The legend says "Run 1..N"; this table is the only record of which
    # source directory each label refers to, so keep them together.
    print(f"\n{'label':<8}{'directory':<14}{'strategy':<15}{'n_nd':>6}"
          f"{'max hold [ms]':>15}{'min impact [m/s]':>18}")
    for label, name, st, n, hmax, imin in summary:
        print(f"{label:<8}{name:<14}{st:<15}{n:>6}{hmax:>15.3f}{imin:>18.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

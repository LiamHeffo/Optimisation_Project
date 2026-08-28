"""Run-to-run distribution of windowed hypervolume across the 10 trials.

Each trial's non-dominated archive is restricted to individuals whose
piston impact speed falls in a window (default 10-40 m/s), and the 2-D
hypervolume of that sub-front is computed.  The ten resulting scalars are
shown twice:

* panel (a) — every run as its own labelled point, sorted by value, with
  the mean and +/-1 sd overlaid.  Nothing is hidden by binning and each
  run stays traceable back to its directory.
* panel (b) — the empirical CDF, which assumes no binning at all and
  reads as "fraction of runs reaching at least this hypervolume".

Run colours and "Run N" labels match ``plot_paper_pareto_summary.py``, so
a run can be followed across the figures.

Caveats this figure is built to make visible rather than hide
--------------------------------------------------------------
The window is a deliberate choice, and it does not contain every run's
solutions.  In the shipped data **run 0098 (Run 3) has no archived
individual between 10 and 40 m/s at all** — its whole front sits below
10 m/s — so its hypervolume is a structural zero, not a bad result.  Two
further runs rest on only 3-4 points.  Both conditions are annotated on
the figure and listed in the stdout table; ``--min-points`` sets the
threshold at which a run is marked.

That matters for interpretation: the spread across the ten values is
driven substantially by whether a run happened to place points inside
the window, not only by how good those points were.  A hypervolume
computed over each run's full front (no window) is defined for all ten
and varies far less; ``--window 0 inf`` produces it for comparison.

Reference point — the window's worst corner: hold time 0 (normalised
f_0 = 1) and the window's upper impact-speed edge.  Both objectives are
minimised in normalised space, so hypervolume is the area dominated by
the sub-front relative to that corner, and larger is better.

Usage
-----
    python3 src/plot_paper_hv_distribution.py
    python3 src/plot_paper_hv_distribution.py --window 0 10
    python3 src/plot_paper_hv_distribution.py --window 0 inf
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem.config import APPROX_IDEAL_2D, APPROX_NADIR_2D  # noqa: E402
from plot_paper_pareto_summary import (  # noqa: E402
    DEFAULT_EXCLUDE, RESULTS_DIR, FIGURES_DIR, PALETTE, MARKERS,
    non_dominated,
)

# See the note in plot_paper_design_density.py: importing the Pareto module
# applies its rcParams globally, so any key that should differ is set here.
rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
})


def hypervolume_2d(F: np.ndarray, ref: tuple[float, float]) -> float:
    """Exact 2-D hypervolume of a minimisation front against ``ref``.

    Points not strictly dominating the reference contribute nothing and are
    dropped first.  The remainder are swept in order of increasing f_0,
    accumulating the rectangle each one adds beyond the running best f_1 —
    O(n log n) and exact, so no external HV library is needed for d = 2.
    """
    if len(F) == 0:
        return 0.0
    F = F[(F[:, 0] < ref[0]) & (F[:, 1] < ref[1])]
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 0])]
    prev_f1, total = ref[1], 0.0
    for f0, f1 in F:
        if f1 < prev_f1:
            total += (ref[0] - f0) * (prev_f1 - f1)
            prev_f1 = f1
    return total


def load_archive(run_dir: Path) -> np.ndarray:
    """Return the run's non-dominated archive as (N, 2) normalised objectives."""
    path = run_dir / "summary" / "archive.csv"
    if not path.exists():
        return np.empty((0, 2))
    pts = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                pts.append((float(row["f_0"]), float(row["f_1"])))
            except (KeyError, TypeError, ValueError):
                continue
    F = np.asarray(pts, dtype=float).reshape(-1, 2)
    return F[non_dominated(F)] if len(F) else F


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out", type=Path,
                    default=FIGURES_DIR / "paper_hv_distribution.eps")
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE)
    ap.add_argument("--window", nargs=2, type=float, default=(10.0, 40.0),
                    metavar=("LO", "HI"),
                    help="impact-speed window in m/s (default 10 40); "
                         "use 0 inf for the unwindowed front")
    ap.add_argument("--min-points", type=int, default=5,
                    help="runs with fewer points than this inside the window "
                         "are flagged on the figure (default 5)")
    ap.add_argument("--also-png", action="store_true")
    args = ap.parse_args()

    lo_v, hi_v = float(args.window[0]), float(args.window[1])
    ideal = np.asarray(APPROX_IDEAL_2D, dtype=float)
    nadir = np.asarray(APPROX_NADIR_2D, dtype=float)

    # Window edges expressed in the normalised objective the archive stores.
    # f_1 = (v - ideal_1) / (nadir_1 - ideal_1), monotonically increasing in v,
    # so the window maps across directly.
    def v_to_f1(v):
        return (v - ideal[1]) / (nadir[1] - ideal[1])

    ref = (1.0, min(v_to_f1(hi_v), 1.0))

    runs = sorted(d for d in args.results_dir.iterdir()
                  if d.is_dir() and d.name not in args.exclude)
    if not runs:
        print(f"No run directories under {args.results_dir}", file=sys.stderr)
        return 1

    rows = []
    for i, run_dir in enumerate(runs):
        F = load_archive(run_dir)
        if len(F) == 0:
            print(f"  {run_dir.name}: empty archive — skipped", file=sys.stderr)
            continue
        v = F[:, 1] * (nadir[1] - ideal[1]) + ideal[1]     # back to m/s
        sel = (v >= lo_v) & (v <= hi_v)
        hv = hypervolume_2d(F[sel], ref)
        rows.append({
            "label": f"Run {i + 1}", "dir": run_dir.name,
            "colour": PALETTE[i], "marker": MARKERS[i],
            "n_total": len(F), "n_win": int(sel.sum()), "hv": hv,
        })

    hv = np.array([r["hv"] for r in rows], dtype=float)
    mean, sd = hv.mean(), hv.std(ddof=1)

    fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(7.2, 3.6))

    # ── panel (a): one labelled point per run, sorted ────────────────────
    order = np.argsort(hv)
    ax_s.axvspan(mean - sd, mean + sd, color="0.91", zorder=0, lw=0)
    ax_s.axvline(mean, color="0.45", lw=0.9, ls=(0, (4, 2)), zorder=1)
    for y, k in enumerate(order):
        r = rows[k]
        ax_s.plot(r["hv"], y, linestyle="none", marker=r["marker"],
                  markersize=5.5, markerfacecolor=r["colour"],
                  markeredgecolor="0.15", markeredgewidth=0.4, zorder=3)
    ax_s.set_yticks(range(len(rows)))
    ax_s.set_yticklabels([rows[k]["label"] for k in order])
    ax_s.set_ylim(-0.7, len(rows) - 0.3)
    ax_s.grid(axis="x", color="0.88", lw=0.5, zorder=0)
    ax_s.set_axisbelow(True)

    # Flag the runs the window barely covers — without this the reader
    # cannot tell a genuinely poor run from one the window missed.
    for y, k in enumerate(order):
        r = rows[k]
        if r["n_win"] == 0:
            ax_s.annotate("no individuals in window",
                          xy=(r["hv"], y), xytext=(6, 0),
                          textcoords="offset points", fontsize=6.5,
                          color="#b3200f", va="center")
        elif r["n_win"] < args.min_points:
            ax_s.annotate(f"$n$={r['n_win']}", xy=(r["hv"], y), xytext=(6, 0),
                          textcoords="offset points", fontsize=6.5,
                          color="0.35", va="center")

    ax_s.set_xlabel("Hypervolume  (normalised objective space)")
    ax_s.set_title(f"(a) per-run value, mean $\\pm$ 1 s.d.", pad=6)

    # ── panel (b): empirical CDF ─────────────────────────────────────────
    xs = np.sort(hv)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax_e.step(np.concatenate([[xs[0]], xs]), np.concatenate([[0.0], ys]),
              where="post", color="#1f3d7a", lw=1.2, zorder=3)
    ax_e.plot(xs, ys, linestyle="none", marker="o", markersize=3.6,
              markerfacecolor="#1f3d7a", markeredgecolor="white",
              markeredgewidth=0.4, zorder=4)
    ax_e.axvline(mean, color="0.45", lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax_e.set_ylim(0, 1.02)
    ax_e.set_xlabel("Hypervolume  (normalised objective space)")
    ax_e.set_ylabel("Fraction of runs $\\leq$ value")
    ax_e.set_title(f"(b) empirical CDF ($n$ = {len(rows)} runs)", pad=6)
    ax_e.grid(color="0.88", lw=0.5, zorder=0)
    ax_e.set_axisbelow(True)

    win_txt = (f"impact speed window {lo_v:g}-{hi_v:g} m s$^{{-1}}$"
               if np.isfinite(hi_v) else f"impact speed $\\geq$ {lo_v:g} m s$^{{-1}}$")
    fig.suptitle(f"Run-to-run hypervolume distribution  ({win_txt})",
                 fontsize=9.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="eps")
    if args.also_png:
        fig.savefig(args.out.with_suffix(".png"), dpi=400)
    plt.close(fig)

    # ── stdout report ────────────────────────────────────────────────────
    print(f"\nWrote {args.out}")
    print(f"Window: {lo_v:g}-{hi_v:g} m/s   reference point (f0, f1) = "
          f"({ref[0]:.4f}, {ref[1]:.4f})")
    if args.exclude:
        print(f"Excluded: {', '.join(args.exclude)}")
    print(f"\n{'label':<8}{'directory':<14}{'n_nd':>6}{'n_window':>10}"
          f"{'hypervolume':>14}")
    for r in rows:
        flag = "  <-- window empty" if r["n_win"] == 0 else (
            "  <-- few points" if r["n_win"] < args.min_points else "")
        print(f"{r['label']:<8}{r['dir']:<14}{r['n_total']:>6}"
              f"{r['n_win']:>10}{r['hv']:>14.5f}{flag}")

    q1, med, q3 = np.percentile(hv, [25, 50, 75])
    print(f"\nmean   = {mean:.5f}      sd  = {sd:.5f}"
          f"      CV = {sd / mean:.1%}" if mean else "")
    print(f"median = {med:.5f}      IQR = {q1:.5f} - {q3:.5f}")
    print(f"min    = {hv.min():.5f}      max = {hv.max():.5f}")

    # Normality: reported because the motivating question was whether the
    # spread looks normal, but n = 10 gives the test very little power --
    # it will fail to reject almost any unimodal shape.  Treat a "pass" as
    # "not obviously non-normal", never as evidence of normality.
    try:
        from scipy import stats as _st
        w, p = _st.shapiro(hv)
        print(f"\nShapiro-Wilk: W = {w:.4f}, p = {p:.3f}  "
              f"(n = {len(hv)}; very low power -- indicative only)")
    except Exception as exc:                       # scipy optional
        print(f"\nShapiro-Wilk unavailable ({exc})")

    n_empty = sum(1 for r in rows if r["n_win"] == 0)
    n_few = sum(1 for r in rows if 0 < r["n_win"] < args.min_points)
    if n_empty or n_few:
        print(f"\nCaution: {n_empty} run(s) have no individuals in the window "
              f"(hypervolume is a structural zero, not a poor result) and "
              f"{n_few} rest on fewer than {args.min_points} points. "
              f"The spread partly reflects window coverage, not run quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

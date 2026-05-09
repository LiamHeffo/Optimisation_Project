"""
Per-generation diagnostics for the Chocat 2015 constraint-handling technique.

The strategy fills ``strategy.cht_diag_buffer`` with one record per
(parent, phase) CHT invocation.  This module:

* drains the buffer at end-of-generation, tagging each record with the
  generation number,
* appends them to two on-disk CSVs (``cht_per_call.csv`` and
  ``cht_per_gen.csv``),
* every ``plot_interval`` generations, regenerates a six-panel summary
  PNG that visualises whether the CHT is doing what it should (volume
  preserved, anisotropy growing, infeasibility falling, eigenvector
  alignment with the boundary).

CSV schemas are stable so downstream analysis can rely on them.
"""

from __future__ import annotations

import csv
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# CSV schemas
# ─────────────────────────────────────────────────────────────────────────────

PER_CALL_FIELDS = [
    "generation", "phase", "parent_idx",
    "n_violators", "shrink_applied", "psd_fallback",
    "log_det_C_before", "log_det_C_after",
    "condition_number_before", "condition_number_after",
    "min_eigenvalue_after", "max_eigenvalue_after",
    "effective_pool_weight",
    "violation_axis_angle_deg",
    # JSON-encoded list-valued columns: stored as strings so downstream
    # tooling (pandas) can json.loads() them on read.
    "eigenvalues_before", "eigenvalues_after",
    "principal_axis_before", "principal_axis_after",
    "mean_violation_direction",
    "per_constraint_active_count",
]

PER_GEN_FIELDS = [
    "generation",
    "n_cht_calls",
    "n_resample_iterations",
    "n_infeasible_post_resample",
    "n_lambda",
    "infeasibility_rate",
    "n_psd_fallback",
    # Aggregates across all (call, parent) records this generation.
    "mean_log_det_drift",       # mean(log_det_after - log_det_before)
    "max_abs_log_det_drift",    # max |drift| — sanity for volume preservation
    "mean_condition_number_after",
    "max_condition_number_after",
    "mean_violation_axis_angle_deg",
    "mean_effective_pool_weight",
    "any_shrink_applied",
]


# ─────────────────────────────────────────────────────────────────────────────
# Buffer drain → CSV append
# ─────────────────────────────────────────────────────────────────────────────

def _jsonify(v):
    """JSON-encode list-valued cells; pass through scalars as-is."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return json.dumps(list(v))
    return v


def append_per_call_rows(csv_path: Path, gen: int, records: list[dict]) -> None:
    """Append the drained per-call records to cht_per_call.csv.

    Writes a header row on first call (when the file does not yet exist).
    """
    csv_path = Path(csv_path)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_CALL_FIELDS)
        if is_new:
            writer.writeheader()
        for rec in records:
            row = {"generation": gen}
            for k in PER_CALL_FIELDS[1:]:
                row[k] = _jsonify(rec.get(k))
            writer.writerow(row)


def _summarise(gen: int, records: list[dict],
                n_resample_iterations: int,
                n_infeasible_post_resample: int,
                n_lambda: int) -> dict:
    """Reduce a generation's worth of per-call records to one summary row.

    Per-parent variability is collapsed by mean/max so the per-gen CSV
    stays small and easy to plot.  The richer per-call signals are still
    available in cht_per_call.csv for any analysis that needs them.
    """
    drifts, conds, angles, pool_ws = [], [], [], []
    n_psd_fallback = 0
    any_shrink = False
    for r in records:
        a, b = r.get("log_det_C_after"), r.get("log_det_C_before")
        if a is not None and b is not None:
            drifts.append(a - b)
        c = r.get("condition_number_after")
        if c is not None:
            conds.append(c)
        ang = r.get("violation_axis_angle_deg")
        if ang is not None:
            angles.append(ang)
        pw = r.get("effective_pool_weight")
        if pw is not None:
            pool_ws.append(pw)
        if r.get("psd_fallback"):
            n_psd_fallback += 1
        if r.get("shrink_applied"):
            any_shrink = True

    def _mean(xs):  return float(np.mean(xs)) if xs else None
    def _max(xs):   return float(np.max(xs))  if xs else None
    def _absmax(xs): return float(np.max(np.abs(xs))) if xs else None

    return {
        "generation":                     gen,
        "n_cht_calls":                    len(records),
        "n_resample_iterations":          n_resample_iterations,
        "n_infeasible_post_resample":     n_infeasible_post_resample,
        "n_lambda":                       n_lambda,
        "infeasibility_rate":             (n_infeasible_post_resample / n_lambda) if n_lambda else None,
        "n_psd_fallback":                 n_psd_fallback,
        "mean_log_det_drift":             _mean(drifts),
        "max_abs_log_det_drift":          _absmax(drifts),
        "mean_condition_number_after":    _mean(conds),
        "max_condition_number_after":     _max(conds),
        "mean_violation_axis_angle_deg":  _mean(angles),
        "mean_effective_pool_weight":     _mean(pool_ws),
        "any_shrink_applied":             any_shrink,
    }


def append_per_gen_row(csv_path: Path, gen: int, records: list[dict],
                        n_resample_iterations: int,
                        n_infeasible_post_resample: int,
                        n_lambda: int) -> dict:
    """Append the per-gen summary row.  Returns the row dict for caller use."""
    csv_path = Path(csv_path)
    row = _summarise(gen, records, n_resample_iterations,
                     n_infeasible_post_resample, n_lambda)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_GEN_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return row


def drain_and_persist(strategy, gen: int, out_dir: Path,
                       n_resample_iterations: int,
                       n_infeasible_post_resample: int,
                       n_lambda: int) -> list[dict]:
    """One-call helper invoked by main.py after each generation.

    Drains ``strategy.cht_diag_buffer`` (clearing it), writes both CSVs,
    and returns the drained records so the caller can keep them around
    for plotting if it wants to.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(strategy.cht_diag_buffer)
    strategy.cht_diag_buffer.clear()

    append_per_call_rows(out_dir / "cht_per_call.csv", gen, records)
    append_per_gen_row(
        out_dir / "cht_per_gen.csv", gen, records,
        n_resample_iterations, n_infeasible_post_resample, n_lambda,
    )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _to_float(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_list(s):
    if s is None or s == "":
        return None
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return None


def plot_cht_diagnostics(out_dir: Path, current_gen: int) -> Path | None:
    """Generate the six-panel CHT diagnostic figure from the on-disk CSVs.

    Reading from CSV (rather than holding the entire history in memory)
    means this function can be called as a pure side-effect at any
    plotting cadence without changing the main-loop's data structures.

    Returns the path of the saved figure, or None if there is no data
    yet.
    """
    out_dir = Path(out_dir)
    per_call_rows = _read_csv(out_dir / "cht_per_call.csv")
    per_gen_rows  = _read_csv(out_dir / "cht_per_gen.csv")

    if not per_call_rows or not per_gen_rows:
        return None

    # ── Build per-(gen, parent) tidy arrays ──────────────────────────────
    gens_pc, parents_pc = [], []
    log_det_after, cond_after, angle_pc, pool_w_pc = [], [], [], []
    eigs_after_one_parent_by_gen = {}   # {gen: vp_after} for parent 0 only
    constraint_counts_by_gen = {}       # {gen: [counts...]} aggregated across parents
    for r in per_call_rows:
        g = int(r["generation"])
        p = int(r["parent_idx"])
        gens_pc.append(g); parents_pc.append(p)
        log_det_after.append(_to_float(r["log_det_C_after"]))
        cond_after.append(_to_float(r["condition_number_after"]))
        angle_pc.append(_to_float(r["violation_axis_angle_deg"]))
        pool_w_pc.append(_to_float(r["effective_pool_weight"]))
        if p == 0 and r.get("phase") == "post_eval":
            ev = _to_list(r["eigenvalues_after"])
            if ev is not None:
                eigs_after_one_parent_by_gen[g] = ev
        cc = _to_list(r["per_constraint_active_count"])
        if cc is not None:
            agg = constraint_counts_by_gen.setdefault(g, np.zeros(len(cc)))
            constraint_counts_by_gen[g] = agg + np.array(cc)

    gens_pc = np.array(gens_pc); parents_pc = np.array(parents_pc)
    log_det_after = np.array(log_det_after, dtype=float)
    cond_after    = np.array(cond_after,    dtype=float)
    angle_pc      = np.array(angle_pc,      dtype=float)

    # ── Per-gen series ───────────────────────────────────────────────────
    pg_gens, pg_infeas, pg_resample, pg_psd = [], [], [], []
    for r in per_gen_rows:
        pg_gens.append(int(r["generation"]))
        pg_infeas.append(_to_float(r["infeasibility_rate"]))
        pg_resample.append(_to_float(r["n_resample_iterations"]))
        pg_psd.append(_to_float(r["n_psd_fallback"]))
    pg_gens = np.array(pg_gens)
    pg_infeas = np.array(pg_infeas, dtype=float)
    pg_resample = np.array(pg_resample, dtype=float)

    # ── Figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=120)
    fig.suptitle(f"CHT diagnostics through generation {current_gen}", fontsize=14)

    unique_parents = sorted(set(parents_pc.tolist()))

    # 1) Volume preservation — log_det per parent
    ax = axes[0, 0]
    for p in unique_parents:
        mask = parents_pc == p
        ax.plot(gens_pc[mask], log_det_after[mask], lw=0.7, alpha=0.7,
                label=f"parent {p}")
    ax.set_title("log det(C) — volume should be approximately preserved")
    ax.set_xlabel("Generation"); ax.set_ylabel("log det(C) after CHT")
    ax.grid(alpha=0.3)

    # 2) Condition number per parent
    ax = axes[0, 1]
    for p in unique_parents:
        mask = parents_pc == p
        ax.plot(gens_pc[mask], cond_after[mask], lw=0.7, alpha=0.7,
                label=f"parent {p}")
    ax.set_yscale("log")
    ax.set_title("Condition number λ_max/λ_min — rising = anisotropy growing")
    ax.set_xlabel("Generation"); ax.set_ylabel("κ(C)")
    ax.grid(alpha=0.3, which="both")

    # 3) Eigenvalue spectrum heatmap for parent 0 (post_eval phase only)
    ax = axes[0, 2]
    if eigs_after_one_parent_by_gen:
        gs = sorted(eigs_after_one_parent_by_gen.keys())
        spectrum = np.array([eigs_after_one_parent_by_gen[g] for g in gs]).T  # (n_dim, n_gen)
        # log-scale colour: eigenvalues span many orders of magnitude
        im = ax.imshow(np.log10(spectrum + 1e-300), aspect="auto",
                       origin="lower",
                       extent=[gs[0], gs[-1], 0, spectrum.shape[0]],
                       cmap="viridis")
        plt.colorbar(im, ax=ax, label="log10(eigenvalue)")
        ax.set_title("Eigenvalue spectrum (parent 0, post_eval)")
        ax.set_xlabel("Generation"); ax.set_ylabel("Eigenvalue index")
    else:
        ax.text(0.5, 0.5, "no parent-0 post_eval data yet",
                ha="center", va="center", transform=ax.transAxes)

    # 4) Infeasibility rate + resample iterations on twin axes
    ax = axes[1, 0]
    ax.plot(pg_gens, pg_infeas, color="C3", label="infeasibility rate")
    ax.set_xlabel("Generation"); ax.set_ylabel("infeasibility rate", color="C3")
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(pg_gens, pg_resample, color="C0", lw=0.8, alpha=0.7,
             label="resample iters")
    ax2.set_ylabel("# resample iterations", color="C0")
    ax.set_title("Infeasibility & resample iterations")

    # 5) Per-constraint activity heatmap
    ax = axes[1, 1]
    if constraint_counts_by_gen:
        gs = sorted(constraint_counts_by_gen.keys())
        n_constraints = len(constraint_counts_by_gen[gs[0]])
        mat = np.array([constraint_counts_by_gen[g] for g in gs]).T  # (m, n_gen)
        im = ax.imshow(mat, aspect="auto", origin="lower",
                       extent=[gs[0], gs[-1], 0, n_constraints],
                       cmap="magma")
        plt.colorbar(im, ax=ax, label="# violators (summed across parents/calls)")
        ax.set_title("Per-constraint activity")
        ax.set_xlabel("Generation"); ax.set_ylabel("Constraint index")
    else:
        ax.text(0.5, 0.5, "no per-constraint data yet",
                ha="center", va="center", transform=ax.transAxes)

    # 6) Principal axis vs mean violation direction angle
    ax = axes[1, 2]
    valid = np.isfinite(angle_pc)
    if np.any(valid):
        ax.scatter(gens_pc[valid], angle_pc[valid], s=4, alpha=0.4,
                   c=parents_pc[valid], cmap="tab10")
        ax.axhline(90, color="k", ls="--", lw=0.8, alpha=0.6,
                   label="orthogonal (target)")
        ax.set_ylim(0, 92)
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Angle: principal axis vs mean violation direction")
    ax.set_xlabel("Generation"); ax.set_ylabel("angle (deg)")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / f"cht_diagnostics_gen_{current_gen:04d}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

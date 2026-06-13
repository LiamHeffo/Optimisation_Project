"""
Per-generation diagnostics for the Arnold & Hansen 2012 constraint-handling
technique.

Mirrors ``cht_diagnostics.py`` structurally: the strategy fills
``strategy.arnold_diag_buffer`` with one record per (parent, infeasible
offspring) Eq. 7 invocation; this module drains the buffer at
end-of-generation and appends to two on-disk CSVs:

* ``arnold_per_call.csv`` — one row per Eq. 7 call (raw signal)
* ``arnold_per_gen.csv``  — one row per generation (aggregates)

Plotting lives in ``plotting.py`` for parity with the Chocat path and
to share the matplotlib helpers already there.
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
    "generation", "parent_idx", "lineage_id",
    "m_active",            # number of constraints active in this Eq. 7 call
    "A_delta_fro",         # ‖A_new − A_old‖_F
    "shrink_applied", "psd_fallback",
    "active_js",           # JSON list of constraint indices that were active
    "v_norms",             # JSON list of ‖v_j‖ for each active j
]

PER_GEN_FIELDS = [
    "generation",
    "n_arnold_calls",
    "n_infeasible",          # total infeasible offspring observed this gen
    "n_lambda",
    "infeasibility_rate",
    "n_psd_fallback",
    "n_no_active_js",        # calls that hit no finite-positive g_j (e.g. +inf-only)
    "mean_m_active",
    "mean_A_delta_fro",
    "max_A_delta_fro",
    "any_shrink_applied",
]


def _jsonify(v):
    """JSON-encode list-valued cells; pass scalars through."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return json.dumps(list(v))
    return v


def append_per_call_rows(csv_path: Path, gen: int, records: list[dict]) -> None:
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
                n_infeasible: int, n_lambda: int) -> dict:
    """Reduce a generation's worth of per-call records to one summary row.

    ``n_infeasible`` is the number of infeasible offspring observed this
    generation (i.e. how many slots produced no selection candidate),
    not the number of Eq. 7 calls — those may differ when an infeasible
    has no finite-positive g_j (the call records a "no active js" entry
    but no shrinkage was applied).
    """
    fros, m_actives = [], []
    n_psd_fallback = 0
    n_no_active_js = 0
    any_shrink = False
    for r in records:
        f = r.get("A_delta_fro")
        if f is not None:
            fros.append(float(f))
        ma = r.get("m_active")
        if ma is not None:
            m_actives.append(int(ma))
            if int(ma) == 0:
                n_no_active_js += 1
        if r.get("psd_fallback"):
            n_psd_fallback += 1
        if r.get("shrink_applied"):
            any_shrink = True

    def _mean(xs): return float(np.mean(xs)) if xs else None
    def _max(xs):  return float(np.max(xs))  if xs else None

    return {
        "generation":           gen,
        "n_arnold_calls":       len(records),
        "n_infeasible":         n_infeasible,
        "n_lambda":             n_lambda,
        "infeasibility_rate":   (n_infeasible / n_lambda) if n_lambda else None,
        "n_psd_fallback":       n_psd_fallback,
        "n_no_active_js":       n_no_active_js,
        "mean_m_active":        _mean(m_actives),
        "mean_A_delta_fro":     _mean(fros),
        "max_A_delta_fro":      _max(fros),
        "any_shrink_applied":   any_shrink,
    }


def append_per_gen_row(csv_path: Path, gen: int, records: list[dict],
                        n_infeasible: int, n_lambda: int) -> dict:
    csv_path = Path(csv_path)
    row = _summarise(gen, records, n_infeasible, n_lambda)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_GEN_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return row


def drain_and_persist(strategy, gen: int, out_dir: Path,
                       n_infeasible: int, n_lambda: int) -> list[dict]:
    """One-call helper invoked by main.py after each generation.

    Drains ``strategy.arnold_diag_buffer`` (clearing it), writes both CSVs,
    and returns the drained records.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(getattr(strategy, "arnold_diag_buffer", []))
    if hasattr(strategy, "arnold_diag_buffer"):
        strategy.arnold_diag_buffer.clear()

    append_per_call_rows(out_dir / "arnold_per_call.csv", gen, records)
    append_per_gen_row(
        out_dir / "arnold_per_gen.csv", gen, records,
        n_infeasible, n_lambda,
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


def plot_arnold_diagnostics(out_dir: Path) -> Path | None:
    """Generate a four-panel Arnold diagnostic figure from on-disk CSVs.

    Panels:
      (1) Infeasibility rate per generation — how often parents draw an
          infeasible offspring (Arnold mode "wastes" generations on this).
      (2) Mean & max A_delta_fro per generation — magnitude of the
          Eq. 7 subtractive step on the Cholesky factor.
      (3) m_active per generation (mean) — how many constraints are
          driving each Eq. 7 call.
      (4) Per-constraint mean ‖v_j‖ over time — the central question
          "is the v_j filter actually learning the boundary?".  Only
          constraints that became active at least once are drawn.

    Returns the path of the saved figure, or None if no data yet.
    """
    out_dir = Path(out_dir)
    per_call_rows = _read_csv(out_dir / "arnold_per_call.csv")
    per_gen_rows  = _read_csv(out_dir / "arnold_per_gen.csv")
    if not per_gen_rows:
        return None

    # ── per-gen aggregates ──
    gens = np.array([int(r["generation"]) for r in per_gen_rows])
    rate = np.array([_to_float(r["infeasibility_rate"]) or 0.0
                     for r in per_gen_rows])
    mean_fro = np.array([_to_float(r["mean_A_delta_fro"]) or 0.0
                         for r in per_gen_rows])
    max_fro  = np.array([_to_float(r["max_A_delta_fro"]) or 0.0
                         for r in per_gen_rows])
    mean_m   = np.array([_to_float(r["mean_m_active"]) or 0.0
                         for r in per_gen_rows])

    # ── per-call: accumulate mean ‖v_j‖ by (gen, j) ──
    # v_norms is a list aligned with active_js: row k → ‖v_{active_js[k]}‖.
    # Pool by gen first (averaging across parents and across calls), then
    # plot one line per j-index that ever appeared.
    sums:   dict[tuple[int, int], float] = {}
    counts: dict[tuple[int, int], int]   = {}
    for r in per_call_rows:
        g = int(r["generation"])
        active = _to_list(r.get("active_js")) or []
        norms  = _to_list(r.get("v_norms"))   or []
        for j, vn in zip(active, norms):
            key = (g, int(j))
            sums[key]   = sums.get(key, 0.0) + float(vn)
            counts[key] = counts.get(key, 0) + 1

    # constraint indices that ever appeared
    js_seen = sorted({k[1] for k in sums.keys()})

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=120)

    # (1) infeasibility rate
    ax = axes[0, 0]
    ax.plot(gens, rate, color="#0a4")
    ax.set_title("Arnold: infeasibility rate per generation")
    ax.set_xlabel("generation")
    ax.set_ylabel("n_infeasible / λ")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)

    # (2) A_delta_fro
    ax = axes[0, 1]
    ax.plot(gens, mean_fro, label="mean", color="#06c")
    ax.plot(gens, max_fro,  label="max",  color="#c30", alpha=0.7)
    ax.set_title("‖A_new − A‖_F per Eq. 7 call (per-gen mean / max)")
    ax.set_xlabel("generation")
    ax.set_ylabel("Frobenius norm")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.legend()
    ax.grid(alpha=0.3)

    # (3) m_active
    ax = axes[1, 0]
    ax.plot(gens, mean_m, color="#a0a")
    ax.set_title("mean m_active per Eq. 7 call")
    ax.set_xlabel("generation")
    ax.set_ylabel("# active constraints")
    ax.grid(alpha=0.3)

    # (4) per-constraint ‖v_j‖
    ax = axes[1, 1]
    if js_seen:
        cmap = plt.get_cmap("tab20")
        for k, j in enumerate(js_seen):
            xs = sorted({g for (g, jj) in sums.keys() if jj == j})
            ys = [sums[(g, j)] / counts[(g, j)] for g in xs]
            ax.plot(xs, ys, label=f"j={j}", color=cmap(k % 20),
                    linewidth=1.2)
        # Many constraints can be active in our 18-element vector; cap
        # legend rows so it fits.
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_title("mean ‖v_j‖ per active constraint (per gen)")
    ax.set_xlabel("generation")
    ax.set_ylabel("‖v_j‖")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig_path = out_dir / "arnold_diagnostics.png"
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path

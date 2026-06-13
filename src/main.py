"""
Main entry point for the X2 free-piston driver multi-objective optimisation.

Experiment configuration
------------------------
Each element of experiment_types is a 4-tuple:
    (sim_type, pop_size, step_size, p4_treatment)

sim_type options:
    'ParentValue'      – infeasible offspring inherit the parent's value for
                         the violated attribute
    'ElitistCrossover' – the most hypervolume-contributing parent donates
    'RandomCrossover'  – a randomly selected parent donates
    'Penalty'          – infeasible individuals receive a fitness penalty
    'CovarianceCHT'    – infeasibles are NOT repaired and NOT evaluated; their
                         constraint vectors feed Chocat 2015's eigenvalue
                         shrinkage of each parent's Cholesky factor (with
                         Adaptation-B Mahalanobis pooling).  See
                         StrategyMultiObjective._chtCovarianceUpdate.

p4_treatment options:
    'hard_bounds_on_p4' – enforce the upper p4 bound via retransformation
    None                – no special treatment

Output layout
-------------
Each invocation creates a fresh, numbered run folder under

    <repo>/Results/<RESULTS_CATEGORY>/<RUN_PREFIX>_NNNN/

with one subfolder per output type (per-gen plots, per-gen CSVs,
convergence, summary).  Auto-numbering uses the largest existing
NNNN + 1.

Module-level setup
------------------
The DEAP creator types (FitnessMulti, Individual) are registered here.
This is intentional: creator.create has a side-effect on a global registry,
so it must run exactly once per process.  Both this file and
evaluation_shortcuts.py rely on those types being registered.
"""

import gc
import os
import pathlib
import resource
import subprocess
import sys
import time
import multiprocessing
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from deap import base, creator, tools

from algorithm.toolbox   import Toolbox
from algorithm.hypervolume import HyperVolume
from algorithm.cmaes     import (
    StrategyMultiObjective,
    is_cht_active, is_al_active, cht_method,
)
from problem.config      import (
    APPROX_IDEAL, APPROX_NADIR,
    APPROX_IDEAL_2D, APPROX_NADIR_2D,
    BOUNDS,
    he_lower, he_upper,
    driver_p_lower, driver_p_upper,
    p4_lower, p4_upper,
    D_throat_lower, D_throat_upper,
    reservoir_lower, reservoir_upper,
    buffer_length_lower, buffer_length_upper,
)
from problem.transforms  import variable_transformation, variable_untransformation, unnormalise_fitness
from problem.evaluate    import evaluate, set_logbook, _PITOT3_FAILURE_SENTINEL
from problem.feasibility import evaluate_constraints, is_feasible
from problem.sampling    import lhs_sample, draw_structurally_feasible_lhs
from plotting            import (
    plot_objective_space,
    plot_objective_space_3d,
    plot_objective_space_heatmap,
    plot_holdtime_impactspeed_2d,
    plot_archive_holdtime_impactspeed_2d,
    plot_current_parent_population_2d,
)
from results_io          import (
    setup_run_directory,
    setup_subfolders,
    write_population_csv,
)
from cht_diagnostics     import drain_and_persist as _cht_drain_and_persist
from cht_diagnostics     import drain_and_persist_al as _al_drain_and_persist
from cht_diagnostics     import plot_cht_diagnostics as _cht_plot
from arnold_diagnostics  import drain_and_persist as _arnold_drain_and_persist
from arnold_diagnostics  import plot_arnold_diagnostics as _arnold_plot
from utils               import parallelization_setup

# ─────────────────────────────────────────────────────────────────────────────
# DEAP type registration  (runs once on import)
# ─────────────────────────────────────────────────────────────────────────────

# FitnessMulti: all three objectives are minimised (weights = -1).
# The evaluator returns normalised values, so minimising maps to:
#   delta_vs1   → 0 is best
#   hold_time   → 0 is best (we negate: longer hold time = smaller normalised value)
#   impact_speed → 0 is best (same negation logic)
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual",   list, fitness=creator.FitnessMulti,
               ind_number=int, sim_type=str, bounds=list)

# 2-objective variants used by the CHT_AL sim_type, where delta_vs1 has
# been moved from a Pareto objective to an Augmented-Lagrangian constraint.
# Registered unconditionally so the namespace is always populated; main()
# picks which class to instantiate at run-time based on sim_type.
creator.create("FitnessMulti2D", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual2D",   list, fitness=creator.FitnessMulti2D,
               ind_number=int, sim_type=str, bounds=list)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons
# ─────────────────────────────────────────────────────────────────────────────

toolbox = Toolbox()
toolbox.register("evaluate", evaluate)

# Reference point for HV computation: the all-zeros point in normalised
# space.  Instantiated as 3-D for legacy sim_types; main() rebinds the
# name to a 2-D HyperVolume when sim_type == 'CHT_AL'.
pop_hypervolumes = HyperVolume(np.array((0, 0, 0)))

normalised = True

# ─────────────────────────────────────────────────────────────────────────────
# Output structure
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_CATEGORY = "parent_value_with_recomb"
RUN_PREFIX       = "pv_w_rec"
SAVE_INTERVAL    = 10

# Legacy 3-objective output structure (delta_vs1 + hold_time + impact_speed).
OUTPUT_FOLDERS = [
    "pareto_3d",
    "pareto_heatmap",
    "pareto_dvs1_holdtime",
    "pareto_dvs1_impactspeed",
    "pareto_holdtime_impactspeed",
    "population",
    "convergence",
    "summary",
    # CHT diagnostics (CSVs + per-SAVE_INTERVAL summary plots).  Created
    # for every run; only populated when sim_type is in CHOCAT_SIM_TYPES.
    "cht_diagnostics",
    # Arnold diagnostics — created for every run, populated only when
    # sim_type is in ARNOLD_SIM_TYPES.  Mirror structure of cht_diagnostics.
    "arnold_diagnostics",
    # Per-generation strategy state (σ and psucc per parent slot).
    # Populated for ALL sim_types so the σ death-spiral hypothesis
    # can be verified independently of constraint-handling choice.
    "strategy_diagnostics",
]

# 2-objective output structure used by both CHT_AL and ArnoldCHT_AL: no
# 3-D Pareto plot, no delta_vs1-vs-* plots — delta_vs1 is now a
# constraint.  Plus al_diagnostics for AL telemetry and arnold_diagnostics
# for the Arnold CHT family.
RESULTS_CATEGORY_AL = "al_cht_recomb"
RUN_PREFIX_AL       = "al_cht"
OUTPUT_FOLDERS_AL = [
    "pareto_holdtime_impactspeed",
    "archive_pareto",
    "current_parents",
    "population",
    "convergence",
    "summary",
    "cht_diagnostics",
    "arnold_diagnostics",
    "strategy_diagnostics",
    "al_diagnostics",
]


def _run_constants(sim_type):
    """Resolve sim_type-specific run constants in one place.

    Returns
    -------
    (results_category, run_prefix, output_folders, Individual_cls,
     ideal_point, nadir_point) :
        Individual_cls is the DEAP class to instantiate for each
        offspring (Individual for 3-objective, Individual2D for AL).
        ideal_point / nadir_point are used by the unnormalisation step
        in the summary writer; they match the dimensionality of the
        selected Individual_cls.
    """
    # AL-active sim_types (CHT_AL and ArnoldCHT_AL) share the 2-objective
    # output structure: delta_vs1 has been moved out of fitness and is
    # handled as an AL constraint, so plots / CSVs are (hold_time,
    # impact_speed) only.
    if is_al_active(sim_type):
        return (
            RESULTS_CATEGORY_AL, RUN_PREFIX_AL, OUTPUT_FOLDERS_AL,
            creator.Individual2D,
            APPROX_IDEAL_2D, APPROX_NADIR_2D,
        )
    return (
        RESULTS_CATEGORY, RUN_PREFIX, OUTPUT_FOLDERS,
        creator.Individual,
        APPROX_IDEAL, APPROX_NADIR,
    )


def _cheap_al_proxy(strategy):
    """Return the (F̄, ḡ_AL) cheap proxy from the current parent set.

    The AL coefficient adaptation in pycma expects values "at the
    distribution mean".  In MO-CMA-ES there is no single mean; per the
    user-confirmed design we average over parents that survived
    selection.  Zero extra heavy evaluations.

    Returns (None, None) when no parent has both valid fitness and a
    g_al value (would only happen pathologically — e.g. the entire
    parent population came from infeasible-on-shock-speed individuals,
    which selection should have rejected anyway).
    """
    Fs, gals = [], []
    for p in strategy.parents:
        if p.fitness.valid and getattr(p, "_g_al", None) is not None:
            Fs.append(sum(p.fitness.values))
            gals.append(np.asarray(p._g_al, dtype=float))
    if not Fs:
        return None, None
    return float(np.mean(Fs)), np.mean(np.stack(gals, axis=0), axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hv_contributions(population, ref=None):
    """Per-individual HV contribution = HV(pop) − HV(pop ∖ {i}).

    The HyperVolume class expects "to maximise" inputs, so we negate the
    fitness values (which are all to-minimise) before passing them in.

    The reference point dimensionality is auto-detected from the first
    feasible individual's fitness tuple — 3-D for legacy sim_types, 2-D
    for CHT_AL.  Caller may override via ``ref``.

    Infeasible individuals (fitness.valid is False) have no fitness and
    therefore no HV contribution; they appear in the returned list as
    None and are omitted from the HV computation entirely.
    """
    if len(population) == 0:
        return []
    feasible_idx = [i for i, ind in enumerate(population) if ind.fitness.valid]
    if not feasible_idx:
        return [None] * len(population)
    fits_neg = np.array([list(population[i].fitness.values) for i in feasible_idx]) * -1
    if ref is None:
        ref = np.zeros(fits_neg.shape[1])
    hv = HyperVolume(ref)
    full = hv.compute(fits_neg)
    contribs_feasible = []
    for k in range(len(feasible_idx)):
        partial = hv.compute(np.delete(fits_neg, k, axis=0))
        contribs_feasible.append(full - partial)
    out = [None] * len(population)
    for k, i in enumerate(feasible_idx):
        out[i] = contribs_feasible[k]
    return out


def _build_pop_row(ind, gen, slot_idx, sigma_used, parent_idx, hv_contribution, bounds):
    """One CSV row for one individual.

    Infeasible individuals (no valid fitness) emit None for every objective
    column so downstream analysis can distinguish "not evaluated" from a
    real zero.  The 'feasible' and 'max_g' columns surface the constraint
    state for post-hoc CHT diagnostics.

    The fitness dimensionality is auto-detected from the individual's
    fitness tuple.  For CHT_AL (2-D fitness) the legacy delta_vs1 columns
    are recovered from ``ind._g_al`` (the AL constraint vector, =
    delta_vs1 - al_tol) so the CSV schema stays stable for downstream
    analysis tools — the columns are identical, just sourced differently.
    """
    raw_vars    = variable_untransformation(ind, bounds)
    scaled_vars = list(ind)

    is_2d_fitness = (
        ind.fitness.valid and len(ind.fitness.values) == 2
    )

    if ind.fitness.valid:
        scaled_objs_raw = list(ind.fitness.values)
        if is_2d_fitness:
            # CHT_AL: fitness is (hold_time, impact_speed) — delta_vs1 is
            # not in fitness.values but is recoverable from g_al.
            raw_2d  = list(unnormalise_fitness(ind.fitness.values,
                                               APPROX_IDEAL_2D, APPROX_NADIR_2D))
            g_al    = getattr(ind, "_g_al", None)
            al_tol  = getattr(ind, "al_tol", 100.0)
            raw_dvs = (float(g_al[0]) + al_tol) if g_al is not None else None
            raw_objs    = [raw_dvs, raw_2d[0], raw_2d[1]]
            # delta_vs1 has no normalisation in 2-D mode; emit None.
            scaled_objs = [None, scaled_objs_raw[0], scaled_objs_raw[1]]
        else:
            scaled_objs = scaled_objs_raw
            raw_objs    = list(unnormalise_fitness(ind.fitness.values,
                                                   APPROX_IDEAL, APPROX_NADIR))
    else:
        scaled_objs = [None, None, None]
        raw_objs    = [None, None, None]

    g = getattr(ind, "_g", None)
    feasible = getattr(ind, "_feasible", None)
    max_g = float(np.max(g)) if g is not None else None

    return {
        "generation":            gen,
        "ind_number":            slot_idx,
        "parent_idx":            parent_idx,
        "chosen":                False,
        "offspring_ind_number":  None,
        "sigma":                 sigma_used,
        "hv_contribution":       hv_contribution,
        "feasible":              feasible,
        "max_g":                 max_g,
        "raw_pct_he":            raw_vars[0],
        "raw_driver_p":          raw_vars[1],
        "raw_p4":                raw_vars[2],
        "raw_d_throat":          raw_vars[3],
        "raw_reservoir_p":       raw_vars[4],
        "raw_buffer_length":     raw_vars[5],
        "scaled_pct_he":         scaled_vars[0],
        "scaled_driver_p":       scaled_vars[1],
        "scaled_p4":             scaled_vars[2],
        "scaled_d_throat":       scaled_vars[3],
        "scaled_reservoir_p":    scaled_vars[4],
        "scaled_buffer_length":  scaled_vars[5],
        "raw_delta_vs1":         raw_objs[0],
        "raw_hold_time":         raw_objs[1],
        "raw_impact_speed":      raw_objs[2],
        "scaled_delta_vs1":      scaled_objs[0],
        "scaled_hold_time":      scaled_objs[1],
        "scaled_impact_speed":   scaled_objs[2],
    }


def _make_snapshot(gen, population, sigmas_per_slot, parent_idx_per_slot, bounds):
    """Build snapshot rows for one generation.

    Each individual is also tagged with (_origin_gen, _snapshot_idx) so that
    later generations can locate this row when they need to fill in the
    'offspring_ind_number' or 'chosen' columns.
    """
    contributions = _hv_contributions(population)
    rows = []
    for i, ind in enumerate(population):
        row = _build_pop_row(
            ind, gen, i,
            sigmas_per_slot[i],
            parent_idx_per_slot[i],
            contributions[i],
            bounds,
        )
        ind._origin_gen   = gen
        ind._snapshot_idx = i
        rows.append(row)
    return rows


def _mark_chosen(gen_snapshots, chosen):
    """Set chosen=True for every snapshot row whose individual survived
    selection.  Idempotent — once True, stays True."""
    for ind in chosen:
        if hasattr(ind, '_origin_gen') and hasattr(ind, '_snapshot_idx'):
            snap = gen_snapshots.get(ind._origin_gen)
            if snap is not None and 0 <= ind._snapshot_idx < len(snap):
                snap[ind._snapshot_idx]["chosen"] = True


def _fill_offspring(gen_snapshots, parents_at_generate, offspring):
    """Write each new offspring's ind_number into its parent's snapshot row.

    Only the *first* offspring is recorded per row.  A long-surviving parent
    that produces one offspring per generation will have its first-gen
    offspring recorded; subsequent ones are tracked instead in any snapshot
    that captures the parent again (e.g. via failure-substitution).
    """
    for off in offspring:
        if not hasattr(off, '_ps'):
            continue
        tag, p_idx = off._ps
        if tag != "o" or p_idx is None or p_idx >= len(parents_at_generate):
            continue
        parent = parents_at_generate[p_idx]
        if not (hasattr(parent, '_origin_gen') and hasattr(parent, '_snapshot_idx')):
            continue
        snap = gen_snapshots.get(parent._origin_gen)
        if snap is None or not (0 <= parent._snapshot_idx < len(snap)):
            continue
        row = snap[parent._snapshot_idx]
        if row.get("offspring_ind_number") is None:
            row["offspring_ind_number"] = off.ind_number


def _save_outputs(bookshelf_gen, gen_snapshots, fitness_history, MU, folders,
                  strategy=None):
    """Write per-generation plots and population CSVs.

    All known snapshots are re-written every save trigger so that lazily-
    filled fields (chosen, offspring_ind_number) propagate to disk as the
    information becomes available.

    Plots are routed by fitness dimensionality (auto-detected from
    fitness_history).  3-D fitness goes to the legacy 5-plot bundle;
    2-D fitness (CHT_AL) goes to a single hold_time-vs-impact_speed
    plot — there is no third axis to scatter on.
    """
    pop_dir = folders["population"]
    for g, rows in gen_snapshots.items():
        write_population_csv(pop_dir / f"population_gen_{g:04d}.csv", rows)

    is_2d = (
        len(fitness_history) > 0
        and len(fitness_history[0]) == 2
    )

    if is_2d:
        # CHT_AL: only one Pareto plot (the 2-D one) — delta_vs1 is no
        # longer a Pareto axis; it is logged in the AL diagnostics csv.
        plot_holdtime_impactspeed_2d(
            fitness_history,
            MU=MU, gen=bookshelf_gen,
            out_dir=folders["pareto_holdtime_impactspeed"],
        )
        # Archive non-dominated front plot (D1).
        if strategy is not None and "archive_pareto" in folders:
            archive_nd = StrategyMultiObjective._archive_nondominated(
                list(getattr(strategy, "external_archive", []))
            )
            archive_fitness = [m["fitness"] for m in archive_nd]
            plot_archive_holdtime_impactspeed_2d(
                archive_fitness,
                gen=bookshelf_gen,
                out_dir=folders["archive_pareto"],
            )
        # Current parent population snapshot — one point per surviving
        # parent.  Independent of fitness_history; reflects the actual
        # state of the search at this generation.  Fires at gen 0 (the
        # initial sentinel-screened population) and every SAVE_INTERVAL
        # gens thereafter, matching the cadence of every other
        # _save_outputs call.
        if strategy is not None and "current_parents" in folders:
            parent_fitness = [
                tuple(p.fitness.values)
                for p in strategy.parents
                if p.fitness.valid
            ]
            plot_current_parent_population_2d(
                parent_fitness,
                gen=bookshelf_gen,
                out_dir=folders["current_parents"],
            )
    else:
        # Legacy 3-objective plots (unchanged behaviour).
        plot_objective_space(fitness_history, 'delta_vs1', 'hold_time',
                             MU=MU, gen=bookshelf_gen, out_dir=folders["pareto_dvs1_holdtime"])
        plot_objective_space(fitness_history, 'delta_vs1', 'impact_speed',
                             MU=MU, gen=bookshelf_gen, out_dir=folders["pareto_dvs1_impactspeed"])
        plot_objective_space(fitness_history, 'hold_time', 'impact_speed',
                             MU=MU, gen=bookshelf_gen, out_dir=folders["pareto_holdtime_impactspeed"])
        plot_objective_space_3d(fitness_history,
                                MU=MU, gen=bookshelf_gen, out_dir=folders["pareto_3d"])
        plot_objective_space_heatmap(fitness_history,
                                     MU=MU, gen=bookshelf_gen, out_dir=folders["pareto_heatmap"])

    # Belt-and-braces: each plot_* function calls plt.close() but only
    # on the current figure.  plt.close('all') guarantees no pyplot
    # state survives a save burst, which over hundreds of generations
    # would otherwise compound into a noticeable RSS drift.
    plt.close('all')


def _append_strategy_per_gen_row(out_dir, gen, strategy, max_mu=None):
    """Append one row of σ, psucc and lineage state to strategy_per_gen.csv.

    Called once per generation immediately after toolbox.update(), so
    self.sigmas / self.psucc reflect the post-update parent set (i.e.
    the parents that will seed the *next* generate() call).

    Schema: generation, mu, summary stats (mean/min/max for σ and
    psucc), then per-slot lineage_<i>, sigma_<i>, psucc_<i>.  μ is
    constant within a run, so the header is fixed at first write.
    max_mu fixes the column count for M1 restarts that grow mu over time.
    """
    import csv
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "strategy_per_gen.csv"

    sigmas = list(strategy.sigmas)
    psucc  = list(strategy.psucc)
    mu     = len(sigmas)
    if max_mu is None:
        max_mu = mu
    lineage_ids = [
        getattr(p, "_lineage_id", None) for p in strategy.parents
    ]

    fieldnames = (
        ["generation", "mu",
         "mean_sigma", "min_sigma", "max_sigma",
         "mean_psucc", "min_psucc", "max_psucc",
         "gens_silent", "sigma_floor_active",
         "archive_size",
         "restart_count", "restart_event_gen", "pop_size_after_restart"]
        + [f"lineage_{i}" for i in range(max_mu)]
        + [f"sigma_{i}"   for i in range(max_mu)]
        + [f"psucc_{i}"   for i in range(max_mu)]
    )

    floor = getattr(strategy, "sigma_floor_silent", None)
    silent = getattr(strategy, "_gens_silent_count", 0)
    silent_thresh = getattr(strategy, "_gens_silent_threshold", 20)
    floor_active = (floor is not None and silent >= silent_thresh)

    row = {
        "generation":            gen,
        "mu":                    mu,
        "mean_sigma":            float(np.mean(sigmas)),
        "min_sigma":             float(np.min(sigmas)),
        "max_sigma":             float(np.max(sigmas)),
        "mean_psucc":            float(np.mean(psucc)),
        "min_psucc":             float(np.min(psucc)),
        "max_psucc":             float(np.max(psucc)),
        "gens_silent":           silent,
        "sigma_floor_active":    bool(floor_active),
        "archive_size":          len(getattr(strategy, "external_archive", [])),
        "restart_count":         getattr(strategy, "_restart_count", 0),
        "restart_event_gen":     getattr(strategy, "_last_restart_gen", "") or "",
        "pop_size_after_restart": mu,
    }
    for i in range(max_mu):
        row[f"lineage_{i}"] = lineage_ids[i] if i < len(lineage_ids) else ""
        row[f"sigma_{i}"]   = float(sigmas[i]) if i < len(sigmas) else ""
        row[f"psucc_{i}"]   = float(psucc[i])  if i < len(psucc)  else ""

    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Main evolution loop
# ─────────────────────────────────────────────────────────────────────────────

def _write_round2_seed_csv(path, strategy):
    """Persist the non-dominated external archive as a round-2 seed CSV."""
    import csv
    archive = list(getattr(strategy, "external_archive", []))
    nd = StrategyMultiObjective._archive_nondominated(archive)
    if not nd:
        with open(path, "w", newline="") as f:
            f.write("design_0,design_1,design_2,design_3,design_4,design_5\n")
        return
    n_design = len(nd[0]["design"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"design_{k}" for k in range(n_design)])
        for m in nd:
            w.writerow(m["design"])


def _load_round2_seed(seed_csv_path, target_mu, pop_init_fn, bounds):
    """Load seed designs from CSV and pad with fresh feasibles if needed."""
    import csv
    from problem.feasibility import evaluate_constraints, is_feasible
    from problem.transforms   import variable_transformation
    designs = []
    with open(seed_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            designs.append([float(row[k]) for k in reader.fieldnames])
    seen = set()
    deduped = []
    for d in designs:
        key = tuple(round(v, 12) for v in d)
        if key not in seen:
            seen.add(key)
            deduped.append(d)
    if len(deduped) >= target_mu:
        return deduped[:target_mu]
    pad_count = target_mu - len(deduped)
    fresh = variable_transformation(pop_init_fn(pad_count), bounds)
    for slot in range(pad_count):
        attempt = 0
        while not is_feasible(evaluate_constraints(fresh[slot], bounds)):
            attempt += 1
            if attempt > 1000:
                raise RuntimeError(
                    "M2 seed padding could not produce a feasible "
                    "individual after 1000 attempts."
                )
            fresh[slot] = variable_transformation(pop_init_fn(1), bounds)[0]
    return deduped + fresh


def _launch_round_two_if_pending(idx, exp, script_path):
    """Find the most recent round-1 run dir and dispatch round-2 subprocess."""
    sim_type = exp[0]
    results_category, run_prefix, *_ = _run_constants(sim_type)
    results_root = pathlib.Path("Results") / results_category
    if not results_root.exists():
        return
    candidates = sorted(
        [p for p in results_root.iterdir()
         if p.is_dir()
         and p.name.startswith(run_prefix)
         and not p.name.endswith("_r2")
         and (p / "ROUND2_SEED_READY").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print("M2: no round-1 marker found — skipping round 2.")
        return
    r1_dir   = candidates[0]
    seed_csv = r1_dir / "summary" / "round2_seed.csv"
    if not seed_csv.exists():
        print(f"M2 WARNING: marker in {r1_dir.name} but no seed CSV — "
              f"skipping round 2.")
        return
    print(f"\nM2: launching round 2 seeded from {r1_dir.name}\n")
    subprocess.run(
        [sys.executable, str(script_path),
         "--experiment-index", str(idx),
         "--round-two", str(seed_csv),
         "--round-one-dir", str(r1_dir)],
        check=False,
    )


def main(experiment_type, *, round_two_seed=None, round_one_dir=None):
    s1 = time.time()

    _is_round_two  = round_two_seed is not None
    _round_two_seed = round_two_seed
    _round_one_dir  = round_one_dir

    # ── Experiment parameters ─────────────────────────────────────────────
    N           = 6
    pop_size    = experiment_type[1]
    sim_type    = experiment_type[0]
    p4_treatment = experiment_type[3]
    step_size   = experiment_type[2]
    # AL constraint tolerance (m/s on delta_vs1).  Only consumed when
    # sim_type == 'CHT_AL'; legacy sim_types ignore it.  Default 100 m/s
    # per the user-confirmed setting.  experiment_type may be a 4-tuple
    # for legacy YAML entries; fall back to the default in that case.
    al_tol = experiment_type[4] if len(experiment_type) > 4 else 100.0
    # CHT covariance shrinkage strength.  None ⇒ let the strategy fall
    # back to its dimension-dependent default (0.5/(n+2)).  Consumed by
    # both CovarianceCHT and CHT_AL.
    cht_gamma = experiment_type[5] if len(experiment_type) > 5 else None
    # Anti-degeneration feature toggles.  Each key in this dict opts in
    # to one of the experimental fixes for the σ death spiral; all
    # default off so omitting the field reproduces baseline behaviour.
    # See the YAML header for the full schema.
    features = experiment_type[6] if len(experiment_type) > 6 else {}
    NGEN        = experiment_type[7] if len(experiment_type) > 7 else 500
    # Optional LHS seed (experiment_type[8]).  None ⇒ fresh OS entropy on
    # every LatinHypercube call (no reproducibility).  An int seeds a
    # single shared numpy Generator used for the run's initial pop and
    # every subsequent M1/M2 LHS draw.  See _lhs_sample.
    lhs_seed    = experiment_type[8] if len(experiment_type) > 8 else None

    # Compute maximum possible mu across all restarts (for CSV header sizing).
    max_mu = pop_size
    _m1_cfg = features.get("restart_on_sigma_collapse") if isinstance(features, dict) else None
    if _m1_cfg is not None:
        max_mu += int(_m1_cfg["pop_increment"]) * int(_m1_cfg["max_restarts"])

    # M2: in round-2, bump pop_size by pop_increment
    if _is_round_two and isinstance(features, dict) and features.get("restart_round_two"):
        pop_size += int(features["restart_round_two"]["pop_increment"])
    MU, LAMBDA  = pop_size, pop_size

    print(f"Step Size = {step_size}")
    print(f'Pop Size = {pop_size}\n')
    if is_al_active(sim_type):
        print(f'AL tolerance (delta_vs1 ≤): {al_tol} m/s')
    if cht_method(sim_type) == 'chocat':
        print(f'cht_gamma = {cht_gamma if cht_gamma is not None else "default (0.5/(n+2))"}')
    if cht_method(sim_type) == 'arnold':
        print('cht_method = Arnold & Hansen 2012 '
              '(β=0.1/(n+2), c_c=1/(n+2) by default)')
    if features:
        active_features = [k for k, v in features.items() if v not in (None, False, 0)]
        if active_features:
            print(f'Active features: {active_features}')
        else:
            print('Active features: none (baseline)')
    else:
        print('Active features: none (baseline)')

    # ── Output directory layout ───────────────────────────────────────────
    # Per-sim_type constants: legacy modes write a 3-objective tree;
    # CHT_AL writes a 2-objective tree with an extra al_diagnostics dir.
    (results_category, run_prefix, output_folders,
     Individual_cls, ideal_point, nadir_point) = _run_constants(sim_type)
    _r2_prefix = run_prefix + "_r2" if _is_round_two else run_prefix
    run_dir = setup_run_directory(results_category, _r2_prefix)
    folders = setup_subfolders(run_dir, output_folders)
    # M2: write round-1 pointer
    if _is_round_two and _round_one_dir is not None:
        (run_dir / "round1_pointer.txt").write_text(str(_round_one_dir))
    print(f"Run directory: {run_dir}")

    # Per-mode HV calculator.  Rebinds the global ``pop_hypervolumes``
    # name to a local 2-D / 3-D variant — the global lookup at the
    # bottom of the loop will hit this local instead.  No global access
    # leaks because no other site in main.py reads pop_hypervolumes.
    pop_hypervolumes = HyperVolume(
        np.zeros(2 if is_al_active(sim_type) else 3)
    )

    # ── Logbook initialisation ────────────────────────────────────────────
    gen_counter = 0
    toolbox.logbook.add('generation',          gen_counter)
    toolbox.logbook.add('fixer_count',         {f"{gen_counter}": 0})
    toolbox.logbook.add('time_spent_fixing',   0)
    toolbox.logbook.add("time taken",          0)
    toolbox.logbook.add("No. individuals that failed objective tests", 0)
    toolbox.logbook.add("No. individuals that produced no hold time",  0)
    toolbox.logbook.add("No. individuals that failed constraint tests", 0)
    toolbox.logbook.add("hypervolume",         [0 for _ in range(1, NGEN + 1)])
    # Per-generation count of offspring that passed the feasibility check
    # (and were therefore evaluated by SPARK + PITOT3).  A persistently
    # low number signals stagnation — the search ellipsoid is wider than
    # the feasible region so almost every offspring is rejected.  Index
    # is zero-based on generation (slot k holds gen k+1's count).
    toolbox.logbook.add("feasible_offspring_count", [0 for _ in range(1, NGEN + 1)])
    # CovarianceCHT only: how many CHT-and-resample iterations were
    # needed each generation before either every offspring became
    # feasible or the cap was hit.  0 means the first batch was already
    # all-feasible; max_iterations means the cap was reached.
    toolbox.logbook.add("resample_iterations", [0 for _ in range(1, NGEN + 1)])

    # Give the evaluate module a reference to the logbook so it can update
    # counters without accessing a global toolbox.
    set_logbook(toolbox.logbook)

    # ── Design-variable bounds ────────────────────────────────────────────
    bounds = BOUNDS

    # ── Population initialisation ─────────────────────────────────────────
    # Per-slot structural-feasibility resample cap used by the M1 restart
    # padding loops below (init now uses oversample-and-filter via
    # draw_structurally_feasible_lhs, so this constant does NOT govern
    # init).  1000 is a generous fp guard against pop_init returning
    # nothing feasible — empirically the cheap structural filter passes
    # ~99% of the time.
    _MAX_RESAMPLE_ATTEMPTS = 1000

    # Single shared numpy Generator drives every LHS draw in this run
    # (initial pop, M1 restart fresh individuals, M2 padding).  Seeded
    # ⇒ same lhs_seed reproduces the entire LHS stream; unseeded ⇒ OS
    # entropy on every call (legacy behaviour).
    if lhs_seed is not None:
        _lhs_rng = np.random.default_rng(int(lhs_seed))
        print(f"LHS RNG: seeded with lhs_seed={int(lhs_seed)}")
    else:
        _lhs_rng = np.random.default_rng()
        print("LHS RNG: unseeded (fresh OS entropy)")

    # Closure used by M1 / M2 / _load_round2_seed call sites.  Each
    # call advances the shared _lhs_rng so successive draws differ.
    def pop_init(n):
        return lhs_sample(n, rng=_lhs_rng)

    i = 0
    if _round_two_seed is not None:
        # M2 round-2: seed initial population from round-1 archive.
        init_pop_transformed = _load_round2_seed(
            _round_two_seed, MU, pop_init, bounds,
        )
    else:
        # LHS oversample-and-filter (k=4): draws 4·MU points, keeps the
        # first MU that pass evaluate_constraints / is_feasible.  Replaces
        # the slot-by-slot uniform resample.  Per-slot replacement would
        # destroy the LHS stratification on the original draw.
        init_pop_transformed = draw_structurally_feasible_lhs(
            MU, pop_init, bounds, oversample=4, max_doublings=3,
        )

    # Use the dimension-appropriate Individual class — Individual2D for
    # CHT_AL (2-objective fitness), Individual for the legacy 3-objective
    # sim_types.  Picked once via _run_constants() above so this is the
    # only branch needed.
    population = [Individual_cls(x) for x in init_pop_transformed]
    initial_population = population

    for ind in population:
        ind.ind_number = i
        ind.bounds     = bounds
        # al_tol is read by problem.evaluate.evaluate() when computing
        # g_al = delta_vs1 - al_tol in CHT_AL mode.  Setting it on every
        # individual (regardless of sim_type) is harmless: legacy paths
        # never consult it.
        ind.al_tol = al_tol
        i += 1

    parallelization_setup(population)

    # ── Initial evaluation + sentinel screen ─────────────────────────────
    # Sentinels are not detected by evaluate_constraints — they arise
    # only after the expensive SPARK / PITOT3 evaluation.  Without
    # screening they enter the strategy with fitness ≈ (1.0, 1.0) or
    # _feasible=False, biasing the initial AL bootstrap and reducing
    # the effective MU.  Cap at 168 full-rebuild attempts — sentinel
    # rate at init can be considerably higher than at restart since
    # the LHS draw has no prior steering.
    _INIT_SENTINEL_ATTEMPT_CAP = 168
    def _tag(ind, fit, g, g_al):
        ind._g, ind._g_al = g, g_al
        ind._feasible = fit is not None
        if fit is not None:
            ind.fitness.values = fit
        if is_al_active(sim_type):
            ind._pitot3_sentinel = (
                g_al is not None
                and len(g_al) > 0
                and (g_al[0] + al_tol) >= (_PITOT3_FAILURE_SENTINEL - 1.0)
            )
            ind._spark_sentinel = (
                fit is not None
                and all(abs(v - 1.0) < 1e-9 for v in fit)
            )
        else:
            ind._pitot3_sentinel = False
            ind._spark_sentinel  = False

    def _bad(ind):
        return (not ind._feasible
                or ind._pitot3_sentinel
                or ind._spark_sentinel)

    def _failure_label(ind):
        """One-word reason for why ``ind`` was flagged bad.  Priority
        order: pitot3 > spark > eval_failed (these are not mutually
        exclusive — a PITOT3 sentinel can coexist with fit=None)."""
        if ind._pitot3_sentinel:
            return "pitot3_sentinel"
        if ind._spark_sentinel:
            return "spark_sentinel"
        if not ind._feasible:
            return "eval_failed"
        return "clean"

    def _summarise_bad(pop):
        """Return (n_clean, n_bad, breakdown_counts, bad_slot_indices)."""
        bad_idx = [k for k, ind in enumerate(pop) if _bad(ind)]
        counts  = {"pitot3_sentinel": 0, "spark_sentinel": 0, "eval_failed": 0}
        for k in bad_idx:
            counts[_failure_label(pop[k])] += 1
        return len(pop) - len(bad_idx), len(bad_idx), counts, bad_idx

    # Append-only log of every individual evaluated during init.  Lets
    # us recover the good (clean) samples even if the sentinel-screen
    # loop exhausts its attempt cap and raises.  Each call flushes a
    # row immediately so a SIGKILL/uncaught raise still leaves a
    # readable file on disk.  Filter by ``result == 'clean'`` and pick
    # the last row per ``slot`` to recover the accepted set.
    _init_log_path = folders["summary"] / "init_population_log.csv"
    _init_log_fields = [
        "attempt", "slot", "ind_number",
        "design_0", "design_1", "design_2",
        "design_3", "design_4", "design_5",
        "fit_0", "fit_1", "fit_2",
        "g_al_0",
        "feasible", "pitot3_sentinel", "spark_sentinel",
        "result",
    ]
    def _log_init_sample(attempt, slot, ind):
        import csv as _csv_init
        new_file = not _init_log_path.exists()
        with _init_log_path.open("a", newline="") as _f:
            _w = _csv_init.DictWriter(_f, fieldnames=_init_log_fields,
                                      extrasaction="ignore")
            if new_file:
                _w.writeheader()
            fit_vals = list(ind.fitness.values) if ind._feasible else []
            g_al = ind._g_al if ind._g_al is not None else []
            _w.writerow({
                "attempt": attempt,
                "slot": slot,
                "ind_number": getattr(ind, "ind_number", ""),
                "design_0": float(ind[0]),
                "design_1": float(ind[1]),
                "design_2": float(ind[2]),
                "design_3": float(ind[3]),
                "design_4": float(ind[4]),
                "design_5": float(ind[5]),
                "fit_0": fit_vals[0] if len(fit_vals) > 0 else "",
                "fit_1": fit_vals[1] if len(fit_vals) > 1 else "",
                "fit_2": fit_vals[2] if len(fit_vals) > 2 else "",
                "g_al_0": float(g_al[0]) if len(g_al) > 0 else "",
                "feasible":        int(bool(ind._feasible)),
                "pitot3_sentinel": int(bool(ind._pitot3_sentinel)),
                "spark_sentinel":  int(bool(ind._spark_sentinel)),
                "result":          _failure_label(ind),
            })

    for ind in population:
        ind.sim_type   = sim_type
        ind.normalised = normalised
    print("\n── Initial population: first expensive evaluation ──")
    init_fits = list(toolbox.map(toolbox.evaluate, population))
    for ind, (fit, g, g_al) in zip(population, init_fits):
        _tag(ind, fit, g, g_al)
    # Log every slot's first-draw evaluation (attempt = 0).
    for _k, ind in enumerate(population):
        _log_init_sample(attempt=0, slot=_k, ind=ind)
    print(f"  init log written to {_init_log_path}")

    n_clean, n_bad, counts, bad_idx = _summarise_bad(population)
    print(f"  first draw: {n_clean}/{len(population)} clean, "
          f"{n_bad} to replace")
    if n_bad > 0:
        print(f"    breakdown: {counts['pitot3_sentinel']} pitot3_sentinel, "
              f"{counts['spark_sentinel']} spark_sentinel, "
              f"{counts['eval_failed']} eval_failed")
        print(f"    bad slots: {bad_idx}")

    _init_sentinel_attempts = 0
    while any(_bad(ind) for ind in population):
        _init_sentinel_attempts += 1
        if _init_sentinel_attempts > _INIT_SENTINEL_ATTEMPT_CAP:
            raise RuntimeError(
                f"Initial population: could not obtain non-sentinel "
                f"feasible individuals after {_INIT_SENTINEL_ATTEMPT_CAP} "
                f"attempts.  Sentinel rate at init is unexpectedly high "
                f"— check SPARK / PITOT3 worker health."
            )
        bad_slots = [k for k, ind in enumerate(population) if _bad(ind)]
        reasons = {k: _failure_label(population[k]) for k in bad_slots}
        print(f"\n  ── resample attempt {_init_sentinel_attempts}/"
              f"{_INIT_SENTINEL_ATTEMPT_CAP}: "
              f"replacing {len(bad_slots)} slot(s) ──")
        print(f"    reasons: "
              + ", ".join(f"slot {k}={reasons[k]}" for k in bad_slots))

        fresh_designs = draw_structurally_feasible_lhs(
            len(bad_slots), pop_init, bounds, oversample=4, max_doublings=3,
        )
        print(f"    fresh LHS draw + structural filter complete "
              f"({len(fresh_designs)} designs)")

        for k_bad, x_new in zip(bad_slots, fresh_designs):
            new_ind = Individual_cls(x_new)
            new_ind.ind_number = k_bad
            new_ind.bounds     = bounds
            new_ind.al_tol     = al_tol
            new_ind.sim_type   = sim_type
            new_ind.normalised = normalised
            population[k_bad]  = new_ind
        parallelization_setup([population[k] for k in bad_slots])
        new_fits = list(toolbox.map(
            toolbox.evaluate, [population[k] for k in bad_slots]
        ))
        for k_bad, (fit, g, g_al) in zip(bad_slots, new_fits):
            _tag(population[k_bad], fit, g, g_al)
        # Log every replacement evaluated this attempt.  Append-only,
        # so a crash mid-loop still preserves all clean rows up to now.
        for k_bad in bad_slots:
            _log_init_sample(attempt=_init_sentinel_attempts,
                             slot=k_bad, ind=population[k_bad])

        # Per-attempt outcome on the replacements.
        still_bad = [k for k in bad_slots if _bad(population[k])]
        n_recovered = len(bad_slots) - len(still_bad)
        print(f"    outcome: {n_recovered}/{len(bad_slots)} replacements "
              f"clean; {len(still_bad)} still bad")
        if still_bad:
            new_reasons = {k: _failure_label(population[k]) for k in still_bad}
            print(f"    persisting reasons: "
                  + ", ".join(f"slot {k}={new_reasons[k]}" for k in still_bad))

    print(f"\n── Initial population finalised: MU={len(population)}, "
          f"all parents non-sentinel feasible "
          f"(resample attempts used: {_init_sentinel_attempts}/"
          f"{_INIT_SENTINEL_ATTEMPT_CAP}) ──\n")

    # ── Strategy and multiprocessing setup ────────────────────────────────
    # n_constraints is the length of the constraint vector returned by
    # problem.feasibility.evaluate_constraints — used by Arnold modes
    # to allocate one v_j accumulator per constraint per parent.
    # Computed once here from a feasible probe so the dimension is
    # exact, not derived from a formula that could drift if
    # evaluate_constraints' layout ever changes.
    _probe_x = np.full(len(population[0]), 1.5)
    n_constraints = len(evaluate_constraints(_probe_x, bounds))

    strategy = StrategyMultiObjective(
        population, sigma=step_size,
        mu=MU, lambda_=LAMBDA,
        sim_type=sim_type, p4_treatment=p4_treatment,
        bounds=bounds,
        al_tol=al_tol,                # consumed only when AL is active
        cht_gamma=cht_gamma,          # None ⇒ strategy default (0.5/(n+2))
        n_constraints=n_constraints,  # required by Arnold modes
        features=features,            # anti-degeneration toggles (see YAML header)
        logbook=toolbox.logbook,      # injected — no global access inside cmaes.py
    )
    toolbox.register("generate", strategy.generate, Individual_cls)
    toolbox.register("update",   strategy.update)

    # Bootstrap AL coefficients from the initial population's data.
    # Idempotent: pycma's set_coefficients short-circuits once
    # _initialized is fully True; we still call it again every generation
    # below until it is, to refine on additional samples.
    if is_al_active(sim_type):
        F_pop  = [sum(ind.fitness.values) for ind in population if ind._feasible]
        G_AL   = [ind._g_al               for ind in population if ind._feasible]
        if F_pop:
            strategy.init_al(F_pop, G_AL)
            print(f"AL bootstrapped: lam={strategy.al.lam}, mu={strategy.al.mu}")

    # maxtasksperchild caps the number of evaluations a worker handles
    # before the Pool kills and respawns it.  This bounds per-worker
    # memory creep from PITOT3 / SPARK / gdtk.gas, all of which retain
    # state across calls (Lua VMs, cached gas-model objects, GasState /
    # Driver / Tube instances).  Without this, after a few hundred
    # generations the workers' RSS sums up to all available system RAM
    # and the kernel OOM-killer terminates the parent process — visible
    # as "Killed" followed by a flood of worker BrokenPipeErrors.
    #
    # 50 is a balance: large enough that the worker startup cost
    # (loading PITOT3, gdtk, etc.) doesn't dominate the per-eval cost,
    # small enough that any single worker's heap stays bounded.  At
    # pop_size=12, each worker handles ~4 generations before being
    # recycled.
    pool = multiprocessing.Pool(maxtasksperchild=50)
    toolbox.register("map", pool.map)

    # ── Snapshot bookkeeping ──────────────────────────────────────────────
    gen_snapshots = {}
    gen_snapshots[0] = _make_snapshot(
        0, population,
        sigmas_per_slot=[step_size] * MU,
        parent_idx_per_slot=[None] * MU,
        bounds=bounds,
    )

    # Seed fitness_history with the initial population so the "every individual
    # ever sampled" plots include the starting points, not just offspring.
    fitness_history = [tuple(ind.fitness.values) for ind in population]

    # Render the gen-0 Pareto plot (and population CSV) immediately after
    # initialisation.  This exposes the LHS-seeded starting cloud BEFORE
    # any selection pressure has acted on it — useful for verifying the
    # init coverage independently of the evolution trajectory.  The
    # archive plot will be empty at this point (no update() has run yet)
    # and renders an "archive empty" annotation.
    _save_outputs(0, gen_snapshots, fitness_history, MU, folders,
                  strategy=strategy)

    # ── Evolution ─────────────────────────────────────────────────────────
    for gen in range(NGEN):
        toolbox.logbook.bookshelf['generation'] += 1
        bookshelf_gen = toolbox.logbook.bookshelf['generation']
        print('\n')
        print('*' * 30)
        print(f"Generation {bookshelf_gen}")
        print('*' * 30)

        # Snapshot strategy state BEFORE generate(): update() mutates
        # self.sigmas in-place, so we need a frozen view of which step size
        # was used to mutate each parent into its offspring this generation.
        sigmas_at_generate  = list(strategy.sigmas)
        parents_at_generate = list(strategy.parents)

        parents    = population
        population = toolbox.generate()

        # Each new offspring's _ps now points at its parent.  Walk back to
        # the parent's earlier snapshot row and record the offspring's
        # ind_number there.
        _fill_offspring(gen_snapshots, parents_at_generate, population)

        i = 0
        # Feature B3 (al_tol_schedule): read the current AL tolerance
        # from the strategy (returns the static value when no schedule
        # is configured).  Per-generation update — offspring need to
        # see the new tol before evaluate() reads x.al_tol.
        current_al_tol = (
            strategy.current_al_tol()
            if hasattr(strategy, "current_al_tol") else al_tol
        )
        for ind in population:
            ind.normalised = normalised
            ind.ind_number = i
            ind.bounds     = bounds
            # Offspring are freshly constructed by strategy.generate() — they
            # do NOT inherit sim_type or al_tol from the parent.  Without
            # these, evaluate() falls into its legacy 3-objective branch
            # for an Individual2D and the length-2-vs-length-3 fitness
            # assignment later trips DEAP's assertion.
            ind.sim_type = sim_type
            ind.al_tol   = current_al_tol
            i += 1

        # Pre-evaluation CHT pass.  Two families:
        #
        # Chocat (CovarianceCHT / CHT_AL): iterative CHT-and-resample
        # loop (Chocat 2015 Algorithm 3 step 3-2).  Infeasible offspring
        # shrink each parent's covariance and are resampled from the
        # tightened distribution; up to 5 iterations per generation.
        # Mutates population in place.
        #
        # Arnold (ArnoldCHT / ArnoldCHT_AL): one-shot per parent.
        # Infeasible offspring update v_{j,i} (Eq. 6) and apply Eq. 7 to
        # A_i once; the slot is then marked infeasible and contributes
        # NO selection candidate this generation.  Per the paper,
        # iteration is complete — no resampling.
        #
        # Cheap in either case because feasibility_check does NOT call
        # SPARK / PITOT3.  Both families tag every Individual with
        # ._g and ._feasible so downstream code is method-agnostic.
        if is_cht_active(sim_type):
            # Feature C1 (cht_resample_tol): Chocat-only.  Permits slight
            # constraint violations during the resample loop; rejected
            # at strategy __init__ for Arnold sim_types.
            resample_tol = (
                features.get("cht_resample_tol") if isinstance(features, dict) else None
            ) or 0.0
            def _check(ind):
                g = evaluate_constraints(ind, bounds)
                return is_feasible(g, tol=resample_tol), g
            if cht_method(sim_type) == 'chocat':
                n_iter = strategy.resample_infeasibles(
                    population, feasibility_check=_check, max_iterations=5,
                )
                print(f"resample iterations this gen = {n_iter}")
                toolbox.logbook.bookshelf['resample_iterations'][gen] = n_iter
            else:  # 'arnold'
                strategy.apply_arnold_infeasibility(
                    population, feasibility_check=_check,
                )
                # Arnold has no resample concept; log 0 so the CSV
                # column stays uniform across sim_types.
                toolbox.logbook.bookshelf['resample_iterations'][gen] = 0

        # Retry logic for transient evaluation failures
        try:
            fitnesses = toolbox.map(toolbox.evaluate, population)
        except Exception:
            time.sleep(1)
            try:
                fitnesses = toolbox.map(toolbox.evaluate, population)
            except Exception:
                time.sleep(1)
                try:
                    fitnesses = toolbox.map(toolbox.evaluate, population)
                except Exception:
                    time.sleep(1)
                    fitnesses = toolbox.map(toolbox.evaluate, population)

        fixed = False
        for i, (ind, result) in enumerate(zip(population, fitnesses)):
            fit, g, g_al = result
            ind._g = g
            ind._g_al = g_al

            if fit is None:
                # Skipped by feasibility short-circuit.  Leave fitness
                # unset so DEAP's selection treats this individual as
                # invalid.  The CHT consumes ind._g to update covariance.
                # In CHT_AL mode, evaluate() already routed PITOT3 / SPARK
                # failures to fit=None, g_al=None — those individuals are
                # excluded from AL coefficient adaptation by construction.
                ind._feasible = False
                continue

            ind._feasible = True
            # Failure-sentinel detection in legacy 3-objective mode:
            # fit[0] is normalised delta_vs1 and == 1.0 means PITOT3 hit
            # its 3500 m/s sentinel.  In CHT_AL mode that path is already
            # caught inside evaluate() (returns fit=None), so skip the
            # check rather than indexing a 2-tuple at slot [0] which would
            # be hold_time, not delta_vs1.
            normalised_shock_speed = (
                fit[0] if not is_al_active(sim_type) else None
            )

            if normalised_shock_speed == 1.0:
                # PITOT3 / SPARK reported the failure sentinel even though
                # the candidate passed our feasibility check.  Recover by
                # substituting the strategy parent that this offspring
                # was generated from.
                #
                # The original recovery used `parents[ind.ind_number]`
                # (the previous generation's offspring batch), which
                # worked when every offspring was guaranteed feasible by
                # repair.  In CovarianceCHT mode infeasibles are kept
                # unevaluated, so an arbitrary previous-gen offspring may
                # have no fitness — DEAP then returns () for fitness.values
                # and the length-3 assignment below crashes.
                #
                # strategy.parents are guaranteed feasible by selection,
                # so they always have a length-3 fitness tuple.  Index
                # via ind._ps[1] (the donor parent recorded in generate()).
                p_idx = ind._ps[1] if hasattr(ind, "_ps") else None
                replacement = (
                    strategy.parents[p_idx]
                    if (p_idx is not None
                        and 0 <= p_idx < len(strategy.parents)
                        and strategy.parents[p_idx].fitness.valid)
                    else None
                )
                if replacement is None:
                    # Genuine corner case: we couldn't find a feasible
                    # replacement.  Fall back to marking this slot as
                    # infeasible so the CHT consumes its violation info
                    # and selection ignores it.
                    ind._feasible = False
                    continue

                new_fitness = replacement.fitness.values
                population[i] = replacement
                population[i].fitness.values = new_fitness
                population[i].ind_number = i
                population[i]._g = getattr(replacement, "_g", g)
                population[i]._feasible = True
                fixed = True
                fitness_history.append(new_fitness)
            else:
                ind.fitness.values = fit
                fitness_history.append(fit)

        # Per-generation feasibility count.  This is the diagnostic that
        # tells stagnation ("HV constant because zero offspring made it
        # through") apart from "rare improvements".  Recorded into the
        # logbook so the convergence_data.txt / summary writers can
        # surface it later.
        n_feasible = sum(
            1 for ind in population if getattr(ind, "_feasible", False)
        )
        print(f"feasible offspring this gen = {n_feasible} / {len(population)}")
        toolbox.logbook.bookshelf['feasible_offspring_count'][gen] = n_feasible

        # Snapshot the just-evaluated population for this generation.
        sigmas_per_slot     = list(sigmas_at_generate)
        parent_idx_per_slot = []
        for ind in population:
            if hasattr(ind, '_ps') and ind._ps[0] == "o":
                parent_idx_per_slot.append(ind._ps[1])
            else:
                parent_idx_per_slot.append(None)
        gen_snapshots[bookshelf_gen] = _make_snapshot(
            bookshelf_gen, population,
            sigmas_per_slot, parent_idx_per_slot, bounds,
        )

        toolbox.update(population)

        # Persist post-update strategy state (σ, psucc, lineage per slot)
        # for every sim_type.  Cheap (~one CSV row per gen) and lets us
        # see the σ trajectory directly — needed to test the
        # death-spiral hypothesis from the CHT analysis.
        _append_strategy_per_gen_row(
            folders["strategy_diagnostics"],
            gen=bookshelf_gen,
            strategy=strategy,
            max_mu=max_mu,
        )

        # ── M1: internal IPOP-style restart ───────────────────────────────
        if strategy.consume_restart_pending():
            print(f"\n{'='*40}\nM1 RESTART at gen {bookshelf_gen} "
                  f"(restart #{strategy._restart_count + 1})\n{'='*40}\n")
            _pop_incr = int(
                features["restart_on_sigma_collapse"]["pop_increment"]
            )
            fresh_untransformed = pop_init(_pop_incr)
            fresh_transformed   = variable_transformation(
                fresh_untransformed, bounds
            )
            for _slot in range(_pop_incr):
                _attempt = 0
                while not is_feasible(
                        evaluate_constraints(fresh_transformed[_slot], bounds)):
                    _attempt += 1
                    if _attempt > _MAX_RESAMPLE_ATTEMPTS:
                        raise RuntimeError(
                            f"M1 restart: could not generate a feasible "
                            f"fresh individual after {_MAX_RESAMPLE_ATTEMPTS} "
                            f"attempts."
                        )
                    fresh_transformed[_slot] = variable_transformation(
                        pop_init(1), bounds
                    )[0]
            fresh_inds = [Individual_cls(x) for x in fresh_transformed]
            for _fi, _ind in enumerate(fresh_inds):
                _ind.ind_number = i + _fi
                _ind.bounds     = bounds
                _ind.al_tol     = current_al_tol
                _ind.sim_type   = sim_type
                _ind.normalised = normalised
            # Evaluate fresh individuals
            fresh_fits = list(toolbox.map(toolbox.evaluate, fresh_inds))
            _restart_attempts = 0
            while any(
                f[0] is None
                or getattr(fresh_inds[_fi], "_pitot3_sentinel", False)
                or getattr(fresh_inds[_fi], "_spark_sentinel",  False)
                for _fi, f in enumerate(fresh_fits)
            ):
                _restart_attempts += 1
                if _restart_attempts > 20:
                    raise RuntimeError(
                        "M1 restart: could not obtain non-sentinel feasible "
                        "fresh individuals after 20 attempts."
                    )
                for _fi, (_fnd, _fresult) in enumerate(
                        zip(fresh_inds, fresh_fits)):
                    _ffit, _fg, _fg_al = _fresult
                    _fnd._g    = _fg
                    _fnd._g_al = _fg_al
                    _fnd._feasible = _ffit is not None
                    if _ffit is not None:
                        _fnd.fitness.values = _ffit
                    if is_al_active(sim_type):
                        _fnd._pitot3_sentinel = (
                            _fg_al is not None
                            and len(_fg_al) > 0
                            and (_fg_al[0] + current_al_tol) >= (_PITOT3_FAILURE_SENTINEL - 1.0)
                        )
                        _fnd._spark_sentinel = (
                            _ffit is not None
                            and all(abs(v - 1.0) < 1e-9 for v in _ffit)
                        )
                    else:
                        _fnd._pitot3_sentinel = False
                        _fnd._spark_sentinel  = False
                    if (not _fnd._feasible
                            or _fnd._pitot3_sentinel
                            or _fnd._spark_sentinel):
                        _new_ut = pop_init(1)
                        _new_tr = variable_transformation(_new_ut, bounds)[0]
                        _att2 = 0
                        while not is_feasible(
                                evaluate_constraints(_new_tr, bounds)):
                            _att2 += 1
                            if _att2 > _MAX_RESAMPLE_ATTEMPTS:
                                raise RuntimeError(
                                    "M1 restart padding: infeasible design.")
                            _new_tr = variable_transformation(
                                pop_init(1), bounds
                            )[0]
                        fresh_inds[_fi] = Individual_cls(_new_tr)
                        fresh_inds[_fi].ind_number = i + _fi
                        fresh_inds[_fi].bounds     = bounds
                        fresh_inds[_fi].al_tol     = current_al_tol
                        fresh_inds[_fi].sim_type   = sim_type
                        fresh_inds[_fi].normalised = normalised
                fresh_fits = list(toolbox.map(toolbox.evaluate, fresh_inds))
            # Final attribute assignment
            for _fi, (_fnd, _fresult) in enumerate(zip(fresh_inds, fresh_fits)):
                _ffit, _fg, _fg_al = _fresult
                _fnd._g    = _fg
                _fnd._g_al = _fg_al
                _fnd._feasible = _ffit is not None
                if _ffit is not None:
                    _fnd.fitness.values = _ffit
                if is_al_active(sim_type):
                    _fnd._pitot3_sentinel = (
                        _fg_al is not None
                        and len(_fg_al) > 0
                        and (_fg_al[0] + current_al_tol) >= (_PITOT3_FAILURE_SENTINEL - 1.0)
                    )
                    _fnd._spark_sentinel = (
                        _ffit is not None
                        and all(abs(v - 1.0) < 1e-9 for v in _ffit)
                    )
                else:
                    _fnd._pitot3_sentinel = False
                    _fnd._spark_sentinel  = False
            strategy.apply_internal_restart(fresh_inds, step_size)
            MU = strategy.mu
            LAMBDA = strategy.lambda_
            # Log restart event
            _restart_csv = folders["summary"] / "restart_events.csv"
            import csv as _csv_mod
            _restart_is_new = not _restart_csv.exists()
            with _restart_csv.open("a", newline="") as _rf:
                _rw = _csv_mod.DictWriter(
                    _rf,
                    fieldnames=["restart_idx", "gen",
                                "mu_before", "mu_after"]
                )
                if _restart_is_new:
                    _rw.writeheader()
                _rw.writerow({
                    "restart_idx": strategy._restart_count,
                    "gen":         bookshelf_gen,
                    "mu_before":   MU - _pop_incr,
                    "mu_after":    MU,
                })
            print(f"M1: pop size now {MU}")

        # ── Augmented Lagrangian coefficient update (CHT_AL only) ────
        # Order matters: this runs AFTER toolbox.update() so the proxy
        # we feed it is the post-selection parent set — i.e. the search
        # distribution that will seed the next generate() call.  This is
        # the closest analogue to the paper's "m^(t+1)" in MOO without
        # paying for an extra centroid evaluation.
        #
        # We also re-call init_al each gen until pycma's set_coefficients
        # decides it is fully initialised (sign_average balanced, see the
        # _initialized array) — pycma short-circuits idempotently once
        # the initial-conditions are met, so the cost is negligible.
        if is_al_active(sim_type):
            F_proxy, g_al_proxy = _cheap_al_proxy(strategy)
            if F_proxy is not None:
                # Refine bootstrap on additional g_al samples whilst not
                # yet fully initialised.  No-op once is_initialized=True.
                if not strategy.al.is_initialized:
                    F_pop_now = [sum(p.fitness.values) for p in strategy.parents
                                 if p.fitness.valid and getattr(p, "_g_al", None) is not None]
                    G_AL_now  = [p._g_al for p in strategy.parents
                                 if p.fitness.valid and getattr(p, "_g_al", None) is not None]
                    if F_pop_now:
                        strategy.init_al(F_pop_now, G_AL_now)
                strategy.update_al(F_proxy, g_al_proxy)
                print(f"AL: lam={strategy.al.lam}, mu={strategy.al.mu}, "
                      f"g_al_proxy={g_al_proxy}")

        # Drain Chocat CHT diagnostics for this generation.  Must happen
        # AFTER update(), because update()'s post-eval CHT pass also
        # appends to the buffer.  drain_and_persist clears the buffer in
        # place, so next generation starts clean.  Cheap when Chocat
        # isn't active (buffer is always empty).
        if cht_method(sim_type) == 'chocat':
            _cht_drain_and_persist(
                strategy,
                gen=bookshelf_gen,
                out_dir=folders["cht_diagnostics"],
                n_resample_iterations=toolbox.logbook.bookshelf['resample_iterations'][gen],
                n_infeasible_post_resample=(LAMBDA - n_feasible),
                n_lambda=LAMBDA,
            )

        # Drain Arnold CHT diagnostics for this generation.  Mirrors the
        # Chocat block above but writes to arnold_per_*.csv.
        if cht_method(sim_type) == 'arnold':
            _arnold_drain_and_persist(
                strategy,
                gen=bookshelf_gen,
                out_dir=folders["arnold_diagnostics"],
                n_infeasible=(LAMBDA - n_feasible),
                n_lambda=LAMBDA,
            )

        # Drain AL diagnostics (one row per generation) — only writes
        # anything when AL is active; for other sim_types the buffer is
        # empty and this is a no-op write of zero rows.
        if is_al_active(sim_type):
            _al_drain_and_persist(
                strategy,
                gen=bookshelf_gen,
                out_dir=folders["al_diagnostics"],
            )

        # Mark every snapshot row whose individual is still in
        # strategy.parents.  This catches both freshly-chosen offspring and
        # surviving older parents.
        _mark_chosen(gen_snapshots, strategy.parents)

        # HV is computed on the elitist parent set (size = mu, constant across
        # generations) rather than raw offspring. This removes the cardinality
        # noise that produced the discrete-plateau jumps in the convergence trace.
        parent_fitnesses = np.array([ind.fitness.values for ind in strategy.parents])
        hypervolume = pop_hypervolumes.compute(parent_fitnesses * -1)
        print(f'hypervolume = {hypervolume}')
        toolbox.logbook.bookshelf['hypervolume'][gen] = hypervolume

        # Periodic outputs every SAVE_INTERVAL generations.
        if bookshelf_gen % SAVE_INTERVAL == 0:
            _save_outputs(bookshelf_gen, gen_snapshots, fitness_history, MU, folders,
                          strategy=strategy)
            # Refresh the CHT diagnostic figure from the CSVs the drain
            # block has been appending to every generation.  The plot is
            # stateless (read-from-disk), so this is a pure side-effect
            # that doesn't need to share state with the main loop.
            # Both CovarianceCHT and CHT_AL drain CHT records (the box+
            # physical constraint handling is identical between them),
            # so both should regenerate the figure.  Arnold runs use a
            # different diagnostic schema; their plot is generated at
            # end-of-run from arnold_per_gen.csv (see plotting.py).
            if cht_method(sim_type) == 'chocat':
                _cht_plot(folders["cht_diagnostics"], current_gen=bookshelf_gen)
            elif cht_method(sim_type) == 'arnold':
                _arnold_plot(folders["arnold_diagnostics"])
            # Force a full GC pass: matplotlib's render buffers and the
            # transient numpy arrays in the CHT covariance update can
            # accumulate as uncollected garbage between gc cycles, and
            # over hundreds of generations that drift adds up to hundreds
            # of MB.  Doing this just after each save burst is the
            # natural pause point in the loop.
            gc.collect()
            # Surface RSS for this Python process so memory growth is
            # visible in real time, not only after an OOM.  ru_maxrss is
            # in KiB on Linux.
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"parent RSS (peak) = {rss_mb:.1f} MB")

    # ── Convergence data ──────────────────────────────────────────────────
    convergence_dir = folders["convergence"]
    summary_dir     = folders["summary"]

    with open(convergence_dir / "convergence_data.txt", "w") as file:
        file.write(f"Simulation Type = {sim_type}\n")
        file.write(f"Step Size = {step_size}\n")
        file.write(f'Pop Size = {pop_size}\n')
        file.write(f'p4 treatment = {p4_treatment}\n')
        file.write(f'Number of generations = {NGEN}\n')
        file.write(f'Current Time = {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
        file.write(f"Hypervolume per generation:\n")
        for gen in range(NGEN):
            file.write(f"Generation {gen + 1}: {toolbox.logbook.bookshelf['hypervolume'][gen]}\n")

        file.write(f"\nFeasible offspring per generation (out of {LAMBDA}):\n")
        for gen in range(NGEN):
            file.write(
                f"Generation {gen + 1}: "
                f"{toolbox.logbook.bookshelf['feasible_offspring_count'][gen]}\n"
            )

        if cht_method(sim_type) == 'chocat':
            file.write("\nCHT resample iterations per generation:\n")
            for gen in range(NGEN):
                file.write(
                    f"Generation {gen + 1}: "
                    f"{toolbox.logbook.bookshelf['resample_iterations'][gen]}\n"
                )

    # ── Hypervolume convergence plot ──────────────────────────────────────
    x_range = NGEN
    tick_interval = x_range / 5

    plt.figure(dpi=200)
    plt.title("Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.ylim((0, 1.1))

    gen_axis = list(range(1, NGEN + 1))
    avg_hv_list = [toolbox.logbook.bookshelf['hypervolume'][g - 1] for g in gen_axis]
    plt.plot(gen_axis, avg_hv_list)
    plt.savefig(convergence_dir / f"convergence_{sim_type}.png")
    plt.close()

    # ── Fixer count plot (non-Penalty runs only) ──────────────────────────
    fixer_count = []
    running_total = 0
    for entry in toolbox.logbook.bookshelf["fixer_count"].keys():
        if entry != '0':
            running_total += toolbox.logbook.bookshelf["fixer_count"][entry]
            fixer_count.append(running_total)

    e1 = time.time()
    toolbox.logbook.bookshelf["time taken"] = e1 - s1

    # Skip the "cumulative fixes" plot for any sim_type that doesn't run
    # the repair while-loop in generate().  Penalty was already excluded
    # by name; CovarianceCHT also bypasses repair (the CHT shrinkage
    # replaces it).  The empty-data check covers both cases and any
    # future no-repair sim_type without needing a name list.
    if fixer_count:
        generation = list(range(1, len(fixer_count) + 1))
        plt.figure(dpi=200)
        plt.title("Cumulative Number of Individuals Fixed")
        plt.xlabel("Generation")
        plt.ylabel("Number of Individuals Fixed")
        plt.plot(generation, fixer_count)
        plt.gca().xaxis.set_major_locator(MultipleLocator(tick_interval))
        plt.savefig(summary_dir / f"cma_es_mo_fpd_{sim_type}RunningTotal.png")
        plt.close()

    # ── Output summary text file ──────────────────────────────────────────
    # ideal_point / nadir_point come from _run_constants() at the top of
    # main(); they match the dimensionality of fitness_history's tuples
    # (3-D for legacy, 2-D for CHT_AL) so unnormalise_fitness works
    # without further branching.
    initial_dimensionalised_fitness = [
        unnormalise_fitness(ind, ideal_point, nadir_point)
        for ind in fitness_history[:MU]
    ]
    final_dimensionalised_fitness = [
        unnormalise_fitness(ind, ideal_point, nadir_point)
        for ind in fitness_history[-MU:]
    ]
    # Header used in the per-population objective tables below.  In
    # AL-active modes delta_vs1 has been moved out of fitness_history
    # (it is a constraint, not an objective) — so the header omits it.
    objectives_header = (
        "Driver Hold Time (ms) | Piston Impact Speed (m/s)"
        if is_al_active(sim_type)
        else "Residual of Shock Speed (m/s) | Driver Hold Time (ms) | Piston Impact Speed (m/s)"
    )
    # ms-conversion column index (hold_time): index 1 in 3-D, index 0 in 2-D.
    holdtime_col_idx = 0 if is_al_active(sim_type) else 1

    sig_figs = 6

    def _fmt_var(variable, index):
        """Format a single design variable for the output table."""
        col_widths = [
            'Percent Helium ', 'Driver Pressure (MPa) ', ' p4 (MPa) ',
            ' Throat Diameter (mm) ', ' Reservoir Pressure (MPa) ', ' Buffer Length (mm) '
        ]
        scale = [1, 1e-6, 1e-6, 1e3, 1e-6, 1e3]
        variable = round(variable * scale[index], sig_figs - str(variable * scale[index]).find('.'))
        white_space = int(np.round((len(col_widths[index]) - len(f'{variable}')) / 2))
        sep = '|' if index in [1, 3, 4] else ''
        return sep + ' ' * white_space + f'{variable}' + ' ' * white_space

    with open(summary_dir / "output.txt", "w") as file:
        file.write(f"Simulation Type = {sim_type}\n")
        file.write(f"step size = {experiment_type[2]}\n")
        file.write(f'pop size = {pop_size}\n')
        file.write(f'p4 treatment = {p4_treatment}\n')
        file.write(f"time taken = {toolbox.logbook.bookshelf['time taken']}\n")
        file.write(f'Number of generations = {NGEN}\n')
        file.write(f'Current Time = {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
        file.write(f"Number of individuals that produced no hold time = "
                   f"{toolbox.logbook.bookshelf['No. individuals that produced no hold time']}\n")
        file.write(f"Fixer Count = {running_total}\n")
        file.write('*' * 100 + '\n')
        file.write("INITIAL POPULATION:\n")
        file.write(
            "Percent Helium |Driver Pressure (MPa) | p4 (MPa) | Throat Diameter (mm) "
            "| Reservoir Pressure (MPa) | Buffer Length (mm)\n"
        )
        for ind in initial_population:
            string = ''.join(
                _fmt_var(v, idx)
                for idx, v in enumerate(variable_untransformation(ind, bounds))
            )
            file.write(string + '\n')

        file.write('\n')
        file.write(objectives_header + "\n")
        for ind in initial_dimensionalised_fitness:
            row = []
            for idx, obj in enumerate(ind):
                obj = round(obj, sig_figs - str(obj).find('.'))
                if idx == holdtime_col_idx:
                    obj *= 1e3   # seconds → milliseconds
                row.append(f'{obj}')
            file.write('  '.join(row) + '\n')

        file.write('\n' + '*' * 100 + '\n')
        file.write("FINAL POPULATION:\n")
        file.write(
            "Percent Helium | Driver Pressure (MPa) | p4 (MPa) | Throat Diameter (mm) "
            "| Reservoir Pressure (MPa) | Buffer length (mm)\n"
        )
        for ind in strategy.parents:
            string = ''.join(
                _fmt_var(v, idx)
                for idx, v in enumerate(variable_untransformation(ind, bounds))
            )
            file.write(string + '\n')

        file.write('\n')
        file.write(objectives_header + "\n")
        for ind in final_dimensionalised_fitness:
            row = []
            for idx, obj in enumerate(ind):
                obj = round(obj, sig_figs - str(obj).find('.'))
                if idx == holdtime_col_idx:
                    obj *= 1e3   # seconds → milliseconds
                row.append(f'{obj}')
            file.write('  '.join(row) + '\n')

    # Final save catches any in-flight snapshots (e.g. when NGEN is not a
    # multiple of SAVE_INTERVAL) and re-writes earlier CSVs with any newly
    # available chosen / offspring data.
    _save_outputs(toolbox.logbook.bookshelf['generation'],
                  gen_snapshots, fitness_history, MU, folders,
                  strategy=strategy)

    # ── Flush archive to CSV ─────────────────────────────────────────────
    # Archive exists for any AL-active sim_type (CHT_AL / ArnoldCHT_AL).
    if is_al_active(sim_type):
        strategy.flush_archive_to_csv(summary_dir)

    # ── M2 end-of-round-1 hook ───────────────────────────────────────────
    if (isinstance(features, dict)
            and features.get("restart_round_two")
            and not _is_round_two):
        seed_csv_path = summary_dir / "round2_seed.csv"
        _write_round2_seed_csv(seed_csv_path, strategy)
        (run_dir / "ROUND2_SEED_READY").touch()
        print(f"M2: round-1 seed written to {seed_csv_path}")

    print('\n\nEND OF SIM')
    print('*' * 60)
    print('\n\n')

    return strategy.parents


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _config_path = pathlib.Path(__file__).parent.parent / "config" / "experiments.yaml"
    with open(_config_path) as _f:
        _config = yaml.safe_load(_f)

    # 9-tuple: (sim_type, pop_size, step_size, p4_treatment, al_tol,
    #           cht_gamma, features_dict, n_gen, lhs_seed).
    # ``n_gen``     : number of generations (default 500 for back-compat).
    # ``al_tol``    : (CHT_AL only) constraint tolerance ε (m/s).
    # ``cht_gamma`` : (CovarianceCHT / CHT_AL) shrinkage strength; None means
    #                 "use the strategy's dimension-dependent default".
    # ``features``  : dict of optional anti-degeneration feature toggles
    #                 (eigenvalue floor, lam floor, etc.).  See the YAML
    #                 header comment for the full menu.  Empty dict =
    #                 baseline behaviour (no features enabled).
    # ``lhs_seed``  : int | None.  Seeds the numpy Generator that drives every
    #                 LatinHypercube call in this run (initial pop, M1
    #                 restart, M2 padding).  None ⇒ fresh OS entropy.
    experiment_types = [
        (
            exp["sim_type"],
            exp["pop_size"],
            exp["step_size"],
            exp["p4_treatment"],
            exp.get("al_tol", 100.0),
            exp.get("cht_gamma", None),
            exp.get("features", {}) or {},
            exp.get("n_gen", 500),
            exp.get("lhs_seed", None),
        )
        for exp in _config["experiments"]
    ]

    _rt_seed = None
    _rt_r1   = None
    if "--round-two" in sys.argv:
        _rt_seed = sys.argv[sys.argv.index("--round-two") + 1]
    if "--round-one-dir" in sys.argv:
        _rt_r1 = sys.argv[sys.argv.index("--round-one-dir") + 1]

    if "--experiment-index" in sys.argv:
        # ── Worker mode ───────────────────────────────────────────────────
        # This branch runs when the dispatcher below launched us as a child
        # subprocess.  We execute exactly one experiment and then exit,
        # letting the OS reclaim every byte of RAM the run accumulated.
        idx = int(sys.argv[sys.argv.index("--experiment-index") + 1])
        solutions = main(experiment_types[idx],
                         round_two_seed=_rt_seed,
                         round_one_dir=_rt_r1)

    else:
        # ── Dispatcher mode ───────────────────────────────────────────────
        # Run each experiment in a fresh Python interpreter so that memory
        # (PITOT3 Lua VMs, matplotlib caches, gdtk gas-model objects, the
        # multiprocessing worker pool) is fully reclaimed between experiments.
        # Without this, successive runs in the same process accumulate RSS
        # until the kernel OOM-kills the parent (~3× slowdown by run 4).
        #
        # sys.executable   — same interpreter that is running this script,
        #                    so virtual-environment / conda paths are preserved.
        # Path(__file__).resolve() — absolute path to main.py, works
        #                    regardless of the working directory the user
        #                    invoked us from.
        _script = pathlib.Path(__file__).resolve()

        for i, experiment_type in enumerate(experiment_types):
            print(f"\n{'=' * 60}")
            print(f"Experiment {i + 1} / {len(experiment_types)}: {experiment_type}")
            print(f"{'=' * 60}\n")
            result = subprocess.run(
                [sys.executable, str(_script), "--experiment-index", str(i)],
                check=False,           # don't raise — report and continue
            )
            if result.returncode != 0:
                print(
                    f"\nWARNING: experiment {i + 1} exited with code "
                    f"{result.returncode}.  Continuing with the next one.\n"
                )
            # M2: check for round-2 marker and dispatch if present.
            features_dict = experiment_type[6] or {}
            if features_dict.get("restart_round_two"):
                _launch_round_two_if_pending(i, experiment_type, _script)

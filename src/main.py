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
from algorithm.cmaes     import StrategyMultiObjective
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
from problem.evaluate    import evaluate, set_logbook
from problem.feasibility import evaluate_constraints, is_feasible
from plotting            import (
    plot_objective_space,
    plot_objective_space_3d,
    plot_objective_space_heatmap,
    plot_holdtime_impactspeed_2d,
)
from results_io          import (
    setup_run_directory,
    setup_subfolders,
    write_population_csv,
)
from cht_diagnostics     import drain_and_persist as _cht_drain_and_persist
from cht_diagnostics     import drain_and_persist_al as _al_drain_and_persist
from cht_diagnostics     import plot_cht_diagnostics as _cht_plot
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
    # for every run; only populated when sim_type == 'CovarianceCHT'.
    "cht_diagnostics",
    # Per-generation strategy state (σ and psucc per parent slot).
    # Populated for ALL sim_types so the σ death-spiral hypothesis
    # can be verified independently of constraint-handling choice.
    "strategy_diagnostics",
]

# CHT_AL output structure: 2-objective (no 3-D Pareto plot, no
# delta_vs1-vs-* plots — delta_vs1 is now a constraint), plus a new
# al_diagnostics directory mirroring cht_diagnostics for AL telemetry.
RESULTS_CATEGORY_AL = "al_cht_recomb"
RUN_PREFIX_AL       = "al_cht"
OUTPUT_FOLDERS_AL = [
    "pareto_holdtime_impactspeed",
    "population",
    "convergence",
    "summary",
    "cht_diagnostics",
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
    if sim_type == 'CHT_AL':
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


def _save_outputs(bookshelf_gen, gen_snapshots, fitness_history, MU, folders):
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


def _append_strategy_per_gen_row(out_dir, gen, strategy):
    """Append one row of σ, psucc and lineage state to strategy_per_gen.csv.

    Called once per generation immediately after toolbox.update(), so
    self.sigmas / self.psucc reflect the post-update parent set (i.e.
    the parents that will seed the *next* generate() call).

    Schema: generation, mu, summary stats (mean/min/max for σ and
    psucc), then per-slot lineage_<i>, sigma_<i>, psucc_<i>.  μ is
    constant within a run, so the header is fixed at first write.
    """
    import csv
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "strategy_per_gen.csv"

    sigmas = list(strategy.sigmas)
    psucc  = list(strategy.psucc)
    mu     = len(sigmas)
    lineage_ids = [
        getattr(p, "_lineage_id", None) for p in strategy.parents
    ]

    fieldnames = (
        ["generation", "mu",
         "mean_sigma", "min_sigma", "max_sigma",
         "mean_psucc", "min_psucc", "max_psucc"]
        + [f"lineage_{i}" for i in range(mu)]
        + [f"sigma_{i}"   for i in range(mu)]
        + [f"psucc_{i}"   for i in range(mu)]
    )

    row = {
        "generation": gen,
        "mu":         mu,
        "mean_sigma": float(np.mean(sigmas)),
        "min_sigma":  float(np.min(sigmas)),
        "max_sigma":  float(np.max(sigmas)),
        "mean_psucc": float(np.mean(psucc)),
        "min_psucc":  float(np.min(psucc)),
        "max_psucc":  float(np.max(psucc)),
    }
    for i in range(mu):
        row[f"lineage_{i}"] = lineage_ids[i]
        row[f"sigma_{i}"]   = float(sigmas[i])
        row[f"psucc_{i}"]   = float(psucc[i])

    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Main evolution loop
# ─────────────────────────────────────────────────────────────────────────────

def main(experiment_type):
    s1 = time.time()

    # ── Experiment parameters ─────────────────────────────────────────────
    N           = 6
    pop_size    = experiment_type[1]
    MU, LAMBDA  = pop_size, pop_size
    NGEN        = 500
    sim_type    = experiment_type[0]
    p4_treatment = experiment_type[3]
    step_size   = experiment_type[2]
    # AL constraint tolerance (m/s on delta_vs1).  Only consumed when
    # sim_type == 'CHT_AL'; legacy sim_types ignore it.  Default 100 m/s
    # per the user-confirmed setting.  experiment_type may be a 4-tuple
    # for legacy YAML entries; fall back to the default in that case.
    al_tol = experiment_type[4] if len(experiment_type) > 4 else 100.0

    print(f"Step Size = {step_size}")
    print(f'Pop Size = {pop_size}\n')
    if sim_type == 'CHT_AL':
        print(f'AL tolerance (delta_vs1 ≤): {al_tol} m/s')

    # ── Output directory layout ───────────────────────────────────────────
    # Per-sim_type constants: legacy modes write a 3-objective tree;
    # CHT_AL writes a 2-objective tree with an extra al_diagnostics dir.
    (results_category, run_prefix, output_folders,
     Individual_cls, ideal_point, nadir_point) = _run_constants(sim_type)
    run_dir = setup_run_directory(results_category, run_prefix)
    folders = setup_subfolders(run_dir, output_folders)
    print(f"Run directory: {run_dir}")

    # Per-mode HV calculator.  Rebinds the global ``pop_hypervolumes``
    # name to a local 2-D / 3-D variant — the global lookup at the
    # bottom of the loop will hit this local instead.  No global access
    # leaks because no other site in main.py reads pop_hypervolumes.
    pop_hypervolumes = HyperVolume(
        np.zeros(2 if sim_type == 'CHT_AL' else 3)
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
    def pop_init(MU):
        percent_he_list     = np.random.uniform(he_lower,            he_upper,            (MU, 1))
        D_throat_list       = np.random.uniform(D_throat_lower,      D_throat_upper,      (MU, 1))
        driver_p_list       = np.random.uniform(driver_p_lower,      driver_p_upper,      (MU, 1))
        buffer_length_list  = np.random.uniform(buffer_length_lower, buffer_length_upper, (MU, 1))

        p4_list = np.zeros_like(percent_he_list)
        for i in range(MU):
            if 1190.63 * driver_p_list[i] < p4_upper:
                p4_list[i] = np.random.uniform(14.62 * driver_p_list[i], 1190.63 * driver_p_list[i])
            else:
                p4_list[i] = np.random.uniform(14.62 * driver_p_list[i], p4_upper)

        reservoir_p_list = [np.random.uniform(driver_p_list[i], reservoir_upper) for i in range(MU)]

        return [
            [percent_he_list[i][0], driver_p_list[i][0], p4_list[i][0],
             D_throat_list[i][0],   reservoir_p_list[i][0], buffer_length_list[i][0]]
            for i in range(MU)
        ]

    i = 0
    init_pop_untransformed = pop_init(MU)
    init_pop_transformed   = variable_transformation(init_pop_untransformed, bounds)

    # The strategy assumes every parent it starts with is feasible — Pareto
    # selection later relies on every parent having a valid fitness, and
    # length(self.parents) must equal mu so per-parent state arrays don't
    # shrink and break later generate() calls.  pop_init enforces some
    # constraints by construction (driver_p < reservoir_p, p4 within
    # bounds) but not the compression-ratio range, so a small fraction
    # (~1%) of initial individuals fail the feasibility check.  Resample
    # any infeasible slots one at a time until the whole pop is feasible.
    from problem.feasibility import evaluate_constraints, is_feasible
    _MAX_RESAMPLE_ATTEMPTS = 1000
    for slot in range(MU):
        attempt = 0
        while not is_feasible(evaluate_constraints(init_pop_transformed[slot], bounds)):
            attempt += 1
            if attempt > _MAX_RESAMPLE_ATTEMPTS:
                raise RuntimeError(
                    f"Could not generate a feasible initial individual for slot "
                    f"{slot} after {_MAX_RESAMPLE_ATTEMPTS} attempts.  Check "
                    f"that pop_init's sampling ranges are consistent with "
                    f"problem.feasibility.evaluate_constraints."
                )
            # pop_init(1) returns a length-1 list; replace just this slot.
            init_pop_transformed[slot] = variable_transformation(pop_init(1), bounds)[0]

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

    for ind in population:
        ind.sim_type   = sim_type
        ind.normalised = normalised
        fit, g, g_al = toolbox.evaluate(ind)
        ind._g = g
        ind._g_al = g_al
        ind._feasible = fit is not None
        if ind._feasible:
            ind.fitness.values = fit

    # ── Strategy and multiprocessing setup ────────────────────────────────
    strategy = StrategyMultiObjective(
        population, sigma=step_size,
        mu=MU, lambda_=LAMBDA,
        sim_type=sim_type, p4_treatment=p4_treatment,
        bounds=bounds,
        al_tol=al_tol,                # consumed only when sim_type=='CHT_AL'
        logbook=toolbox.logbook,      # injected — no global access inside cmaes.py
    )
    toolbox.register("generate", strategy.generate, Individual_cls)
    toolbox.register("update",   strategy.update)

    # Bootstrap AL coefficients from the initial population's data.
    # Idempotent: pycma's set_coefficients short-circuits once
    # _initialized is fully True; we still call it again every generation
    # below until it is, to refine on additional samples.
    if sim_type == 'CHT_AL':
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
        for ind in population:
            ind.normalised = normalised
            ind.ind_number = i
            ind.bounds     = bounds
            i += 1

        # CHT-and-resample loop (Chocat 2015 Algorithm 3 step 3-2): for
        # CovarianceCHT and CHT_AL, infeasible offspring drive a covariance
        # shrinkage of each parent's Cholesky factor and are then
        # resampled from the tighter distribution.  Cheap because the
        # feasibility check does NOT call SPARK / PITOT3 — it only
        # evaluates the constraint vector via problem.feasibility.
        # Mutates population in place and tags every Individual with
        # ._g and ._feasible so the post-eval loop and update()'s
        # post-resample CHT can both consume them.
        #
        # In CHT_AL the resample only operates on box+physical g (the
        # 18-element vector); delta_vs1 is handled separately by the AL
        # in selection, not via CHT shrinkage.
        if sim_type in ('CovarianceCHT', 'CHT_AL'):
            def _check(ind):
                g = evaluate_constraints(ind, bounds)
                return is_feasible(g), g
            n_iter = strategy.resample_infeasibles(
                population, feasibility_check=_check, max_iterations=5,
            )
            print(f"resample iterations this gen = {n_iter}")
            toolbox.logbook.bookshelf['resample_iterations'][gen] = n_iter

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
                fit[0] if sim_type != 'CHT_AL' else None
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
        )

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
        if sim_type == 'CHT_AL':
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

        # Drain CHT diagnostics for this generation.  Must happen AFTER
        # update(), because update()'s post-eval CHT pass also appends to
        # the buffer.  drain_and_persist clears the buffer in place, so
        # next generation starts clean.  Cheap when sim_type isn't
        # CovarianceCHT or CHT_AL (buffer is always empty).
        if sim_type in ('CovarianceCHT', 'CHT_AL'):
            _cht_drain_and_persist(
                strategy,
                gen=bookshelf_gen,
                out_dir=folders["cht_diagnostics"],
                n_resample_iterations=toolbox.logbook.bookshelf['resample_iterations'][gen],
                n_infeasible_post_resample=(LAMBDA - n_feasible),
                n_lambda=LAMBDA,
            )

        # Drain AL diagnostics (one row per generation) — only writes
        # anything when sim_type == 'CHT_AL'; for other sim_types the
        # buffer is empty and this is a no-op write of zero rows.
        if sim_type == 'CHT_AL':
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
            _save_outputs(bookshelf_gen, gen_snapshots, fitness_history, MU, folders)
            # Refresh the CHT diagnostic figure from the CSVs the drain
            # block has been appending to every generation.  The plot is
            # stateless (read-from-disk), so this is a pure side-effect
            # that doesn't need to share state with the main loop.
            if sim_type == 'CovarianceCHT':
                _cht_plot(folders["cht_diagnostics"], current_gen=bookshelf_gen)
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

        if sim_type == 'CovarianceCHT':
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
    # CHT_AL mode delta_vs1 has been moved out of fitness_history (it
    # is a constraint, not an objective) — so the header omits it.
    objectives_header = (
        "Driver Hold Time (ms) | Piston Impact Speed (m/s)"
        if sim_type == 'CHT_AL'
        else "Residual of Shock Speed (m/s) | Driver Hold Time (ms) | Piston Impact Speed (m/s)"
    )
    # ms-conversion column index (hold_time): index 1 in 3-D, index 0 in 2-D.
    holdtime_col_idx = 0 if sim_type == 'CHT_AL' else 1

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
                  gen_snapshots, fitness_history, MU, folders)

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

    # 5-tuple: (sim_type, pop_size, step_size, p4_treatment, al_tol).
    # ``al_tol`` is consumed only by the CHT_AL sim_type; legacy entries
    # without the field default to 100 m/s and ignore it.
    experiment_types = [
        (
            exp["sim_type"],
            exp["pop_size"],
            exp["step_size"],
            exp["p4_treatment"],
            exp.get("al_tol", 100.0),
        )
        for exp in _config["experiments"]
    ]

    if "--experiment-index" in sys.argv:
        # ── Worker mode ───────────────────────────────────────────────────
        # This branch runs when the dispatcher below launched us as a child
        # subprocess.  We execute exactly one experiment and then exit,
        # letting the OS reclaim every byte of RAM the run accumulated.
        idx = int(sys.argv[sys.argv.index("--experiment-index") + 1])
        solutions = main(experiment_types[idx])

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

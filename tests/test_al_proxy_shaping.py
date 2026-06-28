"""
Unit tests for B4: FRONT-ALAMO-aligned shaping of the AL update proxy.

Two independent knobs, both defaulting off (legacy = mean over all
surviving parents):

  * ``al_proxy_front_only`` — summarise only the first non-dominated front
    (w.r.t. the AL-augmented fitness) instead of all parents.
  * ``al_proxy_g_quantile`` — aggregate g_al with an upper quantile (the
    noise-robust surrogate for FRONT-ALAMO's worst-violation) instead of
    the mean.

These exercise ``StrategyMultiObjective.al_first_front`` and
``main._cheap_al_proxy`` in isolation — no L1d / heavy evaluator.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import main
from main import _cheap_al_proxy
from deap import creator
from algorithm.cmaes import StrategyMultiObjective
from problem.config import BOUNDS
from problem.feasibility import evaluate_constraints, is_feasible
from problem.transforms import variable_transformation


def _feasible_norm_pop(n, rng):
    """Return n feasible normalised [1,2]^6 individuals (as plain lists)."""
    from init_population_l1d import lhs_unit_to_physical
    out = []
    while len(out) < n:
        phys = lhs_unit_to_physical(rng.random(6))
        x_norm = variable_transformation([phys], BOUNDS)[0]
        if is_feasible(evaluate_constraints(x_norm, BOUNDS)):
            out.append([float(v) for v in x_norm])
    return out


def _make_parents(x_norms, fitness_values, g_als, sentinel=None):
    """Build Individual2D parents with explicit fitness + g_al."""
    parents = []
    for i, x in enumerate(x_norms):
        ind = creator.Individual2D(x)
        ind.ind_number = i
        ind.bounds = BOUNDS
        ind.sim_type = 'ArnoldCHT_AL'
        ind.al_tol = 100.0
        ind.fitness.values = tuple(fitness_values[i])
        ind._feasible = True
        ind._g = evaluate_constraints(x, BOUNDS)
        ind._g_al = np.array(g_als[i], dtype=float)
        ind._raw_delta_vs1 = float(g_als[i][0]) + ind.al_tol
        if sentinel and sentinel[i]:
            ind._pitot3_sentinel = True
        parents.append(ind)
    return parents


def _make_strategy(parents, features=None):
    n_constraints = len(evaluate_constraints(parents[0], BOUNDS))
    return StrategyMultiObjective(
        parents, sigma=0.1, mu=len(parents), lambda_=len(parents),
        sim_type='ArnoldCHT_AL', p4_treatment=None, bounds=BOUNDS,
        al_tol=100.0, n_constraints=n_constraints, features=features or {},
    )


# ── al_first_front ──────────────────────────────────────────────────────

def test_first_front_raw_objectives_when_penalty_zero():
    """With AL not bootstrapped every penalty is 0, so al_first_front is the
    raw-objective first front.  p2 is dominated by both p0 and p1."""
    rng = np.random.default_rng(0)
    x = _feasible_norm_pop(3, rng)
    fits = [(0.1, 0.5), (0.5, 0.1), (0.9, 0.9)]   # minimise both → p2 dominated
    parents = _make_parents(x, fits, g_als=[[10.0], [20.0], [1000.0]])
    strat = _make_strategy(parents)

    front = strat.al_first_front(parents)
    assert {p.ind_number for p in front} == {0, 1}


def test_first_front_uses_al_augmented_fitness():
    """The front must be defined on the AL-augmented fitness (the same
    criterion _select uses), not the raw one.  Here all three points are
    mutually non-dominated raw, but a large penalty on p1 (via its g_al)
    pushes it off the augmented front."""
    rng = np.random.default_rng(1)
    x = _feasible_norm_pop(3, rng)
    fits = [(0.1, 0.5), (0.3, 0.3), (0.5, 0.1)]   # trade-off: all non-dom raw
    parents = _make_parents(x, fits, g_als=[[0.0], [1000.0], [0.0]])
    strat = _make_strategy(parents)

    # Force a per-individual penalty equal to g_al (shadows the method).
    strat._al_penalty = lambda g: 0.0 if g is None else float(np.asarray(g)[0])

    front = strat.al_first_front(parents)
    assert {p.ind_number for p in front} == {0, 2}   # p1 penalised off front


# ── _cheap_al_proxy: aggregator ─────────────────────────────────────────

def test_proxy_mean_is_legacy_default():
    rng = np.random.default_rng(2)
    x = _feasible_norm_pop(4, rng)
    g = [[10.0], [20.0], [30.0], [100.0]]
    parents = _make_parents(x, [(0.5, 0.5)] * 4, g)
    strat = _make_strategy(parents)                 # no B4 flags

    F, g_proxy, stats = _cheap_al_proxy(strat)
    assert np.isclose(g_proxy[0], np.mean([10, 20, 30, 100]))
    assert stats["n_proxy_set"] == 4
    assert stats["proxy_front_only"] is False
    assert stats["proxy_g_quantile"] is None
    # Distribution stats always span the full clean set.
    assert stats["g_al_max"] == 100.0


def test_proxy_quantile_aggregator():
    rng = np.random.default_rng(3)
    x = _feasible_norm_pop(4, rng)
    g = [[10.0], [20.0], [30.0], [100.0]]
    parents = _make_parents(x, [(0.5, 0.5)] * 4, g)
    strat = _make_strategy(parents, {'al_proxy_g_quantile': 0.9})

    F, g_proxy, stats = _cheap_al_proxy(strat)
    expected = np.quantile([10.0, 20.0, 30.0, 100.0], 0.9)
    assert np.isclose(g_proxy[0], expected)
    assert not np.isclose(g_proxy[0], np.mean([10, 20, 30, 100]))   # ≠ mean
    assert stats["proxy_g_quantile"] == 0.9


# ── _cheap_al_proxy: set restriction ────────────────────────────────────

def test_proxy_front_only_restricts_set():
    """front_only drops the dominated, high-g_al p2 from the aggregate."""
    rng = np.random.default_rng(4)
    x = _feasible_norm_pop(3, rng)
    fits = [(0.1, 0.5), (0.5, 0.1), (0.9, 0.9)]    # p2 dominated
    parents = _make_parents(x, fits, g_als=[[10.0], [20.0], [1000.0]])
    strat = _make_strategy(parents, {'al_proxy_front_only': True})

    F, g_proxy, stats = _cheap_al_proxy(strat)
    assert stats["n_proxy_set"] == 2                # only the front
    assert stats["proxy_front_only"] is True
    assert np.isclose(g_proxy[0], np.mean([10.0, 20.0]))   # p2's 1000 excluded
    # but the full-set spread is still reported
    assert stats["g_al_max"] == 1000.0
    assert stats["n_feasible_parents"] == 3


def test_proxy_front_only_plus_quantile():
    rng = np.random.default_rng(5)
    x = _feasible_norm_pop(3, rng)
    fits = [(0.1, 0.5), (0.5, 0.1), (0.9, 0.9)]
    parents = _make_parents(x, fits, g_als=[[10.0], [20.0], [1000.0]])
    strat = _make_strategy(
        parents, {'al_proxy_front_only': True, 'al_proxy_g_quantile': 0.9})

    F, g_proxy, stats = _cheap_al_proxy(strat)
    assert stats["n_proxy_set"] == 2
    assert np.isclose(g_proxy[0], np.quantile([10.0, 20.0], 0.9))


# ── sentinel handling unchanged ─────────────────────────────────────────

def test_proxy_excludes_sentinels():
    rng = np.random.default_rng(6)
    x = _feasible_norm_pop(3, rng)
    g = [[10.0], [20.0], [3400.0]]                  # p2 is the sentinel
    parents = _make_parents(x, [(0.5, 0.5)] * 3, g,
                            sentinel=[False, False, True])
    strat = _make_strategy(parents)

    F, g_proxy, stats = _cheap_al_proxy(strat)
    assert stats["n_feasible_parents"] == 2         # sentinel filtered out
    assert np.isclose(g_proxy[0], np.mean([10.0, 20.0]))
    assert stats["g_al_max"] == 20.0                # 3400 never enters

# tests/test_lhs_init.py
#
# Unit tests for src/problem/sampling.py — Latin Hypercube sampling
# over the X2 design space, plus the oversample-and-filter helper
# used to seed the initial population.
#
# Tests are isolated:
#   * Shape / bound / conditional-bound contracts on lhs_sample.
#   * Marginal stratification: one sample per axis quantile bin
#     for the unconstrained axes (LHS guarantee).
#   * Reproducibility under a fixed numpy Generator seed.
#   * Discrepancy: LHS centred L2 discrepancy is strictly lower
#     than i.i.d. uniform on average (the empirical reason for
#     switching).
#   * draw_structurally_feasible_lhs: returns exactly n designs,
#     all passing evaluate_constraints / is_feasible.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from problem.sampling import lhs_sample, draw_structurally_feasible_lhs
from problem.config import (
    BOUNDS,
    he_lower, he_upper,
    driver_p_lower, driver_p_upper,
    p4_upper,
    D_throat_lower, D_throat_upper,
    reservoir_upper,
    buffer_length_lower, buffer_length_upper,
)
from problem.feasibility import evaluate_constraints, is_feasible


# Column order in the sample tuple:
PCT_HE, DRV_P, P4, D_THROAT, RES_P, BUF_LEN = 0, 1, 2, 3, 4, 5


def _seeded_rng(seed=12345):
    return np.random.default_rng(seed)


# ─────────────────────────────────────────────────────────────────
# lhs_sample: shape and basic bound contracts
# ─────────────────────────────────────────────────────────────────

def test_lhs_sample_zero_returns_empty():
    assert lhs_sample(0, rng=_seeded_rng()) == []


def test_lhs_sample_shape_for_typical_pop():
    samples = lhs_sample(24, rng=_seeded_rng())
    assert len(samples) == 24
    for x in samples:
        assert len(x) == 6


def test_lhs_sample_unconditional_axes_within_bounds():
    """The 4 unconditional axes (percent_he, driver_p, D_throat,
    buffer_length) must lie within their fixed physical bounds."""
    samples = lhs_sample(50, rng=_seeded_rng())
    for x in samples:
        assert he_lower            <= x[PCT_HE]   <= he_upper
        assert driver_p_lower      <= x[DRV_P]    <= driver_p_upper
        assert D_throat_lower      <= x[D_THROAT] <= D_throat_upper
        assert buffer_length_lower <= x[BUF_LEN]  <= buffer_length_upper


def test_lhs_sample_p4_in_conditional_band():
    """p4 must lie in [14.62·driver_p, min(1190.63·driver_p, p4_upper)]
    for every sample — the same conditional band the legacy pop_init
    produced."""
    samples = lhs_sample(50, rng=_seeded_rng())
    for x in samples:
        drv_p = x[DRV_P]
        p4_lo = 14.62 * drv_p
        p4_hi = min(1190.63 * drv_p, p4_upper)
        # Allow a generous fp tolerance — LHS u may be ~0 or ~1.
        assert p4_lo - 1e-6 <= x[P4] <= p4_hi + 1e-6


def test_lhs_sample_reservoir_above_driver():
    """reservoir_p must be >= driver_p and <= reservoir_upper."""
    samples = lhs_sample(50, rng=_seeded_rng())
    for x in samples:
        assert x[DRV_P] - 1e-6 <= x[RES_P] <= reservoir_upper + 1e-6


# ─────────────────────────────────────────────────────────────────
# Marginal stratification: the headline LHS guarantee
# ─────────────────────────────────────────────────────────────────

def test_lhs_sample_marginal_stratification_unconditional_axes():
    """For each unconditional axis, sort the n samples and assert that
    exactly one sample falls into each of the n equal-probability
    strata.  This is the LHS contract: project onto any single axis,
    get one point per quantile bin.

    Conditional axes (p4, reservoir_p) are excluded — they are
    LHS-stratified in the latent [0,1] cube but mapped through a
    per-sample range, so the physical-units marginal is not
    necessarily uniform across strata."""
    n = 30
    samples = lhs_sample(n, rng=_seeded_rng())

    axes = {
        "percent_he":    (PCT_HE,   he_lower,            he_upper),
        "driver_p":      (DRV_P,    driver_p_lower,      driver_p_upper),
        "D_throat":      (D_THROAT, D_throat_lower,      D_throat_upper),
        "buffer_length": (BUF_LEN,  buffer_length_lower, buffer_length_upper),
    }
    for name, (idx, lo, hi) in axes.items():
        values = sorted(x[idx] for x in samples)
        # Stratum k is (lo + k·w, lo + (k+1)·w) for w = (hi-lo)/n.
        w = (hi - lo) / n
        for k, v in enumerate(values):
            stratum_lo = lo + k * w
            stratum_hi = lo + (k + 1) * w
            # Allow tiny fp slack at the boundaries.
            assert stratum_lo - 1e-9 <= v <= stratum_hi + 1e-9, (
                f"axis {name}: sorted value at rank {k} = {v} not in "
                f"stratum [{stratum_lo}, {stratum_hi}]"
            )


# ─────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────

def test_lhs_sample_reproducible_under_fixed_seed():
    """Two runs with the same int seed must produce bit-identical
    samples.  This is the property lhs_seed in experiments.yaml
    relies on for ablation reproducibility."""
    a = lhs_sample(24, rng=np.random.default_rng(7))
    b = lhs_sample(24, rng=np.random.default_rng(7))
    for xa, xb in zip(a, b):
        for va, vb in zip(xa, xb):
            assert va == vb


def test_lhs_sample_differs_with_different_seeds():
    """Sanity check: different seeds produce different draws.  Without
    this the reproducibility test could silently pass on a broken
    sampler that always returns the same values."""
    a = lhs_sample(24, rng=np.random.default_rng(7))
    b = lhs_sample(24, rng=np.random.default_rng(8))
    assert any(va != vb for xa, xb in zip(a, b) for va, vb in zip(xa, xb))


# ─────────────────────────────────────────────────────────────────
# Discrepancy: LHS should beat i.i.d. uniform on average
# ─────────────────────────────────────────────────────────────────

def test_lhs_lower_centred_discrepancy_than_uniform():
    """Compute centred L2 discrepancy (scipy.stats.qmc.discrepancy)
    on the latent [0,1]^6 cube for LHS and for i.i.d. uniform, each
    averaged over 20 seeds.  LHS should be strictly lower on average
    — this is the empirical justification for switching."""
    from scipy.stats.qmc import LatinHypercube, discrepancy
    n_seeds = 20
    n = 24
    d = 6

    lhs_vals, unif_vals = [], []
    for s in range(n_seeds):
        rng_lhs  = np.random.default_rng(1000 + s)
        rng_unif = np.random.default_rng(2000 + s)
        u_lhs    = LatinHypercube(d=d, optimization="random-cd",
                                  seed=rng_lhs).random(n)
        u_unif   = rng_unif.random((n, d))
        lhs_vals.append(discrepancy(u_lhs, method="CD"))
        unif_vals.append(discrepancy(u_unif, method="CD"))

    assert np.mean(lhs_vals) < np.mean(unif_vals), (
        f"LHS mean CD = {np.mean(lhs_vals):.4f} is not lower than "
        f"uniform mean CD = {np.mean(unif_vals):.4f}"
    )


# ─────────────────────────────────────────────────────────────────
# draw_structurally_feasible_lhs
# ─────────────────────────────────────────────────────────────────

def test_draw_structurally_feasible_returns_exactly_n_feasible():
    """The oversample-and-filter helper must return exactly n designs,
    every one of which passes is_feasible(evaluate_constraints(...))."""
    n = 24
    rng = _seeded_rng()
    def pop_init_fn(k):
        return lhs_sample(k, rng=rng)
    designs = draw_structurally_feasible_lhs(
        n, pop_init_fn, BOUNDS, oversample=4, max_doublings=3,
    )
    assert len(designs) == n
    for x in designs:
        assert is_feasible(evaluate_constraints(x, BOUNDS))

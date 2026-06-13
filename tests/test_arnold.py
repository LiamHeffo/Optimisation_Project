# tests/test_arnold.py
#
# Unit tests for the Arnold & Hansen 2012 constraint-handling integration:
#
#   * Module-level sim_type classification helpers.
#   * Strategy __init__ validation (unknown sim_type, incompatible
#     feature flags, missing n_constraints under Arnold).
#   * _arnold_update_v (Eq. 6): low-pass filter accumulation, +inf skip.
#   * _arnold_update_A (Eq. 7): multi-rank subtractive update; v_j
#     direction is the one that shrinks; PSD guard / no-op contract.
#   * apply_arnold_infeasibility: feasible offspring untouched; infeasible
#     offspring drive v + A update and are marked infeasible (no
#     resampling).
#
# Each test isolates one property so failures localise.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from algorithm.cmaes import (
    StrategyMultiObjective,
    is_cht_active, is_al_active, cht_method,
    CHOCAT_SIM_TYPES, ARNOLD_SIM_TYPES, AL_ENABLED_SIM_TYPES,
)


N = 6
M = 18   # 6 physical + 2*6 box for the X2 problem


# ─────────────────────────────────────────────────────────────────────────────
# Module-level classification helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_classification_helpers_chocat():
    assert is_cht_active('CovarianceCHT')
    assert is_cht_active('CHT_AL')
    assert not is_al_active('CovarianceCHT')
    assert     is_al_active('CHT_AL')
    assert cht_method('CovarianceCHT') == 'chocat'
    assert cht_method('CHT_AL')        == 'chocat'


def test_classification_helpers_arnold():
    assert is_cht_active('ArnoldCHT')
    assert is_cht_active('ArnoldCHT_AL')
    assert not is_al_active('ArnoldCHT')
    assert     is_al_active('ArnoldCHT_AL')
    assert cht_method('ArnoldCHT')    == 'arnold'
    assert cht_method('ArnoldCHT_AL') == 'arnold'


def test_classification_helpers_non_cht():
    for st in ('ParentValue', 'ElitistCrossover', 'RandomCrossover', 'Penalty'):
        assert not is_cht_active(st)
        assert not is_al_active(st)
        assert cht_method(st) is None


def test_classification_constants_partition():
    """Every CHT sim_type lives in exactly one of the two families."""
    assert set(CHOCAT_SIM_TYPES).isdisjoint(set(ARNOLD_SIM_TYPES))
    # Every AL-active sim_type must be a CHT sim_type.
    assert set(AL_ENABLED_SIM_TYPES).issubset(
        set(CHOCAT_SIM_TYPES) | set(ARNOLD_SIM_TYPES)
    )


# ─────────────────────────────────────────────────────────────────────────────
# __init__ validation
# ─────────────────────────────────────────────────────────────────────────────

class _AttrList(list):
    """list that accepts arbitrary attributes (_lineage_id etc.).

    Plain np.ndarray rejects attribute setting (NumPy >= 1.20), so the
    strategy's lineage-tagging loop fails on raw arrays.  Wrapping the
    parents in a list subclass keeps the design-vector semantics and
    permits attribute setting — mirrors what DEAP's Individual class
    does in production.
    """
    pass


def _make_strategy(sim_type='ArnoldCHT', n_constraints=M, **kwargs):
    """Minimal Arnold-aware strategy for unit testing.

    All paths through cmaes that touch SPARK / PITOT3 / DEAP fitness are
    avoided here — we only exercise the constraint-handling internals.
    """
    rng = np.random.RandomState(0)
    parents = [
        _AttrList(rng.uniform(1.4, 1.6, size=N).tolist())
        for _ in range(3)
    ]
    return StrategyMultiObjective(
        population=parents,
        sigma=0.5,
        mu=3, lambda_=3,
        sim_type=sim_type,
        n_constraints=n_constraints,
        **kwargs,
    )


def test_unknown_sim_type_rejected():
    with pytest.raises(ValueError, match="Unknown sim_type"):
        _make_strategy(sim_type='nonsense')


def test_arnold_requires_n_constraints():
    rng = np.random.RandomState(0)
    parents = [
        _AttrList(rng.uniform(1.4, 1.6, size=N).tolist()) for _ in range(3)
    ]
    with pytest.raises(ValueError, match="n_constraints"):
        StrategyMultiObjective(
            population=parents, sigma=0.5, mu=3, lambda_=3,
            sim_type='ArnoldCHT',
            # n_constraints intentionally omitted
        )


def test_chocat_feature_rejected_under_arnold():
    with pytest.raises(ValueError, match="Chocat-only"):
        _make_strategy(
            sim_type='ArnoldCHT',
            features={'cht_eigenvalue_floor': 0.5},
        )


def test_box_reflective_repair_rejected_under_arnold():
    with pytest.raises(ValueError, match="Chocat-only"):
        _make_strategy(
            sim_type='ArnoldCHT',
            features={'box_reflective_repair': True},
        )


def test_psucc_exclude_resampled_rejected_under_arnold():
    with pytest.raises(ValueError, match="Chocat-only"):
        _make_strategy(
            sim_type='ArnoldCHT',
            features={'psucc_exclude_resampled': True},
        )


def test_chocat_feature_accepted_under_chocat():
    """Same flag that errors under Arnold is fine under Chocat."""
    s = _make_strategy(
        sim_type='CovarianceCHT', n_constraints=M,
        features={'cht_eigenvalue_floor': 0.5},
    )
    assert s.cht_eigenvalue_floor == 0.5


def test_per_parent_v_allocated_for_arnold():
    s = _make_strategy(sim_type='ArnoldCHT')
    assert len(s.v) == 3          # one per parent
    assert len(s.v[0]) == M       # one per constraint
    assert s.v[0][0].shape == (N,)
    assert np.all(s.v[0][0] == 0)


def test_v_not_allocated_for_non_arnold():
    s = _make_strategy(sim_type='CovarianceCHT')
    assert s.v == []


def test_arnold_default_parameters_match_paper():
    """β = 0.1/(n+2), c_c = 1/(n+2) at n=6 → 0.0125 and 0.125."""
    s = _make_strategy(sim_type='ArnoldCHT')
    assert s.arnold_beta == pytest.approx(0.1 / (N + 2))
    assert s.arnold_cc   == pytest.approx(1.0 / (N + 2))


# ─────────────────────────────────────────────────────────────────────────────
# _arnold_update_v (Eq. 6)
# ─────────────────────────────────────────────────────────────────────────────

def test_v_only_updates_on_finite_positive_g():
    """Only constraints with isfinite(g_j) > 0 get their v_j updated."""
    s = _make_strategy(sim_type='ArnoldCHT')
    Az = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    g = np.zeros(M)
    g[2] = 0.5   # active
    g[5] = np.inf  # box-cascade — must be skipped
    g[7] = -0.1  # inactive

    active_js = s._arnold_update_v(parent_idx=0, Az=Az, g=g)

    assert active_js == [2]
    assert np.allclose(s.v[0][2], s.arnold_cc * Az)
    assert np.allclose(s.v[0][5], 0.0)
    assert np.allclose(s.v[0][7], 0.0)


def test_v_low_pass_accumulation():
    """Repeated violations in the same direction must grow ‖v_j‖ but
    bounded by Az (geometric series sum)."""
    s = _make_strategy(sim_type='ArnoldCHT')
    Az = np.array([1.0, 0, 0, 0, 0, 0])
    g = np.zeros(M); g[0] = 1.0
    norms = []
    # After n iterations of EMA with rate c_c=0.125 on a constant signal:
    #   ‖v_n‖ = (1 − (1−c_c)^n) · ‖Az‖
    # Pick n large enough that (1−c_c)^n ≪ tolerance below.
    n_iter = 300
    for _ in range(n_iter):
        s._arnold_update_v(parent_idx=0, Az=Az, g=g)
        norms.append(np.linalg.norm(s.v[0][0]))
    assert all(b >= a - 1e-12 for a, b in zip(norms, norms[1:]))   # monotonic
    # Asymptotic limit is ‖Az‖; loosely check approach.
    assert norms[-1] == pytest.approx(np.linalg.norm(Az), rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# _arnold_update_A (Eq. 7)
# ─────────────────────────────────────────────────────────────────────────────

def test_update_A_no_active_js_is_noop():
    s = _make_strategy(sim_type='ArnoldCHT')
    A, invA = np.eye(N), np.eye(N)
    A_new, invA_new = s._arnold_update_A(
        A, invA, parent_idx=0, active_js=[],
    )
    assert np.array_equal(A_new, A)
    assert np.array_equal(invA_new, invA)


def test_update_A_shrinks_along_v_direction():
    """After Eq. 7, the variance of A·z in the unit direction of v_j
    must be strictly smaller than before."""
    s = _make_strategy(sim_type='ArnoldCHT')
    # Pre-load v[0][3] with a known direction by simulating one Eq. 6
    # call.  Use a strong c_c so the v is well-developed in one step.
    direction = np.array([1.0, 0, 0, 0, 0, 0])
    s.v[0][3] = direction.copy()

    A_old   = np.eye(N).copy()
    invA_old = np.eye(N).copy()

    A_new, invA_new = s._arnold_update_A(
        A_old.copy(), invA_old.copy(),
        parent_idx=0, active_js=[3],
    )

    # ‖A_new · direction‖ < ‖A_old · direction‖ — shrinkage in v_j's
    # direction.
    before = np.linalg.norm(A_old @ direction)
    after  = np.linalg.norm(A_new @ direction)
    assert after < before


def test_update_A_preserves_orthogonal_directions():
    """A direction orthogonal to v_j should be barely affected."""
    s = _make_strategy(sim_type='ArnoldCHT')
    s.v[0][3] = np.array([1.0, 0, 0, 0, 0, 0])
    ortho = np.array([0.0, 1.0, 0, 0, 0, 0])

    A_new, _ = s._arnold_update_A(
        np.eye(N), np.eye(N),
        parent_idx=0, active_js=[3],
    )

    before = np.linalg.norm(np.eye(N) @ ortho)
    after  = np.linalg.norm(A_new @ ortho)
    # The Eq. 7 outer product (v_j w_j^T)/(w_j^T w_j) is rank-1 along
    # v_j's row index, so the orthogonal component shrinks far less.
    # We just assert that the dominant shrinkage is on v_j's direction.
    assert abs(after - before) < (
        np.linalg.norm(np.eye(N) @ s.v[0][3])
        - np.linalg.norm(A_new @ s.v[0][3])
    )


def test_update_A_inverse_consistency():
    """invCholesky_new must remain the inverse of A_new (Cholesky invariant)."""
    s = _make_strategy(sim_type='ArnoldCHT')
    s.v[0][7] = np.array([0.0, 1.0, 0, 0, 0, 0])
    A_new, invA_new = s._arnold_update_A(
        np.eye(N), np.eye(N), parent_idx=0, active_js=[7],
    )
    assert np.allclose(invA_new @ A_new, np.eye(N), atol=1e-8)


def test_diag_record_appended():
    s = _make_strategy(sim_type='ArnoldCHT')
    s.v[0][2] = np.array([1.0, 0, 0, 0, 0, 0])
    assert s.arnold_diag_buffer == []
    s._arnold_update_A(
        np.eye(N), np.eye(N), parent_idx=0, active_js=[2],
    )
    assert len(s.arnold_diag_buffer) == 1
    rec = s.arnold_diag_buffer[0]
    assert rec['m_active'] == 1
    assert rec['active_js'] == [2]
    assert rec['shrink_applied'] is True
    assert rec['psd_fallback']   is False
    assert rec['A_delta_fro'] > 0


# ─────────────────────────────────────────────────────────────────────────────
# apply_arnold_infeasibility — full lifecycle slot
# ─────────────────────────────────────────────────────────────────────────────

class _FakeInd(list):
    """Minimal stand-in for a DEAP Individual that we never evaluate."""
    pass


def _make_offspring_from(strat, parent_idx, dx):
    """Build a fake offspring at parent + dx with the _Az + _ps tags
    that apply_arnold_infeasibility expects."""
    p = strat.parents[parent_idx]
    x = np.array(p) + dx
    ind = _FakeInd(x.tolist())
    ind._Az = np.array(dx, dtype=float)
    ind._ps = ("o", parent_idx)
    ind._lineage_id = 999 + parent_idx
    return ind


def test_apply_arnold_feasible_untouched():
    """A feasible offspring keeps _feasible=True and does not change v or A."""
    s = _make_strategy(sim_type='ArnoldCHT')
    A_before = [Ai.copy() for Ai in s.A]
    v_before = [[vij.copy() for vij in row] for row in s.v]

    off = _make_offspring_from(s, parent_idx=0, dx=np.zeros(N))

    def _feas(ind):
        return True, -np.ones(M)

    s.apply_arnold_infeasibility([off], feasibility_check=_feas)

    assert off._feasible is True
    for k in range(3):
        assert np.array_equal(s.A[k], A_before[k])
        for j in range(M):
            assert np.array_equal(s.v[k][j], v_before[k][j])


def test_apply_arnold_infeasible_marks_and_updates():
    """An infeasible offspring is marked, drives v[0][j] update and
    A[0] change.  No resampling — the offspring itself is not mutated."""
    s = _make_strategy(sim_type='ArnoldCHT')
    A0_before = s.A[0].copy()

    dx = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    off = _make_offspring_from(s, parent_idx=0, dx=dx)
    x_pre = list(off)

    def _feas(ind):
        g = -np.ones(M)
        g[3] = 0.7   # constraint 3 violated
        return False, g

    s.apply_arnold_infeasibility([off], feasibility_check=_feas)

    assert off._feasible is False
    assert np.allclose(off._g[3], 0.7)
    # No resampling — design vector is unchanged.
    assert list(off) == x_pre
    # v[0][3] now carries c_c · dx.
    assert np.allclose(s.v[0][3], s.arnold_cc * dx)
    # A[0] changed (Eq. 7 fired).
    assert not np.array_equal(s.A[0], A0_before)


def test_apply_arnold_pure_box_cascade_no_shrink():
    """If the only violations are +inf cascades (box bound) with no
    finite-positive entry, no Eq. 7 call should shrink A.  Records a
    diagnostic but with m_active=0."""
    s = _make_strategy(sim_type='ArnoldCHT')
    A0_before = s.A[0].copy()

    off = _make_offspring_from(s, parent_idx=0, dx=np.full(N, 0.6))

    def _feas(ind):
        g = -np.ones(M)
        g[0] = np.inf   # cascade
        g[1] = np.inf
        return False, g

    s.apply_arnold_infeasibility([off], feasibility_check=_feas)

    assert off._feasible is False
    assert np.array_equal(s.A[0], A0_before)   # untouched
    assert len(s.arnold_diag_buffer) == 1
    assert s.arnold_diag_buffer[0]['m_active'] == 0
    assert s.arnold_diag_buffer[0]['shrink_applied'] is False

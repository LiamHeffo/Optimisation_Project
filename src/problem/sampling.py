"""
Latin Hypercube sampling utilities for the X2 design space.

Two top-level helpers:

    lhs_sample(n_samples, *, rng=None, lhs_optimization="random-cd")
        Draw ``n_samples`` PHYSICAL-space points via LHS over [0,1]^6.
        Handles the conditional axis bounds for p4 and reservoir_p the
        same way the legacy pop_init did.

    draw_structurally_feasible_lhs(n, pop_init_fn, bounds,
                                   oversample=4, max_doublings=3)
        Oversample-and-filter wrapper around any pop_init_fn (typically
        a closure that calls lhs_sample): returns ``n`` normalised-space
        designs that pass evaluate_constraints / is_feasible.  Per-slot
        rejection would destroy LHS stratification; this preserves it
        on the original draw and yields a near-LHS subset.

Both helpers are kept side-effect-free (no DEAP imports, no global
RNG mutation) so they can be unit-tested in isolation.
"""

from problem.config import (
    he_lower, he_upper,
    driver_p_lower, driver_p_upper,
    p4_upper,
    D_throat_lower, D_throat_upper,
    reservoir_upper,
    buffer_length_lower, buffer_length_upper,
)
from problem.feasibility import evaluate_constraints, is_feasible
from problem.transforms import variable_transformation


def lhs_sample(n_samples, *, rng=None, lhs_optimization="random-cd"):
    """Latin Hypercube sample over the X2 design space, in PHYSICAL units.

    Output column order matches the legacy pop_init:
        [percent_he, driver_p, p4, D_throat, reservoir_p, buffer_length]

    p4 and reservoir_p are mapped through their CONDITIONAL bounds
    (driven by the just-sampled driver_p), preserving the original
    pop_init constraint semantics.  LHS stratification is preserved
    in the latent [0,1]^6 cube; in physical units the conditional
    axes are stratified over their per-sample support.

    n_samples == 0 returns [].  n_samples == 1 degenerates to a single
    uniform draw — used by the per-slot fallback paths.
    """
    if n_samples <= 0:
        return []
    from scipy.stats.qmc import LatinHypercube
    sampler = LatinHypercube(d=6, optimization=lhs_optimization, seed=rng)
    u = sampler.random(n_samples)

    out = []
    for i in range(n_samples):
        he      = he_lower            + u[i, 0] * (he_upper            - he_lower)
        drv_p   = driver_p_lower      + u[i, 1] * (driver_p_upper      - driver_p_lower)
        Dthroat = D_throat_lower      + u[i, 3] * (D_throat_upper      - D_throat_lower)
        buf_len = buffer_length_lower + u[i, 5] * (buffer_length_upper - buffer_length_lower)

        p4_hi = min(1190.63 * drv_p, p4_upper)
        p4_lo = 14.62 * drv_p
        p4    = p4_lo + u[i, 2] * (p4_hi - p4_lo)

        res_p = drv_p + u[i, 4] * (reservoir_upper - drv_p)

        out.append([he, drv_p, p4, Dthroat, res_p, buf_len])
    return out


def draw_structurally_feasible_lhs(n, pop_init_fn, bounds,
                                   oversample=4, max_doublings=3):
    """Return ``n`` normalised-space designs that pass the cheap
    structural feasibility check (evaluate_constraints / is_feasible).

    Strategy: draw n·oversample raw samples via pop_init_fn, transform
    to normalised [1,2]^6, keep the first n that pass.  If fewer than
    n pass, double oversample and retry up to max_doublings times.
    """
    k = oversample
    kept = []
    for attempt in range(max_doublings + 1):
        size = n * k
        raw = pop_init_fn(size)
        transformed = variable_transformation(raw, bounds)
        kept = [
            x for x in transformed
            if is_feasible(evaluate_constraints(x, bounds))
        ]
        if len(kept) >= n:
            return kept[:n]
        k *= 2
    raise RuntimeError(
        f"LHS init: only {len(kept)} of {n} structurally feasible after "
        f"{max_doublings} doublings (final oversample factor {k // 2})."
    )

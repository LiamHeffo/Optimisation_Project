"""
Multi-Objective CMA-ES strategy (MO-CMA-ES).

Implements the (mu + lambda)-MO-CMA-ES algorithm from:
    Voss, Hansen, Igel — "Improved Step Size Adaptation for the MO-CMA-ES", 2010.

The strategy object is used with a generate–update loop:

    strategy = StrategyMultiObjective(population, sigma, ...)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    for gen in range(NGEN):
        offspring = toolbox.generate()
        # ... evaluate offspring ...
        toolbox.update(offspring)

Note on X2-specific coupling:
    check_feasibility() and crossover() contain X2-specific logic for the p4
    pressure constraint and its coordinate-coupled transformation.  These
    reference variable_untransformation() from problem.transforms.  If this
    strategy is ever reused for a different problem, those two methods should
    be parameterised via callables passed to __init__.
"""

import time
import numpy as np
import scipy.linalg
from deap import tools

from problem.transforms import variable_untransformation


class StrategyMultiObjective(object):
    """Multiobjective CMA-ES strategy.

    Parameters
    ----------
    population : list
        Initial parent population of DEAP individuals.
    sigma : float
        Initial step size (same for all parents).
    mu : int, optional
        Number of parents to keep (defaults to len(population)).
    lambda_ : int, optional
        Number of offspring per generation (defaults to 1).
    sim_type : str
        Constraint-handling strategy: 'ParentValue', 'ElitistCrossover',
        'RandomCrossover', or 'Penalty'.
    p4_treatment : str or None
        Special treatment for the p4 variable.  Use 'hard_bounds_on_p4'
        to enforce the upper bound via re-transformed coordinates.
    bounds : list of (lo, hi) tuples
        Physical-space bounds for each design variable.
    logbook : Logbook, optional
        If provided, generation statistics are written to
        logbook.bookshelf during generate().
    indicator : callable, optional
        Hypervolume indicator function (defaults to deap.tools.hypervolume).

    CMA-ES hyperparameters (all optional, sensible defaults shown):

    +----------+---------------------------+-------------------------------+
    | d        | 1.0 + N/2                 | Step-size damping             |
    | ptarg    | 1/(5 + 0.5)               | Target success rate           |
    | cp       | ptarg / (2 + ptarg)       | Step-size learning rate       |
    | cc       | 2 / (N + 2)               | Cumulation time horizon       |
    | ccov     | 2 / (N^2 + 6)             | Covariance matrix learning    |
    | pthresh  | 0.44                      | Success rate threshold        |
    +----------+---------------------------+-------------------------------+
    """

    def __init__(self, population, sigma, **params):
        self.sim_type     = params.get("sim_type", 1)
        self.p4_treatment = params.get("p4_treatment")
        self.bounds       = params.get("bounds")
        self.logbook      = params.get("logbook", None)   # injected — no global access

        print(f'self.sim_type = {self.sim_type}')
        print(f'self.p4_treatment = {self.p4_treatment}')
        print(f"bounds = {self.bounds}")

        # Defensive: every initial parent must be feasible (have a valid
        # fitness).  If not, _select() will filter it out, the per-parent
        # state arrays will shrink below mu, and a subsequent generate()
        # will IndexError on the missing slot.  Failing loudly here is
        # vastly easier to debug than that downstream symptom.
        for i, p in enumerate(population):
            if not getattr(p, "_feasible", True):
                raise ValueError(
                    f"StrategyMultiObjective received an infeasible initial "
                    f"parent at index {i}.  Every parent must satisfy the "
                    f"problem's feasibility check before being passed to the "
                    f"strategy — resample or repair it first."
                )

        self.parents = population
        self.dim = len(self.parents[0])

        # Selection
        self.mu      = params.get("mu",      len(self.parents))
        self.lambda_ = params.get("lambda_", 1)

        # Step-size control
        self.d      = params.get("d",      1.0 + self.dim / 2.0)
        self.ptarg  = params.get("ptarg",  1.0 / (5.0 + 0.5))
        self.cp     = params.get("cp",     self.ptarg / (2.0 + self.ptarg))

        # Covariance matrix adaptation
        self.cc     = params.get("cc",     2.0 / (self.dim + 2.0))
        self.ccov   = params.get("ccov",   2.0 / (self.dim ** 2 + 6.0))
        self.pthresh = params.get("pthresh", 0.44)

        # Rank-mu_MO,succ recombination (Voss 2009).
        # d_steps controls the Mahalanobis neighbourhood radius (eq. 13–17).
        self.d_steps = params.get("d_steps", self.dim + 3)

        # CHT (Chocat 2015) covariance shrinkage strength.  Analogous to
        # beta in the (1+1)-CMA-ES paper (Arnold & Hansen 2012, Table 1):
        # beta = 0.1 / (n + 2).  Higher values shrink more aggressively
        # along violating directions.
        self.cht_gamma = params.get("cht_gamma", 0.1 / (self.dim + 2.0))

        # Per-parent internal state
        self.sigmas      = [sigma] * len(population)
        self.A           = [np.identity(self.dim) for _ in range(len(population))]
        self.invCholesky = [np.identity(self.dim) for _ in range(len(population))]
        self.pc          = [np.zeros(self.dim)    for _ in range(len(population))]
        self.psucc       = [self.ptarg]            * len(population)

        self.indicator = params.get("indicator", tools.hypervolume)
        self.time_spent_fixing = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def generate(self, ind_init):
        """Generate lambda_ offspring (one per parent) from the current strategy.

        Each offspring is tagged with ``_ps = ("o", parent_index)`` so that
        update() can trace it back to its parent.
        """
        arz = np.random.randn(self.lambda_, self.dim)
        individuals = list()

        for i, p in enumerate(self.parents):
            p._ps = "p", i

        if self.lambda_ == self.mu:
            for i in range(self.lambda_):
                mutation = self.sigmas[i] * np.dot(self.A[i], arz[i])
                new_individual = self.parents[i] + mutation
                repaired = False

                # CovarianceCHT replaces the repair while-loop: infeasible
                # offspring pass through and feed the CHT covariance update
                # (Phase 3).  Penalty mode also bypasses repair (its handler
                # is in evaluate.py).
                if self.sim_type not in ('Penalty', 'CovarianceCHT'):
                    s = time.time()
                    while True:
                        if not self.check_feasibility(new_individual)[0]:
                            repaired = True
                            # Record repair attempts in the logbook bookshelf
                            if self.logbook is not None:
                                gen_key = f'{self.logbook.bookshelf["generation"]}'
                                if gen_key in self.logbook.bookshelf["fixer_count"]:
                                    self.logbook.bookshelf["fixer_count"][gen_key] += 1
                                else:
                                    self.logbook.bookshelf["fixer_count"][gen_key] = 1

                            bad_attribute = self.check_feasibility(new_individual)[1]
                            new_individual = self.crossover(new_individual, bad_attribute, i)
                        else:
                            break
                    e = time.time()

                    if self.logbook is not None:
                        gen_key = f'{self.logbook.bookshelf["generation"]}'
                        if gen_key not in self.logbook.bookshelf["fixer_count"]:
                            self.logbook.bookshelf["fixer_count"][gen_key] = 0
                        self.logbook.bookshelf['time_spent_fixing'] += e - s

                individuals.append(ind_init(new_individual))
                individuals[-1]._ps = "o", i
                individuals[-1]._repaired = repaired

        else:
            # Random-parent variant: pick parents from the first Pareto front
            ndom = tools.sortLogNondominated(self.parents, len(self.parents), first_front_only=True)
            for i in range(self.lambda_):
                j = np.random.randint(0, len(ndom))
                _, p_idx = ndom[j]._ps
                individuals.append(
                    ind_init(
                        self.parents[p_idx]
                        + self.sigmas[p_idx] * np.dot(self.A[p_idx], arz[i])
                    )
                )
                individuals[-1]._ps = "o", p_idx
                individuals[-1]._repaired = False

        return individuals

    def update(self, population):
        """Update covariance matrices and step sizes from the evaluated population."""
        chosen, not_chosen = self._select(population + self.parents)

        cp, cc, ccov = self.cp, self.cc, self.ccov
        d, ptarg, pthresh = self.d, self.ptarg, self.pthresh

        last_steps    = [self.sigmas[ind._ps[1]]       if ind._ps[0] == "o" else None for ind in chosen]
        sigmas        = [self.sigmas[ind._ps[1]]       if ind._ps[0] == "o" else None for ind in chosen]
        invCholesky   = [self.invCholesky[ind._ps[1]].copy() if ind._ps[0] == "o" else None for ind in chosen]
        A             = [self.A[ind._ps[1]].copy()     if ind._ps[0] == "o" else None for ind in chosen]
        pc            = [self.pc[ind._ps[1]].copy()    if ind._ps[0] == "o" else None for ind in chosen]
        psucc         = [self.psucc[ind._ps[1]]        if ind._ps[0] == "o" else None for ind in chosen]

        # Snapshot parent state for the rank-mu_MO,succ update (Voss 2009).
        # The per-offspring loop below mutates self.sigmas in-place, so we
        # need a frozen view of (x_k^(g), sigma_k^(g)) at update-entry time.
        # Repaired offspring are excluded — their (x' - x) is not a clean
        # Gaussian step (temporary; revisit when box-constraint handling
        # is refactored).
        parents_snapshot = [np.array(p) for p in self.parents]
        sigmas_snapshot  = list(self.sigmas)
        successful_steps = [
            (ind._ps[1], np.array(ind))
            for ind in chosen
            if ind._ps[0] == "o"
            and not getattr(ind, "_repaired", False)
        ]

        # Infeasible-offspring pool for the CHT update.  Built once from
        # the full offspring population (not 'chosen', which excludes
        # them).  Each entry is (donor_parent_idx, x_offspring, g_vector).
        # For non-CHT sim_types the pool is empty (all offspring are
        # feasible by repair construction or have no _g), so the CHT
        # call below is a no-op and existing behaviour is preserved.
        infeasible_pool = [
            (ind._ps[1], np.array(ind), ind._g)
            for ind in population
            if ind._ps[0] == "o"
            and hasattr(ind, "_g")
            and np.any(np.asarray(ind._g) > 0)
        ]

        for i, ind in enumerate(chosen):
            t, p_idx = ind._ps
            if t == "o":
                psucc[i] = (1.0 - cp) * psucc[i] + cp
                sigmas[i] = sigmas[i] * np.exp((psucc[i] - ptarg) / (d * (1.0 - ptarg)))
                print(f"sigmas: {sigmas[i]}")

                # CHT covariance shrinkage (Chocat 2015) — slot in BEFORE
                # rank-mu_succ and rank-one so subsequent updates operate
                # on the constraint-aware geometry.  Matches Chocat
                # Algorithm 3 step-3-2 -> step-3-4 ordering.  Only fires
                # when the sim_type opts in AND there are infeasibles.
                if self.sim_type == 'CovarianceCHT' and infeasible_pool:
                    A[i], invCholesky[i] = self._chtCovarianceUpdate(
                        A[i], invCholesky[i], p_idx,
                        parents_snapshot, sigmas_snapshot, infeasible_pool,
                    )

                # Rank-mu_MO,succ recombination (Voss 2009): blend in
                # information from neighbouring successful offspring before
                # applying the standard rank-one Cholesky update below.
                A[i], invCholesky[i] = self._rankMuSuccUpdate(
                    A[i], invCholesky[i], p_idx,
                    parents_snapshot, sigmas_snapshot, successful_steps,
                )

                if psucc[i] < pthresh:
                    xp = np.array(ind)
                    x  = np.array(self.parents[p_idx])
                    pc[i] = (1.0 - cc) * pc[i] + np.sqrt(cc * (2.0 - cc)) * (xp - x) / last_steps[i]
                    invCholesky[i], A[i] = self._rankOneUpdate(invCholesky[i], A[i], 1 - ccov, ccov, pc[i])
                else:
                    pc[i] = (1.0 - cc) * pc[i]
                    pc_weight = cc * (2.0 - cc)
                    invCholesky[i], A[i] = self._rankOneUpdate(invCholesky[i], A[i], 1 - ccov + pc_weight, ccov, pc[i])

                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx] + cp
                self.sigmas[p_idx] = self.sigmas[p_idx] * np.exp(
                    (self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg))
                )

        for ind in not_chosen:
            t, p_idx = ind._ps
            if t == "o":
                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx]
                self.sigmas[p_idx] = self.sigmas[p_idx] * np.exp(
                    (self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg))
                )

        self.parents     = chosen
        self.sigmas      = [sigmas[i]      if ind._ps[0] == "o" else self.sigmas[ind._ps[1]]      for i, ind in enumerate(chosen)]
        self.invCholesky = [invCholesky[i] if ind._ps[0] == "o" else self.invCholesky[ind._ps[1]] for i, ind in enumerate(chosen)]
        self.A           = [A[i]           if ind._ps[0] == "o" else self.A[ind._ps[1]]           for i, ind in enumerate(chosen)]
        self.pc          = [pc[i]          if ind._ps[0] == "o" else self.pc[ind._ps[1]]          for i, ind in enumerate(chosen)]
        self.psucc       = [psucc[i]       if ind._ps[0] == "o" else self.psucc[ind._ps[1]]       for i, ind in enumerate(chosen)]

    # ─────────────────────────────────────────────────────────────────────────
    # X2-specific feasibility repair (p4 pressure constraint)
    # ─────────────────────────────────────────────────────────────────────────

    def crossover(self, individual, attribute_index, i):
        """Repair an infeasible individual by borrowing the offending attribute
        from a selected parent, then re-transforming p4 if necessary."""

        def swap_attribute(individual, crossover_individual_index, attribute_index):
            crossover_individual = self.parents[crossover_individual_index]

            if attribute_index == 2 and self.p4_treatment == "hard_bounds_on_p4":
                # When swapping p4, the new driver_p changes the transformation,
                # so we must retransform the donor's p4 into the child's driver_p frame.
                swapped_p4_natural = variable_untransformation(self.parents[i], self.bounds)[2]
                child_driver_p_nat = variable_untransformation(individual, self.bounds)[1]
                swapped_p4_transformed = (
                    (swapped_p4_natural - 14.62 * child_driver_p_nat)
                    / (1190.63 * child_driver_p_nat - 14.62 * child_driver_p_nat)
                    + 1
                )

                if not 1 <= swapped_p4_transformed <= 2:
                    # Donor p4 is also infeasible in the child's frame; copy both
                    individual[1] = crossover_individual[1]
                    individual[attribute_index] = crossover_individual[attribute_index]
                else:
                    individual[attribute_index] = swapped_p4_transformed
            else:
                individual[attribute_index] = crossover_individual[attribute_index]

        if self.sim_type == 'ElitistCrossover':
            ref = np.array([ind.fitness.wvalues for ind in self.parents]) * -1
            ref = np.max(ref, axis=0) + 1
            crossover_individual_index = self.indicator(self.parents, ref=ref)
            swap_attribute(individual, crossover_individual_index, attribute_index)

        if self.sim_type == 'RandomCrossover':
            attribute_list = [
                (pop_idx, ind[attribute_index])
                for pop_idx, ind in enumerate(self.parents)
            ]
            crossover_individual_index = attribute_list[np.random.randint(0, len(attribute_list))][0]
            swap_attribute(individual, crossover_individual_index, attribute_index)

        if self.sim_type == 'ParentValue':
            swap_attribute(individual, i, attribute_index)

        return individual

    def check_feasibility(self, new_individual):
        """Return (True, 0) if feasible, or (False, bad_index) otherwise.

        Checks the hard p4 upper bound (if p4_treatment is set) and then the
        normalised [1, 2] bounds on all variables.
        """
        if self.p4_treatment == "hard_bounds_on_p4":
            p4_upper = self.bounds[2][1]
            p4_real_value = variable_untransformation(new_individual, self.bounds)[2]
            if p4_real_value >= p4_upper:
                return (False, 2)

        for j in range(len(new_individual)):
            if not (1 <= new_individual[j] <= 2):
                return (False, j)

        return (True, 0)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _select(self, candidates):
        """Select mu individuals from candidates using Pareto ranking + hypervolume.

        Infeasible candidates (those with ``_feasible == False``, set by the
        evaluation pipeline when the constraint vector is violated and no
        fitness was computed) are filtered out before the Pareto sort.
        DEAP's ``tools.sortLogNondominated`` requires every individual to
        have a valid fitness — including infeasibles would crash the sort.

        Filtered infeasibles are appended directly to ``not_chosen`` so they
        still surface to ``update()`` via the population it received, and
        their ``_g`` vectors feed the CHT covariance update.
        """
        # Partition: feasibles drive selection; infeasibles bypass it.
        feasible    = [ind for ind in candidates if getattr(ind, "_feasible", True)]
        infeasibles = [ind for ind in candidates if not getattr(ind, "_feasible", True)]

        if len(feasible) <= self.mu:
            return feasible, infeasibles

        pareto_fronts = tools.sortLogNondominated(feasible, len(feasible))

        chosen = []
        mid_front = None
        not_chosen = list(infeasibles)
        full = False

        for front in pareto_fronts:
            if len(chosen) + len(front) <= self.mu and not full:
                chosen += front
            elif mid_front is None and len(chosen) < self.mu:
                mid_front = front
                full = True
            else:
                not_chosen += front

        k = self.mu - len(chosen)

        if k > 0:
            ref = np.array([ind.fitness.wvalues for ind in feasible]) * -1
            ref = np.max(ref, axis=0) + 1

            for _ in range(len(mid_front) - k):
                idx = self.indicator(mid_front, ref=ref)
                not_chosen.append(mid_front.pop(idx))

            chosen += mid_front

        return chosen, not_chosen

    def _rankMuSuccUpdate(self, A, invCholesky, parent_idx, parents_snapshot,
                          sigmas_snapshot, successful_steps):
        """Rank-mu_MO,succ update of parent_idx's covariance (Voss 2009, eq. 8).

        Replaces C_i with (1 - sum_w) * C_i + Z, where Z is a weighted sum of
        outer products of normalised steps from successful offspring across
        the population, weighted by Mahalanobis closeness in C_i's metric.

        The hybrid scheme: reconstruct C = A A^T, blend, re-Cholesky.  Cheap
        for small n; preserves the existing rank-one Cholesky path that fires
        immediately after this in update().

        Parameters
        ----------
        A, invCholesky : ndarray
            Current Cholesky factor of C_i and its inverse.
        parent_idx : int
            Index of the parent whose covariance is being updated.
        parents_snapshot, sigmas_snapshot : list
            Snapshots of self.parents and self.sigmas taken at the start of
            update(), so values are not corrupted by mid-loop mutation.
        successful_steps : list of (donor_parent_idx, x_offspring)
            Offspring deemed successful and not repaired.  Each contributes a
            step (x' - x_donor) / sigma_donor to the rank-mu aggregate.

        Note on success criterion
        -------------------------
        Voss 2009 defines a successful offspring as one that dominates its
        parent in the joint Q^(g) ranking (indicator I(a' < a)).  We instead
        reuse the existing success-rate test (psucc < pthresh) that already
        drives our sigma adaptation, for consistency.  Empirical impact has
        not been measured; revisit if the recombination underperforms.
        """
        n = self.dim
        mu_succ = len(successful_steps)
        if mu_succ == 0:
            return A, invCholesky

        x_i = np.array(parents_snapshot[parent_idx])
        sigma_i = sigmas_snapshot[parent_idx]

        w_pp  = np.zeros(mu_succ)   # w''_ij  (eq. 15)
        steps = np.zeros((mu_succ, n))

        scale = np.sqrt(self.d_steps * n)
        for k, (donor_idx, x_off) in enumerate(successful_steps):
            x_off  = np.asarray(x_off)
            x_don  = np.asarray(parents_snapshot[donor_idx])
            sig_don = sigmas_snapshot[donor_idx]
            # Mahalanobis distance under C_i (eq. 10): ||invCholesky · diff|| / sigma_i
            d_M = np.linalg.norm(invCholesky @ (x_off - x_i)) / sigma_i
            w_pp[k]  = np.exp(-d_M / scale)         # h(x) = e^{-x}, eq. 17
            steps[k] = (x_off - x_don) / sig_don

        # Normalise (eq. 16).  Denominator includes the (mu - mu_succ) zero-weight slots.
        denom = self.mu - mu_succ + np.sum(w_pp)
        if denom <= 0:
            return A, invCholesky
        w_p = w_pp / denom

        # mu_eff and degeneracy-guard rescale (eq. 19, 23).
        sum_w_p_sq = np.sum(w_p ** 2)
        if sum_w_p_sq <= 0:
            return A, invCholesky
        mu_eff = (np.sum(w_p) ** 2) / sum_w_p_sq
        rescale = min(1.0, (2.0 * mu_eff - 1.0) / ((n + 2) ** 2 + mu_eff))
        w = w_p * rescale

        sum_w = np.sum(w)
        Z = np.einsum("k,ki,kj->ij", w, steps, steps)
        # Note: This reads as:
        #   for k in range(mu_succ):
        #       Z += w[k] * np.outer(steps[k], steps[k])

        C_old = A @ A.T
        C_new = (1.0 - sum_w) * C_old + Z
        C_new = (C_new + C_new.T) / 2.0   # symmetrise for numerical safety

        try:
            A_new = np.linalg.cholesky(C_new)
        except np.linalg.LinAlgError:
            # Blend produced a non-PSD matrix (rare; usually means sum_w ~ 1
            # with degenerate Z).  Skip the rank-mu step this generation.
            return A, invCholesky
        invCholesky_new = np.linalg.solve(A_new, np.eye(n))
        return A_new, invCholesky_new

    def _chtCovarianceUpdate(self, A, invCholesky, parent_idx,
                              parents_snapshot, sigmas_snapshot,
                              infeasible_offspring, gamma=None):
        """Chocat 2015 CHT covariance update with Adaptation-B pooling.

        For parent ``parent_idx`` with current Cholesky factor ``A``,
        shrink the search ellipsoid along eigenvectors that point into
        directions where infeasible offspring landed.  Hypervolume of the
        ellipsoid is preserved by an explicit determinant rescale, so
        only the *shape* of the search distribution changes.

        Adaptation-B pooling
        --------------------
        Chocat assumes one global (m, C); we have per-parent (xᵢ, σᵢ, Cᵢ).
        Each parent updates from *all* infeasible offspring across the
        swarm, weighted by Mahalanobis closeness in this parent's metric
        (same trick as ``_rankMuSuccUpdate``).  Distant offspring
        contribute ~0; the parent's own offspring contributes most.

        Algorithm (mapping to Chocat eq. numbers)
        -----------------------------------------
        1.  Eigendecompose Cᵢ = P D² Pᵀ (eq. 8–9).
        2.  Mahalanobis pool weight per offspring: exp(-d_M / scale).
        3.  Per-constraint rank weights wᵢⱼ from eq. 13, multiplied by the
            pool weight.
        4.  Eigenvalue shrinkage along violation projections (eq. 12),
            clamped at ε·vp_i to keep S strictly positive-definite.
        5.  Hypervolume rescale [det(C)/det(S)]^(1/n) computed in
            log-space for numerical safety (eq. 11).
        6.  Re-Cholesky with PSD-failure fallback (matches the existing
            try/except pattern in _rankMuSuccUpdate).

        Parameters
        ----------
        A, invCholesky : (n, n) ndarray
            Current Cholesky factor of Cᵢ and its inverse.
        parent_idx : int
            Index of the parent whose Cholesky is being updated.
        parents_snapshot, sigmas_snapshot : list
            Frozen views of self.parents and self.sigmas at update-entry,
            so values are not corrupted by mid-loop mutation.
        infeasible_offspring : list of (donor_idx, x_offspring, g_vector)
            Every infeasible offspring this generation, regardless of
            which parent generated it.
        gamma : float, optional
            Shrinkage strength.  Defaults to self.cht_gamma.

        Returns
        -------
        A_new, invCholesky_new : (n, n) ndarray
            Updated Cholesky and its inverse.  Returns (A, invCholesky)
            unchanged if there are no violators or if the update would
            yield a non-PSD matrix.
        """
        n = self.dim
        if not infeasible_offspring:
            return A, invCholesky
        if gamma is None:
            gamma = self.cht_gamma

        x_i = np.asarray(parents_snapshot[parent_idx], dtype=float)
        sigma_i = sigmas_snapshot[parent_idx]
        m = len(infeasible_offspring[0][2])      # number of constraints

        # ── Step 1: eigendecompose C_i ────────────────────────────────────
        C = A @ A.T
        C = 0.5 * (C + C.T)                       # symmetrise for numerical safety
        vp, P = np.linalg.eigh(C)                 # ascending eigenvalues
        vp = np.maximum(vp, 0.0)                  # any tiny negatives -> 0
        sqrt_vp = np.sqrt(vp)

        # ── Step 2: Mahalanobis pool weight per offspring ────────────────
        scale = np.sqrt(self.d_steps * n)
        n_off = len(infeasible_offspring)
        pool_w = np.zeros(n_off)
        steps  = np.zeros((n_off, n))
        for k, (_donor_idx, x_off, _g_off) in enumerate(infeasible_offspring):
            x_off = np.asarray(x_off, dtype=float)
            steps[k] = x_off - x_i
            d_M = np.linalg.norm(invCholesky @ steps[k]) / sigma_i
            pool_w[k] = np.exp(-d_M / scale)

        # ── Steps 3–4: per-constraint shrinkage along eigenvectors ───────
        sqrt_vp_new = sqrt_vp.copy()
        for j in range(m):
            # Find offspring that violate constraint j (g_off[j] > 0).
            violators = []
            for k, (_donor_idx, _x_off, g_off) in enumerate(infeasible_offspring):
                gj = g_off[j]
                # +inf marks "physical constraint short-circuited because box
                # violated" — those still count as violations of j, but we
                # rank by the box violation magnitude instead via pool_w.
                if np.isfinite(gj) and gj > 0.0:
                    violators.append((k, gj))
                elif np.isinf(gj) and gj > 0.0:
                    violators.append((k, np.finfo(float).max))

            if not violators:
                continue

            # Sort worst-violator-first (largest g_j gets the highest rank
            # weight w_1j per Chocat eq. 13).
            violators.sort(key=lambda kt: -kt[1])
            mu_cj = len(violators)

            # Chocat eq. 13: w_ij = (ln(mu_cj + 1) - ln(rank+1))
            # / (mu_cj·ln(mu_cj+1) - sum_k ln(k+1))
            # Reduces to a logarithmically-decaying weight; sums to 1.
            ranks = np.arange(mu_cj)
            num   = np.log(mu_cj + 1) - np.log(ranks + 1)
            denom = num.sum()
            if denom <= 0:
                continue
            w_rank = num / denom

            # Modulate by pool weight (Adaptation B): an offspring far
            # from this parent in C_i's metric contributes less.
            w = w_rank * np.array([pool_w[kt[0]] for kt in violators])
            w_sum = w.sum()
            if w_sum <= 0:
                continue

            # For each eigenvector, shrink the corresponding eigenvalue
            # by the weighted projection of the unit step direction onto
            # that eigenvector.
            #
            # Why the unit-direction normalisation
            # ------------------------------------
            # Chocat uses raw Proj_{e_i}[z_l - m] in their eq. 12, but
            # they have one global (m, C) so there is no Adaptation-B
            # pooling to worry about.  Here, with per-parent Cᵢ and pool
            # weighting, raw projections grow linearly with step
            # magnitude while pool_w decays only exponentially: distant
            # offspring would dominate the update for any moderate
            # distance, defeating the "parent learns most from its own
            # neighbourhood" intent.  Using the unit step direction
            # bounds projection magnitude in [0, 1] so pool_w controls
            # the absolute contribution.  The "bigger violations get
            # more signal" property is already captured by w_rank from
            # eq. 13 (worst violator gets the largest rank weight).
            for i_eig in range(n):
                e = P[:, i_eig]
                proj_sum = 0.0
                for wk, (k, _gj) in zip(w, violators):
                    step = steps[k]
                    step_norm = np.linalg.norm(step)
                    if step_norm < 1e-15:
                        continue
                    proj_sum += wk * abs(e @ step) / step_norm
                # Numerical floor: never drive an eigenvalue below
                # epsilon * its prior magnitude.  Without this, large
                # projections can push sqrt_vp_new[i_eig] negative,
                # which would make S not PSD.
                sqrt_vp_new[i_eig] = max(
                    sqrt_vp_new[i_eig] - gamma * proj_sum * sqrt_vp[i_eig],
                    1e-10 * max(sqrt_vp[i_eig], 1e-30),
                )

        # If nothing changed, skip the rest.
        if np.allclose(sqrt_vp_new, sqrt_vp):
            return A, invCholesky

        # ── Step 5: hypervolume-preserving rescale (eq. 11) ──────────────
        vp_new = sqrt_vp_new ** 2
        # log-space division avoids overflow / divide-by-zero when any
        # eigenvalue is at the numerical floor.
        eps = 1e-300
        log_factor = (np.sum(np.log(vp + eps))
                      - np.sum(np.log(vp_new + eps))) / n
        S = (P * vp_new) @ P.T
        S = 0.5 * (S + S.T)
        C_new = np.exp(log_factor) * S
        C_new = 0.5 * (C_new + C_new.T)

        # ── Step 6: re-Cholesky with PSD-failure fallback ───────────────
        try:
            A_new = np.linalg.cholesky(C_new)
        except np.linalg.LinAlgError:
            return A, invCholesky
        invCholesky_new = scipy.linalg.solve_triangular(
            A_new, np.eye(n), lower=True,
        )
        return A_new, invCholesky_new

    def _rankOneUpdate(self, invCholesky, A, alpha, beta, v):
        """Rank-one update of the Cholesky factor and its inverse."""
        w = np.dot(invCholesky, v)

        if w.max() > 1e-20:
            w_inv   = np.dot(w, invCholesky)
            norm_w2 = np.sum(w ** 2)
            a       = np.sqrt(alpha)
            root    = np.sqrt(1 + beta / alpha * norm_w2)
            b       = a / norm_w2 * (root - 1)

            A = a * A + b * np.outer(v, w)
            invCholesky  = (
                1.0 / a * invCholesky
                - b / (a ** 2 + a * b * norm_w2) * np.outer(w, w_inv)
            )

        return invCholesky, A

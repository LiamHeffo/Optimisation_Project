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

# pycma's Augmented Lagrangian (Atamna et al 2017 / Dufossé & Hansen 2020).
# Imported at module level (cheap), but only instantiated when
# sim_type == 'CHT_AL'.  The class is a single-objective construct; this
# strategy adapts it to the multi-objective setting by adding the same
# scalar AL penalty to every objective component (uniform translation in
# objective space, which preserves Pareto dominance among same-AL
# individuals while pushing infeasibles uniformly worse).
from cma.constraints_handler import AugmentedLagrangian


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

        # CHT diagnostics: per-call records appended by _chtCovarianceUpdate
        # and drained by main.py once per generation.  Each entry is one dict
        # describing one (parent_idx, phase) invocation.  See
        # _chtCovarianceUpdate for the schema.
        self.cht_diag_buffer = []

        # Lineage tracking: each individual carries a stable, monotonically
        # increasing ID from the moment it is created.  Plotting per-lineage
        # gives smooth trajectories that end when the lineage is displaced
        # by selection — vastly more interpretable than per-slot trajectories
        # which silently switch identities at every reassignment event.
        for i, p in enumerate(self.parents):
            p._lineage_id = i
        self._next_lineage_id = len(self.parents)

        # ─────────────────────────────────────────────────────────────────
        # Augmented Lagrangian state (CHT_AL sim_type only)
        # ─────────────────────────────────────────────────────────────────
        # AL is layered ON TOP of CHT: CHT shrinks the covariance using the
        # 18-element box+physical g vector; AL adapts a Lagrangian on the
        # 1-element g_AL = delta_vs1 - al_tol.  The two never share data.
        #
        # set_algorithm(3) selects the g-CDF-based mu-update (muplus3 /
        # muminus3), which has no scalar-f dependency — making it usable
        # in multi-objective settings where there is no single fitness.
        # set_dufosse2020() then overrides chi_domega = 2^(1/sqrt(n)) and
        # k1 = 10 per Section 4.2 of Dufossé & Hansen 2020.
        self.al_tol = float(params.get("al_tol", 100.0))
        if self.sim_type == 'CHT_AL':
            self.al = AugmentedLagrangian(self.dim, equality=False)
            self.al.set_algorithm(3)
            self.al.set_dufosse2020()
            # Quiet pycma's internal logging — we maintain our own
            # per-generation diagnostics (see al_diag_buffer below).
            self.al.logging = 0
        else:
            self.al = None
        # Per-generation AL diagnostic records (appended by update_al,
        # drained by main.py mirroring the CHT pattern).
        self.al_diag_buffer = []

    # ─────────────────────────────────────────────────────────────────────────
    # Augmented Lagrangian helpers (CHT_AL sim_type only)
    # ─────────────────────────────────────────────────────────────────────────

    def _al_penalty(self, g_al):
        """Return the scalar AL penalty Σₖ AL(g_AL_k) for one individual.

        For our m=1 case this is a single term.  Returns 0.0 cleanly when
        AL is disabled or coefficients have not been bootstrapped yet, so
        callers can use it unconditionally.

        Why we gate on ``lam is not None`` and not ``al.is_initialized``:
        pycma's ``is_initialized`` flag only flips to True when the
        empirical sign-average of g (across recent calls) is balanced —
        which can take ``2 + n`` generations to satisfy in the
        all-infeasible regime.  But ``set_coefficients`` populates lam/mu
        immediately on the first call, and pycma's ``al(g)`` callable
        returns a real penalty as soon as those exist.  Gating on
        ``is_initialized`` would silently zero out the penalty for the
        first several generations of a real run.
        """
        if (self.al is None
                or self.al.lam is None
                or g_al is None):
            return 0.0
        return float(sum(self.al(np.asarray(g_al, dtype=float))))

    def init_al(self, F_pop, G_AL_pop):
        """Bootstrap the AL coefficients from one generation's worth of data.

        ``F_pop`` is a list of scalar fitness aggregates per individual
        (in MOO, we use ``sum(fitness.values)`` — pycma's
        ``set_coefficients`` only uses ``iqr(F)`` as a magnitude scale,
        so any reasonable scalar surrogate works).  ``G_AL_pop`` is a
        list of g_al vectors (each length 1 for our case).

        No-op if AL is disabled.  Idempotent — pycma's
        ``set_coefficients`` skips work once coefficients are fully set.
        """
        if self.al is None:
            return
        if len(F_pop) == 0 or len(G_AL_pop) == 0:
            return
        self.al.set_coefficients(np.asarray(F_pop, dtype=float),
                                 np.asarray(G_AL_pop, dtype=float))

    def update_al(self, F_proxy_scalar, g_al_proxy):
        """Per-generation update of γ and μ from the parent-centroid proxy.

        With ``set_algorithm(3)`` the μ-update is g-only (muplus3 /
        muminus3 use the empirical CDF of recent g values).  ``F_proxy_scalar``
        is therefore unused inside pycma's update branch we selected, but
        we still pass it through so the ``self.f`` cached state stays
        consistent for any future algorithm switch.
        """
        if self.al is None or not self.al.is_initialized:
            return
        self.al.update(float(F_proxy_scalar),
                       np.asarray(g_al_proxy, dtype=float))
        # Snapshot for diagnostics — captured here rather than at the
        # call site so the format stays consistent across cmaes.py
        # internals.  ``lam`` and ``mu`` are length-m numpy arrays.
        self.al_diag_buffer.append({
            "g_al_proxy":     np.asarray(g_al_proxy, dtype=float).tolist(),
            "f_proxy_scalar": float(F_proxy_scalar),
            "lam":            self.al.lam.tolist(),
            "mu":             self.al.mu.tolist(),
            "al_pen_proxy":   self._al_penalty(g_al_proxy),
            "count":          int(self.al.count),
        })

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
                individuals[-1]._lineage_id = self._next_lineage_id
                self._next_lineage_id += 1

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
                individuals[-1]._lineage_id = self._next_lineage_id
                self._next_lineage_id += 1

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
                # σ is now logged per-generation to strategy_per_gen.csv;
                # see _append_strategy_per_gen_row in main.py.
                # print(f"sigmas: {sigmas[i]}")

                # CHT covariance shrinkage (Chocat 2015) — slot in BEFORE
                # rank-mu_succ and rank-one so subsequent updates operate
                # on the constraint-aware geometry.  Matches Chocat
                # Algorithm 3 step-3-2 -> step-3-4 ordering.  Only fires
                # when the sim_type opts in AND there are infeasibles.
                if self.sim_type == 'CovarianceCHT' and infeasible_pool:
                    # The C being updated belongs to chosen[i] — the new
                    # occupant of slot i.  Record by *its* lineage so the
                    # diagnostic trace tracks the right individual.
                    A[i], invCholesky[i] = self._chtCovarianceUpdate(
                        A[i], invCholesky[i], p_idx,
                        parents_snapshot, sigmas_snapshot, infeasible_pool,
                        diag_phase="post_eval",
                        lineage_id=getattr(ind, "_lineage_id", None),
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
    # CHT resample loop (Chocat 2015 Algorithm 3 step 3-2)
    # ─────────────────────────────────────────────────────────────────────────

    def resample_infeasibles(self, population, feasibility_check,
                              max_iterations=5):
        """Iteratively shrink covariance and resample infeasible offspring.

        Implements Chocat 2015 Algorithm 3 step 3-2's resample branch:
        when a generation produces infeasible offspring, the constraint
        violation directions are fed into the CHT covariance update,
        which shrinks each parent's Cᵢ along those directions.  The
        infeasible slots are then resampled from the new (tighter)
        distribution.  Repeat until all feasible or max_iterations is
        reached.

        Each iteration's CHT operates on a *fresh* set of offspring (the
        previous infeasibles were resampled), so there is no
        double-counting of constraint signal across iterations.  After
        this loop returns, update()'s post-evaluation CHT call will
        operate on whatever infeasibles remain — also a unique signal,
        not a re-application of the loop's data.

        Parameters
        ----------
        population : list of Individual
            Output of generate(): one offspring per parent (lambda_=mu).
            Mutated in place.
        feasibility_check : callable(ind) -> (feasible_bool, g_vector)
            Cheap check that does NOT call SPARK / PITOT3.  Closure over
            `bounds` provided by the caller, so the strategy stays
            domain-agnostic about the contents of g.
        max_iterations : int, default 5
            Hard cap on resample passes per generation.  At λ=12 and
            modest cht_gamma, 1–3 iterations typically suffice; the cap
            bounds wall-clock cost in pathological cases.

        Returns
        -------
        n_iterations : int
            How many CHT-and-resample passes were performed before
            either every offspring became feasible or the cap was hit.
            Useful as a per-generation diagnostic.

        Side effects
        ------------
        - self.A[i] and self.invCholesky[i] are mutated by each
          iteration's CHT call (in place).
        - Each individual in `population` has ind._g and ind._feasible
          set on every iteration — the final values reflect the post-
          loop state.
        """
        n = self.dim

        # First pass: tag every offspring with feasibility info so the
        # caller can rely on _g / _feasible regardless of whether we
        # actually iterate.
        for ind in population:
            feasible, g = feasibility_check(ind)
            ind._g = g
            ind._feasible = feasible

        for iteration in range(max_iterations):
            infeasible_slots = [
                i for i, ind in enumerate(population)
                if not ind._feasible
            ]
            if not infeasible_slots:
                return iteration

            # Build the CHT pool from the CURRENT infeasibles.
            infeasible_pool = [
                (population[i]._ps[1], np.array(population[i]), population[i]._g)
                for i in infeasible_slots
            ]

            # Snapshot parent positions and sigmas (they don't change in
            # the loop, but _chtCovarianceUpdate expects snapshots).
            parents_snapshot = [np.array(p) for p in self.parents]
            sigmas_snapshot  = list(self.sigmas)

            # Apply CHT to every parent's Cholesky factor.  Adaptation-B
            # pooling: each parent learns from every infeasible across
            # the swarm, weighted by Mahalanobis closeness.
            for parent_idx in range(len(self.parents)):
                self.A[parent_idx], self.invCholesky[parent_idx] = (
                    self._chtCovarianceUpdate(
                        self.A[parent_idx], self.invCholesky[parent_idx],
                        parent_idx, parents_snapshot, sigmas_snapshot,
                        infeasible_pool,
                        diag_phase=f"resample_iter_{iteration}",
                        lineage_id=getattr(
                            self.parents[parent_idx], "_lineage_id", None,
                        ),
                    )
                )

            # Resample only the infeasible slots from the now-tighter
            # distribution.  Mutate the existing Individual objects in
            # place so ind_number / _ps / DEAP fitness slot survive.
            for i in infeasible_slots:
                p_idx = population[i]._ps[1]
                z = np.random.randn(n)
                mutation = self.sigmas[p_idx] * np.dot(self.A[p_idx], z)
                new_x = self.parents[p_idx] + mutation
                for k in range(n):
                    population[i][k] = float(new_x[k])
                # Re-check feasibility for this slot.
                feasible, g = feasibility_check(population[i])
                population[i]._g = g
                population[i]._feasible = feasible

        # Cap reached; return how many iterations were spent.
        return max_iterations

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

        In CHT_AL mode the Pareto sort and HV indicator operate on the
        AL-augmented fitness — i.e. f_i + Σₖ AL(g_AL_k) for every
        objective component — so infeasibles (in the AL sense) are biased
        worse but the Pareto structure among same-AL individuals is
        preserved.  The augmentation is implemented by monkey-swapping
        ``ind.fitness.values`` for the duration of the sort and restoring
        afterwards via try/finally.  This pattern keeps DEAP's
        ``sortLogNondominated`` and the HV indicator untouched.

        For all other sim_types this is a no-op; the original raw fitness
        is sorted exactly as before.
        """
        # ── AL augmentation: enter ───────────────────────────────────────
        # Build a snapshot of (id(ind) -> original fitness.values) so we
        # can restore even if downstream code throws.  We only augment in
        # CHT_AL mode AND only after AL has been initialised (the very
        # first generation runs raw, since lam=0 and mu=0 make AL == 0
        # anyway and the bootstrapping happens after gen 0 selection).
        original_fitness = {}
        if self.sim_type == 'CHT_AL' and self.al is not None and self.al.is_initialized:
            for ind in candidates:
                if not ind.fitness.valid:
                    continue
                g_al = getattr(ind, "_g_al", None)
                pen = self._al_penalty(g_al)
                if pen == 0.0:
                    continue
                original_fitness[id(ind)] = ind.fitness.values
                ind.fitness.values = tuple(v + pen for v in ind.fitness.values)

        try:
            return self._select_pareto(candidates)
        finally:
            # ── AL augmentation: exit ────────────────────────────────────
            # Restore original fitness on every path (success or exception).
            # Without this, downstream HV plots and CSV writers would log
            # AL-shifted values as if they were raw — wrong.
            for ind in candidates:
                key = id(ind)
                if key in original_fitness:
                    ind.fitness.values = original_fitness[key]

    def _select_pareto(self, candidates):
        """The original Pareto + HV selection, factored out so _select can
        wrap it with the AL augmentation.

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
                              infeasible_offspring, gamma=None,
                              diag_phase=None, lineage_id=None):
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
            # No work to do; emit a no-op record so the per-gen totals stay
            # honest about how often CHT was called with an empty pool.
            self._record_cht_diag(parent_idx, diag_phase, n_violators=0,
                                   lineage_id=lineage_id)
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
        # Snapshot pre-update spectrum for the diagnostics record.  np.eigh
        # returns eigenvalues ascending, so vp[-1] is largest.
        vp_before  = vp.copy()
        principal_axis_before = P[:, -1].copy()

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
        per_constraint_active_count = [0] * m   # diagnostic: who drove shrinkage
        for j in range(m):
            # Find offspring that violate constraint j (g_off[j] > 0).
            #
            # Note on +inf entries: src/problem/feasibility.py sets the
            # six physical-space constraints (j=0..5) to +inf whenever
            # ANY box bound is violated, because the un-transformation
            # has driver_p-coupled divisions that aren't well-defined
            # outside the box.  An earlier version of this loop treated
            # those +inf entries as genuine violations of j=0..5, which
            # caused the SAME box-violating offspring's step direction
            # to be shrunk seven times (once for each cascaded j=0..5
            # entry plus once for the actual box constraint), producing
            # ~5x over-shrinkage of the corresponding eigenvalue.  The
            # actual box constraint already carries the directional
            # signal — we don't need the cascade to amplify it — so we
            # skip +inf entries.
            violators = []
            for k, (_donor_idx, _x_off, g_off) in enumerate(infeasible_offspring):
                gj = g_off[j]
                if np.isfinite(gj) and gj > 0.0:
                    violators.append((k, gj))

            per_constraint_active_count[j] = len(violators)
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

        # If nothing changed, skip the rest.  Still record a diag entry so
        # we can see how often shrinkage was a no-op.
        if np.allclose(sqrt_vp_new, sqrt_vp):
            self._record_cht_diag(
                parent_idx, diag_phase,
                n_violators=n_off,
                vp_before=vp_before, vp_after=vp_before,
                principal_axis_before=principal_axis_before,
                principal_axis_after=principal_axis_before,
                pool_w=pool_w, steps=steps, infeasible_offspring=infeasible_offspring,
                per_constraint_active_count=per_constraint_active_count,
                shrink_applied=False, psd_fallback=False,
                lineage_id=lineage_id,
            )
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
            self._record_cht_diag(
                parent_idx, diag_phase,
                n_violators=n_off,
                vp_before=vp_before, vp_after=vp_before,
                principal_axis_before=principal_axis_before,
                principal_axis_after=principal_axis_before,
                pool_w=pool_w, steps=steps, infeasible_offspring=infeasible_offspring,
                per_constraint_active_count=per_constraint_active_count,
                shrink_applied=False, psd_fallback=True,
                lineage_id=lineage_id,
            )
            return A, invCholesky
        invCholesky_new = scipy.linalg.solve_triangular(
            A_new, np.eye(n), lower=True,
        )

        # Post-update spectrum (the rescaled C_new) for the diag record.
        vp_after, P_after = np.linalg.eigh(C_new)
        vp_after = np.maximum(vp_after, 0.0)
        principal_axis_after = P_after[:, -1]

        self._record_cht_diag(
            parent_idx, diag_phase,
            n_violators=n_off,
            vp_before=vp_before, vp_after=vp_after,
            principal_axis_before=principal_axis_before,
            principal_axis_after=principal_axis_after,
            pool_w=pool_w, steps=steps, infeasible_offspring=infeasible_offspring,
            per_constraint_active_count=per_constraint_active_count,
            shrink_applied=True, psd_fallback=False,
            lineage_id=lineage_id,
        )
        return A_new, invCholesky_new

    def _record_cht_diag(self, parent_idx, phase, n_violators,
                          vp_before=None, vp_after=None,
                          principal_axis_before=None, principal_axis_after=None,
                          pool_w=None, steps=None, infeasible_offspring=None,
                          per_constraint_active_count=None,
                          shrink_applied=False, psd_fallback=False,
                          lineage_id=None):
        """Append one CHT diagnostic record to ``self.cht_diag_buffer``.

        Computes the derived Tier 1 / Tier 2 quantities (log-det,
        condition number, mean violation direction, principal-axis vs
        violation-direction angle) from the raw inputs.  Centralising the
        derivation here keeps the algebra in one place and the
        _chtCovarianceUpdate body readable.

        All array inputs are stored as plain Python lists so the buffer
        is JSON/CSV-friendly.
        """
        eps = 1e-300

        def _safe_logdet(vp):
            return float(np.sum(np.log(np.maximum(vp, eps)))) if vp is not None else None

        def _cond(vp):
            if vp is None or len(vp) == 0:
                return None
            vmax = float(np.max(vp))
            vmin = float(np.min(vp[vp > 0])) if np.any(vp > 0) else eps
            return vmax / max(vmin, eps)

        # Mean violation direction: weighted average of unit step vectors,
        # weighted by pool_w * max(g_off).  Captures "which direction did
        # the violations come from" so we can compare against the
        # post-CHT principal axis (Tier 2 mechanism check).
        mean_violation_direction = None
        violation_axis_angle_deg = None
        effective_pool_weight = None
        if (pool_w is not None and steps is not None
                and infeasible_offspring is not None and len(steps) > 0):
            effective_pool_weight = float(np.mean(pool_w))
            v_acc = np.zeros(self.dim)
            for k, (_donor_idx, _x_off, g_off) in enumerate(infeasible_offspring):
                step = steps[k]
                norm = np.linalg.norm(step)
                if norm < 1e-15:
                    continue
                # Severity: largest finite violation, fall back to 1.0 if
                # only +inf box-violations are present.
                finite_g = [g for g in g_off if np.isfinite(g) and g > 0]
                severity = max(finite_g) if finite_g else 1.0
                v_acc += pool_w[k] * severity * (step / norm)
            v_norm = np.linalg.norm(v_acc)
            if v_norm > 1e-15:
                mean_violation_direction = (v_acc / v_norm).tolist()
                if principal_axis_after is not None:
                    cos_t = float(np.clip(
                        np.abs(np.dot(v_acc / v_norm, principal_axis_after)),
                        0.0, 1.0,
                    ))
                    violation_axis_angle_deg = float(np.degrees(np.arccos(cos_t)))

        rec = {
            "phase":                   phase,
            "parent_idx":              int(parent_idx),
            "lineage_id":              int(lineage_id) if lineage_id is not None else None,
            "n_violators":             int(n_violators),
            "shrink_applied":          bool(shrink_applied),
            "psd_fallback":            bool(psd_fallback),
            "log_det_C_before":        _safe_logdet(vp_before),
            "log_det_C_after":         _safe_logdet(vp_after),
            "condition_number_before": _cond(vp_before),
            "condition_number_after":  _cond(vp_after),
            "min_eigenvalue_after":    float(np.min(vp_after)) if vp_after is not None else None,
            "max_eigenvalue_after":    float(np.max(vp_after)) if vp_after is not None else None,
            "eigenvalues_before":      vp_before.tolist() if vp_before is not None else None,
            "eigenvalues_after":       vp_after.tolist()  if vp_after  is not None else None,
            "principal_axis_before":   principal_axis_before.tolist() if principal_axis_before is not None else None,
            "principal_axis_after":    principal_axis_after.tolist()  if principal_axis_after  is not None else None,
            "mean_violation_direction": mean_violation_direction,
            "violation_axis_angle_deg": violation_axis_angle_deg,
            "effective_pool_weight":   effective_pool_weight,
            "per_constraint_active_count": list(per_constraint_active_count) if per_constraint_active_count is not None else None,
        }
        self.cht_diag_buffer.append(rec)

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

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

                if self.sim_type != 'Penalty':
                    s = time.time()
                    while True:
                        if not self.check_feasibility(new_individual)[0]:
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

        for i, ind in enumerate(chosen):
            t, p_idx = ind._ps
            if t == "o":
                psucc[i] = (1.0 - cp) * psucc[i] + cp
                sigmas[i] = sigmas[i] * np.exp((psucc[i] - ptarg) / (d * (1.0 - ptarg)))
                print(f"sigmas: {sigmas[i]}")

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
        """Select mu individuals from candidates using Pareto ranking + hypervolume."""
        if len(candidates) <= self.mu:
            return candidates, []

        pareto_fronts = tools.sortLogNondominated(candidates, len(candidates))

        chosen = []
        mid_front = None
        not_chosen = []
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
            ref = np.array([ind.fitness.wvalues for ind in candidates]) * -1
            ref = np.max(ref, axis=0) + 1

            for _ in range(len(mid_front) - k):
                idx = self.indicator(mid_front, ref=ref)
                not_chosen.append(mid_front.pop(idx))

            chosen += mid_front

        return chosen, not_chosen

    def _rankOneUpdate(self, invCholesky, A, alpha, beta, v):
        """Rank-one update of the Cholesky factor and its inverse."""
        w = np.dot(invCholesky, v)

        if w.max() > 1e-20:
            w_inv   = np.dot(w, invCholesky)
            norm_w2 = np.sum(w ** 2)
            a       = np.sqrt(alpha)
            root    = np.sqrt(1 + beta / alpha * norm_w2)
            b       = a / norm_w2 * (root - 1)

            A            = a * A + b * np.outer(v, w)
            invCholesky  = (
                1.0 / a * invCholesky
                - b / (a ** 2 + a * b * norm_w2) * np.outer(w, w_inv)
            )

        return invCholesky, A

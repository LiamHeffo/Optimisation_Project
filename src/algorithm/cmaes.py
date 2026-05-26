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


def _eval_schedule(schedule, current_gen, fallback):
    """Linearly interpolate a scheduled parameter at ``current_gen``.

    Two schedule forms are accepted:

    - **3-element** ``[start, end, n_gens]`` (legacy):
        linear from ``start`` at gen 0 to ``end`` at gen ``n_gens``,
        saturating at ``end`` afterwards.

    - **4-element** ``[start, end, start_gen, end_gen]``:
        constant at ``start`` until gen ``start_gen``, linear from
        ``start`` to ``end`` over ``[start_gen, end_gen]``, constant
        at ``end`` after.  Use for delayed-onset schedules — e.g.
        a tightening that only kicks in after the population has
        reached a feasible region, or a γ-decay that holds full
        strength during the early CHT-pull phase.

    ``schedule = None`` returns ``fallback`` (the static value).
    """
    if schedule is None:
        return fallback
    if len(schedule) == 3:
        start, end, n_gens = schedule
        start_gen, end_gen = 0, n_gens
    elif len(schedule) == 4:
        start, end, start_gen, end_gen = schedule
    else:
        raise ValueError(
            f"schedule must be a 3- or 4-element list, got {len(schedule)}: "
            f"{schedule}"
        )
    if current_gen < start_gen:
        return float(start)
    span = max(1, end_gen - start_gen)
    progress = min(1.0, max(0.0, (current_gen - start_gen) / span))
    return float(start * (1.0 - progress) + end * progress)


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
        # along violating directions.  The 0.5 prefactor (5× the
        # Arnold-Hansen default) was chosen empirically for the X2
        # constrained problem where the box+phys-feasible region is
        # narrow enough that the default shrinkage was too gentle to
        # keep up with offspring drift.
        #
        # ``cht_gamma=None`` is treated identically to "key missing" so
        # main.py can pass through whatever the YAML supplies (or omits)
        # without having to know the dimension-dependent default.
        _cht_gamma = params.get("cht_gamma")
        self.cht_gamma = (
            _cht_gamma if _cht_gamma is not None
            else 0.5 / (self.dim + 2.0)
        )

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

        # ─────────────────────────────────────────────────────────────────
        # Anti-degeneration feature toggles (each defaults to off)
        # ─────────────────────────────────────────────────────────────────
        # Sourced from the ``features`` dict in experiments.yaml.  All
        # default off so omission == legacy behaviour.  Each feature is
        # documented at its use-site; the toggles are listed here for a
        # single grep target.
        features = params.get("features", {}) or {}
        # A1: floor each eigenvalue at (factor * pre-CHT eigenvalue) per
        # call.  None ⇒ legacy 1e-10 floor (effectively unbounded shrink).
        self.cht_eigenvalue_floor = features.get("cht_eigenvalue_floor")
        # A2: after CHT shrinkage, pull eigenvalues toward their mean by
        # this factor.  None ⇒ no isotropy relaxation.
        self.cht_isotropy_alpha = features.get("cht_isotropy_alpha")
        # A3: linearly interpolate cht_gamma from start → end over
        # n_gens generations.  Schema: [start_gamma, end_gamma, n_gens].
        # None ⇒ static cht_gamma.
        self.cht_gamma_schedule = features.get("cht_gamma_schedule")
        # A4: when κ(C) exceeds this threshold, apply an isotropy
        # correction even if no infeasibles are present.  None ⇒ off.
        self.cht_kappa_trigger = features.get("cht_kappa_trigger")
        # B1: maintain self.al.lam ≥ factor * self.al.mu so the AL stays
        # engaged even after the centroid is comfortably feasible.
        # None ⇒ legacy clamp-to-zero behaviour.
        self.al_lam_floor_factor = features.get("al_lam_floor_factor")
        # B2: multiply self.al.mu by this factor each generation while
        # self.al.lam is clamped at zero.  None ⇒ no decay (mu frozen).
        self.al_mu_decay = features.get("al_mu_decay")
        # B3: linearly interpolate al_tol from start → end over n_gens.
        # Schema: [start_tol, end_tol, n_gens].  None ⇒ static al_tol.
        self.al_tol_schedule = features.get("al_tol_schedule")
        # C1: tolerance for is_feasible() inside the CHT resample loop.
        # Higher ⇒ more permissive (fewer offspring re-sampled / fewer
        # CHT calls).  None or 0.0 ⇒ strict feasibility.
        self.cht_resample_tol = features.get("cht_resample_tol")
        # C2: when an offspring's design vector lands outside [1, 2]^n,
        # reflect it back into the box rather than letting the CHT take
        # the hit.  Box bounds only (physical constraints stay strict).
        self.box_reflective_repair = bool(features.get("box_reflective_repair", False))
        # C3: when an offspring was resampled by the CHT loop, do not
        # let it contribute to its donor parent's psucc / σ update.
        # Decouples σ adaptation from CHT-induced "successes".
        self.psucc_exclude_resampled = bool(features.get("psucc_exclude_resampled", False))
        # Group-C reversal toggle: treat sentinel offspring in the
        # not-chosen branch as honest failures rather than skipping the
        # update entirely.  Reframes SPARK/PITOT3 failures as information
        # about local feasibility (the region around the design is
        # unsimulable) rather than as numerical artefacts to be filtered
        # out.  When True, sentinel not-chosen offspring drive psucc
        # down and σ down, restoring the pre-Fix-S3 behaviour on the
        # failure side only.  Chosen-branch Fix-S3 (no "+" signal from
        # sentinel successes) remains unconditionally — a sentinel that
        # somehow survives selection is still a numerical artefact, not
        # a real success.  Default False keeps existing Fix-S3 behaviour.
        self.psucc_sentinel_as_failure = bool(features.get("psucc_sentinel_as_failure", False))
        # F1: floor each parent's σ when both the AL and CHT are quiet
        # for >= F1_SILENT_GENS_THRESHOLD consecutive generations.
        # None ⇒ no floor (legacy behaviour).
        self.sigma_floor_silent = features.get("sigma_floor_silent")
        # F3: ranking criterion within a non-dominated front.
        # "hv_contribution" preserves legacy Voss 2009 behaviour.
        # "crowding" switches to NSGA-II crowding-distance ranking,
        # which removes the knee-seeking bias of HV-contribution.
        self.selection_mode = features.get("selection", "hv_contribution")
        if self.selection_mode not in ("hv_contribution", "crowding"):
            raise ValueError(
                f"features.selection must be 'hv_contribution' or "
                f"'crowding', got {self.selection_mode!r}"
            )

        # Silent-regime counter and threshold for F1.  Counted by
        # update(); reset whenever either AL or CHT fires.
        self._gens_silent_count = 0
        self._gens_silent_threshold = 20

        # Generation counter used by schedule-based features (A3, B3).
        # Incremented by update() once per generation.
        self._generation = 0

        # ─────────────────────────────────────────────────────────────────
        # D1: External non-dominated archive
        # ─────────────────────────────────────────────────────────────────
        # Strategy elitism only retains the top-mu under the *current*
        # selection criterion — front members that were once Pareto-
        # optimal can be displaced when the front advances or selection
        # criteria drift (e.g. AL penalty state changes).  The archive
        # remembers every individual that has ever sat on the non-dom
        # front in *raw* objective space, capped and pruned by crowding
        # distance so the archive preserves spread rather than knee bias.
        #
        # Updated once per generation at the end of update().  Flushed
        # to disk via flush_archive_to_csv() once at end of run.
        self.archive_cap = int(params.get("archive_cap", 100))
        self.external_archive = []

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

    def current_al_tol(self):
        """Return the AL tolerance for the current generation.

        Feature B3 (al_tol_schedule): linearly interpolate al_tol from
        start → end across n_gens generations.  Schema:
        ``[start_tol, end_tol, n_gens]`` in features dict.  Saturates at
        ``end_tol`` after ``self._generation >= n_gens``.

        None ⇒ static ``self.al_tol`` (the legacy single value).
        Called from main.py once per generation to retag each offspring's
        ``ind.al_tol`` before evaluation.
        """
        return _eval_schedule(
            self.al_tol_schedule, self._generation, self.al_tol,
        )

    def update_al(self, F_proxy_scalar, g_al_proxy, proxy_stats=None):
        """Per-generation update of γ and μ from the parent-centroid proxy.

        With ``set_algorithm(3)`` the μ-update is g-only (muplus3 /
        muminus3 use the empirical CDF of recent g values).  ``F_proxy_scalar``
        is therefore unused inside pycma's update branch we selected, but
        we still pass it through so the ``self.f`` cached state stays
        consistent for any future algorithm switch.

        Gating rationale: we used to short-circuit on
        ``not self.al.is_initialized``, but that flag stays False until
        pycma's empirical sign-average is balanced — which can take many
        generations in the all-infeasible bootstrap regime.  Skipping the
        update during that window starves pycma's ``g_history`` deque of
        data (the CDF-based μ-adaptation reads from it), and also leaves
        the al_diag_buffer empty so no CSV row ever gets written.  The
        right gate is ``self.al.lam is None``: once ``set_coefficients``
        has populated lam/mu, ``al.update`` is safe — pycma handles its
        own internal short-circuits when mu is still zero.
        """
        if self.al is None or self.al.lam is None:
            return
        self.al.update(float(F_proxy_scalar),
                       np.asarray(g_al_proxy, dtype=float))

        # Feature B1 (al_lam_floor_factor): prevent λ from clamping to
        # zero once the centroid drifts inside the feasible region.
        # pycma's ``update()`` sets λ = max(λ + μ·g/dgamma, 0).  At
        # convergence λ→0 and the AL contributes nothing to selection
        # for the rest of the run.  This feature enforces
        # λ ≥ factor·μ, so a small attractive pressure toward the
        # constraint boundary persists indefinitely.  None ⇒ legacy
        # clamp-to-zero.
        if self.al_lam_floor_factor is not None and self.al.mu is not None:
            for k in range(len(self.al.lam)):
                lam_floor = self.al_lam_floor_factor * self.al.mu[k]
                if self.al.lam[k] < lam_floor:
                    self.al.lam[k] = lam_floor

        # Feature B2 (al_mu_decay): when λ is at (or below) its clamp
        # value, multiply μ by this factor each generation.  Lets the
        # penalty "forget" any early-run peak — otherwise μ stays
        # frozen at its maximum and any future re-engagement of the
        # AL fires with disproportionate force.  Skipped when B1 keeps
        # λ above the threshold.  None ⇒ μ frozen (legacy).
        if self.al_mu_decay is not None and self.al.mu is not None:
            # Threshold: legacy behaviour treats λ==0 as "clamped"; with
            # B1 enabled we use the feature's own floor as the marker.
            if self.al_lam_floor_factor is not None:
                threshold = self.al_lam_floor_factor * self.al.mu  # array
                clamped = np.asarray(self.al.lam) <= np.asarray(threshold) + 1e-12
            else:
                clamped = np.asarray(self.al.lam) < 1e-12
            for k in range(len(self.al.mu)):
                if clamped[k]:
                    self.al.mu[k] *= self.al_mu_decay

        # Snapshot for diagnostics — captured here rather than at the
        # call site so the format stays consistent across cmaes.py
        # internals.  ``lam`` and ``mu`` are length-m numpy arrays.
        # Always appended (no is_initialized gate), so the al_per_gen.csv
        # captures the full lam/mu trajectory including the bootstrap
        # window where they may legitimately be zero.
        #
        # Diag-2: include per-gen population g_al statistics (min/max/
        # std/n_feasible_parents) so we can post-hoc evaluate whether
        # the mean proxy is masking bimodality.
        # Diag-3: include pen_to_f_ratio = (μ_AL · max(|g|)²) / max(|F|)
        # so we can see when the penalty starts dominating f in absolute
        # terms (sentinel-driven scaling drift).
        stats = proxy_stats or {}
        g_max_abs = float(np.max(np.abs(np.asarray(g_al_proxy, dtype=float))))
        F_abs_max = stats.get("F_proxy_abs_max")
        mu_arr = np.asarray(self.al.mu, dtype=float)
        if F_abs_max is not None and F_abs_max > 0 and len(mu_arr) > 0:
            pen_to_f = float(np.max(mu_arr) * (g_max_abs ** 2) / F_abs_max)
        else:
            pen_to_f = float("nan")

        self.al_diag_buffer.append({
            "g_al_proxy":         np.asarray(g_al_proxy, dtype=float).tolist(),
            "f_proxy_scalar":     float(F_proxy_scalar),
            "lam":                self.al.lam.tolist(),
            "mu":                 self.al.mu.tolist(),
            "al_pen_proxy":       self._al_penalty(g_al_proxy),
            "count":              int(self.al.count),
            "is_initialized":     bool(self.al.is_initialized),
            "g_al_min":           stats.get("g_al_min"),
            "g_al_max":           stats.get("g_al_max"),
            "g_al_std":           stats.get("g_al_std"),
            "n_feasible_parents": stats.get("n_feasible_parents"),
            "pen_to_f_ratio":     pen_to_f,
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

                # Feature C2 (box_reflective_repair): reflect coordinates
                # back into [1, 2]^n before evaluation, instead of letting
                # the CHT take the hit.  Preserves the (σ, A) information
                # that produced the offspring while keeping it inside the
                # normalised box.  The while-loop handles the rare case
                # where reflection itself overshoots (very large σ); in
                # practice 1–2 iterations are sufficient.  Box bounds
                # only — physical constraints stay strict so the CHT
                # still adapts to them.
                if self.box_reflective_repair:
                    for k in range(self.dim):
                        reflections = 0
                        while (new_individual[k] < 1.0
                               or new_individual[k] > 2.0) and reflections < 8:
                            if new_individual[k] < 1.0:
                                new_individual[k] = 2.0 - new_individual[k]
                            elif new_individual[k] > 2.0:
                                new_individual[k] = 4.0 - new_individual[k]
                            reflections += 1

                # CovarianceCHT (and CHT_AL, which layers AL on top of CHT)
                # replace the repair while-loop: infeasible offspring pass
                # through and feed the CHT covariance update (Phase 3).
                # Without this guard, CHT_AL would enter the loop, call
                # crossover() — which has no CHT_AL branch — get the
                # individual back unchanged, and spin forever.
                # Penalty mode also bypasses repair (its handler is in
                # evaluate.py).
                if self.sim_type not in ('Penalty', 'CovarianceCHT', 'CHT_AL'):
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
        # Snapshot the full candidate pool BEFORE selection mutates state.
        # Used at the end of update() to refresh the external archive
        # (D1) — we want every evaluated individual considered for
        # archive admission, not just the selected parents.
        archive_candidates_snapshot = list(population) + list(self.parents)

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
                # Feature C3 (psucc_exclude_resampled): an offspring that
                # only became feasible after CHT resampling shouldn't be
                # rewarded with a psucc / σ increment — its "success" is
                # an artefact of the CHT-tightened distribution, not of
                # the donor parent's σ.  Skip both the σ-up signal and
                # the rank-one update below.  The CHT itself still
                # consumes the resample data via the infeasible pool.
                #
                # Fix-S3: sentinel-aware psucc.  An offspring whose
                # heavy evaluators (PITOT3 or SPARK) failed carries a
                # numerical-failure signal, not an objective signal —
                # neither rewarding it (psucc up) nor punishing it
                # (psucc down) reflects what σ-adaptation should track.
                # Skip the donor's psucc / σ update entirely.
                #
                # Reading the flags set in main.py after evaluation,
                # rather than re-detecting here, keeps the sentinel
                # definition consistent across filter sites (and
                # correctly catches PITOT3-only sentinels whose fitness
                # is real but g_al is the sentinel value).
                is_sentinel = (
                    getattr(ind, "_pitot3_sentinel", False)
                    or getattr(ind, "_spark_sentinel", False)
                )
                skip_psucc = (
                    is_sentinel
                    or (self.psucc_exclude_resampled
                        and getattr(ind, "_resampled", False))
                )
                if not skip_psucc:
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
                if self.sim_type in ('CovarianceCHT', 'CHT_AL') and infeasible_pool:
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

                # Feature A4 (cht_kappa_trigger): once the condition
                # number of C_i exceeds the threshold, re-isotropise
                # toward a less-anisotropic shape — even if there are
                # no current infeasibles to feed the regular CHT.
                # Acts as an escape valve for anisotropy lock-in.
                if (self.cht_kappa_trigger is not None
                        and self.sim_type in ('CovarianceCHT', 'CHT_AL')):
                    A[i], invCholesky[i] = self._chtIsotropyCorrection(
                        A[i], invCholesky[i],
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

                # Same C3 gate for the global per-parent state update.
                if not skip_psucc:
                    self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx] + cp
                    self.sigmas[p_idx] = self.sigmas[p_idx] * np.exp(
                        (self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg))
                    )

        for ind in not_chosen:
            t, p_idx = ind._ps
            if t == "o":
                # Feature C3 also applies to the failure side: a
                # resampled-then-dominated offspring shouldn't shrink
                # the donor's σ either.  CHT covariance has already
                # consumed its constraint signal — that's enough.
                #
                # Fix-S3 (failure branch): by default, sentinel offspring
                # should not shrink the donor's σ — both PITOT3 and SPARK
                # encode "numerical failure" not "honest objective
                # failure", and skipping breaks the sentinel trap that
                # otherwise locks σ → 0 around failure regions.
                #
                # Toggle psucc_sentinel_as_failure inverts this: when
                # True, sentinels are treated as honest failures and
                # contribute the psucc decay + σ shrinkage below.  The
                # interpretation is that an unsimulable design is itself
                # information about local feasibility — shrinking σ
                # away from that region is the correct CMA response.
                if (getattr(ind, "_pitot3_sentinel", False)
                        or getattr(ind, "_spark_sentinel", False)):
                    if not self.psucc_sentinel_as_failure:
                        continue
                if (self.psucc_exclude_resampled
                        and getattr(ind, "_resampled", False)):
                    continue
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

        # Increment the generation counter once per update() invocation.
        # Schedule-based features (A3 cht_gamma_schedule, B3 al_tol_schedule)
        # read this to interpolate their parameters over time.  Counted
        # here rather than in main.py so the strategy is self-contained.
        self._generation += 1

        # ── F1: σ floor in silent regime ─────────────────────────────────
        # Track whether this generation was "silent" — neither AL nor
        # CHT did any work — and clamp σ once the silent streak has
        # reached the threshold.  The clamp prevents σ from collapsing
        # below sigma_floor_silent purely because psucc keeps dropping
        # in the absence of strict-Pareto offspring.  Direct attack on
        # the σ collapse component of the diversity-degeneration story.
        al_silent  = (self.al is None
                      or self.al.lam is None
                      or all(float(l) == 0.0 for l in self.al.lam))
        cht_silent = (len(infeasible_pool) == 0)
        if al_silent and cht_silent:
            self._gens_silent_count += 1
        else:
            self._gens_silent_count = 0

        if (self.sigma_floor_silent is not None
                and self._gens_silent_count >= self._gens_silent_threshold):
            floor = float(self.sigma_floor_silent)
            for k in range(len(self.sigmas)):
                if self.sigmas[k] < floor:
                    self.sigmas[k] = floor

        # ── D1: External archive refresh ─────────────────────────────────
        # Use the pre-selection snapshot so individuals that didn't
        # survive selection (but were Pareto-optimal at evaluation time)
        # still get a chance at archive admission.
        self._update_archive(archive_candidates_snapshot)

    # ─────────────────────────────────────────────────────────────────────────
    # External non-dominated archive (D1)
    # ─────────────────────────────────────────────────────────────────────────

    def _update_archive(self, candidates):
        """Refresh the external non-dominated archive with new candidates.

        Each candidate is admitted only if it has a valid fitness AND
        ``_feasible`` is True (we don't want SPARK/PITOT3 sentinel
        failures polluting the archive).  Non-domination is recomputed
        across (archive ∪ new entries) on *raw* fitness — the archive
        should reflect what we have actually discovered in objective
        space, independent of the AL coefficient state at the time of
        capture.  When size exceeds archive_cap, prune by crowding
        distance so the archive retains spread rather than knee bias.
        """
        new_entries = []
        for ind in candidates:
            if not getattr(ind, "_feasible", True):
                continue
            if not ind.fitness.valid:
                continue
            # D1 + sentinel filter: archive only "trustworthy" points —
            # both heavy evaluators succeeded.  PITOT3-only sentinels
            # have real fitness but unknown constraint state; SPARK
            # sentinels have fitness at the nadir.  Neither belongs in
            # an archive of "good points we have ever found".
            if (getattr(ind, "_pitot3_sentinel", False)
                    or getattr(ind, "_spark_sentinel", False)):
                continue
            g_al = getattr(ind, "_g_al", None)
            new_entries.append({
                "gen_found":  self._generation,
                "design":     [float(x) for x in ind],
                "fitness":    tuple(float(v) for v in ind.fitness.values),
                "g_al":       ([float(x) for x in g_al]
                               if g_al is not None else None),
                "lineage_id": getattr(ind, "_lineage_id", None),
            })
        if not new_entries:
            return

        # Drop new-entry duplicates (same objective values) before union
        # so the archive can't grow unbounded from re-admission of an
        # unchanged elite parent each generation.  Keep the earliest-found.
        seen = {(m["fitness"][0], m["fitness"][1])
                for m in self.external_archive}
        deduped_new = []
        for m in new_entries:
            key = (m["fitness"][0], m["fitness"][1])
            if key in seen:
                continue
            seen.add(key)
            deduped_new.append(m)
        if not deduped_new:
            return

        pool = list(self.external_archive) + deduped_new
        nd = self._archive_nondominated(pool)
        if len(nd) > self.archive_cap:
            nd = self._archive_prune_by_crowding(nd, self.archive_cap)
        self.external_archive = nd

    @staticmethod
    def _archive_nondominated(pool):
        """Return the non-dominated subset of pool (minimisation).

        Pool is a list of dicts with a ``fitness`` tuple (any length).
        O(N²) — fine for N ≤ a few hundred which is our archive cap.
        """
        n = len(pool)
        keep = [True] * n
        fits = [p["fitness"] for p in pool]
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                if (all(fits[j][k] <= fits[i][k] for k in range(len(fits[i])))
                        and any(fits[j][k] < fits[i][k] for k in range(len(fits[i])))):
                    keep[i] = False
                    break
        return [pool[i] for i in range(n) if keep[i]]

    @staticmethod
    def _archive_prune_by_crowding(pool, target_size):
        """Reduce pool to target_size by repeatedly dropping the
        lowest-crowding member.  Boundary members in each objective
        carry inf crowding so the extremes are protected — the archive
        preserves spread under pruning.
        """
        pool = list(pool)
        while len(pool) > target_size:
            fits = [p["fitness"] for p in pool]
            n = len(fits)
            n_obj = len(fits[0])
            crowding = [0.0] * n
            for m in range(n_obj):
                order = sorted(range(n), key=lambda i: fits[i][m])
                crowding[order[0]]  = float("inf")
                crowding[order[-1]] = float("inf")
                f_min, f_max = fits[order[0]][m], fits[order[-1]][m]
                denom = f_max - f_min
                if denom == 0.0:
                    continue
                for k in range(1, n - 1):
                    if crowding[order[k]] == float("inf"):
                        continue
                    crowding[order[k]] += (
                        (fits[order[k + 1]][m] - fits[order[k - 1]][m]) / denom
                    )
            drop = min(range(n), key=lambda i: crowding[i])
            pool = pool[:drop] + pool[drop + 1:]
        return pool

    def flush_archive_to_csv(self, out_dir):
        """Write the external archive to ``{out_dir}/archive.csv``.

        Called once at end of run from main.py.  Schema:
          gen_found, lineage_id, f_0, f_1, ..., design_0..design_{n-1},
          g_al_0..g_al_{m-1}
        """
        import csv as _csv
        from pathlib import Path as _Path

        out_dir = _Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.external_archive:
            # Write a header-only file so post-hoc tools can distinguish
            # "feature off / nothing to archive" from "file missing".
            csv_path = out_dir / "archive.csv"
            with csv_path.open("w", newline="") as f:
                f.write("gen_found,lineage_id,f_0,f_1\n")
            return

        n_obj    = len(self.external_archive[0]["fitness"])
        n_design = len(self.external_archive[0]["design"])
        g_sample = next((m["g_al"] for m in self.external_archive
                         if m["g_al"] is not None), None)
        n_gal    = len(g_sample) if g_sample else 0

        fieldnames = (
            ["gen_found", "lineage_id"]
            + [f"f_{k}" for k in range(n_obj)]
            + [f"design_{k}" for k in range(n_design)]
            + [f"g_al_{k}" for k in range(n_gal)]
        )
        with (out_dir / "archive.csv").open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for m in self.external_archive:
                row = {"gen_found": m["gen_found"],
                       "lineage_id": m["lineage_id"]}
                for k in range(n_obj):
                    row[f"f_{k}"] = m["fitness"][k]
                for k in range(n_design):
                    row[f"design_{k}"] = m["design"][k]
                g = m["g_al"] if m["g_al"] is not None else []
                for k in range(n_gal):
                    row[f"g_al_{k}"] = g[k] if k < len(g) else None
                w.writerow(row)

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
                # Feature C3 (psucc_exclude_resampled): tag the offspring
                # so update() can later opt-out of feeding it into the
                # donor parent's psucc / σ adaptation.  Tagging is
                # unconditional (cheap, just sets a bool); the feature
                # flag is checked at consumption time.  See update().
                population[i]._resampled = True

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
            if self.selection_mode == "crowding":
                # F3 (NSGA-II): rank mid_front by crowding distance and
                # keep the k highest-crowding members.  Boundary points
                # carry infinite crowding so the front extremes survive,
                # which is exactly the anti-knee-bias property the
                # diversity analysis identified as the key structural fix.
                crowding = self._crowding_distance(mid_front)
                order = sorted(range(len(mid_front)),
                               key=lambda i: -crowding[i])
                chosen     += [mid_front[i] for i in order[:k]]
                not_chosen += [mid_front[i] for i in order[k:]]
            else:
                ref = np.array([ind.fitness.wvalues for ind in feasible]) * -1
                ref = np.max(ref, axis=0) + 1

                for _ in range(len(mid_front) - k):
                    idx = self.indicator(mid_front, ref=ref)
                    not_chosen.append(mid_front.pop(idx))

                chosen += mid_front

        return chosen, not_chosen

    @staticmethod
    def _crowding_distance(front):
        """NSGA-II crowding distance (Deb et al. 2002) on a list of
        DEAP individuals with valid fitness.values tuples.

        Returns a list of crowding distances, one per input individual,
        in the same order as the input.  Boundary points get ``inf`` so
        they are always preferred under "keep highest crowding" rules.

        For ``len(front) <= 2`` every point is a boundary by definition.
        """
        n = len(front)
        if n == 0:
            return []
        if n <= 2:
            return [float("inf")] * n
        n_obj = len(front[0].fitness.values)
        crowding = [0.0] * n
        for m in range(n_obj):
            order = sorted(range(n), key=lambda i: front[i].fitness.values[m])
            crowding[order[0]]  = float("inf")
            crowding[order[-1]] = float("inf")
            f_min = front[order[0]].fitness.values[m]
            f_max = front[order[-1]].fitness.values[m]
            denom = f_max - f_min
            if denom == 0.0:
                # Objective constant across the front → no spread signal
                # in this dimension; skip without polluting the sum.
                continue
            for k in range(1, n - 1):
                if crowding[order[k]] == float("inf"):
                    continue
                crowding[order[k]] += (
                    (front[order[k + 1]].fitness.values[m]
                     - front[order[k - 1]].fitness.values[m]) / denom
                )
        return crowding

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

    def _chtIsotropyCorrection(self, A, invCholesky):
        """Pull a covariance Cholesky factor back toward isotropy when
        its condition number exceeds ``self.cht_kappa_trigger``.

        Feature A4 escape valve.  Independent of infeasibility — fires
        purely on covariance shape.  Mechanic:

            C = A·Aᵀ ;  C = P diag(vp) Pᵀ  (eigendecompose)
            if κ(C) > kappa_trigger:
                vp ← (1-α) vp + α mean(vp)        # blend toward isotropy
                rescale by exp(log_factor / n)    # preserve det(C)
                A ← cholesky(C_new)               # re-factor

        Uses ``self.cht_isotropy_alpha`` if set, else defaults to 0.1.
        Returns ``(A, invCholesky)`` unchanged on a PSD-fallback path
        (matches the rest of the strategy's defensive style).
        """
        n = self.dim
        C = A @ A.T
        C = 0.5 * (C + C.T)
        vp, P = np.linalg.eigh(C)
        vp = np.maximum(vp, 0.0)
        vp_min = vp[vp > 0].min() if np.any(vp > 0) else 1e-300
        kappa = vp[-1] / max(vp_min, 1e-300)
        if kappa <= self.cht_kappa_trigger:
            return A, invCholesky
        alpha = self.cht_isotropy_alpha if self.cht_isotropy_alpha is not None else 0.1
        vp_mean = float(np.mean(vp))
        vp_new = (1.0 - alpha) * vp + alpha * vp_mean
        eps = 1e-300
        log_factor = (np.sum(np.log(vp + eps))
                      - np.sum(np.log(vp_new + eps))) / n
        S = (P * vp_new) @ P.T
        S = 0.5 * (S + S.T)
        C_new = np.exp(log_factor) * S
        C_new = 0.5 * (C_new + C_new.T)
        try:
            A_new = np.linalg.cholesky(C_new)
        except np.linalg.LinAlgError:
            return A, invCholesky
        invCholesky_new = scipy.linalg.solve_triangular(
            A_new, np.eye(n), lower=True,
        )
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
            # Feature A3 (cht_gamma_schedule): linearly interpolate γ
            # from start to end across n_gens generations.  Aggressive
            # early shrinkage to find feasibility, gentle later
            # shrinkage to avoid anisotropy lock-in.  Schema:
            # [start_gamma, end_gamma, n_gens].  Saturates at end_gamma
            # after self._generation >= n_gens.  None ⇒ static γ.
            gamma = _eval_schedule(
                self.cht_gamma_schedule, self._generation, self.cht_gamma,
            )

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
                #
                # Feature A1 (cht_eigenvalue_floor): caps per-call
                # shrinkage to a configurable fraction of the prior
                # eigenvalue.  E.g. 0.5 means no axis can lose more than
                # 50% of its size in a single CHT call.  Hard-bounds
                # anisotropy build-up.  None ⇒ legacy 1e-10 floor.
                psd_floor = 1e-10 * max(sqrt_vp[i_eig], 1e-30)
                if self.cht_eigenvalue_floor is not None:
                    feature_floor = self.cht_eigenvalue_floor * sqrt_vp[i_eig]
                    effective_floor = max(psd_floor, feature_floor)
                else:
                    effective_floor = psd_floor
                sqrt_vp_new[i_eig] = max(
                    sqrt_vp_new[i_eig] - gamma * proj_sum * sqrt_vp[i_eig],
                    effective_floor,
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

        # Feature A2 (cht_isotropy_alpha): pull eigenvalues toward their
        # mean before the rescale.  Counteracts anisotropy build-up:
        # vp_relaxed = (1-α) vp + α mean(vp).  α=0 is a no-op; α=1 makes
        # the covariance fully isotropic in one step (extreme).  The
        # subsequent log_factor rescale still preserves det(C).
        if self.cht_isotropy_alpha is not None:
            alpha = self.cht_isotropy_alpha
            vp_mean = np.mean(vp_new)
            vp_new = (1.0 - alpha) * vp_new + alpha * vp_mean

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

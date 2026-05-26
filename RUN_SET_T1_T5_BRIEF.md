# Run Set T1–T5 — Brief for Context Resumption

**Date generated:** 2026-05-24
**Branch:** `CHT_AL`
**Companion docs (read first if resuming cold):**
- `INVESTIGATION_SUMMARY.md` — overall state of the X2 MO-CMA-ES + AL investigation.
- `~/Documents/Optimisation Vault/Notes/Feature Test 00b…10b` — per-run analyses of the previous sentinel-fix rerun set (0028–0033).
- `~/Documents/Optimisation Vault/Notes/Feature Test Summary - Sentinel-Fix Rerun (runs 00b-10b) vs Original.md` — cross-cutting comparison.

## 1. Where we are

Phase 1 of the sentinel-fix programme produced six runs (0028–0033). Key findings carried forward:

- **The sentinel fixes (S1, S2, S3) shifted the failure modes rather than removing them.** Baseline μ_AL inverted from 8×10⁻⁵ → 3 074; κ(C) inflated from 5.2 → 15.4; A3 (γ-schedule alone) produced κ = 18.3 because γ-decay is *exit-rate* control, not *equilibrium-κ* control.
- **A2 (`cht_isotropy_alpha`) was robust** — κ peak 1.82 across ~270 active CHT generations.
- **B3 (`al_tol_schedule`) was the best single feature** — final HV 0.095, archive of 15, the only run with sustained CHT and AL engagement.
- **0033 (C3) froze** when the initial population was all-sentinel: Fix-S3 prevented psucc adaptation, AL bootstrap caught all sentinels, σ stayed at 0.05 for 400 gens.
- **Two diagnostic bugs were uncovered**: the `n_spark_sentinels` column counts surviving parents (almost always 0 by Pareto geometry) rather than offspring batches (~50% sentinel rate at gen 400 across all runs); and the Pareto plot's purple highlight was the last *offspring* batch, not the surviving parent set.

## 2. Code changes since last context save

All changes committed to `CHT_AL` working tree.

| File | Change |
|---|---|
| `src/algorithm/cmaes.py` | New `_eval_schedule(schedule, gen, fallback)` helper supporting 3-element legacy `[start, end, n_gens]` and 4-element `[start, end, start_gen, end_gen]` forms. Both `current_al_tol` and the γ-schedule call site simplified to use it. |
| `src/algorithm/cmaes.py` | New feature toggle `psucc_sentinel_as_failure` (default False). When True, sentinel offspring in the *not-chosen* branch contribute the standard psucc decay + σ shrinkage. Chosen-branch Fix-S3 stays unconditionally active (sentinels can never produce a "+" signal). |
| `src/main.py` | New `_write_parents_csv` writes `parents_gen_NNNN.csv` per save trigger with surviving μ-set fitness, sigma, psucc, sentinel flags, designs, g_al. New `"parents"` subfolder added to both `OUTPUT_FOLDERS` lists. |
| `src/main.py` | `_save_outputs` threads `strategy` and `bounds` through, writes parents CSV, passes `[ind.fitness.values for ind in strategy.parents]` to the Pareto plot. |
| `src/plotting.py` | `plot_holdtime_impactspeed_2d` accepts optional `parent_fitness` kwarg; purple highlight now shows the surviving parents when supplied, with a legend identifying the slice. |
| `src/main.py` | `NGEN = 500` (was 400). |
| `config/experiments.yaml` | Schema docs updated for the 4-element schedule form and the new toggle. Previous rerun set commented out as yaml history. Active runs are now T1–T5 below. |

Still **not** fixed: the `n_spark_sentinels` diagnostic still counts parents only. Offspring-batch sentinel counter is a deferred edit.

## 3. The five active tests

All share: `sim_type: CHT_AL`, `step_size: 0.05`, `p4_treatment: null`, `psucc_sentinel_as_failure: true`. NGEN = 500.

| ID | pop | initial ε | ε schedule | γ schedule | Purpose |
|---|---|---|---|---|---|
| T1 | 12 | 100 | — | — | Reference: just the new toggle on top of the baseline. |
| T2 | 12 | 1500 | `[1500, 100, 100, 450]` | — | Delayed ε tightening. Holds wide for 100 gens, narrows over gens 100–450, settles at 100 for the last 50. |
| T3 | 12 | 1500 | `[1500, 100, 100, 450]` | `[0.0125, 0.0, 100, 450]` | T2 + γ decays to 0 over same window. Final 50 gens run with no CHT shrinkage. |
| T4 | 24 | 1500 | `[1500, 100, 100, 450]` | — | T2 with doubled pop. |
| T5 | 24 | 1500 | `[1500, 100, 100, 450]` | `[0.0125, 0.0, 100, 450]` | T3 with doubled pop — full stack. |

## 4. Why these tests

The five tests target three failure modes uncovered in phase 1, each by an independent mechanism:

**psucc-as-failure (all five).** Treats SPARK/PITOT3 sentinels in the not-chosen branch as honest failures rather than skipping them. Conceptual frame: an unsimulable design *is* information about local feasibility. Reverses the "Fix-S3 trap" that froze 0033 and re-enables σ adaptation in regimes where >50% of offspring are sentinels.

**Delayed ε tightening (T2–T5).** Direct attack on the AL μ-inversion (0028's μ peaked at 3 074 because the AL bootstrapped against a too-tight ε on an early infeasible population). A wide initial ε = 1500 means the bootstrap sees mostly AL-feasible individuals (delta_vs1 < 1500 m/s is the easy regime); μ stays moderate. The 100-gen delay lets the population settle into the well-behaved region before any constraint pressure is applied; the 350-gen ramp gives ample time for the AL to track each tightening step.

**γ → 0 schedule (T3, T5).** Direct attack on κ inflation (15.4 baseline, 18.3 A3). Eventually setting γ = 0 means the CHT stops carving covariance entirely — the per-parent C is then governed purely by CMA-ES rank-μ/rank-1 updates. If κ falls in the γ = 0 tail, the κ inflation is reversible. If not, the carving is path-dependent.

**Pop = 24 (T4, T5).** Tests whether front clustering across 0028–0032 was driven by under-sampling (μ = 12 is small) rather than by the constraint-handling mechanisms. Doubles both the parent pool's expressive capacity and the per-gen evaluation budget.

## 5. What to look for in each test

### T1 — baseline + psucc_sentinel_as_failure
- **σ trajectory**: does σ collapse faster or slower than 0028's? Compare mean σ at gen 100, 200, 400.
- **psucc per slot**: do per-parent psucc values now show floor-hitting (any reach ~0)? If yes, the trap mechanism is back. If no, the sentinels are providing useful information without destroying adaptation.
- **HV**: better, worse, or same as 0028 (HV = 0.293)? If better, the not-chosen-branch information was load-bearing. If worse, the not-chosen-branch suppression was protective.
- **Sentinel offspring rate**: read directly from `population_gen_*.csv` (count rows with `scaled_hold_time ≥ 0.99999`). Compare with 0028's late-gen 50% rate. A lower rate means σ retreated faster from sentinel regions.

### T2 — delayed ε tightening
- **μ_AL peak**: hypothesis is < 100, much smaller than 0028's 3 074. Read from `al_diagnostics/al_per_gen.csv` (`mu` column).
- **gens λ > 0 trajectory**: does λ go to 0 quickly under wide ε, then re-engage as the schedule tightens? Look for a multi-phase λ pattern in `al_per_gen.csv`.
- **pen_to_f_ratio max**: should drop dramatically below 0028's 9.8×10⁶ if the moderate-μ hypothesis is right.
- **HV trajectory shape**: expect late-run HV gains (gens 200–450) similar to 0032 (B3), the previous best.

### T3 — T2 + γ → 0
- **κ trajectory**: peak should arrive before gen 100 (full γ phase). Then κ should *decrease* during γ-decay phase, especially in gens 400–500 when γ = 0.
- **Final κ**: hypothesis is < 5 at gen 500. If κ stays high in the γ = 0 tail, the anisotropy is locked in and γ-decay is insufficient.
- **CHT-active gens**: should drop dramatically after gen 450 (effectively zero work to do).
- **Combined HV**: best of the set if both hypotheses hold; comparable to T2 if γ-control wasn't the bottleneck.

### T4 — T2 with pop = 24
- **Archive size and Δ**: doubling pop should roughly double the per-gen non-dominated point yield. Archive should grow faster.
- **Sentinel rate per gen**: same fraction (~50%) but absolute count doubles. Does the offspring batch still purge sentinels through selection or does the larger pool let more sentinels survive into parents?
- **HV vs T2**: if T4 beats T2 significantly, pop-scaling was the binding constraint. If similar, the constraint-handling mechanisms were the binding constraint.

### T5 — T3 with pop = 24
- **All of T3's signals + T4's signals.** This is the full-stack test.
- **Compare to T3**: marginal benefit of larger pop given the constraint-handling improvements are already in place.
- **Final Pareto front shape**: read the new `parents_gen_0500.csv`. Are the 24 surviving parents spread along the front (good) or clustered (still has knee-bias problem)?

## 6. Key artefacts the new code produces

Per run directory `Results/al_cht_recomb/al_cht_NNNN/`:

- `parents/parents_gen_NNNN.csv` — surviving μ-set with full state (NEW).
- `pareto_holdtime_impactspeed/pareto_holdtime_impactspeed_gen_NNNN.png` — now highlights parents (NEW behaviour, same path).
- `population/population_gen_NNNN.csv` — pre-selection offspring batch (unchanged).
- `strategy_diagnostics/strategy_per_gen.csv` — per-parent σ and psucc (unchanged; **sentinel columns still count parents only**).
- `al_diagnostics/al_per_gen.csv` — λ, μ, proxy stats (unchanged).
- `cht_diagnostics/cht_per_gen.csv` — κ, n_cht_calls (unchanged).
- `summary/archive.csv`, `summary/diversity_metrics.txt` (unchanged).

## 7. Cross-test comparisons to set up after runs complete

| Comparison | What it tells us |
|---|---|
| T1 vs 0028 | Effect of `psucc_sentinel_as_failure` alone. |
| T2 vs 0032 (B3 rerun) | Effect of moving from `[100, 20, 500]` schedule to `[1500, 100, 100, 450]` schedule. |
| T3 vs T2 | Marginal value of γ-decay on top of delayed ε. |
| T4 vs T2 | Marginal value of pop=24. |
| T5 vs T3 | Marginal value of pop=24 when CHT is already gentled. |
| T5 vs all of 0028–0032 | Best-of-set comparison for the full-stack run. |

## 8. Known open issues

1. **`n_spark_sentinels` diagnostic still counts parents only.** Offspring-batch counter not yet added. To get offspring sentinel rates, parse `population_gen_NNNN.csv` and count rows where `scaled_hold_time ≥ 0.99999 ∧ scaled_impact_speed ≥ 0.99999`.
2. **No backfill of `parents_gen_*.csv` for runs 0028–0033.** The data wasn't recorded; reconstruction would require joining `population_gen_*.csv` `chosen=True` rows with their `_origin_gen` snapshot. Not yet automated.
3. **Layer-3 features (P1 σ-kick on persistent all-sentinel, P2 multi-sample AL proxy) not yet implemented.** Likely needed only if T5 still fails.

## 9. Numbering convention

Tests T1–T5 run as runs 0034–0038 (next sequence). The "T" prefix in this document is for plan-level identification; the on-disk run prefix is `al_cht_NNNN`. Per-run analysis notes should be titled `Feature Test T1 - ...md` through `Feature Test T5 - ...md` in `~/Documents/Optimisation Vault/Notes/`.

---
*Compact summary generated 2026-05-24 to bridge context windows. The five tests are the active set in `config/experiments.yaml`; the previous rerun set is commented out below them.*

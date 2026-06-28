# Session brief — constraint handling, step-size, init/sampling, `delta_vs1`/`al_tol`

**Branch:** `l1d_cht_al`  ·  **Date:** 2026-06-28  ·  **Purpose:** hand-off so this
investigation can resume in a fresh context window.

This session was **investigation only — no source files were modified.** Everything
below is what we *learned by reading the code*, plus the next steps it sets up.

---

## 0. What is currently active (verified from code, not from folder names)

The dispatcher ([src/main.py:1701-1726](src/main.py#L1701)) builds one experiment per
entry under `experiments:` in [config/experiments.yaml](config/experiments.yaml). The
live config has **two identical `ArnoldCHT_AL` entries** (the YAML `*arnold` alias
repeats the `&arnold` anchor, [experiments.yaml:166-175](config/experiments.yaml#L166)):

```
sim_type: ArnoldCHT_AL   pop_size: 24   step_size: 0.1   al_tol: 4900
features: { psucc_sentinel_as_failure: true, al_tol_schedule: [4900, 100.0, 100, 250] }
```

Verified three ways (all agree):
- `cht_method('ArnoldCHT_AL')` → `'arnold'` ([cmaes.py:58-80](src/algorithm/cmaes.py#L58)).
- Runtime dispatch takes the **Arnold** branch — `apply_arnold_infeasibility`, *not*
  `resample_infeasibles` ([main.py:1174-1188](src/main.py#L1174)).
- `is_al_active` / `is_cht_active` both True.

**Active = Arnold & Hansen 2012 covariance constraint handling + Augmented Lagrangian.**
Infeasibles are *consumed before evaluation and dropped* (no resample).

> ⚠️ **Run-vs-config mismatch to remember:** the newest run on disk, `al_cht_0080`
> (had `al_per_gen.csv` open in a spreadsheet), is a **Chocat** run (`cht_diagnostics/`,
> no `arnold_diagnostics/`) — a *historical* run under a different config. A fresh launch
> today is **Arnold**.

---

## 1. `population_gen_xxxx.csv` — why `raw_delta_vs1` is blank for some rows

- The snapshot is the **pre-selection λ batch** ([main.py:1308](src/main.py#L1308)), so it
  includes infeasible offspring.
- Objective columns are filled only when `ind.fitness.valid`
  ([_build_pop_row, main.py:384](src/main.py#L384)). Arnold **drops** infeasibles before the
  L1d sim runs, so they have **no fitness and no `delta_vs1`** → blank by design
  (distinguishes "not evaluated" from "zero"). Real example: a gen had 18/24 populated, 6 blank.
- In 2-D mode `raw_delta_vs1` is *reconstructed*: `raw_dvs = g_al[0] + al_tol`
  ([main.py:393](src/main.py#L393)).
- **Gap:** `_build_pop_row` computes `feasible` and `max_g` but
  [results_io.py POPULATION_FIELDS](src/results_io.py#L22) omits them, and the writer uses
  `extrasaction="ignore"` → **those columns are silently dropped**, so the CSV gives no
  in-file reason for a blank.
- **Suggested (not done):** add `"feasible"` + `"max_g"` to `POPULATION_FIELDS`.

---

## 2. Step size — active levers + why σ behaves the way it does

Success-rule (1+1-style) step control, per-parent. Constants for N=6
([cmaes.py:192-199](src/algorithm/cmaes.py#L192)): `d=4.0`, `ptarg=1/5.5≈0.1818`,
`cp≈0.0833`. Update: `σ ← σ·exp((psucc−ptarg)/(d·(1−ptarg)))`. **Equilibrium at
psucc=ptarg≈0.18** (above → σ grows, below → σ shrinks).

**Active toggles:** `psucc_sentinel_as_failure: true` (inert in L1d — no PITOT3/SPARK
sentinels fire) + `al_tol_schedule` (indirect). **OFF:** `sigma_floor_silent`,
`psucc_exclude_resampled` (C3), `cht_kappa_trigger`, `cht_gamma_schedule`,
`box_reflective_repair` (C2).

**Empirical σ (from `strategy_per_gen.csv`):**
- **Arnold (0078/0079):** σ **collapses → ~0** by gen ~300 (psucc → 0). Infeasibles are
  dropped from σ-adaptation entirely ([cmaes.py:1500-1502](src/algorithm/cmaes.py#L1500)),
  so as the front matures, feasible offspring rarely beat parents → σ collapses.
- **Chocat (0080):** σ **inflates** (gen 73 mean 0.145 > 0.10 start) and stays large
  (gen 290 max 0.25). Cause = **resample-driven psucc inflation**: infeasibles are
  resampled to near-parent feasible points that usually survive → manufactured
  "successes". The fix built for this, **C3 `psucc_exclude_resampled`**, is OFF (and is a
  *Chocat-only* lever — it does nothing in Arnold).

---

## 3. Initialisation / sampling

**Two distinct mechanisms.**

**Phase A — gen 0 (initialisation):**
[initialise_population_sobol, init_population_l1d.py:337](src/init_population_l1d.py#L337) —
scrambled **Sobol** (low-discrepancy) sequence + **L1d rejection** (keep only genuine
ruptures), adaptive refill sized by observed acceptance rate, hard ceiling
`max_eval_factor·pop_size`. Unit→physical mapping uses **conditional bounds**
([lhs_unit_to_physical:117](src/init_population_l1d.py#L117)): `p4` and `reservoir_p` are
drawn relative to the just-drawn `driver_p` (so `reservoir_p ≥ driver_p`, p4 in the
compression window by construction). Output `.npz` → `main.py --seed-npz`
([main.py:929](src/main.py#L929)); seed rows are already normalised and are **not**
re-transformed. Unseeded fallback = uniform-conditional + one-slot feasibility resample.

**Phase B — per generation (sampling):**
[generate, cmaes.py:582](src/algorithm/cmaes.py#L582). With λ=μ=24, **one Gaussian
offspring per parent**: `x' = x_parent + σ_i · A_i · z`, `z~N(0,I)`, `C_i = A_iA_iᵀ`.
Under the active Arnold config there is **no in-generate repair** (CHT-active guard at
[cmaes.py:621](src/algorithm/cmaes.py#L621) skips it; C2 reflection is off) — infeasibles
flow to `apply_arnold_infeasibility` and are dropped.

**Two coordinate spaces:** physical (Pa/m/%) for L1d; normalised `[1,2]⁶` for CMA-ES
(`variable_transformation` / `variable_untransformation`).

---

## 4. `init_arnold.npz` diagnostics (`src/L1d_Outputs/init_arnold.npz`)

Keys: `x_phys (24,6)`, `x_norm (24,6)`, `objectives (24,3)`, `attempts (24,)`,
`n_evaluated=640`, `acceptance_rate=0.0375`, `method='sobol'`, `seed=1`.

- **Cost:** 24 keepers from **640** L1d runs → **3.75%** acceptance (616 rejected).
  Below the ~5% rupture estimate; effective eval-factor ≈ 27 (under the 64 default ceiling).
- **Health:** 24/24 feasible (`max(g) = −3.9e-4`); `x_norm ∈ [1.0007, 1.9933] ⊂ [1,2]⁶`;
  `x_norm == transform(x_phys)` to machine zero; no PITOT3 sentinels (no `delta_vs1==3500`).
- **`attempts` all 1** (Sobol path: resample cost lives in `n_evaluated`).
- **Objectives:** `t_hold` 0–0.59 ms (some exactly 0); `impact` 8.8–288 m/s;
  `delta_vs1` **1746–3953** (mean 3002) — **see §5 for correct interpretation.**

---

## 5. ⭐ `delta_vs1` and `al_tol` — the corrected understanding (most important)

**`delta_vs1` is NOT the shock speed. It is the absolute deviation from a 4900 m/s target.**

Pipeline ([l1d_job.py:436-457](src/problem/l1d_job.py#L436)):
```
transducers at (1.5, 2.5) m  ->  arrival = first p > 2× quiescent fill p (mean of first 50)
vs1 = |2.5 − 1.5| / |t_arr2 − t_arr1|              (time-of-flight average speed)
delta_vs1 = | vs1 − 4900 |                          (VS1_TARGET = 4900, l1d_job.py:77)
```
So `delta_vs1` is a **goal-distance to MINIMISE** (0 = perfect). It is an **absolute
value → V-shaped, non-smooth at 4900, and folds too-slow & too-fast onto the same
score** (sign of `vs1 − 4900` is lost).

**AL wiring** ([evaluate.py:182-183](src/problem/evaluate.py#L182)):
`g_al = [delta_vs1 − al_tol]`. pycma convention `g_al ≤ 0` feasible
([_al_penalty, cmaes.py:420](src/algorithm/cmaes.py#L420)), so:
```
feasible  ⟺  delta_vs1 ≤ al_tol  ⟺  | vs1 − 4900 | ≤ al_tol
```
**`al_tol` = allowed deviation band (m/s) around 4900.** Not computed from data — it's a
**scheduled homotopy** ([_eval_schedule, cmaes.py:83](src/algorithm/cmaes.py#L83)),
applied to every offspring each gen ([main.py:1126-1140](src/main.py#L1126)):
```
al_tol_schedule = [4900, 100.0, 100, 250]   (4-elem: start, end, start_gen, end_gen)
  gen < 100 : al_tol = 4900     (band ±4900 — any rupture passes, AL ~dormant)
  100–250   : linear 4900 → 100 (band tightens)
  gen ≥ 250 : al_tol = 100      (target vs1 ∈ [4800, 5000])
```
`raw_delta_vs1 (CSV) = g_al + al_tol = delta_vs1 = |vs1 − 4900|`.

### Correction to the earlier npz reading (record this)
Earlier in the session `delta_vs1` was mis-described as the shock speed to push *up*
toward 4900. **It is the deviation to push *down* toward 0.** Re-read correctly:

| value | wrong (earlier) | correct |
|---|---|---|
| `delta_vs1=3953` | "best, near target" | **worst** seed (vs1 is 3953 m/s off 4900) |
| `delta_vs1=1746` | — | **best** seed (closest to target) |
| seed AL feasibility | "0/24, violates widely" | **24/24 feasible at gen 0** (all dev < al_tol=4900) |
| implied vs1 (if slow side) | — | best ≈ 3154, mean ≈ 1898, worst ≈ 947 m/s |

The real difficulty: seed is feasible while band is wide, but as `al_tol → 100` the
deviations (≥1746) become violations, and the stale-ε note says the deviation **floors
~2412** → vs1 can't get within 2412 of 4900 → the final ±100 band is **unreachable** for
these designs. The homotopy just exposes that floor gradually.

---

## 6. Next steps (what this brief sets up)

1. **Un-fold `vs1`** — the `abs()` hides which side of 4900 the population sits on.
   Re-derive `vs1 = 4900 ± delta_vs1` for the seed, and confirm against raw traces by
   reading `history-loc-0000/0001.data` in a `src/L1d_Outputs/DEAP_*/` dir. Expectation:
   weak-driver shocks **below** 4900. Confirms which basin of the V we're stuck in.
2. Decide whether the `abs()` folding should be replaced with a **signed** constraint
   (e.g. `vs1 ≥ 4900` one-sided) so the optimiser can see direction.
3. Re-examine whether the **~2412 floor** is physical (driver can't make the shock) or an
   artifact of transducer placement `(1.5, 2.5) m` / the 2×-quiescent detector.
4. (Optional, separate) the σ story: enable **C3** only if studying *Chocat*; for Arnold,
   σ-collapse is the live issue, not σ-inflation.

## Quick reference — key files
- Active config: [config/experiments.yaml:166-175](config/experiments.yaml#L166)
- CHT classify / dispatch: [cmaes.py:58-80](src/algorithm/cmaes.py#L58), [main.py:1174-1188](src/main.py#L1174)
- `delta_vs1`: [l1d_job.py:436-457](src/problem/l1d_job.py#L436)
- AL constraint + tol: [evaluate.py:173-184](src/problem/evaluate.py#L173), [cmaes.py:83-118](src/algorithm/cmaes.py#L83), [main.py:1126-1140](src/main.py#L1126)
- σ update: [cmaes.py:675-882](src/algorithm/cmaes.py#L675)
- Init/sampling: [init_population_l1d.py:117](src/init_population_l1d.py#L117), [:337](src/init_population_l1d.py#L337); [generate, cmaes.py:582](src/algorithm/cmaes.py#L582)
- Pop CSV: [main.py:363-437](src/main.py#L363), [results_io.py:22](src/results_io.py#L22)

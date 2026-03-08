# X2 Free-Piston Driver — Multi-Objective Optimisation

Multi-objective evolutionary optimisation of the X2 free-piston driver facility
using the MO-CMA-ES algorithm (Voss, Hansen, Igel, 2010).

---

## Problem Description

**Design variables** (6 total, searched in a normalised [1, 2]⁶ space):

| Variable | Physical range |
|---|---|
| Percent helium in driver gas | 70 – 100 % |
| Driver fill pressure | 1 000 – 2 740 000 Pa |
| Diaphragm burst pressure p4 | coupled to driver pressure |
| Throat diameter | 0.050 – 0.085 m |
| Reservoir pressure | driver_p – 8 000 000 Pa |
| Buffer length | 0.05 – 0.15 m |

**Objectives** (three, all minimised):

| Objective | Evaluator | Ideal → Nadir |
|---|---|---|
| Shock speed residual \|vs − 4900\| (m/s) | PITOT3 | 0 → 3500 m/s |
| Driver hold time (maximised, so minimise negation) | SPARK | 0.01 → 0 s |
| Piston impact speed (m/s) | SPARK | 0 → 350 m/s |

---

## Repository Structure

```
config/
  experiments.yaml      # experiment configurations (pop size, sigma, strategy)

src/
  main.py               # entry point and evolution loop
  utils.py              # shared utilities (feasibility, parallelisation setup)
  plotting.py           # Pareto front scatter plots
  algorithm/
    cmaes.py            # MO-CMA-ES strategy (generate / update)
    toolbox.py          # DEAP-style operator registry and Logbook
    hypervolume.py      # hypervolume indicator (Fonseca et al., 2006)
    penalty.py          # OBSOLETE — see inline Penalty handling in cmaes.py
  problem/
    config.py           # bounds, reference points, PITOT3 base configuration
    transforms.py       # physical ↔ normalised space conversions
    evaluate.py         # PITOT3 constraint + SPARK objective evaluators

scripts/
  evaluation_shortcuts.py   # quick single-design evaluation for debugging
  images_for_paper.py       # publication-quality figure generation
  Running_Metric.py         # HV / IGD monitoring callback
  file_deleter.py           # PITOT3_Outputs cleanup utility

gas_models/             # Lua gas model files for SPARK and PITOT3
exploration/            # legacy experimental scripts (not maintained)
results/                # output plots and data from completed runs
reference/              # thesis and reference materials
```

---

## Constraint-Handling Strategies

| `sim_type` | Description |
|---|---|
| `ParentValue` | Infeasible offspring inherit the parent's value for the violated attribute |
| `ElitistCrossover` | The highest hypervolume-contributing parent donates the violating attribute |
| `RandomCrossover` | A randomly selected parent donates the violating attribute |
| `Penalty` | Infeasible individuals receive a fitness penalty (handled inline in `cmaes.py`) |

---

## How to Run

### Dependencies

Install Python packages:

```bash
pip install -r requirements.txt
```

The following tools must also be installed and available on `$PATH`:

- **gdtk / Eilmer** (provides `gdtk.gas` and PITOT3): https://gdtk.uqcloud.net
- **SPARK** (piston dynamics solver)

### Running the optimisation

```bash
cd src
python main.py
```

Experiment configurations (population size, step size, strategy) are read from
`config/experiments.yaml`. Edit that file to change what runs — no need to
touch `main.py`.

### Running a single design evaluation

```bash
python scripts/evaluation_shortcuts.py
```

Edit the candidate design point `x` at the top of that file before running.

---

## Outputs

Each completed run writes results into a timestamped subdirectory under
`results/MOO_CMA_ES_<sim_type>/`, containing:

- `convergence_data.txt` — hypervolume per generation
- `convergence_<sim_type>.png` — convergence plot
- `output.txt` — initial and final population design variables and objectives
- Pareto front scatter plots for each pair of objectives

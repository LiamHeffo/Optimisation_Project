import os
import sys

# Allow this script to be run directly from scripts/ without installing the
# package.  Adds src/ to the module search path so that 'problem', 'algorithm',
# 'utils', etc. resolve correctly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from deap import base, creator

from problem.evaluate   import evaluate
from problem.transforms import variable_transformation, unnormalise_fitness
from problem.config     import BOUNDS, APPROX_IDEAL, APPROX_NADIR

# Register DEAP types (safe to call even if already registered by main.py —
# DEAP issues a RuntimeWarning but does not raise).
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual",   list, fitness=creator.FitnessMulti,
               ind_number=int, sim_type=str, bounds=list)

bounds = BOUNDS  # 6-variable bounds imported from problem.config

# Candidate design points in physical space:
#   [percent_He, driver_p (Pa), p4 (Pa), D_throat (m), reservoir_p (Pa), buffer_length (m)]
# x = [(80,   0.077e6,   35.7e6, 0.0849, 6.08e6, 0.10)]
x = [(99.9, 0.3285e6, 29.2e6,  0.0517, 7.61e6, 0.10)]
# x = [(100,  0.3498e6,  29.87e6, 0.052,  7.95e6, 0.10)]
# x = [(100,  0.3035e6,  28.37e6, 0.0558, 7.36e6, 0.10)]
# x = [(100,  0.3164e6,  28.62e6, 0.0506, 7.12e6, 0.10)]
# x = [(100,  0.3480e6,  29.86e6, 0.0518, 7.92e6, 0.10)]
# x = [(100,  0.3686e6,  28.78e6, 0.0508, 7.85e6, 0.10)]

x_transformed = variable_transformation(x, bounds)
candidate = creator.Individual(x_transformed[0])

candidate.sim_type   = 'ParentValue'
candidate.normalised = True
candidate.ind_number = 0       # required by constraint_function for its working directory
candidate.bounds     = bounds  # required by evaluate() → constraint/objective functions

fit, g = evaluate(candidate)
if fit is None:
    print(f"infeasible: g = {g}")
else:
    print(unnormalise_fitness(fit, APPROX_IDEAL, APPROX_NADIR))
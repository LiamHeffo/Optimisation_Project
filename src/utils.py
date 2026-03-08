"""
General-purpose utility functions used across the optimisation pipeline.

  distance()            – Euclidean-squared distance to the feasible boundary
  closest_feasible()    – Clamp an individual back into [MIN_BOUND, MAX_BOUND]
  valid()               – Check whether an individual lies in the normalised domain
  parallelization_setup() – Create per-individual working directories for PITOT3
"""

import os
import numpy as np

from problem.config import MIN_BOUND, MAX_BOUND


def distance(feasible_ind, original_ind):
    """Euclidean-squared distance between a feasible point and an infeasible one."""
    return sum((f - o) ** 2 for f, o in zip(feasible_ind, original_ind))


def closest_feasible(individual):
    """Return the nearest feasible individual by clamping each component to [MIN_BOUND, MAX_BOUND]."""
    print(f'Individual: {individual}')
    feasible_ind = np.array(individual)
    feasible_ind = np.maximum(MIN_BOUND, feasible_ind)
    feasible_ind = np.minimum(MAX_BOUND, feasible_ind)
    print(f"feasible individual: {feasible_ind}")
    return feasible_ind


def valid(individual):
    """Return True if the individual lies within the normalised search domain [1, 2]^N."""
    if any(individual < MIN_BOUND) or any(individual > MAX_BOUND):
        return False
    return True


def parallelization_setup(population):
    """
    Create a per-individual subdirectory inside PITOT3_Outputs/ so that
    concurrent PITOT3 evaluations do not overwrite each other's files.
    """
    starting_working_directory = os.getcwd()
    if 'PITOT3_Outputs' not in os.listdir(starting_working_directory):
        os.mkdir('PITOT3_Outputs')

    os.chdir(starting_working_directory + '/PITOT3_Outputs')

    for i, ind in enumerate(population):
        run_folder = f'DEAP_tests_{i}'
        if not os.path.exists(run_folder):
            os.mkdir(run_folder)

    os.chdir(starting_working_directory)

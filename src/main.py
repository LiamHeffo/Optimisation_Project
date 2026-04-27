"""
Main entry point for the X2 free-piston driver multi-objective optimisation.

Experiment configuration
------------------------
Each element of experiment_types is a 4-tuple:
    (sim_type, pop_size, step_size, p4_treatment)

sim_type options:
    'ParentValue'      – infeasible offspring inherit the parent's value for
                         the violated attribute
    'ElitistCrossover' – the most hypervolume-contributing parent donates
    'RandomCrossover'  – a randomly selected parent donates
    'Penalty'          – infeasible individuals receive a fitness penalty

p4_treatment options:
    'hard_bounds_on_p4' – enforce the upper p4 bound via retransformation
    None                – no special treatment

Module-level setup
------------------
The DEAP creator types (FitnessMulti, Individual) are registered here.
This is intentional: creator.create has a side-effect on a global registry,
so it must run exactly once per process.  Both this file and
evaluation_shortcuts.py rely on those types being registered.
"""

import os
import pathlib
import time
import multiprocessing
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from deap import base, creator, tools

from algorithm.toolbox   import Toolbox
from algorithm.hypervolume import HyperVolume
from algorithm.cmaes     import StrategyMultiObjective
from problem.config      import (
    APPROX_IDEAL, APPROX_NADIR, BOUNDS,
    he_lower, he_upper,
    driver_p_lower, driver_p_upper,
    p4_lower, p4_upper,
    D_throat_lower, D_throat_upper,
    reservoir_lower, reservoir_upper,
    buffer_length_lower, buffer_length_upper,
)
from problem.transforms  import variable_transformation, variable_untransformation, unnormalise_fitness
from problem.evaluate    import evaluate, set_logbook
from plotting            import plot_objective_space
from utils               import parallelization_setup

# ─────────────────────────────────────────────────────────────────────────────
# DEAP type registration  (runs once on import)
# ─────────────────────────────────────────────────────────────────────────────

# FitnessMulti: all three objectives are minimised (weights = -1).
# The evaluator returns normalised values, so minimising maps to:
#   delta_vs1   → 0 is best
#   hold_time   → 0 is best (we negate: longer hold time = smaller normalised value)
#   impact_speed → 0 is best (same negation logic)
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual",   list, fitness=creator.FitnessMulti,
               ind_number=int, sim_type=str, bounds=list)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons
# ─────────────────────────────────────────────────────────────────────────────

toolbox = Toolbox()
toolbox.register("evaluate", evaluate)

# Reference point for HV computation: the all-zeros point in normalised space
pop_hypervolumes = HyperVolume(np.array((0, 0, 0)))

normalised = True

# ─────────────────────────────────────────────────────────────────────────────
# Main evolution loop
# ─────────────────────────────────────────────────────────────────────────────

def main(experiment_type):
    s1 = time.time()

    # ── Experiment parameters ─────────────────────────────────────────────
    N           = 6
    pop_size    = experiment_type[1]
    MU, LAMBDA  = pop_size, pop_size
    NGEN        = 10
    sim_type    = experiment_type[0]
    p4_treatment = experiment_type[3]
    step_size   = experiment_type[2]

    print(f"Step Size = {step_size}")
    print(f'Pop Size = {pop_size}')

    # ── Logbook initialisation ────────────────────────────────────────────
    gen_counter = 0
    toolbox.logbook.add('generation',          gen_counter)
    toolbox.logbook.add('fixer_count',         {f"{gen_counter}": 0})
    toolbox.logbook.add('time_spent_fixing',   0)
    toolbox.logbook.add("time taken",          0)
    toolbox.logbook.add("No. individuals that failed objective tests", 0)
    toolbox.logbook.add("No. individuals that produced no hold time",  0)
    toolbox.logbook.add("No. individuals that failed constraint tests", 0)
    toolbox.logbook.add("hypervolume",         [0 for _ in range(1, NGEN + 1)])

    # Give the evaluate module a reference to the logbook so it can update
    # counters without accessing a global toolbox.
    set_logbook(toolbox.logbook)

    # ── Design-variable bounds ────────────────────────────────────────────
    bounds = BOUNDS

    # ── Population initialisation ─────────────────────────────────────────
    def pop_init(MU):
        percent_he_list     = np.random.uniform(he_lower,            he_upper,            (MU, 1))
        D_throat_list       = np.random.uniform(D_throat_lower,      D_throat_upper,      (MU, 1))
        driver_p_list       = np.random.uniform(driver_p_lower,      driver_p_upper,      (MU, 1))
        buffer_length_list  = np.random.uniform(buffer_length_lower, buffer_length_upper, (MU, 1))

        p4_list = np.zeros_like(percent_he_list)
        for i in range(MU):
            if 1190.63 * driver_p_list[i] < p4_upper:
                p4_list[i] = np.random.uniform(14.62 * driver_p_list[i], 1190.63 * driver_p_list[i])
            else:
                p4_list[i] = np.random.uniform(14.62 * driver_p_list[i], p4_upper)

        reservoir_p_list = [np.random.uniform(driver_p_list[i], reservoir_upper) for i in range(MU)]

        return [
            [percent_he_list[i][0], driver_p_list[i][0], p4_list[i][0],
             D_throat_list[i][0],   reservoir_p_list[i][0], buffer_length_list[i][0]]
            for i in range(MU)
        ]

    i = 0
    init_pop_untransformed = pop_init(MU)
    init_pop_transformed   = variable_transformation(init_pop_untransformed, bounds)

    population = [creator.Individual(x) for x in init_pop_transformed]
    initial_population = population

    for ind in population:
        ind.ind_number = i
        ind.bounds     = bounds
        i += 1

    parallelization_setup(population)

    for ind in population:
        ind.sim_type   = sim_type
        ind.normalised = normalised
        ind.fitness.values = toolbox.evaluate(ind)

    # ── Strategy and multiprocessing setup ────────────────────────────────
    strategy = StrategyMultiObjective(
        population, sigma=step_size,
        mu=MU, lambda_=LAMBDA,
        sim_type=sim_type, p4_treatment=p4_treatment,
        bounds=bounds,
        logbook=toolbox.logbook,      # injected — no global access inside cmaes.py
    )
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update",   strategy.update)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    fitness_history = []

    # ── Evolution ─────────────────────────────────────────────────────────
    for gen in range(NGEN):
        toolbox.logbook.bookshelf['generation'] += 1
        print('\n')
        print('*' * 30)
        print(f"Generation {toolbox.logbook.bookshelf['generation']}")
        print('*' * 30)

        parents    = population
        population = toolbox.generate()

        i = 0
        for ind in population:
            ind.normalised = normalised
            ind.ind_number = i
            ind.bounds     = bounds
            i += 1

        # Retry logic for transient evaluation failures
        try:
            fitnesses = toolbox.map(toolbox.evaluate, population)
        except Exception:
            time.sleep(1)
            try:
                fitnesses = toolbox.map(toolbox.evaluate, population)
            except Exception:
                time.sleep(1)
                try:
                    fitnesses = toolbox.map(toolbox.evaluate, population)
                except Exception:
                    time.sleep(1)
                    fitnesses = toolbox.map(toolbox.evaluate, population)

        fixed = False
        for i, (ind, fit) in enumerate(zip(population, fitnesses)):
            normalised_shock_speed = fit[0]

            if normalised_shock_speed == 1.0:
                # Evaluation failed — substitute the parent
                replacement = parents[ind.ind_number]
                new_fitness = replacement.fitness.values
                population[i] = replacement
                population[i].fitness.values = new_fitness
                fixed = True
                fitness_history.append(new_fitness)
            else:
                ind.fitness.values = fit
                fitness_history.append(fit)

        toolbox.update(population)

        fitness_copy = list(fitnesses)
        fitness_copy = [fit for fit in fitness_copy if fit[0] != 1.0]

        avg_hypervolume = pop_hypervolumes.compute(np.array(fitness_copy) * -1)
        print(f'average hypervolume = {avg_hypervolume}')
        toolbox.logbook.bookshelf['hypervolume'][gen] = avg_hypervolume

        # Intermediate Pareto scatter plots every 50 generations (200–750)
        if gen % 50 == 0 and 200 <= gen <= 750:
            starting_working_directory = os.getcwd()
            if starting_working_directory[-1] in [f'{i}' for i in range(0, 12)]:
                starting_working_directory = starting_working_directory[:35]

            os.chdir(starting_working_directory + '/Scatter_Plots')
            plot_objective_space(fitness_history, 'delta_vs1',  'hold_time',    MU=MU, sim_type=sim_type, gen=gen)
            plot_objective_space(fitness_history, 'delta_vs1',  'impact_speed', MU=MU, sim_type=sim_type, gen=gen)
            plot_objective_space(fitness_history, 'hold_time',  'impact_speed', MU=MU, sim_type=sim_type, gen=gen)
            os.chdir(starting_working_directory)

    # ── Post-processing ───────────────────────────────────────────────────
    starting_working_directory = os.getcwd()
    title_string = f'MOO_CMA_ES_{sim_type}'

    if title_string not in os.listdir(starting_working_directory):
        os.mkdir(title_string)
    os.chdir(starting_working_directory + '/' + title_string)

    new_working_directory = os.getcwd()
    previous_experiment_list = [
        d for d in os.listdir()
        if d[:-2] == title_string
    ]

    if len(previous_experiment_list) == 0:
        test_name = f'MOO_CMA_ES_{sim_type}_1'
        os.mkdir(test_name)
    else:
        experiment_number = np.max([int(d[-1]) for d in previous_experiment_list]) + 1
        test_name = f'MOO_CMA_ES_{sim_type}_{experiment_number}'
        os.mkdir(test_name)

    os.chdir(new_working_directory + '/' + test_name)

    # ── Convergence data ──────────────────────────────────────────────────
    with open('convergence_data.txt', 'w') as file:
        file.write(f"Simulation Type = {sim_type}\n")
        file.write(f"Step Size = {step_size}\n")
        file.write(f'Pop Size = {pop_size}\n')
        file.write(f'p4 treatment = {p4_treatment}\n')
        file.write(f'Number of generations = {NGEN}\n')
        file.write(f'Current Time = {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
        file.write(f"Average Hypervolume per generation:\n")
        for gen in range(NGEN):
            file.write(f"Generation {gen + 1}: {toolbox.logbook.bookshelf['hypervolume'][gen]}\n")

    # ── Hypervolume convergence plot ──────────────────────────────────────
    x_range = NGEN
    tick_interval = x_range / 5

    plt.figure(dpi=800)
    plt.title("Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.ylim((0, 1.1))

    gen_axis = list(range(1, NGEN + 1))
    avg_hv_list = [toolbox.logbook.bookshelf['hypervolume'][g - 1] for g in gen_axis]
    plt.plot(gen_axis, avg_hv_list)
    plt.savefig(f"convergence_{sim_type}.png")
    plt.close()

    # ── Final Pareto scatter plots ────────────────────────────────────────
    plot_objective_space(fitness_history, 'delta_vs1', 'hold_time',    MU=MU, sim_type=sim_type, gen=NGEN)
    plot_objective_space(fitness_history, 'delta_vs1', 'impact_speed', MU=MU, sim_type=sim_type, gen=NGEN)
    plot_objective_space(fitness_history, 'hold_time', 'impact_speed', MU=MU, sim_type=sim_type, gen=NGEN)

    # ── Fixer count plot (non-Penalty runs only) ──────────────────────────
    fixer_count = []
    running_total = 0
    for entry in toolbox.logbook.bookshelf["fixer_count"].keys():
        if entry != '0':
            running_total += toolbox.logbook.bookshelf["fixer_count"][entry]
            fixer_count.append(running_total)

    e1 = time.time()
    toolbox.logbook.bookshelf["time taken"] = e1 - s1

    if sim_type != "Penalty":
        generation = list(range(1, NGEN + 1))
        plt.figure(dpi=800)
        plt.title("Cumulative Number of Individuals Fixed")
        plt.xlabel("Generation")
        plt.ylabel("Number of Individuals Fixed")
        plt.plot(generation, fixer_count)
        plt.gca().xaxis.set_major_locator(MultipleLocator(tick_interval))
        plt.savefig(f"cma_es_mo_fpd_{sim_type}RunningTotal.png")
        plt.close()

    # ── Output summary text file ──────────────────────────────────────────
    initial_dimensionalised_fitness = [
        unnormalise_fitness(ind, APPROX_IDEAL, APPROX_NADIR)
        for ind in fitness_history[:MU]
    ]
    final_dimensionalised_fitness = [
        unnormalise_fitness(ind, APPROX_IDEAL, APPROX_NADIR)
        for ind in fitness_history[-MU:]
    ]

    sig_figs = 6

    def _fmt_var(variable, index):
        """Format a single design variable for the output table."""
        col_widths = [
            'Percent Helium ', 'Driver Pressure (MPa) ', ' p4 (MPa) ',
            ' Throat Diameter (mm) ', ' Reservoir Pressure (MPa) ', ' Buffer Length (mm) '
        ]
        scale = [1, 1e-6, 1e-6, 1e3, 1e-6, 1e3]
        variable = round(variable * scale[index], sig_figs - str(variable * scale[index]).find('.'))
        white_space = int(np.round((len(col_widths[index]) - len(f'{variable}')) / 2))
        sep = '|' if index in [1, 3, 4] else ''
        return sep + ' ' * white_space + f'{variable}' + ' ' * white_space

    with open('output.txt', 'w') as file:
        file.write(f"Simulation Type = {sim_type}\n")
        file.write(f"step size = {experiment_type[2]}\n")
        file.write(f'pop size = {pop_size}\n')
        file.write(f'p4 treatment = {p4_treatment}\n')
        file.write(f"time taken = {toolbox.logbook.bookshelf['time taken']}\n")
        file.write(f'Number of generations = {NGEN}\n')
        file.write(f'Current Time = {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
        file.write(f"Number of individuals that produced no hold time = "
                   f"{toolbox.logbook.bookshelf['No. individuals that produced no hold time']}\n")
        file.write(f"Fixer Count = {running_total}\n")
        file.write('*' * 100 + '\n')
        file.write("INITIAL POPULATION:\n")
        file.write(
            "Percent Helium |Driver Pressure (MPa) | p4 (MPa) | Throat Diameter (mm) "
            "| Reservoir Pressure (MPa) | Buffer Length (mm)\n"
        )
        for ind in initial_population:
            string = ''.join(
                _fmt_var(v, idx)
                for idx, v in enumerate(variable_untransformation(ind, bounds))
            )
            file.write(string + '\n')

        file.write('\n')
        file.write("Residual of Shock Speed (m/s) | Driver Hold Time (ms) | Piston Impact Speed (m/s)\n")
        for ind in initial_dimensionalised_fitness:
            row = []
            for idx, obj in enumerate(ind):
                obj = round(obj, sig_figs - str(obj).find('.'))
                if idx == 1:
                    obj *= 1e3
                row.append(f'{obj}')
            file.write('  '.join(row) + '\n')

        file.write('\n' + '*' * 100 + '\n')
        file.write("FINAL POPULATION:\n")
        file.write(
            "Percent Helium | Driver Pressure (MPa) | p4 (MPa) | Throat Diameter (mm) "
            "| Reservoir Pressure (MPa) | Buffer length (mm)\n"
        )
        for ind in strategy.parents:
            string = ''.join(
                _fmt_var(v, idx)
                for idx, v in enumerate(variable_untransformation(ind, bounds))
            )
            file.write(string + '\n')

        file.write('\n')
        file.write("Residual of Shock Speed (m/s) | Driver Hold Time (ms) | Piston Impact Speed (m/s)\n")
        for ind in final_dimensionalised_fitness:
            row = []
            for idx, obj in enumerate(ind):
                obj = round(obj, sig_figs - str(obj).find('.'))
                if idx == 1:
                    obj *= 1e3
                row.append(f'{obj}')
            file.write('  '.join(row) + '\n')

    os.chdir(starting_working_directory)

    print('\n\nEND OF SIM')
    print('*' * 60)
    print('\n\n')

    return strategy.parents


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _config_path = pathlib.Path(__file__).parent.parent / "config" / "experiments.yaml"
    with open(_config_path) as _f:
        _config = yaml.safe_load(_f)

    experiment_types = [
        (exp["sim_type"], exp["pop_size"], exp["step_size"], exp["p4_treatment"])
        for exp in _config["experiments"]
    ]

    for experiment_type in experiment_types:
        solutions = main(experiment_type)

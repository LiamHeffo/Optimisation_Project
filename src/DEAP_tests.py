import spark
import numpy as np
import yaml, os, random, time
import array
import random
import time
import itertools
from copy import deepcopy
from functools import wraps
from itertools import repeat
import multiprocessing
from functools import partial
from collections import defaultdict
from yaml.loader import Reader, Scanner, Parser, Composer, SafeConstructor, Resolver
from pitot3_utils.pitot3_classes import Facility, Driver, Diaphragm, Facility_State, Tube, Nozzle, Test_Section
from pitot3_utils.pitot3_classes import eilmer4_CEAGas_input_file_creator, expansion_tube_test_time_calculator, \
    state_output_for_final_output, pitot3_results_output, cleanup_function, pitot3_remote_run_creator
from pitot3 import StrictBoolSafeLoader
from gdtk.gas import GasModel, GasState, GasFlow
from deap import base, creator, tools
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

normalised = True

# Problem size
N = 6

MIN_BOUND = np.ones(N)
MAX_BOUND = np.ones(N) + 1

APPROX_IDEAL = (0, 0.01, 0)
APPROX_NADIR = (3500, 0, 350)

def plot_objective_space(fitness_history, obj1, obj2, **kwargs):
        
        # Extract keyword arguments
        MU = kwargs.get('MU', 10)  # default MU=10 if not provided
        sim_type = kwargs.get('sim_type', 'Parent_Value')
        gen = kwargs.get('gen', 0)
        normalised = kwargs.get('normalised', True)

        if obj1 == 'delta_vs1' and obj2 == 'hold_time':
            plt.figure(dpi=800)
            plt.title("Pareto Frontier")
            plt.xlabel("Normalised Residual of Shock Speed")
            plt.ylabel("Normalised Hold Time")

            delta_vs_history = [entry[0] for entry in fitness_history]
            hold_time_history = [entry[1] for entry in fitness_history]

            if normalised:
                plt.xlim((0, 1.1))
                plt.ylim((0, 1.1))
            else:
                plt.ylim((-0.005, 2*np.max(hold_time_history)))

            plt.scatter(delta_vs_history, hold_time_history, facecolors='none', edgecolors='lightblue')
            plt.scatter(delta_vs_history[-MU:], hold_time_history[-MU:], facecolors='none', edgecolors='green')

            plt.savefig(f"cma_es_mo_fpd_{sim_type}_1_{gen}.png")
            plt.close()

        elif obj1 == 'delta_vs1' and obj2 == 'impact_speed':

            plt.figure(dpi=800)
            plt.title("Pareto Frontier")
            plt.xlabel("Normalised Residual of Shock Speed")
            plt.ylabel("Normalised Impact Speed")

            delta_vs_history = [entry[0] for entry in fitness_history]
            impact_speed_history = [entry[2] for entry in fitness_history]

            if normalised:
                plt.xlim((0, 1.1))
                plt.ylim((0, 1.1))
            else:
                plt.ylim((-0.005, 2*np.max(impact_speed_history)))

            plt.scatter(delta_vs_history, impact_speed_history, facecolors='none', edgecolors='lightblue')
            plt.scatter(delta_vs_history[-MU:], impact_speed_history[-MU:], facecolors='none', edgecolors='orange')

            plt.savefig(f"cma_es_mo_fpd_{sim_type}_2_{gen}.png")
            plt.close()

        elif obj1 == 'hold_time' and obj2 == 'impact_speed':

            hold_time_history = [entry[1] for entry in fitness_history]
            impact_speed_history = [entry[2] for entry in fitness_history]


            plt.figure(dpi=800)
            plt.title("Pareto Frontier")
            plt.xlabel("Normalised Hold Time")
            plt.ylabel("Normalised Impact Speed")

            if normalised:
                plt.xlim((0, 1.1))
                plt.ylim((0, 1.1))
            else:
                plt.ylim((-0.005, 2*np.max(impact_speed_history)))

            plt.scatter(hold_time_history, impact_speed_history, facecolors='none', edgecolors='lightblue')
            plt.scatter(hold_time_history[-MU:], impact_speed_history[-MU:], facecolors='none', edgecolors='purple')

            plt.savefig(f"cma_es_mo_fpd_{sim_type}_3_{gen}.png")
            plt.close()

def base_config_dict():
    config_data = {'mode': 'fully_theoretical', 'output_filename': 'optimisation_test_setup',
                   'facility': 'x2_nrst_85_mm_shock_tube', 'driver_condition': 'custom_from_dict',
                   'test_gas_gas_model': 'CEAGas', 'test_gas_name': 'he-with-ions', 'p1': 150e3}
    return config_data

def base_driver_dict(x):
    driver_dict = {'percent_He': x[0], 'driver_p': x[1], 'p4': x[2], 'D_throat': x[3], 'reservoir_p': x[4], 'buffer_length': x[5]}

    driver_dict['driver_fill_composition'] = {'He': float(driver_dict['percent_He'] / 100),
                                              'Ar': float(1 - driver_dict['percent_He'] / 100)}
    driver_dict['driver_condition_name'] = 'x2lwp-2.0mm-0'  # I think this is only for the output...
    driver_dict['driver_condition_type'] = 'empirical'
    driver_dict['driver_gas_model'] = 'thermally-perfect-preset'
    driver_dict['driver_fill_gas_name'] = 'he-ar'
    driver_dict['driver_speciesList'] = list(driver_dict['driver_fill_composition'].keys())
    driver_dict['driver_inputUnits'] = 'moles'
    driver_dict['driver_withIons'] = False
    driver_dict['M_throat'] = 1.0

    return driver_dict

def distance(feasible_ind, original_ind):
    """A distance function to the feasibility region."""
    return sum((f - o)**2 for f, o in zip(feasible_ind, original_ind))

def closest_feasible(individual):
    """A function returning a valid individual from an invalid one."""
    print(f'Individual: {individual}')
    feasible_ind = np.array(individual)
    feasible_ind = np.max(MIN_BOUND, feasible_ind)
    feasible_ind = np.min(MAX_BOUND, feasible_ind)
    print(f"feasible individual: {feasible_ind}")
    return feasible_ind

def valid(individual):
    """Determines if the individual is valid or not."""
    if any(individual < MIN_BOUND) or any(individual > MAX_BOUND):
        return False
    return True

def parallelization_setup(population):
    ####################################################################################################################
    # Building separate files for parallelization:
    ####################################################################################################################

    starting_working_directory = os.getcwd()
    if 'PITOT3_Outputs' not in [directory for directory in os.listdir(starting_working_directory)]:
        os.mkdir('PITOT3_Outputs')

    os.chdir(starting_working_directory + '/' + 'PITOT3_Outputs')

    test_names = []

    # we store the config that we are changing for each simulation in case we need it later on...
    lists_to_iterate_through = population

    for i, ind in enumerate(lists_to_iterate_through):

        ind_number = i

        test_name = f'DEAP_tests_{ind_number}'

        test_names.append(test_name)

        run_folder = test_name

        if not os.path.exists(run_folder):
            os.mkdir(run_folder)

    os.chdir(starting_working_directory)

def variable_transformation(pop, bounds):

    # print(f'bounds[0] = {bounds[0][0]}')
    new_population = []
    for x in pop:
        x_new_0 = (x[0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0]) + 1 # percent_he
        x_new_1 = (x[1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0]) + 1 # driver_p
        x_new_2 = (x[2] - 14.62 * x[1]) / (1190.63 * x[1] - 14.62 * x[1]) + 1 # p4
        x_new_3 = (x[3] - bounds[3][0]) / (bounds[3][1] - bounds[3][0]) + 1 # D_throat
        x_new_4 = (x[4] - x[1]) / (bounds[4][1] - x[1]) + 1 # reservoir_p
        x_new_5 = (x[5] - bounds[5][0]) / (bounds[5][1] - bounds[5][0]) + 1 # buffer_length

        x_new = [x_new_0, x_new_1, x_new_2, x_new_3, x_new_4, x_new_5]
        new_population.append(x_new)

    return new_population

def variable_untransformation(x, bounds):
    # print(f"Untransforming x = {x} with bounds = {bounds}")
    x_new_0 = np.abs(x[0] - 1) * (bounds[0][1] - bounds[0][0]) + bounds[0][0]
    x_new_1 = np.abs(x[1] - 1) * (bounds[1][1] - bounds[1][0]) + bounds[1][0]
    x_new_2 = np.abs(x[2] - 1) * (1190.63 * x_new_1 - 14.62 * x_new_1) + 14.62 * x_new_1
    x_new_3 = np.abs(x[3] - 1) * (bounds[3][1] - bounds[3][0]) + bounds[3][0]
    x_new_4 = np.abs(x[4] - 1) * (bounds[4][1] - x_new_1) + x_new_1
    x_new_5 = np.abs(x[5] - 1) * (bounds[5][1] - bounds[5][0]) + bounds[5][0]

    x_new = [x_new_0, x_new_1, x_new_2, x_new_3, x_new_4, x_new_5]

    return x_new

def normalise_fitness(fitness, ideal_point, nadir_point):
    return tuple((np.array(fitness) - np.array(ideal_point)) / (np.array(nadir_point) - np.array(ideal_point)))

def unnormalise_fitness(fitness, ideal_point, nadir_point):
    return tuple(np.array(fitness) * (np.array(nadir_point) - np.array(ideal_point)) + np.array(ideal_point))

def constraint_function(x1, bounds):
    ####################################################################################################################
    # pre-processing
    ####################################################################################################################
    ind_number = x1.ind_number
    test_name = f"DEAP_tests_{ind_number}"

    # where we start out...
    starting_working_directory = os.getcwd()
    if starting_working_directory[-1] in [f'{i}' for i in range(0, 12)]:
        starting_working_directory = starting_working_directory[:35]

    # change directory to the one of the simulation
    os.chdir(starting_working_directory + '/' + 'PITOT3_Outputs' + '/' + test_name)

    ####################################################################################################################
    # Transforming variables back to standard dimensions:
    ####################################################################################################################

    # t, p_idx = x._ps
    # print(f"t = {t}")

    # he_lower, he_upper = 70, 100
    # D_throat_lower, D_throat_upper = 0.05, 0.085
    # driver_p_lower, driver_p_upper = 1000, (40/14.62)*1e6
    # p4_lower, p4_upper = 150e3, 40e6
    # reservoir_lower, reservoir_upper = 1000, 8e6

    # bounds = [(he_lower, he_upper), (driver_p_lower, driver_p_upper),
    #           (p4_lower, p4_upper), (D_throat_lower, D_throat_upper), (reservoir_lower, reservoir_upper)]

    x = variable_untransformation(x1, bounds)
    # print(f"\n x1 = {x1}, \n x = {x}")

    ####################################################################################################################
    # Running some heuristic checks:
    ####################################################################################################################
    driver_dict = base_driver_dict(x)

    if driver_dict['driver_p'] > driver_dict['reservoir_p']:
        print("bad guess")
        return 3500

    if 0.085 < driver_dict['D_throat'] < 0:
        print("bad guess")
        return 3500

    pressure_ratio = driver_dict["p4"] / driver_dict["driver_p"]
    compression_ratio = pressure_ratio**(1/1.667)                                                                                                           
    if 5 < compression_ratio < 70:
        pass
    else:
        print("bad guess")
        return 3500
    
    if driver_dict['p4'] > bounds[2][1]:
        print("bad guess")
        return 3500

    ####################################################################################################################
    # Building config dictionaries
    ####################################################################################################################

    config_data = base_config_dict()

    # building driver gas model:
    preset_gas_models_folder = '$PITOT3_DATA/preset_gas_models'
    driver_gmodel_location = '{0}/thermally-perfect-{1}-gas-model.lua'.format(preset_gas_models_folder,
                                                                              driver_dict['driver_fill_gas_name'])
    gmodel = GasModel(os.path.expandvars(driver_gmodel_location))

    # Performing Isentropic Compression:
    state4i = GasState(gmodel)
    state4i.p = driver_dict['driver_p']
    state4i.T = 298.15

    driver_fill_composition_massf = {}
    molecular_mass_dict = {'Ar': 39.948, 'He': 4.002602}
    total_molecular_mass = 0.0

    for species in driver_dict['driver_speciesList']:
        total_molecular_mass += molecular_mass_dict[species] * driver_dict['driver_fill_composition'][species]

    for species in driver_dict['driver_speciesList']:
        driver_fill_composition_massf[species] = driver_dict['driver_fill_composition'][species] * (
                molecular_mass_dict[species] / total_molecular_mass)

    state4i.massf = driver_fill_composition_massf
    state4i.update_thermo_from_pT()
    state4i.update_sound_speed()
    gamma = state4i.gamma

    T4 = state4i.T * (driver_dict['p4'] / state4i.p) ** (1.0 - (1.0 / gamma))  # K
    driver_dict['T4'] = T4

    ####################################################################################################################
    # Setting up driver object:
    ####################################################################################################################
    facility_name = config_data['facility']

    # facility_yaml_filename = '{0}/{1}.yaml'.format(facilities_folder, facility_name)
    facility_yaml_filename = '{0}/facilities/{1}.yaml'.format('$PITOT3_DATA', facility_name)

    facility_yaml_file = open(os.path.expandvars(facility_yaml_filename))
    facility_input_data = yaml.load(facility_yaml_file, Loader=yaml.FullLoader)
    facility = Facility(facility_input_data)

    # get some values which we may need while working through the calculation
    facility_type = facility.get_facility_type()

    # we also need to add the facility type to the config data so we have it later on...
    config_data['facility_type'] = facility_type

    D_shock_tube = facility.shock_tube_diameter

    outputUnits = 'moles'

    # load the species molecular weights file here so we can use it to get mole fractions when needed...
    species_molecular_weights_filename = '$PITOT3_DATA/PITOT3_species_molecular_weights.yaml'
    species_molecular_weights_file = open(os.path.expandvars(species_molecular_weights_filename))
    species_MW_dict = yaml.load(species_molecular_weights_file, Loader=StrictBoolSafeLoader)

    T_0 = float(298.15)
    p_0 = float(101325)

    driver = Driver(driver_dict, p_0=p_0, T_0=T_0, preset_gas_models_folder=preset_gas_models_folder,
                    outputUnits=outputUnits, species_MW_dict=species_MW_dict, D_shock_tube=D_shock_tube)

    ###################################################################################################################
    # Now processing the driver gas:
    ###################################################################################################################

    state3s = driver.get_exit_state()

    ####################################################################################################################
    # Now setting up tube object:
    ####################################################################################################################

    if facility:
        shock_tube_length, shock_tube_diameter = facility.get_shock_tube_length_and_diameter()
    else:
        shock_tube_length = None
        shock_tube_diameter = None

    test_gas_gas_model = config_data['test_gas_gas_model']
    test_gas_filename = None
    test_gas_name = config_data['test_gas_name']

    p1 = float(config_data['p1'])
    shock_tube_fill_state_name = 's1'
    T1 = 298.15

    shock_tube_tube_name = 'shock_tube'
    shock_tube_expand_to = 'flow_behind_shock'
    shock_tube_expansion_factor = 1.0

    shock_tube_unsteady_expansion_steps = 100
    vs1_guess_1 = 2000
    vs1_guess_2 = 5000
    vs1_limits = [400, 20000]
    vs1_tolerance = 1e-4

    shock_tube_shocked_state_name = 's2'
    shock_tube_unsteadily_expanded_state_name = 's3'

    shock_tube = Tube(tube_name=shock_tube_tube_name, tube_length=shock_tube_length, tube_diameter=shock_tube_diameter,
                      fill_pressure=p1, fill_temperature=T1, fill_gas_model=test_gas_gas_model,
                      fill_gas_name=test_gas_name, fill_gas_filename=test_gas_filename,
                      fill_state_name=shock_tube_fill_state_name, shocked_fill_state_name=shock_tube_shocked_state_name,
                      entrance_state_name=driver.get_exit_state_name(),
                      entrance_state=state3s,
                      unsteadily_expanded_entrance_state_name=shock_tube_unsteadily_expanded_state_name,
                      expand_to=shock_tube_expand_to, expansion_factor=shock_tube_expansion_factor,
                      preset_gas_models_folder=preset_gas_models_folder,
                      unsteady_expansion_steps=shock_tube_unsteady_expansion_steps,
                      vs_guess_1=vs1_guess_1, vs_guess_2=vs1_guess_2, vs_limits=vs1_limits, vs_tolerance=vs1_tolerance,
                      outputUnits=outputUnits, species_MW_dict=species_MW_dict)

    ####################################################################################################################
    # Now calculating shock speed for given inputs
    ####################################################################################################################
    try:
        shock_tube.calculate_shock_speed_and_related_states()
    except Exception as e:
        print(f"{e}")
        print(f"x = {x}")
        return 3500

    ####################################################################################################################
    # returning the constraint function value:
    ####################################################################################################################

    os.chdir(starting_working_directory)

    # print(f"The end working directory is {os.getcwd()}\n")


    return np.abs(shock_tube.vs - 3585)

def objective_function(x, bounds):
    ####################################################################################################################
    # Transforming variables back to standard dimensions:
    ####################################################################################################################
    # he_lower, he_upper = 70, 100
    # D_throat_lower, D_throat_upper = 0.05, 0.085
    # driver_p_lower, driver_p_upper = 1000, (40/14.62)*1e6
    # p4_lower, p4_upper = 150e3, 40e6
    # reservoir_lower, reservoir_upper = 1000, 8e6

    # bounds = [(he_lower, he_upper), (driver_p_lower, driver_p_upper),
    #           (p4_lower, p4_upper), (D_throat_lower, D_throat_upper), (reservoir_lower, reservoir_upper)]

    x = variable_untransformation(x, bounds)
    # print(f"x = {x}")


    # driver_condition_dict = {'percent_He' : x[0], 'driver_p' : x[1], 'p4' : x[2], 'D_throat' : x[3], 'reservoir_p': x[4]}
    driver_condition_dict = {'percent_He': x[0], 'driver_p': x[1], 'p4': x[2], 'D_throat': x[3], 'reservoir_p': x[4], 'buffer_length': x[5]}


    # running some heuristic tests:
    if driver_condition_dict['driver_p'] > driver_condition_dict['reservoir_p']:
        # print("bad guess")
        # print(f"driver p = {driver_condition_dict['driver_p']}")
        # print(f"reservoir p = {driver_condition_dict['reservoir_p']}")
        toolbox.logbook.bookshelf["No. individuals that failed objective tests"] += 1
        toolbox.logbook.bookshelf["No. individuals that produced no hold time"] += 1
        return 0, 350

    if 0.0849 < driver_condition_dict['D_throat'] < 0:
        # print("bad guess")
        toolbox.logbook.bookshelf["No. individuals that failed objective tests"] += 1
        toolbox.logbook.bookshelf["No. individuals that produced no hold time"] += 1
        return 0, 350

    pressure_ratio = driver_condition_dict["p4"] / driver_condition_dict["driver_p"]
    compression_ratio = pressure_ratio**(1/1.667)                                                                                                           
    if 5 < compression_ratio < 70:
        pass
    else:
        # print("bad guess")
        # print(f"driver p = {driver_condition_dict['driver_p']}")
        # print(f"p4 = {driver_condition_dict['p4']}")
        # print(f'cmopression ratio = {compression_ratio}')
        toolbox.logbook.bookshelf["No. individuals that failed objective tests"] += 1
        toolbox.logbook.bookshelf["No. individuals that produced no hold time"] += 1
        return 0, 350
    
    if driver_condition_dict['p4'] > bounds[2][1]:
        print('p4 too high')
        toolbox.logbook.bookshelf["No. individuals that failed objective tests"] += 1
        toolbox.logbook.bookshelf["No. individuals that produced no hold time"] += 1
        return 0, 350

    fill_condition = {
        "p_drvr_0": driver_condition_dict['driver_p'],  
        "T_drvr_0": 298.15,  # K
        "composition_drvr": {'He': float(driver_condition_dict['percent_He'] / 100),
                             'Ar': float(1 - driver_condition_dict['percent_He'] / 100)},
        "composition_units": 'molef',  # default is massf
        "p_rsvr_0": driver_condition_dict['reservoir_p'],
        "T_rsvr_0": 298.15,  # K
        "p_rupture": driver_condition_dict['p4'] 
    }

    facility = {
        "L_drvr": 4.475,  # m (see docs for which dimension this is)
        "D_piston": 0.2568,  # m (equivalent/equal to D_drvr)
        "V_drvr_0": spark.calculateInitialDriverVolume(4.475, 0.2568, 0.112, 85 / 1000, L_buffer=float(driver_condition_dict['buffer_length']),
                                                       D_buffer=50 / 1000),
        # <-- see the docs for description of this function
        "m_piston": 10.5,  # kg
        "L_buffer": float(driver_condition_dict['buffer_length']),  # 50 mm buffers
        "D_star": driver_condition_dict['D_throat'],  
        "D_driven": 85 / 1000  # 85 mm
    }
    rupture_model = {
        "model": "drewry",  # a drewry model requires that D_driven is set in facility dict
        "K": 0.93,
        "rho": 8649,  # kg/m^3
        "time_model": "linear",
        "tau": 2.0 / 1000,  # 2.5 mm petal/diaphragm thickness
        "b": 0.06
    }
    settings = {
        "rsvr_gm": "ideal_air",
        "drvr_gm": "mixed_he_ar",
        "max_piston_cycles": 3,  # this halting criteria is ignored if the diaphragm ruptures
        "percent_time_on_buffers_before_halting": 0.05,
        # the sim's other stopping criteria: % of total sim time after which to stop the sim if the piston is on the buffers
        "effective_inflection_velocity_tolerance": 3,
        # has to be zero in this type of optimiser, # m/s (a piston will never exactly inflect at 0.0, so what velocity is deemed sufficient?)
        't_hold_sim': True
    }

    sim = spark.createSimulation(
        condition_dict=fill_condition,
        facility_dict=facility,
        simulation_settings_dict=settings,
        diaphragm_model_dict=rupture_model,
    )

    try:
        sim.run()
        diaphragm_rupture_flag = sim.flags.diaphragm_ruptured
        impact_flag = sim.flags.impact_occurred
        # print(f"diaphragm_rupture_flag = {diaphragm_rupture_flag}, impact_flag = {impact_flag}")
        # print(f'halting reason = {sim.halting_information.reason}')
    except Exception as e:
        print(f"{e}")
        print(f"x = {x}")

    if diaphragm_rupture_flag and impact_flag:
        t_hold = sim.t_hold
        impact_speed = round(sim.results.vel_buffer_strike_max,3)
        return t_hold, impact_speed
    else:
        # toolbox.logbook.bookshelf["No. individuals that produced no hold time"] += 1
        return 0, 350

def evaluate(x):
    if x.sim_type == "Penalty":
        if valid(x):                
            delta_vs = constraint_function(x, x.bounds)
            hold_time, impact_speed = objective_function(x, x.bounds)
            normalised_fitness = normalise_fitness((delta_vs, hold_time, impact_speed), APPROX_IDEAL, APPROX_NADIR)
            # print(f'fitness: {fitness}')
            # print(f'normalised fitness: {normalised_fitness}')
            # return normalise_fitness((constraint_function(x), objective_function(x)), APPROX_IDEAL, APPROX_NADIR)
            return normalised_fitness
        else:
            return ClosestValidPenalty.wrapper(x)
    else:
        # print(f'x.bounds = {x.bounds}')
        delta_vs = constraint_function(x, x.bounds)
        hold_time, impact_speed = objective_function(x, x.bounds)
        normalised_fitness = normalise_fitness((delta_vs, hold_time, impact_speed), APPROX_IDEAL, APPROX_NADIR)
        # print(f'fitness: {fitness}')
        # print(f'normalised fitness: {normalised_fitness}')
        # return normalise_fitness((constraint_function(x), objective_function(x)), APPROX_IDEAL, APPROX_NADIR)
        return normalised_fitness


class Toolbox(object):
    """A toolbox for evolution that contains the evolutionary operators. At
    first the toolbox contains a :meth:`~deap.toolbox.clone` method that
    duplicates any element it is passed as argument, this method defaults to
    the :func:`copy.deepcopy` function. and a :meth:`~deap.toolbox.map`
    method that applies the function given as first argument to every items
    of the iterables given as next arguments, this method defaults to the
    :func:`map` function. You may populate the toolbox with any other
    function by using the :meth:`~deap.base.Toolbox.register` method.

    Concrete usages of the toolbox are shown for initialization in the
    :ref:`creating-types` tutorial and for tools container in the
    :ref:`next-step` tutorial.
    """

    def __init__(self):
        self.register("clone", deepcopy)
        self.register("map", map)
        self.logbook = Logbook()

    def register(self, alias, function, *args, **kargs):
        """Register a *function* in the toolbox under the name *alias*. You
        may provide default arguments that will be passed automatically when
        calling the registered function. Fixed arguments can then be overridden
        at function call time.

        :param alias: The name the operator will take in the toolbox. If the
                      alias already exist it will overwrite the operator
                      already present.
        :param function: The function to which refer the alias.
        :param argument: One or more argument (and keyword argument) to pass
                         automatically to the registered function when called,
                         optional.

        The following code block is an example of how the toolbox is used. ::

            >>> def func(a, b, c=3):
            ...     print a, b, c
            ...
            >>> tools = Toolbox()
            >>> tools.register("myFunc", func, 2, c=4)
            >>> tools.myFunc(3)
            2 3 4

        The registered function will be given the attributes :attr:`__name__`
        set to the alias and :attr:`__doc__` set to the original function's
        documentation. The :attr:`__dict__` attribute will also be updated
        with the original function's instance dictionary, if any.
        """
        pfunc = partial(function, *args, **kargs)
        pfunc.__name__ = alias
        pfunc.__doc__ = function.__doc__

        if hasattr(function, "__dict__") and not isinstance(function, type):
            # Some functions don't have a dictionary, in these cases
            # simply don't copy it. Moreover, if the function is actually
            # a class, we do not want to copy the dictionary.
            pfunc.__dict__.update(function.__dict__.copy())

        setattr(self, alias, pfunc)

    def unregister(self, alias):
        """Unregister *alias* from the toolbox.

        :param alias: The name of the operator to remove from the toolbox.
        """
        delattr(self, alias)

    def decorate(self, alias, *decorators):
        """Decorate *alias* with the specified *decorators*, *alias*
        has to be a registered function in the current toolbox.

        :param alias: The name of the operator to decorate.
        :param decorator: One or more function decorator. If multiple
                          decorators are provided they will be applied in
                          order, with the last decorator decorating all the
                          others.

        .. note::
            Decorate a function using the toolbox makes it unpicklable, and
            will produce an error on pickling. Although this limitation is not
            relevant in most cases, it may have an impact on distributed
            environments like multiprocessing.
            A function can still be decorated manually before it is added to
            the toolbox (using the @ notation) in order to be picklable.
        """
        pfunc = getattr(self, alias)
        function, args, kargs = pfunc.func, pfunc.args, pfunc.keywords
        for decorator in decorators:
            function = decorator(function)
        self.register(alias, function, *args, **kargs)

    # def hold(self, alias, object):
    #     self.dict[f'{alias}'] = object
    #     setattr(self, alias, object)


class Logbook(list):
    """Evolution records as a chronological list of dictionaries.

    Data can be retrieved via the :meth:`select` method given the appropriate
    names.

    The :class:`Logbook` class may also contain other logbooks referred to
    as chapters. Chapters are used to store information associated to a
    specific part of the evolution. For example when computing statistics
    on different components of individuals (namely :class:`MultiStatistics`),
    chapters can be used to distinguish the average fitness and the average
    size.
    """

    def __init__(self):
        self.buffindex = 0
        self.chapters = defaultdict(Logbook)
        self.bookshelf = {}
        """Dictionary containing the sub-sections of the logbook which are also
        :class:`Logbook`. Chapters are automatically created when the right hand
        side of a keyworded argument, provided to the *record* function, is a
        dictionary. The keyword determines the chapter's name. For example, the
        following line adds a new chapter "size" that will contain the fields
        "max" and "mean". ::

            logbook.record(gen=0, size={'max' : 10.0, 'mean' : 7.5})

        To access a specific chapter, use the name of the chapter as a
        dictionary key. For example, to access the size chapter and select
        the mean use ::

            logbook.chapters["size"].select("mean")

        Compiling a :class:`MultiStatistics` object returns a dictionary
        containing dictionaries, therefore when recording such an object in a
        logbook using the keyword argument unpacking operator (**), chapters
        will be automatically added to the logbook.
        ::

            >>> fit_stats = Statistics(key=attrgetter("fitness.values"))
            >>> size_stats = Statistics(key=len)
            >>> mstats = MultiStatistics(fitness=fit_stats, size=size_stats)
            >>> # [...]
            >>> record = mstats.compile(population)
            >>> logbook.record(**record)
            >>> print logbook
              fitness          length
            ------------    ------------
            max     mean    max     mean
            2       1       4       3

        """

        self.columns_len = None
        self.header = None
        """Order of the columns to print when using the :data:`stream` and
        :meth:`__str__` methods. The syntax is a single iterable containing
        string elements. For example, with the previously
        defined statistics class, one can print the generation and the
        fitness average, and maximum with
        ::

            logbook.header = ("gen", "mean", "max")

        If not set the header is built with all fields, in arbitrary order
        on insertion of the first data. The header can be removed by setting
        it to :data:`None`.
        """

        self.log_header = True
        """Tells the log book to output or not the header when streaming the
        first line or getting its entire string representation. This defaults
        :data:`True`.
        """

    def record(self, **infos):
        """Enter a record of event in the logbook as a list of key-value pairs.
        The information are appended chronologically to a list as a dictionary.
        When the value part of a pair is a dictionary, the information contained
        in the dictionary are recorded in a chapter entitled as the name of the
        key part of the pair. Chapters are also Logbook.
        """
        apply_to_all = {k: v for k, v in infos.items() if not isinstance(v, dict)}
        for key, value in list(infos.items()):
            if isinstance(value, dict):
                chapter_infos = value.copy()
                chapter_infos.update(apply_to_all)
                self.chapters[key].record(**chapter_infos)
                del infos[key]
        self.append(infos)

    def select(self, *names):
        """Return a list of values associated to the *names* provided
        in argument in each dictionary of the Statistics object list.
        One list per name is returned in order.
        ::

            >>> log = Logbook()
            >>> log.record(gen=0, mean=5.4, max=10.0)
            >>> log.record(gen=1, mean=9.4, max=15.0)
            >>> log.select("mean")
            [5.4, 9.4]
            >>> log.select("gen", "max")
            ([0, 1], [10.0, 15.0])

        With a :class:`MultiStatistics` object, the statistics for each
        measurement can be retrieved using the :data:`chapters` member :
        ::

            >>> log = Logbook()
            >>> log.record(**{'gen': 0, 'fit': {'mean': 0.8, 'max': 1.5},
            ... 'size': {'mean': 25.4, 'max': 67}})
            >>> log.record(**{'gen': 1, 'fit': {'mean': 0.95, 'max': 1.7},
            ... 'size': {'mean': 28.1, 'max': 71}})
            >>> log.chapters['size'].select("mean")
            [25.4, 28.1]
            >>> log.chapters['fit'].select("gen", "max")
            ([0, 1], [1.5, 1.7])
        """
        if len(names) == 1:
            return [entry.get(names[0], None) for entry in self]
        return tuple([entry.get(name, None) for entry in self] for name in names)

    @property
    def stream(self):
        """Retrieve the formatted not streamed yet entries of the database
        including the headers.
        ::

            >>> log = Logbook()
            >>> log.append({'gen' : 0})
            >>> print log.stream  # doctest: +NORMALIZE_WHITESPACE
            gen
            0
            >>> log.append({'gen' : 1})
            >>> print log.stream  # doctest: +NORMALIZE_WHITESPACE
            1
        """
        startindex, self.buffindex = self.buffindex, len(self)
        return self.__str__(startindex)

    def __delitem__(self, key):
        if isinstance(key, slice):
            for i, in range(*key.indices(len(self))):
                self.pop(i)
                for chapter in self.chapters.values():
                    chapter.pop(i)
        else:   
            self.pop(key)
            for chapter in self.chapters.values():
                chapter.pop(key)

    def pop(self, index=0):
        """Retrieve and delete element *index*. The header and stream will be
        adjusted to follow the modification.

        :param item: The index of the element to remove, optional. It defaults
                     to the first element.

        You can also use the following syntax to delete elements.
        ::

            del log[0]
            del log[1::5]
        """
        if index < self.buffindex:
            self.buffindex -= 1
        return super(self.__class__, self).pop(index)

    def __txt__(self, startindex):
        columns = self.header
        if not columns:
            columns = sorted(self[0].keys()) + sorted(self.chapters.keys())
        if not self.columns_len or len(self.columns_len) != len(columns):
            self.columns_len = [len(c) for c in columns]

        chapters_txt = {}
        offsets = defaultdict(int)
        for name, chapter in self.chapters.items():
            chapters_txt[name] = chapter.__txt__(startindex)
            if startindex == 0:
                offsets[name] = len(chapters_txt[name]) - len(self)

        str_matrix = []
        for i, line in enumerate(self[startindex:]):
            str_line = []
            for j, name in enumerate(columns):
                if name in chapters_txt:
                    column = chapters_txt[name][i + offsets[name]]
                else:
                    value = line.get(name, "")
                    string = "{0:n}" if isinstance(value, float) else "{0}"
                    column = string.format(value)
                self.columns_len[j] = max(self.columns_len[j], len(column))
                str_line.append(column)
            str_matrix.append(str_line)

        if startindex == 0 and self.log_header:
            header = []
            nlines = 1
            if len(self.chapters) > 0:
                nlines += max(map(len, chapters_txt.values())) - len(self) + 1
            header = [[] for i in range(nlines)]
            for j, name in enumerate(columns):
                if name in chapters_txt:
                    length = max(len(line.expandtabs()) for line in chapters_txt[name])
                    blanks = nlines - 2 - offsets[name]
                    for i in range(blanks):
                        header[i].append(" " * length)
                    header[blanks].append(name.center(length))
                    header[blanks + 1].append("-" * length)
                    for i in range(offsets[name]):
                        header[blanks + 2 + i].append(chapters_txt[name][i])
                else:
                    length = max(len(line[j].expandtabs()) for line in str_matrix)
                    for line in header[:-1]:
                        line.append(" " * length)
                    header[-1].append(name)
            str_matrix = chain(header, str_matrix)

        template = "\t".join("{%i:<%i}" % (i, k) for i, k in enumerate(self.columns_len))
        text = [template.format(*line) for line in str_matrix]

        return text

    def __str__(self, startindex=0):
        text = self.__txt__(startindex)
        return "\n".join(text)

    def add(self, alias, object):
        self.bookshelf[f'{alias}'] = object


class StrategyMultiObjective(object):
    """Multiobjective CMA-ES strategy based on the paper [Voss2010]_. It
    is used similarly as the standard CMA-ES strategy with a generate-update
    scheme.

    :param population: An initial population of individual.
    :param sigma: The initial step size of the complete system.
    :param mu: The number of parents to use in the evolution. When not
               provided it defaults to the length of *population*. (optional)
    :param lambda_: The number of offspring to produce at each generation.
                    (optional, defaults to 1)
    :param indicator: The indicator function to use. (optional, default to
                      :func:`~deap.tools.hypervolume`)

    Other parameters can be provided as described in the next table

    +----------------+---------------------------+----------------------------+
    | Parameter      | Default                   | Details                    |
    +================+===========================+============================+
    | ``d``          | ``1.0 + N / 2.0``         | Damping for step-size.     |
    +----------------+---------------------------+----------------------------+
    | ``ptarg``      | ``1.0 / (5 + 1.0 / 2.0)`` | Target success rate.       |
    +----------------+---------------------------+----------------------------+
    | ``cp``         | ``ptarg / (2.0 + ptarg)`` | Step size learning rate.   |
    +----------------+---------------------------+----------------------------+
    | ``cc``         | ``2.0 / (N + 2.0)``       | Cumulation time horizon.   |
    +----------------+---------------------------+----------------------------+
    | ``ccov``       | ``2.0 / (N**2 + 6.0)``    | Covariance matrix learning |
    |                |                           | rate.                      |
    +----------------+---------------------------+----------------------------+
    | ``pthresh``    | ``0.44``                  | Threshold success rate.    |
    +----------------+---------------------------+----------------------------+

    .. [Voss2010] Voss, Hansen, Igel, "Improved Step Size Adaptation
       for the MO-CMA-ES", 2010.

    """
    def __init__(self, population, sigma, **params):
        self.sim_type = params.get("sim_type", 1)
        print(f'self.sim_type = {self.sim_type}')
        self.p4_treatment = params.get("p4_treatment")
        print(f'self.p4_treatment = {self.p4_treatment}')
        self.bounds = params.get("bounds")
        print(f"bounds = {self.bounds}")
        self.parents = population
        self.dim = len(self.parents[0])

        # Selection
        self.mu = params.get("mu", len(self.parents))
        self.lambda_ = params.get("lambda_", 1)

        # Step size control

        # Can I Change these to help with the step size problem?

        self.d = params.get("d", 1.0 + self.dim / 2.0)
        self.ptarg = params.get("ptarg", 1.0 / (5.0 + 0.5))
        self.cp = params.get("cp", self.ptarg / (2.0 + self.ptarg))

        # Covariance matrix adaptation
        self.cc = params.get("cc", 2.0 / (self.dim + 2.0))
        self.ccov = params.get("ccov", 2.0 / (self.dim ** 2 + 6.0))
        self.pthresh = params.get("pthresh", 0.44)

        # Internal parameters associated to the mu parent
        self.sigmas = [sigma] * len(population)
        # Lower Cholesky matrix (Sampling matrix)
        self.A = [np.identity(self.dim) for _ in range(len(population))]
        # Inverse Cholesky matrix (Used in the update of A)
        self.invCholesky = [np.identity(self.dim) for _ in range(len(population))]
        self.pc = [np.zeros(self.dim) for _ in range(len(population))]
        self.psucc = [self.ptarg] * len(population)

        self.indicator = params.get("indicator", tools.hypervolume)

        self.time_spent_fixing = 0

    def generate(self, ind_init):
        """Generate a population of :math:`\lambda` individuals of type
        *ind_init* from the current strategy.

        :param ind_init: A function object that is able to initialize an
                         individual from a list.
        :returns: A list of individuals with a private attribute :attr:`_ps`.
                  This last attribute is essential to the update function, it
                  indicates that the individual is an offspring and the index
                  of its parent.
        """
        arz = np.random.randn(self.lambda_, self.dim)
        individuals = list()

        # Make sure every parent has a parent tag and index
        for i, p in enumerate(self.parents):
            p._ps = "p", i

        # Each parent produces an offspring
        if self.lambda_ == self.mu:
            for i in range(self.lambda_):
                # print("Z", list(arz[i]))
                # print(f"A[i] = {self.A[i]}")
                # print(f"sigmas = {self.sigmas[i]}")
                mutation = self.sigmas[i] * np.dot(self.A[i], arz[i])
                # print("mutation", mutation)
                # print("parent", self.parents[i])
                new_individual = self.parents[i] + mutation
                # print("new_individual", new_individual)
                # print('*'*30, '\n')

                # checks to see if the new individual is feasible or not
                if self.sim_type != 'Penalty':

                    s = time.time()
                    while True:
                        # print(f'\nself.check_feasibility(new_individual)[0] = {self.check_feasibility(new_individual)[0]}')
                        if not self.check_feasibility(new_individual)[0]:

                            if f'{toolbox.logbook.bookshelf["generation"]}' in [gen for gen in toolbox.logbook.bookshelf["fixer_count"].keys()]:
                                toolbox.logbook.bookshelf["fixer_count"][f"{toolbox.logbook.bookshelf['generation']}"] += 1
                            else:
                                toolbox.logbook.bookshelf['fixer_count'][f"{toolbox.logbook.bookshelf['generation']}"] = 1

                            bad_attribute = self.check_feasibility(new_individual)[1]
                            # print(f'bad attribute = {bad_attribute}')
                            # print("parent", self.parents[i])
                            # print(f'parent p4 = {variable_untransformation(self.parents[i], self.bounds)[2]}')
                            new_individual = self.crossover(new_individual, bad_attribute, i)
                            # print(f'new_individual = {new_individual}\n')
                        else:
                            break
                    e = time.time()
                    if f'{toolbox.logbook.bookshelf["generation"]}' not in [gen for gen in toolbox.logbook.bookshelf["fixer_count"].keys()]:
                        toolbox.logbook.bookshelf['fixer_count'][f"{toolbox.logbook.bookshelf['generation']}"] = 0

                    toolbox.logbook.bookshelf['time_spent_fixing'] += e - s

                individuals.append(ind_init(new_individual))
                individuals[-1]._ps = "o", i

        # Parents producing an offspring are chosen at random from the first front
        else:
            ndom = tools.sortLogNondominated(self.parents, len(self.parents), first_front_only=True)
            for i in range(self.lambda_):
                j = np.random.randint(0, len(ndom))
                _, p_idx = ndom[j]._ps
                individuals.append(ind_init(self.parents[p_idx] + self.sigmas[p_idx] * np.dot(self.A[p_idx], arz[i])))
                individuals[-1]._ps = "o", p_idx

        return individuals

    def _select(self, candidates):
        if len(candidates) <= self.mu:
            return candidates, []

        pareto_fronts = tools.sortLogNondominated(candidates, len(candidates))

        chosen = list()
        mid_front = None
        not_chosen = list()
            

        # Fill the next population (chosen) with the fronts until there is not enough space
        # When an entire front does not fit in the space left we rely on the hypervolume
        # for this front
        # The remaining fronts are explicitly not chosen
        full = False
        for front in pareto_fronts:
            # print(f'\nlen(front) = {len(front)}')
            # print(f'len(chosen) = {len(chosen)}')
            # print(f'len(not_chosen) = {len(not_chosen)}')
            # for ind in front:
            #     print(f'ind = {ind}\nfitness = {ind.fitness.values}, _ps = {ind._ps}')
            # print('\nPrinting selected individuals:')
            # for selected in chosen:
            #     print(f'selected = {selected}\nfitness = {selected.fitness.values}, _ps = {selected._ps}')

            if len(chosen) + len(front) <= self.mu and not full:
                chosen += front
            elif mid_front is None and len(chosen) < self.mu:
                mid_front = front
                # With this front, we selected enough individuals
                full = True
            else:
                not_chosen += front

        # Separate the mid front to accept only k individuals
        k = self.mu - len(chosen)

        if k > 0:
            # reference point is chosen in the complete population
            # as the worst in each dimension +1
            ref = np.array([ind.fitness.wvalues for ind in candidates]) * -1
            # print(f"ref = {ref}")
            ref = np.max(ref, axis=0) + 1
            # print(f"ref = {ref}")
            # print(f"ref type = {type(ref)}")

            for _ in range(len(mid_front) - k):
                idx = self.indicator(mid_front, ref=ref)
                not_chosen.append(mid_front.pop(idx))

            chosen += mid_front
        # print('\n final selection:')
        # print(f'len(chosen) = {len(chosen)}')
        # print(f'len(not_chosen) = {len(not_chosen)}')
        return chosen, not_chosen

    def _rankOneUpdate(self, invCholesky, A, alpha, beta, v):
        w = np.dot(invCholesky, v)

        # Under this threshold, the update is mostly noise
        if w.max() > 1e-20:
            w_inv = np.dot(w, invCholesky)
            norm_w2 = np.sum(w ** 2)
            a = np.sqrt(alpha)
            root = np.sqrt(1 + beta / alpha * norm_w2)
            b = a / norm_w2 * (root - 1)

            A = a * A + b * np.outer(v, w)
            invCholesky = 1.0 / a * invCholesky - b / (a ** 2 + a * b * norm_w2) * np.outer(w, w_inv)

        return invCholesky, A

    def update(self, population):
        """Update the current covariance matrix strategies from the
        *population*.

        :param population: A list of individuals from which to update the
                           parameters.
        """
        chosen, not_chosen = self._select(population + self.parents)

        # print(f"Chosen: {chosen}, Not chosen: {not_chosen}")

        cp, cc, ccov = self.cp, self.cc, self.ccov
        d, ptarg, pthresh = self.d, self.ptarg, self.pthresh

        # Make copies for chosen offspring only
        last_steps = [self.sigmas[ind._ps[1]] if ind._ps[0] == "o" else None for ind in chosen]
        sigmas = [self.sigmas[ind._ps[1]] if ind._ps[0] == "o" else None for ind in chosen]
        invCholesky = [self.invCholesky[ind._ps[1]].copy() if ind._ps[0] == "o" else None for ind in chosen]
        A = [self.A[ind._ps[1]].copy() if ind._ps[0] == "o" else None for ind in chosen]
        pc = [self.pc[ind._ps[1]].copy() if ind._ps[0] == "o" else None for ind in chosen]
        psucc = [self.psucc[ind._ps[1]] if ind._ps[0] == "o" else None for ind in chosen]

        # Update the internal parameters for successful offspring
        for i, ind in enumerate(chosen):
            t, p_idx = ind._ps

            # Only the offspring update the parameter set
            if t == "o":
                # print(f"ind = {ind}, p_idx = {p_idx}")
                # Update (Success = 1 since it is chosen)
                psucc[i] = (1.0 - cp) * psucc[i] + cp
                # print(f"psucc: {psucc[i]}")
                sigmas[i] = sigmas[i] * np.exp((psucc[i] - ptarg) / (d * (1.0 - ptarg)))
                print(f"sigmas: {sigmas[i]}")

                if psucc[i] < pthresh:
                    xp = np.array(ind)
                    x = np.array(self.parents[p_idx])
                    pc[i] = (1.0 - cc) * pc[i] + np.sqrt(cc * (2.0 - cc)) * (xp - x) / last_steps[i]
                    invCholesky[i], A[i] = self._rankOneUpdate(invCholesky[i], A[i], 1 - ccov, ccov, pc[i])
                else:
                    pc[i] = (1.0 - cc) * pc[i]
                    pc_weight = cc * (2.0 - cc)
                    invCholesky[i], A[i] = self._rankOneUpdate(invCholesky[i], A[i], 1 - ccov + pc_weight, ccov, pc[i])

                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx] + cp
                self.sigmas[p_idx] = self.sigmas[p_idx] * np.exp((self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg)))

        # It is unnecessary to update the entire parameter set for not chosen individuals
        # Their parameters will not make it to the next generation
        for ind in not_chosen:
            t, p_idx = ind._ps

            # Only the offspring update the parameter set
            if t == "o":
                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx]
                self.sigmas[p_idx] = self.sigmas[p_idx] * np.exp((self.psucc[p_idx] - ptarg) / (d * (1.0 - ptarg)))

        # Make a copy of the internal parameters
        # The parameter is in the temporary variable for offspring and in the original one for parents
        self.parents = chosen
        self.sigmas = [sigmas[i] if ind._ps[0] == "o" else self.sigmas[ind._ps[1]] for i, ind in enumerate(chosen)]
        self.invCholesky = [invCholesky[i] if ind._ps[0] == "o" else self.invCholesky[ind._ps[1]] for i, ind in enumerate(chosen)]
        self.A = [A[i] if ind._ps[0] == "o" else self.A[ind._ps[1]] for i, ind in enumerate(chosen)]
        self.pc = [pc[i] if ind._ps[0] == "o" else self.pc[ind._ps[1]] for i, ind in enumerate(chosen)]
        self.psucc = [psucc[i] if ind._ps[0] == "o" else self.psucc[ind._ps[1]] for i, ind in enumerate(chosen)]

    def crossover(self, individual, attribute_index, i):

        def swap_attribute(individual, crossover_individual_index, attribute_index):
            crossover_individual = self.parents[crossover_individual_index]

            if attribute_index == 2 and self.p4_treatment == "hard_bounds_on_p4":
            # This fixes an issue that arises when you try to swap a p4 value that is above the hard limit when its in its transformed coordinates
                swapped_p4_natural_coordinates = variable_untransformation(self.parents[i], self.bounds)[2]
                child_driver_p_nat_coords = variable_untransformation(individual, self.bounds)[1]
                # print(f'swapped p4 in natural coordinates = {swapped_p4_natural_coordinates}')
                # print(f'child driver p in natural coordinates = {child_driver_p_nat_coords}')
                swapped_p4_transformed = (swapped_p4_natural_coordinates - 14.62 * child_driver_p_nat_coords) / (1190.63 * child_driver_p_nat_coords - 14.62 * child_driver_p_nat_coords) + 1 # p4
                # print(f'Swapped p4 in transformed coordinates = {swapped_p4_transformed}')

                if not 1 <= swapped_p4_transformed <= 2:
                    individual[1] = crossover_individual[1] # retains driver p
                    individual[attribute_index] = crossover_individual[attribute_index] # retains p4
                    # print(f"We needed to fix driver p as well")
                    # print(f'parent driver p = {variable_untransformation(crossover_individual, self.bounds)[1]}')
                    # print(f'New individual = {individual}')
                    # print(f'New individual in natural coordinates = {variable_untransformation(individual, self.bounds)}')
                else:
                    individual[attribute_index] = swapped_p4_transformed
                # print(f'untransforming swapped p4 = {variable_untransformation(individual, self.bounds)[2]}')
            else:    
                individual[attribute_index] = crossover_individual[attribute_index]

        # attribute_list = collect_attributes(self.parents, attribute_index)

        if self.sim_type == 'ElitistCrossover':
            # calculates the reference point for hypervolume calculation using the worst feasible individual
            ref = np.array([ind.fitness.wvalues for ind in self.parents]) * -1
            # print(f"ref = {ref}")
            ref = np.max(ref, axis=0) + 1
            # print(f"ref = {ref}")
            crossover_individual_index = self.indicator(self.parents, ref=ref)
            swap_attribute(individual, crossover_individual_index, attribute_index)

        if self.sim_type == 'RandomCrossover':
            attribute_list = [(population_index, ind[attribute_index]) for population_index, ind in
                              enumerate(self.parents)]
            crossover_individual_index = attribute_list[np.random.randint(0, len(attribute_list))][0]
            swap_attribute(individual, crossover_individual_index, attribute_index) # recieves a new feasible attribute from a
            # randomly selected individual from the parent population

        if self.sim_type == 'ParentValue':
            swap_attribute(individual, i, attribute_index) # maintains the feasible attribute from it's parent
        return individual

    def check_feasibility(self, new_individual):

        # print(f'new individual = {new_individual}')
    
        if self.p4_treatment == "hard_bounds_on_p4":   
            p4_upper = self.bounds[2][1] 
            p4_real_value = variable_untransformation(new_individual, self.bounds)[2]

            # Checking constraint on p4
            if p4_real_value < p4_upper:
                pass
            else:
                # print(f'p4 = {p4_real_value}')
                return (False, 2)
        # Checking constraint on all other variables, including compression ratio
        for j in range(len(new_individual)):
            if 1 <= new_individual[j] <= 2:
                pass
            else:
                return (False, j)

        return (True, 0)
    


class ClosestValidPenalty(object):
    r"""This decorator returns penalized fitness for invalid individuals and the
    original fitness value for valid individuals. The penalized fitness is made
    of the fitness of the closest valid individual added with a weighted
    (optional) *distance* penalty. The distance function, if provided, shall
    return a value growing as the individual moves away the valid zone.

    :param feasibility: A function returning the validity status of any
                        individual.
    :param feasible: A function returning the closest feasible individual
                     from the current invalid individual.
    :param alpha: Multiplication factor on the distance between the valid and
                  invalid individual.
    :param distance: A function returning the distance between the individual
                     and a given valid point. The distance function can also return a sequence
                     of length equal to the number of objectives to affect multi-objective
                     fitnesses differently (optional, defaults to 0).
    :returns: A decorator for evaluation function.

    This function relies on the fitness weights to add correctly the distance.
    The fitness value of the ith objective is defined as

    .. math::

       f^\mathrm{penalty}_i(\mathbf{x}) = f_i(\operatorname{valid}(\mathbf{x})) - \\alpha w_i d_i(\operatorname{valid}(\mathbf{x}), \mathbf{x})

    where :math:`\mathbf{x}` is the individual,
    :math:`\operatorname{valid}(\mathbf{x})` is a function returning the closest
    valid individual to :math:`\mathbf{x}`, :math:`\\alpha` is the distance
    multiplicative factor and :math:`w_i` is the weight of the ith objective.
    """

    def __init__(self, feasibility, feasible, alpha, distance=None):
        self.fbty_fct = feasibility
        self.fbl_fct = feasible
        self.alpha = alpha
        self.dist_fct = distance


    def wrapper(individual, *args, **kwargs):
        if self.fbty_fct(individual):
            return func(individual, *args, **kwargs)

        f_ind = self.fbl_fct(individual)
        print("individual", f_ind)
        f_fbl = func(f_ind, *args, **kwargs)
        print("feasible", f_fbl)

        weights = tuple(1.0 if w >= 0 else -1.0 for w in individual.fitness.weights)

        if len(weights) != len(f_fbl):
            raise IndexError("Fitness weights and computed fitness are of different size.")

        dists = tuple(0 for w in individual.fitness.weights)
        if self.dist_fct is not None:
            dists = self.dist_fct(f_ind, individual)
            if not isinstance(dists, Sequence):
                dists = repeat(dists)

        print("penalty ", tuple(  - w * self.alpha * d for f, w, d in zip(f_fbl, weights, dists)))
        print("returned", tuple(f - w * self.alpha * d for f, w, d in zip(f_fbl, weights, dists)))
        return tuple(f - w * self.alpha * d for f, w, d in zip(f_fbl, weights, dists))


class HyperVolume:
    """
    Hypervolume computation based on variant 3 of the algorithm in the paper:
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep
    algorithm for the hypervolume indicator. In IEEE Congress on Evolutionary
    Computation, pages 1157-1163, Vancouver, Canada, July 2006.

    Minimization is implicitly assumed here!

    """

    def __init__(self, referencePoint):
        """Constructor."""
        self.referencePoint = referencePoint
        self.list = []

    def compute(self, front):
        """Returns the hypervolume that is dominated by a non-dominated front.

        Before the HV computation, front and reference point are translated, so
        that the reference point is [0, ..., 0].

        """

        def weaklyDominates(point, other):
            for i in range(len(point)):
                if point[i] > other[i]:
                    return False
            return True

        relevantPoints = []
        referencePoint = self.referencePoint
        dimensions = len(referencePoint)
        #######
        # fmder: Here it is assumed that every point dominates the reference point
        # for point in front:
        #     # only consider points that dominate the reference point
        #     if weaklyDominates(point, referencePoint):
        #         relevantPoints.append(point)
        relevantPoints = front
        # fmder
        #######
        if any(referencePoint):
            # shift points so that referencePoint == [0, ..., 0]
            # this way the reference point doesn't have to be explicitly used
            # in the HV computation

            #######
            # fmder: Assume relevantPoints are numpy array
            # for j in xrange(len(relevantPoints)):
            #     relevantPoints[j] = [relevantPoints[j][i] - referencePoint[i] for i in xrange(dimensions)]
            # print(f"Relevant points: {relevantPoints}")
            # print(f"referencePoint: {referencePoint}")
            relevantPoints -= referencePoint
            # fmder
            #######

        self.preProcess(relevantPoints)
        bounds = [-1.0e308] * dimensions
        hyperVolume = self.hvRecursive(dimensions - 1, len(relevantPoints), bounds)
        return hyperVolume

    def hvRecursive(self, dimIndex, length, bounds):
        """Recursive call to hypervolume calculation.

        In contrast to the paper, the code assumes that the reference point
        is [0, ..., 0]. This allows the avoidance of a few operations.

        """
        hvol = 0.0
        sentinel = self.list.sentinel
        if length == 0:
            return hvol
        elif dimIndex == 0:
            # special case: only one dimension
            # why using hypervolume at all?
            return -sentinel.next[0].cargo[0]
        elif dimIndex == 1:
            # special case: two dimensions, end recursion
            q = sentinel.next[1]
            h = q.cargo[0]
            p = q.next[1]
            while p is not sentinel:
                pCargo = p.cargo
                hvol += h * (q.cargo[1] - pCargo[1])
                if pCargo[0] < h:
                    h = pCargo[0]
                q = p
                p = q.next[1]
            hvol += h * q.cargo[1]
            return hvol
        else:
            remove = self.list.remove
            reinsert = self.list.reinsert
            hvRecursive = self.hvRecursive
            p = sentinel
            q = p.prev[dimIndex]
            while q.cargo is not None:
                if q.ignore < dimIndex:
                    q.ignore = 0
                q = q.prev[dimIndex]
            q = p.prev[dimIndex]
            while length > 1 and (q.cargo[dimIndex] > bounds[dimIndex] or q.prev[dimIndex].cargo[dimIndex] >= bounds[dimIndex]):
                p = q
                remove(p, dimIndex, bounds)
                q = p.prev[dimIndex]
                length -= 1
            qArea = q.area
            qCargo = q.cargo
            qPrevDimIndex = q.prev[dimIndex]
            if length > 1:
                hvol = qPrevDimIndex.volume[dimIndex] + qPrevDimIndex.area[dimIndex] * (qCargo[dimIndex] - qPrevDimIndex.cargo[dimIndex])
            else:
                qArea[0] = 1
                qArea[1:dimIndex+1] = [qArea[i] * -qCargo[i] for i in range(dimIndex)]
            q.volume[dimIndex] = hvol
            if q.ignore >= dimIndex:
                qArea[dimIndex] = qPrevDimIndex.area[dimIndex]
            else:
                qArea[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                if qArea[dimIndex] <= qPrevDimIndex.area[dimIndex]:
                    q.ignore = dimIndex
            while p is not sentinel:
                pCargoDimIndex = p.cargo[dimIndex]
                hvol += q.area[dimIndex] * (pCargoDimIndex - q.cargo[dimIndex])
                bounds[dimIndex] = pCargoDimIndex
                reinsert(p, dimIndex, bounds)
                length += 1
                q = p
                p = p.next[dimIndex]
                q.volume[dimIndex] = hvol
                if q.ignore >= dimIndex:
                    q.area[dimIndex] = q.prev[dimIndex].area[dimIndex]
                else:
                    q.area[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                    if q.area[dimIndex] <= q.prev[dimIndex].area[dimIndex]:
                        q.ignore = dimIndex
            hvol -= q.area[dimIndex] * q.cargo[dimIndex]
            return hvol

    def preProcess(self, front):
        """Sets up the list data structure needed for calculation."""
        dimensions = len(self.referencePoint)
        nodeList = _MultiList(dimensions)
        nodes = [_MultiList.Node(dimensions, point) for point in front]
        for i in range(dimensions):
            self.sortByDimension(nodes, i)
            nodeList.extend(nodes, i)
        self.list = nodeList

    def sortByDimension(self, nodes, i):
        """Sorts the list of nodes by the i-th value of the contained points."""
        # build a list of tuples of (point[i], node)
        decorated = [(node.cargo[i], node) for node in nodes]
        # sort by this value
        decorated.sort()
        # write back to original list
        nodes[:] = [node for (_, node) in decorated]


class _MultiList:
    """A special data structure needed by FonsecaHyperVolume.

    It consists of several doubly linked lists that share common nodes. So,
    every node has multiple predecessors and successors, one in every list.

    """

    class Node:

        def __init__(self, numberLists, cargo=None):
            self.cargo = cargo
            self.next = [None] * numberLists
            self.prev = [None] * numberLists
            self.ignore = 0
            self.area = [0.0] * numberLists
            self.volume = [0.0] * numberLists

        def __str__(self):
            return str(self.cargo)

        def __lt__(self, other):
            return all(self.cargo < other.cargo)

    def __init__(self, numberLists):
        """Constructor.

        Builds 'numberLists' doubly linked lists.

        """
        self.numberLists = numberLists
        self.sentinel = _MultiList.Node(numberLists)
        self.sentinel.next = [self.sentinel] * numberLists
        self.sentinel.prev = [self.sentinel] * numberLists

    def __str__(self):
        strings = []
        for i in range(self.numberLists):
            currentList = []
            node = self.sentinel.next[i]
            while node != self.sentinel:
                currentList.append(str(node))
                node = node.next[i]
            strings.append(str(currentList))
        stringRepr = ""
        for string in strings:
            stringRepr += string + "\n"
        return stringRepr

    def __len__(self):
        """Returns the number of lists that are included in this _MultiList."""
        return self.numberLists

    def getLength(self, i):
        """Returns the length of the i-th list."""
        length = 0
        sentinel = self.sentinel
        node = sentinel.next[i]
        while node != sentinel:
            length += 1
            node = node.next[i]
        return length

    def append(self, node, index):
        """Appends a node to the end of the list at the given index."""
        lastButOne = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = lastButOne
        # set the last element as the new one
        self.sentinel.prev[index] = node
        lastButOne.next[index] = node

    def extend(self, nodes, index):
        """Extends the list at the given index with the nodes."""
        sentinel = self.sentinel
        for node in nodes:
            lastButOne = sentinel.prev[index]
            node.next[index] = sentinel
            node.prev[index] = lastButOne
            # set the last element as the new one
            sentinel.prev[index] = node
            lastButOne.next[index] = node

    def remove(self, node, index, bounds):
        """Removes and returns 'node' from all lists in [0, 'index'[."""
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node

    def reinsert(self, node, index, bounds):
        """
        Inserts 'node' at the position it had in all lists in [0, 'index'[
        before it was removed. This method assumes that the next and previous
        nodes of the node that is reinserted are in the list.

        """
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]


creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))

# Creates an individual type whose data structure is a list
creator.create("Individual", list, fitness=creator.FitnessMulti, ind_number=int, sim_type=str, bounds=list)

# Toolbox is a container for operators that can be used throughout the simulation
# Also convenient to enable simulation information to be stored in vitro
toolbox = Toolbox()
toolbox.register("evaluate", evaluate)
pop_hypervolumes = HyperVolume(np.array((0, 0, 0)))
# print(f'ref point = {np.array(normalise_fitness(APPROX_IDEAL, APPROX_IDEAL, APPROX_NADIR))}')
# toolbox.decorate("evaluate", ClosestValidPenalty(valid, closest_feasible, 1.0e+6, distance))

def main(experiment_type):

    s1 = time.time()
    ###########################################################
    # Sim Parameters
    ###########################################################
    N = 6
    # pop_size = experiment_type[1]
    pop_size = experiment_type[1]
    MU, LAMBDA = pop_size, pop_size
    NGEN = 750
    sim_type = experiment_type[0]
    p4_treatment = experiment_type[3]
    step_size = experiment_type[2]
    print(f"Step Size = {step_size}")
    print(f'Pop Size = {pop_size}')

    ###########################################################
    # Stats:
    ###########################################################
    gen_counter = 0

    toolbox.logbook.add('generation', gen_counter)
    toolbox.logbook.add('fixer_count', {f"{gen_counter}": 0})
    toolbox.logbook.add('time_spent_fixing', 0)
    toolbox.logbook.add("time taken", 0)
    toolbox.logbook.add("No. individuals that failed objective tests", 0)
    toolbox.logbook.add("No. individuals that produced no hold time", 0)
    toolbox.logbook.add("No. individuals that failed constraint tests", 0)
    # toolbox.logbook.add("average constraint value", [0 for _ in range(1, NGEN+1)])
    # toolbox.logbook.add("std constraint value", [0 for _ in range(1, NGEN+1)])
    # toolbox.logbook.add("average objective value", [0 for _ in range(1, NGEN+1)])
    # toolbox.logbook.add("std objective value", [0 for _ in range(1, NGEN+1)])
    toolbox.logbook.add("hypervolume", [0 for _ in range(1, NGEN+1)])

    ###########################################################
    # Initialisation
    ##########################################################

    he_lower, he_upper = 70, 100
    D_throat_lower, D_throat_upper = 0.05, 0.085
    driver_p_lower, driver_p_upper = 1000, (40/14.62)*1e6
    p4_lower, p4_upper = 150e3, 40e6
    reservoir_lower, reservoir_upper = 1000, 8e6
    buffer_length_lower, buffer_length_upper = 0.05, 0.15

    bounds = [(he_lower, he_upper), (driver_p_lower, driver_p_upper),
              (p4_lower, p4_upper), (D_throat_lower, D_throat_upper), (reservoir_lower, reservoir_upper), (buffer_length_lower, buffer_length_upper)]

    def pop_init(MU):
        percent_he_list = np.random.uniform(he_lower, he_upper, (MU, 1))
        D_throat_list = np.random.uniform(D_throat_lower, D_throat_upper, (MU, 1))
        driver_p_list = np.random.uniform(driver_p_lower, driver_p_upper, (MU, 1))
        buffer_length_list = np.random.uniform(buffer_length_lower, buffer_length_upper, (MU, 1))
        # print(f'buffer_length_list = {buffer_length_list}')
        p4_list = np.zeros_like(percent_he_list)
        for i in range(MU):
            if 1190.63*driver_p_list[i] < p4_upper:
                p4_list[i] = np.random.uniform(14.62 * driver_p_list[i], 1190.63 * driver_p_list[i])
                # print(f'p4 -1- {i} = {p4_list[i]*10**(-6)}')
            else:
                p4_list[i] = np.random.uniform(14.62 * driver_p_list[i], p4_upper)
                # print(f'p4 -2- {i} = {p4_list[i]*10**(-6)}')

        reservoir_p_list = [np.random.uniform(driver_p_list[i], reservoir_upper) for i in range(MU)]

        return [[percent_he_list[i][0], driver_p_list[i][0], p4_list[i][0], D_throat_list[i][0], reservoir_p_list[i][0], buffer_length_list[i][0]]
                for i in range(MU)]

    i = 0
    init_pop_untransformed = pop_init(MU)
    init_pop_transformed = variable_transformation(init_pop_untransformed, bounds)
    # print(f'Initial population in transformed coordinates = {init_pop_transformed}')

    # time.sleep( 10)
    # print('\n')
    # for printing_ind in init_pop_untransformed:
        # print(f"Initial population = {printing_ind}")
    # print('\n')
    # for printing_ind in init_pop_transformed:
        # print(f"Transformed population = {printing_ind}")
    # check_variable_transformation = [variable_untransformation(ind_i, bounds) for ind_i in init_pop_transformed]
    # print('\n')
    # for printing_ind in check_variable_transformation:
        # print(f"Untransformed population = {printing_ind}")
    population = [creator.Individual(x) for x in init_pop_transformed]
    initial_population = population
    for ind in population:
        ind.ind_number = i
        ind.bounds = bounds
        i += 1

    parallelization_setup(population)

    for ind in population:
        ind.sim_type = sim_type
        ind.normalised = normalised
        ind.fitness.values = toolbox.evaluate(ind) # When I call this, I pass information into the fitness functions

    strategy = StrategyMultiObjective(population, sigma=step_size, mu=MU, lambda_=LAMBDA, sim_type=sim_type, p4_treatment=p4_treatment, bounds=bounds)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    fitness_history = []

    # constraint_stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    # constraint_stats.register("avg", np.mean)
    # constraint_stats.register("std", np.std)
    # constraint_stats.register("min", np.min)
    # constraint_stats.register("max", np.max)

    # objective_stats = tools.Statistics(lambda ind: ind.fitness.values[1])
    # objective_stats.register("avg", np.mean)
    # objective_stats.register("std", np.std)
    # objective_stats.register("min", np.min)
    # objective_stats.register("max", np.max)

    # toolbox.logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # # Objects that will compile the data
    # sigma = np.ndarray((NGEN, LAMBDA))
    # std = np.ndarray((NGEN, N))

    ###########################################################
    # Evolution:
    ###########################################################
    for gen in range(NGEN):
        toolbox.logbook.bookshelf['generation'] += 1
        print('\n')
        print('*'*30)
        print(f"Generation {toolbox.logbook.bookshelf['generation']}")
        print('*' * 30)

        # Generating a new population
        parents = population
        population = toolbox.generate()

        i = 0
        for ind in population:
            ind.normalised = normalised
            ind.ind_number = i
            ind.bounds = bounds
            i += 1

        try:
            fitnesses = toolbox.map(toolbox.evaluate, population)
        except:
            time.sleep(1)
            try:
                fitnesses = toolbox.map(toolbox.evaluate, population)
            except:
                time.sleep(1)
                try:
                    fitnesses = toolbox.map(toolbox.evaluate, population)
                except:
                    time.sleep(1)
                    fitnesses = toolbox.map(toolbox.evaluate, population)

        # print(f'\n')
        # print(f'New population fitnesses')
        fixed = False
        for i, (ind, fit) in enumerate(zip(population, fitnesses)):
            normalised_shock_speed = fit[0]

            if normalised_shock_speed == 1.0:
                # Replace individual in population with parent
                replacement = parents[ind.ind_number]
                new_fitness = replacement.fitness.values

                # with open("diagnostics1.txt", "a") as file:  # use "a" to avoid overwriting every time
                #     print(f'Individual {ind.ind_number} failed objective tests', file=file)
                #     print(f'individual = {ind}', file=file)
                #     print(f'fitness = {fit}', file=file)
                #     print(f'parents ind.fitness values = {new_fitness}', file=file)
                #     print(f'New Individual = {replacement}, New fitness = {new_fitness}', file=file)
                #     print(f'\nPotential individuals (i.e., the parents to select from) are:', file=file)
                #     for pot_ind in parents:
                #         print(f'{pot_ind}', file=file)

                population[i] = replacement
                population[i].fitness.values = new_fitness  # Ensure it's explicitly re-set

                fixed = True
                fitness_history.append(new_fitness)
            else:
                new_fitness = fit
                ind.fitness.values = new_fitness
                fitness_history.append(new_fitness)

        # selects new population and updates parameters based on new fitnesses
        # if fixed:
        #     with open("diagnostics2.txt", "w") as file:
        #         file.write(f"Simulation Type = {sim_type}\n")
        #         file.write(f"step size = {experiment_type[2]}\n")
        #         file.write(f'pop size = {pop_size}\n')
        #         file.write(f'p4 treatment = {p4_treatment}\n')
        #         file.write(f"time taken = {toolbox.logbook.bookshelf['time taken']}\n")
        #         file.write(f'Number of generations = {gen}\n')
        #         file.write(f'Current Time = {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
        #         print('\nwe fixed something. Printing population as diagnostics', file=file)
        #         for ind in population:
        #             print(f'Individual = {ind}, ind_number = {ind.ind_number}', file=file)
        #             print(f'fitness = {ind.fitness.values}', file=file)

        toolbox.update(population)
        fitness_copy = fitnesses

        for fit in fitness_copy:
            if fit[0] == 1.0:
                fitness_copy.remove(fit)

        avg_hypervolume = pop_hypervolumes.compute(np.array(fitness_copy) * -1)
        print(f'average hypervolume = {avg_hypervolume}')
        toolbox.logbook.bookshelf['hypervolume'][gen] = avg_hypervolume

        if gen % 50 == 0 and 200 <= gen <= 750:

            starting_working_directory = os.getcwd()
            if starting_working_directory[-1] in [f'{i}' for i in range(0, 12)]:
                starting_working_directory = starting_working_directory[:35]

            # change directory to the one of the simulation
            os.chdir(starting_working_directory + '/' + 'Scatter_Plots')

            plot_objective_space(fitness_history, 'delta_vs1', 'hold_time', MU=MU, sim_type=sim_type, gen=gen)
            plot_objective_space(fitness_history, 'delta_vs1', 'impact_speed', MU=MU, sim_type=sim_type, gen=gen)
            plot_objective_space(fitness_history, 'hold_time', 'impact_speed', MU=MU, sim_type=sim_type, gen=gen)

            os.chdir(starting_working_directory)



        # Update the hall of fame and the statistics with the
        # currently evaluated population
        # constraint_record = constraint_stats.compile(population)
        # objective_record = objective_stats.compile(population)

        # toolbox.logbook.record(evals=len(population), gen=gen, **constraint_record)
        # toolbox.logbook.record(evals=len(population), gen=gen, **objective_record)

        # toolbox.logbook.bookshelf['average constraint value'][gen] = constraint_record['avg']
        # toolbox.logbook.bookshelf['average objective value'][gen] = objective_record['avg']
        # toolbox.logbook.bookshelf['std constraint value'][gen] = constraint_record['std']
        # toolbox.logbook.bookshelf['std objective value'][gen] = objective_record['std']

        # Save more data along the evolution for later plotting
        # sigma[gen] = strategy.sigmas
        # std[gen, :N] = np.std(population, axis=0)

    ###################################################
    # post processing
    ###################################################

    starting_working_directory = os.getcwd()
    title_string = f'MOO_CMA_ES_{sim_type}'

    if title_string not in [directory for directory in os.listdir(starting_working_directory)]:
        os.mkdir(title_string)
    os.chdir(starting_working_directory + '/' + title_string)

    previous_experiment_list = []
    os.chdir(starting_working_directory + '/' + title_string)
    new_working_directory = os.getcwd()

    for directory in os.listdir():
        directory_copy = directory[:-2]
        if directory_copy == title_string:
            previous_experiment_list.append(directory)
    if len(previous_experiment_list) == 0:
        test_name = f'MOO_CMA_ES_{sim_type}_1'
        os.mkdir(test_name)
    else:
        experiment_number_list = [int(directory[-1]) for directory in previous_experiment_list]
        experiment_number = np.max(experiment_number_list) + 1
        test_name = f'MOO_CMA_ES_{sim_type}_{experiment_number}'
        os.mkdir(test_name)

    os.chdir(new_working_directory + '/' + test_name)

    #################################################
    # Writing convergence data to file
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

    ##################################################
    # Generic plot settings:

    x_range = NGEN
    no_ticks = 6
    tick_interval = x_range / (no_ticks - 1)

    ##################################################
    # scatter history of every individual

    plt.figure(dpi=800)
    plt.title("Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")

    plt.ylim((0, 1.1))
    gen = [i for i in range(1, NGEN+1)]
    avg_hypervolume_list = [toolbox.logbook.bookshelf['hypervolume'][generation - 1] for generation in gen]
    plt.plot(gen, avg_hypervolume_list)

    plt.savefig(f"convergence_{sim_type}.png")
    plt.close()

    ##################################################
    # scatter history of every individual

    plot_objective_space(fitness_history, 'delta_vs1', 'hold_time', MU=MU, sim_type=sim_type, gen=NGEN)

    plot_objective_space(fitness_history, 'delta_vs1', 'impact_speed', MU=MU, sim_type=sim_type, gen=NGEN)

    plot_objective_space(fitness_history, 'hold_time', 'impact_speed', MU=MU, sim_type=sim_type, gen=NGEN)

    ###################################################
    # printing information

    fixer_count = []
    running_total = 0
    for entry in toolbox.logbook.bookshelf["fixer_count"].keys():
        if entry == '0':
            pass
        else:
            running_total += toolbox.logbook.bookshelf["fixer_count"][entry]
            fixer_count.append(running_total)

    e1 = time.time()
    toolbox.logbook.bookshelf["time taken"] = e1 - s1

    initial_dimensionalised_fitness = [unnormalise_fitness(ind, APPROX_IDEAL, APPROX_NADIR) for ind in fitness_history[:MU]]
    final_dimensionalised_fitness = [unnormalise_fitness(ind, APPROX_IDEAL, APPROX_NADIR) for ind in fitness_history[-MU:]]
    # print(f'strtategy.parents = {strategy.parents}\n\n\n, dimensionalised_fitness = {dimensionalised_fitness}')

    # print(f"Final Population = {strategy.parents, dimensionalised_fitness}\n")

    sig_figs = 6

    with open('output.txt', 'w') as file:
        file.write(f"Simulation Type = {sim_type}\n")
        file.write(f"step size = {experiment_type[2]}\n")
        file.write(f'pop size = {pop_size}\n')
        file.write(f'p4 treatment = {p4_treatment}\n')
        file.write(f"time taken = {toolbox.logbook.bookshelf['time taken']}\n")
        file.write(f'Number of generations = {NGEN}\n')
        file.write(f'Current Time = {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
        file.write(f"Number of individuals that produced no hold time = {toolbox.logbook.bookshelf['No. individuals that produced no hold time']}\n")
        file.write(f"Fixer Count = {running_total}\n")
        file.write('*'*100)
        file.write('\n')
        file.write("INITIAL POPULATION:\n")
        file.write("Percent Helium |Driver Pressure (MPa) | p4 (MPa) | Throat Diameter (mm) | Reservoir Pressure (MPa) | Buffer Length (mm)\n")
        for ind in initial_population:
            string = f''
            for index, variable in enumerate(variable_untransformation(ind, bounds)):
                if index == 0:
                    variable = round(variable, sig_figs - str(variable).find('.'))
                    white_space = np.round((len('Percent Helium ') - len(f'{variable}')) / 2)
                    string += ' '*int(white_space) + f'{variable}' + ' '*int(white_space)
                elif index == 1:
                    variable *= 10 ** (-6)
                    variable = round(variable, sig_figs - str(variable).find('.'))
                    white_space = np.round((len('Driver Pressure (MPa) ') - len(f'{variable}')) / 2)
                    string += '|' + ' ' *int(white_space) + f'{variable}' + ' '*int(white_space) + '|'
                elif index == 2:
                    variable *= 10 ** (-6)
                    variable = round(variable, sig_figs - str(variable).find('.'))
                    white_space = np.round((len(' p4 (MPa) ') - len(f'{variable}')) / 2)
                    string += ' ' * int(white_space) + f'{variable}' + ' ' * int(white_space)
                elif index == 3:
                    variable *= 10 ** (3)
                    variable = round(variable, sig_figs - str(variable).find('.'))
                    white_space = np.round((len(' Throat Diameter (mm) ') - len(f'{variable}')) / 2)
                    string += '|' + ' ' * int(white_space) + f'{variable}' + ' ' * int(white_space)
                elif index == 4:
                    variable *= 10 ** (-6)
                    variable = round(variable, sig_figs - str(variable).find('.'))
                    white_space = np.round((len(' Reservoir Pressure (MPa) ') - len(f'{variable}')) / 2)
                    string += '|' + ' ' * int(white_space) + f'{variable}' + ' ' * int(white_space)
                elif index == 5:
                    variable *= 10**(3)
                    variable = round(variable, sig_figs - str(variable).find('.'))
                    white_space = np.round((len(' Buffer Length (mm) ') - len(f'{variable}')) / 2)
                    string += ' ' * int(white_space) + f'{variable}' + ' ' * int(white_space) 
            file.write(string)
            file.write('\n')

        file.write('\n')
        file.write(
            "Residual of Shock Speed (m/s) | Driver Hold Time (ms) | Piston Impact Speed (m/s)\n")
        for ind in initial_dimensionalised_fitness:
            string = f''
            for index, objective in enumerate(ind):
                objective = round(objective, sig_figs - str(objective).find('.'))
                if index == 0:
                    white_space = np.round((len('Residual of Shock Speed (m/s)') - len(f'{objective}')) / 2)
                    string += ' '*int(white_space) + f'{objective}' + ' '*int(white_space)
                elif index == 1:
                    objective *= 10**(3)
                    white_space = np.round((len(' Driver Hold Time (ms) ') - len(f'{objective}')) / 2)
                    string += ' '*int(white_space) + f'{objective}' + ' '*int(white_space)
                elif index == 2:
                    white_space = np.round((len(' Piston Impact Speed (m/s) ') - len(f'{objective}')) / 2)
                    string += ' '*int(white_space) + f'{objective}' + ' '*int(white_space)
            file.write(string)
            file.write('\n')


        file.write('\n')
        file.write('*' * 100)
        file.write('\n')

        file.write("FINAL POPULATION:\n")
        file.write(
            "Percent Helium | Driver Pressure (MPa) | p4 (MPa) | Throat Diameter (mm) | Reservoir Pressure (MPa) | Buffer length (mm)\n")
        for ind in strategy.parents:
            string = f''
            for index, variable in enumerate(variable_untransformation(ind, bounds)):
                if index == 0:
                    variable = round(variable, sig_figs - str(variable).find('.'))
                    white_space = np.round((len('Percent Helium ') - len(f'{variable}')) / 2)
                    print(f"white space = {white_space}")
                    string += ' ' * (int(white_space)) + f'{variable}' + ' ' * (int(white_space))
                elif index == 1:
                    variable *= 10 ** (-6)
                    variable = round(variable, sig_figs - str(variable).find('.'))
                    white_space = np.round((len(' Driver Pressure (MPa) ') - len(f'{variable}')) / 2)
                    print(f"white space = {white_space}")
                    string += ' ' * (int(white_space)) + f'{variable}' + ' ' * int(white_space)
                elif index == 2:
                    variable *= 10 ** (-6)
                    variable = round(variable, sig_figs - str(variable).find('.'))
                    white_space = np.round((len(' p4 (MPa) ') - len(f'{variable}')) / 2)
                    print(f"white space = {white_space}")
                    string += ' ' * int(white_space) + f'{variable}' + ' ' * int(white_space)
                elif index == 3:
                    variable *= 10 ** (3)
                    variable = round(variable, sig_figs - str(variable).find('.'))
                    white_space = np.round((len(' Throat Diameter (mm) ') - len(f'{variable}')) / 2)
                    string += ' ' * int(white_space) + f'{variable}' + ' ' * int(white_space)
                elif index == 4:
                    variable *= 10**(-6)
                    variable = round(variable, sig_figs - str(variable).find('.'))
                    white_space = np.round((len(' Reservoir Pressure (MPa) ') - len(f'{variable}')) / 2)
                    string += ' ' * int(white_space) + f'{variable}' + ' ' * int(white_space)
                elif index == 5:
                    variable *= 10**(3)
                    variable = round(variable, sig_figs - str(variable).find('.'))
                    white_space = np.round((len(' Buffer Length (mm) ') - len(f'{variable}')) / 2)
                    string += ' ' * int(white_space) + f'{variable}' + ' ' * int(white_space)    
            file.write(string)
            file.write('\n')


        file.write('\n')
        file.write(
            "Residual of Shock Speed (m/s) | Driver Hold Time (ms) | Piston Impact Speed (m/s)\n")
        for ind in final_dimensionalised_fitness:
            string = f''
            for index, objective in enumerate(ind):
                objective = round(objective, sig_figs - str(objective).find('.'))
                if index == 0:
                    white_space = np.round((len('Residual of Shock Speed (m/s)') - len(f'{objective}')) / 2)
                    string += ' ' * int(white_space) + f'{objective}' + ' ' * int(white_space)
                elif index == 1:
                    objective *= 10**(3)
                    white_space = np.round((len(' Driver Hold Time (s) ') - len(f'{objective}')) / 2)
                    string += ' '* int(white_space) + f'{objective}' + ' '* int(white_space)
                elif index == 2:
                    white_space = np.round((len(' Piston Impact Speed (m/s) ') - len(f'{objective}')) / 2)
                    string += ' '*int(white_space) + f'{objective}' + ' '*int(white_space)
            file.write(string)
            file.write('\n')


    ###################################################
    # cumulative fixer count per gen

    generation = [gen for gen in range(1, NGEN+1)]

    if sim_type != "Penalty":
        plt.figure(dpi=800)
        plt.title("Cumulative Number of Individuals Fixed")
        plt.xlabel("Generation")
        plt.ylabel("Number of Individuals Fixed")
        plt.plot(generation, fixer_count)

        plt.gca().xaxis.set_major_locator(MultipleLocator(tick_interval))
        # plt.xticks(np.arange(0, NGEN, NGEN/10))

        plt.savefig(f"cma_es_mo_fpd_{sim_type}RunningTotal.png")
        plt.close()

    #####################################################
    # constraint evolution over time

    # plt.figure(dpi=400)
    # plt.title("Constraint Value over time")
    # plt.xlabel("Generation")
    # plt.ylabel("Normalised Residual of Shock Speed")

    # plt.plot(generation, toolbox.logbook.bookshelf["average constraint value"], label="Average Shock Speed Residual", color="red")
    # plt.plot(generation, toolbox.logbook.bookshelf["std constraint value"], label="Std Shock Speed Residual", color="blue")
    # if normalised:
    #     # plt.xlim((0, 1.1))
    #     plt.ylim((0, 1.1))

    # plt.gca().xaxis.set_major_locator(MultipleLocator(tick_interval))
    # # plt.xticks(np.arange(0, NGEN, NGEN/10))
    # plt.legend()

    # plt.savefig(f"cma_es_mo_fpd_{sim_type}ConstraintEvolution.png")
    # plt.close()


    #####################################################
    # objective evolution over time

    # plt.figure(dpi=400)
    # plt.title("Objective Value over time")
    # plt.xlabel("Generation")
    # plt.ylabel("Normalised Hold Time")

    # plt.plot(generation, toolbox.logbook.bookshelf["average objective value"], label="Average Hold Time",
    #          color="red")
    # plt.plot(generation, toolbox.logbook.bookshelf["std objective value"], label="Std Hold Time",
    #          color="blue")
    # if normalised:
    #     # plt.xlim((0, 1.1))
    #     plt.ylim((0, 1.1))
    # else:
    #     plt.ylim((-0.005, 2 * np.max(objective_history)))

    # plt.gca().xaxis.set_major_locator(MultipleLocator(tick_interval))
    # # plt.xticks(np.arange(0, NGEN, NGEN / 10))
    # plt.legend()

    # plt.savefig(f"cma_es_mo_fpd_{sim_type}ObjectiveEvolution.png")
    # plt.close()


    #####################################################
    # objective and constraint evolution in objective space

    # plt.figure(dpi=400)
    # plt.title("Evolution of Average Fitness")
    # plt.xlabel("Normalised Residual of Shock Speed")
    # plt.ylabel("Normalised Hold Time")

    # for i in range(NGEN - 1):
    #     plt.plot(toolbox.logbook.bookshelf["average constraint value"][i:i+2],
    #              toolbox.logbook.bookshelf["average objective value"][i:i+2], color='blue', alpha=(i+1)/NGEN)

    # plt.scatter(toolbox.logbook.bookshelf["average constraint value"][-1],
    #             toolbox.logbook.bookshelf["average objective value"][-1], color='black', s=10)

    # if normalised:
    #     plt.xlim((0, 1.1))
    #     plt.ylim((0, 1.1))
    # else:
    #     plt.ylim((-0.005, 2 * np.max(objective_history)))
    #     plt.xlim((0, 1500))

    # plt.savefig(f"cma_es_mo_fpd_{sim_type}ObjectiveSpaceEvolution.png")
    # plt.close()


    # #####################################################
    # # std of each variable over time

    # plt.figure(dpi=400)
    # plt.title("Evolution of STD of Each Variable")
    # plt.ylabel("Normalised STD")
    # plt.xlabel("Generation")

    # # percent he
    # plt.plot(generation, [element[0] for element in std], color='green', label='percent_he')
    # # driver_p
    # plt.plot(generation, [element[1] for element in std], color='red', label='driver_p')
    # # p4
    # plt.plot(generation, [element[2] for element in std], color='blue', label='p4')
    # # D_throat
    # plt.plot(generation, [element[3] for element in std], color='black', label='D_throat')
    # # reservoir_p
    # plt.plot(generation, [element[4] for element in std], color='orange', label='reservoir_p')

    # plt.legend()

    # plt.savefig(f"cma_es_mo_fpd_{sim_type}StdVariableSpaceEvolution.png")
    # plt.close()


    #####################################################
    # average sigma of each generation

    # plt.figure(dpi=400)
    # plt.title("Evolution of Average Step Size")
    # plt.ylabel("Normalised Average Step Size")
    # plt.xlabel("Generation")

    # plt.plot(generation, [np.average(element) for element in sigma])
    # plt.savefig(f"cma_es_mo_fpd_{sim_type}StdVariableSpaceEvolution.png")
    # plt.close()

    os.chdir(starting_working_directory)

    print('\n\nEND OF SIM')
    print('*'*60)
    print('\n\n')

    return strategy.parents

if __name__ == "__main__":
    sigma_factors = [0.005, 0.005, 0.005]
    # experiment_types = [('Penalty', 12, penalty_factors[4]), ('Penalty', 12, penalty_factors[3]),
    #                     ('Penalty', 12, penalty_factors[2]), ('Penalty', 12, penalty_factors[1]),
    #                     ('Penalty', 12, penalty_factors[0])]
    # experiment_types = [('ParentValue', 12, sigma_factors[0]), ('ParentValue', 12, sigma_factors[1]),
    #                     ('ParentValue', 12, sigma_factors[2]), ('ParentValue', 12, sigma_factors[3]),
    #                     ('ParentValue', 12, sigma_factors[4])]
    experiment_types = [('ParentValue', 24, sigma_factors[0], "hard_bounds_on_p4"), ('ParentValue', 36, sigma_factors[1], "hard_bounds_on_p4"),
                    ('ParentValue', 36, sigma_factors[2], "hard_bounds_on_p4")]

    for experiment_type in experiment_types:
        solutions = main(experiment_type)

        # ClosestValidPenality = ClosestValidPenalty(valid, closest_feasible, experiment_type[2], distance)
        # # try:
        #     solutions = main(experiment_type)
        # except Exception as e:
        #     print(f"An error occurred during the simulation: {e}")
        #     pass

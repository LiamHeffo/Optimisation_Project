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

def base_config_dict():
    config_data = {'mode': 'fully_theoretical', 'output_filename': 'optimisation_test_setup',
                   'facility': 'x2_nrst_85_mm_shock_tube', 'driver_condition': 'custom_from_dict',
                   'test_gas_gas_model': 'CEAGas', 'test_gas_name': 'he-with-ions', 'p1': 150e3}
    return config_data

def base_driver_dict(x):
    driver_dict = {'percent_He': x[0], 'driver_p': x[1], 'p4': x[2], 'D_throat': x[3], 'reservoir_p': x[4]}

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

def constraint_function(x):

    driver_dict = base_driver_dict(x)

    # print(f'driver dict = {driver_dict}')

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

    return np.abs(shock_tube.vs - 4900)


x = (80, 0.077e6, 35.7e6, 0.085, 6.08e6)  # percent_He, driver_p, p4, D_throat, reservoir_p
# x = (99.9, 0.3285e6, 29.2e6, 0.0517, 7.61e6) #Opt-500-x12#1
# x = (99.9, 0.3498e6, 29.87e6, 0.052, 7.95e6) #Opt-500-x12#2
# x = (99.9, 0.3035e6, 28.37e6, 0.0558, 7.36e6) #Opt-500-x12#3
# x = (99.9, 0.3164e6, 28.62e6, 0.0506, 7.12e6) #Opt-500-x12#4
# x = (99.9, 0.3480e6, 29.86e6, 0.0518, 7.92e6) #Opt-500-x36#1
# x = (99.9, 0.3686e6, 28.78e6, 0.0508, 7.85e6) #Opt-500-x36#2
# x = (100, 0.26068e6, 26.4e6, 0.0512, 5.947e6)   #Opt-750-x36#1

print(f"Shock Speed Residual = {constraint_function(x)}")
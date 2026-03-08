# First party modules
from pyoptsparse import Optimization, NSGA2

import spark
import numpy as np
import yaml, os, random, time

from yaml.loader import Reader, Scanner, Parser, Composer, SafeConstructor, Resolver

from pitot3_utils.pitot3_classes import Facility, Driver, Diaphragm, Facility_State, Tube, Nozzle, Test_Section
from pitot3_utils.pitot3_classes import eilmer4_CEAGas_input_file_creator, expansion_tube_test_time_calculator, \
    state_output_for_final_output, pitot3_results_output, cleanup_function
from pitot3 import StrictBoolSafeLoader

from gdtk.gas import GasModel, GasState, GasFlow


def constraint_function(driver_dict):
    ####################################################################################################################
    # Building driver dictionary
    ####################################################################################################################

    config_data = {'mode': 'fully_theoretical', 'output_filename': 'optimisation_test_setup',
                   'facility': 'x2_nrst_85_mm_shock_tube', 'driver_condition': 'custom_from_dict',
                   'test_gas_gas_model': 'CEAGas', 'test_gas_name': 'n2-o2-with-ions', 'p1': 10000.0}

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
    vs1_guess_2 = 3000
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

    shock_tube.calculate_shock_speed_and_related_states()

    ####################################################################################################################
    # returning the constraint function value:
    ####################################################################################################################

    return np.abs(shock_tube.vs - 2500)


def objective_function(driver_condition_dict):
    fill_condition = {
        "p_drvr_0": driver_condition_dict['driver_p'],  # 77.2 kPa
        "T_drvr_0": 298.15,  # K
        "composition_drvr": {'He': float(driver_condition_dict['percent_He'] / 100),
                             'Ar': float(1 - driver_dict['percent_He'] / 100)},
        "composition_units": 'molef',  # default is massf
        "p_rsvr_0": driver_condition_dict['reservoir_p'],
        "T_rsvr_0": 298.15,  # K
        "p_rupture": driver_condition_dict['p4']  # 35.6e6, # 35.6 MPa 2.5mm scored to 0.2mm depth.
    }

    facility = {
        "L_drvr": 4.475,  # m (see docs for which dimension this is)
        "D_piston": 0.2568,  # m (equivalent/equal to D_drvr)
        "V_drvr_0": spark.calculateInitialDriverVolume(4.475, 0.2568, 0.112, 85 / 1000, L_buffer=45 / 1000,
                                                       D_buffer=50 / 1000),
        # <-- see the docs for description of this function
        "m_piston": 10.5,  # kg
        "L_buffer": 50 / 1000,  # 50 mm buffers
        "D_star": driver_condition_dict['D_throat'],  # 65 mm
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
        "max_piston_cycles": 2,  # this halting criteria is ignored if the diaphragm ruptures
        "percent_time_on_buffers_before_halting": 0.05,
        # the sim's other stopping criteria: % of total sim time after which to stop the sim if the piston is on the buffers
        "effective_inflection_velocity_tolerance": 3
        # has to be zero in this type of optimiser, # m/s (a piston will never exactly inflect at 0.0, so what velocity is deemed sufficient?)
    }

    sim = spark.createSimulation(
        condition_dict=fill_condition,
        facility_dict=facility,
        simulation_settings_dict=settings,
        diaphragm_model_dict=rupture_model,
    )

    sim.run()
    diaphragm_rupture_flag = sim.flags.diaphragm_ruptured

    if diaphragm_rupture_flag:
        t_hold = sim.t_hold
        return t_hold
    else:
        return 0

def objfunc(var_dict):
    funcs = {}
    funcs["obj"] = objective_function(var_dict)
    funcs["constr"] = constraint_function(var_dict)
    fail = False

    return funcs, fail



# INITIAL GUESS
driver_dict = {'percent_He' : 80, 'driver_p' : 110.3e3, 'p4' : 15.5e6, 'D_throat' : 0.085, 'reservoir_p':2.8e6}

optProb = Optimization("Optimisation_Test_TP", objfunc)

optProb.addVar('percent_He', varType='c', value=driver_dict['percent_He'], lower=0, upper=100, scale=0.1)
optProb.addVar('driver_p', varType='c', value=driver_dict['driver_p'], lower=0, upper=40e6, scale=1*10**(-5))
optProb.addVar('p4', varType='c', value=driver_dict['p4'], lower=0, upper=40e6, scale=1*10**(-7))
optProb.addVar('reservoir_p', varType='c', value=driver_dict['reservoir_p'], lower=0, upper=40e6, scale=1*10**(-6))
optProb.addVar('D_throat', varType='c', value=driver_dict['D_throat'], lower=0, upper=0.085, scale=1*10**(2))

#optProb.addCon('con', lower=0, upper=None, scale=1, linear=False, wrt=None, jac=None)

optProb.addObj("obj")
optProb.addObj("constr")

print(optProb)

optOptions = {'PopSize':10, 'maxGen':100, 'pCross_real': 0.6, 'pMut_real': 0.2, 'eta_c':10.0, 'eta_m': 20.0, 'seed': 0,
              'xinit':0}
opt = NSGA2(options=optOptions)

sol = opt(optProb, sens='FD')

print(sol)


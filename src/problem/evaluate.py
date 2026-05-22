"""
X2 free-piston driver evaluation functions.

constraint_function(x, bounds)
    Calls PITOT3 to compute |vs - 4900| (shock speed residual).

objective_function(x, bounds)
    Calls SPARK to compute (hold_time, impact_speed).

evaluate(x)
    Combines both, applies normalisation, handles the Penalty sim_type.

Logbook injection
-----------------
objective_function() records run-time counters (failed evaluations, etc.)
into a Logbook bookshelf.  Rather than reading a global, it uses the module-
level _logbook variable.  Call set_logbook(logbook) from main.py after the
Toolbox is constructed.
"""

import contextlib
import os
import sys
import yaml
import numpy as np

import spark


# SPARK's sim.run() prints a "t_hold = ..." line per evaluation, which
# floods the terminal at λ ~ 12 × hundreds of generations.  We silence
# its stdout below.  stderr is left untouched so real errors still
# surface, and the genuine exception-handler print() inside
# objective_function() is routed to stderr explicitly.
_DEVNULL = open(os.devnull, "w")
from gdtk.gas import GasModel, GasState
from pitot3_utils.pitot3_classes import (
    Facility, Driver, Tube,
)
from pitot3 import StrictBoolSafeLoader

from problem.config import (
    base_config_dict, base_driver_dict,
    APPROX_IDEAL, APPROX_NADIR,
    APPROX_IDEAL_2D, APPROX_NADIR_2D,
    BOUNDS,
)
from problem.transforms import variable_untransformation, normalise_fitness

# PITOT3 returns this sentinel value (m/s) when its shock-speed solver
# fails or any of the heuristic pre-checks bail.  Any delta_vs1 ==
# _PITOT3_FAILURE_SENTINEL is treated as a failed evaluation (not a real
# constraint reading); in CHT_AL mode the individual is _feasible=False
# and excluded from AL coefficient adaptation.
_PITOT3_FAILURE_SENTINEL = 3500
from problem.feasibility import evaluate_constraints, is_feasible
from utils import valid
from algorithm.penalty import ClosestValidPenalty

# ─────────────────────────────────────────────────────────────────────────────
# Logbook injection — avoids accessing a global toolbox from inside the
# evaluation functions.  Set this once from main.py after toolbox creation.
# ─────────────────────────────────────────────────────────────────────────────

_logbook = None


def set_logbook(logbook):
    """Register the run logbook so that objective_function can update counters."""
    global _logbook
    _logbook = logbook


# ─────────────────────────────────────────────────────────────────────────────
# Constraint function — PITOT3 shock-speed residual
# ─────────────────────────────────────────────────────────────────────────────

def constraint_function(x1, bounds):
    """Evaluate the shock-speed constraint via PITOT3.

    Parameters
    ----------
    x1 : Individual
        Normalised-space individual; must carry an .ind_number attribute.
    bounds : list of (lo, hi)
        Physical-space bounds (used by variable_untransformation).

    Returns
    -------
    float
        |vs - vs1| in m/s, or 3500 (penalty) on failure.
    """
    ind_number = x1.ind_number
    test_name = f"DEAP_tests_{ind_number}"

    # Anchor the working-directory paths on __file__ rather than on
    # os.getcwd().  Workers chdir into PITOT3 test directories during
    # their evaluations, so cwd is unreliable across calls in the same
    # worker — the previous getcwd-then-string-slice hack only happened
    # to work when the project path was exactly 35 chars long.  Using
    # __file__ resolves the same path every call and matches where
    # parallelization_setup() (utils.py) creates the directories:
    # <src/>/PITOT3_Outputs/DEAP_tests_<i>.
    starting_working_directory = os.getcwd()              # for cwd restore at exit
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(src_dir, "PITOT3_Outputs", test_name)
    os.makedirs(target_dir, exist_ok=True)                # defensive: create if missing
    os.chdir(target_dir)

    x = variable_untransformation(x1, bounds)

    # ── Heuristic pre-checks ──────────────────────────────────────────────
    driver_dict = base_driver_dict(x)

    if driver_dict['driver_p'] > driver_dict['reservoir_p']:
        print("bad guess")
        os.chdir(starting_working_directory)
        return 3500

    if 0.085 < driver_dict['D_throat'] < 0:
        print("bad guess")
        os.chdir(starting_working_directory)
        return 3500

    pressure_ratio = driver_dict["p4"] / driver_dict["driver_p"]
    compression_ratio = pressure_ratio ** (1 / 1.667)
    if not (5 < compression_ratio < 70):
        print("bad guess")
        os.chdir(starting_working_directory)
        return 3500

    if driver_dict['p4'] > bounds[2][1]:
        print("bad guess")
        os.chdir(starting_working_directory)
        return 3500

    # ── Build PITOT3 configuration ────────────────────────────────────────
    config_data = base_config_dict()

    preset_gas_models_folder = '$PITOT3_DATA/preset_gas_models'
    driver_gmodel_location = '{0}/thermally-perfect-{1}-gas-model.lua'.format(
        preset_gas_models_folder, driver_dict['driver_fill_gas_name']
    )
    gmodel = GasModel(os.path.expandvars(driver_gmodel_location))

    # Isentropic compression to find T4
    state4i = GasState(gmodel)
    state4i.p = driver_dict['driver_p']
    state4i.T = 298.15

    molecular_mass_dict = {'Ar': 39.948, 'He': 4.002602}
    total_molecular_mass = sum(
        molecular_mass_dict[sp] * driver_dict['driver_fill_composition'][sp]
        for sp in driver_dict['driver_speciesList']
    )
    state4i.massf = {
        sp: driver_dict['driver_fill_composition'][sp] * (molecular_mass_dict[sp] / total_molecular_mass)
        for sp in driver_dict['driver_speciesList']
    }
    state4i.update_thermo_from_pT()
    state4i.update_sound_speed()
    gamma = state4i.gamma

    driver_dict['T4'] = state4i.T * (driver_dict['p4'] / state4i.p) ** (1.0 - 1.0 / gamma)

    # ── Facility and tube setup ───────────────────────────────────────────
    facility_yaml_filename = '$PITOT3_DATA/facilities/{0}.yaml'.format(config_data['facility'])
    facility_yaml_file = open(os.path.expandvars(facility_yaml_filename))
    facility_input_data = yaml.load(facility_yaml_file, Loader=yaml.FullLoader)
    facility = Facility(facility_input_data)

    config_data['facility_type'] = facility.get_facility_type()
    D_shock_tube = facility.shock_tube_diameter

    outputUnits = 'moles'
    species_molecular_weights_filename = '$PITOT3_DATA/PITOT3_species_molecular_weights.yaml'
    species_MW_dict = yaml.load(
        open(os.path.expandvars(species_molecular_weights_filename)),
        Loader=StrictBoolSafeLoader,
    )

    T_0, p_0 = 298.15, 101325.0

    driver = Driver(
        driver_dict, p_0=p_0, T_0=T_0,
        preset_gas_models_folder=preset_gas_models_folder,
        outputUnits=outputUnits, species_MW_dict=species_MW_dict,
        D_shock_tube=D_shock_tube,
    )

    state3s = driver.get_exit_state()

    shock_tube_length, shock_tube_diameter = facility.get_shock_tube_length_and_diameter()

    shock_tube = Tube(
        tube_name='shock_tube',
        tube_length=shock_tube_length,
        tube_diameter=shock_tube_diameter,
        fill_pressure=float(config_data['p1']),
        fill_temperature=298.15,
        fill_gas_model=config_data['test_gas_gas_model'],
        fill_gas_name=config_data['test_gas_name'],
        fill_gas_filename=None,
        fill_state_name='s1',
        shocked_fill_state_name='s2',
        entrance_state_name=driver.get_exit_state_name(),
        entrance_state=state3s,
        unsteadily_expanded_entrance_state_name='s3',
        expand_to='flow_behind_shock',
        expansion_factor=1.0,
        preset_gas_models_folder=preset_gas_models_folder,
        unsteady_expansion_steps=100,
        vs_guess_1=2000, vs_guess_2=5000,
        vs_limits=[400, 20000], vs_tolerance=1e-4,
        outputUnits=outputUnits, species_MW_dict=species_MW_dict,
    )

    try:
        shock_tube.calculate_shock_speed_and_related_states()
    except Exception as e:
        print(f"{e}")
        print(f"x = {x}")
        os.chdir(starting_working_directory)
        return 3500

    os.chdir(starting_working_directory)
    return np.abs(shock_tube.vs - 4900)


# ─────────────────────────────────────────────────────────────────────────────
# Objective function — SPARK piston dynamics
# ─────────────────────────────────────────────────────────────────────────────

def objective_function(x, bounds):
    """Evaluate hold time and piston impact speed via SPARK.

    Parameters
    ----------
    x : Individual
        Normalised-space individual.
    bounds : list of (lo, hi)
        Physical-space bounds.

    Returns
    -------
    (hold_time, impact_speed) : (float, float)
        Physical values.  Returns (0, 350) for failed/infeasible evaluations.
    """
    x = variable_untransformation(x, bounds)

    driver_condition_dict = {
        'percent_He':    x[0],
        'driver_p':      x[1],
        'p4':            x[2],
        'D_throat':      x[3],
        'reservoir_p':   x[4],
        'buffer_length': x[5],
    }

    def _log_failure():
        if _logbook is not None:
            _logbook.bookshelf["No. individuals that failed objective tests"] += 1
            _logbook.bookshelf["No. individuals that produced no hold time"]  += 1

    # ── Heuristic pre-checks ──────────────────────────────────────────────
    if driver_condition_dict['driver_p'] > driver_condition_dict['reservoir_p']:
        _log_failure()
        return 0, 350

    if 0.0849 < driver_condition_dict['D_throat'] < 0:
        _log_failure()
        return 0, 350

    pressure_ratio = driver_condition_dict["p4"] / driver_condition_dict["driver_p"]
    compression_ratio = pressure_ratio ** (1 / 1.667)
    if not (5 < compression_ratio < 70):
        _log_failure()
        return 0, 350

    if driver_condition_dict['p4'] > bounds[2][1]:
        print('p4 too high')
        _log_failure()
        return 0, 350

    # ── SPARK simulation setup ────────────────────────────────────────────
    fill_condition = {
        "p_drvr_0":          driver_condition_dict['driver_p'],
        "T_drvr_0":          298.15,
        "composition_drvr":  {
            'He': float(driver_condition_dict['percent_He'] / 100),
            'Ar': float(1 - driver_condition_dict['percent_He'] / 100),
        },
        "composition_units": 'molef',
        "p_rsvr_0":          driver_condition_dict['reservoir_p'],
        "T_rsvr_0":          298.15,
        "p_rupture":         driver_condition_dict['p4'],
    }

    facility = {
        "L_drvr":    4.475,
        "D_piston":  0.2568,
        "V_drvr_0":  spark.calculateInitialDriverVolume(
            4.475, 0.2568, 0.112, 85 / 1000,
            L_buffer=float(driver_condition_dict['buffer_length']),
            D_buffer=50 / 1000,
        ),
        "m_piston":  10.5,
        "L_buffer":  float(driver_condition_dict['buffer_length']),
        "D_star":    driver_condition_dict['D_throat'],
        "D_driven":  85 / 1000,
    }

    rupture_model = {
        "model":      "drewry",
        "K":          0.93,
        "rho":        8649,
        "time_model": "linear",
        "tau":        2.0 / 1000,
        "b":          0.06,
    }

    settings = {
        "rsvr_gm":                              "ideal_air",
        "drvr_gm":                              "mixed_he_ar",
        "max_piston_cycles":                    3,
        "percent_time_on_buffers_before_halting": 0.05,
        "effective_inflection_velocity_tolerance": 3,
        "t_hold_sim":                           True,
    }

    sim = spark.createSimulation(
        condition_dict=fill_condition,
        facility_dict=facility,
        simulation_settings_dict=settings,
        diaphragm_model_dict=rupture_model,
    )

    diaphragm_rupture_flag = False
    impact_flag = False
    try:
        with contextlib.redirect_stdout(_DEVNULL):
            sim.run()
        diaphragm_rupture_flag = sim.flags.diaphragm_ruptured
        impact_flag = sim.flags.impact_occurred
    except Exception as e:
        # Route to stderr so real failures aren't swallowed by the
        # stdout redirect above.
        print(f"{e}", file=sys.stderr)
        print(f"x = {x}", file=sys.stderr)

    if diaphragm_rupture_flag and impact_flag:
        t_hold = sim.t_hold
        impact_speed = round(sim.results.vel_buffer_strike_max, 3)
        return t_hold, impact_speed
    else:
        return 0, 350


# ─────────────────────────────────────────────────────────────────────────────
# Combined evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(x):
    """Evaluate constraints and (if feasible) fitness for one individual.

    Returns
    -------
    (fit, g, g_al) : 3-tuple
        fit  : tuple of objective values, or None.
                 - For legacy sim_types: a 3-tuple
                   (delta_vs1, hold_time, impact_speed), normalised.
                 - For 'CHT_AL': a 2-tuple (hold_time, impact_speed),
                   normalised — delta_vs1 is no longer a Pareto objective.
                 None means the individual is infeasible (box+phys violated,
                 or in CHT_AL mode the shock-speed simulation failed).
                 SPARK / PITOT3 were not run, or were run but produced
                 the failure sentinel; either way no fitness is recorded.
        g    : np.ndarray, the box+physical constraint vector (length 18
                 for n=6).  Always returned regardless of feasibility —
                 the CHT consumes g for the covariance update.
        g_al : np.ndarray of length 1, or None.
                 - For 'CHT_AL': np.array([delta_vs1 - al_tol]) — the
                   Augmented-Lagrangian constraint vector (one entry).
                 - For all other sim_types: None.
                 None when fit is None, since AL coefficient adaptation
                 only consumes paired (f, g_al) data from successful
                 evaluations.

    Behaviour change (CHT_AL phase)
    -------------------------------
    The 3-tuple return is uniform across sim_types so callers can always
    write ``fit, g, g_al = evaluate(x)``.  Legacy sim_types receive
    g_al=None and ignore it.
    """
    g = evaluate_constraints(x, x.bounds)

    if x.sim_type == "Penalty":
        # Penalty mode keeps its existing behaviour: always produce a
        # fitness, using the closest-valid penalty for box-violators.
        if valid(x):
            delta_vs = constraint_function(x, x.bounds)
            hold_time, impact_speed = objective_function(x, x.bounds)
            fit = normalise_fitness(
                (delta_vs, hold_time, impact_speed), APPROX_IDEAL, APPROX_NADIR,
            )
        else:
            fit = ClosestValidPenalty.wrapper(x)
        return fit, g, None

    if x.sim_type == "CHT_AL":
        # AL path: delta_vs1 becomes a constraint; objectives are 2-D.
        # Box+phys infeasible => no PITOT3 / SPARK, no AL data.
        if not is_feasible(g):
            return None, g, None

        # Heavy-evaluator failures are encoded via the existing sentinel
        # returns rather than rejected as infeasible:
        #   - PITOT3 failure  ⇒ delta_vs1 = 3500 m/s
        #                       ⇒ g_al = 3500 - al_tol ≈ +3400  (a large
        #                         positive, which the AL will penalise as
        #                         a major constraint violation)
        #   - SPARK  failure  ⇒ (hold_time, impact_speed) = (0, 350)
        #                       ⇒ fit_2d normalises to (1, 1)  (worst
        #                         possible values on both axes; Pareto-
        #                         dominated by every successful candidate)
        # Letting these flow through the AL machinery is more robust than
        # rejecting them outright, since random initial points often hit
        # numerical-failure regions before the search converges to the
        # well-behaved part of the design space.
        delta_vs = constraint_function(x, x.bounds)
        hold_time, impact_speed = objective_function(x, x.bounds)
        fit_2d = normalise_fitness(
            (hold_time, impact_speed), APPROX_IDEAL_2D, APPROX_NADIR_2D,
        )
        # AL constraint vector: g_AL_k(x) = delta_vs1(x) - al_tol  (≤ 0).
        # We use the un-normalised delta_vs1 here so the AL coefficients
        # adapt on the natural scale of the constraint; pycma's
        # set_coefficients() will derive its own scaling from iqr(F)/iqr(G).
        al_tol = getattr(x, "al_tol", 100.0)
        g_al = np.array([delta_vs - al_tol], dtype=float)
        return fit_2d, g, g_al

    # Non-Penalty, non-AL path (legacy 3-objective sim_types).
    if not is_feasible(g):
        return None, g, None

    delta_vs = constraint_function(x, x.bounds)
    hold_time, impact_speed = objective_function(x, x.bounds)
    fit = normalise_fitness(
        (delta_vs, hold_time, impact_speed), APPROX_IDEAL, APPROX_NADIR,
    )
    return fit, g, None

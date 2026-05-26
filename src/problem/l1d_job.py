"""
L1d4 job-script templating, subprocess driver, and output parsing for the
X2 free-piston driver optimisation.

Three responsibilities:

1. ``write_job_script(...)`` — materialise an L1d4 input script for one
   individual.  The script is syntactically a valid Python script that
   ``l1d4-prep`` ingests via ``exec()`` with the L1d helper objects
   (``add_break_point``, ``Piston``, ``GasSlug`` …) bound into the
   globals.

2. ``parse_l1d_outputs(...)`` — read the simulation outputs and return
   ``(t_hold, impact_speed, vs1, ok)``.  Failure to extract any of the
   three returns the existing SPARK/PITOT3 sentinel values so the
   downstream CHT_AL machinery sees the same encoding it does today.

3. ``run_l1d(...)`` — the top-level wrapper called by ``evaluate.py``.
   Owns the per-individual working directory, the three L1d subprocess
   invocations, the timeout, and the cleanup.

Per-individual filesystem layout::

    L1d_Outputs/DEAP_<i>/
    ├── DEAP_<i>.py                  ← the templated job script
    ├── ideal_air.lua                ← symlinked gas models
    ├── mixed_he_ar.lua
    ├── cea-lut-air.lua
    ├── piston-0000-history.data     ← produced by --piston-history
    └── DEAP_<i>/                    ← L1d's job-name subdirectory
        ├── diaphragm-0000.data
        ├── history-loc-0000.data    ← primary-diaphragm station
        ├── history-loc-0001.data    ← shock-tube transducer 1
        ├── history-loc-0002.data    ← shock-tube transducer 2
        ├── piston-0000.data
        └── slug-*.data
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from problem.l1d_geometry import build_break_points


# ─────────────────────────────────────────────────────────────────────────────
# Module-level configuration constants (edit these to retune the bring-up)
# ─────────────────────────────────────────────────────────────────────────────

# Failure sentinels — must match the values the rest of the pipeline
# (evaluate.py, main.py _detect_sentinels) expects.
SENTINEL_OBJECTIVE = (0.0, 350.0)
SENTINEL_VS_DELTA  = 3500.0
SENTINEL_TUPLE     = (*SENTINEL_OBJECTIVE, SENTINEL_VS_DELTA, False)
VS1_TARGET         = 4900.0

# Gas-model file names (these match the user's existing prep-gas outputs
# in the project root).  cea-lut-air.lua is not yet generated; absence
# raises a clear error rather than silently failing inside l1d4-prep.
GAS_MODEL_FILES = ("ideal_air.lua", "mixed_he_ar.lua", "cea-lut-air.lua")

# Per-evaluation budget.  L1d's t_finish is set to 28 ms simulated time;
# wall-clock varies with mesh.  The 15-min wall budget mirrors the SPARK
# path; widen if mesh_scale_factor is raised.
EVAL_TIMEOUT_S       = 15 * 60
PREP_TIMEOUT_S       = 2 * 60
POSTPROCESS_TIMEOUT_S = 60

# Mesh & wall-resolution controls.
MESH_SCALE_FACTOR    = 5      # per-slug ncells multiplier
TUBE_N               = 4000   # tube-wall mesh resolution

# Time-stepping constants.
T_FINISH             = 100.0e-3
T_SWITCH             = 23.0e-3

# Provisional X2-default transducer x-positions (relative to PD at x=0).
TRANSDUCER_XS = (4.231, 4.746)

# Keep failed job directories on disk for post-hoc debugging.
KEEP_FAILED_JOBS     = False

# Keep every job directory (successful or failed) on disk.  Useful during
# bring-up when you want to inspect the L1d outputs after each call.
# Must be False before kicking off parallel CMA-ES runs — otherwise
# L1d_Outputs/ grows unbounded (λ workers × hundreds of generations).
KEEP_ALL_JOBS        = True

# Stream the stdout/stderr of every L1d subprocess directly to the parent
# terminal.  Useful during bring-up so you can watch the run progress.
# Flip to False before kicking off parallel CMA-ES runs — otherwise the
# λ workers will interleave their output and the terminal becomes
# unreadable, and the captured-stderr error path below won't have the
# diagnostic text to print on subprocess failure.
STREAM_L1D_OUTPUT    = True


# ─────────────────────────────────────────────────────────────────────────────
# Job-script template
# ─────────────────────────────────────────────────────────────────────────────
# The L1d4-prep tool exec()'s this file with the L1d helper API bound
# into globals.  Everything inside {{double-braces}} is a literal brace
# that needs to survive .format(); everything inside {single-braces} is a
# placeholder that write_job_script() fills in.

JOB_SCRIPT_TEMPLATE = '''\
# L1d4 job script — auto-generated for individual {ind_number}.
# Do not edit by hand; regenerated per evaluation.
import math

config.title = "X2 driver opt: individual {ind_number}"

# ─── Gas models ──────────────────────────────────────────────────────────
gm_ideal_air = add_gas_model("ideal_air.lua")
gm_he_ar     = add_gas_model("mixed_he_ar.lua")
gm_cea_air   = add_gas_model("cea-lut-air.lua")

# Per-individual He/Ar mass fractions derived from percent_He.
massf_he_ar = config.gmodels[gm_he_ar].molef2massf(
    {{"He": {percent_He:.6f} / 100.0, "Ar": 1.0 - {percent_He:.6f} / 100.0}}
)

# ─── Tube geometry (volume-conserving break-points from Hodson 2025) ─────
{break_point_calls}
tube.n = {tube_n}

# ─── Gas slugs and gas-path objects ─────────────────────────────────────
T_amb = {T_amb:.4f}

left_wall = VelocityEnd(x0={reservoir_start_x:.4f}, vel=0.0)

res_gas = GasSlug(
    gmodel_id=gm_ideal_air,
    p={reservoir_p:.6e}, T=T_amb, vel=0.0,
    ncells={n_res}, viscous_effects=1, hcells=1,
    label="reservoir",
)

piston = Piston(
    mass=10.524, diam={D_compression:.4f},
    xL0={piston_xL0:.4f}, xR0={piston_xR0:.4f}, vel0=0.0,
    front_seal_f=0.2,
    front_seal_area=0.020 * {D_compression:.4f} * math.pi,
    x_buffer={x_buffer:.6f}, on_buffer=0,
    label="lightweight piston",
)

driver_gas = GasSlug(
    gmodel_id=gm_he_ar,
    p={driver_p:.6e}, T=T_amb, vel=0.0, massf=massf_he_ar,
    ncells={n_drv}, cluster_strength=1.05, to_end_R=True,
    viscous_effects=1, hcells=1,
    label="driver gas",
)

primary_diaphragm = Diaphragm(x0={pd_x:.4f}, p_burst={p4:.6e}, state=0)

test_gas = GasSlug(
    gmodel_id=gm_cea_air,
    p={test_gas_p1:.6e}, T=T_amb, vel=0.0,
    ncells={n_test}, cluster_strength=1.05, to_end_L=True,
    viscous_effects=0, hcells=1,
    label="test gas",
)

right_free = FreeEnd(x0={shock_tube_end_x:.4f})

assemble_gas_path(left_wall, res_gas, piston, driver_gas,
                  primary_diaphragm, test_gas, right_free)

# ─── Loss regions ───────────────────────────────────────────────────────
add_loss_region({launcher_loss_x0:.4f}, {launcher_loss_x1:.4f}, 0.1)
add_loss_region({diaphragm_loss_x0:.6f}, {pd_x:.4f}, 0.7)

# ─── History locations (indices wired in parse_l1d_outputs) ──────────────
add_history_loc({pd_x:.4f})                  # idx 0 — primary-diaphragm pressure
add_history_loc({transducer_x_1:.4f})        # idx 1 — vs1 ToF station 1
add_history_loc({transducer_x_2:.4f})        # idx 2 — vs1 ToF station 2

# ─── Time stepping ──────────────────────────────────────────────────────
config.dt_init   = 1.0e-10
config.max_time  = {t_finish:.4e}
config.max_step  = 25_000_000
add_cfl_value(0.0, 0.25)
add_dt_plot(0.0,         2.0e-4, 2.0e-4)
add_dt_plot({t_switch:.4e}, 5.0e-5, 5.0e-6)
'''


# ─────────────────────────────────────────────────────────────────────────────
# Writing the job script
# ─────────────────────────────────────────────────────────────────────────────

def write_job_script(out_path, params, ind_number):
    """Materialise the L1d job script for one individual.

    Returns the ``derived`` dict from ``build_break_points`` so the
    parser knows where the primary-diaphragm and transducer stations sit.
    """
    bps, derived = build_break_points(
        buffer_length=params["buffer_length"],
        D_throat=params["D_throat"],
    )
    bp_lines = "\n".join(
        f"add_break_point({x:.6f}, {d:.6f})" for x, d in bps
    )

    # Launcher loss region spans the constant-D=0.1561 launcher tube.
    # End of T3 ramp (small-D side) → start of launcher exit ramp.
    launcher_loss_x0 = next(x for x, d in bps if abs(d - 0.1561) < 1e-4)
    launcher_loss_x1 = next(x for x, d in reversed(bps) if abs(d - 0.1561) < 1e-4)

    script = JOB_SCRIPT_TEMPLATE.format(
        ind_number=ind_number,
        break_point_calls=bp_lines,
        tube_n=TUBE_N,
        T_amb=298.15,
        reservoir_start_x=-8.7188,
        D_compression=0.2568,
        piston_xL0=derived["piston_xL0"],
        piston_xR0=derived["piston_xR0"],
        x_buffer=derived["x_inner_buffer"],
        pd_x=derived["pd_x"],
        shock_tube_end_x=5.0,
        launcher_loss_x0=launcher_loss_x0,
        launcher_loss_x1=launcher_loss_x1,
        diaphragm_loss_x0=derived["x_outer_buffer"],
        transducer_x_1=params["transducer_xs"][0],
        transducer_x_2=params["transducer_xs"][1],
        t_finish=T_FINISH,
        t_switch=T_SWITCH,
        n_res=39 * params["mesh_scale"],
        n_drv=46 * params["mesh_scale"],
        n_test=34 * params["mesh_scale"],
        percent_He=params["percent_He"],
        driver_p=params["driver_p"],
        reservoir_p=params["reservoir_p"],
        p4=params["p4"],
        test_gas_p1=params["test_gas_p1"],
    )

    Path(out_path).write_text(script)
    return derived


# ─────────────────────────────────────────────────────────────────────────────
# Parsing L1d outputs
# ─────────────────────────────────────────────────────────────────────────────

def _load_history_loc(job_dir, idx):
    """Load one history-loc-NNNN.data file, returning (t, p) columns.

    Verified column layout (L1d 4.0): ``1:t  2:vel  3:L_bar  4:rho  5:p
    6:T  ...``.  We only use ``t`` (index 0) and ``p`` (index 4).
    """
    path = os.path.join(job_dir, f"history-loc-{idx:04d}.data")
    arr = np.loadtxt(path, comments="#")
    return arr[:, 0], arr[:, 4]


def _load_times_map(job_dir):
    """Load ``times.data`` as a ``tindx → time`` dict.

    L1d writes ``diaphragm-NNNN.data`` and ``piston-NNNN.data`` without
    embedded timestamps; the times live in a separate ``times.data`` file
    and are joined by tindx.  We need this join to recover t_burst from
    the diaphragm state-flip.
    """
    arr = np.loadtxt(os.path.join(job_dir, "times.data"), comments="#")
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return {int(row[0]): float(row[1]) for row in arr}


def parse_l1d_outputs(job_dir, p_burst, transducer_xs, p_band=0.10):
    """Extract ``(t_hold, impact_speed, vs1, ok)`` from L1d outputs.

    Parameters
    ----------
    job_dir : str
        Path to L1d's <job_name>/ subdirectory (NOT the per-individual
        L1d_Outputs/DEAP_<i>/ parent).  This is where the diaphragm,
        slug, piston, and history-loc files live.
    p_burst : float
        Rupture pressure (Pa).  Defines the hold-time band centre.
    transducer_xs : (float, float)
        x-positions of the two shock-tube transducer stations, used as
        the path length for the vs1 time-of-flight calculation.
    p_band : float
        Fractional half-width of the hold-time band (default ±10%).

    Returns
    -------
    (t_hold, impact_speed, vs_delta, ok) : tuple
        On success ``ok=True`` and the three numbers are real measurements.
        On any failure the SPARK/PITOT3 sentinel tuple is returned, so
        downstream code (main._detect_sentinels) sees the same encoding.
    """
    parent = os.path.dirname(job_dir)
    piston_file = os.path.join(parent, "piston-0000-history.data")

    # 1. Impact speed: first row in piston-history where on_buffer == 1.
    try:
        piston = np.loadtxt(piston_file, comments="#")
    except OSError:
        return SENTINEL_TUPLE
    if piston.ndim == 1:
        piston = piston[np.newaxis, :]
    impact_rows = np.where(piston[:, 6] == 1)[0]
    if impact_rows.size == 0:
        return SENTINEL_TUPLE
    impact_speed = abs(float(piston[impact_rows[0], 3]))

    # 2. t_burst from diaphragm state-flip.  L1d writes diaphragm state
    #    as just (tindx, state); the time is in a separate times.data
    #    file and must be joined by tindx.
    diaphragm_file = os.path.join(job_dir, "diaphragm-0000.data")
    try:
        diaphragm = np.loadtxt(diaphragm_file, comments="#")
        times_map = _load_times_map(job_dir)
    except OSError:
        return SENTINEL_TUPLE
    if diaphragm.ndim == 1:
        diaphragm = diaphragm[np.newaxis, :]
    burst_rows = np.where(diaphragm[:, 1] == 1)[0]
    if burst_rows.size == 0:
        return SENTINEL_TUPLE
    burst_tindx = int(diaphragm[burst_rows[0], 0])
    if burst_tindx not in times_map:
        return SENTINEL_TUPLE
    t_burst = times_map[burst_tindx]

    # 3. Hold time: pressure trace at PD station (history-loc-0000),
    #    integrating dt over samples where p ∈ [(1-p_band), (1+p_band)] · p_burst.
    try:
        t_pd, p_pd = _load_history_loc(job_dir, idx=0)
    except OSError:
        return SENTINEL_TUPLE
    in_band = (p_pd >= (1.0 - p_band) * p_burst) & (p_pd <= (1.0 + p_band) * p_burst)
    in_band[t_pd < t_burst] = False
    if not in_band.any():
        t_hold = 0.0
    else:
        dt = np.diff(t_pd)
        t_hold = float(np.sum(dt[in_band[:-1]]))

    # 4. vs1: shock arrival times at the two transducer stations.  Use
    #    a "first-crossing of 2× quiescent fill pressure" detector — the
    #    quiescent region is the first 50 samples (well before burst).
    arrivals = []
    for hist_idx in (1, 2):
        try:
            t_s, p_s = _load_history_loc(job_dir, idx=hist_idx)
        except OSError:
            return SENTINEL_TUPLE
        p_quiescent = max(float(p_s[:50].mean()), 1.0)
        crossings = np.where(p_s > 2.0 * p_quiescent)[0]
        if crossings.size == 0:
            return SENTINEL_TUPLE
        arrivals.append(float(t_s[crossings[0]]))

    dt_arrival = abs(arrivals[1] - arrivals[0])
    if dt_arrival <= 0.0:
        return SENTINEL_TUPLE
    vs1 = abs(transducer_xs[1] - transducer_xs[0]) / dt_arrival

    return t_hold, impact_speed, abs(vs1 - VS1_TARGET), True


# ─────────────────────────────────────────────────────────────────────────────
# Per-individual subprocess driver
# ─────────────────────────────────────────────────────────────────────────────

def _stage_gas_models(job_root, source_dir):
    """Symlink the three .lua gas-model files into the per-individual
    job directory so l1d4-prep finds them in cwd.

    Raises FileNotFoundError with a clear message if any are missing —
    a typo in the file name will otherwise surface as a confusing
    l1d4-prep traceback.
    """
    for fname in GAS_MODEL_FILES:
        src = Path(source_dir) / fname
        dst = Path(job_root) / fname
        if not src.is_file():
            raise FileNotFoundError(
                f"Gas-model file '{fname}' not found at {src}. "
                f"Run prep-gas to generate it, or update GAS_MODEL_FILES."
            )
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())


def run_l1d(x_phys, ind_number, *, test_gas_p1, transducer_xs=TRANSDUCER_XS,
            mesh_scale=MESH_SCALE_FACTOR, gas_model_dir=None,
            output_root=None):
    """Run one L1d evaluation end to end.

    Parameters
    ----------
    x_phys : sequence of 6 floats
        Physical-space design vector ``[percent_He, driver_p, p4,
        D_throat, reservoir_p, buffer_length]``.
    ind_number : int
        Population index, used to name the per-individual work directory.
    test_gas_p1 : float
        Shock-tube fill pressure (Pa).  Passed in by the caller so it
        stays single-source (read from problem.config in evaluate.py).
    transducer_xs : (float, float)
        Shock-tube transducer x-positions (m).
    mesh_scale : int
        Per-slug ncells multiplier.
    gas_model_dir : str or None
        Directory holding the .lua gas-model files.  Defaults to the
        project root.
    output_root : str or None
        Parent directory under which L1d_Outputs/DEAP_<i>/ is created.
        Defaults to the project src/ directory.

    Returns
    -------
    (t_hold, impact_speed, vs_delta, ok) : tuple
        On success ok=True; on any failure mode the SPARK/PITOT3 sentinel
        tuple is returned so the downstream pipeline sees the existing
        encoding.
    """
    percent_He, driver_p, p4, D_throat, reservoir_p, buffer_length = x_phys
    params = {
        "percent_He":    percent_He,
        "driver_p":      driver_p,
        "p4":            p4,
        "D_throat":      D_throat,
        "reservoir_p":   reservoir_p,
        "buffer_length": buffer_length,
        "test_gas_p1":   test_gas_p1,
        "mesh_scale":    mesh_scale,
        "transducer_xs": tuple(transducer_xs),
    }

    if gas_model_dir is None:
        gas_model_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    if output_root is None:
        output_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

    job_name = f"DEAP_{ind_number}"
    job_root = os.path.join(output_root, "L1d_Outputs", job_name)
    job_inner = os.path.join(job_root, job_name)
    os.makedirs(job_root, exist_ok=True)

    success = False
    try:
        script_path = os.path.join(job_root, f"{job_name}.py")
        try:
            write_job_script(script_path, params, ind_number)
        except ValueError as e:
            # build_break_points raises on infeasible orifice geometry.
            print(f"L1d geometry infeasible for ind {ind_number}: {e}",
                  file=sys.stderr)
            return SENTINEL_TUPLE

        _stage_gas_models(job_root, gas_model_dir)

        # When STREAM_L1D_OUTPUT is True the subprocesses inherit the
        # parent's stdout/stderr so progress is visible live in the
        # terminal; when False, both streams are captured so the error
        # path below can print the last 2000 chars of stderr on failure.
        capture = not STREAM_L1D_OUTPUT
        cwd = os.getcwd()
        os.chdir(job_root)
        try:
            subprocess.run(
                ["l1d4-prep", f"--job={job_name}"],
                check=True, timeout=PREP_TIMEOUT_S, capture_output=capture,
            )
            subprocess.run(
                ["l1d4", "--run-simulation", f"--job={job_name}"],
                check=True, timeout=EVAL_TIMEOUT_S, capture_output=capture,
            )
            subprocess.run(
                ["l1d4", "--piston-history",
                 f"--job={job_name}", "--pindx=0"],
                check=True, timeout=POSTPROCESS_TIMEOUT_S,
                capture_output=capture,
            )
        finally:
            os.chdir(cwd)

        result = parse_l1d_outputs(
            job_dir=job_inner,
            p_burst=p4,
            transducer_xs=transducer_xs,
        )
        success = result[3]
        return result

    except subprocess.CalledProcessError as e:
        print(f"L1d subprocess failed for ind {ind_number}: "
              f"cmd={e.cmd} rc={e.returncode}", file=sys.stderr)
        # When STREAM_L1D_OUTPUT=True the subprocess inherits stderr, so
        # e.stderr is None and the user has already seen the error text
        # in the terminal.  Only re-emit captured stderr when we have it.
        if e.stderr:
            print(e.stderr.decode(errors="replace")[-2000:], file=sys.stderr)
        return SENTINEL_TUPLE
    except subprocess.TimeoutExpired as e:
        print(f"L1d timeout for ind {ind_number}: cmd={e.cmd}",
              file=sys.stderr)
        return SENTINEL_TUPLE
    except Exception as e:
        print(f"L1d unexpected failure for ind {ind_number}: {e}",
              file=sys.stderr)
        return SENTINEL_TUPLE
    finally:
        keep = KEEP_ALL_JOBS or ((not success) and KEEP_FAILED_JOBS)
        if not keep:
            shutil.rmtree(job_root, ignore_errors=True)

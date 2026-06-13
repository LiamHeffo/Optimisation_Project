"""
Standalone L1d4 integration test with SPARK comparison.

Runs a single L1d simulation using exactly the same code path that the
CMA-ES uses (evaluate._evaluate_l1d → run_l1d → parse_l1d_outputs), then
runs a SPARK simulation of the same design and overlays results for
direct comparison.

The design vector is the mid-point of the physical BOUNDS — a neutral,
physically reasonable starting point with no prior knowledge required.

Run from the project root on the l1d_cht_al branch:

    cd /home/x-lab-user/Optimisation_Project
    git checkout l1d_cht_al
    python tests/test_l1d_single_run.py

Prerequisites:
  - The three .lua gas model files must be present in the project root.
  - SPARK must be importable (it lives in /home/x-lab-user/spark/src).

Outputs (written to src/L1d_Outputs/DEAP_0/):
  - pressure_histories.png   : L1d pressure at all three history locations
  - piston_kinematics.png    : L1d piston position and velocity
  - spark_vs_l1d_driver.png  : SPARK driver pressure overlaid on L1d PD trace
"""
import contextlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from problem.config import BOUNDS, base_config_dict
from problem.l1d_job import run_l1d, TRANSDUCER_XS, SENTINEL_TUPLE


# ── Design vector ─────────────────────────────────────────────────────────────
# Mid-point of each physical bound.  Guaranteed inside the valid range.

# x_phys = np.array([0.5 * (lo + hi) for lo, hi in BOUNDS])

x_phys = np.array([
    80.0,       # percent_He     [%]
    0.0103e6,      # driver_p       [Pa]
    3.5e6,     # p4 (burst)     [Pa]
    0.065,      # D_throat       [m]
    7.94e6,      # reservoir_p    [Pa]
    0.05,       # buffer_length  [m]
])

VAR_LABELS = [
    "percent_He   [%]  ",
    "driver_p     [Pa] ",
    "p4           [Pa] ",
    "D_throat     [m]  ",
    "reservoir_p  [Pa] ",
    "buffer_length[m]  ",
]

print("=" * 60)
print("L1d single-run test — design vector (physical mid-bounds)")
print("=" * 60)
for label, val in zip(VAR_LABELS, x_phys):
    print(f"  {label}: {val:.4g}")
print()

percent_He    = x_phys[0]
driver_p      = x_phys[1]
p4            = x_phys[2]   # burst pressure
D_throat      = x_phys[3]
reservoir_p   = x_phys[4]
buffer_length = x_phys[5]

test_gas_p1  = float(base_config_dict()['p1'])
output_root  = Path("src/L1d_Outputs/DEAP_0")
job_subdir   = output_root / "DEAP_0"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  L1d simulation
# ─────────────────────────────────────────────────────────────────────────────

print(f"test_gas_p1  = {test_gas_p1:.4g} Pa")
print(f"transducers at x = {TRANSDUCER_XS} m (relative to PD)")
print()
print("── L1d simulation ──────────────────────────────────────")

t_hold, impact_speed, delta_vs, ok = run_l1d(
    x_phys=x_phys,
    ind_number=0,
    test_gas_p1=test_gas_p1,
    transducer_xs=TRANSDUCER_XS,
)

print()
sentinel_t, sentinel_imp, sentinel_dvs, _ = SENTINEL_TUPLE

if not ok:
    print("L1d RESULT: FAILED (run_l1d returned ok=False)")
    print(f"  t_hold       = {t_hold}  (sentinel = {sentinel_t})")
    print(f"  impact_speed = {impact_speed}  (sentinel = {sentinel_imp})")
    print(f"  delta_vs     = {delta_vs}  (sentinel = {sentinel_dvs})")
else:
    print("L1d RESULT: OK")
    print(f"  t_hold       = {t_hold * 1e3:.3f} ms")
    print(f"  impact_speed = {impact_speed:.1f} m/s")
    print(f"  delta_vs     = {delta_vs:.1f} m/s  (|vs1 - 4900|; target = 0)")
    if t_hold <= 0:
        print("  WARNING: t_hold = 0 — diaphragm may not have burst or window never entered")
    if impact_speed >= sentinel_imp:
        print("  WARNING: impact_speed at or above sentinel value")
    if delta_vs >= sentinel_dvs:
        print("  WARNING: delta_vs at or above sentinel value — vs1 parse may have failed")

print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SPARK simulation
# ─────────────────────────────────────────────────────────────────────────────
# SPARK is a 0D ODE model of the driver: it tracks piston position/velocity
# and driver pressure as scalars (no spatial resolution).  p_drvr from SPARK
# is therefore the driver pressure everywhere in the driver — directly
# comparable to history-loc-0000 (the primary-diaphragm station) in L1d.

print("── SPARK simulation ─────────────────────────────────────")

spark_ok = False
spark_sim = None

try:
    spark_src = "/home/x-lab-user/spark/src"
    if spark_src not in sys.path:
        sys.path.insert(0, spark_src)
    import spark

    fill_condition = {
        "p_drvr_0":         driver_p,
        "T_drvr_0":         298.15,
        "composition_drvr": {
            "He": float(percent_He / 100.0),
            "Ar": float(1.0 - percent_He / 100.0),
        },
        "composition_units": "molef",
        "p_rsvr_0":         reservoir_p,
        "T_rsvr_0":         298.15,
        "p_rupture":        p4,
    }

    facility = {
        "L_drvr":   4.475,
        "D_piston": 0.2568,
        "V_drvr_0": spark.calculateInitialDriverVolume(
            4.475, 0.2568, 0.112, 85 / 1000,
            L_buffer=float(buffer_length),
            D_buffer=50 / 1000,
        ),
        "m_piston": 10.5,
        "L_buffer": float(buffer_length),
        "D_star":   D_throat,
        "D_driven": 85 / 1000,
    }

    rupture_model = {
        "model":      "drewry",
        "K":          0.93,
        "rho":        8649,
        "time_model": "linear",
        "tau":        2.0e-3,
        "b":          0.06,
    }

    settings = {
        "rsvr_gm":                                "ideal_air",
        "drvr_gm":                                "mixed_he_ar",
        "max_piston_cycles":                      3,
        "percent_time_on_buffers_before_halting": 0.05,
        "effective_inflection_velocity_tolerance": 3,
        "t_hold_sim":                             True,
    }

    spark_sim = spark.createSimulation(
        condition_dict=fill_condition,
        facility_dict=facility,
        simulation_settings_dict=settings,
        diaphragm_model_dict=rupture_model,
    )

    _devnull = open(os.devnull, "w")
    with contextlib.redirect_stdout(_devnull):
        spark_sim.run()
    _devnull.close()

    spark_ok = spark_sim.flags.diaphragm_ruptured and spark_sim.flags.impact_occurred

    if spark_ok:
        print("SPARK RESULT: OK")
        print(f"  t_hold       = {spark_sim.t_hold * 1e3:.3f} ms")
        print(f"  impact_speed = {spark_sim.results.vel_buffer_strike_max:.1f} m/s")
    else:
        print("SPARK RESULT: incomplete — flags:")
        print(f"  diaphragm_ruptured = {spark_sim.flags.diaphragm_ruptured}")
        print(f"  impact_occurred    = {spark_sim.flags.impact_occurred}")
        if spark_sim.results is not None and hasattr(spark_sim.results, 'p_drvr'):
            p_max_MPa = max(spark_sim.results.p_drvr) * 1e-6
            print(f"  max p_drvr reached = {p_max_MPa:.2f} MPa  (p_burst = {p4*1e-6:.2f} MPa)")
        print("  (driver trace will still be overlaid in Figure 3 for diagnostics)")

except Exception as exc:
    print(f"SPARK RESULT: FAILED — {exc}")

print()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Plots
# ─────────────────────────────────────────────────────────────────────────────

output_root.mkdir(parents=True, exist_ok=True)

HISTORY_CONFIGS = [
    (0, f"x = {TRANSDUCER_XS[0]:.3f} m  (ToF station 1)", "ToF transducer 1"),
    (1, f"x = {TRANSDUCER_XS[1]:.3f} m  (ToF station 2)", "ToF transducer 2"),
]


def _load_history(idx):
    """Return (t_ms, p_MPa) for history-loc-000N, or (None, None) if missing."""
    f = job_subdir / f"history-loc-{idx:04d}.data"
    if not f.exists():
        return None, None
    d = np.loadtxt(f, comments="#")
    return d[:, 0] * 1e3, d[:, 4] * 1e-6   # s→ms, Pa→MPa


# ── Diaphragm burst time ──────────────────────────────────────────────────────
# Authoritative source: diaphragm-0000.data, which records (tindx, state) at
# every dt_plot step.  state=0 means intact; state=2 means fully ruptured.
# We find the first tindx where state==2, then look up the simulation time via
# the piston history file — the only dt_plot-sampled file that stores both
# tindx and t in the same row.
#
# The actual rupture instant falls somewhere in the window [t_prev, t_burst].
# dt_tindx_ms is the width of that window (the dt_plot resolution at this
# step), and is reported alongside t_burst_ms as the temporal uncertainty.
t_burst_ms  = None
dt_tindx_ms = None   # dt_plot window width at the burst step

_diaph_file  = job_subdir / "diaphragm-0000.data"
_piston_path = output_root / "piston-0000-history.data"

if _diaph_file.exists() and _piston_path.exists():
    _ddata      = np.loadtxt(_diaph_file, comments="#", dtype=int)  # (tindx, state)
    _pdata_b    = np.loadtxt(_piston_path, comments="#")            # (tindx, t, ...)
    _tindx_col  = _pdata_b[:, 0].astype(int)
    _t_col      = _pdata_b[:, 1]                                    # seconds

    _open_rows = np.where(_ddata[:, 1] == 2)[0]
    if _open_rows.size > 0:
        _burst_tindx = int(_ddata[_open_rows[0], 0])
        _bi   = np.where(_tindx_col == _burst_tindx)[0]
        _bi_p = np.where(_tindx_col == _burst_tindx - 1)[0]
        if _bi.size > 0 and _bi_p.size > 0:
            t_burst_ms  = float(_t_col[_bi[0]])  * 1e3
            dt_tindx_ms = float(_t_col[_bi[0]] - _t_col[_bi_p[0]]) * 1e3
            print(f"Diaphragm burst: tindx={_burst_tindx}, "
                  f"t = {t_burst_ms:.3f} ms  ±{dt_tindx_ms:.4f} ms (dt_plot resolution)")
        else:
            print(f"Diaphragm burst: tindx={_burst_tindx} found in diaphragm file "
                  f"but not matched in piston history")
    else:
        print("Diaphragm burst: state never reached 2 — diaphragm did not rupture")
else:
    _missing = [n for n, f in [("diaphragm-0000.data", _diaph_file),
                                ("piston-0000-history.data", _piston_path)]
                if not f.exists()]
    print(f"Diaphragm burst: cannot determine — missing: {', '.join(_missing)}")
print()

# Reusable legend label (built once, used in every figure).
# The ± term is the dt_plot resolution window — the actual rupture instant
# falls somewhere in [t_burst − dt_tindx, t_burst].
if t_burst_ms is not None and dt_tindx_ms is not None:
    _burst_label = (f"diaphragm burst  "
                    f"t = {t_burst_ms:.3f} ±{dt_tindx_ms:.4f} ms")
elif t_burst_ms is not None:
    _burst_label = f"diaphragm burst  t = {t_burst_ms:.3f} ms"
else:
    _burst_label = None


# ── Figure 1: L1d pressure histories ─────────────────────────────────────────

fig1, axes1 = plt.subplots(len(HISTORY_CONFIGS), 1, figsize=(10, 3 * len(HISTORY_CONFIGS)), sharex=False)
if len(HISTORY_CONFIGS) == 1:
    axes1 = [axes1]
fig1.suptitle("L1d pressure histories — DEAP_0 (mid-bounds design)", fontsize=12)

for ax, (idx, x_label, description) in zip(axes1, HISTORY_CONFIGS):
    t_ms, p_MPa = _load_history(idx)
    ax.set_title(f"loc {idx} — {description}  [{x_label}]")
    ax.set_ylabel("Pressure (MPa)")
    ax.set_xlabel("Time (ms)")
    ax.grid(True, lw=0.4, alpha=0.5)

    if t_ms is None:
        ax.text(0.5, 0.5, f"history-loc-{idx:04d}.data not found",
                ha="center", va="center", transform=ax.transAxes, color="red")
        continue

    ax.plot(t_ms, p_MPa, lw=0.8, color="steelblue")
    if t_burst_ms is not None:
        ax.axvline(t_burst_ms, color="seagreen", lw=0.8, ls=":",
                   label=_burst_label)
        ax.legend(fontsize=8, loc="upper left")

plt.tight_layout()
p1 = output_root / "pressure_histories.png"
fig1.savefig(p1, dpi=150)
plt.close(fig1)
print(f"Saved: {p1}")


# ── Figure 2: L1d piston kinematics ──────────────────────────────────────────
# piston-0000-history.data columns (from gdtk postprocess.d):
#   0:tindx  1:t  2:x  3:vel  4:is_restrain  5:brakes_on  6:on_buffer

fig2, (ax_pos, ax_vel, ax_vx) = plt.subplots(3, 1, figsize=(10, 9))
ax_vel.sharex(ax_pos)   # time axis shared between position and velocity-vs-time
fig2.suptitle("L1d piston kinematics — DEAP_0 (mid-bounds design)", fontsize=12)

piston_file = output_root / "piston-0000-history.data"
if piston_file.exists():
    pdata  = np.loadtxt(piston_file, comments="#")
    pt_ms  = pdata[:, 1] * 1e3            # s → ms
    px_m   = pdata[:, 2]                   # piston CENTRE (what L1d tracks)
    pv_ms  = pdata[:, 3]                   # velocity (m/s)
    on_buf = pdata[:, 6].astype(int)
    # L1d stores the piston centre; we want the front face for physical plots.
    PISTON_HALF_LENGTH = 0.1105            # (PISTON_FRONT_X − piston_xL0) / 2
    px_front = px_m + PISTON_HALF_LENGTH   # front-face position (m)

    buf_idx = np.where(on_buf == 1)[0]

    # ── position vs time (front face) ────────────────────────────────────
    ax_pos.plot(pt_ms, px_front, lw=0.8, color="steelblue")
    if buf_idx.size > 0:
        ax_pos.axvline(pt_ms[buf_idx[0]], color="red", lw=0.8, ls="--",
                       label=f"buffer contact  t = {pt_ms[buf_idx[0]]:.2f} ms")
    if t_burst_ms is not None:
        ax_pos.axvline(t_burst_ms, color="seagreen", lw=0.8, ls=":",
                       label=_burst_label)
    if buf_idx.size > 0 or t_burst_ms is not None:
        ax_pos.legend(fontsize=8, loc="upper left")
    ax_pos.set_ylabel("Piston front-face position (m)")
    ax_pos.grid(True, lw=0.4, alpha=0.5)

    # ── velocity vs time ──────────────────────────────────────────────────
    ax_vel.plot(pt_ms, pv_ms, lw=0.8, color="steelblue")
    if buf_idx.size > 0:
        ax_vel.axvline(pt_ms[buf_idx[0]], color="red", lw=0.8, ls="--",
                       label=f"buffer contact  t = {pt_ms[buf_idx[0]]:.2f} ms")
    if t_burst_ms is not None:
        ax_vel.axvline(t_burst_ms, color="seagreen", lw=0.8, ls=":",
                       label=_burst_label)
    if buf_idx.size > 0 or t_burst_ms is not None:
        ax_vel.legend(fontsize=8, loc="upper left")
    ax_vel.set_ylabel("Piston velocity (m/s)")
    ax_vel.set_xlabel("Time (ms)")
    ax_vel.grid(True, lw=0.4, alpha=0.5)

    # ── velocity vs position (front face) ────────────────────────────────
    # For the phase-plane (x-axis is position, not time) we mark the piston's
    # front-face position at the burst instant rather than a time.
    ax_vx.plot(px_front, pv_ms, lw=0.8, color="steelblue")
    if buf_idx.size > 0:
        ax_vx.axvline(px_front[buf_idx[0]], color="red", lw=0.8, ls="--",
                      label=f"buffer contact  x = {px_front[buf_idx[0]]:.3f} m")
    if t_burst_ms is not None:
        _bi = np.argmin(np.abs(pt_ms - t_burst_ms))
        ax_vx.axvline(px_front[_bi], color="seagreen", lw=0.8, ls=":",
                      label=f"diaphragm burst  x = {px_front[_bi]:.3f} m")
    if buf_idx.size > 0 or t_burst_ms is not None:
        ax_vx.legend(fontsize=8, loc="upper left")
    ax_vx.set_xlabel("Piston front-face position (m)")
    ax_vx.set_ylabel("Piston velocity (m/s)")
    ax_vx.grid(True, lw=0.4, alpha=0.5)
else:
    for ax in (ax_pos, ax_vel, ax_vx):
        ax.text(0.5, 0.5, "piston-0000-history.data not found",
                ha="center", va="center", transform=ax.transAxes, color="red")

plt.tight_layout()
p2 = output_root / "piston_kinematics.png"
fig2.savefig(p2, dpi=150)
plt.close(fig2)
print(f"Saved: {p2}")


# ── Figure 3: SPARK driver pressure vs L1d primary-diaphragm pressure ─────────
# SPARK p_drvr is the whole-driver pressure (0D model, no spatial variation).
# L1d history-loc-0000 is sampled at x=0 (PD face) — the closest equivalent.

fig3, ax3 = plt.subplots(figsize=(10, 4))
fig3.suptitle("Driver pressure: SPARK (0D) vs L1d at primary diaphragm  [mid-bounds design]",
              fontsize=11)

t_pd_ms, p_pd_MPa = _load_history(0)
if t_pd_ms is not None:
    ax3.plot(t_pd_ms, p_pd_MPa, lw=0.8, color="steelblue", label="L1d  (history-loc-0000, x = 0 m)")

if spark_sim is not None and hasattr(spark_sim.results, 'p_drvr'):
    sp_t_ms  = np.array(spark_sim.results.time) * 1e3
    sp_p_MPa = np.array(spark_sim.results.p_drvr) * 1e-6
    label = "SPARK  (0D driver pressure)" if spark_ok else "SPARK  (incomplete — piston halted early)"
    ax3.plot(sp_t_ms, sp_p_MPa, lw=0.8, color="darkorange", label=label)

# Hold-time band
p_lo = p4 * 0.90 * 1e-6
p_hi = p4 * 1.10 * 1e-6
ax3.axhspan(p_lo, p_hi, alpha=0.12, color="orange",
            label=f"±10 % hold band  (p_burst = {p4*1e-6:.1f} MPa)")
ax3.axhline(p4 * 1e-6, color="orange", lw=0.8, ls="--")

if t_burst_ms is not None:
    ax3.axvline(t_burst_ms, color="seagreen", lw=0.8, ls=":",
                label=_burst_label)

ax3.set_xlabel("Time (ms)")
ax3.set_ylabel("Pressure (MPa)")
ax3.legend(fontsize=8)
ax3.grid(True, lw=0.4, alpha=0.5)

plt.tight_layout()
p3 = output_root / "spark_vs_l1d_driver.png"
fig3.savefig(p3, dpi=150)
plt.close(fig3)
print(f"Saved: {p3}")


# ── Figure 4: Driver pressure at PD + 5 upstream stations ────────────────────
# Overlays all six driver-side history locations on one axes so you can see
# the pressure wave build and propagate toward the PD as the piston compresses.
# Indices 3-7 are appended after the ToF transducers in the job template and
# are purely diagnostic — parse_l1d_outputs only reads indices 0-2.

DRIVER_LOCS = [
    (2,  "PD − 0.01 m"),
    (3,  "PD − 0.02 m"),
    (4,  "PD − 0.03 m"),
    (5,  "PD − 0.04 m"),
    (6,  "PD − 0.05 m"),
    (7,  "PD − 0.06 m"),
    (8,  "PD − 0.07 m"),
    (9,  "PD − 0.08 m"),
    (10, "PD − 0.09 m"),
    (11, "PD − 0.10 m"),
]

fig4, ax4 = plt.subplots(figsize=(10, 5))
fig4.suptitle("L1d driver pressure — PD + 5 upstream stations  [mid-bounds design]",
              fontsize=11)

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(DRIVER_LOCS)))
for (idx, label), color in zip(DRIVER_LOCS, colors):
    t_ms, p_MPa = _load_history(idx)
    if t_ms is not None:
        ax4.plot(t_ms, p_MPa, lw=0.8, color=color, label=label)
    else:
        print(f"  [Figure 4] history-loc-{idx:04d}.data not found — skipping {label}")

ax4.axhspan(p4 * 0.90 * 1e-6, p4 * 1.10 * 1e-6, alpha=0.12, color="orange",
            label=f"±10 % hold band  (p_burst = {p4*1e-6:.1f} MPa)")
ax4.axhline(p4 * 1e-6, color="orange", lw=0.8, ls="--")
if t_burst_ms is not None:
    ax4.axvline(t_burst_ms, color="seagreen", lw=0.8, ls=":",
                label=_burst_label)

ax4.set_xlabel("Time (ms)")
ax4.set_ylabel("Pressure (MPa)")
ax4.set_xlim(23, 35)
ax4.legend(fontsize=8)
ax4.grid(True, lw=0.4, alpha=0.5)

plt.tight_layout()
p4_path = output_root / "driver_pressure_profiles.png"
fig4.savefig(p4_path, dpi=150)
plt.close(fig4)
print(f"Saved: {p4_path}")


# ── Figure 5: Driver pressure subplots — one per history location ─────────────

# Pre-load peak pressures for the two fixed reference locations.
# These are always idx 2 (PD−0.01 m) and idx 11 (PD−0.10 m) regardless
# of how many locations are in DRIVER_LOCS.
_, _p_near = _load_history(2)   # PD − 0.01 m
_, _p_far  = _load_history(11)  # PD − 0.10 m
p_peak_near = float(np.max(_p_near)) if _p_near is not None else None
p_peak_far  = float(np.max(_p_far))  if _p_far  is not None else None

n_locs = len(DRIVER_LOCS)
n_cols  = 2
n_rows  = (n_locs + 1) // n_cols   # ceiling division
fig5, axes5_2d = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex=True)
fig5.suptitle("L1d driver pressure — individual subplots per station  [mid-bounds design]",
              fontsize=11)

axes5_flat = axes5_2d.flatten()

colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_locs))
for ax, (idx, label), color in zip(axes5_flat, DRIVER_LOCS, colors):
    t_ms, p_MPa = _load_history(idx)
    ax.set_title(label, fontsize=9)
    ax.set_ylabel("Pressure (MPa)")
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.axhspan(p4 * 0.90 * 1e-6, p4 * 1.10 * 1e-6, alpha=0.12, color="orange")
    ax.axhline(p4 * 1e-6, color="orange", lw=0.8, ls="--")
    if p_peak_near is not None:
        ax.axhline(p_peak_near, color="crimson", lw=0.9, ls="--",
                   label=f"peak PD−0.01 m = {p_peak_near:.3f} MPa")
    if p_peak_far is not None:
        ax.axhline(p_peak_far, color="steelblue", lw=0.9, ls="--",
                   label=f"peak PD−0.10 m = {p_peak_far:.3f} MPa")
    if t_ms is not None:
        ax.plot(t_ms, p_MPa, lw=0.8, color=color)
        ax.set_xlim(15, 25)
    else:
        ax.text(0.5, 0.5, f"history-loc-{idx:04d}.data not found",
                ha="center", va="center", transform=ax.transAxes, color="red")
    if t_burst_ms is not None:
        ax.axvline(t_burst_ms, color="seagreen", lw=0.8, ls=":",
                   label=_burst_label)
    if p_peak_near is not None or p_peak_far is not None or t_burst_ms is not None:
        ax.legend(fontsize=7, loc="upper left")

# x-axis label only on the bottom row
for ax in axes5_2d[-1, :]:
    ax.set_xlabel("Time (ms)")

# hide any unused axes (if n_locs is odd)
for ax in axes5_flat[n_locs:]:
    ax.set_visible(False)

plt.tight_layout()
p5_path = output_root / "driver_pressure_subplots.png"
fig5.savefig(p5_path, dpi=150)
plt.close(fig5)
print(f"Saved: {p5_path}")

print()
print("All outputs in: src/L1d_Outputs/DEAP_0/")

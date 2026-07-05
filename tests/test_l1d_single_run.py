"""
Standalone L1d4 integration test.

Runs a single L1d simulation using exactly the same code path that the
CMA-ES uses (evaluate._evaluate_l1d → run_l1d → parse_l1d_outputs).

The design vector is the mid-point of the physical BOUNDS — a neutral,
physically reasonable starting point with no prior knowledge required.

Run from the project root on the l1d_cht_al branch:

    cd /home/x-lab-user/Optimisation_Project
    git checkout l1d_cht_al
    python tests/test_l1d_single_run.py

Prerequisites:
  - The three .lua gas model files must be present in the project root.

Outputs (written to src/L1d_Outputs/DEAP_0/):
  - pressure_histories.png       : L1d pressure at the two ToF transducer
                                   stations (history-loc-0000 and -0001)
  - piston_kinematics.png        : L1d piston position and velocity
  - driver_pressure_profiles.png : single-panel pressure trace at
                                   history-loc-0002 with ±10 % hold band
                                   and computed t_hold window
"""
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

# x_phys = np.array([
#     80,       # percent_He     [%]
#     77.2e3,      # driver_p       [Pa]
#     35.7e6,     # p4 (burst)     [Pa]
#     0.085,      # D_throat       [m]
#     6.08e6,      # reservoir_p    [Pa]
#     0.045,       # buffer_length  [m]
# ])

x_phys = np.array([
    80.0,       # percent_He     [%]
    92.8e3,      # driver_p       [Pa]
    27.9e6,     # p4 (burst)     [Pa]
    0.085,      # D_throat       [m]
    6.85e6,      # reservoir_p    [Pa]
    0.045,       # buffer_length  [m]
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
print("L1d single-run test")
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
    print(f"  delta_vs     = {delta_vs:.1f} m/s  (|vs1 - 3585|; target = 0)")
    if t_hold <= 0:
        print("  WARNING: t_hold = 0 — diaphragm may not have burst or window never entered")
    if impact_speed >= sentinel_imp:
        print("  WARNING: impact_speed at or above sentinel value")
    if delta_vs >= sentinel_dvs:
        print("  WARNING: delta_vs at or above sentinel value — vs1 parse may have failed")

print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Plots
# ─────────────────────────────────────────────────────────────────────────────

output_root.mkdir(parents=True, exist_ok=True)

HISTORY_CONFIGS = [
    (0, f"x = {TRANSDUCER_XS[0]:.3f} m  (ToF station 1)", "ToF transducer 1"),
    (1, f"x = {TRANSDUCER_XS[1]:.3f} m  (ToF station 2)", "ToF transducer 2"),
]
DRIVER_PROBE_IDXS = (2, 3, 4)   # all driver-side history locs, ordered for overlay


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
    # The ±10 % hold band only has physical meaning on the driver-side
    # probes (idx ∈ DRIVER_PROBE_IDXS), where p_burst is the reference
    # pressure.  The shock-tube transducers see a totally different
    # pressure range, so don't draw it on idx 0 and 1.
    if idx in DRIVER_PROBE_IDXS:
        p_lo_MPa = p4 * 0.90 * 1e-6
        p_hi_MPa = p4 * 1.10 * 1e-6
        ax.axhspan(p_lo_MPa, p_hi_MPa, alpha=0.12, color="orange",
                   label=f"±10 % hold band  (p_burst = {p4*1e-6:.2f} MPa)")
        ax.axhline(p4 * 1e-6, color="orange", lw=0.8, ls="--")
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


# ── Figure 3: Driver-side pressure at history-loc-0002 ───────────────────────
# Single-panel diagnostic for the t_hold metric.  We plot the raw trace, the
# ±10 % hold band around p_burst, the burst instant, and a shaded interval
# spanning the computed t_hold window — defined the same way as in
# parse_l1d_outputs: from t_burst until the first sample where the trace
# leaves the band (or to t_finish if it never does).  If t_hold's number
# matches the visual width of the shaded interval, the parser is correct.

fig4, ax4 = plt.subplots(figsize=(10, 5))
fig4.suptitle("L1d driver-side pressure (PD − 0.02 m)  [mid-bounds design]",
              fontsize=11)

# Canonical t_hold probe: history-loc-0002 (PD − 0.02 m) is what the
# parser reads, and the t_hold window below is derived from it.
t_drv_ms, p_drv_MPa = _load_history(2)
if t_drv_ms is not None:
    ax4.plot(t_drv_ms, p_drv_MPa, lw=1.0, color="steelblue",
             label="L1d  (loc 2 — PD − 0.02 m, canonical t_hold probe)")
else:
    print("  [Figure 3] history-loc-0002.data not found")

if t_drv_ms is None:
    ax4.text(0.5, 0.5, "no driver-side history files found",
             ha="center", va="center", transform=ax4.transAxes, color="red")

ax4.axhspan(p4 * 0.90 * 1e-6, p4 * 1.10 * 1e-6, alpha=0.12, color="orange",
            label=f"±10 % hold band  (p_burst = {p4*1e-6:.2f} MPa)")
ax4.axhline(p4 * 1e-6, color="orange", lw=0.8, ls="--")

# Mirror parse_l1d_outputs's first-exit definition so the shaded interval
# below visualises exactly the t_hold the parser computed.
if t_drv_ms is not None and t_burst_ms is not None and ok:
    p_lo_MPa = p4 * 0.90 * 1e-6
    p_hi_MPa = p4 * 1.10 * 1e-6
    post_burst  = t_drv_ms >= t_burst_ms
    out_of_band = (p_drv_MPa < p_lo_MPa) | (p_drv_MPa > p_hi_MPa)
    exit_idxs   = np.where(post_burst & out_of_band)[0]
    t_exit_ms   = float(t_drv_ms[exit_idxs[0]]) if exit_idxs.size else float(t_drv_ms[-1])
    ax4.axvspan(t_burst_ms, t_exit_ms, alpha=0.18, color="seagreen",
                label=f"t_hold window  ({(t_exit_ms - t_burst_ms):.3f} ms)")

if t_burst_ms is not None:
    ax4.axvline(t_burst_ms, color="seagreen", lw=0.8, ls=":",
                label=_burst_label)

ax4.set_xlabel("Time (ms)")
ax4.set_ylabel("Pressure (MPa)")
ax4.legend(fontsize=8, loc="upper left")
ax4.grid(True, lw=0.4, alpha=0.5)

# Zoom the x-axis to ±3 ms around the diaphragm-burst instant — the only
# physically interesting window for the t_hold metric.  Skip the zoom if
# the diaphragm never burst, so the plot still shows the whole sim and
# the failure is visually obvious.
if t_burst_ms is not None:
    ax4.set_xlim(t_burst_ms - 3.0, t_burst_ms + 3.0)

plt.tight_layout()
p4_path = output_root / "driver_pressure_profiles.png"
fig4.savefig(p4_path, dpi=150)
plt.close(fig4)
print(f"Saved: {p4_path}")


# ── Figures 4..N: Gas-path area profile at selected tindx ────────────────────
# One PNG per chosen tindx, plotting cross-sectional area vs x for BOTH
# the driver-gas slug (slug-0001) and the test-gas slug (slug-0002).
# The gas-path order in the template is [reservoir(0), driver(1), test
# gas(2)], so slugs 1 and 2 together cover everything from the piston
# face to the shock-tube exit (the reservoir is on the upstream side of
# the piston and is plotted separately if you ever need it).
#
# Schedule: burst-50, burst-45, ..., burst-5, burst-1, burst, burst+5,
# burst+10, ..., burst+50.  That's 22 frames straddling the diaphragm
# rupture, sampled every 5 dt_plot steps on either side plus a dense
# pair (burst-1, burst) right at the event.  All 22 share the same x-
# and y-axis limits so the Lagrangian drift of each slug is visible at
# a glance instead of being normalised away by per-plot autoscaling.

SLUG1_FACES_FILE = job_subdir / "slug-0001-faces.data"
SLUG2_FACES_FILE = job_subdir / "slug-0002-faces.data"
TIMES_FILE       = job_subdir / "times.data"
_d_for_area      = job_subdir / "diaphragm-0000.data"


def _stream_slug_faces(path, wanted):
    """Stream a slug-NNNN-faces.data file, returning {tindx: ndarray} only
    for the tindx values in ``wanted``.

    Single linear pass over the file; only buffers rows for blocks we
    actually want, so memory stays bounded by the size of one block
    regardless of how large the file is.
    """
    out = {}
    cur_tindx, cur_rows = None, []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("# tindx"):
                if cur_tindx in wanted and cur_rows:
                    out[cur_tindx] = np.array(cur_rows)
                cur_tindx = int(s.split()[2])
                cur_rows = []
            elif s.startswith("#"):
                continue
            else:
                if cur_tindx in wanted:
                    parts = s.split()
                    cur_rows.append((float(parts[0]), float(parts[1])))
        if cur_tindx in wanted and cur_rows:
            out[cur_tindx] = np.array(cur_rows)
    return out


# Re-derive burst tindx locally to decouple from the earlier block's locals.
_burst_tindx_area = None
if _d_for_area.exists():
    _arr = np.loadtxt(_d_for_area, comments="#", dtype=int)
    if _arr.ndim == 1:
        _arr = _arr[np.newaxis, :]
    _rows = np.where(_arr[:, 1] == 2)[0]
    if _rows.size > 0:
        _burst_tindx_area = int(_arr[_rows[0], 0])

_have_any_slug = SLUG1_FACES_FILE.exists() or SLUG2_FACES_FILE.exists()

if _burst_tindx_area is None or not _have_any_slug:
    print("[Area profiles] skipped — burst tindx unknown or both slug-faces files missing")
else:
    area_dir = output_root / "area_profiles"
    area_dir.mkdir(exist_ok=True)

    schedule = (
        [_burst_tindx_area - 5 * k for k in range(10, 0, -1)]   # burst-50 ... burst-5
        + [_burst_tindx_area - 1, _burst_tindx_area]            # burst-1, burst
        + [_burst_tindx_area + 5 * k for k in range(1, 11)]     # burst+5 ... burst+50
    )
    schedule = [t for t in schedule if t >= 0]   # drop any negative tindxs if burst < 50
    wanted = set(schedule)

    blocks_drv = _stream_slug_faces(SLUG1_FACES_FILE, wanted) if SLUG1_FACES_FILE.exists() else {}
    blocks_tst = _stream_slug_faces(SLUG2_FACES_FILE, wanted) if SLUG2_FACES_FILE.exists() else {}

    # Per-tindx wall-clock time for plot titles.
    tindx_to_t_ms = {}
    if TIMES_FILE.exists():
        _ta = np.loadtxt(TIMES_FILE, comments="#")
        if _ta.ndim == 1:
            _ta = _ta[np.newaxis, :]
        tindx_to_t_ms = {int(row[0]): float(row[1]) * 1e3 for row in _ta}

    # Global axis limits across all selected frames AND both slugs, so
    # every plot frames the same region of (x, area) space.  Without this,
    # autoscaling would hide the slugs' Lagrangian drift in x.
    _all_x, _all_a = [], []
    for _src in (blocks_drv, blocks_tst):
        for _t, _a in _src.items():
            _all_x.append(_a[:, 0])
            _all_a.append(_a[:, 1])
    if _all_x:
        _xs_all   = np.concatenate(_all_x)
        _area_all = np.concatenate(_all_a)
        _x_lo, _x_hi = float(_xs_all.min()), float(_xs_all.max())
        _x_pad = 0.05 * (_x_hi - _x_lo) if _x_hi > _x_lo else 0.01
        _area_cm2_hi = float(_area_all.max()) * 1e4 * 1.05   # m² → cm²
    else:
        _x_lo, _x_hi, _x_pad, _area_cm2_hi = -1.0, 1.0, 0.05, 1.0

    print(f"[Area profiles] generating {len(schedule)} plots at "
          f"tindx ∈ {schedule[0]}..{schedule[-1]}  "
          f"(driver={len(blocks_drv)}, test={len(blocks_tst)})")

    for _tindx in schedule:
        has_drv = _tindx in blocks_drv
        has_tst = _tindx in blocks_tst
        if not (has_drv or has_tst):
            print(f"  tindx={_tindx}: not in either slug file — skipping")
            continue

        _t_ms = tindx_to_t_ms.get(_tindx)
        _t_label = f"t = {_t_ms:.3f} ms" if _t_ms is not None else "t = ?"
        _delta = _tindx - _burst_tindx_area
        if _delta == 0:
            _rel = "burst"
        elif _delta > 0:
            _rel = f"burst + {_delta}"
        else:
            _rel = f"burst − {-_delta}"

        fig_a, ax_a = plt.subplots(figsize=(10, 4))
        fig_a.suptitle(
            f"Gas-path area profile — tindx={_tindx} ({_rel})   [{_t_label}]",
            fontsize=11,
        )

        if has_drv:
            _ad = blocks_drv[_tindx]
            ax_a.plot(_ad[:, 0], _ad[:, 1] * 1e4, lw=1.0, color="crimson",
                      marker=".", ms=2, label="driver slug (slug-0001)")
        if has_tst:
            _at = blocks_tst[_tindx]
            ax_a.plot(_at[:, 0], _at[:, 1] * 1e4, lw=1.0, color="seagreen",
                      marker=".", ms=2, label="test-gas slug (slug-0002)")

        # Primary-diaphragm marker — fixed reference between the two slugs.
        ax_a.axvline(0.0, color="orange", ls="--", lw=0.8, alpha=0.6,
                     label="primary diaphragm (x = 0)")

        ax_a.set_xlim(_x_lo - _x_pad, _x_hi + _x_pad)
        ax_a.set_ylim(0.0, _area_cm2_hi)
        ax_a.set_xlabel("x  (m)")
        ax_a.set_ylabel("Cross-sectional area  (cm²)")
        ax_a.grid(True, lw=0.4, alpha=0.5)
        ax_a.legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        _out = area_dir / f"area_profile_tindx_{_tindx:04d}.png"
        fig_a.savefig(_out, dpi=150)
        plt.close(fig_a)

    print(f"[Area profiles] saved to {area_dir}/")


print()
print("All outputs in: src/L1d_Outputs/DEAP_0/")

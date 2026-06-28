"""
Re-plot L1d outputs from an existing run, without re-running the CFD.

This is the workshopping companion to tests/test_l1d_single_run.py.
It reads .data files from a finished L1d simulation and regenerates the
plots, so you can iterate on t_hold logic without paying the cost of
another simulation.

Pipeline separation:

    [ L1d sim ]  -->  [ .data files ]  <--  [ this script ]
       slow              interface              fast, iterate freely


SUPPORTED LAYOUTS
-----------------
Two on-disk layouts are auto-detected:

  (1) "Nested"  -- the layout written by test_l1d_single_run.py:
         run_dir/
            piston-0000-history.data
            run_dir/                     (history files one level deeper)
               history-loc-*.data
               diaphragm-0000.data
               times.data

  (2) "Flat"    -- the layout in the saved condition_N folders:
         run_dir/
            piston-0000.data             (NB: no -history suffix, no t col)
            history-loc-*.data
            diaphragm-0000.data
            times.data
            condition N                  (free-form spec text -- optional)

If the run_dir contains a "condition*" text file, p_burst is parsed
from it automatically (two formats supported -- see parse_condition_file).
Otherwise the module-level P_BURST fallback is used.


WHAT TO EDIT
------------
  1.  CONFIG block (below) -- only the smoothing knobs and P_BURST
      fallback.  Per-run parameters now come from the condition file.

  2.  compute_t_hold() -- the workshop target.


USAGE
-----
    cd /home/x-lab-user/Optimisation_Project

    # Single run (auto-detect layout, auto-extract p_burst if available)
    python3 src/L1d_Outputs/replot_l1d_outputs.py src/L1d_Outputs/DEAP_0/condition_1

    # Batch: any parent dir containing condition_* subdirs
    python3 src/L1d_Outputs/replot_l1d_outputs.py src/L1d_Outputs/DEAP_0

    # Customise the output suffix (default: _replot)
    python3 src/L1d_Outputs/replot_l1d_outputs.py <path> --suffix _v2
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Resolve src/ on sys.path so problem.t_hold imports cleanly when the
# script is run directly (Path(__file__).parent.parent == .../src).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from problem.t_hold import smooth_pressure, compute_t_hold  # noqa: E402


# ────────────────────────────────────────────────────────────────────────────
# CONFIG -- per-run params are now auto-extracted from "condition*" files
# ────────────────────────────────────────────────────────────────────────────
# Burst pressure FALLBACK (Pa).  Used only when no condition file is found
# in the run directory (e.g. raw DEAP_0/ output from the integration test).
P_BURST = 35.7e6

# Transducer x-positions (m, relative to PD).  Pulled from the run config.
# Used only for plot titles -- the actual data comes from the history files.
TRANSDUCER_XS = (4.470, 5.470)

# Low-pass smoothing applied to the driver-side pressure before the
# band-check (Savitzky-Golay filter).
#   SMOOTH_WINDOW_S = None  -> no smoothing (original behaviour).
#   SMOOTH_WINDOW_S = 2e-4  -> ~0.2 ms window; reasonable for L1d traces
#                              where dt_plot ≈ 10 μs (≈ 20 samples).
# Tune up for noisier traces; tune down if it starts smearing the real exit.
SMOOTH_WINDOW_S  = 5.0e-4
SMOOTH_POLYORDER = 3

# Default output directory to re-plot.  Override on the CLI if you want.
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "DEAP_0"


# smooth_pressure and compute_t_hold now imported from problem.t_hold
# (see top-of-file).  Workshop knobs (SMOOTH_WINDOW_S, SMOOTH_POLYORDER)
# above still control what gets passed in for the diagnostic plots.


# ────────────────────────────────────────────────────────────────────────────
# Discovery -- normalise the two on-disk layouts into a uniform handle
# ────────────────────────────────────────────────────────────────────────────
def discover_run(run_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    """
    Locate the components of a single L1d run, regardless of layout.

    Returns
    -------
    (job_subdir, piston_file, condition_file)
        job_subdir     : directory containing history-loc-*.data, diaphragm
                         and times files.  May equal run_dir (flat layout)
                         or run_dir / run_dir.name (nested layout).
        piston_file    : Path to the piston history file (either name), or
                         None if absent.
        condition_file : Path to the "condition*" spec file (free-form txt
                         that defines p_burst etc.), or None if absent.
    """
    # Layout detection: the presence of history-loc-0002.data is the marker.
    if (run_dir / "history-loc-0002.data").exists():
        job_subdir = run_dir
    elif (run_dir / run_dir.name / "history-loc-0002.data").exists():
        job_subdir = run_dir / run_dir.name
    else:
        return None, None, None

    # Piston file: try both naming conventions in priority order.
    piston_file = None
    for name in ("piston-0000-history.data", "piston-0000.data"):
        candidate = run_dir / name
        if candidate.exists():
            piston_file = candidate
            break

    # Condition spec: any file whose name starts with "condition" (case-
    # insensitive), ignoring directories (the nested layout has a same-
    # named subdir).
    condition_file = None
    for p in sorted(run_dir.iterdir()):
        if p.is_file() and p.name.lower().startswith("condition"):
            condition_file = p
            break

    return job_subdir, piston_file, condition_file


# ────────────────────────────────────────────────────────────────────────────
# Condition-file parsing -- two formats supported
# ────────────────────────────────────────────────────────────────────────────
# Format A (the np.array literal):
#     x_phys = np.array([
#         80.0,       # percent_He     [%]
#         101.3e3,    # driver_p       [Pa]
#         15.5e6,     # p4 (burst)     [Pa]   ← what we want (3rd numeric value)
#         ...
#     ])
#
# Format B (keyword assignment):
#     prim_dia_burst  = 1.8e6      # Primary diaphragm burst pressure [Pa]
#
_PRIM_BURST_RE = re.compile(r"prim_dia_burst\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_NP_ARRAY_RE   = re.compile(r"np\.array\(\s*\[(.*?)\]\s*\)", re.DOTALL)
_NUMBER_RE     = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_condition_file(path: Path) -> float | None:
    """
    Extract p_burst (Pa) from a condition spec file.  Returns None if
    neither format can be matched.

    Tries Format B (keyword) first because it's unambiguous; falls back
    to Format A (third numeric entry in the np.array body, comments
    stripped first).
    """
    try:
        text = path.read_text()
    except OSError:
        return None

    # Format B
    m = _PRIM_BURST_RE.search(text)
    if m:
        return float(m.group(1))

    # Format A
    m = _NP_ARRAY_RE.search(text)
    if m:
        body = m.group(1)
        # Strip Python-style line comments before pulling numbers --
        # otherwise the "10" in "[%]" gets parsed as a value.
        body_clean = re.sub(r"#.*", "", body)
        numbers = _NUMBER_RE.findall(body_clean)
        if len(numbers) >= 3:
            return float(numbers[2])   # third entry = p4 (burst)

    return None


# compute_t_hold lives in problem.t_hold (imported above).  Tune
# DEFAULT_* there to change behaviour for prod + workshop in lockstep,
# or tune SMOOTH_WINDOW_S / SMOOTH_POLYORDER above to override here only.


# ────────────────────────────────────────────────────────────────────────────
# Loaders -- thin wrappers over the .data file format
# ────────────────────────────────────────────────────────────────────────────
def load_history(job_subdir: Path, idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (t_s, p_Pa) for history-loc-NNNN.data, or (None, None) if missing."""
    f = job_subdir / f"history-loc-{idx:04d}.data"
    if not f.exists():
        return None, None
    d = np.loadtxt(f, comments="#")
    return d[:, 0], d[:, 4]   # cols: 0=t (s), 4=p (Pa)


def load_tindx_to_t(job_subdir: Path, piston_file: Path | None) -> dict[int, float]:
    """
    Build a tindx -> simulation-time (s) lookup.

    Source priority:
      1. times.data           (always tindx + t; present in both layouts)
      2. piston-0000-history  (7-col layout: tindx, t, ...; fallback)

    Returns {} if no source is available.  Used by find_burst() and by
    plot_piston_kinematics() when the piston file lacks a t column.
    """
    times_file = job_subdir / "times.data"
    if times_file.exists():
        arr = np.loadtxt(times_file, comments="#")
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return {int(row[0]): float(row[1]) for row in arr}

    if piston_file is not None and piston_file.exists():
        arr = np.loadtxt(piston_file, comments="#")
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.shape[1] >= 7:   # has t column at index 1
            return {int(row[0]): float(row[1]) for row in arr}

    return {}


def find_burst(
    job_subdir: Path,
    tindx_to_t: dict[int, float],
) -> tuple[float | None, float | None, int | None]:
    """
    Return (t_burst_s, dt_resolution_s, burst_tindx) from L1d outputs.

    Authoritative source for the burst event: diaphragm-0000.data
    records (tindx, state) at every dt_plot step.  state == 2 means
    fully ruptured.  We find the first such tindx, then look its time
    up in tindx_to_t (which is format-agnostic -- see load_tindx_to_t).

    The true rupture instant lies in [t_prev, t_burst]; the width of
    that window is the dt_plot resolution and is returned as the
    temporal uncertainty.
    """
    diaph_file = job_subdir / "diaphragm-0000.data"
    if not diaph_file.exists() or not tindx_to_t:
        return None, None, None

    ddata = np.loadtxt(diaph_file, comments="#", dtype=int)
    if ddata.ndim == 1:
        ddata = ddata[np.newaxis, :]

    open_rows = np.where(ddata[:, 1] == 2)[0]
    if open_rows.size == 0:
        return None, None, None

    burst_tindx = int(ddata[open_rows[0], 0])
    if burst_tindx not in tindx_to_t:
        return None, None, burst_tindx

    t_burst_s = tindx_to_t[burst_tindx]
    t_prev    = tindx_to_t.get(burst_tindx - 1)
    dt_res_s  = (t_burst_s - t_prev) if t_prev is not None else None
    return t_burst_s, dt_res_s, burst_tindx


# ────────────────────────────────────────────────────────────────────────────
# Plots
# ────────────────────────────────────────────────────────────────────────────
def _burst_legend_label(t_burst_ms: float | None, dt_ms: float | None) -> str | None:
    if t_burst_ms is None:
        return None
    if dt_ms is None:
        return f"diaphragm burst  t = {t_burst_ms:.3f} ms"
    return f"diaphragm burst  t = {t_burst_ms:.3f} ±{dt_ms:.4f} ms"


def plot_pressure_histories(
    job_subdir: Path,
    output_root: Path,
    t_burst_ms: float | None,
    burst_label: str | None,
    suffix: str,
) -> None:
    """Two-panel figure: pressure at ToF stations 1 and 2."""
    configs = [
        (0, f"x = {TRANSDUCER_XS[0]:.3f} m  (ToF station 1)", "ToF transducer 1"),
        (1, f"x = {TRANSDUCER_XS[1]:.3f} m  (ToF station 2)", "ToF transducer 2"),
    ]
    fig, axes = plt.subplots(len(configs), 1, figsize=(10, 3 * len(configs)))
    fig.suptitle(f"L1d pressure histories  ({output_root.name})", fontsize=12)

    for ax, (idx, x_label, description) in zip(axes, configs):
        t_s, p_Pa = load_history(job_subdir, idx)
        ax.set_title(f"loc {idx} -- {description}  [{x_label}]")
        ax.set_ylabel("Pressure (MPa)")
        ax.set_xlabel("Time (ms)")
        ax.grid(True, lw=0.4, alpha=0.5)
        if t_s is None:
            ax.text(0.5, 0.5, f"history-loc-{idx:04d}.data not found",
                    ha="center", va="center", transform=ax.transAxes, color="red")
            continue
        ax.plot(t_s * 1e3, p_Pa * 1e-6, lw=0.8, color="steelblue")
        if t_burst_ms is not None:
            ax.axvline(t_burst_ms, color="seagreen", lw=0.8, ls=":", label=burst_label)
            ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    out = output_root / f"pressure_histories{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out.name}")


def plot_piston_kinematics(
    piston_file: Path | None,
    tindx_to_t: dict[int, float],
    output_root: Path,
    t_burst_ms: float | None,
    burst_label: str | None,
    suffix: str,
) -> None:
    """
    Three-panel piston diagnostic: x(t), v(t), v(x).

    Handles both piston-file layouts:
      7 cols: tindx, t, x, vel, is_restrain, brakes_on, on_buffer   (DEAP_0)
      6 cols: tindx,    x, vel, is_restrain, brakes_on, on_buffer   (condition_N)

    For the 6-col layout, simulation time is looked up via tindx_to_t
    (sourced from times.data).
    """
    fig, (ax_pos, ax_vel, ax_vx) = plt.subplots(3, 1, figsize=(10, 9))
    ax_vel.sharex(ax_pos)
    fig.suptitle(f"L1d piston kinematics  ({output_root.name})", fontsize=12)

    if piston_file is None or not piston_file.exists():
        for ax in (ax_pos, ax_vel, ax_vx):
            ax.text(0.5, 0.5, "piston file not found",
                    ha="center", va="center", transform=ax.transAxes, color="red")
        plt.tight_layout()
        out = output_root / f"piston_kinematics{suffix}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  saved: {out.name}  (no data)")
        return

    pdata = np.loadtxt(piston_file, comments="#")
    if pdata.shape[1] >= 7:
        # 7-col layout: t is column 1.
        pt_s   = pdata[:, 1]
        px_m   = pdata[:, 2]
        pv_ms  = pdata[:, 3]
        on_buf = pdata[:, 6].astype(int)
    else:
        # 6-col layout: no t column; look it up via tindx -> t.
        tindx  = pdata[:, 0].astype(int)
        pt_s   = np.array([tindx_to_t.get(int(ti), np.nan) for ti in tindx])
        px_m   = pdata[:, 1]
        pv_ms  = pdata[:, 2]
        on_buf = pdata[:, 5].astype(int)

    pt_ms = pt_s * 1e3
    PISTON_HALF_LENGTH = 0.1105
    px_front = px_m + PISTON_HALF_LENGTH

    buf_idx = np.where(on_buf == 1)[0]

    # x(t)
    ax_pos.plot(pt_ms, px_front, lw=0.8, color="steelblue")
    if buf_idx.size > 0:
        ax_pos.axvline(pt_ms[buf_idx[0]], color="red", lw=0.8, ls="--",
                       label=f"buffer contact  t = {pt_ms[buf_idx[0]]:.2f} ms")
    if t_burst_ms is not None:
        ax_pos.axvline(t_burst_ms, color="seagreen", lw=0.8, ls=":", label=burst_label)
    ax_pos.set_ylabel("Piston front-face position (m)")
    ax_pos.grid(True, lw=0.4, alpha=0.5)
    ax_pos.legend(fontsize=8, loc="upper left")

    # v(t)
    ax_vel.plot(pt_ms, pv_ms, lw=0.8, color="steelblue")
    if buf_idx.size > 0:
        ax_vel.axvline(pt_ms[buf_idx[0]], color="red", lw=0.8, ls="--",
                       label=f"buffer contact  t = {pt_ms[buf_idx[0]]:.2f} ms")
    if t_burst_ms is not None:
        ax_vel.axvline(t_burst_ms, color="seagreen", lw=0.8, ls=":", label=burst_label)
    ax_vel.set_ylabel("Piston velocity (m/s)")
    ax_vel.set_xlabel("Time (ms)")
    ax_vel.grid(True, lw=0.4, alpha=0.5)
    ax_vel.legend(fontsize=8, loc="upper left")

    # v(x) phase plane
    ax_vx.plot(px_front, pv_ms, lw=0.8, color="steelblue")
    if buf_idx.size > 0:
        ax_vx.axvline(px_front[buf_idx[0]], color="red", lw=0.8, ls="--",
                      label=f"buffer contact  x = {px_front[buf_idx[0]]:.3f} m")
    if t_burst_ms is not None:
        bi = np.argmin(np.abs(pt_ms - t_burst_ms))
        ax_vx.axvline(px_front[bi], color="seagreen", lw=0.8, ls=":",
                      label=f"diaphragm burst  x = {px_front[bi]:.3f} m")
    ax_vx.set_xlabel("Piston front-face position (m)")
    ax_vx.set_ylabel("Piston velocity (m/s)")
    ax_vx.grid(True, lw=0.4, alpha=0.5)
    ax_vx.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    out = output_root / f"piston_kinematics{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out.name}")


def plot_driver_pressure(
    job_subdir: Path,
    output_root: Path,
    p_burst: float,
    t_burst_s: float | None,
    burst_label: str | None,
    suffix: str,
) -> None:
    """
    Driver-side pressure at loc 0002 with the +-10 % hold band and the
    t_hold window AS COMPUTED BY compute_t_hold().  Edit compute_t_hold
    and re-run this script to see the new window immediately.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"L1d driver-side pressure  ({output_root.name})", fontsize=11)

    t_s, p_Pa = load_history(job_subdir, 2)
    if t_s is None:
        ax.text(0.5, 0.5, "history-loc-0002.data not found",
                ha="center", va="center", transform=ax.transAxes, color="red")
    else:
        # Raw trace -- thin, lightly opaque so the smoothed overlay reads.
        raw_alpha = 0.5 if SMOOTH_WINDOW_S is not None else 1.0
        ax.plot(t_s * 1e3, p_Pa * 1e-6, lw=0.8, color="steelblue",
                alpha=raw_alpha,
                label="L1d  (loc 2 -- driver-side probe, raw)")
        # Smoothed trace -- this is what compute_t_hold's band-check sees.
        if SMOOTH_WINDOW_S is not None:
            p_smooth = smooth_pressure(t_s, p_Pa, SMOOTH_WINDOW_S, SMOOTH_POLYORDER)
            ax.plot(t_s * 1e3, p_smooth * 1e-6, lw=1.4, color="navy",
                    label=f"smoothed  (Savitzky-Golay, "
                          f"window = {SMOOTH_WINDOW_S*1e3:.3f} ms, "
                          f"order {SMOOTH_POLYORDER})")

    # +-10 % band
    ax.axhspan(p_burst * 0.90 * 1e-6, p_burst * 1.10 * 1e-6,
               alpha=0.12, color="orange",
               label=f"±10 % hold band  (p_burst = {p_burst*1e-6:.2f} MPa)")
    ax.axhline(p_burst * 1e-6, color="orange", lw=0.8, ls="--")

    # t_hold window from the workshop function.  Note t_start may be
    # later than t_burst when the delayed-start provision triggers.
    if t_s is not None and t_burst_s is not None:
        t_hold, t_start, t_exit = compute_t_hold(
            t_s, p_Pa, p_burst, t_burst_s,
            smooth_window_s=SMOOTH_WINDOW_S,
            smooth_polyorder=SMOOTH_POLYORDER,
        )
        if t_hold > 0.0:
            ax.axvspan(t_start * 1e3, t_exit * 1e3, alpha=0.18, color="seagreen",
                       label=f"t_hold (compute_t_hold) = {t_hold*1e3:.3f} ms")
            # Mark the settling delay, if any.
            if t_start > t_burst_s:
                ax.axvline(t_start * 1e3, color="purple", lw=0.8, ls="--",
                           label=f"t_start (delayed-start) = {t_start*1e3:.3f} ms")
        else:
            print("  [t_hold] no valid hold established "
                  "(below band at burst, no entry within entry_window_s)")
        print(f"  compute_t_hold:  t_hold = {t_hold*1e3:.3f} ms   "
              f"t_burst = {t_burst_s*1e3:.3f} ms   "
              f"t_start = {t_start*1e3:.3f} ms   "
              f"t_exit = {t_exit*1e3:.3f} ms")

    if t_burst_s is not None:
        ax.axvline(t_burst_s * 1e3, color="seagreen", lw=0.8, ls=":", label=burst_label)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Pressure (MPa)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, lw=0.4, alpha=0.5)
    # Zoom +-10 ms around burst if known.
    if t_burst_s is not None:
        ax.set_xlim(t_burst_s * 1e3 - 10.0, t_burst_s * 1e3 + 10.0)

    plt.tight_layout()
    out = output_root / f"driver_pressure_profiles{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out.name}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
def process_run(run_dir: Path, suffix: str) -> bool:
    """
    Re-plot a single L1d run directory.  Returns True on success, False
    if the run was unrecognisable (no history data).
    """
    job_subdir, piston_file, condition_file = discover_run(run_dir)
    if job_subdir is None:
        print(f"  SKIP: no history-loc-0002.data in {run_dir} (or its same-named subdir)")
        return False

    # Resolve p_burst -- prefer the condition file, fall back to the module constant.
    if condition_file is not None:
        parsed = parse_condition_file(condition_file)
        if parsed is not None:
            p_burst = parsed
            print(f"  condition file: {condition_file.name}  →  "
                  f"p_burst = {p_burst*1e-6:.2f} MPa")
        else:
            p_burst = P_BURST
            print(f"  condition file: {condition_file.name}  (unparseable; "
                  f"falling back to P_BURST = {P_BURST*1e-6:.2f} MPa)")
    else:
        p_burst = P_BURST
        print(f"  no condition file  →  using P_BURST = {P_BURST*1e-6:.2f} MPa")

    print(f"  job_subdir : {job_subdir.relative_to(run_dir.parent) if run_dir.parent in job_subdir.parents or job_subdir == run_dir else job_subdir}")
    print(f"  piston file: {piston_file.name if piston_file else '(none)'}")

    tindx_to_t = load_tindx_to_t(job_subdir, piston_file)
    t_burst_s, dt_res_s, burst_tindx = find_burst(job_subdir, tindx_to_t)

    if t_burst_s is None:
        print("  Diaphragm burst: not detected")
        t_burst_ms = None
        dt_res_ms  = None
    else:
        t_burst_ms = t_burst_s * 1e3
        dt_res_ms  = dt_res_s * 1e3 if dt_res_s is not None else None
        msg = f"  Diaphragm burst: tindx={burst_tindx}, t = {t_burst_ms:.3f} ms"
        if dt_res_ms is not None:
            msg += f" ±{dt_res_ms:.4f} ms (dt_plot resolution)"
        print(msg)

    burst_label = _burst_legend_label(t_burst_ms, dt_res_ms)

    plot_pressure_histories(job_subdir, run_dir, t_burst_ms, burst_label, suffix)
    plot_piston_kinematics(piston_file, tindx_to_t, run_dir, t_burst_ms, burst_label, suffix)
    plot_driver_pressure(job_subdir, run_dir, p_burst, t_burst_s, burst_label, suffix)
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "input_path", nargs="?", default=str(DEFAULT_OUTPUT_ROOT),
        help=("Either a single run directory, OR a parent directory "
              "containing condition_* subdirectories (batch mode).  "
              f"Default: {DEFAULT_OUTPUT_ROOT}"),
    )
    p.add_argument(
        "--suffix", default="_replot",
        help="Suffix added to output PNG names so existing plots are not "
             "overwritten (default: _replot)",
    )
    args = p.parse_args(argv)

    input_path = Path(args.input_path).resolve()
    if not input_path.exists():
        print(f"ERROR: {input_path} does not exist", file=sys.stderr)
        return 1

    # Discovery: if the input dir contains condition_* subdirs, batch them.
    condition_dirs = sorted(
        d for d in input_path.glob("condition_*") if d.is_dir()
    )

    if condition_dirs:
        print(f"Batch mode: {len(condition_dirs)} condition dirs under {input_path}")
        print(f"Output suffix: '{args.suffix}'\n")
        n_ok = 0
        for cdir in condition_dirs:
            print(f"┌── {cdir.name} " + "─" * (60 - len(cdir.name)))
            if process_run(cdir, args.suffix):
                n_ok += 1
            print()
        print(f"Done.  {n_ok}/{len(condition_dirs)} runs processed.")
    else:
        print(f"Single-run mode: {input_path}")
        print(f"Output suffix: '{args.suffix}'\n")
        ok = process_run(input_path, args.suffix)
        if not ok:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

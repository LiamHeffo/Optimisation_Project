"""
Canonical t_hold computation for the X2 driver optimisation.

This module is the single source of truth for the two functions that
characterise the driver-pressure hold window:

  smooth_pressure   -- Savitzky-Golay low-pass filter (filter-then-decide).
  compute_t_hold    -- two-phase settling + dwell t_hold algorithm.

Both the production parser (problem.l1d_job.parse_l1d_outputs) and the
workshop replotter (L1d_Outputs/replot_l1d_outputs.py) import from here,
so the optimiser and the diagnostic plots always agree.

If you re-tune any default below, the change propagates to both
consumers automatically -- which is the point of putting it here.

Defaults reflect the values that were workshopped and validated against
a sweep of historical condition_N runs:

    band_frac        = 0.10     (±10 % band, X2 spec)
    entry_window_s   = 1.0e-3   (1 ms settling tolerance)
    smooth_window_s  = 5.0e-4   (~50 samples at dt_plot = 10 µs)
    smooth_polyorder = 3        (preserves peaks + gradients)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


# Canonical defaults -- the parameters compute_t_hold uses when callers
# don't override.  Re-tune here, both prod and workshop pick it up.
DEFAULT_BAND_FRAC        = 0.10
DEFAULT_ENTRY_WINDOW_S   = 1.0e-3
DEFAULT_SMOOTH_WINDOW_S  = 5.0e-4
DEFAULT_SMOOTH_POLYORDER = 3


def smooth_pressure(
    t: np.ndarray,
    p: np.ndarray,
    window_s: float | None,
    polyorder: int = DEFAULT_SMOOTH_POLYORDER,
) -> np.ndarray:
    """
    Low-pass smooth a pressure trace with a Savitzky-Golay filter.

    Parameters
    ----------
    t         : time array (s) -- used only to derive the sample dt
    p         : raw pressure array (Pa)
    window_s  : smoothing window in SECONDS (not samples).  If None, the
                input is returned unchanged.  Internally converted to an
                odd integer number of samples using dt = median(diff(t)).
    polyorder : Savitzky-Golay polynomial order.  3 preserves peaks and
                gradients well; 2 behaves more like a moving average.

    Why a *seconds* window rather than a samples window?  Trace dt depends
    on the L1d dt_plot setting, which changes between runs.  Specifying
    the filter scale physically (time) makes the smoothing portable.
    """
    if window_s is None or len(p) < polyorder + 3:
        return p
    dt = float(np.median(np.diff(t)))
    if dt <= 0.0:
        return p
    n = int(round(window_s / dt))
    if n < polyorder + 2:
        return p   # window too narrow to be meaningful
    if n % 2 == 0:
        n += 1     # savgol_filter requires an odd window length
    if n > len(p):
        n = len(p) if len(p) % 2 == 1 else len(p) - 1
    return savgol_filter(p, window_length=n, polyorder=polyorder)


def compute_t_hold(
    t: np.ndarray,
    p: np.ndarray,
    p_burst: float,
    t_burst: float,
    band_frac: float = DEFAULT_BAND_FRAC,
    entry_window_s: float = DEFAULT_ENTRY_WINDOW_S,
    smooth_window_s: float | None = DEFAULT_SMOOTH_WINDOW_S,
    smooth_polyorder: int = DEFAULT_SMOOTH_POLYORDER,
) -> tuple[float, float, float]:
    """
    Compute t_hold from a driver-side pressure trace, with low-pass
    smoothing and a "delayed-start" provision for traces that are still
    settling at the burst instant.

    Returns
    -------
    (t_hold, t_start, t_exit)
        t_start = anchor for the hold window.  Equals t_burst when the
                  trace is already in band at burst; equals the entry
                  time when delayed-start triggers; equals t_burst
                  (with t_hold = 0.0) when no valid hold is established.
        t_exit  = first time after t_start that the trace leaves the band,
                  or t[-1] if it never does.
        t_hold  = t_exit - t_start.

    Algorithm
    ---------
    Phase 0 -- Savitzky-Golay smoothing of p (if smooth_window_s is set).
    Phase 1 -- pick t_start:
        if p_smooth(t_burst) < p_lo  (still ramping up):
            search forward for the first sample where p enters the band.
            if no such sample exists  ->  t_hold = 0
            if t_entry - t_burst > entry_window_s  ->  t_hold = 0
            else  ->  t_start = t_entry
        else (already in band, or above):
            t_start = t_burst   # preserve naive first-exit semantics
    Phase 2 -- find first exit after t_start.

    Why "below" only?  Physically the driver pressure ramps UP toward
    p_burst, so the realistic settling case is below-band-at-burst.
    The above-band branch keeps the naive first-exit behaviour intact
    so the new code is a superset of the original.

    TODO -- variants worth exploring if traces remain noisy:
      (a) Low-pass smoothing (Savitzky-Golay) -- IMPLEMENTED.
      (b) Hysteresis: require N consecutive out-of-band samples
          before counting as exit (rejects single-sample spikes).
      (c) Longest contiguous in-band window instead of first-exit.
      (d) Quantile-centred band: centre on np.median of an in-window
          slice rather than the strict p_burst setpoint.
      (e) Slope-aware exit: only count exits with a sustained
          negative slope (genuine decay vs. wave-driven dip).
    """
    # Phase 0 -- optional smoothing.
    p = smooth_pressure(t, p, smooth_window_s, smooth_polyorder)

    p_lo = p_burst * (1.0 - band_frac)
    p_hi = p_burst * (1.0 + band_frac)

    # Locate burst in the sample grid.
    burst_idx = int(np.searchsorted(t, t_burst, side="left"))
    if burst_idx >= len(t):
        return 0.0, t_burst, t_burst   # burst beyond end of trace

    # Phase 1 -- pick t_start.
    if p[burst_idx] < p_lo:
        in_band = (p >= p_lo) & (p <= p_hi)
        post_burst = t >= t_burst
        entry_idxs = np.where(post_burst & in_band)[0]
        if entry_idxs.size == 0:
            return 0.0, t_burst, t_burst   # never entered band

        t_entry = float(t[entry_idxs[0]])
        if (t_entry - t_burst) > entry_window_s:
            return 0.0, t_burst, t_burst   # entry too late

        t_start = t_entry
    else:
        t_start = float(t_burst)

    # Phase 2 -- first exit after t_start.
    post_start = t > t_start
    out_of_band = (p < p_lo) | (p > p_hi)
    exit_idxs = np.where(post_start & out_of_band)[0]
    t_exit = float(t[exit_idxs[0]]) if exit_idxs.size else float(t[-1])
    t_hold = t_exit - t_start
    return t_hold, t_start, t_exit

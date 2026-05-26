"""
Volume-conserving geometry construction for the X2 driver in L1d4.

Implements the methodology from Hodson 2025 §4.3.1: every discontinuous
area change in the simplified-cylindrical X2 driver model is replaced by
a gradual linear ramp whose endpoints are placed to preserve the internal
volume of the cylinder either side.  The single helper
:func:`volume_conserving_ramp` is the executable form of Eq 4.12–4.15.

:func:`build_break_points` walks the X2 transition table once per
individual and emits the (x, D) pairs that the L1d job script turns into
``add_break_point`` calls.  ``buffer_length`` and ``D_throat`` are the
only design variables that affect geometry; everything else is a facility
constant.

Reference data (Table 4.5, no buffer studs, no orifice plate) is
reproduced by ``build_break_points(buffer_length=0.0, D_throat=None)``;
this is asserted by the smoke test at the bottom of the file.
"""
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Facility constants (Hodson Table 4.4 / 4.5)
# ─────────────────────────────────────────────────────────────────────────────

# Reservoir-side area transitions: (x_nominal, R_upstream, R_downstream).
# All are reductions (downstream is smaller), all use V_buffer = 0.
RESERVOIR_GRADIENTS = [
    (-5.8007, 0.158,   0.122),
    (-5.1817, 0.122,   0.090),
    (-5.1407, 0.090,   0.07805),
]

# Buffer-plate gradient: the only transition affected by V_buffer.
BUFFER_PLATE_X_NOMINAL = -0.1120
R_COMPRESSION = 0.2568 / 2   # = 0.1284
R_SHOCK       = 0.0850 / 2   # = 0.0425

# Fixed anchors with no gradient applied.  The launcher exit anchors
# (-4.8332, D=0.1561) and (-4.808, D=0.2568) bracket a pre-baked 25.2 mm
# ramp from Eq 4.9 that L1d realises automatically via its piecewise-
# linear interpretation of consecutive break-points.
RESERVOIR_START_ANCHOR = (-8.7188, 0.316)
LAUNCHER_SMALL_D       = (-4.8332, 0.1561)
LAUNCHER_LARGE_D       = (-4.808,  0.2568)   # = piston rear initial position
PISTON_FRONT_X         = -4.587
PD_X                   = 0.0

# Buffer-stud parameters (X2 standard six-stud configuration).
N_BUFFER_STUDS = 6
D_BUFFER_STUD  = 0.050

# Hodson methodology constant.
M_ASPECT = 2.0

# Provisional orifice-plate parameters.  L_ORIFICE is the axial extent of
# the constant-D_throat section between the two flanking gradients.
L_ORIFICE = 0.010

# Minimal-model shock-tube extent past the primary diaphragm.
L_SHOCK_TUBE_END = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Volume-conserving gradient (Hodson Eq 4.12 – 4.15)
# ─────────────────────────────────────────────────────────────────────────────

def volume_conserving_ramp(R, r, x_nominal, direction, m=M_ASPECT, V_buffer=0.0):
    """One area transition.

    Parameters
    ----------
    R, r : float
        Radii either side of the transition.  ``R > r`` must hold.
    x_nominal : float
        Sharp-discontinuity x-coordinate from the cylindrical (Table 4.4)
        layout.
    direction : int
        ``+1`` for an expansion in the downstream direction (small-D
        upstream, large-D downstream); ``-1`` for a reduction.
    m : float
        Aspect ratio of the gradient (radius:length).  Hodson uses 2.
    V_buffer : float
        Volume to subtract from the upstream cylinder to account for
        buffer studs.  Zero for every transition except the buffer-plate.

    Returns
    -------
    (x_R, x_r) : tuple of float
        x-coordinates of the large-D and small-D endpoints of the ramp.
        For ``direction == -1`` ``x_R < x_r`` (large-D upstream); for
        ``direction == +1`` ``x_r < x_R`` (small-D upstream).
    """
    if R <= r:
        raise ValueError(f"volume_conserving_ramp expects R > r; got R={R}, r={r}")
    m_signed = m * direction
    a = (2 * (R**3 - r**3 + 3 * m_signed * V_buffer / (2 * np.pi))
         ) / (3 * (R**2 - r**2))
    x_R = (R - a) / m_signed + x_nominal
    x_r = (r - a) / m_signed + x_nominal
    return x_R, x_r


def buffer_stud_volume(buffer_length):
    """Total displaced volume of the six X2 buffer studs (Eq 4.13).

    ``buffer_length`` ≤ 0 disables the studs (V_buffer = 0).
    """
    if buffer_length <= 0:
        return 0.0
    return N_BUFFER_STUDS * 0.25 * np.pi * buffer_length * D_BUFFER_STUD**2


# ─────────────────────────────────────────────────────────────────────────────
# Per-individual break-point assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_break_points(buffer_length, D_throat=None):
    """Return ``(break_points, derived)`` for one individual.

    Parameters
    ----------
    buffer_length : float
        Buffer-stud length in metres.  Zero or negative disables the
        studs (V_buffer = 0); otherwise V_buffer feeds Eq 4.12 for the
        buffer-plate transition only.
    D_throat : float or None
        Orifice-plate bore diameter in metres.  ``None`` means no orifice
        plate (the basic Hodson Table 4.5 configuration).

    Returns
    -------
    break_points : list of (x, D) tuples
        In ascending-x order, ready to be emitted as L1d
        ``add_break_point`` calls.
    derived : dict
        Computed positions the job-script template and the parser need
        (piston anchors, primary-diaphragm x, buffer-plate ramp endpoints,
        orifice centre when present).
    """
    V_b = buffer_stud_volume(buffer_length)
    bps = [RESERVOIR_START_ANCHOR]

    # Three reservoir-side gradients (all reductions, all V_buffer = 0).
    for x_nom, R_up, R_dn in RESERVOIR_GRADIENTS:
        x_R, x_r = volume_conserving_ramp(R_up, R_dn, x_nom, direction=-1)
        bps.append((x_R, 2 * R_up))
        bps.append((x_r, 2 * R_dn))

    # Launcher anchors — diameter ramp between them is interpolated by L1d.
    bps.append(LAUNCHER_SMALL_D)
    bps.append(LAUNCHER_LARGE_D)
    bps.append((PISTON_FRONT_X, 2 * R_COMPRESSION))

    # Buffer-plate gradient — the only stud-affected transition.
    x_R_T5, x_r_T5 = volume_conserving_ramp(
        R_COMPRESSION, R_SHOCK, BUFFER_PLATE_X_NOMINAL,
        direction=-1, V_buffer=V_b,
    )
    bps.append((x_R_T5, 2 * R_COMPRESSION))
    bps.append((x_r_T5, 2 * R_SHOCK))

    derived = {
        "piston_xL0":      LAUNCHER_LARGE_D[0],
        "piston_xR0":      PISTON_FRONT_X,
        "pd_x":            PD_X,
        "x_outer_buffer":  x_R_T5,
        "x_inner_buffer":  x_r_T5,
    }

    # Orifice plate, if a D_throat is supplied and is strictly smaller
    # than the shock-tube diameter.  At D_throat == 2*R_SHOCK the orifice
    # is degenerate (no area change), so we skip the gradient calls and
    # treat the design as if no orifice were present.
    if D_throat is not None and D_throat < 2 * R_SHOCK:
        x_orifice_centre = x_r_T5 / 2.0          # midpoint(x_inner_buffer, PD_X=0)
        x_T6_nom = x_orifice_centre - L_ORIFICE / 2.0
        x_T7_nom = x_orifice_centre + L_ORIFICE / 2.0
        R_throat = D_throat / 2.0

        # Upstream gradient (reduction shock-tube → orifice).
        x_R_T6, x_r_T6 = volume_conserving_ramp(
            R_SHOCK, R_throat, x_T6_nom, direction=-1,
        )
        # Downstream gradient (expansion orifice → shock-tube).
        x_R_T7, x_r_T7 = volume_conserving_ramp(
            R_SHOCK, R_throat, x_T7_nom, direction=+1,
        )
        if x_r_T7 <= x_r_T6:
            raise ValueError(
                f"Orifice geometry infeasible at D_throat={D_throat:.4f} m, "
                f"L_orifice={L_ORIFICE:.4f} m: gradients would overlap "
                f"(x_r_T6={x_r_T6:.6f}, x_r_T7={x_r_T7:.6f})."
            )
        bps.append((x_R_T6, 2 * R_SHOCK))
        bps.append((x_r_T6, D_throat))
        bps.append((x_r_T7, D_throat))
        bps.append((x_R_T7, 2 * R_SHOCK))
        derived["x_orifice_centre"] = x_orifice_centre

    bps.append((PD_X,             2 * R_SHOCK))
    bps.append((L_SHOCK_TUBE_END, 2 * R_SHOCK))
    bps.sort(key=lambda xd: xd[0])
    return bps, derived


# ─────────────────────────────────────────────────────────────────────────────
# Self-test — reproduces Table 4.5 (basic config, no studs, no orifice)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bps, derived = build_break_points(buffer_length=0.0, D_throat=None)

    # Hodson Table 4.5, rounded to four decimals as in the paper.
    expected = [
        (-8.7188, 0.3160),
        (-5.8093, 0.3160),
        (-5.7913, 0.2440),
        (-5.1893, 0.2440),
        (-5.1733, 0.1800),
        (-5.1436, 0.1800),
        (-5.1376, 0.1561),
        (-4.8332, 0.1561),
        (-4.8080, 0.2568),
        (-4.5870, 0.2568),
        (-0.1299, 0.2568),
        (-0.0869, 0.0850),
        ( 0.0000, 0.0850),
        ( 5.0000, 0.0850),
    ]

    print(f"Generated {len(bps)} break-points; expected {len(expected)}.")
    if len(bps) != len(expected):
        raise SystemExit(f"FAIL: length mismatch")

    for i, ((x_got, D_got), (x_exp, D_exp)) in enumerate(zip(bps, expected)):
        match = abs(x_got - x_exp) < 5e-4 and abs(D_got - D_exp) < 1e-4
        flag = "OK" if match else "FAIL"
        print(f"  [{i:2d}] x={x_got:+8.4f} (exp {x_exp:+8.4f})  "
              f"D={D_got:.4f} (exp {D_exp:.4f})  {flag}")
        if not match:
            raise SystemExit("Table 4.5 mismatch — methodology not reproduced")

    # Orifice sanity check across the D_throat bound.  At the upper
    # bound (D_throat == 2*R_SHOCK) the orifice is degenerate and is
    # skipped; at smaller D_throat the orifice break-points are added.
    for D_th in (0.085, 0.07, 0.05):
        bps_o, derived_o = build_break_points(0.10, D_throat=D_th)
        has_orifice = "x_orifice_centre" in derived_o
        print(f"\nOrifice config: D_throat={D_th:.3f} m, buffer_length=0.10 m")
        print(f"  total break-points = {len(bps_o)}  (orifice inserted: {has_orifice})")
        print(f"  x_inner_buffer     = {derived_o['x_inner_buffer']:+.5f}")
        if has_orifice:
            print(f"  x_orifice_centre   = {derived_o['x_orifice_centre']:+.5f}")

    print("\nAll smoke tests passed.")

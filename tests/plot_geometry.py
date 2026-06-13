"""Render the X2 driver geometry for a chosen (buffer_length, D_throat).

Usage examples:

    # Default — 0.10 m buffer studs, 0.075 m orifice bore.
    python tests/plot_geometry.py

    # No orifice plate (D_throat ≥ 2*R_SHOCK is treated as degenerate).
    python tests/plot_geometry.py --buffer-length 0.10 --D-throat 0.085

    # No buffer studs, no orifice — reproduces Hodson Table 4.5 geometry.
    python tests/plot_geometry.py --buffer-length 0.0 --D-throat 0.085

The figure is written to src/L1d_Outputs/geometry_check.png by default.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from problem.l1d_geometry import plot_geometry


def main():
    p = argparse.ArgumentParser(description="Render the X2 driver geometry.")
    p.add_argument("--buffer-length", "-b", type=float, default=0.10,
                   help="Buffer stud length (m); 0 disables studs. Default 0.10.")
    p.add_argument("--D-throat", "-d", type=float, default=0.075,
                   help="Orifice bore diameter (m); pass >= 0.085 to disable. "
                        "Default 0.075.")
    p.add_argument("--out", "-o",
                   default="src/L1d_Outputs/geometry_check.png",
                   help="Output PNG path (default src/L1d_Outputs/geometry_check.png).")
    args = p.parse_args()

    D_throat = args.D_throat
    if D_throat >= 0.085:
        # Above the shock-tube diameter the orifice branch is skipped in
        # build_break_points; pass None so the title reflects that clearly.
        D_throat = None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_geometry(
        buffer_length=args.buffer_length,
        D_throat=D_throat,
        save_path=str(out_path),
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

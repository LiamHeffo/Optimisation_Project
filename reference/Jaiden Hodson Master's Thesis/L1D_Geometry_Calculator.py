# Jaiden Hodson 24/04/2025
# L1D4 Volume Preserving Geometry Calculator
import numpy as np

# general parameters
D = 0.2568 # m, outer diameter
d = 0.0850 # m, inner diameter
R = D/2 # m, outer radius
r = d/2 # m, inner radius
x_profile_change = -0.112 # m
direction = -1 # 1 for area expansion relative to downstream direction, -1 for area reduction
ar = 2 # aspect ratio of profile change (higher = sharper)
m = ar * direction

# buffer parameters
buffer_toggle = 1 # 1 with buffers, 0 without buffers
L_b = 0.100 # m, buffer stud length
D_b = 0.050 # m, buffer stud diameter
n_b = 6 # number of buffer studs
V_b = buffer_toggle*n_b*0.25*np.pi*L_b*(D_b**2) # total buffer volume

# calcs
intercept = (2*(R**3 - r**3 + ((3*m*V_b)/(2*np.pi)) ))/(3*(R**2 - r**2))
x_0_R = (R - intercept)/m # sol's @ x=0
x_0_r = (r - intercept)/m
x_R = x_0_R + x_profile_change # sol's @ profile change
x_r = x_0_r + x_profile_change

# solution
print("Generated Breakpoints:")
if direction == 1:
    print(f"add_break_point({round(x_r, 6)}, {d})")
    print(f"add_break_point({round(x_R, 6)}, {D})")
elif direction == -1:
    print(f"add_break_point({round(x_R, 6)}, {D})")
    print(f"add_break_point({round(x_r, 6)}, {d})")


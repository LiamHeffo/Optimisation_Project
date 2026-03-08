# jaiden hodson 7/02/2025
# imports
import numpy as np

# variables
buffer_D = 0.05 # m
buffer_L = 0.140 # m
piston_m = 10.50 # kg

# other important parameters
V_i = 0.25*np.pi*buffer_L*buffer_D**2
n = 6 # number of studs
comp_tube_D = 0.2568 # m
buffer_mount_pitch_D = 0.1705 # m
YS = 40e6 # Pa

# calcuting maximum permissible piston impact velocity
# wall interference condition
D1 = comp_tube_D - buffer_mount_pitch_D
L1 = (4*V_i)/(np.pi*(D1**2))

# adjacent buffer interference condition
D2 = buffer_mount_pitch_D*np.sin(np.pi/n)
L2 = (4*V_i)/(np.pi*(D2**2))

# determining maximum length reduction at buffer fracture
L = [L1, L2]
L_at_max_deformation = max(L)
x_max = buffer_L - L_at_max_deformation

# function that returns max permissible impact velocity from max plastic deformation
def vel_impact_max(x):
    v = np.sqrt(-n*YS*((np.pi*buffer_D**2)/(2*piston_m))*buffer_L*(np.log((buffer_L-x)/buffer_L)))
    return v

# checking if buckling is a possible buffer failure mode via slenderness ratio
Rg = np.sqrt(((np.pi*buffer_D**4)/64)/((np.pi*buffer_D**2)/4))
L_effective = buffer_L*2.0 # 1.0 p-p, 0.7 p-f, 0.5 f-f, 2.0 f-free
SR = L_effective/Rg

# printing results
if SR < 40.0:
    print("Buckling Failure:", "Short Column - No")
else:
    print("Buckling Failure:", "Probable")
print("Slenderness Ratio =", round(SR, 3))
print("Max. Impact Velocity =", round(vel_impact_max(x_max), 3), "m/s")
print("Buffer Length @ Fracture =", round(L_at_max_deformation*1000, 3), "mm")
print("Max. Buffer Strain =", -1*round(100*x_max/buffer_L, 3), "%")

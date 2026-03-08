# File: x2_cylinders.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'

# centreline
x = np.array([-9,1])
y = np.array([0,0])

# piston
x_piston = np.array([-4.808,-4.587,-4.587,-4.808,-4.808])
y_piston = np.array([0.1284,0.1284,-0.1284,-0.1284,0.1284])

# PD
x_pd = np.array([0,0])
y_pd = np.array([0.0425,-0.0425])

# tunnel
x_t = np.array([1,-0.0869,-0.1299,-4.808,-4.8332,-5.1376,-5.1436,-5.1733,-5.1893,
                -5.7913,-5.8093,-8.7188,-8.7188,-5.8093,-5.7913,
                -5.1893,-5.1733,-5.1436,-5.1376,-4.8332,-4.808,-0.1299,-0.0869,1])
y_t = np.array([0.0425,0.0425,0.1284,0.1284,0.07805,0.07805,0.09,0.09,0.122,0.122,0.158,0.158,
                -0.158,-0.158,-0.122,-0.122,-0.09,-0.09,-0.07805,-0.07805,-0.1284,-0.1284,
                -0.0425,-0.0425])

plt.rcParams["figure.figsize"] = (16,6)
plt.rcParams["font.size"]  = 20
#plt.plot(x,y, dashes=[5,3,1,5,1,3], color='k')
plt.plot(x_piston,y_piston, 'k--')
plt.plot(x_pd,y_pd, 'k--')
plt.plot(x_t,y_t, 'k-')
#plt.xlabel('x-location (m)')
#plt.ylabel('Radius (m)')
plt.xlim(-9,0.5)
plt.ylim(-0.3,0.3)
plt.savefig('Breakpoints_Diagram.png', dpi=500)
plt.show()
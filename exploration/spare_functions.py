# Defining the problem
# class ElementwiseTests(ElementwiseProblem):
#     def __init__(self):
#         xl_bounds = {'percent_He' : 0, 'driver_p' : 0, 'p4' : 0, 'D_throat' : 0, 'reservoir_p':0}
#         xu_bounds = {'percent_He' : 100, 'driver_p' : 40e6, 'p4' : 40e6, 'D_throat' : 0.085, 'reservoir_p': 40e6}
#         xl = np.zeros(5)
#         xu = np.zeros(5)
#
#         i = 0
#         for var in xl_bounds:
#             xl[i] = xl_bounds[var]
#             xu[i] = xu_bounds[var]
#             i += 1
#
#
#         super().__init__(n_var=5, n_obj=1, n_constr=0, xl=xl, xu=xu)
#
#         def _evaluate(self, x, out):
#             driver_dict = {'percent_He' : x[0], 'driver_p' : x[1], 'p4' : x[2], 'D_throat' : x[3], 'reservoir_p': x[4]}
#             print(driver_dict)
#             out['F'] = -1*objective_function(driver_dict)
#             print(out['F'])

import numpy as np

def instantiate_individual(Pop_size, Pmax, D_star_max):
    return

def finite_difference_calculation(func, var, x0):
    dx = x0[var]*0.00001

    xp1 = x0.copy()
    xp1[var] = x0[var] + dx
    yp1 = func(x0)

    xm1 = x0.copy()
    xm1[var] = x0[var] - dx
    ym1 = func(xm1)

    y = [ym1, yp1]
    y_diff = np.diff(np.array(y))

    return y_diff/(2*dx)

P = np.linspace(1, 10, 10)
for index, element in enumerate(P, start=3):
    print('index = ', index)
    print('element = ', element)

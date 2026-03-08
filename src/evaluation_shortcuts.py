from deap import creator
from DEAP_tests import evaluate, variable_transformation, unnormalise_fitness

he_lower, he_upper = 70, 100
D_throat_lower, D_throat_upper = 0.05, 0.085
driver_p_lower, driver_p_upper = 1000, (40/14.62)*1e6
p4_lower, p4_upper = 150e3, 40e6
reservoir_lower, reservoir_upper = 1000, 8e6

bounds = [(he_lower, he_upper), (driver_p_lower, driver_p_upper),
            (p4_lower, p4_upper), (D_throat_lower, D_throat_upper), (reservoir_lower, reservoir_upper)]

APPROX_IDEAL = (0, 0.01)
APPROX_NADIR = (3500, 0)

# x = [(80, 0.077e6, 35.7e6, 0.0849, 6.08e6)]
x = [(99.9, 0.3285e6, 29.2e6, 0.0517, 7.61e6)]
# x = [(100, 0.3498e6, 29.87e6, 0.052, 7.95e6)]
# x = [(100, 0.3035e6, 28.37e6, 0.0558, 7.36e6)]
# x = [(100, 0.3164e6, 28.62e6, 0.0506, 7.12e6)]
# x = [(100, 0.3480e6, 29.86e6, 0.0518, 7.92e6)]
# x = [(100, 0.3686e6, 28.78e6, 0.0508, 7.85e6)]


x_transformed = variable_transformation(x, bounds)
candidate = creator.Individual(x_transformed[0]) 

candidate.sim_type = 'ParentValue'
candidate.normalised = True

print(unnormalise_fitness(evaluate(candidate), APPROX_IDEAL, APPROX_NADIR))
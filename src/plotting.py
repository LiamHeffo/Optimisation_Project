"""
Objective-space scatter plots for the Pareto front history.

plot_objective_space(fitness_history, obj1, obj2, **kwargs)

Supported objective-pair combinations:
    ('delta_vs1', 'hold_time')
    ('delta_vs1', 'impact_speed')
    ('hold_time',  'impact_speed')

The last MU entries in fitness_history are treated as the current Pareto
front and highlighted in a distinct colour.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_objective_space(fitness_history, obj1, obj2, **kwargs):
    """Scatter-plot the objective-space history and highlight the current front.

    Parameters
    ----------
    fitness_history : list of tuples
        All evaluated fitness values (normalised or physical).
    obj1, obj2 : str
        Names of the two objectives to plot (see module docstring).
    MU : int, optional
        Population size — the last MU entries are the current front (default 10).
    sim_type : str, optional
        Used in the output filename (default 'Parent_Value').
    gen : int, optional
        Generation number for the filename (default 0).
    normalised : bool, optional
        If True, axes are clamped to [0, 1.1] (default True).
    """
    MU         = kwargs.get('MU',         10)
    sim_type   = kwargs.get('sim_type',   'Parent_Value')
    gen        = kwargs.get('gen',         0)
    normalised = kwargs.get('normalised',  True)

    if obj1 == 'delta_vs1' and obj2 == 'hold_time':
        plt.figure(dpi=800)
        plt.title("Pareto Frontier")
        plt.xlabel("Normalised Residual of Shock Speed")
        plt.ylabel("Normalised Hold Time")

        delta_vs_history   = [entry[0] for entry in fitness_history]
        hold_time_history  = [entry[1] for entry in fitness_history]

        if normalised:
            plt.xlim((0, 1.1))
            plt.ylim((0, 1.1))
        else:
            plt.ylim((-0.005, 2 * np.max(hold_time_history)))

        plt.scatter(delta_vs_history,       hold_time_history,       facecolors='none', edgecolors='lightblue')
        plt.scatter(delta_vs_history[-MU:], hold_time_history[-MU:], facecolors='none', edgecolors='green')

        plt.savefig(f"cma_es_mo_fpd_{sim_type}_1_{gen}.png")
        plt.close()

    elif obj1 == 'delta_vs1' and obj2 == 'impact_speed':
        plt.figure(dpi=800)
        plt.title("Pareto Frontier")
        plt.xlabel("Normalised Residual of Shock Speed")
        plt.ylabel("Normalised Impact Speed")

        delta_vs_history      = [entry[0] for entry in fitness_history]
        impact_speed_history  = [entry[2] for entry in fitness_history]

        if normalised:
            plt.xlim((0, 1.1))
            plt.ylim((0, 1.1))
        else:
            plt.ylim((-0.005, 2 * np.max(impact_speed_history)))

        plt.scatter(delta_vs_history,       impact_speed_history,       facecolors='none', edgecolors='lightblue')
        plt.scatter(delta_vs_history[-MU:], impact_speed_history[-MU:], facecolors='none', edgecolors='orange')

        plt.savefig(f"cma_es_mo_fpd_{sim_type}_2_{gen}.png")
        plt.close()

    elif obj1 == 'hold_time' and obj2 == 'impact_speed':
        plt.figure(dpi=800)
        plt.title("Pareto Frontier")
        plt.xlabel("Normalised Hold Time")
        plt.ylabel("Normalised Impact Speed")

        hold_time_history     = [entry[1] for entry in fitness_history]
        impact_speed_history  = [entry[2] for entry in fitness_history]

        if normalised:
            plt.xlim((0, 1.1))
            plt.ylim((0, 1.1))
        else:
            plt.ylim((-0.005, 2 * np.max(impact_speed_history)))

        plt.scatter(hold_time_history,       impact_speed_history,       facecolors='none', edgecolors='lightblue')
        plt.scatter(hold_time_history[-MU:], impact_speed_history[-MU:], facecolors='none', edgecolors='purple')

        plt.savefig(f"cma_es_mo_fpd_{sim_type}_3_{gen}.png")
        plt.close()

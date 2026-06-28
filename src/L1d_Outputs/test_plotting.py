import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import numpy as np

HERE     = Path(__file__).resolve().parent        # the script's own directory
DATA_DIR = HERE / "DEAP_0" / "DEAP_0"             # grandchild dir
DATA     = DATA_DIR / "history-loc-0002.data"

arr = np.loadtxt(DATA, comments="#", unpack=True)

t = arr[0]
p = arr[4]

plt.plot(t, p)
plt.xlabel("Time [s]")
plt.ylabel("Pressure [Pa]")
plt.title("Pressure vs Time")
plt.grid()
plt.xlim(20e-3, 35e-3)
plt.show()
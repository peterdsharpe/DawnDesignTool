import os, sys

sys.path.insert(0, os.path.abspath(".."))
from design_opt import *

param_range = np.linspace(1, 1.169, 30)
outputs = []
s_opts["max_iter"] = 1000

for i, param in enumerate(param_range):
    print(f"{'-' * 50}\nIteration {i + 1} of {len(param_range)}\n{'-' * 50}")
    opti.set_value(wing_drag_multiplier, param)
    try:
        sol = opti.solve()
        opti.set_initial(sol.value_variables())
        opti.set_initial(opti.lam_g, sol.value(opti.lam_g))
        outputs.append(sol.value(wing_span))
    except:
        outputs.append(np.NaN)
outputs = np.array(outputs)

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

sns.set(font_scale=1)

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.plot(100 * (param_range - 1), outputs, ".-")
plt.axvline(x=0, ls='--', color="gray")
plt.text(
    x=0,
    y=0.95*ax.get_ylim()[1]+(1-0.95)*ax.get_ylim()[0],
    s="Previous Baseline",
    color="gray",
    horizontalalignment='right',
    verticalalignment='top',
    rotation=90
)
plt.axvline(x=16.9, ls='--', color="gray")
plt.text(
    x=16.9,
    y=0.05*ax.get_ylim()[1]+(1-0.05)*ax.get_ylim()[0],
    s="Conservative Worst-Case",
    color="gray",
    horizontalalignment='right',
    verticalalignment='bottom',
    rotation=90
)
plt.axvline(x=16.9 * 0.75, ls='--', color="gray")
plt.text(
    x=16.9 * 0.75,
    y=0.05*ax.get_ylim()[1]+(1-0.05)*ax.get_ylim()[0],
    s="New Baseline?",
    color="gray",
    horizontalalignment='right',
    verticalalignment='bottom',
    rotation=90
)


plt.xlabel(r"Increase in Wing Drag over Baseline [%]")
plt.ylabel(r"Wing Span [m]")
plt.title(r"Effect of Tripped Boundary Layers on Vehicle Sizing")
plt.tight_layout()
plt.legend()
plt.show()

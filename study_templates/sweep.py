import os, sys

sys.path.insert(0, os.path.abspath(".."))
from design_opt import *

param_range = np.linspace(0, 0.25, 26)
outputs = []
s_opts["max_iter"] = 1000

for i, param in enumerate(param_range):
    print(f"{'-' * 50}\nIteration {i + 1} of {len(param_range)}\n{'-' * 50}")
    opti.set_value(max_oxamide, param)
    try:
        sol = opti.solve()
        # opti.set_initial(sol.value_variables())
        # opti.set_initial(opti.lam_g, sol.value(opti.lam_g))
        outputs.append(sol.value(boost_range))
    except:
        outputs.append(np.NaN)
outputs = np.array(outputs)

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

sns.set(font_scale=1)

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
goodpoint = np.logical_not(np.logical_or.reduce([param_range == 0.12, param_range == 0.13, param_range == 0.18]))
plt.plot(param_range[goodpoint], outputs[goodpoint] / 1e3, ".-")
plt.xlabel(r"Max Allowable Oxamide Fraction")
plt.ylabel(r"Boost Range")
plt.title(r"Oxamide Fraction vs. Boost Range")
plt.tight_layout()
plt.legend()
plt.show()

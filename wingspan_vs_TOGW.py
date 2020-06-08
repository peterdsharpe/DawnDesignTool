from design_opt import *

weightings = np.linspace(0, 1, 30)
objective_1 = wing_span
objective_2 = max_mass_total
s_opts["max_iter"] = 1e3  # If you need to interrupt, just use ctrl+c

objective_1_outputs = []
objective_2_outputs = []

for i, weighting in enumerate(weightings):
    print(f"{'-' * 50}\nIteration {i + 1} of {len(weightings)}\n{'-' * 50}")
    objective = (
            (1 - weighting) * objective_1 / 50 +
            (weighting) * objective_2 / 300
    )
    opti.minimize(objective + penalty + 1e-6 * things_to_slightly_minimize)
    try:
        sol = opti.solve()
        # opti.set_initial(sol.value_variables())
        # opti.set_initial(opti.lam_g, sol.value(opti.lam_g))
        objective_1_outputs.append(sol.value(objective_1))
        objective_2_outputs.append(sol.value(objective_2))
    except:
        objective_1_outputs.append(np.NaN)
        objective_2_outputs.append(np.NaN)

objective_1_outputs = np.array(objective_1_outputs)
objective_2_outputs = np.array(objective_2_outputs)

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

sns.set(font_scale=1)

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.plot(objective_1_outputs, objective_2_outputs, ".-")
plt.xlabel(r"Wingspan [m]")
plt.ylabel(r"TOGW [kg]")
plt.title(r"Pareto Plot: Wingspan vs. TOGW")
plt.tight_layout()
plt.legend()
# plt.savefig("C:/Users/User/Downloads/temp.svg")
plt.show()

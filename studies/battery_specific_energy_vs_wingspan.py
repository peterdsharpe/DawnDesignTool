import os, sys
sys.path.append(os.path.abspath("."))
from design_opt import *
import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.express as px
import plotly.graph_objects as go
import dash
import seaborn as sns
sns.set(font_scale=1)


if __name__ == "__main__":

    param_range = np.linspace(600, 250, 60)
    raw_outputs = []

    for param in param_range:
        # opti.set_value(mass_payload, param)
        # opti.set_value(days_to_simulate, param)
        # opti.set_value(structural_mass_margin_multiplier, param)
        opti.set_value(battery_specific_energy_Wh_kg, param)
        try:
            sol = opti.solve()
            opti.set_initial(sol.value_variables())
            opti.set_initial(opti.lam_g, sol.value(opti.lam_g))
            raw_outputs.append(sol.value(wing_span))
        except:
            raw_outputs.append(None)

    outputs = np.array(raw_outputs)

import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.express as px
import plotly.graph_objects as go
import dash
import seaborn as sns
sns.set(font_scale=1)
fig, ax = plt.subplots(1, 1, figsize=(6.4,4.8), dpi=200)
goodpoints = np.array([x is not None for x in raw_outputs])
plt.plot(param_range[goodpoints], outputs[goodpoints], '.-')
# plt.axis("equal")
plt.xlim(325,625)
plt.ylim(25, 75)
plt.xlabel("Battery Specific Energy [Wh/kg]")
plt.ylabel("Wingspan [m]")
plt.title("Influence of Battery Specific Energy on Wingspan, Baseline Mission")
plt.tight_layout()
# plt.legend()
# plt.savefig("C:/Users/User/Downloads/solarsuperlinearityspan.svg")
plt.savefig("C:/Users/User/Downloads/solarsuperlinearityspan.png")
plt.show()

from design_opt import *
import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.express as px
import plotly.graph_objects as go
import dash
import seaborn as sns
sns.set(font_scale=1)


if __name__ == "__main__":

    param_range = np.linspace(300, 600, 30)
    outputs = []

    for param in param_range:
        # opti.set_value(mass_payload, param)
        # opti.set_value(days_to_simulate, param)
        # opti.set_value(structural_mass_margin_multiplier, param)
        opti.set_value(battery_specific_energy_Wh_kg, param)
        try:
            sol = opti.solve()
            opti.set_initial(sol.value_variables())
            outputs.append(sol.value(wing_span))
        except:
            outputs.append(None)

    outputs = np.array(outputs)

    px.line(
        x=param_range,
        y=outputs,

    ).show()

import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.express as px
import plotly.graph_objects as go
import dash
import seaborn as sns
sns.set(font_scale=1)
fig, ax = plt.subplots(1, 1, figsize=(6.4,4.8), dpi=200)
plt.plot(param_range, outputs, '.-')
# plt.axis("equal")
plt.ylim(25, 50)
plt.xlabel("Battery Specific Energy [Wh/kg]")
plt.ylabel("Wingspan [m]")
plt.title("Battery Specific Energy Trade")
plt.tight_layout()
# plt.legend()
# plt.savefig("C:/Users/User/Downloads/solarsuperlinearityspan.svg")
# plt.savefig("C:/Users/User/Downloads/solarsuperlinearityspan.png")
plt.show()

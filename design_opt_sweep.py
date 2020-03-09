from design_opt import *
import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.express as px
import plotly.graph_objects as go
import dash
import seaborn as sns
sns.set(font_scale=1)


if __name__ == "__main__":

    param_range = np.linspace(1, 10, 10)
    outputs = []

    for param in param_range:
        # opti.set_value(battery_specific_energy_Wh_kg, param)
        opti.set_value(days_to_simulate, param)
        try:
            sol = opti.solve()
            opti.set_initial(sol.value_variables())
            outputs.append(sol.value(max_mass_total))
        except:
            outputs.append(None)

    px.line(
        x=param_range,
        y=outputs,

    ).show()
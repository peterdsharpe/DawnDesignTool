from design_opt import *
import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.express as px
import plotly.graph_objects as go
import dash
import seaborn as sns
sns.set(font_scale=1)


if __name__ == "__main__":

    param_range = np.linspace(0, 30, 31)
    # param_range = np.linspace(250, 600, 30)
    # param_range = np.linspace(1, 10, 10)
    outputs = []

    for param in param_range:
        opti.set_value(mass_payload, param)
        # opti.set_value(days_to_simulate, param)
        try:
            sol = opti.solve()
            opti.set_initial(sol.value_variables())
            outputs.append(sol.value(wing.span()))
        except:
            outputs.append(None)

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
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.plot(param_range, outputs, '.-')
plt.xlabel("Payload Mass [kg]")
plt.ylabel("Minimum Possible Wing Span [m]")
plt.title("Solar Airplane: Payload Mass vs. Wing Span")
plt.tight_layout()
plt.legend()
plt.savefig("C:/Users/User/Downloads/solarpayloadspan.png")
plt.show()

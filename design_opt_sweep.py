from design_opt import *
import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.express as px
import plotly.graph_objects as go
import dash
import seaborn as sns
sns.set(font_scale=1)


if __name__ == "__main__":

    # param_range = np.linspace(0, 30, 31)
    # param_range = np.linspace(250, 600, 30)
    # param_range = np.linspace(1, 10, 10)
    param_range = np.linspace(1, 1.3, 10)
    outputs = []

    for param in param_range:
        # opti.set_value(mass_payload, param)
        # opti.set_value(days_to_simulate, param)
        opti.set_value(mass_margin_multiplier, param)
        try:
            sol = opti.solve()
            opti.set_initial(sol.value_variables())
            outputs.append(sol.value(max_mass_total))
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
fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.8), dpi=200)
plt.plot(param_range, outputs/outputs[0], '.-')
plt.axis("equal")
plt.xlabel("Mass Margin Multiplier")
plt.ylabel("Mass / Mass with no margin")
plt.title("Solar Airplane: Mass Margin Superlinearity")
plt.tight_layout()
plt.legend()
plt.savefig("C:/Users/User/Downloads/solarsuperlinearity.svg")
plt.savefig("C:/Users/User/Downloads/solarsuperlinearity.png")
plt.show()

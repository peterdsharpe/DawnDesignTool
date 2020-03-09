from design_opt import *
import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.express as px
import plotly.graph_objects as go
import dash
import seaborn as sns

sns.set(font_scale=1)

if __name__ == "__main__":

    param_range = np.linspace(250, 600, 50)
    outputs = []

    for param in param_range:
        opti.set_value(battery_specific_energy_Wh_kg, param)
        try:
            sol = opti.solve()
            opti.set_initial(sol.value_variables())
            outputs.append(sol.value(wing.span()))
        except:
            outputs.append(None)

    # px.line(
    #     x=param_range,
    #     y=outputs,
    #
    # ).show()

    fig, ax = plt.subplots()
    plt.plot(param_range, outputs, ".-" ,label="Medium Technology Assumptions")
    plt.xlabel("Battery Specific Energy (measured at cell, not pack) [Wh/kg]")
    plt.ylabel("Minimum Possible Wingspan [m]")
    plt.title("Battery Spec. Energy vs. Minimum Wingspan of Solar Airplane")
    plt.legend()
    plt.axvline(x=265, ls='--', color='g')
    plt.text(
        x=265,
        y=0.25 * ax.get_ylim()[1] + (1 - 0.25) * ax.get_ylim()[0],
        s="Aurora Odysseus",
        color='g',
        horizontalalignment='right',
        verticalalignment='center',
        rotation=90
    )
    plt.axvline(x=435, ls='--', color='g')
    plt.text(
        x=435,
        y=0.7 * ax.get_ylim()[1] + (1 - 0.7) * ax.get_ylim()[0],
        s="Airbus Zephyr",
        color='g',
        horizontalalignment='right',
        verticalalignment='center',
        rotation=90
    )

    plt.tight_layout()
    plt.savefig("Battery_vs_span.svg")
    plt.show()

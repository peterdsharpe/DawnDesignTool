from design_opt_solar_dynamic_frozen_to_sweep import *
import plotly.express as px

if __name__ == "__main__":

    param_range = np.linspace(110, 70, 10)
    outputs = []

    for param in param_range:
        opti.set_value(max_span, param)
        try:
            sol = opti.solve()
            opti.set_initial(sol.value_variables())
            outputs.append(sol.value(mass_total))
        except:
            outputs.append(None)

    px.line(
        x=param_range,
        y=outputs,

    ).show()

    outputs = np.array(outputs)
    index = np.argwhere(outputs==np.max(outputs))
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    import plotly.express as px

    style.use("seaborn")

    plt.plot(param_range[:10], outputs[:10], ".-")
    plt.xlabel("Maximum Wing Span [m]")
    plt.ylabel("TOGW [kg]")
    plt.title("Effects of a Limited Wingspan")
    # plt.savefig("C:/Users/User/Downloads/spanlimit.png", dpi=600)
    plt.show()
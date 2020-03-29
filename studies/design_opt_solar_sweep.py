from design_opt_solar import *
import plotly.express as px

if __name__ == "__main__":

    param_range = np.linspace(26, 49, 20)
    outputs = []

    for param in param_range:
        opti.set_value(latitude, param)
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
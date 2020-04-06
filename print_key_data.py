from design_opt import *

try:
    sol = opti.solve()
except:
    sol = opti.debug

if np.abs(sol.value(penalty / objective)) > 0.01:
    print("\nWARNING: high penalty term! P/O = %.3f\n" % sol.value(penalty / objective))

# Find dusk and dawn
try:
    si = sol.value(solar_flux_on_horizontal)
    dusk = np.argwhere(si[:round(len(si) / 2)] < 1)[0, 0]
    dawn = np.argwhere(si[round(len(si) / 2):] > 1)[0, 0] + round(len(si) / 2)
except IndexError:
    print("Could not find dusk and dawn - you likely have a shorter-than-one-day mission.")

# # region Text Postprocessing & Utilities
# ##### Text output
o = lambda x: print(
    "%s: %f" % (x, sol.value(eval(x))))  # A function to Output a scalar variable. Input a variable name as a string
outs = lambda xs: [o(x) for x in xs] and None  # input a list of variable names as strings
print_title = lambda s: print("\n********** %s **********" % s.upper())

print_title("Key Results")
outs([
    "max_mass_total",
    "wing_span",
    "wing_root_chord"
])


def qp(var_name):
    # QuickPlot a variable.
    fig = px.scatter(y=sol.value(eval(var_name)), title=var_name, labels={'y': var_name})
    fig.data[0].update(mode='markers+lines')
    fig.show()


def qp2(x_name, y_name):
    # QuickPlot two variables.
    fig = px.scatter(
        x=sol.value(eval(x_name)),
        y=sol.value(eval(y_name)),
        title="%s vs. %s" % (x_name, y_name),
        labels={'x': x_name, 'y': y_name}
    )
    fig.data[0].update(mode='markers+lines')
    fig.show()


def qp3(x_name, y_name, z_name):
    # QuickPlot two variables.
    fig = px.scatter_3d(
        x=sol.value(eval(x_name)),
        y=sol.value(eval(y_name)),
        z=sol.value(eval(z_name)),
        title="%s vs. %s" % (x_name, y_name),
        labels={'x': x_name, 'y': y_name},
        size_max=18
    )
    fig.data[0].update(mode='markers+lines')
    fig.show()


s = lambda x: sol.value(x)

draw = lambda: airplane.substitute_solution(sol).draw()

# endregion

# Draw mass breakdown
import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.express as px
import plotly.graph_objects as go
import dash
import seaborn as sns

sns.set(font_scale=1)

pie_labels = [
    "Payload",
    "Structural",
    "Propulsion",
    "Power Systems",
    "Avionics"
]
pie_values = [
    sol.value(mass_payload),
    sol.value(mass_structural),
    sol.value(mass_propulsion),
    sol.value(cas.mmax(mass_power_systems)),
    sol.value(mass_avionics),
]
colors = plt.cm.Set2(np.arange(5))
plt.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', colors=colors)
plt.title("Mass Breakdown at Takeoff")
plt.show()

print("\n")
o('wing_span')
o('wing_root_chord')
o('hstab_span')
o('hstab_chord')
o('vstab_span')
o('vstab_chord')
o('max_mass_total')
o('solar_area_fraction')
o('battery_capacity_watt_hours')
o('n_propellers')
o('propeller_diameter')
o('boom_length')
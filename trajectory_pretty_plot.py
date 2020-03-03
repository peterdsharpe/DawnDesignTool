from design_opt_solar import *
import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.express as px
import plotly.graph_objects as go
import dash
style.use("seaborn")

try:
    sol = opti.solve()
except:
    sol = opti.debug

if np.abs(sol.value(penalty / objective)) > 0.01:
    print("\nWARNING: high penalty term! P/O = %.3f\n" % sol.value(penalty / objective))

x = sol.value(x)
y = sol.value(y)
si = sol.value(solar_flux_on_horizontal)
dusk = np.argwhere(si[:round(len(si)/2)] < 1)[0,0]
dawn = np.argwhere(si[round(len(si)/2):] > 1)[0,0]+round(len(si)/2)

plt.plot(x[:dusk], y[:dusk], '.-', color=(103/255,155/255,240/255), label="Day")
plt.plot(x[dawn:], y[dawn:], '.-', color=(103/255,155/255,240/255))
plt.plot(x[dusk-1:dawn+1], y[dusk-1:dawn+1], '.-', color=(7/255,36/255,84/255), label="Night")
plt.legend()
plt.xlabel("Downrange Distance [m]")
plt.ylabel("Altitude [m]")
plt.title("Optimal Trajectory of a Solar-Electric Airplane")
plt.tight_layout()
plt.savefig("optimal_trajectory.svg")
plt.show()
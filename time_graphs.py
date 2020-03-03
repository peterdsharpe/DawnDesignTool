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

import matplotlib.pyplot as plt
import matplotlib.style as style
style.use("seaborn")

plt.figure()
plt.plot(sol.value(time/3600), sol.value(y), ".-")
plt.xlabel("Time after Solar Noon [hours]")
plt.ylabel("Altitude [m]")
plt.title("Altitude over a Day")
plt.tight_layout()
plt.savefig("C:/Users/User/Downloads/altitude.png", dpi=600)
plt.show()

plt.figure()
plt.plot(sol.value(time/3600), sol.value(airspeed), ".-")
plt.xlabel("Time after Solar Noon [hours]")
plt.ylabel("Airspeed [m/s]")
plt.title("Airspeed over a Day")
plt.tight_layout()
plt.savefig("C:/Users/User/Downloads/airspeed.png", dpi=600)
plt.show()

plt.figure()
plt.plot(sol.value(time/3600), sol.value(q), ".-")
plt.xlabel("Time after Solar Noon [hours]")
plt.ylabel("Dynamic Pressure [Pa]")
plt.title("Dynamic Pressure over a Day")
plt.tight_layout()
plt.savefig("C:/Users/User/Downloads/q.png", dpi=600)
plt.show()

plt.figure()
plt.plot(sol.value(time/3600), sol.value(net_power), ".-")
plt.xlabel("Time after Solar Noon [hours]")
plt.ylabel("Net Power [W] (positive is charging)")
plt.title("Net Power over a Day")
plt.tight_layout()
plt.savefig("C:/Users/User/Downloads/net_power.png", dpi=600)
plt.show()

plt.figure()
plt.plot(sol.value(time/3600), 100*sol.value(battery_stored_energy_nondim), ".-")
plt.xlabel("Time after Solar Noon [hours]")
plt.ylabel("Battery Charge %")
plt.title("Battery Charge over a Day")
plt.tight_layout()
plt.savefig("C:/Users/User/Downloads/battery_charge.png", dpi=600)
plt.show()

plt.figure()
plt.plot(sol.value(time/3600), sol.value(wing_Re), ".-")
plt.xlabel("Time after Solar Noon [hours]")
plt.ylabel("Wing Reynolds Number")
plt.title("Wing Reynolds Number over a Day")
plt.tight_layout()
plt.savefig("C:/Users/User/Downloads/wing_Re.png", dpi=600)
plt.show()

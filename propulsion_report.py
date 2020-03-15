from design_opt import *
import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.express as px
import plotly.graph_objects as go
import dash
import seaborn as sns
sns.set(font_scale=1)

sol = opti.solve()

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.plot(sol.value(time/3600), sol.value(power_in), "-", label="Power In (Total)")
plt.plot(sol.value(time/3600), sol.value(power_out), "-", label="Power Out (Total)")
plt.plot(sol.value(time/3600), sol.value(power_out_propulsion), "-", label="Power Out (Propulsion)")
plt.plot(sol.value(time/3600), sol.value(power_out_payload), "-", label="Power Out (Payload)")
plt.plot(sol.value(time/3600), np.tile(sol.value(power_out_avionics),n_timesteps), "-", label="Power Out (Avionics)")
plt.xlabel("Time after Solar Noon [hours]")
plt.ylabel("Power [W]")
plt.title("Power In and Power Out over a Day")
plt.tight_layout()
plt.legend()
plt.savefig("C:/Users/User/Downloads/powerio.png")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.plot(sol.value(time/3600), sol.value(net_power))
plt.xlabel("Time after Solar Noon [hours]")
plt.ylabel("Net Power (pos. is charging) [W]")
plt.title("Net Power over a Day")
plt.tight_layout()
plt.legend()
plt.savefig("C:/Users/User/Downloads/powernet.png")
plt.show()

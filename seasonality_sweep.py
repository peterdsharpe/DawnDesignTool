from design_opt import *
from aerosandbox.tools.carpet_plot_utils import time_limit

latitudes = np.linspace(-80, 80, 50)
day_of_years = np.linspace(0, 365, 60)

payloads = np.empty((
    len(latitudes), len(day_of_years)
))

for i, lat_val in enumerate(latitudes):
    for j, day_val in enumerate(day_of_years):
        print("\n".join([
            "-" * 50,
            f"latitude: {lat_val}",
            f"day of year: {day_val}",
        ]))
        opti.set_value(latitude, lat_val)
        opti.set_value(day_of_year, day_val)
        try:
            with time_limit(10):
                sol = opti.solve()
            opti.set_initial(opti.value_variables())
            opti.set_initial(opti.lam_g, sol.value(opti.lam_g))
            payload_val = sol.value(mass_payload)
        except Exception as e:
            print(e)
            payload_val = np.NaN
        finally:
            payloads[i, j] = payload_val

payloads = np.array(payloads)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette=sns.color_palette("viridis"))

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
plt.contourf(
    day_of_years,
    latitudes,
    payloads
)
plt.xlabel(r"Day of Year")
plt.ylabel(r"Latitude")
plt.title(r"Payload Capability [kg]")
plt.tight_layout()
plt.colorbar()
# plt.savefig("C:/Users/User/Downloads/temp.svg")
plt.show()

np.save("lats_no_traj_opt", latitudes)
np.save("days_no_traj_opt", day_of_years)
np.save("pays_no_traj_opt", payloads)
import aerosandbox as asb
import numpy as np

af = asb.Airfoil(name="HALE_03 (root)", coordinates="HALE_03.dat")

no_trips = af.xfoil_aseq(
    a_start=0,
    a_end=15,
    a_step=0.1,
    Re=300e3,
    max_iter=100,
    verbose=True,
)

trips = af.xfoil_aseq(
    a_start=0,
    a_end=15,
    a_step=0.1,
    Re=300e3,
    xtr_bot=0.05,
    xtr_top=0.05,
    max_iter=100,
    verbose=True,
)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette=sns.color_palette("husl", 2))

def trim_nans(array):
    return array[np.logical_not(np.isnan(array))]

### CL/CD figure
fig, ax = plt.subplots(1, 1, figsize=(4.8, 6), dpi=200)
plt.plot(
    trim_nans(no_trips["Cd"]) * 1e4,
    trim_nans(no_trips["Cl"]),
    label="Nominal"
)
plt.plot(
    trim_nans(trips["Cd"]) * 1e4,
    trim_nans(trips["Cl"]),
    label="Tripped"
)
plt.xlim(0, 400)
plt.axhline(y=1.18, ls='--', color="gray")
plt.text(
    x=0.1 * ax.get_xlim()[1] + (1 - 0.1) * ax.get_xlim()[0],
    y=1.18,
    s=r"Cruise $C_l$",
    color="gray",
    horizontalalignment='left',
    verticalalignment='bottom'
)
plt.xlabel(r"$C_d \times 10^4$")
plt.ylabel(r"$C_l$")
plt.title(
    "Effect of Trips on HALE_03 (root) Airfoil\n"
    r"Root section, $Re=300$k (approx. peak-altitude condition)"
)
plt.tight_layout()
plt.legend()
# plt.savefig("C:/Users/User/Downloads/temp.svg")
plt.show()

### Airfoil figure
fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=200)
x = af.coordinates[:,0]
y=af.coordinates[:,1]
plt.plot(x, y, "-", zorder=11, color='#280887')
plt.axis("equal")
plt.axis("off")
# plt.xlabel(r"$x/c$")
# plt.ylabel(r"$y/c$")
# plt.title("%s Airfoil" % af.name)
plt.tight_layout()
plt.show()

### Calculate cruise drags
cruise_Cl = 1.18

from scipy import interpolate

no_trips_Cd = interpolate.interp1d(
    trim_nans(no_trips["Cl"]),
    trim_nans(no_trips["Cd"]),
    kind="cubic"
)(cruise_Cl)
trips_Cd = interpolate.interp1d(
    trim_nans(trips["Cl"]),
    trim_nans(trips["Cd"]),
    kind="cubic"
)(cruise_Cl)
print(trips_Cd/no_trips_Cd)
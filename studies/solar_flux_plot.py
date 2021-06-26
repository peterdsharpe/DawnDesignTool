import numpy as np
import aerosandbox.library.power_solar as lib_solar
from aerosandbox.tools.pretty_plots import plt, sns, mpl, show_plot

days = np.linspace(0, 365, 600)
lats = np.linspace(-80, 80, 400)
Days, Lats = np.meshgrid(days, lats)
Days = Days.reshape((*Days.shape, 1))
Lats = Lats.reshape((*Lats.shape, 1))

time = np.linspace(0, 86400, 250)
Time = time.reshape((1, 1, -1))

flux = lib_solar.solar_flux_on_horizontal(
    latitude=Lats,
    day_of_year=Days,
    time = Time,
    scattering=True
)

energy = np.trapz(flux, time, axis=2)
energy = np.where(
    energy < 1,
    -1,
    energy
)


energy_kwh = energy / 1000 / 3600

### Payload plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
args = [
    days,
    lats,
    energy_kwh,
]
kwargs = {
    "levels": np.arange(0, 12.1, 1),
    "alpha" : 0.7,
    "extend": "both",
}

viridis = mpl.cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[0, :] = np.array([0, 0, 0, 1])
newcmp = mpl.colors.ListedColormap(newcolors)

CS = plt.contour(*args, **kwargs, colors="k", linewidths=0.5)
CF = plt.contourf(*args, **kwargs, cmap=newcmp)
cbar = plt.colorbar(label=r"Daily Solar Energy [kWh/m$^2$]", extendrect=True)
ax.clabel(CS, inline=1, fontsize=9, fmt=r"%.0f kWh/m$^2$")

plt.xticks(
    np.linspace(0, 365, 13)[:-1],
    (
        "Jan. 1",
        "Feb. 1",
        "Mar. 1",
        "Apr. 1",
        "May 1",
        "June 1",
        "July 1",
        "Aug. 1",
        "Sep. 1",
        "Oct. 1",
        "Nov. 1",
        "Dec. 1"
    ),
    rotation=40
)
lat_label_vals = np.arange(-80, 80.1, 20)
lat_labels = []
for lat in lat_label_vals:
    if lat >= 0:
        lat_labels.append(f"{lat:.0f}N")
    else:
        lat_labels.append(f"{-lat:.0f}S")
plt.yticks(
    lat_label_vals,
    lat_labels
)

plt.suptitle(
    "Total Solar Energy Available per Day",
    y=0.98
)
plt.title(
    "\n".join([
        "Incident radiation on a horizontal surface.",
        "Assumes no cloud cover. Includes atmospheric scattering losses."
    ]),
    fontsize=10
)

show_plot(
    xlabel="Time of Year",
    ylabel="Latitude",
)

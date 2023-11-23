from aerosandbox.tools.pretty_plots import plt, sns, mpl, show_plot
import aerosandbox.numpy as np
from scipy import interpolate
import pandas as pd

run_name = "Wildfire/1km_radius_20dB_5mRes"

debug_mode = False

# Do raw imports
data = pd.read_csv(f"cache/{run_name}.csv")
data.columns = data.columns.str.strip()
days_raw = np.array(data['Days'], dtype=float)
lats_raw = np.array(data['Latitudes'], dtype=float)
spans_raw = np.array(data['Spans'], dtype=float)

# # Transfer to a grid
# res_y = np.sum(np.diff(lats_raw) < 0) + 1
# res_x = len(lats_raw) // res_y
# lats_grid = lats_raw[:res_x]
# days_grid = days_raw[::res_x]
# spans_grid = spans_raw.reshape((res_y, res_x)).T
# assert len(lats_grid) * len(days_grid) == np.product(spans_raw.shape)
#
# # Patch NaNs
# from interpolate_utils import bridge_nans
#
# bridge_nans(spans_grid, depth=2)
#

# Add dummy points
bad_points = [
    (0, 80),
    (0, 70),
    (365, 80),
    (365, 70),
    (244, -80),
    (244, -70),
]
for p in bad_points:
    days_raw = np.append(days_raw, p[0])
    lats_raw = np.append(lats_raw, p[1])
    spans_raw = np.append(spans_raw, 1000)

# # Filter by nan
nan = np.isnan(spans_raw)
# infeasible_value = 100  # Value to assign to NaNs and worse-than-this points
# spans_raw[nan] = infeasible_value
# spans_raw[spans_raw > infeasible_value] = infeasible_value
# nan = np.isnan(spans_raw)

rbf = interpolate.RBFInterpolator(
    np.vstack((
        days_raw[~nan],
        lats_raw[~nan],
    )).T,
    spans_raw[~nan],
    smoothing=50,
)

days_plot = np.linspace(0, 365, 300)
lats_plot = np.linspace(-80, 80, 200)
# Convert to 2D arrays
Days_plot, Lats_plot = np.meshgrid(days_plot, lats_plot)

Spans = rbf(
    np.vstack((
        Days_plot.flatten(),
        Lats_plot.flatten()
    )).T
).reshape(Days_plot.shape)

### Payload plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)
args = [
    Days_plot,
    Lats_plot,
    Spans,
]
kwargs = {
    "levels": np.arange(20, 50.1, 2),
    "alpha" : 0.7,
    "extend": "both",
}

viridis = mpl.cm.get_cmap('viridis_r', 256)
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[-1, :] = np.array([0, 0, 0, 1])
newcmp = mpl.colors.ListedColormap(newcolors)

CS = plt.contour(*args, **kwargs, colors="k", linewidths=0.5)
CF = plt.contourf(*args, **kwargs, cmap=newcmp)
cbar = plt.colorbar(label="Wing Span [m]", extendrect=True)
ax.clabel(CS, inline=1, fontsize=10, fmt="%.0f m")

### Does unstructured linear interpolation; useful for checking RBF accuracy.
# args = [
#     days_raw[~nan],
#     lats_raw[~nan],
#     spans_raw[~nan]
# ]
# CS = plt.tricontour(*args, **kwargs, colors="k", linewidths=0.5)
# CF = plt.tricontourf(*args, **kwargs, cmap=newcmp)
# cbar = plt.colorbar(label="Wing Span [m]", extendrect=True)
# ax.clabel(CS, inline=1, fontsize=10, fmt="%.0f m")

### Plots the location of raw data points. Useful for debugging.
if debug_mode:
    plt.scatter(
        days_raw[~nan],
        lats_raw[~nan],
        c=spans_raw[~nan],
        cmap=newcmp,
        edgecolor="w",
        zorder=4
    )
    plt.clim(*CS.get_clim())

### Plots the region of interest (arctic ice)

# ax.add_patch(
#     plt.Rectangle(
#         (1, -80),
#         width=(55),
#         height=(20),
#         linestyle="--",
#         color="k",
#         linewidth=1,
#         fill=False
#     )
# )
# ax.add_patch(
#     plt.Rectangle(
#         (365-22, -80),
#         width=(20),
#         height=(20),
#         linestyle="--",
#         color="k",
#         linewidth=1,
#         fill=False
#     )
# )
#
# ax.add_patch(
#     plt.Rectangle(
#         (115, 60),
#         width=(130),
#         height=(20),
#         linestyle="--",
#         color="k",
#         linewidth=1,
#         fill=False
#     )
# )

# ### Plots the region of interest (CONUS)
# plt.plot(
#     244,
#     26,
#     ".--k",
#     label="Region of Interest\n& Sizing Case",
# )
# ax.add_patch(
#     plt.Rectangle(
#         (152, 26),
#         width=(244 - 152),
#         height=(49 - 26),
#         linestyle="--",
#         color="k",
#         linewidth=1,
#         fill=False
#     )
# )
## Amazon Mission
# ax.add_patch(
#     plt.Rectangle(
#         (1, -11),
#         width=(362),
#         height=(15),
#         linestyle="--",
#         color="k",
#         linewidth=1,
#         fill=False
#     )
# )

# ### Plot the region of interest (hurricane)
# ax.add_patch(
#     plt.Rectangle(
#         (212, 5),
#         width=(90),
#         height=(45),
#         linestyle="--",
#         color="k",
#         linewidth=1,
#         fill=False
#     )
# )
#
# plt.annotate(
#     text="Infeasible",
#     xy=(174, -55),
#     xycoords="data",
#     ha="center",
#     fontsize=10,
#     color='w',
# )

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
    "Minimum Wingspan Airplane by Mission",
    y=0.98
)
plt.title(
    "\n".join([
        "6 kg payload, min alt set by strat height, no alt. cycling, 450 Wh/kg batteries,",
        "100W continuous payload power, SunPower solar cells on wing and vertical tail, station-keeping in 95% wind"
    ]),
    fontsize=10
)

show_plot(
    xlabel="Time of Year",
    ylabel="Latitude",
    show=True,
)
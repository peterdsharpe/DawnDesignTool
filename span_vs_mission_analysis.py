from aerosandbox.tools.pretty_plots import plt, sns, mpl, show_plot
import aerosandbox.numpy as np
from scipy import interpolate

cache_suffix = "_10kg_payload"

# Do raw imports
lats_raw = np.load(f"cache/lats{cache_suffix}.npy", allow_pickle=True)
days_raw = np.load(f"cache/days{cache_suffix}.npy", allow_pickle=True)
spans_raw = np.load(f"cache/spans{cache_suffix}.npy", allow_pickle=True)

# Transfer to a grid
res_y = np.sum(np.diff(lats_raw) < 0) + 1
res_x = len(lats_raw) // res_y
lats_grid = lats_raw[:res_x]
days_grid = days_raw[::res_x]
spans_grid = spans_raw.reshape((res_y, res_x)).T
assert len(lats_grid) * len(days_grid) == np.product(spans_raw.shape)

# Patch NaNs
from interpolate_utils import bridge_nans

bridge_nans(spans_grid)

# Filter by nan
nan = np.isnan(spans_raw)
infeasible_value = 55  # Value to assign to NaNs and worse-than-this points
spans_raw[nan] = infeasible_value
spans_raw[spans_raw > infeasible_value] = infeasible_value

rbf = interpolate.RBFInterpolator(
    np.vstack((
        days_raw,
        lats_raw,
    )).T,
    spans_grid.flatten(order="F"),
    smoothing=200,
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
#     day_of_years_raw,
#     latitudes_raw,
#     Spans_raw
# ]
# CS = plt.tricontour(*args, **kwargs, colors="k", linewidths=0.5)
# CF = plt.tricontourf(*args, **kwargs, cmap=newcmp)
# cbar = plt.colorbar(label="Wing Span [m]", extendrect=True)
# ax.clabel(CS, inline=1, fontsize=10, fmt="%.0f m")

### Plots the location of raw data points. Useful for debugging.
plt.scatter(
    days_raw[~nan],
    lats_raw[~nan],
    c=spans_raw[~nan],
    cmap=newcmp,
    edgecolor="w",
    zorder=4
)
plt.clim(*CS.get_clim())

### Plots the region of interest (CONUS)
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
#         linewidth=0.5,
#         fill=False
#     )
# )

plt.annotate(
    text="Infeasible",
    xy=(174, -55),
    xycoords="data",
    ha="center",
    fontsize=10,
    color='w',
)

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

plt.xlabel(r"Time of Year")
plt.ylabel(r"Latitude")
plt.suptitle("Minimum Wingspan Airplane by Mission", y=0.98)
plt.title(
    "\n"
    "10 kg payload, min alt set by strat height, 450 Wh/kg batteries,\n Microlink solar cells, station-keeping in 95% wind, no alt. cycling",
    fontsize=10,
)
plt.tight_layout()
plt.show()

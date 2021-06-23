from aerosandbox.tools.pretty_plots import plt, sns, mpl, show_plot
import aerosandbox.numpy as np
from scipy import interpolate

cache_suffix = "_10kg_payload_no_cycling"

# Do raw imports
latitudes_raw = np.load(f"cache/lats{cache_suffix}.npy", allow_pickle=True)
day_of_years_raw = np.load(f"cache/days{cache_suffix}.npy", allow_pickle=True)
Spans_raw = np.load(f"cache/spans{cache_suffix}.npy", allow_pickle=True)

# def annual_cylindrical_distance_metric(
#         a: np.ndarray,
#         b: np.ndarray,
# ) -> np.ndarray:
#     a = np.array([
#         np.cos(a[0] / 365 * 2 * np.pi),
#         np.sin(a[0] / 365 * 2 * np.pi),
#         a[1] / 160 * 2 * np.pi,
#     ])
#     b = np.array([
#         np.cos(b[0] / 365 * 2 * np.pi),
#         np.sin(b[0] / 365 * 2 * np.pi),
#         b[1] / 160 * 2 * np.pi,
#     ])
#     return np.sqrt(
#         np.sum((a - b) ** 2)
#     )
#
#
# rbf = interpolate.Rbf(  # Old RBF implementation
#     np.array(day_of_years_raw),
#     np.array(latitudes_raw),
#     np.array(Spans_raw),
#     norm=annual_cylindrical_distance_metric,
#     function='linear',
#     smooth=0,
# )

rbf = interpolate.RBFInterpolator(
    y=np.vstack((
        np.array(day_of_years_raw),
        np.array(latitudes_raw),
    )).T,
    d=np.array(Spans_raw),
    smoothing=50,
    degree=4
)
day_of_years = np.linspace(0, 365, 300)
latitudes = np.linspace(-80, 80, 200)
# Convert to 2D arrays
Days, Lats = np.meshgrid(day_of_years, latitudes)
# Spans = rbf(Days, Lats).reshape(Days.shape)  # Used for old RBF implementation

Spans = rbf(
    np.vstack((
        Days.flatten(),
        Lats.flatten()
    )).T
).reshape(Days.shape)

### Payload plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)
args = [
    Days,
    Lats,
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
# plt.plot(
#     day_of_years_raw,
#     latitudes_raw,
#     ".",
#     color="r",
#     markeredgecolor="w"
# )

## Plots the region of interest (CONUS)
plt.plot(
    244,
    26,
    ".--k",
    label="Region of Interest\n& Sizing Case",
)
ax.add_patch(
    plt.Rectangle(
        (152, 26),
        width=(244 - 152),
        height=(49 - 26),
        linestyle="--",
        color="k",
        edgecolor="k",
        linewidth=0.5,
        fill=False
    )
)

plt.annotate(
    s="Infeasible",
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

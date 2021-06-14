from design_opt import *
from aerosandbox.tools.carpet_plot_utils import time_limit, patch_nans
import matplotlib as mpl
cache_suffix="_30kg_payload"

def run_sweep():
    latitudes = np.linspace(-80, 80, 15)
    day_of_years = np.linspace(0, 365, 30)
    spans = []
    days = []
    lats = []

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
                span_val = sol.value(wing_span)
                lats.append(lat_val)
                days.append(day_val)
                spans.append(span_val)
            except Exception as e:
                print(e)

    np.save("cache/lats" + cache_suffix, lats)
    np.save("cache/days" + cache_suffix, days)
    np.save("cache/spans" + cache_suffix, spans)


def analyze():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.interpolate import Rbf
    sns.set(palette=sns.color_palette("viridis"))

    # Do raw imports
    latitudes = np.load(f"cache/lats{cache_suffix}.npy", allow_pickle=True)
    day_of_years = np.load(f"cache/days{cache_suffix}.npy", allow_pickle=True)
    Spans = np.load(f"cache/spans{cache_suffix}.npy", allow_pickle=True)

    rbf = Rbf(
        np.array(day_of_years),
        np.array(latitudes),
        np.array(Spans),
        function='cubic',
        smooth=5,
    )
    day_of_years = np.linspace(0, 365, 300)
    latitudes = np.linspace(-80, 80, 300)
    # Convert to 2D arrays
    Days, Lats = np.meshgrid(day_of_years, latitudes)
    Spans = rbf(Days, Lats).reshape(300, 300)

    ### Payload plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)
    args = [
        Days,
        Lats,
        Spans,
    ]

    levels = np.arange(20, 50.1, 2)
    viridis = mpl.cm.get_cmap('viridis_r', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    newcolors[-1, :] = np.array([0, 0, 0, 1])
    newcmp = mpl.colors.ListedColormap(newcolors)
    CS = plt.contour(*args, levels=levels, linewidths=0.5, colors="k", alpha=0.7, extend='both')
    CF = plt.contourf(*args, levels=levels, cmap=newcmp, alpha=0.7, extend='both')
    cbar = plt.colorbar()
    ax.clabel(CS, inline=1, fontsize=10, fmt="%.0f m")

    plt.plot(
        244,
        49,
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
        "30 kg payload, min alt set by strat height, 450 Wh/kg cells,\n 89% batt. packing factor, station-keeping in 95% wind",
        fontsize = 10,
    )
    cbar = plt.colorbar()
    cbar.set_label("Wing Span [m]")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


run_sweep()
analyze()

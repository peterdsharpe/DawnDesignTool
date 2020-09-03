from design_opt import *
from aerosandbox.tools.carpet_plot_utils import time_limit, patch_nans

cache_suffix="_60kft"

### Data acquisition
def run_sweep():
    latitudes = np.linspace(-80, 80, 30)
    day_of_years = np.linspace(0, 365, 30)
    spans = np.empty((
        len(latitudes),
        len(day_of_years)
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
                span_val = sol.value(wing_span)
            except Exception as e:
                print(e)
                span_val = np.NaN
            finally:
                spans[i, j] = span_val

    np.save("cache/lats" + cache_suffix, latitudes)
    np.save("cache/days" + cache_suffix, day_of_years)
    np.save("cache/spans" + cache_suffix, spans)


def analyze():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(palette=sns.color_palette("viridis"))

    # Do raw imports
    latitudes = np.load(f"cache/lats{cache_suffix}.npy")
    day_of_years = np.load(f"cache/days{cache_suffix}.npy")
    Spans = np.load(f"cache/spans{cache_suffix}.npy")

    # Convert to 2D arrays
    Days, Lats = np.meshgrid(day_of_years, latitudes)

    # Patch NaNs and smooth
    Spans = patch_nans(Spans)

    #
    from scipy.ndimage import zoom
    Days = zoom(Days, 10, order=3)
    Lats = zoom(Lats, 10, order=3)
    Spans = zoom(Spans, 10, order=3)

    ### Payload plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)
    args = [
        Days,
        Lats,
        Spans,
    ]
    levels = np.arange(20, 50.1, 2)
    plt.contour(*args, levels=[34], colors="r", linewidths=3)
    CS = plt.contour(*args, levels=levels, linewidths=0.5, colors="k", alpha=0.7, extend='both')
    CF = plt.contourf(*args, levels=levels, cmap="viridis", alpha=0.7, extend='both')
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
        s="Insufficient\nSunlight",
        xy=(174, -55),
        xycoords="data",
        ha="center",
        fontsize=10,
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
        "30 kg payload, 60 kft min. alt., 450 Wh/kg cells, 89% batt. packing factor",
        fontsize = 10,
    )
    cbar = plt.colorbar()
    cbar.set_label("Wing Span [m]")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


# run_sweep()
analyze()

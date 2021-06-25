import numpy as np

from design_opt import *
from aerosandbox.tools.carpet_plot_utils import time_limit, patch_nans

cache_suffix = 'solar_test'

def run_sweep():
        latitudes = np.linspace(-80, 80, 15)
        day_of_years = np.linspace(0, 365, 30)
        # altitudes = np.linspace(1000, 30000, 100)
        lats = []
        days = []
        sun = []

        for i, lat_val in enumerate(latitudes):
            for j, day_val in enumerate(day_of_years):
                #try:
                    latitude = lat_val
                    day_of_year = day_val
                    solar_flux_on_horizontal = lib_solar.solar_flux_on_horizontal(
                        latitude, day_of_year, time, scattering=True)
                    sun.append(opti.value(np.sum(solar_flux_on_horizontal)))
                    lats.append(lat_val)
                    days.append(day_val)
                # except:
                #     pass

        np.save("cache/lats" + cache_suffix, lats)
        np.save("cache/days" + cache_suffix, days)
        np.save("cache/sun" + cache_suffix, sun, allow_pickle=True)


def analyze():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.interpolate import Rbf
    sns.set(palette=sns.color_palette("viridis"))

    # Do raw imports
    latitudes = np.load(f"cache/lats{cache_suffix}.npy")
    days = np.load(f"cache/days{cache_suffix}.npy")
    sun = np.load(f"cache/sun{cache_suffix}.npy")
    sun = np.divide(sun, 1000)

   #  Convert to 2D arrays
    Days, Lats = np.meshgrid(days, latitudes)
    # winds.reshape()
    rbf = Rbf(
        np.array(days),
        np.array(latitudes),
        np.array(sun),
        function='linear',
        smooth=5,
    )
    #
    latitudes = np.linspace(-80, 80, 300)
    day_of_years = np.linspace(0, 365, 300)
    Days, Lats = np.meshgrid(day_of_years, latitudes)

    Sun = rbf(Days, Lats).reshape(300, 300)
    # # Patch NaNs and smooth
    # winds = patch_nans(winds)

    #
    from scipy.ndimage import zoom
    # Alts = zoom(Alts, 10, order=3)
    # Lats = zoom(Lats, 10, order=3)
    # winds = zoom(winds, 10, order=3)

    ### Payload plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)
    args = [
        Days,
        Lats,
        Sun,
    ]
    levels = np.arange(0, 90, 10)
    # plt.contour(*args, levels=[34], colors="r", linewidths=3)
    CS = plt.contour(*args, levels=levels, linewidths=0.5, colors="k", alpha=0.7, extend='both')
    CF = plt.contourf(*args, levels=levels, cmap="viridis", alpha=0.7, extend='both')
    ax.clabel(CS, inline=1, fontsize=10, fmt="%.0f kj")
    # plt.plot(
    #     244,
    #     49,
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
    #         edgecolor="k",
    #         linewidth=0.5,
    #         fill=False
    #     )
    # # )
    # plt.annotate(
    #     s="Mission\nInfeasible",
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

    plt.xlabel(r"Day of Year")
    plt.ylabel(r"Latitude")
    plt.suptitle("Total Solar Flux on Horizontal over Day", y=0.98)
    # plt.title(
    #     "\n"
    #     "30 kg payload, min alt set by strat height, 450 Wh/kg cells,\n 89% batt. packing factor, station-keeping in 95% wind",
    #     fontsize = 10,
    # )
    cbar = plt.colorbar()
    cbar.set_label("Solar Flux on Horizontal [kj]")
    # plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


run_sweep()
analyze()
import numpy as np
import datetime
from design_opt import *
from aerosandbox.visualization.carpet_plot_utils import time_limit, patch_nans
from aerosandbox.modeling.interpolation import InterpolatedModel

cache_suffix = '_wind_at_cruise'

days = np.array([-365, -335., -305., -274., -244., -213., -182., -152., -121., -91.,
                 -60., -32., 1., 32., 60., 91., 121., 152.,
                 182., 213., 244., 274., 305., 335., 366., 397., 425.,
                 456., 486., 517., 547., 578., 609., 639., 670., 700.])

latitudes = np.load('../cache/latitudes.npy')
latitudes = np.flip(latitudes)
altitudes = np.load('../cache/altitudes.npy')
altitudes = np.flip(altitudes)
wind_data = np.load('../cache/wind_speed_array.npy')
wind_data_array = np.dstack((np.flip(wind_data), wind_data))
wind_data_array = np.dstack((wind_data_array, np.flip(wind_data)))
wind_function_95th = InterpolatedModel({"altitudes": altitudes, "latitudes": latitudes, "day_of_year": days},
                                       wind_data_array, "bspline")

height = np.genfromtxt(path + '/cache/strat-height-monthly.csv', delimiter=',')
latitude_list = np.linspace(-80, 80, 50)
months = np.linspace(1, 12, 12)
strat_model = InterpolatedModel({'latitude': latitude_list, 'month': months},
                                height, 'bspline')
# def wind_speed_func(alt):
#     day_array = np.full(shape=alt.shape[0], fill_value=1) * day_of_year
#     latitude_array = np.full(shape=alt.shape[0], fill_value=1) * latitude
#     speed_func = wind_function_95th({"altitudes": alt, "latitudes": latitude_array, "day_of_year": day_array})
#     return speed_func

def run_sweep():
        latitudes = np.linspace(-80, 80, 15)
        day_of_years = np.linspace(0, 365, 30)
        # altitudes = np.linspace(1000, 30000, 100)
        lats = []
        days = []
        winds = []

        for i, lat_val in enumerate(latitudes):
            for j, day_val in enumerate(day_of_years):
                    print("\n".join([
                        "-" * 50,
                        f"latitude: {lat_val}",
                        f"day of year: {day_val}",
                    ]))
                #try:
                    latitude = lat_val
                    day_of_year = day_val
                    day = opti.value(day_of_year)
                    date = datetime.datetime(2020, 1, 1) + datetime.timedelta(day)
                    month = date.month
                    offset_value = 1000
                    min_cruise_altitude = strat_model({'latitude': opti.value(latitude), 'month': month}) * 1000 + offset_value
                    wind = wind_function_95th({'altitudes':min_cruise_altitude, 'latitudes':lat_val, 'day_of_year':day_val})
                    winds.append(opti.value(wind))
                    lats.append(lat_val)
                    days.append(day_val)
                # except:
                #     pass

        np.save("cache/lats" + cache_suffix, lats)
        np.save("cache/days" + cache_suffix, days)
        np.save("cache/winds" + cache_suffix, winds)


def analyze():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.interpolate import Rbf
    sns.set(palette=sns.color_palette("viridis"))

    # Do raw imports
    latitudes = np.load(f"cache/lats{cache_suffix}.npy")
    days = np.load(f"cache/days{cache_suffix}.npy")
    winds = np.load(f"cache/winds{cache_suffix}.npy")

   #  Convert to 2D arrays
    Days, Lats = np.meshgrid(days, latitudes)
    # winds.reshape()
    rbf = Rbf(
        np.array(days),
        np.array(latitudes),
        np.array(winds),
        function='linear',
        smooth=5,
    )
    #
    latitudes = np.linspace(-80, 80, 30)
    day_of_years = np.linspace(0, 365, 30)
    Days, Lats = np.meshgrid(day_of_years, latitudes)

    Winds = rbf(Days, Lats).reshape(30, 30)
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
        Winds,
    ]
    levels = np.arange(20, 60, 5)
    # plt.contour(*args, levels=[34], colors="r", linewidths=3)
    CS = plt.contour(*args, levels=levels, linewidths=0.5, colors="k", alpha=0.7, extend='both')
    CF = plt.contourf(*args, levels=levels, cmap="viridis_r", alpha=0.7, extend='both')
    ax.clabel(CS, inline=1, fontsize=10, fmt="%.0f m/s")
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
    plt.suptitle("95th Percentile Wind Speed at Minimum Cruise Altitude", y=0.98)
    # plt.title(
    #     "\n"
    #     "30 kg payload, min alt set by strat height, 450 Wh/kg cells,\n 89% batt. packing factor, station-keeping in 95% wind",
    #     fontsize = 10,
    # )
    cbar = plt.colorbar()
    cbar.set_label("95th Percentile Wind Speed [m/s]")
    # plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


run_sweep()
analyze()

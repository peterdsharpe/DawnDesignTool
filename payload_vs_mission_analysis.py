import sys
sys.path.append("C:\\Users\\AnnickDewald\\PycharmProjects\\AeroSandbox")
from aerosandbox.tools.pretty_plots import plt, sns, mpl, show_plot
import aerosandbox.numpy as np
from scipy import interpolate
import pandas as pd

debug_mode = False
run_num = 3
run_name = "payload_run"
def plot(run_name, title1, title2, run_number):
    # Do raw imports
    data = pd.read_csv(f"cache/Wildfire/{run_name}_{run_number}.csv")
    data.columns = data.columns.str.strip()
    days_raw = np.array(data['Days'], dtype=float)
    lats_raw = np.array(data['Latitudes'], dtype=float)
    payloads_raw = np.array(data['Payloads'], dtype=float)

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
        payloads_raw = np.append(payloads_raw, -1000)

    # # Filter by nan
    nan = np.isnan(payloads_raw)
    # infeasible_value = 100  # Value to assign to NaNs and worse-than-this points
    # spans_raw[nan] = infeasible_value
    # spans_raw[spans_raw > infeasible_value] = infeasible_value
    # nan = np.isnan(spans_raw)

    rbf = interpolate.RBFInterpolator(
        np.vstack((
            days_raw[~nan],
            lats_raw[~nan],
        )).T,
        payloads_raw[~nan],
        smoothing=0,
        kernel="cubic",
    )

    days_plot = np.linspace(0, 365, 300)
    lats_plot = np.linspace(-80, 80, 200)
    # Convert to 2D arrays
    Days_plot, Lats_plot = np.meshgrid(days_plot, lats_plot)

    Payloads = rbf(
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
        Payloads,
    ]
    kwargs = {
        "levels": np.arange(0, 100, 10),
        "alpha" : 0.7,
        "extend": "both",
    }

    viridis = mpl.cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([0, 0, 0, 1])
    newcmp = mpl.colors.ListedColormap(newcolors)

    CS = plt.contour(*args, **kwargs, colors="k", linewidths=0.5)
    CF = plt.contourf(*args, **kwargs, cmap=newcmp)
    cbar = plt.colorbar(label="Payload [kg]", extendrect=True)
    ax.clabel(CS, inline=1, fontsize=9, fmt="%.0f kg")

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
            c=payloads_raw[~nan],
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

    plt.suptitle(
        "Maximum Payload Airplane by Mission",
        y=0.98
    )
    plt.title(
        "\n".join([
            title1,
            title2
        ]),
        fontsize=10
    )
    plt.xlabel("Day of Year")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"cache/Wildfire/plot_{run_num}.png")
    plt.show()
    plt.close()

def read_excel_data(excel_file):
    df = pd.read_excel(excel_file)
    return df.values.tolist()

if __name__ == '__main__':

    excel_file = "cache/Wildfire/wildfire_runs_payload.xlsx"
    # Read data from Excel
    inputs = read_excel_data(excel_file)
    title_1 = inputs[run_num][8]
    title_2 = inputs[run_num][9]
    plot(run_name, title_1, title_2, run_num)


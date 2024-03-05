import multiprocessing as mp
from ice_dyn_design_opt import *
import aerosandbox.numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib as mpl
import csv
import itertools


### where is parameter cominations file and where should outputs go?
run_name = "lawnmower_new_obj"

### define sweep ranges
max_weighting = 4
number_of_objective_terms = 6

### Turn parallelization on/off.
parallel = False

def create_grid(max, num_objective_terms, run_name):
    # Generating permutations
    permutations = list(itertools.product(range(1, max+1), repeat=num_objective_terms))

    # Writing permutations to a CSV file
    with open(f'outputs\\{run_name}\\0_parameter_combinations.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(permutations)

    print("Permutations saved to permutations.csv")

    # create np.array of permutations where the first column is the index of the run
    permutations = np.array(permutations)
    permutations = np.insert(permutations, 0, np.arange(0, permutations.shape[0]), axis=1)
    files = np.array([run_name] * len(permutations))
    combined_array = np.column_stack((files, permutations))

    return combined_array

def run(run_name, index_val, wingspan_scaling_term, resolution_scaling_term, coverage_scaling_term, day_scaling_term,
        precision_scaling_term, temporal_scaling_term):
    print("\n".join([
        "-" * 50,
        f"Run number: {index_val}",
        f"Wingspan: {wingspan_scaling_term}",
        f"Resolution: {resolution_scaling_term}",
        f"Coverage: {coverage_scaling_term}",
        f"Day: {day_scaling_term}",
        f"Precision: {precision_scaling_term}",
        f"Temporal: {temporal_scaling_term}",
    ]))
    run_num = index_val
    opti.set_value(wingspan_optimization_scaling_term, float(wingspan_scaling_term))
    opti.set_value(azimuth_optimization_scaling_term, float(resolution_scaling_term))
    opti.set_value(coverage_optimization_scaling_term, float(coverage_scaling_term))
    opti.set_value(day_optimization_scaling_term, float(day_scaling_term))
    opti.set_value(precision_optimization_scaling_term, float(precision_scaling_term))
    opti.set_value(temporal_optimization_scaling_term, float(temporal_scaling_term))

    try:
        sol = opti.solve(
            max_iter=30000,
            options={
                "ipopt.max_cpu_time": 60000,
            },
            verbose=False
        )

        print("converged for track = 0 case!")

        opti.set_initial_from_sol(sol)
        opti.set_value(track_scaler, 1)
        try:
            sol = opti.solve(
                max_iter=30000,
                options={
                    "ipopt.max_cpu_time": 60000,
                },
                verbose=False
            )
            print("converged for track = 1 case!")

            def s(x):  # Shorthand for evaluating the value of a quantity x at the optimum
                return sol.value(x)

            span = s(wing_span)
            resolution_val = s(strain_azimuth_resolution)
            coverage_val = s(coverage_area)
            day_val = s(day_of_year)
            precision_val = s(required_strain_precision)
            temporal_val = s(strain_temporal_resolution)

            def output(x: Union[str, List[str]]) -> None:  # Output a scalar variable (give variable name as a string).
                if isinstance(x, list):
                    for xi in x:
                        output(xi)
                    return
                # print(f"{x}: {sol.value(eval(x)):.3f}")

            def fmt(x):
                return f"{s(x):.6g}"

            def plot(
                    x_name: str,
                    y_name: str,
                    xlabel: str,
                    ylabel: str,
                    title: str,
                    save_name: str = None,
                    show: bool = True,
                    plot_day_color=(103 / 255, 155 / 255, 240 / 255),
                    plot_night_color=(7 / 255, 36 / 255, 84 / 255),
            ) -> None:  # Plot a variable x and variable y, highlighting where day and night occur

                # Make the plot axes
                fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=plot_dpi)

                # Evaluate the data, and plot it. Shade according to whether it's day/night.
                x = s(eval(x_name))
                y = s(eval(y_name))
                plot_average_color = tuple([
                    (d + n) / 2
                    for d, n in zip(plot_day_color, plot_night_color)
                ])
                plt.plot(  # Plot a black line through all points
                    x,
                    y,
                    '-',
                    color=plot_average_color,
                )
                plt.plot(  # Emphasize daytime points
                    x[is_daytime],
                    y[is_daytime],
                    '.',
                    color=plot_day_color,
                    label="Day"
                )
                plt.plot(  # Emphasize nighttime points
                    x[is_nighttime],
                    y[is_nighttime],
                    '.',
                    color=plot_night_color,
                    label="Night"
                )

                # Disable offset notation, which makes things hard to read.
                ax.ticklabel_format(useOffset=False)

                # Do specific things for certain variable names.
                if x_name == "hour":
                    ax.xaxis.set_major_locator(
                        ticker.MultipleLocator(base=3)
                    )
                if y_name == "y_km":
                    # Create a rectangle patch
                    rectangle = mpl.patches.Rectangle((0, s(max_imaging_offset) / 1000), s(coverage_length) / 1000,
                                                      s(coverage_width) / 1000, edgecolor='red',
                                                      facecolor='red', linewidth=2, label='Coverage Area')

                    # Add the rectangle patch to the axis
                    ax.add_patch(rectangle)
                #     plt.xlim(-75, 150)
                #     plt.ylim(-25, 125)

                # Do the usual plot things.
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
                plt.legend()
                plt.tight_layout()
                if save_name is not None:
                    plt.savefig(save_name)

                return fig, ax

            # Define the data for the "Outputs" section
            outputs_data = {
                "Wing Span": f"{fmt(wing_span)} meters",
                "Strain Range Resolution": f"{fmt(strain_range_resolution)} meters",
                "Strain Azimuth Resolution": f"{fmt(strain_azimuth_resolution)} meters",
                "Strain Temporal Resolution": f"{fmt(strain_temporal_resolution)} hours",
                "Strain Precision": f"{fmt(required_strain_precision)} 1 / yr",
                "Coverage Area": f"{fmt(coverage_area)} meters^2",
                "Day of Year": f"{fmt(day_of_year)}",
                "Cruise Altitude": f"{fmt(cruise_altitude / 1000)} kilometers",
                "Average Airspeed": f"{fmt(avg_airspeed)} m/s",
                "Wing Root Chord": f"{fmt(wing_root_chord)} meters",
                "mass_TOGW": f"{fmt(mass_total)} kg",
                "Average Cruise L/D": fmt(avg_cruise_LD),
                "CG location": "(" + ", ".join([fmt(xyz) for xyz in mass_props_TOGW.xyz_cg]) + ") m",
            }

            fmtpow = lambda x: fmt(x) + " W"

            # Define a filename for the CSV file with the temporal resolution in the name
            csv_file = f"outputs/{run_name}/outputs_data_run_{run_num}.csv"

            # Write the data to the CSV file
            with open(csv_file, 'w', newline='') as csvfile:
                fieldnames = ["Property", "Value"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write the header row
                writer.writeheader()

                # Write the data rows
                for k, v in outputs_data.items():
                    writer.writerow({"Property": k, "Value": v})
                for k, v in mass_props.items():
                    writer.writerow({"Property": k, "Value": fmt(v.mass) + " kg"})
                for k, v in {
                    "payload power": fmtpow(payload_power),
                    "payload mass": f"{fmt(mass_props['payload'].mass)} kg",
                    "aperture length": f"{fmt(radar_length)} meters",
                    "aperture width": f"{fmt(radar_width)} meters",
                    "SAR range resolution": f"{fmt(range_resolution)} meters",
                    "SAR azimuth resolution": f"{fmt(azimuth_resolution)} meters",
                    "SAR revisit period": f"{fmt(revisit_period)} hours",
                    "pixels in the incoherent averaging window": fmt(N_i),
                    "precision": f"{fmt(max_precision)} 1 / yr",
                    "pulse repetition frequency": f"{fmt(pulse_rep_freq)} Hz",
                    "bandwidth": f"{fmt(bandwidth)} Hz",
                    "center wavelength": f"{fmt(wavelength)} meters",
                    "look angle": f"{fmt(look_angle)} degrees",
                    "swath range": f"{fmt(max_swath_range)} meters",
                    "swath azimuth": f"{fmt(max_swath_azimuth)} meters",
                    "SNR": f"{fmt(max_snr)} dB",
                    "swath overlap": f"{fmt(swath_overlap)}",
                    "coverage length": f"{fmt(coverage_length)} meters",
                    "coverage width": f"{fmt(2 * max_swath_range - (max_swath_range * swath_overlap))} meters",
                }.items():
                    writer.writerow({"Property": k, "Value": v})
                for k, v in {
                    "max_power_in": fmtpow(power_in_after_panels_max),
                    "max_power_out": fmtpow(power_out_propulsion_max),
                    "battery_total_energy": fmtpow(battery_total_energy),
                }.items():
                    writer.writerow({"Property": k, "Value": v})

            print(f"Data from 'Outputs' section saved to {csv_file}")

            ### Draw plots
            plot_dpi = 200

            # Find dusk and dawn
            is_daytime = s(solar_flux_on_horizontal) >= 1  # 1 W/m^2 or greater insolation
            is_nighttime = np.logical_not(is_daytime)

            plot("hour", "z_km",
                 xlabel="Hours after Solar Noon",
                 ylabel="Altitude [km]",
                 title="Altitude over Simulation",
                 save_name=f"outputs/{run_name}/altitude_{run_num}.png"
                 )
            plot("hour", "air_speed",
                 xlabel="Hours after Solar Noon",
                 ylabel="True Airspeed [m/s]",
                 title="True Airspeed over Simulation",
                 save_name=f"outputs/{run_name}/airspeed_{run_num}.png"
                 )
            plot("hour", "net_power",
                 xlabel="Hours after Solar Noon",
                 ylabel="Net Power [W] (positive is charging)",
                 title="Net Power to Battery over Simulation",
                 save_name=f"outputs/{run_name}/net_power_{run_num}.png"
                 )
            plot("hour", "battery_charge_state",
                 xlabel="Hours after Solar Noon",
                 ylabel="State of Charge [%]",
                 title="Battery Charge State over Simulation",
                 save_name=f"outputs/{run_name}/battery_charge_{run_num}.png"
                 )
            plot("hour", "distance",
                 xlabel="hours after Solar Noon",
                 ylabel="Downrange Distance [km]",
                 title="Optimal Trajectory over Simulation",
                 save_name=f"outputs/{run_name}/trajectory_{run_num}.png"
                 )
            plot("x_km", "y_km",
                 xlabel="Downrange Distance in X [km]",
                 ylabel="Downrange Distance in Y [km]",
                 title="Optimal Trajectory over Simulation",
                 save_name=f"outputs/{run_name}/trajectory_{run_num}.png"
                 )

            plt.savefig(f"outputs/{run_name}/mass_pie_chart_{run_num}.png")
        except Exception as e:
            print('Failed to solve with track_scaler = 1')
            print(e)
            span = np.nan

    except Exception as e:
        print("Does not converge with track_scaler = 0")
        print(e)
        span = np.nan

    return index_val, resolution_val, coverage_val, precision_val, temporal_val, span

def run_wrapped(input):
    return run(*input)

if __name__ == '__main__':

    ### Make a data file
    filename = f"outputs/{run_name}/sweep_results.csv"
    l = 20
    with open(filename, "w+") as f:
        f.write(
            f"{'Spatial_Resolutions'.ljust(l)},"
            f"{'Coverage_Areas'.ljust(l)},"
            f"{'Precisions'.ljust(l)},"
            f"{'Temporal_Resolutions'.ljust(l)}, "
            f"{'Spans'.ljust(l)}\n"
        )

    parameter_array = create_grid(max_weighting, number_of_objective_terms, run_name)
    if parallel:
        with mp.Pool(mp.cpu_count()) as p:
            for index, resolution_val, coverage_val, precision_val, temporal_val, span_val in p.imap_unordered(
                    func=run_wrapped,
                    iterable=parameter_array,
            ):
                with open(filename, "a") as f:
                    f.write(
                        f"{str(resolution_val).ljust(l)},"
                        f"{str(coverage_val).ljust(l)},"
                        f"{str(precision_val).ljust(l)},"
                        f"{str(temporal_val).ljust(l)},"
                        f"{str(span_val).ljust(l)}\n"
                    )
    else:
        for input in parameter_array:
            index, resolution_val, coverage_val, precision_val, temporal_val, span_val = run_wrapped(input)
            with open(filename, "a") as f:
                f.write(
                    f"{str(resolution_val).ljust(l)},"
                    f"{str(coverage_val).ljust(l)},"
                    f"{str(precision_val).ljust(l)},"
                    f"{str(temporal_val).ljust(l)},"
                    f"{str(span_val).ljust(l)}\n"
                )

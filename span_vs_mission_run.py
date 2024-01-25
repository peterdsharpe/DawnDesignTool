import multiprocessing as mp
from wildfire_design_opt import *
import aerosandbox.numpy as np
import pandas as pd

### Turn parallelization on/off.
parallel = True

def read_excel_data(excel_file):
    df = pd.read_excel(excel_file)
    return df.values.tolist()

def run(day_val, lat_val):
    print("\n".join([
        "-" * 50,
        f"day of year: {day_val}",
        f"latitude: {lat_val}",
    ]))

    opti.set_value(day_of_year, day_val)
    opti.set_value(latitude, lat_val)

    try:
        # sol = func_timeout(
        #     timeout=60,
        #     func=opti.solve,
        #     args=(),
        #     kwargs={
        #         "max_iter": 200,
        #         "options" : {
        #             "ipopt.max_cpu_time": 60,
        #         },
        #         "verbose" : False
        #     }
        # )
        sol = opti.solve(
            max_iter=2000,
            max_runtime=600,
            verbose=False
        )

        print("Success!")
        if not parallel:
            opti.set_initial_from_sol(sol)

        span = sol.value(wing_span)
    except Exception as e:
        print("Fail!")
        print(e)

        span = np.NaN

    return day_val, lat_val, span


def run_wrapped(input):
    return run(*input)


if __name__ == '__main__':

    run_number = 10

    excel_file = "cache/Wildfire/wildfire_runs.xlsx"

    # read inputs from excel file
    inputs = read_excel_data(excel_file)
    input = inputs[run_number]

    # adjust model inputs to match excel file
    opti.set_value(required_snr, input[0])
    opti.set_value(battery_specific_energy_Wh_kg, input[1])
    opti.set_value(solar_cell_efficiency, input[2])
    opti.set_value(revisit_rate, input[3])
    opti.set_value(coverage_radius, input[4])
    opti.set_value(spatial_resolution, input[5])
    opti.set_value(structural_mass_margin_multiplier, input[6])

    ### Set the run ID
    run_name = f"Wildfire/run_{run_number}"

    ### Make a data file
    filename = f"cache/{run_name}.csv"
    l = 20
    with open(filename, "w+") as f:
        f.write(
            f"{'Days'.ljust(l)},"
            f"{'Latitudes'.ljust(l)},"
            f"{'Spans'.ljust(l)}\n"
        )

    ### Define sweep space
    day_of_years = np.linspace(0, 365, 60)
    latitudes = np.linspace(-80, 80, 40)

    ### Make inputs into 1D lists of inputs
    Day_of_years, Latitudes = np.meshgrid(day_of_years, latitudes)
    inputs = [
        (day, lat)
        for day, lat in zip(Day_of_years.flatten(), Latitudes.flatten())
    ]

    ### Crunch the numbers
    if parallel:
        with mp.Pool(mp.cpu_count()) as p:
            for day_val, lat_val, span_val in p.imap_unordered(
                    func=run_wrapped,
                    iterable=inputs
            ):
                with open(filename, "a") as f:
                    f.write(
                        f"{str(day_val).ljust(l)},"
                        f"{str(lat_val).ljust(l)},"
                        f"{str(span_val).ljust(l)}\n"
                    )

    else:
        for input in inputs:
            day_val, lat_val, span_val = run_wrapped(input)
            with open(filename, "a") as f:
                f.write(
                    f"{str(day_val).ljust(l)},"
                    f"{str(lat_val).ljust(l)},"
                    f"{str(span_val).ljust(l)}\n"
                )

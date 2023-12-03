import multiprocessing as mp
from wildfire_design_opt import *
import aerosandbox.numpy as np
import pandas as pd
import sys


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
            max_iter=200,
            max_runtime=60,
            verbose=False
        )
        print("Success!")
        if not parallel:
            opti.set_initial(opti.value_variables())
            opti.set_initial(opti.lam_g, sol.value(opti.lam_g))

        payload = sol.value(mass_payload)
    except Exception as e:
        print("Fail!")
        print(e)

        payload = np.NaN

    return day_val, lat_val, payload


def run_wrapped(input):
    return run(*input)


if __name__ == '__main__':
    ## name run number

    # adjust model inputs to match excel file

    ### Make a data file
    # file where outputs will be saved
    filename = f"cache/Wildfire/payload_run.csv"
    l = 20
    with open(filename, "w+") as f:
        f.write(
            f"{'Days'.ljust(l)},"
            f"{'Latitudes'.ljust(l)},"
            f"{'Payloads'.ljust(l)}\n"
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
            for day_val, lat_val, payload_val in p.imap_unordered(
                    func=run_wrapped,
                    iterable=inputs
            ):
                with open(filename, "a") as f:
                    f.write(
                        f"{str(day_val).ljust(l)},"
                        f"{str(lat_val).ljust(l)},"
                        f"{str(payload_val).ljust(l)}\n"
                    )

    else:
        for input in inputs:
            day_val, lat_val, payload_val = run_wrapped(input)
            with open(filename, "a") as f:
                f.write(
                    f"{str(day_val).ljust(l)},"
                    f"{str(lat_val).ljust(l)},"
                    f"{str(payload_val).ljust(l)}\n"
                )

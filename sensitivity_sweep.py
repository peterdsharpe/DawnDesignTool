import multiprocessing as mp
from design_opt import *
from func_timeout import func_timeout
import aerosandbox.numpy as np
import pandas as pd

### Set the run ID
run_name = "batt_spec"


def run(batt_val):
    print("\n".join([
        "-" * 50,
        f"battery specific energy: {batt_val}",
    ]))

    opti.set_value(battery_specific_energy_Wh_kg, batt_val)

    try:
        sol = func_timeout(
            timeout=60,
            func=opti.solve,
            args=(),
            kwargs={
                "max_iter": 200,
                "options" : {
                    "ipopt.max_cpu_time": 60,
                },
                "verbose" : False
            }
        )
        print("Success!")

        opti.set_initial(opti.value_variables())
        opti.set_initial(opti.lam_g, sol.value(opti.lam_g))

        span = sol.value(wing_span)
    except Exception as e:
        print("Fail!")
        print(e)

        span = np.NaN

    return batt_val, span

def plot_results(run_name):
    import matplotlib.pyplot as plt
    from matplotlib import style
    import seaborn as sns

    # Do raw imports
    data = pd.read_csv(f"cache/{run_name}.csv")
    data.columns = data.columns.str.strip()
    batt_specs = np.array(data['BattSpec'], dtype=float)
    spans = np.array(data['Spans'], dtype=float)

    sns.set(font_scale=1)

    plt.plot(batt_specs, spans, ".-")
    plt.xlabel(r"Battery Specific Energy [Wh/kg]")
    plt.ylabel(r"Wing Span [m]")
    plt.title(r"Effect of Battery Specific Energy on Wingspan")
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == '__main__':

    ### Make a data file
    filename = f"cache/{run_name}.csv"
    l = 20
    with open(filename, "w+") as f:
        f.write(
            f"{'BattSpec'.ljust(l)}\n"
            f"{'Spans'.ljust(l)}\n"
        )

    ### Define sweep space
    battery_energies = np.linspace(300, 600, 50)

    ### Crunch the numbers
    for batt_val in battery_energies:
            batt_val, span_val = run(batt_val)
            with open(filename, "a") as f:
                f.write(
                    f"{str(batt_val).ljust(l)},"
                    f"{str(span_val).ljust(l)}\n"
                )

    plot_results(run_name)


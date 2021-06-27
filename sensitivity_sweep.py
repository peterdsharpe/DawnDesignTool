import multiprocessing as mp
from design_opt import *
from func_timeout import func_timeout
import aerosandbox.numpy as np
import pandas as pd

### Set the run ID
run_name = "solar_eff"


def run(eff_val):
    print("\n".join([
        "-" * 50,
        f"Solar Cell Efficiency: {eff_val}",
    ]))

    opti.set_value(solar_cell_efficiency, eff_val)

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

    return eff_val, span

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
    plt.xlabel(r"Solar Panel Level Efficiency")
    plt.ylabel(r"Wing Span [m]")
    plt.title(r"Effect of Solar Efficiency on Wingspan")
    plt.tight_layout()
    plt.savefig('/Users/annickdewald/Desktop/Thesis/Photos/' + run_name, dpi=300)
    plt.show()

if __name__ == '__main__':

    ### Make a data file
    filename = f"cache/{run_name}.csv"
    l = 20
    with open(filename, "w+") as f:
        f.write(
            f"{'BattSpec'.ljust(l)},"
            f"{'Spans'.ljust(l)}\n"
        )

    ### Define sweep space
    solar_efficiencies = np.linspace(0.10, 0.30, 50)

    ### Crunch the numbers
    for eff in solar_efficiencies:
            eff, span_val = run(eff)
            with open(filename, "a") as f:
                f.write(
                    f"{str(eff).ljust(l)},"
                    f"{str(span_val).ljust(l)}\n"
                )

    plot_results(run_name)


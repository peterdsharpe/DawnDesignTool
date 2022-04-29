import multiprocessing as mp
from design_opt import *
from func_timeout import func_timeout
import aerosandbox.numpy as np
import pandas as pd

### Set the run ID
run_name = "wind_direction"
parameter = "wind_direction"
unit = "degrees"
sweep_space = np.linspace(0, 400, 20)

def run(val):
    print("\n".join([
        "-" * 50,
        f"{parameter}: {val}",
    ]))

    opti.set_value(eval(parameter), val)

    try:
        sol = func_timeout(
            timeout=240,
            func=opti.solve,
            args=(),
            kwargs={
                "max_iter": 2000,
                "options" : {
                    "ipopt.max_cpu_time": 240,
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

    return val, span

def plot_results(run_name):
    import matplotlib.pyplot as plt
    from matplotlib import style
    import seaborn as sns

    # Do raw imports
    data = pd.read_csv(f"cache/{run_name}.csv")
    data.columns = data.columns.str.strip()
    altitude = np.array(data[parameter], dtype=float)
    # altitude = np.divide(altitude, 1000)
    spans = np.array(data['Spans'], dtype=float)

    sns.set(font_scale=1)

    plt.plot(altitude, spans, ".-", label='Feasible Aircraft')
    # plt.scatter(altitude[43], spans[43], marker = "o", color='r', label='Baseline Design')
    plt.xlabel(f"{parameter} [{unit}]")
    plt.ylabel(r"Wing Span [m]")
    plt.title(f"Effect of {parameter} on Wingspan")
    plt.tight_layout()
    plt.legend()
    plt.savefig('/Users/annickdewald/Desktop/Thesis/Photos/' + run_name, dpi=300)
    plt.show()


if __name__ == '__main__':

    ### Make a data file
    filename = f"cache/{run_name}.csv"
    l = 20
    with open(filename, "w+") as f:
        f.write(
            f"{parameter.ljust(l)},"
            f"{'Spans'.ljust(l)}\n"
        )

    ### Define sweep space
    values = sweep_space

    ### Crunch the numbers
    for val in values:
            val, span_val = run(val)
            with open(filename, "a") as f:
                f.write(
                    f"{str(val).ljust(l)},"
                    f"{str(span_val).ljust(l)}\n"
                )

    plot_results(run_name)


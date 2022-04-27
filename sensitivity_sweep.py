import multiprocessing as mp
from design_opt import *
from func_timeout import func_timeout
import aerosandbox.numpy as np
import pandas as pd

### Set the run ID
run_name = "required_headway_per_day"


def run(dist):
    print("\n".join([
        "-" * 50,
        f"required_headway_per_day: {dist}",
    ]))

    opti.set_value(required_headway_per_day, dist)

    try:
        sol = func_timeout(
            timeout=60,
            func=opti.solve,
            args=(),
            kwargs={
                "max_iter": 2000,
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

    return dist, span

def plot_results(run_name):
    import matplotlib.pyplot as plt
    from matplotlib import style
    import seaborn as sns

    # Do raw imports
    data = pd.read_csv(f"cache/{run_name}.csv")
    data.columns = data.columns.str.strip()
    altitude = np.array(data['Distance'], dtype=float)
    altitude = np.divide(altitude, 1000)
    spans = np.array(data['Spans'], dtype=float)

    sns.set(font_scale=1)

    plt.plot(altitude, spans, ".-", label='Feasible Aircraft')
    plt.scatter(altitude[43], spans[43], marker = "o", color='r', label='Baseline Design')
    plt.xlabel(r"Required Headway Per Day [km]")
    plt.ylabel(r"Wing Span [m]")
    plt.title(r"Effect of Required Headway Per Day on Wingspan")
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
            f"{'Distance'.ljust(l)},"
            f"{'Spans'.ljust(l)}\n"
        )

    ### Define sweep space
    distances = np.linspace(0, 1e6, 50)

    ### Crunch the numbers
    for dist in distances:
            dist, span_val = run(dist)
            with open(filename, "a") as f:
                f.write(
                    f"{str(dist).ljust(l)},"
                    f"{str(span_val).ljust(l)}\n"
                )

    plot_results(run_name)


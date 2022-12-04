import multiprocessing as mp
from design_opt import *
from func_timeout import func_timeout
import aerosandbox.numpy as np
import pandas as pd

### Set the run ID
run_name = "pareto_front3"
parameter = "lam"
unit = "[-]"
sweep_space = np.linspace(0, 1, 11)

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
        revisit = sol.value(revisit_rate)
    except Exception as e:
        print("Fail!")
        print(e)

        span = np.NaN
        revisit = np.NaN

    return val, span, revisit

def plot_results(run_name):
    import matplotlib.pyplot as plt
    from matplotlib import style
    import seaborn as sns

    # Do raw imports
    data = pd.read_csv(f"cache/{run_name}.csv")
    data.columns = data.columns.str.strip()
    values = np.array(data[parameter], dtype=float)
    # altitude = np.divide(altitude, 1000)
    spans = np.array(data['Spans'], dtype=float)
    revisit = np.array(data['Revisit Rate'], dtype=float)


    sns.set(font_scale=1)

    plt.plot(revisit, spans, ".-", label='Pareto Front')
    # plt.scatter(altitude[43], spans[43], marker = "o", color='r', label='Baseline Design')
    plt.xlabel(r"Revisit Rate [-]")
    plt.ylabel(r"Wing Span [m]")
    plt.title(r"Set of Optimal Aircraft")
    plt.tight_layout()
    plt.legend()
    plt.savefig('cache/' + run_name, dpi=300)
    plt.show()


if __name__ == '__main__':

    ### Make a data file
    filename = f"cache/{run_name}.csv"
    l = 20
    with open(filename, "w+") as f:
        f.write(
            f"{parameter.ljust(l)},"
            f"{'Revisit Rate'.ljust(l)},"
            f"{'Spans'.ljust(l)}\n"
        )

    ### Define sweep space
    values = sweep_space

    ### Crunch the numbers
    for val in values:
            val, span_val, revist_val = run(val)
            with open(filename, "a") as f:
                f.write(
                    f"{str(val).ljust(l)},"
                    f"{str(revist_val).ljust(l)},"
                    f"{str(span_val).ljust(l)}\n"
                )

    plot_results(run_name)


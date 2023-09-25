import multiprocessing as mp
from new_design_opt_simple import *
import aerosandbox.numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"

### Set the run ID
run_name = "sweep"

### Turn parallelization on/off.
parallel = True


def run(wingspan_scale, time_scale, space_scale):
    print("\n".join([
        "-" * 50,
        f"Wingspan  Scaling: {wingspan_scale}",
        f"Temporal Resolution Scaling: {time_scale}",
        f"Spatial Resolution Scaling: {space_scale}",
    ]))

    opti.set_value(wingspan_optimization_scaling_term, wingspan_scale)
    opti.set_value(temporal_resolution_optimization_scaling_term, temporal_scale)
    opti.set_value(spatial_resolution_optimization_scaling_term, spatial_scale)

    try:
        sol = opti.solve(
            max_iter=1000,
            max_runtime=600,
            verbose=False
        )

        print("Success!")
        time = opti.value(temporal_resolution)
        space = opti.value(spatial_resolution)
        span = opti.value(wing_span)
        opti.set_initial_from_sol(sol)


    except Exception as e:
        print("Fail!")
        print(e)

        time = np.NaN
        space = np.NaN
        span = np.NaN

    return time, space, span



def plot_results(filename):

    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Create an interactive 3D scatter plot
    fig = px.scatter_3d(df, x='wing_span', y='temporal_resolution', z='spatial_resolution')

    # Customize the appearance if needed
    fig.update_traces(marker=dict(size=5))

    # Set axis labels
    fig.update_layout(
        scene=dict(xaxis_title='Wing Span', yaxis_title='Temporal Resolution', zaxis_title='Spatial Resolution'))

    # Show the interactive plot
    fig.show()


if __name__ == '__main__':

    ### Make a data file
    filename = f"cache/{run_name}.csv"
    l = 20
    with open(filename, "w+") as f:
        f.write(
            f"{'temporal_resolution'.ljust(l)},"
            f"{'spatial_resolution'.ljust(l)},"
            f"{'wing_span'.ljust(l)}\n"
        )

    ### Define sweep space
    df = pd.read_csv('parameter_combinations.csv')

    # read from df and set values
    for index, row in df.iterrows():
        wingspan_scale = row['Wingspan']
        temporal_scale = row['Temporal']
        spatial_scale = row['Spatial']
        time, space, span = run(wingspan_scale,
                                temporal_scale,
                                spatial_scale)
        with open(filename, "a") as f:
            f.write(
                f"{str(time).ljust(l)},"
                f"{str(space).ljust(l)},"
                f"{str(span).ljust(l)}\n"
            )

    plot_results(filename)

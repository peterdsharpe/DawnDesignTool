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
run_name = "circular_sum_grid"

def run(run_name, index_val, wingspan_scaling_term, resolution_scaling_term, coverage_scaling_term, day_scaling_term,
        precision_scaling_term, temporal_scaling_term):
    n_passes = ''.join([n for n in run_name if n.isdigit()])

    print("\n".join([
        "-" * 50,
        f"Run number: {index_val}",
        f"Wingspan: {wingspan_scaling_term}",
        f"Resolution: {resolution_scaling_term}",
        f"Coverage: {coverage_scaling_term}",
        f"Day: {day_scaling_term}",
        f"Precision: {precision_scaling_term}",
        f"Temporal: {temporal_scaling_term}",
        f"number of passes: {n_passes}",
    ]))
    primals = np.load(f"outputs/{run_name}/0_primals.npy")
    duals = np.load(f"outputs/{run_name}/0_duals.npy")
    opti.set_value(wingspan_optimization_scaling_term, float(wingspan_scaling_term))
    opti.set_value(azimuth_optimization_scaling_term, float(resolution_scaling_term))
    opti.set_value(coverage_optimization_scaling_term, float(coverage_scaling_term))
    opti.set_value(duration_optimization_scaling_term, float(day_scaling_term))
    opti.set_value(precision_optimization_scaling_term, float(precision_scaling_term))
    opti.set_value(temporal_optimization_scaling_term, float(temporal_scaling_term))
    opti.set_value(track_scaler, 0)
    try:
        opti.set_value(number_of_passes, float(n_passes))
    except:
        opti.set_value(number_of_passes, 0)
    # opti.set_initial(opti.x, primals)
    # opti.set_initial(opti.lam_g, duals)
    try:
        sol = opti.solve(
            max_iter=30000,
            options={
                "ipopt.max_cpu_time": 6000,
            },
            verbose=True
        )
        print("converged for track = 0 case!")
        opti.set_initial_from_sol(sol)
        opti.set_value(track_scaler, 1)
        try:
            sol = opti.solve(
                max_iter=30000,
                options={
                    "ipopt.max_cpu_time": 10000,
                },
                verbose=True
            )
            print("converged for track = 1 case!")
            primals = sol(opti.x)
            duals = sol(opti.lam_g)
            np.save(f"outputs/{run_name}/0_primals.npy", primals)
            np.save(f"outputs/{run_name}/0_duals.npy", duals)
            print("Saved solution")

        except Exception as e:
            print('Failed to solve with track_scaler = 1')
            print(e)

    except Exception as e:
        print("Does not converge with track_scaler = 0")
        print(e)



if __name__ == '__main__':


    run(run_name, 0, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16)


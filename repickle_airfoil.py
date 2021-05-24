import aerosandbox as asb
import aerosandbox.aerodynamics.aero_2D.xfoil as xfoil
import aerosandbox.numpy as np
import pandas as pd
import aerosandbox.modeling.interpolation
# from aerosandbox.tools.carpet_plot_utils import time_limit, patch_nans # Moved to aerosandbox.visualization.carpet_plot_utils in new ASB version; change if you have errors
from aerosandbox.modeling.interpolation import InterpolatedModel
import dill as pickle
import pathlib
from pathlib import Path
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
# from scipy.interpolate import interp2d
# from scipy.interpolate import CloughTocher2DInterpolator
# from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from aerosandbox.tools.airfoil_fitter.airfoil_fitter import AirfoilFitter

def run_xfoil():
    ### load wing airfoil from datafile
    wing_airfoil = asb.geometry.Airfoil(name="HALE_03", coordinates=r"studies/airfoil_optimizer/HALE_03.dat")

    reynolds = np.geomspace(1000, 100000000, 200)[139:]
    alphas = np.arange(-15, 15, 0.1)
    alpha = alphas[180:]
    alpha = np.append(alpha, alphas[0:180])
    #reynolds = [10000]
    output = {}
    # sweep over wide range of logspaced Re values
    for Re in reynolds:

        # initialize xfoil through asb
        xf = xfoil.XFoil(
            airfoil=wing_airfoil,
            Re=Re,
            n_crit=7,
            mach=0,
            xtr_upper=1,
            xtr_lower=1,
            max_iter=40,
            verbose=True,
            xfoil_command = "/Users/annickdewald/Desktop/Xfoil/bin/xfoil",
        )

        # test range of alpha values
        result = xf.alpha([alpha])

        # pickle each output dict
        import dill as pickle
        import pathlib
        #
        path = str(
                pathlib.Path(__file__).parent.absolute()
        )
        with open(path + f"/cache/airfoil_xfoil_results/airfoil_{Re}.cache", "wb+") as f:
            pickle.dump(result, f)

def get_Xfoil_dat():
    reynolds = np.geomspace(1000, 100000000, 200)
    # alphas = np.arange(-15, 15, 0.1)
    # alpha = alphas[180:]
    # np.append(alpha, alphas[0:180])
    alpha_list = []
    reynolds_list = []
    cl_values = []
    cd_values = []
    cm_values = []
    for Re in reynolds:
        xfoil_function = {}
        path = str(pathlib.Path(__file__).parent.absolute())
        try:
            with open(path + f"/cache/airfoil_xfoil_results/airfoil_{Re}.cache", "rb") as f:
                unpickle = pickle.Unpickler(f)
                xfoil_function = unpickle.load()

                alpha = list(xfoil_function['alpha'])
                reynold = [Re] * len(alpha)
                cl_out = list(xfoil_function['CL'])
                cd_out = list(xfoil_function['CD'])
                cm_out = list(xfoil_function['CM'])

                alpha_list.extend(alpha)
                reynolds_list.extend(reynold)
                cl_values.extend(cl_out)
                cd_values.extend(cd_out)
                cm_values.extend(cm_out)

        except (FileNotFoundError, TypeError):
            pass
    return alpha_list, reynolds_list, cl_values, cd_values, cm_values

def plot_xfoil(alpha_list, reynolds_list, cl_values, cd_values, cm_values):
    fig, ax = plt.subplots()
    ax.tricontour(reynolds_list, alpha_list, cl_values, levels=10, linewidths=0.01, colors='k')
    cntr2 = ax.tricontourf(reynolds_list, alpha_list, cl_values, levels=10)
    fig.colorbar(cntr2, ax=ax, cmap='viridis')
    # ax.plot(reynolds_list, alpha_list, 'ko', ms=1)
    ax.set_xscale("log")
    ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
           title=r"$C_l$ from Xfoil")
    plt.show()

    fig, ax = plt.subplots()
    ax.tricontour(reynolds_list, alpha_list, np.log(cd_values), levels=10, linewidths=0.01, colors='k')
    cntr2 = ax.tricontourf(reynolds_list, alpha_list, np.log(cd_values), levels=10)
    fig.colorbar(cntr2, ax=ax, cmap='viridis')
    # ax.plot(reynolds_list, alpha_list, 'ko', ms=3)
    ax.set_xscale("log")
    ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
           title=r"$C_d$ from Xfoil")
    plt.show()

    fig, ax = plt.subplots()
    ax.tricontour(reynolds_list, alpha_list, cm_values, levels=10, linewidths=0.01, colors='k')
    cntr2 = ax.tricontourf(reynolds_list, alpha_list, cm_values, levels=10)
    fig.colorbar(cntr2, ax=ax, cmap='viridis')
    # ax.plot(reynolds_list, alpha_list, 'ko', ms=3)
    ax.set_xscale("log")
    ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
           title=r"$C_m$ from Xfoil")
    plt.show()

    fig, ax = plt.subplots()
    cl_cd = np.divide(cl_values, cd_values)
    ax.tricontour(reynolds_list, alpha_list, cl_cd, levels=10, linewidths=0.01, colors='k')
    cntr2 = ax.tricontourf(reynolds_list, alpha_list, cl_cd, levels=10)
    fig.colorbar(cntr2, ax=ax, cmap='viridis')
    # ax.plot(reynolds_list, alpha_list, 'ko', ms=3)
    ax.set_xscale("log")
    ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
           title=r"$C_l$/$C_d$ from Xfoil")
    plt.show()

def run_cl(alpha_list, reynolds_list, cl_values):
    cl_rbf = Rbf(
        np.log(np.array(reynolds_list)),
        np.array(alpha_list),
        np.array(cl_values),
        function='linear',
        smooth=1,
    )
    reynolds = np.geomspace(1000, 100000000, 200)
    alphas = np.arange(-15, 15, 0.1)
    Reynolds, Alpha = np.meshgrid(reynolds[::8], alphas[::12], indexing="ij")
    cl_grid = cl_rbf(np.log(Reynolds.flatten()), Alpha.flatten()).reshape(len(reynolds[::8]), len(alphas[::12])).T
    np.save('./cache/cl_function.npy', cl_grid)
    np.save('./cache/alpha.npy', alphas[::12])
    np.save('./cache/reynolds.npy', reynolds[::8])
    return cl_grid


def run_cd(alpha_list, reynolds_list, cd_values):
    cd_rbf = Rbf(
        np.log(np.array(reynolds_list)),
        np.array(alpha_list),
        np.log(np.array(cd_values)),
        function='linear',
        smooth=1,
    )
    reynolds = np.geomspace(1000, 100000000, 200)
    alphas = np.arange(-15, 15, 0.1)
    Reynolds, Alpha = np.meshgrid(reynolds[::8], alphas[::12], indexing="ij")
    cd_grid = cd_rbf(np.log(Reynolds.flatten()), Alpha.flatten()).reshape(len(reynolds[::8]), len(alphas[::12])).T
    np.save('./cache/cd_function.npy', cd_grid)
    np.save('./cache/alpha.npy', alphas[::12])
    np.save('./cache/reynolds.npy', reynolds[::8])
    return cd_grid


def run_cm(alpha_list, reynolds_list, cm_values):
    cm_rbf = Rbf(
        np.log(np.array(reynolds_list)),
        np.array(alpha_list),
        np.array(cm_values),
        function='linear',
        smooth=1,
    )
    reynolds = np.geomspace(1000, 100000000, 200)
    alphas = np.arange(-15, 15, 0.1)
    Reynolds, Alpha = np.meshgrid(reynolds[::8], alphas[::12], indexing="ij")
    cm_grid = cm_rbf(np.log(Reynolds.flatten()), Alpha.flatten()).reshape(len(reynolds[::8]), len(alphas[::12])).T
    np.save('./cache/cm_function.npy', cm_grid)
    np.save('./cache/alpha.npy', alphas[::12])
    np.save('./cache/reynolds.npy', reynolds[::8])
    return cm_grid


def run_cl_plot(grid):


    fig, ax = plt.subplots()
    alphas = np.arange(-15, 15, 0.1)
    reynolds = np.geomspace(1000, 100000000, 200)
    Reynolds, Alpha = np.meshgrid(reynolds[::8], alphas[::12], indexing="ij")
    clr = plt.contourf(
        reynolds[::8],
        alphas[::12],
        grid,
        levels=50,
    )
    plt.contour(
        reynolds[::8],
        alphas[::12],
        grid,
        levels=50,
        colors='black',
        linewidths=0.7
    )
    ax.set_xscale("log")
    ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
           title=r"$C_l$ after RBF")
    fig.colorbar(clr, ax=ax)
    plt.show()

def run_cd_plot(cd_grid):
    fig, ax = plt.subplots()
    alphas = np.arange(-15, 15, 0.1)
    reynolds = np.geomspace(1000, 100000000, 200)
    Reynolds, Alpha = np.meshgrid(reynolds[::8], alphas[::12], indexing="ij")
    clr = plt.contourf(
        reynolds[::8],
        alphas[::12],
        np.exp(cd_grid),
        levels=50,
    )
    plt.contour(
        reynolds[::8],
        alphas[::12],
        np.exp(cd_grid),
        levels=50,
        colors='black',
        linewidths=0.7
    )
    ax.set_xscale("log")
    ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
           title=r"$C_d$ after RBF")
    fig.colorbar(clr, ax=ax)
    plt.show()

def run_cm_plot(cm_grid):
    fig, ax = plt.subplots()
    alphas = np.arange(-15, 15, 0.1)
    reynolds = np.geomspace(1000, 100000000, 200)
    Reynolds, Alpha = np.meshgrid(reynolds[::8], alphas[::12], indexing="ij")
    clr = plt.contourf(
        reynolds[::8],
        alphas[::12],
        cm_grid,
        levels=50,
    )
    plt.contour(
        reynolds[::8],
        alphas[::12],
        cm_grid,
        levels=50,
        colors='black',
        linewidths=0.7
    )
    ax.set_xscale("log")
    ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
           title=r"$C_m$ after RBF")
    fig.colorbar(clr, ax=ax)
    plt.show()

def run_cl_cd_plot(cl_grid, cd_grid):
    fig, ax = plt.subplots()
    alphas = np.arange(-15, 15, 0.1)
    reynolds = np.geomspace(1000, 100000000, 200)
    Reynolds, Alpha = np.meshgrid(reynolds[::8], alphas[::12], indexing="ij")
    clr = plt.contourf(
        reynolds[::8],
        alphas[::12],
        np.divide(cl_grid, np.exp(cd_grid)),
        levels=50,
    )
    plt.contour(
        reynolds[::8],
        alphas[::12],
        np.divide(cl_grid, np.exp(cd_grid)),
        levels=50,
        colors='black',
        linewidths=0.7
    )
    ax.set_xscale("log")
    ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
           title=r"$C_l/C_d$ after RBF")
    fig.colorbar(clr, ax=ax)
    plt.show()


# run_xfoil()

alpha_list, reynolds_list, cl_values, cd_values, cm_values = get_Xfoil_dat()
plot_xfoil(alpha_list, reynolds_list, cl_values, cd_values, cm_values)
sample_resolution = 6
alpha_list = alpha_list[::sample_resolution]
reynolds_list = reynolds_list[::sample_resolution]
cl_values = cl_values[::sample_resolution]
cd_values = cd_values[::sample_resolution]
cm_values = cm_values[::sample_resolution]

cl_grid = run_cl(alpha_list, reynolds_list, cl_values)
run_cl_plot(cl_grid)

cd_grid = run_cl(alpha_list, reynolds_list, cd_values)
run_cd_plot(cd_grid)

cm_grid = run_cl(alpha_list, reynolds_list, cm_values)
run_cm_plot(cm_grid)

run_cl_cd_plot(cl_grid, cd_grid)

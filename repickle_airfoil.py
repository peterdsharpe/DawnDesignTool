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

# from aerosandbox.tools.airfoil_fitter.airfoil_fitter import AirfoilFitter

# #
# # ### load wing airfoil from datafile
# wing_airfoil = asb.geometry.Airfoil(name="HALE_03", coordinates=r"studies/airfoil_optimizer/HALE_03.dat")
# # #
# # reynolds = np.logspace(3, 8, num=100)
# # #reynolds = [10000]
# # output = {}
# # # sweep over wide range of logspaced Re values
# # for Re in reynolds:
# #
# #     # initialize xfoil through asb
# #     xf = xfoil.XFoil(
# #         airfoil=wing_airfoil,
# #         Re=Re,
# #         n_crit=7,
# #         mach=0,
# #         xtr_upper=1,
# #         xtr_lower=1,
# #         max_iter=40,
# #         verbose=True,
# #         xfoil_command = "/Users/annickdewald/Desktop/Xfoil/bin/xfoil",
# #     )
# #
# #     # test range of alpha values
# #     result = xf.alpha([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 1, 0, -1, -2, -3, \
# #                   -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15])
# #
# #     # pickle each output dict
# #     import dill as pickle
# #     import pathlib
# #
# #     path = str(
# #         pathlib.Path(__file__).parent.absolute()
# #     )
# #     with open(path + f"/cache/airfoil_xfoil_results/airfoil_{Re}.cache", "wb+") as f:
# #         pickle.dump(result, f)
#
#
##

reynolds = np.logspace(3, 8, num=100)
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

points = np.column_stack((reynolds_list, alpha_list))
alpha = np.linspace(-15, 15, 31)
grid_alpha_y = np.array(alpha)
reynolds_vert = np.reshape(reynolds, (-1, 1))
grid_reynolds_x = reynolds_vert
for i in range(0, len(reynolds) - 1):
    grid_alpha_y = np.vstack([grid_alpha_y, alpha])
for i in range(0, len(alpha) - 1):
    grid_reynolds_x = np.hstack([grid_reynolds_x, reynolds_vert])

from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from scipy.interpolate import interp2d
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.ndimage import gaussian_filter

# Use cubic interpolator to fill in the gaps
# grid_cl = griddata(points, cl_values, (grid_reynolds_x, grid_alpha_y), method='cubic')
# grid_cd = griddata(points, cd_values, (grid_reynolds_x, grid_alpha_y), method='cubic')
# grid_cm = griddata(points, cm_values, (grid_reynolds_x, grid_alpha_y), method='cubic')
cl_rbf = Rbf(
    np.log(np.array(reynolds_list)),
    np.array(alpha_list),
    np.array(cl_values),
    function='linear',
    smooth=0.5,
)
cd_rbf = Rbf(
    np.log(np.array(reynolds_list)),
    np.array(alpha_list),
    np.array(cd_values),
    function='linear',
    smooth=0.5,
)
cm_rbf = Rbf(
    np.log(np.array(reynolds_list)),
    np.array(alpha_list),
    np.array(cm_values),
    function='linear',
    smooth=0.5,
)
Reynolds, Alpha = np.meshgrid(reynolds, alpha, indexing="ij")
cl_grid = cl_rbf(np.log(Reynolds.flatten()), Alpha.flatten()).reshape(len(reynolds), len(alpha)).T
cd_grid = cd_rbf(np.log(Reynolds.flatten()), Alpha.flatten()).reshape(len(reynolds), len(alpha)).T
cm_grid = cm_rbf(np.log(Reynolds.flatten()), Alpha.flatten()).reshape(len(reynolds), len(alpha)).T

#
# # Use linear interpolator to fill in the edges
# cl_array = np.vstack([grid_reynolds_x.flatten(), grid_alpha_y.flatten(), grid_cl.flatten()]).T
# cl_array = cl_array[~np.isnan(cl_array).any(axis=1), :]
# points = cl_array[:, 0:2]
# values = cl_array[:, 2]
# grid_cl2 = griddata(points, values, (grid_reynolds_x, grid_alpha_y), method='nearest')
#
# cd_array = np.vstack([grid_reynolds_x.flatten(), grid_alpha_y.flatten(), grid_cd.flatten()]).T
# cd_array = cd_array[~np.isnan(cd_array).any(axis=1), :]
# points = cd_array[:, 0:2]
# values = cd_array[:, 2]
# grid_cd2 = griddata(points, values, (grid_reynolds_x, grid_alpha_y), method='nearest')
#
# cm_array = np.vstack([grid_reynolds_x.flatten(), grid_alpha_y.flatten(), grid_cm.flatten()]).T
# cm_array = cm_array[~np.isnan(cm_array).any(axis=1), :]
# points = cm_array[:, 0:2]
# values = cm_array[:, 2]
# grid_cm2 = griddata(points, values, (grid_reynolds_x, grid_alpha_y), method='nearest')
#
# # smooth grid using
# grid_cl3 = gaussian_filter(grid_cl2, sigma=0.9, mode='reflect')
# grid_cd3 = gaussian_filter(grid_cd2, sigma=0.9, mode='reflect')
# grid_cm3 = gaussian_filter(grid_cm2, sigma=0.9, mode='reflect')

cl_function = InterpolatedModel({"alpha": alpha, "reynolds": reynolds, },
                                cl_grid, "bspline")
cd_function = InterpolatedModel({"alpha": alpha, "reynolds": reynolds},
                                cd_grid, "bspline")
cm_function = InterpolatedModel({"alpha": alpha, "reynolds": reynolds},
                                cm_grid, "bspline")

np.save('./cache/cl_function.npy', cl_grid)
np.save('./cache/cd_function.npy', cd_grid)
np.save('./cache/cm_function.npy', cm_grid)
np.save('./cache/alpha.npy', alpha)
np.save('./cache/reynolds.npy', reynolds)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.contourf(
        reynolds,
        alpha,
        cl_rbf(np.log(Reynolds.flatten()), Alpha.flatten()).reshape(len(reynolds), len(alpha)).T
    )
    ax.set_xscale("log")
    ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
           title=r"$C_l$ after RBF")
    plt.show()
    fig, ax = plt.subplots()
    plt.contourf(
        reynolds,
        alpha,
        cd_rbf(np.log(Reynolds.flatten()), Alpha.flatten()).reshape(len(reynolds), len(alpha)).T
    )
    ax.set_xscale("log")
    ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
           title=r"$C_d$ after RBF")
    plt.show()
    fig, ax = plt.subplots()
    plt.contourf(
        reynolds,
        alpha,
        cm_rbf(np.log(Reynolds.flatten()), Alpha.flatten()).reshape(len(reynolds), len(alpha)).T
    )
    ax.set_xscale("log")
    ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
           title=r"$C_m$ after RBF")
    plt.show()

    # fig, ax = plt.subplots()
    # ax.contour(reynolds, alpha, grid_cl2.T, levels=25, linewidths=0.5, colors='k')
    # cs = ax.contourf(reynolds, alpha, grid_cl2.T, levels=25, cmap="viridis")
    # cbar = fig.colorbar(cs)
    # ax.set_xscale("log")
    # ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
    #        title=r"$C_l$ Pre-smoothed")
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # ax.contour(reynolds, alpha, grid_cl3.T, levels=25, linewidths=0.5, colors='k')
    # cs = ax.contourf(reynolds, alpha, grid_cl3.T, levels=25, cmap="viridis")
    # cbar = fig.colorbar(cs)
    # ax.set_xscale("log")
    # ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
    #        title=r"$C_l$ Smoothed")
    # plt.show()

# rbfi = Rbf(reynolds_list, alpha_list, cl_values, function='gaussian', smooth=10)
# alpha = np.linspace(-15, 15, 100)
# di = rbfi(grid_reynolds_x, grid_alpha_y)
# # di = rbfi(reynolds, alpha)
#
# # f = interp2d(reynolds_list, alpha_list, cl_values, kind='cubic')
# # grid_cl = f(reynolds, alpha)
#
# f = interp2d(alpha, reynolds, grid_cl, kind='cubic')
# grid_cl2 = f(alpha, reynolds)


# # assemble Cl, CD, CM into 2d arrays where axis are alpha and renolds number
# cl_array = np.empty((1,31), int)
# cd_array = np.empty((1,31), int)
# cm_array = np.empty((1,31), int)
# reynolds = np.logspace(3, 8, num=100)
# # reynolds = [reynolds[11]]
#
# for Re in reynolds:
#     xfoil_function = {}
#     path = str(pathlib.Path(__file__).parent.absolute())
#     try:
#         with open(path + f"/cache/airfoil_xfoil_results/airfoil_{Re}.cache", "rb") as f:
#             unpickle = pickle.Unpickler(f)
#             xfoil_function = unpickle.load()
#
#             # pull out values from dict
#             alpha = list(xfoil_function['alpha'])
#             print(alpha)
#             cl_out = list(xfoil_function['CL'])
#             cd_out = list(xfoil_function['CD'])
#             print(cd_out)
#             cm_out = list(xfoil_function['CM'])
#             cl = []
#             cd = []
#             cm = []
#             #check if length matches and then correct if not
#             if len(alpha) != 31:
#                 alpha_spec = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 1, 0, -1, -2, -3, \
#                          -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15]
#
#                 for val in alpha_spec:
#                     try:
#                         idx = alpha.index(val)
#                         cl.append(cl_out[idx])
#                         cd.append(cd_out[idx])
#                         cm.append(cm_out[idx])
#                     except ValueError:
#                         print(val)
#                         cl.append(np.nan)
#                         cd.append(np.nan)
#                         cm.append(np.nan)
#             else:
#                 cl = cl_out
#                 cd = cd_out
#                 cm = cm_out
#
#             # reorder from alpha -15 to 15
#             # alpha = np.concatenate([np.flip(alpha[16:]), np.flip(alpha[13:16]), alpha[0:13]])
#             cl = np.concatenate([np.flip(cl[16:]), np.flip(cl[13:16]), cl[0:13]])
#             cd = np.concatenate([np.flip(cd[16:]), np.flip(cd[13:16]), cd[0:13]])
#             cm = np.concatenate([np.flip(cm[16:]), np.flip(cm[13:16]), cm[0:13]])
#             cl = np.reshape(cl, (1, 31))
#             cd = np.reshape(cd, (1, 31))
#             cm = np.reshape(cm, (1, 31))
#             print(cd)
#
#     except (FileNotFoundError, TypeError):
#         cl = np.empty((1, 31))
#         cl[:] = np.NaN
#         cd = np.empty((1, 31))
#         cd[:] = np.NaN
#         cm = np.empty((1, 31))
#         cm[:] = np.NaN
#
#     cl_array = np.append(cl_array, cl, axis=0)
#     cd_array = np.append(cd_array, cd, axis=0)
#     cm_array = np.append(cm_array, cm, axis=0)
#
#
# cl_array = cl_array[1:]
# cd_array = cd_array[1:]
# cm_array = cm_array[1:]
# alpha = np.linspace(-15,15, 31)
# alpha = np.array(alpha)
# reynolds = np.array(reynolds)
# df = pd.DataFrame(cd_array, columns = alpha, index = reynolds)
# cd_arrayb = [[cd_array[j][i] for j in range(len(cd_array))] for i in range(len(cd_array[0]))]
#
# from scipy.interpolate import interp2d
#
# f = interp2d(alpha, reynolds, cl_array, kind='cubic')
# znew = f(alpha, reynolds)

# import matplotlib.pyplot as plt
# from matplotlib import ticker, cm
# fig, ax = plt.subplots()
# contour = ax.contour(reynolds, alpha, cd_arrayb, levels=25, linewidths=0.5, colors='k')
# cs = ax.contourf(reynolds, alpha, cd_arrayb, levels=10, cmap="viridis")
# cbar = fig.colorbar(cs)
#
# ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
#        title=r"$C_d$ Function Outputs")
# contour.set_clim(0, 0.04)

# # use patch nans to put reasonable guesses in there
# cl_array = patch_nans(cl_array)
# cl_array = np.transpose(cl_array)
# cd_array = patch_nans(cd_array)
# cd_array = np.transpose(cd_array)
# cm_array = patch_nans(cm_array)
# cm_array = np.transpose(cm_array)
# # save nparray to .npy files
# np.save('/Users/annickdewald/Desktop/Thesis/DawnDesignTool/cache/cl_function.npy', cl_array)
# np.save('/Users/annickdewald/Desktop/Thesis/DawnDesignTool/cache/cd_function.npy', cd_array)
# np.save('/Users/annickdewald/Desktop/Thesis/DawnDesignTool/cache/cm_function.npy', cm_array)
# np.save('/Users/annickdewald/Desktop/Thesis/DawnDesignTool/cache/alpha.npy', alpha)
# np.save('/Users/annickdewald/Desktop/Thesis/DawnDesignTool/cache/reynolds.npy', reynolds)
#
# # cl_array = np.load('/Users/annickdewald/Desktop/Thesis/DawnDesignTool/cache/cl_function.npy')
# # cd_array = np.load('/Users/annickdewald/Desktop/Thesis/DawnDesignTool/cache/cd_function.npy')
# # cm_array = np.load('/Users/annickdewald/Desktop/Thesis/DawnDesignTool/cache/cm_function.npy')
# # alpha = np.load('/Users/annickdewald/Desktop/Thesis/DawnDesignTool/cache/alpha.npy')
# # reynolds = np.load('/Users/annickdewald/Desktop/Thesis/DawnDesignTool/cache/reynolds.npy')
#
# # create functions of cl, cd, and cm
# cl_function = InterpolatedModel({"alpha": alpha, "reynolds": reynolds,},
#                                               cl_array, "bspline")
# cd_function = InterpolatedModel({"alpha": alpha, "reynolds": reynolds},
#                                               cd_array, "bspline")
# cm_function = InterpolatedModel({"alpha": alpha, "reynolds": reynolds},
#                                               cm_array, "bspline")
# wing_airfoil.CL_function = cl_function
# wing_airfoil.CD_function = cd_function
# wing_airfoil.CM_function = cd_function
# # pickle.dump(wing_airfoil, "/Users/annickdewald/Desktop/Thesis/DawnDesignTool/cache/wing_airfoil.cache",  )
# with open("/Users/annickdewald/Desktop/Thesis/DawnDesignTool/cache/wing_airfoil.cache", "wb+") as f:
#     pickle.dump(wing_airfoil, f)

# import pathlib
#
# path = str(
#     pathlib.Path(__file__).parent.absolute()
# )
# cl_array = np.load(path + '/cache/cl_function.npy')
# cd_array = np.load(path + '/cache/cd_function.npy')
# cm_array = np.load(path + '/cache/cm_function.npy')
# alpha_array = np.load(path + '/cache/alpha.npy')
# reynolds_array = np.load(path + '/cache/reynolds.npy')
# cl_function = InterpolatedModel({"alpha": alpha_array, "reynolds": reynolds_array},
#                                               cl_array, "bspline")
# cd_function = InterpolatedModel({"alpha": alpha_array, "reynolds": reynolds_array},
#                                               cd_array, "bspline")
# cm_function = InterpolatedModel({"alpha": alpha_array, "reynolds": reynolds_array},
#                                               cm_array, "bspline")


# import numpy as np
# alphas = np.arange(-3, 10, 0.25).tolist()
# reynolds = asb.np.geomspace(300000, 1000000, 100)
# cd_array = np.arange(100).reshape(1, 100)
# cl_array = np.arange(100).reshape(1,100)
# #cd_array = np.reshape(cd_array, (99, 80))
# for alpha in alphas:
#     cds = []
#     cls = []
#     for reynold in reynolds:
#         cd = cd_function({'alpha':alpha, 'reynolds':reynold})
#         cl = cl_function({'alpha':alpha, 'reynolds':reynold})
#         cds.append(cd)
#         cls.append(cl)
#     # print(cds)
#     #print(cd_array)
#     cds = np.asarray(cds)
#     cls = np.asarray(cls)
#     cds = cds.reshape(1, 100)
#     cls = cls.reshape(1, 100)
#     cd_array = np.concatenate((cd_array,cds[:None]),axis=0)
#     cl_array = np.concatenate((cl_array, cls[:None]), axis=0)
# alphas = np.asarray(alphas)
# reynolds = np.array(reynolds)
# cd_array =cd_array[1:][:]
# cl_array =cl_array[1:][:]
#
#
# import matplotlib.pyplot as plt
# from matplotlib import ticker, cm
# fig, ax = plt.subplots()
# ax.contour(reynolds, alphas, cd_array, levels=25, linewidths=0.5, colors='k')
# cs = ax.contourf(reynolds, alphas, cd_array, levels=25, cmap="viridis")
# cbar = fig.colorbar(cs)
# ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
#        title=r"$C_d$ Function Outputs")
#
# fig, ax = plt.subplots()
# ax.contour(reynolds, alphas, cl_array, levels=25, linewidths=0.5, colors='k')
# cs = ax.contourf(reynolds, alphas, cl_array, levels=25, cmap="viridis")
# cbar = fig.colorbar(cs)
# ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
#        title=r"$C_l$ Function Outputs")
# # plt.contourf(reynolds, alphas,  cd_array, [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18])

import aerosandbox as asb
import aerosandbox.aerodynamics.aero_2D.xfoil as xfoil
import aerosandbox.numpy as np
import aerosandbox.modeling.interpolation
from aerosandbox.tools.carpet_plot_utils import time_limit, patch_nans
from aerosandbox.modeling.interpolation import InterpolatedModel

#
# from pathlib import Path
# from aerosandbox.tools.airfoil_fitter.airfoil_fitter import AirfoilFitter
#
# ### load wing airfoil from datafile
# wing_airfoil = asb.geometry.Airfoil(name="HALE_03", coordinates=r"studies/airfoil_optimizer/HALE_03.dat")
# #
# reynolds = np.logspace(3, 8, num=100)[50:65]
# #reynolds = [10000]
# output = {}
# # sweep over wide range of logspaced Re values
# for Re in reynolds:
#
#     # initialize xfoil through asb
#     xf = xfoil.XFoil(
#         airfoil=wing_airfoil,
#         Re=Re,
#         n_crit=7,
#         mach=0,
#         xtr_upper=1,
#         xtr_lower=1,
#         max_iter=40,
#         verbose=True,
#         xfoil_command = "/Users/annickdewald/Desktop/Xfoil/bin/xfoil",
#     )
#
#     # test range of alpha values
#     result = xf.alpha([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 1, 0, -1, -2, -3, \
#                   -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15])
#
#     # pickle each output dict
#     import dill as pickle
#     import pathlib
#
#     path = str(
#         pathlib.Path(__file__).parent.absolute()
#     )
#     with open(path + f"/cache/airfoil_xfoil_results/airfoil_{Re}.cache", "wb+") as f:
#         pickle.dump(result, f)


# assemble Cl, CD, CM into 2d arrays where axis are alpha and renolds number
cl_array = np.empty((1,31), int)
cd_array = np.empty((1,31), int)
cm_array = np.empty((1,31), int)
import dill as pickle
import pathlib
reynolds = np.logspace(3, 8, num=100)

for Re in reynolds:
    xfoil_function = {}
    path = str(pathlib.Path(__file__).parent.absolute())
    try:
        with open(path + f"/cache/airfoil_xfoil_results/airfoil_{Re}.cache", "rb") as f:
            unpickle = pickle.Unpickler(f)
            xfoil_function = unpickle.load()

            # pull out values from dict
            alpha = list(xfoil_function['alpha'])
            cl_out = list(xfoil_function['CL'])
            cd_out = list(xfoil_function['CD'])
            cm_out = list(xfoil_function['CM'])
            cl = []
            cd = []
            cm = []
            #check if length matches and then correct if not
            if len(alpha) != 31:
                alpha_spec = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 1, 0, -1, -2, -3, \
                         -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15]

                for val in alpha_spec:
                    try:
                        idx = alpha.index(val)
                        cl.append(cl_out[idx])
                        cd.append(cd_out[idx])
                        cm.append(cm_out[idx])
                    except ValueError:
                        cl.append(np.nan)
                        cd.append(np.nan)
                        cm.append(np.nan)
            else:
                cl = cl_out
                cd = cd_out
                cm = cm_out

            # reorder from alpha -15 to 15
            #alpha = np.concatenate([np.flip(alpha[16:]), np.flip(alpha[13:16]), alpha[0:13]])
            cl = np.concatenate([np.flip(cl[16:]), np.flip(cl[13:16]), cl[0:13]])
            cd = np.concatenate([np.flip(cd[16:]), np.flip(cd[13:16]), cd[0:13]])
            cm = np.concatenate([np.flip(cm[16:]), np.flip(cm[13:16]), cm[0:13]])
            cl = np.reshape(cl, (1, 31))
            cd = np.reshape(cd, (1, 31))
            cm = np.reshape(cm, (1, 31))

    except (FileNotFoundError, TypeError):
        cl = np.empty((1, 31))
        cl[:] = np.NaN
        cd = np.empty((1, 31))
        cd[:] = np.NaN
        cm = np.empty((1, 31))
        cm[:] = np.NaN

    cl_array = np.append(cl_array, cl, axis=0)
    cd_array = np.append(cd_array, cd, axis=0)
    cm_array = np.append(cm_array, cm, axis=0)


cl_array = cl_array[1:]
cd_array = cd_array[1:]
cm_array = cm_array[1:]
alpha = np.array(alpha_spec)
reynolds = np.array(reynolds)


# use patch nans to put reasonable guesses in there
cl_array = patch_nans(cl_array)
cd_array = patch_nans(cd_array)
cm_array = patch_nans(cm_array)
# save nparray to .npy files
np.save('cl_function', cl_array)
np.save('cd_function', cd_array)
np.save('cm_function', cm_array)
np.save('alpha', alpha)
np.save('reynolds', reynolds)
cl_function = InterpolatedModel({"reynolds": reynolds, "alpha": alpha},
                                              cl_array, "bspline")
cd_function = InterpolatedModel({"reynolds": reynolds, "alpha": alpha},
                                              cd_array, "bspline")
cm_function = InterpolatedModel({"reynolds": reynolds, "alpha": alpha},
                                              cm_array, "bspline")
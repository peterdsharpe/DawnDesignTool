import aerosandbox as asb
import aerosandbox.aerodynamics.aero_2D.xfoil as xfoil
import aerosandbox.numpy as np
import aerosandbox.modeling.interpolation

from pathlib import Path
from aerosandbox.tools.airfoil_fitter.airfoil_fitter import AirfoilFitter

# load wing airfoil from datafile
wing_airfoil = asb.geometry.Airfoil(name="HALE_03", coordinates=r"studies/airfoil_optimizer/HALE_03.dat")

reynolds = np.logspace(3, 8, num=50)[25:]
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
        max_iter=20,
        verbose=True,
        xfoil_command = "/Users/annickdewald/Desktop/Xfoil/bin/xfoil",
    )

    # test range of alpha values
    result = xf.alpha([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 1, 0, -1, -2, -3, \
                  -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15])

    # pickle each output dict
    import dill as pickle
    import pathlib

    path = str(
        pathlib.Path(__file__).parent.absolute()
    )
    with open(path + f"/cache/airfoil_{Re}.cache", "wb+") as f:
        pickle.dump(result, f)
#
# # assemble Cl, CD, CM into 2d arrays where axis are alpha and renolds number
# cl_array = np.zeros(31)
# cd_array = np.zeros(31)
# cm_array = np.zeros(31)
# import dill as pickle
# import pathlib
# reynolds = [10000]
# for Re in reynolds:
#     xfoil_function = {}
#     path = str(pathlib.Path(__file__).parent.absolute())
#     with open(path + f"/cache/airfoil_{Re}.cache", "rb") as f:
#         unpickle = pickle.Unpickler(f)
#         xfoil_function = unpickle.load()
#     alpha = xfoil_function['alpha']
#     alpha = np.concatenate([np.flip(alpha[16:]), np.flip(alpha[13:16]), alpha[0:13]])
#     cl = xfoil_function['CL']
#     cl = np.concatenate([np.flip(cl[16:]), np.flip(cl[13:16]), cl[0:13]])
#     cd = xfoil_function['CD']
#     cd = np.concatenate([np.flip(cd[16:]), np.flip(cd[13:16]), cd[0:13]])
#     cm = xfoil_function['CM']
#     cm = np.concatenate([np.flip(cm[16:]), np.flip(cm[13:16]), cm[0:13]])
#     cl_array = np.append( [cl_array], [cl], axis=0)
#     cd_array = np.append([cd_array], [cd], axis=0)
#     cm_array = np.append([cm_array], [cm], axis=0)
# cl_array = cl_array[1:]
# cd_array = cd_array[1:]
# cm_array = cm_array[1:]
# # save nparray to .npy files
# np.save('cl_function', cl_array)
# np.save('cd_function', cd_array)
# np.save('cm_function', cm_array)
# np.save('alpha', alpha)
# np.save('reynolds', reynolds)




# fill gaps in arrays with nan
# use patch nans to put reasonable guesses in there

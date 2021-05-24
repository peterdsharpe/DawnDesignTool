import aerosandbox.numpy as np
from aerosandbox.modeling.interpolation import InterpolatedModel
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

path = str(
    pathlib.Path(__file__).parent.absolute()
)

cl_array = np.load(path + '/cache/cl_function.npy')
cd_array = np.load(path + '/cache/cd_function.npy')
cm_array = np.load(path + '/cache/cm_function.npy')
alpha_array = np.load(path + '/cache/alpha.npy')
reynolds_array = np.load(path + '/cache/reynolds.npy')

alphas = np.linspace(-15, 15, 100)
reynolds = np.geomspace(1000, 100000000, 100)
Reynolds, Alpha = np.meshgrid(reynolds, alphas, indexing="ij")

cl_function = InterpolatedModel({"alpha": alpha_array, "reynolds": np.log(reynolds_array)},
                                              cl_array, "bspline")
cl_values = cl_function({'alpha': Alpha.flatten(), 'reynolds': np.log(Reynolds.flatten())})

cd_function = InterpolatedModel({"alpha": alpha_array, "reynolds": np.log(reynolds_array)},
                                              cd_array, "bspline")
cd_values = cd_function({'alpha': Alpha.flatten(), 'reynolds': np.log(Reynolds.flatten())})

cm_function = InterpolatedModel({"alpha": alpha_array, "reynolds": np.log(reynolds_array)},
                                              cm_array, "bspline")
cm_values = cm_function({'alpha': Alpha.flatten(), 'reynolds': np.log(Reynolds.flatten())})


fig, ax = plt.subplots()

clr = plt.contourf(
    reynolds,
    alphas,
    cl_values.reshape(len(reynolds), len(alphas)).T,
    levels=50,
)
plt.contour(
    reynolds,
    alphas,
    cl_values.reshape(len(reynolds), len(alphas)).T,
    levels=50,
    colors='black',
    linewidths=0.7
)
ax.set_xscale("log")
ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
       title=r"$C_l$ function outputs")
fig.colorbar(clr, ax=ax)
plt.show()

fig, ax = plt.subplots()

clr = plt.contourf(
    reynolds,
    alphas,
    np.exp(cd_values.reshape(len(reynolds), len(alphas)).T),
    levels = 50,
    # norm = LogNorm(),
)
plt.contour(
    reynolds,
    alphas,
    np.exp(cd_values.reshape(len(reynolds), len(alphas)).T),
    levels=50,
    colors='black',
    linewidths=0.7
)
ax.set_xscale("log")
ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
       title=r"$C_d$ function outputs")
fig.colorbar(clr, ax=ax)
plt.show()

fig, ax = plt.subplots()

clr = plt.contourf(
    reynolds,
    alphas,
    cm_values.reshape(len(reynolds), len(alphas)).T,
    levels=50,
)

plt.contour(
    reynolds,
    alphas,
    cm_values.reshape(len(reynolds), len(alphas)).T,
    levels=50,
    colors='black',
    linewidths=0.7
)
ax.set_xscale("log")
ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
       title=r"$C_m$ function outputs")
fig.colorbar(clr, ax=ax)
plt.show()

fig, ax = plt.subplots()

clr = plt.contourf(
    reynolds,
    alphas,
    np.divide(cl_values.reshape(len(reynolds), len(alphas)).T,np.exp(cd_values.reshape(len(reynolds), len(alphas)).T)) ,
    levels=50,
)

plt.contour(
    reynolds,
    alphas,
    np.divide(cl_values.reshape(len(reynolds), len(alphas)).T,np.exp(cd_values.reshape(len(reynolds), len(alphas)).T)),
    levels=50,
    colors='black',
    linewidths=0.7
)
ax.set_xscale("log")
ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
       title=r"$C_l$/$C_d$ function outputs")
fig.colorbar(clr, ax=ax)
plt.show()

# import plotly.graph_objects as go
#
# fig = go.Figure(data=[go.Surface(z=cl_values, x=alphas, y=reynolds)])
# fig.update_layout(title='Cl Function', autosize=True,
#                   width=500, height=500,
#                   margin=dict(l=65, r=50, b=65, t=90))
# #fig.update_yaxes(type="log")
# fig.show()
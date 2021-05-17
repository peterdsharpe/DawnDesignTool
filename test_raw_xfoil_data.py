from repickle_airfoil import *

# fig, ax = plt.subplots()
# plt.tricontourf(
#     np.array(reynolds_list),
#     np.array(alpha_list),
#     np.array(cl_values)
# )
# cbar = fig.colorbar(cs)
# ax.set_xscale("log")
# ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
#        title=r"$C_l$ Raw from XFoil")
# plt.show()
#
# fig, ax = plt.subplots()
# plt.contourf(
#     reynolds,
#     alpha,
#     grid_cl.T
# )
# cbar = fig.colorbar(cs)
# ax.set_xscale("log")
# ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
#        title=r"$C_l$ after griddata()")
# plt.show()



fig, ax = plt.subplots()
cl_rbf = Rbf(
    np.log(np.array(reynolds_list)),
    np.array(alpha_list),
    np.array(cl_values),
    function='linear',
    smooth=1,
)
Reynolds, Alpha = np.meshgrid(reynolds, alpha, indexing="ij")
plt.contourf(
    reynolds,
    alpha,
    cl_rbf(np.log(Reynolds.flatten()), Alpha.flatten()).reshape(len(reynolds), len(alpha)).T
)
ax.set_xscale("log")
ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
       title=r"$C_l$ after RBF")
plt.show()
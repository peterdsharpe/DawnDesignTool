from repickle_airfoil import *

fig, ax = plt.subplots()
plt.tricontourf(
    np.array(reynolds_list),
    np.array(alpha_list),
    np.array(cl_values)
)
cbar = fig.colorbar(cs)
ax.set_xscale("log")
ax.set(xlabel="Reynolds Number", ylabel=r"$\alpha$ (angle)",
       title=r"$C_l$ Pre-Interpolated")
plt.show()
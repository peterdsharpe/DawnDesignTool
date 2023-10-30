import numpy as np
from aerosandbox.tools.pretty_plots import plt, sns, mpl, show_plot

# define lower limit of spatial resolution where there are no benefits to further improving spatial resolution
k_i = 100 # somewhere between 100 - 150 kPa m^0.5 as this is fracture toughness so this condition must be met
## TODO try stepping through a range of k_i values and see how the plot changes with time as stress intensifies
yield_strength = 100 # somewhere between 100 - 150 kPa
poissons_ratio = 0.325 #
A = 2.4 * 10 ** -24 # [Pa^-3s^-1] could be to the -25 if need be as this is melt point value and we don't expect that warm
plastic_radius = k_i ** 2 / (2 * np.pi * yield_strength ** 2) # m

spatial_resolution_lower_limit = 0.1 * plastic_radius

print("spatial resolution lower limit: ", spatial_resolution_lower_limit)

# calculate deviatoric stress and strain rate in ice around a crack
n = 100
plotting_range = 10
# Define a range of r and θ values
r_values = np.linspace(0.001, plotting_range * 1.5, n)  # Adjust the range and number of points as needed
theta_values = np.linspace(0, 2 * np.pi, n)  # Adjust the range and number of points as needed

# Create a grid of r and θ values
theta, radius = np.meshgrid(theta_values, r_values)

# for debugging
# theta = np.pi * 3 / 4
# radius = 50

# Initialize a 100x100 matrix to store the deviatoric stress values
deviatoric_stress_matrix = np.zeros((n, n))
strain_rate_matrix = np.zeros((n, n))

# Initialize arrays to store Cartesian coordinates
x_coords = np.zeros((n, n))
y_coords = np.zeros((n, n))

# Iterate through the grid and calculate deviatoric stress at each point
for i in range(n):
    for j in range(n):
        r = radius[i, j]
        t = theta[i, j]

        # Convert polar coordinates to Cartesian coordinates
        x = r * np.cos(t)
        y = r * np.sin(t)

        # Store Cartesian coordinates
        x_coords[i, j] = x
        y_coords[i, j] = y

        # Calculate the components of the Cauchy stress tensor
        cauchy_stress_tensor = k_i / np.sqrt(2 * np.pi * r) * np.array(
            [
                [np.cos(t / 2) * (1 - np.sin(t / 2) * np.sin(3 * t / 2)), np.sin(t / 2) * np.cos(t / 2) * np.cos(3 * t / 2)],
                [np.sin(t / 2) * np.cos(t / 2) * np.cos(3 * t / 2), np.cos(t / 2) * (1 + np.sin(t / 2) * np.sin(3 * t / 2))]
            ]
        )
        sigma_33 = 0 # poissons_ratio * (cauchy_stress_tensor[0, 0] + cauchy_stress_tensor[1, 1])
        cauchy_stress_tensor = np.append(cauchy_stress_tensor, [[0], [0]], axis=1)
        cauchy_stress_tensor = np.append(cauchy_stress_tensor, [[0, 0, sigma_33]], axis=0)
        # Calculate the eigenvalues and mean stress
        eigenvalues = np.linalg.eigvals(cauchy_stress_tensor)
        I_1 = np.trace(cauchy_stress_tensor)
        I_2 = 0.5 * (np.trace(cauchy_stress_tensor) ** 2 - np.trace(cauchy_stress_tensor ** 2))
        I_3 = np.linalg.det(cauchy_stress_tensor)
        pressure = (I_1) / 3

        deviatoric_stress_tensor = cauchy_stress_tensor - pressure * np.identity(3)

        # Calculate deviatoric stress at the point
        deviatoric_stress = np.sqrt((np.trace((deviatoric_stress_tensor ** 2))) / 2)

        # Store the deviatoric stress in the matrix
        deviatoric_stress_matrix[i, j] = (deviatoric_stress)
        strain_rate_matrix[i, j] =((A * 10 ** 3 * deviatoric_stress ** 3) * 3.154e+7)

# Create a contour plot of deviatoric stress in Cartesian coordinates
viridis = mpl.colormaps.get_cmap('viridis')
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[-1, :] = np.array([139/255, 0/255, 0/255, 0.5])
# newcolors[0, :] = np.array([0/255, 0/255, 0/255, 0.5])
newcmp = mpl.colors.ListedColormap(newcolors)

args = [
    x_coords,
    y_coords,
    deviatoric_stress_matrix,
]
contour_levels = np.logspace(-1, 2, 10)
kwargs = {
    "levels": contour_levels,
    "alpha" : .9,
    "extend": "both",
}
plt.figure(figsize=(8, 8))
CS = plt.contour(*args, **kwargs, colors="k", linewidths=0.2)
contour = plt.contourf(*args, **kwargs, cmap=newcmp, locator=mpl.ticker.LogLocator())
plt.plot([0, 1000], [0, 0], '-w', label='Crack Orientation')

# Add a colorbar
cbar = plt.colorbar(contour, label='Deviatoric Stress [kPa]')

# Set the title and labels
plt.title('Deviatoric Stress Contour Plot Assuming Linear-Elastic Fracture Mechanics')
plt.xlabel('X [meters]')
plt.ylabel('Y [meters]')
plt.xlim(-plotting_range, plotting_range)
plt.ylim(-plotting_range, plotting_range)
plt.legend(loc='upper left')

# Show the plot
plt.show()

args = [
    x_coords,
    y_coords,
    strain_rate_matrix,
]
kwargs = {
    "levels": np.logspace(-17, -6, 12),
    "alpha" : .9,
    "extend": "both",
}
plt.figure(figsize=(8, 8))
CS = plt.contour(*args, **kwargs, colors="k", linewidths=0.2)
contour = plt.contourf(*args, **kwargs, cmap=newcmp, locator=mpl.ticker.LogLocator())
plt.plot([0, 1000], [0, 0], '-w', label='Crack Orientation')

# Add a colorbar
cbar = plt.colorbar(contour, label='Strain Rate [1/year]')

# Set the title and labels
plt.title('Strain Rate Contour Plot')
plt.xlabel('X [meters]')
plt.ylabel('Y [meters]')
plt.xlim(-plotting_range, plotting_range)
plt.ylim(-plotting_range, plotting_range)
plt.legend(loc='upper left')

# Show the plot
plt.show()





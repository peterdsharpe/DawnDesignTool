import numpy as np
import matplotlib.pyplot as plt

# define lower limit of spatial resolution where there are no benefits to further improving spatial resolution
k_i = 100 # somewhere between 100 - 150 kPa/m^0.5
yield_strength = 100 # somewhere between 100 - 150 kPa
poissons_ratio = 0.325 #
plastic_radius = k_i ** 2 / (2 * np.pi * yield_strength ** 2)

spatial_resolution_lower_limit = 0.1 * plastic_radius

print(spatial_resolution_lower_limit)

# Define a range of r and θ values
r_values = np.linspace(0.0001, 150, 100)  # Adjust the range and number of points as needed
theta_values = np.linspace(0, 2 * np.pi, 100)  # Adjust the range and number of points as needed

# Create a grid of r and θ values
theta, radius = np.meshgrid(theta_values, r_values)
theta = np.pi * 3 / 4
radius = 50

# Initialize a 100x100 matrix to store the deviatoric stress values
deviatoric_stress_matrix = np.zeros((100, 100))

# Initialize arrays to store Cartesian coordinates
x_coords = np.zeros((100, 100))
y_coords = np.zeros((100, 100))

# Iterate through the grid and calculate deviatoric stress at each point
for i in range(100):
    for j in range(100):
        r = radius[i, j]
        t = theta[i, j]

        # Convert polar coordinates to Cartesian coordinates
        x = r * np.cos(t)
        y = r * np.sin(t)

        # Store Cartesian coordinates
        x_coords[i, j] = x
        y_coords[i, j] = y

        # Calculate the components of the Cauchy stress tensor
        cauchy_stress_tensor = k_i / (2 * np.pi * r) ** 0.5 * np.array(
            [
                [np.cos(t / 2) * (1 - np.sin(t / 2) * np.sin(3 * t / 2)), np.sin(t / 2) * np.cos(t / 2) * np.cos(3 * t / 2)],
                [np.sin(t / 2) * np.cos(t / 2) * np.cos(3 * t / 2), np.cos(t / 2) * (1 + np.sin(t / 2) * np.sin(3 * t / 2))]
            ]
        )
        sigma_33 = poissons_ratio * (cauchy_stress_tensor[0, 0] + cauchy_stress_tensor[1, 1])
        cauchy_stress_tensor = np.append(cauchy_stress_tensor, [[0], [0]], axis=1)
        cauchy_stress_tensor = np.append(cauchy_stress_tensor, [[0, 0, sigma_33]], axis=0)
        # Calculate the eigenvalues and mean stress
        eigenvalues = np.linalg.eigvals(cauchy_stress_tensor)
        I_1 = eigenvalues[0]
        I_2 = eigenvalues[1]
        I_3 = eigenvalues[2]
        pressure = (I_1) / 3
        # pressure = (cauchy_stress_tensor[0, 0] + cauchy_stress_tensor[1, 1]) / 2

        deviatoric_stress_tensor = cauchy_stress_tensor - pressure * np.identity(3)

        # Calculate deviatoric stress at the point
        deviatoric_stress = np.sqrt(
            (cauchy_stress_tensor[0, 0] - pressure) ** 2 + (cauchy_stress_tensor[1, 1] - pressure) ** 2
        )
        deviatoric_stress = (np.trace(deviatoric_stress_tensor) ** 2 - np.trace((deviatoric_stress_tensor ** 2))) / 2
        eigenvalues_tau = np.linalg.eigvals(deviatoric_stress_tensor)
        I_1_tau = eigenvalues_tau[0]
        # print(deviatoric_stress)
        I_2_tau = eigenvalues_tau[1]
        # deviatoric_stress = I_2_tau

        # Store the deviatoric stress in the matrix
        deviatoric_stress_matrix[i, j] = deviatoric_stress

# Create a contour plot of deviatoric stress in Cartesian coordinates
contour_levels = np.linspace(-10, 150, 25)  # Adjust the levels as needed
plt.figure(figsize=(8, 8))
contour = plt.contourf(x_coords, y_coords, deviatoric_stress_matrix, levels=contour_levels, cmap='viridis', vmin=-10, vmax=150)

# Add a colorbar
cbar = plt.colorbar(contour, label='Deviatoric Stress [kPa]')

# Set the title and labels
plt.title('Deviatoric Stress Contour Plot')
plt.xlabel('X [meters]')
plt.ylabel('Y [meters]')
plt.xlim(-100, 100)
plt.ylim(-100, 100)

# Show the plot
plt.show()





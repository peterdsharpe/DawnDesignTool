import numpy as np
import sys

sys.path.append("C:\\Users\\AnnickDewald\\PycharmProjects\\AeroSandbox")
from aerosandbox.modeling.interpolation import InterpolatedModel

# load the data
sigma0_ice_vals = np.load('sigma0_hh.npy', allow_pickle=True)
sigma0_soil_vals = np.load('sigma0_hh_soil.npy', allow_pickle=True)
wavelengths = np.load('wavelengths.npy', allow_pickle=True)
theta_values = np.load('theta_values.npy', allow_pickle=True)
dict = {'wavelength': wavelengths, 'look_angle': theta_values}

# create the model
sigma0_model_ice = InterpolatedModel(x_data_coordinates={'wavelength': wavelengths,
                                                     'look_angle': theta_values},
                                 y_data_structured=sigma0_ice_vals,
                                 method='bspline')
sigma0_model_soil = InterpolatedModel(x_data_coordinates={'wavelength': wavelengths,
                                                        'look_angle': theta_values},
                                    y_data_structured=sigma0_soil_vals,
                                    method='bspline')

def sigma0_ice(wavelength, look_angle):
    """
    Returns the sigma0 value for a given wavelength and look angle
    :param wavelength: radar wavelength in meters
    :param look_angle: aperture look angle in degrees
    :return: sigma0 value in dB fit to model from PSet3 in Brent's Remote Sensing Class
    """
    return sigma0_model_ice({
        'wavelength': wavelength,
        'look_angle': look_angle
    })
def sigma0_soil(wavelength, look_angle):
    """
    Returns the sigma0 value for a given wavelength and look angle
    :param wavelength: radar wavelength in meters
    :param look_angle: aperture look angle in degrees
    :return: sigma0 value in dB fit to model from PSet3 in Brent's Remote Sensing Class
    """
    return sigma0_model_soil({
        'wavelength': wavelength,
        'look_angle': look_angle
    })

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sigma0_model_val_ice = np.zeros((len(wavelengths), len(theta_values)))
    for i, wavelength in enumerate(wavelengths):
        sigma0_model_val_ice[i, :] = sigma0_ice(wavelength, theta_values)

    sigma0_model_val_soil = np.zeros((len(wavelengths), len(theta_values)))
    for i, wavelength in enumerate(wavelengths):
        sigma0_model_val_soil[i, :] = sigma0_ice(wavelength, theta_values)

    # Plotting
    fig = plt.figure(figsize=(14, 6))
    x_limits = [30, 60]
    y_limits = [0.015, 0.03]

    # Plot for sigma0_hh
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(theta_values, wavelengths)
    ax1.plot_surface(X, Y, sigma0_ice_vals, cmap='plasma')
    ax1.set_title('Sigma0 HH Ice')
    ax1.set_xlabel('Incidence Angle (degrees)')
    ax1.set_ylabel('Wavelength (m)')
    ax1.set_zlabel('Sigma0 HH (dB)')


    # Plot for sigma0_vv
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, sigma0_model_val_ice, cmap='plasma')
    ax2.set_title('Sigma0 HH Ice Fit')
    ax2.set_xlabel('Incidence Angle (degrees)')
    ax2.set_ylabel('Wavelength (m)')
    ax2.set_zlabel('Sigma0 HH Fit Value (dB)')



    plt.show()

    # Plotting
    fig = plt.figure(figsize=(14, 6))
    # x_limits = [30, 60]
    # y_limits = [0.015, 0.03]

    # Plot for sigma0_hh
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(theta_values, wavelengths)
    ax1.plot_surface(X, Y, sigma0_soil_vals, cmap='plasma')
    ax1.set_title('Sigma0 HH Soil')
    ax1.set_xlabel('Incidence Angle (degrees)')
    ax1.set_ylabel('Wavelength (m)')
    ax1.set_zlabel('Sigma0 HH (dB)')


    # Plot for sigma0_vv
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, sigma0_model_val_soil, cmap='plasma')
    ax2.set_title('Sigma0 HH Fit Soil')
    ax2.set_xlabel('Incidence Angle (degrees)')
    ax2.set_ylabel('Wavelength (m)')
    ax2.set_zlabel('Sigma0 HH Fit Value (dB)')



    plt.show()


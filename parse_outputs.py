import pandas as pd
import re

output_file = "sweeps_new_weighting3"
annotation = "airspeed"
sizing = ('look_angle')
runs = [i for i in range(1, 170 + 1)]

# pull the data from the csv output files
wingspans = []
spatial_resolutions = []
coverage_areas = []
wavelengths = []
look_angles = []
SNRs = []
payload_powers = []
airspeeds = []
cruise_altitudes = []
SAR_range_resolutions = []
SAR_azimuth_resolutions = []
swath_range = []

for run_num in runs:
    try:
        csv_file = f"outputs/{output_file}/outputs_data_run_{run_num}.csv"
        df = pd.read_csv(csv_file)
        df = df.T
        df.columns = df.iloc[0,]
        # Remove the first row (column names)
        df = df[1:]
        wingspan = df['Wing Span'].values[0]
        wingspan = re.search(r'([\d.]+)', wingspan).group(1)
        wingspans.append(float(wingspan))
        spatial_resolution = df['Strain Azimuth Resolution'].values[0]
        spatial_resolution = re.search(r'([\d.]+)', spatial_resolution).group(1)
        spatial_resolutions.append(float(spatial_resolution))
        coverage_area = df['Coverage Area'].values[0]
        coverage_areas.append(float(coverage_area.split()[0]) / 1e6)
        wavelength = df['center wavelength'].values[0]
        wavelength = re.search(r'([\d.]+)', wavelength).group(1)
        wavelengths.append(float(wavelength))
        look_angle = df['look angle'].values[0]
        look_angle = re.search(r'([\d.]+)', look_angle).group(1)
        look_angles.append(float(look_angle))
        SNR = df['SNR'].values[0]
        SNR = float(SNR.split()[0])
        SNRs.append(SNR)
        payload_power = df['payload power'].values[0]
        payload_power = re.search(r'([\d.]+)', payload_power).group(1)
        payload_powers.append(float(payload_power))
        airspeed = df['Average Airspeed'].values[0]
        airspeed = re.search(r'([\d.]+)', airspeed).group(1)
        airspeeds.append(float(airspeed))
        cruise_altitude = df['Cruise Altitude'].values[0]
        cruise_altitude = re.search(r'([\d.]+)', cruise_altitude).group(1)
        cruise_altitudes.append(float(cruise_altitude))
        SAR_range_resolution = df['SAR range resolution'].values[0]
        SAR_range_resolution = re.search(r'([\d.]+)', SAR_range_resolution).group(1)
        SAR_range_resolutions.append(float(SAR_range_resolution))
        SAR_azimuth_resolution = df['SAR azimuth resolution'].values[0]
        SAR_azimuth_resolution = re.search(r'([\d.]+)', SAR_azimuth_resolution).group(1)
        SAR_azimuth_resolutions.append(float(SAR_azimuth_resolution))
        swath = df['swath range'].values[0]
        swath = re.search(r'([\d.]+)', swath).group(1)
        swath_range.append(float(swath) / 1e3)



    except FileNotFoundError:
         pass

import matplotlib.pyplot as plt
import numpy as np
plt.close()
# Create colormap
cmap = plt.get_cmap('viridis')  # You can choose any other colormap

# Normalize wingspan values to use in colormap
norm = plt.Normalize(min(wingspans), max(wingspans))

# Create scatter plot
plt.scatter(spatial_resolutions, coverage_areas, c=wingspans, cmap=cmap, norm=norm)

# Set labels and title
plt.xlabel('Spatial Resolution [m]')
plt.ylabel('Coverage Area [km^2]')
plt.title('Pareto Front Results: Spatial Resolution vs Coverage Area')


# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Wingspan [m]')

# Show plot
plt.grid(True)
plt.show()

# Create colormap
cmap = plt.get_cmap('viridis')  # You can choose any other colormap

# Normalize wingspan values to use in colormap
norm = plt.Normalize(min(wingspans), max(wingspans))

# Create scatter plot
plt.scatter(spatial_resolutions, coverage_areas, c=wingspans, cmap=cmap, norm=norm)

if sizing == 'cruise_altitude':
    # Define size of dots based on cruise altitude values
    sizes = (np.array(cruise_altitudes) - 11.5) * 10

    # Create scatter plot
    plt.scatter(spatial_resolutions, coverage_areas, c=wingspans, cmap=cmap, norm=norm,
                s=sizes)
    plt.title('Pareto Front Results: cruise altitude demonstrated by dot size')
if sizing == "airspeed":
    # define size of dots based of airspeed values
    sizes = (np.array(airspeeds) - 14) * 20
    plt.scatter(spatial_resolutions, coverage_areas, c=wingspans, cmap=cmap, norm=norm,
                s=sizes)
    plt.title('Pareto Front Results: airspeed demonstrated by dot size')
if sizing == "payload_power":
    # Define size of dots based on payload power values
    sizes = np.array(payload_powers) * 0.2

    # Create scatter plot
    plt.scatter(spatial_resolutions, coverage_areas, c=wingspans, cmap=cmap, norm=norm,
                s=sizes)
    plt.title('Pareto Front Results: payload power demonstrated by dot size')

if sizing == "snr":
    # Define size of dots based on payload power values
    sizes = (np.array(SNRs) - 9.5) * 5

    # Create scatter plot
    plt.scatter(spatial_resolutions, coverage_areas, c=wingspans, cmap=cmap, norm=norm,
                s=sizes)
    plt.title('Pareto Front Results: SNR demonstrated by dot size')

if sizing == 'SAR_range_resolution':
    # Define size of dots based on SAR range resolution values
    sizes = (np.array(SAR_range_resolutions) - 0.1) * 100

    # Create scatter plot
    plt.scatter(spatial_resolutions, coverage_areas, c=wingspans, cmap=cmap, norm=norm,
                s=sizes)
    plt.title('Pareto Front Results: SAR range resolution demonstrated by dot size')

if sizing == 'SAR_azimuth_resolution':
    # Define size of dots based on SAR azimuth resolution values
    sizes = (np.array(SAR_azimuth_resolutions) - 0.1) * 100

    # Create scatter plot
    plt.scatter(spatial_resolutions, coverage_areas, c=wingspans, cmap=cmap, norm=norm,
                s=sizes)
    plt.title('Pareto Front Results: SAR azimuth resolution demonstrated by dot size')
if sizing == 'swath_range':
    # Define size of dots based on SAR azimuth resolution values
    sizes = (np.array(swath_range) - 40) * 1

    # Create scatter plot
    plt.scatter(spatial_resolutions, coverage_areas, c=wingspans, cmap=cmap, norm=norm,
                s=sizes)
    plt.title('Pareto Front Results: Swath Range demonstrated by dot size')
if sizing == 'look_angle':
    # Define size of dots based on SAR azimuth resolution values
    sizes = (np.array(look_angles) - 25) * 10

    # Create scatter plot
    plt.scatter(spatial_resolutions, coverage_areas, c=wingspans, cmap=cmap, norm=norm,
                s=sizes)
    plt.title('Pareto Front Results: Look Angle demonstrated by dot size')
# Set labels and title
plt.xlabel('Spatial Resolution [m]')
plt.ylabel('Coverage Area [km^2]')
# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Wingspan [m]')

# Add legend for sizing
if sizing == "payload_power":
    sizes_legend = [100, 300, 500]  # Sample payload power values for legend
    for size in sizes_legend:
        plt.scatter([], [], s=size * 0.2, label=str(size) + ' kW')

    plt.legend(title='Payload Power', labelspacing=1, loc='upper left')
if sizing == 'cruise_altitude':
    sizes_legend = [11.6, 13]  # Sample cruise altitude values for legend
    for size in sizes_legend:
        plt.scatter([], [], s=30*(size -11.5), label=str(size) + ' km')

    plt.legend(title='Cruise Altitude', labelspacing=1, loc='upper left')
if sizing == "airspeed":
    sizes_legend = [16, 19, 22]  # Sample airspeed values for legend
    for size in sizes_legend:
        plt.scatter([], [], s=20*(size - 14), label=str(size) + ' m/s')

    plt.legend(title='Airspeed', labelspacing=1, loc='upper left')
if sizing == "snr":
    sizes_legend = [15, 30]  # Sample snr values for legend
    for size in sizes_legend:
        plt.scatter([], [], s=5*(size - 9.5), label=str(size))

    plt.legend(title='SNR', labelspacing=1, loc='upper left')
if sizing == 'SAR_range_resolution':
    sizes_legend = [ 0.4, 0.8]  # Sample SAR range resolution values for legend
    for size in sizes_legend:
        plt.scatter([], [], s=100*(size - 0.1), label=str(size) + ' m')

    plt.legend(title='SAR Range Resolution', labelspacing=1, loc='upper left')
if sizing == 'SAR_azimuth_resolution':
    sizes_legend = [0.2, 0.3, 0.4]  # Sample SAR azimuth resolution values for legend
    for size in sizes_legend:
        plt.scatter([], [], s=100*(size - 0.1), label=str(size) + ' m')

    plt.legend(title='SAR Azimuth Resolution', labelspacing=1, loc='upper left')
if sizing == 'swath_range':
    sizes_legend = [100, 150, 200]  # Sample swath range values for legend
    for size in sizes_legend:
        plt.scatter([], [], s=(size - 40), label=str(size) + ' km')

    plt.legend(title='Swath Range', labelspacing=1, loc='upper left')
if sizing == 'look_angle':
    sizes_legend = [30, 40]  # Sample look angle values for legend
    for size in sizes_legend:
        plt.scatter([], [], s=10*(size - 25), label=str(size) + ' degrees')

    plt.legend(title='Look Angle', labelspacing=1, loc='upper left')
# Show plot
plt.grid(True)
plt.show()


# Create colormap
cmap = plt.get_cmap('viridis')  # You can choose any other colormap

# Normalize wingspan values to use in colormap
norm = plt.Normalize(min(wingspans), max(wingspans))

# Create scatter plot
plt.scatter(spatial_resolutions, coverage_areas, c=wingspans, cmap=cmap, norm=norm)

if annotation == 'cruise_altitude':
    # Annotate points with payload power values
    for i, txt in enumerate(cruise_altitudes):
        plt.annotate(f"{txt} km", (spatial_resolutions[i], coverage_areas[i]))
if annotation == "airspeed":
    # Annotate points with airspeed values
    for i, txt in enumerate(airspeeds):
        plt.annotate(f"{txt} m/s", (spatial_resolutions[i], coverage_areas[i]))
if annotation == "SNR":
    # Annotate points with SNR values
    for i, txt in enumerate(SNRs):
        plt.annotate(f"{txt}", (spatial_resolutions[i], coverage_areas[i]))
if annotation == "payload_power":
    # Annotate points with payload power values
    for i, txt in enumerate(payload_powers):
        plt.annotate(f"{txt} W", (spatial_resolutions[i], coverage_areas[i]))

# Set labels and title
plt.xlabel('Spatial Resolution [m]')
plt.ylabel('Coverage Area [km^2]')
plt.title('Zoomed Region: Spatial Resolution vs Coverage Area')
# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Wingspan [m]')

# Show plot
plt.grid(True)
# plt.show()


import plotly.graph_objs as go

# Create a trace for the scatter plot
trace = go.Scatter3d(
    x=wingspans,
    y=spatial_resolutions,
    z=coverage_areas,
    mode='markers',
    marker=dict(
        size=5,
        color='blue',                # Color of markers
        opacity=0.8                  # Opacity of markers
    )
)

# Define layout
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='Wingspan [m]'),
        yaxis=dict(title='Spatial Resolution [m]'),
        zaxis=dict(title='Coverage Area [km^2]')
    ),
    title='Racetrack Trajectory Sweep Results'
)

# Create figure object
fig = go.Figure(data=[trace], layout=layout)

# Display the plot in a browser
fig.show(renderer='browser')
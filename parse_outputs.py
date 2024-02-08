import pandas as pd
import re

output_file = "remote_sweeps"
runs = [i for i in range(0, 61 + 1)]

# pull the data from the csv output files
wingspans = []
spatial_resolutions = []
coverage_areas = []
wavelengths = []
look_angles = []
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
    except:
        pass

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a figure and axis object
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(wingspans, spatial_resolutions, coverage_areas, c='blue', marker='o')

# Set labels and title
ax.set_xlabel('Wingspan [m]')
ax.set_ylabel('Spatial Resolution [m]')
ax.set_zlabel('Coverage Area [km^2]')
ax.set_title('Racetrack Trajectory Sweep Results')

# Show plot
plt.show()

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
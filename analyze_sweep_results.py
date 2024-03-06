import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import sys
sys.path.append("C:\\Users\\AnnickDewald\\PycharmProjects\\AeroSandbox")
from aerosandbox.tools.pretty_plots import plt, sns, mpl, show_plot


#### inputs
file_name = 'lawnmower_pass=1_test'
variables = ['Spatial_Resolutions', 'Coverage_Areas', 'Days', 'Precisions', 'Temporal_Resolutions','Spans']
x_axis = 'Spatial_Resolutions'
y_axis = 'Coverage_Areas'
color = 'Spans'
size ='Days'
style = 'Precisions'

# Load the data
file = f'outputs\\{file_name}\\sweep_results.csv'

df = pd.read_csv(file)

# drop spaces in the column names
df.columns = df.columns.str.replace(' ', '')
df['Spans'] = df['Spans'].values.astype(float)

# drop rows with NaN values in the columns of interest
df = df.dropna(subset=['Spans'])

# Create colormap
cmap = plt.get_cmap('viridis')  # You can choose any other colormap

# Normalize wingspan values to use in colormap
norm = plt.Normalize(min(df[color]), max(df[color]))

# Create scatter plot
plt.scatter(df[x_axis], df[y_axis], c=df[color], s=df[size], cmap=cmap, norm=norm, alpha=0.7, edgecolors="w", linewidth=0.5)

if color == 'Spans':
    plt.colorbar(label='Span [m]')
if color == 'Temporal_Resolutions':
    plt.colorbar(label='Temporal Resolution [hrs]')
if color == 'Coverage_Areas':
    plt.colorbar(label='Coverage Area [m^2]')
if color == 'Days':
    plt.colorbar(label='Days')
if color == 'Precisions':
    plt.colorbar(label='Precision [1/yr]')
if color == 'Spatial_Resolutions':
    plt.colorbar(label='Spatial Resolution [m]')


if size == 'Spans':
    sizes_legend = [15, 30, 45]
    # add legend for sizes
    plt.scatter([], [], s=15, label='15')
    plt.scatter([], [], s=30, label='30')
    plt.scatter([], [], s=45, label='45')
if size == 'Temporal_Resolutions':
    sizes_legend = df[size].unique()
    # add legend for sizes
    for s in sizes_legend:
        plt.scatter([], [], s=s, label=s)
if size == 'Coverage_Areas':
    sizes_legend = df[size].unique()
    for s in sizes_legend:
        plt.scatter([], [], s=s, label=s)
if size == 'Days':
    unique_sizes = sorted(df[size].unique())
    plt.scatter(df[x_axis], df[y_axis], c=df[color], s=df[size]+20, cmap=cmap,
                norm=norm, alpha=0.7, edgecolors="w", linewidth=0.5)

    # Plot the scatter plot with the 'Size' column
    for size_value in unique_sizes:
        size_data = df[df[size] == size_value]
        plt.scatter(x=[],y=[], s=size_value+20, label=f'{int(size_value)}', color='black')

if size == 'Precisions':
    sizes_legend = df[size].unique()
    for s in sizes_legend:
        plt.scatter([], [], s=s, label=s)
if size == 'Spatial_Resolutions':
    sizes_legend = df[size].unique()
    # add legend for sizes
    for s in sizes_legend:
        plt.scatter([], [], s=s, label=s)

plt.title(f'{y_axis} vs {x_axis} with {color} as color and {size} as size')

if x_axis == 'Spans':
    plt.xlabel('Span [m]')
if x_axis == 'Temporal_Resolutions':
    plt.xlabel('Temporal Resolution [hrs]')
if x_axis == 'Coverage_Areas':
    plt.xlabel('Coverage Area [m^2]')
if x_axis == 'Days':
    plt.xlabel('Days')
if x_axis == 'Precisions':
    plt.xlabel('Precision [1/yr]')
if x_axis == 'Spatial_Resolutions':
    plt.xlabel('Spatial Resolution [m]')

if y_axis == 'Spans':
    plt.ylabel('Span [m]')
if y_axis == 'Temporal_Resolutions':
    plt.ylabel('Temporal Resolution [hrs]')
if y_axis == 'Coverage_Areas':
    plt.ylabel('Coverage Area [m^2]')
if y_axis == 'Days':
    plt.ylabel('Days')
if y_axis == 'Precisions':
    plt.ylabel('Precision [1/yr]')
if y_axis == 'Spatial_Resolutions':
    plt.ylabel('Spatial Resolution [m]')

plt.grid(True)
plt.legend(title=size)
plt.show()

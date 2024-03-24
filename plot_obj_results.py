import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import sys
sys.path.append("C:\\Users\\AnnickDewald\\PycharmProjects\\AeroSandbox")
from aerosandbox.tools.pretty_plots import plt, sns, mpl, show_plot
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.colors as mcolors

#### inputs
file_name = 'run_from_cache'
variables = ['Spatial_Resolutions', 'Coverage_Areas', 'Days', 'Precisions', 'Temporal_Resolutions', 'Spans']
x_axis = 'Coverage_Areas'
y_axis = 'Spans'
lines = 'Temporal_Resolutions'
shape = 'Spatial_Resolutions'

# size ='Days'
# style = 'Precisions'

# Load the data
file = f'outputs\\{file_name}\\sweep_results.csv'
df = pd.read_csv(file)

# drop spaces in the column names
df.columns = df.columns.str.replace(' ', '')
df['Spans'] = df['Spans'].values.astype(float)

# drop rows with NaN values in the columns of interest
df = df.dropna(subset=['Spans'])

df_span = df[df['Run'] < 7]
df_spatial = df[(df['Run'] > 7) & (df['Run'] < 15)]
df_coverage = df[(df['Run'] > 14) & (df['Run'] < 22)]
df_days = df[(df['Run'] > 21) & (df['Run'] < 29)]
df_precision = df[(df['Run'] > 28) & (df['Run'] < 36)]
df_temporal = df[(df['Run'] > 35) & (df['Run'] < 43)]

def plot_parallel_coordinates(df, attribute):
        # Add a column 'Colors' with different colors for each row
        # df['Colors'] = df['Run']

        # Convert DataFrame to dictionary
        data_dict = df.to_dict(orient='list')

        # Create a parallel coordinates plot using graph_objects
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=df['Run'], colorscale='portland'),
            dimensions=list([
                dict(label='Spatial Resolutions', values=df['Spatial_Resolutions']),
                dict(label='Precisions', values=df['Precisions']),
                dict(label='Temporal Resolutions', values=df['Temporal_Resolutions']),
                dict(label='Coverage Areas', values=df['Coverage_Areas']),
                dict(label='Days', values=df['Days']),
                dict(label='Spans', values=df['Spans'])
            ]),
            unselected=dict(line=dict(color='black', opacity=0))
        ))

        # Add labels and title
        fig.update_layout(
            xaxis=dict(title='Variables'),
            yaxis=dict(title='Values'),
            title=f'Parallel Coordinates Plot as Weighting on {attribute} Changes',
        )

        # Save the plot as an HTML file
        fig.write_html(f'C:\\Users\\AnnickDewald\\PycharmProjects\\DawnDesignTool\outputs\\{file_name}\\parallel_coordinates_plot_{attribute}.html')

        # Open the HTML file in the default web browser
        import webbrowser
        webbrowser.open(f'C:\\Users\\AnnickDewald\\PycharmProjects\\DawnDesignTool\outputs\\{file_name}\\parallel_coordinates_plot_{attribute}.html')



# plot_parallel_coordinates(df_span, 'Wing Span')
# plot_parallel_coordinates(df_spatial, 'Spatial Resolution')
# plot_parallel_coordinates(df_coverage, 'Coverage Area')
# plot_parallel_coordinates(df_days, 'Days')
# plot_parallel_coordinates(df_precision, 'Precision')
# plot_parallel_coordinates(df_temporal, 'Temporal Resolution')

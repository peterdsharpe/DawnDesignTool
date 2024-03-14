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
file_name = 'lawnmower_new_obj'
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
df['Index'] = df.index

# sweep ranges of input variables
strain_azimuth_resolutions = np.linspace(10, 100, 4)
coverage_area_requirements = np.linspace(1e6, 1e9, 5)
days_of_year = np.linspace(0, 60, 4)
required_strain_rate_precisions = np.linspace(1e-5, 1e-4, 4)
required_temporal_resolutions = np.linspace(6, 24, 3)
def outliers_time(df,strain_azimuth_resolutions, coverage_area_requirements, days_of_year, required_strain_rate_precisions):
    time_gradients = np.array([])
    for x in strain_azimuth_resolutions:
        for y in coverage_area_requirements:
            for z in days_of_year:
                for a in required_strain_rate_precisions:
                        df_time = df[(df['Spatial_Resolutions'] == x) & (df['Coverage_Areas'] == y) & (df['Days'] == z) & (df['Precisions'] == a)]
                        df_time = df_time.sort_values(by='Temporal_Resolutions')
                        if len(df_time) > 1:
                            # reset index
                            # print(df_time)
                            df_time.reset_index(drop=True, inplace=True)
                            gradient = np.gradient(df_time['Spans'], df_time['Temporal_Resolutions'])
                            time_gradients = np.append(time_gradients, gradient)
                            if (gradient == np.inf).any():
                                print(df_time)
                            if (gradient == np.nan).any():
                                print(df_time)
    time_gradients_mean = np.mean(time_gradients)
    threshold_high = np.quantile(time_gradients, 0.95)
    threshold_low = np.quantile(time_gradients, 0.05)
    time_gradients_outliers_high = time_gradients[time_gradients > threshold_high]
    time_gradients_outliers_low = time_gradients[time_gradients < threshold_low]
    print(f"Mean time gradient: {time_gradients_mean}")
    print(f"High time gradients: {time_gradients_outliers_high}")
    print(f"Low time gradients: {time_gradients_outliers_low}")
    for x in strain_azimuth_resolutions:
        for y in coverage_area_requirements:
            for z in days_of_year:
                for a in required_strain_rate_precisions:
                    df_time = df[(df['Spatial_Resolutions'] == x) & (df['Coverage_Areas'] == y) & (df['Days'] == z) & (
                                df['Precisions'] == a)]
                    df_time = df_time.sort_values(by='Temporal_Resolutions')
                    if len(df_time) > 1:
                        # reset index
                        # print(df_time)
                        df_time.reset_index(drop=True, inplace=True)
                        gradient = np.gradient(df_time['Spans'], df_time['Temporal_Resolutions'])
                        if (gradient < threshold_low).any():
                            # plot scatter plot with x axis temporal resolution and y axis spans
                            plt.plot(df_time['Temporal_Resolutions'], df_time['Spans'])
                            plt.scatter(df_time['Temporal_Resolutions'], df_time['Spans'])
                            plt.xlabel('Temporal Resolutions')
                            plt.ylabel('Spans')
                            plt.title(f'Spatial Resolution: {x}, Coverage Area: {y}, \n Day: {z}, Precision: {a}')
                            plt.show()

def outliers_day(df, strain_azimuth_resolutions, coverage_area_requirements, required_strain_rate_precisions, required_temporal_resolutions):
    # repeat for day of year
    day_gradients = np.array([])
    for x in strain_azimuth_resolutions:
        for y in coverage_area_requirements:
            for z in required_strain_rate_precisions:
                for a in required_temporal_resolutions:
                        df_day = df[(df['Spatial_Resolutions'] == x) & (df['Coverage_Areas'] == y) & (df['Precisions'] == z) & (df['Temporal_Resolutions'] == a)]
                        df_day = df_day.sort_values(by='Days')
                        if len(df_day) >1:
                            # reset index
                            # print(df_time)
                            df_day.reset_index(drop=True, inplace=True)
                            gradient = np.gradient(df_day['Spans'], df_day['Days'])
                            day_gradients = np.append(day_gradients, gradient)
    day_gradients_mean = np.mean(day_gradients)
    threshold_high = np.quantile(day_gradients, 0.95)
    threshold_low = np.quantile(day_gradients, 0.05)
    day_gradients_outliers_high = day_gradients[day_gradients > threshold_high]
    day_gradients_outliers_low = day_gradients[day_gradients < threshold_low]
    print(f"Mean day gradient: {day_gradients_mean}")
    print(f"High day gradients: {day_gradients_outliers_high}")
    print(f"Low day gradients: {day_gradients_outliers_low}")
    for x in strain_azimuth_resolutions:
        for y in coverage_area_requirements:
            for z in required_strain_rate_precisions:
                for a in required_temporal_resolutions:
                    df_day = df[(df['Spatial_Resolutions'] == x) & (df['Coverage_Areas'] == y) & (df['Precisions'] == z) & (
                                df['Temporal_Resolutions'] == a)]
                    df_day = df_day.sort_values(by='Days')
                    if len(df_day) > 1:
                        # reset index
                        # print(df_time)
                        df_day.reset_index(drop=True, inplace=True)
                        gradient = np.gradient(df_day['Spans'], df_day['Days'])
                        if (gradient > threshold_high).any():
                            # plot scatter plot with x axis temporal resolution and y axis spans
                            plt.plot(df_day['Days'], df_day['Spans'])
                            plt.scatter(df_day['Days'], df_day['Spans'])
                            plt.xlabel('Days')
                            plt.ylabel('Spans')
                            plt.title(f'Spatial Resolution: {x}, Coverage Area: {y},\n  Precision: {z}, Temporal Resolution: {a}')
                            plt.show()

def outliers_coverage(df, strain_azimuth_resolutions, days_of_year, required_strain_rate_precisions, required_temporal_resolutions):
    # repeat for coverage area
    coverage_gradients = np.array([])
    for x in strain_azimuth_resolutions:
        for y in days_of_year:
            for z in required_strain_rate_precisions:
                for a in required_temporal_resolutions:
                        df_coverage = df[(df['Spatial_Resolutions'] == x) & (df['Days'] == y) & (df['Precisions'] == z) & (df['Temporal_Resolutions'] == a)]
                        df_coverage = df_coverage.sort_values(by='Coverage_Areas')
                        if len(df_coverage) >1:
                            # reset index
                            # print(df_time)
                            df_coverage.reset_index(drop=True, inplace=True)
                            gradient = np.gradient(df_coverage['Spans'], df_coverage['Coverage_Areas'])
                            coverage_gradients = np.append(coverage_gradients, gradient)
    coverage_gradients_mean = np.mean(coverage_gradients)
    threshold_high = np.quantile(coverage_gradients, 0.95)
    threshold_low = np.quantile(coverage_gradients, 0.05)
    coverage_gradients_outliers_high = coverage_gradients[coverage_gradients > threshold_high]
    coverage_gradients_outliers_low = coverage_gradients[coverage_gradients < threshold_low]
    print(f"Mean coverage gradient: {coverage_gradients_mean}")
    print(f"High coverage gradients: {coverage_gradients_outliers_high}")
    print(f"Low coverage gradients: {coverage_gradients_outliers_low}")
    for x in strain_azimuth_resolutions:
        for y in days_of_year:
            for z in required_strain_rate_precisions:
                for a in required_temporal_resolutions:
                    df_coverage = df[(df['Spatial_Resolutions'] == x) & (df['Days'] == y) & (df['Precisions'] == z) & (
                                df['Temporal_Resolutions'] == a)]
                    df_coverage = df_coverage.sort_values(by='Coverage_Areas')
                    if len(df_coverage) > 1:
                        # reset index
                        # print(df_time)
                        df_coverage.reset_index(drop=True, inplace=True)
                        gradient = np.gradient(df_coverage['Spans'], df_coverage['Coverage_Areas'])
                        if (gradient > threshold_high).any():
                            # plot scatter plot with x axis temporal resolution and y axis spans
                            plt.plot(df_coverage['Coverage_Areas'], df_coverage['Spans'])
                            plt.scatter(df_coverage['Coverage_Areas'], df_coverage['Spans'])
                            plt.xlabel('Coverage Areas')
                            plt.ylabel('Spans')
                            plt.title(f'Spatial Resolution: {x}, Day: {y}, \n Precision: {z}, Temporal Resolution: {a}')
                            plt.show()

def outliers_precision(df, strain_azimuth_resolutions, coverage_area_requirements, days_of_year, required_temporal_resolutions):
    precision_gradients = np.array([])
    for x in strain_azimuth_resolutions:
        for y in coverage_area_requirements:
            for z in days_of_year:
                for a in required_temporal_resolutions:
                        df_precision = df[(df['Spatial_Resolutions'] == x) & (df['Coverage_Areas'] == y) & (df['Days'] == z) & (df['Temporal_Resolutions'] == a)]
                        df_precision = df_precision.sort_values(by='Precisions')
                        if len(df_precision) >1:
                            # reset index
                            # print(df_time)
                            df_precision.reset_index(drop=True, inplace=True)
                            gradient = np.gradient(df_precision['Spans'], df_precision['Precisions'])
                            precision_gradients = np.append(precision_gradients, gradient)
    precision_gradients_mean = np.mean(precision_gradients)
    threshold_high = np.quantile(precision_gradients, 0.95)
    threshold_low = np.quantile(precision_gradients, 0.05)
    precision_gradients_outliers_high = precision_gradients[precision_gradients > threshold_high]
    precision_gradients_outliers_low = precision_gradients[precision_gradients < threshold_low]
    print(f"Mean precision gradient: {precision_gradients_mean}")
    print(f"High precision gradients: {precision_gradients_outliers_high}")
    print(f"Low precision gradients: {precision_gradients_outliers_low}")
    for x in strain_azimuth_resolutions:
        for y in coverage_area_requirements:
            for z in days_of_year:
                for a in required_temporal_resolutions:
                    df_precision = df[(df['Spatial_Resolutions'] == x) & (df['Coverage_Areas'] == y) & (df['Days'] == z) & (
                                df['Temporal_Resolutions'] == a)]
                    df_precision = df_precision.sort_values(by='Precisions')
                    if len(df_precision) > 1:
                        # reset index
                        # print(df_time)
                        df_precision.reset_index(drop=True, inplace=True)
                        gradient = np.gradient(df_precision['Spans'], df_precision['Precisions'])
                        if (gradient < threshold_low).any():
                            # plot scatter plot with x axis temporal resolution and y axis spans
                            plt.plot(df_precision['Precisions'], df_precision['Spans'])
                            plt.scatter(df_precision['Precisions'], df_precision['Spans'])
                            plt.xlabel('Precisions')
                            plt.ylabel('Spans')
                            plt.title(f'Spatial Resolution: {x}, Coverage Area: {y}, \n Day: {z}, Temporal Resolution: {a}')
                            plt.show()
def outliers_spatial_resolutions(df, coverage_area_requirements, days_of_year, required_strain_rate_precisions, required_temporal_resolutions):
    spatial_gradients = np.array([])
    for x in coverage_area_requirements:
        for y in days_of_year:
            for z in required_strain_rate_precisions:
                for a in required_temporal_resolutions:
                        df_spatial = df[(df['Coverage_Areas'] == x) & (df['Days'] == y) & (df['Precisions'] == z) & (df['Temporal_Resolutions'] == a)]
                        df_spatial = df_spatial.sort_values(by='Spatial_Resolutions')
                        if len(df_spatial) >1:
                            # reset index
                            # print(df_time)
                            df_spatial.reset_index(drop=True, inplace=True)
                            gradient = np.gradient(df_spatial['Spans'], df_spatial['Spatial_Resolutions'])
                            spatial_gradients = np.append(spatial_gradients, gradient)
    spatial_gradients_mean = np.mean(spatial_gradients)
    threshold_high = np.quantile(spatial_gradients, 0.95)
    threshold_low = np.quantile(spatial_gradients, 0.05)
    spatial_gradients_outliers_high = spatial_gradients[spatial_gradients > threshold_high]
    spatial_gradients_outliers_low = spatial_gradients[spatial_gradients < threshold_low]
    print(f"Mean spatial gradient: {spatial_gradients_mean}")
    print(f"High spatial gradients: {spatial_gradients_outliers_high}")
    print(f"Low spatial gradients: {spatial_gradients_outliers_low}")
    for x in coverage_area_requirements:
        for y in days_of_year:
            for z in required_strain_rate_precisions:
                for a in required_temporal_resolutions:
                    df_spatial = df[(df['Coverage_Areas'] == x) & (df['Days'] == y) & (df['Precisions'] == z) & (
                                df['Temporal_Resolutions'] == a)]
                    df_spatial = df_spatial.sort_values(by='Spatial_Resolutions')
                    if len(df_spatial) > 1:
                        # reset index
                        df_spatial.reset_index(drop=True, inplace=True)
                        gradient = np.gradient(df_spatial['Spans'], df_spatial['Spatial_Resolutions'])
                        if (gradient < threshold_low).any():
                            # plot scatter plot with x axis temporal resolution and y axis spans
                            plt.plot(df_spatial['Spatial_Resolutions'], df_spatial['Spans'])
                            plt.scatter(df_spatial['Spatial_Resolutions'], df_spatial['Spans'])
                            plt.xlabel('Spatial Resolutions')
                            plt.ylabel('Spans')
                            plt.title(f'Coverage Area: {x}, Day: {y}, \n Precision: {z}, Temporal Resolution: {a}')
                            plt.show()
def plot_outlier_spans(df, quantile=0.95):
        # Find the outliers in span that are in the top 5% of the data
        threshold = df['Spans'].quantile(quantile)
        df_outliers = df[df['Spans'] > threshold]
        df_outliers = df_outliers.drop(columns=['Index'])
        df_outliers.reset_index(drop=True, inplace=True)

        # Add a column 'Colors' with different colors for each row
        df_outliers['Colors'] = df_outliers.index

        # Convert DataFrame to dictionary
        data_dict = df_outliers.to_dict(orient='list')

        # Create a parallel coordinates plot using graph_objects
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=df_outliers['Colors'], colorscale='portland'),
            dimensions=list([
                dict(label='Spatial Resolutions', values=df_outliers['Spatial_Resolutions']),
                dict(label='Coverage Areas', values=df_outliers['Coverage_Areas']),
                dict(label='Days', values=df_outliers['Days']),
                dict(label='Precisions', values=df_outliers['Precisions']),
                dict(label='Temporal Resolutions', values=df_outliers['Temporal_Resolutions']),
                dict(label='Spans', values=df_outliers['Spans'])
            ]),
            unselected=dict(line=dict(color='black', opacity=0))
        ))

        # Add labels and title
        fig.update_layout(
            xaxis=dict(title='Variables'),
            yaxis=dict(title='Values'),
            title='Parallel Coordinates Plot'
        )

        # Save the plot as an HTML file
        fig.write_html('parallel_coordinates_plot.html')

        # Open the HTML file in the default web browser
        import webbrowser
        webbrowser.open('parallel_coordinates_plot.html')


def plot_grouped_data(df, x_axis, y_axis, lines, shape, variables):
    # Group the data by unique combinations of all variables except x_axis, y_axis, lines, and shape
    grouped = df.groupby([var for var in variables if var not in [x_axis, y_axis, lines, shape]])

    # Get unique values of shape variable
    unique_shapes = df[shape].unique()

    # Map each unique shape value to a different marker style
    marker_styles = {shape_value: marker for shape_value, marker in zip(unique_shapes, plt.Line2D.filled_markers)}

    # Plot the grouped data
    for name, group in grouped:
        fig, ax = plt.subplots()
        for line_value, line_group in group.groupby(lines):
            for shape_value, shape_group in line_group.groupby(shape):
                marker = marker_styles.get(shape_value, 'o')  # Default to 'o' if shape value is not in the dictionary
                ax.scatter(
                    shape_group[x_axis],
                    shape_group[y_axis],
                    label=f"{lines} = {line_value}, {shape} = {shape_value}",
                    marker=marker
                )
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        title_vars = [f"{var}={val}" for var, val in zip(variables, name) if var not in [x_axis, y_axis, lines, shape]]
        ax.set_title(', '.join(title_vars))
        ax.legend(title=shape)
        plt.show()


plot_outlier_spans(df, quantile=0)
# outliers_time(df,strain_azimuth_resolutions, coverage_area_requirements, days_of_year, required_strain_rate_precisions)
# outliers_day(df, strain_azimuth_resolutions, coverage_area_requirements, required_strain_rate_precisions, required_temporal_resolutions)
# outliers_coverage(df, strain_azimuth_resolutions, days_of_year, required_strain_rate_precisions, required_temporal_resolutions)
# outliers_precision(df, strain_azimuth_resolutions, coverage_area_requirements, days_of_year, required_temporal_resolutions)
# outliers_spatial_resolutions(df, coverage_area_requirements, days_of_year, required_strain_rate_precisions, required_temporal_resolutions)
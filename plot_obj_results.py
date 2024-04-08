import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import sys
sys.path.append("C:\\Users\\AnnickDewald\\PycharmProjects\\AeroSandbox")
from aerosandbox.tools.pretty_plots import plt, sns, mpl, show_plot
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

#### inputs
file_name = 'sweep_area_n_free'
variables = ['Spatial_Resolutions', 'Coverage_Areas', 'Days', 'Precisions', 'Temporal_Resolutions', 'Spans']
# run_type = 'sweep all'
run_type = 'sweep requirement'
attribute = 'Coverage_Areas'

# Load the data
file = f'outputs\\{file_name}\\sweep_results.csv'
weighting_file = f'outputs\\{file_name}\\0_parameter_combinations.csv'
df = pd.read_csv(file)

# load the weighting file
df_weighting = pd.read_csv(weighting_file, header = None)
if run_type == 'sweep all':

    df_weighting.columns = ['Spans', 'Spatial Resolutions', 'Coverage Areas', 'Days', 'Precisions', 'Temporal Resolutions']
    # find runs to split into different dataframes
    rows_with_zero = df_weighting[(df_weighting == 0).any(axis=1)]

    # drop spaces in the column names
    df.columns = df.columns.str.replace(' ', '')

    # make sure all values are floats
    df['Spans'] = df['Spans'].values.astype(float)
    df['Spatial_Resolutions'] = df['Spatial_Resolutions'].values.astype(float)
    df['Coverage_Areas'] = df['Coverage_Areas'].values.astype(float)
    df['Days'] = df['Days'].values.astype(float)
    df['Precisions'] = df['Precisions'].values.astype(float)
    df['Temporal_Resolutions'] = df['Temporal_Resolutions'].values.astype(float)

    # drop rows with NaN values in the columns of interest
    df = df.dropna(subset=['Spans'])
    df = df.sort_values(by=['Run'])

# df formatting
df.loc[:, 'Coverage_Areas'] = df['Coverage_Areas'].div(1e6).round(2)
df.columns = ['Run', 'Span [m]', 'Spatial_Resolution [m]', 'Coverage_Area [km^2]', 'Duration [days]', 'Precision [1/yr]', 'Temporal_Resolution [hrs]']



    # record the output for the all objectives weighted equally case
    even_weighting = df[df['Run'] == 0]

    # make span df
    df_span = df[df['Run'] < rows_with_zero.index[0]]
    df_span = df_span.set_index('Run')
    df_span.loc[:, 'weight'] = df_weighting.loc[df_span.index, 'Spans']
    df_span = df_span.sort_values(by=['weight'])

    # make spatial df
    df_spatial = df[(df['Run'] > rows_with_zero.index[0] + 1) & (df['Run'] < rows_with_zero.index[1])]
    df_spatial = pd.concat([even_weighting, df_spatial])
    df_spatial = df_spatial.set_index('Run')
    df_spatial.loc[:, 'weight'] = df_weighting.loc[df_spatial.index, 'Spatial Resolutions']
    df_spatial = df_spatial.sort_values(by=['weight'])

    # make coverage df
    df_coverage = df[(df['Run'] > rows_with_zero.index[1] + 1) & (df['Run'] < rows_with_zero.index[2])]
    df_coverage = pd.concat([even_weighting, df_coverage])
    df_coverage = df_coverage.set_index('Run')
    df_coverage.loc[:, 'weight'] = df_weighting.loc[df_coverage.index, 'Coverage Areas']
    df_coverage = df_coverage.sort_values(by=['weight'])

    # make days df
    df_days = df[(df['Run'] > rows_with_zero.index[2] + 1) & (df['Run'] < rows_with_zero.index[3])]
    df_days = pd.concat([even_weighting, df_days])
    df_days = df_days.set_index('Run')
    df_days.loc[:, 'weight'] = df_weighting.loc[df_days.index, 'Days']
    df_days = df_days.sort_values(by=['weight'])

    # make precision df
    df_precision = df[(df['Run'] > rows_with_zero.index[3] + 1) & (df['Run'] < rows_with_zero.index[4])]
    df_precision = pd.concat([even_weighting, df_precision])
    df_precision = df_precision.set_index('Run')
    df_precision.loc[:, 'weight'] = df_weighting.loc[df_precision.index, 'Precisions']
    df_precision = df_precision.sort_values(by=['weight'])

    # make temporal df
    df_temporal = df[(df['Run'] > rows_with_zero.index[4] + 1) & (df['Run'] < rows_with_zero.index[5])]
    df_temporal = pd.concat([even_weighting, df_temporal])
    df_temporal = df_temporal.set_index('Run')
    df_temporal.loc[:, 'weight'] = df_weighting.loc[df_temporal.index, 'Temporal Resolutions']
    df_temporal = df_temporal.sort_values(by=['weight'])

    df = df.set_index('Run')

    # drop ones with only single attribute weighted
    df = df[~df.index.isin(rows_with_zero.index)]

    # what runs converged?
    converged_runs = df.index.values
if run_type == 'sweep requirement':
    # drop spaces in the column names
    df.columns = df.columns.str.replace(' ', '')

    # make sure all values are floats
    df['Spans'] = df['Spans'].values.astype(float)
    df['Spatial_Resolutions'] = df['Spatial_Resolutions'].values.astype(float)
    df['Coverage_Areas'] = df['Coverage_Areas'].values.astype(float)
    df['Days'] = df['Days'].values.astype(float)
    df['Precisions'] = df['Precisions'].values.astype(float)
    df['Temporal_Resolutions'] = df['Temporal_Resolutions'].values.astype(float)

    # drop rows with NaN values in the columns of interest
    df = df.dropna(subset=['Spans'])
    df = df.sort_values(by=['Run'])

    # df formatting
    df.loc[:, 'Coverage_Areas'] = df['Coverage_Areas'].div(1e6).round(2)
    df.columns = ['Run', 'Span [m]', 'Spatial Resolution [m]', 'Coverage Area [km^2]', 'Duration [days]',
                  'Precision [1/yr]', 'Temporal Resolution [hrs]']

    df = df.set_index('Run')

    # what runs converged?
    converged_runs = df.index.values


def load_variables(output_file, runs):

    # pull the additional variables from the csv output files
    wavelengths = []
    look_angles = []
    SNRs = []
    payload_powers = []
    airspeeds = []
    cruise_altitudes = []
    SAR_range_resolutions = []
    SAR_azimuth_resolutions = []
    swath_ranges = []
    swath_azimuths = []
    payload_masses = []
    bandwidths = []
    aperture_widths = []
    aperture_lengths = []
    mass_TOGWs = []
    coverage_lengths = []
    coverage_widths = []
    revisit_period = []
    swath_overlap = []
    PRFs = []
    battery_mass = []

    for run_num in runs:
        try:
            csv_file = f"outputs/{output_file}/outputs_data_run_{run_num}.csv"
            df = pd.read_csv(csv_file)
            df = df.T
            df.columns = df.iloc[0,]
            # Remove the first row (column names)
            df = df[1:]
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
            swath_range = df['swath range'].values[0]
            swath_range = re.search(r'([\d.]+)', swath_range).group(1)
            swath_ranges.append(float(swath_range) / 1e3)
            swath_azimuth = df['swath azimuth'].values[0]
            swath_azimuth = re.search(r'([\d.]+)', swath_azimuth).group(1)
            swath_azimuths.append(float(swath_azimuth) / 1e3)
            payload_mass = df['payload mass'].values[0]
            payload_mass = re.search(r'([\d.]+)', payload_mass).group(1)
            payload_masses.append(float(payload_mass))
            bandwidth = df['bandwidth'].values[0]
            bandwidth = float(bandwidth.split()[0])
            bandwidths.append(float(bandwidth))
            aperture_width = df['aperture width'].values[0]
            aperture_width = re.search(r'([\d.]+)', aperture_width).group(1)
            aperture_widths.append(float(aperture_width))
            aperture_length = df['aperture length'].values[0]
            aperture_length = re.search(r'([\d.]+)', aperture_length).group(1)
            aperture_lengths.append(float(aperture_length))
            mass_TOGW = df['mass_TOGW'].values[0]
            mass_TOGW = re.search(r'([\d.]+)', mass_TOGW).group(1)
            mass_TOGWs.append(float(mass_TOGW))
            coverage_length = df['coverage length'].values[0]
            coverage_length = re.search(r'([\d.]+)', coverage_length).group(1)
            coverage_lengths.append(float(coverage_length) / 1e3)
            coverage_width = df['coverage width'].values[0]
            coverage_width = re.search(r'([\d.]+)', coverage_width).group(1)
            coverage_widths.append(float(coverage_width) / 1e3)
            revisit = df['SAR revisit period'].values[0]
            revisit = re.search(r'([\d.]+)', revisit).group(1)
            revisit_period.append(float(revisit))
            swath_overlap_ = df['swath overlap'].values[0]
            swath_overlap_ = float(swath_overlap_.split()[0])
            swath_overlap.append(swath_overlap_)
            PRF = df['pulse repetition frequency'].values[0]
            PRF = float(PRF.split()[0])
            PRFs.append(PRF)
            battery = df['battery_pack'].values[0]
            battery = re.search(r'([\d.]+)', battery).group(1)
            battery_mass.append(float(battery))

        except FileNotFoundError:
            pass

    # make dict
    dict = {'Run': runs, 'Wavelength [m]': wavelengths, 'Look_Angle [deg]': look_angles, 'SNR [dB]': SNRs, 'Payload Power [W]': payload_powers,
            'Average Airspeed [m/s]': airspeeds, 'Cruise Altitude [km]': cruise_altitudes, 'SAR Range Resolution [m]': SAR_range_resolutions,
            'SAR Azimuth Resolution [m]': SAR_azimuth_resolutions, 'Swath Range [km]': swath_ranges, 'Swath Azimuth [km]': swath_azimuths,
            'Payload Mass [kg]': payload_masses, 'Bandwidth [Hz]': bandwidths, 'Aperture Width [m]': aperture_widths, 'Aperture Length [m]': aperture_lengths,
            'Mass TOGW [kg]': mass_TOGWs, 'Coverage Length [km]': coverage_lengths, 'Coverage Width [km]': coverage_widths, 'Revisit Period [hrs]': revisit_period,
            'Swath Overlap': swath_overlap, 'Pulse Repetition Frequency [Hz]': PRFs, 'Battery Mass [kg]': battery_mass}

    # create dataframe with all the variables of interest
    design_variables = pd.DataFrame(dict)
    design_variables = design_variables.set_index('Run')

    return design_variables


design_variable_df = load_variables(file_name, converged_runs)
big_df = pd.concat([df, design_variable_df], axis=1)
big_df_spans = big_df[big_df.index.isin(df_span.index)]
big_df_spatial = big_df[big_df.index.isin(df_spatial.index)]
big_df_coverage = big_df[big_df.index.isin(df_coverage.index)]
big_df_days = big_df[big_df.index.isin(df_days.index)]
big_df_precision = big_df[big_df.index.isin(df_precision.index)]
big_df_temporal = big_df[big_df.index.isin(df_temporal.index)]

def plot_parallel_coordinates(df, attribute):
        # Add a column 'Colors' with different colors for each row
        # df['Colors'] = df['Run']
        try:
            df = df.drop(columns = ['weight'])
        except:
            pass
        df = df.reset_index()
        df = df.drop(columns=['Run'])

        # Convert DataFrame to dictionary
        data_dict = df.to_dict(orient='list')
        list = []
        for i in df.columns:
            if i == 'Coverage_Area [km^2]':
                values = [ - x for x in data_dict[i]]
                list.append(dict(range =[round(min(values) - 1), round(max(values) + 1)], label='- Coverage_Areas [1/km^2]', values=values))
            elif i == 'Duration [days]':
                values = [- x for x in data_dict[i]]
                list.append(dict(range =[round(min(values) - 1), 0], label='- Duration [1/days]', values=values))
            elif i == 'Precision [1/yr]':
                list.append(dict(range =[round(min(data_dict[i]) - 1e-6, 6), round(max(data_dict[i]), 6)  + 1e-6], label='Precisions [1/yr]', values=data_dict[i]))
            else:
                list.append(dict(range =[round(min(data_dict[i]) - 1), round(max(data_dict[i])+ 1)],label=i, values=data_dict[i]))

        # Create a parallel coordinates plot using graph_objects
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=df.index, colorscale='hsv'),
            dimensions=list,
            unselected=dict(line=dict(color='black', opacity=0.2))
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



def spider(df, *, id_column, title=None, max_values=None, padding=1.25):
    categories = df.dtypes[(df.dtypes == 'float') | (df.dtypes == 'int')].index.tolist()
    categories = categories[:-1]
    data = df[categories].to_dict(orient='list')
    ids = df[id_column].tolist()
    if max_values is None:
        max_values = {key: padding * max(value) for key, value in data.items()}

    normalized_data = {key: np.array(value) / max_values[key] for key, value in data.items()}
    num_vars = len(data.keys())
    tiks = list(data.keys())
    tiks += tiks[:1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    # If you want the first axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]
        values += values[:1]  # Close the plot for a better look
        ax.plot(angles, values, label=f'Weighting {model_name}')
        ax.fill(angles, values, alpha=0.15)
        for _x, _y, t in zip(angles, values, actual_values):
            t = f'{t:.2f}' if isinstance(t, float) else str(t)
            ax.text(_x, _y, t, size='xx-small')

    ax.fill(angles, np.ones(num_vars + 1), alpha=0.05)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(tiks)
    ax.legend(loc='upper right', bbox_to_anchor=(0.05, 0.1))
    if title is not None: plt.suptitle(f"Effects of increasing weighting on {title} in objective function")
    plt.tight_layout()
    plt.show()


def plot_radar(df, attribute):
    # df = df.drop([27])
    df.loc[:, 'Coverage_Areas'] = df['Coverage_Areas'].div(1e6).round(2)
    # df.loc[:, 'Coverage_Areas'] = 1 / df['Coverage_Areas']
    df.loc[:, 'Precisions'] = df['Precisions'].multiply(1e6).round(2)
    # df.loc[:, 'Days'] = 1 /  df['Days']

    columns = ['Spans [m]', 'Spatial Resolutions [m]', 'Coverage Areas [km^2]', 'Duration [days]', 'Precisions [1e6/yr]', 'Temporal Resolutions [hrs]', 'Weight']
    df.columns = columns

    spider(
        df,
        id_column='Weight',
        title=attribute,
        padding=1.1,
    )


def plot_scatter_matrix(df):

    sns.set_theme(style="ticks")

    sns.pairplot(df, diag_kind='kde', plot_kws={'color': 'black'})
    sns.pairplot(df, plot_kws={'color': 'black'})
    plt.show()

def corr_heatmap(df):
    # Correlation Matrix Heatmap
    f, ax = plt.subplots(figsize=(10, 6))
    corr = df.corr()
    hm = sns.heatmap(round(corr, 2), annot=False, ax=ax, vmax=1, vmin=-1, cmap="coolwarm", fmt='.2f',
                     linewidths=.05)
    f.subplots_adjust(top=0.93)
    plt.tight_layout()

    plt.show()

def histogram(df):
    # Plot histograms for each column

    # first check distribution of the data
    std_columns = []
    for column in df.columns:
        std = df[column].std()
        mean = df[column].mean()
        if std/mean < 1e-3:
            std_columns.append(column)

    # Calculate the number of rows and columns needed for subplots
    num_cols = len(df.columns)
    num_rows = (num_cols - 1) // 5 + 1  # Ensure at least 1 row

    # Create subplots
    fig, axes = plt.subplots(num_rows, 5, figsize=(20, num_rows * 4))
    axes = axes.flatten()

    # Plot histograms for each column
    for i, col in enumerate(df.columns):
        if col in std_columns:  # Adjust settings for specific column
            # sns.histplot(df[col], ax=axes[i], kde=True, bins=1)  # Adjust bin size
            sns.histplot(df[col], ax=axes[i], kde=True, bins=10)
            axes[i].set_title(col)
        else:  # Adjust settings for another specific column
            sns.histplot(df[col], ax=axes[i], kde=True, bins=10)
            axes[i].set_title(col)

    # Remove extra empty subplots if necessary
    for j in range(num_cols, num_rows * 5):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()

def show_changes(df, attribute):
    # plot bar chart
    # df = df.drop(columns = ['weight'])
    # Calculate the number of rows and columns needed for subplots
    num_cols = len(df.columns)
    num_rows = (num_cols - 1) // 5 + 1  # Ensure at least 1 row

    # Create subplots
    fig, axes = plt.subplots(num_rows, 5, figsize=(20, num_rows * 4))
    axes = axes.flatten()

    # Plot bar chart for each column
    for i, col in enumerate(df.columns):
            sns.scatterplot(x=df[attribute], y=col, data=df, ax=axes[i], s=100)
            axes[i].set_title(col)
            axes[i].set_xlabel(attribute)
            axes[i].set_ylabel(col)

    # Remove extra empty subplots if necessary
    for j in range(num_cols, num_rows * 5):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()


# plot_parallel_coordinates(df, 'all')
plot_parallel_coordinates(df_span, 'Wing Span')
plot_parallel_coordinates(df_spatial, 'Spatial Resolution')
plot_parallel_coordinates(df_coverage, 'Coverage Area')
plot_parallel_coordinates(df_days, 'Days')
plot_parallel_coordinates(df_precision, 'Precision')
plot_parallel_coordinates(df_temporal, 'Temporal Resolution')
# plot_radar(df_span, 'wingspan')
# plot_radar(df_spatial, 'Spatial Resolution')
# plot_radar(df_coverage, 'Coverage Area')
# plot_radar(df_days, 'Duration')
# plot_radar(df_precision, 'Precision')
# plot_radar(df_temporal, 'Temporal Resolution')
# # plot_scatter_matrix(df)
# # corr_heatmap(df)
# # corr_heatmap(big_df)
# big_df = big_df.drop(columns = ['Wavelength [m]'])
# histogram(big_df)
#
# big_df = big_df.drop(columns = ['Swath Overlap', 'Pulse Repetition Frequency [Hz]'])
# corr_heatmap(big_df)
# show_changes(big_df_spans, 'Span [m]')
# show_changes(big_df_spatial, 'Spatial_Resolution [m]')
# show_changes(big_df_coverage, 'Coverage_Area [km^2]')
# show_changes(big_df_days, 'Duration [days]')
# show_changes(big_df_precision, 'Precision [1/yr]')
# show_changes(big_df_temporal, 'Temporal_Resolution [hrs]')



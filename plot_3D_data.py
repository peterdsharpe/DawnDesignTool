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
import itertools
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, no_update, callback


#### inputs
file_name = ('lawnmower=4_antarctica')
variables = ['Spatial_Resolutions', 'Coverage_Areas', 'Duration', 'Precisions', 'Temporal_Resolutions', 'Spans']
# run_type = 'parameter_sweep'
run_type = 'objective_sweep'

#### read data
# Load the data
file = f'outputs\\{file_name}\\sweep_results.csv'
weighting_file = f'outputs\\{file_name}\\parameter_combinations.csv'
df = pd.read_csv(file)
num_of_racetracks = ''.join([n for n in file_name if n.isdigit()])

# # load the weighting file
# df_weighting = pd.read_csv(weighting_file, header=None)
#
# df_weighting.columns = ['Spatial_Resolutions', 'Coverage_Areas', 'Days', 'Precisions', 'Temporal_Resolutions', 'Spans']

# drop spaces in the column names
df.columns = df.columns.str.replace(' ', '')

# make sure all values are floats
df['Spans'] = df['Spans'].values.astype(float)
df['Spatial_Resolutions'] = df['Spatial_Resolutions'].values.astype(float)
df['Coverage_Areas'] = df['Coverage_Areas'].values.astype(float)
df['Duration'] = df['Duration'].values.astype(float)
df['Precisions'] = df['Precisions'].values.astype(float)
df['Temporal_Resolutions'] = df['Temporal_Resolutions'].values.astype(float)

# drop rows with NaN values in the columns of interest
df = df.dropna(subset=['Spans'])
df = df.sort_values(by=['Run'])


# df formatting
df.loc[:, 'Coverage_Areas'] = df['Coverage_Areas'].div(1e6).round(2)
df.columns = ['Run', 'Wingspan [m]', 'Spatial Resolution [m]', 'Spatial Coverage [km^2]', 'Temporal Coverage [days]',
              'Precision [yr^-1]', 'Temporal Resolution [hrs]']
df = df[['Run', 'Wingspan [m]', 'Spatial Resolution [m]', 'Spatial Coverage [km^2]',
         'Temporal Resolution [hrs]', 'Temporal Coverage [days]', 'Precision [yr^-1]']]

df = df.set_index('Run')

# what runs converged?
converged_runs = df.index.values

df['Spatial Coverage [km^2]'] = df['Spatial Coverage [km^2]'].round(0)
df['Temporal Coverage [days]'] = df['Temporal Coverage [days]'].round(2)
df['Precision [yr^-1]'] = df['Precision [yr^-1]'].round(8)
df['Temporal Resolution [hrs]'] = df['Temporal Resolution [hrs]'].round(2)
df['Spatial Resolution [m]'] = df['Spatial Resolution [m]'].round(2)
df['Wingspan [m]'] = df['Wingspan [m]'].round(2)


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
    number_of_racetracks = []
    offset_distances = []
    max_power_outputs = []
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
            number_of_racetrack = df['Number of Racetracks'].values[0]
            number_of_racetracks.append(float(number_of_racetrack))
            offset_distance = df['imaging offset'].values[0]
            offset_distance = re.search(r'([\d.]+)', offset_distance).group(1)
            offset_distances.append(float(offset_distance) / 1e3)
            max_power_output = df['max_power_out'].values[0]
            max_power_output = re.search(r'([\d.]+)', max_power_output).group(1)
            max_power_outputs.append(float(max_power_output)/ 1e3)



        except FileNotFoundError:
            pass

    # make dict
    dict = {'Run': runs, 'SNR [dB]': SNRs,

            'Mass TOGW [kg]': mass_TOGWs, 'Payload Power [W]': payload_powers, 'Payload Mass [kg]': payload_masses,
            'Battery Mass [kg]': battery_mass, 'Average Airspeed [m/s]': airspeeds,
            'Cruise Altitude [km]': cruise_altitudes, 'Revisit Period [hrs]': revisit_period,
            'Max Power Output [kW]': max_power_outputs,

            'Number of Racetracks': number_of_racetracks, 'Imaging Offset Distance [km]': offset_distances,
            'Look Angle [deg]': look_angles, 'Swath Overlap': swath_overlap, 'Swath Range Dimension [km]': swath_ranges,
            'Swath Azimuth Dimension [km]': swath_azimuths,

            'SAR Range Resolution [m]': SAR_range_resolutions, 'SAR Azimuth Resolution [m]': SAR_azimuth_resolutions,
            'Wavelength [m]': wavelengths, 'Bandwidth [Hz]': bandwidths, 'Pulse Repetition Frequency [Hz]': PRFs,
            'Aperture Width [m]': aperture_widths, 'Aperture Length [m]': aperture_lengths}

    # create dataframe with all the variables of interest
    design_variables = pd.DataFrame(dict)
    design_variables = design_variables.set_index('Run')

    return design_variables

def check_optimality(df_all):
    try:
        df['Temporal Coverage [days]'] = -df['Temporal Coverage [days]']
        df['Spatial Coverage [km^2]'] = -df['Spatial Coverage [km^2]']
    except:
        pass
    costs = df.values
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        # Check if solution i is dominated by any other solution
        is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(np.any(costs[i+1:] > c, axis=1))
        if is_efficient[i] == False:
            df_all = df_all.drop([i])

    return df_all
def plot_scatter_matrix(df):

    # sns.set_theme(style="ticks")
    # Create a PairGrid
    g = sns.PairGrid(df)

    # Map the upper and lower triangle to scatter plots
    g.map_upper(sns.scatterplot, color='black')
    g.map_lower(sns.scatterplot, color='black')

    # Define a custom function for the diagonal
    def custom_hist(x, **kwargs):
        bins = kwargs.pop('bins', 10)
        plt.hist(x, bins=bins, **kwargs)

    # Map the diagonal to the custom function
    g.map_diag(custom_hist, bins=10)
    # g = sns.pairplot(df, diag_kind='auto', plot_kws={'color': 'black'})
    # g = sns.pairplot(df, plot_kws={'color': 'black'})
    if 'Wingspan [m]' in df.columns:
        g.fig.axes[2].invert_xaxis()
        g.fig.axes[4].invert_xaxis()
        g.fig.axes[12].invert_yaxis()
        g.fig.axes[24].invert_yaxis()
    else:
        g.fig.axes[1].invert_xaxis()
        g.fig.axes[3].invert_xaxis()
        g.fig.axes[5].invert_yaxis()
        g.fig.axes[15].invert_yaxis()
    plt.show()

def plot_parallel_coordinates(df, attribute, n):
    # Add a column 'Colors' with different colors for each row
    # df['Colors'] = df['Run']
    try:
        df = df.drop(columns=['weight'])
    except:
        pass
    if attribute[0] == '-':
        df = df.sort_values(by=attribute[1:])
    elif attribute == 'all':
        pass
    else:
        df = df.sort_values(by=attribute)
    df = df.reset_index(drop=True)
    # Convert DataFrame to dictionary
    data_dict = df.to_dict(orient='list')

    # Define color palette based on the number of rows in df
    color_palette = sns.color_palette('turbo', len(df))
    color_palette = [(r * 255, g * 255, b * 255) for r, g, b in color_palette]

    # Normalize color level values
    normalized_values = [i / (len(df) - 1) for i in range(len(df))]

    # Create a list of 2-element lists with normalized values and color string
    colorscale = [[val, f'rgb{rgb_tuple}'] for val, rgb_tuple in zip(normalized_values, color_palette)]

    list = []
    for i in df.columns:
        if i == 'Spatial Coverage [km^2]':
            values = [- x for x in data_dict[i]]
            list.append(dict(range=[round(min(values), 0), 0], label='-Spatial Coverage [km^2]', values=values))
        elif i == 'Temporal Coverage [days]':
            values = [- x for x in data_dict[i]]
            list.append(dict(range=[round(min(values)), 0], label='-Temporal Coverage [days]', values=values))
        elif i == 'Precision [yr^-1]':
            list.append(dict(range=[0, 1.005E-4], label='Precision [yr^-1]', values=data_dict[i]))
        elif i == 'Temporal Resolution [hrs]':
            list.append(dict(range=[0, 24], label='Temporal Resolution [hrs]', values=data_dict[i]))
        elif i == 'Spatial Resolution [m]':
            list.append(dict(range=[0, 50], label='Spatial Resolution [m]', values=data_dict[i]))
        else:
            list.append(dict(range=[0, round(max(data_dict[i]) + 1)], label=i, values=data_dict[i]))

    # bring attribute to the top of the list
    for i in range(len(list)):
        if list[i]['label'] == attribute:
            list.insert(0, list.pop(i))
            break

    # Create a parallel coordinates plot using graph_objects
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=df.index, colorscale=colorscale),  # Use the same color palette
        dimensions=list,
        labelfont=dict(size=20),
        rangefont=dict(size=1),
        tickfont=dict(size=16),
        unselected=dict(line=dict(color='black', opacity=0.2))
    ))

    # Add labels and title
    fig.update_layout(
        xaxis=dict(title='Variables'),
        yaxis=dict(title='Values'),
        # title=f'Parallel Coordinates Plot as Weighting on {attribute} Changes with {n} Racetracks',
        autosize=False,  # Disable autosizing
        width=1800,  # Set width of the plot
        height=1000,  # Set height of the plot
        margin=dict(l=140, r=130, t=80, b=40),  # Adjust margins
    )

    # Save the plot as an HTML file
    fig.write_html(
        f'C:\\Users\\AnnickDewald\\PycharmProjects\\DawnDesignTool\\outputs\\{file_name}\\parallel_coordinates_plot_{attribute}.html')

    # Open the HTML file in the default web browser
    import webbrowser
    webbrowser.open(
        f'C:\\Users\\AnnickDewald\\PycharmProjects\\DawnDesignTool\\outputs\\{file_name}\\parallel_coordinates_plot_{attribute}.html')



def plot_3D( x, y, z):
    # plot the data in three dimensions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(x, y, z)
    # sc = ax.scatter(df[x], df[y], df[z], c=df[z], cmap='viridis', marker='o')
    if z == 'Spatial Coverage [km^2]' or z == 'Temporal Coverage [days]':
        sc = ax.scatter( df[x], df[y],df[z], c=df[z], cmap='inferno')
    else:
        sc = ax.scatter(df[x], df[y],df[z], c=df[z], cmap='inferno_r')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z, labelpad=10)
    fig.colorbar(sc, label=z, pad=0.1)

    utopia_x = df[x].min()
    utopia_y = df[y].min()
    utopia_z = df[z].min()

    if x == 'Spatial Coverage [km^2]' or x == 'Temporal Coverage [days]':
        ax.invert_xaxis()
        utopia_x = df[x].max()
    if y == 'Spatial Coverage [km^2]' or y == 'Temporal Coverage [days]':
        ax.invert_yaxis()
        utopia_y = df[y].max()
    if z == 'Spatial Coverage [km^2]' or z == 'Temporal Coverage [days]':
        ax.invert_zaxis()
        utopia_z = df[z].max()

    ax.scatter(utopia_x, utopia_y, utopia_z, color='green', marker='*', s=200)  # Utopia point
    ax.annotate('Utopia', (utopia_x, utopia_y), xytext=(10, -5), textcoords='offset points', color='green',
                    fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_3D_interactive(x, y, z):
    # Create scatter plot
    scatter = go.Scatter3d(
        x=df[x],
        y=df[y],
        z=df[z],
        mode='markers',
        marker=dict(
            size=5,
            color=df[z],  # Color based on z-value
            colorscale='inferno',  # Colormap
            colorbar=dict(title=z)  # Add colorbar
        )
    )
    # Create figure
    fig = go.Figure(data=[scatter])

    # Set layout
    fig.update_layout(scene=dict(
                        xaxis_title=x,
                        yaxis_title=y,
                        zaxis_title=z))

    # Display the plot in a browser
    fig.show(renderer='browser')

def check_pareto_optimality(x, y):
    # check if the solution is pareto optimal
    x_values = df[x].values
    y_values = df[y].values
    if x == 'Spatial Coverage [km^2]' or x == 'Temporal Coverage [days]':
        x_values = -x_values
    if y == 'Spatial Coverage [km^2]' or y == 'Temporal Coverage [days]':
        y_values = -y_values
    costs = np.column_stack((x_values, y_values))
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    optimal = []
    for i, c in enumerate(costs):
        # Check if solution i is dominated by any other solution
        is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(np.any(costs[i + 1:] > c, axis=1))
        if is_efficient[i] == True:
            optimal.append(costs[i])
    # make all numbers in optimal posivitve
    optimal = [[abs(x), abs(y)] for x, y in optimal]
    return optimal

def plot_2D(x, y, z):
    # plot in 2d with 3rd dimension as color
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if z == 'Spatial Coverage [km^2]' or z == 'Temporal Coverage [days]':
        sc = ax.scatter( df[x], df[y], c=df[z], cmap='inferno')
    else:
        sc = ax.scatter(df[x], df[y], c=df[z], cmap='inferno_r')
    plt.colorbar(sc, label=z)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    # Find the utopia point
    utopia_x = df[x].min()
    utopia_y = df[y].min()

    if x == 'Spatial Coverage [km^2]' or x == 'Temporal Coverage [days]':
        ax.invert_xaxis()
        utopia_x = df[x].max()
    if y == 'Spatial Coverage [km^2]' or y == 'Temporal Coverage [days]':
        ax.invert_yaxis()
        utopia_y = df[y].max()

    ax.scatter(utopia_x, utopia_y, color='green', marker='*', s=200)  # Utopia point
    ax.annotate('Utopia', (utopia_x, utopia_y), xytext=(10, -5), textcoords='offset points', color='green',
                    fontsize=12)
    optimal_points = check_pareto_optimality(x, y)
    optimal_points.sort(key=lambda point: point[0])
    for i in range(len(optimal_points) - 1):
        ax.plot([optimal_points[i][0], optimal_points[i+1][0]], [optimal_points[i][1], optimal_points[i+1][1]],
                color='blue', linestyle='-', linewidth=1)
    if x == 'Spatial Coverage [km^2]' or x == 'Temporal Coverage [days]':
        ax.annotate('Pareto Front', (optimal_points[0][0], optimal_points[0][1]), xytext=(-70, -10), textcoords='offset points', color='blue',
                    fontsize=12)
    else:
        ax.annotate('Pareto Front', (optimal_points[-1][0], optimal_points[-1][1]), xytext=(-70, -10), textcoords='offset points', color='blue',
                    fontsize=12)
    plt.tight_layout()
    plt.show()

def corr_heatmap(df):
    # df['Spatial Coverage [km^2]'] = -df['Spatial Coverage [km^2]']
    # df['Temporal Coverage [days]'] = -df['Temporal Coverage [days]']
    # Correlation Matrix Heatmap
    f, ax = plt.subplots(figsize=(10, 6))
    corr = df.corr()
    if len(df.columns) > 10:
        hm = sns.heatmap(round(corr, 2), annot=False, ax=ax, vmax=1, vmin=-1, cmap="coolwarm", fmt='.2f',
                         linewidths=.05)
    else:
        hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, vmax=1, vmin=-1, cmap="coolwarm", fmt='.2f',
                     linewidths=.05)
    f.subplots_adjust(top=0.93)
    t = f.suptitle('Objective Correlation Heatmap', fontsize=14)
    # put correlation into the cell
    plt.tight_layout()

    plt.show()

def interactive_plot(df):
    column_combinations = list(itertools.combinations(df.columns, 2))
    df = df.reset_index(drop=False)
    for i in column_combinations:
        utopia_x = df[i[0]].min()
        utopia_y = df[i[1]].min()

        hover_data = [col for col in df.columns if col not in i]
        fig = px.scatter(df, x=i[0], y=i[1], hover_data=hover_data)
        fig.update_traces(marker=dict(size=14))  # Set marker size for all points

        if df[i[0]].name == 'Spatial Coverage [km^2]' or df[i[0]].name == 'Temporal Coverage [days]':
            fig.update_xaxes(range=[df[i[0]].max() * 1.1, df[i[0]].min() * 0.9])  # Inverting x-axis range
            utopia_x = df[i[0]].max()
        if df[i[1]].name == 'Spatial Coverage [km^2]' or df[i[1]].name == 'Temporal Coverage [days]':
            fig.update_yaxes(range=[df[i[1]].max() * 1.1, df[i[1]].min() * 0.9])  # Inverting y-axis range
            utopia_y = df[i[1]].max()
        fig.add_trace(go.Scatter(x=[utopia_x], y=[utopia_y], mode='markers+text',
                                 marker=dict(symbol='star', color='green', size=15),
                                 name='Utopia', text=['Utopia'],textfont=dict(size=16),  # Text to display as label
                                 textposition='top center'))
        optimal_points = check_pareto_optimality(df[i[0]].name, df[i[1]].name)
        optimal_points.sort(key=lambda point: point[-1])
        x_values = [point[0] for point in optimal_points]
        y_values = [point[1] for point in optimal_points]

        # Example normalization function
        def normalize(value, min_value, max_value):
            return (value - min_value) / (max_value - min_value)

        # Assuming x_values and y_values are lists of coordinates
        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        max_y = max(y_values)

        normalized_x_values = [normalize(x, min_x, max_x) for x in x_values]
        normalized_y_values = [normalize(y, min_y, max_y) for y in y_values]

        distances_to_utopia = [
            ((x - normalize(utopia_x, min_x, max_x)) ** 2 + (y - normalize(utopia_y, min_y, max_y)) ** 2) ** 0.5
            for x, y in zip(normalized_x_values, normalized_y_values)
        ]

        # Find the index of the ideal point (minimizing distance to Utopia)
        ideal_index = np.argmin(distances_to_utopia)

        # Adding indicator for the closest Pareto optimal point
        fig.add_trace(go.Scatter(x=[x_values[ideal_index]], y=[y_values[ideal_index]], mode='markers+text',
                                 marker=dict(symbol='circle', color='yellow', size=15),
                                 text=['Closest Point <br> to Utopia'], textfont=dict(size=16),
                                 textposition='bottom left'))
        fig.add_trace(
            go.Scatter(x=x_values, y=y_values, mode='lines+text', name='Pareto Optimal Line', text=['Pareto Front'],
                       textposition='bottom right',textfont=dict(size=16), line=dict(color='red', width=1)))
        # fig.update_traces(hoverinfo='none', hover_template=None)

        fig.update_layout(
            title={
                'text': f'Scatter Plot for {i[0]} vs {i[1]}',  # Add a title
                'x': 0.5,  # Center the title
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title=i[0],
            yaxis_title=i[1],
            xaxis=dict(
                title_font=dict(size=18),  # Font size for x-axis label
                tickfont=dict(size=16)  # Font size for x-axis tick marks
            ),
            yaxis=dict(
                title_font=dict(size=18),  # Font size for y-axis label
                tickfont=dict(size=16)  # Font size for y-axis tick marks
            ),
            showlegend=False, # Remove legend
            margin=dict(l=20, r=20, t=40, b=20)  # Adjust margins to achieve tight layout

        )
        # app = Dash(__name__)
        # app.layout = html.Div([
        #     dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
        #     dcc.Tooltip(id="graph-tooltip"),
        # ])
        #
        # @callback(
        #     Output("graph-tooltip", "show"),
        #     Output("graph-tooltip", "bbox"),
        #     Output("graph-tooltip", "children"),
        #     Input("graph-basic-2", "hoverData"),
        # )
        # def display_hover(hoverData):
        #     if hoverData is None:
        #         return False, no_update, no_update
        #
        #     # demo only shows the first point, but other points may also be available
        #     pt = hoverData["points"][0]
        #     bbox = pt["bbox"]
        #     num = pt["pointNumber"]
        #
        #     df_row = df.iloc[num]
        #     img_src = df_row['IMG_URL']
        #     name = df_row['NAME']
        #     form = df_row['FORM']
        #     desc = df_row['DESC']
        #     if len(desc) > 300:
        #         desc = desc[:100] + '...'
        #
        #     children = [
        #         html.Div([
        #             html.Img(src=img_src, style={"width": "100%"}),
        #             html.H2(f"{name}", style={"color": "darkblue", "overflow-wrap": "break-word"}),
        #             html.P(f"{form}"),
        #             html.P(f"{desc}"),
        #         ], style={'width': '200px', 'white-space': 'normal'})
        #     ]
        #
        #     return True, bbox, children
        fig.show(renderer='browser')
def plot_distribution(df):

    df = df.reset_index(drop=True)


    # Calculate the number of rows and columns needed for subplots
    num_cols = len(df.columns)
    cols_in_plot = 7
    num_rows = (num_cols - 1) // cols_in_plot + 1  # Ensure at least 1 row

    # Create subplots
    fig, axes = plt.subplots(num_rows, cols_in_plot, figsize=(4 * cols_in_plot, num_rows * 4))
    axes = axes.flatten()

    # Plot bar chart for each column
    for i, col in enumerate(df.columns):
        for j in range(len(df)):
            sns.histplot(df[col], ax=axes[i], kde=False, bins=10)
        axes[i].set_title(col)

    # Remove extra empty subplots if necessary
    for j in range(num_cols, num_rows * cols_in_plot):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'C:\\Users\\AnnickDewald\\PycharmProjects\\DawnDesignTool\outputs\\{file_name}\\distributions.png')
    plt.show()

design_variable_df = load_variables(file_name, converged_runs)
big_df = pd.concat([df, design_variable_df], axis=1)

if 'fixed_aircraft' in file_name:
    df = df.drop(columns=['Wingspan [m]'])

# Get all combinations of 3 columns
column_combinations = list(itertools.combinations(df.columns, 3))

# Check if 'Wingspan [m]' is in any combination and move it to 2nd position
for i, combination in enumerate(column_combinations):
    if 'Wingspan [m]' in combination:
        # Move 'Wingspan [m]' to the second position
        column_combinations[i] = tuple(col for col in combination if col != 'Wingspan [m]')  # Remove 'Wingspan [m]'
        # Insert 'Wingspan [m]' at the second position
        column_combinations[i] = (column_combinations[i][0], 'Wingspan [m]') + column_combinations[i][1:]

# just give me the first one for now
# column_combinations = column_combinations[:1]
# for combination in column_combinations:
#     # plot_3D(*combination)
#     # plot_3D_interactive(*combination)
#     plot_2D(*combination)
#
# # optimal = check_pareto_optimality('Spatial Coverage [km^2]', 'Temporal Coverage [days]')
# # optimal.sort(key=lambda point: point[0])
# plot_parallel_coordinates(df, 'all', num_of_racetracks)
# plot_scatter_matrix(df)
corr_heatmap(df)
df_pareto_opt = check_optimality(df)

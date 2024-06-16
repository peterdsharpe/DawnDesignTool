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
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table

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

            'Imaging Offset Distance [km]': offset_distances,
            'Look Angle [deg]': look_angles, 'Swath Range Dimension [km]': swath_ranges,
            'Swath Azimuth Dimension [km]': swath_azimuths,

            'SAR Range Resolution [m]': SAR_range_resolutions, 'SAR Azimuth Resolution [m]': SAR_azimuth_resolutions,
            'Aperture Width [m]': aperture_widths, 'Aperture Length [m]': aperture_lengths}

    # create dataframe with all the variables of interest
    design_variables = pd.DataFrame(dict)
    design_variables = design_variables.set_index('Run')

    return design_variables

if __name__ == '__main__':
    # load the variables
    design_variables = load_variables(file_name, converged_runs)

    # merge the dataframes
    df = pd.merge(df, design_variables, on='Run')

    # Rounding!
    # Specify columns with different rounding requirements
    columns_with_different_rounding = [
        'Spatial Coverage [km^2]',
        'Temporal Coverage [days]',
        'Precision [yr^-1]',
        'Temporal Resolution [hrs]',
        'Spatial Resolution [m]',
        'Wingspan [m]',
        'Swath Overlap'
    ]

    # Round specific columns to different decimal places
    df['Spatial Coverage [km^2]'] = df['Spatial Coverage [km^2]'].round(0)
    df['Temporal Coverage [days]'] = df['Temporal Coverage [days]'].round(2)
    df['Precision [yr^-1]'] = df['Precision [yr^-1]'].round(8)
    df['Temporal Resolution [hrs]'] = df['Temporal Resolution [hrs]'].round(2)
    df['Spatial Resolution [m]'] = df['Spatial Resolution [m]'].round(2)
    df['Wingspan [m]'] = df['Wingspan [m]'].round(2)

    # Identify columns that need rounding to 2 decimal places
    columns_to_round = [col for col in df.columns if col not in columns_with_different_rounding]

    # Round remaining columns to 2 decimal places
    df[columns_to_round] = df[columns_to_round].round(2)
    df_percentiles = pd.DataFrame()
    # find the distribution of each design variable
    for col in df.columns:
        percentiles = df[col].rank(pct=True)
        df_percentiles[f'{col}'] = percentiles * 100
    df_percentiles = df_percentiles.round(0)

    # iterate through each row in the df
    df = df.iloc[:10]
    indexes = df.index
    for index in indexes:
        file = f'outputs\\{file_name}\\sweep_results.csv'

        # load the three view plot
        img_3view = mpl.image.imread(f'outputs\\{file_name}\\three_view_{index}.png')
        # crop image
        img_3view = img_3view[800:,0:1400]

        # load trajectory plot
        img_traj = mpl.image.imread(f'outputs\\{file_name}\\trajectory_{index}.png')

        # crop image
        img_traj = img_traj[65:, :]

        # temporal coverage plot
        end_day = df['Temporal Coverage [days]'][index] / 2 - 10
        start_day = 365 + (end_day - df['Temporal Coverage [days]'][index])
        day_of_year = np.linspace(1, 365, 365)
        active = np.ones(365)
        # Step 4: Update the 'active' column based on start_day and end_day
        active[(day_of_year <= start_day) & (day_of_year >= end_day)] = 0
        temporal_coverage_df = pd.DataFrame({
            'day of year': day_of_year,
            'active': active
        })
        # print(start_day, end_day)

        # Create a figure and subplots
        fig = plt.figure(figsize=(10, 8))
        # plt.title(f'Key Design Variables for Run Number {index}')
        gs = mpl.gridspec.GridSpec(3, 2, figure=fig, width_ratios=[2, 1], height_ratios=[5, 5, 1])

        # Add plots to specific subplots
        ax1 = fig.add_subplot(gs[0, 0])  # First plot at position [0, 0]
        ax2 = fig.add_subplot(gs[1, 0])  # Second plot at position
        ax3 = fig.add_subplot(gs[2, 0])  # Third plot at position

        # Plot the loaded image in the upper left quadrant
        ax1.imshow(img_3view)
        ax1.axis('off')  # Turn off axis for the image plot
        ax1.title.set_text('Optimal Aircraft')

        ax2.imshow(img_traj)
        ax2.axis('off')
        ax2.title.set_text('Optimal Trajectory')

        ax3.plot(temporal_coverage_df['day of year'], temporal_coverage_df['active'], linestyle="-", linewidth=0.0001)
        ax3.fill_between(temporal_coverage_df['day of year'], temporal_coverage_df['active'], 0,
                         where=(temporal_coverage_df['active'] > 0), interpolate=True, color='blue', alpha=1)
        # plt.xlabel('Day of Year')
        ax3.set_ylim(0, 1)
        ax3.set_xlim(1, 365)
        ax3.set_xticks(
            np.linspace(0, 365, 13)[:-1],
            (
                "Jan. 1",
                "Feb. 1",
                "Mar. 1",
                "Apr. 1",
                "May 1",
                "June 1",
                "July 1",
                "Aug. 1",
                "Sep. 1",
                "Oct. 1",
                "Nov. 1",
                "Dec. 1"
            ),
            rotation=40
        )
        ax3.set_yticks([])

        ax3.title.set_text('Temporal Coverage')

        # Ensure the subplots have the same size by setting the aspect ratio
        # ax1.set_aspect(aspect='auto')
        # ax2.set_aspect(aspect='auto')

        ax_table = fig.add_subplot(gs[:, 1])  # Table takes up both rows in the second column
        ax_table.axis('off')
        i = int(index)
        i = str(i)
        # ax_table.title.set_text(f'Key Design Variables for Run Number {i}')

        # Create a table subplot on the right
        percentile_data = df_percentiles.loc[index]
        table_data = df.loc[index]  # Get the data for the current row

        table_data = table_data.T
        percentile_data = percentile_data.T
        all_data = pd.concat([table_data, percentile_data], axis=1)

        table = ax_table.table(cellText=all_data.values,
                                rowLabels=all_data.index,
                                colLabels=['Value', 'Percentile'],
                                loc='center')
        # Adjust table properties
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(0.8, 1.5)

        # # Remove the lines around the table
        # for key, cell in table.get_celld().items():
        #     cell.set_linewidth(0)

        # Modify the cell padding
        for (i, j), cell in table.get_celld().items():
            cell.set_text_props(ha='right', va='center')
            cell.PAD = 0.1 # Adjust the padding value as needed
        cmap = plt.cm.get_cmap('coolwarm')
        norm = mpl.colors.Normalize(vmin=0, vmax=100)
        for i in range(0, len(all_data)):
                value = all_data.iloc[i, 1]
                color = cmap(norm(value))
                # change opacity
                color = mpl.colors.to_rgba(color, alpha=1)
                table[(i+1, 1)].set_facecolor(color)

        # Show the plot
        plt.tight_layout()

        plt.savefig(f'outputs\\{file_name}\\outputs_image_run_{index}.png')

        # plt.show()
        plt.close()


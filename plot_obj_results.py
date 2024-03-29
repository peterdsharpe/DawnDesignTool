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



def spider(df, *, id_column, title=None, max_values=None, padding=1.25):
    categories = df.dtypes[(df.dtypes == 'float') | (df.dtypes == 'int')].index.tolist()
    categories = categories[1:]
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
        ax.plot(angles, values, label=f'Run {model_name}')
        ax.fill(angles, values, alpha=0.15)
        for _x, _y, t in zip(angles, values, actual_values):
            t = f'{t:.2f}' if isinstance(t, float) else str(t)
            ax.text(_x, _y, t, size='xx-small')

    ax.fill(angles, np.ones(num_vars + 1), alpha=0.05)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(tiks)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.2))
    if title is not None: plt.suptitle(f"Effects of increasing weighting on {title} in objective function")
    plt.show()


radar = spider
def plot_radar(df, attribute):
    df.loc[:, 'Coverage_Areas'] = df['Coverage_Areas'].div(1e6).round(2)
    df.loc[:, 'Precisions'] = df['Precisions'].multiply(1e3).round(2)
    columns = ['Run', 'Spans [m]', 'Spatial Resolutions [m]', 'Coverage Areas [km^2]', 'Days', 'Precisions [1e3/yr]', 'Temporal Resolutions [hrs]']
    df.columns = columns

    spider(
        df,
        id_column='Run',
        title=attribute,
        padding=1.1
    )


def plot_scatter_matrix(df):
    df.loc[:, 'Coverage_Areas'] = df['Coverage_Areas'].div(1e6).round(2)
    columns = ['Run', 'Spans [m]', 'Spatial Resolutions [m]', 'Coverage Areas [km^2]', 'Days', 'Precisions [1/yr]', 'Temporal Resolutions [hrs]']
    df.columns = columns
    df = df.drop(columns=['Run'])
    df = df.drop([8])
    df = df.drop([10])

    sns.set_theme(style="ticks")

    # df = sns.load_dataset("penguins")
    sns.pairplot(df, diag_kind='kde', plot_kws={'color': 'black'})
    sns.pairplot(df, plot_kws={'color': 'black'})
    plt.show()

def corr_heatmap(df):
    # Correlation Matrix Heatmap
    df.loc[:, 'Coverage_Areas'] = df['Coverage_Areas'].div(1e6).round(2)
    columns = ['Run', 'Spans', 'Spatial Resolutions', 'Coverage Areas', 'Days', 'Precisions', 'Temporal Resolutions']
    df.columns = columns
    df = df.drop(columns=['Run'])
    df = df.drop([8])
    df = df.drop([10])
    heatmap = sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1,cmap='coolwarm', linewidths=0.5)
    # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    # add blank boundary around plot
    plt.tight_layout()

    plt.show()

# plot_parallel_coordinates(df_span, 'Wing Span')
# plot_parallel_coordinates(df_spatial, 'Spatial Resolution')
# plot_parallel_coordinates(df_coverage, 'Coverage Area')
# plot_parallel_coordinates(df_days, 'Days')
# plot_parallel_coordinates(df_precision, 'Precision')
# plot_parallel_coordinates(df_temporal, 'Temporal Resolution')
# plot_radar(df_span, 'Wing Span')
# plot_radar(df_spatial, 'Spatial Resolution')
# plot_radar(df_coverage, 'Coverage Area')
# plot_radar(df_days, 'Days')
# plot_radar(df_precision, 'Precision')
# plot_radar(df_temporal, 'Temporal Resolution')
# plot_scatter_matrix(df)
corr_heatmap(df)
__author__ = "Linus Stoltz"
__version__ = "1.1"
__email__ = "lstoltz@cfrfoundation.org"
from flask import Flask, render_template, jsonify, request, abort
import pandas as pd
# import io
from erddapy import ERDDAP
import plotly
import plotly.express as px
import json
import plotly.graph_objs as go
import pandas as pd
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger.info('Starting WindDash...')

app = Flask(__name__)
app.config['APPLICATION_ROOT'] = '/winddash'
BATHY_PATHY = 'static/bathymetry/ny_gome_contours.shp'


# Global variable to hold the full dataset
full_dataset = None
first_request = True
REFRESH_INTERVAL = timedelta(minutes=30)
last_update = datetime.min  # Initialize to a very old date
mapbox_access_token = 'pk.eyJ1IjoibHN0b2x0eiIsImEiOiJjbTBmY2hmcGowdHh1MndvaWlvejZpNHlyIn0.BLC9SltZf5sYCGGURAWh9w'  # free public token


def truncate_at_first_space(col_name):
    return col_name.split(' ')[0]


def extract_date_from_string(text):
    # Split the string by underscore
    parts = text.split('_')

    # Extract the date part
    date_str = parts[1]

    # Convert the date string to a datetime object
    date_obj = datetime.strptime(date_str, '%Y%m%d')

    return date_obj


def simplify_geometry(geometry, tolerance=0.01):
    if geometry is not None:
        return geometry.simplify(tolerance, preserve_topology=True)
    return geometry


@app.before_request
def load_full_dataset():
    logger.info('request for full dataset...')
    global full_dataset, first_request, latest_observation, cast_count, bathy, last_update
    if first_request or datetime.now() - last_update >= REFRESH_INTERVAL:
        logger.info('first time requesting or refreshing data...')
        first_request = False

        # Build the ERDDAP query URL for the full dataset
        server = 'https://erddap.ondeckdata.com/erddap/'

        try:
            e = ERDDAP(
                server=server,
                protocol="tabledap",
                response="nc",
            )
            e.dataset_id = 'wind_farm_profiles_1m_binned'
            full_dataset = e.to_pandas()
        except Exception as e:
            logger.error(f'Error connecting to ERDDAP server: {e}')
            return
        logger.info('got data from erddap')
        # Do some reshaping a bit to plot on the front ent
        full_dataset.rename(
            columns=lambda x: truncate_at_first_space(x), inplace=True)
        full_dataset['time'] = pd.to_datetime(full_dataset['time'])
        full_dataset['temp_f'] = full_dataset['temperature'] * 9/5 + 32
        full_dataset['chlorophyll'] = np.nan
        # project_id_mapping = {
        # 'cccfa_outer_cape': 'CCCFA',
        # 'shelf_research_fleet': 'CFRF | WHOI'
        #     }
        # full_dataset['project_id_labels'] = full_dataset['project_id'].replace(project_id_mapping)

        # need to create a common time stamp for each profile
        first_observation = full_dataset.groupby(
            'profile_id')['time'].first().reset_index()
        first_observation.rename(columns={'time': 'sample_date'}, inplace=True)
        # and merge it back in
        full_dataset = full_dataset.merge(
            first_observation, on='profile_id', how='left')
        full_dataset['first_observation'] = full_dataset['sample_date'].dt.strftime(
            '%Y-%m-%d %H:%M')
        full_dataset['time_numeric'] = pd.to_numeric(full_dataset['time'])
        latest_observation = full_dataset['time'].max().strftime('%Y-%m-%d')
        cast_count = full_dataset['profile_id'].nunique()
        # full_dataset['extracted_date'] = full_dataset['profile_id'].apply(extract_date_from_string)
        bathy = gpd.read_file(BATHY_PATHY)
        bathy = bathy[bathy['geometry'].notnull()]
        bathy['geometry'] = bathy['geometry'].apply(simplify_geometry)
        # Update last update timestamp
        last_update = datetime.now()
    else:
        logger.info('using cached dataset...')


@app.route('/winddash')
def index():
    logger.info('rendering index.html')
    try:
        return render_template('index.html')
    except Exception as e:
        logger.info('error rendering index.html: %s', e)
        return jsonify({'error': 'Error rendering index.html'}), 500


@app.route('/winddash/filter_data', methods=['POST'])
def filter_data():
    # if request.remote_addr not in ALLOWED_IPS:
    #     abort(403)  # Forbidden

    filtered_data = None
    try:
        global full_dataset
        logger.info('getting dates from front end...')
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        filtered_data = full_dataset[(full_dataset['time'] >= start_date) & (
            full_dataset['time'] <= end_date)]
        if filtered_data.empty:
            # set this  to the last two weeks of data if there is no new data in the last 2 weeks. le sad ...
            last_date = full_dataset['time'].max()
            last_range = last_date - timedelta(days=14)

            filtered_data = full_dataset[(full_dataset['time'] >= last_range) & (
                full_dataset['time'] <= last_date)]
        # fig = create_data_plot(filtered_data)
        # fig = px.line(filtered_data, x='temperature', y='sea_pressure', title='Temperature', color='time')
        logger.info('creating plots')
        fig = create_data_plots(filtered_data)
        plot_data = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return jsonify({'plot_data': plot_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def create_data_plots(plot_df):
    # Normalize the time_numeric column to a range between 0 and 1
    plot_df['time_normalized'] = (plot_df['time_numeric'] - plot_df['time_numeric'].min()) / \
                                 (plot_df['time_numeric'].max() -
                                  plot_df['time_numeric'].min())
    plot_df['hover_text'] = plot_df.apply(
        lambda row: f"""
        Date: {row['first_observation']}<br> 
        ID: {row['profile_id']} <br>
        Survey: {row['survey_id']}""", axis=1)
    # Define the color scale
    colorscale = px.colors.sequential.Rainbow
    try:

        fig = make_subplots(
            rows=3, cols=2,
            specs=[[{}, {}], [{}, {}], [{"type": "mapbox", "colspan": 2}, None]],
            vertical_spacing=0.06,
            row_heights=[0.3, 0.3, 0.353]  # Adjust the heights of the rows
        )

        # List of variables to plot
        variables = ['temp_f', 'absolute_salinity',
                     'density', 'chlorophyll']
        titles = ['Degrees Fahrenheit', 'PSU',
                  'Kilograms per Meter\u00B3', '[Chl-a] Micrograms per Liter']

        # Add traces for each variable
        for i, variable in enumerate(variables):
            row = i // 2 + 1
            col = i % 2 + 1
            # extracted date is the profile_id derivative
            for profile_id, group in plot_df.groupby(['first_observation']):
                color_idx = int(
                    group['time_normalized'].iloc[0] * (len(colorscale) - 1))
                color = colorscale[color_idx]
                # Show legend only for the first variable
                showlegend = (i == 0)
                fig.add_trace(go.Scatter(
                    x=group[variable],
                    y=group['sea_pressure'],
                    mode='lines',
                    line=dict(color=color, width=4),
                    hoverinfo='text',
                    hovertext=group['hover_text'],
                    hoverlabel=dict(
                        font=dict(
                            size=18)  # Set the font size for hover text
                    ),
                    name=f'{profile_id[0]}',
                    legendgroup=str(profile_id[0]),
                    showlegend=showlegend
                ), row=row, col=col)

            # Invert the y-axis for the first subplot
            fig.update_yaxes(title_text='Depth (meters)',autorange='reversed', row=row, col=col)

            xmin = min(plot_df[variable]) - 0.1 * (max(plot_df[variable]) - min(plot_df[variable]))
            xmax = max(plot_df[variable]) + 0.1 * (max(plot_df[variable]) - min(plot_df[variable]))

            fig.update_xaxes(title_text=titles[i], row=row, col=col, range=[xmin,xmax])
            fig.update_yaxes( row=row, col=col)

        # Define the bathymetry values, labels, and line styles
        subset_values = [-200, -100, -50]
        labels = {-200: '200m', -100: '100m', -50: '50m'}

        # Batch process the bathymetry lines
        for value in subset_values:
            shelf_break = bathy[bathy['elev_m'] == value]

            all_lats = []
            all_lons = []

            for _, row in shelf_break.iterrows():
                # Extract coordinates from the geometry
                coords = list(row['geometry'].coords)
                lats, lons = zip(*[(lat, lon) for lon, lat, _ in coords])
                all_lats.extend(lats)
                all_lons.extend(lons)
                all_lats.append(None)  # Add None to separate lines
                all_lons.append(None)

            # Add the trace to the Plotly figure
            fig.add_trace(go.Scattermapbox(
                lat=all_lats,
                lon=all_lons,
                mode='lines',
                line=dict(color='white', width=2),  # Use solid lines
                hovertext=labels[value],
                hoverinfo='text',
                hoverlabel=dict(
                    font=dict(
                        size=20)  # Set the font size for hover text
                ),
                showlegend=False  # Change this to True if you want to show the legend for lines
            ), row=3, col=1)

        # Create the map subplot
        df = plot_df.groupby('profile_id').agg(
            {'first_observation': 'first', 'latitude': 'first',
             'longitude': 'first', 'sea_pressure': 'max',
             'temperature': 'mean', 'absolute_salinity': 'mean'}).reset_index()
        df['hovertext'] = df.apply(
            lambda row: f"Date: {row['first_observation']}<br> {round(row['latitude'],6)}, {round(row['longitude'],6)}<br>Depth = {round(row['sea_pressure'],2)} m<br>Mean T = {round(row['temperature'],2)} C<br> Mean S = {round(row['absolute_salinity'],2)} PSU",
            axis=1
        )
        for profile_id, row in df.iterrows():
            color_idx = int(plot_df[plot_df['first_observation'] == row['first_observation']]
                            ['time_normalized'].iloc[0] * (len(colorscale) - 1))
            color = colorscale[color_idx]
            fig.add_trace(go.Scattermapbox(
                lat=[row['latitude']],
                lon=[row['longitude']],
                mode='markers',
                marker=dict(size=15, color=color),
                hoverinfo='text',
                hovertext=row['hovertext'],
                hoverlabel=dict(
                    font=dict(
                        size=20)  # Set the font size for hover text
                ),
                name=f'{row["first_observation"]} | Map',
                legendgroup=str(row['first_observation']),
                showlegend=False  # Show legend for the map marker
            ), row=3, col=1)

        # Set the mapbox layout
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        # quick stats for the figure
        n_casts_window = plot_df['profile_id'].nunique()

        fig.update_layout(
            autosize=True, # Automatically adjust the size of the figure
            font=dict(
                size=18
            ),  #
            mapbox=dict(
                accesstoken=mapbox_access_token,
                domain={'x': [0, 1], 'y': [0, 0.33]},
                style="mapbox://styles/mapbox/satellite-v9",
                center={"lat": center_lat, "lon": center_lon},
                zoom=7
            ),
            # clickmode='event+select',
            height=3000,
            width=1410,
            showlegend=True,
            title_text=f'There are {n_casts_window} casts during this time period, with {cast_count} casts total! | Latest data: {latest_observation}',
            updatemenus=[dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=["showlegend", True],
                        label="Show Legend",
                        method="relayout"
                    ),
                    dict(
                        args=["showlegend", False],
                        label="Hide Legend",
                        method="relayout"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )],
            annotations=[

                dict(
                    text='Temperature',
                    x=0.16,
                    y=1.02,
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                    font=dict(
                        size=30,
                        # Change the color to red
                        family='Arial, sans-serif',  # Set the font family
                        weight='bold'  # Make the text bold
                    )
                ),
                dict(
                    text='Salinity',
                    x=0.82,
                    y=1.02,
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                    font=dict(
                        size=30,
                        # Change the color to red
                        family='Arial, sans-serif',  # Set the font family
                        weight='bold'  # Make the text bold
                    )
                ),
                dict(
                    text='Density',
                    x=0.16,
                    y=0.68,
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                    font=dict(
                        size=30,
                        # Change the color to red
                        family='Arial, sans-serif',  # Set the font family
                        weight='bold'  # Make the text bold
                    )
                ),
                dict(
                    text='Chlorophyll',
                    x=0.82,
                    y=0.68,
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                    font=dict(
                        size=30,
                        # Change the color to red
                        family='Arial, sans-serif',  # Set the font family
                        weight='bold'  # Make the text bold
                    )
                ),
                dict(
                    text=f'Content developed by Linus Stoltz, Data Manager',
                    x=0.7,
                    y=0.001,
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                    font=dict(
                        size=16,
                        color='white',  # Change the color to red
                        family='Arial, sans-serif',  # Set the font family
                        weight='bold'  # Make the text bold
                    )
                )
            ]
        )
        config = {
            'toImageButtonOptions': {
                'format': 'png',  # one of png, svg, jpeg, webp
                'filename': 'cfrf_winddash',
                'height': 500,
                'width': 700,
                'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }
        # fig.show(config=config)
    except Exception as e:
        logger.error('error: %s', e)
    return fig


if __name__ == '__main__':
    app.run(debug=False, port=5001)

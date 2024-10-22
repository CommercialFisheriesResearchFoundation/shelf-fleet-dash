__author__ = "Linus Stoltz"
__version__ = "1.1"
__email__ = "lstoltz@cfrfoundation.org"
from flask import Flask, render_template, jsonify, request, abort, g
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
from shapely.geometry import LineString, Polygon, MultiPolygon
import numpy as np
import random

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
BOEM_PATHY = 'static/lease_areas/BOEM_Lease_Areas_4_13_2020_WGS1984.shp'


# Global variable to hold the full dataset
full_dataset = None
first_request = True
REFRESH_INTERVAL = timedelta(minutes=30)
last_update = datetime.min  # Initialize to a very old date
mapbox_access_token = 'pk.eyJ1IjoibHN0b2x0eiIsImEiOiJjbTBmY2hmcGowdHh1MndvaWlvejZpNHlyIn0.BLC9SltZf5sYCGGURAWh9w'  # free public token


def truncate_at_first_space(col_name):
    return col_name.split(' ')[0]


def random_color():
    return 'rgba({},{},{},0.6)'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def extract_date_from_string(text):
    # Split the string by underscore
    parts = text.split('_')
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
    #!TODO: Filter data on radio button calls not in the data pull
    #!TODO: Add shape files for South fork onto map frame, probably add in with bathymetry?
    #!TODO: Size the css and front end to not look wonky
    logger.info('request for full dataset...')
    global full_dataset, first_request, bathy, last_update, lease_areas
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
        full_dataset['feet'] = full_dataset['sea_pressure'] * 3.28084
        full_dataset['chlorophyll'] = np.nan

        try:
            full_dataset['time'] = pd.to_datetime(full_dataset['time'])
            full_dataset['year'] = full_dataset['time'].dt.year
            full_dataset['month'] = full_dataset['time'].dt.month
        except Exception as e:
            logger.error('error: %s', e, exc_info=True)

        bathy = gpd.read_file(BATHY_PATHY)
        bathy = bathy[bathy['geometry'].notnull()]
        bathy['geometry'] = bathy['geometry'].apply(simplify_geometry)

        lease_areas = gpd.read_file(BOEM_PATHY)
        lease_areas = lease_areas.to_crs('EPSG:4326')  # need to reproject
        lease_areas = lease_areas[lease_areas['geometry'].notnull()]
        lease_areas['coords'] = lease_areas['geometry'].apply(
            get_polygon_coords)

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


def subset_data(survey_id):

    global full_dataset
    plot_df = None
    selected_columns = ['temp_f', 'feet', 'temperature', 'sea_pressure',
                        'year', 'month', 'absolute_salinity', 'latitude', 'longitude', 'density']
    if survey_id == 'all':

        plot_df = full_dataset.loc[:, selected_columns]
        plot_df = plot_df.groupby(
            ['sea_pressure', 'year', 'month']).mean().reset_index()
        plot_df['year_month'] = pd.to_datetime(
            plot_df[['year', 'month']].assign(day=1))
    else:
        plot_df = full_dataset[full_dataset['survey_id'] == survey_id]
        plot_df = plot_df.loc[:, selected_columns]
        plot_df = plot_df.groupby(
            ['sea_pressure', 'year', 'month']).mean().reset_index()
        plot_df['year_month'] = pd.to_datetime(
            plot_df[['year', 'month']].assign(day=1))
    return plot_df


@app.route('/winddash/create_plot', methods=['POST'])
def filter_data():
    # if request.remote_addr not in ALLOWED_IPS:
    #     abort(403)  # Forbidden
    global full_dataset
    # which plot to generate
    variable = request.form.get('variable')
    logger.info('current variable: %s', variable)

    # filter the df
    survey_id = request.form.get('survey_id')
    plot_df = subset_data(survey_id)

    if variable == 'temperature':
        try:
            logger.info('creating temp...')
            label = 'Temperature'
            fig = create_data_plots(plot_df, 'temp_f', label)
            plot_data = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            logger.info('success! Temp plot made')

            return jsonify({'plot_data': plot_data})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    elif variable == 'salinity':
        try:
            logger.info('creating salinity plots...')
            label = 'Salinity'
            fig = create_data_plots(plot_df, 'absolute_salinity', label)
            plot_data = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            logger.info('success! salt plot made')

            return jsonify({'plot_data': plot_data})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    elif variable == 'density':
        try:
            logger.info('creating Rho plots...')
            label = 'Density'
            fig = create_data_plots(plot_df, 'density', label)
            plot_data = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            logger.info('success! Rho plot made')

            return jsonify({'plot_data': plot_data})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    elif variable == 'map':
        try:
            logger.info('creating map ...')
            label = 'Map'

            fig = create_map(plot_df, survey_id)
            plot_data = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            logger.info('success! Map made')

            return jsonify({'plot_data': plot_data})
        except Exception as e:
            logger.info('error creating map: %s', e)
            return jsonify({'error': str(e)}), 500


def create_map(df, survey_id):
    fig = go.Figure()

    subset_values = [-100, -50, -20]
    labels = {-100: '100m', -50: '50m', -20: '20m'}

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
            line=dict(color='white', width=2),
            hovertext=labels[value],
            hoverinfo='text',
            hoverlabel=dict(
                font=dict(
                    size=20)  # Set the font size for hover text
            ),
            showlegend=False
        ))
    # Loop through lease areas
    for _, row in lease_areas.iterrows():
        # Handle both Polygons and MultiPolygons
        geometry = row['geometry']

        if geometry.geom_type == 'Polygon':
            polygons = [geometry]  # Single polygon as a list
        elif geometry.geom_type == 'MultiPolygon':
            # Extract polygons from MultiPolygon
            polygons = list(geometry.geoms)

        # Add each polygon as a separate trace
        for polygon in polygons:
            coords = polygon.exterior.coords
            lats, lons = zip(*[(lat, lon)
                             for lon, lat in coords])  # Extract lat and lon

            fig.add_trace(go.Scattermapbox(
                mode='lines',
                lon=lons,
                lat=lats,
                fill="toself",
                text=row['Company'],
                hoverinfo='text',
                # Assign random color for each polygon
                line=dict(width=2, color=random_color()),
                hoverlabel=dict(
                    font=dict(
                        size=20)  # Set the font size for hover text
                ),
                showlegend=False
            ))

    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    fig.add_trace(go.Scattermapbox(
        mode='markers',  # Plot each point as a marker
        lat=df['latitude'],
        lon=df['longitude'],
        marker=dict(size=14, color='red'),  # Customize marker size and color
        hoverinfo='text',
        hovertext=f'Survey: {survey_id}',
        hoverlabel=dict(
            font=dict(
                size=20)
        ),
        showlegend=False
    ))

    fig.update_layout(
        mapbox=dict(
            accesstoken=mapbox_access_token,
            style="mapbox://styles/mapbox/satellite-v9",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        annotations=[
            dict(
                text=f'Content developed by Linus Stoltz, Data Manager CFRF',
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
    return fig


def create_data_plots(df=None, variable='temperature', label='label'):

    title_dict = {'Temperature': 'Degrees',
                  'Salinity': 'PSU',
                  'Density': 'kg/m^3'}

    scatter_fig = px.scatter(
        df,
        x='year_month',
        y='sea_pressure',
        # Set the title for the temperature plot
        title=f'CFRF WindDash - {label}',
        labels={variable: title_dict[label]},
        color=variable,  # Color by the temperature value
        color_continuous_scale='Jet'
    )

    # Update marker size
    scatter_fig.update_traces(marker=dict(size=15))

    # Update x-axis to show only year and month
    scatter_fig.update_xaxes(
        title_text='Year-Month',
        tickformat='%Y-%m',  # Format to show year and month
        dtick='M1'  # Set the tick interval to 1 month
    )

    # Invert the y-axis for the plot
    scatter_fig.update_yaxes(title_text='Depth', autorange='reversed')

    # Add a date range slider
    scatter_fig.update_xaxes(
        rangeslider=dict(
            visible=True,
            thickness=0.2
        ),
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    if variable == 'temp_f':
        scatter_fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"marker.color": [df[variable]],
                                   "marker.colorbar.title.text": "Temperature (°F)"}],
                            label="Show °F",
                            method="restyle"
                        ),
                        dict(
                            args=[{"marker.color": [df['temperature']],
                                   "marker.colorbar.title.text": "Temperature (°C)"}],
                            label="Show °C",
                            method="restyle"
                        ),
                        dict(
                            args=[
                                {"y": [df['sea_pressure']], "marker.colorbar.title.text": "Temperature (°F)"}],
                            label="Show meters",
                            method="update"
                        ),
                        dict(
                            args=[
                                {"y": [df['feet']], "marker.colorbar.title.text": "Temperature (°C)"}],
                            label="Show Feet",
                            method="update"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.8,
                    xanchor="left",
                    y=1.2,
                    yanchor="top"
                ),
            ]
        )
    else:
        scatter_fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[
                                {"y": [df['sea_pressure']], "marker.colorbar.title.text": "Temperature (°F)"}],
                            label="Show meters",
                            method="update"
                        ),
                        dict(
                            args=[
                                {"y": [df['feet']], "marker.colorbar.title.text": "Temperature (°C)"}],
                            label="Show Feet",
                            method="update"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.4,
                    xanchor="left",
                    y=1.2,
                    yanchor="top"
                ),
            ]
        )
    

    return scatter_fig


def get_polygon_coords(geometry):
    if geometry.geom_type == 'Polygon':
        return list(geometry.exterior.coords)
    elif geometry.geom_type == 'MultiPolygon':
        coords = []
        for polygon in geometry.geoms:
            coords.extend(list(polygon.exterior.coords))
        return coords
    else:
        return None


if __name__ == '__main__':
    app.run(debug=False, port=5001)

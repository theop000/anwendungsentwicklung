import pandas as pd
import dash
from dash import html, dcc, Input, Output, State, ctx
import plotly.express as px
import json
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Read the countries file
countries_df = pd.read_fwf('./data/countries.txt', 
                          colspecs=[(0, 2), (3, 50)],
                          names=['Code', 'Country'],
                          encoding='utf-8')

# Read the stations file with specific columns
stations_df = pd.read_csv('./data/stations.csv', 
                         usecols=[0, 1, 2, 5],  # Select first 4 columns
                         names=['Station_ID', 'Latitude', 'Longitude', 'Name'])

# Clean station names to keep only the first name before any comma
stations_df['Name'] = stations_df['Name'].str.split(',').str[0]

# Read the inventory file
inventory_df = pd.read_fwf('./data/inventory.txt',
                          colspecs=[(0, 11),    # Station ID
                                  (11, 20),     # Latitude
                                  (20, 30),     # Longitude
                                  (31, 35),     # Element
                                  (36, 40),     # First year
                                  (41, 45)],    # Last year
                          names=['Station_ID', 'Latitude', 'Longitude', 
                                'Element', 'FirstYear', 'LastYear'])

# Filter for TMAX and TMIN
temp_stations = inventory_df[inventory_df['Element'].isin(['TMAX', 'TMIN'])]

# Get unique station IDs that have both TMAX and TMIN
valid_stations = temp_stations.groupby('Station_ID').Element.nunique()
valid_station_ids = valid_stations[valid_stations == 2].index

# Filter stations_df to only include stations with both TMAX and TMIN
stations_df = stations_df[stations_df['Station_ID'].isin(valid_station_ids)]

# Create the Dash app
app = dash.Dash(__name__)

# Create the map figure
fig = px.scatter_map(stations_df,
                     lat='Latitude',
                     lon='Longitude',
                     hover_name='Name',
                     zoom=1,
                     height=800)

fig.update_layout(
    mapbox_style="open-street-map",
    margin={"r":0,"t":0,"l":0,"b":0},
    clickmode='event+select'  # Enable clicking
)

# Update marker size to make them easier to click
fig.update_traces(marker=dict(size=3))

# Define the app layout
app.layout = html.Div([
    # Store component to save the selected stations
    dcc.Store(id='selected-stations-store'),
    # Add store for button state
    dcc.Store(id='button-state', data={'active': False}),
    
    dcc.Tabs([
        dcc.Tab(label='Map View', children=[
            html.H1('Karte - Wetterstationen', 
                    style={'textAlign': 'left', 'marginBottom': 20, 'fontWeight': 'bold'}),
            
            # Flex container for map and sidebar
            html.Div([
                # Map container (left side)
                html.Div([
                    dcc.Graph(id='station-map', figure=fig),
                ], style={'width': '75%', 'display': 'inline-block'}),
                
                # Sidebar container (right side)
                html.Div([
                    html.H3('Search Settings', style={'marginBottom': '20px'}),
                    
                    # Radius slider
                    html.Label('Suchradius (km)', style={'fontWeight': 'bold'}),
                    dcc.Slider(
                        id='radius-slider',
                        min=1,
                        max=100,
                        value=50,
                        step=1,
                        marks={
                            1: '1',
                            25: '25',
                            50: '50',
                            75: '75',
                            100: '100'
                        }
                    ),
                    html.Br(),
                    
                    # Station count slider
                    html.Label('Number of Stations', style={'fontWeight': 'bold'}),
                    dcc.Slider(id='station-count-slider', min=1, max=10, value=5, step=1),
                    html.Br(),
                    html.Br(),
                    
                    # Place Pin button with dynamic style
                    html.Button('Place Pin', 
                               id='place-pin-button',
                               style={
                                   'width': '100%',
                                   'padding': '10px',
                                   'backgroundColor': '#808080',  # Start with gray
                                   'color': 'white',
                                   'border': 'none',
                                   'borderRadius': '1px',
                                   'cursor': 'pointer'
                               }),
                    
                    # Display clicked coordinates
                    html.Div(id='click-data', 
                            style={'marginTop': '20px', 'textAlign': 'center'})
                    
                ], style={
                    'width': '23%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'padding': '20px',
                    'backgroundColor': '#f9f9f9',
                    'borderLeft': '1px solid #ccc',
                    'height': '800px',
                    'marginLeft': '2%'
                })
            ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'})
        ]),
        dcc.Tab(label='Station Data', children=[
            html.H1('Station Data', 
                   style={'textAlign': 'left', 'marginBottom': 20, 'fontWeight': 'bold'}),
            html.Div([
                html.Div(id='station-data-table'),
                html.Div(id='no-selection-message', 
                        children='Please select a location and click "Place Pin" in the Map View tab to see station data.',
                        style={'textAlign': 'center', 'marginTop': '50px', 'color': '#666'})
            ])
        ])
    ], style={'marginBottom': '20px'})
])

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

@app.callback(
    Output('station-map', 'figure'),
    Output('selected-stations-store', 'data'),
    Output('station-data-table', 'children'),
    Output('click-data', 'children'),
    Input('station-map', 'clickData'),
    Input('place-pin-button', 'n_clicks'),
    Input('radius-slider', 'value'),
    Input('station-count-slider', 'value'),
    State('station-map', 'figure'),
    State('button-state', 'data'),
    prevent_initial_call=True
)
def update_map(clickData, n_clicks, radius, station_count, current_fig, button_state):
    if not ctx.triggered_id:
        return current_fig, None, None, "Click on a station to place a pin"
    
    if clickData is None:
        return current_fig, None, None, "Click on a station to place a pin"
    
    try:
        lat = clickData['points'][0]['lat']
        lon = clickData['points'][0]['lon']
    except (KeyError, IndexError):
        return current_fig, None, None, "Click on a station to place a pin"
    
    if ctx.triggered_id == 'station-map':
        # Create new figure with base stations
        new_fig = px.scatter_map(stations_df,
                               lat='Latitude',
                               lon='Longitude',
                               hover_name='Name',
                               zoom=1,
                               height=800)
        
        # Add temporary pin for selected location
        new_fig.add_trace(px.scatter_map(
            pd.DataFrame({'lat': [lat], 'lon': [lon]}),
            lat='lat',
            lon='lon',
            hover_name=None
        ).data[0])
        
        # Update marker appearances
        new_fig.data[0].marker.update(size=3, color='blue')
        new_fig.data[-1].marker.update(size=8, color='red', symbol='circle')
        
        new_fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":0,"l":0,"b":0},
            clickmode='event+select'
        )
        
        return new_fig, None, None, f'Selected coordinates: {lat:.4f}, {lon:.4f}'
    
    elif ctx.triggered_id == 'place-pin-button':
        # Calculate distances for all stations
        stations_df['Distance'] = stations_df.apply(
            lambda row: haversine_distance(lat, lon, row['Latitude'], row['Longitude']), 
            axis=1
        )
        
        # Get closest stations within radius and limit
        closest_stations = stations_df[stations_df['Distance'] <= radius].nsmallest(station_count, 'Distance')
        
        # Create new figure with base stations
        new_fig = px.scatter_map(stations_df,
                               lat='Latitude',
                               lon='Longitude',
                               hover_name='Name',
                               zoom=1,
                               height=800)
        
        # Add user pin
        new_fig.add_trace(px.scatter_map(
            pd.DataFrame({'lat': [lat], 'lon': [lon]}),
            lat='lat',
            lon='lon',
            hover_name=None
        ).data[0])
        
        # Update the pin appearance
        new_fig.data[-1].marker.update(size=15, color='red', symbol='circle')
        new_fig.data[-1].name = 'Selected Location'
        
        # Highlight selected stations
        if len(closest_stations) > 0:
            new_fig.add_trace(px.scatter_map(
                closest_stations,
                lat='Latitude',
                lon='Longitude',
                hover_name='Name'
            ).data[0])
            new_fig.data[-1].marker.update(size=12, color='green')
            new_fig.data[-1].name = 'Selected Stations'
        
        # Update base stations appearance
        new_fig.data[0].marker.update(size=8, color='blue')
        new_fig.data[0].name = 'All Stations'
        
        # Update layout with zoomed view centered on selected station
        new_fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":0,"l":0,"b":0},
            mapbox=dict(
                center=dict(lat=lat, lon=lon),
                zoom=7  # This zoom level shows roughly 300km top to bottom
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            clickmode='event+select'
        )
        
        # Create table for station data
        if len(closest_stations) > 0:
            table = html.Table([
                html.Thead(
                    html.Tr([
                        html.Th('Station Name'),
                        html.Th('Distance (km)'),
                        html.Th('Latitude'),
                        html.Th('Longitude'),
                        html.Th('Station ID')
                    ], style={'backgroundColor': '#f4f4f4'})
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(row['Name']),
                        html.Td(f"{row['Distance']:.2f}"),
                        html.Td(f"{row['Latitude']:.4f}"),
                        html.Td(f"{row['Longitude']:.4f}"),
                        html.Td(row['Station_ID'])
                    ]) for _, row in closest_stations.iterrows()
                ])
            ], style={
                'width': '100%',
                'borderCollapse': 'collapse',
                'border': '1px solid #ddd',
                'marginTop': '20px'
            })
        else:
            table = html.Div('No stations found within the specified radius.',
                           style={'textAlign': 'center', 'marginTop': '50px', 'color': '#666'})
        
        return new_fig, closest_stations.to_dict('records'), table, f'Found {len(closest_stations)} stations within {radius}km'
    
    return current_fig, None, None, f'Selected coordinates: {lat:.4f}, {lon:.4f}'

@app.callback(
    Output('place-pin-button', 'children'),
    Output('place-pin-button', 'style'),
    Output('button-state', 'data'),
    Input('station-map', 'clickData'),
    Input('place-pin-button', 'n_clicks'),
    State('button-state', 'data'),
    prevent_initial_call=True
)
def update_button(clickData, n_clicks, button_state):
    base_style = {
        'width': '100%',
        'padding': '10px',
        'color': 'white',
        'border': 'none',
        'borderRadius': '1px',
        'cursor': 'pointer'
    }
    
    if ctx.triggered_id == 'station-map' and clickData:
        base_style['backgroundColor'] = '#4CAF50'  # Green when station selected
        return 'Confirm Selection', base_style, {'active': True}
    
    if ctx.triggered_id == 'place-pin-button' and button_state['active']:
        base_style['backgroundColor'] = '#808080'  # Gray after confirmation
        return 'Place Pin', base_style, {'active': False}
    
    base_style['backgroundColor'] = '#808080'  # Default gray
    return 'Place Pin', base_style, button_state

# Add CSS styles for the table
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            table td, table th {
                padding: 12px;
                border: 1px solid #ddd;
                text-align: left;
            }
            table tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            table tr:hover {
                background-color: #f5f5f5;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


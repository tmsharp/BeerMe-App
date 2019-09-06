## Import
import time
import pickle

import dash_core_components as dcc
import dash_html_components as html 
from dash.dependencies import Input, Output, State
import numpy as np

from app import app, server
from tabs import content_based_filtering, collab_filt, hybrid
from util import *

# Layout
app.layout = html.Div([

    # css
    html.Link(href="app/assets/favicon.ico", rel="icon"),
    html.Link(rel="stylesheet", href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css",
              integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm",
              crossOrigin="anonymous"),
    html.Link(href='/assets/css/main.css', rel='stylesheet'),
    html.Link(href='/assets/css/loadwheel.css', rel='stylesheet'),
    
    # url
    dcc.Location(id='url', refresh=False),

    # tabs
    dcc.Tabs(id="tabs", value='cbf', children=[
        dcc.Tab(label='Content Based Filtering', value='cbf'),
        dcc.Tab(label='Collaborative Filtering', value='collabfilt'),
        dcc.Tab(label='Hybrid', value='hybrid'),
    ]),

    # page content
    html.Div(id='page-content')

])


### callbacks for page content and url 
# output url for clicking tabs
@app.callback(Output('url', 'pathname'),
              [Input('tabs','value')])
def tab_selection(tab):
    if tab == 'collabfilt':
        return '/collabfilt'
    elif tab == 'cbf':
        return '/cbf'
    elif tab == 'hybrid':
        return '/hybrid'


# display each app's content based on url (which is updated by tab clicks)
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/collabfilt':
        return collab_filt.layout
    elif pathname == '/hybrid':
        return hybrid.layout
    else:
        return content_based_filtering.layout


## run
if __name__ == '__main__':
    server.run(debug=True)
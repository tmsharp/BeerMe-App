## Import
import time
import pickle

import dash_core_components as dcc
import dash_html_components as html 
from dash.dependencies import Input, Output, State
import numpy as np

from app import app, server
from tabs import existing_user, new_user
from util import *

# Layout
app.layout = html.Div(className="row", children=[

    # css
    html.Link(href="app/assets/favicon.ico", rel="icon"),
    # html.Link(rel="stylesheet", href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css",
    #           integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm",
    #           crossOrigin="anonymous"),
    # html.Link(href='/assets/css/main.css', rel='stylesheet'),
    # html.Link(href='/assets/css/loadwheel.css', rel='stylesheet'),
    html.Link(href='/assets/css/normalize.css', rel='stylesheet'),
    html.Link(href='/assets/css/skeleton.css', rel='stylesheet'),
    html.Link(href='/assets/css/codepen.css', rel='stylesheet'),
    html.Link(href='/assets/css/tabs.css', rel='stylesheet'),
    
    # url
    dcc.Location(id='url', refresh=False),

    # header
    html.Div(className='row', style={'background':'#fcc203', 'padding':'10px'}, children = [
        html.Div(style={'display': 'flex'}, children = [
            html.H2('BeerMe', className = "tagline-400", style={'padding-right': '10px'}),
            html.Div(style={'margin-top':'10px'}, children=[
                html.H4('A Recommendation System for Untappd', className="verticalLine-2 tagline")
            ])
        ])
    ]),
    
    # tabs
    dcc.Tabs(id="tabs", value='cbf', children=[
        dcc.Tab(label='Existing User', value='existing-user'),
        dcc.Tab(label='New User', value='new-user'),
    ]),

    # page content
    html.Div(id='page-content', style={'margin-top':'50px', 'margin-bottom':'50px'})

])


### callbacks for page content and url 
# output url for clicking tabs
@app.callback(Output('url', 'pathname'),
              [Input('tabs','value')])
def tab_selection(tab):
    if tab == 'new-user':
        return '/new-user'
    elif tab == 'existing-user':
        return '/existing-user'


# display each app's content based on url (which is updated by tab clicks)
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/new-user':
        return new_user.layout
    else:
        return existing_user.layout


## run
if __name__ == '__main__':
    server.run(debug=True)
## Import
import time
import pickle

import dash_core_components as dcc
import dash_html_components as html 
from dash.dependencies import Input, Output, State
import numpy as np

from app import app, server
from tabs import me
from util import *

# Layout
app.layout = html.Div([

    # css
    html.Link(href="app/assets/favicon.ico", rel="icon"),
    html.Link(rel="stylesheet", href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css",
              integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm",
              crossOrigin="anonymous"),
    html.Link(href='/assets/css/main.css', rel='stylesheet'),
    
    # url
    dcc.Location(id='url', refresh=False),

    # tabs
    dcc.Tabs(id="tabs", value='me', children=[
        dcc.Tab(label='Me', value='me'),
        dcc.Tab(label='Content Based Filtering', value='cbf'),
    ]),

    # page content
    html.Div(id='page-content')

])


### callbacks for page content and url 
# output url for clicking tabs
@app.callback(Output('url', 'pathname'),
              [Input('tabs','value')])
def tab_selection(tab):
   '''
   @doc: callback function for the page buttons in the header
   @args: pathname is a str representing the button path that points to the
      correct html.Div() objects defined above.
   @return: the appropriate html.Div() object as defined above
   '''
   if tab == 'me':
      return '/me'
   elif tab == 'cbf':
      return '/cbf'


# display each app's content based on url (which is updated by tab clicks)
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/cbf':
        pass
    else:
        return me.layout


## run
if __name__ == '__main__':
    server.run(debug=True)
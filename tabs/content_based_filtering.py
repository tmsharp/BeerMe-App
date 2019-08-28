import dash
import dash_core_components as dcc
import dash_html_components as html 
from dash.dependencies import Input, Output, State
import pickle
import numpy as np

from app import app
from util import *

layout = html.Div(className = 'container my-4', children =[
    
   # username section
   html.Div(className='card', children=[
       html.Div(className='col-lg-5 m-4', children=[
            dcc.Dropdown(
                id = 'username-selection-dropdown-cbf',
                options=username_options(),
                multi = False
            )
       ]),
    ]),

    # feature selection
    html.Div(className='card', children=[
       html.Div(className='col-lg-5 m-4', children=[
            dcc.Dropdown(
                id = 'feature-selection-dropdown-cbf',
                options = [{'label': 'Simple', 'value': 'simple'},
                            {'label': 'Categorical Enocidng of Beer Description', 'value': 'cat-encoding'},
                            {'label': 'Count Vectorizer of Beer Description', 'value': 'count-vect'},
                            {'label': 'TFIDF Vectorizer of Beer Description', 'value': 'tfidf-vect'}],
                multi = False
            )
       ]),
    ]),

    # search section
    html.Div(id='search-section-cbf', className='card', children = [
        html.Div(className='card-body', children = [
            html.H2(className='card-title text-center', children = "Find My Beer"),
            html.Div(className='card-text text-center', children = [
                    """
                    Select up to 5 beers and our algorithm will predict which beer you should drink!
                    """
            ]),
            html.Div(className='row justify-content-center', children=[
                html.Div(className='col-lg-5 m-4', children=[
                    dcc.Dropdown(
                        id = 'beer-selection-dropdown-cbf',
                        options = beer_options(),
                        multi = True
                    )
                ]),
            ]),
            html.Div(className='row justify-content-center', children=[
                html.Button('Search', id='search-button-cbf', className='btn btn-outline-primary')
            ]),
                html.Div(className='row justify-content-center my-3', children=[
                html.Div(id='search-results-cbf')
            ]),
            html.Div(className='row justify-content-center my-3', children=[
                html.Img(id='beer-loader-cbf', src='/assets/img/beer-loader.gif'),
            ]),
        ]),
    ]),
])
# end container


## callbacks
@app.callback(Output('beer-loader-cbf', 'style'),
                [Input('search-button-cbf', 'n_clicks')],
                [State('beer-selection-dropdown-cbf', 'value')])
def display_beer_loader(n_clicks, value):
    if value != None:
        if len(value) == 1:
            return {'display': 'none'}
    elif n_clicks != None and value != None and value != []:
        time.sleep(0.25)
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(Output('search-results-cbf', 'children'),
                [Input('search-button-cbf', 'n_clicks')],
                [State('beer-selection-dropdown-cbf', 'value'),
                State('username-selection-dropdown-cbf', 'value'),
                State('feature-selection-dropdown-cbf', 'value')])
def build_model(n_clicks, value, user_of_interest, features):

    print("HERE")
    
    drop_cols =['username', 'beer_name', 'brewery']

    df = IMPORT_CLEAN_STEP(db_path)

    if features == 'simple':
        user_df = df[df['username']==user_of_interest].drop(drop_cols + ['beer_description'], axis=1, inplace=False)
    elif features == 'cat-encoding':
        user_df = cat_encoding(df, 'beer_description', drop_cols)
    elif features == 'count-vect':
        user_df = count_vectorizer(df,'beer_description', drop_cols)
    elif features == 'tfidf-vect':
        user_df = tfidf_vectorizer(df,'beer_description', drop_cols)
    
    print("HEERE")
    run_model(user_df, user_of_interest, 'user_rating')
    print("HEEERE")
import dash
import dash_core_components as dcc
import dash_html_components as html 
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pickle
import numpy as np

from app import app
from util import *

layout = html.Div(className = 'container my-4', children =[

    dcc.Store(id="memory"),
    
    # username section
    html.Div(className='card', children=[
        html.Div(className='col-lg-5 my-4', children=[
            html.H4("Select Your Username"),
            dcc.Dropdown(
                id = 'username-selection-dropdown-collabfilt',
                options=username_options(),
                multi = False
            )
       ]),

        html.Div(className='row justify-content-center', children=[
            html.Button('Test Collaborative Filtering Method', id='model-button-collabfilt', className='btn btn-outline-primary'),
            dcc.Loading(id="loading-model", children=[html.Div(id="loading-model-output")], type="default"),
        ]),
        html.Div(className='row justify-content-center my-3', children=[
            html.Div(id='model-results-collabfilt')
        ]),
    
    ]),

    # prediciton section
    html.Div(className='card', children = [
        html.Div(className='card-body', children = [
            html.H2(className='card-title text-center', children = "Rate My Beer"),
            html.Div(className='card-text text-center', children = [
                    """
                    Select a beer and our algorithm will predict your rating!
                    """
            ]),
            html.Div(className='row justify-content-center', children=[
                html.Div(className='col-lg-5 m-4', children=[
                    dcc.Dropdown(
                        id = 'beer-selection-dropdown-collabfilt',
                        options = beer_options(),
                        multi = False
                    )
                ]),
            ]),
            html.Div(className='row justify-content-center', children=[
                html.Button('Predict', id='prediction-button-collabfilt', className='btn btn-outline-primary')
            ]),
            html.Div(className='row justify-content-center my-3', children=[
                html.Div(id='prediction-results-collabfilt')
            ]),
        ]),
    ]),

    # ranking section
    html.Div(className='card', children = [
        html.Div(className='card-body', children = [
            html.H2(className='card-title text-center', children = "Rank My Beers"),
            html.Div(className='card-text text-center', children = [
                    """
                    Select a few beers and will tell you which one you'll like best!
                    """
            ]),
            html.Div(className='row justify-content-center', children=[
                html.Div(className='col-lg-5 m-4', children=[
                    dcc.Dropdown(
                        id = 'ranking-beer-selection-dropdown-collabfilt',
                        options = beer_options(),
                        multi = True
                    )
                ]),
            ]),
            html.Div(className='row justify-content-center', children=[
                html.Button('Rank', id='ranking-button-collabfilt', className='btn btn-outline-primary')
            ]),
            html.Div(className='row justify-content-center my-3', children=[
                html.Div(id='ranking-results-collabfilt')
            ]),
        ]),
    ]),

])
# end container


## callbacks
@app.callback(Output('beer-loader-collabfilt', 'style'),
                [Input('search-button-collabfilt', 'n_clicks')],
                [State('beer-selection-dropdown-collabfilt', 'value')])
def display_beer_loader(n_clicks, value):
    if value != None:
        if len(value) == 1:
            return {'display': 'none'}
    elif n_clicks != None and value != None and value != []:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(Output('model-results-collabfilt', 'children'),
                [Input('model-button-collabfilt', 'n_clicks')],
                [State('username-selection-dropdown-collabfilt', 'value')])
def build_model(n_clicks, user_of_interest):

    if n_clicks != None:
        
        print("Running...")
       
        query = "SELECT user_rating, beer_name, username FROM prepped_data"
        df = import_table(db_path, query, remove_dups=False)
        mae, quarter, half = collaborative_filtering(df, user_of_interest)

        # structure html and return 
        children = [html.Div("We have created a predictive model based on your taste preferences".format(quarter, half, mae),
                            style={'font-size':'large', 'font-weight':'bold'}),
                    html.Br(),
                    html.Div("Full analysis below:", style={'text-align':'center', 'font-weight':'bold'}),
                    html.Div("Accuracy within 0.25 stars: {:.2f}%".format(quarter), style={'text-align':'center', 'font-size':'small'}),
                    html.Div("Accuracy within 0.50 stars: {:.2f}%".format(half), style={'text-align':'center', 'font-size':'small'}),
                    html.Div("Mean Absolute Error (MAE): {:.2f}".format(mae), style={'text-align':'center', 'font-size':'small'})]
        ret_html = html.Div(children=children)

        return ret_html

@app.callback(Output('prediction-results-collabfilt', 'children'),
                [Input('prediction-button-collabfilt', 'n_clicks')],
                [State('username-selection-dropdown-collabfilt', 'value'),
                State('beer-selection-dropdown-collabfilt', 'value')])
def predict_beer_rating(n_clicks, user_of_interest, beer):

    if n_clicks != None:

        print("Running...")

        query = "SELECT user_rating, beer_name, username FROM prepped_data"
        df = import_table(db_path, query, remove_dups=False)
        df = COSINE_STEP(df, user_of_interest)
        prediction = df[ (df.sort_values('nearest_neighbor_rank')['beer_name'] == beer) & (df['username']!=user_of_interest) ].iloc[0]['user_rating']

        ret_html = html.Div("We predict that your rating for this beer will be {:.2f}".format(float(prediction)),
                             style={'font-size':'large', 'font-weight':'bold'})
        return ret_html


@app.callback(Output('ranking-results-collabfilt', 'children'),
                [Input('ranking-button-collabfilt', 'n_clicks')],
                [State('username-selection-dropdown-collabfilt', 'value'),
                State('ranking-beer-selection-dropdown-collabfilt', 'value')])
def rank_beers(n_clicks, user_of_interest, beers):

    if n_clicks != None:

        print("Running...")

        query = "SELECT user_rating, beer_name, username FROM prepped_data"
        df = import_table(db_path, query, remove_dups=False)
        df = COSINE_STEP(df, user_of_interest)

        d = {"beer_name":[], "predictions":[]}
        for beer in beers:
            prediction = df[ (df.sort_values('nearest_neighbor_rank')['beer_name'] == beer) & (df['username']!=user_of_interest) ].iloc[0]['user_rating']
            d['beer_name'].append(beer)
            d['predictions'].append(prediction)

        beer_df = pd.DataFrame.from_dict(d)
        beer_df.sort_values('predictions', inplace=True, ascending=False)
        top_beer = beer_df.iloc[0,0]
    
        random_responses = ["Our best guess is you're gonna love {}!", 
                            "Forget the other options, {} should be your next one!"]
        rand_ind = np.random.randint(0,len(random_responses))

        children = [html.Div(random_responses[rand_ind].format(str(top_beer)), style={'font-size':'large', 'font-weight':'bold'}), 
                    html.Br(),
                    html.Div("Full analysis below: ", style={'text-align':'center', 'font-weight':'bold'})]
        for beer in beer_df['beer_name']:
            child = html.Div("{} predicted rating: {:.2f}".format(beer, float(beer_df[beer_df['beer_name']==beer]['predictions'])), style={'text-align':'center', 'font-size':'small'})
            children.append(child)

        ret_html = html.Div(children=children)
        return ret_html

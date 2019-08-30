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
                id = 'username-selection-dropdown-cbf',
                options=username_options(),
                multi = False
            )
       ]),

        # feature selection
        html.Div(className='col-lg-5 my-4', children=[
                html.H4("Select Which Feature Selection You'd Like to Use"),
                dcc.Dropdown(
                    id = 'feature-selection-dropdown-cbf',
                    options = [{'label': 'Simple', 'value': 'simple'},
                                {'label': 'Categorical Encoding of Beer Description', 'value': 'cat-encoding'},
                                {'label': 'Count Vectorizer of Beer Description', 'value': 'count-vect'},
                                {'label': 'TFIDF Vectorizer of Beer Description', 'value': 'tfidf-vect'}],
                    multi = False
                )
        ]),

        html.Div(className='row justify-content-center', children=[
            html.Button('Build Content-Based-Filtering Model', id='model-button-cbf', className='btn btn-outline-primary'),
            dcc.Loading(id="loading-model", children=[html.Div(id="loading-model-output")], type="default"),
        ]),
        html.Div(className='row justify-content-center my-3', children=[
            html.Div(id='model-results-cbf')
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
                        multi = False
                    )
                ]),
            ]),
            html.Div(className='row justify-content-center', children=[
                html.Button('Search', id='prediction-button-cbf', className='btn btn-outline-primary')
            ]),
            html.Div(className='row justify-content-center my-3', children=[
                html.Div(id='prediction-results-cbf')
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
    print(n_clicks, value)
    if value != None:
        if len(value) == 1:
            return {'display': 'none'}
    elif n_clicks != None and value != None and value != []:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

# @app.callback(Output("loading-model-output", "children"), 
#             [Input('model-button-cbf', 'n_clicks')])
# def input_triggers_spinner(n_clicks):
#     return value


@app.callback(Output('model-results-cbf', 'children'),
                [Input('model-button-cbf', 'n_clicks')],
                [State('username-selection-dropdown-cbf', 'value'),
                 State('feature-selection-dropdown-cbf', 'value')])
def build_model(n_clicks, user_of_interest, feature_selection):

    if n_clicks != None:
       
        df = import_table(db_path, query = "SELECT * FROM prepped_data WHERE username = '{}'".format(user_of_interest))
    
        drop_cols =['username', 'beer_name', 'brewery']
        if feature_selection == 'simple':
            user_df = df[df['username']==user_of_interest].drop(drop_cols + ['beer_description'], axis=1, inplace=False)
        elif feature_selection == 'cat-encoding':
            user_df = cat_encoding(df, 'beer_description', drop_cols)
        elif feature_selection == 'count-vect':
            user_df = count_vectorizer(df,'beer_description', drop_cols)
        elif feature_selection == 'tfidf-vect':
            user_df = tfidf_vectorizer(df,'beer_description', drop_cols)

        model, mae, quarter, half = run_model(user_df, 'user_rating')

        d={}
        d['model'] = model
        d['feature_selection'] = feature_selection

        with open('model.pkl', 'wb') as file:
            pickle.dump(d, file)

        return "We have created a model with an accuracy of {:.2f}% within 0.25 stars = and {:.2f}% within 0.5 stars (MAE = {:.2f})".format(quarter, half, mae)

@app.callback(Output('prediction-results-cbf', 'children'),
                [Input('prediction-button-cbf', 'n_clicks')],
                [State('beer-selection-dropdown-cbf', 'value')])
def predict_beer_rating(n_clicks, beer):

    if n_clicks != None:
        with open('model.pkl', 'rb') as file:
            d = pickle.load(file)
            model = d['model']
            feature_selection = d['feature_selection']
    
    
        drop_cols =['username', 'beer_name', 'brewery']
        if feature_selection == 'simple':
            query = "SELECT ABV, IBU, global_rating FROM prepped_data WHERE beer_name = '{}'".format(beer)
            beer_df = import_table(db_path, query, remove_dups=False)

            print(beer_df)
            beer_df['global_rating'] = beer_df['global_rating'].mean()

            beer_df = beer_df[~beer_df.duplicated()]
            print(beer_df)
            prediction = model.predict(beer_df)
            print(prediction)

    # elif feature_selection == 'cat-encoding':
    #     user_df = cat_encoding(df, 'beer_description', drop_cols)
    # elif feature_selection == 'count-vect':
    #     user_df = count_vectorizer(df,'beer_description', drop_cols)
    # elif feature_selection == 'tfidf-vect':
    #     user_df = tfidf_vectorizer(df,'beer_description', drop_cols)

        return "We predict that your rating for this beer will be {:.2f}".format(prediction[0])

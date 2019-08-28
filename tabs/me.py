import dash
import dash_core_components as dcc
import dash_html_components as html 
from dash.dependencies import Input, Output, State
from flask_sqlalchemy import SQLAlchemy
import numpy as np

from app import app

from util import *

layout = html.Div(className = 'container my-4', children =[
    
#    # username section
#    html.Div(className='card', children=[
#        html.Div(className='col-lg-5 m-4', children=[
#             dcc.Dropdown(
#                 id = 'username-selection-dropdown',
#                 options=username_options(),
#                 multi = False
#             )
#        ]),
#     ]),
    # search section
    html.Div(id='search-section', className='card', children = [
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
                        id = 'beer-selection-dropdown',
                        options = beer_options(),
                        multi = True
                    )
                ]),
            ]),
            html.Div(className='row justify-content-center', children=[
                html.Button('Search', id='search-button', className='btn btn-outline-primary')
            ]),
                html.Div(className='row justify-content-center my-3', children=[
                html.Div(id='search-results')
            ]),
            html.Div(className='row justify-content-center my-3', children=[
                html.Img(id='beer-loader', src='/assets/img/beer-loader.gif'),
            ]),
        ]),
    ]),
])
# end container


## callbacks
@app.callback(Output('beer-loader', 'style'),
                [Input('search-button', 'n_clicks')],
                [State('beer-selection-dropdown', 'value')])
def display_beer_loader(n_clicks, value):
    if value != None:
        if len(value) == 1:
            return {'display': 'none'}
    elif n_clicks != None and value != None and value != []:
        time.sleep(0.25)
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(Output('search-results', 'children'),
                [Input('search-button', 'n_clicks')],
                [State('beer-selection-dropdown', 'value')])
def display_beer_loader(n_clicks, value):

    if value != None:
        if len(value) == 1:
            return "Please select more than one beer!"

    if n_clicks != None and value != None and value != []:
        
        features = ['ABV', 'IBU', 'global_rating']

        query = """SELECT * FROM user_extract 
                    WHERE beer_name IN {}""".format(tuple(value))

        df = import_table('data/beer.db', query=query)
        df = df[~df.duplicated()]
        df = impute_na(df, features=features)

        # convert IBU from str to float (str due to impute)
        if 'IBU' in features:
            df['IBU'] = df['IBU'].astype(float)

        # take mean of global rating since it changes based on when users data is scraped
        if 'global_rating' in features:
            df = df.groupby('beer_name').mean()
        
        df = df[~df.duplicated()]

        # scale features
        with open('models/X_scaler-final_model.pkl', 'rb') as file:
            X_scaler = pickle.load(file)
        with open('models/y_scaler-final_model.pkl', 'rb') as file:
            y_scaler = pickle.load(file)
        X_scaler.fit(df[features])
        df[features] = X_scaler.transform(df[features])
        
        # import model and predict
        with open('models/final_model.pkl', 'rb') as file:
            model = pickle.load(file)
        preds = model.predict(df[features])

        # inverse scale and rebuild df
        predictions = y_scaler.inverse_transform(preds)
        df[features] = X_scaler.inverse_transform(df[features])
        df['predictions'] = predictions

        # rename global rating
        df.rename({'gloabl_rating': 'average_global_rating'}, inplace=True, axis=1)

        # sort by original order
        df = df.loc[value]

        # sort by predictions
        df = df.sort_values('predictions', ascending=False)

        random_responses = ["Our best guess is you're gonna love {}", 
                            "Have a cold {} for us!", 
                            "Don't even think about any other option, {} is your next cold one!"]
        rand_ind = np.random.randint(0,len(random_responses))

        return random_responses[rand_ind].format(str(df.index[0]))
    
    elif n_clicks != None:
        return ":( Please select at least one beer"

    else:
        pass


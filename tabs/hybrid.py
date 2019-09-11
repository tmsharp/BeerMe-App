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
                id = 'username-selection-dropdown-hybrid',
                options=username_options(),
                multi = False
            )
       ]),

        # feature selection
        html.Div(className='col-lg-5 my-4', children=[
                html.H4("Select Which Feature Selection You'd Like to Use"),
                dcc.Dropdown(
                    id = 'feature-selection-dropdown-hybrid',
                    options = [{'label': 'Simple', 'value': 'simple'},
                                {'label': 'Categorical Encoding of Beer Description', 'value': 'cat-encoding'},
                                {'label': 'Count Vectorizer of Beer Description', 'value': 'count-vect'},
                                {'label': 'TFIDF Vectorizer of Beer Description', 'value': 'tfidf-vect'}],
                    multi = False
                )
        ]),

        html.Div(className='row justify-content-center', children=[
            html.Button('Build Hybrid Model', id='model-button-hybrid', className='btn btn-outline-primary'),
            dcc.Loading(id="loading-model", children=[html.Div(id="loading-model-output")], type="default"),
        ]),
        html.Div(className='row justify-content-center my-3', children=[
            html.Div(id='model-results-hybrid')
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
                        id = 'beer-selection-dropdown-hybrid',
                        options = beer_options(),
                        multi = False
                    )
                ]),
            ]),
            html.Div(className='row justify-content-center', children=[
                html.Button('Predict', id='prediction-button-hybrid', className='btn btn-outline-primary')
            ]),
            html.Div(className='row justify-content-center my-3', children=[
                html.Div(id='prediction-results-hybrid')
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
                        id = 'ranking-beer-selection-dropdown-hybrid',
                        options = beer_options(),
                        multi = True
                    )
                ]),
            ]),
            html.Div(className='row justify-content-center', children=[
                html.Button('Rank', id='ranking-button-hybrid', className='btn btn-outline-primary')
            ]),
            html.Div(className='row justify-content-center my-3', children=[
                html.Div(id='ranking-results-hybrid')
            ]),
        ]),
    ]),

])
# end container


## callbacks
@app.callback(Output('beer-loader-hybrid', 'style'),
                [Input('search-button-hybrid', 'n_clicks')],
                [State('beer-selection-dropdown-hybrid', 'value')])
def display_beer_loader(n_clicks, value):
    print(n_clicks, value)
    if value != None:
        if len(value) == 1:
            return {'display': 'none'}
    elif n_clicks != None and value != None and value != []:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(Output('model-results-hybrid', 'children'),
                [Input('model-button-hybrid', 'n_clicks')],
                [State('username-selection-dropdown-hybrid', 'value'),
                 State('feature-selection-dropdown-hybrid', 'value')])
def build_model(n_clicks, user_of_interest, feature_selection):

    if n_clicks != None:
       
        df = import_table(db_path, query = "SELECT username, beer_name, beer_description, ABV, IBU, global_rating, user_rating FROM prepped_data")
    
        # feature prep
        if feature_selection == 'simple':
            hybrid_df = df.drop(['beer_description'], axis=1, inplace=False)
           
        elif feature_selection == 'cat-encoding':
            df = import_table(db_path, query = "SELECT username, user_rating, beer_name, beer_description, ABV, IBU, global_rating FROM prepped_data")
            hybrid_df = cat_encoding(df, 'beer_description')
        elif feature_selection == 'count-vect':
            df = import_table(db_path, query = "SELECT username, user_rating, beer_name, beer_description, ABV, IBU, global_rating FROM prepped_data")
            hybrid_df = count_vectorizer(df, 'beer_description')
        elif feature_selection == 'tfidf-vect':
            df = import_table(db_path, query = "SELECT username, user_rating, beer_name, beer_description, ABV, IBU, global_rating FROM prepped_data")
            hybrid_df = count_vectorizer(df, 'beer_description')

        print(hybrid_df)
        model_list, mae_list, quarter_list, half_list = run_hybrid(hybrid_df, user_of_interest, 'user_rating')

        mae = min(i for i in mae_list if i > 0)
        ind = mae_list.index(mae)

        print(mae_list, ind)
        print(len(model_list), len(mae_list), len(quarter_list), len(half_list))

        model = model_list[ind]
        quarter = quarter_list[ind]
        half = half_list[ind]

        d={}
        d['model'] = model
        d['feature_selection'] = feature_selection

        with open('hybrid-model.pkl', 'wb') as file:
            pickle.dump(d, file)

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

@app.callback(Output('prediction-results-hybrid', 'children'),
                [Input('prediction-button-hybrid', 'n_clicks')],
                [State('beer-selection-dropdown-hybrid', 'value')])
def predict_beer_rating(n_clicks, beer):

    if n_clicks != None:
        with open('hybrid-model.pkl', 'rb') as file:
            d = pickle.load(file)
            model = d['model']
            feature_selection = d['feature_selection']
    
    
        if feature_selection == 'simple':
            query = "SELECT ABV, IBU, global_rating FROM prepped_data WHERE beer_name = '{}'".format(beer)
            beer_df = import_table(db_path, query, remove_dups=False)
            beer_df['global_rating'] = beer_df['global_rating'].mean()
            beer_df = beer_df[~beer_df.duplicated()]
            prediction = model.predict(beer_df)

        # elif feature_selection == 'cat-encoding':
        #     query = "SELECT ABV, IBU, global_rating, beer_description FROM prepped_data WHERE beer_name = '{}'".format(beer)
        #     beer_df = import_table(db_path, query, remove_dups=False)
        #     beer_df['global_rating'] = beer_df['global_rating'].mean()
        #     beer_df = beer_df[~beer_df.duplicated()]
        #     # feature encoding
        #     beer_df = cat_encoding(beer_df, 'beer_description')
        #     #
        #     print("HERE")
        #     print(beer_df)
        #     prediction = model.predict(beer_df)
        # elif feature_selection == 'count-vect':
        #     user_df = count_vectorizer(df,'beer_description')
        # elif feature_selection == 'tfidf-vect':
        #     user_df = tfidf_vectorizer(df,'beer_description')
        
        ret_html = html.Div("We predict that your rating for this beer will be {:.2f}".format(prediction[0]),
                             style={'font-size':'large', 'font-weight':'bold'})
        return ret_html


@app.callback(Output('ranking-results-hybrid', 'children'),
                [Input('ranking-button-hybrid', 'n_clicks')],
                [State('ranking-beer-selection-dropdown-hybrid', 'value')])
def rank_beers(n_clicks, beers):

    if n_clicks != None:
        with open('hybrid-model.pkl', 'rb') as file:
            d = pickle.load(file)
            model = d['model']
            feature_selection = d['feature_selection']
    
    
        drop_cols =['username', 'beer_name', 'brewery']
        if feature_selection == 'simple':
            query = "SELECT beer_name, ABV, IBU, global_rating FROM prepped_data WHERE beer_name in {}".format(tuple(beers))
            beer_df = import_table(db_path, query, remove_dups=False)

            beer_df['global_rating'] = beer_df.groupby("beer_name").transform(lambda x: x.fillna(x.mean()))['global_rating']
            beer_df = beer_df[~beer_df.duplicated('beer_name')]
            
            predictions = model.predict(beer_df.drop('beer_name', axis=1))
            beer_df['predictions'] = predictions
            beer_df.sort_values('predictions', inplace=True, ascending=False)
            top_beer = beer_df.iloc[0,0]
            
            random_responses = ["Our best guess is you're gonna love {}!", ]
                                # "Forget the other options, {} should be your next cold one!"]
            rand_ind = np.random.randint(0,len(random_responses))

            children = [html.Div(random_responses[rand_ind].format(str(top_beer)), style={'font-size':'large', 'font-weight':'bold'}), 
                        html.Br(),
                        html.Div("Full analysis below: ", style={'text-align':'center', 'font-weight':'bold'})]
            for beer in beer_df['beer_name']:
                child = html.Div("{} predicted rating: {:.2f}".format(beer, float(beer_df[beer_df['beer_name']==beer]['predictions'])), style={'text-align':'center', 'font-size':'small'})
                children.append(child)

            ret_html = html.Div(children=children)
            return ret_html

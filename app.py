"""
This app creates a responsive sidebar layout with dash-bootstrap-components and
some custom css with media queries.

When the screen is small, the sidebar moved to the top of the page, and the
links get hidden in a collapse element. We use a callback to toggle the
collapse when on a small screen, and the custom CSS to hide the toggle, and
force the collapse to stay open when the screen is large.

dcc.Location is used to track the current location. There are two callbacks,
one uses the current location to render the appropriate page content, the other
uses the current location to toggle the "active" properties of the navigation
links.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import dash_auth
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np
# import urllib.parse
# import os
# import datetime

import plotly.graph_objs as go
import plotly.express as px

from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
# import warnings
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim import corpora, models, similarities
from gensim.corpora import Dictionary
from gensim.models import LsiModel, LogEntropyModel
from gensim.models.phrases import Phraser, Phrases
from gensim.models.tfidfmodel import TfidfModel
# from gensim.matutils import sparse2full

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

# ===== Read Data =====
# master = pd.read_pickle('data/20200527-master.pickle')
master = pd.read_pickle('data/20210401-master.pickle')
master = master[['No', 'Symbol', 'Type', 'Year', 'Date', 'Title','Proponents', 'Pillars', 'Topics', 'Available','Source', 'Use', 'FileID']]

files = master[~master['Proponents'].isin(['Secretariat','Chair'])].groupby(['FileID','Year','Proponents','Pillars','Topics']).size().reset_index()

# "Stack" multiple proponents and pillars to different rows
files['ProponentsList'] = files['Proponents'].str.split(',')
files['PillarsList'] = files['Pillars'].str.split(',')
files['TopicList'] = files['Topics'].str.split(',')

proponent = files.apply(lambda x: pd.Series(x['ProponentsList']), axis=1).stack().reset_index(level=1, drop=True)
proponent.name = 'Proponent'
pillar = files.apply(lambda x: pd.Series(x['PillarsList']), axis=1).stack().reset_index(level=1, drop=True)
pillar.name = 'Pillar'
topic = files.apply(lambda x: pd.Series(x['TopicList']), axis=1).stack().reset_index(level=1, drop=True)
topic.name = 'Topic'

files = files.join(pillar)
files = files.join(proponent)
files = files.join(topic)

files['Proponent'] = files['Proponent'].str.strip()
files['Pillar'] = files['Pillar'].str.strip()
files['Topic'] = files['Topic'].str.strip()

# Dictionary for dropdown selections: add "All"
dict_pillar = ['All'] + list(files['Pillar'].unique())
dict_pillar = dict(zip(dict_pillar, dict_pillar))

dict_proponent = list(files['Proponent'].unique())
dict_proponent.sort()
dict_proponent = ['All'] + dict_proponent
dict_proponent = dict(zip(dict_proponent, dict_proponent))

dict_topic = list(files['Topic'].unique())
dict_topic.sort()
dict_topic = dict(zip(dict_topic, dict_topic))


# Text data
textdata = pd.read_pickle('data/20200527-doc-data-keyterms-0.3.pickle')
textdata = textdata [['Doc', 'Text', 'Len', 'KTTextRankStr', 'KTScakeStr',]]


# For Similarity
# fil = pd.read_pickle('data/20210401-master.pickle')
# # files = files[['No', 'Symbol', 'Type', 'Year', 'Date', 'Title','Proponents', 'Pillars', 'Topics', 'Available','Source', 'Use', 'FileKey']]
# fil = fil[['No', 'Symbol', 'Type', 'Year','Date',  'Proponents', 'Pillars', 'Topics', 'FileID']]
# fil['TopicsList'] = fil['Topics'].str.split(',')
# topics = fil.apply(lambda x: pd.Series(x['TopicsList']), axis=1).stack().reset_index(level=1, drop=True)
#
# topics = [x.strip() for x in topics]
# topics = list(set(topics))
# topics.sort()
# dict_topic = dict(zip(topics, topics))



# data = pd.read_pickle('data/data_preprocessing_phrase_20210404.pickle')
data = pd.read_pickle('data/data_preprocessing_phrase_20210404.pickle')
data = data[data['text'].str.len()>100]

data = data.groupby('FileID')['text'].apply(lambda x: ' '.join(x)).reset_index()

data = pd.merge(data[['FileID','text']], master, left_on='FileID',right_on='FileID')
data['Words'] = data['text'].str.split(' ')
# print(data.shape)
# data.head()


# function to calculate similarity
def calc_similarity(ids, docs, kRandom=3, nClusters=3, sortCluster=True):
    # ids: list of IDs identifying texts
    # docs: list of docs

    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    # # TF-IDF
    # tfidf = models.TfidfModel(corpus)
    # index = similarities.MatrixSimilarity(tfidf[corpus])
    # sims = index[tfidf[corpus]]
    # df_sims = pd.DataFrame(sims, index=ids,columns=ids)

    # Log ent
    log_ent = LogEntropyModel(corpus)
    index = similarities.MatrixSimilarity(log_ent[corpus])
    sims = index[log_ent[corpus]]
    df_sims = pd.DataFrame(sims, index=ids, columns=ids)

    if sortCluster == True:
        # Ordered by clusters and distances
        X = df_sims.copy()
        model= KMeans(n_clusters=nClusters, random_state=kRandom)

    #     FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
    #     clusassign = model.fit_predict(X.as_matrix())
    #     min_dist = np.min(cdist(X.as_matrix(), model.cluster_centers_, 'euclidean'), axis=1)

        clusassign = model.fit_predict(X.values)
        min_dist = np.min(cdist(X.values, model.cluster_centers_, 'euclidean'), axis=1)

        Y = pd.DataFrame(min_dist, index=X.index, columns=['Center_euclidean_dist'])
        Z = pd.DataFrame(clusassign, index=X.index, columns=['cluster_ID'])
        A = pd.concat([Y,Z], axis=1)
        A = A.sort_values(['cluster_ID', 'Center_euclidean_dist']).reset_index()

        namelist= A['index'].tolist()
        df_sim_sorted = pd.DataFrame(namelist,columns=['NameSort'])
        df_sim_sorted = pd.merge(df_sim_sorted, df_sims, left_on='NameSort', right_index=True).set_index('NameSort')
        df_sim_sorted = df_sim_sorted[namelist]

        return df_sim_sorted
    else:
        return df_sims






# ===== App =====
# Username and password for login
VALID_USERNAME_PASSWORD_PAIRS = {'wto': 'wto'}

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']
# external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/3.3.7/css/bootstrap.min.css']

# with "__name__" local css under assets is also included
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)
app.title = 'Trade News Tracker'
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=UA-62289743-8"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
          gtag('config', 'UA-62289743-8');
        </script>

        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

server = app.server
app.config.suppress_callback_exceptions = True

# we use the Row and Col components to construct the sidebar header
# it consists of a title, and a toggle, the latter is hidden on large screens
sidebar_header = dbc.Row(
    [
        # dbc.Col(html.H3("Ag Texts", className="display-4000")),
        dbc.Col(html.Img(src=app.get_asset_url("logo.png"), width="130px", style={'margin-left':'15px'})),
        dbc.Col(
            html.Button(
                # use the Bootstrap navbar-toggler classes to style the toggle
                html.Span(className="navbar-toggler-icon"),
                className="navbar-toggler",
                # the navbar-toggler classes don't set color, so we do it here
                style={
                    "color": "rgba(0,0,0,.5)",
                    "bordercolor": "rgba(0,0,0,.1)",
                },
                id="toggle",
            ),
            # the column containing the toggle will be only as wide as the
            # toggle, resulting in the toggle being right aligned
            width="auto",
            # vertically align the toggle in the center
            align="center",
        ),
    ]
)

sidebar = html.Div(
    [
        sidebar_header,
        # we wrap the horizontal rule and short blurb in a div that can be
        # hidden on a small screen
        html.Div(
            [
                html.Br(),
                html.Br(),
                # html.P(
                #     "Follow the trade news with data",
                    # className="lead",
                # ),1
            ],
            id="blurb",
        ),
        # use the Collapse component to animate hiding / revealing links
        dbc.Collapse(
            dbc.Nav(
                [
                    dbc.NavLink("Stats", href="/page-1", id="page-1-link"),
                    dbc.NavLink("Similarity", href="/page-2", id="page-2-link"),
                    dbc.NavLink("Text Data", href="/page-3", id="page-3-link"),
                    dbc.NavLink("WordCloud", href="/page-5", id="page-5-link"),
                    dbc.NavLink("About", href="/page-4", id="page-4-link"),
                ],
                vertical=True,
                pills=False,
            ),
            id="collapse",
            # id="sidebar",
        ),

        html.Div([
                    html.Hr(),
                    html.P(
                        "V.20210425",
                        # className="lead",
                    ),
                ],
            id="blurb-bottom",
            ),
    ],
    id="sidebar",
)

content = html.Div(id="page-content")
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 6)],
    [Input("url", "pathname")],
)

def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False, False, False
    return [pathname == f"/page-{i}" for i in range(1, 6)]

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page-1"]:
        return html.Div([
                        # Chart 1
                        dbc.Row([
                            dbc.Col([
                                html.H3('Summary Stats', style={'font-weight': 'bold'}),
                                html.P(
                                    id="description",
                                    children=dcc.Markdown(
                                      children=(
                                        '''
                                        Members' submissiong, Chair and Secretariat summaries/notes are not included.
                                        ''')
                                    )
                                ),
                                html.Br(),
                                html.H6('Number of Proposals by year', style={'font-weight': 'bold'}),
                            ], lg=10),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Label('Select Pillar:'),
                                dcc.Dropdown(
                                    id='stat-year-dropdown-pillar',
                                    options=[{'label': v, 'value': k}
                                                for k, v in dict_pillar.items()],
                                    multi=False,
                                    value= 'All',
                                ),
                            ], lg=4),
                            dbc.Col([
                                html.Label('Select Proponent:'),
                                dcc.Dropdown(
                                    id='stat-year-dropdown-proponent',
                                    options=[{'label': v, 'value': k}
                                                for k, v in dict_proponent.items()],
                                    multi=False,
                                    value= 'All',
                                ),
                            ], lg=4)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(
                                    id='stat-plot-year-pillar-proponent'
                                ),
                            ], lg=10),
                        ]),

                        # Chart 2
                        dbc.Row([
                            dbc.Col([
                                html.Label('Select topic:'),
                                dcc.Dropdown(
                                    id='stat-year-dropdown-topic',
                                    options=[{'label': v, 'value': k}
                                                for k, v in dict_topic.items()],
                                    multi=False,
                                    value= 'All',
                                ),
                            ], lg=4),
                            dbc.Col([
                                html.Label('Select Proponent:'),
                                dcc.Dropdown(
                                    id='stat-year-dropdown-proponent2',
                                    options=[{'label': v, 'value': k}
                                                for k, v in dict_proponent.items()],
                                    multi=False,
                                    value= 'All',
                                ),
                            ], lg=4)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(
                                    id='stat-plot-year-topic-proponent'
                                ),
                            ], lg=10),
                        ]),



                    ])

    elif pathname in ["/page-2"]:
        return html.Div([
                        dbc.Row([
                            # dbc.Col(lg=1),
                            dbc.Col([
                                html.H3('Similarity within topics', style={'font-weight': 'bold'}),
                                # html.H5('Updata on 14 June 2020'),
                                html.P(
                                    id="description",
                                    children=dcc.Markdown(
                                      children=(
                                        '''
                                        Similarity between two docs in a topic.
                                        ''')
                                    )
                                ),
                                html.Br(),
                                # html.H6('Number of Proposals by year', style={'font-weight': 'bold'}),
                                # dcc.Dropdown(
                                #     id='my-dropdown',
                                #     options=[{'label': v, 'value': k}
                                #                 for k, v in dict_pillar.items()],
                                #     multi=False,
                                #     value= [0,1,2,3,4,5,6,7,8,9],
                                # ),
                            ], lg=10),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Label('Select Topic:'),
                                dcc.Dropdown(
                                    id='plot-year-dropdown-pillar1',
                                    options=[{'label': v, 'value': k}
                                                for k, v in dict_topic.items()],
                                    multi=False,
                                    value= 'COT',
                                ),
                            ], lg=4),
                            # dbc.Col([
                            #     html.Label('Select Proponent:'),
                            #     dcc.Dropdown(
                            #         id='plot-year-dropdown-proponent1',
                            #         options=[{'label': v, 'value': k}
                            #                     for k, v in dict_proponent.items()],
                            #         multi=False,
                            #         value= 'All',
                            #     ),
                            # ], lg=4)
                        ]),
                        dbc.Row([
                            # dbc.Col(lg=1),
                            # dbc.Col([
                            #     dcc.Graph(
                            #         id='top_topics'
                            #     ),
                            # ], lg=3),
                            dbc.Col([
                                dcc.Graph(
                                    id='plot_year1'
                                ),
                            ], lg=10),
                        ]),
                    ])

    elif pathname == "/page-3":
        return html.Div([
                html.H5('Text Data - Preprocessed: stopwords removed; words in original form; without numbers; predefined phrase linked by "_"'),
                dash_table.DataTable(
                    id='table',
                    # columns=[{"name": i, "id": i} for i in textdata.columns],
                    # data=textdata.to_dict('records'),

                    columns=[{"name": i, "id": i} for i in data[['FileID','text']].columns],
                    data=data[['FileID','text']].to_dict('records'),

                    editable=False,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    column_selectable=False,
                    row_selectable=False,
                    row_deletable=False,
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current= 0,
                    page_size= 20,
                    # style_cell_conditional=[
                    #     {'if': {'column_id': 'Member'},
                    #      'width': '100px'},
                    # ]
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto'
                    },
                    style_cell={
                        # 'height': 'auto',
                        'minWidth': '20px', 'maxWidth': '300px',
                        # 'whiteSpace': 'normal',
                        'textAlign': 'left',
                        'verticalAlign': 'top',
                        'fontSize':12,
                    },
                )
            ])

    elif pathname == "/page-4":
        return html.Div([
                        dbc.Row([
                            # dbc.Col(lg=1),
                            dbc.Col([
                                html.H3('About Ag Doc Analyser', style={'font-weight': 'bold'}),
                                # html.H5('Updata on 14 June 2020'),
                                html.P(
                                    id="description",
                                    children=dcc.Markdown(
                                      children=('''
                                                    Adipiscing lacus ut elementum, nec duis, tempor litora turpis dapibus. Imperdiet cursus odio tortor in elementum.
                                                    Egestas nunc eleifend feugiat lectus laoreet, vel nunc taciti integer cras. Hac pede dis, praesent nibh ac dui mauris sit.
                                                    Pellentesque mi, facilisi mauris, elit sociis leo sodales accumsan. Iaculis ac fringilla torquent lorem consectetuer,
                                                    sociosqu phasellus risus urna aliquam, ornare.

                                                    Adipiscing lacus ut elementum, nec duis, tempor litora turpis dapibus. Imperdiet cursus odio tortor in elementum.
                                                    Egestas nunc eleifend feugiat lectus laoreet, vel nunc taciti integer cras. Hac pede dis, praesent nibh ac dui mauris sit.
                                                    Pellentesque mi, facilisi mauris, elit sociis leo sodales accumsan. Iaculis ac fringilla torquent lorem consectetuer,
                                                    sociosqu phasellus risus urna aliquam, ornare.
                                        ''')
                                    )
                                ),
                                # html.Br(),
                                # html.H4('Select a topic or topics, which are defined by a set of keywords:', style={'font-weight': 'bold'}),
                                # dcc.Dropdown(
                                #     id='my-dropdown',
                                #     options=[{'label': v, 'value': k}
                                #                 for k, v in dict_pillar.items()],
                                #     multi=False,
                                #     value= [0,1,2,3,4,5,6,7,8,9],
                                # ),
                            ], lg=8),
                        ]),
                        ]),
    elif pathname in ["/page-5"]:
        # return html.H5("Content to be added page 2.")
        return html.Div([
                        dbc.Row([
                            # dbc.Col(lg=1),
                            dbc.Col([
                                html.H3('WordCloud by topic', style={'font-weight': 'bold'}),
                                # html.H5('Updata on 14 June 2020'),
                                html.P(
                                    id="description",
                                    children=dcc.Markdown(
                                      children=(
                                        '''
                                        Word frequency in a topic.
                                        ''')
                                    )
                                ),
                                html.Br(),
                            ], lg=10),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Label('Select Topic:'),
                                dcc.Dropdown(
                                    id='plot-year-dropdown-pillar2',
                                    options=[{'label': v, 'value': k}
                                                for k, v in dict_topic.items()],
                                    multi=False,
                                    value= 'COT',
                                ),
                            ], lg=4),
                            # dbc.Col([
                            #     html.Label('Select Proponent:'),
                            #     dcc.Dropdown(
                            #         id='plot-year-dropdown-proponent1',
                            #         options=[{'label': v, 'value': k}
                            #                     for k, v in dict_proponent.items()],
                            #         multi=False,
                            #         value= 'All',
                            #     ),
                            # ], lg=4)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(
                                    id='plot_year2'
                                ),
                            ], lg=10),
                        ]),
                    ])

    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


@app.callback(Output('stat-plot-year-pillar-proponent', 'figure'),
             [Input('stat-year-dropdown-pillar', 'value'),
              Input('stat-year-dropdown-proponent', 'value'),
             ])
def update_graph(select_pillar, select_proponent):
    if not select_pillar:
        raise PreventUpdate
    if not select_proponent:
        raise PreventUpdate

    # select_pillar = 'DS'
    # select_pillar = 'All'
    if select_pillar == 'All':
        select_pillar = list(files['Pillar'].unique())
    else:
        select_pillar = [select_pillar]

    # select_proponent = 'Australia'
    # select_proponent = 'All'

    if select_proponent == 'All':
        select_proponent = list(files['Proponent'].unique())
    else:
        select_proponent = [select_proponent]

    df_plot = files[(files['Pillar'].isin(select_pillar)) &
                    (files['Proponent'].isin(select_proponent))].groupby('Year')['FileID'].nunique().reset_index(name='Count')

    figure = {
        'data': [go.Bar(
            x=df_plot['Year'].tolist(),
            y=df_plot['Count'].tolist(),
        )],
        'layout':go.Layout(
            # title= 'Number of news items by topics',
            yaxis = dict(
                # autorange=True,
                automargin=True
            ),
            xaxis=dict(
                title='',
                tickmode='linear'
            )
        )
    }
    return figure













@app.callback(Output('stat-plot-year-topic-proponent', 'figure'),
             [Input('stat-year-dropdown-topic', 'value'),
              Input('stat-year-dropdown-proponent2', 'value'),
             ])
def update_graph(select_topic, select_proponent):
    if not select_topic:
        raise PreventUpdate
    if not select_proponent:
        raise PreventUpdate

    # select_topic = 'DS'
    # select_topic = 'All'
    if select_topic == 'All':
        select_topic = list(files['Topic'].unique())
    else:
        select_topic = [select_topic]

    # select_proponent = 'Australia'
    # select_proponent = 'All'

    if select_proponent == 'All':
        select_proponent = list(files['Proponent'].unique())
    else:
        select_proponent = [select_proponent]

    df_plot = files[(files['Topic'].isin(select_topic)) &
                    (files['Proponent'].isin(select_proponent))].groupby('Year')['FileID'].nunique().reset_index(name='Count')

    figure = {
        'data': [go.Bar(
            x=df_plot['Year'].tolist(),
            y=df_plot['Count'].tolist(),
        )],
        'layout':go.Layout(
            # title= 'Number of news items by topics',
            yaxis = dict(
                # autorange=True,
                automargin=True
            ),
            xaxis=dict(
                title='',
                tickmode='linear'
            )
        )
    }
    return figure


















# Similarity
@app.callback(Output('plot_year1', 'figure'),
             [Input('plot-year-dropdown-pillar1', 'value'),
              # Input('plot-year-dropdown-proponent1', 'value'),
             ])
def update_graph(select_pillar1):
    if not select_pillar1:
        raise PreventUpdate
    # if not select_proponent1:
    #     raise PreventUpdate

    # # select_pillar = 'DS'
    # # select_pillar = 'All'
    # if select_pillar == 'All':
    #     select_pillar = list(files['Pillar'].unique())
    # else:
    #     select_pillar = [select_pillar]
    #
    # # select_proponent = 'Australia'
    # # select_proponent = 'All'
    #
    # if select_proponent == 'All':
    #     select_proponent = list(files['Proponent'].unique())
    # else:
    #     select_proponent = [select_proponent]

    # topic = 'TRQ'
    topic = select_pillar1

    selected = data[data['Topics'].str.contains(topic)].sort_values('Year')
    selected_symbols = selected['Symbol'].tolist()
    selected_docs = selected['Words'].tolist()
    sim_sorted = calc_similarity(selected_symbols, selected_docs, kRandom=3, nClusters=3, sortCluster=False)

    fig = px.imshow(sim_sorted)
    fig.update_layout(
        height=800,
        width=800,
        font=dict(
            family="Courier New, monospace",
            size=8,
            color="RebeccaPurple"
                ),
        xaxis=dict(autorange='reversed')
        )
    return fig






# wordcloud
@app.callback(Output('plot_year2', 'figure'),
             [Input('plot-year-dropdown-pillar2', 'value'),
              # Input('plot-year-dropdown-proponent1', 'value'),
             ])
def update_graph(select_pillar2):
    if not select_pillar2:
        raise PreventUpdate

    topic = select_pillar2
    text = " ".join(review for review in data[data["Topics"].str.contains(topic)]['text'])
    # Generate a word cloud image
    wordcloud = WordCloud(background_color="white", width=900, height=600).generate(text)

    # Display the generated image:
    # the matplotlib way:
    # plt.figure(figsize=[18,9])
    # plt=px.imshow(wordcloud, interpolation='bilinear')
    fig=px.imshow(wordcloud)
    fig.update_layout(
        width=800,
        height=650,
        
        yaxis_visible=False,
        xaxis_visible=False,
        font=dict(
            family="Courier New, monospace",
            size=8,
            color="RebeccaPurple"
                ),
        )
    return fig














# General modules
@app.callback(
    Output("collapse", "is_open"),
    [Input("toggle", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True)

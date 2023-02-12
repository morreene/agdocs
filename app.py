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
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_auth

import pandas as pd
import numpy as np
# import urllib.parse
# import os
# import datetime

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.cluster import KMeans
# from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
# import warnings
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# from gensim import corpora, models, similarities
from gensim import corpora,  similarities

# from gensim.corpora import Dictionary
# from gensim.models import LsiModel, LogEntropyModel
from gensim.models import LogEntropyModel

# from gensim.models.phrases import Phraser, Phrases
# from gensim.models.tfidfmodel import TfidfModel
# from gensim.matutils import sparse2full

# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from wordcloud import WordCloud

# import matplotlib.pyplot as plt

# ===== Read Data =====

DASH_MASTER = 'data/docmaster.pickle'
DASH_DATA = 'data/data-preprocessed.pickle'
DASH_WORD_FREQ = 'data/word-frequency.pickle'
DASH_TFIDF = 'data/tfidf.pickle'

master = pd.read_pickle(DASH_MASTER)
master = master[['No', 'Symbol', 'Type', 'Year', 'Date', 'Title','Authors', 'Pillars', 'Topics', 'Available','Source', 'Use', 'FileID']]

# files = master[~master['Authors'].isin(['Secretariat','Chair'])][['FileID','Year','Authors','Pillars','Topics']]
files = master[['FileID','Year','Authors','Pillars','Topics']].copy()

member = master[~master['Authors'].isin(['Secretariat','Chair'])]['Authors'].tolist()

# "Stack" multiple proponents and pillars to different rows
files['AuthorsList'] = files['Authors'].str.split(',')
files['PillarsList'] = files['Pillars'].str.split(',')
files['TopicsList'] = files['Topics'].str.split(',')

proponent = files.apply(lambda x: pd.Series(x['AuthorsList']), axis=1).stack().reset_index(level=1, drop=True)
proponent.name = 'Author'
pillar = files.apply(lambda x: pd.Series(x['PillarsList']), axis=1).stack().reset_index(level=1, drop=True)
pillar.name = 'Pillar'
topic = files.apply(lambda x: pd.Series(x['TopicsList']), axis=1).stack().reset_index(level=1, drop=True)
topic.name = 'Topic'

files = files.join(pillar)
files = files.join(proponent)
files = files.join(topic)

files['Author'] = files['Author'].str.strip()
files['Pillar'] = files['Pillar'].str.strip()
files['Topic'] = files['Topic'].str.strip()

# Dictionary for dropdown selections: add "All"
dict_pillar = ['All'] + list(files['Pillar'].unique())
dict_pillar = dict(zip(dict_pillar, dict_pillar))

dict_proponent = list(files['Author'].unique())
dict_proponent.sort()
dict_proponent = ['Chair','Secretariat','All Members & Groups'] + dict_proponent
dict_proponent = dict(zip(dict_proponent, dict_proponent))

dict_topic = list(files['Topic'].unique())
dict_topic.sort()
dict_topic = ['All'] + dict_topic
dict_topic = dict(zip(dict_topic, dict_topic))

dict_year = list(files['Year'].astype(str).unique())
dict_year.sort()
dict_year = ['All'] + dict_year
dict_year = dict(zip(dict_year, dict_year))



# Text data
def load_data():

    data = pd.read_pickle(DASH_DATA, compression='zip')
    # data = data[data['text'].str.len()>100]
    #
    # # Consolidate paras to doc
    # data1 = data.groupby('FileID')['text'].apply(lambda x: ' '.join(x)).reset_index()
    # data2 = data.groupby('FileID')['Text'].apply(lambda x: ' '.join(x)).reset_index()
    # data = pd.merge(data1,data2,on='FileID')
    #
    # data = pd.merge(data[['FileID','Text','text']], master, left_on='FileID',right_on='FileID')
    # data['Words'] = data['text'].str.split(' ')
    return data

# for terms frequency
df_cv = pd.read_pickle(DASH_WORD_FREQ, compression='zip')
all_terms = list(df_cv.columns)

allfileid = files['FileID'].unique().tolist()

# for tf-idf keywords
def load_tfidf():
    tfidf = pd.read_pickle(DASH_TFIDF)
    # allfileid = tfidf['FileID'].unique().tolist()
    return tfidf

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

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']
# external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/3.3.7/css/bootstrap.min.css']

# with "__name__" local css under assets is also included
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

# Username and password for login
VALID_USERNAME_PASSWORD_PAIRS = {'wto': 'wto'}
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
                [   dbc.NavLink("About",            href="/page-1", id="page-1-link"),
                    dbc.NavLink("Text Data",        href="/page-2", id="page-2-link"),
                    dbc.NavLink("Document Stats",   href="/page-3", id="page-3-link"),
                    dbc.NavLink("Similarity",       href="/page-4", id="page-4-link"),
                    dbc.NavLink("WordCloud",        href="/page-5", id="page-5-link"),
                    dbc.NavLink("Term Freqency",    href="/page-7", id="page-7-link"),
                    dbc.NavLink("Keywords TF-IDF",  href="/page-8", id="page-8-link"),
                    dbc.NavLink("Network: Member",  href="/page-6", id="page-6-link"),
                    dbc.NavLink("Network: Docs",    href="/page-9", id="page-9-link"),
                ],
                vertical=True,
                pills=False,
            ),
            id="collapse",
            # id="sidebar",
        ),

        html.Div([  
            # html.Hr(),
                    html.P(
                        "Version 20230212",
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
    [Output(f"page-{i}-link", "active") for i in range(1, 9)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False, False, False, False, False, False
    return [pathname == f"/page-{i}" for i in range(1, 9)]






@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page-1"]:
        return html.Div([
                dbc.Container([
                            html.H4("About AgDocs Analyzer - the data, charts and app", className="display-about"),
                            html.P(
                                "Getting insights of 20+ year history from the docs",
                                className="lead",
                            ),
                            html.Hr(className="my-2"),
                            dcc.Markdown(
                                '''
                                This tool, developed by MAIS (ERSD), presents the results of a joint project between AGCD and ERSD-MAIS. The project aimed to:

                                * Compile all documents related to agriculture negotiations in the WTO since 1997, when the analysis and exchange of information process began.
                                * Create a method and tool for quantitative analysis of these documents, organized by nature, topic, and content.

                                ##### Text Data
                                This section displays the full collection of documents and the extracted text features. Only the text
                                is extracted from Word/PDF documents, while tables and formatting are ignored. The "text" column contains 
                                processed text data, produced by a NLP program, including cleaning, standardizing, removing stopwords, stemming, and other techniques.

                                ##### Document Statistics
                                This section includes a series of charts that provide summary statistics of the documents, including 
                                counts of documents by year, member, and pillar (or category). It should be noted that the documents in the 
                                collection are of different nature, although efforts have been made to exclude identical documents, such as those with marginal revisions.

                                ##### Similarity
                                This section allows for the comparison of similarity or difference between two documents.

                                ##### WordCloud
                                This section displays the words with the highest occurrences in the collection.

                                ##### Term Frequency
                                This section shows the frequency at which a word is used across the collection over time.

                                ##### Keywords TF-IDF
                                This section displays the keywords of the documents in the collection.

                                ##### Network: Member
                                This section shows the connections between members through jointly authored and negotiated documents.

                                ##### Network: Docs
                                This section shows the links between documents through cross-references.


                                ***contact: @wto.org***
                                '''
                                ),
                        ])
        ])
    elif pathname == "/page-2":
        if 'data' in globals():
            del data
        data = load_data()
        data = data.rename(columns={'text':'Normalized Text'})
        return html.Div([
                html.H3('Text Data', style={'font-weight': 'bold'}),
                html.P('Preprocessed: stopwords removed; words in original form; without numbers; predefined phrase linked by "_"'),
                dash_table.DataTable(
                    id='table',
                    # columns=[{"name": i, "id": i} for i in textdata.columns],
                    # data=textdata.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in data[['FileID','Text','Normalized Text']].columns],
                    data=data[['FileID','Text','Normalized Text']].to_dict('records'),

                    # columns=[{"name": i, "id": i} for i in data[['FileID','Text','text']].columns],
                    # data=data[['FileID','Text','text']].to_dict('records'),

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

    elif pathname in ["/page-3"]:
        return html.Div([
                        # Chart 1
                        dbc.Row([
                            dbc.Col([
                                html.H3('Summary Statistics of Documents', style={'font-weight': 'bold'}),
                                html.P(
                                    id="description",
                                    children=dcc.Markdown(
                                      children=(
                                        '''
                                        Member submisions, and Chair and Secretariat summaries/notes.
                                        ''')
                                    )
                                ),
                                html.Br(),
                                html.H6('Number of documents by year', style={'font-weight': 'bold'}),
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
                                html.Label('Select Author:'),
                                dcc.Dropdown(
                                    id='stat-year-dropdown-proponent',
                                    options=[{'label': v, 'value': k}
                                                for k, v in dict_proponent.items()],
                                    multi=False,
                                    value= 'All Members & Groups',
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
                                html.Label('Select Author:'),
                                dcc.Dropdown(
                                    id='stat-year-dropdown-proponent2',
                                    options=[{'label': v, 'value': k}
                                                for k, v in dict_proponent.items()],
                                    multi=False,
                                    value= 'All Members & Groups',
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


                        # Chart 3
                        dbc.Row([
                            dbc.Col([
                                html.H6('Number of documents by proponent', style={'font-weight': 'bold'}),
                                html.Label('Select Year:'),
                                dcc.Dropdown(
                                    id='stat-3-dropdown-year',
                                    options=[{'label': v, 'value': k}
                                                for k, v in dict_year.items()],
                                    multi=False,
                                    value= 'All',
                                ),
                            ], lg=4),
                            # dbc.Col([
                            #     html.Label('Select Author:'),
                            #     dcc.Dropdown(
                            #         id='stat-year-dropdown-proponent2',
                            #         options=[{'label': v, 'value': k}
                            #                     for k, v in dict_proponent.items()],
                            #         multi=False,
                            #         value= 'All Members & Groups',
                            #     ),
                            # ], lg=4)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(
                                    id='stat-3-proponent'
                                ),
                            ], lg=10),
                        ]),


                        # Chart 4
                        dbc.Row([
                            dbc.Col([
                                html.Br(),
                                html.Br(),
                                html.H6('Number of documents by topic', style={'font-weight': 'bold'}),
                                html.Label('Select Year:'),
                                dcc.Dropdown(
                                    id='stat-4-dropdown-year',
                                    options=[{'label': v, 'value': k}
                                                for k, v in dict_year.items()],
                                    multi=False,
                                    value= 'All',
                                ),
                            ], lg=4),
                            # dbc.Col([
                            #     html.Label('Select Author:'),
                            #     dcc.Dropdown(
                            #         id='stat-year-dropdown-proponent2',
                            #         options=[{'label': v, 'value': k}
                            #                     for k, v in dict_proponent.items()],
                            #         multi=False,
                            #         value= 'All Members & Groups',
                            #     ),
                            # ], lg=4)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(
                                    id='stat-4-topic'
                                ),
                            ], lg=10),
                        ]),





                    ])

    elif pathname in ["/page-4"]:
        # if 'data' in globals():
        #     del data
        # data = load_data()
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
                            #     html.Label('Select Author:'),
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
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(
                                    id='plot_year2'
                                ),
                            ], lg=10),
                        ]),
                    ])


    elif pathname in ["/page-6"]:
        return html.Div([
                        # html.H1('Title'),
                        html.H3('Networks: proposal proponents', style={'font-weight': 'bold'}),
                        # html.Embed(src = "assets/network_proponent.html", width=850, height=850),
                        html.Embed(src = "assets/Proponents Network - year(1997-2001) minEdge(1) KeepGroup(No).html", width=1000, height=700),
                        html.Embed(src = "assets/Proponents Network - year(1997-2001) minEdge(1) KeepGroup(Yes).html", width=1000, height=700),
                        html.Embed(src = "assets/Proponents Network - year(2002-2008) minEdge(1) KeepGroup(No).html", width=1000, height=700),
                        html.Embed(src = "assets/Proponents Network - year(2002-2008) minEdge(1) KeepGroup(Yes).html", width=1000, height=700),
                        html.Embed(src = "assets/Proponents Network - year(2009-2015) minEdge(1) KeepGroup(No).html", width=1000, height=700),
                        html.Embed(src = "assets/Proponents Network - year(2009-2015) minEdge(1) KeepGroup(Yes).html", width=1000, height=700),
                        html.Embed(src = "assets/Proponents Network - year(2016-2021) minEdge(1) KeepGroup(No).html", width=1000, height=700),
                        html.Embed(src = "assets/Proponents Network - year(2016-2021) minEdge(1) KeepGroup(Yes).html", width=1000, height=700),
                        ])
    elif pathname in ["/page-9"]:
        return html.Div([
                        # html.H1('Title'),
                        html.H3('Networks: document cross reference', style={'font-weight': 'bold'}),
                        # html.Embed(src = "assets/network_proponent.html", width=850, height=850),
                        html.Embed(src = "assets/cross_reference.html", width=850, height=850)
                        ])


    elif pathname in ["/page-7"]:
        return html.Div([
                        dbc.Row([
                            dbc.Col([
                                    html.H3('Term Frequency', style={'font-weight': 'bold'}),
                                    html.P(
                                        id="description",
                                        children=dcc.Markdown(
                                          children=(
                                            '''
                                            Term frequency across time
                                            ''')
                                        )
                                    ),

                            ]),

                            ]),
                        dbc.Row([
                                dbc.Col([
                                        dbc.Input(id='term-freq-input', value='tariff ams', type='text'),
                                        dbc.Button(id='term-freq-button', type='submit', children='Submit', className="mr-2"),
                                        html.P(id='term-freq-invalid'),
                                        ], lg=6),
                                ]),
                        dbc.Row([
                                dbc.Col([
                                        dcc.Graph(
                                            id='term-freq-plot'
                                            ),
                                        # dbc.Button(id='term-freq-button', type='submit', children='Submit', className="mr-2"),
                                        ], lg=10),
                                ])
                        ])
    elif pathname in ["/page-8"]:
        return html.Div([
                        dbc.Row([
                            dbc.Col([
                                    html.H3('TF-IDF keywords', style={'font-weight': 'bold'}),
                                    html.P(
                                        id="description2",
                                        children=dcc.Markdown(
                                          children=(
                                            '''
                                            Keywords based on TF-IDF. Select documents
                                            ''')
                                        )
                                    ),
                            ]),

                            ]),
                        dbc.Row([
                                dbc.Col([
                                        html.P(id='tfidf-invalid'),
                                        dcc.Dropdown(id='tfidf-dropdown',
                                                     multi=True,
                                                     value=['AIE-1', 'AIE-2','AIE-3','AIE-4','AIE-5'],
                                                     placeholder='Select members',
                                                     options=[{'label': country, 'value': country}
                                                              for country in allfileid]),
                                        ],lg=10),
                                ]),
                        dbc.Row([
                                dbc.Col([
                                        dcc.Graph(
                                            id='tfidf-plot'
                                            ),
                                        ], lg=10),
                                ])
                        ])


    # If the user tries to reach a different page, return a 404 message
    return dbc.Container(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

# Callbacks for interactive pages
# Stats 1 - Pillar
@app.callback(Output('stat-plot-year-pillar-proponent', 'figure'),
             [Input('stat-year-dropdown-pillar', 'value'),
              Input('stat-year-dropdown-proponent', 'value'),
             ])
def update_graph1(select_pillar, select_proponent):
    # if not select_pillar:
    #     raise PreventUpdate
    # if not select_proponent:
    #     raise PreventUpdate

    if select_pillar == 'All':
        select_pillar = list(files['Pillar'].unique())
    else:
        select_pillar = [select_pillar]

    if select_proponent == 'All Members & Groups':
        select_proponent = member
    else:
        select_proponent = [select_proponent]

    df_plot = files[(files['Pillar'].isin(select_pillar)) &
                    (files['Author'].isin(select_proponent))].groupby('Year')['FileID'].nunique().reset_index(name='Count')

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

# Stats 2 - Topic
@app.callback(Output('stat-plot-year-topic-proponent', 'figure'),
             [Input('stat-year-dropdown-topic', 'value'),
              Input('stat-year-dropdown-proponent2', 'value'),
             ])
def update_graph2(select_topic, select_proponent):
    # if not select_topic:
    #     raise PreventUpdate
    # if not select_proponent:
    #     raise PreventUpdate

    if select_topic == 'All':
        select_topic = list(files['Topic'].unique())
    else:
        select_topic = [select_topic]

    if select_proponent == 'All Members & Groups':
        select_proponent = member
    else:
        select_proponent = [select_proponent]

    df_plot = files[(files['Topic'].isin(select_topic)) &
                    (files['Author'].isin(select_proponent))].groupby('Year')['FileID'].nunique().reset_index(name='Count')

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
















# Stats 3 - Author
@app.callback(Output('stat-3-proponent', 'figure'),
             [Input('stat-3-dropdown-year', 'value'),
              # Input('stat-year-dropdown-proponent2', 'value'),
             ])
def update_graph3(select_year):
    # if not select_year:
    #     raise PreventUpdate
    # # if not select_proponent:
    # #     raise PreventUpdate

    if select_year == 'All':
        select_year = list(files['Year'].unique())
    else:
        # select_year = [int(select_year)]
        select_year = [select_year]



    # if select_proponent == 'All Members & Groups':
    #     select_proponent = member
    # else:
    #     select_proponent = [select_proponent]

    df_plot = files[(files['Year'].isin(select_year))
                    # &  (files['Author'].isin(select_proponent))
                    ].groupby('Author')['FileID'].nunique().reset_index(name='Count')

    df_plot = df_plot.sort_values('Count')

    figure = {
        'data': [go.Bar(
            x=df_plot['Author'].tolist(),
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
                tickmode='linear',
                tickfont = dict(size=8)
            )
        )
    }
    return figure



# Stats 3 - topic
@app.callback(Output('stat-4-topic', 'figure'),
             [Input('stat-4-dropdown-year', 'value'),
              # Input('stat-year-dropdown-proponent2', 'value'),
             ])
def update_graph4(select_year):
    # if not select_year:
    #     raise PreventUpdate
    # # if not select_proponent:
    # #     raise PreventUpdate

    if select_year == 'All':
        select_year = list(files['Year'].unique())
    else:
        # select_year = [int(select_year)]
        select_year = [select_year]



    # if select_proponent == 'All Members & Groups':
    #     select_proponent = member
    # else:
    #     select_proponent = [select_proponent]

    df_plot = files[(files['Year'].isin(select_year) & (files['Topic']!='X'))
                    # &  (files['Author'].isin(select_proponent))
                    ].groupby('Topic')['FileID'].nunique().reset_index(name='Count')

    df_plot = df_plot.sort_values('Count')

    figure = {
        'data': [go.Bar(
            x=df_plot['Topic'].tolist(),
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
                tickmode='linear',
                tickfont = dict(size=8)
            )
        )
    }
    return figure

























# Similarity
@app.callback(Output('plot_year1', 'figure'),
             [Input('plot-year-dropdown-pillar1', 'value'),
             ])
def update_graph_sim(select_pillar1):
    # if not select_pillar1:
    #     raise PreventUpdate
    # topic = 'TRQ'

    topic = select_pillar1

    if 'data' in globals():
        del data
    data = load_data()


    selected = data[data['Topics'].str.contains(topic)].sort_values('Year')
    selected_symbols = selected['Symbol'].tolist()
    selected_docs = selected['Words'].tolist()
    sim_sorted = calc_similarity(selected_symbols, selected_docs, kRandom=3, nClusters=3, sortCluster=False)

    figure = px.imshow(sim_sorted)
    figure.update_layout(
        height=800,
        width=800,
        font=dict(
            family="Courier New, monospace",
            size=8,
            color="RebeccaPurple"
                ),
        xaxis=dict(autorange='reversed')
        )
    return figure

# WordCloud
@app.callback(Output('plot_year2', 'figure'),
             [Input('plot-year-dropdown-pillar2', 'value'),
             ])
def update_graph_cloud(select_pillar2):
    # if not select_pillar2:
    #     raise PreventUpdate

    if 'data' in globals():
        del data
    data = load_data()

    topic = select_pillar2
    text = " ".join(review for review in data[data["Topics"].str.contains(topic)]['text'])
    # Generate a word cloud image
    wordcloud = WordCloud(background_color="white", width=900, height=600).generate(text)

    # Display the generated image:
    figure=px.imshow(wordcloud)
    figure.update_layout(
                    width=800,
                    height=650,
                    xaxis_visible=False,
                    yaxis_visible=False,
                    font=dict(
                              family="Courier New, monospace",
                              size=8,
                              color="RebeccaPurple"
                             ),
                    )
    return figure



# Term frequency
@app.callback(
        [dash.dependencies.Output('term-freq-plot', 'figure'),
         dash.dependencies.Output('term-freq-invalid', 'children')
         ],
        [dash.dependencies.Input('term-freq-button', 'n_clicks')],
        [dash.dependencies.State('term-freq-input', 'value')]
    )

def update_output(clicks, terms):
    terms = terms.strip().split(' ')
    invalid = set(terms) - set(all_terms)
    terms = list(set(terms) - invalid)

    # terms = ['competition','ams','food']
    freq = df_cv.loc[:,terms].div(df_cv["SUM"], axis=0)
    freq = freq.join(df_cv['Date'])
    freq= freq.groupby('Date')[terms].mean()

    fig = px.line(freq.ewm(span = 50).mean())
    
    
    # fig = go.Figure()

    # fig.add_trace(go.Scatter(x=x, y=y + 5, name="spline",
    #                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
    #                 hoverinfo='text+name',
    #                 line_shape='spline'))
    

    if len(invalid) >0:
        invalid = 'Invalid term(s): ' + ' '.join(list(invalid))
    else:
        invalid = ' '
    return fig, invalid


# itf-idf
# @app.callback(
#         [dash.dependencies.Output('tfidf-plot', 'figure'),
#          dash.dependencies.Output('tfidf-invalid', 'children')
#          ],
#         [dash.dependencies.Input('tfidf-button', 'n_clicks')],
#         [dash.dependencies.State('tfidf-dropdown', 'value')]
#     )

@app.callback(
        [Output('tfidf-plot', 'figure'),
         Output('tfidf-invalid', 'children')
         ],
        # [Input('tfidf-button', 'n_clicks')],
        [Input('tfidf-dropdown', 'value')]
    )

# def update_output(clicks, terms):
def update_output(terms):
    # print(terms)

    # terms = terms.strip().split(' ')
    # invalid = set(terms) - set(all_terms)
    # terms = list(set(terms) - invalid)

    # terms = ['competition','ams','food']

    # files = ['AIE-1', 'AIE-2','AIE-3','AIE-4','AIE-5','AIE-6','AIE-7','AIE-8','AIE-9',]
    # file = 'AIE-1'
    tfidf = load_tfidf()
    vis_tfidf = tfidf.set_index('word').groupby(['FileID']).tfidf.nlargest(10).reset_index()

    files = terms
    # print(files)
    rows = 2
    rows = len(terms)//5 + 1
    cols = 5

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=files)
    done = False
    for r in range(rows):
        for c in range(cols):
            # print(c+cols*r)
            if c+cols*r+1 > len(files):
                done = True
                break
            trace = vis_tfidf[vis_tfidf['FileID']==files[c+cols*r]][['word','tfidf']].set_index('word').sort_values('tfidf',ascending=True)
            fig.add_trace(go.Bar(y=trace.index, x=trace.tfidf, orientation='h',
                                 marker=dict(color='#7FC97F'
                                 # 'rgba(58, 71, 80, 0.6)'
                                 )
                                 ),
                          row=r+1, col=c+1,
                         )
        if done:
            break

    fig.update_layout(height=rows*300, width=1200, title_text="Side By Side Subplots", showlegend=False,
                      font=dict(family="Courier New, monospace",size=9,
                                # color="RebeccaPur
                               )
                     )
    invalid = ' '
    return fig, invalid













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

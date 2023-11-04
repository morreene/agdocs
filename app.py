from flask import Flask, session
from flask_session import Session
from dash import Dash, dcc, html
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
# to delete dash_table
from dash import html, dcc, dash_table
import urllib.parse
import pandas as pd




import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots




#################################################
#####     configurations
#################################################



#################################################
#####     Functions
#################################################



#################################################
#####     Load data 
#################################################

DASH_MASTER = 'data/docmaster.pickle'
DASH_DATA = 'data/data-preprocessed.pickle'
DASH_WORD_FREQ = 'data/word-frequency.pickle'
DASH_TFIDF = 'data/tfidf.pickle'

master = pd.read_pickle(DASH_MASTER)
master = master[['No', 'Symbol', 'Type', 'Year', 'Date', 'Title','Authors', 'Pillars', 'Topics', 'FileID']]

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

data = pd.read_pickle(DASH_DATA, compression='zip')


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



#################################################
##### Dash App
#################################################

# Hardcoded users (for demo purposes)
USERS = {"admin": "admin", "w": "w", "wto": "wto"}

server = Flask(__name__)
server.config['SECRET_KEY'] = 'supersecretkey'
server.config['SESSION_TYPE'] = 'filesystem'

Session(server)

# dash app
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']
app = Dash(__name__, server=server, 
        #    external_stylesheets=[dbc.themes.BOOTSTRAP], 
           external_stylesheets = external_stylesheets,
           suppress_callback_exceptions=True
           )

app.title = 'AgDocs Dataset'
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=UA-62289743-10"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
          gtag('config', 'UA-62289743-10');
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

### sidebar
sidebar_header = dbc.Row([
    html.A([dbc.Col(html.Img(src=app.get_asset_url("logo.png"),  width="180px", style={'margin-left':'15px', 'margin-bottom':'50px'}))], href="/page-1"),
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
])

sidebar = html.Div([
                    sidebar_header,
                    # use the Collapse component to animate hiding / revealing links
                    dbc.Collapse(
                        dbc.Nav([
                                dbc.NavLink("Home",                 href="/page-1", id="page-1-link"),
                                dbc.NavLink("Project Report",       href="/page-2", id="page-2-link"),
                                dbc.NavLink("Document Inventory",   href="/page-3", id="page-3-link"),
                                dbc.NavLink("Document Texts",       href="/page-4", id="page-4-link"),
                                dbc.NavLink("Summary Statistics",     href="/page-5", id="page-5-link"),
                                # dbc.NavLink("Stats: by topic",      href="/page-6", id="page-6-link"),
                                # dbc.NavLink("Stats: doc size",      href="/page-7", id="page-7-link"),
                                dbc.NavLink("Network - Authors",      href="/page-6", id="page-6-link"),
                                dbc.NavLink("Network - Documents",   href="/page-7", id="page-7-link"),
                            ], vertical=True, pills=False,
                        ), id="collapse",
                    ),
                    html.Div([html.P("V1.0 (20231104)",
                                # className="lead",
                            ),],id="blurb-bottom",
                    ),
                ], id="sidebar",
            )

content = html.Div(id="page-content")

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 8)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False, False, False, 
    return [pathname == f"/page-{i}" for i in range(1, 8)]

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Location(id='logout-url', refresh=False),  # Added logout URL component
    # login facet
    dbc.Container(
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Sign in to AgDocs", className="card-title"),
                            html.Br(),
                            dbc.Form(
                                [
                                    dbc.Row([
                                            dbc.Col([
                                                    dbc.Input(type="text", id="username", placeholder="Username", style={"width": 300}),
                                                ], className="mb-3",
                                            )
                                        ]
                                    ),
                                    dbc.Row([
                                            dbc.Col([
                                                    dbc.Input(type="password",  id="password", placeholder="Password",style={"width": 300}),
                                                ], className="mb-3",
                                            )
                                        ]
                                    ),
                                    dbc.Button(id='login-button', children='Sign in', n_clicks=0, color="primary", className="my-custom-button", style={"width": 300}),
                                ], 
                            ),
                        ], className="d-grid gap-2 col-8 mx-auto",
                    ),
                    className="text-center",
                    style={"width": "500px", "margin": "auto", "background-color": "#e4f5f2"},
                ), width=6, className="mt-5",
            )
        ), id='login-facet',className="login-page",
    ),

    html.Div([sidebar, content], id='page-layout', style={'display': 'none'}),
])

@app.callback(
    [Output('login-facet', 'style'),
    Output('page-layout', 'style')],
    [Input('login-button', 'n_clicks')],
    [State('username', 'value'), State('password', 'value')]
)
def update_output(n_clicks, username, password):
    if n_clicks > 0:
        if username in USERS and USERS[username] == password:
            session['authed'] = True
    if session.get('authed', False):
        return  {'display': 'none'}, {'display': 'block'}
    else:
        return {}, {'display': 'none'}

# render content according to path
@app.callback(Output("page-content", "children"),
              Output("logout-url", "pathname"),  # Added callback output for logout URL
              [Input("url", "pathname"), Input("logout-url", "pathname")])
def render_page_content(pathname, logout_pathname):
    if logout_pathname == "/logout":  # Handle logout
        session.pop('authed', None)
        return dcc.Location(pathname="/login", id="redirect-to-login"), "/logout"
    elif pathname in ["/","/login", "/page-1"]:
        return html.Div([
                dbc.Container([
                            html.H4("About AgDocs Dataset", className="display-about"),
                            html.P(
                                "Getting insights of 20+ year history from the docs",
                                className="lead",
                            ),
                            html.Hr(className="my-2"),
                            dcc.Markdown(
                                '''
                                    This site presents the results of a joint project between AGCD and ERSD-MAIS. The project aimed to:

                                    * Compile all documents related to agriculture negotiations in the WTO since 1997, when the analysis and exchange of information process began.
                                    * Explore a method and tool for quantitative analysis of these documents, organized by nature, topic, and content.

                                    ##### Document texts
                                    This section displays the full collection of documents and the extracted text features. Only the text
                                    is extracted from Word/PDF documents, while tables and formatting are ignored. 

                                    ##### Document Statistics
                                    This section includes a series of charts that provide summary statistics of the documents, including 
                                    counts of documents by year, member, and pillar (or category). It should be noted that the documents in the 
                                    collection are of different nature, although efforts have been made to exclude identical documents, such as those with marginal revisions.

                                    ##### Network: Member
                                    This section shows the connections between members through jointly authored and negotiated documents.

                                    ##### Network: Docs
                                    This section shows the links between documents through cross-references.
                                '''
                                ),
                        ])
        ]), pathname

    elif pathname == "/page-2":
        return html.Div([
                        html.H3('Report', style={'font-weight': 'bold'}),
                        html.Embed(src = "assets/html/INTSUBAG750ERSD3.htm", width=850, height=850),
                        # html.Embed(src = "assets/cross_reference.html", width=850, height=850)
                        ]), pathname

    elif pathname == "/page-3":
        return html.Div([
                html.H3('Inventory', style={'font-weight': 'bold'}),
                html.Br(),
                # html.P('Preprocessed: stopwords removed; words in original form; without numbers; predefined phrase linked by "_"'),
                dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in master.columns],
                    data=master.to_dict('records'),

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
                    page_size= 50,

                    # Freeze the first row
                    fixed_rows={'headers': True},

                    # style_cell_conditional=[
                    #     {'if': {'column_id': 'Title'},
                    #      'width': '180px'},
                    # ],
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto'
                    },
                    style_cell={
                        # 'height': 'auto',
                        'width': 'auto',
                        'minWidth': '50px', 
                        'maxWidth': '300px',
                        # 'whiteSpace': 'normal',
                        'textAlign': 'left',
                        'verticalAlign': 'top',
                        'fontSize':12,
                    },
                )
            ]), pathname
    elif pathname == "/page-4":
        # if 'data' in globals():
        #     del data
        # data = load_data()
        # data = data.rename(columns={'text':'Normalized Text'})
        return html.Div([
                html.H3('Document texts', style={'font-weight': 'bold'}),
                html.Br(),
                # html.P('Preprocessed: stopwords removed; words in original form; without numbers; predefined phrase linked by "_"'),
                dash_table.DataTable(
                    id='table',
                    # columns=[{"name": i, "id": i} for i in textdata.columns],
                    # data=textdata.to_dict('records'),
                    # columns=[{"name": i, "id": i} for i in data[['FileID','Text','Normalized Text']].columns],
                    # data=data[['FileID','Text','Normalized Text']].to_dict('records'),
                    columns=[{"name": i, "id": i} for i in data[['Symbol','Title','Text']].columns],
                    data=data[['Symbol','Title','Text']].to_dict('records'),

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

                    # # Freeze the first row
                    # fixed_rows={'headers': True},

                    style_cell_conditional=[
                        {'if': {'column_id': 'Symbol'},
                         'width': '80px'},
                    ],

                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto'
                    },
                    style_cell={
                        # 'height': 'auto',
                        # 'minWidth': '20px', 
                        # 'maxWidth': '300px',
                        # 'whiteSpace': 'normal',
                        'textAlign': 'left',
                        'verticalAlign': 'top',
                        'fontSize':12,
                    },
                )
            ]), pathname

    elif pathname in ["/page-5"]:
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

                        # # Chart 3
                        # dbc.Row([
                        #     dbc.Col([
                        #         html.H6('Number of documents by proponent', style={'font-weight': 'bold'}),
                        #         html.Label('Select Year:'),
                        #         dcc.Dropdown(
                        #             id='stat-3-dropdown-year',
                        #             options=[{'label': v, 'value': k}
                        #                         for k, v in dict_year.items()],
                        #             multi=False,
                        #             value= 'All',
                        #         ),
                        #     ], lg=4),
                        #     # dbc.Col([
                        #     #     html.Label('Select Author:'),
                        #     #     dcc.Dropdown(
                        #     #         id='stat-year-dropdown-proponent2',
                        #     #         options=[{'label': v, 'value': k}
                        #     #                     for k, v in dict_proponent.items()],
                        #     #         multi=False,
                        #     #         value= 'All Members & Groups',
                        #     #     ),
                        #     # ], lg=4)
                        # ]),
                        # dbc.Row([
                        #     dbc.Col([
                        #         dcc.Graph(
                        #             id='stat-3-proponent'
                        #         ),
                        #     ], lg=10),
                        # ]),


                        # # Chart 4
                        # dbc.Row([
                        #     dbc.Col([
                        #         html.Br(),
                        #         html.Br(),
                        #         html.H6('Number of documents by topic', style={'font-weight': 'bold'}),
                        #         html.Label('Select Year:'),
                        #         dcc.Dropdown(
                        #             id='stat-4-dropdown-year',
                        #             options=[{'label': v, 'value': k}
                        #                         for k, v in dict_year.items()],
                        #             multi=False,
                        #             value= 'All',
                        #         ),
                        #     ], lg=4),
                        # ]),
                        # dbc.Row([
                        #     dbc.Col([
                        #         dcc.Graph(
                        #             id='stat-4-topic'
                        #         ),
                        #     ], lg=10),
                        # ]),
                    ]), pathname


    # elif pathname in ["/page-6"]:
    #     return html.Div([
    #                         html.H4("Stats 2", className="display-about"),
    #                 ]), pathname


    # elif pathname in ["/page-7"]:
    #     return html.Div([
    #                         html.H4("Stats 3", className="display-about"),
    #                 ]), pathname

    elif pathname in ["/page-6"]:
        return html.Div([
                        html.H3('Network: Author of documents', style={'font-weight': 'bold'}),
                        html.Embed(src = "assets/html/Years 1997-2001 Keep Member groups True.html", width=1000, height=700),
                        html.Embed(src = "assets/html/Years 2002-2008 Keep Member groups True.html", width=1000, height=700),
                        html.Embed(src = "assets/html/Years 2009-2015 Keep Member groups True.html", width=1000, height=700),
                        html.Embed(src = "assets/html/Years 2016-2023 Keep Member groups True.html", width=1000, height=700),
                        ]), pathname

    elif pathname in ["/page-7"]:
        return html.Div([
                        html.H3('Network: document cross reference', style={'font-weight': 'bold'}),
                        html.Embed(src = "assets/html/networkx_graph.html", width=850, height=850),
                        # html.Embed(src = "assets/cross_reference.html", width=850, height=850)
                        ]), pathname

    else:
        # If the user tries to reach a different page, return a 404 message
        return dbc.Container(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        ), pathname


#################################################
#####    Page Search
#################################################


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




























































































































































































# call back for returning results
@app.callback(
        [Output("search-results", "children"),  
        #  Output("top-space", "style"),
         Output("sample-queries", "style")
         ],
        [Input("search-button", "n_clicks"),
         Input('search-box', 'n_submit'), ], 
        [State("search-box", "value"),
        State('radio-select-top', 'value')]
        )
def search(n_clicks, n_submit, search_terms, top):
    # Check if the search button was clicked
    if (n_clicks <=0 and n_submit is None) or search_terms=='' or search_terms is None:
        return "",  None
    else:
        df = search_docs(search_terms, top = top)
        
        csv_string = df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)

        df['meta'] = df['member'] + '\n' + df['symbol'] + '\n' + df['date'] + '\n Score: ' + df['score'].astype(str) 
        df['text'] = df['text'] + '\n\n [Topic]: ' + df['topic']

        matches = df[['meta', 'text']]
        matches.columns = ['Meta','Text (Paragraph)']

        # Display the results in a datatable
        return html.Div(style={'width': '100%'},
                        children=[
                            # html.P('Find ' + str(len(matches)) +" paragraphs, with score ranging from " + str(df['score'].min()) + ' to ' + str(df['score'].max())),
                            # html.A('Download CSV', id='download-link', download="rawdata.csv", href=csv_string, target="_blank",),
                            html.Br(),
                            dbc.Row(
                                [
                                    dbc.Col(html.P('Find ' + str(len(matches)) +" paragraphs, with scores from " + str(df['score'].min()) + ' to ' + str(df['score'].max())), width={"size": 9, "offset": 0}),
                                    dbc.Col(html.A('Download CSV', id='download-link', download="rawdata.csv", href=csv_string, target="_blank"), width={"size": 3, "offset": 0}),
                                ],
                                justify="between",
                                style={"margin-bottom": "20px"},
                            ),

                            html.Br(),
                            dash_table.DataTable(
                                    id="search-results-table",
                                    columns=[{"name": col, "id": col} for col in matches.columns],
                                    data=matches.to_dict("records"),

                                    editable=False,
                                    # filter_action="native",

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
                                    style_table={'width': '900px'},
                                    style_header={'fontWeight': 'bold'},
                                    style_cell={
                                        # 'height': 'auto',
                                        # 'minWidth': '50px', 
                                        # 'maxWidth': '800px',
                                        # # 'width': '100px',
                                        # 'whiteSpace': 'normal',
                                        'textAlign': 'left',
                                        'fontSize': '14px',
                                        'verticalAlign': 'top',
                                        'whiteSpace': 'pre-line'
                                    },
                                    style_cell_conditional=[
                                        # {'if': {'column_id': 'Symbol'},
                                        #  'width': '50px'},
                                        # {'if': {'column_id': 'Member'},
                                        #  'width': '90px'},
                                        # {'if': {'column_id': 'Date'},
                                        #  'width': '80px'},
                                        # {'if': {'column_id': 'Section/Topic'},
                                        #  'width': '200px'},
                                        {'if': {'column_id': 'Text (Paragraph)'},
                                        'width': '1000px'},
                                        # {'if': {'column_id': 'Score'},
                                        #  'width': '80px', 'textAlign': 'right'},
                                    ],
                                    style_data_conditional=[
                                        {
                                            'if': {'row_index': 'odd'},
                                            'backgroundColor': 'rgb(250, 250, 250)',
                                        }
                                    ],
                                    style_as_list_view=True,
                                )
                            ]
                ),  {'display': 'none'}

#################################################
#####     Page Chat
#################################################

# call back for returning results
@app.callback(
        [Output("search-results2", "children"),  
         Output("sample-queries2", "style")
        ],
        [Input("search-button2", "n_clicks"),
         Input("search-box2", "n_submit")
        ], 
        [State("search-box2", "value"),
         State('radio-select-top2', 'value')
        ]
        )
def chat(n_clicks, n_submit, query, model):
    # Check if the search button was clicked
    # if (n_clicks <=0 and n_submit is None) or search_terms=='' or search_terms is None:
    if (n_clicks <=0 and n_submit is None) or query=='' or query is None:
        return "",  None
    else:
        # ChatGPT only
        prompt = f"""
                    Answer the following question.
                    If you don't know the answer, just say that you don't know. 
                    ---
                    QUESTION: {query}   
                """
        chatgpt = get_completion(prompt, model)        

        # chatgpt = complete(search_terms, model)
        # # print(chatgpt)

        # ChatGPT plus TPR
        prompt = retrieve(query)
        chatgpttpr = get_completion(prompt, model)
        # query_with_contexts = retrieve(search_terms)
        # chatgpttpr = complete(query_with_contexts, model)
    return html.Div(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(html.H5("Answer by ChatGPT"), width={"size": 6, "offset": 0}),
                    dbc.Col(html.H5("Answer by ChatGPT based on TPR reports"), width={"size": 6, "offset": 0}),
                ],
                justify="between",
                style={"margin-bottom": "20px"},
            ),
            dbc.Row(
                [
                    dbc.Col(html.P(dcc.Markdown(chatgpt)), width={"size": 6, "offset": 0}),
                    dbc.Col(html.P(dcc.Markdown(chatgpttpr)), width={"size": 6, "offset": 0}),
                ],
                justify="between",
            ),
        ],
    )
),  {'display': 'none'}
 


#################################################
#####     Page tags
#################################################

@app.callback(
    # Output('table', 'data'),
    Output("search-results3", "children"),
    [Input({'type': 'tag', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State({'type': 'tag', 'index': dash.dependencies.ALL}, 'children')]
)
def update_table(*args):
    ctx = dash.callback_context

    if not ctx.triggered:
        return None # df.to_dict('records')

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    tag_clicked = ctx.states[button_id + '.children']
    # print(tag_clicked)

    df = search_docs(tags[tag_clicked], top = 200)
    
    csv_string = df.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)

    df['meta'] = df['member'] + '\n' + df['symbol'] + '\n' + df['date'] + '\n Score: ' + df['score'].astype(str) 
    df['text'] = df['text'] + '\n\n [Topic]: ' + df['topic']

    matches = df[['meta', 'text']]
    matches.columns = ['Meta','Text (Paragraph)']

    # Display the results in a datatable
    return html.Div(style={'width': '100%'},
                     children=[
                        html.Br(),
                        dbc.Row(
                            [
                                dbc.Col(html.P('Find ' + str(len(matches)) +" paragraphs, with scores from " + str(df['score'].min()) + ' to ' + str(df['score'].max())), width={"size": 9, "offset": 0}),
                                dbc.Col(html.A('Download CSV', id='download-link', download="rawdata.csv", href=csv_string, target="_blank"), width={"size": 3, "offset": 0}),
                            ],
                            justify="between",
                            style={"margin-bottom": "20px"},
                        ),
                        html.Br(),
                        dash_table.DataTable(
                                id="search-results-table",
                                columns=[{"name": col, "id": col} for col in matches.columns],
                                data=matches.to_dict("records"),

                                editable=False,
                                # filter_action="native",

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

                                style_table={'width': '100%'},
                                style_header={'fontWeight': 'bold'},
                                style_cell={
                                    # 'height': 'auto',
                                    # 'minWidth': '50px', 
                                    # 'maxWidth': '800px',
                                    # # 'width': '100px',
                                    # 'whiteSpace': 'normal',
                                    'textAlign': 'left',
                                    'fontSize': '14px',
                                    'verticalAlign': 'top',
                                    'whiteSpace': 'pre-line'
                                    # 'whiteSpace': 'nowrap',
                                    
                                },
                                # style_cell_conditional=[
                                #     # {'if': {'column_id': 'Symbol'},
                                #     #  'width': '50px'},
                                #     # {'if': {'column_id': 'Member'},
                                #     #  'width': '90px'},
                                #     # {'if': {'column_id': 'Date'},
                                #     #  'width': '80px'},
                                #     # {'if': {'column_id': 'Section/Topic'},
                                #     #  'width': '200px'},
                                #     {'if': {'column_id': 'Text (Paragraph)'},
                                #     'width': '1000px'},
                                #     # {'if': {'column_id': 'Score'},
                                #     #  'width': '80px', 'textAlign': 'right'},
                                # ],
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': 'rgb(250, 250, 250)',
                                    }
                                ],
                                style_as_list_view=True,
                            )
                        ]
            )


#################################################
# end of function page
#################################################














#################################################
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
    app.run_server(port=8888, debug=True)
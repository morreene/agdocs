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
import dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_auth

import pandas as pd
import time
import datetime
import os
import urllib.parse

# import urllib
# import sqlite3
# from machine_learning import topic_extraction,create_dict_list_of_topics

import plotly.graph_objs as go
import networkx as nx

# global final_reddit_topic_df
# global top_post_df
global dict_topics


# Read data
data = pd.read_pickle('data/data_keyterms_topics_senti.pickle')
topics = pd.read_pickle('data/topics.pickle')
# network = pd.read_pickle('data/data_network.pickle')
network = pd.read_pickle('data/data_network1.pickle')

# Variables
data['Key Terms'] = data['KTScakeStr']
dict_topics = dict(zip(topics['Topic'], topics['TopicName']))
weeks = data['WeekStr'].unique().tolist()
weeks.sort(reverse=True)

# Dash app
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']
# with "__name__" local css under assets is also included
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

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
        dbc.Col(html.Img(src=app.get_asset_url("logo.png"), width="180px", style={'margin-left':'15px'})),
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
                    dbc.NavLink("About", href="/page-1", id="page-1-link"),
                    dbc.NavLink("Summarization", href="/page-2", id="page-2-link"),
                    dbc.NavLink("Topics", href="/page-3", id="page-3-link"),
                    dbc.NavLink("Sentiment", href="/page-4", id="page-4-link"),
                    dbc.NavLink("Network", href="/page-5", id="page-5-link"),

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
                        "V.20210423 based on Plotly Dash",
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
                dbc.Jumbotron([
                            html.H4("About the Data and the App", className="display-about"),
                            html.P(
                                "Tracking what happening in trade with NLP algorithms",
                                className="lead",
                            ),
                            html.Hr(className="my-2"),
                            dcc.Markdown(
                                '''
                                The news tracker analyzes over two thousand news pieces, which were curated by professional editors, with natural language processing (NLP) and machine learning algorithms and tools:

                                * **Topic model:** topics identified by unsupervised machine learning models ([Latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) and [Non\-negative matrix factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)).
                                   The number of news reports under those topics could be tracked over time;
                                * **Sentiment analysis:** how sentiment changes over time;
                                * **Name network:** names are linked because they are mentioned together in one news report;
                                * **Key terms: ** automatically extract key phrases to [summarize](https://en.wikipedia.org/wiki/Automatic_summarization) news reports.
                                '''
                                ),
                        ])
        ])

    elif pathname == "/page-2":
        return html.Div([
                            html.Div(id='test14'),
                            html.H4('News summarized by key terms', style={'font-weight': 'bold'}),
                            dcc.Markdown('''"Key Terms" are automatically extracted from news text by a NLP algorithm -[sCAKE](https://arxiv.org/abs/1811.10831v1). '''),
                                            # [SGRank](https://www.aclweb.org/anthology/S15-1013/) or
                                            # [TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
                            dcc.Dropdown(
                                id='dropdown-week-summarization',
                                options= [{'label': i, 'value': i} for i in weeks],
                                multi=True,
                                clearable=False,
                                value= [max(weeks)],
                            ),
                            dcc.Dropdown(
                                id='dropdown-topic-summarization',
                                options=[{'label': v, 'value': k}
                                            for k, v in dict_topics.items()],
                                multi=True,
                                value= [0,1,2,3,4,5,6,7,8,9],
                            ),
                            html.Div(id='summarization-table-container', style= {'margin': '15px'}),
        ])

    elif pathname == "/page-3":
        return html.Div([
                    dbc.Row([
                                html.H4('Select a topic or topics, which are defined by a set of keywords:', style={'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id='topic-topic-dropdown',
                                    options=[{'label': v, 'value': k}
                                                for k, v in dict_topics.items()],
                                    multi=True,
                                    value= [0,1,2,3,4,5,6,7,8,9],
                                ),
                        ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(
                                id='top_topics'
                            ),
                        ], lg=3),
                        dbc.Col([
                            dcc.Graph(
                                id='top_topics_timeline'
                            ),
                        ], lg=9),
                    ]),
            ])
    elif pathname == "/page-4":
        return html.Div([
                            dbc.Col([
                                html.H4('Sentiment Timeline: sentiment score - positive(+1), neutral(0), or negative(-1)', style={'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id='sentiment-topic-dropdown',
                                    options=[{'label': v, 'value': k}
                                                for k, v in dict_topics.items()],
                                    multi=True,
                                    value= [0,1,2,3,4,5,6,7,8,9],
                                ),
                                dcc.Graph(
                                    id='sentiment'
                                ),
                            ], lg=12),
            ])
    elif pathname == "/page-5":
        return html.Div([
                        dbc.Col([
                            html.H4('Network: how names are menteioned together in one piece of news', style={'font-weight': 'bold'}),
                            dcc.Dropdown(
                                id='dropdown-week-names',
                                options= [{'label': i, 'value': i} for i in weeks],
                                multi=True,
                                clearable=False,
                                value= [max(weeks)],
                            ),

                            dcc.Dropdown(
                                id='dropdown-topic-names',
                                options=[{'label': v, 'value': k}
                                            for k, v in dict_topics.items()],
                                multi=True,
                                value= [0,1,2,3,4,5,6,7,8,9],
                            ),
                            dcc.Graph(
                                id='names'
                            ),
                        ], lg=12),
            ])

    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

# Summarization
@app.callback(Output('summarization-table-container', 'children'),
                [Input('dropdown-topic-summarization', 'value'), Input('dropdown-week-summarization', 'value')])
def generate_table(selected_dropdown_value, dropdown_week, max_rows=10):
    if not selected_dropdown_value:
        raise PreventUpdate
    dff = data[(data['Topic'].isin(selected_dropdown_value)) & (data['WeekStr'].isin(dropdown_week))][['Date', 'Title', 'Agency', 'Topic','Key Terms']]
    dff['Topic'] = dff['Topic'].astype(str)

    return html.Div([
            dash_table.DataTable(
                    id='tab',
                    columns=[
                        {"name": i, "id": i, "deletable": False, "selectable": False} for i in dff.columns
                    ],
                    data = dff.to_dict('records'),
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
                    style_cell={
                        # 'height': 'auto',
                        'minWidth': '20px', 'maxWidth': '500px',
                        # 'whiteSpace': 'normal',
                        'textAlign': 'left',
                        'fontSize':12,
                    },
                    style_cell_conditional=[
                        {'if': {'column_id': 'Title'},
                         'width': '400px'},
                        {'if': {'column_id': 'Agency'},
                         'width': '100px'},
                        {'if': {'column_id': 'Date'},
                         'width': '90px'},
                        {'if': {'column_id': 'Topic'},
                         'width': '60px'},
                        {'if': {'column_id': 'Key Terms'},
                         'font-weight':'bold'},
                    ],
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto'
                    },
                    # style_cell_conditional=[
                    #     {
                    #         'if': {'column_id': c},
                    #         'textAlign': 'left'
                    #     } for c in ['Date', 'Region']
                    # ],
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                )
            ])

# Topics Bar chart
@app.callback(Output('top_topics', 'figure'), [Input('topic-topic-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    if not selected_dropdown_value:
        raise PreventUpdate

    df = data[data['Topic'].isin(selected_dropdown_value)]['Topic']
    df_plot = df.value_counts().reset_index()
    df_plot['label'] = df_plot['index'].map(dict_topics)
    df_plot['label'] = 'Topic ' + df_plot['index'].astype(str)

    figure = {
        'data': [go.Bar(
            x=df_plot['Topic'].tolist(),
            y=df_plot['label'].tolist(),
            orientation='h'
        )],
        'layout':go.Layout(
            title= 'Number of news items by topics',
            yaxis = dict(
                # autorange=True,
                automargin=True
            )
        )
    }
    return figure

# Topics - Scatter
@app.callback(Output('top_topics_timeline', 'figure'), [Input('topic-topic-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    if not selected_dropdown_value:
        raise PreventUpdate

    df = data[data['Topic'].isin(selected_dropdown_value)]
    df_plot = df.groupby(['Week','Topic']).size().reset_index(name='Count')
    trace_list = []
    for value in selected_dropdown_value:
        trace = go.Scatter(
            y=df_plot[df_plot['Topic']==value]['Count'].tolist(),
            x=df_plot[df_plot['Topic']==value]['Week'].tolist(),
            name = 'Topic' + str(value),
            stackgroup='one'
        )
        trace_list.append(trace)

    layout = dict(title='Timeline of topics - weekly total',
                  xaxis=dict(title='Week'),
                  yaxis=dict(title='Number of news items'),
                  )
    figure = dict(data=trace_list,layout=layout)
    return figure


# Sentiment
@app.callback(Output('sentiment', 'figure'), [Input('sentiment-topic-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    if not selected_dropdown_value:
        raise PreventUpdate

    df = data[data['Topic'].isin(selected_dropdown_value)].copy()
    df = df[df['Day_of_Week']<5][['Date','compound','Title']]
    # df['mean'] = df['compound'].expanding().mean()
    avg = df.groupby('Date')['compound'].mean().reset_index(name='avg')

    trace1 = go.Scatter(
        y=df['compound'].tolist(),
        x=df['Date'].tolist(),
        mode='markers',
        name='sentiment score',
        hovertext=df['Title'].tolist(),
        hoverinfo="text",
        marker_color='rgba(31,119,180, 0.4)'
    )
    trace2 = go.Scatter(
        y=avg['avg'].tolist(),
        x=avg['Date'].tolist(),
        name='average',
        line = dict(color='rgba(214,39,40,0.9)', width=4),
        hoverinfo='skip'
    )

    layout = dict(
                  xaxis=dict(title=''),
                  yaxis=dict(title='Sentiment score'),
                  margin=dict(t = 30),
                  # width: 1000,
                  height=600,
                  )
    figure = dict(data=[trace2,trace1],layout=layout)
    return figure



# Name Network

@app.callback(Output('names', 'figure'),
                [Input('dropdown-topic-names', 'value'), Input('dropdown-week-names', 'value')])
def update_graph(selected_topic, selected_week):
    if not selected_topic:
        raise PreventUpdate
    dff = network[network['WeekStr'].isin(selected_week)]
    # dff['Topic'] = dff['Topic'].astype(str)
    dff = dff.groupby(['Name0', 'Name1']).size().reset_index(name='Count')
    dff = dff[dff['Count']>3]


# @app.callback(Output('names', 'figure'),
#                 [Input('network-topic-dropdown', 'value')]
#                 )
# def update_graph(selected_dropdown_value):
    # if not selected_dropdown_value:
    #     raise PreventUpdate
    A = list(dff["Name1"].unique())
    B = list(dff["Name0"].unique())
    print(A)
    node_list = list(set(A+B))

    G = nx.Graph()
    for i in node_list:
        G.add_node(i)

    for i,j in dff.iterrows():
        G.add_edges_from([(j["Name0"],j["Name1"])])

    pos = nx.spring_layout(G, k=0.5, iterations=50)
    for n, p in pos.items():
        G.nodes[n]['pos'] = p

    # Standard plotly networkx code
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[1]]['pos']
        x1, y1 = G.nodes[edge[0]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
    annotations = list(G.nodes)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text = annotations,
        textposition="top right",
        textfont=dict(
            size=8,
            color="darkcyan"
        ),

        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' | YlGnBu
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' | Darkmint
            colorscale='Darkmint',
            reversescale=False,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # hover text
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    # node_trace.text = node_text

    figure = dict(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    # title='<br>Network graph made with Python',
                    # width=500,
                    height=800,
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    # annotations=[ dict(
                    #     # text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    #     showarrow=False,
                    #     xref="paper", yref="paper",
                    #     x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return figure













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

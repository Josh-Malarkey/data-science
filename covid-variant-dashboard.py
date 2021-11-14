"""
Author: Josh Stine
Date Created: 10/28/2021

resources used:
https://github.com/Coding-with-Adam/Dash-by-Plotly/tree/master/Dash%20Components/Dropdown
https://www.youtube.com/watch?v=UYH_dNSX1DM
"""

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas
import plotly.graph_objects as go

# import and transform the data for the way we need it in order to create the dashboard
variants = ['Beta','Alpha','Gamma','Delta','Kappa','Epsilon','Eta','Iota','Lambda','others']
colors = ["001219","005f73","0a9396","94d2bd","e9d8a6","ee9b00","ca6702","bb3e03","ae2012","E2797D"]
covid_variant_data = pandas.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/variants/covid-variants.csv').dropna()
covid_variant_data = covid_variant_data[covid_variant_data['variant'].isin(variants)]
countries = covid_variant_data['location'].unique()
country_dict = []
for country in countries:
    country_dict.append({'label': country, 'value': country})

# create the dashboard as an app and initialize server
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    # div for dropdown
    html.Div([
        html.Br(),
        html.Div(id='country_data'),
        html.Br(),
        html.Label(['Choose Country:'], style={'font-weight': 'bold','text-align': 'center'}),
        dcc.Dropdown(id='country_dropdown',
                     options=country_dict,
                     optionHeight=35,
                     value='United States',
                     disabled=False,
                     multi=False,
                     searchable=True,
                     search_value='United States',
                     placeholder='Please select a Country',
                     clearable=True,
                     style={'width':'50%', 'align':'right', 'padding-left':'5px'})
    ]),
    # div for graph
    html.Div([
        dcc.Graph(id='covid_variant_graph')
    ]),
    # div for data source
    html.Div([
        html.Span(['Data Source: '], style={'font-size': '16px', 'font-weight':'bold'}),
        html.A('https://github.com/owid/covid-19-data/blob/master/public/data/variants/covid-variants.csv',
               href='https://github.com/owid/covid-19-data/blob/master/public/data/variants/covid-variants.csv'
               , target='_blank'),
        html.Br(),
        html.P('*** This chart is sourced directly from the repository so it will update when new data is uploaded ***'
               , style={'font-size':'12px', 'font-weight':'bold'})
    ])
])

# connect the dropdown to the graph
@app.callback(
    Output(component_id='covid_variant_graph', component_property='figure'),
    [Input(component_id='country_dropdown', component_property='value')]
)

def build_dash(country):
    # select subset of data based on country and add trace lines for each variant
    country_variant_data = covid_variant_data[covid_variant_data['location']==country]
    trace_lines = []
    for index, variant in enumerate(variants):
        # create a subset of data by variant
        variant_data = country_variant_data[country_variant_data['variant']==variant]
        # sum the data by date in case there are any duplicates
        variant_data_sum = variant_data.groupby(country_variant_data['date']).sum()
        # add date back in as a column since it moves to index with .groupby()
        variant_data_sum.reset_index(level=0, inplace=True)
        # add back in the country and variant columns since .sum() drops all non numerics
        variant_data_sum['location'] = country
        variant_data_sum['variant'] = variant
        trace = go.Scatter(
                x=variant_data_sum['date'],
                y=variant_data_sum['num_sequences'],
                name=variant,
                fill='tonexty',
                stackgroup=index,
                line=dict(color="#"+colors[index])
        )
        trace_lines.append(trace)
    layout = dict(
            title=country + ' COVID Variant Trends',
            annotations=[
                go.layout.Annotation(
                    text='Double-click on variant to isolate<br>Single-click to remove from plot',
                    align='left',
                    bordercolor='black',
                    borderwidth=1.5,
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=1,
                    y=1.15
                ),
                go.layout.Annotation(
                    text='Slide dates on x-axis to update chart zoom',
                    align='left',
                    bordercolor='black',
                    borderwidth=1.5,
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=1,
                    y=-0.25
                )
            ]
    )
    fig = go.Figure(data=trace_lines, layout=layout)
    fig.update_layout(title_font_size=24, title_x=0.5)
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Number of Sequences')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

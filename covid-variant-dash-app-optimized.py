"""
Author: Josh Malarkey (Stine)
Date Created: 10/28/2021
Date Updated Last: 1/29/2022
"""

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from datetime import datetime
import flask
import logging
import pandas
import plotly.express as px
import plotly.graph_objects as go
import requests


"""Set up log file"""
logger = logging.getLogger(__name__)
log_file_name = 'optimized-covid-variant-dash-app_logfile.txt'
logging.basicConfig(filename=log_file_name, level=logging.DEBUG)


"""Extract, Transform, and Load Data into the application"""
# https://coolors.co/001219-005f73-0a9396-94d2bd-e9d8a6-ee9b00-ca6702-bb3e03-ae2012-e2797d
colors = ["5e548e", "001219", "005f73", "0a9396", "94d2bd", "e9d8a6", "ee9b00", "ca6702", "bb3e03", "ae2012", "E2797D",
          "e26d5c", "4f5d75"]

# get case data
logging.info('\n\nBEGIN extracting, transforming, loading data - ' + datetime.now().strftime('%m/%d/%Y, %H:%M:%S'))
covid_case_data = pandas.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/jhu/full_data.csv')
countries = covid_case_data['location'].unique()
covid_case_data = covid_case_data[covid_case_data['location'].isin(countries)]

# get vaccination data
vaccination_data = pandas.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv')
vaccination_data = vaccination_data[vaccination_data['location'].isin(countries)]

# dictionary for country dropdown list
country_dict = []
for country in countries:
    country_dict.append({'label': country, 'value': country})

# get variant data for variants with greek names
try:
    response = requests.get('http://raw.githubusercontent.com/owid/covid-19-data/master/public/data/internal/megafile--variants.json')
    code = response.status_code
    if response.status_code == 200:
        covid_variant_data = pandas.read_json('http://raw.githubusercontent.com/owid/covid-19-data/master/public/data/internal/megafile--variants.json')
    else:
        covid_variant_data = pandas.read_json('http://raw.githubusercontent.com/Josh-Stine/heroku-apps/main/variants.json')
except requests.ConnectionError as exception:
    covid_variant_data = pandas.read_json('http://raw.githubusercontent.com/Josh-Stine/heroku-apps/master/variants.json')
if 'non_who' in covid_variant_data.columns:
    covid_variant_data.rename(columns={'non_who':'(Non-WHO) + org strain'}, inplace=True)
variants = covid_variant_data.columns[2:]
variants_of_interest = []
for variant in variants:
    if '_' not in variant and 'others' not in variant:
        variants_of_interest.append(variant)
variants_of_interest.sort()


# create dictionary to map color for each variant
color_variant_map = dict()
color_index = 0
for variant in variants_of_interest:
    # handle case for more variants than colors
    if color_index >= len(colors) - 1:
        color_index = 0
    color_variant_map[variant] = '#' + colors[color_index]
    color_index += 1

# sort/load variant data for world map
map_variant_data = pandas.read_csv('https://raw.githubusercontent.com/Josh-Stine/heroku-apps/main/dominant_variants_timeseries.csv')
map_variant_data.sort_values(by=['date','location','variant'],inplace=True)
logging.info('END extracting, transforming, loading data - ' + datetime.now().strftime('%m/%d/%Y, %H:%M:%S'))


"""HTML Layout of the app"""
logging.info('\nBEGIN building HTML layout of the app - ' + datetime.now().strftime('%m/%d/%Y, %H:%M:%S'))
# create the dashboard as an app and initialize server
server = flask.Flask(__name__)  # define flask app.server
app = dash.Dash(__name__, server=server)

app.layout = html.Div([
    html.Div([
        # div for title
        html.Div(
            html.H1(['COVID-19 Dashboard: Country and Global Data']),
            style={'display': 'inline-block', 'width': '100%',
                   'text-align': 'center', 'font-family': 'Verdana, Arial, Sans-Serif'}
        )
    ]),
    # div for graphs
    html.Div([
        # global variant map and chart
        html.Div([
            html.Div(
                children=dcc.Graph(id='global_covid_variant_graph'),
                style={'display':'inline-block', 'width':'50%'}
            ),
            html.Div(
                children=dcc.Graph(id='covid_variant_map'),
                style={'display': 'inline-block','width':'50%'}
            )
        ]),
        html.Div([
            # div for dropdown
            html.Div([
                html.Div([
                    html.Label(['Choose Country:'], style={'font-weight': 'bold', 'text-align': 'center',
                                                           'padding-left': '5em', 'margin-top':'10px'}),
                    html.Div(children=dcc.Dropdown(id='country_dropdown',
                                 options=country_dict,
                                 optionHeight=35,
                                 value='United States',
                                 disabled=False,
                                 multi=False,
                                 searchable=True,
                                 search_value='United States',
                                 placeholder='Please select a Country',
                                 clearable=True,
                                 style={'padding-left': '5em', 'margin-bottom':'0px'})
                        )
                ], style={'display':'inline-block', 'width':'30%'}),
                # div for disclaimer
                html.Div([
                    html.Div([
                        html.Span(
                            [
                                '*** These charts are sourced directly from the GitHub repo linked at the bottom; they automatically '
                                ' refresh when new data is uploaded ***'], style={'font-size': '15px','font-style':'italic'})

                    ], style={'float': 'right','align-items':'center','justify-content':'center','padding-right':'5em',
                              'margin-top':'10px'}),
                ], style={'display': 'inline-block', 'width': '70%'}
                )
            ]),
            # div for country variant graph
            html.Div([
                html.Div(style={'width':'10%','display':'inline-block'}),
                html.Div(
                    children=dcc.Graph(id='country_covid_variant_graph'),
                    style={'width':'80%', 'display':'inline-block'}
                ),
                html.Div(style={'width':'10%','display':'inline-block'})
            ])
        ]),
        # case charts
        html.Div(
            children=dcc.Graph(id='country_covid_case_graph'),
            className='case_graph',
            style={'width':'50%', 'display':'inline-block'}
        ),
        html.Div(
            children=dcc.Graph(id='global_covid_case_graph'),
            className='case_graph',
            style={'width':'50%', 'display':'inline-block'}
        ),
        # vaccination charts
        html.Div(
            children=dcc.Graph(id='country_covid_vaccination_graph'),
            className='vaccination_graph',
            style={'width':'50%', 'display':'inline-block'}
        ),
        html.Div(
            children=dcc.Graph(id='global_covid_vaccination_graph'),
            className='vaccination_graph',
            style={'width':'50%', 'display':'inline-block'}
        )
    ]),
    html.Div([
        # div for data source link and author credit
        html.Div([
            html.Div([
                html.Span(['Author: Josh Malarkey'], style={'font-size': '14px', 'font-weight': 'bold'}),
                html.Br(),
                html.Span(['See more at my GitHub portfolio: '], style={'font-size': '14px', 'font-style': 'italic'}),
                html.A('https://josh-malarkey.github.io/', href='https://josh-malarkey.github.io/', target='_blank'),
                html.Br(),
                html.Br(),
                html.Span(['Data Source: '], style={'font-size': '14px', 'font-weight': 'bold'}),
                html.A('https://github.com/owid/covid-19-data/blob/master/public/data/',
                href='https://github.com/owid/covid-19-data/blob/master/public/data/', target='_blank')
            ], style={'float':'center'}),
        ], style={'display': 'inline-block', 'width':'100%'})
    ])
])
logging.info('END building HTML layout of the app - ' + datetime.now().strftime('%m/%d/%Y, %H:%M:%S'))


"""Interactive functionality of the app"""
logging.info('\nBEGIN building interactive functionality of the app - ' + datetime.now().strftime('%m/%d/%Y, %H:%M:%S'))
# create the range slider that allows each chart to be quick-filtered based on the intervals below
range_slider = dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )


# build the charts
@app.callback([
    Output(component_id='covid_variant_map', component_property='figure'),
    Output(component_id='country_covid_variant_graph', component_property='figure'),
    Output(component_id='global_covid_variant_graph', component_property='figure'),
    Output(component_id='country_covid_case_graph', component_property='figure'),
    Output(component_id='global_covid_case_graph', component_property='figure'),
    Output(component_id='country_covid_vaccination_graph', component_property='figure'),
    Output(component_id='global_covid_vaccination_graph', component_property='figure')
    ], [Input(component_id='country_dropdown', component_property='value')]
)
def build_charts(country):

    # BUILD THE VARIANT MAP
    variant_map_fig = px.choropleth(data_frame=map_variant_data, locationmode='country names', locations='location',
                        color='variant', hover_name='location', animation_frame='date', animation_group='date',
                        hover_data=['location', 'date', 'variant', 'percent_of_sequences'],
                        color_discrete_map=color_variant_map, title='Dominant COVID Variants 2020-Present',
                        projection='equirectangular'
                        )
    variant_map_fig.update_layout(title_font_size=24, title_x=0.5, showlegend=False, height=500)
    variant_map_fig['layout']['sliders'][0]['pad'] = dict(l=40, b=20, t=20)

    # BUILD VARIANT CHARTS
    country_tracelines = []
    global_tracelines = []
    color_index = 0
    for variant in variants_of_interest:
        # init global and country data
        country_variant_data = pandas.DataFrame(covid_variant_data[covid_variant_data['location'] == country])
        # create a subset of data by variant
        country_variant_data_subset = pandas.concat([country_variant_data[variant], country_variant_data['date']], axis=1)
        global_variant_data_subset = pandas.concat([covid_variant_data[variant], covid_variant_data['date']], axis=1)
        # sum the data by date in case there are any duplicates
        country_variant_data_sum = country_variant_data_subset.groupby(country_variant_data['date']).sum()
        global_variant_data_avg = global_variant_data_subset.groupby(covid_variant_data['date']).mean()
        # add date back in as a column since it moves to index with .groupby()
        country_variant_data_sum.reset_index(level=0, inplace=True)
        global_variant_data_avg.reset_index(level=0, inplace=True)
        # make trace lines and add them to respective arrays
        country_trace = go.Scatter(
                x=country_variant_data_sum['date'],
                y=country_variant_data_sum[variant],
                name=variant,
                fill='tonexty',
                stackgroup='variants',
                line=dict(color=color_variant_map[variant])
        )
        global_trace = go.Scatter(
                x=global_variant_data_avg['date'],
                y=global_variant_data_avg[variant],
                name=variant,
                fill='tonexty',
                stackgroup='variants',
                line=dict(color=color_variant_map[variant])
        )
        country_tracelines.append(country_trace)
        global_tracelines.append(global_trace)
        color_index += 1
    country_variant_fig = go.Figure(data=country_tracelines, layout=dict(title=country + ' COVID Variant Trends', height=500))
    country_variant_fig.update_layout(title_font_size=24, title_x=0.5, xaxis=range_slider,
                                      legend=dict(orientation="v", traceorder='normal'))
    country_variant_fig.update_yaxes(title_text='% of Positive Tests Sequenced')
    global_variant_fig = go.Figure(data=global_tracelines, layout=dict(title='Global COVID Variant Trends', height=500))
    global_variant_fig.update_layout(title_font_size=24, title_x=0.5, xaxis=range_slider,
                                     legend=dict(orientation="v", traceorder='normal', xanchor='left', yanchor='top'))
    global_variant_fig.update_yaxes(title_text='% of Positive Tests Sequenced')

    # BUILD CASE CHARTS
    country_case_data = covid_case_data[covid_case_data['location'] == country]
    country_case_data_sum = country_case_data.groupby(country_case_data['date']).sum()
    country_case_data_sum.reset_index(level=0, inplace=True)
    country_case_data_sum['location'] = country
    country_new_cases = go.Scatter(
        x=country_case_data_sum['date'],
        y=country_case_data_sum['new_cases'],
        name=country,
        fill='tonexty',
        stackgroup=0,
        line=dict(color="#" + colors[0])
    )
    country_case_fig = go.Figure(data=country_new_cases, layout=dict(title=country + ' COVID New Case Trends'))
    country_case_fig.update_layout(title_font_size=24, title_x=0.5, xaxis=range_slider)
    country_case_fig.update_yaxes(title_text='Number of New Cases')

    global_case_data_sum = covid_case_data.groupby(covid_case_data['date']).sum()
    global_case_data_sum.reset_index(level=0, inplace=True)
    global_new_cases = go.Scatter(
            x=global_case_data_sum['date'],
            y=global_case_data_sum['new_cases'],
            name='Global',
            fill='tonexty',
            stackgroup=0,
            line=dict(color="#"+colors[3])
    )
    global_case_fig = go.Figure(data=global_new_cases, layout=dict(title='Global COVID New Case Trends'))
    global_case_fig.update_layout(title_font_size=24, title_x=0.5, xaxis=range_slider)
    global_case_fig.update_yaxes(title_text='Number of New Cases')

    # BUILD VAXX CHARTS
    country_vax_data = vaccination_data[vaccination_data['location'] == country]
    country_vax_data_sum = country_vax_data.groupby(country_vax_data['date']).sum()
    country_vax_data_sum.reset_index(level=0, inplace=True)
    country_vax_data_sum['location'] = country
    country_people_boosted = go.Scatter(
        x=country_vax_data_sum['date'],
        y=country_vax_data_sum['total_boosters_per_hundred'],
        name='People<br>Boosted<br>per 100<br>',
        fill='tonexty',
        stackgroup=0,
        line=dict(color="#" + colors[2])
    )
    country_people_fully_vaccinated = go.Scatter(
        x=country_vax_data_sum['date'],
        y=country_vax_data_sum['people_fully_vaccinated_per_hundred'],
        name='People Fully<br>Vaccinated<br>per 100<br>',
        fill='tonexty',
        stackgroup=1,
        line=dict(color="#" + colors[4])
    )
    country_people_vaccinated = go.Scatter(
        x=country_vax_data_sum['date'],
        y=country_vax_data_sum['people_vaccinated_per_hundred'],
        name='People<br>Vaccinated<br>per 100',
        fill='tonexty',
        stackgroup=2,
        line=dict(color="#" + colors[5])
    )
    country_vaxx_fig = go.Figure(data=[country_people_boosted, country_people_fully_vaccinated, country_people_vaccinated],
                    layout=dict(title=country + ' COVID Vaccination Trends'))
    country_vaxx_fig.update_layout(title_font_size=24, title_x=0.5, xaxis=range_slider)
    country_vaxx_fig.update_yaxes(title_text='Number of Vaccinations per 100 People')

    global_vaxx_data_sum = vaccination_data.groupby(vaccination_data['date']).mean()
    global_vaxx_data_sum.reset_index(level=0, inplace=True)
    global_people_boosted = go.Scatter(
        x=global_vaxx_data_sum['date'],
        y=global_vaxx_data_sum['total_boosters_per_hundred'],
        name='People<br>Boosted<br>per 100<br>',
        fill='tonexty',
        stackgroup=0,
        line=dict(color="#" + colors[2])
    )
    global_people_fully_vaccinated = go.Scatter(
        x=global_vaxx_data_sum['date'],
        y=global_vaxx_data_sum['people_fully_vaccinated_per_hundred'],
        name='People Fully<br>Vaccinated<br>per 100<br>',
        fill='tonexty',
        stackgroup=1,
        line=dict(color="#" + colors[4])
    )
    global_people_vaccinated = go.Scatter(
        x=global_vaxx_data_sum['date'],
        y=global_vaxx_data_sum['people_vaccinated_per_hundred'],
        name='People<br>Vaccinated<br>per 100',
        fill='tonexty',
        stackgroup=2,
        line=dict(color="#" + colors[5])
    )
    global_vaxx_fig = go.Figure(data=[global_people_boosted, global_people_fully_vaccinated, global_people_vaccinated],
                    layout=dict(title='Global COVID Vaccination Trends'))
    global_vaxx_fig.update_layout(title_font_size=24, title_x=0.5, xaxis=range_slider)
    global_vaxx_fig.update_yaxes(title_text='Number of Vaccinations per 100 People')
    return variant_map_fig, country_variant_fig, global_variant_fig, country_case_fig, global_case_fig, \
           country_vaxx_fig, global_vaxx_fig


logging.info('END building interactive functionality of the app - ' + datetime.now().strftime('%m/%d/%Y, %H:%M:%S'))


if __name__ == '__main__':
    app.run_server(debug=True)

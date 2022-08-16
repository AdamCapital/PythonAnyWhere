#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importar bibliotecas
import pandas as pd
import numpy as np
import scipy
# from xbbg import blp
from datetime import date

import plotly.express as px


# from plotly.offline import iplot
# import plotly.offline as py
# import plotly.graph_objs as go
# import kaleido


# In[ ]:


def style_negative(v, props=''):
    return props if v[0] == '-' and len(v) > 1 else None


# In[ ]:


# importar dados
data = pd.DataFrame(
    pd.read_table('Z:\\Dados_Coletas_Economia\\Portfolio_Analysis\\OverviewResume28Jul2022.txt', delimiter='	'))

# limpar dados
data = data[['Product', 'Description', 'Name', 'ExpiryDate', 'Amount',
             'YestAmount', 'Financial Price', 'Price', 'Yesterday Fin Price',
             'YesterdayPrice', 'YesterdayPosition', 'Position', 'PL', 'PL Pct']]

product_List = 'Taxa|Despesa|Administration|Fee|Diferença de Marcação|Dividendos|Provision'
data = data[~data['Product'].str.contains(product_List, regex=True)].reset_index(drop=True)

# In[ ]:


# importar tickers
dict_names_tickers = pd.read_excel('Z:\\Dados_Coletas_Economia\\Portfolio_Analysis\\dict_names_tickers.xlsx')

# In[ ]:


# juntando dados
assets_tickers = pd.merge(data, dict_names_tickers, on='Product', how='left')

# In[ ]:


import_tickers = pd.read_csv('import_tickers.csv', sep=',')

# In[ ]:

'''
# importar dados dos ativos
start_date = pd.to_datetime('2015-01-01')
end_date = date.today()

import_tickers = blp.bdh(list(assets_tickers['Ticker'].unique()), 'Px_Last', start_date, end_date)
import_tickers.columns = import_tickers.columns.droplevel(1)
import_tickers.columns.name = None
import_tickers.to_csv('Z://Dados_Coletas_Economia//Portfolio_Analysis//import_tickers.csv', sep=',', index=False)
#import_tickers = pd.read_csv('Z://Dados_Coletas_Economia//Portfolio_Analysis//import_tickers.csv', sep = ',')
'''

# In[ ]:


# retirando os retornos
import_tickers_ret = import_tickers.apply(lambda x: np.log(x / x.shift()))
# import_tickers_ret = import_tickers_ret.fillna(0)


# In[ ]:


# Retornos acumulados
import_tickers_ret_acum = import_tickers_ret.apply(lambda x: 100 * x.cumsum())
import_tickers_ret_acum.iloc[0, :] = 0
import_tickers_ret_acum.fillna(method='ffill', inplace=True)
import_tickers_ret_acum = import_tickers_ret_acum.reset_index().rename(columns={'index': 'date'})
import_tickers_ret_acum

# In[ ]:


# Correlações
cross_corr_df = import_tickers_ret.corr()
cross_corr_df = cross_corr_df.apply(lambda x: round(x * 100, 0)).fillna('-')
# cross_corr_df = cross_corr_df.reset_index()
# cross_corr_df = cross_corr_df.rename(columns = {'index':'Correl'})
cross_corr_df

cross_corr_heatmap = px.imshow(cross_corr_df, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
cross_corr_heatmap.update_layout(
    title_text="Correlations (%)", title_x=0.5,
    font={'size': 14},
    yaxis_title=" ",
    xaxis_title=" ",
    legend_title=" ",
    font_color="black",
    hoverlabel={'bgcolor': "black", 'font_size': 16}
)
cross_corr_heatmap.update_xaxes(tickangle=55)

# In[ ]:


# Estatísticas Descritivas
stats = pd.concat([import_tickers_ret.apply(lambda x: x.mean()),
                   import_tickers_ret.apply(lambda x: x.median()),
                   import_tickers_ret.apply(lambda x: x.max()),
                   import_tickers_ret.apply(lambda x: x.min()),
                   import_tickers_ret.apply(lambda x: x.var()),
                   import_tickers_ret.apply(lambda x: x.std()),
                   import_tickers_ret.apply(lambda x: x.skew()),
                   import_tickers_ret.apply(lambda x: x.kurtosis())], axis=1).T
stats.index = ['Mean', 'Median', 'Max', 'Min', 'Variance', 'Std. Dev', 'Skewness', 'Kurtosis']
stats1 = stats.iloc[:6].apply(lambda x: round(x * 100, 2).astype(str)) + '%'
stats2 = stats.iloc[6:].apply(lambda x: round(x, 2).astype(str))
stats = pd.concat([stats1, stats2], axis=0).reset_index()
stats = stats.rename(columns={'index': 'Stats'})
stats

# In[ ]:


# Maximum Drawdown
Roll_Max = import_tickers.fillna(method='ffill').cummax()
Daily_Drawdown = (100 * (import_tickers.fillna(method='ffill') / Roll_Max - 1)).reset_index().rename({'index': 'date'},
                                                                                                     axis=1)
Daily_Drawdown

# In[ ]:


from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = 'Adam Capital - Asset Allocation Tool'

app.layout = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col(html.Img(src=app.get_asset_url('logo_vertical.png')),
                    align="start",
                    width={'size': 1, 'offset': 0}),
            dbc.Col(
                html.H1('Asset Allocation Tool (1.0 version) - Last Update: ' + str(date.today().strftime('%m/%d/%Y'))),
                align="center",
                width={'size': "auto", 'offset': 1})
        ])
    ]),
    dcc.Tabs(children=[
        dcc.Tab(label='Asset Overview', children=[
            html.P("Asset:"),
            dcc.Dropdown(list(import_tickers_ret.columns), "PETR4 Equity", id="asset"),
            html.Div([
                dbc.Row([
                    dbc.Col(dcc.Graph(id="graph_hist"),
                            width={'size': 6}),
                    dbc.Col(dcc.Graph(id="graph_cumul"),
                            width={'size': 6})
                ])
            ]),
            html.Div([
                dbc.Row([
                    dbc.Col(dcc.Graph(id="graph_mdd"),
                            width={'size': 6}),
                    dbc.Col(dcc.Graph(id="graph_vol"),
                            width={'size': 6})
                ])
            ])
        ]),
        dcc.Tab(label='Portfolio Analysis', children=[
            html.Div([
                dbc.Row([
                    dbc.Col(dbc.Table.from_dataframe(stats))
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=cross_corr_heatmap))
                ])
            ])
        ])
    ])
    # html.P("Chart Type:"),
    # dcc.Dropdown(id="std", min=1, max=3, value=1,
    #           marks={1: '1', 3: '3'}),
])


# histograma dos retornos
@app.callback(
    Output("graph_hist", "figure"),
    Input("asset", "value"))
def display_hist(asset):
    subset = import_tickers_ret[asset].dropna()
    fig = px.histogram(subset, x=asset, marginal='box', histnorm='percent', color_discrete_sequence=["#29abe2"])
    fig['data'][1]['x'] = np.around(fig['data'][1]['x'], 2)

    quant = 1

    fig.update_layout(title_text='Returns Histogram - ' + str(asset), title_x=0.5,
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      font={'size': 16},
                      yaxis_title="Probability (%)",
                      xaxis_title="Returns",
                      legend_title="",
                      font_color="black",
                      hoverlabel={'bgcolor': "white", 'font_size': 16},
                      xaxis={'range': [import_tickers_ret[asset].dropna().quantile(quant / 100),
                                       import_tickers_ret[asset].dropna().quantile(1 - quant / 100)]}, )

    fig.add_vline(x=0, line_width=.5, line_dash="dash", line_color="black")
    fig.add_vline(x=import_tickers_ret[asset].std(), line_width=.5, line_dash="dash", line_color="black")
    fig.add_vline(x=import_tickers_ret[asset].std() * -1, line_width=.5, line_dash="dash", line_color="black")
    fig.add_vline(x=import_tickers_ret[asset].std() * 2, line_width=.5, line_dash="dash", line_color="black")
    fig.add_vline(x=import_tickers_ret[asset].std() * -2, line_width=.5, line_dash="dash", line_color="black")

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=True, gridwidth=1, gridcolor='#E7E7E7')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', showgrid=True, gridwidth=1, gridcolor='#E7E7E7')

    return fig


# retornos acumulados
@app.callback(
    Output("graph_cumul", "figure"),
    Input("asset", "value"))
def display_cumul(asset):
    fig = px.line(import_tickers_ret_acum, x='date', y=asset)

    fig['data'][0]['line']['color'] = "#29abe2"

    fig.update_layout(title_text='Cumulative Returns - ' + str(asset), title_x=0.5,
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      font={'size': 16},
                      yaxis_title="Cumulative Returns (%)",
                      xaxis_title="Date",
                      legend_title="",
                      font_color="black",
                      hoverlabel={'bgcolor': "white", 'font_size': 16},
                      legend_x=0,
                      legend_y=1)

    fig.add_trace(go.Scatter(x=import_tickers_ret_acum['date'],
                             y=import_tickers_ret_acum[asset].rolling(window=66).mean(),
                             mode='lines',
                             name='MA 3m'))

    fig['data'][1]['line']['width'] = 2.5

    fig.add_hline(y=import_tickers_ret_acum[asset].mean(), line_width=.5, line_dash="dash", line_color="black")

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=True, gridwidth=1, gridcolor='#E7E7E7')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', showgrid=True, gridwidth=1, gridcolor='#E7E7E7')

    return fig


# Underwater chart
@app.callback(
    Output("graph_mdd", "figure"),
    Input("asset", "value"))
def display_mdd(asset):
    fig = px.area(Daily_Drawdown, x='date', y=asset)

    fig['data'][0]['line']['color'] = "#29abe2"

    fig.update_layout(title_text='Maximum Drawdown - ' + str(asset), title_x=0.5,
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      font={'size': 16},
                      yaxis_title="Maximum Drawdown (%)",
                      xaxis_title="Date",
                      legend_title="",
                      font_color="black",
                      hoverlabel={'bgcolor': "white", 'font_size': 16},
                      legend_x=0,
                      legend_y=1)

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=True, gridwidth=1, gridcolor='#E7E7E7')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', showgrid=True, gridwidth=1, gridcolor='#E7E7E7')

    return fig


# Volatilidade móvel
@app.callback(
    Output("graph_vol", "figure"),
    Input("asset", "value"))
def display_vol(asset):
    vol = import_tickers_ret[asset].copy().dropna().reset_index().rename({'index': 'date'}, axis=1)
    vol['1w'] = vol[asset].rolling(window=5).std()
    vol['2m'] = vol[asset].rolling(window=22).std()

    meanvol = vol['1w'].mean()

    vol = pd.melt(vol, id_vars=['date', asset], value_vars=['1w', '2m'])

    fig = px.line(vol, x='date', y='value', color='variable')

    fig['data'][0]['line']['color'] = "#29abe2"
    fig['data'][1]['line']['width'] = 2.5

    fig.update_layout(title_text='Volatility of Returns - ' + str(asset), title_x=0.5,
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      font={'size': 16},
                      yaxis_title="Volatility of Returns (%)",
                      xaxis_title="Date",
                      legend_title="",
                      font_color="black",
                      hoverlabel={'bgcolor': "white", 'font_size': 16},
                      legend_x=0,
                      legend_y=1)

    fig.add_hline(y=meanvol, line_width=.5, line_dash="dash", line_color="black")

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=True, gridwidth=1, gridcolor='#E7E7E7')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', showgrid=True, gridwidth=1, gridcolor='#E7E7E7')

    return fig


app.run_server(debug=True, use_reloader=False)

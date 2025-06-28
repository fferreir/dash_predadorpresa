import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy import heaviside
from textwrap import dedent
import plotly.graph_objects as go


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME, requests_pathname_prefix='/dash_predadorpresa/'])
server = app.server
cabecalho = html.H1("Modelo Predador-Presa (Lotka-Volterra)",className="bg-primary text-white p-2 mb-4")

descricao = dcc.Markdown(
    '''
    As equações de Lotka-Volterra são um par de equações diferenciais frequentemente utilizadas para descrever dinâmicas
    nos sistemas biológicos, especialmente quando duas espécies interagem: uma como presa e outra como predadora.
    Os modelos mais básicos para predador-presa de duas espécies são chamados de Lokta-Volterra, e consideram que a
    única fonte de alimento da espécie predadora é a população da presa e que não há competição alguma entre
    indivíduos da mesma espécie
    ''', mathjax=True
)

parametros = dcc.Markdown(
    '''
    * r: taxa de crescimento da população de presas
    * c: taxa relacionada à diminuição das presas por conta da ação dos predadores
    * b: taxa relacionada ao aumento da pop. predadores. Consideramos que o sucesso reprodutivo dos predadores está diretamente ligado à atividade de predação.
    * m: mortalidade da população de predadores


    ''', mathjax=True
)
cond_inicial = dcc.Markdown(
    '''
    * Presas = 20
    * Predadores = 5
    ''', mathjax=True
)

perguntas = dcc.Markdown(
    '''
    1. Considerando uma taxa de natalidade das presas de 80% por ano (r = 0.80), observe o que ocorre com as duas populações,
    quando partimos de uma situação inicial com 20 presas e 5 predadores, para c=0.1, b=0.02 e m=0.50. Qual o período de
    oscilação e qual o número máximo que as populações de presas e de predadores atingem?
    2. Mantenha as mesmas taxas do item 1 e considere as seguintes condições iniciais: N = (20 presas, 10 predadores),
    N = (10 presas, 100 predadores), N = (100 presas, 10 predadores). Verifique o que ocorre.
    3. Refaça o item 1, modificando a taxa b para b=0.08. Analise também o período de oscilação e o tamanho máximo das populações.
    ''', mathjax=True
)

textos_descricao = html.Div(
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    descricao, title="Descrição do modelo"
                ),
                dbc.AccordionItem(
                    parametros, title="Parâmetros do modelo"
                ),
                dbc.AccordionItem(
                    cond_inicial, title="Condições iniciais"
                ),
                dbc.AccordionItem(
                    perguntas, title="Perguntas"
                ),
            ],
            start_collapsed=True,
        )
    )

ajuste_condicoes_iniciais = html.Div(
        [
            html.P("Ajuste das condições iniciais", className="card-header border-dark mb-3"),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Número inicial de presas''', mathjax=True), html_for="s_init"),
                    dcc.Slider(id="presa", min=5, max=100, value=20, tooltip={"placement": "bottom", "always_visible": False}),
                ],
                className="m-2",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Número inicial de predadores ''', mathjax=True), html_for="i_init"),
                    dcc.Slider(id="pred", min=5, max=100, value=5, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),

        ],
        className="card border-dark mb-3",
    )

ajuste_parametros = html.Div(
        [
            html.P("Ajuste dos parâmetros", className="card-header border-dark mb-3"),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Taxa de crescimento da população de presas (r)''', mathjax=True), html_for="alpha"),
                    dcc.Slider(id="r", min=0.4, max=1.2, value=0.8, tooltip={"placement": "bottom", "always_visible": False}),
                ],
                className="m-2",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Taxa relacionada à diminuição de presas por conta da predação (c) ''', mathjax=True), html_for="beta"),
                    dcc.Slider(id="c", min=0.01, max=0.2, value=0.1, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Taxa relacionada ao aumento da pop. predadores (b)''', mathjax=True), html_for="gamma"),
                    dcc.Slider(id="b", min=0.01, max=0.08, value=0.02,  tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Taxa de mortalidade da população de predadores (m)''', mathjax=True), html_for="delta"),
                    dcc.Slider(id="m", min=0.3, max=0.7, value=0.5, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),
        ],
        className="card border-dark mb-3",
    )

def ode_sys(t, state, r, c, b, m):
    pred, presa=state
    dpred_dt=b*pred*presa-m*pred
    dpresa_dt=r*presa - c*pred*presa
    return [dpred_dt, dpresa_dt]

@app.callback(Output('population_chart', 'figure'),
              [Input('presa', 'value'),
              Input('pred', 'value'),
              Input('r', 'value'),
              Input('c', 'value'),
              Input('b', 'value'),
              Input('m', 'value')])
def gera_grafico(presa, pred, r, c, b, m):
    t_begin = 0.
    t_end = 70.
    t_span = (t_begin, t_end)
    t_nsamples = 10000
    t_eval = np.linspace(t_begin, t_end, t_nsamples)
    sol = solve_ivp(fun=ode_sys,
                    t_span=t_span,
                    y0=[pred, presa],
                    args=(r,c,b,m),
                    t_eval=t_eval,
                    method='Radau')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='Predador',
                             line=dict(color='#00b400', width=4, dash='dashdot')))
    fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name ='Presa',
                             line=dict(color='#ff0000', width=4)))
    fig.update_layout(title='Dinâmica Modelo Predador-Presa (Lotka-Volterra)',
                       xaxis_title='Tempo',
                       yaxis_title='População')
    return fig

app.layout = dbc.Container([
                cabecalho,
                dbc.Row([
                        dbc.Col(html.Div(ajuste_parametros), width=3),
                        dbc.Col(html.Div([ajuste_condicoes_iniciais,html.Div(textos_descricao)]), width=3),
                        dbc.Col(dcc.Graph(id='population_chart'), width=6),
                        ]),
              ], fluid=True),


if __name__ == '__main__':
    app.run(debug=False)

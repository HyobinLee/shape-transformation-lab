import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np

app = dash.Dash(__name__)
server = app.server

# 초기 점 위치
def get_initial_point():
    return 3.0, 1.0

# 대칭 변환 행렬 함수
def get_symmetry_matrix(axis_type, angle_degree=None):
    if axis_type == "x축":
        return np.array([[1, 0], [0, -1]])
    elif axis_type == "y축":
        return np.array([[-1, 0], [0, 1]])
    elif axis_type == "y=ax" and angle_degree is not None:
        theta = np.radians(angle_degree)
        cos2 = np.cos(2 * theta)
        sin2 = np.sin(2 * theta)
        return np.array([[cos2, sin2], [sin2, -cos2]])
    else:
        return np.identity(2)

# 직선 방정식 생성 (y = ax)
def get_line_coordinates(axis_type, angle_degree=None):
    x = np.linspace(-6, 6, 100)
    if axis_type == "x축":
        return x, np.zeros_like(x)
    elif axis_type == "y축":
        return np.zeros_like(x), x
    elif axis_type == "y=ax" and angle_degree is not None:
        a = np.tan(np.radians(angle_degree))
        return x, a * x
    else:
        return x, np.zeros_like(x)

# 그래프 생성
def generate_figure(x0, y0, axis1, angle1, axis2, angle2):
    point = np.array([x0, y0])
    A = get_symmetry_matrix(axis1, angle1)
    B = get_symmetry_matrix(axis2, angle2)
    transformed = B @ (A @ point)

    x_line1, y_line1 = get_line_coordinates(axis1, angle1)
    x_line2, y_line2 = get_line_coordinates(axis2, angle2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[x0], y=[y0], mode='markers',
                             marker=dict(size=10, color='blue'),
                             name='원래 점', uid="original"))
    fig.add_trace(go.Scatter(x=[transformed[0]], y=[transformed[1]], mode='markers',
                             marker=dict(size=10, color='red'),
                             name='변환된 점'))
    fig.add_trace(go.Scatter(x=x_line1, y=y_line1, mode='lines',
                             line=dict(color='green', dash='dash'),
                             name='첫 번째 축'))
    fig.add_trace(go.Scatter(x=x_line2, y=y_line2, mode='lines',
                             line=dict(color='green', dash='dot'),
                             name='두 번째 축'))

    fig.update_layout(
        dragmode='drawopenpath',
        clickmode='event+select',
        xaxis=dict(range=[-6, 6], scaleanchor='y'),
        yaxis=dict(range=[-6, 6]),
        height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True
    )
    return fig

# 앱 레이아웃
app.layout = html.Div([
    html.H2("🟨 두 번의 대칭이동 시뮬레이터 (Dash 기반)"),

    html.Div([
        html.Label("첫 번째 대칭축:"),
        dcc.Dropdown(
            id='axis1',
            options=[{'label': x, 'value': x} for x in ['x축', 'y축', 'y=ax']],
            value='x축'
        ),
        dcc.Slider(id='angle1', min=-90, max=90, step=1, value=45,
                   marks={-90: '-90°', 0: '0°', 90: '90°'},
                   tooltip={"placement": "bottom"}),

        html.Br(),

        html.Label("두 번째 대칭축:"),
        dcc.Dropdown(
            id='axis2',
            options=[{'label': x, 'value': x} for x in ['x축', 'y축', 'y=ax']],
            value='y축'
        ),
        dcc.Slider(id='angle2', min=-90, max=90, step=1, value=-45,
                   marks={-90: '-90°', 0: '0°', 90: '90°'},
                   tooltip={"placement": "bottom"})
    ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        dcc.Graph(
            id='graph',
            config={'editable': True},
            figure=generate_figure(*get_initial_point(), 'x축', 45, 'y축', -45)
        )
    ], style={'width': '58%', 'display': 'inline-block'})
])

# 콜백
@app.callback(
    Output('graph', 'figure'),
    Input('graph', 'relayoutData'),
    Input('axis1', 'value'),
    Input('angle1', 'value'),
    Input('axis2', 'value'),
    Input('angle2', 'value'),
    State('graph', 'figure')
)
def update_graph(relayoutData, axis1, angle1, axis2, angle2, fig):
    x0, y0 = get_initial_point()

    # 추후 확장 가능: 드래그한 점 위치 읽기
    if relayoutData and 'shapes[0].x0' in relayoutData:
        x0 = relayoutData['shapes[0].x0']
        y0 = relayoutData['shapes[0].y0']

    return generate_figure(x0, y0, axis1, angle1, axis2, angle2)

# 실행
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)


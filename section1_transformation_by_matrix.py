import streamlit as st
import numpy as np
import plotly.graph_objects as go

import lab_ui
from lab_ui import chart, equal_axes, format_number


def parse_point(text, fallback):
    """"1,1" 같은 좌표 입력을 읽는다. 못 읽으면 기본값으로 되돌린다.

    학생은 반드시 이상한 값을 넣는다 — `1;1`, 빈칸, `1,1,1`. 예전에는 그것이
    그대로 예외가 되어 화면에 빨간 traceback 이 떴다. 이 앱의 방침대로
    예외를 흘리지 않고 기본값으로 되돌린 뒤 무엇이 잘못됐는지 알려 준다.

    Returns:
        (좌표 ndarray, 오류 메시지 또는 None)
    """
    parts = [p.strip() for p in (text or '').split(',')]
    if len(parts) != 2:
        return np.array(fallback, dtype=float), f"'{text}' — 쉼표로 x, y 두 개를 넣어 주세요 (예: 1,1)"
    try:
        return np.array([float(parts[0]), float(parts[1])]), None
    except ValueError:
        return np.array(fallback, dtype=float), f"'{text}' — 숫자만 넣어 주세요 (예: 1,1)"


def run_transformation_by_matrix():
    st.header("🟩 (1) 행렬을 통한 일차변환 시뮬레이터")
    st.markdown("여러 도형을 여러 행렬로 일차변환하는 실험을 해 보세요.")

    # ✅ Plotly 버전 시각화 함수
    def plot_shape(shape_type, shape, transformed, matrix, a=1, b=1, c=0,
                   rainbow=False, numbers=False):
        fig = go.Figure()

        # 직선일 경우 변환된 점 하나 강조.
        # a, b 가 둘 다 0이면 직선이 아니므로(호출부에서 이미 안내한다) 건너뛴다.
        new_point = None
        if shape_type == "직선" and not (a == 0 and b == 0):
            base_point = np.array([0, c / b]) if b != 0 else np.array([c / a, 0])
            new_point = np.dot(base_point, matrix.T)

        # 축 범위를 **먼저** 정한다. 무지개를 "보이는 영역 안에서의 호의 길이"로
        # 칠하려면 어디까지가 화면인지 알아야 하기 때문이다.
        all_x = np.concatenate([shape[:, 0], transformed[:, 0]])
        all_y = np.concatenate([shape[:, 1], transformed[:, 1]])
        if new_point is not None:
            all_x = np.append(all_x, new_point[0])
            all_y = np.append(all_y, new_point[1])
        x_center, y_center = np.mean(all_x), np.mean(all_y)
        half_range = max(np.ptp(all_x), np.ptp(all_y)) * 0.75
        half_range = min(half_range, 20)  # 최대 20으로 제한
        if half_range < 1:
            half_range = 2
        x_range = [x_center - half_range, x_center + half_range]
        y_range = [y_center - half_range, y_center + half_range]

        # ✅ 무지개 대응 — 호의 길이 기준으로 다시 뽑아 색을 입힌다.
        #    변환은 다시 뽑은 점에 그대로 적용하면 되므로, 색 배열 하나가
        #    변환 전후를 잇는다.
        t = None
        src, dst = shape, transformed
        if rainbow:
            src, t = lab_ui.arclength_parameter(shape, x_range, y_range)
            if t is not None:
                dst = src @ matrix.T
        style = lab_ui.channel_style(rainbow=t is not None)
        closed = lab_ui.is_closed(shape)

        def draw(points, role, name):
            common = dict(x=points[:, 0], y=points[:, 1], name=name,
                          mode='lines+markers')
            if t is None:
                color = style[role]['color']
                fig.add_trace(go.Scatter(
                    line=dict(color=color, dash=style[role]['dash']),
                    marker=dict(color=color, symbol=style[role]['symbol'], size=6),
                    **common))
            else:
                # Plotly 는 선 색을 점마다 줄 수 없다(line.color 는 trace 당 하나).
                # 그래서 선은 옅은 무채색으로 두고 색은 마커가 진다.
                fig.add_trace(go.Scatter(
                    line=dict(color='lightgray', width=1, dash=style[role]['dash']),
                    marker=lab_ui.rainbow_marker(t, closed, size=7,
                                                 symbol=style[role]['symbol']),
                    **common))

        draw(src, 'before', '변환전 도형')
        draw(dst, 'after', '변환후 도형')

        # 웨이포인트 번호 — 색을 못 봐도 대응이 읽히고, 변환 후 번호 간격이
        # 흐트러진 정도로 어디가 늘어났는지를 수로 읽을 수 있다.
        if t is not None and numbers:
            idx = lab_ui.waypoint_indices(10, len(src))
            for points, color in ((src, 'black'), (dst, 'dimgray')):
                fig.add_trace(go.Scatter(
                    x=points[idx, 0], y=points[idx, 1],
                    mode='text', text=[str(k) for k in range(len(idx))],
                    textfont=dict(size=11, color=color),
                    hoverinfo='skip', showlegend=False,
                ))

        if new_point is not None:
            fig.add_trace(go.Scatter(
                x=[new_point[0]], y=[new_point[1]],
                mode='markers', name='변환된 점',
                marker=dict(color='red', size=10, symbol='circle')
            ))

        # 도형 종류가 바뀌면 시야를 되돌리고, 행렬만 바꿀 때는 확대해 둔
        # 시야를 그대로 둔다.
        equal_axes(fig, view_key=f"s1-{shape_type}",
                   x_range=x_range, y_range=y_range)
        fig.update_layout(
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        return fig



    ##################### (1) ########################
    # ✅ 메뉴별 콘텐츠


    col1, spacer, col2 = st.columns([1.4, 0.3, 2])  # 좌:입력 / 우:출력

    with col1:
        st.subheader("# 도형 입력 ___________________")
        shape_type = st.selectbox("도형 종류를 선택하세요", ["삼각형", "사각형", "원", "직선"])

        # 잘못된 입력은 예외를 내지 않고 기본값으로 되돌린 뒤 여기 모아서 알린다.
        problems = []

        def point(label, default):
            value, problem = parse_point(st.text_input(label, default), default.split(','))
            if problem:
                problems.append(problem)
            return value

        # 직선 계수는 다른 도형일 때도 plot_shape 에 넘어가므로 기본값을 둔다.
        a, b, c = 1.0, 1.0, 1.0

        if shape_type == "삼각형":
            A = point("점 A 좌표 (예: 1,1)", "1,1")
            B = point("점 B 좌표 (예: 1,2)", "1,2")
            C = point("점 C 좌표 (예: 2,1)", "2,1")
            shape = np.array([A, B, C, A])
        elif shape_type == "사각형":
            A = point("점 A 좌표 (예: 1,1)", "1,1")
            B = point("점 B 좌표 (예: 1,2)", "1,2")
            C = point("점 C 좌표 (예: 2,2)", "2,2")
            D = point("점 D 좌표 (예: 2,1)", "2,1")
            shape = np.array([A, B, C, D, A])
        elif shape_type == "원":
            center = point("원 중심 좌표 (예: 1,1)", "1,1")
            radius = st.number_input("반지름", value=2.0, step=0.1, format="%.1f")
            theta = np.linspace(0, 2*np.pi, 200)
            shape = np.stack([center[0] + radius * np.cos(theta),
                                center[1] + radius * np.sin(theta)], axis=1)
        elif shape_type == "직선":
            st.markdown("직선의 형태: $ax + by = c$")
            a = st.number_input("계수 a", value=1.0, step=0.1, format="%.1f")
            b = st.number_input("계수 b", value=1.0, step=0.1, format="%.1f")
            c = st.number_input("상수 c", value=2.0, step=0.1, format="%.1f")
            if a == 0 and b == 0:
                # 0x + 0y = c 는 직선이 아니다. 예전에는 여기서 c/a 가 0으로 나눠졌다.
                problems.append("a 와 b 가 둘 다 0이면 직선이 아닙니다. a = 1 로 두고 그립니다.")
                a = 1.0
            x_vals = np.linspace(-20, 20, 400)
            if b != 0:
                y_vals = (c - a * x_vals) / b
            else:
                x_vals = np.full(400, c / a)
                y_vals = np.linspace(-20, 20, 400)
            shape = np.stack([x_vals, y_vals], axis=1)

        for problem in problems:
            st.warning(problem)

        st.subheader("# 2×2 변환 행렬 입력 ___________________")
        a11 = st.number_input("a11", value=1.0, step=0.5, format="%.1f")
        a12 = st.number_input("a12", value=-1.0, step=0.5, format="%.1f")
        a21 = st.number_input("a21", value=1.0, step=0.5, format="%.1f")
        a22 = st.number_input("a22", value=2.0, step=0.5, format="%.1f")
        matrix = np.array([[a11, a12], [a21, a22]])

        # ✅ 관찰 보조 (렌즈)
        st.divider()
        rainbow_on = st.toggle(
            "🌈 무지개 대응 보기", value=False, key="s1_rainbow",
            help="도형을 호의 길이에 따라 무지개로 칠하고, 변환 후에도 같은 색을 "
                 "물려줍니다. 어느 부분이 어디로 갔는지 색으로 따라갈 수 있습니다.",
        )
        numbers_on = rainbow_on and st.toggle(
            "🔢 번호 붙이기", value=False, key="s1_numbers",
            help="등간격 10곳에 번호를 붙입니다. 변환 후 번호 간격이 얼마나 "
                 "흐트러졌는지로 어디가 늘어나고 어디가 눌렸는지 알 수 있습니다.",
        )

    with col2:
        #변환적용
        transformed = np.dot(shape, matrix.T)

        st.subheader("시각화 결과")
        st.subheader("수식 표시")
        st.latex(
            fr"""\text{{입력된 행렬}} =
    \begin{{bmatrix}}
    {a11} & {a12} \\
    {a21} & {a22}
    \end{{bmatrix}}"""
        )

        if shape_type == "원":
            st.latex(rf"(x - {format_number(center[0])})^2 + (y - {format_number(center[1])})^2 = {format_number(radius)}^2")
        elif shape_type == "직선":
            st.latex(rf"\text{{입력된 직선:}} \quad {format_number(a)}x + {format_number(b)}y = {format_number(c)}")

        fig = plot_shape(shape_type, shape, transformed, matrix, a, b, c,
                         rainbow=rainbow_on, numbers=numbers_on)
        chart(fig, key="s1_chart")
        if rainbow_on:
            st.caption(
                "🌈 색 = 변환 전 도형에서의 위치(호의 길이 기준). ○ 변환 전 / △ 변환 후. "
                "**직선에서는 색 간격이 그대로 남고, 원에서는 흐트러집니다 — 왜일까요?**"
            )




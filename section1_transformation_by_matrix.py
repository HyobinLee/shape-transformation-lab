import streamlit as st
import numpy as np
import plotly.graph_objects as go

from lab_ui import CHART_SIZE, chart, equal_axes, format_number


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
    def plot_shape(shape_type, shape, transformed, matrix, a=1, b=1, c=0):
        fig = go.Figure()

        # 원래 도형
        fig.add_trace(go.Scatter(
            x=shape[:, 0], y=shape[:, 1],
            mode='lines+markers',
            name='변환전 도형',
            line=dict(color='blue'),
            marker=dict(color='blue')
        ))

        # 변환된 도형
        fig.add_trace(go.Scatter(
            x=transformed[:, 0], y=transformed[:, 1],
            mode='lines+markers',
            name='변환후 도형',
            line=dict(color='red', dash='dash'),
            marker=dict(color='red')
        ))

        # 직선일 경우 변환된 점 하나 강조.
        # a, b 가 둘 다 0이면 직선이 아니므로(호출부에서 이미 안내한다) 건너뛴다.
        new_point = None
        if shape_type == "직선" and not (a == 0 and b == 0):
            if b != 0:
                base_point = np.array([0, c / b])
            else:
                base_point = np.array([c / a, 0])
            new_point = np.dot(base_point, matrix.T)
            fig.add_trace(go.Scatter(
                x=[new_point[0]], y=[new_point[1]],
                mode='markers',
                name='변환된 점',
                marker=dict(color='red', size=10, symbol='circle')
            ))

        # 축 범위 조절
        all_x = np.concatenate([shape[:, 0], transformed[:, 0]])
        all_y = np.concatenate([shape[:, 1], transformed[:, 1]])
        if new_point is not None:
            all_x = np.append(all_x, new_point[0])
            all_y = np.append(all_y, new_point[1])
        x_center = np.mean(all_x)
        y_center = np.mean(all_y)
        x_range = np.ptp(all_x)
        y_range = np.ptp(all_y)
        half_range = max(x_range, y_range) * 0.75
        half_range = min(half_range, 20)  # 최대 20으로 제한
        if half_range < 1:
            half_range = 2
        # 도형 종류가 바뀌면 시야를 되돌리고, 행렬만 바꿀 때는 확대해 둔
        # 시야를 그대로 둔다.
        equal_axes(
            fig, view_key=f"s1-{shape_type}",
            x_range=[x_center - half_range, x_center + half_range],
            y_range=[y_center - half_range, y_center + half_range],
        )
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

        fig = plot_shape(shape_type, shape, transformed, matrix, a, b, c)
        chart(fig, key="s1_chart")




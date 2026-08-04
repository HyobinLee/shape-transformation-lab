import streamlit as st
import numpy as np
import plotly.graph_objs as go

from lab_ui import chart, equal_axes

#: 복소평면에서 보여 줄 범위.
VIEW = 8.0


def run_rotation_translation():
    st.header("🟥 (4) 회전과 평행이동 시뮬레이터")
    st.markdown("평행이동&회전이동&평행이동은 회전이동일까요? 회전의 기준점은?")

    # ✅ 좌: 입력 / 우: 그래프
    left_col, right_col = st.columns([1, 1.5])

    with left_col:
        # 윗줄: alpha, theta, beta 입력
        st.subheader("🔧 α, θ, β 값을 정해 보세요.")
        st.latex(r"w = (\cos\theta + i\sin\theta)(z + \alpha) + \beta")

        upper_col1, upper_col2, upper_col3 = st.columns(3)

        with upper_col1:
            st.markdown("**α (회전 이전 평행이동)**")
            alpha_re = st.number_input("Re(α)", value=1.0, step=0.5, format="%.2f", key="alpha_re")
            alpha_im = st.number_input("Im(α)", value=0.0, step=0.5, format="%.2f", key="alpha_im")

        with upper_col2:
            st.markdown("**θ (회전각, 도)**")
            theta_deg = st.number_input("회전각 θ", value=45.0, step=5.0, format="%.1f")
            theta_rad = np.radians(theta_deg)
            cos_theta = np.cos(theta_rad)
            sin_theta = np.sin(theta_rad)

        with upper_col3:
            st.markdown("**β (회전 이후 평행이동)**")
            beta_re = st.number_input("Re(β)", value=0.0, step=0.5, format="%.2f", key="beta_re")
            beta_im = st.number_input("Im(β)", value=-1.0, step=0.5, format="%.2f", key="beta_im")

        st.divider()

        # 아랫줄: z 입력
        st.subheader("🖱 입력 복소수 z 와 변환 결과 w 시각화")
        st.markdown("**z = x + iy**")
        x = st.number_input("x (실수 부분)", value=2.0, step=0.5, format="%.2f", key="z_x")
        y = st.number_input("y (허수 부분)", value=1.0, step=0.5, format="%.2f", key="z_y")

    with right_col:
        # ✅ 복소수 정의 및 변환
        z = complex(x, y)
        alpha = complex(alpha_re, alpha_im)
        beta = complex(beta_re, beta_im)
        rotator = complex(cos_theta, sin_theta)
        w = rotator * (z + alpha) + beta

        # ✅ 시각화
        fig = go.Figure()

        # ✅ x축, y축 선 (xref, yref 명시)
        fig.add_shape(
            type="line", x0=-8, y0=0, x1=8, y1=0,
            line=dict(color="black", width=1), layer="below",
            xref="x", yref="y"
        )
        fig.add_shape(
            type="line", x0=0, y0=-8, x1=0, y1=8,
            line=dict(color="black", width=1), layer="below",
            xref="x", yref="y"
        )

        # ✅ 점들 추가
        fig.add_trace(go.Scatter(x=[z.real], y=[z.imag], mode='markers',
                                marker=dict(size=12, color='blue'), name='입력 z'))
        fig.add_trace(go.Scatter(x=[w.real], y=[w.imag], mode='markers',
                                marker=dict(size=12, color='red'), name='변환 결과 w'))
        fig.add_trace(go.Scatter(x=[alpha.real], y=[alpha.imag], mode='markers',
                                marker=dict(size=10, color='purple', symbol='x'), name='첫번째 평행이동 α'))
        fig.add_trace(go.Scatter(x=[beta.real], y=[beta.imag], mode='markers',
                                marker=dict(size=10, color='orange', symbol='x'), name='두번째 평행이동 β'))

        # ✅ 고정 레이아웃. 다른 섹션과 같은 크기로 맞춘다 —
        #    섹션을 옮길 때마다 페이지 높이가 튀면 그 자체가 깜빡임으로 보인다.
        equal_axes(fig, view_key="s4", x_range=[-VIEW, VIEW], y_range=[-VIEW, VIEW])
        fig.update_layout(
            xaxis_title="Re",
            yaxis_title="Im",
            yaxis_constrain='domain',
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True,
            title="복소평면에서의 회전+평행이동 변환 시각화"
        )

        chart(fig, key="s4_chart")

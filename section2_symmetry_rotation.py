import streamlit as st
import numpy as np
import plotly.graph_objs as go

import lab_ui
from lab_ui import chart, equal_axes

#: 좌표평면에서 보여 줄 범위. 학생 입력도 이 범위로 제한한다.
VIEW = 5.0


# ✅ 대칭 행렬 생성 함수 (축의 종류와 각도 입력 → 행렬)
def reflection_matrix(axis_type, angle_deg=None):
    if axis_type == 'x축':
        return np.array([[1, 0], [0, -1]])
    elif axis_type == 'y축':
        return np.array([[-1, 0], [0, 1]])
    elif axis_type == '직선y=ax':
        # 각도 → 라디안 → 기울기
        theta_rad = np.radians(angle_deg)
        a = np.tan(theta_rad)
        norm = 1 + a**2
        return (1 / norm) * np.array([[1 - a**2, 2*a], [2*a, a**2 - 1]])
    # 모르는 축이면 예외를 던지지 않고 항등변환으로 되돌린다.
    # (예전에는 암묵적으로 None 을 돌려줘서 뒤의 `@` 에서 죽었다.)
    return np.eye(2)

# ✅ 시뮬레이터 실행 함수
def run_symmetry_rotation():
    st.header("(2) 두 번의 대칭이동 시뮬레이터")
    st.caption("두 축 대칭의 결과가 회전과 같음을 시각적으로 관찰해 보세요.")



    # 초기 점.
    #
    # 그래프 클릭과 숫자 입력이 **같은 세션 키**를 공유해야 한다. 예전에는
    # 클릭이 별도의 `selected_point` 만 갱신했는데, 재실행 때
    # `number_input(value=..., key=...)` 이 저장돼 있던 옛 위젯값을 돌려주고
    # 그것이 클릭 결과를 덮어써서 **클릭이 전혀 먹지 않았다.**
    # 그래서 기본값은 위젯이 만들어지기 전에 한 번만 세션에 심고,
    # `value=` 는 넘기지 않는다.
    if "input_x" not in st.session_state:
        st.session_state["input_x"] = 2.0
        st.session_state["input_y"] = 1.0

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("🖱 입력 설정")
        st.markdown("⬇ **초기점과 2개의 대칭축을 입력하여 변환된 점을 관찰해 보세요.**")

        # ✅ 초기 점 좌표 입력 (그래프를 클릭해도 여기 값이 따라 바뀐다)
        st.markdown(f"🔵 **초기점 좌표를 입력하세요.({-VIEW:.0f}와 {VIEW:.0f} 사이) "
                    "— 그래프를 클릭해도 됩니다.**")
        x0 = st.number_input("x 좌표", min_value=-VIEW, max_value=VIEW,
                             step=0.3, format="%.2f", key="input_x")
        y0 = st.number_input("y 좌표", min_value=-VIEW, max_value=VIEW,
                             step=0.3, format="%.2f", key="input_y")

        axis1 = st.selectbox("첫 번째 대칭축", ["x축", "y축", "직선y=ax"], key="axis1")
        if axis1 == "직선y=ax":
            angle1 = st.number_input("x축과 이루는 각도 θ₁ (도)", value=45.0,
                                     step=0.1, format="%.1f", key="angle1")
        else:
            angle1 = 45.0

        axis2 = st.selectbox("두 번째 대칭축", ["x축", "y축", "직선y=ax"], key="axis2")
        if axis2 == "직선y=ax":
            angle2 = st.number_input("x축과 이루는 각도 θ₂ (도)", value=-45.0,
                                     step=0.1, format="%.1f", key="angle2")
        else:
            angle2 = -45.0

        st.markdown("🔵 입력점 | 🟢 1차 대칭 | 🔴 최종 대칭 결과")
        st.markdown("🟣 축1 (보라색 선), 🟠 축2 (주황색 선)")

        # ✅ 관찰 보조 (렌즈)
        #
        # 점 하나만 보면 "회전"이라는 말이 은유로만 들린다. 점을 옮겨 가며
        # 자취를 남기면 입력 자취와 최종 자취가 나란히 놓이고, 그것이 문자
        # 그대로 회전임이 보인다.
        st.divider()
        trail_on = lab_ui.trail_controls("s2")

    with col2:
        # 행렬
        R1 = reflection_matrix(axis1, angle1 if angle1 is not None else 1.0)
        R2 = reflection_matrix(axis2, angle2 if angle2 is not None else 1.0)

        # 대칭 계산
        P0 = np.array([x0, y0])
        P1 = R1 @ P0
        P2 = R2 @ P1

        # 자취 적재. 값이 실제로 바뀌었을 때만 쌓인다(trail_push 가 처리).
        if trail_on:
            lab_ui.trail_push("s2", ((P0[0], P0[1]), (P2[0], P2[1])), limit=150)

        # 그래프 생성
        fig = go.Figure()
        fig.update_layout(title="좌표 평면")

        # 지난 자취를 먼저 깔아 현재 점이 위에 오게 한다.
        if trail_on:
            history = lab_ui.trail_items("s2")
            if history:
                for trace in lab_ui.trail_traces([[p0] for p0, _ in history],
                                                 lab_ui.BEFORE, "입력점 자취"):
                    fig.add_trace(trace)
                for trace in lab_ui.trail_traces([[p2] for _, p2 in history],
                                                 lab_ui.AFTER, "최종점 자취"):
                    fig.add_trace(trace)

        # 🎯 대칭축 시각화 함수
        def draw_axis(fig, axis, angle, name, color):
            
                   
            # ── 2) 통일된 axis_norm으로 분기 ──
            if axis == "x축":
                fig.add_trace(go.Scatter(
                    x=[-5, 5], y=[0, 0], mode='lines',
                    line=dict(color=color, width=2), name=name
                ))
            elif axis == "y축":
                fig.add_trace(go.Scatter(
                    x=[0, 0], y=[-5, 5], mode='lines',
                    line=dict(color=color, width=2), name=name
                ))
            else:
                if angle is None or not np.isfinite(angle):   # 🔧 추가된 보호 조건
                    angle = 45.0
 
                theta = np.radians(angle)
                a = np.tan(theta)




                # ── 기울기에 따라 화면 안에 들어오도록 축 범위 조정 ──
                if abs(a) <= 1:
                    x1=-5; x2=5
                    y1=a*x1; y2=a*x2
                    
                else:
                    y1=-5; y2=5
                    x1=y1/a; x2=y2/a
                                           
                fig.add_trace(go.Scatter(
                    x=[x1, x2], y=[y1, y2], mode='lines',
                    line=dict(color=color, width=2), name=name
                ))

               


        # 점 시각화
        fig.add_trace(go.Scatter(x=[P0[0]], y=[P0[1]], mode='markers',
                                 marker=dict(color='blue', size=10), name='입력점'))
        fig.add_trace(go.Scatter(x=[P1[0]], y=[P1[1]], mode='markers',
                                 marker=dict(color='green', size=10), name='1차 대칭'))
        fig.add_trace(go.Scatter(x=[P2[0]], y=[P2[1]], mode='markers',
                                 marker=dict(color='red', size=10), name='최종 결과'))

        draw_axis(fig, axis1, angle1, "🟣 축1", "purple")
            
        draw_axis(fig, axis2, angle2, "🟠 축2", "orange")

        # ✅ 원점과 입력점, 최종점 연결선 추가
        fig.add_trace(go.Scatter(
            x=[0, P0[0]], y=[0, P0[1]], mode='lines',
            line=dict(color='blue', width=2, dash='dot'),
            name='입력점→원점'
        ))

        fig.add_trace(go.Scatter(
            x=[0, P2[0]], y=[0, P2[1]], mode='lines',
            line=dict(color='red', width=2, dash='dot'),
            name='최종점→원점'
        ))


        equal_axes(fig, view_key="s2", x_range=[-VIEW, VIEW], y_range=[-VIEW, VIEW])

        # 그래프를 클릭하면 그 자리로 초기점을 옮긴다.
        #
        # 예전에는 `streamlit-plotly-events`(iframe 컴포넌트) + 명시적
        # `st.rerun()` 이라 재실행이 두 번 돌고 iframe 이 매번 다시 로드돼
        # 눈에 띄게 깜빡였다. Streamlit 이 기본으로 제공하는 선택 이벤트를
        # 쓰면 재실행이 한 번이고 의존성도 하나 준다.
        event = chart(fig, key="s2_chart", on_select="rerun", selection_mode="points")
        points = event.selection["points"] if event and event.selection else []
        if points:
            # 위젯 키를 직접 갱신해야 숫자 입력칸에도 반영된다.
            new_x = float(np.clip(points[0]["x"], -VIEW, VIEW))
            new_y = float(np.clip(points[0]["y"], -VIEW, VIEW))
            if (new_x, new_y) != (x0, y0):
                st.session_state["input_x"] = new_x
                st.session_state["input_y"] = new_y
                st.rerun()

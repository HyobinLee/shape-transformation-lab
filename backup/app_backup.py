import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import plotly.graph_objects as go


# ✅ 페이지 설정
st.set_page_config(page_title="도형 변환 실험실", layout="wide")
st.title("🔄 도형 변환 실험실")

# ✅ 한글 폰트 설정
font_path = './fonts/나눔 글꼴/나눔고딕/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf'
font_prop = None
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False

# ✅ 숫자 포맷 함수
def format_number(n):
    return f"{n:.1f}".rstrip('0').rstrip('.') if n % 1 != 0 else str(int(n))

# ✅ Plotly 버전 시각화 함수
def plot_shape(shape_type, shape, transformed, matrix, font_family, a=1, b=1, c=0):
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

    # 직선일 경우 변환된 점 하나 강조
    if shape_type == "직선":
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
    if shape_type == "직선":
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
    fig.update_layout(
        width=600,
        height=600,
        xaxis=dict(
            range=[x_center - half_range, x_center + half_range],
            zeroline=True,
            zerolinecolor='gray',
            showgrid=True,  # ✅ 보조선 추가
            gridcolor='lightgray',  # ✅ 선 색상 설정 (선택)
        ),
        yaxis=dict(
            range=[y_center - half_range, y_center + half_range],
            zeroline=True,
            zerolinecolor='gray',
            scaleanchor='x'   # ✅ 이 위치가 맞습니다!
        ),
        font=dict(family=font_family),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    return fig

# ✅ 사이드바 메뉴
menu = st.sidebar.radio("📂 메뉴를 선택하세요", [
    "행렬을 통한 일차변환",
    "행렬을 통한 두 번의 대칭이동",
    "복소평면에서의 변환",
    "복소평면에서 평행이동과 회전이동의 결합"
])

##################### (1) ########################
# ✅ 메뉴별 콘텐츠
if menu == "행렬을 통한 일차변환":
    st.subheader("🏠 행렬을 통한 일차변환")
    st.write("여러 도형을 여러 행렬로 일차변환을 실행해 보세요.")

    col1, spacer, col2 = st.columns([1.4, 0.3, 2])  # 좌:입력 / 우:출력

    with col1:
        st.subheader("도형 입력")
        shape_type = st.selectbox("도형 종류를 선택하세요", ["삼각형", "사각형", "원", "직선"])

        if shape_type == "삼각형":
            A = np.array(st.text_input("점 A 좌표 (예: 1,1)", "1,1").split(','), dtype=float)
            B = np.array(st.text_input("점 B 좌표 (예: 1,2)", "1,2").split(','), dtype=float)
            C = np.array(st.text_input("점 C 좌표 (예: 2,1)", "2,1").split(','), dtype=float)
            shape = np.array([A, B, C, A])
        elif shape_type == "사각형":
            A = np.array(st.text_input("점 A 좌표 (예: 1,1)", "1,1").split(','), dtype=float)
            B = np.array(st.text_input("점 B 좌표 (예: 1,2)", "1,2").split(','), dtype=float)
            C = np.array(st.text_input("점 C 좌표 (예: 2,2)", "2,2").split(','), dtype=float)
            D = np.array(st.text_input("점 D 좌표 (예: 2,1)", "2,1").split(','), dtype=float)
            shape = np.array([A, B, C, D, A])
        elif shape_type == "원":
            center = np.array(st.text_input("원 중심 좌표 (예: 1,1)", "1,1").split(','), dtype=float)
            radius = st.number_input("반지름", value=2.0, step=0.1, format="%.1f")
            theta = np.linspace(0, 2*np.pi, 200)
            shape = np.stack([center[0] + radius * np.cos(theta),
                              center[1] + radius * np.sin(theta)], axis=1)
        elif shape_type == "직선":
            st.markdown("직선의 형태: $ax + by = c$")
            a = st.number_input("계수 a", value=1.0, step=0.1, format="%.1f")
            b = st.number_input("계수 b", value=1.0, step=0.1, format="%.1f")
            c = st.number_input("상수 c", value=2.0, step=0.1, format="%.1f")
            x_vals = np.linspace(-20, 20, 400)
            if b != 0:
                y_vals = (c - a * x_vals) / b
            else:
                x_vals = np.full(400, c / a)
                y_vals = np.linspace(-5, 5, 400)
            shape = np.stack([x_vals, y_vals], axis=1)

        st.subheader("2×2 변환 행렬 입력")
        a11 = st.number_input("a11", value=1.0, step=0.1, format="%.1f")
        a12 = st.number_input("a12", value=-1.0, step=0.1, format="%.1f")
        a21 = st.number_input("a21", value=1.0, step=0.1, format="%.1f")
        a22 = st.number_input("a22", value=2.0, step=0.1, format="%.1f")
        matrix = np.array([[a11, a12], [a21, a22]])

    with col2:
        # 직선 관련 계수 기본값 선언 (에러 방지용)
        if shape_type == "직선":
            a, b, c = a, b, c
        else:
            a, b, c = 1, 1, 1

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

        fig = plot_shape(shape_type, shape, transformed, matrix, 'NanumGothic', a, b, c)
        st.plotly_chart(fig, use_container_width=True)









################ (2) ####################
elif menu == "행렬을 통한 두 번의 대칭이동":
    st.subheader("🔁 행렬을 통한 두 번의 대칭이동")
    st.write("두 번의 대칭을 조합한 결과는 반드시 회전 변환이 될까요?")






################## (3) #####################
elif menu == "복소평면에서의 변환":
    st.subheader("🔷 복소평면에서의 변환")
    st.write("복소수를 이용한 여러 변환을 실험할 수 있습니다.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.title("🔁 복소평면에서의 변환 실험")
        st.markdown("복소수 $z = x + iy$ 로 정의된 도형을 복소함수 $w = f(z)$ 를 통해 변환해 보세요.")

        # ✅ 도형 정의식 입력
        st.subheader("1️⃣ z의 자취 : x, y의 관계식")
        st.caption("!!Warning!! 곱은 *로, 제곱은 **로, 등호는 ==로 표기하세요.(파이썬표기법)")
        definition = st.text_input("예: 2*y == x**2 + 1", value="x**2 + y**2 == 1", key="definition_input")

        # ✅ 복소함수 입력
        st.subheader("2️⃣ 복소함수식 입력 : w = f(z)")
        st.caption("!!Warning!! 허수 i는 1j로 표기하세요.(파이썬 표기법)")
        fz_input = st.text_input("w =", value="(z - 1j)**2", key="function_input")

    # ✅ 자동 정의역 추정 및 마스킹
    Z_selected = None
    final_range = None
    max_attempts = 10
    # 원본 정의식 보존
    original = definition
    for attempt in range(max_attempts):
        range_size = 8 + attempt * 2
        N = 800
        x = np.linspace(-range_size, range_size, N)
        y = np.linspace(-range_size, range_size, N)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        eps = (2 * range_size) / (N - 1)
        eps *= 2  # 허용오차 배율 조정 (라인도 두께 보장)
        try:
            local_vars = {"x": X, "y": Y, "np": np, "i": 1j}
            # 등식 비교일 때 동적 eps 사용
            if "==" in original:
                left, right = original.split("==")
                L = eval(left, local_vars)
                R = eval(right, local_vars)
                mask = np.abs(L - R) < eps
            else:
                mask = eval(original, local_vars)
            mask = np.array(mask, dtype=bool)
            if mask.sum() > 0:
                Z_selected = Z[mask]
                final_range = range_size
                break
        except Exception:
            continue

    if Z_selected is None or Z_selected.size == 0:
        st.error("오류 : 식을 다시 확인해 주세요.")
    else:
        # ✅ 복소함수 적용
        try:
            W = eval(fz_input, {"z": Z_selected, "np": np})
        except Exception as e:
            st.error(f"복소함수 적용 오류: {e}")
            W = None

        # ✅ 시각화
        with col2:
            if W is not None and getattr(W, 'size', 0) > 0:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=Z_selected.real, y=Z_selected.imag,
                    mode='markers', marker=dict(size=4, color='blue'),
                    name='변환 전 도형 z'
                ))
                fig.add_trace(go.Scatter(
                    x=W.real, y=W.imag,
                    mode='markers', marker=dict(size=4, color='red'),
                    name='변환 후 도형 w'
                ))

                # 축 및 그리드, 스케일 동기화
                all_re = np.concatenate([Z_selected.real, W.real])
                all_im = np.concatenate([Z_selected.imag, W.imag])
                x_min, x_max = all_re.min(), all_re.max()
                y_min, y_max = all_im.min(), all_im.max()
                margin = max(x_max - x_min, y_max - y_min) * 0.1
                fig.update_xaxes(
                    range=[x_min - margin, x_max + margin],
                    zeroline=True, zerolinecolor='gray',
                    showgrid=True, gridcolor='lightgray',
                    scaleanchor='y', scaleratio=1
                )
                fig.update_yaxes(
                    range=[y_min - margin, y_max + margin],
                    zeroline=True, zerolinecolor='gray',
                    showgrid=True, gridcolor='lightgray',
                    scaleanchor='x', scaleratio=1
                )
                fig.update_layout(
                    title='복소함수를 통한 도형 변환',
                    xaxis_title='Re', yaxis_title='Im',
                    width=600, height=600, showlegend=True
                )
                st.plotly_chart(fig)
            else:
                st.warning("복소함수 적용 결과가 없습니다.")



















############### (4) ##################
elif menu == "복소평면에서 평행이동과 회전이동의 결합":
    st.subheader("🌀 평행이동 + 회전이동")
    st.write("복소수의 덧셈과 곱셈을 통해 평행이동과 회전을 결합한 변환은 여전히 회전이동이 될까요?")










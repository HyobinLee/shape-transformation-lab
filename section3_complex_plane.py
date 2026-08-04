import streamlit as st
import numpy as np
import plotly.graph_objects as go

from expression_parser import ExpressionError, compile_complex_function, compile_locus
from lab_ui import chart, equal_axes


@st.cache_resource(max_entries=64)
def _locus(text):
    """자취 정의식을 컴파일한다. 같은 식을 다시 타이핑해도 다시 컴파일하지 않는다.

    반환값이 클로저라 `cache_data` 로는 다룰 수 없어 `cache_resource` 를 쓴다.
    학생이 식을 계속 고쳐 넣으므로 `max_entries` 로 상한을 둔다.
    """
    return compile_locus(text)


@st.cache_resource(max_entries=64)
def _complex_function(text):
    return compile_complex_function(text)


def run_complex_plane():
    st.header("🟦 (3) 복소평면에서의 이동 시뮬레이터")
    st.markdown("복소수 $z = x + iy$ 로 정의된 도형을 복소함수 $w = f(z)$ 를 통해 변환해 보세요.")

    ################## (3) #####################

    #st.subheader("🔷 복소평면에서의 변환")
    #st.write("복소수를 이용한 여러 변환을 실험할 수 있습니다.")

    col1, col2 = st.columns([1, 1])

    with col1:
        #st.title("🔁 복소평면에서의 변환 실험")
        #st.markdown("복소수 $z = x + iy$ 로 정의된 도형을 복소함수 $w = f(z)$ 를 통해 변환해 보세요.")

        # ✅ 도형 정의식 입력
        st.subheader("# z의 자취 : x, y의 관계식 ___________________")
        st.markdown('<span style="color: purple;">⚠️ 곱은 *로, 제곱은 **로, 등호는 ==로 표기하세요.(파이썬 표기법)</span>', unsafe_allow_html=True)
        st.caption("부등식(`y > x**2`)과 조건 결합(`&`, `|`)도 쓸 수 있고, "
                   "sin·cos·exp·log·sqrt·abs 함수를 쓸 수 있습니다.")
        definition = st.text_input("예: 2*y == x**2 + 1", value="x**2 + y**2 == 1", key="definition_input")

        # ✅ 복소함수 입력
        st.subheader("# 복소함수식 입력 : w = f(z) ___________________")
        st.markdown('<span style="color: purple;">⚠️ 허수 i는 1j로 표기하세요.(파이썬 표기법)</span>', unsafe_allow_html=True)
        st.caption("`i` 로 써도 됩니다. conj(z)(켤레복소수), abs(z), re(z), im(z), arg(z) 도 쓸 수 있습니다.")
        fz_input = st.text_input("w =", value="(z - 1j)**2", key="function_input")

    # ✅ 수식 컴파일 (eval 을 쓰지 않는 안전한 파서)
    #    범위를 넓혀 가며 여러 번 평가하므로, 컴파일은 반복 밖에서 한 번만 한다.
    locus_mask = None
    apply_fz = None
    try:
        locus_mask = _locus(definition)
    except ExpressionError as e:
        st.error(f"자취 정의식 오류 : {e}")
    try:
        apply_fz = _complex_function(fz_input)
    except ExpressionError as e:
        st.error(f"복소함수식 오류 : {e}")

    # ✅ 자동 정의역 추정 및 마스킹
    #
    # 격자는 400×400 으로 시작한다. 대개 첫 시도에 자취를 잡으므로, 처음부터
    # 800×800 을 돌리면 매 재실행마다 학생을 기다리게 만든다. 못 잡았을 때만
    # 범위를 넓히고 마지막 두 번은 격자도 촘촘히 한다.
    Z_selected = None
    final_range = None
    max_attempts = 10
    if locus_mask is not None:
        for attempt in range(max_attempts):
            range_size = 8 + attempt * 2
            N = 400 if attempt < max_attempts - 2 else 800
            x = np.linspace(-range_size, range_size, N)
            y = np.linspace(-range_size, range_size, N)
            X, Y = np.meshgrid(x, y)
            Z = X + 1j * Y

            eps = (2 * range_size) / (N - 1)
            eps *= 2  # 허용오차 배율 조정 (라인도 두께 보장)
            try:
                mask = locus_mask(X, Y, eps)
            except ExpressionError as e:
                st.error(f"자취 정의식 오류 : {e}")
                break
            except Exception:
                continue
            if mask.sum() > 0:
                Z_selected = Z[mask]
                final_range = range_size
                break

    if locus_mask is None or apply_fz is None:
        pass  # 위에서 이미 원인을 안내했다.
    elif Z_selected is None or Z_selected.size == 0:
        st.error("오류 : 식을 만족하는 점을 찾지 못했습니다. 식을 다시 확인해 주세요.")
    else:
        # ✅ 복소함수 적용
        try:
            W = apply_fz(Z_selected)
        except ExpressionError as e:
            st.error(f"복소함수식 오류 : {e}")
            W = None
        except Exception as e:
            st.error(f"복소함수 적용 오류: {e}")
            W = None

        # ✅ 시각화
        with col2:
            if W is not None and getattr(W, 'size', 0) > 0:
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
                # 식이 바뀌면 시야를 되돌리고, 그렇지 않으면 확대해 둔 채로 둔다.
                equal_axes(
                    fig, view_key=f"s3-{definition}-{fz_input}",
                    x_range=[x_min - margin, x_max + margin],
                    y_range=[y_min - margin, y_max + margin],
                )
                fig.update_layout(
                    title='복소함수를 통한 도형 변환',
                    xaxis_title='Re', yaxis_title='Im',
                    showlegend=True,
                )
                chart(fig, key="s3_chart")
                st.caption(
                    f"자동으로 잡은 정의역: 실수부·허수부 모두 "
                    f"[−{final_range}, {final_range}] 안에서 찾았습니다."
                )
            else:
                st.warning("복소함수 적용 결과가 없습니다.")






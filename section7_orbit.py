"""(7) 되풀이하면 어디로 가는가 — 반복 사상과 궤도.

관찰시키려는 명제:

    같은 변환을 계속 적용하면 궤도가 생긴다. 궤도는 **고정점**을 중심으로
    돌거나, 빨려들거나, 튕겨 나간다. 운명을 가르는 것은 **배율의 크기가
    1보다 큰가 작은가** 하나뿐이다.

§4 는 "회전의 기준점은?" 이라 묻고 답을 주지 않는다. 이 섹션은 답을
알려주는 대신 **눈에 보이게** 만든다 — 배율을 1보다 작게 하면 궤도가 나선을
그리며 한 점으로 빨려들고, 그 점이 곧 고정점이다.

식은 §4 와 같고 배율 r 하나만 더 붙는다:

    w = r·e^{iθ}(z + α) + β
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import lab_ui
from lab_ui import chart

#: 궤도가 화면 밖으로 완전히 달아났다고 보는 크기. inf/nan 이 Plotly 로
#: 흘러가면 축이 통째로 깨지므로 그 전에 끊는다.
ESCAPE = 1e6

PRESETS = {
    "정다각형": dict(r=1.0, theta=90.0, ax=0.0, ay=0.0, bx=2.0, by=0.0),
    "닫히지 않는 궤도": dict(r=1.0, theta=360 / np.sqrt(2), ax=0.0, ay=0.0, bx=2.0, by=0.0),
    "고정점으로 수렴": dict(r=0.9, theta=35.0, ax=1.0, ay=0.0, bx=0.0, by=-1.0),
    "평행이동 (중심 없음)": dict(r=1.0, theta=0.0, ax=0.6, ay=0.4, bx=0.0, by=0.0),
    "발산": dict(r=1.12, theta=28.0, ax=0.5, ay=0.0, bx=0.0, by=0.0),
}


# ── 수학 ─────────────────────────────────────────────────────────────────────
#
# 고정점 공식은 §4 에도 같은 것이 있다. 합쳐 두지 않는 것은 의도된 선택이다 —
# 각 섹션 파일이 하나의 독립된 수업 자료로 읽히는 편이, 공통 모듈을 오가며
# 읽어야 하는 코드보다 이 프로젝트의 목적에 맞기 때문이다.

def fixed_point(rho, alpha, beta):
    """w = ρ(z+α)+β 를 제자리에 두는 점.

    z = ρ(z+α)+β 를 풀면 z*(1−ρ) = ρα+β 이므로 z* = (ρα+β)/(1−ρ).

    Returns:
        고정점, 또는 **ρ = 1 이면 None** — 순수 평행이동에는 중심이 없다.
        그것이 이 섹션에서 만나야 할 퇴화 경우다.
    """
    if abs(1 - rho) < 1e-12:
        return None
    return (rho * alpha + beta) / (1 - rho)


def orbit(z0, rho, alpha, beta, steps):
    """z₀ 에서 시작해 같은 변환을 `steps` 번 되풀이한 궤도.

    발산하면 거기서 멈춘다. 끝까지 계산해 봐야 inf 만 쌓이고, 그 inf 가
    그림의 축 범위를 통째로 망가뜨린다.

    Returns:
        ``(궤도 (m, ) 복소배열, 발산했는가)``
    """
    points = [complex(z0)]
    for _ in range(steps):
        nxt = rho * (points[-1] + alpha) + beta
        if not np.isfinite(nxt.real) or not np.isfinite(nxt.imag) or abs(nxt) > ESCAPE:
            return np.array(points), True
        points.append(nxt)
    return np.array(points), False


def julia_escape(c, span, size, max_iter=60):
    """f(z) = z² + c 의 탈출시간. 발산하지 않는 점들의 경계가 줄리아 집합이다.

    numpy 벡터 연산이라 400×400×60 이 눈 깜짝할 사이에 끝난다.
    """
    axis = np.linspace(-span, span, size)
    Z = axis[None, :] + 1j * axis[:, None]
    escaped = np.zeros(Z.shape, dtype=int)
    alive = np.ones(Z.shape, dtype=bool)
    for k in range(max_iter):
        Z[alive] = Z[alive] ** 2 + c
        just_escaped = alive & (np.abs(Z) > 2)
        escaped[just_escaped] = k
        alive &= ~just_escaped
        Z[~alive] = 0          # 더 키우지 않는다 (overflow 경고 방지)
    escaped[alive] = max_iter
    return axis, escaped


# ── 화면 ─────────────────────────────────────────────────────────────────────

def _orbit_figure(path, star, span, max_steps):
    """궤도를 프레임으로 그린다 — n 슬라이더가 브라우저 안에서만 돈다."""
    fig = go.Figure()

    if star is not None:
        fig.add_trace(go.Scatter(
            x=[star.real], y=[star.imag], mode='markers',
            marker=dict(size=13, color='mediumpurple', symbol='star'),
            name='고정점 z*'))

    # 색을 시작(파랑) → 끝(빨강) 으로 흘려 궤도의 방향이 보이게 한다.
    t = np.linspace(0, 1, len(path))
    moving = len(fig.data)
    fig.add_trace(go.Scatter(
        x=path.real, y=path.imag, mode='lines+markers',
        line=dict(color='lightgray', width=1),
        marker=dict(size=7, color=t, colorscale='turbo', cmin=0, cmax=1,
                    showscale=False),
        name='궤도'))
    fig.add_trace(go.Scatter(
        x=[path.real[0]], y=[path.imag[0]], mode='markers',
        marker=dict(size=12, color=lab_ui.BEFORE, symbol='circle'),
        name='시작점 z₀'))

    frames = []
    for n in range(1, len(path) + 1):
        frames.append(go.Frame(
            name=f"{n}",
            data=[go.Scatter(x=path.real[:n], y=path.imag[:n],
                             marker=dict(size=7, color=t[:n], colorscale='turbo',
                                         cmin=0, cmax=1, showscale=False))],
            traces=[moving]))
    fig.frames = frames

    lab_ui.equal_axes(fig, view_key="s7", x_range=[-span, span], y_range=[-span, span])
    fig.update_layout(
        margin=dict(l=10, r=10, t=64, b=10),
        legend=dict(orientation='h', y=-0.05),
        updatemenus=[dict(
            type='buttons', direction='left', x=0.0, y=1.11, xanchor='left',
            buttons=[dict(label='▶ 반복', method='animate',
                          args=[None, dict(frame=dict(duration=70, redraw=False),
                                           fromcurrent=True,
                                           transition=dict(duration=0))]),
                     dict(label='⏸', method='animate',
                          args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             mode='immediate')])])],
        sliders=[dict(
            active=len(frames) - 1, x=0.14, len=0.84, y=1.07, xanchor='left',
            currentvalue=dict(prefix='n = ', font=dict(size=13)),
            steps=[dict(method='animate', label=f"{n}",
                        args=[[f"{n}"], dict(mode='immediate',
                                             frame=dict(duration=0, redraw=False),
                                             transition=dict(duration=0))])
                   for n in range(1, len(path) + 1)])],
    )
    return fig


def run_orbit():
    st.header("🟧 (7) 되풀이하면 어디로 가는가")
    st.markdown("**같은 변환을 계속 적용하면 점은 어디로 갈까요? 무엇이 그 운명을 가를까요?**")
    st.latex(r"w = r(\cos\theta + i\sin\theta)(z + \alpha) + \beta")

    with st.expander("🔍 관찰 과제", expanded=False):
        st.markdown(
            "1. r = 1 로 두고 θ 를 90°, 60°, 72° 로 바꿔 보세요. 궤도가 **닫히는** 이유가 뭘까요?\n"
            "2. θ 를 254.56°(= 360/√2)로 해 보세요. 왜 아무리 반복해도 안 닫힐까요?\n"
            "3. r 을 1보다 **작게** 해 보세요. 궤도가 한 점으로 빨려듭니다 — **그 점이 어디인가요?**\n"
            "4. θ = 0 으로 두면 그 점이 사라집니다. 왜일까요?"
        )

    defaults = dict(r=0.9, theta=35.0, ax=1.0, ay=0.0, bx=0.0, by=-1.0)
    for key, value in defaults.items():
        st.session_state.setdefault(f"s7_{key}", value)

    col1, col2 = st.columns([1, 1.5])

    with col1:
        preset = st.pills("프리셋", list(PRESETS), key="s7_preset")
        if preset and st.session_state.get("s7_applied") != preset:
            for key, value in PRESETS[preset].items():
                st.session_state[f"s7_{key}"] = float(value)
            st.session_state["s7_applied"] = preset
            st.rerun()

        st.subheader("변환")
        pair = st.columns(2)
        r = pair[0].number_input("배율 r", min_value=0.0, max_value=3.0,
                                 step=0.05, format="%.3f", key="s7_r")
        theta_deg = pair[1].number_input("회전각 θ (도)", step=5.0, format="%.2f",
                                         key="s7_theta")
        pair = st.columns(2)
        alpha_re = pair[0].number_input("Re(α)", step=0.5, format="%.2f", key="s7_ax")
        alpha_im = pair[1].number_input("Im(α)", step=0.5, format="%.2f", key="s7_ay")
        pair = st.columns(2)
        beta_re = pair[0].number_input("Re(β)", step=0.5, format="%.2f", key="s7_bx")
        beta_im = pair[1].number_input("Im(β)", step=0.5, format="%.2f", key="s7_by")

        steps = st.slider("반복 횟수 n", min_value=1, max_value=200, value=60, key="s7_n")

        rho = r * np.exp(1j * np.radians(theta_deg))
        alpha = complex(alpha_re, alpha_im)
        beta = complex(beta_re, beta_im)
        star = fixed_point(rho, alpha, beta)

        st.divider()
        st.subheader("계기판")
        st.markdown(f"- 배율 $|\\rho|$ = **{abs(rho):.4f}**")
        if star is not None:
            st.markdown(
                "- 고정점에서 잰 거리의 비 "
                f"$\\dfrac{{|z_{{n+1}}-z^*|}}{{|z_n-z^*|}}$ = **{abs(rho):.4f}**"
            )
            st.caption("n 을 아무리 밀어도 이 비가 변하지 않습니다. 왜 그럴까요?")
        else:
            st.caption("고정점이 없어 잴 기준이 없습니다.")

        reveal = st.toggle("🔎 고정점 밝히기", value=False, key="s7_reveal")
        if reveal:
            if star is None:
                st.info("**고정점이 없습니다** — ρ = 1 이면 순수 평행이동이고, "
                        "평행이동에는 중심이 없습니다.")
            else:
                st.latex(r"z^*=\frac{\rho\alpha+\beta}{1-\rho}="
                         rf"{star.real:.3f}{star.imag:+.3f}i")
                st.latex(r"z_n - z^* = \rho^n\,(z_0 - z^*)")
                st.caption("고정점에서 재면 이 변환은 그냥 곱하기 하나입니다.")

    with col2:
        st.session_state.setdefault("s7_z0", (2.0, 1.0))
        z0 = complex(*st.session_state["s7_z0"])
        path, diverged = orbit(z0, rho, alpha, beta, steps)

        # 축 범위는 **처음 몇 점**으로 잡는다. 궤도 전체로 잡으면 발산할 때
        # 시작 부분이 점 하나로 뭉개진다.
        head = path[:min(len(path), 12)]
        reference = np.concatenate([head, [star] if star is not None else []])
        span = float(max(3.0, np.abs(reference).max() * 1.6))
        span = min(span, 40.0)

        event = chart(_orbit_figure(path, star, span, steps), key="s7_chart",
                      on_select="rerun", selection_mode="points")
        picked = event.selection["points"] if event and event.selection else []
        if picked:
            new = (float(picked[0]["x"]), float(picked[0]["y"]))
            if new != st.session_state["s7_z0"]:
                st.session_state["s7_z0"] = new
                st.rerun()

        if diverged:
            st.warning(f"궤도가 {len(path) - 1}번 만에 화면 밖으로 달아났습니다. "
                       "$|\\rho| > 1$ 이면 언제나 이렇게 됩니다.")
        st.caption("그래프를 클릭해 시작점을 옮겨 보세요. 색은 시작(파랑)에서 끝(빨강)으로 흐릅니다.")

    with st.expander("✅ 확인", expanded=False):
        st.markdown(
            "- $|\\rho| < 1$ → 나선을 그리며 고정점으로 **수렴**합니다. "
            "§4 가 물었던 '회전의 기준점'이 눈으로 드러나는 순간입니다.\n"
            "- $|\\rho| = 1$ → 고정점 둘레를 **맴돕니다.** θ 가 360°의 유리수 배면 궤도가 닫혀 "
            "정다각형이 되고, 무리수 배면 영원히 닫히지 않고 원을 촘촘히 메웁니다.\n"
            "- $|\\rho| > 1$ → **발산**합니다.\n"
            "- θ = 0 → ρ = 1 이라 분모가 0. 고정점이 없습니다 = 순수 평행이동.\n\n"
            "그리고 이 모든 것이 한 줄로 설명됩니다 — "
            "$z_n - z^* = \\rho^n (z_0 - z^*)$. **고정점에서 재면 그냥 곱하기 하나입니다.**"
        )

    with st.expander("🧭 더 나아가기 — 되풀이를 z² + c 로 바꾸면", expanded=False):
        st.markdown(
            "지금까지는 **일차** 변환을 되풀이했습니다. $f(z) = z^2 + c$ 로 바꾸고 "
            "'발산하지 않는 시작점들'을 모아 보면 어떤 모양이 나올까요?"
        )
        st.caption("교과 범위 밖입니다. 궁금하면 켜 보세요.")
        if st.toggle("줄리아 집합 그리기", value=False, key="s7_julia"):
            pair = st.columns(2)
            c_re = pair[0].slider("Re(c)", -1.5, 1.0, -0.4, 0.01, key="s7_cre")
            c_im = pair[1].slider("Im(c)", -1.2, 1.2, 0.6, 0.01, key="s7_cim")
            axis, escaped = julia_escape(complex(c_re, c_im), span=1.7, size=400)
            julia = go.Figure(go.Heatmap(x=axis, y=axis, z=escaped,
                                         colorscale='turbo', showscale=False,
                                         hoverinfo='skip'))
            lab_ui.equal_axes(julia, view_key="s7-julia",
                              x_range=[-1.7, 1.7], y_range=[-1.7, 1.7], size=520)
            julia.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            chart(julia, key="s7_julia_chart")
            st.caption("색 = 발산하기까지 걸린 반복 횟수. 검은 부분이 끝내 발산하지 않는 점들입니다.")

import streamlit as st
import numpy as np
import plotly.graph_objs as go

import lab_ui
from lab_ui import chart, equal_axes

#: 복소평면에서 보여 줄 범위.
VIEW = 8.0


def fixed_point(rho, alpha, beta):
    """w = ρ(z+α)+β 를 제자리에 두는 점 — 이 섹션의 헤더가 묻는 '회전의 기준점'.

    z = ρ(z+α)+β 를 z 에 대해 풀면 z*(1−ρ) = ρα+β 이므로 z* = (ρα+β)/(1−ρ).

    ρ = 1(θ = 0)이면 분모가 0이 된다. 예외를 던지지 않고 None 을 돌려주며,
    그 경우가 곧 **"평행이동에는 중심이 없다"** 는 답이다.

    (섹션 7 에도 같은 함수가 있다. 각 섹션이 하나의 독립된 수업 자료로 읽히는
    편이 낫다는 이 프로젝트의 방침에 따른 의도된 중복이다.)
    """
    if abs(1 - rho) < 1e-12:
        return None
    return (rho * alpha + beta) / (1 - rho)


def run_rotation_translation():
    st.header("🟥 (4) 회전과 평행이동 시뮬레이터")
    st.markdown("평행이동&회전이동&평행이동은 회전이동일까요? 회전의 기준점은?")

    # ✅ 좌: 입력 / 우: 그래프
    left_col, right_col = st.columns([1, 1.5])

    with left_col:
        # 윗줄: alpha, theta, beta 입력
        st.subheader("🔧 α, θ, β 값을 정해 보세요.")
        st.latex(r"w = (\cos\theta + i\sin\theta)(z + \alpha) + \beta")

        # 위젯 기본값은 만들어지기 전에 세션에 심는다. 프리셋과 클릭이 같은
        # 키를 갱신해야 하기 때문이다.
        for key, value in (("alpha_re", 1.0), ("alpha_im", 0.0), ("theta_deg", 45.0),
                           ("beta_re", 0.0), ("beta_im", -1.0)):
            st.session_state.setdefault(key, value)

        preset = st.pills("프리셋", ["회전이 되는 경우", "θ = 0 (회전 아님)", "제자리로 돌아옴"],
                          key="s4_preset")
        presets = {
            "회전이 되는 경우": dict(alpha_re=1.0, alpha_im=0.0, theta_deg=45.0,
                              beta_re=0.0, beta_im=-1.0),
            # θ=0 이면 ρ=1 이라 분모가 0 — 중심이 사라진다. 이 섹션의 퇴화 경우.
            "θ = 0 (회전 아님)": dict(alpha_re=1.0, alpha_im=0.5, theta_deg=0.0,
                                 beta_re=0.5, beta_im=0.0),
            # ρα+β = 0 이면 고정점이 원점이다.
            "제자리로 돌아옴": dict(alpha_re=0.0, alpha_im=0.0, theta_deg=60.0,
                              beta_re=0.0, beta_im=0.0),
        }
        if preset and st.session_state.get("s4_applied") != preset:
            for key, value in presets[preset].items():
                st.session_state[key] = float(value)
            st.session_state["s4_applied"] = preset
            st.rerun()

        upper_col1, upper_col2, upper_col3 = st.columns(3)

        with upper_col1:
            st.markdown("**α (회전 이전 평행이동)**")
            alpha_re = st.number_input("Re(α)", step=0.5, format="%.2f", key="alpha_re")
            alpha_im = st.number_input("Im(α)", step=0.5, format="%.2f", key="alpha_im")

        with upper_col2:
            st.markdown("**θ (회전각, 도)**")
            theta_deg = st.number_input("회전각 θ", step=5.0, format="%.1f", key="theta_deg")
            theta_rad = np.radians(theta_deg)
            cos_theta = np.cos(theta_rad)
            sin_theta = np.sin(theta_rad)

        with upper_col3:
            st.markdown("**β (회전 이후 평행이동)**")
            beta_re = st.number_input("Re(β)", step=0.5, format="%.2f", key="beta_re")
            beta_im = st.number_input("Im(β)", step=0.5, format="%.2f", key="beta_im")

        st.divider()

        # 아랫줄: z 입력. 그래프를 클릭해도 같은 값이 바뀌므로 기본값은
        # 위젯이 만들어지기 전에 한 번만 심고 value= 는 넘기지 않는다.
        st.subheader("🖱 입력 복소수 z 와 변환 결과 w 시각화")
        st.markdown("**z = x + iy** — 그래프를 클릭해도 됩니다.")
        for key, value in (("z_x", 2.0), ("z_y", 1.0)):
            st.session_state.setdefault(key, value)
        x = st.number_input("x (실수 부분)", step=0.5, format="%.2f", key="z_x")
        y = st.number_input("y (허수 부분)", step=0.5, format="%.2f", key="z_y")

        st.divider()
        steps_on = st.toggle(
            "🟢 중간 단계 보기", value=True, key="s4_steps",
            help="z → z+α → 회전 → +β 를 차례로 보여 줍니다. "
                 "이 섹션의 주제가 '합성'이니, 중간이 보여야 합니다.",
        )
        trail_on = lab_ui.trail_controls("s4")

        # 헤더가 던진 질문에 학생이 스스로 도달하게 하는 순서:
        # ① 추측한다 → ② 앱이 그 점을 변환해 제자리인지 보여 준다 → ③ 확인한다.
        guess_on = st.toggle(
            "🎯 회전 중심 추측하기", value=False, key="s4_guess",
            help="추측한 점을 실제로 변환해서 제자리에 남는지 확인해 줍니다.",
        )
        reveal = st.toggle("🔎 정답 보기", value=False, key="s4_reveal")

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

        # ✅ 자취 — 입력을 옮겨 가며 남기면 두 자취가 나란히 놓이고,
        #    그 겹침에서 회전 중심의 위치가 눈으로 짐작된다.
        if trail_on:
            lab_ui.trail_push("s4", ((z.real, z.imag), (w.real, w.imag)), limit=150)
            history = lab_ui.trail_items("s4")
            if history:
                for trace in lab_ui.trail_traces([[p] for p, _ in history],
                                                 lab_ui.BEFORE, "입력 자취"):
                    fig.add_trace(trace)
                for trace in lab_ui.trail_traces([[q] for _, q in history],
                                                 lab_ui.AFTER, "결과 자취"):
                    fig.add_trace(trace)

        # ✅ 중간 단계 — 이 섹션의 주제는 '합성'인데, z 와 w 만 있으면
        #    합성이라는 사실 자체가 화면에 없다. 초록 = 중간 단계.
        if steps_on:
            after_shift = z + alpha
            after_turn = rotator * after_shift
            fig.add_trace(go.Scatter(
                x=[z.real, after_shift.real, after_turn.real, w.real],
                y=[z.imag, after_shift.imag, after_turn.imag, w.imag],
                mode='lines+markers',
                line=dict(color=lab_ui.MIDDLE, width=2, dash='dot'),
                marker=dict(size=9, color=lab_ui.MIDDLE, symbol='diamond'),
                name='① +α  ② 회전  ③ +β'))

        # ✅ 점들 추가
        fig.add_trace(go.Scatter(x=[z.real], y=[z.imag], mode='markers',
                                marker=dict(size=12, color=lab_ui.BEFORE), name='입력 z'))
        fig.add_trace(go.Scatter(x=[w.real], y=[w.imag], mode='markers',
                                marker=dict(size=12, color=lab_ui.AFTER,
                                            symbol='triangle-up'), name='변환 결과 w'))
        fig.add_trace(go.Scatter(x=[alpha.real], y=[alpha.imag], mode='markers',
                                marker=dict(size=10, color='purple', symbol='x'), name='첫번째 평행이동 α'))
        fig.add_trace(go.Scatter(x=[beta.real], y=[beta.imag], mode='markers',
                                marker=dict(size=10, color='orange', symbol='x'), name='두번째 평행이동 β'))

        # ✅ 회전 중심 — 추측 먼저, 정답은 나중에.
        star = fixed_point(rotator, alpha, beta)
        if guess_on:
            st.session_state.setdefault("s4_guess_point", (0.0, 0.0))
            guess = complex(*st.session_state["s4_guess_point"])
            moved = rotator * (guess + alpha) + beta
            fig.add_trace(go.Scatter(
                x=[guess.real, moved.real], y=[guess.imag, moved.imag],
                mode='lines+markers',
                line=dict(color='dimgray', width=2),
                marker=dict(size=11, color='dimgray', symbol='x'),
                name='내 추측 → 그 상'))
        if reveal and star is not None:
            fig.add_trace(go.Scatter(
                x=[star.real], y=[star.imag], mode='markers',
                marker=dict(size=15, color='mediumpurple', symbol='star'),
                name='회전 중심 z*'))

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

        event = chart(fig, key="s4_chart", on_select="rerun", selection_mode="points")
        picked = event.selection["points"] if event and event.selection else []
        if picked:
            # 추측 모드에서는 클릭이 '중심 후보'를, 아니면 입력점 z 를 옮긴다.
            new = (float(np.clip(picked[0]["x"], -VIEW, VIEW)),
                   float(np.clip(picked[0]["y"], -VIEW, VIEW)))
            target = "s4_guess_point" if guess_on else None
            if target:
                if new != st.session_state.get(target):
                    st.session_state[target] = new
                    st.rerun()
            elif new != (x, y):
                st.session_state["z_x"], st.session_state["z_y"] = new
                st.rerun()

        if guess_on:
            guess = complex(*st.session_state.get("s4_guess_point", (0.0, 0.0)))
            moved = rotator * (guess + alpha) + beta
            distance = abs(moved - guess)
            st.caption(
                f"추측한 점과 그 상 사이의 거리: **{distance:.4f}** — "
                "0에 가까울수록 제자리에 남는 점입니다. 그래프를 클릭해 옮겨 보세요."
            )
            if distance < 0.05:
                st.success("거의 제자리입니다. 이 점이 무엇일까요?")

        if reveal:
            if star is None:
                st.info("**회전 중심이 없습니다.** θ = 0 이면 회전이 아니라 순수 평행이동이고, "
                        "평행이동에는 중심이 없습니다.")
            else:
                st.latex(r"z^*=\frac{\rho\alpha+\beta}{1-\rho}="
                         rf"{star.real:.3f}{star.imag:+.3f}i")
                st.caption("평행이동→회전→평행이동의 합성은 결국 이 점을 중심으로 하는 "
                           "**한 번의 회전**이었습니다.")

        st.caption("🟢 초록 마름모 = 중간 단계(①+α ②회전 ③+β). "
                   "z 와 w 만 보면 '합성'이라는 이 섹션의 주제가 보이지 않습니다.")

"""(5) 일차변환의 고유공간 — 변환이 자기 자신으로 보내는 직선.

관찰시키려는 명제:

    일차변환에는 **자기 자신으로 옮겨지는 직선**이 있다. 그 위의 점은 아무리
    변환해도 그 직선을 벗어나지 못하고, 그 직선 위에서 변환은 단순한
    상수배일 뿐이다. 그런 직선은 두 개일 수도, 하나뿐일 수도, 아예 없을
    수도, 평면 전체일 수도 있다.

제목이 "고유벡터"가 아니라 **"고유공간"** 인 것이 이 섹션의 설계를 결정한다.
관찰 대상은 벡터 하나가 아니라 **직선(부분공간)** 이므로, 학생이 도달해야 할
문장은 "Av = λv 인 v 가 있다" 가 아니라 "**A 가 통째로 제자리에 두는 직선이
있다**" 이다. 그래서 뷰가 셋이다 — 찾고(A), 정말 못 벗어나는지 확인하고(B),
그 직선을 기준으로 세상을 다시 본다(C).
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import lab_ui
from lab_ui import chart

#: 훑기 프레임 수. 2° 간격이면 충분하고, 프레임은 브라우저 안에서만 도므로
#: Streamlit 재실행이 0회다.
FRAMES = 90

#: 퇴화까지 포함해 네 가지 경우를 전부 만나게 하는 프리셋.
PRESETS = {
    "두 직선": np.array([[1.0, 1.0], [0.0, 2.0]]),
    "결손(전단)": np.array([[2.0, 1.0], [0.0, 2.0]]),
    "평면 전체": np.array([[2.0, 0.0], [0.0, 2.0]]),
    "고유공간 없음": np.array([[0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]]),
    "λ=0 포함": np.array([[1.0, 1.0], [1.0, 1.0]]),
}


# ── 수학 ─────────────────────────────────────────────────────────────────────

def deviation_curve(A, theta):
    """어긋난 각 g(θ) — v(θ) 가 만드는 **직선**과 Av(θ) 가 만드는 **직선**의 각.

    부호 있는 각을 ±90° 로 접는다. 고유값이 음수면 점은 원점 건너편으로
    튀지만 **직선은 그대로**이기 때문이다. 이 한 줄이 곧 "고유공간은 방향이
    아니라 직선이다" 라는 명제의 코드판이다.

    Av = 0 인 방향(λ=0)에서는 `atan2(0, 0) = 0` 이 되어 우연히 옳은 답을 낸다.
    핵도 λ=0 의 고유공간이므로 g = 0 이 맞다.

    Returns:
        라디안 배열. 값이 0인 곳이 고유직선.
    """
    v = np.column_stack([np.cos(theta), np.sin(theta)])
    Av = v @ np.asarray(A, dtype=float).T
    cross = v[:, 0] * Av[:, 1] - v[:, 1] * Av[:, 0]
    dot = np.einsum('ij,ij->i', v, Av)
    angle = np.arctan2(cross, dot)
    return (angle + np.pi / 2) % np.pi - np.pi / 2


def _direction_for(A, lam):
    """(A - λI)v = 0 을 만족하는 단위벡터. A = λI 면 방향이 하나로 안 정해져 None."""
    a, b, c, d = np.asarray(A, dtype=float).ravel()
    rows = [np.array([a - lam, b]), np.array([c, d - lam])]
    row = max(rows, key=np.linalg.norm)
    if np.linalg.norm(row) < 1e-12:
        return None
    v = np.array([-row[1], row[0]])
    return v / np.linalg.norm(v)


def eigen_structure(A, tol=1e-9):
    """2×2 행렬의 고유공간을 분류한다.

    "중근인가", "결손인가" 는 부동소수점에서 정확히 판정할 수 없다. 판별식이
    딱 0이 되는 일은 거의 없으므로 **행렬 크기에 견준 상대 기준**을 쓴다.

    Returns:
        dict — `kind` 는 'two_lines' | 'defective' | 'plane' | 'none',
        `dim` 은 고유공간의 차원(0·1·2), `directions` 는 고유직선의 방향
        단위벡터들, `eigenvalues` 는 실고유값(복소면 빈 리스트).
    """
    A = np.asarray(A, dtype=float)
    a, b, c, d = A.ravel()
    trace, det = a + d, a * d - b * c
    discriminant = (a - d) ** 2 + 4 * b * c
    scale = max(1.0, float(np.linalg.norm(A))) ** 2

    if discriminant > tol * scale:
        root = np.sqrt(discriminant)
        lams = [(trace + root) / 2, (trace - root) / 2]
        directions = [v for v in (_direction_for(A, lam) for lam in lams) if v is not None]
        return dict(kind='two_lines', dim=1, eigenvalues=lams, directions=directions,
                    trace=trace, det=det, discriminant=discriminant)

    if discriminant < -tol * scale:
        return dict(kind='none', dim=0, eigenvalues=[], directions=[],
                    trace=trace, det=det, discriminant=discriminant)

    # 중근. A 가 λI 면 모든 방향이 고유방향이고(평면 전체), 아니면 직선 하나뿐이다.
    lam = trace / 2
    if np.allclose(A, lam * np.eye(2), atol=1e-9 * scale):
        return dict(kind='plane', dim=2, eigenvalues=[lam, lam], directions=[],
                    trace=trace, det=det, discriminant=discriminant)
    direction = _direction_for(A, lam)
    return dict(kind='defective', dim=1, eigenvalues=[lam, lam],
                directions=[direction] if direction is not None else [],
                trace=trace, det=det, discriminant=discriminant)


def count_zero_crossings(g):
    """g 가 0을 **가로지르는** 횟수. 닿기만 하는 것(접함)은 세지 않는다.

    가로지름 = 서로 다른 두 고유직선, 접함 = 결손. 이 구분이 이 섹션의 핵심이라
    수를 세는 방식도 그것을 따라야 한다.

    처음과 끝을 이어 **순환으로** 센다. g 는 π 주기이기 때문이다 —
    v(θ+π) = −v(θ) 라 방향만 뒤집히고 직선은 같다. 이어 세지 않으면 고유직선이
    하필 θ=0 에 놓였을 때(대각행렬이 흔히 그렇다) 구간 끝에서 잘려 반만 세어진다.

    Args:
        g: [0, π) 를 **끝점 없이** 훑은 값. 끝점을 넣으면 같은 직선이 두 번 세어진다.
    """
    sign = np.sign(np.asarray(g, dtype=float))
    sign = sign[sign != 0]
    if len(sign) < 2:
        return 0
    return int(np.count_nonzero(sign != np.roll(sign, 1)))


# ── 화면 ─────────────────────────────────────────────────────────────────────

def _sweep_figure(A, structure, span):
    """뷰 A — 훑기. 위: 단위원 위의 v 와 Av. 아래: 어긋난 각 곡선.

    θ 슬라이더를 Plotly 프레임으로 만든다. 프레임을 미리 다 계산해 그림에
    넣어 두면 조작이 **브라우저 안에서만** 일어나므로 재실행이 0회이고,
    따라서 깜빡임도 0이다.
    """
    theta = np.linspace(0, np.pi, FRAMES)
    g = deviation_curve(A, theta)
    unit = np.linspace(0, 2 * np.pi, 200)

    fig = make_subplots(rows=2, cols=1, row_heights=[0.62, 0.38],
                        vertical_spacing=0.12,
                        subplot_titles=("v 와 Av", "어긋난 각 g(θ)"))

    # 0: 단위원
    fig.add_trace(go.Scatter(x=np.cos(unit), y=np.sin(unit), mode='lines',
                             line=dict(color='lightgray', width=1),
                             name='단위원', hoverinfo='skip'), row=1, col=1)
    # 1~: 고유직선 (있으면)
    for k, direction in enumerate(structure['directions']):
        end = direction * span
        fig.add_trace(go.Scatter(x=[-end[0], end[0]], y=[-end[1], end[1]],
                                 mode='lines',
                                 line=dict(color='mediumpurple', width=2, dash='dot'),
                                 name=f'고유직선 {k + 1}'), row=1, col=1)

    # 어긋난 각 곡선과 0선
    fig.add_trace(go.Scatter(x=np.degrees(theta), y=np.degrees(g), mode='lines',
                             line=dict(color='black', width=2),
                             name='g(θ)', hoverinfo='skip'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[0, 180], y=[0, 0], mode='lines',
                             line=dict(color='mediumpurple', width=1, dash='dot'),
                             name='g = 0', hoverinfo='skip'), row=2, col=1)

    # 움직이는 세 개. 인덱스를 기억해 두었다가 프레임에서 이것만 갱신한다.
    moving = len(fig.data)
    v0 = np.array([np.cos(theta[0]), np.sin(theta[0])])
    Av0 = A @ v0
    fig.add_trace(go.Scatter(x=[0, v0[0]], y=[0, v0[1]], mode='lines+markers',
                             line=dict(color=lab_ui.BEFORE, width=3),
                             marker=dict(size=9, symbol='circle'), name='v'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, Av0[0]], y=[0, Av0[1]], mode='lines+markers',
                             line=dict(color=lab_ui.AFTER, width=3, dash='dash'),
                             marker=dict(size=9, symbol='triangle-up'), name='Av'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[0.0], y=[np.degrees(g[0])], mode='markers',
                             marker=dict(size=11, color='black'),
                             name='현재 θ'), row=2, col=1)

    frames = []
    for k, th in enumerate(theta):
        v = np.array([np.cos(th), np.sin(th)])
        Av = A @ v
        frames.append(go.Frame(
            name=f"{k}",
            data=[
                go.Scatter(x=[0, v[0]], y=[0, v[1]]),
                go.Scatter(x=[0, Av[0]], y=[0, Av[1]]),
                go.Scatter(x=[np.degrees(th)], y=[np.degrees(g[k])]),
            ],
            traces=[moving, moving + 1, moving + 2],
        ))
    fig.frames = frames

    fig.update_xaxes(range=[-span, span], zeroline=True, zerolinecolor='gray',
                     showgrid=True, gridcolor='lightgray', row=1, col=1)
    fig.update_yaxes(range=[-span, span], zeroline=True, zerolinecolor='gray',
                     showgrid=True, gridcolor='lightgray',
                     scaleanchor='x', scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="θ (도)", range=[0, 180], dtick=45, row=2, col=1)
    fig.update_yaxes(title_text="g (도)", range=[-95, 95], dtick=45, row=2, col=1)

    fig.update_layout(
        height=760, uirevision="s5-sweep", margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation='h', y=-0.08),
        updatemenus=[dict(
            type='buttons', direction='left', x=0.0, y=1.10, xanchor='left',
            buttons=[
                dict(label='▶ 훑기', method='animate',
                     args=[None, dict(frame=dict(duration=45, redraw=False),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label='⏸', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate')]),
            ])],
        sliders=[dict(
            active=0, x=0.12, len=0.86, y=1.06, xanchor='left',
            currentvalue=dict(prefix='θ = ', suffix='°', font=dict(size=13)),
            steps=[dict(method='animate', label=f"{np.degrees(th):.0f}",
                        args=[[f"{k}"], dict(mode='immediate',
                                             frame=dict(duration=0, redraw=False),
                                             transition=dict(duration=0))])
                   for k, th in enumerate(theta)],
        )],
    )
    return fig


def _stay_figure(A, structure, span, point, trail_on):
    """뷰 B — 고유직선 위의 점은 정말 그 직선을 벗어나지 못하는가."""
    fig = go.Figure()

    for k, direction in enumerate(structure['directions']):
        end = direction * span * 1.4
        fig.add_trace(go.Scatter(x=[-end[0], end[0]], y=[-end[1], end[1]],
                                 mode='lines',
                                 line=dict(color='mediumpurple', width=2, dash='dot'),
                                 name=f'고유직선 {k + 1}'))

    if trail_on:
        history = lab_ui.trail_items("s5")
        if history:
            for trace in lab_ui.trail_traces([[p] for p, _ in history],
                                             lab_ui.BEFORE, "입력 자취"):
                fig.add_trace(trace)
            for trace in lab_ui.trail_traces([[q] for _, q in history],
                                             lab_ui.AFTER, "상의 자취"):
                fig.add_trace(trace)

    image = A @ point
    fig.add_trace(go.Scatter(x=[0, point[0]], y=[0, point[1]], mode='lines+markers',
                             line=dict(color=lab_ui.BEFORE, width=2),
                             marker=dict(size=10, symbol='circle'), name='점 p'))
    fig.add_trace(go.Scatter(x=[0, image[0]], y=[0, image[1]], mode='lines+markers',
                             line=dict(color=lab_ui.AFTER, width=2, dash='dash'),
                             marker=dict(size=10, symbol='triangle-up'), name='Ap'))

    lab_ui.equal_axes(fig, view_key="s5-stay",
                      x_range=[-span, span], y_range=[-span, span])
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation='h', y=-0.05))
    return fig


def _basis_figure(A, structure, span):
    """뷰 C — 고유기저로 보면 변환이 두 방향 늘이기일 뿐이다."""
    fig = go.Figure()
    directions = structure['directions']
    if len(directions) < 2:
        return None

    u, w = directions[0], directions[1]
    ticks = np.arange(-3, 4)
    for coefficient in ticks:
        for base, other, color in ((u, w, 'darkorange'), (w, u, 'teal')):
            start = base * coefficient - other * 3
            end = base * coefficient + other * 3
            fig.add_trace(go.Scatter(
                x=[start[0], end[0]], y=[start[1], end[1]], mode='lines',
                line=dict(color=color, width=1), opacity=0.45,
                hoverinfo='skip', showlegend=False))
            image_start, image_end = A @ start, A @ end
            fig.add_trace(go.Scatter(
                x=[image_start[0], image_end[0]], y=[image_start[1], image_end[1]],
                mode='lines', line=dict(color=color, width=1, dash='dash'),
                opacity=0.9, hoverinfo='skip', showlegend=False))

    for direction, name in ((u, '고유직선 1'), (w, '고유직선 2')):
        end = direction * span * 1.4
        fig.add_trace(go.Scatter(x=[-end[0], end[0]], y=[-end[1], end[1]],
                                 mode='lines',
                                 line=dict(color='mediumpurple', width=2.5),
                                 name=name))

    lab_ui.equal_axes(fig, view_key="s5-basis",
                      x_range=[-span, span], y_range=[-span, span])
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation='h', y=-0.05))
    return fig


def run_eigenspace():
    st.header("🟪 (5) 일차변환의 고유공간")
    st.markdown("**행렬이 통째로 제자리에 두는 직선이 있을까요? 몇 개나 있을까요?**")

    with st.expander("🔍 관찰 과제", expanded=False):
        st.markdown(
            "1. θ 를 0°에서 180°까지 밀어 보세요. **v 와 Av 가 같은 직선 위에 겹치는** 순간이 있나요?\n"
            "2. 아래 곡선이 0을 **가로지르는지 · 닿기만 하는지 · 아예 안 닿는지** 보세요. "
            "세 경우가 뜻하는 것이 각각 무엇일까요?\n"
            "3. 프리셋을 하나씩 눌러 보세요. **'결손(전단)' 은 중근인데 왜 고유직선이 하나뿐일까요?**"
        )

    # 프리셋으로 행렬을 바꾸려면 위젯이 만들어지기 전에 세션에 심어야 한다.
    for key, value in zip(("m11", "m12", "m21", "m22"), (1.0, 1.0, 0.0, 2.0)):
        st.session_state.setdefault(key, value)

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("행렬 A")
        preset = st.pills("프리셋", list(PRESETS), key="s5_preset",
                          help="네 가지 경우를 전부 만나 보세요.")
        if preset and st.session_state.get("s5_applied") != preset:
            values = PRESETS[preset].ravel()
            for key, value in zip(("m11", "m12", "m21", "m22"), values):
                st.session_state[key] = float(value)
            st.session_state["s5_applied"] = preset
            st.rerun()

        top = st.columns(2)
        a11 = top[0].number_input("a11", step=0.5, format="%.2f", key="m11")
        a12 = top[1].number_input("a12", step=0.5, format="%.2f", key="m12")
        bottom = st.columns(2)
        a21 = bottom[0].number_input("a21", step=0.5, format="%.2f", key="m21")
        a22 = bottom[1].number_input("a22", step=0.5, format="%.2f", key="m22")
        A = np.array([[a11, a12], [a21, a22]])

        structure = eigen_structure(A)
        st.latex(rf"A=\begin{{bmatrix}}{a11:g}&{a12:g}\\{a21:g}&{a22:g}\end{{bmatrix}}")

        st.subheader("계기판")
        st.markdown(
            f"- 대각합 $\\mathrm{{tr}}\\,A$ = **{structure['trace']:.3f}**\n"
            f"- 행렬식 $\\det A$ = **{structure['det']:.3f}**\n"
            f"- 판별식 $D=(a-d)^2+4bc$ = **{structure['discriminant']:.3f}**"
        )
        st.caption("행렬을 아무렇게나 바꿔도 λ₁+λ₂ = tr A 와 λ₁λ₂ = det A 는 깨지지 않습니다. 확인해 보세요.")

        st.divider()
        reveal = st.toggle("🔎 정체 밝히기", value=False, key="s5_reveal",
                           help="고유값과 고유공간의 이름을 보여 줍니다. 먼저 그림으로 추측해 보세요.")
        if reveal:
            names = {'two_lines': "서로 다른 두 고유직선 (각각 1차원)",
                     'defective': "고유직선 하나뿐 — 중근인데도 (결손)",
                     'plane': "평면 전체가 고유공간 (2차원)",
                     'none': "실고유공간 없음"}
            st.info(f"**{names[structure['kind']]}**")
            crossings = count_zero_crossings(
                deviation_curve(A, np.linspace(0, np.pi, 2000, endpoint=False)))
            st.caption(f"g(θ) 가 0을 가로지른 횟수: **{crossings}회**")
            if structure['eigenvalues']:
                st.latex(r"\lambda = " + ",\\ ".join(
                    f"{lam:.3f}" for lam in structure['eigenvalues']))
            else:
                st.caption("고유값이 실수가 아닙니다 — 지켜지는 방향이 하나도 없다는 뜻입니다.")

    with col2:
        span = float(max(2.5, min(6.0, np.abs(A).max() * 1.6)))
        view_a, view_b, view_c = st.tabs(["① 훑기", "② 정말 못 벗어나는가", "③ 고유기저로 보기"])

        with view_a:
            chart(_sweep_figure(A, structure, span), key="s5_sweep")
            st.caption(
                "각을 ±90° 로 접어 재기 때문에, 고유값이 음수여서 점이 원점 건너편으로 튀어도 "
                "**직선이 그대로면 g = 0** 입니다. 고유공간은 방향이 아니라 직선이니까요."
            )

        with view_b:
            if structure['directions']:
                default = structure['directions'][0] * 2.0
            else:
                default = np.array([2.0, 1.0])
            st.session_state.setdefault("s5_point", (float(default[0]), float(default[1])))

            trail_on = lab_ui.trail_controls("s5")
            point = np.array(st.session_state["s5_point"])
            if trail_on:
                image = A @ point
                lab_ui.trail_push("s5", ((point[0], point[1]),
                                         (float(image[0]), float(image[1]))), limit=150)

            event = chart(_stay_figure(A, structure, span, point, trail_on),
                          key="s5_stay", on_select="rerun", selection_mode="points")
            picked = event.selection["points"] if event and event.selection else []
            if picked:
                new = (float(picked[0]["x"]), float(picked[0]["y"]))
                if new != st.session_state["s5_point"]:
                    st.session_state["s5_point"] = new
                    st.rerun()

            st.caption(
                "고유직선(보라 점선) **위를 클릭해 가며** 자취를 남겨 보세요. "
                "상의 자취가 같은 직선 위에만 쌓입니다 — 이것이 '공간'이라 부르는 이유입니다. "
                "직선에서 **살짝 벗어난** 곳에서도 해 보세요."
            )

        with view_c:
            figure = _basis_figure(A, structure, span)
            if figure is None:
                st.warning(
                    "고유직선이 둘이 아니라 격자를 만들 수 없습니다. "
                    "**대각화가 늘 되는 것은 아닙니다.**"
                )
                st.caption("프리셋에서 '두 직선' 이나 'λ=0 포함' 을 골라 보세요.")
            else:
                chart(figure, key="s5_basis")
                st.caption(
                    "실선 = 고유기저 격자, 점선 = 그 격자의 상. "
                    "**격자가 휘지 않고 두 방향으로 늘어나기만 합니다.**"
                )
                if reveal:
                    st.latex(r"A = PDP^{-1},\quad D=\begin{bmatrix}\lambda_1&0\\0&\lambda_2\end{bmatrix}")

    with st.expander("✅ 확인", expanded=False):
        st.markdown(
            "- g(θ) 가 0을 **가로지름 2번** → 서로 다른 두 고유직선\n"
            "- **닿았다 되돌아옴(접함)** → 고유직선 하나뿐 (결손). 중근이어도 직선은 하나입니다\n"
            "- **한 번도 안 닿음** → 실고유공간 없음. 지켜지는 방향이 없다는 것이 곧 회전입니다\n"
            "- **항상 0** (평평한 가로선) → 평면 전체가 고유공간\n\n"
            "'λ=0 포함' 프리셋에서 고유공간 하나는 **붕괴하는 방향**(핵)입니다. "
            "$\\det A = 0$ 인 것과 같은 사건이에요."
        )

    with st.expander("🧭 더 나아가기 — 직교를 지키는 쌍", expanded=False):
        st.markdown(
            "고유방향은 **없을 수도** 있었습니다. 그런데 서로 직교하던 두 방향이 "
            "변환 후에도 직교하는 쌍은 **어떤 행렬에도 반드시 있습니다.**"
        )
        theta = np.linspace(0, np.pi, 361)
        v = np.column_stack([np.cos(theta), np.sin(theta)])
        w = np.column_stack([-np.sin(theta), np.cos(theta)])
        product = np.einsum('ij,ij->i', v @ A.T, w @ A.T)
        pair = go.Figure()
        pair.add_trace(go.Scatter(x=np.degrees(theta), y=product, mode='lines',
                                  line=dict(color='black'), name='Av · Aw'))
        pair.add_trace(go.Scatter(x=[0, 180], y=[0, 0], mode='lines',
                                  line=dict(color='mediumpurple', dash='dot'),
                                  name='0'))
        pair.update_layout(height=280, uirevision="s5-pair",
                           xaxis_title="θ (도)", yaxis_title="Av · Aw",
                           margin=dict(l=10, r=10, t=10, b=10))
        chart(pair, key="s5_pair")
        st.caption(
            "0을 지나는 θ 가 그 쌍입니다. 그 쌍이 단위원의 상인 **타원의 장축·단축**이고, "
            "특이값분해의 정체입니다. 고유공간과 달리 이것은 언제나 존재합니다."
        )

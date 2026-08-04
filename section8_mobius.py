"""(8) 원은 원으로 간다 — 뫼비우스 변환.

관찰시키려는 명제:

    w = (az+b)/(cz+d) 는 **원과 직선을 원과 직선으로** 보낸다.
    이 변환의 눈으로 보면 **직선은 무한대를 지나는 원**이다.

§3 과 성격이 정반대이며 그래서 보완적이다. §3 은 자유 타이핑(표현력)이고
여기는 구조화된 손잡이(발견)다. §3 에서 `1/z` 를 타이핑할 수는 있지만,
**격자를 통째로 보는** 어포던스가 없어 "원 → 원" 이 보이지 않는다.

도형 하나가 아니라 **족(family)** 을 보내는 것이 이 섹션의 핵심이다.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import lab_ui
from lab_ui import chart

#: 분모가 0에 너무 가까운 점은 무한대로 날아간다. 그대로 두면 축이 깨진다.
NEAR_POLE = 1e-6
#: 화면 밖으로 한참 나간 점은 잘라 낸다.
FAR = 60.0

PRESETS = {
    "1/z (반전)": dict(a_re=0.0, a_im=0.0, b_re=1.0, b_im=0.0,
                     c_re=1.0, c_im=0.0, d_re=0.0, d_im=0.0),
    "케일리 변환": dict(a_re=1.0, a_im=0.0, b_re=0.0, b_im=-1.0,
                    c_re=1.0, c_im=0.0, d_re=0.0, d_im=1.0),
    "(z−1)/(z+1)": dict(a_re=1.0, a_im=0.0, b_re=-1.0, b_im=0.0,
                        c_re=1.0, c_im=0.0, d_re=1.0, d_im=0.0),
    "c=0 (그냥 일차식)": dict(a_re=1.0, a_im=1.0, b_re=1.0, b_im=0.0,
                         c_re=0.0, c_im=0.0, d_re=1.0, d_im=0.0),
    "ad−bc=0 (한 점으로)": dict(a_re=1.0, a_im=0.0, b_re=2.0, b_im=0.0,
                            c_re=2.0, c_im=0.0, d_re=4.0, d_im=0.0),
}


# ── 수학 ─────────────────────────────────────────────────────────────────────

def mobius(z, a, b, c, d):
    """w = (az+b)/(cz+d). 극 근처는 NaN 으로 두어 그리는 쪽에서 끊게 한다."""
    z = np.asarray(z, dtype=complex)
    denominator = c * z + d
    with np.errstate(all='ignore'):
        w = (a * z + b) / denominator
    w[np.abs(denominator) < NEAR_POLE] = np.nan
    w[~np.isfinite(w)] = np.nan
    return w


def fixed_points(a, b, c, d):
    """제자리에 남는 점 — cz² + (d−a)z − b = 0 의 해. 최대 2개.

    c = 0 이면 이차항이 사라져 일차방정식이 되고(고정점 1개, 나머지 하나는
    무한대), a = d 까지 같으면 평행이동이라 유한한 고정점이 아예 없다.
    """
    if abs(c) < 1e-12:
        if abs(d - a) < 1e-12:
            return []
        return [b / (d - a)]
    discriminant = np.sqrt((d - a) ** 2 + 4 * b * c)
    return [(-(d - a) + s * discriminant) / (2 * c) for s in (1, -1)]


def circle_family(count, span, points=240):
    """동심원 + 방사선. 원과 직선이 어떻게 서로 오가는지 보려면 둘 다 있어야 한다."""
    curves = []
    for radius in np.linspace(span / count, span, count):
        angle = np.linspace(0, 2 * np.pi, points)
        curves.append(('원', radius * np.exp(1j * angle)))
    for phi in np.linspace(0, np.pi, count, endpoint=False):
        t = np.linspace(-span, span, points)
        curves.append(('방사선', t * np.exp(1j * phi)))
    return curves


def grid_family(count, span, points=240):
    """직교 좌표격자. 직선만 있는 족이라 '직선 → 원' 이 더 또렷하다."""
    curves = []
    for value in np.linspace(-span, span, count):
        t = np.linspace(-span, span, points)
        curves.append(('세로', value + 1j * t))
        curves.append(('가로', t + 1j * value))
    return curves


def clip(w):
    """화면 밖으로 달아난 부분을 NaN 으로 끊는다.

    끊지 않고 이으면 극을 가로지르는 곡선이 화면을 가로지르는 가짜 선으로
    이어져, 있지도 않은 구조를 보여 주게 된다.
    """
    w = np.array(w, dtype=complex)
    w[np.abs(w) > FAR] = np.nan
    return w


# ── 화면 ─────────────────────────────────────────────────────────────────────

def run_mobius():
    st.header("🟨 (8) 원은 원으로 간다 — 뫼비우스 변환")
    st.latex(r"w=\frac{az+b}{cz+d}")
    st.markdown("**원을 보내면 무엇이 될까요? 직선을 보내면요?**")

    with st.expander("🔍 관찰 과제", expanded=False):
        st.markdown(
            "1. `1/z` 프리셋을 누르고 **동심원**을 보세요. 원은 무엇이 되나요?\n"
            "2. 그중 **원점을 지나는** 원(방사선)만 다르게 행동합니다. 무엇이 되나요?\n"
            "3. 격자로 바꿔 보세요. **직선이 원이 됩니다.** 그렇다면 직선은 원의 일종일까요?\n"
            "4. `ad−bc=0` 프리셋을 눌러 보세요. 왜 모든 것이 한 점으로 뭉개질까요?"
        )

    defaults = dict(a_re=0.0, a_im=0.0, b_re=1.0, b_im=0.0,
                    c_re=1.0, c_im=0.0, d_re=0.0, d_im=0.0)
    for key, value in defaults.items():
        st.session_state.setdefault(f"s8_{key}", value)

    col1, col2 = st.columns([1, 1.5])

    with col1:
        preset = st.pills("프리셋", list(PRESETS), key="s8_preset")
        if preset and st.session_state.get("s8_applied") != preset:
            for key, value in PRESETS[preset].items():
                st.session_state[f"s8_{key}"] = float(value)
            st.session_state["s8_applied"] = preset
            st.rerun()

        st.subheader("계수")
        coefficients = {}
        for name in "abcd":
            pair = st.columns(2)
            coefficients[name] = complex(
                pair[0].number_input(f"Re({name})", step=0.5, format="%.2f",
                                     key=f"s8_{name}_re"),
                pair[1].number_input(f"Im({name})", step=0.5, format="%.2f",
                                     key=f"s8_{name}_im"))
        a, b, c, d = (coefficients[n] for n in "abcd")

        st.divider()
        family = st.radio("보낼 족", ["동심원 + 방사선", "직교 격자"],
                          key="s8_family", horizontal=True)
        count = st.slider("촘촘함", 4, 12, 7, key="s8_count")
        show_source = st.toggle("변환 전 족도 함께 보기", value=True, key="s8_source")

        st.subheader("계기판")
        determinant = a * d - b * c
        st.markdown(f"- $ad-bc$ = **{determinant.real:.3f}{determinant.imag:+.3f}i**")
        if abs(determinant) < 1e-9:
            st.warning("$ad-bc=0$ 입니다 — 이 변환은 뫼비우스 변환이 아닙니다. "
                       "무엇이 일어나는지 그림에서 확인해 보세요.")

        reveal = st.toggle("🔎 고정점 밝히기", value=False, key="s8_reveal")
        if reveal:
            points = fixed_points(a, b, c, d)
            if not points:
                st.info("유한한 고정점이 없습니다 — 평행이동이기 때문입니다.")
            else:
                st.latex(r"cz^2+(d-a)z-b=0")
                for k, point in enumerate(points, start=1):
                    st.latex(rf"z_{k}^*={point.real:.3f}{point.imag:+.3f}i")
                if abs(c) < 1e-12:
                    st.caption("c = 0 이라 고정점이 하나뿐입니다. 나머지 하나는 무한대에 있습니다.")

    with col2:
        span = 3.0
        curves = (circle_family(count, span) if family.startswith("동심원")
                  else grid_family(count, span))

        fig = go.Figure()
        # 족마다 색을 유지해 **어느 선이 어느 원이 되었는지** 추적되게 한다.
        palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e',
                   '#17becf', '#8c564b', '#e377c2']
        for index, (kind, z) in enumerate(curves):
            color = palette[index % len(palette)]
            if show_source:
                source = clip(z)
                fig.add_trace(go.Scatter(
                    x=source.real, y=source.imag, mode='lines',
                    line=dict(color=color, width=1, dash='dot'),
                    opacity=0.35, hoverinfo='skip', showlegend=False))
            w = clip(mobius(z, a, b, c, d))
            fig.add_trace(go.Scatter(
                x=w.real, y=w.imag, mode='lines',
                line=dict(color=color, width=2),
                hoverinfo='skip', showlegend=False))

        if reveal:
            for point in fixed_points(a, b, c, d):
                if abs(point) < FAR:
                    fig.add_trace(go.Scatter(
                        x=[point.real], y=[point.imag], mode='markers',
                        marker=dict(size=14, color='mediumpurple', symbol='star'),
                        name='고정점'))
            if abs(c) > 1e-12:
                pole = -d / c
                if abs(pole) < FAR:
                    fig.add_trace(go.Scatter(
                        x=[pole.real], y=[pole.imag], mode='markers',
                        marker=dict(size=13, color='black', symbol='x'),
                        name='극 (무한대로 가는 점)'))

        view = 6.0
        lab_ui.equal_axes(fig, view_key=f"s8-{family}",
                          x_range=[-view, view], y_range=[-view, view])
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title='Re', yaxis_title='Im', showlegend=reveal)
        chart(fig, key="s8_chart")
        st.caption("점선 = 변환 전 족, 실선 = 변환 후. **같은 색끼리가 서로 대응합니다.**")

    with st.expander("✅ 확인", expanded=False):
        st.markdown(
            "- 뫼비우스 변환은 **원과 직선을 원과 직선으로** 보냅니다. "
            "둘을 한데 묶어 '원'이라 부르면(직선 = 무한대를 지나는 원) "
            "**원은 언제나 원으로 갑니다.**\n"
            "- `1/z` 에서 **원점을 지나는 원만** 직선이 됩니다. 원점이 무한대로 가기 때문입니다.\n"
            "- 극 $z=-d/c$ 가 무한대로 날아가고, 그 근처에서 격자가 폭발적으로 벌어집니다.\n"
            "- 고정점은 $cz^2+(d-a)z-b=0$ 의 해라 **최대 2개**입니다. "
            "섹션 7 에서 본 고정점과 같은 개념입니다.\n"
            "- $ad-bc=0$ 이면 분자와 분모가 비례해 **상이 한 점으로 뭉개집니다.** "
            "섹션 1 의 $\\det A=0$ 과 같은 사건입니다.\n\n"
            "섹션 3 에서 `1/z` 를 직접 타이핑해 도형 하나를 보내 볼 수도 있습니다. "
            "거기는 자유롭게 쓰는 곳이고, 여기는 **족을 통째로 보는** 곳입니다."
        )

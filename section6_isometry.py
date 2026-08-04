"""(6) 거울을 몇 번 놓아야 하는가 — 평면 등거리변환의 분류.

관찰시키려는 명제:

    평면의 모든 등거리변환은 **평행이동·회전·대칭·미끄럼대칭 넷 중 하나**다.
    거울(대칭축)을 몇 개 어떻게 놓았는지가 그것을 결정하며, **개수의 홀짝이
    방향 보존 여부를 가른다.**

§2 는 원점을 지나는 축만 다룰 수 있어(모든 반사가 원점을 고정) 평행이동과
미끄럼대칭이 원리적으로 등장할 수 없다. **축의 위치를 풀어 주는 것만으로
이론의 절반이 새로 열린다** — 그것이 이 섹션이 §2 와 갈라지는 지점이다.

§2 를 대체하지 않고 병존시킨다. §2 는 점 하나·원점 통과 축의 입문판이고
여기는 도형·자유 배치의 심화판이다.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import lab_ui
from lab_ui import chart

VIEW = 6.0

#: 방향이 보이는 비대칭 도형. 점 하나로는 대칭의 방향 뒤집힘을 관찰할 수 없다.
FLAG = np.array([
    [0.0, 0.0], [0.0, 2.4], [1.4, 2.4], [1.4, 1.9], [0.5, 1.9],
    [0.5, 1.35], [1.15, 1.35], [1.15, 0.85], [0.5, 0.85], [0.5, 0.0], [0.0, 0.0],
])


# ── 수학 ─────────────────────────────────────────────────────────────────────
#
# 아핀변환을 3×3 동차좌표 행렬로 다루면 합성이 행렬곱 하나로 끝난다.

def reflection(angle_deg, offset):
    """각도 `angle_deg`, 원점으로부터의 부호 있는 거리 `offset` 인 직선에 대한 반사.

    축 위의 한 점은 c = offset·(−sinφ, cosφ) 이고, 선형부는

        L = [[cos2φ, sin2φ], [sin2φ, −cos2φ]]

    반사가 c 를 제자리에 두어야 하므로 평행성분은 t = (I − L)c 다.
    """
    phi = np.radians(angle_deg)
    L = np.array([[np.cos(2 * phi), np.sin(2 * phi)],
                  [np.sin(2 * phi), -np.cos(2 * phi)]])
    c = offset * np.array([-np.sin(phi), np.cos(phi)])
    t = (np.eye(2) - L) @ c
    M = np.eye(3)
    M[:2, :2], M[:2, 2] = L, t
    return M


def apply(M, points):
    """동차좌표 행렬을 (N, 2) 점들에 적용한다."""
    points = np.asarray(points, dtype=float)
    return points @ M[:2, :2].T + M[:2, 2]


def classify_isometry(M, tol=1e-9):
    """등거리변환의 정체를 밝힌다.

    det L = +1 이면 방향을 보존한다 — L 이 항등이면 평행이동, 아니면 회전이고
    중심은 (I − L)⁻¹t 다. det L = −1 이면 방향을 뒤집는다 — 미끄럼 성분
    (t·u)u 가 0이면 순수 대칭, 아니면 미끄럼대칭이다. 여기서 u 는 L 의 고유값
    +1 방향, 즉 축의 방향이다.

    회전인지 평행이동인지의 경계는 **수치 허용오차로** 판정해야 한다. 각도
    슬라이더가 만드는 부동소수점 오차 때문에 L 이 정확히 항등이 되는 일은
    거의 없다.

    Returns:
        dict — `kind` 는 'translation' | 'rotation' | 'reflection' | 'glide',
        회전이면 `center`·`angle`, 평행이동이면 `vector`,
        대칭·미끄럼대칭이면 `axis_point`·`axis_angle`·`glide`.
    """
    M = np.asarray(M, dtype=float)
    L, t = M[:2, :2], M[:2, 2]
    orientation = float(np.linalg.det(L))

    if orientation > 0:                       # 방향 보존
        if np.allclose(L, np.eye(2), atol=1e-7):
            return dict(kind='translation', preserves=True, vector=t)
        angle = float(np.degrees(np.arctan2(L[1, 0], L[0, 0])))
        center = np.linalg.solve(np.eye(2) - L, t)
        return dict(kind='rotation', preserves=True, center=center, angle=angle)

    # 방향 반전. 축의 방향 u 는 L 의 고유값 +1 방향.
    phi = np.arctan2(L[1, 0], L[0, 0]) / 2     # L 은 2φ 로 쓰여 있다
    u = np.array([np.cos(phi), np.sin(phi)])
    glide = float(t @ u)
    perpendicular = t - glide * u
    axis_point = perpendicular / 2             # 반사는 축을 수직으로 2배 옮긴다
    kind = 'reflection' if abs(glide) < 1e-7 else 'glide'
    return dict(kind=kind, preserves=False, axis_point=axis_point,
                axis_angle=float(np.degrees(phi)), glide=glide * u,
                glide_length=abs(glide))


# ── 화면 ─────────────────────────────────────────────────────────────────────

NAMES = {'translation': "평행이동", 'rotation': "회전",
         'reflection': "대칭", 'glide': "미끄럼대칭"}


def _axis_points(angle_deg, offset, span):
    phi = np.radians(angle_deg)
    direction = np.array([np.cos(phi), np.sin(phi)])
    base = offset * np.array([-np.sin(phi), np.cos(phi)])
    reach = span * 2
    return np.array([base - direction * reach, base + direction * reach])


def run_isometry():
    st.header("🟫 (6) 거울을 몇 번 놓아야 하는가")
    st.markdown("**거울을 여러 개 놓고 도형을 비춰 보세요. 결과는 결국 무엇이 될까요?**")

    with st.expander("🔍 관찰 과제", expanded=False):
        st.markdown(
            "1. 거울 **2개를 평행하게** 놓아 보세요(각도를 같게, 거리를 다르게). "
            "결과가 회전인가요?\n"
            "2. 거울 2개를 **교차**시켜 보세요. 도형이 얼마나 돌았나요? "
            "두 축의 사잇각과 비교해 보세요.\n"
            "3. 거울을 **3개**로 늘려 보세요. 새로운 종류의 변환이 나오나요?\n"
            "4. 거울 개수의 **홀짝**과 도형이 뒤집혔는지 사이에 어떤 관계가 있나요?"
        )

    col1, col2 = st.columns([1, 1.5])

    with col1:
        count = st.radio("거울 개수", [1, 2, 3], index=1, horizontal=True, key="s6_count")

        st.pills("프리셋", ["두 축 평행", "두 축 교차", "미끄럼대칭"], key="s6_preset")
        presets = {
            "두 축 평행": [(90.0, -1.0), (90.0, 1.0), (0.0, 0.0)],
            "두 축 교차": [(0.0, 0.0), (45.0, 0.0), (0.0, 0.0)],
            # 직선의 각은 180° 주기라 -90~90 으로 적는다 (120° ≡ -60°).
            "미끄럼대칭": [(0.0, 0.0), (60.0, 1.5), (-60.0, -1.0)],
        }
        preset = st.session_state.get("s6_preset")
        if preset and st.session_state.get("s6_applied") != preset:
            for k, (angle, offset) in enumerate(presets[preset], start=1):
                st.session_state[f"s6_a{k}"] = angle
                st.session_state[f"s6_d{k}"] = offset
            st.session_state["s6_applied"] = preset
            st.rerun()

        mirrors = []
        for k in range(1, count + 1):
            st.session_state.setdefault(f"s6_a{k}", [0.0, 45.0, 90.0][k - 1])
            st.session_state.setdefault(f"s6_d{k}", 0.0)
            st.markdown(f"**거울 {k}**")
            pair = st.columns(2)
            angle = pair[0].slider(f"각도 φ{k} (도)", -90.0, 90.0, step=1.0, key=f"s6_a{k}")
            offset = pair[1].slider(f"원점과의 거리 d{k}", -4.0, 4.0, step=0.1, key=f"s6_d{k}")
            mirrors.append((angle, offset))

        st.caption("👉 **거리 d 를 0이 아니게** 해 보세요. 원점을 지나지 않는 거울은 "
                   "섹션 2 에서는 놓을 수 없었습니다.")

        # 거울을 놓은 순서대로 합성한다. 마지막에 놓은 거울이 가장 나중에 적용된다.
        total = np.eye(3)
        for angle, offset in mirrors:
            total = reflection(angle, offset) @ total
        result = classify_isometry(total)

        st.divider()
        reveal = st.toggle("🔎 정체 밝히기", value=False, key="s6_reveal",
                           help="결과 변환의 이름과 그 기하 요소를 보여 줍니다. 먼저 추측해 보세요.")

    with col2:
        shape = FLAG - np.array([0.7, 1.2])          # 원점 근처로 옮겨 둔다
        stages = [shape]
        running = np.eye(3)
        for angle, offset in mirrors:
            running = reflection(angle, offset) @ running
            stages.append(apply(running, shape))

        fig = go.Figure()

        for k, (angle, offset) in enumerate(mirrors):
            line = _axis_points(angle, offset, VIEW)
            color = ['purple', 'orange', 'teal'][k]
            fig.add_trace(go.Scatter(x=line[:, 0], y=line[:, 1], mode='lines',
                                     line=dict(color=color, width=2),
                                     name=f'거울 {k + 1}'))

        # 중간 단계는 초록 계열로 옅게, 처음은 파랑, 마지막은 빨강.
        for k, stage in enumerate(stages):
            if k == 0:
                style = dict(color=lab_ui.BEFORE, dash='solid', width=2.5)
                name, symbol = '원래 도형', 'circle'
            elif k == len(stages) - 1:
                style = dict(color=lab_ui.AFTER, dash='dash', width=2.5)
                name, symbol = '최종 결과', 'triangle-up'
            else:
                style = dict(color=lab_ui.MIDDLE, dash='dot', width=1.5)
                name, symbol = f'{k}번째 거울 뒤', 'diamond'
            fig.add_trace(go.Scatter(
                x=stage[:, 0], y=stage[:, 1], mode='lines+markers',
                line=style, marker=dict(size=5, symbol=symbol, color=style['color']),
                opacity=1.0 if k in (0, len(stages) - 1) else 0.55, name=name))

        if reveal:
            if result['kind'] == 'rotation':
                center = result['center']
                fig.add_trace(go.Scatter(
                    x=[center[0]], y=[center[1]], mode='markers',
                    marker=dict(size=15, color='mediumpurple', symbol='star'),
                    name='회전 중심'))
            elif result['kind'] == 'translation':
                fig.add_trace(go.Scatter(
                    x=[0, result['vector'][0]], y=[0, result['vector'][1]],
                    mode='lines+markers',
                    line=dict(color='mediumpurple', width=3),
                    marker=dict(size=9, symbol='arrow-bar-up'),
                    name='평행이동 벡터'))
            else:
                point, phi = result['axis_point'], result['axis_angle']
                line = _axis_points(phi, float(np.hypot(*point)) * np.sign(
                    point @ np.array([-np.sin(np.radians(phi)), np.cos(np.radians(phi))])
                    or 1.0), VIEW)
                fig.add_trace(go.Scatter(x=line[:, 0], y=line[:, 1], mode='lines',
                                         line=dict(color='mediumpurple', width=3),
                                         name='결과의 축'))
                if result['kind'] == 'glide':
                    glide = result['glide']
                    fig.add_trace(go.Scatter(
                        x=[point[0], point[0] + glide[0]],
                        y=[point[1], point[1] + glide[1]],
                        mode='lines+markers',
                        line=dict(color='mediumpurple', width=3, dash='dash'),
                        marker=dict(size=9), name='미끄럼 벡터'))

        lab_ui.equal_axes(fig, view_key=f"s6-{count}",
                          x_range=[-VIEW, VIEW], y_range=[-VIEW, VIEW])
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation='h', y=-0.05))
        chart(fig, key="s6_chart")

        st.caption("🔵 원래 도형 · 🟢 중간 단계 · 🔴 최종 결과. "
                   "도형이 **뒤집혔는지** 보세요 — 거울 개수와 어떤 관계인가요?")

        if reveal:
            st.info(f"**{NAMES[result['kind']]}** — "
                    f"방향을 {'보존합니다' if result['preserves'] else '뒤집습니다'}.")
            if result['kind'] == 'rotation':
                st.latex(rf"\text{{중심}}=({result['center'][0]:.3f},\ "
                         rf"{result['center'][1]:.3f}),\quad "
                         rf"\text{{각}}={result['angle']:.2f}^\circ")
            elif result['kind'] == 'translation':
                st.latex(rf"\text{{이동}}=({result['vector'][0]:.3f},\ "
                         rf"{result['vector'][1]:.3f}),\quad "
                         rf"\text{{거리}}={np.hypot(*result['vector']):.3f}")
            else:
                st.latex(rf"\text{{축의 각}}={result['axis_angle']:.2f}^\circ,\quad "
                         rf"\text{{미끄럼}}={result['glide_length']:.3f}")

    with st.expander("✅ 확인", expanded=False):
        st.markdown(
            "| 거울 배치 | 결과 |\n| --- | --- |\n"
            "| 2개, 평행 | **평행이동.** 이동거리 = 두 축 간격의 **2배** |\n"
            "| 2개, 교차 | 교점을 중심으로 하는 **회전**, 각 = 사잇각의 **2배** |\n"
            "| 3개 | **대칭 또는 미끄럼대칭** — 세 번 놓아도 새 종류는 안 나옵니다 |\n"
            "| 짝수 개 | 방향 **보존** |\n"
            "| 홀수 개 | 방향 **반전** |\n\n"
            "결국 평면의 등거리변환은 **평행이동·회전·대칭·미끄럼대칭 넷뿐**입니다. "
            "거울 몇 개를 어떻게 놓든 이 넷 밖으로 나갈 수 없습니다.\n\n"
            "섹션 2 에서 본 '두 번 대칭 = 회전, 각은 2배' 는 두 축이 **원점에서** 만나는 "
            "특수한 경우였습니다. 여기서는 교점이 어디든 성립합니다."
        )

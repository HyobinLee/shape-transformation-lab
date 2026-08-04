"""섹션마다 똑같이 반복되는 화면 배관.

섹션이 4개에서 8개로 늘면서, `key`/`uirevision`/축 설정 같은 상용구를 각
섹션이 따로 들고 있으면 **한 군데만 빠뜨려도 그 섹션만 깜빡이거나 그 섹션만
축 비율이 어긋납니다.** 그래서 배관만 여기로 모읍니다.

`docs/260804_0200_plans.md` 7절이 정한 경계를 지킵니다.

    1. 학생이 읽는 문구는 여기 두지 않는다. (섹션 공통 UI 라벨은 예외 —
       모든 섹션에서 글자 그대로 같아야 하는 것들이다.)
    2. 변환·판정·공식은 여기 두지 않는다. 무엇을 변환할지는 섹션이 정한다.
    3. 섹션 파일만 읽고도 그 수업이 통째로 이해되어야 한다. 이 성질이 깨지는
       추출은 되돌린다.

즉 여기 있는 것은 **"어떻게 보이는가"** 뿐이고, **"무엇을 보여 주는가"** 는
전부 `sectionN_*.py` 에 남습니다.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

#: 모든 섹션이 쓰는 정사각 그래프 한 변(px). 섹션마다 달라지면 사이드바로
#: 섹션을 옮길 때마다 페이지 높이가 튀어 깜빡임처럼 보인다.
CHART_SIZE = 600

#: 변환 전 = 파랑, 변환 후 = 빨강, 중간 단계 = 초록.
#: 섹션이 바뀌어도 색의 뜻이 유지되므로 학생이 색만 보고 읽을 수 있다.
BEFORE, AFTER, MIDDLE = 'blue', 'red', 'green'
#: 자취가 색상 채널을 무지개에 양보할 때 쓰는 무채색.
TRAIL_GRAY = '#9aa0a6'


def format_number(n):
    """소수점 이하가 없으면 정수처럼, 있으면 한 자리까지 보여 준다."""
    return f"{n:.1f}".rstrip('0').rstrip('.') if n % 1 != 0 else str(int(n))


def equal_axes(fig, view_key, x_range=None, y_range=None, size=CHART_SIZE):
    """가로세로 1:1 등축으로 고정하고, 갱신해도 시야가 초기화되지 않게 한다.

    이 앱에서 등축 스케일은 미관이 아니라 **정확성 요건**이다. 회전과 대칭이
    눈으로 회전·대칭처럼 보여야 하기 때문이다.

    Args:
        fig: 손볼 Plotly Figure.
        view_key: `uirevision` 에 넣을 값. **이 값이 그대로면 학생이 확대·이동해
            둔 시야가 갱신 후에도 유지되고, 값이 바뀌면 시야가 초기화된다.**
            그러므로 "보던 것이 달라졌으니 시야를 되돌려도 좋은" 기준
            (예: 도형 종류)을 넣고, 매 조작마다 바뀌는 값(행렬 성분 등)은
            넣지 않는다.
        x_range, y_range: 축 범위. None 이면 Plotly 가 자동으로 잡는다.
        size: 정사각 그래프 한 변(px).
    """
    axis = dict(zeroline=True, zerolinecolor='gray',
                showgrid=True, gridcolor='lightgray')
    fig.update_xaxes(range=x_range, **axis)
    fig.update_yaxes(range=y_range, scaleanchor='x', scaleratio=1, **axis)
    fig.update_layout(uirevision=view_key, width=size, height=size)
    return fig


def chart(fig, key, **kwargs):
    """그래프를 그린다. `key` 는 선택이 아니라 필수다.

    `key` 가 없으면 프론트엔드가 갱신된 그래프를 **같은 요소로 알아보지 못해
    통째로 다시 마운트**한다. 그것이 이 앱에서 보이던 번쩍임의 직접 원인이었다.
    함수 인자로 강제해 두면 새 섹션에서 빠뜨릴 수 없다.

    `use_container_width` 는 넘기지 않는다 — Streamlit 1.60 부터
    `width='stretch'` 가 기본값이라 아무것도 넘기지 않는 것이 맞다.
    """
    return st.plotly_chart(fig, key=key, **kwargs)


# ── 채널 예산 ────────────────────────────────────────────────────────────────
#
# 관찰 보조 장치(자취·무지개)는 전부 그림의 시각 채널을 소비한다. 특히 무지개는
# "파랑 = 변환 전 / 빨강 = 변환 후" 라는 이 앱의 가장 오래된 관용구가 쓰던
# 색상 채널을 통째로 가져간다. 어느 조합에서 어느 채널이 무엇을 뜻하는지
# 여기 한곳에서 정해 두지 않으면 섹션마다 다르게 해석된다.

def channel_style(rainbow=False):
    """변환 전/후를 어느 채널로 구분할지 정한다.

    무지개가 켜지면 색상은 '원본에서의 위치'가 가져가므로, 전/후 구분은
    **마커 모양과 선 스타일**이 진다. 관용구를 버리는 것이 아니라 다른
    채널로 옮기는 것이다.

    무지개가 꺼져 있어도 모양·선 스타일은 그대로 준다. 색만으로 구분하면
    적록색각이상 학생이 읽을 수 없기 때문에, 색은 늘 **중복 부호화**된다.

    Returns:
        ``{'before': {...}, 'after': {...}}`` — 각각 `symbol`, `dash` 를 갖고,
        무지개가 꺼져 있을 때만 `color` 를 갖는다.
    """
    before = dict(symbol='circle', dash='solid')
    after = dict(symbol='triangle-up', dash='dash')
    if not rainbow:
        before['color'] = BEFORE
        after['color'] = AFTER
    return {'before': before, 'after': after}


# ── 무지개 대응 ──────────────────────────────────────────────────────────────

def is_closed(points, tol=1e-9):
    """첫 점과 끝 점이 같으면 닫힌 곡선으로 본다.

    닫힌 곡선에는 **순환 색상표**를 써야 한다. 그러지 않으면 시작과 끝이
    만나는 자리에 색의 이음매가 생겨, 도형에 없는 특징을 있는 것처럼
    보이게 만든다.
    """
    points = np.asarray(points, dtype=float)
    return len(points) > 2 and bool(np.allclose(points[0], points[-1], atol=tol))


def colorscale(closed):
    """닫힌 곡선이면 순환 색상표, 열린 곡선이면 순차 색상표."""
    return 'hsv' if closed else 'turbo'


def _longest_visible_run(points, x_range, y_range):
    """상자 안에 들어온 **가장 긴 연속 구간**만 남긴다.

    화면 밖까지 포함해 정규화하면 무지개의 일부만 화면에 나타난다. §1 의
    직선이 그 예로, x 를 [-20, 20] 으로 만들어 두고 화면은 훨씬 좁게 잡는다.

    잘린 구간이 여러 개일 때 굳이 다 잇지 않고 가장 긴 것 하나만 쓴다.
    이어 붙이면 화면 밖에서 건너뛴 만큼 색이 튀어, 대응이 오히려 흐려진다.
    """
    if x_range is None or y_range is None:
        return points
    inside = ((points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
              (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]))
    if inside.all() or not inside.any():
        return points

    best_start = best_len = run_start = run_len = 0
    for k, is_in in enumerate(inside):
        if is_in:
            run_len = run_len + 1 if run_len else 1
            run_start = k - run_len + 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        else:
            run_len = 0
    return points[best_start:best_start + best_len] if best_len >= 2 else points


def arclength_parameter(points, x_range=None, y_range=None, n=140):
    """폴리라인을 **보이는 영역 안에서의 호의 길이**로 매개변수화한다.

    인덱스 순서로 색을 입히면 표본이 촘촘한 구간에 색이 몰린다. 호의 길이로
    잡아야 화면에서 색 띠의 폭이 고르고, 변환 후 그 폭이 흐트러진 정도가 곧
    그 자리에서 도형이 얼마나 늘어났는지가 된다.

    Args:
        points: (N, 2) 폴리라인.
        x_range, y_range: 파이썬이 계산한 축 범위. **브라우저의 실시간
            뷰포트가 아니다** — 학생이 확대해도 색은 다시 계산되지 않는다.
            색은 점의 이름표이므로, 확대할 때마다 이름이 바뀌면 추적하려던
            대응 자체가 무너진다.
        n: 다시 뽑을 점 개수.

    Returns:
        ``(resampled (n, 2), t (n,))`` — `t` 는 [0, 1] 등간격.
        **색을 입힐 만한 길이가 없으면 `(원본, None)`** 을 돌려준다
        (도형이 한 점으로 붕괴한 경우 등). 부르는 쪽에서 단색으로 되돌린다.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points, None

    visible = _longest_visible_run(points, x_range, y_range)
    steps = np.linalg.norm(np.diff(visible, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = cumulative[-1]
    if not np.isfinite(total) or total <= 1e-12:
        return points, None

    t = np.linspace(0.0, 1.0, n)
    target = t * total
    resampled = np.column_stack([
        np.interp(target, cumulative, visible[:, 0]),
        np.interp(target, cumulative, visible[:, 1]),
    ])
    return resampled, t


def angle_parameter(points, center=None):
    """순서 없는 점구름을 **중심에 대한 편각**으로 매개변수화한다.

    §3·§8 의 자취는 격자 마스크에서 나오므로(`Z[mask]`) 점에 순서가 없고,
    따라서 호의 길이가 정의되지 않는다. 최근접 이웃으로 꿰어 곡선을 만드는
    방법은 자취가 갈라지는 순간 엉뚱하게 이어지고, 면적형 자취에는 애초에
    성립하지 않는다. 편각이 정직하다.

    Returns:
        ``t (N,)`` — [0, 1). 점이 모두 중심에 몰려 있으면 None.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return None
    center = np.mean(points, axis=0) if center is None else np.asarray(center, dtype=float)
    offset = points - center
    if not np.any(np.linalg.norm(offset, axis=1) > 1e-12):
        return None
    angle = np.arctan2(offset[:, 1], offset[:, 0])
    return (angle + np.pi) / (2 * np.pi)


def rainbow_marker(t, closed, size=5, **kwargs):
    """`t` 를 색으로 바꾼 마커 설정.

    **변환 전과 후에 같은 `t` 를 넘겨야 한다.** 변환 후에 다시 계산하면
    대응이 깨지고, 이 기능은 대응을 보여 주는 것이 전부다.
    """
    return dict(size=size, color=t, colorscale=colorscale(closed),
                cmin=0.0, cmax=1.0, showscale=False, **kwargs)


def waypoint_indices(count, total):
    """등간격 웨이포인트로 쓸 인덱스.

    색을 전혀 못 보는 학생도 번호로 대응을 읽을 수 있게 한다. 게다가 번호는
    색보다 나은 구석이 있다 — 변환 후 **번호 간격이 흐트러진 정도**로
    어디가 늘어나고 어디가 눌렸는지를 정량적으로 읽을 수 있다.
    """
    if total <= 0 or count <= 0:
        return np.array([], dtype=int)
    return np.unique(np.linspace(0, total - 1, count).astype(int))


# ── 자취 남기기 ──────────────────────────────────────────────────────────────

def _trail_key(name):
    return f"trail__{name}"


def trail_push(name, entry, limit=100):
    """자취에 한 항목을 쌓는다. **직전과 같으면 쌓지 않는다.**

    Streamlit 은 아무 위젯이나 건드려도 재실행되므로, 무심코 짜면 같은 점이
    재실행마다 중복 적재된다. 값이 실제로 달라졌을 때만 넣어야 한다.

    Args:
        name: 섹션별로 다른 이름. 섹션 간 자취가 섞이지 않게 한다.
        entry: 해시 비교가 되는 값(좌표 튜플 등).
        limit: 남길 개수 상한. 상한이 없으면 수업 한 시간이면 느려진다.
    """
    key = _trail_key(name)
    history = st.session_state.setdefault(key, [])
    if history and history[-1] == entry:
        return
    history.append(entry)
    if len(history) > limit:
        del history[:len(history) - limit]


def trail_items(name):
    return st.session_state.get(_trail_key(name), [])


def trail_clear(name):
    st.session_state[_trail_key(name)] = []


def trail_controls(name):
    """자취 토글·지우기 버튼. 켜져 있는지를 돌려준다."""
    left, right = st.columns([2, 1])
    with left:
        on = st.toggle("자취 남기기", value=False, key=f"trail_on__{name}")
    with right:
        if st.button("자취 지우기", key=f"trail_clear__{name}"):
            trail_clear(name)
    return on


def trail_traces(points_list, color, name="자취", max_opacity=0.55):
    """지난 자취를 나이순으로 옅게 그린 trace 들.

    오래된 것일수록 옅다. 무지개가 함께 켜져 있으면 `color` 에
    `TRAIL_GRAY` 를 넘겨 자취가 색상 채널을 양보하게 한다 — 색상이 두 가지를
    동시에 뜻하는 상황을 만들지 않는다.
    """
    traces = []
    count = len(points_list)
    for k, points in enumerate(points_list):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        age = (k + 1) / count  # 1 에 가까울수록 최근
        traces.append(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='lines' if len(points) > 1 else 'markers',
            line=dict(color=color, width=1),
            marker=dict(color=color, size=5),
            opacity=max_opacity * age,
            hoverinfo='skip',
            showlegend=(k == count - 1),
            name=name,
        ))
    return traces

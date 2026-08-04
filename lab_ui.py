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

import streamlit as st

#: 모든 섹션이 쓰는 정사각 그래프 한 변(px). 섹션마다 달라지면 사이드바로
#: 섹션을 옮길 때마다 페이지 높이가 튀어 깜빡임처럼 보인다.
CHART_SIZE = 600


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

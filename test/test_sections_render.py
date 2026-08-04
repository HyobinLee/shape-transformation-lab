"""네 섹션이 이상한 입력을 받아도 예외 없이 그려지는지 확인한다.

교육용 도구의 사용자는 반드시 이상한 값을 넣는다. 그래서 이 검증이 보는 것은
계산의 정확성이 아니라 **앱이 죽지 않는가** 하나다. 학생 화면에 빨간
traceback 이 뜨는 순간 그 수업은 거기서 끝난다.

`pytest` 없이 그냥 돌아간다:

    python test/test_sections_render.py

통과하면 마지막 줄에 ALL PASS 가 찍히고 종료 코드가 0이다.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

MENUS = [
    "1. 행렬을 통한 일차변환",
    "2. 행렬을 통한 대칭/회전변환",
    "3. 복소평면에서의 이동",
    "4. 복소평면에서의 회전/평행이동",
]

failures = []


def check(label, condition, detail=""):
    mark = "OK  " if condition else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail and not condition else ''}")
    if not condition:
        failures.append(label)


def load(menu, **widgets):
    """앱을 해당 섹션으로 띄우고, 주어진 위젯 값을 넣은 뒤 다시 실행한다."""
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(menu)
    at.run()
    for key, value in widgets.items():
        at.session_state[key] = value
    if widgets:
        at.run()
    return at


def describe(at):
    return " | ".join(e.value for e in at.exception) if at.exception else ""


print("=== 1. 네 섹션이 기본값으로 뜨는가 ===")
for menu in MENUS:
    at = load(menu)
    check(menu, not at.exception, describe(at))

print("\n=== 2. 섹션 1 — 학생이 넣을 법한 잘못된 좌표 ===")
# 위젯 라벨이 아니라 순서로 접근한다. 삼각형이 기본이라 좌표 칸이 셋이다.
for bad in ["1;1", "", "1,1,1", "abc", "1,"]:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(MENUS[0])
    at.run()
    at.text_input[0].set_value(bad)
    at.run()
    ok = not at.exception and len(at.warning) > 0
    check(f"좌표 '{bad}' → 경고로 안내하고 계속", ok, describe(at) or "경고가 없다")

print("\n=== 3. 섹션 1 — 직선 a=b=0 (0으로 나누던 자리) ===")
at = AppTest.from_file(APP, default_timeout=60)
at.run()
at.sidebar.radio[0].set_value(MENUS[0])
at.run()
at.selectbox[0].set_value("직선")
at.run()
at.number_input[0].set_value(0.0)   # 계수 a
at.number_input[1].set_value(0.0)   # 계수 b
at.run()
check("a=0, b=0 → 죽지 않고 안내", not at.exception and len(at.warning) > 0, describe(at))

print("\n=== 4. 섹션 2 — 클릭 경로가 숫자 칸에 반영되는가 ===")
at = load(MENUS[1])
at.session_state["input_x"] = 4.0
at.session_state["input_y"] = -3.0
at.run()
check("클릭 좌표가 유지된다",
      at.session_state["input_x"] == 4.0 and at.session_state["input_y"] == -3.0,
      f"x={at.session_state['input_x']}, y={at.session_state['input_y']}")

print("\n=== 5. 섹션 2 — 모르는 축 종류가 들어와도 죽지 않는가 ===")
from section2_symmetry_rotation import reflection_matrix  # noqa: E402
import numpy as np  # noqa: E402
check("reflection_matrix('없는축') → 항등행렬",
      np.allclose(reflection_matrix("없는축"), np.eye(2)))

print("\n=== 6. 섹션 3 — 잘못된 수식은 한국어로 안내 ===")
for bad_locus in ["x**2 +", "__import__('os')", "zzz == 1"]:
    at = load(MENUS[2], definition_input=bad_locus)
    check(f"자취식 '{bad_locus}'", not at.exception and len(at.error) > 0,
          describe(at) or "오류 안내가 없다")

print("\n=== 7. 모든 그래프에 안정적인 key 가 있는가 ===")
# key 가 없으면 갱신 때마다 차트가 통째로 다시 마운트되어 번쩍인다.
for menu in MENUS:
    at = load(menu)
    charts = at.get("plotly_chart")
    check(f"{menu} — 차트 {len(charts)}개 모두 key 보유",
          len(charts) > 0 and all(c.proto.id for c in charts))

print("\n" + "=" * 52)
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL PASS")

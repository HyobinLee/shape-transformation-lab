"""모든 섹션이 이상한 입력을 받아도 예외 없이 그려지는지 확인한다.

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
    "5. 일차변환의 고유공간",
    "6. 거울을 몇 번 놓아야 하는가",
    "7. 되풀이하면 어디로 가는가",
    "8. 원은 원으로 간다 (뫼비우스)",
]

failures = []


def check(label, condition, detail=""):
    mark = "OK  " if condition else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail and not condition else ''}")
    if not condition:
        failures.append(label)


def open_section(menu, timeout=90):
    """앱을 띄우고 해당 섹션으로 이동한다.

    사이드바가 묶음(기초/심화) → 섹션 두 단계라 라디오도 두 개다.
    """
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    group = "기초" if menu in MENUS[:4] else "심화"
    at.sidebar.radio[0].set_value(group)
    at.run()
    at.sidebar.radio[1].set_value(menu)
    at.run()
    return at


def load(menu, **widgets):
    """앱을 해당 섹션으로 띄우고, 주어진 위젯 값을 넣은 뒤 다시 실행한다."""
    at = open_section(menu)
    for key, value in widgets.items():
        at.session_state[key] = value
    if widgets:
        at.run()
    return at


def describe(at):
    return " | ".join(e.value for e in at.exception) if at.exception else ""


FREE, DISCOVERY = "🔬 자유 탐구", "🧭 단계별 탐구"


def open_s1(mode=FREE, **widgets):
    """섹션 1 을 특정 모드로 연다.

    §1 은 화면이 둘이다 — 예전 실험실(자유 탐구)과 답을 감춘 3단계 경로.
    기본값은 단계 모드이므로, 예전 위젯을 건드리는 검증은 모드를 명시해야 한다.
    """
    at = open_section(MENUS[0])
    at.session_state["s1_mode"] = mode
    at.run()
    for key, value in widgets.items():
        at.session_state[key] = value
    if widgets:
        at.run()
    return at


print("=== 1. 모든 섹션이 기본값으로 뜨는가 ===")
for menu in MENUS:
    at = load(menu)
    check(menu, not at.exception, describe(at))

print("\n=== 2. 섹션 1 — 학생이 넣을 법한 잘못된 좌표 ===")
# 위젯 라벨이 아니라 순서로 접근한다. 삼각형이 기본이라 좌표 칸이 셋이다.
for bad in ["1;1", "", "1,1,1", "abc", "1,"]:
    at = open_section(MENUS[0])
    at.text_input[0].set_value(bad)
    at.run()
    ok = not at.exception and len(at.warning) > 0
    check(f"좌표 '{bad}' → 경고로 안내하고 계속", ok, describe(at) or "경고가 없다")

print("\n=== 3. 섹션 1 — 직선 a=b=0 (0으로 나누던 자리) ===")
at = open_s1(FREE)
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

print("\n=== 8. 관찰 보조 장치를 켜도 죽지 않는가 ===")
# 무지개는 도형 종류마다 다른 경로를 탄다 — 닫힌 곡선/열린 곡선/점구름.
for shape in ["삼각형", "사각형", "원", "직선"]:
    at = open_s1(FREE)
    at.selectbox[0].set_value(shape)
    at.run()
    at.session_state["s1_rainbow"] = True
    at.run()
    at.session_state["s1_numbers"] = True
    at.run()
    check(f"섹션 1 무지개+번호 — {shape}", not at.exception, describe(at))

at = open_s1(FREE)
at.session_state["s1_rainbow"] = True
at.run()
# 도형이 한 점으로 붕괴해도(모든 성분 0) 0으로 나누지 않아야 한다.
for k in range(4):
    at.number_input[k if len(at.number_input) > k else 0].set_value(0.0)
at.run()
check("섹션 1 무지개 + det=0 행렬", not at.exception, describe(at))

at = load(MENUS[2])
at.session_state["s3_rainbow"] = False
at.run()
check("섹션 3 무지개 끄기", not at.exception, describe(at))

print("\n=== 9. 자취가 쌓이고 상한을 지키는가 ===")
at = open_section(MENUS[1])
at.session_state["trail_on__s2"] = True
at.run()
for x in [1.0, 2.0, 3.0, 3.0, 3.0]:      # 같은 값 반복은 한 번만 쌓여야 한다
    at.session_state["input_x"] = x
    at.run()
try:
    # AppTest 의 session_state 프록시에는 .get() 이 없다 (앱 안의 st.session_state 와 다르다).
    history = at.session_state["trail__s2"]
except (KeyError, AttributeError):
    history = []
check("자취가 쌓인다", len(history) > 0, len(history))
check("같은 점 반복은 한 번만", len(history) <= 4, len(history))
check("섹션 2 자취 켜고도 예외 없음", not at.exception, describe(at))

print("\n=== 10. 섹션 5 — 네 가지 경우를 전부 그리는가 ===")
import json  # noqa: E402

from section5_eigenspace import PRESETS  # noqa: E402

for name, A in PRESETS.items():
    at = open_section(MENUS[4])
    at.session_state["s5_reveal"] = True
    for key, value in zip(("m11", "m12", "m21", "m22"), A.ravel()):
        at.session_state[key] = float(value)
    at.run()
    check(f"섹션 5 — {name}", not at.exception, describe(at))

at = load(MENUS[4])
spec = json.loads(at.get("plotly_chart")[0].proto.spec)
check("훑기 그래프에 프레임이 실린다", len(spec.get("frames", [])) > 0,
      len(spec.get("frames", [])))
check("훑기 슬라이더가 있다", len(spec["layout"].get("sliders", [])) == 1)
check("uirevision 이 있다", spec["layout"].get("uirevision") is not None)

print("\n=== 11. 섹션 1 — 연속 변환 프레임과 계기판 ===")
at = open_s1(FREE, s1_morph=True)
spec = json.loads(at.get("plotly_chart")[0].proto.spec)
check("연속 변환 프레임", len(spec.get("frames", [])) == 41, len(spec.get("frames", [])))
check("넓이 계기판 3칸", len(at.get("metric")) == 3, len(at.get("metric")))
check("연속 변환 켜고도 예외 없음", not at.exception, describe(at))

print("\n=== 12. 섹션 7 — 다섯 프리셋과 발산 방어 ===")
from section7_orbit import PRESETS as ORBIT_PRESETS  # noqa: E402

for name, values in ORBIT_PRESETS.items():
    at = open_section(MENUS[6])
    at.session_state["s7_reveal"] = True
    for key, value in values.items():
        at.session_state[f"s7_{key}"] = float(value)
    at.run()
    check(f"섹션 7 — {name}", not at.exception, describe(at))

at = load(MENUS[6], s7_r=2.5, s7_n=200)      # 확실히 발산시킨다
check("섹션 7 발산해도 죽지 않는다", not at.exception, describe(at))
check("발산을 학생에게 알린다", len(at.warning) > 0)

at = load(MENUS[6], s7_julia=True)
check("줄리아 집합을 켜도 죽지 않는다", not at.exception, describe(at))

print("\n=== 13. 섹션 4 — 고정점 추측·확인 경로 ===")
at = load(MENUS[3], s4_guess=True, s4_reveal=True)
check("추측+정답 토글 동시", not at.exception, describe(at))

at = load(MENUS[3], s4_reveal=True, theta_deg=0.0)
check("theta=0 → 중심이 없다고 안내", not at.exception and len(at.info) > 0, describe(at))

at = load(MENUS[3], **{"trail_on__s4": True, "s4_steps": True})
check("중간 단계 + 자취 동시", not at.exception, describe(at))

print("\n=== 14. 섹션 6 — 거울 1~3개와 프리셋 ===")
for count in (1, 2, 3):
    at = load(MENUS[5], s6_count=count, s6_reveal=True)
    check(f"거울 {count}개 + 정체 밝히기", not at.exception, describe(at))

for preset in ("두 축 평행", "두 축 교차", "미끄럼대칭"):
    at = load(MENUS[5], s6_preset=preset, s6_reveal=True, s6_count=3)
    check(f"섹션 6 프리셋 — {preset}", not at.exception, describe(at))

# 축이 겹치면 항등변환이 된다 — 회전과 평행이동의 경계.
at = load(MENUS[5], s6_count=2, s6_reveal=True,
          s6_a1=30.0, s6_d1=1.0, s6_a2=30.0, s6_d2=1.0)
check("같은 거울 두 번(항등)도 죽지 않는다", not at.exception, describe(at))

print("\n=== 15. 섹션 2 — 호·도형·계기판 ===")
at = load(MENUS[1], s2_shape=True, s2_arc=True, s2_reveal=True)
check("도형+호+숫자 동시", not at.exception, describe(at))
check("불변량 계기판 3칸", len(at.get("metric")) == 3, len(at.get("metric")))
check("섹션 6 으로 가는 안내가 있다", len(at.info) > 0)

at = load(MENUS[1], s2_shape=True, axis1="y축", axis2="x축")
check("축 종류를 바꿔도 호가 그려진다", not at.exception, describe(at))

print("\n=== 16. 섹션 8 — 프리셋과 극 방어 ===")
from section8_mobius import PRESETS as MOBIUS_PRESETS  # noqa: E402

for name, values in MOBIUS_PRESETS.items():
    at = open_section(MENUS[7])
    at.session_state["s8_reveal"] = True
    for key, value in values.items():
        at.session_state[f"s8_{key}"] = float(value)
    at.run()
    check(f"섹션 8 — {name}", not at.exception, describe(at))

at = load(MENUS[7], s8_family="직교 격자", s8_count=12, s8_reveal=True)
check("격자 족 + 촘촘하게", not at.exception, describe(at))

# 계수가 전부 0이면 어디서나 0/0 이다. 그래도 죽지 않아야 한다.
at = load(MENUS[7], **{f"s8_{n}_{p}": 0.0 for n in "abcd" for p in ("re", "im")})
check("계수가 전부 0이어도 죽지 않는다", not at.exception, describe(at))

at = load(MENUS[7], s8_preset="ad−bc=0 (한 점으로)")
check("ad−bc=0 을 학생에게 알린다", len(at.warning) > 0)

print("\n=== 17. 섹션 1 — 단계별 탐구 ===")
import section1_transformation_by_matrix as s1  # noqa: E402

at = open_s1(DISCOVERY)
check("단계 모드가 기본으로 뜬다", not at.exception, describe(at))
check("1단계 앞부분에는 도형 종류가 잠겨 있다", len(at.selectbox) == 0, len(at.selectbox))
check("행렬 입력 칸이 없다(고정행렬)", not any(w.label.startswith("a1") for w in at.number_input))

# 답을 감추는 규약 — 단계 모드에는 계기판도 기저 상자도 없다.
check("계기판이 감춰져 있다", len(at.get("metric")) == 0, len(at.get("metric")))
check("기저 상자 토글이 감춰져 있다",
      not any("기저" in t.label for t in at.toggle), [t.label for t in at.toggle])

for stage in (1, 2, 3):
    at = open_s1(DISCOVERY)
    at.session_state["s1"]["stage"] = stage
    at.run()
    check(f"{stage}단계가 그려진다", not at.exception, describe(at))
    check(f"{stage}단계 차트에 key 가 있다",
          len(at.get("plotly_chart")) == 1 and at.get("plotly_chart")[0].proto.id)

# 3단계는 방향각을 돌려 붕괴를 찾는 자리다. 임계각에서 상이 한 점이 되어도
# 축 범위 계산이 0으로 나뉘지 않아야 한다.
at = open_s1(DISCOVERY)
at.session_state["s1"]["stage"] = 3
at.run()
at.session_state["s1d_theta_exact"] = 180.0 + np.degrees(np.arctan(-2.0))   # A2 의 임계각
at.run()
check("3단계 임계각(상이 한 점)에서도 죽지 않는다", not at.exception, describe(at))

# 학생이 발견을 말하면 앱이 숫자로 재고 승인한다 — 키 없이 규칙 폴백으로.
at = open_s1(DISCOVERY)
at.session_state["s1"]["stage"] = 2
at.run()
s1._handle_submission(at.session_state["s1"], 2, s1.MATRICES["A2"],
                      "도형이 다 한 줄로 납작해져요")
check("올바른 관찰이 승인된다", "S2_FLAT" in at.session_state["s1"]["reached"][2])
s1._handle_submission(at.session_state["s1"], 2, s1.MATRICES["A2"],
                      "그 직선 기울기가 0.5 인 것 같아")
check("틀린 관찰은 승인되지 않는다", "S2_SLOPE" not in at.session_state["s1"]["reached"][2])
check("틀렸을 때 다음 힌트를 하나 준다", at.session_state["s1"]["hint_idx"])
at.run()
check("대화 뒤에도 죽지 않는다", not at.exception, describe(at))

# 잔상은 같은 입력이 반복돼도 쌓이지 않아야 한다.
at = open_s1(DISCOVERY, s1d_ghosts=True)
at.run()
at.run()
check("잔상이 재실행마다 중복 적재되지 않는다",
      len(at.session_state["s1"]["ghosts"]) <= 1, len(at.session_state["s1"]["ghosts"]))

for bad in ["1;1", "abc"]:
    at = open_s1(DISCOVERY)
    at.text_input[0].set_value(bad)
    at.run()
    check(f"단계 모드 좌표 '{bad}' → 경고로 안내하고 계속",
          not at.exception and len(at.warning) > 0, describe(at))

print("\n" + "=" * 52)
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL PASS")

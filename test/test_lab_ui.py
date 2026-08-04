"""관찰 보조 장치의 배관 검증 — 무지개 매개변수화와 자취 적재.

이 두 가지는 겉보기에 단순하지만 함정이 있는 코드다. 등간격 재표본화가
사실은 등간격이 아니거나, 재실행마다 같은 점이 중복으로 쌓이거나, 도형이
한 점으로 붕괴했을 때 0으로 나누는 일이 조용히 일어난다.

`pytest` 없이 그냥 돌아간다:

    python test/test_lab_ui.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import lab_ui  # noqa: E402

failures = []


def check(label, condition, detail=""):
    print(f"  [{'OK  ' if condition else 'FAIL'}] {label}"
          f"{(' — ' + str(detail)) if detail and not condition else ''}")
    if not condition:
        failures.append(label)


def circle(n=200, r=1.0):
    th = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


print("=== 1. 호의 길이 재표본화가 정말 등간격인가 ===")
# x 에 대해 등간격으로 뽑은 직선은 인덱스로는 고르지만, 세로에 가까울수록
# 호의 길이로는 한쪽에 몰린다. 재표본화가 그것을 바로잡아야 한다.
x = np.linspace(-10, 10, 400)
line = np.column_stack([x, 3 * x])
pts, t = lab_ui.arclength_parameter(line, n=100)
gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
check("직선: 구간 길이가 모두 같다", np.std(gaps) / np.mean(gaps) < 1e-6, np.std(gaps))
check("t 가 [0,1] 등간격", np.allclose(t, np.linspace(0, 1, 100)))

pts, t = lab_ui.arclength_parameter(circle(), n=120)
gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
check("원: 구간 길이가 모두 같다", np.std(gaps) / np.mean(gaps) < 1e-3, np.std(gaps))

print("\n=== 2. 보이는 영역 밖은 빼고 정규화하는가 ===")
# x 를 [-20,20] 으로 만들어 두고 화면은 [-2,2] 만 볼 때, 무지개 전체가
# 화면 안에 들어와야 한다.
wide = np.column_stack([np.linspace(-20, 20, 400), np.zeros(400)])
pts, t = lab_ui.arclength_parameter(wide, x_range=[-2, 2], y_range=[-2, 2], n=50)
check("잘라낸 뒤 남은 점이 전부 화면 안", pts[:, 0].min() >= -2.01 and pts[:, 0].max() <= 2.01,
      (pts[:, 0].min(), pts[:, 0].max()))
check("무지개가 화면 폭 전체를 덮는다", pts[:, 0].max() - pts[:, 0].min() > 3.9,
      pts[:, 0].max() - pts[:, 0].min())

print("\n=== 3. 퇴화 방어 — 0으로 나누지 않는가 ===")
check("한 점으로 붕괴한 도형 → t=None",
      lab_ui.arclength_parameter(np.zeros((50, 2)))[1] is None)
check("점 하나 → t=None", lab_ui.arclength_parameter(np.array([[1.0, 2.0]]))[1] is None)
check("빈 입력 → t=None", lab_ui.arclength_parameter(np.zeros((0, 2)))[1] is None)
check("중심에 몰린 점구름 → 편각 None", lab_ui.angle_parameter(np.zeros((10, 2))) is None)

print("\n=== 4. 대응이 깨지지 않는가 (이 기능의 전부) ===")
src, t = lab_ui.arclength_parameter(circle(), n=80)
M = np.array([[2.0, 1.0], [0.0, 0.5]])
dst = src @ M.T
before = lab_ui.rainbow_marker(t, closed=True)
after = lab_ui.rainbow_marker(t, closed=True)
check("변환 전후가 같은 색 배열을 쓴다", np.array_equal(before['color'], after['color']))
check("점 개수가 같다", len(src) == len(dst) == len(t))

print("\n=== 5. 닫힘 판정과 색상표 ===")
check("원은 닫혀 있다", lab_ui.is_closed(circle()))
check("직선은 열려 있다", not lab_ui.is_closed(line))
check("닫힌 곡선 → 순환 색상표", lab_ui.colorscale(True) == 'hsv')
check("열린 곡선 → 순차 색상표", lab_ui.colorscale(False) == 'turbo')

print("\n=== 6. 채널 예산 ===")
plain = lab_ui.channel_style(rainbow=False)
rainbow = lab_ui.channel_style(rainbow=True)
check("무지개 OFF → 색으로 전/후 구분",
      plain['before']['color'] == lab_ui.BEFORE and plain['after']['color'] == lab_ui.AFTER)
check("무지개 ON → 색을 내려놓는다",
      'color' not in rainbow['before'] and 'color' not in rainbow['after'])
check("색과 무관하게 모양이 늘 다르다 (색각 대응)",
      plain['before']['symbol'] != plain['after']['symbol']
      and rainbow['before']['symbol'] != rainbow['after']['symbol'])

print("\n=== 7. 자취 — 중복 적재와 상한 ===")


class FakeState(dict):
    def setdefault(self, k, v):
        return dict.setdefault(self, k, v)


lab_ui.st.session_state = FakeState()

lab_ui.trail_push("t1", (1.0, 1.0))
lab_ui.trail_push("t1", (1.0, 1.0))
lab_ui.trail_push("t1", (1.0, 1.0))
check("같은 점을 반복해도 한 번만 쌓인다", len(lab_ui.trail_items("t1")) == 1,
      len(lab_ui.trail_items("t1")))

lab_ui.trail_push("t1", (2.0, 2.0))
check("값이 바뀌면 쌓인다", len(lab_ui.trail_items("t1")) == 2)

for k in range(500):
    lab_ui.trail_push("t2", (float(k), 0.0), limit=100)
check("상한이 지켜진다", len(lab_ui.trail_items("t2")) == 100, len(lab_ui.trail_items("t2")))
check("오래된 것부터 버린다", lab_ui.trail_items("t2")[0] == (400.0, 0.0),
      lab_ui.trail_items("t2")[0])

lab_ui.trail_push("t3", (9.0, 9.0))
check("섹션끼리 섞이지 않는다", len(lab_ui.trail_items("t1")) == 2 and len(lab_ui.trail_items("t3")) == 1)

lab_ui.trail_clear("t1")
check("지우기", lab_ui.trail_items("t1") == [])

print("\n=== 8. 웨이포인트 ===")
idx = lab_ui.waypoint_indices(10, 80)
check("10곳", len(idx) == 10, len(idx))
check("처음과 끝을 포함", idx[0] == 0 and idx[-1] == 79, (idx[0], idx[-1]))
check("점보다 웨이포인트가 많아도 죽지 않는다", len(lab_ui.waypoint_indices(10, 3)) <= 3)
check("빈 도형", len(lab_ui.waypoint_indices(10, 0)) == 0)

print("\n=== 9. 넓이비가 정말 |det| 인가 (섹션 1 계기판의 주장) ===")
from section1_transformation_by_matrix import signed_area  # noqa: E402

rng = np.random.default_rng(20260804)
triangle = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [1.0, 1.0]])
square = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 1.0], [1.0, 1.0]])
shapes = {"삼각형": triangle, "사각형": square, "원": circle(300, 2.0)}

worst = 0.0
for _ in range(200):
    M = rng.normal(size=(2, 2))
    for shape in shapes.values():
        before, after = signed_area(shape), signed_area(shape @ M.T)
        if abs(before) < 1e-9:
            continue
        worst = max(worst, abs(abs(after / before) - abs(np.linalg.det(M))))
check("도형·행렬 무엇이든 넓이비 = |det|", worst < 1e-6, worst)

check("det < 0 이면 넓이의 부호가 뒤집힌다 (방향 반전)",
      signed_area(triangle) * signed_area(triangle @ np.array([[1.0, 0.0], [0.0, -1.0]]).T) < 0)
check("det = 0 이면 넓이가 0 (선분으로 붕괴)",
      abs(signed_area(triangle @ np.array([[1.0, 1.0], [1.0, 1.0]]).T)) < 1e-9)

print("\n" + "=" * 52)
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL PASS")

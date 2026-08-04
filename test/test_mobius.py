"""섹션 8의 수학 검증 — 뫼비우스 변환.

이 섹션의 주장은 정리다: **원과 직선은 원과 직선으로 간다.** 그래서 검증도
그것을 직접 확인한다 — 원을 보낸 뒤 그 상이 정말 원(또는 직선)인지 재 본다.

    python test/test_mobius.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from section8_mobius import fixed_points, mobius  # noqa: E402

failures = []
rng = np.random.default_rng(20260804)


def check(label, condition, detail=""):
    condition = bool(condition)
    text = str(detail)
    print(f"  [{'OK  ' if condition else 'FAIL'}] {label}"
          f"{f' — {text}' if (not condition and text) else ''}")
    if not condition:
        failures.append(label)


def circle(center, radius, n=400):
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return center + radius * np.exp(1j * angle)


def fit_circle_or_line(w):
    """점들이 원 위에 있으면 (중심, 반지름), 직선 위에 있으면 None 을 돌려준다.

    원·직선을 한꺼번에 다루려면 |z|² 항을 포함한 일반형을 쓴다.

        A(x² + y²) + Bx + Cy + D = 0

    A = 0 이면 직선, 아니면 원이다. 최소 특이벡터로 계수를 찾는다.
    """
    w = w[np.isfinite(w)]
    x, y = w.real, w.imag
    M = np.column_stack([x ** 2 + y ** 2, x, y, np.ones_like(x)])
    _, singular, vh = np.linalg.svd(M, full_matrices=False)
    A, B, C, D = vh[-1]
    residual = singular[-1] / max(singular[0], 1e-300)
    if abs(A) < 1e-8:
        return None, residual                      # 직선
    center = complex(-B / (2 * A), -C / (2 * A))
    radius = np.sqrt(max((B ** 2 + C ** 2) / (4 * A ** 2) - D / A, 0.0))
    return (center, radius), residual


print("=== 1. 원은 원 또는 직선으로 간다 (이 섹션의 정리) ===")
worst = 0.0
for _ in range(300):
    a, b, c, d = (rng.normal() + 1j * rng.normal() for _ in range(4))
    if abs(a * d - b * c) < 1e-3:
        continue
    center, radius = rng.normal() + 1j * rng.normal(), abs(rng.normal()) + 0.3
    z = circle(center, radius)
    w = mobius(z, a, b, c, d)
    w = w[np.isfinite(w)]
    if len(w) < 50 or np.max(np.abs(w)) > 1e6:
        continue
    _, residual = fit_circle_or_line(w)
    worst = max(worst, residual)
check("무작위 300개에서 상이 정확히 원 또는 직선", worst < 1e-8, worst)

print("\n=== 2. 직선도 원 또는 직선으로 간다 ===")
worst = 0.0
for _ in range(300):
    a, b, c, d = (rng.normal() + 1j * rng.normal() for _ in range(4))
    if abs(a * d - b * c) < 1e-3:
        continue
    t = np.linspace(-40, 40, 800)
    z = (rng.normal() + 1j * rng.normal()) + t * np.exp(1j * rng.uniform(0, np.pi))
    w = mobius(z, a, b, c, d)
    w = w[np.isfinite(w)]
    if len(w) < 50 or np.max(np.abs(w)) > 1e6:
        continue
    _, residual = fit_circle_or_line(w)
    worst = max(worst, residual)
check("무작위 300개에서 상이 정확히 원 또는 직선", worst < 1e-6, worst)

print("\n=== 3. 1/z — 원점을 지나는 원만 직선이 된다 ===")
shape, _ = fit_circle_or_line(mobius(circle(0j, 1.0), 0, 1, 1, 0))
check("원점 중심 단위원 → 원 (직선 아님)", shape is not None, shape)

# 원점을 지나는 원: 중심 1, 반지름 1
z = circle(1 + 0j, 1.0)
z = z[np.abs(z) > 1e-6]                # 원점 자신은 무한대로 간다
shape, residual = fit_circle_or_line(mobius(z, 0, 1, 1, 0))
check("원점을 지나는 원 → 직선", shape is None, shape)
check("그 직선이 정확하다", residual < 1e-8, residual)

# 원점을 지나지 않는 원은 원으로 남는다.
shape, _ = fit_circle_or_line(mobius(circle(3 + 0j, 1.0), 0, 1, 1, 0))
check("원점을 지나지 않는 원 → 원", shape is not None, shape)

print("\n=== 4. 고정점 ===")
worst = 0.0
for _ in range(300):
    a, b, c, d = (rng.normal() + 1j * rng.normal() for _ in range(4))
    if abs(a * d - b * c) < 1e-3 or abs(c) < 1e-3:
        continue
    for point in fixed_points(a, b, c, d):
        image = mobius(np.array([point]), a, b, c, d)[0]
        if np.isfinite(image):
            worst = max(worst, abs(image - point) / max(1.0, abs(point)))
check("f(z*) = z* (상대오차)", worst < 1e-6, worst)

check("c=0, a≠d → 고정점 1개 (나머지는 무한대)",
      len(fixed_points(2 + 0j, 1 + 0j, 0j, 1 + 0j)) == 1)
check("c=0, a=d → 평행이동이라 유한한 고정점 없음",
      fixed_points(1 + 0j, 3 + 0j, 0j, 1 + 0j) == [])
check("1/z 의 고정점은 ±1",
      sorted(abs(p - s) < 1e-9 for p in fixed_points(0j, 1 + 0j, 1 + 0j, 0j)
             for s in (1 + 0j, -1 + 0j)).count(True) == 2)

print("\n=== 5. 퇴화: ad−bc=0 이면 상이 한 점 ===")
w = mobius(circle(0j, 1.0), 1 + 0j, 2 + 0j, 2 + 0j, 4 + 0j)
w = w[np.isfinite(w)]
check("모든 점이 같은 곳으로 간다", np.ptp(w.real) < 1e-9 and np.ptp(w.imag) < 1e-9,
      f"{np.ptp(w.real):.2e}, {np.ptp(w.imag):.2e}")
check("그 값은 a/c", abs(w[0] - 0.5) < 1e-9, w[0])

print("\n=== 6. 극과 방어 ===")
w = mobius(np.array([0j]), 0j, 1 + 0j, 1 + 0j, 0j)   # 1/0
check("극에서 NaN 을 돌려준다 (예외 아님)", np.isnan(w[0]))

w = mobius(np.array([-1 + 0j, 0j, 1 + 0j]), 1 + 0j, 0j, 1 + 0j, 1 + 0j)  # z/(z+1)
check("극 옆의 점들은 멀쩡하다", np.isnan(w[0]) and np.all(np.isfinite(w[1:])), w)

w = mobius(np.array([1e300 + 0j]), 1 + 0j, 0j, 1 + 0j, 1 + 0j)
check("아주 큰 입력도 유한하거나 NaN", np.isnan(w[0]) or np.isfinite(w[0]), w)

check("c=d=0 이어도 죽지 않는다",
      np.all(np.isnan(mobius(np.array([1 + 0j]), 1 + 0j, 1 + 0j, 0j, 0j))))

print("\n" + "=" * 52)
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL PASS")

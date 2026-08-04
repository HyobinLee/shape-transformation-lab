"""섹션 7(과 §4)의 수학 검증 — 고정점과 궤도.

이 섹션의 결론은 한 줄이다:

    z_n − z* = ρⁿ (z₀ − z*)

**고정점에서 재면 이 변환은 그냥 곱하기 하나다.** 그 한 줄에서 수렴·회전·발산이
모두 따라 나오므로, 검증도 그것을 직접 겨눈다.

    python test/test_orbit.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from section4_rotation_translation import fixed_point as fixed_point_s4  # noqa: E402
from section7_orbit import (  # noqa: E402
    ESCAPE, PRESETS, fixed_point, julia_escape, orbit,
)

failures = []


def check(label, condition, detail=""):
    condition = bool(condition)
    text = str(detail)
    print(f"  [{'OK  ' if condition else 'FAIL'}] {label}"
          f"{f' — {text}' if (not condition and text) else ''}")
    if not condition:
        failures.append(label)


def rho_of(preset):
    return preset["r"] * np.exp(1j * np.radians(preset["theta"]))


print("=== 1. 고정점이 정말 제자리에 남는가 ===")
rng = np.random.default_rng(20260804)
worst = 0.0
for _ in range(400):
    rho = rng.normal() + 1j * rng.normal()
    alpha = rng.normal() + 1j * rng.normal()
    beta = rng.normal() + 1j * rng.normal()
    star = fixed_point(rho, alpha, beta)
    if star is None:
        continue
    worst = max(worst, abs(rho * (star + alpha) + beta - star))
check("무작위 400개에서 f(z*) = z*", worst < 1e-8, worst)

print("\n=== 2. §4 와 §7 의 공식이 같은 답을 내는가 (의도된 중복) ===")
same = all(
    (fixed_point(r, a, b) is None and fixed_point_s4(r, a, b) is None)
    or abs(fixed_point(r, a, b) - fixed_point_s4(r, a, b)) < 1e-12
    for r, a, b in [(rng.normal() + 1j * rng.normal(),
                     rng.normal() + 1j * rng.normal(),
                     rng.normal() + 1j * rng.normal()) for _ in range(100)]
)
check("두 섹션의 fixed_point 가 일치", same)

print("\n=== 3. ρ = 1 이면 고정점이 없다 (평행이동에는 중심이 없다) ===")
check("ρ = 1 → None", fixed_point(1 + 0j, 2 + 3j, -1j) is None)
check("θ = 0, r = 1 → None", fixed_point(rho_of(PRESETS["평행이동 (중심 없음)"]),
                                         0.6 + 0.4j, 0j) is None)
check("ρ = 1 에 아주 가까워도 죽지 않는다",
      fixed_point(1 + 1e-13j, 1 + 0j, 0j) is None)

print("\n=== 4. 이 섹션의 결론: z_n − z* = ρⁿ (z₀ − z*) ===")
rho, alpha, beta = 0.9 * np.exp(1j * 0.6), 1 + 0j, -1j
star = fixed_point(rho, alpha, beta)
path, diverged = orbit(2 + 1j, rho, alpha, beta, 60)
predicted = star + rho ** np.arange(len(path)) * (path[0] - star)
check("궤도가 공식과 일치", np.allclose(path, predicted, atol=1e-9),
      float(np.max(np.abs(path - predicted))))

distance = np.abs(path - star)
ratios = distance[1:] / distance[:-1]
check("|z_{n+1}−z*| / |z_n−z*| 가 늘 |ρ| (계기판의 주장)",
      np.allclose(ratios, abs(rho), atol=1e-9), float(np.std(ratios)))

print("\n=== 5. |ρ| 가 운명을 가르는가 ===")
path, _ = orbit(2 + 1j, 0.8 * np.exp(0.5j), 1 + 0j, -1j, 200)
star = fixed_point(0.8 * np.exp(0.5j), 1 + 0j, -1j)
check("|ρ| < 1 → 고정점으로 수렴", abs(path[-1] - star) < 1e-6, abs(path[-1] - star))

rho = np.exp(1j * np.pi / 2)              # 90°, 네 번이면 제자리
path, _ = orbit(2 + 1j, rho, 0j, 2 + 0j, 8)
check("|ρ| = 1, θ=90° → 4번마다 닫힌다 (정사각형)",
      abs(path[4] - path[0]) < 1e-9 and abs(path[8] - path[0]) < 1e-9,
      abs(path[4] - path[0]))

rho = np.exp(1j * 2 * np.pi / np.sqrt(2))  # 무리수 배 → 영원히 안 닫힌다
path, _ = orbit(2 + 0j, rho, 0j, 0j, 400)
closest = min(abs(path[k] - path[0]) for k in range(1, len(path)))
check("무리수 배 → 400번을 돌아도 제자리로 안 온다", closest > 1e-3, closest)

path, diverged = orbit(2 + 1j, 1.2 * np.exp(0.3j), 0.5 + 0j, 0j, 500)
check("|ρ| > 1 → 발산하고, 발산했다고 알려준다", diverged)
check("발산해도 유한한 값만 남긴다",
      np.all(np.isfinite(path)) and np.all(np.abs(path) <= ESCAPE),
      float(np.max(np.abs(path))))

print("\n=== 6. 퇴화 방어 ===")
for label, kwargs in (
    ("r = 0 (한 점으로 붕괴)", dict(rho=0j, alpha=1 + 0j, beta=2 + 0j)),
    ("아주 큰 배율", dict(rho=1e6 + 0j, alpha=0j, beta=0j)),
    ("아주 작은 배율", dict(rho=1e-12 + 0j, alpha=0j, beta=1 + 1j)),
):
    try:
        path, _ = orbit(1 + 1j, kwargs["rho"], kwargs["alpha"], kwargs["beta"], 100)
        ok = np.all(np.isfinite(path)) and len(path) >= 1
    except Exception as exc:                    # noqa: BLE001
        ok, path = False, exc
    check(f"{label} → 죽지 않는다", ok, path)

path, _ = orbit(0j, 0j, 0j, 0j, 50)
check("모든 값이 0이어도 죽지 않는다", np.all(path == 0))

print("\n=== 7. 줄리아 집합 (심화 토글) ===")
axis, escaped = julia_escape(-0.4 + 0.6j, span=1.7, size=120, max_iter=40)
check("탈출시간이 유한하고 범위 안", np.all(np.isfinite(escaped))
      and escaped.min() >= 0 and escaped.max() <= 40)
check("발산하지 않는 점이 존재한다 (집합이 비지 않았다)", np.any(escaped == 40))
check("멀리 있는 점은 곧바로 발산한다", escaped[0, 0] < 5, escaped[0, 0])

_, filled = julia_escape(0j, span=1.7, size=120, max_iter=40)
# c = 0 이면 줄리아 집합은 정확히 단위원 내부다.
inside = filled[60, 60] == 40                    # 원점
check("c = 0 → 원점은 집합 안", inside)

print("\n" + "=" * 52)
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL PASS")

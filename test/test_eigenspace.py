"""섹션 5의 수학 검증 — 고유공간 분류와 어긋난 각 곡선.

이 섹션의 중심 주장은 **"g(θ) 곡선의 모양만으로 고유공간의 정체가 전부
읽힌다"** 이다. 그러니 검증도 그 주장 자체를 겨눈다 — 네 가지 경우에서
곡선이 각각 가로지르는지·닿는지·안 닿는지·평평한지.

    python test/test_eigenspace.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from section5_eigenspace import (  # noqa: E402
    PRESETS, count_zero_crossings, deviation_curve, eigen_structure,
)

failures = []
#: g 는 π 주기이므로 끝점을 넣지 않는다 — 넣으면 같은 직선을 두 번 세게 된다.
THETA = np.linspace(0, np.pi, 2000, endpoint=False)


def check(label, condition, detail=""):
    print(f"  [{'OK  ' if condition else 'FAIL'}] {label}"
          f"{(' — ' + str(detail)) if detail and not condition else ''}")
    if not condition:
        failures.append(label)


print("=== 1. 네 가지 경우를 분류하는가 ===")
expected = {
    "두 직선": ('two_lines', 1),
    "결손(전단)": ('defective', 1),
    "평면 전체": ('plane', 2),
    "고유공간 없음": ('none', 0),
    "λ=0 포함": ('two_lines', 1),
}
for name, (kind, dim) in expected.items():
    s = eigen_structure(PRESETS[name])
    check(f"{name} → {kind}", s['kind'] == kind and s['dim'] == dim,
          f"{s['kind']}/{s['dim']}")

print("\n=== 2. g(θ) 곡선의 모양이 분류와 맞는가 ===")
g = deviation_curve(PRESETS["두 직선"], THETA)
check("두 직선 → 0을 가로지름 2번", count_zero_crossings(g) == 2, count_zero_crossings(g))

g = deviation_curve(PRESETS["결손(전단)"], THETA)
touches = np.min(np.abs(g)) < 1e-6
check("결손 → 0에 닿는다", touches, np.min(np.abs(g)))
check("결손 → 그런데 가로지르지는 않는다 (접함)", count_zero_crossings(g) == 0,
      count_zero_crossings(g))

g = deviation_curve(PRESETS["고유공간 없음"], THETA)
check("회전 → 0에 한 번도 안 닿는다", np.min(np.abs(g)) > 0.5, np.min(np.abs(g)))

g = deviation_curve(PRESETS["평면 전체"], THETA)
check("평면 전체 → 항상 0", np.allclose(g, 0.0), np.max(np.abs(g)))

# 음수 배율도 직선은 그대로 둔다. 각을 ±90°로 접는 이유가 이것이다.
g = deviation_curve(-2 * np.eye(2), THETA)
check("음의 배율(-2I) → 그래도 항상 0", np.allclose(g, 0.0), np.max(np.abs(g)))

print("\n=== 3. 찾은 고유방향이 정말 고유방향인가 ===")
for name in ("두 직선", "결손(전단)", "λ=0 포함"):
    A = PRESETS[name]
    s = eigen_structure(A)
    ok = True
    for v in s['directions']:
        Av = A @ v
        cross = v[0] * Av[1] - v[1] * Av[0]   # 평행하면 0
        ok = ok and abs(cross) < 1e-9
    check(f"{name}: Av 가 v 와 평행", ok and len(s['directions']) > 0,
          len(s['directions']))

print("\n=== 4. numpy 의 답과 일치하는가 ===")
rng = np.random.default_rng(20260804)
mismatch = 0
for _ in range(300):
    A = rng.normal(size=(2, 2)) * rng.choice([0.5, 1.0, 4.0])
    s = eigen_structure(A)
    numpy_real = np.count_nonzero(np.abs(np.linalg.eigvals(A).imag) < 1e-9)
    ours = 0 if s['kind'] == 'none' else 2
    if ours != numpy_real:
        mismatch += 1
check("무작위 300개에서 실고유값 유무가 일치", mismatch == 0, f"{mismatch}건 불일치")

print("\n=== 5. g 의 영점이 고유방향과 같은 자리인가 ===")
worst = 0.0
for _ in range(200):
    A = rng.normal(size=(2, 2))
    s = eigen_structure(A)
    if s['kind'] != 'two_lines':
        continue
    g = deviation_curve(A, THETA)
    zeros = THETA[np.where(np.diff(np.sign(g)) != 0)[0]]
    for v in s['directions']:
        angle = np.arctan2(v[1], v[0]) % np.pi
        if len(zeros):
            worst = max(worst, float(np.min(np.abs(zeros - angle))))
check("영점 위치가 고유방향과 일치 (오차 < 0.01 rad)", worst < 0.01, worst)

print("\n=== 6. 퇴화 방어 ===")
for label, A in (("영행렬", np.zeros((2, 2))),
                 ("항등행렬", np.eye(2)),
                 ("아주 큰 행렬", np.full((2, 2), 1e12)),
                 ("아주 작은 행렬", np.full((2, 2), 1e-12))):
    try:
        s = eigen_structure(A)
        g = deviation_curve(A, THETA)
        ok = s['kind'] in ('two_lines', 'defective', 'plane', 'none') and np.all(np.isfinite(g))
    except Exception as exc:                      # noqa: BLE001
        ok, s = False, exc
    check(f"{label} → 죽지 않고 분류", ok, s)

check("영행렬은 평면 전체 (모든 방향이 λ=0)",
      eigen_structure(np.zeros((2, 2)))['kind'] == 'plane')

print("\n" + "=" * 52)
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL PASS")

"""섹션 6의 수학 검증 — 등거리변환의 분류.

이 섹션의 주장은 정리 그 자체다:

    평면의 모든 등거리변환은 평행이동·회전·대칭·미끄럼대칭 넷 중 하나이고,
    거울 개수의 홀짝이 방향 보존 여부를 가른다.

그래서 검증도 정리를 직접 확인한다 — 평행한 두 축이 정말 간격의 2배만큼
평행이동을 만드는지, 교차하는 두 축이 정말 사잇각의 2배만큼 교점 둘레로
돌리는지, 거울 셋이 정말 넷 밖으로 나가지 않는지.

    python test/test_isometry.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from section6_isometry import apply, classify_isometry, reflection  # noqa: E402

failures = []
rng = np.random.default_rng(20260804)
PROBE = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.3, -1.7], [-3.1, 0.4]])


def check(label, condition, detail=""):
    condition = bool(condition)
    text = str(detail)
    print(f"  [{'OK  ' if condition else 'FAIL'}] {label}"
          f"{f' — {text}' if (not condition and text) else ''}")
    if not condition:
        failures.append(label)


def compose(mirrors):
    total = np.eye(3)
    for angle, offset in mirrors:
        total = reflection(angle, offset) @ total
    return total


print("=== 1. 반사가 정말 등거리변환인가 ===")
worst = 0.0
for _ in range(200):
    M = reflection(rng.uniform(-90, 90), rng.uniform(-4, 4))
    moved = apply(M, PROBE)
    for i in range(len(PROBE)):
        for j in range(i + 1, len(PROBE)):
            before = np.linalg.norm(PROBE[i] - PROBE[j])
            after = np.linalg.norm(moved[i] - moved[j])
            worst = max(worst, abs(before - after))
check("거리를 보존한다", worst < 1e-9, worst)

worst = 0.0
for _ in range(200):
    angle, offset = rng.uniform(-90, 90), rng.uniform(-4, 4)
    M = reflection(angle, offset)
    worst = max(worst, float(np.max(np.abs(apply(M, apply(M, PROBE)) - PROBE))))
check("두 번 반사하면 제자리 (M² = I)", worst < 1e-9, worst)

print("\n=== 2. 축 위의 점은 움직이지 않는가 ===")
worst = 0.0
for _ in range(200):
    angle, offset = rng.uniform(-90, 90), rng.uniform(-4, 4)
    phi = np.radians(angle)
    on_axis = (offset * np.array([-np.sin(phi), np.cos(phi)])
               + rng.uniform(-5, 5) * np.array([np.cos(phi), np.sin(phi)]))
    worst = max(worst, float(np.max(np.abs(apply(reflection(angle, offset),
                                                 on_axis[None, :]) - on_axis))))
check("축 위의 점은 제자리", worst < 1e-9, worst)

print("\n=== 3. 평행한 두 축 → 평행이동, 거리는 간격의 2배 ===")
for angle in (0.0, 30.0, -75.0):
    for d1, d2 in ((-1.0, 1.0), (0.5, 2.0), (-3.0, -0.5)):
        result = classify_isometry(compose([(angle, d1), (angle, d2)]))
        distance = float(np.hypot(*result['vector'])) if result['kind'] == 'translation' else -1
        check(f"φ={angle:g}, d={d1:g}→{d2:g} : 평행이동 거리 = 2|Δd|",
              result['kind'] == 'translation'
              and abs(distance - 2 * abs(d2 - d1)) < 1e-9,
              f"{result['kind']} / {distance:.4f} vs {2 * abs(d2 - d1):.4f}")

print("\n=== 4. 교차하는 두 축 → 회전, 각은 사잇각의 2배, 중심은 교점 ===")
for phi1, phi2 in ((0.0, 45.0), (20.0, -35.0), (-60.0, 10.0)):
    result = classify_isometry(compose([(phi1, 0.0), (phi2, 0.0)]))
    expected = ((2 * (phi2 - phi1)) + 180) % 360 - 180
    check(f"φ={phi1:g}→{phi2:g} : 회전각 = 2(φ₂−φ₁)",
          result['kind'] == 'rotation' and abs(result['angle'] - expected) < 1e-7,
          f"{result.get('angle')} vs {expected}")
    check(f"φ={phi1:g}→{phi2:g} : 중심이 원점(교점)",
          np.allclose(result['center'], 0, atol=1e-9), result.get('center'))

# 원점에서 만나지 않는 경우 — §2 로는 만들 수 없던 상황.
phi1, phi2, d1 = 0.0, 60.0, 0.0
# 두 축의 교점을 (2, 0) 으로 만든다: 축1 은 x축, 축2 는 (2,0) 을 지나는 60° 선.
phi = np.radians(phi2)
d2 = float(np.array([2.0, 0.0]) @ np.array([-np.sin(phi), np.cos(phi)]))
result = classify_isometry(compose([(phi1, d1), (phi2, d2)]))
check("교점이 원점이 아니어도 중심 = 교점",
      result['kind'] == 'rotation' and np.allclose(result['center'], [2.0, 0.0], atol=1e-9),
      result.get('center'))

print("\n=== 5. 거울 1개 → 원래 축을 복원하는가 ===")
worst_angle = worst_point = 0.0
for _ in range(200):
    angle, offset = rng.uniform(-89, 89), rng.uniform(-4, 4)
    result = classify_isometry(reflection(angle, offset))
    if result['kind'] != 'reflection':
        worst_angle = 999
        break
    difference = (result['axis_angle'] - angle + 90) % 180 - 90
    worst_angle = max(worst_angle, abs(difference))
    # 복원한 축이 원래 축과 같은 직선인지: 원래 축 위의 점이 그 위에 있는가
    phi = np.radians(angle)
    normal = np.array([-np.sin(phi), np.cos(phi)])
    worst_point = max(worst_point, abs(result['axis_point'] @ normal - offset))
check("축의 각도를 복원", worst_angle < 1e-7, worst_angle)
check("축의 위치를 복원", worst_point < 1e-9, worst_point)
check("미끄럼 성분이 0 (순수 대칭)",
      classify_isometry(reflection(37.0, 1.5))['glide_length'] < 1e-9)

print("\n=== 6. 거울 3개 → 대칭 또는 미끄럼대칭 (새 종류는 없다) ===")
kinds = set()
for _ in range(400):
    mirrors = [(rng.uniform(-90, 90), rng.uniform(-4, 4)) for _ in range(3)]
    kinds.add(classify_isometry(compose(mirrors))['kind'])
check("거울 3개는 늘 방향 반전형", kinds <= {'reflection', 'glide'}, kinds)
check("미끄럼대칭이 실제로 나온다", 'glide' in kinds, kinds)

print("\n=== 7. 홀짝이 방향 보존을 가르는가 ===")
ok = True
for count in (1, 2, 3, 4, 5):
    for _ in range(60):
        mirrors = [(rng.uniform(-90, 90), rng.uniform(-4, 4)) for _ in range(count)]
        result = classify_isometry(compose(mirrors))
        ok = ok and (result['preserves'] == (count % 2 == 0))
check("짝수 → 보존, 홀수 → 반전", ok)

print("\n=== 8. 분류가 넷 밖으로 나가지 않는가 ===")
kinds = set()
for count in (1, 2, 3, 4, 5, 6):
    for _ in range(120):
        mirrors = [(rng.uniform(-90, 90), rng.uniform(-4, 4)) for _ in range(count)]
        kinds.add(classify_isometry(compose(mirrors))['kind'])
check("네 가지뿐", kinds <= {'translation', 'rotation', 'reflection', 'glide'}, kinds)

# 무작위로는 평행이동이 **나올 수 없다.** 두 축이 정확히 평행해야 하는데 그럴
# 확률은 0이기 때문이다(측도 0). 학생이 슬라이더로 각도를 맞춰야만 보게 되는
# 경우이므로, 여기서도 일부러 평행하게 놓아 확인한다.
check("무작위 표본에는 평행이동이 없다 (정확히 평행할 확률 0)",
      'translation' not in kinds, kinds)
for _ in range(60):
    angle = rng.uniform(-90, 90)
    mirrors = [(angle, rng.uniform(-4, 4)), (angle, rng.uniform(-4, 4))]
    kinds.add(classify_isometry(compose(mirrors))['kind'])
check("일부러 평행하게 놓으면 평행이동이 나온다", 'translation' in kinds, kinds)
check("그러면 네 가지가 모두 나온다", len(kinds) == 4, kinds)

print("\n=== 9. 분류가 실제 변환과 맞는가 (되돌려 확인) ===")
worst = 0.0
for _ in range(400):
    count = int(rng.integers(1, 5))
    mirrors = [(rng.uniform(-90, 90), rng.uniform(-4, 4)) for _ in range(count)]
    M = compose(mirrors)
    result = classify_isometry(M)
    if result['kind'] == 'rotation':
        # 중심은 정말 제자리에 남아야 한다.
        center = result['center']
        worst = max(worst, float(np.max(np.abs(apply(M, center[None, :]) - center))))
    elif result['kind'] == 'translation':
        worst = max(worst, float(np.max(np.abs(apply(M, PROBE) - (PROBE + result['vector'])))))
    else:
        # 축 위의 점은 미끄럼 벡터만큼만 움직여야 한다.
        phi = np.radians(result['axis_angle'])
        direction = np.array([np.cos(phi), np.sin(phi)])
        on_axis = result['axis_point'] + rng.uniform(-3, 3) * direction
        expected = on_axis + result['glide']
        worst = max(worst, float(np.max(np.abs(apply(M, on_axis[None, :]) - expected))))
check("회전 중심·이동벡터·축이 모두 실제와 일치", worst < 1e-7, worst)

print("\n=== 10. 퇴화 방어 ===")
check("같은 거울 두 번 → 항등(평행이동 0)",
      classify_isometry(compose([(30.0, 1.0), (30.0, 1.0)]))['kind'] == 'translation')
check("같은 거울 두 번 → 이동량 0",
      np.allclose(classify_isometry(compose([(30.0, 1.0), (30.0, 1.0)]))['vector'], 0, atol=1e-9))
check("아주 가까운 두 축도 죽지 않는다",
      classify_isometry(compose([(0.0, 0.0), (1e-9, 0.0)]))['kind']
      in ('translation', 'rotation'))
check("90° 축(수직선)도 처리",
      classify_isometry(reflection(90.0, 2.0))['kind'] == 'reflection')

print("\n" + "=" * 52)
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL PASS")

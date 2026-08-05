"""§1 단계별 탐구가 내세우는 **문장**을 잰다.

이 앱에서 가장 나쁜 실패는 화면에 적어 둔 수학이 틀리는 것이다. 실제로 이 기능의
최초 설계안에는 거짓 명제가 셋 있었다 — "일차변환은 도형의 종류를 보존한다"(원은
타원이 된다), 상직선의 기울기가 $a_{11}/a_{21}$(뒤집혔다), "붕괴직선은 상직선과
직교한다"(대칭행렬에서만 참). 셋 다 **대칭행렬이나 특수한 예만 보면 참으로 보인다.**
그래서 코드가 아니라 주장을 겨눠서, 난수로 매번 다시 잰다.

    python test/test_section1_discovery.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import section1_transformation_by_matrix as s1   # noqa: E402

FAILURES = []


def check(label, condition):
    print(("  PASS  " if condition else "  FAIL  ") + label)
    if not condition:
        FAILURES.append(label)


M = s1.MATRICES
RNG = np.random.default_rng(20260805)


def section(title):
    print(f"\n[{title}]")


# ──────────────────────────────────────────────────────────────────
section("1단계 — 무엇이 살아남는가")

sigma = np.linalg.svd(M["A1"], compute_uv=False)
check("A1 은 가역이고 닮음변환이 아니다 (길이·각이 보존되지 않는다)",
      abs(np.linalg.det(M["A1"])) > 1e-9 and abs(sigma[0] - sigma[1]) > 1e-9)
check("A1 은 대칭행렬이 아니다 (대칭이면 학생이 특수한 예만 보게 된다)",
      M["A1"][0, 1] != M["A1"][1, 0])

theta = np.linspace(0, 2 * np.pi, 721)
circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
for name, expect_circle in (("A1", False), ("R", True)):
    radii = np.linalg.norm(circle @ M[name].T, axis=1)
    stays_circle = np.allclose(radii, radii[0], atol=1e-9)
    check(f"{name}: 원의 상이 {'원' if expect_circle else '타원'} 이다 "
          f"(반축비 {radii.max() / radii.min():.3f})", stays_circle == expect_circle)

check("F 는 det < 0 이다 (방향이 뒤집히는 장면)", np.linalg.det(M["F"]) < 0)

for name in ("A1", "R", "F"):
    p, q, u = (RNG.uniform(-6, 6, size=(50, 2)) for _ in range(3))
    first = (p + u) @ M[name].T - p @ M[name].T
    second = (q + u) @ M[name].T - q @ M[name].T
    check(f"{name}: 평행한 두 선분의 상이 여전히 평행하다",
          np.allclose(s1.cross2(first, second), 0, atol=1e-8))

# ──────────────────────────────────────────────────────────────────
section("2단계 — 상은 원점을 지나는 직선 하나")

EXPECTED_IMAGE_SLOPE = {"A2": 2.0, "B": -2.0, "C": None, "S": 2.0, "D": 3.0}
for name, slope in EXPECTED_IMAGE_SLOPE.items():
    matrix = M[name]
    check(f"{name}: det = 0", s1.is_singular(matrix))
    points = RNG.uniform(-9, 9, size=(400, 2)) @ matrix.T
    direction = s1.image_direction(matrix)
    check(f"{name}: 어떤 점을 넣어도 상이 한 직선 위에 있다(원점 통과)",
          np.allclose(s1.cross2(points, direction), 0, atol=1e-9))
    actual = s1.slope_of(direction)
    check(f"{name}: 상직선의 기울기 = a21/a11 = {slope}",
          (actual is None and slope is None) or
          (actual is not None and slope is not None and abs(actual - slope) < 1e-9))
    check(f"{name}: 유계 도형(원)의 상이 선분이다",
          np.allclose(s1.cross2(circle @ matrix.T, direction), 0, atol=1e-9))

check("C 의 상직선은 x = 0 이다", s1.slope_of(s1.image_direction(M["C"])) is None)
check("고정행렬 중 영행렬은 없다",
      all(np.linalg.norm(M[name]) > 1e-9 for name in M))

# ──────────────────────────────────────────────────────────────────
section("3단계 — 한 점으로 붕괴할 조건")

EXPECTED_COLLAPSE_SLOPE = {"A2": -2.0, "B": -1 / 3, "S": -0.5, "D": -0.5}
t = np.linspace(-20, 20, 300)[:, None]
for name, slope in EXPECTED_COLLAPSE_SLOPE.items():
    matrix = M[name]
    check(f"{name}: 붕괴 기울기 = -a11/a12 = {slope:+.4g}",
          abs(s1.slope_of(s1.kernel_direction(matrix)) - slope) < 1e-9)
    check(f"{name}: 둘째 줄로 만든 -a21/a22 도 같은 값이다",
          abs(-matrix[1, 0] / matrix[1, 1] - slope) < 1e-9)

    direction = s1.kernel_direction(matrix)
    for base in ([1.0, 0.5], [-3.0, 2.0], [4.0, -1.0]):
        image = (np.array(base) + t * direction) @ matrix.T
        check(f"{name}: P={base} 에서도 한 점으로 붕괴한다 (위치와 무관)",
              np.allclose(image, image[0], atol=1e-9))
        check(f"{name}: P={base} 의 붕괴점이 2단계의 상직선 위에 있다",
              abs(float(s1.cross2(image[0], s1.image_direction(matrix)))) < 1e-9)

    off = np.array([1.0, slope + 0.25])          # 임계각을 살짝 벗어나면
    image = (np.array([1.0, 0.5]) + t * off) @ matrix.T
    check(f"{name}: 임계 방향이 아니면 점이 되지 않는다",
          not np.allclose(image, image[0], atol=1e-6))

# 오개념을 죽이는 한 쌍
perpendicular = {name: abs(float(s1.image_direction(M[name]) @ s1.kernel_direction(M[name]))) < 1e-9
                 for name in ("S", "D", "A2", "B")}
check("S(대칭)에서는 상직선과 붕괴직선이 직교한다", perpendicular["S"])
check("D(비대칭)에서는 직교하지 않는다 — 반례 국면이 성립한다", not perpendicular["D"])
check("S 와 D 는 첫 줄이 같아 붕괴 기울기가 같다",
      np.allclose(M["S"][0], M["D"][0]))
check("2·3단계 관찰행렬 A2 에서도 직교하지 않는다", not perpendicular["A2"])

# H5 가 성립할 조건 — A2 와 B 를 견줘서 규칙이 보이려면 둘 다 달라야 한다
check("A2 와 B 는 상 기울기가 다르다",
      s1.slope_of(s1.image_direction(M["A2"])) != s1.slope_of(s1.image_direction(M["B"])))
check("A2 와 B 는 붕괴 기울기가 다르다 (같으면 H5 의 비교가 무의미해진다)",
      abs(s1.slope_of(s1.kernel_direction(M["A2"]))
          - s1.slope_of(s1.kernel_direction(M["B"]))) > 1e-9)
check("B 는 핵과 상이 일치하지 않는다",
      abs(float(s1.cross2(s1.image_direction(M["B"]), s1.kernel_direction(M["B"])))) > 1e-9)

# §5 로 가는 다리
for name in ("A2", "B", "S", "D"):
    matrix, v = M[name], s1.image_direction(M[name])
    check(f"{name}: 상직선은 고유값 tr(A) 의 고유공간이다",
          np.allclose(matrix @ v, matrix.trace() * v))

# ──────────────────────────────────────────────────────────────────
section("검증기 — 학생의 주장을 숫자로 판정한다")

check("A2 에서 '기울기 2' 는 맞다", s1.check_claim("S2_SLOPE", 2.0, M["A2"]) is True)
check("A2 에서 '기울기 0.5' 는 틀리다", s1.check_claim("S2_SLOPE", 0.5, M["A2"]) is False)
check("A2 에서 '붕괴 기울기 -2' 는 맞다", s1.check_claim("S3_SLOPE", -2.0, M["A2"]) is True)
check("A1(가역)에서 '선분이 된다' 는 틀리다", s1.check_claim("S2_FLAT", True, M["A1"]) is False)
check("A1 에서 '원은 타원이 된다' 는 맞다", s1.check_claim("S1_CIRCLE", "타원", M["A1"]) is True)
check("R(회전)에서 '원은 타원이 된다' 는 틀리다", s1.check_claim("S1_CIRCLE", "타원", M["R"]) is False)
check("S(대칭)에서 '직교한다' 는 맞다", s1.check_claim(s1.PERPENDICULAR, True, M["S"]) is True)
check("D(비대칭)에서 '직교한다' 는 틀리다", s1.check_claim(s1.PERPENDICULAR, True, M["D"]) is False)
check("D 에서 '직교하지 않는다' 는 맞다", s1.check_claim(s1.PERPENDICULAR, "아니다", M["D"]) is True)
check("A2 에서 '위치에 달렸다' 는 틀리다",
      s1.check_claim("S3_DIRECTION_ONLY", "위치", M["A2"]) is False)

# ──────────────────────────────────────────────────────────────────
section("힌트 사다리 — 마지막 칸 말고는 정답을 흘리지 않는다")

FORBIDDEN = {
    "1": ["타원"],
    "2-flat": ["선분", "붕괴"],
    "2-irreversible": ["되돌릴 수 없", "역행렬"],
    "2-oneline": ["같은 직선", "일치"],
    "2-line": ["원점", "a_{21}", "a_{11}"],
    "3": ["-a_{11}", "직교", "수직", "핵"],
}
for ladder, tokens in FORBIDDEN.items():
    for index, hint in enumerate(s1.HINTS[ladder][:-1]):        # 마지막 칸은 면제
        leaked = [token for token in tokens if token in hint]
        check(f"{ladder} H{index + 1} 이 정답 낱말을 흘리지 않는다"
              + (f" — {leaked}" if leaked else ""), not leaked)

check("모든 목표에 사다리가 배정되어 있다",
      all(gid in s1.LADDER_OF for stage in (1, 2, 3) for gid in s1.required_goals(stage)))
check("사다리 id 가 모두 실제로 존재한다",
      all(ladder in s1.HINTS for ladder in s1.LADDER_OF.values()))
check("모든 목표에 학생용 문장이 있다",
      all(gid in s1.GOAL_TEXT for stage in (1, 2, 3) for gid in s1.required_goals(stage)))

# ──────────────────────────────────────────────────────────────────
section("폴백 — 키가 없어도 대화가 굴러간다")

samples = [
    (1, "원을 넣었더니 원이 아니라 찌그러진 타원이 됐어", "S1_CIRCLE"),
    (2, "도형이 다 한 줄로 납작해져요", "S2_FLAT"),
    (2, "상이 전부 같은 직선 위에 있어", "S2_ONE_LINE"),
    (2, "그 직선은 원점을 지나", "S2_ORIGIN"),
    (2, "기울기가 2 인 것 같아", "S2_SLOPE"),
    (3, "직선이 한점으로 뭉쳐졌어", "S3_EXISTS"),
    (3, "위치는 상관없고 방향만 중요해", "S3_DIRECTION_ONLY"),
]
for stage, sentence, expected in samples:
    reading = s1._rule_based_claims(sentence, stage)
    ids = [claim["id"] for claim in reading["claims"]]
    check(f"[{stage}단계] “{sentence}” → {expected}", expected in ids)

reading = s1._rule_based_claims("기울기가 2 인 것 같아", 2)
value = next(c["value"] for c in reading["claims"] if c["id"] == "S2_SLOPE")
check("폴백이 뽑은 기울기 값이 검증기를 통과한다",
      s1.check_claim("S2_SLOPE", value, M["A2"]) is True)
check("답을 알려 달라는 요청을 알아본다",
      s1._rule_based_claims("그냥 답 알려줘", 2)["asks_answer"])

# ──────────────────────────────────────────────────────────────────
section("진행 판정 — 더 찾아낸 학생이 갇히지 않는다")

full = set(s1.required_goals(2))
check("필수를 다 찾으면 단계가 끝난다", s1.stage_complete(full, 2))
check("하나라도 빠지면 끝나지 않는다", not s1.stage_complete(full - {"S2_SLOPE"}, 2))

# 2단계에서 3단계의 함정(직선이 한 점이 됨)을 스스로 발견한 학생. 보너스가 같은
# 자리에 쌓이므로, 개수만 세면 진행 막대가 1을 넘고 3단계가 영영 열리지 않았다.
bonus = full | {s1.BONUS_COLLAPSE}
check("보너스를 찾아도 진행률이 1을 넘지 않는다",
      s1.found_count(bonus, 2) == len(s1.required_goals(2)))
check("보너스를 찾은 학생도 다음 단계가 열린다", s1.stage_complete(bonus, 2))
check("3단계 반례 명제도 진행률을 망가뜨리지 않는다",
      s1.found_count(set(s1.required_goals(3)) | {s1.PERPENDICULAR}, 3)
      == len(s1.required_goals(3)))

# ──────────────────────────────────────────────────────────────────
print()
if FAILURES:
    print(f"*** {len(FAILURES)} FAILED ***")
    for label in FAILURES:
        print("  -", label)
    sys.exit(1)
print("ALL PASS")

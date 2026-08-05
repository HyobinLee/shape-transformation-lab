"""(1) 행렬을 통한 일차변환 — 자유 탐구실 + 3단계 발견 경로.

이 섹션은 화면이 둘이다.

- **🔬 자유 탐구**: 예전부터 있던 실험실. 기저 상자·계기판·모든 캡션이 살아 있다.
- **🧭 단계별 탐구**: 학생이 스스로 발견하도록 답을 감춘 3단계 경로. 발견한 것을
  챗봇에게 말로 옮기면, **앱이 숫자로 채점하고** 모델은 말투만 입힌다.

두 화면을 나눈 이유는 [docs/260805_2020_plans.md](docs/260805_2020_plans.md) 3.1절에
있다 — 감추는 것과 삭제하는 것은 다르다. 단계 모드가 감추는 기저 상자와 계기판은
이 앱이 §1 의 핵심으로 내건 명제(열 = 기저벡터의 상, 넓이비 = 행렬식)를 확인하는
유일한 수단이므로, 자유 탐구 쪽에 그대로 남겨 둔다.
"""

from collections import deque

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import gemini_client
import lab_ui
from lab_ui import chart, equal_axes, format_number

EPS = 1e-9


# ═══════════════════════════════════════════════════════════════════
# 이 수업의 수학
#
# 아래 행렬들은 아무거나 고른 것이 아니다. 각각이 학생에게 보여 줄 장면을
# 하나씩 맡고 있고, 그 성질이 깨지면 그 장면이 통째로 사라진다. 그래서
# test/test_section1_discovery.py 가 이 표의 모든 성질을 매번 다시 잰다.
# ═══════════════════════════════════════════════════════════════════

def _matrix(a11, a12, a21, a22):
    return np.array([[a11, a12], [a21, a22]], dtype=float)


_ANGLE = np.pi / 3
MATRICES = {
    # 1단계 — 가역. 비대칭이고 회전도 닮음도 아니어야 한다. 회전행렬을 고정행렬로
    # 고르면 "길이와 각이 보존된다"는 **틀린** 일반화를 학생에게 유도하게 된다.
    "A1": _matrix(1, -1, 1, 2),
    # 1단계 검증 — 원이 원으로 가는 특수한 경우. "아까 결론이 틀린 건가요?" 라는
    # 되물음을 만드는 것이 이 행렬의 전부다.
    "R": _matrix(np.cos(_ANGLE), -np.sin(_ANGLE), np.sin(_ANGLE), np.cos(_ANGLE)),
    # 1단계 검증 — det < 0. 무지개+번호를 켜면 번호 순서가 뒤집힌 것이 보인다.
    "F": _matrix(0, 1, 1, 0),
    # 2·3단계 — 비가역. 대칭행렬을 쓰면 3단계에서 "상직선과 붕괴직선은 직교한다"는
    # 오개념이 우연히 참이 되어 버리므로 **반드시 비대칭**이어야 한다.
    "A2": _matrix(2, 1, 4, 2),
    # 2·3단계 검증 — 상 기울기(-2)도 붕괴 기울기(-1/3)도 A2 와 다르다. 처음 후보였던
    # [[2,1],[-4,-2]] 는 tr=0 이라 핵과 상이 일치해 버려서, 붕괴 기울기가 A2 와
    # 똑같이 -2 로 나온다. 비교에서 아무 정보도 나오지 않는 행렬이었다.
    "B": _matrix(1, 3, -2, -6),
    # 2단계 검증 전용 — 상직선이 x=0 이라 "기울기"를 말할 수 없다. 첫 줄이 영벡터라
    # 3단계 규칙(-a11/a12)이 둘째 줄로 넘어가므로 3단계에서는 쓰지 않는다.
    "C": _matrix(0, 0, 3, 1),
    # 3단계 반례 국면 — S 와 D 는 **첫 줄이 같다.** 그래서 붕괴 기울기는 둘 다
    # -1/2 로 같고 상직선만 다르다. 이 한 쌍을 나란히 보는 것만으로 "붕괴 방향은
    # 첫 줄만으로 정해지며 상직선과는 무관하다"가 드러난다.
    "S": _matrix(1, 2, 2, 4),   # 대칭  → 상직선과 붕괴직선이 직교한다
    "D": _matrix(1, 2, 3, 6),   # 비대칭 → 직교하지 않는다
}

STAGE_MATRIX = {1: "A1", 2: "A2", 3: "A2"}
VERIFY_PRESETS = {1: ["A1", "R", "F"], 2: ["A2", "B", "C"], 3: ["A2", "B", "S", "D"]}


def image_direction(matrix):
    """상직선의 방향벡터. 비가역행렬에서 상은 원점을 지나는 직선 하나다."""
    first, second = matrix[:, 0], matrix[:, 1]
    return first if np.linalg.norm(first) > EPS else second


def kernel_direction(matrix):
    """한 점으로 붕괴하는 방향. 영벡터가 아닌 가로줄에 수직이다."""
    row = matrix[0] if np.linalg.norm(matrix[0]) > EPS else matrix[1]
    return np.array([row[1], -row[0]])


def slope_of(vector):
    """기울기. 수직이면 None 을 돌려준다 — 0으로 나누지 않기 위해서다."""
    if abs(vector[0]) < EPS:
        return None
    return float(vector[1] / vector[0])


def is_singular(matrix):
    return abs(float(np.linalg.det(matrix))) < 1e-12


def cross2(u, v):
    """평면 벡터의 외적(스칼라). 0이면 두 벡터가 나란하다.

    `np.cross` 는 2차원 벡터에 대해 NumPy 2.0 에서 폐기 예고되었으므로 쓰지 않는다.
    """
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


# ═══════════════════════════════════════════════════════════════════
# 각 단계가 도달해야 할 명제
#
# 여기 적힌 문장이 곧 이 수업이다. 학생에게는 발견하기 전까지 보이지 않고,
# `✅ 확인` 에서만 노출된다.
# ═══════════════════════════════════════════════════════════════════

GOALS = {
    1: [
        ("S1_LINEAR", "직선은 직선으로, 삼각형은 삼각형으로 간다 — 꼭짓점의 개수가 보존된다."),
        ("S1_PARALLEL", "평행한 두 변은 변환 후에도 평행하다."),
        ("S1_METRIC", "길이와 각은 보존되지 않는다."),
        ("S1_CIRCLE", "그래서 원은 원이 아니라 **타원**이 된다."),
    ],
    2: [
        ("S2_FLAT", "유계인 도형은 모두 **선분**이 된다."),
        ("S2_IRREVERSIBLE", "서로 다른 두 점이 같은 점으로 간다 — **되돌릴 수 없다**."),
        ("S2_ONE_LINE", "어떤 도형을 넣어도 상은 **모두 같은 하나의 직선 위**에 놓인다."),
        ("S2_ORIGIN", "그 직선은 **원점을 지난다**."),
        ("S2_SLOPE", "그 직선의 기울기는 $a_{21}/a_{11}$ 이다 ($a_{11}=0$ 이면 $x=0$)."),
    ],
    3: [
        ("S3_EXISTS", "직선이 **한 점**으로 대응되는 경우가 있다."),
        ("S3_DIRECTION_ONLY", "그 조건은 직선의 **방향**에만 달려 있고 위치와는 무관하다."),
        ("S3_ON_IMAGE_LINE", "붕괴한 점은 2단계에서 찾은 상직선 위에 있다."),
        ("S3_SLOPE", "붕괴하는 직선의 기울기는 $-a_{11}/a_{12}$ 이다."),
        ("S3_ROWS", "둘째 줄로 만든 $-a_{21}/a_{22}$ 도 같은 값이다 — 비가역이란 두 줄이 평행하다는 뜻이다."),
    ],
}

# 2단계에서 학생이 스스로 3단계의 함정을 발견하는 경우. 필수가 아니라 보너스다.
BONUS_COLLAPSE = "S2_COLLAPSE"
# 3단계 반례 국면에서만 다루는 명제. 필수 목록에는 넣지 않는다.
PERPENDICULAR = "S3_PERP"

GOAL_TEXT = {gid: text for goals in GOALS.values() for gid, text in goals}
GOAL_TEXT[BONUS_COLLAPSE] = "직선이 한 점으로 대응되는 경우가 있다. (3단계에서 다룰 내용을 먼저 발견했다)"
GOAL_TEXT[PERPENDICULAR] = "상직선과 붕괴직선이 직교하는 것은 $a_{12}=a_{21}$ 인 행렬에서만이다."


def required_goals(stage):
    return [gid for gid, _ in GOALS[stage]]


def found_count(reached, stage):
    """이 단계에서 **필수 목표를** 몇 개 찾았는가.

    보너스(2단계에서 3단계의 함정을 먼저 발견한 경우)와 반례 국면의 명제도 같은
    자리에 쌓이므로, 그냥 개수를 세면 필수 목표 수를 넘어선다. 그러면 진행 막대가
    1을 넘고 **다음 단계가 영영 열리지 않는다** — 스스로 더 찾아낸 학생만 갇힌다.
    """
    return len(set(required_goals(stage)) & set(reached))


def stage_complete(reached, stage):
    return set(required_goals(stage)) <= set(reached)


# ═══════════════════════════════════════════════════════════════════
# 힌트 사다리
#
# **모델은 이 목록에서 고르기만 한다.** 즉흥으로 힌트를 지어내게 두면 무엇이
# 새어 나갔는지 아무도 추적할 수 없다. 마지막 칸을 뺀 모든 칸에 정답 낱말이
# 들어 있지 않은지는 test/test_section1_discovery.py 가 검사한다.
# ═══════════════════════════════════════════════════════════════════

HINTS = {
    "1": [
        "꼭짓점 좌표를 여러 가지로 바꿔 보세요. 어떤 좌표를 넣어도 변환 후 도형이 늘 갖는 성질이 있나요?",
        "마주 보는 변이 평행한 사각형을 넣어 보세요. 변환 후에도 평행한가요? 변의 길이와 각은 어떤가요?",
        "이번엔 도형의 종류를 바꿔 보세요. 삼각형·사각형·직선·원 넷 중 셋은 같은 이야기를 하고 하나만 다릅니다. 어느 것이 다른가요?",
        "원을 넣었을 때 나온 곡선을 자세히 보세요. 그것도 원인가요? 중심에서 가장 먼 곳과 가장 가까운 곳의 거리가 같은가요?",
        "삼각형은 삼각형이 되는데 원은 원이 되지 않았습니다. 삼각형을 삼각형이게 하는 성질과, 원을 원이게 하는 성질은 각각 무엇인가요?",
    ],
    "2-flat": [
        "변환 후 도형의 안쪽을 색칠할 수 있나요?",
        "삼각형 말고 사각형과 원도 넣어 보세요. 셋 다 같은 일이 일어나나요?",
        "1단계에서 삼각형은 삼각형이 되었는데, 지금은 무엇이 되었나요? 한 문장으로 적어 보세요. "
        "아직 모든 도형에서 그렇다고 확신할 수는 없으니 **'지금까지 넣어 본 도형은 …'** 으로 시작해 보세요.",
    ],
    "2-irreversible": [
        "🌈 무지개 대응을 켜고 원을 넣어 보세요. 변환 전에는 색이 한 바퀴 도는데, 변환 후에는 어떻게 늘어서 있나요?",
        "🔢 번호도 켜 보세요. 선분 위의 한 자리에 번호가 몇 개 겹쳐 있나요?",
        "변환 후 그림만 보고 처음 도형이 삼각형이었는지 원이었는지 알아낼 수 있을까요?",
    ],
    "2-oneline": [
        "🖐 이전 상 남기기를 켜고, 도형을 여러 개 바꿔 가며 넣어 보세요. 결과들이 어떻게 놓이나요?",
        "도형을 화면 구석으로 멀리 옮겨 보세요(예: 원의 중심을 (8,8)로). 선분도 따라 멀어지나요?",
        "지금까지 나온 선분들을 각각 양쪽으로 길게 늘였다고 상상해 보세요. 서로 어떤 관계인가요?",
    ],
    "2-line": [
        "그 직선이 지나는 점 중에 특별한 점이 있나요? 좌표평면에서 가장 특별한 점 하나를 떠올려 보세요.",
        "이제 기울기를 알아봅시다. 그 직선 위의 점 **하나만 정확히 알면** 됩니다. 아주 단순한 점을 하나 골라 넣어 보세요.",
        "삼각형의 세 꼭짓점을 **(0,0), (1,0), (0,1)** 로 넣어 보세요. (1,0)은 어디로 갔나요?",
        "(1,0)이 간 자리의 좌표를 행렬의 네 수와 비교해 보세요. 어느 두 수인가요?",
        "원점과 $(a_{11},\\,a_{21})$ 을 지나는 직선의 기울기는 얼마인가요?",
    ],
    "3": [
        "$\\theta$ 를 0°에서 180°까지 천천히 돌려 보세요. 변환 후 그림의 **길이**가 어떻게 변하나요?",
        "가장 짧아지는 각 근처에서 아주 조금씩 움직여 보세요. 무엇이 되나요?",
        "그 각을 적어 두고, 이번엔 직선이 지나는 점 $P$ 를 다른 곳으로 옮겨 보세요. 여전히 점이 되나요?",
        "그 점은 어디에 찍히나요? 2단계에서 찾은 직선과 무슨 관계인가요?",
        "행렬을 $B$ 로 바꾸고 다시 찾아보세요. 붕괴하는 직선의 **기울기**와 행렬의 네 수 사이에 규칙이 보이나요?",
        "첫 줄의 두 수 $a_{11},\\,a_{12}$ **만** 써서 그 기울기를 나타낼 수 있나요?",
        "둘째 줄 $a_{21},\\,a_{22}$ 로도 같은 방법으로 만들어 보세요. 왜 같은 값이 나올까요?",
    ],
}

# 어느 목표가 막혔을 때 어느 사다리를 타는가. 순서가 곧 수업의 순서다.
LADDER_OF = {
    "S1_LINEAR": "1", "S1_PARALLEL": "1", "S1_METRIC": "1", "S1_CIRCLE": "1",
    "S2_FLAT": "2-flat", "S2_IRREVERSIBLE": "2-irreversible",
    "S2_ONE_LINE": "2-oneline", "S2_ORIGIN": "2-line", "S2_SLOPE": "2-line",
    "S3_EXISTS": "3", "S3_DIRECTION_ONLY": "3", "S3_ON_IMAGE_LINE": "3",
    "S3_SLOPE": "3", "S3_ROWS": "3",
}

# 말풍선은 **보고해 달라는 초대**일 뿐이다. "무엇을 보라"는 말은 전부 힌트
# 사다리의 몫이다. 이 경계가 무너지면 힌트 소진 상태를 관리할 이유가 없어진다.
BUBBLES = {
    "start": "뭔가 알아내면 나에게 알려 줘!",
    "busy": "지금까지 뭐 봤어?",
    "stage_changed": "행렬이 바뀌었어. 알려 줄 거 생기면 불러.",
    "after_hint": "해 봤어?",
    "almost": "하나 더 있을 것 같은데.",
}
WIGGLE_AFTER_OPS = 5     # 의미 있는 조작 몇 번 뒤에 말을 걸 것인가


# ═══════════════════════════════════════════════════════════════════
# 검증기 — 판정은 코드가 한다
#
# 모델에게 참·거짓을 맡기지 않는다. 실제로 이 계획의 초안은 "붕괴직선은
# 상직선과 직교한다"고 적고 있었고(대칭행렬만 넣어 보면 참으로 보인다),
# 모델도 똑같이 틀린다. 그래서 학생의 주장은 **지금 화면의 행렬로 직접
# 계산해서** 판정한다.
# ═══════════════════════════════════════════════════════════════════

def _sample_shapes(rng, count=64):
    """난수 삼각형들. 주장을 '증명'하지 않고 '재기' 위한 표본이다."""
    return rng.uniform(-6, 6, size=(count, 3, 2))


def _slopes_match(claimed, actual):
    """학생이 말한 기울기와 실제. 수직은 None 으로 표현한다."""
    if actual is None or claimed is None:
        return actual is None and claimed is None
    return abs(float(claimed) - actual) < 1e-6


def check_claim(claim_id, value, matrix):
    """주장 하나를 지금 행렬로 판정한다. 판정할 수 없으면 None.

    Returns:
        True(맞다) / False(틀리다) / None(이 단계에서 다룰 주장이 아니다)
    """
    rng = np.random.default_rng(20260805)
    determinant = float(np.linalg.det(matrix))
    singular = is_singular(matrix)

    if claim_id == "S1_LINEAR":
        # 직선 위의 세 점은 변환 후에도 한 직선 위에 있는가.
        base = rng.uniform(-6, 6, size=(40, 2))
        direction = rng.uniform(-3, 3, size=(40, 2))
        collinear = np.stack([base, base + direction, base + 2 * direction], axis=1) @ matrix.T
        skew = cross2(collinear[:, 1] - collinear[:, 0], collinear[:, 2] - collinear[:, 0])
        return bool(np.allclose(skew, 0, atol=1e-8)) and not singular

    if claim_id == "S1_PARALLEL":
        # 방향이 같은 두 선분의 상이 여전히 나란한가. 끝점의 상에서 직접 잰다.
        u = rng.uniform(-3, 3, size=(40, 2))
        p, q = rng.uniform(-6, 6, size=(40, 2)), rng.uniform(-6, 6, size=(40, 2))
        first = (p + u) @ matrix.T - p @ matrix.T
        second = (q + u) @ matrix.T - q @ matrix.T
        return bool(np.allclose(cross2(first, second), 0, atol=1e-8)
                    and np.all(np.linalg.norm(first, axis=1) > 1e-8) and not singular)

    if claim_id == "S1_METRIC":
        # 길이·각이 안 변하는 것은 닮음변환(특잇값이 같은 경우)뿐이다.
        sigma = np.linalg.svd(matrix, compute_uv=False)
        return bool(abs(sigma[0] - sigma[1]) > 1e-9)

    if claim_id == "S1_CIRCLE":
        sigma = np.linalg.svd(matrix, compute_uv=False)
        is_ellipse = abs(sigma[0] - sigma[1]) > 1e-9
        said_ellipse = str(value).strip() in ("타원", "ellipse", "타원이다", "True", "true")
        return bool(is_ellipse == said_ellipse)

    if claim_id in ("S2_FLAT", "S2_ONE_LINE", "S2_ORIGIN", "S3_EXISTS", BONUS_COLLAPSE):
        if not singular:
            return False
        images = (_sample_shapes(rng).reshape(-1, 2)) @ matrix.T
        direction = image_direction(matrix)
        on_one_line = np.allclose(cross2(images, direction), 0, atol=1e-8)
        return bool(on_one_line)            # 원점을 지나는 것은 일차변환의 정의상 따라온다

    if claim_id == "S2_IRREVERSIBLE":
        return bool(abs(determinant) < 1e-12)

    if claim_id == "S2_SLOPE":
        return _slopes_match(value, slope_of(image_direction(matrix))) if singular else False

    if claim_id == "S3_SLOPE":
        return _slopes_match(value, slope_of(kernel_direction(matrix))) if singular else False

    if claim_id == "S3_DIRECTION_ONLY":
        if not singular:
            return False
        # 평행한 두 직선이 (서로 다른) 각각의 점으로 붕괴하는가.
        direction = kernel_direction(matrix)
        t = np.linspace(-9, 9, 50)[:, None]
        first = (np.array([1.0, 0.5]) + t * direction) @ matrix.T
        second = (np.array([-3.0, 2.0]) + t * direction) @ matrix.T
        collapses = np.allclose(first, first[0], atol=1e-8) and np.allclose(second, second[0], atol=1e-8)
        said_direction = str(value).strip() in ("방향", "direction", "기울기", "True", "true")
        return bool(collapses and said_direction)

    if claim_id == "S3_ON_IMAGE_LINE":
        if not singular:
            return False
        t = np.linspace(-9, 9, 50)[:, None]
        point = ((np.array([1.0, 0.5]) + t * kernel_direction(matrix)) @ matrix.T)[0]
        return bool(abs(float(cross2(point, image_direction(matrix)))) < 1e-8)

    if claim_id == "S3_ROWS":
        if not singular or abs(matrix[0, 1]) < EPS or abs(matrix[1, 1]) < EPS:
            return False
        return bool(abs(-matrix[0, 0] / matrix[0, 1] + matrix[1, 0] / matrix[1, 1]) < 1e-9)

    if claim_id == PERPENDICULAR:
        # 학생이 "직교한다"고 말했을 때, **지금 이 행렬에서** 정말 그런가.
        perpendicular = abs(float(image_direction(matrix) @ kernel_direction(matrix))) < 1e-9
        said_yes = str(value).strip() not in ("False", "false", "아니다", "아니오", "no")
        return bool(perpendicular == said_yes)

    return None


# 주장이 틀렸을 때 무엇을 해 보라고 할 것인가. **답을 고쳐 주지 않고 반례로 민다.**
REBUTTALS = {
    "S1_CIRCLE": "원을 하나 넣고, 가장 길쭉한 방향과 가장 납작한 방향의 길이를 견줘 봐.",
    "S2_SLOPE": "네가 말한 기울기로 원점을 지나는 직선을 그렸다고 치고, (1,0)이 간 자리가 정말 그 위에 있는지 봐.",
    "S3_SLOPE": "그 기울기로 직선을 만들어 $\\theta$ 에 넣어 봐. 점이 되니?",
    "S3_DIRECTION_ONLY": "$P$ 만 옮기고 $\\theta$ 는 그대로 둬 봐. 그래도 점이 되는지.",
    PERPENDICULAR: "$S$ 와 $D$ 를 차례로 넣어 봐. 첫 줄이 같은 두 행렬인데 결과도 같니?",
    "S2_FLAT": "삼각형 말고 원도 넣어 봐.",
}
DEFAULT_REBUTTAL = "화면에서 그게 정말 그런지 한 번 더 확인해 볼래?"


# ═══════════════════════════════════════════════════════════════════
# 모델에게 주는 지시 — 번역만 시킨다
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """너는 고등학생과 함께 일차변환을 탐구하는 **동료**다. 교사가 아니다.
너는 정답을 모르며, 학생의 문장을 아래 목록의 주장(claim)으로 옮기는 일만 한다.
참·거짓 판정은 앱이 직접 계산해서 하므로 너는 절대 판정하지 마라.

반드시 아래 JSON 객체 하나만 출력한다.
{
  "claims": [{"id": "<아래 목록의 id>", "value": <수 또는 문자열 또는 true/false>}],
  "restate": "<학생 말을 한 문장으로 되풀이. 새로운 정보를 덧붙이지 마라>",
  "asks_answer": <학생이 답을 알려 달라고 했으면 true>,
  "sample_thin": <학생이 한두 개만 해 보고 성급히 일반화하면 true>
}

규칙:
- 목록에 없는 id 를 지어내지 마라. 옮길 수 없으면 claims 를 빈 배열로 둬라.
- 학생이 수(기울기 등)를 말했으면 value 에 그 수를 넣어라. 수직이라고 했으면 "수직".
- "납작해져요", "다 한 줄로 뭉개져요" 같은 일상어도 정확한 주장으로 옮겨라.
- restate 에 목표 명제나 힌트를 흘리지 마라. 학생이 한 말만 되풀이한다.

옮길 수 있는 주장 목록:
"""


def _claim_menu(stage):
    ids = required_goals(stage) + ([BONUS_COLLAPSE] if stage == 2 else [])
    if stage == 3:
        ids = ids + [PERPENDICULAR]
    return "\n".join(f'- {gid}: {GOAL_TEXT[gid]}' for gid in ids)


def _rule_based_claims(text, stage):
    """모델 없이도 대화가 굴러가게 하는 폴백.

    키가 없거나 네트워크가 죽어도 **판정기는 코드에 있으므로** 이 폴백은 반쪽이
    아니다. 알아듣는 폭만 좁아진다.
    """
    lowered = text.replace(" ", "")
    claims = []

    def add(claim_id, value=True):
        claims.append({"id": claim_id, "value": value})

    numbers = []
    for token in text.replace("=", " ").replace(",", " ").split():
        try:
            numbers.append(float(token.strip("y=x기울기의은는이가.")))
        except ValueError:
            continue

    if stage == 1:
        if "타원" in lowered:
            add("S1_CIRCLE", "타원")
        if "평행" in lowered:
            add("S1_PARALLEL")
        if any(word in lowered for word in ("길이", "각도", "각이", "변한")):
            add("S1_METRIC")
        if any(word in lowered for word in ("삼각형은삼각형", "직선은직선", "꼭짓점", "종류")):
            add("S1_LINEAR")
    elif stage == 2:
        if any(word in lowered for word in ("선분", "한줄", "납작", "뭉개", "붕괴", "직선이돼")):
            add("S2_FLAT")
        if any(word in lowered for word in ("되돌", "역행렬", "복원", "겹쳐", "같은점")):
            add("S2_IRREVERSIBLE")
        if any(word in lowered for word in ("같은직선", "한직선", "모두같", "일치")):
            add("S2_ONE_LINE")
        if "원점" in lowered:
            add("S2_ORIGIN")
        if "기울기" in lowered and numbers:
            add("S2_SLOPE", numbers[0])
        if "점" in lowered and any(word in lowered for word in ("한점", "점하나")):
            add(BONUS_COLLAPSE)
    else:
        if any(word in lowered for word in ("한점", "점하나", "점이돼", "점으로")):
            add("S3_EXISTS")
        if "방향" in lowered or "기울기만" in lowered:
            add("S3_DIRECTION_ONLY", "방향")
        if "기울기" in lowered and numbers:
            add("S3_SLOPE", numbers[0])
        if any(word in lowered for word in ("둘째줄", "두번째줄", "아래줄", "같은값")):
            add("S3_ROWS")
        if "직교" in lowered or "수직" in lowered:
            add(PERPENDICULAR, "아니다" if any(w in lowered for w in ("아니", "안", "않")) else True)

    return {"claims": claims, "restate": "", "asks_answer": "알려줘" in lowered or "답" in lowered,
            "sample_thin": False}


def interpret(text, stage):
    """학생 문장 → 주장 목록. 모델이 있으면 모델이, 없으면 규칙이 옮긴다."""
    if not gemini_client.is_available():
        return _rule_based_claims(text, stage), "rules"
    try:
        parsed = gemini_client.ask_json_cached(SYSTEM_PROMPT + _claim_menu(stage), text)
    except gemini_client.GeminiUnavailable:
        return _rule_based_claims(text, stage), "rules"

    claims = []
    allowed = set(GOAL_TEXT)
    for item in parsed.get("claims") or []:
        if isinstance(item, dict) and item.get("id") in allowed:
            claims.append({"id": item["id"], "value": item.get("value", True)})
    return {"claims": claims,
            "restate": str(parsed.get("restate") or "")[:200],
            "asks_answer": bool(parsed.get("asks_answer")),
            "sample_thin": bool(parsed.get("sample_thin"))}, "gemini"


# ═══════════════════════════════════════════════════════════════════
# 상태
# ═══════════════════════════════════════════════════════════════════

def _state():
    if "s1" not in st.session_state:
        st.session_state["s1"] = {
            "stage": 1,
            "reached": {1: set(), 2: set(), 3: set()},
            "hint_idx": {},
            "ops": 0,
            "log": [],
            "ghosts": deque(maxlen=3),
            "ghost_sig": None,
            "last_sig": None,
            "bubble": "start",
            "matrix_key": STAGE_MATRIX[1],
            "counterexample_shown": False,
        }
    return st.session_state["s1"]


def _note_operation(state, signature):
    """의미 있는 조작만 센다. 재실행마다 세면 학생이 가만히 있어도 숫자가 오른다."""
    if signature != state["last_sig"]:
        state["last_sig"] = signature
        state["ops"] += 1
        if state["ops"] >= WIGGLE_AFTER_OPS and state["bubble"] == "start":
            state["bubble"] = "busy"


def _push_ghost(state, transformed, signature):
    if signature != state["ghost_sig"]:
        state["ghost_sig"] = signature
        state["ghosts"].append(np.array(transformed, copy=True))


def _next_hint(state, stage):
    """막힌 첫 목표의 사다리에서 **다음 한 칸만** 꺼낸다."""
    for goal_id in required_goals(stage):
        if goal_id in state["reached"][stage]:
            continue
        ladder = LADDER_OF[goal_id]
        index = state["hint_idx"].get(ladder, 0)
        if index < len(HINTS[ladder]):
            state["hint_idx"][ladder] = index + 1
            return HINTS[ladder][index]
        return None      # 사다리를 다 썼다 — `✅ 확인` 으로 착지시킨다
    return None


# ═══════════════════════════════════════════════════════════════════
# 그리기
# ═══════════════════════════════════════════════════════════════════

def parse_point(text, fallback):
    """"1,1" 같은 좌표 입력을 읽는다. 못 읽으면 기본값으로 되돌린다.

    학생은 반드시 이상한 값을 넣는다 — `1;1`, 빈칸, `1,1,1`. 예전에는 그것이
    그대로 예외가 되어 화면에 빨간 traceback 이 떴다. 이 앱의 방침대로
    예외를 흘리지 않고 기본값으로 되돌린 뒤 무엇이 잘못됐는지 알려 준다.

    Returns:
        (좌표 ndarray, 오류 메시지 또는 None)
    """
    parts = [p.strip() for p in (text or '').split(',')]
    if len(parts) != 2:
        return np.array(fallback, dtype=float), f"'{text}' — 쉼표로 x, y 두 개를 넣어 주세요 (예: 1,1)"
    try:
        return np.array([float(parts[0]), float(parts[1])]), None
    except ValueError:
        return np.array(fallback, dtype=float), f"'{text}' — 숫자만 넣어 주세요 (예: 1,1)"


def signed_area(points):
    """신발끈 공식으로 다각형의 넓이를 잰다.

    **부호를 살려 둔다.** 절댓값만 쓰면 넓이가 몇 배가 됐는지는 알 수 있어도
    꼭짓점을 도는 **방향이 뒤집혔는지**는 알 수 없는데, 행렬식이 음수라는 것이
    바로 그 뜻이기 때문이다. 계기판에서 두 사실을 같이 읽으려면 부호가 필요하다.
    """
    points = np.asarray(points, dtype=float)
    closed = points if lab_ui.is_closed(points) else np.vstack([points, points[0]])
    x, y = closed[:, 0], closed[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def plot_shape(shape_type, shape, transformed, matrix, a=1, b=1, c=0,
               rainbow=False, numbers=False, basis=True, morph=False,
               ghosts=None, view_key=None):
    fig = go.Figure()

    # 직선일 경우 변환된 점 하나 강조.
    # a, b 가 둘 다 0이면 직선이 아니므로(호출부에서 이미 안내한다) 건너뛴다.
    new_point = None
    if shape_type == "직선" and not (a == 0 and b == 0):
        base_point = np.array([0, c / b]) if b != 0 else np.array([c / a, 0])
        new_point = np.dot(base_point, matrix.T)

    # 축 범위를 **먼저** 정한다. 무지개를 "보이는 영역 안에서의 호의 길이"로
    # 칠하려면 어디까지가 화면인지 알아야 하기 때문이다.
    all_x = np.concatenate([shape[:, 0], transformed[:, 0]])
    all_y = np.concatenate([shape[:, 1], transformed[:, 1]])
    if new_point is not None:
        all_x = np.append(all_x, new_point[0])
        all_y = np.append(all_y, new_point[1])
    x_center, y_center = np.mean(all_x), np.mean(all_y)
    half_range = max(np.ptp(all_x), np.ptp(all_y)) * 0.75
    half_range = min(half_range, 20)  # 최대 20으로 제한
    if half_range < 1:
        half_range = 2
    x_range = [x_center - half_range, x_center + half_range]
    y_range = [y_center - half_range, y_center + half_range]

    # ✅ 잔상 — **직전 상들을 무채색으로 남긴다.**
    #
    # "어떤 도형을 넣어도 상이 같은 직선 위에 놓인다"는 관찰은 기억에만 의존해서는
    # 서지 않는다. 도형을 바꾸는 순간 이전 결과가 사라지기 때문이다. 색은 파랑/빨강/
    # 무지개가 이미 다 쓰고 있으므로 잔상은 채도를 포기하고 회색만 쓴다.
    for ghost in (ghosts or []):
        fig.add_trace(go.Scatter(
            x=ghost[:, 0], y=ghost[:, 1], mode='lines',
            line=dict(color='rgba(128,128,128,0.45)', width=1),
            hoverinfo='skip', showlegend=False, name='이전 상'))

    # ✅ 무지개 대응 — 호의 길이 기준으로 다시 뽑아 색을 입힌다.
    #    변환은 다시 뽑은 점에 그대로 적용하면 되므로, 색 배열 하나가
    #    변환 전후를 잇는다.
    t = None
    src, dst = shape, transformed
    if rainbow:
        src, t = lab_ui.arclength_parameter(shape, x_range, y_range)
        if t is not None:
            dst = src @ matrix.T
    style = lab_ui.channel_style(rainbow=t is not None)
    closed = lab_ui.is_closed(shape)

    def draw(points, role, name):
        common = dict(x=points[:, 0], y=points[:, 1], name=name,
                      mode='lines+markers')
        if t is None:
            color = style[role]['color']
            fig.add_trace(go.Scatter(
                line=dict(color=color, dash=style[role]['dash']),
                marker=dict(color=color, symbol=style[role]['symbol'], size=6),
                **common))
        else:
            # Plotly 는 선 색을 점마다 줄 수 없다(line.color 는 trace 당 하나).
            # 그래서 선은 옅은 무채색으로 두고 색은 마커가 진다.
            fig.add_trace(go.Scatter(
                line=dict(color='lightgray', width=1, dash=style[role]['dash']),
                marker=lab_ui.rainbow_marker(t, closed, size=7,
                                             symbol=style[role]['symbol']),
                **common))

    draw(src, 'before', '변환전 도형')
    after_index = len(fig.data)          # 연속 변환에서 갱신할 trace
    draw(dst, 'after', '변환후 도형')

    # ✅ 기저 상자 — 이 섹션에서 가장 중요한 구조.
    #
    # e₁, e₂ 가 만드는 단위정사각형과 그 상인 평행사변형을 겹쳐 그린다.
    # **행렬의 열이 곧 기저벡터가 간 자리**라는 사실이 그림에 없으면,
    # 학생에게 행렬은 숫자 네 개일 뿐이다. a11 을 밀었을 때 화살표 하나가
    # 따라 움직이는 것을 보는 것만으로 행렬의 의미가 바뀐다.
    #
    # 단계 모드에서는 이것이 곧 2단계의 정답이므로 호출부가 basis=False 로 끈다.
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    basis_index = None
    if basis:
        fig.add_trace(go.Scatter(
            x=square[:, 0], y=square[:, 1], mode='lines',
            line=dict(color=lab_ui.BEFORE, width=1.5),
            fill='toself', fillcolor='rgba(0,0,255,0.07)',
            name='e₁, e₂ 단위정사각형'))
        basis_index = len(fig.data)
        image = square @ matrix.T
        fig.add_trace(go.Scatter(
            x=image[:, 0], y=image[:, 1], mode='lines',
            line=dict(color=lab_ui.AFTER, width=1.5, dash='dash'),
            fill='toself', fillcolor='rgba(255,0,0,0.07)',
            name='Ae₁, Ae₂ 평행사변형'))

    # 웨이포인트 번호 — 색을 못 봐도 대응이 읽히고, 변환 후 번호 간격이
    # 흐트러진 정도로 어디가 늘어났는지를 수로 읽을 수 있다.
    if t is not None and numbers:
        idx = lab_ui.waypoint_indices(10, len(src))
        for points, color in ((src, 'black'), (dst, 'dimgray')):
            fig.add_trace(go.Scatter(
                x=points[idx, 0], y=points[idx, 1],
                mode='text', text=[str(k) for k in range(len(idx))],
                textfont=dict(size=11, color=color),
                hoverinfo='skip', showlegend=False,
            ))

    if new_point is not None:
        fig.add_trace(go.Scatter(
            x=[new_point[0]], y=[new_point[1]],
            mode='markers', name='변환된 점',
            marker=dict(color='red', size=10, symbol='circle')
        ))

    # ✅ 연속 변환 — 항등행렬에서 A 까지 (1−t)I + tA 로 건너간다.
    #
    # 프레임을 미리 다 계산해 그림에 넣어 두므로 슬라이더 조작이 브라우저
    # 안에서만 일어난다. 재실행 0회 = 깜빡임 0.
    #
    # 이 경로는 **유일하지 않다**는 점을 화면에 밝혀야 한다. 회전이라면
    # 각도를 따라 도는 경로가 더 자연스럽고, 여기 경로는 도중에 det 가 0을
    # 지나며 도형이 납작해질 수도 있다 — 그 자체가 좋은 토론거리다.
    if morph:
        steps = np.linspace(0, 1, 41)
        frames, updated = [], [after_index] + ([basis_index] if basis else [])
        for k, step in enumerate(steps):
            M = (1 - step) * np.eye(2) + step * matrix
            data = [go.Scatter(x=(src @ M.T)[:, 0], y=(src @ M.T)[:, 1])]
            if basis:
                moved = square @ M.T
                data.append(go.Scatter(x=moved[:, 0], y=moved[:, 1]))
            frames.append(go.Frame(name=f"{k}", data=data, traces=updated))
        fig.frames = frames
        fig.update_layout(
            updatemenus=[dict(
                type='buttons', direction='left', x=0.0, y=1.12, xanchor='left',
                buttons=[dict(label='▶ 변환', method='animate',
                              args=[None, dict(frame=dict(duration=55, redraw=False),
                                               fromcurrent=True,
                                               transition=dict(duration=0))]),
                         dict(label='⏸', method='animate',
                              args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                 mode='immediate')])])],
            sliders=[dict(
                active=len(steps) - 1, x=0.14, len=0.84, y=1.08, xanchor='left',
                currentvalue=dict(prefix='t = ', font=dict(size=13)),
                steps=[dict(method='animate', label=f"{s:.2f}",
                            args=[[f"{k}"], dict(mode='immediate',
                                                 frame=dict(duration=0, redraw=False),
                                                 transition=dict(duration=0))])
                       for k, s in enumerate(steps)])],
        )

    # 도형 종류가 바뀌면 시야를 되돌리고, 행렬만 바꿀 때는 확대해 둔
    # 시야를 그대로 둔다.
    equal_axes(fig, view_key=view_key or f"s1-{shape_type}",
               x_range=x_range, y_range=y_range)
    fig.update_layout(
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=10 if not morph else 60, b=0),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# 도형 입력 (두 모드가 함께 쓴다)
# ═══════════════════════════════════════════════════════════════════

def shape_inputs(prefix, shape_types=("삼각형", "사각형", "원", "직선")):
    """도형 하나를 만든다.

    Returns:
        (도형 종류, 점 배열, (a, b, c), 오류 메시지 목록)
    """
    problems = []
    shape_type = (shape_types[0] if len(shape_types) == 1 else
                  st.selectbox("도형 종류를 선택하세요", list(shape_types), key=f"{prefix}_type"))

    def point(label, default):
        value, problem = parse_point(
            st.text_input(label, default, key=f"{prefix}_{label}"), default.split(','))
        if problem:
            problems.append(problem)
        return value

    # 직선 계수는 다른 도형일 때도 plot_shape 에 넘어가므로 기본값을 둔다.
    a, b, c = 1.0, 1.0, 1.0

    if shape_type == "삼각형":
        A = point("점 A 좌표 (예: 1,1)", "1,1")
        B = point("점 B 좌표 (예: 1,2)", "1,2")
        C = point("점 C 좌표 (예: 2,1)", "2,1")
        shape = np.array([A, B, C, A])
    elif shape_type == "사각형":
        A = point("점 A 좌표 (예: 1,1)", "1,1")
        B = point("점 B 좌표 (예: 1,2)", "1,2")
        C = point("점 C 좌표 (예: 2,2)", "2,2")
        D = point("점 D 좌표 (예: 2,1)", "2,1")
        shape = np.array([A, B, C, D, A])
    elif shape_type == "원":
        center = point("원 중심 좌표 (예: 1,1)", "1,1")
        radius = st.number_input("반지름", value=2.0, step=0.1, format="%.1f", key=f"{prefix}_r")
        theta = np.linspace(0, 2 * np.pi, 200)
        shape = np.stack([center[0] + radius * np.cos(theta),
                          center[1] + radius * np.sin(theta)], axis=1)
    else:
        st.markdown("직선의 형태: $ax + by = c$")
        a = st.number_input("계수 a", value=1.0, step=0.1, format="%.1f", key=f"{prefix}_a")
        b = st.number_input("계수 b", value=1.0, step=0.1, format="%.1f", key=f"{prefix}_b")
        c = st.number_input("상수 c", value=2.0, step=0.1, format="%.1f", key=f"{prefix}_c")
        if a == 0 and b == 0:
            # 0x + 0y = c 는 직선이 아니다. 예전에는 여기서 c/a 가 0으로 나눠졌다.
            problems.append("a 와 b 가 둘 다 0이면 직선이 아닙니다. a = 1 로 두고 그립니다.")
            a = 1.0
        x_vals = np.linspace(-20, 20, 400)
        if b != 0:
            y_vals = (c - a * x_vals) / b
        else:
            x_vals = np.full(400, c / a)
            y_vals = np.linspace(-20, 20, 400)
        shape = np.stack([x_vals, y_vals], axis=1)

    return shape_type, shape, (a, b, c), problems


def line_through(point, degrees):
    """점 하나와 방향각으로 직선을 만든다 — 3단계 전용.

    $ax+by=c$ 로 받지 않는 이유가 둘이다. 첫째, 계수 $a,b,c$ 가 행렬 성분과
    문자가 겹쳐 3단계에서 학생이 반드시 혼동한다. 둘째, **방향을 연속으로
    돌릴 수 있어야** 상의 길이가 줄어드는 것을 느끼며 임계각을 찾아갈 수 있다.
    계수를 밀어서는 그 온도가 생기지 않는다.
    """
    radian = np.deg2rad(degrees)
    direction = np.array([np.cos(radian), np.sin(radian)])
    t = np.linspace(-20, 20, 400)[:, None]
    return np.asarray(point, dtype=float) + t * direction


# ═══════════════════════════════════════════════════════════════════
# 챗봇
# ═══════════════════════════════════════════════════════════════════

CHAT_CSS = """
<style>
/* 플로팅은 **연출**이다. 이 CSS 가 통째로 안 먹어도 챗봇은 화면 안에
   인라인으로 앉아 기능을 100% 유지한다. st-emotion-cache-* 해시를 잡는
   흔한 방식 대신 st.container(key=...) 가 붙여 주는 안정적인 클래스만 쓴다. */
.st-key-s1_chatdock {
    position: fixed; right: 24px; bottom: 24px; z-index: 999;
    width: 320px; text-align: right;
}
.st-key-s1_chatdock [data-testid="stPopover"] { display: inline-block; }
.st-key-s1_chatdock [data-testid="stPopover"] button {
    border-radius: 50%; width: 56px; height: 56px; font-size: 24px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    animation: s1-wiggle 6s ease-in-out infinite;
}
.s1-bubble {
    display: inline-block; margin-bottom: 8px; padding: 8px 12px;
    background: #fff8d6; color: #333; border-radius: 12px; font-size: 13px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18); max-width: 260px; text-align: left;
}
/* 6초 중 5.4초는 가만히 있는다. 위젯을 건드릴 때마다 재실행되면서
   애니메이션이 재시작하는데, 대부분 정지 상태라야 그게 눈에 안 띈다. */
@keyframes s1-wiggle {
    0%, 88%, 100% { transform: rotate(0deg); }
    91% { transform: rotate(-10deg); }
    94% { transform: rotate(10deg); }
    97% { transform: rotate(-6deg); }
}
/* 흔들리는 아이콘은 어떤 학생에게는 그냥 방해물이다. */
@media (prefers-reduced-motion: reduce) {
    .st-key-s1_chatdock [data-testid="stPopover"] button { animation: none; }
}
</style>
"""


def _say(state, role, text):
    state["log"].append((role, text))


def _handle_submission(state, stage, matrix, text):
    """학생 문장 하나를 처리한다 — 옮기고, 재고, 답한다."""
    _say(state, "user", text)
    state["ops"] = 0
    reading, source = interpret(text, stage)

    if reading["asks_answer"]:
        hint = _next_hint(state, stage)
        _say(state, "assistant",
             "나도 몰라. 대신 어디를 볼지 하나만 골라 줄게.\n\n"
             + (f"👉 {hint}" if hint else "이번 건 아래 `✅ 확인` 을 같이 열어 보자."))
        state["bubble"] = "after_hint"
        return

    if not reading["claims"]:
        _say(state, "assistant",
             "무슨 말인지 잘 모르겠어. 이런 식으로 말해 줄래? "
             "예: “원을 넣었더니 원이 아니라 찌그러진 모양이 됐어”, “상이 다 한 직선 위에 있어”.")
        return

    approved, rejected = [], []
    for claim in reading["claims"]:
        verdict = check_claim(claim["id"], claim.get("value", True), matrix)
        if verdict is True:
            if claim["id"] not in state["reached"][stage]:
                state["reached"][stage].add(claim["id"])
            approved.append(claim["id"])
        elif verdict is False:
            rejected.append(claim["id"])

    lines = []
    if reading["restate"]:
        lines.append(f"_{reading['restate']}_")

    for claim_id in approved:
        if claim_id == BONUS_COLLAPSE:
            lines.append("🎉 **그건 원래 3단계에서 다룰 내용인데 네가 먼저 찾았어.** "
                         "선생님이 숨겨 둔 걸 스스로 발견한 거야. 3단계에서 이어서 파 보자 — "
                         "지금은 2단계에서 볼 게 아직 남았어.")
        elif claim_id == PERPENDICULAR:
            lines.append("✅ 맞아. 그런데 그게 **늘** 그런지는 다른 행렬로도 확인해 보자.")
        else:
            lines.append(f"✅ 확인했어 — 앱이 직접 계산해 보니 맞아. ({GOAL_TEXT[claim_id]})")

    for claim_id in rejected:
        lines.append(f"🤔 그건 화면이랑 안 맞는 것 같아. {REBUTTALS.get(claim_id, DEFAULT_REBUTTAL)}")

    if reading["sample_thin"]:
        lines.append("아직 몇 개 안 해 봤지? 두어 개 더 넣어 보고 와.")

    remaining = [gid for gid in required_goals(stage) if gid not in state["reached"][stage]]
    if approved and remaining:
        lines.append(f"이 단계에서 찾을 게 **{len(remaining)}개** 더 남았어.")
        state["bubble"] = "almost" if len(remaining) == 1 else "busy"
    elif not remaining:
        lines.append("이 단계는 다 찾았어. 위에서 다음으로 넘어가자! 🎈")

    if rejected or (not approved and remaining):
        hint = _next_hint(state, stage)
        if hint:
            lines.append(f"👉 {hint}")
            state["bubble"] = "after_hint"

    if source == "rules":
        lines.append("_(지금은 내가 좀 둔해. 그래도 확인은 해 줄게.)_")

    _say(state, "assistant", "\n\n".join(lines))


def chat_dock(state, stage, matrix):
    """오른쪽 아래에 떠 있는 챗봇."""
    st.markdown(CHAT_CSS, unsafe_allow_html=True)
    with st.container(key="s1_chatdock"):
        st.markdown(f'<div class="s1-bubble">{BUBBLES[state["bubble"]]}</div>',
                    unsafe_allow_html=True)
        with st.popover("💬"):
            st.caption("네가 쓴 문장은 답변을 만들기 위해 외부 서비스로 전송돼. "
                       "그림을 조작해서 알아낸 걸 말로 적어 줘.")
            if not state["log"]:
                st.chat_message("assistant").markdown(
                    "지금까지 뭘 알아냈어? 확신 없어도 괜찮고, 틀려도 괜찮아. 본 대로 적어 줘.\n\n"
                    "_나도 답은 몰라. 네가 찾은 걸 같이 확인해 보는 역할이야._")
            for role, text in state["log"][-12:]:
                st.chat_message(role).markdown(text)
            submitted = st.chat_input("알아낸 걸 적어 줘", key="s1_chat_input")
    if submitted:
        _handle_submission(state, stage, matrix, submitted)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# 단계별 탐구
# ═══════════════════════════════════════════════════════════════════

STAGE_TITLES = {
    1: "1단계 — 행렬 하나를 정해 놓고, 도형을 바꿔 본다",
    2: "2단계 — 행렬을 바꿨다. 이번엔 무엇이 일어나는가",
    3: "3단계 — 직선을 돌려 본다",
}
STAGE_INTROS = {
    1: "아래 행렬은 고정입니다. 도형만 바꿔 가며 **변하지 않는 것**을 찾아보세요.",
    2: "행렬이 바뀌었습니다. 역시 고정이고, 도형만 바꿉니다.",
    3: "직선 하나를 정해 놓고 **방향각 $\\theta$** 를 돌려 봅니다.",
}


def _stage_header(state):
    stage = state["stage"]
    st.subheader(STAGE_TITLES[stage])
    st.markdown(STAGE_INTROS[stage])

    found, total = found_count(state["reached"][stage], stage), len(required_goals(stage))
    st.progress(found / total, text=f"찾은 것 {found} / {total}")

    columns = st.columns(3)
    for index in (1, 2, 3):
        unlocked = index == 1 or stage_complete(state["reached"][index - 1], index - 1)
        if columns[index - 1].button(f"{index}단계", disabled=not unlocked,
                                     use_container_width=True,
                                     type="primary" if index == stage else "secondary",
                                     key=f"s1_go{index}"):
            state["stage"] = index
            state["bubble"] = "stage_changed"
            state["ghosts"].clear()
            state["matrix_key"] = STAGE_MATRIX[index]
            st.rerun()
    return stage


def _matrix_picker(state, stage):
    """검증 국면 — 이 단계의 목표를 다 찾은 뒤에만 다른 행렬이 열린다.

    귀납적 일반화를 정직하게 만들려면 관찰(고정) 뒤에 검증(여러 개)이 와야 한다.
    그러나 검증을 **먼저** 열어 주면 변인이 둘로 늘어 관찰 자체가 서지 않는다.
    """
    if not stage_complete(state["reached"][stage], stage):
        state["matrix_key"] = STAGE_MATRIX[stage]
        return MATRICES[state["matrix_key"]]

    st.caption("🔓 검증 국면 — 다른 행렬에서도 같은 이야기가 성립하는지 확인해 보세요.")
    options = VERIFY_PRESETS[stage]
    if state["matrix_key"] not in options:
        state["matrix_key"] = options[0]
    chosen = st.pills("행렬", options, default=state["matrix_key"],
                      selection_mode="single", key=f"s1_pills{stage}")
    state["matrix_key"] = chosen or state["matrix_key"]
    return MATRICES[state["matrix_key"]]


def _counterexample_notice(state, stage):
    """3단계의 오개념을 학생 손으로 죽이는 자리. 선택이 아니라 필수다."""
    if stage != 3 or "S3_SLOPE" not in state["reached"][3]:
        return
    st.warning(
        "지금까지 써 본 행렬이 $a_{12} = a_{21}$ 인 것뿐일 수 있습니다. "
        "위에서 **S** 와 **D** 를 차례로 넣어 보세요. 첫 줄이 같은 두 행렬입니다.\n\n"
        "**두 행렬에서 붕괴하는 직선의 기울기는 같나요, 다른가요? "
        "2단계의 상직선과는 늘 수직인가요?**")


def _confirm_expander(state, stage):
    with st.expander("✅ 확인 — 다 찾았거나 막혔을 때만 열어 보세요"):
        for goal_id, text in GOALS[stage]:
            mark = "✅" if goal_id in state["reached"][stage] else "⬜"
            st.markdown(f"{mark} {text}")
        if stage == 2:
            st.markdown("---")
            st.markdown(
                "이제 **📦 기저 상자**를 켜 보세요. 여러분이 찾은 그 직선이 "
                "$e_1 = (1,0)$ 이 간 자리였다는 것이 보입니다.")
            st.checkbox("📦 기저 상자 보기", key="s1_confirm_basis")
            st.caption("각주: 첫 열이 영벡터($a_{11}=a_{21}=0$)이면 상직선은 둘째 열이 정합니다.")
        if stage == 3:
            st.markdown("---")
            st.markdown(
                "직선이 한 점으로 대응될 조건은 **기울기가 $-a_{11}/a_{12}$** 일 때 — 즉 방향벡터가 "
                "$(a_{12},\\,-a_{11})$ 일 때, 다시 말해 **행렬의 각 가로줄이 그 직선의 법선 방향과 "
                "나란할 때**입니다. 2단계의 직선과 수직인 것은 $a_{12}=a_{21}$ 인 특별한 행렬에서만 "
                "일어납니다.")
            st.caption("이 두 직선은 §5 에서 이름을 얻습니다.")


def _discovery():
    state = _state()
    stage = _stage_header(state)

    left, _, right = st.columns([1.4, 0.3, 2])

    with left:
        matrix = _matrix_picker(state, stage)
        st.latex(r"A = \begin{bmatrix} %s & %s \\ %s & %s \end{bmatrix}" % (
            format_number(matrix[0, 0]), format_number(matrix[0, 1]),
            format_number(matrix[1, 0]), format_number(matrix[1, 1])))

        st.subheader("# 도형 입력 ___________________")
        if stage == 3:
            # 원점을 지나지 않고 축과 평행하지 않은 직선. 그래야 붕괴한 점이
            # 원점이 아닌 곳에 찍혀서 "그 점이 상직선 위에 있다"가 보인다.
            px = st.number_input("P 의 x", value=1.0, step=0.5, format="%.2f", key="s1d_px")
            py = st.number_input("P 의 y", value=0.5, step=0.5, format="%.2f", key="s1d_py")
            theta = st.slider("방향각 θ (도)", 0.0, 180.0, 30.0, step=0.1, key="s1d_theta")
            theta = st.number_input("θ 정밀 입력", value=float(theta), step=0.1,
                                    format="%.2f", key="s1d_theta_exact")
            shape_type, shape, coefficients = "직선", line_through((px, py), theta), (0.0, 0.0, 0.0)
            problems = []
        else:
            # 1단계 앞부분에서는 도형 종류를 잠근다. 종류와 좌표를 동시에 열면
            # 무엇 때문에 무엇이 변했는지 학생이 분리할 수 없다.
            unlocked = stage == 2 or bool({"S1_LINEAR", "S1_PARALLEL"} & state["reached"][1])
            types = ("삼각형", "사각형", "원", "직선") if unlocked else ("삼각형",)
            if not unlocked:
                st.caption("먼저 **좌표만** 바꿔 보세요. 뭔가 하나 알아내면 도형 종류가 열립니다.")
            shape_type, shape, coefficients, problems = shape_inputs(f"s1d{stage}", types)

        for problem in problems:
            st.warning(problem)

        st.divider()
        ghosts_on = st.toggle("🖐 이전 상 남기기", value=stage == 2, key="s1d_ghosts",
                              help="직전 3개의 상을 옅은 회색으로 함께 그립니다.")
        rainbow_on = st.toggle("🌈 무지개 대응 보기", value=False, key="s1d_rainbow",
                               help="도형을 호의 길이에 따라 무지개로 칠하고, 변환 후에도 같은 "
                                    "색을 물려줍니다.")
        numbers_on = rainbow_on and st.toggle("🔢 번호 붙이기", value=False, key="s1d_numbers")

    transformed = shape @ matrix.T
    signature = (stage, state["matrix_key"], shape_type,
                 float(np.round(shape, 3).sum()), float(np.round(shape, 3).std()))
    _note_operation(state, signature)
    if ghosts_on:
        _push_ghost(state, transformed, signature)

    with right:
        st.subheader("시각화 결과")
        figure = plot_shape(
            shape_type, shape, transformed, matrix, *coefficients,
            rainbow=rainbow_on, numbers=numbers_on,
            basis=st.session_state.get("s1_confirm_basis", False),
            morph=False,
            ghosts=list(state["ghosts"])[:-1] if ghosts_on else None,
            view_key=f"s1d-{stage}-{shape_type}")
        chart(figure, key="s1d_chart")

        _counterexample_notice(state, stage)
        _confirm_expander(state, stage)

    chat_dock(state, stage, matrix)


# ═══════════════════════════════════════════════════════════════════
# 자유 탐구 (예전 화면 그대로)
# ═══════════════════════════════════════════════════════════════════

def _free_lab():
    st.markdown("여러 도형을 여러 행렬로 일차변환하는 실험을 해 보세요.")
    col1, spacer, col2 = st.columns([1.4, 0.3, 2])  # 좌:입력 / 우:출력

    with col1:
        st.subheader("# 도형 입력 ___________________")
        shape_type, shape, (a, b, c), problems = shape_inputs("s1f")
        for problem in problems:
            st.warning(problem)

        st.subheader("# 2×2 변환 행렬 입력 ___________________")
        a11 = st.number_input("a11", value=1.0, step=0.5, format="%.1f")
        a12 = st.number_input("a12", value=-1.0, step=0.5, format="%.1f")
        a21 = st.number_input("a21", value=1.0, step=0.5, format="%.1f")
        a22 = st.number_input("a22", value=2.0, step=0.5, format="%.1f")
        matrix = np.array([[a11, a12], [a21, a22]])

        # ✅ 관찰 보조 (렌즈)
        st.divider()
        basis_on = st.toggle(
            "📦 기저 상자 보기", value=False, key="s1_basis",
            help="e₁, e₂ 가 만드는 단위정사각형과 그 상을 함께 그립니다. "
                 "a11 을 밀면 어느 화살표가 따라 움직이는지 보세요.",
        )
        morph_on = st.toggle(
            "🎞 연속 변환 보기", value=False, key="s1_morph",
            help="항등행렬에서 A 까지 (1−t)I + tA 로 건너가는 중간 상태를 보여 줍니다.",
        )
        rainbow_on = st.toggle(
            "🌈 무지개 대응 보기", value=False, key="s1_rainbow",
            help="도형을 호의 길이에 따라 무지개로 칠하고, 변환 후에도 같은 색을 "
                 "물려줍니다. 어느 부분이 어디로 갔는지 색으로 따라갈 수 있습니다.",
        )
        numbers_on = rainbow_on and st.toggle(
            "🔢 번호 붙이기", value=False, key="s1_numbers",
            help="등간격 10곳에 번호를 붙입니다. 변환 후 번호 간격이 얼마나 "
                 "흐트러졌는지로 어디가 늘어나고 어디가 눌렸는지 알 수 있습니다.",
        )

    with col2:
        transformed = np.dot(shape, matrix.T)

        st.subheader("시각화 결과")
        st.subheader("수식 표시")
        st.latex(
            fr"""\text{{입력된 행렬}} =
    \begin{{bmatrix}}
    {a11} & {a12} \\
    {a21} & {a22}
    \end{{bmatrix}}"""
        )

        if shape_type == "직선":
            st.latex(rf"\text{{입력된 직선:}} \quad {format_number(a)}x + {format_number(b)}y = {format_number(c)}")

        fig = plot_shape(shape_type, shape, transformed, matrix, a, b, c,
                         rainbow=rainbow_on, numbers=numbers_on,
                         basis=basis_on, morph=morph_on)
        chart(fig, key="s1_chart")
        if rainbow_on:
            st.caption(
                "🌈 색 = 변환 전 도형에서의 위치(호의 길이 기준). ○ 변환 전 / △ 변환 후. "
                "**직선에서는 색 간격이 그대로 남고, 원에서는 흐트러집니다 — 왜일까요?**"
            )
        if morph_on:
            st.caption(
                "⚠️ 이 경로는 **유일하지 않습니다.** 항등행렬에서 A 로 가는 길은 여럿이고, "
                "회전이라면 각도를 따라 도는 길이 더 자연스럽습니다. "
                "도중에 도형이 납작해지는 행렬도 있는데, 그때 $\\det$ 는 어떤 값을 지날까요?"
            )

        # ✅ 넓이비 계기판
        #
        # 도형을 삼각형 → 사각형 → 원으로 바꿔 가며 비가 그대로인지 보는 것이
        # 목적이다. 넓이가 없는 직선에서는 잴 것이 없으므로 건너뛴다.
        if shape_type != "직선":
            before_area = signed_area(shape)
            after_area = signed_area(transformed)
            determinant = float(np.linalg.det(matrix))
            st.subheader("계기판")
            gauge = st.columns(3)
            gauge[0].metric("변환 전 넓이", f"{abs(before_area):.3f}")
            gauge[1].metric("변환 후 넓이", f"{abs(after_area):.3f}")
            if abs(before_area) > 1e-9:
                gauge[2].metric("넓이비", f"{abs(after_area / before_area):.3f}")
            st.markdown(f"$\\det A$ = **{determinant:.3f}**")
            if determinant < 0:
                st.caption("↩️ 행렬식이 음수입니다 — 꼭짓점을 도는 **방향이 뒤집혔습니다.**")
            elif abs(determinant) < 1e-9:
                st.caption("⚠️ 행렬식이 0입니다 — 도형이 선분으로 붕괴합니다. 되돌릴 수 없습니다.")
            st.caption("도형 종류를 바꿔 가며 넓이비를 보세요. 무엇과 같아지나요?")


def run_transformation_by_matrix():
    st.header("🟩 (1) 행렬을 통한 일차변환 시뮬레이터")
    mode = st.radio("모드", ["🧭 단계별 탐구", "🔬 자유 탐구"],
                    horizontal=True, key="s1_mode", label_visibility="collapsed")
    if mode.startswith("🧭"):
        _discovery()
    else:
        _free_lab()

"""expression_parser 의 동작과 방어를 확인하는 스크립트.

pytest 없이 그냥 실행한다. 저장소 루트에서:

    python test/test_expression_parser.py

모두 통과하면 마지막 줄에 ALL PASS 가 찍히고 종료 코드는 0,
하나라도 실패하면 실패 목록과 함께 종료 코드 1.

확인하는 것은 네 가지다.
  1. 학생이 실제로 넣을 법한 자취 정의식이 제대로 계산되는가
  2. 복소함수식의 결과가 numpy 로 직접 계산한 값과 일치하는가
  3. **코드 실행을 노린 입력이 전부 차단되는가** (이 파일의 핵심)
  4. 잘못된 입력에 한국어 안내가 나가는가
"""

import os
import sys
from pathlib import Path

# 저장소 루트를 import 경로에 넣는다 (이 파일은 test/ 안에 있다).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from expression_parser import ExpressionError, compile_complex_function, compile_locus

# 윈도우 콘솔(cp949)에서도 한글이 깨지지 않게.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

fails = []


def make_grid(r=8, N=400):
    axis = np.linspace(-r, r, N)
    X, Y = np.meshgrid(axis, axis)
    eps = (2 * r) / (N - 1) * 2
    return X, Y, eps


X, Y, EPS = make_grid()
Z = np.array([1 + 1j, 0 + 0j, 2 - 1j, -1 + 0.5j])


print("=== 1. 자취 정의식: 정상 입력 ===")
LOCUS_CASES = [
    "x**2 + y**2 == 1",        # 앱 기본값 (원)
    "2*y == x**2 + 1",         # 입력창 예시 (포물선)
    "y == x",
    "x**2/4 + y**2/9 == 1",    # 타원
    "x**2 - y**2 == 1",        # 쌍곡선
    "y == sin(x)",
    "y == exp(x)",
    "y == np.cos(x)",          # 옛 코드가 열어 두었던 np. 접두 호환
    "sqrt(x**2+y**2) == 3",
    "y > x**2",                # 부등식
    "(x > 0) & (y > 0)",       # 조건 결합
    "(x > 0) and (y > 0)",     # and → &
    "0 < y < 1",               # 연쇄 비교
    "x**2 + y**2 - 4",         # 비교 없음 → '= 0' 자취로 해석
    "abs(x) + abs(y) == 2",
]
for src in LOCUS_CASES:
    try:
        mask = compile_locus(src)(X, Y, EPS)
        count = int(mask.sum())
        ok = count > 0 and mask.dtype == bool and mask.shape == X.shape
        print(f"  {'OK  ' if ok else 'FAIL'}  {src:<28} 점 {count}")
        if not ok:
            fails.append(src)
    except Exception as e:
        print(f"  FAIL  {src:<28} {type(e).__name__}: {e}")
        fails.append(src)


print("\n=== 2. 복소함수식: numpy 직접 계산과 일치하는가 ===")
FUNCTION_CASES = [
    ("(z - 1j)**2", lambda W: np.allclose(W, (Z - 1j) ** 2)),      # 앱 기본값
    ("z**2", lambda W: np.allclose(W, Z ** 2)),
    ("(z - i)**2", lambda W: np.allclose(W, (Z - 1j) ** 2)),        # i 표기
    ("2*z + 3 - 1j", lambda W: np.allclose(W, 2 * Z + 3 - 1j)),
    ("conj(z)", lambda W: np.allclose(W, np.conj(Z))),
    ("exp(z)", lambda W: np.allclose(W, np.exp(Z))),
    ("abs(z)", lambda W: np.allclose(W, np.abs(Z))),
    ("re(z)", lambda W: np.allclose(W, Z.real)),
    ("z*(cos(pi/3) + 1j*sin(pi/3))", lambda W: np.allclose(W, Z * np.exp(1j * np.pi / 3))),
    ("5", lambda W: np.allclose(W, 5)),                             # 상수 → 브로드캐스트
    ("1/z", lambda W: np.isinf(W[1]) or np.isnan(W[1])),            # 0 나누기에도 안 죽음
]
for src, matches in FUNCTION_CASES:
    try:
        W = compile_complex_function(src)(Z)
        ok = W.shape == Z.shape and W.dtype == complex and matches(W)
        print(f"  {'OK  ' if ok else 'FAIL'}  {src:<30} -> {W[0]}")
        if not ok:
            fails.append(src)
    except Exception as e:
        print(f"  FAIL  {src:<30} {type(e).__name__}: {e}")
        fails.append(src)


print("\n=== 3. 보안: 코드 실행 시도는 전부 차단되어야 함 ===")
ATTACKS = [
    "__import__('os').system('echo pwned')",
    "().__class__.__bases__[0].__subclasses__()",
    "open('/etc/passwd').read()",
    "eval('1+1')",
    "exec('import os')",
    "[c for c in ().__class__.__mro__]",
    "(lambda: 1)()",
    "x.__class__",
    "globals()",
    "os.system('ls')",
    "np.__loader__",
    "'a'*10",
    "x if True else y",
    "{'a':1}",
    "[1,2,3]",
    "print(1)",
    "x := 5",
    "10**10**10",   # 거대 정수로 앱을 멈추게 하는 입력
    "2**99999",
]
for src in ATTACKS:
    for label, compile_fn in (("자취", compile_locus), ("f(z)", compile_complex_function)):
        try:
            compiled = compile_fn(src)
        except ExpressionError as e:
            print(f"  OK    [{label}] 차단: {src[:32]:<34} ({e})")
            continue
        except Exception as e:
            print(f"  FAIL  [{label}] {src!r} 예상 밖 예외 {type(e).__name__}: {e}")
            fails.append(f"SECURITY-EXC {src}")
            continue
        # 컴파일이 통과했다면 그 자체가 실패다.
        try:
            result = compiled(X, Y, EPS) if label == "자취" else compiled(Z)
        except Exception as e:
            result = f"<계산 단계 {type(e).__name__}>"
        print(f"  FAIL  [{label}] {src!r} 가 통과함 -> {result!r}")
        fails.append(f"SECURITY {src}")


print("\n=== 4. 잘못된 입력: 한국어로 안내되어야 함 ===")
BAD_INPUTS = ["", "   ", "x**2 +", "y == zzz", "foo(x)", "x $ y", "x**2 == ", "y = x"]
for src in BAD_INPUTS:
    try:
        compile_locus(src)
        print(f"  FAIL  {src!r} 가 통과함")
        fails.append(f"BAD {src}")
    except ExpressionError as e:
        print(f"  OK    {src!r:<12} -> {e}")
    except Exception as e:
        print(f"  FAIL  {src!r} 예상 밖 예외 {type(e).__name__}: {e}")
        fails.append(f"BAD-EXC {src}")


print("\n=== 5. 부작용 확인 ===")
print("  'pwned' 파일이 생기지 않았는가:", not os.path.exists("pwned"))

print("\n" + "=" * 52)
if fails:
    print(f"FAILURES {len(fails)}: {fails}")
else:
    print("ALL PASS")
sys.exit(1 if fails else 0)

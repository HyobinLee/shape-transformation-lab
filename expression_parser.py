"""사용자가 입력한 수식을 `eval` 없이 안전하게 계산 가능한 형태로 바꾸는 파서.

섹션 3(복소평면에서의 이동)은 학생이 도형의 자취와 복소함수를 직접 타이핑하게
한다. 그 표현력은 이 앱의 핵심이므로 유지하되, 입력 문자열을 그대로 `eval` 하면
임의 코드 실행 경로가 된다. 그래서 두 단계를 거친다.

    1. Python `ast` 로 파싱해 **허용 목록에 없는 구문을 전부 거부**한다.
       (속성 접근, 첨자, 람다, 컴프리헨션, 문자열 등은 여기서 막힌다.)
    2. 통과한 식만 sympy 로 넘겨 numpy 벡터 함수로 컴파일한다.

1단계가 실질적인 방어선이고, sympy 는 그 위에서 수식을 다루는 역할을 한다.
sympy 의 `parse_expr` 자체는 내부적으로 `eval` 을 쓰므로 단독으로는 안전하지
않다 — 반드시 1단계를 먼저 통과시켜야 한다.

앱 전체의 방침대로, 잘못된 입력에는 예외를 흘리지 않고 학생이 읽을 수 있는
한국어 메시지를 담은 `ExpressionError` 를 돌려준다.
"""

import ast

import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations


class ExpressionError(Exception):
    """사용자 입력 수식이 허용되지 않거나 계산할 수 없을 때."""


# ── 허용 목록 ────────────────────────────────────────────────────────────────

#: 학생이 쓸 수 있는 함수 이름 → sympy 함수.
ALLOWED_FUNCTIONS = {
    'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
    'asin': sympy.asin, 'acos': sympy.acos, 'atan': sympy.atan,
    'arcsin': sympy.asin, 'arccos': sympy.acos, 'arctan': sympy.atan,
    'sinh': sympy.sinh, 'cosh': sympy.cosh, 'tanh': sympy.tanh,
    'exp': sympy.exp, 'log': sympy.log, 'ln': sympy.log, 'sqrt': sympy.sqrt,
    'abs': sympy.Abs, 'Abs': sympy.Abs, 'sign': sympy.sign,
    'floor': sympy.floor, 'ceil': sympy.ceiling, 'ceiling': sympy.ceiling,
    'conj': sympy.conjugate, 'conjugate': sympy.conjugate,
    're': sympy.re, 'Re': sympy.re, 'real': sympy.re,
    'im': sympy.im, 'Im': sympy.im, 'imag': sympy.im,
    'arg': sympy.arg,
}

#: 함수가 아닌 이름(상수) → sympy 객체. 변수(x, y, z)는 호출부에서 더해진다.
ALLOWED_CONSTANTS = {
    'pi': sympy.pi,
    'e': sympy.E, 'E': sympy.E,
    'i': sympy.I, 'I': sympy.I,
}

#: 산술에 허용되는 이항 연산자.
_ARITH_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
#: 조건(마스크)을 결합하는 데 허용되는 이항 연산자.
_LOGIC_OPS = (ast.BitAnd, ast.BitOr, ast.BitXor)

_COMPARISONS = {
    ast.Lt: sympy.StrictLessThan,
    ast.LtE: sympy.LessThan,
    ast.Gt: sympy.StrictGreaterThan,
    ast.GtE: sympy.GreaterThan,
}

#: 수식 길이 상한.
_MAX_SOURCE_LENGTH = 500
#: 상수 지수의 상한. `2**99999` 처럼 거대 정수를 만들어 앱을 멈추게 하는 입력 차단용.
_MAX_CONSTANT_EXPONENT = 1000
#: 상수끼리의 거듭제곱 결과의 상한. `(2**500)**500` 같은 우회 차단용.
_MAX_CONSTANT_MAGNITUDE = 1e15

_SYMPY_GLOBALS = {
    'Symbol': sympy.Symbol, 'Integer': sympy.Integer,
    'Float': sympy.Float, 'Rational': sympy.Rational,
    **ALLOWED_FUNCTIONS, **ALLOWED_CONSTANTS,
}


# ── 1단계: AST 검문 및 정규화 ────────────────────────────────────────────────

def _normalize(node):
    """허용되지 않은 구문을 거부하면서, 파이썬다운 표기를 sympy 가 다룰 수 있는
    형태로 바꾼 새 AST 를 돌려준다.

    바꾸는 것:
      * ``np.sin`` / ``numpy.sin`` → ``sin``      (옛 코드가 `np` 를 열어 두었던 것과의 호환)
      * ``a and b`` / ``a or b`` / ``not a`` → ``a & b`` / ``a | b`` / ``~a``
      * ``0 < y < 1`` (연쇄 비교) → ``(0 < y) & (y < 1)``

    뒤의 두 가지는 파이썬이 그 자리에서 진리값을 요구하는 표기라, sympy 의
    관계식으로는 그대로 평가할 수 없기 때문에 풀어 쓴다.
    """
    if isinstance(node, ast.Expression):
        return ast.Expression(body=_normalize(node.body))

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex, bool)):
            return node
        raise ExpressionError('숫자만 쓸 수 있습니다. 문자열이나 다른 값은 넣을 수 없습니다.')

    if isinstance(node, ast.Name):
        return node

    if isinstance(node, ast.Attribute):
        # `np.sin` 형태만 허용하고 앞의 `np.` 를 떼어 낸다.
        if isinstance(node.value, ast.Name) and node.value.id in ('np', 'numpy'):
            return ast.Name(id=node.attr, ctx=ast.Load())
        raise ExpressionError(f"'.' 을 이용한 표기는 쓸 수 없습니다: '{ast.unparse(node)}'")

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ARITH_OPS + _LOGIC_OPS):
            raise ExpressionError(f"쓸 수 없는 연산자입니다: '{type(node.op).__name__}'")
        if isinstance(node.op, ast.Pow):
            _check_exponent(node)
        return ast.BinOp(left=_normalize(node.left), op=node.op,
                         right=_normalize(node.right))

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return ast.UnaryOp(op=ast.Invert(), operand=_normalize(node.operand))
        if isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
            return ast.UnaryOp(op=node.op, operand=_normalize(node.operand))
        raise ExpressionError(f"쓸 수 없는 연산자입니다: '{type(node.op).__name__}'")

    if isinstance(node, ast.BoolOp):
        op = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        values = [_normalize(v) for v in node.values]
        combined = values[0]
        for right in values[1:]:
            combined = ast.BinOp(left=combined, op=op, right=right)
        return combined

    if isinstance(node, ast.Compare):
        # 연쇄 비교를 두 항씩 끊어 & 로 잇는다.
        operands = [_normalize(node.left)] + [_normalize(c) for c in node.comparators]
        parts = [
            ast.Compare(left=operands[k], ops=[op], comparators=[operands[k + 1]])
            for k, op in enumerate(node.ops)
        ]
        combined = parts[0]
        for part in parts[1:]:
            combined = ast.BinOp(left=combined, op=ast.BitAnd(), right=part)
        return combined

    if isinstance(node, ast.Call):
        # `np.cos(x)` 의 앞부분을 먼저 떼어 낸 뒤에 이름을 확인한다.
        func = _normalize(node.func) if isinstance(node.func, ast.Attribute) else node.func
        if not isinstance(func, ast.Name) or func.id not in ALLOWED_FUNCTIONS:
            raise ExpressionError(f"쓸 수 없는 함수입니다: '{ast.unparse(node.func)}'")
        if node.keywords:
            raise ExpressionError('함수에 이름 붙은 인자는 쓸 수 없습니다.')
        return ast.Call(func=func, args=[_normalize(a) for a in node.args],
                        keywords=[])

    raise ExpressionError(f'쓸 수 없는 표현입니다: {type(node).__name__}')


def _static_float(node):
    """상수만으로 이루어진 식이면 그 값을 float(또는 complex)로, 아니면 None을 돌려준다.

    일부러 **float 영역에서만** 계산한다. 파이썬 정수로 계산하면 `10**10**10`
    같은 입력이 여기서 이미 멈춰 버리지만, float 은 넘치면 `inf` 가 될 뿐이라
    안전하게 크기만 가늠할 수 있다.
    """
    if isinstance(node, ast.Constant):
        return complex(node.value) if isinstance(node.value, complex) else float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _static_float(node.operand)
        if value is None:
            return None
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ARITH_OPS):
        left, right = _static_float(node.left), _static_float(node.right)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            return left ** right
        except (OverflowError, ZeroDivisionError, ValueError):
            return None
    return None


def _check_exponent(node):
    """거듭제곱이 거대 정수로 번지지 않는지 확인한다.

    지수에 변수가 들어 있으면(`2**x`) sympy 가 기호로 두므로 위험하지 않다.
    상수뿐일 때만 크기를 따진다.
    """
    exponent = _static_float(node.right)
    if exponent is not None and abs(exponent) > _MAX_CONSTANT_EXPONENT:
        raise ExpressionError(f'지수가 너무 큽니다 (최대 {_MAX_CONSTANT_EXPONENT}).')

    whole = _static_float(node)
    if whole is not None and not (abs(whole) <= _MAX_CONSTANT_MAGNITUDE):
        raise ExpressionError('계산 결과가 너무 큽니다.')


def _check_names(tree, variables):
    """AST 에 남은 이름이 전부 허용 목록 안에 있는지 확인한다."""
    allowed = set(variables) | set(ALLOWED_CONSTANTS) | set(ALLOWED_FUNCTIONS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in allowed:
            raise ExpressionError(
                f"모르는 이름입니다: '{node.id}'. "
                f"쓸 수 있는 변수는 {', '.join(sorted(variables))} 입니다."
            )


def _parse_to_ast(text, variables):
    """입력 문자열을 검문·정규화된 AST 로 바꾼다."""
    text = (text or '').strip()
    if not text:
        raise ExpressionError('식을 입력해 주세요.')
    if len(text) > _MAX_SOURCE_LENGTH:
        raise ExpressionError(f'식이 너무 깁니다 (최대 {_MAX_SOURCE_LENGTH}자).')
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError:
        raise ExpressionError('식의 문법이 맞지 않습니다. 괄호와 연산자를 확인해 주세요.')

    tree = _normalize(tree)
    _check_names(tree, variables)
    return ast.fix_missing_locations(tree)


# ── 2단계: sympy 로 컴파일 ───────────────────────────────────────────────────

def _to_sympy(node, symbols):
    """검문을 통과한 산술 AST 조각을 sympy 식으로 바꾼다."""
    source = ast.unparse(node)
    try:
        expr = parse_expr(source, local_dict=dict(symbols),
                          global_dict=_SYMPY_GLOBALS,
                          transformations=standard_transformations)
    except ExpressionError:
        raise
    except Exception:
        raise ExpressionError(f"식을 이해하지 못했습니다: '{source}'")
    if isinstance(expr, bool):
        raise ExpressionError(f"'{source}' 는 수식이 아닙니다.")
    return expr


def _build_predicate(node, symbols, eps):
    """AST 를 참/거짓 조건(sympy Boolean)으로 바꾼다.

    비교가 없는 산술식은 '자취'로 보고 ``|식| < eps`` 로 다룬다. 부동소수점에서
    등식이 정확히 성립하는 격자점은 사실상 없으므로, 등식은 언제나 격자 간격에
    비례한 허용오차로 근사 판정해야 한다.
    """
    if isinstance(node, ast.Compare):
        op = type(node.ops[0])
        left = _to_sympy(node.left, symbols)
        right = _to_sympy(node.comparators[0], symbols)
        if op is ast.Eq:
            return sympy.Abs(left - right) < eps
        if op is ast.NotEq:
            return sympy.Abs(left - right) >= eps
        if op in _COMPARISONS:
            return _COMPARISONS[op](left, right)
        raise ExpressionError('쓸 수 없는 비교입니다. ==, <, <=, >, >= 를 써 주세요.')

    if isinstance(node, ast.BinOp) and isinstance(node.op, _LOGIC_OPS):
        left = _build_predicate(node.left, symbols, eps)
        right = _build_predicate(node.right, symbols, eps)
        if isinstance(node.op, ast.BitAnd):
            return sympy.And(left, right)
        if isinstance(node.op, ast.BitOr):
            return sympy.Or(left, right)
        return sympy.Xor(left, right)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return sympy.Not(_build_predicate(node.operand, symbols, eps))

    # 비교가 없는 식 → 자취 '식 = 0' 으로 해석.
    return sympy.Abs(_to_sympy(node, symbols)) < eps


def _lambdify(args, expr):
    try:
        return sympy.lambdify(args, expr, 'numpy')
    except Exception:
        raise ExpressionError('식을 계산할 수 있는 형태로 바꾸지 못했습니다.')


# ── 공개 API ─────────────────────────────────────────────────────────────────

def compile_locus(text):
    """자취 정의식(예: ``x**2 + y**2 == 1``)을 마스크 함수로 컴파일한다.

    Returns:
        ``mask(X, Y, eps) -> bool ndarray`` — `X`, `Y` 격자에서 식을 만족하는
        점을 True 로 표시한다. `eps` 는 등식 판정의 허용오차.

    Raises:
        ExpressionError: 허용되지 않은 구문이거나 해석할 수 없을 때.
    """
    x, y = sympy.symbols('x y', real=True)
    eps = sympy.Symbol('eps', positive=True)
    symbols = {'x': x, 'y': y}

    tree = _parse_to_ast(text, variables=symbols)
    predicate = _build_predicate(tree.body, symbols, eps)
    func = _lambdify((x, y, eps), predicate)

    def mask(X, Y, tolerance):
        with np.errstate(all='ignore'):
            result = func(X, Y, tolerance)
        result = np.asarray(result)
        if result.dtype != bool:
            result = result.astype(bool)
        if result.shape != np.shape(X):
            result = np.broadcast_to(result, np.shape(X))
        return result

    return mask


def compile_complex_function(text):
    """복소함수식(예: ``(z - 1j)**2``)을 계산 함수로 컴파일한다.

    Returns:
        ``apply(Z) -> complex ndarray`` — 복소수 배열 `Z` 에 함수를 적용한다.

    Raises:
        ExpressionError: 허용되지 않은 구문이거나 해석할 수 없을 때.
    """
    z = sympy.Symbol('z')
    symbols = {'z': z}

    tree = _parse_to_ast(text, variables=symbols)
    expr = _to_sympy(tree.body, symbols)
    func = _lambdify((z,), expr)

    def apply(Z):
        with np.errstate(all='ignore'):
            result = func(Z)
        result = np.asarray(result)
        if result.shape != np.shape(Z):
            result = np.broadcast_to(result, np.shape(Z))
        return result.astype(complex, copy=False)

    return apply

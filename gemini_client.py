"""Gemini REST 호출을 한곳에 모은다 — 네트워크·키·타임아웃의 경계.

[expression_parser.py](expression_parser.py) 가 **보안 경계**라서 분리된 것과 같은
논리로 분리했다. 섹션은 "무엇을 물을지"만 넘기고, 키·모델명·재시도·실패 처리는
전부 여기서 끝난다. 섹션 파일에 `urllib` 이 등장하기 시작하면 그 수업이 무엇에
대한 것인지 읽을 수 없게 된다.

**학생이 읽는 문구는 여기 두지 않는다.** 실패했다는 사실만 예외로 알리고, 그걸
어떤 말로 학생에게 전할지는 섹션이 정한다 — `lab_ui.py` 에 그은 것과 같은 경계다.

의존성을 늘리지 않는다. Gemini 는 그냥 JSON 을 받는 HTTPS 엔드포인트이므로
표준 라이브러리 `urllib` 로 충분하다. SDK 하나를 위해 `pip install` 목록이
길어지는 것이 이 프로젝트에서는 더 큰 비용이다([intent.md](docs/intent.md) 3-(e)).
"""

import json
import urllib.error
import urllib.request

import streamlit as st

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
FALLBACK_MODEL = "gemini-3.5-flash"   # secrets 에 GEMINI_MODEL 이 없을 때만 쓴다
TIMEOUT_SEC = 8.0
RETRIES = 1                            # 첫 시도 + 재시도 1회


class GeminiUnavailable(Exception):
    """키가 없거나, 네트워크가 죽었거나, 응답이 규격에 맞지 않는다.

    호출부는 이 예외 하나만 잡으면 된다. 무엇이 잘못됐는지는 메시지에 담기지만
    **그 메시지를 학생에게 그대로 보여 주지는 않는다** — 섹션이 자기 말투로 옮긴다.
    """


def _secret(name, default=None):
    """secrets.toml 이 아예 없어도 죽지 않게 감싼다.

    `st.secrets[...]` 는 파일이 없으면 KeyError 가 아니라 별도 예외를 던진다.
    학생이 이 저장소를 클론해서 키 없이 실행하는 것이 정상 경로이므로, 여기서
    조용히 기본값으로 되돌아가는 것이 이 앱의 방침에 맞는다.
    """
    try:
        return st.secrets[name]
    except Exception:
        return default


def is_available():
    """키가 있는가. 섹션은 이 값으로 '똑똑한 모드/둔한 모드'를 가른다."""
    return bool(_secret("GEMINI_API_KEY"))


def model_name():
    return _secret("GEMINI_MODEL", FALLBACK_MODEL)


def _post(url, payload, api_key):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SEC) as response:
        return json.loads(response.read().decode("utf-8"))


def ask_json(system, user, *, temperature=0.2, max_output_tokens=600):
    """구조화된 JSON 하나를 받아 온다. 실패하면 GeminiUnavailable.

    **왜 JSON 만 받는가**: 이 앱에서 모델은 판정하지 않는다. 학생의 문장을 미리
    정해 둔 주장(claim) 목록으로 옮기는 번역기일 뿐이고, 참·거짓은 섹션의
    검증기가 숫자로 정한다. 자유 문장을 받으면 그 경계가 흐려진다.
    """
    api_key = _secret("GEMINI_API_KEY")
    if not api_key:
        raise GeminiUnavailable("GEMINI_API_KEY 가 없습니다.")

    url = ENDPOINT.format(model=model_name())
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
        },
    }

    last_error = None
    for attempt in range(RETRIES + 1):
        try:
            data = _post(url, payload, api_key)
            break
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")[:200]
            last_error = f"HTTP {error.code}: {body}"
            if error.code in (400, 401, 403, 404):
                # 키·모델명·요청 형식 문제는 다시 보낸다고 달라지지 않는다.
                raise GeminiUnavailable(last_error) from error
        except Exception as error:               # 타임아웃, DNS, 끊긴 연결
            last_error = f"{type(error).__name__}: {error}"
        if attempt == RETRIES:
            raise GeminiUnavailable(last_error or "알 수 없는 실패")

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as error:
        # 안전필터에 걸리면 candidates 가 비어 온다. 이것도 '사용 불가'로 묶는다.
        raise GeminiUnavailable(f"응답 형식이 예상과 다릅니다: {str(data)[:200]}") from error

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise GeminiUnavailable(f"JSON 이 아닙니다: {text[:200]}") from error

    if not isinstance(parsed, dict):
        raise GeminiUnavailable(f"객체가 아닙니다: {text[:200]}")
    return parsed


@st.cache_data(show_spinner=False, ttl=3600, max_entries=256)
def ask_json_cached(system, user, *, temperature=0.2):
    """같은 (지시, 학생 문장) 쌍은 다시 묻지 않는다.

    학생은 같은 문장을 여러 번 제출한다(오타 수정, 새로고침). 그때마다 요금을
    내고 3초를 기다릴 이유가 없다. 캐시는 예외를 저장하지 않으므로 실패는
    그대로 매번 다시 시도된다 — 네트워크가 돌아오면 즉시 회복된다.
    """
    return ask_json(system, user, temperature=temperature)

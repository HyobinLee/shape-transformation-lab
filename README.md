# shape-transformation-lab

행렬과 복소평면을 이용하는 도형의 변환을 관찰하고 실험함.
Fork: 2026-08-03 by Hyobinlee

여덟 개의 실험실이 있습니다. **기초(1~4)** 는 교과서의 문장을 눈으로 확인하게 하고,
**심화(5~8)** 는 각각 기초 하나를 이어받아 그 뒤에 숨은 구조를 학생이 직접 찾게 합니다.

| | 기초 | 심화 |
| --- | --- | --- |
| 행렬 | 1. 일차변환 | **5. 고유공간** — 변환이 제자리에 두는 직선은 몇 개인가 |
| 대칭 | 2. 대칭/회전변환 | **6. 등거리변환** — 거울을 몇 개 놓아도 결과는 넷뿐이다 |
| 복소평면 | 3. 복소함수 사상 | **8. 뫼비우스** — 원은 원으로 간다. 직선도 원일까 |
| 합성 | 4. 회전+평행이동 | **7. 반복과 궤도** — 되풀이하면 어디로 가는가 |

Streamlit 기반 웹 앱입니다. **Docker는 필요하지 않습니다** (배경은 아래 [Docker에 대하여](#docker에-대하여) 참고).

---

## 빠른 시작 (로컬 venv)

Python **3.11 이상**이 설치되어 있으면 됩니다. 저장소를 클론/동기화한 뒤, 저장소 루트에서:

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

> `Activate.ps1` 실행이 막히면 (실행 정책 오류):
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` 를 먼저 실행.
>
> `python` 이 없다고 나오면 `py -3` 또는 `python3` 으로 대체.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 **[http://localhost:8501](http://localhost:8501)** 로 접속. 소스를 저장하면 Streamlit이 자동으로 다시 로드합니다.

종료는 터미널에서 `Ctrl+C`, venv 해제는 `deactivate`.

---

## 다른 IDE / 다른 PC로 옮길 때 (마이그레이션)

`.venv/` 는 **저장소에 포함되지 않습니다**(`.gitignore` 처리). 가상환경에는 절대경로가 하드코딩되어 있어 복사해서 옮기면 깨지므로, 새 환경에서는 **항상 새로 만드는 것이 정상 절차**입니다.

### 절차

1. 저장소를 클론(또는 동기화)한다.
2. 위 [빠른 시작](#빠른-시작-로컬-venv)의 명령을 그대로 실행해 `.venv` 를 새로 만든다.
3. IDE에서 인터프리터를 `.venv` 로 지정한다 (아래 표 참고).
4. `streamlit run app.py` 로 확인한다.

동기화 시 **절대 옮기지 말아야 할 것**: `.venv/`, `__pycache__/`, `*.pyc`. 모두 각 머신에서 재생성되는 산출물입니다.

### IDE별 인터프리터 지정

| IDE | 방법 |
| --- | --- |
| **VS Code** | `Ctrl+Shift+P` → `Python: Select Interpreter` → `.venv` 선택. 이후 통합 터미널이 자동으로 venv를 활성화합니다. |
| **PyCharm** | `Settings` → `Project` → `Python Interpreter` → ⚙ → `Add Local Interpreter` → `Existing` → `.venv/Scripts/python.exe` (Win) / `.venv/bin/python` (mac·Linux) |
| **Cursor / Windsurf** | VS Code와 동일 |
| **JupyterLab** | venv 활성화 후 `pip install ipykernel && python -m ipykernel install --user --name shape-lab` |
| **터미널만 사용** | 매번 `Activate.ps1` / `source .venv/bin/activate` 후 작업 |

### 환경이 꼬였을 때 (초기화)

venv는 언제든 통째로 지우고 다시 만들어도 안전합니다.

```powershell
# Windows
Remove-Item -Recurse -Force .venv
```

```bash
# macOS / Linux
rm -rf .venv
```

이후 [빠른 시작](#빠른-시작-로컬-venv) 명령을 다시 실행.

### 의존성을 추가했다면

```bash
pip install <패키지>
pip freeze > requirements-lock.txt   # 선택: 정확한 버전 기록용
```

`requirements.txt` 에는 패키지 이름만 추가하고, 버전 고정이 필요하면 별도 lock 파일을 쓰는 편이 IDE·OS 간 이동에 유리합니다.

---

## 프로젝트 구조

```text
app.py                                # 사이드바 라우터 (진입점)
lab_ui.py                             # 섹션 공통 화면 배관 (수업 내용은 들어가지 않음)
expression_parser.py                  # 섹션3의 수식 입력을 안전하게 파싱 (AST 검문 + sympy)
gemini_client.py                      # 섹션1 탐구 챗봇의 네트워크 경계 (표준 urllib)

section1_transformation_by_matrix.py  # 기초 1. 행렬을 통한 일차변환
section2_symmetry_rotation.py         # 기초 2. 행렬을 통한 대칭/회전변환
section3_complex_plane.py             # 기초 3. 복소평면에서의 이동
section4_rotation_translation.py      # 기초 4. 복소평면에서의 회전/평행이동
section5_eigenspace.py                # 심화 5. 일차변환의 고유공간          (1 을 이어받음)
section6_isometry.py                  # 심화 6. 거울을 몇 번 놓아야 하는가   (2 를 이어받음)
section7_orbit.py                     # 심화 7. 되풀이하면 어디로 가는가     (4 를 이어받음)
section8_mobius.py                    # 심화 8. 원은 원으로 간다 (뫼비우스)  (3 을 이어받음)

test/                                 # 실행 스크립트 형태의 검증 (pytest 아님)
backup/                               # 미사용 이전 버전 보관
docs/intent.md                        # 프로젝트 구성과 개발 철학
docs/260804_0200_plans.md             # 확장 계획과 그 근거
requirements.txt
```

- [app.py](app.py) 는 라우팅만 담당하고, 실제 화면은 각 `section*.py` 의 `run_*()` 함수에 있습니다. **기능 수정은 대부분 해당 섹션 파일만 고치면 됩니다.**
- [lab_ui.py](lab_ui.py) 에는 섹션마다 반복되는 **화면 배관만** 들어갑니다 — 축 설정, 그래프 `key`/`uirevision`, 자취, 무지개 대응. **수학과 학생이 읽는 문구는 전부 섹션 파일에 남습니다.** 그래야 섹션 파일 하나만 읽어도 그 수업이 통째로 이해됩니다.
- [expression_parser.py](expression_parser.py) 는 섹션3이 학생의 수식 입력(`x**2 + y**2 == 1`, `(z - 1j)**2`)을 `eval` 없이 계산하기 위해 쓰는 모듈입니다. 자세한 배경은 [docs/intent.md](docs/intent.md) 참고.
- `backup/` 의 파일들(섹션 분리 이전의 단일 파일 버전, 섹션2를 Dash로 시도한 실험)은 `app.py` 에서 import되지 않습니다. **`backup/app_backup.py` 에는 교체 이전의 `eval` 이 그대로 남아 있으니 거기서 코드를 되살려 쓰지 마세요.**

### 관찰 보조 장치 (모든 섹션 공통)

기능이 아니라 **보는 방식**입니다. 전부 토글이고, 기본값은 꺼짐입니다.

| 장치 | 하는 일 |
| --- | --- |
| 🌈 **무지개 대응** | 변환 전 도형을 위치에 따라 무지개로 칠하고 변환 후에도 같은 색을 물려줍니다. 어느 점이 어디로 갔는지 색으로 따라갈 수 있습니다. |
| 🔢 **번호 붙이기** | 등간격 10곳에 번호를 붙입니다. 색을 못 봐도 대응이 읽히고, 번호 간격이 흐트러진 정도로 늘어남을 수로 읽습니다. |
| ✏️ **자취 남기기** | 점을 옮긴 자국을 남깁니다. 점 도구가 자취 도구가 됩니다. |
| 🔎 **정체 밝히기** | 정답(고유값, 회전 중심, 변환의 이름)을 보여 줍니다. **먼저 추측한 뒤에 켜세요.** |

무지개를 켜면 색상 채널을 그쪽이 가져가므로, 변환 전/후 구분은 **마커 모양**(○ 전 / △ 후)이 맡습니다.

### 검증 실행

pytest는 쓰지 않습니다. 그냥 스크립트로 돌아갑니다.

```bash
python test/test_expression_parser.py   # 파서 — 정상 입력·보안 차단·오류 안내
python test/test_lab_ui.py              # 무지개 매개변수화, 자취, 넓이비 = |det|
python test/test_eigenspace.py          # 섹션 5 — 고유공간 분류, 어긋난 각 곡선
python test/test_isometry.py            # 섹션 6 — 등거리변환 분류 (정리 자체를 확인)
python test/test_orbit.py               # 섹션 4·7 — 고정점과 궤도
python test/test_mobius.py              # 섹션 8 — 원이 정말 원으로 가는지 재 봄
python test/test_section1_discovery.py  # 섹션 1 단계별 탐구 — 명제·힌트 누설·판정기
python test/test_sections_render.py     # 여덟 섹션이 이상한 입력에도 죽지 않는지
```

각각 마지막 줄에 `ALL PASS` 가 찍히면 성공입니다 (종료 코드 0).
`test_sections_render.py` 는 Streamlit의 `AppTest` 로 앱을 헤드리스로 띄우므로 조금 느립니다(1~2분).

---

## 알려진 주의사항

- **Streamlit 1.60 이상이 필요합니다.** `st.fragment`, `st.plotly_chart(on_select=...)`, `width` 기본값이 모두 최근 버전에 들어왔습니다. `requirements.txt` 에 하한을 명시해 두었습니다.
- 그래프의 한글은 **브라우저에 설치된 폰트**로 그려집니다(Plotly). 저장소에 폰트 파일을 넣을 필요가 없어 `fonts/` 는 추적하지 않습니다.
- 그래프를 **클릭**하면 입력점이 그 자리로 옮겨 갑니다(섹션 2·4·5·7). 숫자 칸에도 그대로 반영됩니다.
- 슬라이더 중 일부(섹션 1의 `t`, 섹션 5의 `θ`, 섹션 7의 `n`)는 **브라우저 안에서만** 돕니다. 서버를 오가지 않으므로 화면이 깜빡이지 않고, ▶ 버튼으로 재생할 수도 있습니다.

---

## Docker에 대하여

이 저장소는 **GitHub Codespaces용 Streamlit 템플릿**에서 fork되어, 원래 `.devcontainer/devcontainer.json` 을 포함하고 있었습니다. 그 파일이 있으면 VS Code가 "Reopen in Container"를 제안하고 **그때만** 로컬 Docker Desktop이 필요해집니다.

앱 자체는 Docker에 의존하지 않으므로 로컬 개발에는 위의 venv 방식만으로 충분합니다. 이에 따라 Dev Container 설정(`.devcontainer/`)은 제거했으며, 어떤 IDE에서도 venv만으로 동일하게 개발할 수 있습니다.

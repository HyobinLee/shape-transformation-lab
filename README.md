# shape-transformation-lab

행렬과 복소평면을 이용하는 도형의 변환을 관찰하고 실험함.
Fork: 2026-08-03 by Hyobinlee

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
section1_transformation_by_matrix.py  # 1. 행렬을 통한 일차변환
section2_symmetry_rotation.py         # 2. 행렬을 통한 대칭/회전변환
section3_complex_plane.py             # 3. 복소평면에서의 이동
section4_rotation_translation.py      # 4. 복소평면에서의 회전/평행이동
expression_parser.py                  # 섹션3의 수식 입력을 안전하게 파싱 (AST 검문 + sympy)
fonts/                                # 나눔고딕 (matplotlib 한글용)
test/                                 # 실행 스크립트 형태의 검증 (pytest 아님)
backup/                               # 미사용 이전 버전 보관
docs/intent.md                        # 프로젝트 구성과 개발 철학
requirements.txt
```

- [app.py](app.py) 는 라우팅만 담당하고, 실제 화면은 각 `section*.py` 의 `run_*()` 함수에 있습니다. **기능 수정은 대부분 해당 섹션 파일만 고치면 됩니다.**
- [expression_parser.py](expression_parser.py) 는 섹션3이 학생의 수식 입력(`x**2 + y**2 == 1`, `(z - 1j)**2`)을 `eval` 없이 계산하기 위해 쓰는 모듈입니다. 자세한 배경은 [docs/intent.md](docs/intent.md) 참고.
- `backup/app_backup.py` (섹션 분리 이전의 단일 파일 버전), `dash_symmetry_tool.py` (섹션2를 Dash로 시도한 실험) 는 `app.py` 에서 import되지 않습니다.

### 검증 실행

파서를 고쳤다면 아래를 실행해 보세요. pytest는 쓰지 않고 그냥 스크립트로 돌아갑니다.

```bash
python test/test_expression_parser.py
```

정상 입력·보안 차단·오류 안내를 모두 확인하고, 마지막 줄에 `ALL PASS` 가 찍히면 성공입니다 (종료 코드 0).

---

## 알려진 주의사항

- 한글 폰트는 저장소의 `fonts/` 에 들어 있는 `NanumGothic.ttf` 를 섹션1·3이 직접 읽어 씁니다. `requirements.txt` 의 `fonts` 는 이와 무관한 PyPI 더미 패키지이므로 지워도 됩니다. 폰트를 코드에서 지정하려면:

  ```python
  matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # macOS: 'AppleGothic'
  matplotlib.rcParams['axes.unicode_minus'] = False
  ```

- Streamlit의 `use_container_width` 인자는 deprecated 되었습니다. 향후 `width='stretch'` / `width='content'` 로 교체 필요.

---

## Docker에 대하여

이 저장소는 **GitHub Codespaces용 Streamlit 템플릿**에서 fork되어, 원래 `.devcontainer/devcontainer.json` 을 포함하고 있었습니다. 그 파일이 있으면 VS Code가 "Reopen in Container"를 제안하고 **그때만** 로컬 Docker Desktop이 필요해집니다.

앱 자체는 Docker에 의존하지 않으므로 로컬 개발에는 위의 venv 방식만으로 충분합니다. 이에 따라 Dev Container 설정(`.devcontainer/`)은 제거했으며, 어떤 IDE에서도 venv만으로 동일하게 개발할 수 있습니다.

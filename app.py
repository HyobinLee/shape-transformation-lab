import streamlit as st

# ✅ 페이지 설정
st.set_page_config(
    page_title="도형 변환 실험실",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ 페이지 제목
st.title("🔄 도형 변환 실험실")

# ✅ 사이드바 메뉴
#
# 심화 섹션은 각각 기초 섹션 하나를 이어받는다. 5→1, 6→2, 7→4, 8→3.
# 항목이 여덟이 되면 평평한 목록은 읽히지 않으므로 두 묶음으로 나눈다.
GROUPS = {
    "기초": [
        "1. 행렬을 통한 일차변환",
        "2. 행렬을 통한 대칭/회전변환",
        "3. 복소평면에서의 이동",
        "4. 복소평면에서의 회전/평행이동",
    ],
    "심화": [
        "5. 일차변환의 고유공간",
        "6. 거울을 몇 번 놓아야 하는가",
        "7. 되풀이하면 어디로 가는가",
        "8. 원은 원으로 간다 (뫼비우스)",
    ],
}

group = st.sidebar.radio("📂 묶음", list(GROUPS), horizontal=True, key="group")
menu = st.sidebar.radio("실험을 선택하세요", GROUPS[group], key=f"menu_{group}")

st.sidebar.caption(
    "**심화**는 기초를 하나씩 이어받습니다 — 5는 1을, 6은 2를, 7은 4를, 8은 3을."
)

# ✅ 선택에 따라 해당 시뮬레이터 실행
#
# import 를 분기 안에 두어 고른 섹션만 불러온다. 위에서 여덟을 다 import 하면
# 쓰지도 않을 sympy 까지 매번 로드해 첫 화면이 늦게 뜬다.
if menu == "1. 행렬을 통한 일차변환":
    from section1_transformation_by_matrix import run_transformation_by_matrix
    run_transformation_by_matrix()
elif menu == "2. 행렬을 통한 대칭/회전변환":
    from section2_symmetry_rotation import run_symmetry_rotation
    run_symmetry_rotation()
elif menu == "3. 복소평면에서의 이동":
    from section3_complex_plane import run_complex_plane
    run_complex_plane()
elif menu == "4. 복소평면에서의 회전/평행이동":
    from section4_rotation_translation import run_rotation_translation
    run_rotation_translation()
elif menu == "5. 일차변환의 고유공간":
    from section5_eigenspace import run_eigenspace
    run_eigenspace()
elif menu == "6. 거울을 몇 번 놓아야 하는가":
    from section6_isometry import run_isometry
    run_isometry()
elif menu == "7. 되풀이하면 어디로 가는가":
    from section7_orbit import run_orbit
    run_orbit()
elif menu == "8. 원은 원으로 간다 (뫼비우스)":
    from section8_mobius import run_mobius
    run_mobius()

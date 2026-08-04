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
menu = st.sidebar.radio("📂 실험을 선택하세요", [
    "1. 행렬을 통한 일차변환",
    "2. 행렬을 통한 대칭/회전변환",
    "3. 복소평면에서의 이동",
    "4. 복소평면에서의 회전/평행이동",
    "5. 일차변환의 고유공간",
    "7. 되풀이하면 어디로 가는가",
])

# ✅ 선택에 따라 해당 시뮬레이터 실행
#
# import 를 분기 안에 두어 고른 섹션만 불러온다. 위에서 넷을 다 import 하면
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
elif menu == "7. 되풀이하면 어디로 가는가":
    from section7_orbit import run_orbit
    run_orbit()

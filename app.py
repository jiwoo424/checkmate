import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

st.set_page_config(layout="wide")

# TOML에서 네비게이션 불러오기
nav = get_nav_from_toml("pages.toml")

# 페이지 탐색 없이 직접 페이지 추가
add_page_title(nav)  # 제목 추가

# 페이지 탐색을 직접 구현
page = st.sidebar.selectbox("Select a page", nav)

# 선택된 페이지에 따라 로직 실행
if page == "Home":
    st.write("🏠 Welcome to the Home page")
elif page == "About":
    st.write("ℹ️ About this app")
else:
    st.write(f"🚧 {page} page is under construction")
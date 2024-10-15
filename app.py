import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

st.set_page_config(layout="wide")

nav = get_nav_from_toml("pages.toml")

add_page_title(nav)

# 페이지 탐색을 직접 구현
page = st.sidebar.selectbox("Select a page", nav)

# 선택된 페이지에 따라 콘텐츠 출력
if page == "Home":
    st.write("🏠 Welcome to the Home page")
elif page == "About":
    st.write("ℹ️ About this app")
else:
    st.write(f"🚧 {page} page is under construction")
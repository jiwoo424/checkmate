import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

st.set_page_config(layout="wide")

# nav 생성 확인
try:
    nav = get_nav_from_toml("pages.toml")
    st.write("Navigation object created successfully")
except Exception as e:
    st.error(f"Error in creating navigation object: {e}")

# navigation 객체 생성 확인
try:
    pg = st.navigation(nav)
    st.write("Page navigation object created successfully")
except Exception as e:
    st.error(f"Error in creating page navigation object: {e}")

# 페이지 타이틀 추가 확인
try:
    add_page_title(pg)
except Exception as e:
    st.error(f"Error in adding page title: {e}")

# pg 실행 확인
try:
    pg.run()
except Exception as e:
    st.error(f"Error in running page navigation: {e}")

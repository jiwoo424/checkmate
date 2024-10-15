import streamlit as st
from st_pages import Page, show_pages, add_page_title
add_page_title()

show_pages(
    [
        Page("app.py", "Home", "🏠"),
        Page("page1.py", "Page 2", ":books:"),
        Page("page2.py", "Page 2", ":books:"),

    ]
)


st.title("전세/월세 사기계약 방지를 위한 부동산계약서 검토-분석 서비스 ")
st.write(""" 명품인재 x 업스테이지 LLM Innovators Challenge """,unsafe_allow_html=True)
st.write(""" <p> team <b style="color:red">체크메이트</b></p>""",unsafe_allow_html=True)
st.divider()

import streamlit as st
from st_pages import Page, show_pages, add_page_title
add_page_title()

show_pages(
    [
        Page("app.py", "서비스 소개", "🏠"),
        Page("page1.py", "계약서 업로드", "📑"),
        Page("page2.py", "법률 용어 질문", "📖"),

    ]
)


st.title("전세/월세 사기계약 방지를 위한 부동산계약서 검토-분석 서비스 ")
st.write(""" 명품인재 x 업스테이지 LLM Innovators Challenge """,unsafe_allow_html=True)
st.write(""" <p> team <b style="color:red">체크메이트</b></p>""",unsafe_allow_html=True)
st.divider()

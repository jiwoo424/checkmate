import streamlit as st
from st_pages import Page, show_pages, add_page_title
add_page_title()

# show_pages(
#     [
#         Page("app.py", "서비스 소개", "🏠"),
#         Page("page1.py", "계약서 업로드", "📑"),
#         Page("page2.py", "법률 용어 질문", "📖"),

#     ]
# )




# 현재 페이지를 세션 상태에 저장
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"

# 페이지 전환 함수
def switch_page(page):
    st.session_state["current_page"] = page

# 페이지 선택 메뉴 (왼쪽 사이드바)
with st.sidebar:
    st.button("서비스 소개", on_click=lambda: switch_page("home"))
    st.button("계약서 업로드", on_click=lambda: switch_page("upload"))
    st.button("법률 용어 질문", on_click=lambda: switch_page("question"))


# 각 페이지별로 분기
if st.session_state["current_page"] == "home":
    st.title("전세/월세 사기계약 방지를 위한 부동산계약서 검토-분석 서비스")
    st.write("명품인재 x 업스테이지 LLM Innovators Challenge")
    st.write("<p>team <b style='color:red'>체크메이트</b></p>", unsafe_allow_html=True)



elif st.session_state["current_page"] == "upload":
    st.title("계약서 업로드")
    file = st.file_uploader("계약서를 업로드하세요", type=["jpg", "jpeg", "png"])

    if file is not None:
        img = Image.open(file)
        st.image(img)

        if "uploaded_file_path" not in st.session_state:
            st.session_state["uploaded_file_path"] = {}

        file_path = f"tmp/{file.name}"
        img.save(file_path)
        
        st.session_state["uploaded_file_path"]["path"] = file_path
        st.success("계약서가 업로드되었습니다!")

elif st.session_state["current_page"] == "question":
    st.title("법률 용어 질문")

    if "uploaded_file_path" in st.session_state and "path" in st.session_state["uploaded_file_path"]:
        file_path = st.session_state["uploaded_file_path"]["path"]
        st.write("업로드된 계약서 미리보기:")

        img = Image.open(file_path)
        st.image(img)
        
        query = st.text_input("질문을 입력하세요")
        if query:
            st.write(f"'{query}'에 대한 추가 정보를 확인합니다...")
    else:
        st.warning("계약서를 먼저 업로드해주세요. (업로드 페이지로 이동)")


# st.title("전세/월세 사기계약 방지를 위한 부동산계약서 검토-분석 서비스 ")
# st.write(""" 명품인재 x 업스테이지 LLM Innovators Challenge """,unsafe_allow_html=True)
# st.write(""" <p> team <b style="color:red">체크메이트</b></p>""",unsafe_allow_html=True)
# st.divider()

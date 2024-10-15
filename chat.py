import streamlit as st
import re
from langchain_upstage import ChatUpstage

from CLAUSE import explain_legal_term

st.title("전세/월세 사기계약 방지를 위한 부동산계약서 검토-분석 서비스 ")
st.write(""" 명품인재 x 업스테이지 LLM Innovators Challenge """,unsafe_allow_html=True)
st.write(""" <p> team <b style="color:red">체크메이트</b></p>""",unsafe_allow_html=True)
st.divider()

my_expander = st.expander(" 테스트 ")
with my_expander:  

    # 세션 상태에 메시지 저장
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "더 알고 싶은 법률 용어는?"}]

# 저장된 메시지 출력
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

                # 사용자 입력 처리
    if prompt := st.chat_input("메시지를 입력하세요", key="chat_input"):

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # 법률 용어 설명 함수 호출 (예: explain_legal_term 함수)
        msg = explain_legal_term(prompt)
        
        # AI 응답 메시지 추가
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
import streamlit as st
import re
from langchain_upstage import ChatUpstage

from CLAUSE import explain_legal_term



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
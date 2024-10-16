import streamlit as st
from st_pages import Page, show_pages, add_page_title
from PIL import Image

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from PIL import Image
import requests
from flask import Flask, request, jsonify
import re
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma


import streamlit as st
import re
from langchain_upstage import ChatUpstage

import langchain
langchain.verbose = False
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


from OCR import extract_clauses_as_dict
from CLAUSE import extract_legal_terms, legal_explanations, generate_clause_explanation, terms_df, explain_legal_term
from DETECTION import initialize_embeddings, load_vector_store, detection



persist_directory = "./chroma_data"
persist_directory_db = "./chroma_db"


api_key = st.secrets['API_KEY']
embeddings = initialize_embeddings(api_key)
vector_store = load_vector_store(persist_directory, embeddings)
db = load_vector_store(persist_directory_db, embeddings)
retriever = db.as_retriever()

def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())



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
        current_time = datetime.now().isoformat().replace(':', '_')
        file.name = current_time + '.jpg'

        save_uploaded_file('tmp', file)

        img = Image.open(file)
        st.image(img)
        
        if "uploaded_file_path" not in st.session_state:
            st.session_state["uploaded_file_path"] = {}
            
        img.save(file_path)
        
        file_path = os.path.join('tmp', file.name)
    

        # OCR API 호출
        def extract_text_from_document(api_key, filename):
            url = "https://api.upstage.ai/v1/document-ai/ocr"
            headers = {"Authorization": f"Bearer {api_key}"}
            files = {"document": open(filename, "rb")}
            response = requests.post(url, headers=headers, files=files)
            return response.json()

        
        api_key = st.secrets['API_KEY']
        ocr_result = extract_text_from_document(api_key, file_path)

        def extract_ocr_text(ocr_result):
            ocr_text = " ".join(page['text'] for page in ocr_result['pages'])
            return ocr_text

        # OCR 결과에서 텍스트 추출
        ocr_text = extract_ocr_text(ocr_result)
        
        
        # 최종적으로 조항을 분리하고 결과를 딕셔너리로 저장
        final_classified_text = extract_clauses_as_dict(ocr_text)
        
        # final_classified_text에서 'type'이 '조항'인 항목들의 'content'를 추출하여 risky_clause 리스트에 저장
        clauses = []

        for key, clause in final_classified_text.items():
            clauses.append(clause)  # 조항 내용을 리스트에 추가

        first_line = ocr_text.split('\n')[0]
        title = re.match(r'[가-힣]+', first_line).group()
        total_clauses = len(clauses)
        num_risky = 0

        detection_results = []
        
        for clause in clauses:
                results = detection(clause, vector_store, embeddings)
                detection_results.append(results)
                if results[3] == 1:
                    num_risky += 1
        st.write(f"해당 계약서는 {title}입니다.")
        st.write(f"총 {total_clauses}개의 조항 중 {num_risky}개의 위험 조항이 감지되었습니다.")

        for i, clause in enumerate(clauses):
            sim_clause, judgment, reason, detection_result = detection_results[i]
            # 조항 출력 스타일 결정 (위험 조항인 경우 빨간색 테두리)
            if detection_result == 1:
                st.markdown(
                    f"<div style='padding: 10px; border: 2px solid red; border-radius: 5px; background-color: #ffe6e6;'>{clause}</div>", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f0f0f0;'>{clause}</div>", 
                    unsafe_allow_html=True
                )

            # 조항에서 법률 용어 추출 및 설명 가져오기
            legal_terms = extract_legal_terms(clause, terms_df)
            term_explanations = legal_explanations(legal_terms, terms_df)

            # 위험 조항인 경우 추가 정보 출력
            if detection_result == 1:
                explanation = generate_clause_explanation(clause, term_explanations, True, sim_clause, judgment)
                st.write("")
                st.write("**조항 해설**")
                st.write(explanation)
                st.write("**⚠️ 유사한 위험 조항 발견**")
                st.write(f"유사 조항: {sim_clause}")
                st.write(f"전문가 견해: {judgment}")
                reason = reason.split('<sep>')
                for r in reason:
                    context_docs = retriever.invoke(r)
                    r = context_docs[0].metadata['source'] + " " + r
                    st.write("**법적 근거**")
                    st.write(r)
                my_expander = st.expander("단어 사전")
                with my_expander:
                    if term_explanations:
                        for term, explanation in term_explanations.items():
                            st.write(f"**{term}**: {explanation}")

                                                    
            else:
                explanation = generate_clause_explanation(clause, term_explanations)
                st.write("")
                st.write("**조항 해설**")
                st.write(explanation)
                my_expander = st.expander("단어 사전")
                with my_expander:
                    if term_explanations:
                        for term, explanation in term_explanations.items():
                            st.write(f"**{term}**: {explanation}")


            st.divider()
            

elif st.session_state["current_page"] == "question":
    st.title("법률 용어 질문")

    if "uploaded_file_path" in st.session_state and "path" in st.session_state["uploaded_file_path"]:
        file_path = st.session_state["uploaded_file_path"]["path"]
        st.write("업로드된 계약서 미리보기:")

        img = Image.open(file_path)
        st.image(img)
        
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "단어를 입력해주세요."}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("메시지를 입력하세요", key="chat_input"):

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
                
            msg = explain_legal_term(prompt)
                
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
            
            
    else:
        st.warning("계약서를 먼저 업로드해주세요. (업로드 페이지로 이동)")


# st.title("전세/월세 사기계약 방지를 위한 부동산계약서 검토-분석 서비스 ")
# st.write(""" 명품인재 x 업스테이지 LLM Innovators Challenge """,unsafe_allow_html=True)
# st.write(""" <p> team <b style="color:red">체크메이트</b></p>""",unsafe_allow_html=True)
# st.divider()

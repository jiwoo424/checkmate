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
from langchain.vectorstores import Chroma

import langchain
langchain.verbose = False
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


from OCR import extract_clauses_with_order, clean_text, classify_remaining_text, process_ocr_text
from CLAUSE import extract_legal_terms, legal_explanations, generate_explanation, terms_df
from DETECTION import initialize_embeddings, load_vector_store, detection

# ChromaDB 경로 설정 (이미 생성된 데이터베이스 경로를 지정)
persist_directory = "./chroma_data"
persist_directory_db = "./chroma_db"


# 임베딩 모델 및 벡터 스토어 로드
api_key = st.secrets['API_KEY']
embeddings = initialize_embeddings(api_key)
vector_store = load_vector_store(persist_directory, embeddings)
db = load_vector_store(persist_directory_db, embeddings)
retriever = db.as_retriever()
	
st.title("전세/월세 사기계약 방지를 위한 부동산계약서 검토-분석 서비스 ")
st.write(""" 명품인재 x 업스테이지 LLM Innovators Challenge """,unsafe_allow_html=True)
st.write(""" <p> team <b style="color:red">체크메이트</b></p>""",unsafe_allow_html=True)
st.write("________________________________")
st.subheader('검토-분석이 필요한 계약서는?')
file = st.file_uploader('계약서를 업로드 하세요', type=['jpg', 'jpeg', 'png'])

def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())

if file is not None:
    current_time = datetime.now().isoformat().replace(':', '_')
    file.name = current_time + '.jpg'

    save_uploaded_file('tmp', file)

    img = Image.open(file)
    st.image(img)
    
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

    # OCR 텍스트 추출
    def extract_ocr_text(ocr_result):
        ocr_text = " ".join(page['text'] for page in ocr_result['pages'])
        return ocr_text

    # OCR 결과에서 텍스트 추출
    ocr_text = extract_ocr_text(ocr_result)
    
    # 최종적으로 조항을 분리하고 결과를 딕셔너리로 저장
    final_classified_text = process_ocr_text(ocr_text)
    
    # final_classified_text에서 'type'이 '조항'인 항목들의 'content'를 추출하여 risky_clause 리스트에 저장
    clauses = []

    for key, item in final_classified_text.items():
        if item['type'] == '조항':
            clauses.append(item['content'])


    for i, clause in enumerate(clauses):
        # 위험 조항 감지
        sim_clause, judgment, reason, detection_result = detection(clause, vector_store, embeddings)

        # 조항 출력 스타일 결정 (위험 조항인 경우 빨간색 테두리)
        if detection_result == 1:
            st.markdown(
                f"<div style='padding: 10px; border: 2px solid red; border-radius: 5px; background-color: #ffe6e6;'>{clause}</div>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>{clause}</div>", 
                unsafe_allow_html=True
        )
        # 조항에서 법률 용어 추출 및 설명 가져오기
        legal_terms = extract_legal_terms(clause, terms_df)
        term_explanations = legal_explanations(legal_terms, terms_df)
        
        # 용어 설명 출력
        if term_explanations:
            st.write("### 용어 설명")
            for term, explanation in term_explanations.items():
                st.write(f"**{term}**: {explanation}")
        # LangChain을 사용하여 조항 설명 생성
        explanation,reason = generate_explanation(clause)
        st.write("### 조항에 대한 설명")
        st.write(explanation)




        # 구분선 추가
        st.write("________________________________")

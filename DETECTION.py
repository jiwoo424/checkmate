import streamlit as st
import os
from langchain.vectorstores import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain_upstage import UpstageEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

# API 키 가져오기
api_key = st.secrets['API_KEY']

# 임베딩 모델 초기화
embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage", api_key=api_key)

# Chroma 데이터베이스 경로 설정
persist_directory_db = "/chroma_db"
vector_store = Chroma(
    persist_directory=persist_directory_db,
    embedding_function=embeddings
)

# 위험 조항 감지 함수
def detection(clause, threshold=0.8):
    # 주어진 조항과 가장 유사한 위험 조항 문서를 검색
    results = vector_store.similarity_search(clause, k=1)

    # 유사한 조항이 없는 경우 예외 처리
    if not results:
        return None, None, None, 0

    sim_clause = results[0].page_content  # 유사 조항
    query_vector = embeddings.embed_query(clause)  # 주어진 조항의 벡터
    stored_vector = embeddings.embed_query(sim_clause)  # 유사 조항의 벡터

    # 두 조항 간의 코사인 유사도 계산
    cosine_sim = cosine_similarity([query_vector], [stored_vector])[0][0]

    # 유사한 조항일 경우에만 위험 조항으로 감지하고 정보를 반환
    if cosine_sim > threshold:
        judgment = results[0].metadata.get('illdcssBasiss')
        reason = results[0].metadata.get('relateLaword')
        return sim_clause, judgment, reason, 1
    else:
        return None, None, None, 0
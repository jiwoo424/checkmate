import streamlit as st
import os
from langchain.embeddings import UpstageEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain_upstage import UpstageEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3


api_key = st.secrets['API_KEY']

persist_directory_db = "/chroma_db"
db = Chroma(embedding_function=embeddings, persist_directory=persist_directory_db)
retriever = db.as_retriever()

persist_directory_data = "/content/drive/MyDrive/Colab Notebooks/LLM/chroma_data" # 저장경로
vector_store = Chroma(
    persist_directory=persist_directory_data,
    embedding_function=embeddings
    )

def detection(clause,threshold = 0.8):
  # 주어진 조항과 가장 유사한 위험 조항 문서를 불러옴
  results = vector_store.similarity_search(clause,k=1)[0]
  sim_clause = results.page_content # 유사 조항
  query_vector = embeddings.embed_query(clause) # 주어진 조항의 벡터
  stored_vector = embeddings.embed_query(sim_clause) # 유사 조항의 벡터

  # 두 조항 간의 코사인 유사도
  cosine_sim = cosine_similarity([query_vector], [stored_vector])

  # 유사한 조항일 경우에만 위험 조항으로 감지하고 정보를 출력함
  if cosine_sim > threshold:
      judgment = results.metadata['illdcssBasiss']
      reason = results.metadata['relateLaword']
      return sim_clause, judgment, reason, 1
  else: return None, None, None, 0
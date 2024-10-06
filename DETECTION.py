import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain_upstage import UpstageEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

# 벡터 스토어 및 임베딩 설정 함수

api_key = st.secrets['API_KEY']

def setup_vector_store(persist_directory, api_key):
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage", api_key=api_key)
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return embeddings, vector_store


# 위험 조항 탐지 함수
def detection(clause, embeddings, vector_store):
    results = vector_store.similarity_search(clause, k=1)[0]
    sim_clause = results.page_content
    query_vector = embeddings.embed_query(clause)
    stored_vector = embeddings.embed_query(sim_clause)

    cosine_sim = cosine_similarity([query_vector], [stored_vector])[0][0]

    if cosine_sim > 0.8:
        judgment = results.metadata['illdcssBasiss']
        return sim_clause, judgment, 1
    else:
        return None, None, 0

import numpy as np
import streamlit as st
from langchain_upstage import UpstageEmbeddings

api_key = st.secrets['API_KEY']
agreements = pd.read_pickle("agreements.pkl")


def get_embedding(text):
    embeddings = UpstageEmbeddings(api_key=api_key, model="solar-embedding-1-large-passage")
    response = embeddings.embed_documents([text])  # Upstage는 리스트 형식 입력 요구
    return np.array(response[0]) 


def recommend_clause(clause, agreements, threshold=0.4):
    embedded_clause = get_embedding(clause)
    embeddings = agreements['feature'].values
    distances = [np.linalg.norm(embedded_clause - emb) for emb in embeddings]
    idx = np.argmin(distances)
    if distances[idx] < threshold:
        return idx, distances[idx]
    else:
        return None, None

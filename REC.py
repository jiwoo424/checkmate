import numpy as np
import streamlit as st
import pandas as pd
from langchain_upstage import UpstageEmbeddings

api_key = st.secrets['API_KEY']
agreements = pd.read_pickle("agreements.pkl")


def get_embedding(text):
    embeddings = UpstageEmbeddings(api_key=api_key, model="solar-embedding-1-large-passage")
    response = embeddings.embed_documents([text])
    return np.array(response[0]) 

def recommend_clause(clause, agreements= agreements.loc[:2], threshold=0.4):
    indices_ini = []
    embedded_clause = [get_embedding(text) for text in clause]
    embeddings = agreements['feature'].values
    distances = [np.linalg.norm(embedded_clause - emb) for emb in embeddings]
    idx = np.argmin(distances)
    if distances[idx] < threshold:
        indices_ini.append(idx)
        indices_ini = list(set(indices_ini))
    return indices_ini
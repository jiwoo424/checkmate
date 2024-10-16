import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain_upstage import UpstageEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

api_key = st.secrets['API_KEY']

def initialize_embeddings(api_key):
    return UpstageEmbeddings(model="solar-embedding-1-large-passage", api_key=api_key)

persist_directory_db = "./chroma_db"


def load_vector_store(persist_directory, embeddings):
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


def detection(clause, vector_store, embeddings, threshold=0.85):
    results = vector_store.similarity_search(clause, k=1)
    
    if not results:
        return None, None, None, 0

    sim_clause = results[0].page_content
    query_vector = embeddings.embed_query(clause)
    stored_vector = embeddings.embed_query(sim_clause)

    cosine_sim = cosine_similarity([query_vector], [stored_vector])[0][0]

    if cosine_sim > threshold:
        judgment = results[0].metadata.get('illdcssBasiss')
        reason = results[0].metadata.get('relateLaword')
        return sim_clause, judgment, reason, 1
    else:
        return None, None, None, 0
import numpy as np
from langchain_upstage import UpstageEmbeddings

def get_embedding(text):
    embeddings = UpstageEmbeddings(api_key=os.environ["API_KEY"], model="solar-embedding-1-large-passage")
    response = embeddings.embed_documents([text])  # Upstage는 리스트 형식 입력 요구
    return np.array(response[0]) 


def recommend(clause, df, threshold=0.4):
    embedded_clause = get_embedding(clause)
    embeddings = df['feature'].values
    distances = [np.linalg.norm(embedded_clause - emb) for emb in embeddings]
    idx = np.argmin(distances)
    if distances[idx] < threshold:
        return idx, distances[idx]
    else:
        return None, None

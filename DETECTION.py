import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from langchain.vectorstores import Chroma
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
from langchain_upstage import UpstageEmbeddings

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



api_key = st.secrets['API_KEY']


df = pd.read_csv("조항.csv")
data = df.copy()[df.clauseField == 5]
embeddings = UpstageEmbeddings(model="solar-embedding-1-large-query")

documents = []
for index, row in data.iterrows():
    sentences = row['clauseArticle'].split("<sep>")
    sentences = [sentence.strip() for sentence in sentences]
    for sentence in sentences:
      documents.append(Document(page_content=sentence, metadata={"illdcssBasiss": row['illdcssBasiss']}))



directory = "faiss_data"

dimension = embeddings.embedding_ctx_length
index = faiss.IndexFlatL2(dimension)
vector_store = FAISS(embedding_function=embeddings, index=index,
                     docstore= InMemoryDocstore(),
                     index_to_docstore_id={})
vector_store.add_documents(documents=documents)

vector_store.save_local(directory)



# vector_store = FAISS.load_local(directory, embeddings)

# input_sentence = data.clauseArticle.iloc[1].split('<sep>')[0]
input_sentence = "제7조(기타) 제3항 법률적인 문제가 발생시는 갑의 소재지 법원으로 한다."
input_vector = embeddings.embed_query(input_sentence)
sim_sentence = vector_store.similarity_search_with_score(input_sentence, k=1)[0][0].page_content
stored_vector = embeddings.embed_query(sim_sentence)

cosine_sim = cosine_similarity([input_vector], [stored_vector])
print(f"코사인 유사도: {cosine_sim[0][0]}")
l2 = np.sum((np.array(input_vector) - np.array(stored_vector))**2)
print(f"L2 거리: {l2}")
results = vector_store.similarity_search_with_score(input_sentence, k=2)  # 상위 1개의 유사한 문장 검색

print("질문:", input_sentence)
print("답변:", results[0][0].metadata['illdcssBasiss'])
print("출처:", results[0][0].page_content)
print(results[0][1])

print("답변:", results[1][0].metadata['illdcssBasiss'])
print("출처:", results[1][0].page_content)
print(results[1][1])

input_sentence == results[0][0].page_content
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from langchain.vectorstores import Chroma
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
from langchain_upstage import UpstageEmbeddings

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/LLM/조항.csv")
data = df.copy()[df.clauseField == 5]
embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")

# documents = []
# for index, row in data.iterrows():
#     sentences = row['clauseArticle'].split("<sep>")
#     sentences = [sentence.strip() for sentence in sentences]
#     for sentence in sentences:
#       documents.append(Document(page_content=sentence, metadata={"illdcssBasiss": row['illdcssBasiss']}))

persist_directory = "/content/drive/MyDrive/Colab Notebooks/LLM/chroma_data"

# vector_store = Chroma.from_documents(
#     documents=documents,
#     embedding=embeddings,
#     persist_directory=persist_directory
# )


# vector_store.persist()

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

def detection(clause):
  results = vector_store.similarity_search(clause,k=1)[0]
  sim_clause = results.page_content
  query_vector = embeddings.embed_query(clause)
  stored_vector = embeddings.embed_query(clause)

  cosine_sim = cosine_similarity([input_vector], [stored_vector])

  if cosine_sim > 0.8:
      judgment = results.metadata['illdcssBasiss']
      return sim_clause, judgment, 1
  else: return None, None, 0
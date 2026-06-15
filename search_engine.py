from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

db = Chroma(persist_directory = "./my_vector_db", embedding_function = embeddings_model)

query = "What do the rebellious artists do in the lower levels?"

print("~" * 60)
print(f"Searching database using the query: '{query}'")

results = db.similarity_search(query, k = 1)

if results:
    print(f"Result: {results[0].page_content}")
else:
    print("No results found")
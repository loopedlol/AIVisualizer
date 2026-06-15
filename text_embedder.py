import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

passage_path = "random_passage.txt"

if not os.path.exists(passage_path):
    print(f"Error: Could not find {passage_path}.")
    exit()

with open(passage_path, "r", encoding="utf-8") as file:
    long_text = file.read()


print("~" * 60)
print(f"File has been loaded. Current character count is {len(long_text)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 50)
chunks = text_splitter.split_text(long_text)

embeddings_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}: {chunk}")

print("Set 2")

db = Chroma.from_texts(
    texts=chunks, 
    embedding=embeddings_model, 
    persist_directory="./my_vector_db"
)
print("Saved vectors to file")

print("~" * 60)
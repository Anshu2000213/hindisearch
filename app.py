import os
from flask import Flask, render_template, request
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)

# Initialize ChromaDB client
client = chromadb.Client()
model = SentenceTransformer('bert-base-multilingual-cased')

def read_files_from_folder(folder_path):
    file_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                content = file.read()
                file_data.append({"file_name": file_name, "content": content})
    return file_data

# Load the model
# model = SentenceTransformer('bert-base-multilingual-cased')


# Read data files and create embeddings
folder_path = "data"
file_data = read_files_from_folder(folder_path)

documents = []
embeddings = []
metadatas = []
ids = []

for index, data in enumerate(file_data):
    documents.append(data['content'])
    embedding = model.encode(data['content']).tolist()
    embeddings.append(embedding)
    metadatas.append({'source': data['file_name']})
    ids.append(str(index + 1))

# Create collection in ChromaDB
pet_collection_emb = client.create_collection("Animal_collection_emb")

pet_collection_emb.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']

        # Perform search using embeddings
        input_em = model.encode(query).tolist()
        results = pet_collection_emb.query(
            query_embeddings=[input_em],
            n_results=2
        )

        return render_template('results.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)

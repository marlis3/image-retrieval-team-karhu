import faiss
import numpy as np

d = 512 # Dimension of the embeddings
index = faiss.IndexFlatL2(d) # Construct the index

def add_embedding(embedding):
    """Adds an embedding to the FAISS index"""
    index.add(np.array([embedding]))

def search(query_embedding, k=5):
    """Searches for the most similar embeddings in the FAISS index"""
    distances, indices = index.search(np.array([query_embedding]), k)
    return distances, indices
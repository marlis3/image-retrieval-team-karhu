import faiss
import numpy as np

d = 512 # Dimension of the embeddings
index = faiss.IndexFlatL2(d) # Construct the index
stored_images = [] # Store image filenames

def add_image(embedding, image_name):
    """Adds an image embedding to FAISS."""
    embedding = np.array(embedding, dtype="float32").reshape(1, -1)  # Ensure correct shape
    index.add(embedding)  # Store in FAISS
    
    stored_images.append(image_name)  # Store image filename
    print(f"Added image: {image_name}, Total images in FAISS: {index.ntotal}")

def search_similar_images(query_embedding, k=5):
    """Finds similar images in FAISS."""
    print(f"Searching FAISS (Total stored images: {index.ntotal})")
    
    if index.ntotal == 0:
        print("FAISS index is empty! No images available.")
        return [], []

    query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)  # Ensure correct shape
    distances, indices = index.search(query_embedding, k)

    # Retrieve image filenames of matched cases
    similar_images = [stored_images[i] for i in indices[0]]
    return similar_images, distances[0].tolist()
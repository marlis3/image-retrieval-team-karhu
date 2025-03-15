# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# Install necessary dependencies
# !pip install torch torchvision transformers datasets huggingface_hub kaggle faiss-gpu kagglehub

# Import libraries
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import random
import kagglehub
# -

# Download the CheXpert dataset using KaggleHub
dataset_download_path = kagglehub.dataset_download("ashery/chexpert")
print("Path to dataset files:", dataset_download_path)

import os

# +
# Load pre-trained CLIP model
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Define the dataset path
dataset_path = Path("/kaggle/input/chexpert/train")

image_paths = []
amount_of_images: int = 1200
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(".jpg"):
            image_paths.append(os.path.join(root, file))
            if len(image_paths) >= amount_of_images:  # Stop after finding 30 images
                break
    if len(image_paths) >= amount_of_images:
        break

print(f"Selected {len(image_paths)} images for testing.")


# +
# Function to generate embeddings using CLIP
def generate_embedding(image):
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt", padding=True)
        embedding = model.get_image_features(**inputs)
    return embedding.numpy()

# Generate embeddings for the subset of images
embeddings = []
for image_path in image_paths:
    image_path = Path(image_path)

    # Skip hidden files (e.g., ._filename)
    if image_path.name.startswith("."):
        print(f"Skipping hidden file: {image_path}")
        continue 
    try:
        # Load the image as a PIL image
        image = Image.open(image_path).convert("RGB")
        
        # Generate the embedding
        embedding = generate_embedding(image)
        embeddings.append(embedding)
    except UnidentifiedImageError:
        print(f"Skipping invalid image file: {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# +
# Convert embeddings to a numpy array
if embeddings:  # Check if embeddings list is not empty
    embeddings = np.vstack(embeddings)
    print("Embeddings shape:", embeddings.shape)
else:
    print("No valid images found.")

# Normalize embeddings (required for FAISS)
faiss.normalize_L2(embeddings)

# Build FAISS index
dimension = embeddings.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)  # L2 distance index
index.add(embeddings)
print("FAISS index built with", index.ntotal, "embeddings")


# +
# Test basic image search
def search_similar_images(query_embedding, k=5):
    # Normalize the query embedding
    faiss.normalize_L2(query_embedding)
    # Search for similar images
    distances, indices = index.search(query_embedding, k)
    return distances, indices

# Random number for selecting a query image
rand_num = random.randint(0, amount_of_images - 1)

# Example: Search for similar images using the first image as a query
query_image_path = image_paths[rand_num]
query_image = Image.open(query_image_path).convert("RGB")
query_embedding = generate_embedding(query_image).reshape(1, -1)  # Reshape for FAISS

# Perform the search
distances, indices = search_similar_images(query_embedding, k=5)

# Display the results
print("Query image:", query_image_path)
print("Top 5 similar images:")
for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"{i + 1}: {image_paths[idx]} (Distance: {distance:.4f})")

from IPython.display import display

# Path to the image
image_path1 = image_paths[idx + 1]
image_path2 = image_paths[idx + 2]

# Open the image
image1 = Image.open(image_path1)
image2 = Image.open(image_path2)

# Display the image
display(image1)
display(image2)
# -



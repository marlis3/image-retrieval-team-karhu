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
# !pip install torch torchvision transformers datasets huggingface_hub kaggle faiss-gpu kagglehub requests

# Import libraries
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import random
import kagglehub
import requests
import os
# -

# Download the CheXpert dataset using KaggleHub
dataset_download_path = kagglehub.dataset_download("ashery/chexpert")
print("Path to dataset files:", dataset_download_path)

# +
# Load pre-trained CLIP model
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Define the dataset path
dataset_path = Path("/kaggle/input/train")

image_paths = []
amount_of_images = 2000
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(".jpg"):
            image_paths.append(os.path.join(root, file))
            # Stop after finding set amount of images
            if len(image_paths) >= amount_of_images:  
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
# !pip install biopython
from Bio import Entrez
from Bio import Medline

# Function to fetch scientific papers from PubMed
def fetch_pubmed_papers(query, num_results=5):
    """
    Fetch scientific papers from PubMed based on a query.
    
    Args:
        query (str): The search query (e.g., "chest X-ray pneumonia").
        num_results (int): Number of papers to fetch (default: 5).
    
    Returns:
        list: A list of dictionaries containing paper details.
    """
    # Set your email (required by PubMed API)
    Entrez.email = "your_email@example.com"
    
    # Search PubMed
    handle = Entrez.esearch(db="pubmed", term=query, retmax=num_results)
    record = Entrez.read(handle)
    handle.close()
    
    # Get PubMed IDs of the papers
    id_list = record["IdList"]
    
    # Fetch details for each paper
    papers = []
    if id_list:
        handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="medline", retmode="text")
        records = Medline.parse(handle)
        
        for record in records:
            paper = {
                "title": record.get("TI", "No title available"),
                "abstract": record.get("AB", "No abstract available"),
                "authors": record.get("AU", []),
                "year": record.get("DP", "").split(" ")[0],  # Extract year from date
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{record.get('PMID', '')}"
            }
            papers.append(paper)
        
        handle.close()
    
    return papers


# +
from pathlib import Path

def extract_keywords(image):
    """
    Use CLIP to generate a text description of the image.
    """
    # Define possible medical conditions

    conditions = [
        "Pneumonia", "Enlarged Cardiomediastinum", "Lung Opacity", "Lung Lesion", 
        "Pneumothorax", "Pleular Other", "Cardiomegaly", "Edema", "Consolidation", 
        "Atelectasis", "Pleural effusion", "Fracture"
    ]    
    
    # Preprocess the image and conditions
    inputs = processor(
        text=conditions,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.get_image_features(inputs["pixel_values"])
        text_features = model.get_text_features(inputs["input_ids"])
    
    # Compute similarity between the image and each condition
    similarities = (image_features @ text_features.T).softmax(dim=-1)
    
    # Get the most relevant condition
    most_relevant_index = similarities.argmax().item()
    most_relevant_condition = conditions[most_relevant_index]
    
    return most_relevant_condition

# Test basic image search
def search_similar_images(query_embedding, k=5):
    # Normalize the query embedding
    faiss.normalize_L2(query_embedding)
    # Search for similar images
    distances, indices = index.search(query_embedding, k)
    return distances, indices


# +
# Random number for selecting a query image
rand_num = random.randint(0, amount_of_images - 1)

# Example: Search for similar images using a random image as a query
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

# Display the query image and top similar images
display(query_image)
for idx in indices[0]:
    display(Image.open(image_paths[idx]))

# Extract keywords from the query image
keywords = extract_keywords(query_image)
print(f"Extracted keywords: {keywords}")

# Fetch scientific papers based on the keywords
papers = fetch_pubmed_papers(keywords, num_results=3)

# Display scientific papers
print("\nScientific papers related to the query:")
for i, paper in enumerate(papers):
    print(f"{i + 1}. Title: {paper['title']}")
    print(f"   Abstract: {paper['abstract']}")
    print(f"   Authors: {', '.join(paper['authors'])}")
    print(f"   Year: {paper['year']}")
    print(f"   URL: {paper['url']}")
    print("-" * 50)

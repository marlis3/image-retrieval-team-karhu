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
# !pip install torch torchvision transformers datasets huggingface_hub kaggle faiss-gpu kagglehub requests biopython sacremoses

# Import libraries
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer
import faiss
import random
import kagglehub
import os
import datetime
from Bio import Entrez
from Bio import Medline

# +
# Download the CheXpert dataset using KaggleHub
dataset_download_path = kagglehub.dataset_download("ashery/chexpert")

# Define the dataset path
dataset_path = Path("/kaggle/input/train")

# Load pre-trained CLIP model
clip_model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# Load image paths
image_paths = []
amount_of_images = 4000
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
def generate_embedding(image):
    """Generate CLIP embeddings for an image."""
    with torch.no_grad():
        inputs = clip_processor(images=image, return_tensors="pt", padding=True)
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

# Convert embeddings to a numpy array
if embeddings:  # Check if embeddings list is not empty
    embeddings = np.vstack(embeddings)
    print("Embeddings shape:", embeddings.shape)
else:
    print("No valid images found.")

# Normalize embeddings (required for FAISS) and Build FAISS index
faiss.normalize_L2(embeddings)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("FAISS index built with", index.ntotal, "embeddings")


# -

def extract_keywords(image, top_k=3, confidence_threshold=0.1):
    """
    Use CLIP to extract multiple relevant medical conditions from the image.
    
    Args:
        image: Input PIL image
        top_k: Number of top conditions to return
        confidence_threshold: Minimum similarity score to include a condition
        
    Returns:
        list: List of tuples (condition, confidence_score) sorted by relevance
    """
    # Define possible medical conditions
    conditions = [
        "Pneumonia", "Enlarged Cardiomediastinum", "Lung Opacity", "Lung Lesion",
        "Pneumothorax", "Pleural Other", "Cardiomegaly", "Edema", "Consolidation",
        "Atelectasis", "Pleural effusion", "Fracture", "Normal"  # Added normal case
    ]
    
    # Preprocess the image and conditions
    inputs = clip_processor(
        text=conditions,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.get_image_features(inputs["pixel_values"])
        text_features = model.get_text_features(inputs["input_ids"])
    
    # Compute similarity scores
    similarities = (image_features @ text_features.T).squeeze(0)
    probs = similarities.softmax(dim=0)
    
    # Get top k conditions
    top_probs, top_indices = probs.topk(top_k)
    
    # Filter by confidence threshold and prepare results
    results = []
    for prob, idx in zip(top_probs, top_indices):
        if prob > confidence_threshold:
            results.append({
                'condition': conditions[idx],
                'confidence': prob.item()
            })
    
    # Sort by confidence (descending)
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return results if results else [{'condition': 'Uncertain', 'confidence': 1.0}]


# +
def fetch_pubmed_papers(query, num_results=5, recent_years=None):
    """
    Fetch scientific papers from PubMed based on query/conditions, sorted by relevance.
    
    Args:
        query: Can be either:
               - A string (single query term)
               - A dictionary like {'condition': '...', 'confidence': ...}
               - A list of such dictionaries
        num_results (int): Number of papers to fetch (default: 5)
        recent_years (int): If set, only return papers from last X years
    
    Returns:
        list: A list of dictionaries containing paper details
    """
    # Extract condition names from different input formats
    if isinstance(query, dict) and 'condition' in query:
        # Single condition dictionary
        search_terms = [query['condition']]
    elif isinstance(query, list) and all(isinstance(x, dict) and 'condition' in x for x in query):
        # List of condition dictionaries - extract just the names
        search_terms = [x['condition'] for x in query]
    else:
        # Assume it's already a string or list of strings
        search_terms = [query] if isinstance(query, str) else query
    
    # Combine terms with OR for PubMed search
    pubmed_query = " OR ".join(f'"{term}"' for term in search_terms)
    
    # Set your email (required by PubMed API)
    Entrez.email = "your_email@example.com"  # Replace with your actual email
    
    # Build the complete search query
    search_term = f'({pubmed_query}[Title/Abstract]) AND ("english"[Language])'
    
    if recent_years:
        current_year = datetime.datetime.now().year
        search_term += f' AND ("{current_year - recent_years}"[Date - Publication] : "3000"[Date - Publication])'
    
    try:
        # Search PubMed - sort by relevance
        handle = Entrez.esearch(db="pubmed", 
                              term=search_term, 
                              retmax=num_results,
                              sort='relevance')
        record = Entrez.read(handle)
        handle.close()
        
        # Get PubMed IDs of the papers
        id_list = record["IdList"]
        
        if not id_list:
            print(f"No papers found for: {', '.join(search_terms)}")
            return []
            
        # Fetch details for each paper
        papers = []
        handle = Entrez.efetch(db="pubmed", 
                              id=",".join(id_list), 
                              rettype="medline", 
                              retmode="text")
        records = Medline.parse(handle)
        
        for record in records:
            # Get citation count
            citation_count = 0
            try:
                cite_handle = Entrez.elink(dbfrom="pubmed", 
                                         id=record.get("PMID", ""), 
                                         cmd="neighbor_score")
                cite_results = Entrez.read(cite_handle)
                if cite_results and cite_results[0]["LinkSetDb"]:
                    citation_count = int(cite_results[0]["LinkSetDb"][0]["Link"][0]["Score"])
                cite_handle.close()
            except Exception as e:
                print(f"Could not get citation count for PMID {record.get('PMID', '')}: {e}")
            
            paper = {
                "title": record.get("TI", "No title available"),
                "abstract": record.get("AB", "No abstract available"),
                "authors": record.get("AU", []),
                "year": record.get("DP", "").split(" ")[0] if record.get("DP") else "Unknown",
                "journal": record.get("TA", ""),
                "citation_count": citation_count,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{record.get('PMID', '')}",
                "pmid": record.get("PMID", "")
            }
            papers.append(paper)
            
        handle.close()
        
        # Sort papers by citation count (higher first)
        papers.sort(key=lambda x: x["citation_count"], reverse=True)
        
        return papers[:num_results]
        
    except Exception as e:
        print(f"Error fetching papers from PubMed: {e}")
        return []

def search_similar_images(query_embedding, k=5):
    """Search for similar images using FAISS index."""
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
print()
print("Top 5 similar images:")
print()
for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"Image {i + 1}:")
    print(f"Distance: {distance:.4f} Path to image: {image_paths[idx]}")
    print()

# Display the query image and top similar images
display(query_image)
for idx in indices[0]:
    display(Image.open(image_paths[idx]))

# Extract keywords from the query image
keywords = extract_keywords(query_image)
print("Conditions found from query image:\n")
for kw in keywords:
    print(f"{kw['condition']}: {kw['confidence']*100:.1f}% confidence")
print()
print("-" * 50)

# Fetch scientific papers based on the keywords
papers = fetch_pubmed_papers(keywords, num_results=3)

# Display scientific papers
print("\nScientific papers related to the query:")
print()
for i, paper in enumerate(papers):
    print(f"{i + 1}. Title: {paper['title']}\n")
    print(f"   Abstract: {paper['abstract']}\n")
    print(f"   Authors: {', '.join(paper['authors'])}\n")
    print(f"   Year: {paper['year']}\n")
    print(f"   URL: {paper['url']}\n")
    print("-" * 50)

# +
# Text search function for getting more information about a certain condition
# INCOMPLETE

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load BioGPT-Large (causal LM)
model_name = "microsoft/BioGPT-Large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# List of supported conditions with possible synonyms
conditions = {
    "Pneumonia": ["pneumonia", "lung infection"],
    "Enlarged Cardiomediastinum": ["enlarged cardiomediastinum", "widened mediastinum"],
    "Lung Opacity": ["lung opacity", "lung shadow"],
    "Lung Lesion": ["lung lesion", "lung nodule"],
    "Pneumothorax": ["pneumothorax", "collapsed lung"],
    "Pleural Other": ["pleural other", "pleural abnormality"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart"],
    "Edema": ["edema", "fluid retention"],
    "Consolidation": ["consolidation", "lung consolidation"],
    "Atelectasis": ["atelectasis", "lung collapse"],
    "Pleural effusion": ["pleural effusion", "fluid in lungs"],
    "Fracture": ["fracture", "bone break"]
}

def generate_medical_info(condition, query_type="description"):
    """Generate clean medical information without repetitions"""
    # System message that won't be shown to user
    system_msg = "You are a medical expert providing accurate information."
    
    if query_type == "description":
        instruction = f"""Describe {condition} in this exact format:
1. Definition: [2-3 sentence definition]
2. Causes: [bullet points of main causes]
3. Symptoms: [key symptoms]
4. Diagnosis: [primary diagnostic methods]
5. Treatment: [first-line treatments]"""
    else:
        instruction = f"""Detail treatments for {condition} in this format:
1. Medications: [drug classes with examples]
2. Procedures: [clinical procedures]
3. Lifestyle: [supportive measures]"""
    
    # Combine with clear stop sequence
    prompt = f"{system_msg}\n{instruction}\n\nResponse:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.4,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.5,  # Strongly prevent repetition
            no_repeat_ngram_size=3,  # Prevent 3-word repeats
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Clean and format the output
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text.split("Response:")[-1].strip()
    
    # Additional cleaning to handle any artifacts
    stop_phrases = ["\n\n", "###", "<|endoftext|>"]
    for phrase in stop_phrases:
        response = response.split(phrase)[0]
    
    return response
    

def process_query(user_query):
    """Process user input and return medical info and relevant papers."""
    # Check for the mentioned condition
    detected_condition = None
    for condition, synonyms in conditions.items():
        if any(synonym.lower() in user_query.lower() for synonym in synonyms):
            detected_condition = condition
            break
    
    if not detected_condition:
        return (f"Sorry, I can only provide information on these conditions: {', '.join(conditions.keys())}. Please provide more specific information.", [])
    
    print(f"Fetching information about {detected_condition}...\n")
    medical_info = generate_medical_info(detected_condition)
    print(medical_info)
    
    print("\nFetching related scientific papers...\n")
    papers = fetch_pubmed_papers(detected_condition, num_results=3)
    
    return medical_info, papers

def main():
    user_query = input("Enter your query: ")
    medical_info, papers = process_query(user_query)
    
    print("\nScientific papers related to the query:")
    for i, paper in enumerate(papers):
        print(f"{i + 1}. Title: {paper['title']}")
        print(f"   Abstract: {paper['abstract']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   Year: {paper['year']}")
        print(f"   URL: {paper['url']}")
        print("-" * 50)

if __name__ == "__main__":
    main()

"""
Configuration settings for the Medical X-ray Retrieval System
"""
import os
import logging

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATABASE_DIR = os.path.join(DATA_DIR, "database")

# Create necessary directories
for directory in [DATA_DIR, DATABASE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model configuration
# Update this path to match your actual model location
MODEL_PATH = ""  # Will update with actual model
USE_HUGGINGFACE = True

# Database configuration
FAISS_INDEX_PATH = os.path.join(DATABASE_DIR, "index.faiss")
METADATA_PATH = os.path.join(DATABASE_DIR, "metadata.pkl")
IMAGE_DB_PATH = os.path.join(DATABASE_DIR, "image_db.pkl")

# Data configuration
# Update this path to match your actual testing data location
CHEXPERT_DIR = ""  # Update this path with your data

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# App configuration
APP_TITLE = "Medical X-ray Retrieval System"
APP_ICON = "🏥"
APP_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Available findings for checkboxes
CLINICAL_FINDINGS = [
    "No Finding",
    "Pneumonia",
    "Cardiomegaly",
    "Pleural Effusion",
    "Atelectasis",
    "Pneumothorax",
    "Edema",
    "Consolidation",
    "Lung Opacity",
    "Lung Lesion",
    "Fracture",
    "Support Devices"
]

# Example text queries
EXAMPLE_QUERIES = [
    # Basic findings
    "chest x-ray with pneumonia",
    "normal chest x-ray with no findings",
    "chest x-ray with pleural effusion",
    "chest x-ray with cardiomegaly",
    "chest x-ray with lung opacity",
    
    # Multiple findings
    "chest x-ray with pneumonia and pleural effusion",
    "chest x-ray with cardiomegaly and pulmonary edema",
    "chest x-ray with atelectasis and lung opacity",
]
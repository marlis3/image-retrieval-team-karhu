import os
import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
from io import BytesIO
import base64
import time
import logging
import hashlib

logger = logging.getLogger("chexpert-loader")

# Hardcoded path to your CheXpert dataset
CHEXPERT_DIR = "/Users/jarinatkpoyta/project-karhu/archive"

def create_subset_from_local_data(model, index, metadata_db, image_db, save_database,
                                 sample_size=50, random_seed=42, use_train=False):
    """
    Create a subset from a local CheXpert dataset and add it to the database.
    
    Args:
        model: The BiomedCLIP model adapter
        index: FAISS index
        metadata_db: Metadata database
        image_db: Image database
        save_database: Function to save the database
        sample_size: Number of images to sample
        random_seed: Random seed for reproducibility
        use_train: Whether to use train.csv (True) or valid.csv (False)
    
    Returns:
        Updated index, metadata_db, and image_db
    """
    try:
        # Determine which CSV file to use
        csv_file = os.path.join(CHEXPERT_DIR, "train.csv" if use_train else "valid.csv")
        
        # Check if the CSV file exists
        if not os.path.exists(csv_file):
            st.error(f"CSV file not found: {csv_file}")
            return index, metadata_db, image_db
        
        # Load CSV file
        st.info(f"Loading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Display basic info about the dataset
        st.info(f"CSV file contains {len(df)} total entries")
        
        # Replace CheXpert path prefix if present
        df["Path"] = df["Path"].str.replace("CheXpert-v1.0-small/", "", regex=False)
        
        # Filter for frontal view for consistent images
        if "Frontal/Lateral" in df.columns:
            df = df[df["Frontal/Lateral"] == "Frontal"]
            st.info(f"Filtered to {len(df)} frontal view images")
        
        # Replace -1 (uncertain) values with NaN
        df = df.replace(-1, float('nan'))
        
        # Take a random sample
        if sample_size < len(df):
            df = df.sample(sample_size, random_state=random_seed)
            st.info(f"Sampled {sample_size} images")
        
        # Define the findings columns
        findings_cols = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        
        # Check which columns exist in the dataset
        valid_findings_cols = [col for col in findings_cols if col in df.columns]
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each image
        added_count = 0
        skipped_count = 0
        duplicate_count = 0
        
        # Track already processed images using their file paths
        processed_paths = set()
        
        # Also create a set of image hashes already in the database
        existing_image_hashes = set()
        for idx in metadata_db:
            if 'file_path' in metadata_db[idx]:
                processed_paths.add(metadata_db[idx]['file_path'])
            if 'image_hash' in metadata_db[idx]:
                existing_image_hashes.add(metadata_db[idx]['image_hash'])
        
        for i, (idx, row) in enumerate(df.iterrows()):
            try:
                # Update progress
                progress = (i + 1) / len(df)
                progress_bar.progress(progress)
                status_text.text(f"Processing image {i+1}/{len(df)}")
                
                # Construct image path (try several possible paths)
                # Path 1: Direct from CSV
                img_path = os.path.join(CHEXPERT_DIR, row['Path'])
                
                # Path 2: With CheXpert-v1.0-small prefix
                if not os.path.exists(img_path):
                    img_path = os.path.join(CHEXPERT_DIR, "CheXpert-v1.0-small", row['Path'])
                
                # Path 3: Just the filename in the root directory
                if not os.path.exists(img_path):
                    img_path = os.path.join(CHEXPERT_DIR, os.path.basename(row['Path']))
                
                # If none of the paths exist, skip this image
                if not os.path.exists(img_path):
                    status_text.text(f"Skipping missing image: {os.path.basename(row['Path'])}")
                    skipped_count += 1
                    continue
                
                # Check if this file path has already been processed
                if img_path in processed_paths:
                    status_text.text(f"Skipping duplicate image (by path): {os.path.basename(row['Path'])}")
                    duplicate_count += 1
                    continue
                
                # Add to processed paths
                processed_paths.add(img_path)
                
                # Load image
                image = Image.open(img_path).convert("RGB")
                
                # Calculate image hash (using a simple perceptual hash method)
                img_hash = calculate_image_hash(image)
                
                # Check if this image hash already exists in the database
                if img_hash in existing_image_hashes:
                    status_text.text(f"Skipping duplicate image (by content): {os.path.basename(row['Path'])}")
                    duplicate_count += 1
                    continue
                
                # Add to existing image hashes
                existing_image_hashes.add(img_hash)
                
                # Create metadata
                metadata = {}
                
                # Store only essential metadata, excluding the Path field
                for col, val in row.items():
                    # Skip path information
                    if col != "Path":
                        metadata[col] = val
                
                # Add file_path and image_hash for internal use
                metadata['file_path'] = img_path  # Store file path for duplicate checking
                metadata['image_hash'] = img_hash  # Store image hash for duplicate checking
                
                # Process findings in a user-friendly way
                specific_findings = []
                findings_dict = {}
                
                for col in valid_findings_cols:
                    if col in row:
                        # Store findings in a user-friendly way (present/absent/uncertain)
                        if row[col] == 1.0:
                            findings_dict[col] = "Present"
                            specific_findings.append(col)
                        elif row[col] == 0.0:
                            findings_dict[col] = "Absent"
                        else:  # NaN values were previously -1 (uncertain)
                            findings_dict[col] = "Uncertain"
                
                # Add these to metadata
                metadata["findings_status"] = findings_dict
                
                # Create text description
                findings_text = "this is a photo of "
                
                if not specific_findings or (len(specific_findings) == 1 and specific_findings[0] == 'No Finding'):
                    findings_text += "normal chest x-ray with no findings"
                else:
                    findings_text += "chest x-ray with " + ", ".join([f.lower() for f in specific_findings if f != 'No Finding'])
                
                metadata["findings_text"] = findings_text
                
                # Add to database
                embedding = model.encode_image(image)[0]
                embedding = embedding / np.linalg.norm(embedding)
                index.add(np.array([embedding], dtype=np.float32))
                
                # Current index is ntotal - 1
                current_idx = index.ntotal - 1
                
                # Create thumbnail and convert to base64
                image_copy = image.copy()
                image_copy.thumbnail((224, 224))
                buffered = BytesIO()
                image_copy.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # Store metadata and image
                metadata_db[current_idx] = metadata
                image_db[current_idx] = img_base64
                
                added_count += 1
                
            except Exception as e:
                status_text.text(f"Error processing {os.path.basename(row.get('Path', 'image'))}: {str(e)}")
                skipped_count += 1
        
        # Save database
        save_database(index, metadata_db, image_db)
        
        # Show summary
        st.success(f"Added {added_count} images to database. Skipped {skipped_count} images. Detected {duplicate_count} duplicates.")
        
        return index, metadata_db, image_db
        
    except Exception as e:
        st.error(f"Error creating subset: {str(e)}")
        return index, metadata_db, image_db

def calculate_image_hash(image, hash_size=8):
    """
    Calculate a simple perceptual hash for an image.
    This helps identify duplicate images even if they have different filenames.
    
    Args:
        image: PIL Image object
        hash_size: Size of the hash (smaller = faster but less accurate)
        
    Returns:
        String representation of the perceptual hash
    """
    # Resize the image to a small square
    image = image.resize((hash_size, hash_size), Image.LANCZOS)
    
    # Convert to grayscale
    image = image.convert("L")
    
    # Get pixel data
    pixels = list(image.getdata())
    
    # Calculate average pixel value
    avg_pixel = sum(pixels) / len(pixels)
    
    # Create binary hash (1 if pixel value is greater than average, 0 otherwise)
    binary_hash = ''.join('1' if pixel > avg_pixel else '0' for pixel in pixels)
    
    # Convert binary hash to hexadecimal for compact storage
    hex_hash = hex(int(binary_hash, 2))[2:].zfill(hash_size**2 // 4)
    
    return hex_hash

# Function to add to your app.py
def add_chexpert_loader_to_sidebar(model, index, metadata_db, image_db, save_database):
    """Add the CheXpert loader UI to the sidebar"""
    st.subheader("Load CheXpert Data")
    
    # Tell the user where we're looking for the dataset
    st.info(f"Using CheXpert dataset at: {CHEXPERT_DIR}")
    
    # Choose between train and validation set
    dataset_option = st.radio("Dataset to use", ["Validation Set (smaller)", "Training Set (larger)"])
    use_train = dataset_option == "Training Set (larger)"
    
    # Number of images to load
    sample_size = st.slider("Number of images to load", min_value=10, max_value=200, value=50)
    
    # Display database stats
    if st.button("Show Database Statistics"):
        # Count unique image paths and hashes
        unique_paths = set()
        unique_hashes = set()
        findings_count = {}
        
        for idx in metadata_db:
            # Track unique files
            if 'file_path' in metadata_db[idx]:
                unique_paths.add(metadata_db[idx]['file_path'])
            if 'image_hash' in metadata_db[idx]:
                unique_hashes.add(metadata_db[idx]['image_hash'])
            
            # Count findings
            if 'findings_status' in metadata_db[idx]:
                for finding, status in metadata_db[idx]['findings_status'].items():
                    if status == "Present":
                        findings_count[finding] = findings_count.get(finding, 0) + 1
        
        # Display basic stats
        st.info(f"""
        Database statistics:
        - Total images: {index.ntotal}
        - Unique file paths: {len(unique_paths)}
        - Unique image hashes: {len(unique_hashes)}
        - Potential duplicates: {index.ntotal - len(unique_hashes)}
        """)
        
        # Display findings distribution
        if findings_count:
            st.subheader("Findings Distribution")
            findings_df = pd.DataFrame({
                'Finding': list(findings_count.keys()),
                'Count': list(findings_count.values())
            }).sort_values(by='Count', ascending=False)
            
            st.dataframe(findings_df)
    
    # Button to start loading
    if st.button("Load CheXpert Subset"):
        with st.spinner(f"Loading {sample_size} images from local CheXpert dataset..."):
            index, metadata_db, image_db = create_subset_from_local_data(
                model, index, metadata_db, image_db, save_database,
                sample_size=sample_size, use_train=use_train
            )
        st.success(f"CheXpert images loaded. Total images in database: {index.ntotal}")
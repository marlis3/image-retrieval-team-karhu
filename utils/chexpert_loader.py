"""
Utilities for loading CheXpert dataset with balanced sampling of findings
"""
import os
import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
from io import BytesIO
import base64
import time
import logging

logger = logging.getLogger("medical-retrieval.chexpert")

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

def create_subset_from_local_data(model, db, sample_size=50, random_seed=42, use_train=True, balance_findings=True):
    """
    Create a subset from a local CheXpert dataset and add it to the database.
    
    Args:
        model: The BiomedCLIP model adapter
        db: Vector database instance
        sample_size: Number of images to sample
        random_seed: Random seed for reproducibility
        use_train: Whether to use train.csv (True) or valid.csv (False)
        balance_findings: Whether to balance the dataset by findings
    
    Returns:
        None: The database is updated in-place
    """
    try:
        # Import from where CheXpert is installed
        from config.config import CHEXPERT_DIR
        chexpert_dir = CHEXPERT_DIR
        
        # Determine which CSV file to use
        csv_file = os.path.join(chexpert_dir, "train.csv" if use_train else "valid.csv")
        
        # Check if the CSV file exists
        if not os.path.exists(csv_file):
            st.error(f"CSV file not found: {csv_file}")
            return
        
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
        
        # Define the findings columns
        findings_cols = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        
        # Check which columns exist in the dataset
        valid_findings_cols = [col for col in findings_cols if col in df.columns]
        
        # Balanced sampling
        if balance_findings and sample_size < len(df):
            st.info("Performing balanced sampling by findings...")
            
            # First, create a list to store our balanced sample
            balanced_sample = []
            findings_count = {}
            
            # Initialize counts for each finding
            for col in valid_findings_cols:
                findings_count[col] = 0
            
            # Calculate target count per finding
            # We'll aim for an equal distribution across findings
            # But ensure we don't exceed our sample size
            target_per_finding = max(3, min(sample_size // len(valid_findings_cols), 10))
            
            # For each finding, sample images where that finding is present
            np.random.seed(random_seed)
            for finding in valid_findings_cols:
                if finding == 'No Finding':
                    continue  # We'll handle normal cases separately
                
                # Get images with this finding
                finding_df = df[df[finding] == 1.0]
                
                if len(finding_df) > 0:
                    # Sample from this finding group
                    if len(finding_df) > target_per_finding:
                        finding_sample = finding_df.sample(target_per_finding, random_state=random_seed)
                    else:
                        finding_sample = finding_df
                    
                    # Add to our balanced sample
                    balanced_sample.append(finding_sample)
                    findings_count[finding] = len(finding_sample)
                    
                    st.info(f"Added {len(finding_sample)} images with {finding}")
            
            # Add normal cases (No Finding)
            normal_df = df[df['No Finding'] == 1.0]
            if len(normal_df) > 0:
                normal_count = min(len(normal_df), target_per_finding)
                normal_sample = normal_df.sample(normal_count, random_state=random_seed)
                balanced_sample.append(normal_sample)
                findings_count['No Finding'] = normal_count
                st.info(f"Added {normal_count} normal images (No Finding)")
            
            # Combine all samples
            df = pd.concat(balanced_sample)
            
            # If we haven't reached our sample size, add more random images
            if len(df) < sample_size:
                remaining = sample_size - len(df)
                
                # Get images not already in our sample
                remaining_df = pd.concat([df, df]).drop_duplicates(keep=False)
                
                if len(remaining_df) > 0:
                    random_sample = remaining_df.sample(min(remaining, len(remaining_df)), 
                                                       random_state=random_seed)
                    df = pd.concat([df, random_sample])
                    st.info(f"Added {len(random_sample)} additional random images to reach target sample size")
            
            # If we have too many, randomly subsample
            if len(df) > sample_size:
                df = df.sample(sample_size, random_state=random_seed)
            
            st.info(f"Final balanced sample: {len(df)} images")
            
            # Display findings distribution
            findings_summary = []
            for finding in valid_findings_cols:
                count = len(df[df[finding] == 1.0])
                findings_summary.append(f"{finding}: {count}")
            
            st.info("Distribution of findings in sample: " + ", ".join(findings_summary))
            
        elif sample_size < len(df):
            # Simple random sampling if not balanced
            df = df.sample(sample_size, random_state=random_seed)
            st.info(f"Randomly sampled {sample_size} images")
        
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
        for idx in db.metadata_db:
            if 'file_path' in db.metadata_db[idx]:
                processed_paths.add(db.metadata_db[idx]['file_path'])
            if 'image_hash' in db.metadata_db[idx]:
                existing_image_hashes.add(db.metadata_db[idx]['image_hash'])
        
        for i, (idx, row) in enumerate(df.iterrows()):
            try:
                # Update progress
                progress = (i + 1) / len(df)
                progress_bar.progress(progress)
                status_text.text(f"Processing image {i+1}/{len(df)}")
                
                # Construct image path (try several possible paths)
                # Path 1: Direct from CSV
                img_path = os.path.join(chexpert_dir, row['Path'])
                
                # Path 2: With CheXpert-v1.0-small prefix
                if not os.path.exists(img_path):
                    img_path = os.path.join(chexpert_dir, "CheXpert-v1.0-small", row['Path'])
                
                # Path 3: Just the filename in the root directory
                if not os.path.exists(img_path):
                    img_path = os.path.join(chexpert_dir, os.path.basename(row['Path']))
                
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
                image_id = db.add_image(model, image, metadata)
                
                if image_id >= 0:
                    added_count += 1
                    logger.info(f"Added image with ID: {image_id}")
                else:
                    skipped_count += 1
                    logger.warning(f"Failed to add image: {img_path}")
                
            except Exception as e:
                status_text.text(f"Error processing {os.path.basename(row.get('Path', 'image'))}: {str(e)}")
                logger.error(f"Error processing image: {str(e)}")
                skipped_count += 1
        
        # Show summary
        st.success(f"Added {added_count} images to database. Skipped {skipped_count} images. Detected {duplicate_count} duplicates.")
        logger.info(f"Added {added_count} images, skipped {skipped_count}, duplicates {duplicate_count}")
        
    except Exception as e:
        st.error(f"Error creating subset: {str(e)}")
        logger.error(f"Error creating subset: {str(e)}")
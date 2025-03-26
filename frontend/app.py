# app.py - Simplified Streamlit Medical Image Retrieval Application
import streamlit as st
import torch
import numpy as np
import os
import pandas as pd
from PIL import Image
import faiss
import base64
from io import BytesIO
import time
import pickle
import logging
import json

# Set Kaggle configuration directory
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.kaggle')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical-retrieval-app")

# Import the fixed model adapter
from model_adapter import BiomedCLIPAdapter
from chexpert_loader import add_chexpert_loader_to_sidebar

# App configuration and constants
MODEL_PATH = "model_checkpoints/best_biomedclip_big.pth"
DATABASE_DIR = "database"
FAISS_INDEX_PATH = os.path.join(DATABASE_DIR, "index.faiss")
METADATA_PATH = os.path.join(DATABASE_DIR, "metadata.pkl")
IMAGE_DB_PATH = os.path.join(DATABASE_DIR, "image_db.pkl")

# Create database directory if it doesn't exist
os.makedirs(DATABASE_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Medical X-ray Retrieval System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def base64_to_image(base64_str):
    """Convert base64 string to PIL Image"""
    img_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_data))

def calculate_image_hash(image, hash_size=8):
    """
    Calculate a simple perceptual hash for an image.
    This helps identify duplicate images even if they have different filenames.
    """
    # Resize the image to a small square
    image = image.resize((hash_size, hash_size), Image.LANCZOS)
    
    # Convert to grayscale
    image = image.convert("L")
    
    # Get pixel data
    pixels = list(image.getdata())
    
    # Calculate average pixel value
    avg_pixel = sum(pixels) / len(pixels)
    
    # Create binary hash
    binary_hash = ''.join('1' if pixel > avg_pixel else '0' for pixel in pixels)
    
    # Convert binary hash to hexadecimal for compact storage
    hex_hash = hex(int(binary_hash, 2))[2:].zfill(hash_size**2 // 4)
    
    return hex_hash

@st.cache_resource
def load_model():
    """Load the BiomedCLIP model"""
    try:
        return BiomedCLIPAdapter(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def load_database():
    """Load or initialize the vector database"""
    # Initialize empty database
    index = None
    metadata_db = {}
    image_db = {}
    
    # Try to load existing database
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH) and os.path.exists(IMAGE_DB_PATH):
            logger.info("Loading existing database...")
            index = faiss.read_index(FAISS_INDEX_PATH)
            
            with open(METADATA_PATH, 'rb') as f:
                metadata_db = pickle.load(f)
            
            with open(IMAGE_DB_PATH, 'rb') as f:
                image_db = pickle.load(f)
                
            logger.info(f"Database loaded with {index.ntotal} images")
        else:
            # Create new FAISS index for 256-dimensional embeddings
            logger.info("Creating new database...")
            index = faiss.IndexFlatIP(256)  # Inner product (cosine) similarity for normalized vectors
            logger.info("New database initialized")
    except Exception as e:
        logger.error(f"Error loading database: {str(e)}")
        # Create new index if loading fails
        index = faiss.IndexFlatIP(256)
    
    return index, metadata_db, image_db

def save_database(index, metadata_db, image_db):
    """Save the database to disk"""
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
        
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata_db, f)
        
        with open(IMAGE_DB_PATH, 'wb') as f:
            pickle.dump(image_db, f)
            
        logger.info(f"Database saved with {index.ntotal} images")
        return True
    except Exception as e:
        logger.error(f"Error saving database: {str(e)}")
        return False

def search_by_image(model, index, metadata_db, image_db, query_image, max_results=10):
    """Search for similar images in the database"""
    try:
        # Get embedding for query image
        embedding = model.encode_image(query_image)[0]
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        # Search in index
        k = min(max_results, index.ntotal) if index.ntotal > 0 else 0
        if k == 0:
            return []
        
        scores, indices = index.search(np.array([embedding], dtype=np.float32), k)
        
        # Collect results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(image_db):
                results.append({
                    'id': int(idx),
                    'score': float(scores[0][i]),
                    'image': image_db.get(int(idx), None),
                    'metadata': metadata_db.get(int(idx), {})
                })
        
        return results
    except Exception as e:
        logger.error(f"Error in image search: {str(e)}")
        return []

def search_by_text(model, index, metadata_db, image_db, query_text, max_results=10):
    """Search for images matching a text description"""
    try:
        # Prepare query text
        if not query_text.lower().startswith("this is a photo of"):
            query_text = "this is a photo of " + query_text
        
        # Get embedding for query text
        embedding = model.encode_text(query_text)[0]
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        # Search in index
        k = min(max_results, index.ntotal) if index.ntotal > 0 else 0
        if k == 0:
            return []
        
        scores, indices = index.search(np.array([embedding], dtype=np.float32), k)
        
        # Collect results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(image_db):
                results.append({
                    'id': int(idx),
                    'score': float(scores[0][i]),
                    'image': image_db.get(int(idx), None),
                    'metadata': metadata_db.get(int(idx), {})
                })
        
        return results
    except Exception as e:
        logger.error(f"Error in text search: {str(e)}")
        return []
    
def multimodal_search(model, index, metadata_db, image_db, query_image, query_text, max_results=10, weight_image=0.5):
    """
    Search for similar images using both image and text inputs.
    
    Args:
        model: The BiomedCLIP model adapter
        index: FAISS index
        metadata_db: Metadata database
        image_db: Image database
        query_image: The query image (PIL.Image)
        query_text: The query text
        max_results: Maximum number of results to return
        weight_image: Weight for image similarity (0.0-1.0), text is 1-weight_image
    
    Returns:
        List of results
    """
    try:
        # Get embedding for query image
        image_embedding = model.encode_image(query_image)[0]
        
        # Normalize image embedding
        image_embedding = image_embedding / np.linalg.norm(image_embedding)
        
        # Prepare query text
        if not query_text.lower().startswith("this is a photo of"):
            query_text = "this is a photo of " + query_text
        
        # Get embedding for query text
        text_embedding = model.encode_text(query_text)[0]
        
        # Normalize text embedding
        text_embedding = text_embedding / np.linalg.norm(text_embedding)
        
        # Combine embeddings with weighted average
        combined_embedding = (weight_image * image_embedding) + ((1 - weight_image) * text_embedding)
        
        # Normalize the combined embedding
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
        
        # Search in index
        k = min(max_results, index.ntotal) if index.ntotal > 0 else 0
        if k == 0:
            return []
        
        scores, indices = index.search(np.array([combined_embedding], dtype=np.float32), k)
        
        # Collect results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(image_db):
                results.append({
                    'id': int(idx),
                    'score': float(scores[0][i]),
                    'image': image_db.get(int(idx), None),
                    'metadata': metadata_db.get(int(idx), {})
                })
        
        return results
    except Exception as e:
        logger.error(f"Error in multimodal search: {str(e)}")
        return []

def format_metadata_for_display(metadata):
    """Format metadata for better display in the UI"""
    formatted = {}
    
    # Skip internal fields and path information
    skip_fields = ['Path', 'path', 'file_path', 'image_hash', 'findings_status']  # Add findings_status to skip
    
    # Process findings separately
    findings = []
    
    # Process findings_status field if it exists
    if 'findings_status' in metadata:
        for finding, status in metadata['findings_status'].items():
            if status == 'Present':
                findings.append(finding)
        # Avoid duplicate display of findings
        skip_fields.extend(list(metadata['findings_status'].keys()))
    # Standard findings processing for backward compatibility
    else:
        # Process each metadata field for old-style findings (with value 1.0 or 1)
        for key, value in metadata.items():
            if key in ['No Finding', 'Pneumonia', 'Cardiomegaly', 'Pleural Effusion', 
                      'Atelectasis', 'Pneumothorax', 'Edema', 'Consolidation', 
                      'Lung Opacity', 'Lung Lesion', 'Fracture', 'Support Devices',
                      'no_finding', 'pneumonia', 'cardiomegaly', 'pleural_effusion',
                      'atelectasis', 'pneumothorax', 'edema', 'consolidation',
                      'lung_opacity', 'lung_lesion', 'fracture', 'support_devices']:
                # If it's a finding with value 1, add to findings list
                if value == 1.0 or value == 1:
                    # Convert underscore format to title case
                    proper_name = key.replace('_', ' ').title()
                    findings.append(proper_name)
                skip_fields.append(key)
    
    # Process the rest of the metadata
    for key, value in metadata.items():
        # Skip fields we don't want to display
        if key in skip_fields:
            continue
            
        # Skip NaN values
        if isinstance(value, float) and np.isnan(value):
            continue
            
        # Format the rest
        formatted[key.replace('_', ' ').title()] = value
    
    # Add findings as a formatted list if there are any
    if findings:
        formatted['Findings'] = ', '.join(findings)
    elif 'Findings' not in formatted and 'findings_text' not in metadata:
        formatted['Findings'] = 'None reported'
    
    return formatted

def add_image_to_database(model, index, metadata_db, image_db, image, metadata=None):
    """Add an image to the database"""
    try:
        # Calculate image hash to check for duplicates
        img_hash = calculate_image_hash(image)
        
        # Check if this image hash already exists in the database
        for idx in metadata_db:
            if metadata_db[idx].get('image_hash') == img_hash:
                return -2  # Duplicate image
        
        # Get embedding for image
        embedding = model.encode_image(image)[0]
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        # Add to index
        index.add(np.array([embedding], dtype=np.float32))
        
        # Current index is ntotal - 1
        current_idx = index.ntotal - 1
        
        # Create thumbnail and convert to base64
        image_copy = image.copy()
        image_copy.thumbnail((224, 224))
        img_base64 = image_to_base64(image_copy)
        
        # Add image hash to metadata
        if metadata is None:
            metadata = {}
        metadata['image_hash'] = img_hash
        
        # Store metadata and image
        metadata_db[current_idx] = metadata
        image_db[current_idx] = img_base64
        
        # Save database
        save_database(index, metadata_db, image_db)
        
        return current_idx
    except Exception as e:
        logger.error(f"Error adding image to database: {str(e)}")
        return -1

def delete_image_from_database(index, metadata_db, image_db, image_id):
    """Delete an image from the database"""
    try:
        # FAISS doesn't support direct deletion, so we rebuild the index
        all_ids = list(range(index.ntotal))
        keep_ids = [idx for idx in all_ids if idx != image_id]
        
        if not keep_ids:  # If this was the only image
            new_index = faiss.IndexFlatIP(256)
            new_metadata_db = {}
            new_image_db = {}
        else:
            # Get embeddings for all IDs to keep
            embeddings = []
            for i in keep_ids:
                # We have to search for this ID to get its embedding
                d, i_found = index.search(np.array([index.reconstruct(i)], dtype=np.float32), 1)
                embeddings.append(index.reconstruct(i))
            
            # Create new index
            new_index = faiss.IndexFlatIP(256)
            new_index.add(np.array(embeddings, dtype=np.float32))
            
            # Create new metadata and image databases
            new_metadata_db = {}
            new_image_db = {}
            
            for new_id, old_id in enumerate(keep_ids):
                new_metadata_db[new_id] = metadata_db[old_id]
                new_image_db[new_id] = image_db[old_id]
        
        # Save new database
        save_database(new_index, new_metadata_db, new_image_db)
        
        return new_index, new_metadata_db, new_image_db
    except Exception as e:
        logger.error(f"Error deleting image: {str(e)}")
        return index, metadata_db, image_db

def display_results(results, index, metadata_db, image_db):
    """Display search results in a grid"""
    if not results:
        st.warning("No results found.")
        return index, metadata_db, image_db  # Return the original objects
    
    st.success(f"Found {len(results)} results")
    
    # Track if we need to refresh (after deletion)
    needs_refresh = False
    
    # Display results in a grid (3 columns)
    cols_per_row = 3
    
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(results):
                with cols[j]:
                    result = results[idx]
                    
                    # Display image
                    if result['image']:
                        try:
                            image = base64_to_image(result['image'])
                            # Convert similarity score to percentage
                            score_percent = f"{result['score'] * 100:.2f}"
                            st.image(image, caption=f"Match: {score_percent}%", use_container_width=True)
                        except:
                            st.error("Could not display image")
                    
                    # Display metadata in an expander
                    with st.expander("View Details"):
                        st.write(f"Image ID: {result['id']}")
                        st.write(f"Similarity: {int(result['score'] * 100)}%")
                        
                        # Format metadata nicely
                        formatted_metadata = format_metadata_for_display(result.get('metadata', {}))
                        for key, value in formatted_metadata.items():
                            st.write(f"**{key}:** {value}")
                    
                    # Add button to delete
                    if st.button(f"Delete", key=f"delete_{result['id']}"):
                        new_index, new_metadata_db, new_image_db = delete_image_from_database(
                            index, metadata_db, image_db, result['id']
                        )
                        
                        # Update the variables with the return values
                        index, metadata_db, image_db = new_index, new_metadata_db, new_image_db
                        
                        st.success(f"Image {result['id']} deleted")
                        needs_refresh = True
    
    # Check if we need to refresh the page
    if needs_refresh:
        st.rerun()
    
    return index, metadata_db, image_db  # Return possibly modified objects

# Main application
def main():
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check the model path and try again.")
        return
    
    # Load or initialize database
    index, metadata_db, image_db = load_database()
    
    # Sidebar
    with st.sidebar:
        st.title("Medical X-ray Retrieval")
        st.write("Use this tool to search for similar medical X-ray images.")
        
        # Database stats
        st.subheader("Database")
        st.metric("Total Images", index.ntotal if index else 0)
        
        # Clear database button
        if index and index.ntotal > 0:
            if st.button("Clear Database"):
                index = faiss.IndexFlatIP(256)
                metadata_db = {}
                image_db = {}
                save_database(index, metadata_db, image_db)
                st.success("Database cleared successfully.")
                st.rerun()
        
        # Add the simplified CheXpert loader
        st.subheader("Load CheXpert Data")
        
        # Hardcoded path - no need for user input
        chexpert_dir = "/Users/jarinatkpoyta/project-karhu/archive"
        st.info(f"Using CheXpert at: {chexpert_dir}")
        
        # Simple options
        csv_option = st.radio("Dataset to use", ["valid.csv (smaller)", "train.csv (larger)"])
        use_train = "train.csv" in csv_option
        csv_file = os.path.join(chexpert_dir, "train.csv" if use_train else "valid.csv")
        
        sample_size = st.slider("Number of images", min_value=10, max_value=100, value=30)
        
        # Option to check for duplicates
        check_duplicates = st.checkbox("Skip duplicate images", value=True)
        
        if st.button("Load Images"):
            if os.path.exists(csv_file):
                with st.spinner(f"Loading {sample_size} images..."):
                    from chexpert_loader import create_subset_from_local_data
                    index, metadata_db, image_db = create_subset_from_local_data(
                        model, index, metadata_db, image_db, save_database,
                        sample_size=sample_size, use_train=use_train
                    )
                st.success(f"Images loaded: {index.ntotal} total")
            else:
                st.error(f"CSV file not found: {csv_file}")
                
        # Show database stats button
        if index and index.ntotal > 0:
            if st.button("Show Database Stats"):
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
                    for key, value in metadata_db[idx].items():
                        if key in ['No Finding', 'Pneumonia', 'Cardiomegaly', 'Pleural Effusion', 
                                'Atelectasis', 'Pneumothorax', 'Edema', 'Consolidation', 
                                'Lung Opacity', 'Lung Lesion', 'Fracture', 'Support Devices',
                                'no_finding', 'pneumonia', 'cardiomegaly', 'pleural_effusion',
                                'atelectasis', 'pneumothorax', 'edema', 'consolidation',
                                'lung_opacity', 'lung_lesion', 'fracture', 'support_devices']:
                            if value == 1.0 or value == 1:
                                proper_name = key.replace('_', ' ').title()
                                findings_count[proper_name] = findings_count.get(proper_name, 0) + 1
                
                # Display basic stats
                st.info(f"""
                Database statistics:
                - Total images: {index.ntotal}
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
    
    # Main area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Image Search", "Text Search", "Add to Database", "Multimodal Search"])
    
    # Tab 1: Image Search
    with tab1:
        st.header("Search by Image")
        st.write("Upload an X-ray image to find similar cases in the database.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"], key="image_search")
            
            if uploaded_file:
                query_image = Image.open(uploaded_file).convert("RGB")
                st.image(query_image, caption="Query Image", use_container_width=True)
                
                max_results = st.slider("Number of results", 1, 50, 10, key="image_results")
                
                if st.button("Search", key="search_image"):
                    if index.ntotal == 0:
                        st.warning("The database is empty. Please add some images first.")
                    else:
                        with st.spinner("Searching..."):
                            start_time = time.time()
                            results = search_by_image(model, index, metadata_db, image_db, query_image, max_results)
                            elapsed_time = time.time() - start_time
                        
                        st.info(f"Search completed in {elapsed_time:.2f} seconds")
                        display_results(results, index, metadata_db, image_db)
    
    # Tab 2: Text Search
    with tab2:
        st.header("Search by Text")
        st.write("Describe what you're looking for to find matching X-ray images.")
        
        # Example queries
        example_queries = [
            "chest x-ray with pneumonia",
            "normal chest x-ray with no findings",
            "chest x-ray with pleural effusion",
            "chest x-ray with cardiomegaly",
            "chest x-ray with lung opacity"
        ]
        
        selected_example = st.selectbox(
            "Select an example query or enter your own below:", 
            [""] + example_queries
        )
        
        query_text = st.text_input(
            "Enter your query:", 
            value=selected_example if selected_example else ""
        )
        
        if query_text:
            max_results = st.slider("Number of results", 1, 50, 10, key="text_results")
            
            if st.button("Search", key="search_text"):
                if index.ntotal == 0:
                    st.warning("The database is empty. Please add some images first.")
                else:
                    with st.spinner("Searching..."):
                        start_time = time.time()
                        results = search_by_text(model, index, metadata_db, image_db, query_text, max_results)
                        elapsed_time = time.time() - start_time
                    
                    st.info(f"Search completed in {elapsed_time:.2f} seconds")
                    display_results(results, index, metadata_db, image_db)
    
    # Tab 3: Add to Database
    with tab3:
        st.header("Add to Database")
        st.write("Upload X-ray images to add to the database.")
        
        upload_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"], key="upload")
        
        if upload_file:
            image = Image.open(upload_file).convert("RGB")
            st.image(image, caption="Image to Add", width=300)
            
            # Metadata form
            st.subheader("Image Metadata")
            
            col1, col2 = st.columns(2)
            
            with col1:
                patient_id = st.text_input("Patient ID")
                age = st.number_input("Age", min_value=0, max_value=120)
                sex = st.selectbox("Sex", ["", "Male", "Female", "Other"])
            
            with col2:
                modality = st.selectbox("Modality", ["", "X-ray", "CT", "MRI", "Ultrasound"])
                view = st.selectbox("View", ["", "Frontal", "Lateral", "AP", "PA"])
                date = st.date_input("Date")
            
            # Findings checkboxes
            st.subheader("Clinical Findings")
            col_a, col_b = st.columns(2)
            
            with col_a:
                no_finding = st.checkbox("No Finding")
                pneumonia = st.checkbox("Pneumonia")
                cardiomegaly = st.checkbox("Cardiomegaly")
                effusion = st.checkbox("Pleural Effusion")
                atelectasis = st.checkbox("Atelectasis")
                pneumothorax = st.checkbox("Pneumothorax")
            
            with col_b:
                edema = st.checkbox("Edema")
                consolidation = st.checkbox("Consolidation")
                lung_opacity = st.checkbox("Lung Opacity")
                lung_lesion = st.checkbox("Lung Lesion")
                fracture = st.checkbox("Fracture")
                support_devices = st.checkbox("Support Devices")
            
            # Notes
            notes = st.text_area("Additional Notes")
            
            # Add to database button
            if st.button("Add to Database"):
                metadata = {
                    "filename": upload_file.name,
                    "uploaded_at": str(time.time()),
                }
                
                # Add form data to metadata
                if patient_id:
                    metadata["patient_id"] = patient_id
                if age > 0:
                    metadata["age"] = age
                if sex:
                    metadata["sex"] = sex
                if modality:
                    metadata["modality"] = modality
                if view:
                    metadata["view"] = view
                if date:
                    metadata["date"] = str(date)
                
                # Process findings in a more user-friendly way
                findings = []
                findings_status = {}
                
                finding_map = {
                    "No Finding": no_finding,
                    "Pneumonia": pneumonia,
                    "Cardiomegaly": cardiomegaly,
                    "Pleural Effusion": effusion,
                    "Atelectasis": atelectasis,
                    "Pneumothorax": pneumothorax,
                    "Edema": edema,
                    "Consolidation": consolidation,
                    "Lung Opacity": lung_opacity,
                    "Lung Lesion": lung_lesion,
                    "Fracture": fracture,
                    "Support Devices": support_devices
                }
                
                for finding, value in finding_map.items():
                    if value:
                        findings.append(finding)
                        metadata[finding] = 1  # Keep original format for compatibility
                    else:
                        metadata[finding] = 0  # Keep original format for compatibility
                
                # Create text description
                if findings and not no_finding:
                    findings_text = "chest x-ray with " + ", ".join([f.lower() for f in findings if f != "No Finding"])
                else:
                    findings_text = "normal chest x-ray with no findings"
                
                metadata["findings_text"] = findings_text
                
                if notes:
                    metadata["notes"] = notes
                
                # Add to database
                with st.spinner("Adding to database..."):
                    image_id = add_image_to_database(model, index, metadata_db, image_db, image, metadata)
                
                if image_id >= 0:
                    st.success(f"Image added to database with ID: {image_id}")
                    st.info(f"Total images in database: {index.ntotal}")
                elif image_id == -2:
                    st.warning("This image appears to be a duplicate of an existing image in the database.")
                else:
                    st.error("Failed to add image to database")
    # Tab 4: Multimodal Search
    with tab4:
            st.header("Multimodal Search")
            st.write("Search using both an image and text description to find the most relevant cases.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Image Input")
                uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"], key="multimodal_image")
                
                if uploaded_file:
                    query_image = Image.open(uploaded_file).convert("RGB")
                    st.image(query_image, caption="Query Image", width=300)
            
            with col2:
                st.subheader("Text Input")
                
                # Example queries for user convenience
                example_queries = [
                    "chest x-ray with pneumonia",
                    "normal chest x-ray with no findings",
                    "chest x-ray with pleural effusion",
                    "chest x-ray with cardiomegaly",
                    "chest x-ray with lung opacity"
                ]
                
                selected_example = st.selectbox(
                    "Select an example query or enter your own below:", 
                    [""] + example_queries,
                    key="multimodal_examples"
                )
                
                query_text = st.text_input(
                    "Enter additional details or specific conditions to look for:",
                    value=selected_example if selected_example else "",
                    key="multimodal_text"
                )
                
                # Weight slider to balance image vs text importance
                weight_image = st.slider(
                    "Image/Text Weight Balance", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.5, 
                    step=0.1,
                    help="Higher values give more importance to the image, lower values to the text"
                )
            
            # Number of results
            max_results = st.slider("Number of results", 1, 50, 10, key="multimodal_results")
            
            # Search button
            if st.button("Search", key="search_multimodal"):
                if not uploaded_file:
                    st.warning("Please upload an image for multimodal search.")
                elif not query_text:
                    st.warning("Please enter text for multimodal search.")
                elif index.ntotal == 0:
                    st.warning("The database is empty. Please add some images first.")
                else:
                    with st.spinner("Searching..."):
                        start_time = time.time()
                        results = multimodal_search(
                            model, 
                            index, 
                            metadata_db, 
                            image_db, 
                            query_image, 
                            query_text, 
                            max_results, 
                            weight_image
                        )
                        elapsed_time = time.time() - start_time
                    
                    st.info(f"Search completed in {elapsed_time:.2f} seconds using {int(weight_image*100)}% image and {int((1-weight_image)*100)}% text weighting")
                    display_results(results, index, metadata_db, image_db)

if __name__ == "__main__":
    main()
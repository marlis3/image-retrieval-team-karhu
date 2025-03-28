"""
Main application file for the Medical X-ray Retrieval System
"""
import streamlit as st
import os
import logging
import sys

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create root logger
logger = logging.getLogger("medical-retrieval-app")
logger.setLevel(logging.DEBUG)

# Initialize directories
logger.info("Initializing application directories...")
os.makedirs("data", exist_ok=True)
os.makedirs("data/database", exist_ok=True)
os.makedirs("model_checkpoints", exist_ok=True)

# Import configuration
from config.config import (
    APP_TITLE,
    APP_ICON, 
    APP_LAYOUT, 
    INITIAL_SIDEBAR_STATE,
    MODEL_PATH,
    CHEXPERT_DIR
)

# Log configuration settings
logger.info(f"Model path: {MODEL_PATH}")
logger.info(f"CheXpert directory: {CHEXPERT_DIR}")

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    logger.warning(f"Model file not found: {MODEL_PATH}")
else:
    logger.info(f"Model file found: {MODEL_PATH}")

# Check if CheXpert directory exists
if not os.path.exists(CHEXPERT_DIR):
    logger.warning(f"CheXpert directory not found: {CHEXPERT_DIR}")
else:
    logger.info(f"CheXpert directory found: {CHEXPERT_DIR}")
    # Check for CSV files
    train_csv = os.path.join(CHEXPERT_DIR, "train.csv")
    valid_csv = os.path.join(CHEXPERT_DIR, "valid.csv")
    
    if os.path.exists(train_csv):
        logger.info(f"Found train.csv: {train_csv}")
    else:
        logger.warning(f"train.csv not found: {train_csv}")
        
    if os.path.exists(valid_csv):
        logger.info(f"Found valid.csv: {valid_csv}")
    else:
        logger.warning(f"valid.csv not found: {valid_csv}")

# Import backend components
from backend.model_adapter import BiomedCLIPAdapter
from backend.database import VectorDatabase
from backend.search import SearchEngine

# Import UI components
from components.sidebar import render_sidebar
from components.image_search import render_image_search_tab
from components.text_search import render_text_search_tab
from components.database_management import render_database_mgmt_tab

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

@st.cache_resource
def load_model():
    """Load the BiomedCLIP model"""
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        model = BiomedCLIPAdapter(MODEL_PATH)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

@st.cache_resource
def initialize_database():
    """Initialize the vector database"""
    logger.info("Initializing database")
    return VectorDatabase()

def main():
    """Main application function"""
    
    # Display application header
    st.title("Medical X-ray Retrieval System")
    
    # Load model
    with st.spinner("Loading ML model..."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check the model path and try again.")
        st.error(f"Expected model path: {MODEL_PATH}")
        return
    
    # Initialize database
    with st.spinner("Initializing database..."):
        db = initialize_database()
    
    # Initialize search engine
    search_engine = SearchEngine(model, db)
    
    # Sidebar
    with st.sidebar:
        render_sidebar(db, model)
    
    # Main area with tabs
    tab1, tab2, tab3 = st.tabs(["Image Search", "Text Search", "Add to Database"])
    
    # Tab 1: Image Search
    with tab1:
        render_image_search_tab(search_engine)
    
    # Tab 2: Text Search
    with tab2:
        render_text_search_tab(search_engine)
    
    # Tab 3: Add to Database
    with tab3:
        render_database_mgmt_tab(model, db)

if __name__ == "__main__":
    main()
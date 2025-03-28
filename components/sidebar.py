"""
Sidebar UI components for the Medical X-ray Retrieval System
"""
import streamlit as st
import os
import pandas as pd
import time
import logging
from config.config import CHEXPERT_DIR

logger = logging.getLogger("medical-retrieval.sidebar")

def render_sidebar(db, model):
    """
    Render the application sidebar
    
    Args:
        db: Database instance
        model: Model instance
        
    Returns:
        None
    """
    st.title("Medical X-ray Retrieval")
    st.write("Use this tool to search for similar medical X-ray images.")
    
    # Database stats
    st.subheader("Database")
    st.metric("Total Images", db.index.ntotal if db.index else 0)
    
    # Clear database button
    if db.index and db.index.ntotal > 0:
        if st.button("Clear Database"):
            if db.clear_database():
                st.success("Database cleared successfully.")
                st.rerun()
            else:
                st.error("Failed to clear the database.")
    
    # Add the CheXpert loader section
    st.subheader("Load CheXpert Data")
    
    # Hardcoded path - no need for user input
    chexpert_dir = CHEXPERT_DIR
    st.info(f"Using CheXpert at: {chexpert_dir}")
    
    # Simple options
    csv_option = st.radio("Dataset to use", ["valid.csv (smaller)", "train.csv (larger)"])
    use_train = "train.csv" in csv_option
    csv_file = os.path.join(chexpert_dir, "train.csv" if use_train else "valid.csv")
    
    sample_size = st.slider("Number of images", min_value=10, max_value=100, value=50)
    
    # Option to check for duplicates
    check_duplicates = st.checkbox("Skip duplicate images", value=True)
    
    if st.button("Load Images"):
        if os.path.exists(csv_file):
            # Import here to avoid circular imports
            from utils.chexpert_loader import create_subset_from_local_data
            create_subset_from_local_data(
                model, db, sample_size=sample_size, use_train=use_train
            )
            st.success(f"Images loaded: {db.index.ntotal} total")
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"CSV file not found: {csv_file}")
            
    # Show database stats button
    if db.index and db.index.ntotal > 0:
        if st.button("Show Database Stats"):
            # Get database statistics
            stats = db.get_database_stats()
            
            # Display basic stats
            st.info(f"""
            Database statistics:
            - Total images: {stats['total_images']}
            - Unique image hashes: {stats['unique_hashes_count']}
            - Potential duplicates: {stats['potential_duplicates']}
            """)
            
            # Display findings distribution
            if stats["findings_count"]:
                st.subheader("Findings Distribution")
                findings_df = pd.DataFrame({
                    'Finding': list(stats["findings_count"].keys()),
                    'Count': list(stats["findings_count"].values())
                }).sort_values(by='Count', ascending=False)
                
                st.dataframe(findings_df)
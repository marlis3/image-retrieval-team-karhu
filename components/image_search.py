"""
Image search UI component for the Medical X-ray Retrieval System
"""
import streamlit as st
import time
from PIL import Image
from components.results import display_results

def render_image_search_tab(search_engine):
    """
    Render the image search tab
    
    Args:
        search_engine: SearchEngine instance
        
    Returns:
        None
    """
    st.header("Search by Image")
    st.write("Upload an X-ray image to find similar cases in the database.")
            
    uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"], key="image_search")
        
    if uploaded_file:
        query_image = Image.open(uploaded_file).convert("RGB")
        st.image(query_image, caption="Query Image", use_container_width=True)
            
        max_results = st.slider("Number of results", 1, 50, 10, key="image_results")
            
        if st.button("Search", key="search_image"):
            if search_engine.db.index.ntotal == 0:
                st.warning("The database is empty. Please add some images first.")
            else:
                with st.spinner("Searching..."):
                    start_time = time.time()
                    results = search_engine.search_by_image(query_image, max_results)
                    elapsed_time = time.time() - start_time
                    
                st.info(f"Search completed in {elapsed_time:.2f} seconds")
                display_results(results)
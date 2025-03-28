"""
Text search UI component for the Medical X-ray Retrieval System
"""
import streamlit as st
import time
from components.results import display_results
from config.config import EXAMPLE_QUERIES

def render_text_search_tab(search_engine):
    """
    Render the text search tab
    
    Args:
        search_engine: SearchEngine instance
        
    Returns:
        None
    """
    st.header("Search by Text")
    st.write("Describe what you're looking for to find matching X-ray images.")
    
    # Example queries
    selected_example = st.selectbox(
        "Select an example query or enter your own below:", 
        [""] + EXAMPLE_QUERIES
    )
    
    query_text = st.text_input(
        "Enter your query:", 
        value=selected_example if selected_example else ""
    )
    
    if query_text:
        max_results = st.slider("Number of results", 1, 50, 10, key="text_results")
        
        if st.button("Search", key="search_text"):
            if search_engine.db.index.ntotal == 0:
                st.warning("The database is empty. Please add some images first.")
            else:
                with st.spinner("Searching..."):
                    start_time = time.time()
                    results = search_engine.search_by_text(query_text, max_results)
                    elapsed_time = time.time() - start_time
                
                st.info(f"Search completed in {elapsed_time:.2f} seconds")
                display_results(results)
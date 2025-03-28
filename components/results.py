"""
Results display UI component for the Medical X-ray Retrieval System
"""
import streamlit as st
from backend.utils import base64_to_image, format_metadata_for_display

def display_results(results):
    """
    Display search results in a grid
    
    Args:
        results (list): List of search results
        
    Returns:
        None
    """
    if not results:
        st.warning("No results found.")
        return
    
    st.success(f"Found {len(results)} results")
    
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
import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd

st.set_page_config(layout="wide", page_title="Medical Image Retrieval System")

st.title("Medical X-ray Retrieval System")
st.markdown("Compare X-rays with similar past cases using fine-tuned CLIP embeddings")

# Create tabs for different search types
tab1, tab2, tab3 = st.tabs(["Image Search", "Text Search", "Add New Case"])

with tab1:
    st.header("Search by Image")
    
    uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"], key="image_search")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
            
            # Add a search button
            if st.button("Find Similar Cases"):
                with st.spinner("Analyzing image and retrieving similar cases..."):
                    # Ensure image file is sent properly
                    uploaded_file.seek(0)  # Reset file pointer
                    files = {"file": uploaded_file.read()}  # Read the actual file contents

                    # Make API request to backend
                    response = requests.post("http://localhost:8000/search_by_image/", files=files)

                    if response.status_code == 200:
                        results = response.json()
                        st.session_state.search_results = results
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
    
    with col2:
        if 'search_results' in st.session_state:
            results = st.session_state.search_results
            if results and len(results['similar_images']) > 0:
                st.success(f"Found {len(results['similar_images'])} similar cases")

                for i, (image_name, similarity) in enumerate(zip(results['similar_images'], results['distances'])):
                    with st.container():
                        cols = st.columns([1, 3])
                        with cols[0]:
                            st.metric("Case", image_name)
                            st.metric("Similarity", f"{similarity:.4f}")
                        with cols[1]:
                            st.markdown(f"**Case #{image_name}**")
                            st.markdown(f"Similarity Score: {similarity:.4f}")
                            st.markdown(f"*Additional metadata retrieval needs backend support*")
                        st.divider()
            else:
                st.warning("No similar cases found. Try another image.")

with tab2:
    st.header("Search by Description")
    
    query = st.text_area("Describe the medical condition or X-ray features", 
                         "Example: Bilateral perihilar opacities consistent with pulmonary edema")
    
    if st.button("Search"):
        with st.spinner("Retrieving similar cases..."):
            response = requests.post("http://localhost:8000/text_search/", 
                                     json={"query": query})
            
            if response.status_code == 200:
                # For demo purposes, just a placeholder
                st.success("Found 5 similar cases")
                st.markdown("*Note: Text search endpoint needs to be implemented in your backend*")
                
                # Example display of results
                df = pd.DataFrame({
                    "Case ID": [101, 102, 103, 104, 105],
                    "Similarity": [0.92, 0.87, 0.85, 0.81, 0.78],
                    "Diagnosis": ["Pulmonary edema", "Congestive heart failure", 
                                 "Cardiomegaly", "Pleural effusion", "Pneumonia"]
                })
                st.table(df)
            else:
                st.error("Text search not yet implemented in backend")

with tab3:
    st.header("Add New Case to Database")
    
    uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"], key="add_case")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="X-ray to Add", use_column_width=True)
        
        # Add case button
        if st.button("Add to Database"):
            with st.spinner("Adding case..."):
                # Prepare the data for the API request
                files = {"file": uploaded_file.read()}

                # Send to backend
                response = requests.post("http://localhost:8000/add_image/", files=files)

                if response.status_code == 200:
                    st.success("Case successfully added to the database!")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")


# Add an "About" section with information about the system
with st.expander("About this System"):
    st.markdown("""
    **Medical Image Retrieval System Using Fine-tuned Embeddings**
    
    This system uses Vision Transformers (ViT) and CLIP models fine-tuned on medical images to 
    provide accurate similarity search for X-rays. The embeddings are stored in a FAISS index
    for efficient retrieval.
    
    The system can help doctors compare new X-rays with similar past cases to aid in diagnosis
    and treatment planning.
    """)
"""
Database management UI component for the Medical X-ray Retrieval System
"""
import streamlit as st
import time
from PIL import Image
from config.config import CLINICAL_FINDINGS

def render_database_mgmt_tab(model, db):
    """
    Render the database management tab
    
    Args:
        model: Model instance
        db: Database instance
        
    Returns:
        None
    """
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
        
        # Split findings into columns for better layout
        st.subheader("Clinical Findings")
        col_count = 2
        cols = st.columns(col_count)
        
        # Create a dictionary to hold checkbox values
        findings_values = {}
        
        # Distribute findings checkboxes among columns
        findings_per_col = len(CLINICAL_FINDINGS) // col_count + (1 if len(CLINICAL_FINDINGS) % col_count else 0)
        
        for i, finding in enumerate(CLINICAL_FINDINGS):
            col_idx = i // findings_per_col
            with cols[col_idx]:
                findings_values[finding] = st.checkbox(finding)
        
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
            
            # Process findings
            findings = []
            findings_status = {}
            
            for finding, value in findings_values.items():
                if value:
                    findings.append(finding)
                    metadata[finding] = 1  # Keep original format for compatibility
                    findings_status[finding] = 'Present'
                else:
                    metadata[finding] = 0  # Keep original format for compatibility
                    findings_status[finding] = 'Absent'
            
            # Add findings status object
            metadata["findings_status"] = findings_status
            
            # Create text description
            if findings and not findings_values.get("No Finding", False):
                findings_text = "chest x-ray with " + ", ".join([f.lower() for f in findings if f != "No Finding"])
            else:
                findings_text = "normal chest x-ray with no findings"
            
            metadata["findings_text"] = findings_text
            
            if notes:
                metadata["notes"] = notes
            
            # Add to database
            with st.spinner("Adding to database..."):
                image_id = db.add_image(model, image, metadata)
            
            if image_id >= 0:
                st.success(f"Image added to database with ID: {image_id}")
                st.info(f"Total images in database: {db.index.ntotal}")
            elif image_id == -2:
                st.warning("This image appears to be a duplicate of an existing image in the database.")
            else:
                st.error("Failed to add image to database")
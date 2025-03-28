"""
Utility functions for the Medical X-ray Retrieval System
"""
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger("medical-retrieval.utils")

def image_to_base64(image):
    """
    Convert PIL Image to base64 string
    
    Args:
        image (PIL.Image): The image to convert
        
    Returns:
        str: Base64 encoded string of the image
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def base64_to_image(base64_str):
    """
    Convert base64 string to PIL Image
    
    Args:
        base64_str (str): Base64 encoded string of an image
        
    Returns:
        PIL.Image: The decoded image
    """
    img_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_data))

def calculate_image_hash(image, hash_size=8):
    """
    Calculate a simple perceptual hash for an image.
    
    This helps identify duplicate images even if they have different filenames.
    
    Args:
        image (PIL.Image): The image to hash
        hash_size (int): The size of the hash (default: 8)
        
    Returns:
        str: Hexadecimal hash of the image
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

def format_metadata_for_display(metadata):
    """
    Format metadata for better display in the UI
    
    Args:
        metadata (dict): Raw metadata dictionary
        
    Returns:
        dict: Formatted metadata for display
    """
    formatted = {}
    
    # Skip internal fields and path information
    skip_fields = ['Path', 'path', 'file_path', 'image_hash', 'findings_status']
    
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
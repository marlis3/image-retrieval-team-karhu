from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np

# Load the processor and model from Hugging Face
model = CLIPModel.from_pretrained("marlis3/clip-for-project-model")
processor = CLIPProcessor.from_pretrained("marlis3/clip-for-project-processor")

def get_image_embedding(image):
    """Converts an image to an embedding using the fine-tuned CLIP model."""
    inputs = processor(images=image, return_tensors="pt")  # Convert to tensor
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)  # Get embedding

    return embedding.cpu().numpy().astype("float32").squeeze(0)
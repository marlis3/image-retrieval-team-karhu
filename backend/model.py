from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np

# Load the processor and model from Hugging Face
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image):
    """Converts an image to an embedding"""
    inputs = processor(images=image, padding=True)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
        return embedding.numpy().astype(np.float32)
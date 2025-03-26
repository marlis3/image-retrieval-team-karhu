import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_adapter")

class BiomedCLIP(nn.Module):
    def __init__(self, model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", projection_dim=256):
        super(BiomedCLIP, self).__init__()
        
        logger.info(f"Initializing BiomedCLIP base model from {model_name}")
        # Load base model directly from Hugging Face Hub
        self.model, self.preprocess = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer(model_name)
        self.projection_dim = projection_dim
        
        # Get model dimensions
        with torch.no_grad():
            # Dummy forward pass to get dimensions
            dummy_image = torch.zeros(1, 3, 224, 224)
            dummy_text = self.tokenizer(["test"], context_length=256)
            image_features, text_features, _ = self.model(dummy_image, dummy_text)
            self.vision_hidden_size = image_features.shape[1]
            self.text_hidden_size = text_features.shape[1]
        
        logger.info(f"Model dimensions: vision={self.vision_hidden_size}, text={self.text_hidden_size}")
        
        # Add projection layers for both vision and text
        self.vision_projection = nn.Linear(self.vision_hidden_size, projection_dim)
        self.text_projection = nn.Linear(self.text_hidden_size, projection_dim)
        
        # Initialize weights
        self.vision_projection.weight.data.normal_(mean=0.0, std=0.02)
        self.vision_projection.bias.data.zero_()
        self.text_projection.weight.data.normal_(mean=0.0, std=0.02)
        self.text_projection.bias.data.zero_()
    
    def forward(self, images, text_prompts):
        """Forward pass for contrastive learning"""
        image_embeds = None
        text_embeds = None
        
        # Process images through base model
        if images is not None:
            with torch.no_grad():
                image_features, _, _ = self.model(images, None)
            
            # Add medical domain projection
            image_embeds = self.vision_projection(image_features)
            # Normalize embeddings
            image_embeds = F.normalize(image_embeds, p=2, dim=1)
        
        # Process text to get embeddings
        if text_prompts is not None:
            with torch.no_grad():
                text_tokens = self.tokenizer(text_prompts, context_length=256).to(images.device if images is not None else next(self.parameters()).device)
                _, text_features, _ = self.model(None, text_tokens)
            
            # Project text features
            text_embeds = self.text_projection(text_features)
            # Normalize embeddings
            text_embeds = F.normalize(text_embeds, p=2, dim=1)
        
        return image_embeds, text_embeds
    
    def encode_images(self, images):
        """Encode images for retrieval or similarity tasks"""
        if images is None:
            return None
            
        with torch.no_grad():
            image_features, _, _ = self.model(images, None)
        
        image_embeds = self.vision_projection(image_features)
        return F.normalize(image_embeds, p=2, dim=1)
    
    def encode_text(self, text_prompts):
        """Encode text for retrieval or similarity tasks"""
        if text_prompts is None:
            return None
            
        device = next(self.parameters()).device
        with torch.no_grad():
            text_tokens = self.tokenizer(text_prompts, context_length=256).to(device)
            _, text_features, _ = self.model(None, text_tokens)
        
        # Project text features
        text_embeds = self.text_projection(text_features)
        
        return F.normalize(text_embeds, p=2, dim=1)
    
    def preprocess_image(self, image):
        """Apply the model's preprocessing to a PIL image"""
        return self.preprocess(image)


class BiomedCLIPAdapter:
    """Adapter class to load and use the fine-tuned BiomedCLIP model"""
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        logger.info(f"Loading model on {self.device}...")
        self.model = self._load_model(model_path)
        logger.info("Model loaded successfully!")
    
    def _load_model(self, model_path):
        """Load the fine-tuned BiomedCLIP model"""
        try:
            # First create a new instance of the model
            model = BiomedCLIP()
            
            # Then load the fine-tuned weights
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"Successfully loaded weights from {model_path}")
            else:
                logger.warning(f"Model file {model_path} not found. Using default weights.")
            
            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, image):
        """Apply preprocessing to an image before encoding"""
        return self.model.preprocess_image(image)
    
    def encode_image(self, image):
        """Encode an image to get embedding"""
        # Apply preprocessing if image is a PIL Image
        from PIL import Image as PILImage
        if isinstance(image, PILImage.Image):
            processed_image = self.preprocess_image(image).unsqueeze(0).to(self.device)
        else:
            # Assume it's already a tensor
            processed_image = image.to(self.device)
        
        with torch.no_grad():
            image_embedding = self.model.encode_images(processed_image)
            return image_embedding.cpu().numpy()
    
    def encode_text(self, text):
        """Encode text to get embedding"""
        if isinstance(text, str):
            text = [text]
            
        with torch.no_grad():
            text_embedding = self.model.encode_text(text)
            return text_embedding.cpu().numpy()


# Testing function to check if model loading works
def test_model_loading(model_path):
    """Test if the model loads correctly"""
    try:
        logger.info("Testing model loading...")
        adapter = BiomedCLIPAdapter(model_path)
        
        # Test with a dummy image
        import numpy as np
        from PIL import Image
        
        # Create a dummy image (black square)
        dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Test image encoding
        image_embedding = adapter.encode_image(dummy_image)
        logger.info(f"Image embedding shape: {image_embedding.shape}")
        
        # Test text encoding
        text_embedding = adapter.encode_text("this is a test")
        logger.info(f"Text embedding shape: {text_embedding.shape}")
        
        logger.info("Model loading test passed!")
        return True
    except Exception as e:
        logger.error(f"Model loading test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test model loading if running this file directly
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "model_checkpoints/best_biomedclip_big.pth"
    
    test_model_loading(model_path)
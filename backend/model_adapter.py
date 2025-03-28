"""
BiomedCLIP model adapter for the Medical X-ray Retrieval System
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
import os
import logging
from PIL import Image as PILImage
import re
from huggingface_hub import hf_hub_download

# Configure logging
logger = logging.getLogger("medical-retrieval.model")

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
            
        logger.info(f"Loading model from {model_path} on {self.device}...")
        
        # Check if it's a path to load or a direct HF model
        if model_path.startswith("hf://"):
            # Extract just the repo ID from the path
            path_parts = model_path.replace("hf://", "").split("/")
            
            if len(path_parts) >= 2:
                repo_id = path_parts[0]
                filename = "/".join(path_parts[1:])
                
                # We need to download the model file first
                try:
                    # Try to download the weights from Hugging Face
                    logger.info(f"Downloading model weights from Hugging Face: {repo_id}, file: {filename}")
                    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
                    logger.info(f"Downloaded model weights to: {local_path}")
                    
                    # Now load the model with the downloaded weights
                    self.model = self._load_custom_model(local_path)
                except Exception as e:
                    logger.error(f"Failed to download model from Hugging Face: {str(e)}")
                    logger.warning("Falling back to default model")
                    self.model = self._create_default_model()
            else:
                logger.error(f"Invalid Hugging Face URL format: {model_path}")
                logger.warning("Falling back to default model")
                self.model = self._create_default_model()
        else:
            # Local path
            self.model = self._load_model(model_path)
            
        logger.info("Model loaded successfully!")
    
    def _create_default_model(self):
        """Create the default BiomedCLIP model"""
        try:
            # Use the Microsoft BiomedCLIP model
            model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            logger.info(f"Creating default model from {model_name}")
            
            model = BiomedCLIP(model_name=model_name)
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error creating default model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _load_custom_model(self, model_path):
        """Load a custom fine-tuned model from a path"""
        try:
            # First create a base model
            model = BiomedCLIP()
            
            # Then load the fine-tuned weights
            logger.info(f"Loading custom model weights from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Check if the state_dict is wrapped (common in some cases)
            if "model" in state_dict:
                state_dict = state_dict["model"]
                
            model.load_state_dict(state_dict)
            
            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            return model
        except Exception as e:
            logger.error(f"Error loading custom model: {str(e)}")
            logger.error("Falling back to default model")
            return self._create_default_model()
    
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
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def preprocess_image(self, image):
        """Apply preprocessing to an image before encoding"""
        return self.model.preprocess_image(image)
    
    def encode_image(self, image):
        """
        Encode an image to get embedding
        
        Args:
            image (PIL.Image or torch.Tensor): The image to encode
            
        Returns:
            numpy.ndarray: The image embedding
        """
        try:
            # Apply preprocessing if image is a PIL Image
            if isinstance(image, PILImage.Image):
                processed_image = self.preprocess_image(image).unsqueeze(0).to(self.device)
            else:
                # Assume it's already a tensor
                processed_image = image.to(self.device)
            
            with torch.no_grad():
                image_embedding = self.model.encode_images(processed_image)
                return image_embedding.cpu().numpy()
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise
    
    def encode_text(self, text):
        """
        Encode text to get embedding
        
        Args:
            text (str or list): The text to encode
            
        Returns:
            numpy.ndarray: The text embedding
        """
        try:
            if isinstance(text, str):
                text = [text]
                
            with torch.no_grad():
                text_embedding = self.model.encode_text(text)
                return text_embedding.cpu().numpy()
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise
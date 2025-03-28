"""
Database operations for the Medical X-ray Retrieval System
"""
import os
import pickle
import faiss
import numpy as np
import logging
from PIL import Image
import time
from .utils import calculate_image_hash, image_to_base64
from config.config import FAISS_INDEX_PATH, METADATA_PATH, IMAGE_DB_PATH

logger = logging.getLogger("medical-retrieval.database")

class VectorDatabase:
    def __init__(self):
        self.index = None
        self.metadata_db = {}
        self.image_db = {}
        self.load_database()
    
    def load_database(self):
        """Load or initialize the vector database"""
        try:
            if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH) and os.path.exists(IMAGE_DB_PATH):
                logger.info("Loading existing database...")
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                
                with open(METADATA_PATH, 'rb') as f:
                    self.metadata_db = pickle.load(f)
                
                with open(IMAGE_DB_PATH, 'rb') as f:
                    self.image_db = pickle.load(f)
                    
                logger.info(f"Database loaded with {self.index.ntotal} images")
            else:
                # Create new FAISS index for 256-dimensional embeddings
                logger.info("Creating new database...")
                self.index = faiss.IndexFlatIP(256)  # Inner product (cosine) similarity for normalized vectors
                logger.info("New database initialized")
        except Exception as e:
            logger.error(f"Error loading database: {str(e)}")
            # Create new index if loading fails
            self.index = faiss.IndexFlatIP(256)
    
    def save_database(self):
        """Save the database to disk"""
        try:
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            
            with open(METADATA_PATH, 'wb') as f:
                pickle.dump(self.metadata_db, f)
            
            with open(IMAGE_DB_PATH, 'wb') as f:
                pickle.dump(self.image_db, f)
                
            logger.info(f"Database saved with {self.index.ntotal} images")
            return True
        except Exception as e:
            logger.error(f"Error saving database: {str(e)}")
            return False
    
    def add_image(self, model, image, metadata=None):
        """
        Add an image to the database
        
        Args:
            model: The BiomedCLIP model adapter
            image (PIL.Image): The image to add
            metadata (dict, optional): Metadata for the image
            
        Returns:
            int: Image ID if successful, -1 on error, -2 if duplicate
        """
        try:
            # Calculate image hash to check for duplicates
            img_hash = calculate_image_hash(image)
            
            # Check if this image hash already exists in the database
            for idx in self.metadata_db:
                if self.metadata_db[idx].get('image_hash') == img_hash:
                    return -2  # Duplicate image
            
            # Get embedding for image
            embedding = model.encode_image(image)[0]
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            # Add to index
            self.index.add(np.array([embedding], dtype=np.float32))
            
            # Current index is ntotal - 1
            current_idx = self.index.ntotal - 1
            
            # Create thumbnail and convert to base64
            image_copy = image.copy()
            image_copy.thumbnail((224, 224))
            img_base64 = image_to_base64(image_copy)
            
            # Add image hash to metadata
            if metadata is None:
                metadata = {}
            metadata['image_hash'] = img_hash
            
            # Store metadata and image
            self.metadata_db[current_idx] = metadata
            self.image_db[current_idx] = img_base64
            
            # Save database
            self.save_database()
            
            return current_idx
        except Exception as e:
            logger.error(f"Error adding image to database: {str(e)}")
            return -1
    
    def delete_image(self, image_id):
        """
        Delete an image from the database
        
        Args:
            image_id (int): ID of the image to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # FAISS doesn't support direct deletion, so we rebuild the index
            all_ids = list(range(self.index.ntotal))
            keep_ids = [idx for idx in all_ids if idx != image_id]
            
            if not keep_ids:  # If this was the only image
                self.index = faiss.IndexFlatIP(256)
                self.metadata_db = {}
                self.image_db = {}
            else:
                # Get embeddings for all IDs to keep
                embeddings = []
                for i in keep_ids:
                    # We have to search for this ID to get its embedding
                    d, i_found = self.index.search(np.array([self.index.reconstruct(i)], dtype=np.float32), 1)
                    embeddings.append(self.index.reconstruct(i))
                
                # Create new index
                new_index = faiss.IndexFlatIP(256)
                new_index.add(np.array(embeddings, dtype=np.float32))
                
                # Create new metadata and image databases
                new_metadata_db = {}
                new_image_db = {}
                
                for new_id, old_id in enumerate(keep_ids):
                    new_metadata_db[new_id] = self.metadata_db[old_id]
                    new_image_db[new_id] = self.image_db[old_id]
                
                # Update database objects
                self.index = new_index
                self.metadata_db = new_metadata_db
                self.image_db = new_image_db
            
            # Save database
            self.save_database()
            
            return True
        except Exception as e:
            logger.error(f"Error deleting image: {str(e)}")
            return False
    
    def clear_database(self):
        """
        Clear the entire database
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.index = faiss.IndexFlatIP(256)
            self.metadata_db = {}
            self.image_db = {}
            self.save_database()
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return False
    
    def get_database_stats(self):
        """
        Get statistics about the database
        
        Returns:
            dict: Database statistics
        """
        stats = {
            "total_images": self.index.ntotal,
            "unique_paths": set(),
            "unique_hashes": set(),
            "findings_count": {}
        }
        
        for idx in self.metadata_db:
            # Track unique files
            if 'file_path' in self.metadata_db[idx]:
                stats["unique_paths"].add(self.metadata_db[idx]['file_path'])
            if 'image_hash' in self.metadata_db[idx]:
                stats["unique_hashes"].add(self.metadata_db[idx]['image_hash'])
            
            # Count findings
            for key, value in self.metadata_db[idx].items():
                if key in ['No Finding', 'Pneumonia', 'Cardiomegaly', 'Pleural Effusion', 
                           'Atelectasis', 'Pneumothorax', 'Edema', 'Consolidation', 
                           'Lung Opacity', 'Lung Lesion', 'Fracture', 'Support Devices',
                           'no_finding', 'pneumonia', 'cardiomegaly', 'pleural_effusion',
                           'atelectasis', 'pneumothorax', 'edema', 'consolidation',
                           'lung_opacity', 'lung_lesion', 'fracture', 'support_devices']:
                    if value == 1.0 or value == 1:
                        proper_name = key.replace('_', ' ').title()
                        stats["findings_count"][proper_name] = stats["findings_count"].get(proper_name, 0) + 1
        
        # Add calculated statistics
        stats["unique_hashes_count"] = len(stats["unique_hashes"])
        stats["potential_duplicates"] = stats["total_images"] - stats["unique_hashes_count"]
        
        return stats
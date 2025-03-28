"""
Search functionality for the Medical X-ray Retrieval System
"""
import numpy as np
import logging

logger = logging.getLogger("medical-retrieval.search")

class SearchEngine:
    def __init__(self, model, database):
        """
        Initialize the search engine
        
        Args:
            model: BiomedCLIP model adapter
            database: Vector database instance
        """
        self.model = model
        self.db = database
    
    def search_by_image(self, query_image, max_results=10):
        """
        Search for similar images in the database
        
        Args:
            query_image (PIL.Image): The query image
            max_results (int): Maximum number of results to return
            
        Returns:
            list: Search results
        """
        try:
            # Get embedding for query image
            embedding = self.model.encode_image(query_image)[0]
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            # Search in index
            k = min(max_results, self.db.index.ntotal) if self.db.index.ntotal > 0 else 0
            if k == 0:
                return []
            
            scores, indices = self.db.index.search(np.array([embedding], dtype=np.float32), k)
            
            # Collect results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.db.image_db):
                    results.append({
                        'id': int(idx),
                        'score': float(scores[0][i]),
                        'image': self.db.image_db.get(int(idx), None),
                        'metadata': self.db.metadata_db.get(int(idx), {})
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error in image search: {str(e)}")
            return []
    
    def search_by_text(self, query_text, max_results=10):
        """
        Search for images matching a text description
        
        Args:
            query_text (str): The text query
            max_results (int): Maximum number of results to return
            
        Returns:
            list: Search results
        """
        try:
            # Prepare query text
            if not query_text.lower().startswith("this is a photo of"):
                query_text = "this is a photo of " + query_text
            
            # Get embedding for query text
            embedding = self.model.encode_text(query_text)[0]
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            # Search in index
            k = min(max_results, self.db.index.ntotal) if self.db.index.ntotal > 0 else 0
            if k == 0:
                return []
            
            scores, indices = self.db.index.search(np.array([embedding], dtype=np.float32), k)
            
            # Collect results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.db.image_db):
                    results.append({
                        'id': int(idx),
                        'score': float(scores[0][i]),
                        'image': self.db.image_db.get(int(idx), None),
                        'metadata': self.db.metadata_db.get(int(idx), {})
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
            return []
            
    def multimodal_search(self, query_image, query_text, max_results=10, weight_image=0.5):
        """
        Search for similar images using both image and text inputs.
        
        Args:
            query_image (PIL.Image): The query image
            query_text (str): The query text
            max_results (int): Maximum number of results to return
            weight_image (float): Weight for image similarity (0.0-1.0), text is 1-weight_image
        
        Returns:
            list: Search results
        """
        try:
            # Get embedding for query image
            image_embedding = self.model.encode_image(query_image)[0]
            
            # Normalize image embedding
            image_embedding = image_embedding / np.linalg.norm(image_embedding)
            
            # Prepare query text
            if not query_text.lower().startswith("this is a photo of"):
                query_text = "this is a photo of " + query_text
            
            # Get embedding for query text
            text_embedding = self.model.encode_text(query_text)[0]
            
            # Normalize text embedding
            text_embedding = text_embedding / np.linalg.norm(text_embedding)
            
            # Combine embeddings with weighted average
            combined_embedding = (weight_image * image_embedding) + ((1 - weight_image) * text_embedding)
            
            # Normalize the combined embedding
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
            
            # Search in index
            k = min(max_results, self.db.index.ntotal) if self.db.index.ntotal > 0 else 0
            if k == 0:
                return []
            
            scores, indices = self.db.index.search(np.array([combined_embedding], dtype=np.float32), k)
            
            # Collect results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.db.image_db):
                    results.append({
                        'id': int(idx),
                        'score': float(scores[0][i]),
                        'image': self.db.image_db.get(int(idx), None),
                        'metadata': self.db.metadata_db.get(int(idx), {})
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error in multimodal search: {str(e)}")
            return []
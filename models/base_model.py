"""
Base Model Class
Abstract base class for all AI vision models in the benchmark
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)

class BaseVisionModel(ABC):
    """Abstract base class for all vision models"""
    
    def __init__(self, model_name: str, model_type: str = "unknown"):
        """
        Initialize the base model
        
        Args:
            model_name: Name of the model (e.g., "GPT-4o", "YOLOv8")
            model_type: Type of model ("commercial" or "opensource")
        """
        self.model_name = model_name
        self.model_type = model_type
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        
    @abstractmethod
    def detect_objects(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Detect objects in an image based on a text prompt
        
        Args:
            image_path: Path to the input image
            prompt: Text description of what to find
            
        Returns:
            Dictionary containing detection results with the following structure:
            {
                'detections': [
                    {
                        'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
                        'confidence': float,        # Confidence score
                        'class': str,              # Object class/label
                        'description': str         # Text description
                    }
                ],
                'response_text': str,              # Raw text response from model
                'processing_time': float,          # Time taken in seconds
                'metadata': {}                     # Additional model-specific data
            }
        """
        pass
        
    @abstractmethod 
    def track_person(self, video_path: str, prompt: str) -> Dict[str, Any]:
        """
        Track a person in a video based on a text description
        
        Args:
            video_path: Path to the input video
            prompt: Text description of the person to track
            
        Returns:
            Dictionary containing tracking results with the following structure:
            {
                'tracks': [
                    {
                        'frame_number': int,
                        'timestamp': float,        # Time in seconds
                        'bbox': [x1, y1, x2, y2], # Bounding box coordinates
                        'confidence': float        # Confidence score
                    }
                ],
                'response_text': str,              # Raw text response from model
                'processing_time': float,          # Time taken in seconds
                'metadata': {}                     # Additional model-specific data
            }
        """
        pass
        
    def _validate_image_path(self, image_path: str) -> Path:
        """Validate that image path exists and is readable"""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")
        return path
        
    def _validate_video_path(self, video_path: str) -> Path:
        """Validate that video path exists and is readable"""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {video_path}")
        return path
        
    def _time_execution(self, func, *args, **kwargs):
        """Time the execution of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        self.logger.info(f"Execution time: {execution_time:.2f} seconds")
        return result, execution_time
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about this model"""
        return {
            'name': self.model_name,
            'type': self.model_type,
            'class': self.__class__.__name__
        }
        
    def health_check(self) -> bool:
        """
        Perform a basic health check on the model
        Returns True if model is ready, False otherwise
        """
        try:
            # Override in subclasses for model-specific health checks
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

class CommercialModel(BaseVisionModel):
    """Base class for commercial API-based models"""
    
    def __init__(self, model_name: str, api_config: Dict[str, Any]):
        """
        Initialize commercial model with API configuration
        
        Args:
            model_name: Name of the model
            api_config: Configuration dictionary containing API keys and settings
        """
        super().__init__(model_name, "commercial")
        self.api_config = api_config
        self._setup_api_client()
        
    @abstractmethod
    def _setup_api_client(self):
        """Setup the API client - implement in subclasses"""
        pass
        
    def _handle_api_error(self, error: Exception, context: str = ""):
        """Handle API errors with logging and retries"""
        error_msg = f"API error in {context}: {error}"
        self.logger.error(error_msg)
        raise Exception(error_msg) from error

class OpenSourceModel(BaseVisionModel):
    """Base class for open-source models"""
    
    def __init__(self, model_name: str):
        """Initialize open-source model"""
        super().__init__(model_name, "opensource")
        self._load_model()
        
    @abstractmethod
    def _load_model(self):
        """Load the model weights and initialize - implement in subclasses"""
        pass
        
    def _preprocess_image(self, image_path: str) -> Any:
        """Preprocess image for the model - override in subclasses"""
        from PIL import Image
        return Image.open(image_path)
        
    def _preprocess_video(self, video_path: str) -> Any:
        """Preprocess video for the model - override in subclasses"""
        import cv2
        return cv2.VideoCapture(video_path) 
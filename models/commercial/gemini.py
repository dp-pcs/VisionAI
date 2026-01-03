"""
Google Gemini Vision Model Implementation
Commercial model using Google's Gemini with vision capabilities
"""

import base64
import time
import os
from typing import Dict, List, Any
from pathlib import Path
import logging

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from ..base_model import CommercialModel

logger = logging.getLogger(__name__)


class GeminiModel(CommercialModel):
    """Google Gemini Vision model implementation"""

    def __init__(self, api_config: Dict[str, Any]):
        """Initialize Gemini model"""
        super().__init__("Gemini", api_config)

    def _setup_api_client(self):
        """Setup Gemini API client"""
        if not genai:
            raise ImportError("Google GenerativeAI library not installed. Install with: pip install google-generativeai")

        # Load API key from environment variable
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set. Please add it to your .env file")

        genai.configure(api_key=api_key)
        self.model_name = self.api_config.get('model', 'gemini-2.0-flash')
        self.model = genai.GenerativeModel(self.model_name)

        self.logger.info(f"Initialized Gemini client with model: {self.model_name}")

    def detect_objects(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Detect objects in image using Gemini Vision

        Args:
            image_path: Path to the input image
            prompt: Text description of what to find

        Returns:
            Dictionary containing detection results
        """
        try:
            # Validate image path
            image_path_obj = self._validate_image_path(image_path)

            # Load image
            image_data = self._load_image(image_path_obj)

            # Create the prompt for object detection
            detection_prompt = self._create_detection_prompt(prompt)

            # Time the API call
            start_time = time.time()

            # Call Gemini API
            response = self.model.generate_content([detection_prompt, image_data])

            processing_time = time.time() - start_time

            # Parse the response
            response_text = response.text
            detections = self._parse_detection_response(response_text)

            result = {
                'detections': detections,
                'response_text': response_text,
                'processing_time': processing_time,
                'metadata': {
                    'model': self.model_name,
                    'finish_reason': response.candidates[0].finish_reason.name if response.candidates else 'unknown'
                }
            }

            self.logger.info(f"Successfully processed image in {processing_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error in detect_objects: {e}")
            return {
                'detections': [],
                'response_text': '',
                'processing_time': 0,
                'metadata': {'error': str(e)}
            }

    def track_person(self, video_path: str, prompt: str) -> Dict[str, Any]:
        """
        Track person in video using Gemini Vision
        Gemini 2.0 supports native video input

        Args:
            video_path: Path to the input video
            prompt: Text description of the person to track

        Returns:
            Dictionary containing tracking results
        """
        try:
            # Validate video path
            video_path_obj = self._validate_video_path(video_path)

            # Check file size - Gemini has limits
            file_size = video_path_obj.stat().st_size
            max_size = 20 * 1024 * 1024  # 20MB limit for inline

            start_time = time.time()

            if file_size > max_size:
                # Use File API for larger videos
                self.logger.info("Video is large, using File API...")
                video_file = genai.upload_file(str(video_path_obj))

                # Wait for processing
                while video_file.state.name == "PROCESSING":
                    time.sleep(2)
                    video_file = genai.get_file(video_file.name)

                if video_file.state.name == "FAILED":
                    raise ValueError("Video processing failed")

                tracking_prompt = self._create_tracking_prompt(prompt)
                response = self.model.generate_content([tracking_prompt, video_file])

                # Clean up
                genai.delete_file(video_file.name)
            else:
                # Upload video directly
                video_data = self._load_video(video_path_obj)
                tracking_prompt = self._create_tracking_prompt(prompt)
                response = self.model.generate_content([tracking_prompt, video_data])

            processing_time = time.time() - start_time

            response_text = response.text
            tracks = self._parse_tracking_response(response_text)

            result = {
                'tracks': tracks,
                'response_text': response_text,
                'processing_time': processing_time,
                'metadata': {
                    'model': self.model_name,
                    'video_size_mb': file_size / (1024 * 1024)
                }
            }

            self.logger.info(f"Successfully processed video in {processing_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error in track_person: {e}")
            return {
                'tracks': [],
                'response_text': '',
                'processing_time': 0,
                'metadata': {'error': str(e)}
            }

    def _load_image(self, image_path: Path):
        """Load image for Gemini API"""
        import PIL.Image
        return PIL.Image.open(image_path)

    def _load_video(self, video_path: Path):
        """Load video for Gemini API"""
        with open(video_path, 'rb') as f:
            video_bytes = f.read()

        # Determine mime type
        suffix = video_path.suffix.lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/avi',
            '.mov': 'video/quicktime',
            '.webm': 'video/webm'
        }
        mime_type = mime_types.get(suffix, 'video/mp4')

        return {'mime_type': mime_type, 'data': video_bytes}

    def _create_detection_prompt(self, user_prompt: str) -> str:
        """Create a detailed prompt for object detection"""
        return f"""You are an expert computer vision analyst. Analyze this image and {user_prompt}

Please provide:
1. A detailed description of what you see
2. Specific locations or regions where the requested items are found
3. Any text you can read from the image
4. Bounding box coordinates if possible (format: x1,y1,x2,y2 as percentages 0-100)

Be specific and thorough in your analysis."""

    def _create_tracking_prompt(self, user_prompt: str) -> str:
        """Create a detailed prompt for person tracking"""
        return f"""You are analyzing a video for person tracking. {user_prompt}

Please:
1. Identify if the target person is visible in the video
2. Describe when they appear and disappear (timestamps)
3. Track their movement through the video
4. Note any distinguishing features or clothing details
5. Provide approximate bounding box locations at key moments (format: timestamp, x1,y1,x2,y2)
6. Assess confidence level (high/medium/low)

Provide a detailed timeline of the person's appearance in the video."""

    def _parse_detection_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse Gemini response to extract detection information"""
        detections = []
        import re

        # Look for bounding box coordinates
        bbox_pattern = r'(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)'
        matches = re.findall(bbox_pattern, response_text)

        for match in matches:
            coords = [int(c) for c in match]
            if all(0 <= c <= 100 for c in coords):  # Percentage coordinates
                detections.append({
                    'bbox': coords,
                    'confidence': 0.8,
                    'class': 'detected_object'
                })

        # If no structured detections, create a general one
        if not detections and response_text:
            detections.append({
                'bbox': [0, 0, 100, 100],
                'confidence': 0.8,
                'class': 'detected_object',
                'description': response_text[:500]
            })

        return detections

    def _parse_tracking_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse Gemini response to extract tracking information"""
        tracks = []
        import re

        # Look for timestamp patterns (e.g., "0:05", "1:30", "00:15")
        timestamp_pattern = r'(\d+):(\d+)'
        matches = re.findall(timestamp_pattern, response_text)

        for i, match in enumerate(matches):
            minutes, seconds = int(match[0]), int(match[1])
            timestamp = minutes * 60 + seconds

            # Estimate confidence based on keywords near this timestamp
            confidence = 0.8
            if 'clearly' in response_text.lower() or 'visible' in response_text.lower():
                confidence = 0.9
            elif 'partially' in response_text.lower() or 'obscured' in response_text.lower():
                confidence = 0.6

            tracks.append({
                'frame_number': i,
                'timestamp': timestamp,
                'bbox': [0, 0, 100, 100],
                'confidence': confidence
            })

        return tracks

    def health_check(self) -> bool:
        """Perform health check on Gemini API"""
        try:
            response = self.model.generate_content("Hello")
            return response.text is not None
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

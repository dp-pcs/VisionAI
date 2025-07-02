"""
OpenAI GPT-4o Vision Model Implementation
Commercial model using OpenAI's GPT-4o with vision capabilities
"""

import base64
import time
from typing import Dict, List, Any
from pathlib import Path
import logging

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

from ..base_model import CommercialModel

logger = logging.getLogger(__name__)

class GPT4OModel(CommercialModel):
    """OpenAI GPT-4o Vision model implementation"""
    
    def __init__(self, api_config: Dict[str, Any]):
        """Initialize GPT-4o model"""
        super().__init__("GPT-4o", api_config)
        
    def _setup_api_client(self):
        """Setup OpenAI API client"""
        if not openai:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
            
        api_key = self.api_config.get('api_key')
        if not api_key or api_key == "your-openai-api-key-here":
            raise ValueError("OpenAI API key not configured. Please update config.yaml")
            
        self.client = OpenAI(api_key=api_key)
        self.model = self.api_config.get('model', 'gpt-4o')
        self.max_tokens = self.api_config.get('max_tokens', 4096)
        
        self.logger.info(f"Initialized GPT-4o client with model: {self.model}")
        
    def detect_objects(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Detect objects in image using GPT-4o Vision
        
        Args:
            image_path: Path to the input image
            prompt: Text description of what to find
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Validate image path
            image_path_obj = self._validate_image_path(image_path)
            
            # Encode image to base64
            image_base64 = self._encode_image_base64(image_path_obj)
            
            # Create the prompt for object detection
            detection_prompt = self._create_detection_prompt(prompt)
            
            # Time the API call
            start_time = time.time()
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": detection_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens
            )
            
            processing_time = time.time() - start_time
            
            # Parse the response
            response_text = response.choices[0].message.content
            detections = self._parse_detection_response(response_text)
            
            result = {
                'detections': detections,
                'response_text': response_text,
                'processing_time': processing_time,
                'metadata': {
                    'model': self.model,
                    'usage': response.usage.dict() if response.usage else {},
                    'finish_reason': response.choices[0].finish_reason
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
        Track person in video using GPT-4o Vision
        Note: GPT-4o currently doesn't support direct video processing,
        so we'll extract frames and analyze them
        
        Args:
            video_path: Path to the input video
            prompt: Text description of the person to track
            
        Returns:
            Dictionary containing tracking results
        """
        try:
            # Validate video path
            video_path_obj = self._validate_video_path(video_path)
            
            # Extract frames from video
            frames = self._extract_video_frames(video_path_obj, max_frames=10)
            
            if not frames:
                return {
                    'tracks': [],
                    'response_text': 'No frames could be extracted from video',
                    'processing_time': 0,
                    'metadata': {'error': 'Frame extraction failed'}
                }
            
            # Create the prompt for person tracking
            tracking_prompt = self._create_tracking_prompt(prompt)
            
            start_time = time.time()
            tracks = []
            response_texts = []
            
            # Analyze each frame
            for frame_idx, (frame_data, timestamp) in enumerate(frames):
                try:
                    frame_base64 = self._encode_frame_base64(frame_data)
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"{tracking_prompt}\n\nFrame {frame_idx + 1} at {timestamp:.2f}s:"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{frame_base64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=self.max_tokens
                    )
                    
                    response_text = response.choices[0].message.content
                    response_texts.append(response_text)
                    
                    # Parse tracking information from response
                    track_info = self._parse_tracking_response(response_text, frame_idx, timestamp)
                    if track_info:
                        tracks.extend(track_info)
                        
                except Exception as e:
                    self.logger.warning(f"Error processing frame {frame_idx}: {e}")
                    continue
                    
            processing_time = time.time() - start_time
            
            result = {
                'tracks': tracks,
                'response_text': '\n'.join(response_texts),
                'processing_time': processing_time,
                'metadata': {
                    'model': self.model,
                    'frames_processed': len(frames),
                    'tracks_found': len(tracks)
                }
            }
            
            self.logger.info(f"Successfully processed video with {len(frames)} frames in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in track_person: {e}")
            return {
                'tracks': [],
                'response_text': '',
                'processing_time': 0,
                'metadata': {'error': str(e)}
            }
            
    def _encode_image_base64(self, image_path: Path) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    def _encode_frame_base64(self, frame_data: bytes) -> str:
        """Encode frame data to base64 string"""
        return base64.b64encode(frame_data).decode('utf-8')
        
    def _create_detection_prompt(self, user_prompt: str) -> str:
        """Create a detailed prompt for object detection"""
        return f"""
You are an expert computer vision analyst. Analyze this image and {user_prompt}

Please provide:
1. A detailed description of what you see
2. Specific locations or regions where the requested items are found
3. Any text you can read from the image
4. Bounding box coordinates if possible (format: x1,y1,x2,y2)

Be specific and thorough in your analysis.
"""
        
    def _create_tracking_prompt(self, user_prompt: str) -> str:
        """Create a detailed prompt for person tracking"""
        return f"""
You are analyzing a video frame for person tracking. {user_prompt}

Please:
1. Identify if the target person is visible in this frame
2. Describe their location and pose
3. Provide bounding box coordinates if possible (format: x1,y1,x2,y2)
4. Note any distinguishing features or clothing details
5. Assess confidence level (high/medium/low)

Focus on the specific person described in the request.
"""
        
    def _parse_detection_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse GPT-4o response to extract detection information"""
        detections = []
        
        # This is a simplified parser - in practice, you'd want more sophisticated parsing
        lines = response_text.split('\n')
        
        current_detection = {}
        for line in lines:
            line = line.strip()
            
            # Look for bounding box coordinates
            if 'box' in line.lower() or 'coordinate' in line.lower():
                # Try to extract coordinates (x1,y1,x2,y2)
                import re
                coords = re.findall(r'\d+', line)
                if len(coords) >= 4:
                    bbox = [int(c) for c in coords[:4]]
                    current_detection['bbox'] = bbox
                    
            # Look for confidence or description
            if current_detection and ('confidence' in line.lower() or 'description' in line.lower()):
                current_detection['description'] = line
                detections.append(current_detection)
                current_detection = {}
                
        # If we don't have structured detections, create a general one
        if not detections and response_text:
            detections.append({
                'bbox': [0, 0, 100, 100],  # Placeholder
                'confidence': 0.8,
                'class': 'detected_object',
                'description': response_text[:200]  # First 200 chars
            })
            
        return detections
        
    def _parse_tracking_response(self, response_text: str, frame_number: int, timestamp: float) -> List[Dict[str, Any]]:
        """Parse GPT-4o response to extract tracking information"""
        tracks = []
        
        # Simple parsing - look for indications that person was found
        if any(keyword in response_text.lower() for keyword in ['visible', 'found', 'see', 'located']):
            # Try to extract bounding box
            import re
            coords = re.findall(r'\d+', response_text)
            
            bbox = [0, 0, 100, 100]  # Default
            if len(coords) >= 4:
                bbox = [int(c) for c in coords[:4]]
                
            # Estimate confidence based on keywords
            confidence = 0.8
            if 'high' in response_text.lower():
                confidence = 0.9
            elif 'low' in response_text.lower():
                confidence = 0.6
                
            tracks.append({
                'frame_number': frame_number,
                'timestamp': timestamp,
                'bbox': bbox,
                'confidence': confidence
            })
            
        return tracks
        
    def _extract_video_frames(self, video_path: Path, max_frames: int = 10) -> List[tuple]:
        """Extract frames from video for analysis"""
        try:
            import cv2
            
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return []
                
            # Select frames to sample
            frame_indices = []
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                step = total_frames // max_frames
                frame_indices = list(range(0, total_frames, step))[:max_frames]
                
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert frame to JPEG bytes
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    timestamp = frame_idx / fps if fps > 0 else 0
                    frames.append((frame_bytes, timestamp))
                    
            cap.release()
            return frames
            
        except Exception as e:
            self.logger.error(f"Error extracting video frames: {e}")
            return []
            
    def health_check(self) -> bool:
        """Perform health check on GPT-4o API"""
        try:
            # Simple test call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return response.choices[0].message.content is not None
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False 
"""
Anthropic Claude Vision Model Implementation
Commercial model using Anthropic's Claude with vision capabilities
"""

import base64
import time
import os
from typing import Dict, List, Any
from pathlib import Path
import logging

try:
    import anthropic
except ImportError:
    anthropic = None

from ..base_model import CommercialModel

logger = logging.getLogger(__name__)


class ClaudeModel(CommercialModel):
    """Anthropic Claude Vision model implementation"""

    def __init__(self, api_config: Dict[str, Any]):
        """Initialize Claude model"""
        super().__init__("Claude", api_config)

    def _setup_api_client(self):
        """Setup Anthropic API client"""
        if not anthropic:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")

        # Load API key from environment variable
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set. Please add it to your .env file")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = self.api_config.get('model', 'claude-sonnet-4-20250514')
        self.max_tokens = self.api_config.get('max_tokens', 4096)

        self.logger.info(f"Initialized Claude client with model: {self.model}")

    def detect_objects(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """
        Detect objects in image using Claude Vision

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
            image_base64, media_type = self._encode_image_base64(image_path_obj)

            # Create the prompt for object detection
            detection_prompt = self._create_detection_prompt(prompt)

            # Time the API call
            start_time = time.time()

            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": detection_prompt
                            }
                        ]
                    }
                ]
            )

            processing_time = time.time() - start_time

            # Parse the response
            response_text = response.content[0].text
            detections = self._parse_detection_response(response_text)

            result = {
                'detections': detections,
                'response_text': response_text,
                'processing_time': processing_time,
                'metadata': {
                    'model': self.model,
                    'usage': {
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens': response.usage.output_tokens
                    },
                    'stop_reason': response.stop_reason
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
        Track person in video using Claude Vision
        Note: Claude doesn't support direct video processing,
        so we extract frames and analyze them

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
                    frame_base64 = base64.b64encode(frame_data).decode('utf-8')

                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": frame_base64
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": f"{tracking_prompt}\n\nFrame {frame_idx + 1} at {timestamp:.2f}s:"
                                    }
                                ]
                            }
                        ]
                    )

                    response_text = response.content[0].text
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

    def _encode_image_base64(self, image_path: Path) -> tuple:
        """Encode image to base64 string and return with media type"""
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Determine media type
        suffix = image_path.suffix.lower()
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_types.get(suffix, 'image/jpeg')

        return image_data, media_type

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
        return f"""You are analyzing a video frame for person tracking. {user_prompt}

Please:
1. Identify if the target person is visible in this frame
2. Describe their location and pose
3. Provide bounding box coordinates if possible (format: x1,y1,x2,y2 as percentages 0-100)
4. Note any distinguishing features or clothing details
5. Assess confidence level (high/medium/low)

Focus on the specific person described in the request."""

    def _parse_detection_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse Claude response to extract detection information"""
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
                    'confidence': 0.85,
                    'class': 'detected_object'
                })

        # If no structured detections, create a general one
        if not detections and response_text:
            detections.append({
                'bbox': [0, 0, 100, 100],
                'confidence': 0.85,
                'class': 'detected_object',
                'description': response_text[:500]
            })

        return detections

    def _parse_tracking_response(self, response_text: str, frame_number: int, timestamp: float) -> List[Dict[str, Any]]:
        """Parse Claude response to extract tracking information"""
        tracks = []

        # Simple parsing - look for indications that person was found
        if any(keyword in response_text.lower() for keyword in ['visible', 'found', 'see', 'located', 'present']):
            import re
            coords = re.findall(r'\d+', response_text)

            bbox = [0, 0, 100, 100]  # Default
            if len(coords) >= 4:
                potential_bbox = [int(c) for c in coords[:4]]
                if all(0 <= c <= 100 for c in potential_bbox):
                    bbox = potential_bbox

            # Estimate confidence based on keywords
            confidence = 0.85
            if 'clearly' in response_text.lower() or 'high' in response_text.lower():
                confidence = 0.95
            elif 'partially' in response_text.lower() or 'low' in response_text.lower():
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
        """Perform health check on Claude API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return response.content[0].text is not None
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

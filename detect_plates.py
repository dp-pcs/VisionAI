#!/usr/bin/env python3
"""
License Plate Detection Script
Uses YOLOv8 to detect vehicles and license plates in video
"""

import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO
import os
import base64


def detect_plates(video_path: str, output_dir: str = "results", save_frames: bool = True):
    """
    Detect license plates in a video using YOLOv8

    Args:
        video_path: Path to input video
        output_dir: Directory to save results
        save_frames: Whether to save frames with detections
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load YOLOv8 model - use the largest model for best accuracy
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8x.pt")  # Downloads automatically if not present

    # Classes we care about for license plate detection
    # YOLO COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
    vehicle_classes = [2, 3, 5, 7]

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {video_path.name}")
    print(f"  FPS: {fps}, Total frames: {total_frames}")

    # For output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video_path = output_path / f"{video_path.stem}_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(out_video_path), fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    detections_log = []

    print("\nProcessing frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO detection
        results = model(frame, verbose=False)

        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Check if it's a vehicle
                if cls in vehicle_classes and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = model.names[cls]

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Log detection
                    timestamp = frame_count / fps
                    detection = {
                        'frame': frame_count,
                        'timestamp': f"{timestamp:.2f}s",
                        'class': class_name,
                        'confidence': f"{conf:.2f}",
                        'bbox': [x1, y1, x2, y2]
                    }
                    detections_log.append(detection)

                    # Extract vehicle region for closer inspection
                    if save_frames and conf > 0.7:
                        vehicle_crop = frame[y1:y2, x1:x2]
                        crop_path = output_path / f"vehicle_frame{frame_count}_{class_name}.jpg"
                        cv2.imwrite(str(crop_path), vehicle_crop)

        # Write annotated frame to output video
        out_video.write(frame)

        # Progress update
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out_video.release()

    print(f"\nDone! Processed {frame_count} frames")
    print(f"Found {len(detections_log)} vehicle detections")
    print(f"\nOutput saved to:")
    print(f"  Video: {out_video_path}")
    print(f"  Crops: {output_path}/vehicle_*.jpg")

    # Print summary of unique detections
    if detections_log:
        print("\nSample detections:")
        for det in detections_log[:10]:
            print(f"  {det['timestamp']} - {det['class']} ({det['confidence']})")


def detect_plates_with_ocr(video_path: str, output_dir: str = "results",
                           frame_interval: int = 5, model_choice: str = "claude",
                           show_preview: bool = False, detailed: bool = False):
    """
    Detect license plates and attempt OCR using vision LLMs

    Args:
        video_path: Path to video
        output_dir: Output directory
        frame_interval: Check every N frames (lower = more thorough)
        model_choice: Which LLM to use (claude, gpt4o, gemini)
        show_preview: Show live preview window
        detailed: Get full vehicle make/model/plate analysis
    """
    from dotenv import load_dotenv
    load_dotenv()

    video_path = Path(video_path)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return

    # Load YOLO
    print("Step 1: Detecting vehicles with YOLO...")
    model = YOLO("yolov8x.pt")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {fps:.1f} FPS, {total_frames} frames, {total_frames/fps:.1f}s duration")

    if show_preview:
        # Calculate preview window size (scale down if too large)
        max_width = 1200
        scale = min(1.0, max_width / frame_width)
        preview_width = int(frame_width * scale)
        preview_height = int(frame_height * scale)
        cv2.namedWindow('License Plate Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('License Plate Detection', preview_width, preview_height)
        print("Press 'q' to quit, 'p' to pause/resume")

    vehicle_frames = []
    full_frames_for_ocr = []  # Also keep full frames
    tracked_vehicles = []  # Track unique vehicles with their last seen position/time
    frame_count = 0
    paused = False

    def is_same_vehicle(bbox1, bbox2, time1, time2, iou_threshold=0.3, time_threshold=2.0):
        """Check if two detections are likely the same vehicle"""
        # If too much time has passed, treat as different vehicle
        if abs(time2 - time1) > time_threshold:
            return False

        # Calculate IoU (Intersection over Union)
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return False  # No intersection

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        iou = intersection / union if union > 0 else 0
        return iou > iou_threshold

    def find_or_create_vehicle(bbox, timestamp, class_name):
        """Find existing tracked vehicle or create new one"""
        for i, tv in enumerate(tracked_vehicles):
            if is_same_vehicle(tv['last_bbox'], bbox, tv['last_seen'], timestamp):
                # Update existing vehicle
                tracked_vehicles[i]['last_bbox'] = bbox
                tracked_vehicles[i]['last_seen'] = timestamp
                tracked_vehicles[i]['appearances'] += 1
                return tv['id'], False  # Return ID and flag indicating not new

        # Create new vehicle
        new_id = len(tracked_vehicles) + 1
        tracked_vehicles.append({
            'id': new_id,
            'first_seen': timestamp,
            'last_seen': timestamp,
            'last_bbox': bbox,
            'class': class_name,
            'appearances': 1
        })
        return new_id, True  # Return ID and flag indicating new vehicle

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp = frame_count / fps
            display_frame = frame.copy()

            # Run YOLO on every frame for smooth preview, but only save at interval
            results = model(frame, verbose=False)

            frame_has_vehicle = False

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Vehicle classes: car, motorcycle, bus, truck
                    if cls in [2, 3, 5, 7] and conf > 0.4:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_name = model.names[cls]
                        bbox = [x1, y1, x2, y2]

                        # Check if this is a new or existing vehicle
                        vehicle_id, is_new = find_or_create_vehicle(bbox, timestamp, class_name)

                        # Draw bounding box on display frame
                        color = (0, 255, 0) if is_new else (0, 200, 255)  # Green for new, orange for tracked
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"V{vehicle_id} {class_name}: {conf:.2f}"
                        cv2.putText(display_frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # Only save if this is a NEW vehicle (first appearance)
                        # OR if we're at an interval and haven't saved this vehicle yet
                        if is_new and frame_count % frame_interval == 0:
                            frame_has_vehicle = True

                            # Expand bounding box to capture plate below vehicle
                            height = y2 - y1
                            width = x2 - x1
                            y2_expanded = min(frame.shape[0], y2 + int(height * 0.1))
                            x1_expanded = max(0, x1 - int(width * 0.05))
                            x2_expanded = min(frame.shape[1], x2 + int(width * 0.05))

                            vehicle_crop = frame[y1:y2_expanded, x1_expanded:x2_expanded]

                            if vehicle_crop.size > 0:
                                vehicle_frames.append({
                                    'frame': frame_count,
                                    'timestamp': timestamp,
                                    'image': vehicle_crop.copy(),
                                    'class': class_name,
                                    'confidence': conf,
                                    'bbox': bbox,
                                    'vehicle_id': vehicle_id
                                })

            # Also save full frames with vehicles for context
            if frame_has_vehicle:
                full_frames_for_ocr.append({
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'image': frame.copy()
                })

            # Add status overlay
            status_text = f"Frame: {frame_count}/{total_frames} | Time: {timestamp:.1f}s | Unique Vehicles: {len(tracked_vehicles)}"
            cv2.putText(display_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

            if show_preview:
                cv2.imshow('License Plate Detection', display_frame)

        # Handle keyboard input
        if show_preview:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord(' '):  # Space also pauses
                paused = not paused
        else:
            # Print progress without preview
            if frame_count % 30 == 0:
                print(f"  Processing frame {frame_count}/{total_frames}...", end='\r')

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()
    print(f"\nTracking summary: {len(tracked_vehicles)} unique vehicles detected")
    print(f"Saved {len(vehicle_frames)} vehicle crops for analysis")

    if not vehicle_frames and not full_frames_for_ocr:
        print("No vehicles detected! Trying full frame analysis...")
        # Fall back to analyzing raw frames
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        while cap.isOpened() and len(full_frames_for_ocr) < 20:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 15 == 0:  # Every 15 frames
                full_frames_for_ocr.append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'image': frame.copy()
                })
        cap.release()

    # Step 2: Use LLM to analyze vehicles
    mode_desc = "make/model/plates" if detailed else "license plates"
    print(f"\nStep 2: Analyzing {mode_desc} with {model_choice.upper()}...")

    vehicles_found = []
    plates_found = []

    # Analyze vehicle crops (more focused for make/model)
    print(f"\nAnalyzing {min(len(vehicle_frames), 25)} vehicle crops...")
    for i, vf in enumerate(vehicle_frames[:25]):
        # Skip if image is too small
        if vf['image'].shape[0] < 50 or vf['image'].shape[1] < 50:
            continue

        if detailed:
            # Full vehicle analysis
            result = analyze_vehicle_full(vf['image'], model_choice, is_crop=True, plates_only=False)
            summary = format_vehicle_summary(result, vf['timestamp'])
            print(f"  {summary}")

            vehicles_found.append({
                'frame': vf['frame'],
                'timestamp': vf['timestamp'],
                'make': result.get('make', 'UNKNOWN'),
                'model': result.get('model', 'UNKNOWN'),
                'year': result.get('year', 'UNKNOWN'),
                'color': result.get('color', 'UNKNOWN'),
                'type': result.get('type', 'UNKNOWN'),
                'plate': result.get('plate', 'NO_PLATE'),
                'plate_state': result.get('plate_state', 'UNKNOWN'),
                'confidence': result.get('confidence', 'LOW'),
                'notes': result.get('notes', ''),
                'source': 'vehicle_crop'
            })

            # Save crop
            crop_path = output_path / f"vehicle_{vf['frame']}_{result.get('make', 'unknown')}.jpg"
            cv2.imwrite(str(crop_path), vf['image'])
        else:
            # Plate-only analysis (original behavior)
            result = analyze_frame_for_plates(vf['image'], model_choice, is_crop=True)
            print(f"  Frame {vf['frame']} ({vf['timestamp']:.1f}s) [{vf['class']}]: {result}")

            if result and "NO" not in result.upper()[:10]:
                plates_found.append({
                    'frame': vf['frame'],
                    'timestamp': vf['timestamp'],
                    'result': result,
                    'source': 'vehicle_crop'
                })
                crop_path = output_path / f"vehicle_plate_frame{vf['frame']}.jpg"
                cv2.imwrite(str(crop_path), vf['image'])

    # Also check full frames for context (especially good for plates in background)
    if not detailed:
        print(f"\nAnalyzing {min(len(full_frames_for_ocr), 10)} full frames...")
        for i, ff in enumerate(full_frames_for_ocr[:10]):
            result = analyze_frame_for_plates(ff['image'], model_choice)
            print(f"  Frame {ff['frame']} ({ff['timestamp']:.1f}s): {result}")

            if result and "NO" not in result.upper()[:10]:
                plates_found.append({
                    'frame': ff['frame'],
                    'timestamp': ff['timestamp'],
                    'result': result,
                    'source': 'full_frame'
                })
                crop_path = output_path / f"plate_frame{ff['frame']}.jpg"
                cv2.imwrite(str(crop_path), ff['image'])

    # Print results
    print(f"\n{'='*70}")

    if detailed:
        print(f"VEHICLE REPORT: Found {len(vehicles_found)} vehicles")
        print('='*70)
        print(f"{'Time':<8} {'Make/Model':<30} {'Color':<12} {'Plate':<15}")
        print('-'*70)

        for v in vehicles_found:
            time_str = f"{v['timestamp']:.1f}s"
            make_model = f"{v['make']} {v['model']}"[:28]
            color = v['color'][:10] if v['color'] else ''
            plate = v['plate'][:13] if v['plate'] else 'NO_PLATE'
            print(f"{time_str:<8} {make_model:<30} {color:<12} {plate:<15}")

        # Save report to file
        report_path = output_path / "vehicle_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Vehicle Detection Report\n")
            f.write(f"Video: {video_path.name}\n")
            f.write(f"Model: {model_choice}\n")
            f.write(f"{'='*70}\n\n")

            for v in vehicles_found:
                f.write(f"Timestamp: {v['timestamp']:.1f}s (Frame {v['frame']})\n")
                f.write(f"  Make: {v['make']}\n")
                f.write(f"  Model: {v['model']}\n")
                f.write(f"  Year: {v['year']}\n")
                f.write(f"  Color: {v['color']}\n")
                f.write(f"  Type: {v['type']}\n")
                f.write(f"  Plate: {v['plate']}\n")
                f.write(f"  State: {v['plate_state']}\n")
                f.write(f"  Confidence: {v['confidence']}\n")
                if v['notes']:
                    f.write(f"  Notes: {v['notes']}\n")
                f.write("\n")

        print(f"\nReport saved to: {report_path}")

        # Also save as CSV
        csv_path = output_path / "vehicle_report.csv"
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'frame', 'make', 'model', 'year', 'color',
                'type', 'plate', 'plate_state', 'confidence', 'notes'
            ])
            writer.writeheader()
            for v in vehicles_found:
                writer.writerow({
                    'timestamp': v['timestamp'],
                    'frame': v['frame'],
                    'make': v['make'],
                    'model': v['model'],
                    'year': v['year'],
                    'color': v['color'],
                    'type': v['type'],
                    'plate': v['plate'],
                    'plate_state': v['plate_state'],
                    'confidence': v['confidence'],
                    'notes': v['notes']
                })
        print(f"CSV saved to: {csv_path}")

    else:
        print(f"RESULTS: Found {len(plates_found)} potential plates")
        print('='*70)
        for p in plates_found:
            print(f"  {p['timestamp']:.1f}s ({p['source']}): {p['result']}")

        if not plates_found:
            print("\nNo plates detected. Tips:")
            print("  - Try with --detailed for make/model identification")
            print("  - Try with --interval 2 for more frames")
            print("  - Check results/ folder for saved frames")


def analyze_frame_for_plates(image, model_choice: str = "claude", is_crop: bool = False):
    """Send image to LLM for plate detection - returns raw response"""
    result = analyze_vehicle_full(image, model_choice, is_crop, plates_only=True)
    return result.get('raw_response', 'NO_PLATE')


def analyze_vehicle_full(image, model_choice: str = "claude", is_crop: bool = False, plates_only: bool = False):
    """
    Analyze vehicle for make/model and license plate
    Returns structured data about the vehicle
    """
    # Encode image
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    if plates_only:
        # Original plate-only prompt for backward compatibility
        if is_crop:
            prompt = """This is a cropped image of a vehicle. Look carefully for a license plate.
If you can see ANY part of a license plate, tell me:
1. What characters you can read (even partial)
2. The plate format/style if recognizable (US state, European, etc.)

Format your response as: PLATE: [characters] or PARTIAL: [what you can see] or NO_PLATE"""
        else:
            prompt = """Look at this image carefully. Find any license plates on vehicles.
For EACH plate you can see, tell me:
1. The plate text (even if partial)
2. Which vehicle it's on

Format: PLATE: [text] - [vehicle description]
Or if no plates visible: NO_PLATE

Be thorough - check all vehicles including those in background."""
    else:
        # Full vehicle analysis prompt
        prompt = """Analyze this vehicle image and provide the following information in this EXACT format:

MAKE: [manufacturer name or UNKNOWN]
MODEL: [model name or UNKNOWN]
YEAR: [approximate year/range or UNKNOWN]
COLOR: [primary color]
TYPE: [sedan/suv/truck/van/motorcycle/other]
PLATE: [plate text if visible, or UNREADABLE if plate exists but can't be read, or NO_PLATE if no plate visible]
PLATE_STATE: [state/region if identifiable, or UNKNOWN]
CONFIDENCE: [HIGH/MEDIUM/LOW for make/model identification]
NOTES: [any other relevant details like damage, distinctive features, partial plate info]

Be as specific as possible with make and model. If you can narrow it down to a few possibilities, list them.
For plates, include any characters you can partially read in NOTES even if the full plate is UNREADABLE."""

    try:
        if model_choice == "claude":
            raw = _call_claude(image_base64, prompt)
        elif model_choice == "gpt4o":
            raw = _call_gpt4o(image_base64, prompt)
        elif model_choice == "gemini":
            raw = _call_gemini(image_base64, prompt)
        else:
            raw = _call_claude(image_base64, prompt)

        if plates_only:
            return {'raw_response': raw}

        # Parse the structured response
        return parse_vehicle_response(raw)

    except Exception as e:
        return {
            'raw_response': f"ERROR: {e}",
            'make': 'ERROR',
            'model': 'ERROR',
            'plate': 'ERROR',
            'error': str(e)
        }


def parse_vehicle_response(response: str) -> dict:
    """Parse the structured vehicle analysis response"""
    result = {
        'raw_response': response,
        'make': 'UNKNOWN',
        'model': 'UNKNOWN',
        'year': 'UNKNOWN',
        'color': 'UNKNOWN',
        'type': 'UNKNOWN',
        'plate': 'NO_PLATE',
        'plate_state': 'UNKNOWN',
        'confidence': 'LOW',
        'notes': ''
    }

    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().upper()
            value = value.strip()

            if key == 'MAKE':
                result['make'] = value
            elif key == 'MODEL':
                result['model'] = value
            elif key == 'YEAR':
                result['year'] = value
            elif key == 'COLOR':
                result['color'] = value
            elif key == 'TYPE':
                result['type'] = value
            elif key == 'PLATE':
                result['plate'] = value
            elif key == 'PLATE_STATE':
                result['plate_state'] = value
            elif key == 'CONFIDENCE':
                result['confidence'] = value
            elif key == 'NOTES':
                result['notes'] = value

    return result


def format_vehicle_summary(vehicle_info: dict, timestamp: float) -> str:
    """Format vehicle info into a readable summary line"""
    make = vehicle_info.get('make', 'UNKNOWN')
    model = vehicle_info.get('model', 'UNKNOWN')
    color = vehicle_info.get('color', '')
    plate = vehicle_info.get('plate', 'NO_PLATE')
    year = vehicle_info.get('year', '')

    # Build description
    parts = []
    if color and color != 'UNKNOWN':
        parts.append(color)
    if year and year != 'UNKNOWN':
        parts.append(year)
    if make and make != 'UNKNOWN':
        parts.append(make)
    if model and model != 'UNKNOWN':
        parts.append(model)

    vehicle_desc = ' '.join(parts) if parts else 'Unknown Vehicle'

    # Format plate info
    if plate in ['NO_PLATE', 'UNREADABLE', 'UNKNOWN']:
        plate_desc = f"plate {plate.lower().replace('_', ' ')}"
    else:
        plate_desc = f"plate: {plate}"

    return f"{timestamp:.1f}s - {vehicle_desc} - {plate_desc}"


def _call_claude(image_base64: str, prompt: str) -> str:
    import anthropic
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    return response.content[0].text


def _call_gpt4o(image_base64: str, prompt: str) -> str:
    from openai import OpenAI
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    return response.choices[0].message.content


def _call_gemini(image_base64: str, prompt: str) -> str:
    import google.generativeai as genai
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    import PIL.Image
    import io
    image_bytes = base64.b64decode(image_base64)
    image = PIL.Image.open(io.BytesIO(image_bytes))

    response = model.generate_content([prompt, image])
    return response.text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect license plates in video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", "-o", default="results", help="Output directory")
    parser.add_argument("--ocr", action="store_true", help="Use LLM Vision for OCR")
    parser.add_argument("--model", "-m", choices=["claude", "gpt4o", "gemini"],
                        default="claude", help="Which LLM to use (default: claude)")
    parser.add_argument("--interval", "-i", type=int, default=5,
                        help="Check every N frames (lower=more thorough, default: 5)")
    parser.add_argument("--no-save", action="store_true", help="Don't save cropped frames")
    parser.add_argument("--watch", "-w", action="store_true",
                        help="Show live preview window while processing")
    parser.add_argument("--detailed", "-d", action="store_true",
                        help="Get full vehicle analysis: make, model, color, plate")

    args = parser.parse_args()

    if args.ocr:
        detect_plates_with_ocr(args.video, args.output, args.interval, args.model,
                              args.watch, args.detailed)
    else:
        detect_plates(args.video, args.output, save_frames=not args.no_save)

# predict.py: YOLO Prediction Module for Flask Integration
import os
import glob
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from ultralytics import YOLO

def find_latest_model(upload_folder='uploads'):
    """
    Returns the fixed model path 'runs/detect/fault/weights/best.pt' (modified to use fixed path instead of searching).
    """
    # Hardcode the fixed model path (ignoring dynamic search and upload_folder for the path)
    fixed_path = 'uploads/runs/detect/fault/weights/best.pt'
    if os.path.exists(fixed_path):
        return fixed_path
    else:
        return None

def run_prediction(image_path, upload_folder='uploads'):
    """
    Run YOLO prediction on image_path.
    Returns dict: {'original': base64_str, 'detected': base64_str, 'predictions': list, 'detections': list}
    or raises ValueError on error.
    """
    # Find latest model (now returns fixed path)
    weights_path = find_latest_model(upload_folder)
    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError(f"No trained model found. Expected at runs/detect/fault/weights/best.pt")  # Adjusted message for fixed path

    # Load model
    model = YOLO(weights_path)
    
    # Run prediction
    results = model.predict(
        source=image_path,
        conf=0.5,  # Confidence threshold
        save=False,  # Don't save to disk, process in memory
        verbose=False
    )
    
    if not results or len(results) == 0:
        raise ValueError("No results from prediction.")
    
    r = results[0]
    detections = []
    if r.boxes is not None:
        for box in r.boxes:
            # Extract class, conf, bbox
            cls_id = int(box.cls[0])
            class_name = r.names[cls_id] if r.names else f'class_{cls_id}'  # Use model names if available
            conf = float(box.conf[0]) * 100  # As percentage
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            detections.append({
                'class': class_name,
                'confidence': round(conf, 2),
                'bbox': [round(x, 2) for x in bbox]  # For JS positioning (note: scale to image size if needed)
            })
    
    # Annotated image (with boxes)
    annotated_img = r.plot()  # Returns numpy array in BGR format
    annotated_img = annotated_img[..., ::-1]  # Convert BGR to RGB (fix for color swapping)
    annotated_pil = Image.fromarray(annotated_img)
    buffered = BytesIO()
    annotated_pil.save(buffered, format="JPEG", quality=85)
    detected_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Original image
    with open(image_path, 'rb') as f:
        orig_data = f.read()
    original_base64 = base64.b64encode(orig_data).decode('utf-8')  # Keep original format, but HTML assumes JPEG
    
    return {
        'original': original_base64,
        'detected': detected_base64,
        'predictions': detections,
        'detections': detections  # For HTML's addBoundingBoxes
    }
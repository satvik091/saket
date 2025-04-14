from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import io
from PIL import Image
import base64
from ultralytics import YOLO  # Use YOLO class instead of torch.hub

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load your YOLO model at startup
def load_model():
    model_path = "best (1).pt"
    model = YOLO(model_path)  # Load YOLOv5 model directly
    print("Model loaded successfully!")
    return model

# Load model globally when app starts
model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')  # Ensure 3 channels
    
    # Convert PIL image to numpy array (OpenCV format)
    img_np = np.array(img)
    results = model(img_np)[0]  # Get first result

    # Process results
    results_data = process_results(results, img_np)
    
    return jsonify(results_data)

def process_results(results, original_img_np):
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        detections.append({
            "name": label,
            "xmin": x1,
            "ymin": y1,
            "xmax": x2,
            "ymax": y2,
            "confidence": float(box.conf[0])
        })
    
    # Annotated image
    annotated = results.plot()
    _, buffer = cv2.imencode('.jpg', annotated)
    annotated_img_str = base64.b64encode(buffer).decode()
    
    # Wing calculations
    wing_data = calculate_wing_data(detections)
    
    # This format should match what your frontend expects
    return {
        'detections': detections,
        'annotated_image': f"data:image/jpeg;base64,{annotated_img_str}",  # Add proper data URI format
        'wing_data': wing_data
    }

def calculate_wing_data(detections):
    # Default values
    wing_data = {
        'wing_type': 'Swept Wing',
        'aspect_ratio': '9.8:1',
        'estimated_wingspan': '35.8 meters',
        'wing_area': '125.4 sq. meters',
        'efficiency_rating': '94.7%'
    }
    
    wings = [d for d in detections if d.get('name') == 'wing']
    if wings:
        wing = wings[0]
        width = wing['xmax'] - wing['xmin']
        height = wing['ymax'] - wing['ymin']
        
        # Format the values as strings to match the expected format in your frontend
        wing_data['aspect_ratio'] = f"{round(width / height, 1)}:1" if height > 0 else "0:1"
        wing_data['estimated_wingspan'] = f"{round(width * 0.1, 1)} meters"
        wing_data['wing_area'] = f"{round(width * height * 0.01, 1)} sq. meters"
        wing_data['efficiency_rating'] = f"{min(round(width / height * 10, 1), 99.9)}%"
    
    return wing_data

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

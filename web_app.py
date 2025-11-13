"""
🌐 Flask Web/Desktop Interface - Plate Detection Demo
Interactive web app untuk test model dengan UI yang modern & responsif
"""

from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from plate_rotation_detector import PlateRotationDetector
import time
import base64
from io import BytesIO
import json
from datetime import datetime


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Load model globally
print("="*70)
print("🚀 PLATE DETECTION WEB APP")
print("="*70)
print("\n📥 Loading model...")
possible_paths = [
    "best.pt",
    "best_model_epoch170.pt",
    "runs/plate_detection/yolov11_ultimate_v1/weights/epoch170.pt"
]

model = None
for path in possible_paths:
    if Path(path).exists():
        model = YOLO(path)
        print(f"✅ Model loaded from: {path}")
        print(f"   Size: {Path(path).stat().st_size / (1024*1024):.2f} MB")
        break

rotation_detector = PlateRotationDetector(debug=False)
print(f"✅ Rotation detector loaded")

# Statistics
stats = {
    'total_processed': 0,
    'total_detections': 0,
    'avg_confidence': 0,
    'avg_time': 0,
    'history': []
}


def img_to_base64(image):
    """Convert OpenCV image to base64"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{img_base64}"


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect():
    """API endpoint untuk detection"""
    try:
        start_time = time.time()
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Cannot read image'}), 400
        
        original_image = image.copy()
        h, w = image.shape[:2]
        
        # Detect rotation
        corrected_image, angle, rot_confidence = rotation_detector.preprocess(image)
        
        # Run detection
        results = model.predict(corrected_image, conf=0.25, verbose=False)
        boxes = results[0].boxes
        
        # Parse detections
        detections = []
        for box in boxes:
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            detections.append({
                'confidence': f"{conf:.2%}",
                'confidence_value': conf,
                'x1': float(xyxy[0]),
                'y1': float(xyxy[1]),
                'x2': float(xyxy[2]),
                'y2': float(xyxy[3]),
                'width': float(xyxy[2] - xyxy[0]),
                'height': float(xyxy[3] - xyxy[1])
            })
        
        # Create annotated image
        annotated = results[0].plot()
        annotated_base64 = img_to_base64(annotated)
        
        # Calculate stats
        processing_time = time.time() - start_time
        
        # Update global stats
        stats['total_processed'] += 1
        stats['total_detections'] += len(detections)
        
        if detections:
            confidences = [d['confidence_value'] for d in detections]
            stats['avg_confidence'] = np.mean(confidences)
        
        stats['avg_time'] = processing_time
        
        # Add to history
        history_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filename': file.filename,
            'detections': len(detections),
            'confidence': f"{stats['avg_confidence']:.2%}",
            'time': f"{processing_time*1000:.2f}ms",
            'rotation': angle
        }
        stats['history'].append(history_entry)
        # Keep only last 50 entries
        if len(stats['history']) > 50:
            stats['history'] = stats['history'][-50:]
        
        return jsonify({
            'success': True,
            'image': {
                'width': w,
                'height': h,
                'rotation': angle
            },
            'detections': detections,
            'stats': {
                'count': len(detections),
                'rotation_angle': angle,
                'rotation_confidence': f"{rot_confidence:.2%}",
                'processing_time': f"{processing_time*1000:.2f}ms",
                'confidence_avg': f"{stats['avg_confidence']:.2%}" if detections else "N/A"
            },
            'annotated_image': annotated_base64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get overall statistics"""
    return jsonify({
        'total_processed': stats['total_processed'],
        'total_detections': stats['total_detections'],
        'avg_confidence': f"{stats['avg_confidence']:.2%}",
        'avg_time': f"{stats['avg_time']*1000:.2f}ms",
        'history': stats['history'],
        'model_info': {
            'name': 'YOLOv11n (Epoch 170)',
            'precision': '81.64%',
            'mAP50': '49.14%',
            'speed': '1.30ms (GPU)',
            'size': '16.08 MB'
        }
    })


@app.route('/api/batch', methods=['POST'])
def batch_detect():
    """Batch detection from folder"""
    try:
        folder = request.json.get('folder')
        if not folder or not Path(folder).exists():
            return jsonify({'error': 'Invalid folder path'}), 400
        
        pattern = request.json.get('pattern', '*.jpg')
        images = list(Path(folder).glob(pattern))[:20]  # Limit to 20 images
        
        results_list = []
        total_detections = 0
        total_time = 0
        
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            start = time.time()
            
            # Detect rotation
            corrected, angle, rot_conf = rotation_detector.preprocess(img)
            
            # Run detection
            results = model.predict(corrected, conf=0.25, verbose=False)
            detections = len(results[0].boxes)
            
            elapsed = time.time() - start
            total_time += elapsed
            total_detections += detections
            
            results_list.append({
                'filename': img_path.name,
                'detections': detections,
                'rotation': angle,
                'time': f"{elapsed*1000:.2f}ms"
            })
        
        avg_time = total_time / len(results_list) if results_list else 0
        
        return jsonify({
            'total_images': len(results_list),
            'total_detections': total_detections,
            'avg_time': f"{avg_time*1000:.2f}ms",
            'avg_detections_per_image': f"{total_detections/len(results_list):.2f}" if results_list else "0",
            'results': results_list
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/info')
def model_info():
    """Get model information"""
    return jsonify({
        'model': 'YOLOv11n',
        'epoch': 170,
        'status': 'Production Ready ✅',
        'metrics': {
            'precision': '81.64%',
            'mAP50': '49.14%',
            'recall': 'High',
            'speed': '1.30ms (RTX 3080 Ti)',
            'size': '16.08 MB'
        },
        'features': [
            '✓ Automatic rotation detection (0°/90°/180°/270°)',
            '✓ High precision plate detection',
            '✓ Real-time inference',
            '✓ Rotation-invariant detection',
            '✓ Multi-plate detection',
            '✓ Batch processing'
        ],
        'input_size': 640,
        'confidence_threshold': 0.25,
        'max_detections': 10,
        'framework': 'PyTorch + YOLOv11'
    })


@app.route('/api/history')
def get_history():
    """Get detection history"""
    return jsonify({
        'history': stats['history'],
        'count': len(stats['history'])
    })


@app.route('/api/reset', methods=['POST'])
def reset_stats():
    """Reset all statistics"""
    stats['total_processed'] = 0
    stats['total_detections'] = 0
    stats['avg_confidence'] = 0
    stats['avg_time'] = 0
    stats['history'] = []
    
    return jsonify({'success': True, 'message': 'Statistics reset'})


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 WEB SERVER STARTING")
    print("="*70)
    print("\n📱 Open your browser and go to:")
    print("   ➜ http://localhost:5000")
    print("\n✨ Features:")
    print("   • Upload images")
    print("   • Real-time detection")
    print("   • View statistics")
    print("   • Detection history")
    print("   • Batch processing")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

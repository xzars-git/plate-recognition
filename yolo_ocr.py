"""
YOLO-based OCR for License Plate Character Recognition
Replaces PaddleOCR with YOLOv11 character detection

Usage:
    from yolo_ocr import YOLO_OCR
    
    ocr = YOLO_OCR('runs/character_detect/yolo11n_chars/weights/best.pt')
    text = ocr.read_text(plate_image)
    print(f"Detected: {text}")
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class YOLO_OCR:
    """
    YOLO-based OCR for character recognition in license plates
    """
    
    def __init__(self, model_path='runs/character_detect/yolo11n_chars/weights/best.pt'):
        """
        Initialize YOLO character detection model
        
        Args:
            model_path: Path to trained YOLO character detection model
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"🔤 Loading YOLO OCR model: {model_path}")
        self.model = YOLO(model_path)
        
        # Character mapping (class ID → character)
        self.class_to_char = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
            5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
            15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
            20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
            25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
            30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
            35: 'Z'
        }
        
        print("   ✅ YOLO OCR ready!")
    
    def read_text(self, plate_img, conf_threshold=0.5, min_chars=3, max_chars=15):
        """
        Detect characters in plate image and return text
        
        Args:
            plate_img: Cropped plate image (numpy array)
            conf_threshold: Confidence threshold for character detection
            min_chars: Minimum number of characters expected
            max_chars: Maximum number of characters expected
            
        Returns:
            str: Recognized text (e.g., "H2251RB")
        """
        result = self.read_text_with_details(plate_img, conf_threshold, 
                                             min_chars, max_chars)
        return result['text']
    
    def read_text_with_details(self, plate_img, conf_threshold=0.5, 
                               min_chars=3, max_chars=15):
        """
        Same as read_text but returns detailed information
        
        Args:
            plate_img: Cropped plate image (numpy array)
            conf_threshold: Confidence threshold for character detection
            min_chars: Minimum number of characters expected
            max_chars: Maximum number of characters expected
            
        Returns:
            dict: {
                'text': str,                    # Full text
                'characters': list,             # List of character dicts
                'avg_conf': float,              # Average confidence
                'num_chars': int                # Number of characters
            }
        """
        # Run YOLO detection
        results = self.model(plate_img, verbose=False)[0]
        
        # Extract detections
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return {
                'text': '',
                'characters': [],
                'avg_conf': 0.0,
                'num_chars': 0
            }
        
        # Get character detections
        characters = []
        confidences = []
        
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            
            # Get class (character)
            cls = int(box.cls[0])
            char = self.class_to_char.get(cls, '?')
            
            characters.append({
                'char': char,
                'x_center': x_center,
                'y_center': y_center,
                'conf': conf,
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'class': cls
            })
            confidences.append(conf)
        
        # Check if we have reasonable number of characters
        if len(characters) < min_chars or len(characters) > max_chars:
            # Still return what we got, but might be unreliable
            pass
        
        # Sort characters by x coordinate (left to right)
        characters.sort(key=lambda c: c['x_center'])
        
        # Combine characters into text
        text = ''.join([c['char'] for c in characters])
        
        # Calculate average confidence
        avg_conf = np.mean(confidences) if confidences else 0.0
        
        return {
            'text': text,
            'characters': characters,
            'avg_conf': float(avg_conf),
            'num_chars': len(characters)
        }
    
    def visualize(self, plate_img, result_dict, show_conf=True):
        """
        Draw character bounding boxes on image
        
        Args:
            plate_img: Original plate image
            result_dict: Output from read_text_with_details()
            show_conf: Whether to show confidence scores
            
        Returns:
            Annotated image
        """
        img = plate_img.copy()
        
        # Draw each character
        for char_info in result_dict['characters']:
            x1, y1, x2, y2 = char_info['bbox']
            char = char_info['char']
            conf = char_info['conf']
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw character label
            if show_conf:
                label = f"{char} {conf:.2f}"
            else:
                label = char
            
            cv2.putText(img, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 0), 2)
        
        # Draw full text at bottom
        text = result_dict['text']
        avg_conf = result_dict['avg_conf']
        
        if show_conf:
            full_label = f"{text} (avg: {avg_conf:.2f})"
        else:
            full_label = text
        
        cv2.putText(img, full_label, (10, img.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                   (0, 255, 255), 2)
        
        return img
    
    def batch_read(self, plate_images, conf_threshold=0.5):
        """
        Process multiple plate images at once (faster with batching)
        
        Args:
            plate_images: List of plate images
            conf_threshold: Confidence threshold
            
        Returns:
            List of text strings
        """
        # Run batch prediction
        results = self.model(plate_images, verbose=False)
        
        texts = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                texts.append('')
                continue
            
            # Extract characters
            characters = []
            for box in boxes:
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x_center = (x1 + x2) / 2
                
                cls = int(box.cls[0])
                char = self.class_to_char.get(cls, '?')
                
                characters.append({
                    'char': char,
                    'x_center': x_center
                })
            
            # Sort and combine
            characters.sort(key=lambda c: c['x_center'])
            text = ''.join([c['char'] for c in characters])
            texts.append(text)
        
        return texts
    
    def predict(self, source, **kwargs):
        """
        Alias for read_text (compatible with PaddleOCR API)
        
        Args:
            source: Image path or numpy array
            **kwargs: Additional arguments
            
        Returns:
            List of results in PaddleOCR-like format
        """
        # Load image if path
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
        else:
            img = source
        
        # Get results
        result = self.read_text_with_details(img)
        
        # Format similar to PaddleOCR output
        ocr_result = []
        for char_info in result['characters']:
            bbox = char_info['bbox']
            char = char_info['char']
            conf = char_info['conf']
            
            # Convert bbox to PaddleOCR format
            x1, y1, x2, y2 = bbox
            bbox_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            ocr_result.append([bbox_points, (char, conf)])
        
        return [[ocr_result]] if ocr_result else [[]]


# Convenience function
def create_yolo_ocr(model_path='runs/character_detect/yolo11n_chars/weights/best.pt'):
    """
    Factory function to create YOLO_OCR instance
    
    Args:
        model_path: Path to trained model
        
    Returns:
        YOLO_OCR instance
    """
    return YOLO_OCR(model_path)


# Test function
def test_yolo_ocr():
    """Quick test of YOLO OCR"""
    import time
    
    print("\n🧪 Testing YOLO OCR...")
    
    # Create OCR
    ocr = YOLO_OCR('runs/character_detect/yolo11n_chars/weights/best.pt')
    
    # Test image
    test_img_path = 'dataset/plate_text_dataset/dataset/H2251RB.jpg'
    if not Path(test_img_path).exists():
        print(f"❌ Test image not found: {test_img_path}")
        return
    
    img = cv2.imread(test_img_path)
    
    # Benchmark
    num_tests = 100
    start = time.time()
    for _ in range(num_tests):
        result = ocr.read_text_with_details(img)
    elapsed = time.time() - start
    
    avg_time = elapsed / num_tests * 1000  # ms
    
    print(f"\n📊 Results:")
    print(f"   Text: {result['text']}")
    print(f"   Characters: {result['num_chars']}")
    print(f"   Avg Confidence: {result['avg_conf']:.3f}")
    print(f"   Avg Time: {avg_time:.2f}ms")
    print(f"   FPS: {1000/avg_time:.1f}")
    
    # Visualize
    vis_img = ocr.visualize(img, result)
    cv2.imshow('YOLO OCR Test', vis_img)
    print("\n👁️ Press any key to close visualization...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_yolo_ocr()

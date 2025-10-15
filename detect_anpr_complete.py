"""
Complete ANPR Pipeline - 2 Stages
Stage 1: Detect plate location dengan YOLOv11
Stage 2: Read text dengan PaddleOCR
"""

from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
from pathlib import Path

class ANPRSystem:
    def __init__(self, plate_detection_model, use_gpu=True):
        """
        Initialize ANPR System
        
        Args:
            plate_detection_model: Path ke YOLOv11 model untuk plate detection
            use_gpu: Gunakan GPU atau tidak
        """
        # Load YOLOv11 untuk plate detection
        self.plate_detector = YOLO(plate_detection_model)
        
        # Load PaddleOCR untuk character recognition
        # use_angle_cls=True untuk detect teks yang miring
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',  # 'en' untuk huruf latin
            use_gpu=use_gpu,
            show_log=False
        )
        
        print("✅ ANPR System initialized!")
        print(f"   Plate Detector: {plate_detection_model}")
        print(f"   OCR Engine: PaddleOCR")
    
    def detect_plate(self, image, conf_threshold=0.25):
        """
        Stage 1: Detect plate location
        
        Args:
            image: Input image (numpy array atau path)
            conf_threshold: Confidence threshold
            
        Returns:
            List of cropped plate images
        """
        # Predict dengan YOLO
        results = self.plate_detector(image, conf=conf_threshold, verbose=False)
        
        plates = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Crop plate region
                if isinstance(image, str):
                    img = cv2.imread(image)
                else:
                    img = image.copy()
                
                plate_img = img[y1:y2, x1:x2]
                plates.append({
                    'image': plate_img,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })
        
        return plates
    
    def read_plate_text(self, plate_image):
        """
        Stage 2: Read text dari cropped plate
        
        Args:
            plate_image: Cropped plate image
            
        Returns:
            Recognized text
        """
        # OCR dengan PaddleOCR
        result = self.ocr.ocr(plate_image, cls=True)
        
        if result is None or len(result) == 0:
            return ""
        
        # Gabungkan semua text yang terdeteksi
        texts = []
        for line in result:
            if line:
                for word_info in line:
                    text = word_info[1][0]  # text
                    conf = word_info[1][1]  # confidence
                    if conf > 0.5:  # Filter low confidence
                        texts.append(text)
        
        # Gabungkan dan bersihkan
        full_text = ''.join(texts)
        full_text = full_text.replace(' ', '').upper()  # Remove spaces, uppercase
        
        return full_text
    
    def process(self, image_path, visualize=True):
        """
        Complete ANPR pipeline
        
        Args:
            image_path: Path ke gambar input
            visualize: Tampilkan hasil atau tidak
            
        Returns:
            Dict dengan hasil deteksi
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Cannot read image: {image_path}")
            return None
        
        # Stage 1: Detect plates
        plates = self.detect_plate(image)
        
        if len(plates) == 0:
            print("⚠️  No plate detected!")
            return {'plates': []}
        
        # Stage 2: Read text dari setiap plate
        results = []
        for idx, plate_info in enumerate(plates):
            text = self.read_plate_text(plate_info['image'])
            
            results.append({
                'plate_number': text,
                'bbox': plate_info['bbox'],
                'confidence': plate_info['confidence']
            })
            
            print(f"   Plate {idx+1}: {text} (conf: {plate_info['confidence']:.2f})")
        
        # Visualize
        if visualize:
            vis_image = image.copy()
            for result in results:
                x1, y1, x2, y2 = result['bbox']
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw text
                text = result['plate_number']
                cv2.putText(vis_image, text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            return {
                'plates': results,
                'visualization': vis_image
            }
        
        return {'plates': results}

def main():
    print("="*60)
    print("🚗 COMPLETE ANPR SYSTEM - 2 Stages")
    print("="*60)
    
    # Path ke model plate detection (hasil training stage 1)
    plate_model = 'runs/plate_detection/yolov11_stage1/weights/best.pt'
    
    # Check if model exists
    if not Path(plate_model).exists():
        print(f"\n❌ Model not found: {plate_model}")
        print("   Please train plate detection model first:")
        print("   python train_plate_detection.py")
        return
    
    # Initialize ANPR system
    anpr = ANPRSystem(plate_model, use_gpu=True)
    
    # Test pada gambar
    print("\n" + "="*60)
    print("📷 Testing ANPR System")
    print("="*60)
    
    test_image = input("\nMasukkan path gambar: ")
    
    if not Path(test_image).exists():
        print(f"❌ File not found: {test_image}")
        return
    
    # Process
    print(f"\n🔍 Processing: {test_image}")
    result = anpr.process(test_image, visualize=True)
    
    if result and len(result['plates']) > 0:
        print(f"\n✅ Detected {len(result['plates'])} plate(s):")
        for idx, plate in enumerate(result['plates']):
            print(f"   {idx+1}. {plate['plate_number']} (confidence: {plate['confidence']:.2f})")
        
        # Save visualization
        output_path = 'anpr_result.jpg'
        cv2.imwrite(output_path, result['visualization'])
        print(f"\n💾 Result saved: {output_path}")
        
        # Show image
        cv2.imshow('ANPR Result', result['visualization'])
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\n⚠️  No plates detected!")

if __name__ == '__main__':
    main()

"""
Test ANPR Model dengan Webcam - Real-time Detection
Stage 1: Deteksi plat nomor dengan YOLOv11
Stage 2: Baca teks dengan PaddleOCR (optional)
"""

import cv2
from ultralytics import YOLO
from pathlib import Path
import time

class WebcamANPR:
    def __init__(self, plate_model_path, use_ocr=False):
        """
        Initialize Webcam ANPR
        
        Args:
            plate_model_path: Path ke model YOLOv11 untuk plate detection
            use_ocr: Apakah menggunakan OCR untuk baca teks (butuh PaddleOCR)
        """
        print("="*60)
        print("🎥 Webcam ANPR System")
        print("="*60)
        
        # Check if model exists
        if not Path(plate_model_path).exists():
            raise FileNotFoundError(f"Model not found: {plate_model_path}")
        
        # Load YOLO model
        print(f"\n📦 Loading model: {plate_model_path}")
        self.model = YOLO(plate_model_path)
        print("✅ Model loaded!")
        
        # OCR (optional)
        self.use_ocr = use_ocr
        if use_ocr:
            try:
                from paddleocr import PaddleOCR
                print("\n📦 Loading PaddleOCR...")
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en', 
                                    use_gpu=False, show_log=False)
                print("✅ OCR loaded!")
            except ImportError:
                print("⚠️  PaddleOCR not installed. OCR disabled.")
                print("   Install with: pip install paddleocr paddlepaddle")
                self.use_ocr = False
        
        # Stats
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
    def read_plate_text(self, plate_img):
        """Read text from cropped plate using OCR"""
        if not self.use_ocr or plate_img is None:
            return ""
        
        try:
            result = self.ocr.ocr(plate_img, cls=True)
            if result is None or len(result) == 0:
                return ""
            
            texts = []
            for line in result:
                if line:
                    for word_info in line:
                        text = word_info[1][0]
                        conf = word_info[1][1]
                        if conf > 0.5:
                            texts.append(text)
            
            full_text = ''.join(texts).replace(' ', '').upper()
            return full_text
        except:
            return ""
    
    def process_frame(self, frame):
        """Process single frame"""
        # Detect plates
        results = self.model(frame, conf=0.25, verbose=False)
        
        # Draw results
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Draw box
                color = (0, 255, 0)  # Green
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"Plate {conf:.2f}"
                
                # OCR (if enabled)
                if self.use_ocr and x2 > x1 and y2 > y1:
                    plate_crop = frame[y1:y2, x1:x2]
                    text = self.read_plate_text(plate_crop)
                    if text:
                        label = f"{text} {conf:.2f}"
                
                # Draw label
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated_frame, (x1, y1-20), (x1+w, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def run(self, camera_id=0):
        """
        Run webcam detection
        
        Args:
            camera_id: ID kamera (0 untuk default webcam)
        """
        print(f"\n🎥 Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        # Set resolution (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "="*60)
        print("✅ Camera opened successfully!")
        print("="*60)
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'o' - Toggle OCR (if available)")
        print("\nPress any key in the window to start...")
        print("="*60)
        
        self.start_time = time.time()
        self.frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame!")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Calculate FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                self.fps = self.frame_count / elapsed_time
            
            # Draw FPS
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(processed_frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw OCR status
            if self.use_ocr:
                cv2.putText(processed_frame, "OCR: ON", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('ANPR - Webcam Detection', processed_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n🛑 Stopping...")
                break
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('o'):
                if self.use_ocr:
                    self.use_ocr = False
                    print("OCR: OFF")
                else:
                    try:
                        from paddleocr import PaddleOCR
                        self.use_ocr = True
                        print("OCR: ON")
                    except:
                        print("⚠️  OCR not available")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("✅ Session complete!")
        print(f"   Frames processed: {self.frame_count}")
        print(f"   Average FPS: {self.fps:.1f}")
        print("="*60)

def main():
    """Main function"""
    print("="*60)
    print("🎥 ANPR Webcam Test")
    print("="*60)
    
    # Path ke model di root folder
    model_path = 'best.pt'
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"\n❌ Model not found: {model_path}")
        print("\nSearching for available models...")
        
        # Cari di folder runs juga (backup)
        runs_dir = Path('runs/plate_detection')
        if runs_dir.exists():
            available_models = list(runs_dir.glob('**/weights/best.pt'))
            if available_models:
                print("\nAvailable models in runs folder:")
                for run in sorted(available_models):
                    print(f"  - {run}")
                
                # Use the latest model
                latest_model = max(available_models, 
                                 key=lambda p: p.stat().st_mtime)
                print(f"\n💡 Using latest model: {latest_model}")
                model_path = str(latest_model)
            else:
                print("\n❌ No trained model found!")
                print("   Please train a model first: python train_plate_detection.py")
                return
        else:
            print("\n❌ No models found!")
            print("   Please ensure best.pt exists in root folder")
            print("   Or train a model: python train_plate_detection.py")
            return
    
    # Initialize webcam ANPR
    use_ocr = input("\nEnable OCR text recognition? (y/n, default=n): ").lower() == 'y'
    
    try:
        anpr = WebcamANPR(model_path, use_ocr=use_ocr)
        anpr.run(camera_id=0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

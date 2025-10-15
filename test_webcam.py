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
        
        # OCR (optional) - LOAD SEKALI DI SINI
        self.use_ocr = use_ocr
        self.ocr = None
        if use_ocr:
            try:
                from paddleocr import PaddleOCR
                print("\n📦 Loading PaddleOCR...")
                # Minimal parameters for compatibility
                self.ocr = PaddleOCR(lang='en', use_gpu=False, show_log=False)
                print("✅ OCR loaded!")
            except ImportError:
                print("⚠️  PaddleOCR not installed. OCR disabled.")
                print("   Install with: pip install paddleocr paddlepaddle")
                self.use_ocr = False
                self.ocr = None
            except Exception as e:
                print(f"⚠️  PaddleOCR error: {e}")
                print("   OCR disabled.")
                self.use_ocr = False
                self.ocr = None
        
        # Stats
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Optimization: resize target untuk inference
        self.inference_size = 640  # YOLO input size
        
        # Skip frame untuk performa lebih baik
        self.skip_frames = 2  # Process 1 dari setiap 3 frame (balance)
        self.last_result = None  # Cache hasil terakhir
        
    def read_plate_text(self, plate_img):
        """Read text from cropped plate using OCR"""
        if not self.use_ocr or self.ocr is None or plate_img is None:
            return ""
        
        try:
            # Use new PaddleOCR API: predict() instead of ocr()
            result = self.ocr.predict(plate_img)
            
            if not result or len(result) == 0:
                return ""
            
            # Extract text from result
            page_result = result[0]  # First page
            if 'rec_texts' not in page_result:
                return ""
            
            # Join all detected texts
            texts = page_result['rec_texts']
            rec_scores = page_result.get('rec_scores', [1.0] * len(texts))
            
            # Filter by confidence
            filtered_texts = []
            for text, score in zip(texts, rec_scores):
                if score > 0.5:
                    filtered_texts.append(text)
            
            full_text = ''.join(filtered_texts).replace(' ', '').upper()
            return full_text
        except Exception as e:
            return ""
    
    def process_frame(self, frame):
        """Process single frame - OPTIMIZED (in-place drawing)"""
        # Get original dimensions
        h, w = frame.shape[:2]
        
        # Resize untuk inference (lebih cepat)
        scale = self.inference_size / max(h, w)
        if scale < 1.0:  # Only resize if image is larger
            new_w = int(w * scale)
            new_h = int(h * scale)
            inference_frame = cv2.resize(frame, (new_w, new_h))
        else:
            inference_frame = frame
            scale = 1.0
        
        # Detect plates (pada frame yang sudah di-resize)
        results = self.model(inference_frame, conf=0.25, verbose=False)
        
        # Draw results LANGSUNG pada frame (in-place, no copy!)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box (scale kembali ke ukuran original)
                x1, y1, x2, y2 = box.xyxy[0]
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
                conf = float(box.conf[0])
                
                # Draw box
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"Plate {conf:.2f}"
                
                # OCR (if enabled) - gunakan frame original untuk crop
                if self.use_ocr and self.ocr is not None and x2 > x1 and y2 > y1:
                    plate_crop = frame[y1:y2, x1:x2]
                    text = self.read_plate_text(plate_crop)
                    if text:
                        label = f"{text} {conf:.2f}"
                
                # Draw label
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1-20), (x1+label_w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame  # Return reference, tidak bikin copy baru
    
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
        
        # OPTIMIZATION: Limit webcam FPS untuk reduce overhead
        cap.set(cv2.CAP_PROP_FPS, 30)  # Max 30 FPS dari webcam
        
        print("\n" + "="*60)
        print("✅ Camera opened successfully!")
        print("="*60)
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'o' - Toggle OCR (if available)")
        print("  '+' - Increase quality (slower)")
        print("  '-' - Decrease quality (faster)")
        print("\nPress any key in the window to start...")
        print("="*60)
        
        self.start_time = time.time()
        self.frame_count = 0
        frame_counter = 0  # Untuk skip frames
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame!")
                break
            
            self.frame_count += 1
            
            # Skip frames untuk performa (process setiap N frame)
            should_process = (frame_counter % (self.skip_frames + 1) == 0)
            
            if should_process:
                # Process frame (in-place)
                self.process_frame(frame)
                self.last_result = frame
            
            frame_counter += 1
            
            # Calculate FPS
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.fps = self.frame_count / elapsed
            
            # Always display untuk smooth FPS
            display_frame = self.last_result if self.last_result is not None else frame
            
            # Draw overlay
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            cv2.putText(display_frame, f"Skip: {self.skip_frames}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if self.use_ocr and self.ocr is not None:
                cv2.putText(display_frame, "OCR: ON", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('ANPR - Webcam Detection', display_frame)
            
            # Handle keyboard (polling setiap frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n🛑 Stopping...")
                break
            elif key == ord('s'):
                save_frame = self.last_result if self.last_result is not None else frame
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, save_frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('o'):
                # Toggle OCR on/off (TIDAK perlu reload model)
                if self.ocr is not None:
                    self.use_ocr = not self.use_ocr
                    status = "ON" if self.use_ocr else "OFF"
                    print(f"OCR: {status}")
                else:
                    print("⚠️  OCR not loaded. Cannot toggle.")
            elif key == ord('+') or key == ord('='):
                # Increase quality (lebih lambat)
                if self.skip_frames > 0:
                    self.skip_frames -= 1
                    print(f"Skip frames: {self.skip_frames} (higher quality)")
            elif key == ord('-'):
                # Decrease quality (lebih cepat)
                if self.skip_frames < 5:
                    self.skip_frames += 1
                    print(f"Skip frames: {self.skip_frames} (higher speed)")
        
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

"""
Test ANPR Model dengan Webcam - ONNX Version
Lebih cepat menggunakan ONNX Runtime untuk inference
Stage 1: Deteksi plat nomor dengan YOLOv11 ONNX
Stage 2: Baca teks dengan PaddleOCR (optional)
"""

import cv2
from ultralytics import YOLO
from pathlib import Path
import time

class WebcamANPR_ONNX:
    def __init__(self, onnx_model_path):
        """
        Initialize Webcam ANPR dengan ONNX model
        
        Args:
            onnx_model_path: Path ke model ONNX untuk plate detection
        """
        print("="*60)
        print("🎥 Webcam ANPR System (ONNX)")
        print("="*60)
        
        # Check if model exists
        if not Path(onnx_model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        
        # Load ONNX model
        print(f"\n📦 Loading ONNX model: {onnx_model_path}")
        self.model = YOLO(onnx_model_path, task='detect')
        print("✅ ONNX model loaded!")
        
        # Stats
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.inference_times = []
        
        # Optimization settings
        self.inference_size = 640
        self.skip_frames = 1  # ONNX lebih cepat, skip lebih sedikit
        self.last_result = None
        
    def process_frame(self, frame):
        """Process single frame - OPTIMIZED with ONNX"""
        h, w = frame.shape[:2]
        
        # Resize untuk inference
        scale = self.inference_size / max(h, w)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            inference_frame = cv2.resize(frame, (new_w, new_h))
        else:
            inference_frame = frame
            scale = 1.0
        
        # ONNX Inference dengan timing
        inference_start = time.time()
        results = self.model(inference_frame, conf=0.25, verbose=False)
        inference_time = (time.time() - inference_start) * 1000  # ms
        self.inference_times.append(inference_time)
        
        # Keep only last 30 inference times
        if len(self.inference_times) > 30:
            self.inference_times.pop(0)
        
        # Draw results LANGSUNG pada frame (in-place)
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
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label (simple, no OCR)
                label = f"Plate {conf:.2f}"
                
                # Draw label
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1-20), (x1+label_w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self, camera_id=0):
        """
        Run webcam detection dengan ONNX
        
        Args:
            camera_id: ID kamera (0 untuk default webcam)
        """
        print(f"\n🎥 Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*60)
        print("✅ Camera opened successfully!")
        print("="*60)
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  '+' - Increase quality (slower)")
        print("  '-' - Decrease quality (faster)")
        print("  'i' - Show inference stats")
        print("\nPress any key in the window to start...")
        print("="*60)
        
        self.start_time = time.time()
        self.frame_count = 0
        frame_counter = 0
        show_stats = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame!")
                break
            
            self.frame_count += 1
            
            # Skip frame logic
            should_process = (frame_counter % (self.skip_frames + 1) == 0)
            
            if should_process:
                self.process_frame(frame)
                self.last_result = frame
            
            frame_counter += 1
            
            # Calculate FPS
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.fps = self.frame_count / elapsed
            
            # Display frame
            display_frame = self.last_result if self.last_result is not None else frame
            
            # Draw FPS overlay
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Draw model info
            cv2.putText(display_frame, "ONNX Runtime", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.putText(display_frame, f"Skip: {self.skip_frames}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Inference stats (optional)
            if show_stats and len(self.inference_times) > 0:
                avg_inference = sum(self.inference_times) / len(self.inference_times)
                min_inference = min(self.inference_times)
                max_inference = max(self.inference_times)
                
                cv2.putText(display_frame, f"Inference: {avg_inference:.1f}ms", (10, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Min/Max: {min_inference:.1f}/{max_inference:.1f}ms", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('ANPR - Webcam Detection (ONNX)', display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n🛑 Stopping...")
                break
            elif key == ord('s'):
                save_frame = self.last_result if self.last_result is not None else frame
                filename = f"screenshot_onnx_{int(time.time())}.jpg"
                cv2.imwrite(filename, save_frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('+') or key == ord('='):
                if self.skip_frames > 0:
                    self.skip_frames -= 1
                    print(f"Skip frames: {self.skip_frames} (higher quality)")
            elif key == ord('-'):
                if self.skip_frames < 5:
                    self.skip_frames += 1
                    print(f"Skip frames: {self.skip_frames} (higher speed)")
            elif key == ord('i'):
                show_stats = not show_stats
                status = "ON" if show_stats else "OFF"
                print(f"Inference stats: {status}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print("\n" + "="*60)
        print("✅ Session complete!")
        print("="*60)
        print(f"   Frames processed: {self.frame_count}")
        print(f"   Average FPS: {self.fps:.1f}")
        
        if len(self.inference_times) > 0:
            avg_inference = sum(self.inference_times) / len(self.inference_times)
            min_inference = min(self.inference_times)
            max_inference = max(self.inference_times)
            print(f"\n📊 Inference Statistics:")
            print(f"   Average: {avg_inference:.1f}ms")
            print(f"   Min: {min_inference:.1f}ms")
            print(f"   Max: {max_inference:.1f}ms")
            print(f"   Theoretical max FPS: {1000/avg_inference:.1f}")
        
        print("="*60)


def main():
    """Main function"""
    print("="*60)
    print("🎥 ANPR Webcam Test (ONNX)")
    print("="*60)
    
    # Path ke ONNX model
    model_path = 'best.onnx'
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"\n❌ ONNX model not found: {model_path}")
        print("\n💡 Tips:")
        print("   1. Export PyTorch model to ONNX:")
        print("      python convert_to_onnx.py")
        print("   2. Or use existing best.pt:")
        print("      python test_webcam.py")
        return
    
    # Get file size
    model_size = Path(model_path).stat().st_size / (1024*1024)
    print(f"\n✅ ONNX model found: {model_path} ({model_size:.1f} MB)")
    
    # Initialize webcam ANPR (no OCR option)
    try:
        anpr = WebcamANPR_ONNX(model_path)
        anpr.run(camera_id=0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

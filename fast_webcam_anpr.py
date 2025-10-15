"""
Fast Webcam ANPR - Optimized for Real-time Performance
Using threading for OCR to prevent webcam freezing
"""

from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import time
import threading
import queue
from collections import deque

class FastWebcamANPR:
    """Real-time ANPR dengan OCR asynchronous untuk performa optimal"""
    
    def __init__(self, model_path='best.pt', enable_ocr=False):
        """
        Initialize ANPR system
        
        Args:
            model_path: Path ke model YOLOv11
            enable_ocr: Enable OCR (default False untuk speed)
        """
        print("="*60)
        print("🚀 FAST WEBCAM ANPR - OPTIMIZED")
        print("="*60)
        
        # Load detection model
        print(f"\n📦 Loading detection model: {model_path}")
        self.model = YOLO(model_path)
        print("✅ Detection model loaded!")
        
        # OCR settings
        self.use_ocr = enable_ocr
        self.ocr = None
        self.ocr_queue = queue.Queue(maxsize=5)  # Limit queue size
        self.ocr_results = {}  # Cache OCR results by bbox
        self.ocr_lock = threading.Lock()  # Thread-safe access to results
        self.ocr_thread = None
        self.ocr_running = False
        self.ocr_loaded = False
        
        if self.use_ocr:
            print("\n📦 Loading OCR model in background (won't block camera)...")
            # Load OCR in background thread to prevent freezing
            self.ocr_running = True
            self.ocr_thread = threading.Thread(target=self._init_and_run_ocr, daemon=True)
            self.ocr_thread.start()
            print("✅ OCR loading in background...")
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.frame_count = 0
        self.start_time = time.time()
        
        print("\n" + "="*60)
        print("✅ FAST ANPR READY!")
        print("="*60)
    
    def _init_and_run_ocr(self):
        """Initialize OCR and run worker (in background thread)"""
        try:
            # Load OCR model
            print("   Loading PaddleOCR models...")
            self.ocr = PaddleOCR(lang='en')
            self.ocr_loaded = True
            print("   ✅ OCR ready!")
            
            # Run OCR worker loop
            self._ocr_worker()
            
        except Exception as e:
            print(f"   ❌ OCR init error: {e}")
            import traceback
            traceback.print_exc()
            self.ocr_loaded = False
            self.use_ocr = False
    
    def _ocr_worker(self):
        """Background worker untuk OCR processing"""
        while self.ocr_running:
            try:
                # Wait until OCR is loaded
                if not self.ocr_loaded:
                    time.sleep(0.1)
                    continue
                
                # Get plate image from queue (with timeout)
                item = self.ocr_queue.get(timeout=0.1)
                
                if item is None:  # Stop signal
                    break
                
                bbox_key, plate_img = item
                
                # Process OCR
                text = self._read_plate_text_sync(plate_img)
                
                # Debug print
                if text:
                    print(f"   ✅ OCR: {text} (key: {bbox_key})")
                
                # Store result (thread-safe)
                with self.ocr_lock:
                    self.ocr_results[bbox_key] = {
                        'text': text,
                        'timestamp': time.time()
                    }
                
                # Clean old results (keep only last 10 seconds)
                current_time = time.time()
                with self.ocr_lock:
                    old_keys = [k for k, v in self.ocr_results.items() 
                               if current_time - v['timestamp'] > 10]
                    for k in old_keys:
                        del self.ocr_results[k]
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"   ⚠️ OCR worker error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def _read_plate_text_sync(self, plate_img):
        """Synchronous OCR processing (called by worker thread)"""
        if plate_img is None or self.ocr is None or not self.ocr_loaded:
            return ""
        
        try:
            # Preprocess for better OCR (fast operations only)
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            
            # Resize if too small
            h, w = gray.shape
            if w < 200:
                scale = 200 / w
                new_w = 200
                new_h = int(h * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Enhance contrast (fast)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Convert back to BGR for PaddleOCR
            processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            # OCR
            result = self.ocr.predict(processed)
            
            if not result or len(result) == 0:
                return ""
            
            page_result = result[0]
            if 'rec_texts' not in page_result:
                return ""
            
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
            print(f"   ⚠️ OCR processing error: {e}")
            return ""
    
    def _bbox_to_key(self, bbox):
        """Convert bbox to string key for caching"""
        x1, y1, x2, y2 = bbox
        # Round to nearest 10 pixels for cache matching
        x1, y1, x2, y2 = map(lambda x: round(x/10)*10, [x1, y1, x2, y2])
        return f"{x1}_{y1}_{x2}_{y2}"
    
    def process_frame(self, frame):
        """Process frame dengan OCR asynchronous"""
        frame_start = time.time()
        
        # Detect plates (fast)
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
                
                # Default label
                label = f"Plate {conf:.2f}"
                
                # OCR (asynchronous)
                if self.use_ocr and x2 > x1 and y2 > y1:
                    bbox_key = self._bbox_to_key([x1, y1, x2, y2])
                    
                    # Check if OCR result exists in cache (thread-safe)
                    ocr_text = None
                    with self.ocr_lock:
                        if bbox_key in self.ocr_results:
                            ocr_text = self.ocr_results[bbox_key]['text']
                    
                    if ocr_text:
                        # OCR completed
                        label = f"{ocr_text} {conf:.2f}"
                        color = (0, 255, 255)  # Yellow for OCR'd plates
                    elif self.ocr_loaded:
                        # Queue for OCR processing (non-blocking)
                        if not self.ocr_queue.full():
                            plate_crop = frame[y1:y2, x1:x2].copy()
                            try:
                                self.ocr_queue.put_nowait((bbox_key, plate_crop))
                                label = f"Processing... {conf:.2f}"
                                color = (255, 165, 0)  # Orange for processing
                            except queue.Full:
                                label = f"Queue Full {conf:.2f}"
                                color = (0, 165, 255)  # Orange-red
                        else:
                            label = f"Queue Full {conf:.2f}"
                            color = (0, 165, 255)
                    else:
                        # OCR not loaded yet
                        label = f"Loading OCR... {conf:.2f}"
                        color = (128, 128, 128)  # Gray
                
                # Draw box with appropriate color
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1-25), (x1+w+10, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1+5, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Calculate FPS
        frame_time = time.time() - frame_start
        self.frame_times.append(frame_time)
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Draw FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw OCR queue status
        if self.use_ocr:
            ocr_status = "LOADED" if self.ocr_loaded else "LOADING..."
            with self.ocr_lock:
                cached_count = len(self.ocr_results)
            
            queue_text = f"OCR: {ocr_status} | Queue: {self.ocr_queue.qsize()}/5 | Cached: {cached_count}"
            cv2.putText(annotated_frame, queue_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        self.frame_count += 1
        
        return annotated_frame
    
    def run(self, camera_id=0):
        """Run webcam detection"""
        print(f"\n🎥 Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Optimize for speed
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("\n" + "="*60)
        print("✅ Camera opened successfully!")
        print("="*60)
        print(f"\n📹 Resolution: {actual_width}x{actual_height}")
        if self.use_ocr:
            print(f"🔤 OCR: ENABLED (loading in background...)")
            print(f"   Wait ~5-10 seconds for OCR to be ready")
        else:
            print(f"🔤 OCR: DISABLED")
        
        print("\n" + "="*60)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'o' - Toggle OCR")
        print("  'c' - Clear OCR cache")
        print("  'd' - Debug info")
        print("\nCamera will start immediately (OCR loads in background)")
        print("Press any key in the window to start...")
        print("="*60)
        
        # Create window
        window_name = "Fast ANPR - Optimized for Speed"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Wait for key press
        ret, frame = cap.read()
        if ret:
            cv2.imshow(window_name, frame)
            cv2.waitKey(0)
        
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("\n❌ Failed to grab frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display
                cv2.imshow(window_name, processed_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                
                elif key == ord('s'):
                    screenshot_count += 1
                    filename = f"screenshot_{screenshot_count}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"\n📸 Screenshot saved: {filename}")
                
                elif key == ord('o'):
                    if self.use_ocr:
                        self.use_ocr = False
                        print("\n🔤 OCR disabled")
                    else:
                        if not self.ocr_loaded:
                            print("\n⚠️  OCR still loading! Wait a moment...")
                        else:
                            self.use_ocr = True
                            print("\n🔤 OCR enabled")
                
                elif key == ord('c'):
                    if self.use_ocr:
                        # Clear queue and cache (thread-safe)
                        while not self.ocr_queue.empty():
                            try:
                                self.ocr_queue.get_nowait()
                            except queue.Empty:
                                break
                        with self.ocr_lock:
                            self.ocr_results.clear()
                        print("\n🗑️  OCR cache cleared")
                
                elif key == ord('d'):
                    # Debug info
                    print("\n" + "="*60)
                    print("🔍 DEBUG INFO:")
                    print("="*60)
                    print(f"OCR Enabled: {self.use_ocr}")
                    print(f"OCR Loaded: {self.ocr_loaded}")
                    print(f"OCR Thread Alive: {self.ocr_thread.is_alive() if self.ocr_thread else False}")
                    print(f"Queue Size: {self.ocr_queue.qsize()}/5")
                    with self.ocr_lock:
                        print(f"Cached Results: {len(self.ocr_results)}")
                        if self.ocr_results:
                            print("\nCached OCR Results:")
                            for key, val in list(self.ocr_results.items())[:5]:
                                print(f"  {key}: {val['text']}")
                    print("="*60)
        
        finally:
            # Cleanup
            if self.ocr_thread:
                self.ocr_running = False
                self.ocr_queue.put(None)  # Stop signal
                self.ocr_thread.join(timeout=2)
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Final stats
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            
            print("\n" + "="*60)
            print("📊 STATISTICS:")
            print("="*60)
            print(f"Total frames processed: {self.frame_count}")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Average FPS: {avg_fps:.1f}")
            if self.use_ocr:
                with self.ocr_lock:
                    print(f"OCR results cached: {len(self.ocr_results)}")
            print("="*60)

def main():
    """Main function"""
    
    print("="*60)
    print("🚀 FAST WEBCAM ANPR")
    print("="*60)
    
    print("\n⚡ Optimizations:")
    print("  ✅ Asynchronous OCR (non-blocking)")
    print("  ✅ OCR result caching")
    print("  ✅ Queue management (max 5)")
    print("  ✅ Background worker thread")
    print("  ✅ Fast preprocessing")
    print("  ✅ Reduced buffer size")
    
    # Ask for OCR
    enable_ocr = input("\n🔤 Enable OCR? (y/n, default=y): ").strip().lower()
    enable_ocr = enable_ocr != 'n'
    
    if enable_ocr:
        print("\n💡 TIP: OCR runs in background thread")
        print("   Webcam will stay smooth (~18 FPS)")
        print("   OCR results appear when ready (1-3 seconds)")
    
    # Initialize system
    anpr = FastWebcamANPR(
        model_path='best.pt',
        enable_ocr=enable_ocr
    )
    
    # Run
    anpr.run(camera_id=0)
    
    print("\n✅ Done!")

if __name__ == '__main__':
    main()

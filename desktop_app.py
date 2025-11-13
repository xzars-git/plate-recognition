#!/usr/bin/env python
"""
🖥️ Desktop GUI App - Plate Detection Demo
Aplikasi desktop dengan Tkinter untuk test model plate detection
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from plate_rotation_detector import PlateRotationDetector
import time
from datetime import datetime
import threading


class PlateDetectionDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Plate Detection Desktop App - Epoch 170")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.current_image = None
        self.current_image_path = None
        self.model = None
        self.rotation_detector = None
        
        # Webcam variables
        self.webcam_active = False
        self.webcam_capture = None
        self.webcam_thread = None
        
        # Rotation toggle
        self.enable_rotation = tk.BooleanVar(value=False)  # Default OFF
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_detections': 0,
            'confidences': [],
            'times': [],
            'history': []
        }
        
        # Initialize UI
        self.setup_ui()
        
        # Load model in background
        self.load_model_async()
    
    def setup_ui(self):
        """Setup user interface"""
        # Title Bar
        title_frame = tk.Frame(self.root, bg='#667eea', height=80)
        title_frame.pack(fill='x', side='top')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🎯 Plate Detection Desktop App",
            font=('Arial', 24, 'bold'),
            fg='white',
            bg='#667eea'
        )
        title_label.pack(pady=20)
        
        subtitle_label = tk.Label(
            title_frame,
            text="YOLOv11n Epoch 170 | Precision: 81.64% | mAP50: 49.14% | Speed: 1.30ms",
            font=('Arial', 10),
            fg='white',
            bg='#667eea'
        )
        subtitle_label.pack()
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left Panel - Controls & Info
        left_panel = tk.Frame(main_container, bg='white', width=400)
        left_panel.pack(side='left', fill='both', padx=(0, 5))
        left_panel.pack_propagate(False)
        
        self.setup_left_panel(left_panel)
        
        # Right Panel - Image Display & Results
        right_panel = tk.Frame(main_container, bg='white')
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.setup_right_panel(right_panel)
    
    def setup_left_panel(self, parent):
        """Setup left control panel"""
        # Model Status
        status_frame = tk.LabelFrame(parent, text="📊 Model Status", font=('Arial', 12, 'bold'), bg='white', fg='#667eea')
        status_frame.pack(fill='x', padx=10, pady=10)
        
        self.status_label = tk.Label(
            status_frame,
            text="⏳ Loading model...",
            font=('Arial', 10),
            bg='white',
            fg='#f59e0b',
            justify='left'
        )
        self.status_label.pack(anchor='w', padx=10, pady=10)
        
        # Upload Section
        upload_frame = tk.LabelFrame(parent, text="📸 Upload Image", font=('Arial', 12, 'bold'), bg='white', fg='#667eea')
        upload_frame.pack(fill='x', padx=10, pady=10)
        
        self.upload_btn = ttk.Button(
            upload_frame,
            text="📂 Select Image",
            command=self.select_image,
            state='disabled'
        )
        self.upload_btn.pack(fill='x', padx=10, pady=10)
        
        self.detect_btn = ttk.Button(
            upload_frame,
            text="🔍 Detect Plates",
            command=self.detect_plates,
            state='disabled'
        )
        self.detect_btn.pack(fill='x', padx=10, pady=(0, 10))
        
        self.batch_btn = ttk.Button(
            upload_frame,
            text="📁 Batch Detect Folder",
            command=self.batch_detect,
            state='disabled'
        )
        self.batch_btn.pack(fill='x', padx=10, pady=(0, 10))
        
        self.webcam_btn = ttk.Button(
            upload_frame,
            text="📹 Start Webcam Detection",
            command=self.toggle_webcam,
            state='disabled'
        )
        self.webcam_btn.pack(fill='x', padx=10, pady=(0, 10))
        
        # Rotation toggle checkbox
        self.rotation_check = tk.Checkbutton(
            upload_frame,
            text="🔄 Enable Auto-Rotation",
            variable=self.enable_rotation,
            font=('Arial', 9),
            bg='white',
            fg='#667eea'
        )
        self.rotation_check.pack(anchor='w', padx=10, pady=(0, 10))
        
        # Webcam status
        self.webcam_status = tk.Label(
            upload_frame,
            text="",
            font=('Arial', 9),
            bg='white',
            fg='#10b981'
        )
        self.webcam_status.pack(padx=10, pady=(0, 10))
        
        # Statistics Section
        stats_frame = tk.LabelFrame(parent, text="📈 Statistics", font=('Arial', 12, 'bold'), bg='white', fg='#667eea')
        stats_frame.pack(fill='x', padx=10, pady=10)
        
        stats_grid = tk.Frame(stats_frame, bg='white')
        stats_grid.pack(fill='x', padx=10, pady=10)
        
        # Row 1
        tk.Label(stats_grid, text="Images Processed:", font=('Arial', 9), bg='white', fg='#666').grid(row=0, column=0, sticky='w', pady=2)
        self.stat_processed = tk.Label(stats_grid, text="0", font=('Arial', 9, 'bold'), bg='white', fg='#667eea')
        self.stat_processed.grid(row=0, column=1, sticky='e', pady=2)
        
        tk.Label(stats_grid, text="Total Detections:", font=('Arial', 9), bg='white', fg='#666').grid(row=1, column=0, sticky='w', pady=2)
        self.stat_detections = tk.Label(stats_grid, text="0", font=('Arial', 9, 'bold'), bg='white', fg='#667eea')
        self.stat_detections.grid(row=1, column=1, sticky='e', pady=2)
        
        tk.Label(stats_grid, text="Avg Confidence:", font=('Arial', 9), bg='white', fg='#666').grid(row=2, column=0, sticky='w', pady=2)
        self.stat_confidence = tk.Label(stats_grid, text="0%", font=('Arial', 9, 'bold'), bg='white', fg='#667eea')
        self.stat_confidence.grid(row=2, column=1, sticky='e', pady=2)
        
        tk.Label(stats_grid, text="Avg Time:", font=('Arial', 9), bg='white', fg='#666').grid(row=3, column=0, sticky='w', pady=2)
        self.stat_time = tk.Label(stats_grid, text="0ms", font=('Arial', 9, 'bold'), bg='white', fg='#667eea')
        self.stat_time.grid(row=3, column=1, sticky='e', pady=2)
        
        stats_grid.columnconfigure(1, weight=1)
        
        reset_btn = ttk.Button(
            stats_frame,
            text="🔄 Reset Statistics",
            command=self.reset_stats
        )
        reset_btn.pack(fill='x', padx=10, pady=(0, 10))
        
        # History Section
        history_frame = tk.LabelFrame(parent, text="📋 Detection History", font=('Arial', 12, 'bold'), bg='white', fg='#667eea')
        history_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.history_text = scrolledtext.ScrolledText(
            history_frame,
            height=10,
            font=('Consolas', 8),
            bg='#f8f9ff',
            fg='#333'
        )
        self.history_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def setup_right_panel(self, parent):
        """Setup right display panel"""
        # Image Display
        image_frame = tk.LabelFrame(parent, text="🖼️ Detection Result", font=('Arial', 12, 'bold'), bg='white', fg='#667eea')
        image_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.image_canvas = tk.Canvas(image_frame, bg='#f8f9ff', highlightthickness=0)
        self.image_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Detection Details
        details_frame = tk.LabelFrame(parent, text="📍 Detection Details", font=('Arial', 12, 'bold'), bg='white', fg='#667eea')
        details_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.details_text = scrolledtext.ScrolledText(
            details_frame,
            height=8,
            font=('Consolas', 9),
            bg='#f8f9ff',
            fg='#333'
        )
        self.details_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def load_model_async(self):
        """Load model in background thread"""
        def load():
            try:
                # Load YOLO model
                possible_paths = [
                    "best.pt",
                    "best_model_epoch170.pt",
                    "runs/plate_detection/yolov11_ultimate_v1/weights/epoch170.pt"
                ]
                
                for path in possible_paths:
                    if Path(path).exists():
                        self.model = YOLO(path)
                        model_size = Path(path).stat().st_size / (1024*1024)
                        break
                
                # Load rotation detector
                self.rotation_detector = PlateRotationDetector(debug=False)
                
                # Update UI
                self.root.after(0, self.on_model_loaded)
                
            except Exception as e:
                self.root.after(0, lambda: self.on_model_error(str(e)))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def on_model_loaded(self):
        """Called when model is loaded"""
        self.status_label.config(
            text="✅ Model Ready\n✅ Rotation Detector Ready",
            fg='#10b981'
        )
        self.upload_btn.config(state='normal')
        self.detect_btn.config(state='normal')
        self.batch_btn.config(state='normal')
        self.webcam_btn.config(state='normal')
    
    def on_model_error(self, error):
        """Called when model loading fails"""
        self.status_label.config(
            text=f"❌ Error loading model:\n{error}",
            fg='#ef4444'
        )
        messagebox.showerror("Error", f"Failed to load model:\n{error}")
    
    def select_image(self):
        """Select image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.load_and_display_image(file_path)
    
    def load_and_display_image(self, path):
        """Load and display image on canvas"""
        try:
            # Read with OpenCV
            self.current_image = cv2.imread(path)
            if self.current_image is None:
                messagebox.showerror("Error", "Cannot read image file")
                return
            
            # Display original
            self.display_image(self.current_image)
            
            # Clear details
            self.details_text.delete('1.0', tk.END)
            self.details_text.insert('1.0', f"Image loaded: {Path(path).name}\n")
            self.details_text.insert(tk.END, f"Size: {self.current_image.shape[1]}x{self.current_image.shape[0]}\n")
            self.details_text.insert(tk.END, "Click 'Detect Plates' to start detection\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image:\n{str(e)}")
    
    def display_image(self, cv_image, is_result=False):
        """Display OpenCV image on canvas"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600
        
        h, w = rgb_image.shape[:2]
        scale = min(canvas_width / w, canvas_height / h) * 0.95
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to PIL and Tkinter
        pil_image = Image.fromarray(resized)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        # Display on canvas
        self.image_canvas.delete('all')
        x = (canvas_width - new_w) // 2
        y = (canvas_height - new_h) // 2
        self.image_canvas.create_image(x, y, anchor='nw', image=self.tk_image)
    
    def detect_plates(self):
        """Run plate detection"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
        
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        # Disable buttons
        self.detect_btn.config(state='disabled', text="⏳ Detecting...")
        self.root.update()
        
        try:
            start_time = time.time()
            
            # Detect rotation (optional based on checkbox)
            if self.enable_rotation.get():
                corrected_image, angle, rot_confidence = self.rotation_detector.preprocess(self.current_image)
            else:
                corrected_image = self.current_image
                angle = 0
                rot_confidence = "N/A (Disabled)"
            
            # Run detection
            results = self.model.predict(corrected_image, conf=0.25, verbose=False)
            boxes = results[0].boxes
            
            # Parse detections
            detections = []
            for box in boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                detections.append({
                    'confidence': conf,
                    'x1': float(xyxy[0]),
                    'y1': float(xyxy[1]),
                    'x2': float(xyxy[2]),
                    'y2': float(xyxy[3]),
                    'width': float(xyxy[2] - xyxy[0]),
                    'height': float(xyxy[3] - xyxy[1])
                })
            
            # Get annotated image
            annotated = results[0].plot()
            
            # Calculate stats
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats['total_processed'] += 1
            self.stats['total_detections'] += len(detections)
            
            if detections:
                confidences = [d['confidence'] for d in detections]
                self.stats['confidences'].extend(confidences)
            
            self.stats['times'].append(processing_time)
            
            # Add to history
            filename = Path(self.current_image_path).name if self.current_image_path else "Unknown"
            history_entry = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'filename': filename,
                'detections': len(detections),
                'confidence': np.mean([d['confidence'] for d in detections]) if detections else 0,
                'time': processing_time * 1000,
                'rotation': angle
            }
            self.stats['history'].append(history_entry)
            
            # Display results
            self.display_image(annotated, is_result=True)
            self.update_details(detections, angle, rot_confidence, processing_time)
            self.update_stats_display()
            self.update_history_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection error:\n{str(e)}")
        finally:
            self.detect_btn.config(state='normal', text="🔍 Detect Plates")
    
    def batch_detect(self):
        """Batch detect on folder"""
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        
        if not folder_path:
            return
        
        # Get all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(folder_path).glob(ext))
            image_files.extend(Path(folder_path).glob(ext.upper()))
        
        if not image_files:
            messagebox.showwarning("Warning", "No images found in folder!")
            return
        
        # Confirm
        if not messagebox.askyesno("Batch Detection", f"Found {len(image_files)} images.\nStart batch detection?"):
            return
        
        # Disable buttons
        self.batch_btn.config(state='disabled', text="⏳ Processing...")
        self.root.update()
        
        try:
            for i, img_path in enumerate(image_files):
                # Load image
                self.current_image_path = str(img_path)
                self.current_image = cv2.imread(str(img_path))
                
                if self.current_image is not None:
                    self.detect_plates()
                
                # Update progress
                self.status_label.config(
                    text=f"⏳ Processing: {i+1}/{len(image_files)}"
                )
                self.root.update()
            
            messagebox.showinfo("Complete", f"Batch detection complete!\nProcessed {len(image_files)} images")
            
        except Exception as e:
            messagebox.showerror("Error", f"Batch detection error:\n{str(e)}")
        finally:
            self.batch_btn.config(state='normal', text="📁 Batch Detect Folder")
            self.status_label.config(text="✅ Model Ready\n✅ Rotation Detector Ready")
    
    def update_details(self, detections, angle, rot_confidence, proc_time):
        """Update detection details text"""
        self.details_text.delete('1.0', tk.END)
        
        # Header
        self.details_text.insert('1.0', f"{'='*60}\n", 'header')
        self.details_text.insert(tk.END, f"🎯 DETECTION RESULTS\n", 'header')
        self.details_text.insert(tk.END, f"{'='*60}\n\n", 'header')
        
        # Stats
        self.details_text.insert(tk.END, f"📊 Statistics:\n")
        self.details_text.insert(tk.END, f"   • Total Detections: {len(detections)}\n")
        self.details_text.insert(tk.END, f"   • Rotation Angle: {angle}°\n")
        self.details_text.insert(tk.END, f"   • Rotation Confidence: {rot_confidence}\n")
        self.details_text.insert(tk.END, f"   • Processing Time: {proc_time*1000:.2f}ms\n\n")
        
        if detections:
            self.details_text.insert(tk.END, f"📍 Detected Plates:\n")
            for i, det in enumerate(detections, 1):
                self.details_text.insert(tk.END, f"\n   Plate {i}:\n")
                self.details_text.insert(tk.END, f"      Confidence: {det['confidence']:.2%}\n")
                self.details_text.insert(tk.END, f"      Position: ({det['x1']:.0f}, {det['y1']:.0f})\n")
                self.details_text.insert(tk.END, f"      Size: {det['width']:.0f} × {det['height']:.0f} px\n")
        else:
            self.details_text.insert(tk.END, f"❌ No plates detected\n")
    
    def update_stats_display(self):
        """Update statistics display"""
        self.stat_processed.config(text=str(self.stats['total_processed']))
        self.stat_detections.config(text=str(self.stats['total_detections']))
        
        if self.stats['confidences']:
            avg_conf = np.mean(self.stats['confidences'])
            self.stat_confidence.config(text=f"{avg_conf:.2%}")
        
        if self.stats['times']:
            avg_time = np.mean(self.stats['times']) * 1000
            self.stat_time.config(text=f"{avg_time:.2f}ms")
    
    def update_history_display(self):
        """Update history display"""
        self.history_text.delete('1.0', tk.END)
        
        if not self.stats['history']:
            self.history_text.insert('1.0', "No history yet...")
            return
        
        # Show last 20 entries
        for entry in reversed(self.stats['history'][-20:]):
            line = f"{entry['timestamp']} | {entry['filename'][:20]:20} | "
            line += f"Det:{entry['detections']} | Conf:{entry['confidence']:.2%} | "
            line += f"Rot:{entry['rotation']}° | {entry['time']:.1f}ms\n"
            self.history_text.insert(tk.END, line)
    
    def reset_stats(self):
        """Reset all statistics"""
        if messagebox.askyesno("Reset", "Reset all statistics?"):
            self.stats = {
                'total_processed': 0,
                'total_detections': 0,
                'confidences': [],
                'times': [],
                'history': []
            }
            self.update_stats_display()
            self.update_history_display()
    
    def toggle_webcam(self):
        """Toggle webcam detection on/off"""
        if not self.webcam_active:
            self.start_webcam()
        else:
            self.stop_webcam()
    
    def start_webcam(self):
        """Start webcam detection"""
        try:
            # Try to open webcam
            self.webcam_capture = cv2.VideoCapture(0)
            
            if not self.webcam_capture.isOpened():
                messagebox.showerror("Error", "Cannot open webcam!\nMake sure camera is connected.")
                return
            
            # Set resolution
            self.webcam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.webcam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.webcam_active = True
            self.webcam_btn.config(text="⏹️ Stop Webcam")
            self.webcam_status.config(text="🔴 LIVE", fg='#ef4444')
            
            # Disable other buttons
            self.upload_btn.config(state='disabled')
            self.detect_btn.config(state='disabled')
            self.batch_btn.config(state='disabled')
            
            # Start webcam thread
            self.webcam_thread = threading.Thread(target=self.webcam_loop, daemon=True)
            self.webcam_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam:\n{str(e)}")
    
    def stop_webcam(self):
        """Stop webcam detection"""
        self.webcam_active = False
        
        if self.webcam_capture:
            self.webcam_capture.release()
            self.webcam_capture = None
        
        self.webcam_btn.config(text="📹 Start Webcam Detection")
        self.webcam_status.config(text="")
        
        # Re-enable buttons
        self.upload_btn.config(state='normal')
        self.detect_btn.config(state='normal')
        self.batch_btn.config(state='normal')
    
    def webcam_loop(self):
        """Main webcam detection loop"""
        frame_count = 0
        fps_start = time.time()
        fps = 0
        
        while self.webcam_active:
            try:
                ret, frame = self.webcam_capture.read()
                
                if not ret:
                    break
                
                # Detect rotation (optional based on checkbox)
                if self.enable_rotation.get():
                    corrected_frame, angle, rot_confidence = self.rotation_detector.preprocess(frame)
                else:
                    corrected_frame = frame
                    angle = 0
                    rot_confidence = "N/A"
                
                # Run detection every frame (you can skip frames for better performance)
                start_time = time.time()
                results = self.model.predict(corrected_frame, conf=0.25, verbose=False)
                processing_time = time.time() - start_time
                
                # Get annotated frame
                annotated_frame = results[0].plot()
                
                # Calculate FPS
                frame_count += 1
                if frame_count >= 10:
                    elapsed = time.time() - fps_start
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()
                
                # Draw info on frame
                boxes = results[0].boxes
                num_detections = len(boxes)
                
                # Add text overlay
                y_offset = 30
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 35
                
                cv2.putText(annotated_frame, f"Detections: {num_detections}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 35
                
                cv2.putText(annotated_frame, f"Rotation: {angle} deg", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 35
                
                cv2.putText(annotated_frame, f"Time: {processing_time*1000:.1f}ms", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Display on GUI
                self.root.after(0, lambda f=annotated_frame: self.display_image(f))
                
                # Update stats if detection found
                if num_detections > 0:
                    # Update stats
                    self.stats['total_processed'] += 1
                    self.stats['total_detections'] += num_detections
                    
                    confidences = [float(box.conf[0]) for box in boxes]
                    self.stats['confidences'].extend(confidences)
                    self.stats['times'].append(processing_time)
                    
                    # Add to history (limit to avoid spam)
                    if len(self.stats['history']) == 0 or \
                       (time.time() - self.stats.get('last_history_time', 0)) > 2.0:
                        
                        history_entry = {
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'filename': 'WEBCAM',
                            'detections': num_detections,
                            'confidence': np.mean(confidences),
                            'time': processing_time * 1000,
                            'rotation': angle
                        }
                        self.stats['history'].append(history_entry)
                        self.stats['last_history_time'] = time.time()
                        
                        # Update displays
                        self.root.after(0, self.update_stats_display)
                        self.root.after(0, self.update_history_display)
                    
                    # Update details
                    detections = []
                    for box in boxes:
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'confidence': conf,
                            'x1': float(xyxy[0]),
                            'y1': float(xyxy[1]),
                            'x2': float(xyxy[2]),
                            'y2': float(xyxy[3]),
                            'width': float(xyxy[2] - xyxy[0]),
                            'height': float(xyxy[3] - xyxy[1])
                        })
                    
                    self.root.after(0, lambda: self.update_details(detections, angle, rot_confidence, processing_time))
                
                # Small delay to not overwhelm GUI
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Webcam error: {e}")
                break
        
        # Cleanup
        if self.webcam_capture:
            self.webcam_capture.release()



def main():
    root = tk.Tk()
    app = PlateDetectionDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
🖼️ Test Model with Image Upload
Upload gambar dan lihat hasil deteksi dari model yang sudah di-train
"""

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO
from pathlib import Path
import time

class ImageTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧪 Test Plate Detection Model")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a2e')
        
        # State
        self.model = None
        self.current_image = None
        self.photo = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup UI"""
        
        # Top bar - Model selector & Upload
        top_frame = tk.Frame(self.root, bg='#16213e', height=100)
        top_frame.pack(fill='x', padx=10, pady=10)
        top_frame.pack_propagate(False)
        
        # Model selector
        tk.Label(
            top_frame,
            text="🤖 Select Model:",
            font=('Segoe UI', 11),
            bg='#16213e',
            fg='white'
        ).pack(side='left', padx=10, pady=30)
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            top_frame,
            textvariable=self.model_var,
            font=('Segoe UI', 10),
            width=50,
            state='readonly'
        )
        self.model_combo.pack(side='left', padx=5)
        
        # Scan for models
        self.scan_models()
        
        # Load model button
        load_btn = tk.Button(
            top_frame,
            text="📥 Load Model",
            font=('Segoe UI', 10, 'bold'),
            bg='#0f3460',
            fg='white',
            activebackground='#1a4d7a',
            cursor='hand2',
            padx=15,
            pady=10,
            command=self.load_model
        )
        load_btn.pack(side='left', padx=10)
        
        # Upload button
        upload_btn = tk.Button(
            top_frame,
            text="📁 Upload Image",
            font=('Segoe UI', 10, 'bold'),
            bg='#006400',
            fg='white',
            activebackground='#008000',
            cursor='hand2',
            padx=15,
            pady=10,
            command=self.upload_image
        )
        upload_btn.pack(side='left', padx=5)
        
        # Status label
        self.status_label = tk.Label(
            top_frame,
            text="⚠️ Load model first",
            font=('Segoe UI', 10),
            bg='#16213e',
            fg='#ff9900'
        )
        self.status_label.pack(side='right', padx=20)
        
        # Main content - Image display
        main_frame = tk.Frame(self.root, bg='#16213e')
        main_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Canvas
        canvas_frame = tk.Frame(main_frame, bg='#0f3460')
        canvas_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.info_label = tk.Label(
            canvas_frame,
            text="👆 Upload an image to test detection",
            font=('Segoe UI', 12),
            bg='#0f3460',
            fg='#00d4ff'
        )
        self.info_label.pack(pady=10)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            bg='#0f3460',
            highlightthickness=2,
            highlightbackground='#00d4ff'
        )
        self.canvas.pack(fill='both', expand=True, pady=(0, 10))
        
        # Results panel
        result_frame = tk.Frame(main_frame, bg='#16213e', width=300)
        result_frame.pack(side='right', fill='y', padx=(10, 0))
        result_frame.pack_propagate(False)
        
        tk.Label(
            result_frame,
            text="📊 Detection Results",
            font=('Segoe UI', 12, 'bold'),
            bg='#16213e',
            fg='white'
        ).pack(pady=10)
        
        # Results text
        self.results_text = tk.Text(
            result_frame,
            font=('Consolas', 9),
            bg='#0f3460',
            fg='white',
            wrap='word',
            height=30
        )
        self.results_text.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
    def scan_models(self):
        """Scan for available models"""
        models = []
        
        # Check common locations
        locations = [
            Path('runs/quick_test/plate_color_50ep/weights/best.pt'),
            Path('runs/quick_test/plate_color_50ep/weights/last.pt'),
            Path('runs/plate_detection/yolov11_ultimate_v1/weights/best.pt'),
            Path('best.pt'),
            Path('yolo11n.pt'),
        ]
        
        for loc in locations:
            if loc.exists():
                models.append(str(loc))
        
        # Also scan runs directory
        runs_dir = Path('runs')
        if runs_dir.exists():
            for weight_file in runs_dir.rglob('weights/best.pt'):
                model_path = str(weight_file)
                if model_path not in models:
                    models.append(model_path)
        
        if not models:
            models = ['No models found - train first!']
        
        self.model_combo['values'] = models
        if models:
            self.model_combo.current(0)
    
    def load_model(self):
        """Load selected model"""
        model_path = self.model_var.get()
        
        if not model_path or model_path == 'No models found - train first!':
            self.status_label.config(
                text="❌ No model selected",
                fg='#ff0000'
            )
            return
        
        try:
            self.status_label.config(
                text="⏳ Loading model...",
                fg='#ffff00'
            )
            self.root.update()
            
            self.model = YOLO(model_path)
            
            self.status_label.config(
                text=f"✅ Model loaded: {Path(model_path).name}",
                fg='#00ff00'
            )
            
            # Display model info
            self.results_text.delete('1.0', 'end')
            self.results_text.insert('end', f"✅ Model Loaded\n")
            self.results_text.insert('end', f"{'='*40}\n\n")
            self.results_text.insert('end', f"Path: {model_path}\n")
            self.results_text.insert('end', f"Size: {Path(model_path).stat().st_size / 1024 / 1024:.2f} MB\n\n")
            self.results_text.insert('end', f"📋 Model Info:\n")
            self.results_text.insert('end', f"  • Type: {self.model.task}\n")
            self.results_text.insert('end', f"  • Classes: {len(self.model.names)}\n")
            self.results_text.insert('end', f"  • Names: {self.model.names}\n\n")
            self.results_text.insert('end', f"🎯 Ready to test!\n")
            self.results_text.insert('end', f"Upload an image to detect plates.\n")
            
        except Exception as e:
            self.status_label.config(
                text=f"❌ Failed to load model",
                fg='#ff0000'
            )
            self.results_text.delete('1.0', 'end')
            self.results_text.insert('end', f"❌ ERROR\n")
            self.results_text.insert('end', f"{'='*40}\n\n")
            self.results_text.insert('end', f"{str(e)}\n")
    
    def upload_image(self):
        """Upload and process image"""
        if not self.model:
            self.status_label.config(
                text="❌ Load model first!",
                fg='#ff0000'
            )
            return
        
        # Select image
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Load image
            self.current_image = Image.open(file_path)
            
            # Run detection
            self.status_label.config(
                text="⏳ Detecting...",
                fg='#ffff00'
            )
            self.root.update()
            
            start_time = time.time()
            results = self.model(file_path, conf=0.1, iou=0.5, verbose=False)
            inference_time = (time.time() - start_time) * 1000
            
            # Draw results
            img_with_boxes = self.current_image.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Draw box
                    draw.rectangle(
                        [(x1, y1), (x2, y2)],
                        outline='#00ff00',
                        width=3
                    )
                    
                    # Draw label
                    label = f"{self.model.names[cls]} {conf:.2f}"
                    draw.text(
                        (x1, y1 - 25),
                        label,
                        fill='#00ff00',
                        font=font
                    )
                    
                    detections.append({
                        'class': self.model.names[cls],
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
            
            # Display image
            self.display_image(img_with_boxes)
            
            # Update results
            self.results_text.delete('1.0', 'end')
            self.results_text.insert('end', f"🎯 Detection Results\n")
            self.results_text.insert('end', f"{'='*40}\n\n")
            self.results_text.insert('end', f"📷 Image: {Path(file_path).name}\n")
            self.results_text.insert('end', f"⏱️ Inference: {inference_time:.2f}ms\n")
            self.results_text.insert('end', f"🔍 Detections: {len(detections)}\n\n")
            
            if detections:
                for i, det in enumerate(detections):
                    self.results_text.insert('end', f"Detection #{i+1}:\n")
                    self.results_text.insert('end', f"  • Class: {det['class']}\n")
                    self.results_text.insert('end', f"  • Confidence: {det['confidence']:.4f}\n")
                    x1, y1, x2, y2 = det['bbox']
                    self.results_text.insert('end', f"  • BBox: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})\n")
                    self.results_text.insert('end', f"  • Size: {x2-x1:.0f}x{y2-y1:.0f}px\n\n")
            else:
                self.results_text.insert('end', "⚠️ No detections found.\n\n")
                self.results_text.insert('end', "Try:\n")
                self.results_text.insert('end', "  • Using a different image\n")
                self.results_text.insert('end', "  • Training with more data\n")
                self.results_text.insert('end', "  • Lowering confidence threshold\n")
            
            self.status_label.config(
                text=f"✅ Detected {len(detections)} plates in {inference_time:.0f}ms",
                fg='#00ff00'
            )
            
        except Exception as e:
            self.status_label.config(
                text="❌ Detection failed",
                fg='#ff0000'
            )
            self.results_text.delete('1.0', 'end')
            self.results_text.insert('end', f"❌ ERROR\n")
            self.results_text.insert('end', f"{'='*40}\n\n")
            self.results_text.insert('end', f"{str(e)}\n")
    
    def display_image(self, img):
        """Display image on canvas"""
        # Get canvas size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 800, 600
        
        # Resize to fit
        scale_x = (canvas_w - 40) / img.width
        scale_y = (canvas_h - 40) / img.height
        scale = min(scale_x, scale_y, 1.0)
        
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Center
        offset_x = (canvas_w - new_w) // 2
        offset_y = (canvas_h - new_h) // 2
        
        # Display
        self.photo = ImageTk.PhotoImage(img_resized)
        self.canvas.delete('all')
        self.canvas.create_image(
            offset_x, offset_y,
            image=self.photo,
            anchor='nw'
        )
        
        # Update info
        self.info_label.config(
            text=f"📷 {img.width}x{img.height}px | Resized to {new_w}x{new_h}px"
        )


def main():
    root = tk.Tk()
    app = ImageTestApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

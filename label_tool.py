#!/usr/bin/env python
"""
🎨 Modern Labeling Tool - Roboflow Style
Upload gambar, labeling, pilih destination dengan GUI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from pathlib import Path
import shutil
import json
from datetime import datetime


class ModernLabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("🏷️ Plate Labeling Tool - Roboflow Style")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a2e')
        
        # State
        self.images = []
        self.current_idx = 0
        self.boxes = []  # Current image boxes
        self.all_labels = {}  # All labels dict {image_path: [boxes]}
        self.drawing = False
        self.start_x = None
        self.start_y = None
        self.temp_box = None
        
        # Canvas state
        self.canvas_image = None
        self.photo = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Dataset destinations
        self.destinations = {
            'Plate Detection YOLO - Train': 'dataset/plate_detection_yolo/train',
            'Plate Detection YOLO - Val': 'dataset/plate_detection_yolo/val',
            'Plate Detection Augmented - Train': 'dataset/plate_detection_augmented/train',
            'Plate Detection Augmented - Val': 'dataset/plate_detection_augmented/val',
            'Custom Path...': None
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup modern UI layout"""
        
        # ====================================================================
        # TOP BAR - Upload & Destination
        # ====================================================================
        top_frame = tk.Frame(self.root, bg='#16213e', height=80)
        top_frame.pack(fill='x', padx=10, pady=10)
        top_frame.pack_propagate(False)
        
        # Upload button
        upload_btn = tk.Button(
            top_frame, 
            text="📁 Upload Images",
            font=('Segoe UI', 12, 'bold'),
            bg='#0f3460',
            fg='white',
            activebackground='#1a4d7a',
            activeforeground='white',
            cursor='hand2',
            padx=20,
            pady=10,
            command=self.upload_images
        )
        upload_btn.pack(side='left', padx=10, pady=15)
        
        # Destination selector
        dest_frame = tk.Frame(top_frame, bg='#16213e')
        dest_frame.pack(side='left', padx=20, pady=15)
        
        tk.Label(
            dest_frame, 
            text="🎯 Save to:", 
            font=('Segoe UI', 10),
            bg='#16213e',
            fg='white'
        ).pack(side='left', padx=5)
        
        self.dest_var = tk.StringVar(value=list(self.destinations.keys())[0])
        dest_combo = ttk.Combobox(
            dest_frame,
            textvariable=self.dest_var,
            values=list(self.destinations.keys()),
            font=('Segoe UI', 10),
            width=35,
            state='readonly'
        )
        dest_combo.pack(side='left', padx=5)
        dest_combo.bind('<<ComboboxSelected>>', self.on_destination_change)
        
        # Stats
        self.stats_label = tk.Label(
            top_frame,
            text="📊 0 images | 0 labeled",
            font=('Segoe UI', 10),
            bg='#16213e',
            fg='#00d4ff'
        )
        self.stats_label.pack(side='right', padx=20)
        
        # ====================================================================
        # MAIN CONTENT - Split view
        # ====================================================================
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Left panel - Image list
        left_panel = tk.Frame(main_frame, bg='#16213e', width=300)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        tk.Label(
            left_panel,
            text="📚 Image List",
            font=('Segoe UI', 12, 'bold'),
            bg='#16213e',
            fg='white'
        ).pack(pady=10)
        
        # Image listbox with scrollbar
        list_frame = tk.Frame(left_panel, bg='#16213e')
        list_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.image_listbox = tk.Listbox(
            list_frame,
            font=('Consolas', 9),
            bg='#0f3460',
            fg='white',
            selectbackground='#00d4ff',
            selectforeground='black',
            yscrollcommand=scrollbar.set,
            activestyle='none'
        )
        self.image_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.image_listbox.yview)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        # Right panel - Canvas + Controls
        right_panel = tk.Frame(main_frame, bg='#16213e')
        right_panel.pack(side='left', fill='both', expand=True)
        
        # Canvas frame
        canvas_frame = tk.Frame(right_panel, bg='#0f3460')
        canvas_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Info label
        self.info_label = tk.Label(
            canvas_frame,
            text="👆 Upload images to start labeling",
            font=('Segoe UI', 11),
            bg='#0f3460',
            fg='#00d4ff'
        )
        self.info_label.pack(pady=5)
        
        # Canvas
        self.canvas = tk.Canvas(
            canvas_frame,
            bg='#0f3460',
            highlightthickness=2,
            highlightbackground='#00d4ff'
        )
        self.canvas.pack(fill='both', expand=True, pady=(0, 10))
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        
        # ====================================================================
        # BOTTOM BAR - Controls & Keyboard shortcuts
        # ====================================================================
        bottom_frame = tk.Frame(self.root, bg='#16213e', height=100)
        bottom_frame.pack(fill='x', padx=10, pady=(0, 10))
        bottom_frame.pack_propagate(False)
        
        # Control buttons
        btn_frame = tk.Frame(bottom_frame, bg='#16213e')
        btn_frame.pack(pady=15)
        
        buttons = [
            ('⬅️ Previous (P)', self.prev_image, '#0f3460'),
            ('🗑️ Delete Box (D)', self.delete_last_box, '#8b0000'),
            ('🔄 Clear All (C)', self.clear_boxes, '#8b4513'),
            ('💾 Save & Next (S)', self.save_and_next, '#006400'),
            ('➡️ Next (N)', self.next_image, '#0f3460'),
        ]
        
        for text, cmd, color in buttons:
            btn = tk.Button(
                btn_frame,
                text=text,
                font=('Segoe UI', 10, 'bold'),
                bg=color,
                fg='white',
                activebackground=self.darken_color(color),
                activeforeground='white',
                cursor='hand2',
                padx=15,
                pady=8,
                command=cmd
            )
            btn.pack(side='left', padx=5)
        
        # Keyboard shortcuts hint
        shortcuts_text = "⌨️  Shortcuts: [P] Previous | [N] Next | [S] Save & Next | [D] Delete Box | [C] Clear All | [Q] Quit"
        tk.Label(
            bottom_frame,
            text=shortcuts_text,
            font=('Consolas', 9),
            bg='#16213e',
            fg='#888888'
        ).pack(pady=(5, 0))
        
        # Bind keyboard
        self.root.bind('<p>', lambda e: self.prev_image())
        self.root.bind('<n>', lambda e: self.next_image())
        self.root.bind('<s>', lambda e: self.save_and_next())
        self.root.bind('<d>', lambda e: self.delete_last_box())
        self.root.bind('<c>', lambda e: self.clear_boxes())
        self.root.bind('<q>', lambda e: self.quit_app())
        
    def darken_color(self, hex_color):
        """Darken hex color for hover effect"""
        rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        darker = tuple(max(0, int(c * 0.7)) for c in rgb)
        return f'#{darker[0]:02x}{darker[1]:02x}{darker[2]:02x}'
    
    def upload_images(self):
        """Upload multiple images"""
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not files:
            return
        
        # Add to list
        for file in files:
            if file not in self.images:
                self.images.append(file)
                
                # Add to listbox
                filename = Path(file).name
                status = "✅" if file in self.all_labels else "⭕"
                self.image_listbox.insert('end', f"{status} {filename}")
        
        self.update_stats()
        
        if self.current_idx == 0 and len(self.images) > 0:
            self.load_image(0)
    
    def on_destination_change(self, event=None):
        """Handle destination change"""
        dest_name = self.dest_var.get()
        
        if dest_name == 'Custom Path...':
            custom_path = filedialog.askdirectory(title="Select Destination Folder")
            if custom_path:
                self.destinations['Custom Path...'] = custom_path
                messagebox.showinfo("✅ Destination Set", f"Labels will be saved to:\n{custom_path}")
            else:
                # Reset to previous
                self.dest_var.set(list(self.destinations.keys())[0])
    
    def on_image_select(self, event):
        """Handle image selection from list"""
        selection = self.image_listbox.curselection()
        if selection:
            idx = selection[0]
            self.load_image(idx)
    
    def load_image(self, idx):
        """Load image to canvas"""
        if idx < 0 or idx >= len(self.images):
            return
        
        self.current_idx = idx
        image_path = self.images[idx]
        
        # Load image
        try:
            img = Image.open(image_path)
            
            # Resize to fit canvas
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            
            if canvas_w <= 1 or canvas_h <= 1:
                canvas_w, canvas_h = 1000, 600
            
            # Calculate scale
            scale_x = (canvas_w - 40) / img.width
            scale_y = (canvas_h - 40) / img.height
            self.scale = min(scale_x, scale_y, 1.0)
            
            new_w = int(img.width * self.scale)
            new_h = int(img.height * self.scale)
            
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center image
            self.offset_x = (canvas_w - new_w) // 2
            self.offset_y = (canvas_h - new_h) // 2
            
            # Load existing boxes
            self.boxes = self.all_labels.get(image_path, []).copy()
            
            # Draw
            self.photo = ImageTk.PhotoImage(img_resized)
            self.canvas.delete('all')
            self.canvas_image = self.canvas.create_image(
                self.offset_x, self.offset_y, 
                image=self.photo, 
                anchor='nw'
            )
            
            # Draw existing boxes
            self.redraw_boxes()
            
            # Update info
            filename = Path(image_path).name
            box_count = len(self.boxes)
            self.info_label.config(
                text=f"📷 {filename} | Image {idx+1}/{len(self.images)} | 🏷️ {box_count} boxes"
            )
            
            # Highlight in listbox
            self.image_listbox.selection_clear(0, 'end')
            self.image_listbox.selection_set(idx)
            self.image_listbox.see(idx)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
    
    def on_mouse_down(self, event):
        """Start drawing box"""
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
    
    def on_mouse_drag(self, event):
        """Draw box while dragging"""
        if not self.drawing:
            return
        
        # Delete previous temp box
        if self.temp_box:
            self.canvas.delete(self.temp_box)
        
        # Draw new temp box
        self.temp_box = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='#00ff00',
            width=3,
            dash=(5, 5)
        )
    
    def on_mouse_up(self, event):
        """Finish drawing box"""
        if not self.drawing:
            return
        
        self.drawing = False
        
        # Delete temp box
        if self.temp_box:
            self.canvas.delete(self.temp_box)
            self.temp_box = None
        
        # Get box coordinates (relative to image)
        x1 = min(self.start_x, event.x) - self.offset_x
        y1 = min(self.start_y, event.y) - self.offset_y
        x2 = max(self.start_x, event.x) - self.offset_x
        y2 = max(self.start_y, event.y) - self.offset_y
        
        # Validate box
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            return  # Too small
        
        # Convert to normalized YOLO format
        img_w = self.photo.width()
        img_h = self.photo.height()
        
        # Clamp to image bounds
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        
        # YOLO format (x_center, y_center, width, height) - normalized
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        
        # Add box
        self.boxes.append([0, x_center, y_center, width, height])
        
        # Redraw
        self.redraw_boxes()
        
        # Update info
        self.info_label.config(
            text=f"📷 {Path(self.images[self.current_idx]).name} | "
                 f"Image {self.current_idx+1}/{len(self.images)} | "
                 f"🏷️ {len(self.boxes)} boxes"
        )
    
    def redraw_boxes(self):
        """Redraw all boxes on canvas"""
        # Clear old boxes
        self.canvas.delete('box')
        self.canvas.delete('label')
        
        if not self.photo:
            return
        
        img_w = self.photo.width()
        img_h = self.photo.height()
        
        for i, box in enumerate(self.boxes):
            # Convert from YOLO to pixel coords
            _, x_center, y_center, width, height = box
            
            x1 = (x_center - width/2) * img_w + self.offset_x
            y1 = (y_center - height/2) * img_h + self.offset_y
            x2 = (x_center + width/2) * img_w + self.offset_x
            y2 = (y_center + height/2) * img_h + self.offset_y
            
            # Draw box
            color = '#00ff00' if i == len(self.boxes) - 1 else '#00d4ff'
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color,
                width=2,
                tags='box'
            )
            
            # Draw label
            self.canvas.create_text(
                x1, y1 - 5,
                text=f"Plate #{i+1}",
                fill=color,
                font=('Segoe UI', 9, 'bold'),
                anchor='sw',
                tags='label'
            )
    
    def delete_last_box(self):
        """Delete last drawn box"""
        if self.boxes:
            self.boxes.pop()
            self.redraw_boxes()
            self.info_label.config(
                text=f"📷 {Path(self.images[self.current_idx]).name} | "
                     f"Image {self.current_idx+1}/{len(self.images)} | "
                     f"🏷️ {len(self.boxes)} boxes"
            )
    
    def clear_boxes(self):
        """Clear all boxes"""
        if self.boxes and messagebox.askyesno("Clear All", "Delete all boxes on this image?"):
            self.boxes = []
            self.redraw_boxes()
            self.info_label.config(
                text=f"📷 {Path(self.images[self.current_idx]).name} | "
                     f"Image {self.current_idx+1}/{len(self.images)} | "
                     f"🏷️ 0 boxes"
            )
    
    def save_and_next(self):
        """Save labels and move to next"""
        self.save_current()
        self.next_image()
    
    def save_current(self):
        """Save current image labels"""
        if self.current_idx >= len(self.images):
            return
        
        image_path = self.images[self.current_idx]
        
        # Save to dict
        self.all_labels[image_path] = self.boxes.copy()
        
        # Update listbox status
        filename = Path(image_path).name
        status = "✅" if self.boxes else "⭕"
        self.image_listbox.delete(self.current_idx)
        self.image_listbox.insert(self.current_idx, f"{status} {filename}")
        
        # Save to destination
        self.save_to_destination(image_path, self.boxes)
        
        self.update_stats()
    
    def save_to_destination(self, image_path, boxes):
        """Save labels and image to selected destination"""
        dest_name = self.dest_var.get()
        dest_path = self.destinations.get(dest_name)
        
        if not dest_path:
            return
        
        dest_path = Path(dest_path)
        labels_dir = dest_path / 'labels'
        images_dir = dest_path / 'images'
        
        # Create dirs
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Save label file
        img_path = Path(image_path)
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        with open(label_path, 'w') as f:
            for box in boxes:
                f.write(' '.join(map(str, box)) + '\n')
        
        # Copy image
        dest_image = images_dir / img_path.name
        if not dest_image.exists():
            shutil.copy(image_path, dest_image)
    
    def prev_image(self):
        """Previous image"""
        if self.current_idx > 0:
            self.save_current()
            self.load_image(self.current_idx - 1)
    
    def next_image(self):
        """Next image"""
        if self.current_idx < len(self.images) - 1:
            self.save_current()
            self.load_image(self.current_idx + 1)
    
    def update_stats(self):
        """Update statistics"""
        total = len(self.images)
        labeled = sum(1 for img in self.images if img in self.all_labels and self.all_labels[img])
        self.stats_label.config(text=f"📊 {total} images | {labeled} labeled")
    
    def quit_app(self):
        """Quit application"""
        if messagebox.askyesno("Quit", "Save current progress and quit?"):
            self.save_current()
            self.root.destroy()


def main():
    root = tk.Tk()
    app = ModernLabelingTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()

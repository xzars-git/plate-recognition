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
        self.original_photo = None  # Store original for zooming
        self.scale = 1.0
        self.zoom_level = 1.0  # Zoom multiplier
        self.offset_x = 0
        self.offset_y = 0
        
        # Drawing mode
        self.draw_mode = 'box'  # 'box' or 'polygon'
        self.polygon_points = []  # Points for polygon mode
        self.temp_polygon_items = []  # Temporary visual items
        
        # Pan/drag state
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.image_offset_x = 0
        self.image_offset_y = 0
        
        # Box/polygon dragging state
        self.dragging_shape = False
        self.selected_box_idx = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        
        # Auto-export color crops toggle
        self.auto_export_crops = tk.BooleanVar(value=True)
        self.color_export_dir = Path('dataset/plate_colors')
        
        # Dataset destinations
        self.destinations = {
            'Plate Detection (Color) - Train': 'dataset/plate_detection_color/train',
            'Plate Detection (Color) - Val': 'dataset/plate_detection_color/val',
            'Plate Detection YOLO - Train': 'dataset/plate_detection_yolo/train',
            'Plate Detection YOLO - Val': 'dataset/plate_detection_yolo/val',
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
        
        # Drawing mode toggle
        mode_frame = tk.Frame(top_frame, bg='#16213e')
        mode_frame.pack(side='left', padx=10)
        
        tk.Label(
            mode_frame,
            text="✏️ Mode:",
            font=('Segoe UI', 10),
            bg='#16213e',
            fg='white'
        ).pack(side='left', padx=5)
        
        self.mode_var = tk.StringVar(value='box')
        mode_box = tk.Radiobutton(
            mode_frame,
            text="□ Box",
            variable=self.mode_var,
            value='box',
            font=('Segoe UI', 9),
            bg='#16213e',
            fg='white',
            selectcolor='#0f3460',
            activebackground='#16213e',
            activeforeground='white',
            command=lambda: self.set_draw_mode('box')
        )
        mode_box.pack(side='left', padx=2)
        
        mode_poly = tk.Radiobutton(
            mode_frame,
            text="⬟ Polygon",
            variable=self.mode_var,
            value='polygon',
            font=('Segoe UI', 9),
            bg='#16213e',
            fg='white',
            selectcolor='#0f3460',
            activebackground='#16213e',
            activeforeground='white',
            command=lambda: self.set_draw_mode('polygon')
        )
        mode_poly.pack(side='left', padx=2)
        
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
        
        # Plate color selector
        tk.Label(
            dest_frame,
            text="🎨 Warna Plat:",
            font=('Segoe UI', 10),
            bg='#16213e',
            fg='white'
        ).pack(side='left', padx=10)
        
        self.color_values = ['white', 'black', 'red', 'yellow']
        self.color_labels = ['⚪ Putih', '⚫ Hitam', '🔴 Merah', '🟡 Kuning']
        self.color_var = tk.StringVar(value=self.color_values[0])
        color_combo = ttk.Combobox(
            dest_frame,
            textvariable=self.color_var,
            values=self.color_labels,
            font=('Segoe UI', 10),
            width=12,
            state='readonly'
        )
        color_combo.pack(side='left', padx=5)
        color_combo.current(0)
        
        # Auto-export crops checkbox
        auto_export_cb = tk.Checkbutton(
            dest_frame,
            text="📦 Auto-export crops",
            variable=self.auto_export_crops,
            font=('Segoe UI', 9),
            bg='#16213e',
            fg='white',
            selectcolor='#0f3460',
            activebackground='#16213e',
            activeforeground='white'
        )
        auto_export_cb.pack(side='left', padx=10)
        
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
        self.canvas.bind('<Button-3>', self.on_pan_start)      # Right click to pan
        self.canvas.bind('<B3-Motion>', self.on_pan_move)      # Drag to pan
        self.canvas.bind('<ButtonRelease-3>', self.on_pan_end) # Release pan
        self.canvas.bind('<Control-Button-1>', self.on_shape_drag_start)  # Ctrl+Left to drag shape
        self.canvas.bind('<Control-B1-Motion>', self.on_shape_drag_move)  # Drag shape
        self.canvas.bind('<Control-ButtonRelease-1>', self.on_shape_drag_end)  # Release shape
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)  # Zoom
        self.canvas.bind('<Button-4>', self.on_mouse_wheel)    # Linux scroll up
        self.canvas.bind('<Button-5>', self.on_mouse_wheel)    # Linux scroll down
        
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
        shortcuts_text = "⌨️  Shortcuts: [1-4] Warna | [P] Previous | [N] Next | [S] Save & Next | [D] Delete Box | [C] Clear All | [Q] Quit | [F/Enter] Finish Polygon | [Esc] Cancel"
        tk.Label(
            bottom_frame,
            text=shortcuts_text,
            font=('Consolas', 9),
            bg='#16213e',
            fg='#888888'
        ).pack(pady=(5, 0))
        
        # Color shortcuts hint
        color_hint = "🎨 Warna: [1] ⚪ Putih | [2] ⚫ Hitam | [3] 🔴 Merah | [4] 🟡 Kuning | 🔍 Zoom: Mouse Wheel | 🖐️ Pan: Right+Drag | 📦 Move Box: Ctrl+Drag"
        tk.Label(
            bottom_frame,
            text=color_hint,
            font=('Consolas', 9),
            bg='#16213e',
            fg='#00d4ff'
        ).pack(pady=(0, 5))
        
        # Bind keyboard
        self.root.bind('<p>', lambda e: self.prev_image())
        self.root.bind('<n>', lambda e: self.next_image())
        self.root.bind('<s>', lambda e: self.save_and_next())
        self.root.bind('<d>', lambda e: self.delete_last_box())
        self.root.bind('<c>', lambda e: self.clear_boxes())
        self.root.bind('<q>', lambda e: self.quit_app())
        self.root.bind('<Return>', lambda e: self.finish_polygon())
        self.root.bind('<Escape>', lambda e: self.cancel_polygon())
        self.root.bind('<f>', lambda e: self.finish_polygon())  # [F] to close polygon
        
        # Bind color shortcuts (1-4)
        self.root.bind('1', lambda e: self.set_color(0))
        self.root.bind('2', lambda e: self.set_color(1))
        self.root.bind('3', lambda e: self.set_color(2))
        self.root.bind('4', lambda e: self.set_color(3))
        
    def set_draw_mode(self, mode):
        """Set drawing mode (box or polygon)"""
        self.draw_mode = mode
        self.cancel_polygon()  # Clear any ongoing polygon
        if hasattr(self, 'info_label'):
            mode_text = "📦 Box Mode" if mode == 'box' else "⬟ Polygon Mode (Click points, click first dot to close)"
            current = self.info_label.cget('text')
            self.info_label.config(text=f"{mode_text} | {current}")
            self.root.after(3000, lambda: self.load_image(self.current_idx))
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        if not self.original_photo:
            return
        
        # Get zoom direction
        if event.num == 4 or event.delta > 0:
            zoom_factor = 1.1  # Zoom in
        elif event.num == 5 or event.delta < 0:
            zoom_factor = 0.9  # Zoom out
        else:
            return
        
        # Update zoom level
        new_zoom = self.zoom_level * zoom_factor
        if 0.5 <= new_zoom <= 5.0:  # Limit zoom range
            self.zoom_level = new_zoom
            self.refresh_canvas()
    
    def refresh_canvas(self):
        """Refresh canvas with current zoom level"""
        if not self.original_photo or self.current_idx >= len(self.images):
            return
        
        # Get original image
        image_path = self.images[self.current_idx]
        img = Image.open(image_path)
        
        # Apply zoom
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 1000, 600
        
        # Calculate scale with zoom
        scale_x = (canvas_w - 40) / img.width * self.zoom_level
        scale_y = (canvas_h - 40) / img.height * self.zoom_level
        self.scale = min(scale_x, scale_y)
        
        new_w = int(img.width * self.scale)
        new_h = int(img.height * self.scale)
        
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Apply pan offset
        self.offset_x = (canvas_w - new_w) // 2 + self.image_offset_x
        self.offset_y = (canvas_h - new_h) // 2 + self.image_offset_y
        
        # Update display
        self.photo = ImageTk.PhotoImage(img_resized)
        self.canvas.delete('all')
        self.canvas_image = self.canvas.create_image(
            self.offset_x, self.offset_y,
            image=self.photo,
            anchor='nw'
        )
        
        # Redraw boxes and polygon points
        self.redraw_boxes()
        self.redraw_polygon_points()
        
        # Update info with zoom level
        filename = Path(image_path).name
        box_count = len(self.boxes)
        zoom_pct = int(self.zoom_level * 100)
        self.info_label.config(
            text=f"📷 {filename} | Image {self.current_idx+1}/{len(self.images)} | 🏷️ {box_count} boxes | 🔍 {zoom_pct}%"
        )
    
    def finish_polygon(self):
        """Finish drawing polygon - SAVE ORIGINAL SHAPE (not bounding box)"""
        if self.draw_mode != 'polygon' or len(self.polygon_points) < 3:
            return
        
        # Convert canvas coordinates to relative image coordinates
        img_w = self.photo.width()
        img_h = self.photo.height()
        
        normalized_points = []
        for x, y in self.polygon_points:
            # Remove offsets to get image-relative coordinates
            img_x = x - self.offset_x
            img_y = y - self.offset_y
            
            # Normalize to 0-1 range
            norm_x = img_x / img_w
            norm_y = img_y / img_h
            
            # Clamp to valid range
            norm_x = max(0, min(1, norm_x))
            norm_y = max(0, min(1, norm_y))
            
            normalized_points.append([norm_x, norm_y])
        
        # Get selected color
        color_idx = self.color_labels.index(self.color_var.get()) if hasattr(self, 'color_labels') else 0
        color = self.color_values[color_idx] if hasattr(self, 'color_values') else 'white'
        
        # Store as polygon with original shape
        self.boxes.append({
            'type': 'polygon',
            'points': normalized_points,
            'color': color
        })
        
        # Auto-save
        if self.current_idx < len(self.images):
            image_path = self.images[self.current_idx]
            self.all_labels[image_path] = self.boxes.copy()
            self.save_to_destination(image_path, self.boxes)
            
            filename = Path(image_path).name
            status = "✅" if self.boxes else "⭕"
            self.image_listbox.delete(self.current_idx)
            self.image_listbox.insert(self.current_idx, f"{status} {filename}")
            self.image_listbox.selection_set(self.current_idx)
            self.update_stats()
        
        # Clear polygon and redraw
        self.cancel_polygon()
        self.redraw_boxes()
        
        # Update info
        zoom_pct = int(self.zoom_level * 100)
        self.info_label.config(
            text=f"📷 {Path(self.images[self.current_idx]).name} | "
                 f"Image {self.current_idx+1}/{len(self.images)} | "
                 f"🏷️ {len(self.boxes)} boxes | 🔍 {zoom_pct}% | ✅ Auto-saved (Polygon)"
        )
    
    def cancel_polygon(self):
        """Cancel polygon drawing"""
        self.polygon_points = []
        for item in self.temp_polygon_items:
            self.canvas.delete(item)
        self.temp_polygon_items = []
    
    def darken_color(self, hex_color):
        """Darken hex color for hover effect"""
        rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        darker = tuple(max(0, int(c * 0.7)) for c in rgb)
        return f'#{darker[0]:02x}{darker[1]:02x}{darker[2]:02x}'
    
    def set_color(self, idx):
        """Set plate color by index (0-3)"""
        if 0 <= idx < len(self.color_labels):
            self.color_var.set(self.color_labels[idx])
            # Update info to show selected color (no popup)
            color_indo = self.color_labels[idx]
            if hasattr(self, 'info_label'):
                current_text = self.info_label.cget('text')
                self.info_label.config(text=f"🎨 {color_indo} dipilih | {current_text}")
                # Reset after 2 seconds
                self.root.after(2000, lambda: self.info_label.config(text=current_text))
    
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
            
            # Reset zoom and pan
            self.zoom_level = 1.0
            self.image_offset_x = 0
            self.image_offset_y = 0
            
            # Draw
            self.photo = ImageTk.PhotoImage(img_resized)
            self.original_photo = img  # Store original for zooming
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
            zoom_pct = int(self.zoom_level * 100)
            self.info_label.config(
                text=f"📷 {filename} | Image {idx+1}/{len(self.images)} | 🏷️ {box_count} boxes | 🔍 {zoom_pct}%"
            )
            
            # Highlight in listbox
            self.image_listbox.selection_clear(0, 'end')
            self.image_listbox.selection_set(idx)
            self.image_listbox.see(idx)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
    
    def on_mouse_down(self, event):
        """Start drawing box or add polygon point"""
        if self.draw_mode == 'polygon':
            # Check if clicking near first point (close polygon)
            if len(self.polygon_points) >= 3:
                first_point = self.polygon_points[0]
                distance = ((event.x - first_point[0])**2 + (event.y - first_point[1])**2)**0.5
                
                if distance < 10:  # Close enough to first point
                    # Finish polygon automatically
                    self.finish_polygon()
                    return
            
            # Add point to polygon
            self.polygon_points.append((event.x, event.y))
            
            # Clear old temp items and redraw everything
            for item in self.temp_polygon_items:
                self.canvas.delete(item)
            self.temp_polygon_items = []
            
            # Draw all points
            for i, (px, py) in enumerate(self.polygon_points):
                point = self.canvas.create_oval(
                    px - 3, py - 3,
                    px + 3, py + 3,
                    fill='#00ff00', outline='#00ff00'
                )
                self.temp_polygon_items.append(point)
            
            # Draw all lines (including closing line if 3+ points)
            for i in range(len(self.polygon_points)):
                if len(self.polygon_points) < 2:
                    break
                    
                next_idx = (i + 1) % len(self.polygon_points)
                
                # Only draw closing line if we have 3+ points
                if i == len(self.polygon_points) - 1 and len(self.polygon_points) < 3:
                    break
                
                p1 = self.polygon_points[i]
                p2 = self.polygon_points[next_idx]
                line = self.canvas.create_line(
                    p1[0], p1[1], p2[0], p2[1],
                    fill='#00ff00', width=2,
                    dash=(5, 5) if i == len(self.polygon_points) - 1 else ()
                )
                self.temp_polygon_items.append(line)
            
            # Update info
            zoom_pct = int(self.zoom_level * 100)
            hint = "(Press [F] or click first point to close)" if len(self.polygon_points) >= 3 else "(Keep clicking points)"
            self.info_label.config(
                text=f"⬟ Polygon: {len(self.polygon_points)} points {hint} | 🔍 {zoom_pct}%"
            )
        else:
            # Box mode
            self.drawing = True
            self.start_x = event.x
            self.start_y = event.y
    
    def on_mouse_drag(self, event):
        """Draw box while dragging"""
        if self.draw_mode == 'polygon' or not self.drawing:
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
        if self.draw_mode == 'polygon' or not self.drawing:
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
        
        # Get selected color
        color_idx = self.color_labels.index(self.color_var.get()) if hasattr(self, 'color_labels') else 0
        color = self.color_values[color_idx] if hasattr(self, 'color_values') else 'white'
        
        # Add box with type marker
        self.boxes.append({
            'type': 'box',
            'yolo': [0, x_center, y_center, width, height],
            'color': color
        })
        
        # Auto-save after drawing box
        if self.current_idx < len(self.images):
            image_path = self.images[self.current_idx]
            self.all_labels[image_path] = self.boxes.copy()
            self.save_to_destination(image_path, self.boxes)
            
            # Update listbox status
            filename = Path(image_path).name
            status = "✅" if self.boxes else "⭕"
            self.image_listbox.delete(self.current_idx)
            self.image_listbox.insert(self.current_idx, f"{status} {filename}")
            self.image_listbox.selection_set(self.current_idx)
            self.update_stats()
        
        # Redraw
        self.redraw_boxes()
        
        # Update info
        zoom_pct = int(self.zoom_level * 100)
        self.info_label.config(
            text=f"📷 {Path(self.images[self.current_idx]).name} | "
                 f"Image {self.current_idx+1}/{len(self.images)} | "
                 f"🏷️ {len(self.boxes)} boxes | 🔍 {zoom_pct}% | ✅ Auto-saved"
        )
    
    def export_color_crop(self, image_path, box, box_idx):
        """Export plate crop to color dataset folder"""
        if not self.auto_export_crops.get():
            return
        
        try:
            # Load original image
            img = Image.open(image_path)
            img_w, img_h = img.size
            
            # Get bounding box coordinates
            box_type = box.get('type', 'box') if isinstance(box, dict) else 'box'
            color = box.get('color', 'white') if isinstance(box, dict) else 'white'
            
            if box_type == 'polygon':
                # Convert polygon to bounding box
                points = box['points']
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                
                x_min, y_min = min(xs), min(ys)
                x_max, y_max = max(xs), max(ys)
            else:
                # Box format
                if 'yolo' in box:
                    _, x_center, y_center, width, height = box['yolo']
                else:
                    _, x_center, y_center, width, height = box
                
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
            
            # Convert to pixels
            x1 = int(x_min * img_w)
            y1 = int(y_min * img_h)
            x2 = int(x_max * img_w)
            y2 = int(y_max * img_h)
            
            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
            
            # Validate crop size
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                return  # Too small
            
            # Crop plate
            plate_crop = img.crop((x1, y1, x2, y2))
            
            # Resize to standard size (96x96)
            plate_resized = plate_crop.resize((96, 96), Image.Resampling.LANCZOS)
            
            # Save to color folder
            # 80% train, 20% val
            img_name = Path(image_path).stem
            split = 'train' if hash(img_name) % 5 != 0 else 'val'
            
            color_dir = self.color_export_dir / split / color
            color_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = color_dir / f"{img_name}_{box_idx}.jpg"
            plate_resized.save(output_file, quality=95)
            
        except Exception as e:
            print(f"Failed to export color crop: {e}")
    
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
        
        # Save label files
        img_path = Path(image_path)
        label_path = labels_dir / f"{img_path.stem}.txt"
        json_path = labels_dir / f"{img_path.stem}.json"
        
        # Save YOLO .txt format (convert polygons to bounding boxes for YOLO compatibility)
        with open(label_path, 'w') as f:
            for box in boxes:
                box_type = box.get('type', 'box') if isinstance(box, dict) else 'box'
                
                if box_type == 'polygon':
                    # Convert polygon to bounding box for YOLO
                    points = box['points']
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    yolo = [0, x_center, y_center, width, height]
                elif 'yolo' in box:
                    yolo = box['yolo']
                else:
                    yolo = box
                
                f.write(' '.join(map(str, yolo)) + '\n')
        
        # Save JSON with full info (polygon or box)
        json_boxes = []
        for idx, box in enumerate(boxes):
            if isinstance(box, dict):
                box_type = box.get('type', 'box')
                if box_type == 'polygon':
                    json_boxes.append({
                        'type': 'polygon',
                        'points': box['points'],
                        'color': box.get('color', 'white')
                    })
                else:
                    json_boxes.append({
                        'type': 'box',
                        'yolo': box.get('yolo', box),
                        'color': box.get('color', 'white')
                    })
            else:
                json_boxes.append({
                    'type': 'box',
                    'yolo': box,
                    'color': 'white'
                })
            
            # Auto-export color crop
            self.export_color_crop(image_path, box, idx)
        
        with open(json_path, 'w') as jf:
            json.dump({'boxes': json_boxes}, jf, indent=2)
        
        # Copy image
        dest_image = images_dir / img_path.name
        if not dest_image.exists():
            shutil.copy(image_path, dest_image)
        
        # ====================================================================
        # AUTO-EXPORT COLOR CROPS - Save plate color crops if enabled
        # ====================================================================
        if self.auto_export_crops.get() and boxes and 'color' in boxes[0]:
            # Create color export directory if it doesn't exist
            self.color_export_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract dominant color from the first box (assuming it's the plate box)
            plate_color = boxes[0]['color']
            color_name = plate_color.capitalize()  # Capitalize for folder name
            
            # Create subdirectory for the color if it doesn't exist
            color_dir = self.color_export_dir / color_name
            color_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the image to the color directory
            shutil.copy(image_path, color_dir / img_path.name)
            
            # Save a separate label file in the color directory
            color_label_path = color_dir / f"{img_path.stem}.txt"
            with open(color_label_path, 'w') as f:
                for box in boxes:
                    if box.get('color') == plate_color:  # Only save the plate box
                        box_type = box.get('type', 'box') if isinstance(box, dict) else 'box'
                        
                        if box_type == 'polygon':
                            # Convert polygon to bounding box for YOLO
                            points = box['points']
                            xs = [p[0] for p in points]
                            ys = [p[1] for p in points]
                            
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            
                            x_center = (x_min + x_max) / 2
                            y_center = (y_min + y_max) / 2
                            width = x_max - x_min
                            height = y_max - y_min
                            
                            yolo = [0, x_center, y_center, width, height]
                        elif 'yolo' in box:
                            yolo = box['yolo']
                        else:
                            yolo = box
                        
                        f.write(' '.join(map(str, yolo)) + '\n')
    
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
    
    def on_pan_start(self, event):
        """Start panning with right mouse button"""
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor='fleur')  # Change cursor to move
    
    def on_pan_move(self, event):
        """Pan the image"""
        if not self.panning:
            return
        
        # Calculate pan delta
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        
        # Update image offset
        self.image_offset_x += dx
        self.image_offset_y += dy
        
        # Update start position for next move
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        
        # Refresh canvas
        self.refresh_canvas()
    
    def on_pan_end(self, event):
        """End panning"""
        self.panning = False
        self.canvas.config(cursor='')  # Reset cursor
    
    def redraw_polygon_points(self):
        """Redraw polygon points after zoom/pan"""
        if not self.polygon_points or self.draw_mode != 'polygon':
            return
        
        # Clear old polygon items
        for item in self.temp_polygon_items:
            self.canvas.delete(item)
        self.temp_polygon_items = []
        
        # Redraw points and lines
        for i, (x, y) in enumerate(self.polygon_points):
            # Draw point
            point = self.canvas.create_oval(
                x - 3, y - 3,
                x + 3, y + 3,
                fill='#00ff00', outline='#00ff00'
            )
            self.temp_polygon_items.append(point)
            
            # Draw line to next point (including closing line)
            if len(self.polygon_points) > 1:
                next_idx = (i + 1) % len(self.polygon_points)
                next_x, next_y = self.polygon_points[next_idx]
                line = self.canvas.create_line(
                    x, y, next_x, next_y,
                    fill='#00ff00', width=2
                )
                self.temp_polygon_items.append(line)
    
    def on_shape_drag_start(self, event):
        """Start dragging a box/polygon with Ctrl+Left Click"""
        if not self.photo or not self.boxes:
            return
        
        # Find which box/polygon was clicked
        click_x = event.x
        click_y = event.y
        
        img_w = self.photo.width()
        img_h = self.photo.height()
        
        for idx, box in enumerate(self.boxes):
            box_type = box.get('type', 'box') if isinstance(box, dict) else 'box'
            
            if box_type == 'polygon':
                # Check if click is inside polygon
                points = box['points']
                canvas_points = []
                for norm_x, norm_y in points:
                    x = norm_x * img_w + self.offset_x
                    y = norm_y * img_h + self.offset_y
                    canvas_points.append((x, y))
                
                # Simple point-in-polygon check (bounding box)
                xs = [p[0] for p in canvas_points]
                ys = [p[1] for p in canvas_points]
                if min(xs) <= click_x <= max(xs) and min(ys) <= click_y <= max(ys):
                    self.dragging_shape = True
                    self.selected_box_idx = idx
                    self.drag_offset_x = click_x
                    self.drag_offset_y = click_y
                    self.canvas.config(cursor='hand2')
                    return
            else:
                # Check if click is inside rectangle
                if 'yolo' in box:
                    _, x_center, y_center, width, height = box['yolo']
                else:
                    _, x_center, y_center, width, height = box
                
                x1 = (x_center - width/2) * img_w + self.offset_x
                y1 = (y_center - height/2) * img_h + self.offset_y
                x2 = (x_center + width/2) * img_w + self.offset_x
                y2 = (y_center + height/2) * img_h + self.offset_y
                
                if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                    self.dragging_shape = True
                    self.selected_box_idx = idx
                    self.drag_offset_x = click_x
                    self.drag_offset_y = click_y
                    self.canvas.config(cursor='hand2')
                    return
    
    def on_shape_drag_move(self, event):
        """Move the selected box/polygon"""
        if not self.dragging_shape or self.selected_box_idx is None:
            return
        
        # Calculate movement delta
        dx = event.x - self.drag_offset_x
        dy = event.y - self.drag_offset_y
        
        # Update drag start position
        self.drag_offset_x = event.x
        self.drag_offset_y = event.y
        
        # Convert delta to normalized coordinates
        img_w = self.photo.width()
        img_h = self.photo.height()
        norm_dx = dx / img_w
        norm_dy = dy / img_h
        
        # Update box/polygon position
        box = self.boxes[self.selected_box_idx]
        box_type = box.get('type', 'box') if isinstance(box, dict) else 'box'
        
        if box_type == 'polygon':
            # Move all polygon points
            for point in box['points']:
                point[0] += norm_dx
                point[1] += norm_dy
        else:
            # Move box center
            if 'yolo' in box:
                box['yolo'][1] += norm_dx  # x_center
                box['yolo'][2] += norm_dy  # y_center
            else:
                box[1] += norm_dx
                box[2] += norm_dy
        
        # Redraw
        self.redraw_boxes()
    
    def on_shape_drag_end(self, event):
        """End dragging and save"""
        if not self.dragging_shape:
            return
        
        self.dragging_shape = False
        self.selected_box_idx = None
        self.canvas.config(cursor='')
        
        # Auto-save
        if self.current_idx < len(self.images):
            image_path = self.images[self.current_idx]
            self.all_labels[image_path] = self.boxes.copy()
            self.save_to_destination(image_path, self.boxes)
            
            # Update info
            zoom_pct = int(self.zoom_level * 100)
            self.info_label.config(
                text=f"📷 {Path(image_path).name} | "
                     f"Image {self.current_idx+1}/{len(self.images)} | "
                     f"🏷️ {len(self.boxes)} boxes | 🔍 {zoom_pct}% | ✅ Moved & saved"
            )
    
    def redraw_boxes(self):
        """Redraw all boxes/polygons on canvas"""
        # Clear old boxes
        self.canvas.delete('box')
        self.canvas.delete('label')
        
        if not self.photo:
            return
        
        img_w = self.photo.width()
        img_h = self.photo.height()
        
        # Color mapping
        color_map = {'white':'#ffffff', 'black':'#333333', 'red':'#ff3333', 'yellow':'#ffcc00'}
        color_indo = {'white':'Putih', 'black':'Hitam', 'red':'Merah', 'yellow':'Kuning'}
        
        for i, box in enumerate(self.boxes):
            # Detect box type
            box_type = box.get('type', 'box') if isinstance(box, dict) else 'box'
            color_name = box.get('color', 'white') if isinstance(box, dict) else 'white'
            display_color = color_map.get(color_name, '#00ff00')
            indo_name = color_indo.get(color_name, 'Putih')
            
            if box_type == 'polygon':
                # Draw polygon with original shape
                points = box['points']
                canvas_points = []
                for norm_x, norm_y in points:
                    x = norm_x * img_w + self.offset_x
                    y = norm_y * img_h + self.offset_y
                    canvas_points.extend([x, y])
                
                # Draw polygon
                self.canvas.create_polygon(
                    canvas_points,
                    outline=display_color,
                    fill='',
                    width=3,
                    tags='box'
                )
                
                # Label at first point
                if canvas_points:
                    self.canvas.create_text(
                        canvas_points[0], canvas_points[1] - 5,
                        text=f"Plat #{i+1} ({indo_name}) ⬟",
                        fill=display_color,
                        font=('Segoe UI', 9, 'bold'),
                        anchor='sw',
                        tags='label'
                    )
            else:
                # Draw rectangle (box mode or old format)
                if 'yolo' in box:
                    _, x_center, y_center, width, height = box['yolo']
                else:
                    _, x_center, y_center, width, height = box
                
                x1 = (x_center - width/2) * img_w + self.offset_x
                y1 = (y_center - height/2) * img_h + self.offset_y
                x2 = (x_center + width/2) * img_w + self.offset_x
                y2 = (y_center + height/2) * img_h + self.offset_y
                
                # Draw box
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline=display_color,
                    width=3,
                    tags='box'
                )
                
                # Draw label
                self.canvas.create_text(
                    x1, y1 - 5,
                    text=f"Plat #{i+1} ({indo_name}) □",
                    fill=display_color,
                    font=('Segoe UI', 9, 'bold'),
                    anchor='sw',
                    tags='label'
                )
    
    def delete_last_box(self):
        """Delete last drawn box"""
        if self.boxes:
            self.boxes.pop()
            
            # Auto-save after deleting
            if self.current_idx < len(self.images):
                image_path = self.images[self.current_idx]
                self.all_labels[image_path] = self.boxes.copy()
                self.save_to_destination(image_path, self.boxes)
                
                # Update listbox status
                filename = Path(image_path).name
                status = "✅" if self.boxes else "⭕"
                self.image_listbox.delete(self.current_idx)
                self.image_listbox.insert(self.current_idx, f"{status} {filename}")
                self.image_listbox.selection_set(self.current_idx)
                self.update_stats()
            
            self.redraw_boxes()
            self.info_label.config(
                text=f"📷 {Path(self.images[self.current_idx]).name} | "
                     f"Image {self.current_idx+1}/{len(self.images)} | "
                     f"🏷️ {len(self.boxes)} boxes | ✅ Auto-saved"
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
    
    def on_shape_drag_start(self, event):
        """Start dragging a box/polygon with Ctrl+Left Click"""
        if not self.photo or not self.boxes:
            return
        
        # Find which box/polygon was clicked
        click_x = event.x
        click_y = event.y
        
        img_w = self.photo.width()
        img_h = self.photo.height()
        
        for idx, box in enumerate(self.boxes):
            box_type = box.get('type', 'box') if isinstance(box, dict) else 'box'
            
            if box_type == 'polygon':
                # Check if click is inside polygon
                points = box['points']
                canvas_points = []
                for norm_x, norm_y in points:
                    x = norm_x * img_w + self.offset_x
                    y = norm_y * img_h + self.offset_y
                    canvas_points.append((x, y))
                
                # Simple point-in-polygon check (bounding box)
                xs = [p[0] for p in canvas_points]
                ys = [p[1] for p in canvas_points]
                if min(xs) <= click_x <= max(xs) and min(ys) <= click_y <= max(ys):
                    self.dragging_shape = True
                    self.selected_box_idx = idx
                    self.drag_offset_x = click_x
                    self.drag_offset_y = click_y
                    self.canvas.config(cursor='hand2')
                    return
            else:
                # Check if click is inside rectangle
                if 'yolo' in box:
                    _, x_center, y_center, width, height = box['yolo']
                else:
                    _, x_center, y_center, width, height = box
                
                x1 = (x_center - width/2) * img_w + self.offset_x
                y1 = (y_center - height/2) * img_h + self.offset_y
                x2 = (x_center + width/2) * img_w + self.offset_x
                y2 = (y_center + height/2) * img_h + self.offset_y
                
                if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                    self.dragging_shape = True
                    self.selected_box_idx = idx
                    self.drag_offset_x = click_x
                    self.drag_offset_y = click_y
                    self.canvas.config(cursor='hand2')
                    return
    
    def on_shape_drag_move(self, event):
        """Move the selected box/polygon"""
        if not self.dragging_shape or self.selected_box_idx is None:
            return
        
        # Calculate movement delta
        dx = event.x - self.drag_offset_x
        dy = event.y - self.drag_offset_y
        
        # Update drag start position
        self.drag_offset_x = event.x
        self.drag_offset_y = event.y
        
        # Convert delta to normalized coordinates
        img_w = self.photo.width()
        img_h = self.photo.height()
        norm_dx = dx / img_w
        norm_dy = dy / img_h
        
        # Update box/polygon position
        box = self.boxes[self.selected_box_idx]
        box_type = box.get('type', 'box') if isinstance(box, dict) else 'box'
        
        if box_type == 'polygon':
            # Move all polygon points
            for point in box['points']:
                point[0] += norm_dx
                point[1] += norm_dy
        else:
            # Move box center
            if 'yolo' in box:
                box['yolo'][1] += norm_dx  # x_center
                box['yolo'][2] += norm_dy  # y_center
            else:
                box[1] += norm_dx
                box[2] += norm_dy
        
        # Redraw
        self.redraw_boxes()
    
    def on_shape_drag_end(self, event):
        """End dragging and save"""
        if not self.dragging_shape:
            return
        
        self.dragging_shape = False
        self.selected_box_idx = None
        self.canvas.config(cursor='')
        
        # Auto-save
        if self.current_idx < len(self.images):
            image_path = self.images[self.current_idx]
            self.all_labels[image_path] = self.boxes.copy()
            self.save_to_destination(image_path, self.boxes)
            
            # Update info
            zoom_pct = int(self.zoom_level * 100)
            self.info_label.config(
                text=f"📷 {Path(image_path).name} | "
                     f"Image {self.current_idx+1}/{len(self.images)} | "
                     f"🏷️ {len(self.boxes)} boxes | 🔍 {zoom_pct}% | ✅ Moved & saved"
            )
    
def main():
    root = tk.Tk()
    app = ModernLabelingTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()

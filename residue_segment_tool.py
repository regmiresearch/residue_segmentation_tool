import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
from glob import glob
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from segment_anything import sam_model_registry, SamPredictor
import torch
import tempfile
import shutil


DEFAULT_FONT = ("Segoe UI", 10)
BUTTON_STYLE = {"font": DEFAULT_FONT, "width": 20, "padx": 2, "pady": 2}

class ResidueSegmentationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Residue Segmentation Tool")
        self.root.geometry("1300x750")
        self.root.configure(bg="#f8f9fa")

        self.sam_predictor = None
        self.sam_loaded = False
        self.mask_inverted = False

        # State
        self.image_paths = []
        self.image_index = 0
        self.image = None
        self.original_shape = None
        self.gray = None
        self.preview_mask = None
        self.final_mask = None
        self.filename = ""
        self.overlay_var = tk.IntVar()
        self.roi_enabled = False
        self.roi = None
        self.tk_img_display_size = (600, 600)
        self.last_display_size = (600, 600)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.hist_canvas = None
        self.mode = tk.StringVar(value="otsu")
        self.start_x = self.start_y = 0
        self.end_x = self.end_y = 0
        self.rectangle_overlay = None

        # Editing tools
        self.edit_mode = False
        self.edit_polygon = []
        self.polygon_lines = []
        self.polygon_history = []  # Stores each polygon mask
        self.segmentation_mask = None        # Base mask from Otsu/Canny/etc
        self.polygon_mask_layers = []        # List of polygon masks for undo
        self.brush_mode = False
        self.erase_mode = False  # NEW
        self.brush_size = 10

        self.build_layout()
    def build_layout(self):
        control_canvas = tk.Canvas(self.root, width=260, bg="#f8f9fa", highlightthickness=0)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=control_canvas.yview)
        self.control_frame = tk.Frame(control_canvas, bg="#f8f9fa")

        self.control_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )
        control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=scrollbar.set)

        control_canvas.pack(side="left", fill="y")
        scrollbar.pack(side="left", fill="y")

        self.image_frame = tk.Frame(self.root, bg="#f8f9fa")
        self.image_frame.pack(side="right", expand=True)

        image_area = tk.Frame(self.image_frame, bg="#f8f9fa")
        image_area.grid(row=0, column=0, columnspan=2)

        self.canvas = tk.Canvas(image_area, width=600, height=600, bg="#ffffff", highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        self.mask_label = tk.Label(image_area, bg="#ffffff", width=600, height=600)
        self.mask_label.grid(row=0, column=1, padx=10, pady=10)

        self.cover_label = tk.Label(image_area, text="", font=("Segoe UI", 12, "bold"), fg="#2e8b57", bg="#f8f9fa")
        self.cover_label.grid(row=1, column=1, pady=(0, 10))

        self.hist_frame = tk.Frame(self.image_frame, bg="#f8f9fa", width=600, height=200)
        self.hist_frame.grid_propagate(False)
        self.hist_frame.grid(row=1, column=0, padx=10, pady=10)

        self.canvas.bind("<Button-1>", self.handle_left_click)
        self.canvas.bind("<Button-3>", self.edit_mask_finalize)
        self.canvas.bind("<B1-Motion>", self.handle_brush_or_draw)
        self.canvas.bind("<ButtonRelease-1>", self.finish_draw)

        self.build_controls()

    def build_controls(self):
        self.edit_label = tk.Label(self.control_frame, text="Edit Mode: OFF", font=("Segoe UI", 10, "bold"), fg="red", bg="#f8f9fa")
        self.edit_label.pack(pady=(0, 5))

        for section_title, widgets in [
            ("Navigation", [("Load Folder", self.load_folder), ("Previous", self.prev_image), ("Next", self.next_image)]),
            ("Mask Handling", [("Confirm Mask", self.confirm_mask), ("Save Image+Mask", self.save_data), ("Overlay on Mask", self.update_display, "check"),("Undo Last Polygon", self.undo_last_polygon),
                ("Reset Mask", self.reset_preview_mask), ("Invert Mask", self.toggle_invert_mask), ("Batch Segment Folder", self.batch_segment_folder)]),
            ("ROI Tools", [("Enable ROI Selection", self.enable_roi), ("Apply Crop to ROI", self.apply_crop)]),
        ]:
            frame = tk.LabelFrame(self.control_frame, text=section_title, font=DEFAULT_FONT, bg="#f8f9fa", bd=2, relief="ridge")
            frame.pack(pady=12, fill="x", padx=5)
            for label, command, *wtype in widgets:
                if wtype and wtype[0] == "check":
                    tk.Checkbutton(frame, text=label, variable=self.overlay_var, command=command,
                                   font=DEFAULT_FONT, bg="#f8f9fa").pack(anchor="w", padx=5, pady=2)
                else:
                    tk.Button(frame, text=label, command=command, **BUTTON_STYLE).pack(pady=2)

        seg = tk.LabelFrame(self.control_frame, text="Segmentation Mode", font=DEFAULT_FONT, bg="#f8f9fa", bd=2, relief="ridge")
        seg.pack(pady=12, fill="x", padx=5)
        for text, value in [("Otsu", "otsu"), ("Canny", "canny"), ("Manual", "manual"), ("SAM", "sam")]:
            tk.Radiobutton(seg, text=text, variable=self.mode, value=value,
                           command=self.update_segmentation, font=DEFAULT_FONT, anchor="w",
                           bg="#f8f9fa").pack(anchor="w", padx=10, pady=1)

        for label_text, attr, default in [("Canny Low", 'canny_low', 100), ("Canny High", 'canny_high', 200), ("Manual Threshold", 'manual_thresh', 127)]:
            tk.Label(self.control_frame, text=label_text, font=DEFAULT_FONT, bg="#f8f9fa").pack(pady=(10, 0))
            scale = tk.Scale(self.control_frame, from_=0, to=255, orient='horizontal', bg="#f8f9fa",
                             command=lambda e: self.update_segmentation())
            setattr(self, attr, scale)
            scale.set(default)
            scale.pack(pady=2)

        self.edit_button = tk.Button(self.control_frame, text="Enter Edit Mode", command=self.toggle_edit_mode, **BUTTON_STYLE)
        self.edit_button.pack(pady=12)

        self.brush_button = tk.Button(self.control_frame, text="Activate Brush Tool", command=self.toggle_brush_mode, **BUTTON_STYLE)
        self.brush_button.pack(pady=6)

        tk.Label(self.control_frame, text="Brush Size", font=DEFAULT_FONT, bg="#f8f9fa").pack()
        self.brush_size_slider = tk.Scale(self.control_frame, from_=1, to=50, orient='horizontal',
                                          bg="#f8f9fa", command=self.set_brush_size)
        self.brush_size_slider.set(self.brush_size)
        self.brush_size_slider.pack(pady=(0, 6))

        self.erase_button = tk.Button(self.control_frame, text="Switch to Erase Mode", command=self.toggle_erase_mode, **BUTTON_STYLE)
        self.erase_button.pack(pady=4)

    def batch_segment_folder(self):
        if not self.image_paths:
            print("No folder loaded.")
            return

        progress_win = tk.Toplevel(self.root)
        progress_win.title("Batch Segmentation Progress")
        progress_label = tk.Label(progress_win, text="Processing images...", font=("Segoe UI", 10))
        progress_label.pack(padx=20, pady=10)
        progress_bar = tk.Label(progress_win, text="", font=("Segoe UI", 10), fg="green")
        progress_bar.pack(pady=5)
        progress_win.update()

        image_dir = os.path.join(self.script_dir, "images_batch")
        mask_dir = os.path.join(self.script_dir, "masks_batch")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for i, path in enumerate(self.image_paths):
            filename = os.path.basename(path)
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            if self.mode.get() == "otsu":
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif self.mode.get() == "manual":
                t = self.manual_thresh.get()
                _, mask = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
            elif self.mode.get() == "canny":
                low, high = self.canny_low.get(), self.canny_high.get()
                edges = cv2.Canny(gray, low, high)
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(edges, kernel, iterations=1)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
            elif self.mode.get() == "sam":
                self.load_sam_model()
                self.sam_predictor.set_image(image)
                H, W = image.shape[:2]
                input_box = np.array([0, 0, W-1, H-1])
                masks, _, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                mask = (masks[0] * 255).astype(np.uint8)

            # Save
            cv2.imwrite(os.path.join(image_dir, filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(mask_dir, filename), mask)

            progress_bar.config(text=f"{i + 1} / {len(self.image_paths)} images processed...")
            progress_win.update()

        progress_label.config(text="Batch processing complete!")
        progress_bar.config(fg="blue")
        tk.Button(progress_win, text="Close", command=progress_win.destroy).pack(pady=10)

        print("Batch segmentation complete.")

    def toggle_erase_mode(self):
        self.erase_mode = not self.erase_mode
        if self.erase_mode:
            self.erase_button.config(text="Switch to Add Mode", fg="red")
            print("Brush in Erase mode")
        else:
            self.erase_button.config(text="Switch to Erase Mode", fg="black")
            print("Brush in Add mode")

    def toggle_invert_mask(self):
        if self.preview_mask is not None:
            self.preview_mask = 255 - self.preview_mask
            self.mask_inverted = not self.mask_inverted
            self.update_display()


    def load_sam_model(self):
        try:
            import torch
            import tempfile
            import shutil
            import os
            import sys
            from segment_anything import sam_model_registry, SamPredictor

            # Determine the path to the bundled .pth model
            if getattr(sys, 'frozen', False):
                # Running from .exe
                model_src_path = os.path.join(sys._MEIPASS, "sam_vit_b.pth")
            else:
                # Running from script
                model_src_path = os.path.abspath("sam_vit_b.pth")

            # Copy the .pth file to a writable temp file
            temp_model_path = os.path.join(tempfile.gettempdir(), "sam_temp.pth")
            shutil.copyfile(model_src_path, temp_model_path)

            # Now load using torch.load from extracted location
            sam = sam_model_registry["vit_b"](checkpoint=temp_model_path)
            sam.eval()
            self.sam_predictor = SamPredictor(sam)
            print("✅ SAM model loaded successfully.")

        except Exception as e:
            print(f"❌ Failed to load SAM model: {e}")
            self.sam_predictor = None



    def apply_brush(self, event):
        if not self.brush_mode or self.preview_mask is None:
            return

        x_disp, y_disp = event.x, event.y
        disp_w, disp_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        img_h, img_w = self.preview_mask.shape[:2]
        scale_x = img_w / self.last_display_size[0]
        scale_y = img_h / self.last_display_size[1]

        x = int(x_disp * scale_x)
        y = int(y_disp * scale_y)

        value = 0 if self.erase_mode else 255
        radius = self.brush_size_slider.get()
        cv2.circle(self.preview_mask, (x, y), radius, value, -1)
        self.update_display()

    def handle_brush_or_draw(self, event):
        if self.brush_mode:
            self.apply_brush(event)
        elif self.roi_enabled:
            self.update_draw(event)

    def toggle_edit_mode(self, force_off=False):
        if force_off:
            self.edit_mode = True  # temporarily set to ON so the toggle logic below will turn it OFF
        self.edit_mode = not self.edit_mode
        if self.edit_mode:
            self.edit_polygon = []
            self.clear_polygon_lines()
            self.edit_label.config(text="Edit Mode: ON", fg="green")
            self.edit_button.config(text="Exit Edit Mode")
            self.canvas.config(cursor="crosshair")
            print("Edit mode enabled. Click to place points. Right-click to finish polygon.")
        else:
            self.edit_label.config(text="Edit Mode: OFF", fg="red")
            self.edit_button.config(text="Enter Edit Mode")
            self.canvas.config(cursor="")
            self.clear_polygon_lines()
            print("Edit mode disabled.")

    def handle_left_click(self, event):
        if self.edit_mode:
            self.edit_mask_click(event)
        else:
            self.start_draw(event)

    def edit_mask_click(self, event):
        if not self.edit_mode:
            return

        # Calculate scaling
        x_disp, y_disp = event.x, event.y
        disp_w, disp_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        img_h, img_w = self.preview_mask.shape[:2]
        scale_x = img_w / self.last_display_size[0]
        scale_y = img_h / self.last_display_size[1]

        # Convert to image coordinates
        x_img = int(x_disp * scale_x)
        y_img = int(y_disp * scale_y)
        self.edit_polygon.append((x_img, y_img))

        # Draw line on canvas (display coordinates)
        if len(self.edit_polygon) > 1:
            x1_disp, y1_disp = int(self.edit_polygon[-2][0] / scale_x), int(self.edit_polygon[-2][1] / scale_y)
            x2_disp, y2_disp = int(self.edit_polygon[-1][0] / scale_x), int(self.edit_polygon[-1][1] / scale_y)
            line = self.canvas.create_line(x1_disp, y1_disp, x2_disp, y2_disp, fill="red", width=2)
            self.polygon_lines.append(line)

        print(f"Point added (image coords): ({x_img}, {y_img})")

    def edit_mask_finalize(self, event):
        if not self.edit_mode or len(self.edit_polygon) < 3:
            print("Not enough points to form polygon.")
            return

        pts = np.array([self.edit_polygon], dtype=np.int32)
        layer = np.zeros_like(self.preview_mask)
        cv2.fillPoly(layer, pts, 255)
        self.polygon_mask_layers.append(layer)

        self.rebuild_preview_mask()
        self.clear_polygon_lines()
        self.edit_polygon = []
        print("Polygon added and stored.")

    def clear_polygon_lines(self):
        for line in self.polygon_lines:
            self.canvas.delete(line)
        self.polygon_lines = []

    def undo_last_polygon(self):
        if not self.polygon_mask_layers:
            print("No polygon to undo.")
            return

        self.polygon_mask_layers.pop()
        self.rebuild_preview_mask()
        print("Undid last polygon.")

    def reset_preview_mask(self):
        self.preview_mask = np.zeros_like(self.gray)
        self.polygon_history.clear()
        self.update_display()
        print("Preview mask reset.")


    def confirm_mask(self):
        self.final_mask = self.preview_mask.copy()
        total_pixels = self.final_mask.size
        residue_pixels = np.count_nonzero(self.final_mask == 255)
        percent_cover = (residue_pixels / total_pixels) * 100
        self.cover_label.config(text=f"Crop Residue Cover = {percent_cover:.2f}%")
        print(f"Confirmed mask. Crop Residue Cover = {percent_cover:.2f}%")

    def update_histogram(self):
        if self.hist_canvas:
            self.hist_canvas.get_tk_widget().destroy()

        fig = Figure(figsize=(3.5, 2.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.hist(self.gray.ravel(), bins=256, range=(0, 255), color='gray')
        ax.set_title("Grayscale Histogram", fontsize=10)
        ax.set_xlabel("Pixel Intensity", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        fig.tight_layout()

        self.hist_canvas = FigureCanvasTkAgg(fig, master=self.hist_frame)
        self.hist_canvas.draw()
        self.hist_canvas.get_tk_widget().pack()

    def load_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.image_paths = sorted([f for ext in ('*.png', '*.jpg', '*.jpeg')
                                       for f in glob(os.path.join(folder, ext))])
            self.image_index = 0
            self.load_image()

    def load_image(self):
        path = self.image_paths[self.image_index]
        self.filename = os.path.basename(path)
        self.image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        self.original_shape = self.image.shape[:2]
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.preview_mask = np.zeros_like(self.gray)
        self.final_mask = None
        self.roi = None
        self.cover_label.config(text="")
        self.update_segmentation()
        self.update_histogram()

    def next_image(self):
        if self.image_index < len(self.image_paths) - 1:
            self.image_index += 1
            self.load_image()

    def prev_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.load_image()

    def update_segmentation(self):
        if self.gray is None or self.image is None:
            return

        if self.mode.get() == "otsu":
            _, self.segmentation_mask = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.mode.get() == "manual":
            t = self.manual_thresh.get()
            _, self.segmentation_mask = cv2.threshold(self.gray, t, 255, cv2.THRESH_BINARY)
        elif self.mode.get() == "canny":
            low, high = self.canny_low.get(), self.canny_high.get()
            edges = cv2.Canny(self.gray, low, high)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.segmentation_mask = np.zeros_like(self.gray)
            cv2.drawContours(self.segmentation_mask, contours, -1, 255, thickness=cv2.FILLED)

        elif self.mode.get() == "sam":
            self.load_sam_model()
            if self.sam_predictor is None:
                print("SAM model not loaded.")
                return

            input_image = self.image.copy()
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
            self.sam_predictor.set_image(input_image)

            H, W = input_image.shape[:2]
            input_box = np.array([0, 0, W-1, H-1])  # Use full image
            masks, _, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            self.segmentation_mask = (masks[0] * 255).astype(np.uint8)


        disable = self.mode.get() == "sam"
        for slider in [self.canny_low, self.canny_high, self.manual_thresh]:
            slider.config(state="disabled" if disable else "normal")

        self.rebuild_preview_mask()

    def update_display(self):
        if self.image is None or self.preview_mask is None:
            print("update_display skipped: image or preview_mask is None")
            return
    
        self.display_image(self.image)

        if self.overlay_var.get():
            overlay = self.image.copy()
            red = np.zeros_like(overlay)
            red[:, :, 0] = 255
            alpha = 0.4
            mask_bool = self.preview_mask.astype(bool)
            overlay[mask_bool] = cv2.addWeighted(
                overlay[mask_bool], 1 - alpha, red[mask_bool], alpha, 0
            )
            display_right = overlay
        else:
            display_right = cv2.cvtColor(self.preview_mask, cv2.COLOR_GRAY2RGB)

        img_pil = Image.fromarray(display_right)
        img_pil.thumbnail(self.tk_img_display_size)
        self.last_display_size = img_pil.size
        tk_img = ImageTk.PhotoImage(img_pil)
        self.mask_label.configure(image=tk_img)
        self.mask_label.image = tk_img

    def display_image(self, img):
        if img is None:
            print("display_image skipped: image is None")
            return
        self.canvas.delete("all")
        img_pil = Image.fromarray(img)
        img_pil.thumbnail(self.tk_img_display_size)
        self.last_display_size = img_pil.size
        self.tk_canvas_img = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_canvas_img)


    def enable_roi(self):
        self.roi_enabled = True
        print("ROI selection enabled. Draw on the original image.")

    def start_draw(self, event):
        if not self.roi_enabled:
            return
        self.start_x, self.start_y = event.x, event.y
        self.rectangle_overlay = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                              outline="red", width=2)

    def update_draw(self, event):
        if not self.roi_enabled or self.rectangle_overlay is None:
            return
        self.end_x, self.end_y = event.x, event.y
        self.canvas.coords(self.rectangle_overlay, self.start_x, self.start_y, self.end_x, self.end_y)

    def rebuild_preview_mask(self):
    # Start with segmentation mask
        self.preview_mask = self.segmentation_mask.copy()

        # Overlay all polygon layers
        for layer in self.polygon_mask_layers:
            self.preview_mask = cv2.bitwise_or(self.preview_mask, layer)

        self.update_display()


    def finish_draw(self, event):
        if not self.roi_enabled:
            return
        self.end_x, self.end_y = event.x, event.y
        disp_w, disp_h = self.last_display_size
        img_h, img_w = self.image.shape[:2]
        scale_x = img_w / disp_w
        scale_y = img_h / disp_h

        x1 = int(min(self.start_x, self.end_x) * scale_x)
        x2 = int(max(self.start_x, self.end_x) * scale_x)
        y1 = int(min(self.start_y, self.end_y) * scale_y)
        y2 = int(max(self.start_y, self.end_y) * scale_y)
        self.roi = (x1, y1, x2, y2)
        print(f"ROI selected: {self.roi}")

    def apply_crop(self):
        if self.roi is None:
            print("No ROI selected.")
            return
        x1, y1, x2, y2 = self.roi
        self.image = cv2.resize(self.image[y1:y2, x1:x2], (self.original_shape[1], self.original_shape[0]))
        self.gray = cv2.resize(self.gray[y1:y2, x1:x2], (self.original_shape[1], self.original_shape[0]))
        self.preview_mask = cv2.resize(self.preview_mask[y1:y2, x1:x2], (self.original_shape[1], self.original_shape[0]))

        self.canvas.delete("all")
        self.rectangle_overlay = None
        self.update_segmentation()
        self.update_display()
        self.update_histogram()
        print("Applied crop and resized to original image shape.")
        self.roi_enabled = False

    def save_data(self):
        if self.final_mask is None:
            print("Please confirm a mask before saving.")
            return

        image_dir = os.path.join(self.script_dir, "images")
        mask_dir = os.path.join(self.script_dir, "masks")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        img_path = os.path.join(image_dir, self.filename)
        mask_path = os.path.join(mask_dir, self.filename)

        # ✅ Handle mask inversion here
        final_mask_to_save = 255 - self.final_mask if self.mask_inverted else self.final_mask

        cv2.imwrite(img_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, final_mask_to_save)

        print(f"Saved image to: {img_path}")
        print(f"Saved mask to: {mask_path}")


    def toggle_brush_mode(self):
        self.brush_mode = not self.brush_mode
        if self.brush_mode:
            self.brush_button.config(text="Deactivate Brush Tool", fg="green")
            self.edit_mode = False
            self.toggle_edit_mode(force_off=True)
            print("Brush mode ON. Use mouse to draw on mask.")
        else:
            self.brush_button.config(text="Activate Brush Tool", fg="black")
            print("Brush mode OFF.")

    def set_brush_size(self, val):
        self.brush_size = int(val)

if __name__ == "__main__":
    root = tk.Tk()
    app = ResidueSegmentationTool(root)
    root.mainloop()
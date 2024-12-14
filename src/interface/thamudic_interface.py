import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
from pathlib import Path
import cv2
import numpy as np
import threading
from typing import Optional
from config import GUI_CONFIG
from predict import ThamudicPredictor
from advanced_preprocessing import AdvancedImageProcessor

class ModernThamudicGUI:
    def __init__(self, model_path: str, label_mapping_path: str):
        self.window = tk.Tk()
        self.window.title(GUI_CONFIG['window_title'])
        self.window.geometry(GUI_CONFIG['window_size'])
        
        # Set theme
        self.style = ttk.Style()
        self.style.theme_use(GUI_CONFIG['theme'])
        
        # Initialize components
        self.predictor = ThamudicPredictor(model_path, label_mapping_path)
        self.processor = AdvancedImageProcessor()
        
        # Current image
        self.current_image: Optional[np.ndarray] = None
        self.current_image_path: Optional[str] = None
        self.processed_image: Optional[np.ndarray] = None
        
        self._create_widgets()
        self._setup_layout()
    
    def _create_widgets(self):
        # Create main container
        self.main_container = ttk.Frame(self.window, padding="10")
        
        # Create left panel for image display
        self.left_panel = ttk.Frame(self.main_container)
        self.image_label = ttk.Label(self.left_panel)
        self.image_label.pack(pady=5)
        
        # Create right panel for controls and results
        self.right_panel = ttk.Frame(self.main_container)
        
        # Control buttons
        self.button_frame = ttk.Frame(self.right_panel)
        self.load_btn = ttk.Button(
            self.button_frame,
            text="Load Image",
            command=self._load_image
        )
        self.process_btn = ttk.Button(
            self.button_frame,
            text="Process Image",
            command=self._process_image
        )
        self.recognize_btn = ttk.Button(
            self.button_frame,
            text="Recognize Text",
            command=self._recognize_text,
            state='disabled'
        )
        
        # Results area
        self.results_frame = ttk.LabelFrame(
            self.right_panel,
            text="Recognition Results",
            padding="5"
        )
        self.results_text = tk.Text(
            self.results_frame,
            height=15,
            width=40,
            wrap=tk.WORD
        )
        
        # Progress bar and status
        self.progress_frame = ttk.Frame(self.right_panel)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            self.progress_frame,
            textvariable=self.status_var
        )
    
    def _setup_layout(self):
        # Configure main container
        self.main_container.grid(
            row=0, column=0,
            sticky='nsew',
            padx=5, pady=5
        )
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        
        # Configure panels
        self.left_panel.grid(row=0, column=0, sticky='nsew', padx=5)
        self.right_panel.grid(row=0, column=1, sticky='nsew', padx=5)
        self.main_container.columnconfigure(0, weight=2)
        self.main_container.columnconfigure(1, weight=1)
        
        # Configure buttons
        self.button_frame.pack(fill='x', pady=5)
        self.load_btn.pack(side='left', padx=2)
        self.process_btn.pack(side='left', padx=2)
        self.recognize_btn.pack(side='left', padx=2)
        
        # Configure results area
        self.results_frame.pack(fill='both', expand=True, pady=5)
        self.results_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Configure progress area
        self.progress_frame.pack(fill='x', pady=5)
        self.progress_bar.pack(fill='x', pady=2)
        self.status_label.pack(fill='x')
    
    def _load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                self.current_image_path = file_path
                
                # Display image
                self._display_image(self.current_image)
                
                self.status_var.set("Image loaded successfully")
                self.process_btn.config(state='normal')
                self.recognize_btn.config(state='disabled')
                self.results_text.delete(1.0, tk.END)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def _process_image(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            # Process image
            self.processed_image = self.processor.preprocess_image(self.current_image)
            
            # Display processed image
            self._display_image(self.processed_image)
            
            self.status_var.set("Image processed successfully")
            self.recognize_btn.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def _recognize_text(self):
        if self.processed_image is None:
            messagebox.showwarning("Warning", "Please process the image first")
            return
        
        self.status_var.set("Recognizing text...")
        self.progress_var.set(0)
        self.results_text.delete(1.0, tk.END)
        
        # Run recognition in a separate thread
        thread = threading.Thread(target=self._run_recognition)
        thread.start()
    
    def _run_recognition(self):
        try:
            # Segment characters
            characters = self.processor.segment_characters(self.processed_image)
            
            results = []
            for i, char_img in enumerate(characters):
                # Update progress
                progress = (i + 1) / len(characters) * 100
                self.window.after(0, self.progress_var.set, progress)
                
                # Predict character
                char, conf = self.predictor.predict_single_character(char_img)
                results.append((char, conf))
            
            # Update results
            self.window.after(0, self._update_results, results)
            
        except Exception as e:
            self.window.after(0, self._show_error, str(e))
    
    def _update_results(self, results):
        text = "Recognition Results:\n\n"
        text += "Complete Text: " + ''.join(char for char, _ in results) + "\n\n"
        text += "Character Confidences:\n"
        for char, conf in results:
            text += f"{char}: {conf:.2f}\n"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, text)
        self.status_var.set("Recognition completed")
        self.progress_var.set(100)
    
    def _display_image(self, image: np.ndarray):
        # Convert to RGB
        if len(image.shape) == 2:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize while maintaining aspect ratio
        display_image = self._resize_image(
            display_image,
            GUI_CONFIG['max_display_size']
        )
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image=Image.fromarray(display_image))
        self.image_label.configure(image=photo)
        self.image_label.image = photo
    
    def _resize_image(self, image: np.ndarray, max_size: int) -> np.ndarray:
        height, width = image.shape[:2]
        if height > width:
            if height > max_size:
                ratio = max_size / height
                new_size = (int(width * ratio), max_size)
            else:
                return image
        else:
            if width > max_size:
                ratio = max_size / width
                new_size = (max_size, int(height * ratio))
            else:
                return image
        
        return cv2.resize(image, new_size)
    
    def _show_error(self, error_message: str):
        messagebox.showerror("Error", f"Recognition failed: {error_message}")
        self.status_var.set("Ready")
        self.progress_var.set(0)
    
    def run(self):
        self.window.mainloop()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Thamudic Script Recognition GUI')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--labels', type=str, required=True,
                      help='Path to label mapping file')
    
    args = parser.parse_args()
    
    app = ModernThamudicGUI(args.model, args.labels)
    app.run()

if __name__ == '__main__':
    main()

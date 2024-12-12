import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
from pathlib import Path
from predict import ThamudicPredictor
import cv2
import numpy as np
import threading
from typing import Optional

class ThamudicGUI:
    def __init__(self, model_path: str, label_mapping_path: str):
        self.window = tk.Tk()
        self.window.title("Thamudic Script Recognition")
        self.window.geometry("800x600")
        
        # Initialize predictor
        self.predictor = ThamudicPredictor(model_path, label_mapping_path)
        
        # Current image
        self.current_image: Optional[np.ndarray] = None
        self.current_image_path: Optional[str] = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky='nsew')
        
        # Image display area
        self.image_label = tk.Label(self.window)  # Initialize the Label
        self.image_label.pack()  # Add it to the window layout
        
        # Buttons
        ttk.Button(main_frame, text="Load Image", command=self._load_image).grid(
            row=1, column=0, pady=5, padx=5
        )
        ttk.Button(main_frame, text="Recognize", command=self._recognize_text).grid(
            row=1, column=1, pady=5, padx=5
        )
        
        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Recognition Results", padding="5")
        results_frame.grid(row=2, column=0, columnspan=2, sticky='nsew', pady=10)
        
        self.results_text = tk.Text(results_frame, height=10, width=60)
        self.results_text.grid(row=0, column=0, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky='nsew', pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=4, column=0, columnspan=2, pady=5)
    
    def _load_image(self):
        """
        Load an image file for recognition
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                self.current_image_path = file_path
                
                # Display image
                display_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                display_image = self._resize_image(display_image, max_size=400)
                
                print(f"Display image type: {type(display_image)}")  # Log the type of the image
                photo = ImageTk.PhotoImage(image=Image.fromarray(display_image))
              #  self.image_label.configure(image=photo)
               # self.image_label.image = photo  # Keep a reference
                
                self.status_var.set("Image loaded successfully")
                self.results_text.delete(1.0, tk.END)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def _recognize_text(self):
        """
        Run text recognition on the loaded image
        """
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        self.status_var.set("Processing...")
        self.progress_var.set(0)
        self.results_text.delete(1.0, tk.END)
        
        # Run recognition in a separate thread
        thread = threading.Thread(target=self._run_recognition)
        thread.start()
    
    def _run_recognition(self):
        """
        Run recognition in a separate thread
        """
        try:
            if self.current_image_path is None:
                self._show_error("No image loaded. Please load an image before recognition.")
                return

            # Create temporary file for visualization
            output_path = Path(self.current_image_path).parent / "temp_visualization.jpg"
            
            # Run prediction
            predictions = self.predictor.predict_inscription(
                self.current_image_path,
                str(output_path)
            )
            
            # Update results
            self.window.after(0, self._update_results, predictions)
            
            # Load and display annotated image
            annotated_image = cv2.imread(str(output_path))
            self.window.after(0, self._update_image, annotated_image)
            
            # Clean up
            output_path.unlink()
            
        except Exception as e:
            self.window.after(0, self._show_error, str(e))
    
    def _update_results(self, predictions):
        """
        Update results in the GUI
        """
        text = "Recognition Results:\n\n"
        text += "Complete Text: " + ''.join(char for char, _ in predictions) + "\n\n"
        text += "Character Confidences:\n"
        for char, conf in predictions:
            text += f"{char}: {conf:.2f}\n"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, text)
        self.status_var.set("Recognition completed")
        self.progress_var.set(100)
    
    def _update_image(self, image):
        """
        Update the displayed image
        """
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        display_image = self._resize_image(display_image, max_size=400)
        
        print(f"Display image type: {type(display_image)}")  # Log the type of the image
        photo = ImageTk.PhotoImage(image=Image.fromarray(display_image))
      #  self.image_label.configure(image=photo)
      #  self.image_label.image = photo  # Keep a reference
    
    def _show_error(self, error_message):
        """
        Show error message in GUI
        """
        messagebox.showerror("Error", f"Recognition failed: {error_message}")
        self.status_var.set("Ready")
        self.progress_var.set(0)
    
    @staticmethod
    def _resize_image(image: np.ndarray, max_size: int) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        """
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
    
    def run(self):
        """
        Start the GUI application
        """
        self.window.mainloop()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Thamudic Recognition GUI')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--label_mapping', type=str, required=True,
                        help='Path to label mapping file')
    
    args = parser.parse_args()
    
    app = ThamudicGUI(args.model_path, args.label_mapping)
    app.run()

if __name__ == '__main__':
    main()

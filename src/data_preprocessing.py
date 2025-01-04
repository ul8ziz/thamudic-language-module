import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from typing import List, Tuple, Union

class ThamudicPreprocessor:
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.augmentation = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
            ], p=0.2),
        ])

    def preprocess_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Preprocess a single image for the model."""
        # Handle both file path and numpy array
        if isinstance(image, str):
            # Read image from file
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not read image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            # If image is already loaded as numpy array
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                if image.dtype == np.float32 or image.dtype == np.float64:
                    image = (image * 255).astype(np.uint8)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        else:
            raise TypeError(f"Expected str or ndarray, got {type(image)}")
        
        # Apply preprocessing
        image = self._apply_basic_preprocessing(image)
        
        # Apply augmentation if training
        augmented = self.augmentation(image=image)
        processed_image = augmented['image']
        
        return processed_image

    def _apply_basic_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply basic preprocessing steps."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Noise removal
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Convert back to RGB
        processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        
        return processed

    def segment_characters(self, image: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
        """Segment individual characters from the image and return their bounding boxes."""
        # Convert to binary
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate average contour area
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            avg_area = np.mean(areas)
            min_area = avg_area * 0.1  # Min area threshold
        else:
            min_area = 100  # Default min area
        
        # Sort contours by x-coordinate (right to left for Arabic)
        def get_contour_precedence(contour):
            x, y, w, h = cv2.boundingRect(contour)
            return (-x)  # Negative for right-to-left order
        
        contours = sorted(contours, key=get_contour_precedence)
        
        # Extract character regions and their bounding boxes
        characters = []
        bounding_boxes = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter based on area and aspect ratio
            aspect_ratio = float(w)/h if h > 0 else 0
            if area > min_area and 0.2 < aspect_ratio < 5:
                # Add padding around the character
                padding = 5
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(image.shape[1], x + w + padding)
                y_end = min(image.shape[0], y + h + padding)
                
                # Extract and resize character region
                char_region = image[y_start:y_end, x_start:x_end]
                char_region = cv2.resize(char_region, self.image_size)
                
                characters.append(char_region)
                bounding_boxes.append((x_start, y_start, x_end - x_start, y_end - y_start))
        
        return characters, bounding_boxes

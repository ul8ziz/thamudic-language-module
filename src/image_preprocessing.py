import cv2
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import List, Tuple, Optional, Union
from PIL import Image

class ThamudicImagePreprocessor:
    def __init__(self):
        self.train_transforms = self.get_train_transforms()
        self.val_transforms = self.get_val_transforms()

    def get_train_transforms(self, height: int = 128, width: int = 128) -> T.Compose:
        return T.Compose([
            T.Resize((height, width), antialias=True),
            T.RandomApply([
                T.RandomRotation(10),
                T.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                )
            ], p=0.5),
            T.RandomApply([
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            T.RandomApply([
                T.RandomPerspective(distortion_scale=0.2)
            ], p=0.3),
            T.RandomApply([
                T.RandomAdjustSharpness(sharpness_factor=2)
            ], p=0.3),
            T.RandomAutocontrast(p=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def get_val_transforms(self, height: int = 128, width: int = 128) -> T.Compose:
        return T.Compose([
            T.Resize((height, width), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def ensure_tensor(self, image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
        """
        Ensure the image is a tensor with the correct format
        """
        if isinstance(image, Image.Image):
            return F.to_tensor(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 2:  # Add channel dimension if needed
                image = image.unsqueeze(0)
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def apply_transforms(self, image: Union[torch.Tensor, Image.Image], transforms: T.Compose) -> torch.Tensor:
        """
        Apply torchvision transforms to an image
        
        Args:
            image: Input image (PIL Image or tensor)
            transforms: Torchvision transforms to apply
            
        Returns:
            Transformed image as tensor
        """
        # If image is already a tensor, no need to convert it first
        if isinstance(image, torch.Tensor):
            transformed = transforms(image)
            return transformed
        
        # If image is PIL Image, apply transforms directly
        if isinstance(image, Image.Image):
            # Convert PIL Image to Tensor
            image_tensor = T.ToTensor()(image)
            transformed = transforms(image_tensor)
            return transformed
            
        raise ValueError(f"Unsupported image type: {type(image)}")

    def preprocess_image(self, image_input: Union[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess a single image for the Thamudic recognition model
        
        Args:
            image_input: Either a path to an image file (str) or a numpy array containing the image
        """
        # Handle different input types
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('L')  # Convert to grayscale
            if image is None:
                raise ValueError(f"Could not read image at {image_input}")
        elif isinstance(image_input, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(image_input)
            if image.mode != 'L':
                image = image.convert('L')  # Convert to grayscale if needed
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Apply preprocessing
        processed = self.apply_transforms(image, self.val_transforms)
        return processed

    def segment_inscription(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Segment an inscription image into individual characters
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours from right to left (for Arabic/Thamudic text direction)
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0], reverse=True)
        
        characters = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 100:  # Filter out noise
                char_image = image[y:y+h, x:x+w]
                characters.append(char_image)
        
        return characters

    def enhance_inscription(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance the quality of inscription images
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened

    def batch_process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Process all images in a directory
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for img_path in input_path.glob("*.jpg"):
            try:
                # Load and preprocess image
                processed = self.preprocess_image(str(img_path))
                
                # Convert tensor to numpy array for saving
                processed_np = processed.numpy().transpose(1, 2, 0)  # CHW -> HWC
                processed_np = (processed_np * 255).astype(np.uint8)
                
                # Save processed image
                output_file = output_path / img_path.name
                cv2.imwrite(str(output_file), processed_np)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

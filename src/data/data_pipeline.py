import os
import json
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import albumentations as A
from typing import Tuple, List, Any
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import tensorflow as tf

def preprocess_image(image, target_size=(128, 128)):
    """
    تحسين معالجة الصور قبل التدريب
    """
    # Convert to float32 and normalize
    image = tf.cast(image, tf.float32) / 255.0
    
    # Add padding to make the image square while preserving aspect ratio
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    max_dim = tf.maximum(height, width)
    pad_height = (max_dim - height) // 2
    pad_width = (max_dim - width) // 2
    
    # Add padding
    paddings = tf.convert_to_tensor([[pad_height, max_dim - height - pad_height],
                                   [pad_width, max_dim - width - pad_width],
                                   [0, 0]])
    image = tf.pad(image, paddings, mode='CONSTANT', constant_values=1.0)
    
    # Resize to target size
    image = tf.image.resize(image, target_size, method='bilinear')
    
    # Ensure correct number of channels
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    
    return image

def load_data(data_dir, label_mapping_file, target_size=(128, 128)):
    """
    تحميل البيانات مع تحسينات في المعالجة
    """
    print("\nLoading and preprocessing data...")
    
    # Load label mapping
    with open(label_mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
        
    # Create label mapping from thamudic letters
    label_mapping = {str(letter['index'] + 1): letter['symbol'] for letter in mapping_data['thamudic_letters']}
    
    images = []
    labels = []
    label_names = []
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Get list of letter directories
    letter_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    letter_dirs.sort(key=lambda x: int(x.split('_')[1]))  # Sort by letter number
    
    if not letter_dirs:
        raise ValueError(f"No letter directories found in {data_dir}")
    
    # Track progress
    total_letters = len(letter_dirs)
    processed_letters = 0
    
    for letter_dir in letter_dirs:
        processed_letters += 1
        print(f"\rProcessing letter: {processed_letters}/{total_letters}", end="")
        
        # Extract letter number from directory name
        letter_num = letter_dir.split('_')[1]
        
        # Get label from mapping
        symbol = label_mapping.get(letter_num)
        if symbol is None:
            print(f"\nWarning: No label mapping found for letter {letter_num}")
            continue
            
        # Find the corresponding letter info
        letter_info = next((letter for letter in mapping_data['thamudic_letters'] 
                          if letter['index'] == int(letter_num) - 1), None)
        
        if letter_info is None:
            print(f"\nWarning: No letter information found for letter {letter_num}")
            continue
            
        letter_path = os.path.join(data_dir, letter_dir)
        image_files = [f for f in os.listdir(letter_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        for image_file in image_files:
            image_path = os.path.join(letter_path, image_file)
            try:
                # Load and preprocess image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"\nWarning: Could not load image {image_path}")
                    continue
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply preprocessing
                processed_image = preprocess_image(image, target_size)
                
                images.append(processed_image)
                labels.append(int(letter_num) - 1)  # Use 0-based index for labels
                label_names.append(letter_info['symbol'])
                
            except Exception as e:
                print(f"\nError processing image {image_path}: {str(e)}")
                continue
    
    print("\nData loading completed!")
    
    if not images:
        raise ValueError("No valid images were loaded! Check the data directory structure and image files.")
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Total images: {len(images)}")
    print(f"Number of classes: {len(np.unique(labels))}")
    print(f"Image shape: {images[0].shape}")
    
    # Print class distribution
    class_counts = {}
    for label in labels:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    print("\nClass distribution:")
    for label, count in class_counts.items():
        letter_info = mapping_data['thamudic_letters'][label]
        try:
            print(f"Class {label} ({letter_info['name']}): {count} images")
        except UnicodeEncodeError:
            print(f"Class {label}: {count} images")
    
    return images, labels, label_names

def augment_image(image, num_augmentations=3):
    """
    Generate augmented versions of the input image
    
    Args:
        image (np.ndarray): Input image
        num_augmentations (int): Number of augmented versions required
    
    Returns:
        List[np.ndarray]: List of augmented images
    """
    augmentations = [
        A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=15, 
                p=0.5
            ),
        ])
        for _ in range(num_augmentations)
    ]
    
    augmented_images = []
    for aug in augmentations:
        # Ensure the image is of type uint8
        image_uint8 = (image * 255).astype(np.uint8)
        
        augmented = aug(image=image_uint8)['image']
        augmented_normalized = augmented / 255.0
        augmented_images.append(augmented_normalized)
    
    return augmented_images

def split_data(
    images: List[np.ndarray], 
    labels: List[int], 
    test_size: float = 0.2, 
    val_size: float = 0.2
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Split the data into training, testing, and validation sets
    
    Args:
        images (List[np.ndarray]): Images
        labels (List[int]): Labels
        test_size (float): Test set size as a proportion of the total dataset
        val_size (float): Validation set size as a proportion of the training set
    
    Returns:
        Tuple[Any, Any, Any, Any, Any, Any]: 
        x_train, x_test, x_val, y_train, y_test, y_val
    """
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Calculate class statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_labels)
    n_samples = len(labels)
    min_samples = min(counts)
    
    print(f"Total samples: {n_samples}")
    print(f"Number of classes: {n_classes}")
    print(f"Minimum samples per class: {min_samples}")
    
    # Create indices for each class
    class_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    
    # Initialize empty arrays for each split
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Split each class proportionally
    for label in unique_labels:
        indices = class_indices[label]
        n_samples_class = len(indices)
        
        # Calculate split sizes
        n_test = max(1, int(n_samples_class * 0.1))  # 10% for test
        n_val = max(1, int((n_samples_class - n_test) * 0.1))  # 10% of remaining for val
        n_train = n_samples_class - n_test - n_val
        
        # Shuffle indices
        np.random.shuffle(indices)
        
        # Split indices
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
    
    # Convert indices to arrays
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    # Create final splits
    x_train = images[train_indices]
    y_train = labels[train_indices]
    x_val = images[val_indices]
    y_val = labels[val_indices]
    x_test = images[test_indices]
    y_test = labels[test_indices]
    
    print(f"Training samples: {len(x_train)} (labels: {len(y_train)})")
    print(f"Validation samples: {len(x_val)} (labels: {len(y_val)})")
    print(f"Testing samples: {len(x_test)} (labels: {len(y_test)})")
    
    # Verify shapes
    print("\nData shapes:")
    print(f"x_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"x_val: {x_val.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"x_test: {x_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    return x_train, x_test, x_val, y_train, y_test, y_val

class ThamudicDataset(Dataset):
    def __init__(self, data_dir: str, label_mapping_file: str, transform=None, train: bool = True):
        self.data_dir = Path(data_dir) / 'thamudic'
        self.transform = transform
        self.train = train
        
        # Load label mapping
        with open(label_mapping_file, 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)
            
        # Get all image paths
        self.image_paths = []
        for img_path in self.data_dir.glob('**/*.png'):
            if img_path.stem.startswith('letter_'):
                self.image_paths.append(img_path)
        
        if not self.image_paths:
            raise ValueError(f"No images found in {data_dir}")
            
        print(f"Found {len(self.image_paths)} images in {'training' if train else 'validation'} set")
        
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = str(self.image_paths[idx])
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        label_str = self.image_paths[idx].stem
        label = int(self.label_mapping[label_str])
        return image, label

class AdvancedImageProcessor:
    def __init__(self):
        self.augmentation = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2.0, p=0.5),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
            ], p=0.2),
        ])

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(bilateral)
        
        # Apply augmentation if available
        if self.augmentation:
            augmented = self.augmentation(image=enhanced)
            enhanced = augmented['image']
            
        return enhanced

def get_data_loaders(data_dir: str, label_mapping_file: str, batch_size: int = 32, num_workers: int = 4):
    """Create train and validation data loaders"""
    train_dataset = ThamudicDataset(data_dir, label_mapping_file, train=True)
    val_dataset = ThamudicDataset(data_dir, label_mapping_file, train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader

def main():
    # Load the data
    images, labels, label_names = load_data(data_dir='path_to_data', label_mapping_file='path_to_label_mapping', target_size=(128, 128))
    
    # Split the data
    x_train, x_test, x_val, y_train, y_test, y_val = split_data(images, labels)
    
    print("Data information:")
    print(f"Total number of images: {len(images)}")
    print(f"Number of training images: {len(x_train)}")
    print(f"Number of testing images: {len(x_test)}")
    print(f"Number of validation images: {len(x_val)}")

if __name__ == "__main__":
    main()

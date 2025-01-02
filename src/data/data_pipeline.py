import os
import json
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import albumentations as A
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
import tensorflow_addons as tfa
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def augment_image(img):
    """
    توليد نسخة معدلة من الصورة مع تحسينات متقدمة
    """
    try:
        # Random flip
        if tf.random.uniform([], 0, 1) > 0.5:
            img = tf.image.flip_left_right(img)
        
        # Random rotation
        angle = tf.random.uniform([], -0.2, 0.2)
        img = tfa.image.rotate(img, angle)
        
        # Random brightness and contrast
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        
        # Random saturation and hue
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_hue(img, 0.1)
        
        # Ensure values are in [0,1]
        img = tf.clip_by_value(img, 0.0, 1.0)
        
        return img
        
    except Exception as e:
        logging.error(f"Error during augmentation: {str(e)}")
        return None

def preprocess_image(image_path):
    """
    معالجة الصورة وتحويلها إلى التنسيق المناسب
    """
    try:
        # Read image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        
        # Convert to float32
        img = tf.cast(img, tf.float32)
        
        # Normalize to [0,1]
        img = img / 255.0
        
        # Get image dimensions
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        
        # Calculate scaling factor
        target_size = tf.constant([128, 128], dtype=tf.int32)
        scale = tf.minimum(
            tf.cast(target_size[0], tf.float32) / tf.cast(h, tf.float32),
            tf.cast(target_size[1], tf.float32) / tf.cast(w, tf.float32)
        )
        
        # Calculate new dimensions
        new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
        new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
        
        # Resize image
        img = tf.image.resize(img, [new_h, new_w])
        
        # Calculate padding
        pad_h = target_size[0] - new_h
        pad_w = target_size[1] - new_w
        
        # Pad image
        paddings = [[0, pad_h], [0, pad_w], [0, 0]]
        img = tf.pad(img, paddings, mode='CONSTANT', constant_values=1.0)
        
        # Ensure output shape
        img = tf.ensure_shape(img, [128, 128, 3])
        
        return img
        
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None

def load_data(data_dir, label_mapping_file):
    """
    تحميل وتحسين البيانات مع معالجة متقدمة للصور
    """
    try:
        # Load label mapping
        with open(label_mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            letter_mapping = mapping_data['thamudic_letters']
        
        images = []
        labels = []
        label_counts = {}
        
        # Process each letter directory
        for letter_idx in range(len(letter_mapping)):
            letter_dir = f"letter_{letter_idx + 1}"  # Directory names start from 1
            letter_path = os.path.join(data_dir, letter_dir)
            
            if not os.path.exists(letter_path):
                logging.warning(f"Directory not found: {letter_path}")
                continue
                
            # Get all image files
            img_files = [f for f in os.listdir(letter_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not img_files:
                logging.warning(f"No images found in {letter_path}")
                continue
            
            # Process each image
            for img_file in img_files:
                img_path = os.path.join(letter_path, img_file)
                img = preprocess_image(img_path)
                
                if img is not None:
                    images.append(img)
                    labels.append(letter_idx)  # Use 0-based index for labels
                    label_counts[letter_idx] = label_counts.get(letter_idx, 0) + 1
                    
                    # Generate augmented images for underrepresented classes
                    if label_counts[letter_idx] < 10:  # Minimum samples per class
                        for _ in range(3):  # Generate 3 augmented versions
                            aug_img = augment_image(img)
                            if aug_img is not None:
                                images.append(aug_img)
                                labels.append(letter_idx)
                                label_counts[letter_idx] += 1
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Print dataset statistics
        logging.info("\nDataset Statistics:")
        logging.info(f"Total images: {len(images)}")
        logging.info(f"Number of classes: {len(letter_mapping)}")
        logging.info(f"Images per class: {dict(Counter(labels))}")
        
        if len(images) == 0:
            raise ValueError("No valid images were loaded")
            
        # Convert labels to categorical
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(letter_mapping))
            
        return images, labels, len(letter_mapping)
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

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

class ThamudicDataset:
    """
    مجموعة بيانات الخط الثمودي مع معالجة متقدمة للصور
    """
    def __init__(self, data_dir: str, label_mapping_file: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load label mapping
        with open(label_mapping_file, 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)
            
        # Get all image paths
        self.image_paths = []
        self.labels = []
        
        # Create label to index mapping
        self.label_to_index = {str(k): i for i, k in enumerate(sorted(self.label_mapping.keys()))}
        
        # Load all images and labels
        for letter_dir in os.listdir(self.data_dir):
            if not letter_dir.startswith('letter_'):
                continue
                
            letter_num = letter_dir.split('_')[1]
            if letter_num not in self.label_to_index:
                continue
                
            letter_path = self.data_dir / letter_dir
            for img_file in letter_path.glob('*.png'):
                self.image_paths.append(img_file)
                self.labels.append(self.label_to_index[letter_num])
        
        if not self.image_paths:
            raise ValueError(f"No images found in {data_dir}")
            
        logging.info(f"Found {len(self.image_paths)} images")
        logging.info(f"Number of classes: {len(set(self.labels))}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        img = preprocess_image(img_path)
        
        # Apply additional transforms if specified
        if self.transform:
            img = self.transform(image=img)['image']
        
        return img, label
        
    def get_class_weights(self):
        """
        حساب أوزان الفئات للتعامل مع عدم التوازن في البيانات
        """
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        class_weights = total_samples / (len(class_counts) * class_counts)
        return class_weights

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
    images, labels, num_classes = load_data(data_dir='path_to_data', label_mapping_file='path_to_label_mapping')
    
    # Split the data
    x_train, x_test, x_val, y_train, y_test, y_val = split_data(images, labels)
    
    print("Data information:")
    print(f"Total number of images: {len(images)}")
    print(f"Number of training images: {len(x_train)}")
    print(f"Number of testing images: {len(x_test)}")
    print(f"Number of validation images: {len(x_val)}")

if __name__ == "__main__":
    main()

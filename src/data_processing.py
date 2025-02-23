"""
Data processing utilities for Thamudic character recognition
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import logging
import json
from typing import Tuple

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'data'
MODEL_CONFIG = {
    'image_size': 224,
    'batch_size': 32,
    'num_classes': 28,
    'validation_split': 0.2
}

class ThamudicDataset(Dataset):
    """Dataset class for Thamudic character images"""
    
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Load letter mapping to ensure correct order
        mapping_path = self.root_dir.parent / 'letter_mapping.json'
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            self.classes = [f"letter_{item['letter']}" for item in mapping_data['thamudic_letters']]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()
        
        # Calculate class weights for imbalanced dataset
        self.class_weights = self._calculate_class_weights()
        
    def _load_samples(self):
        """Load all image samples and their labels"""
        samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = class_dir / img_name
                    samples.append((str(img_path), self.class_to_idx[class_name]))
        return samples
    
    def _calculate_class_weights(self):
        """Calculate class weights to handle imbalanced dataset"""
        class_counts = {}
        total_samples = len(self.samples)
        
        # Count samples per class
        for _, class_idx in self.samples:
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        
        # Calculate weights
        weights = torch.zeros(len(self.classes))
        for class_idx, count in class_counts.items():
            weights[class_idx] = total_samples / (len(self.classes) * count)
        
        return weights
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        img_path, class_idx = self.samples[idx]
        
        try:
            # Load and convert image
            with Image.open(img_path).convert('RGB') as img:
                # Apply transformations
                if self.transform:
                    img = self.transform(img)
                else:
                    # Default transformations if none provided
                    default_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                    ])
                    img = default_transform(img)
                
                return img, class_idx
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            # Return a blank image in case of error
            return torch.zeros((3, 224, 224)), class_idx

def create_data_loaders(
    data_dir: str,
    transform=None,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation
    
    Args:
        data_dir: Path to the data directory
        transform: Transformations to apply to the images
        batch_size: Batch size
        val_split: Validation split
        num_workers: Number of workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    dataset = ThamudicDataset(data_dir, transform=transform)
    
    # Split the data
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
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

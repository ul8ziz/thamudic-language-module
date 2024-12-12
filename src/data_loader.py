import os
import json
from typing import Tuple, Dict, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from image_preprocessing import ThamudicImagePreprocessor

class ThamudicDataset(Dataset):
    def __init__(self, data_dir: str, label_mapping_file: str, transform: Optional[ThamudicImagePreprocessor] = None, train: bool = True):
        self.data_dir = Path(data_dir) / 'thamudic'
        self.preprocessor = transform if transform else ThamudicImagePreprocessor()
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
        
        # Load and preprocess image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Apply transformations
        transforms = self.preprocessor.get_train_transforms() if self.train else self.preprocessor.get_val_transforms()
        image = self.preprocessor.apply_transforms(image, transforms)
            
        # Get label from filename (format: letter_X.png where X is the index)
        label_str = self.image_paths[idx].stem  # Get filename without extension
        try:
            # Use label mapping to get the correct class index
            label = int(self.label_mapping[label_str])
        except KeyError:
            raise ValueError(f"Label not found in mapping for image: {img_path}")
        
        return image, label

def get_data_loaders(
    data_dir: str,
    label_mapping_file: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders using separate directories
    """
    # Create image preprocessor
    preprocessor = ThamudicImagePreprocessor()
    
    # Create datasets
    train_dataset = ThamudicDataset(
        os.path.join(data_dir, 'train'),
        label_mapping_file,
        transform=preprocessor,
        train=True
    )
    
    val_dataset = ThamudicDataset(
        os.path.join(data_dir, 'val'),
        label_mapping_file,
        transform=preprocessor,
        train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def create_label_mapping(data_dir: str, output_file: str) -> Dict[str, int]:
    """
    Create a mapping of Thamudic characters to numerical labels
    """
    # Get all unique labels from filenames
    labels = set()
    for img_path in Path(data_dir).rglob('*.png'):
        if img_path.stem.startswith('letter_'):
            labels.add(img_path.stem)
    
    # Create mapping
    label_mapping = {label: idx for idx, label in enumerate(sorted(labels))}
    
    # Save mapping
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=4)
    
    print(f"Created label mapping with {len(label_mapping)} classes")
    return label_mapping

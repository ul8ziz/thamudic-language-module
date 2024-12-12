import os
import shutil
from pathlib import Path
import random

def split_dataset(data_dir, train_ratio=0.8):
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    
    # Create validation directory if it doesn't exist
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all image files
    image_files = list(train_dir.glob('*.png'))
    random.shuffle(image_files)
    
    # Calculate split point
    split_idx = int(len(image_files) * train_ratio)
    val_files = image_files[split_idx:]
    
    # Move validation files
    for file_path in val_files:
        shutil.move(str(file_path), str(val_dir / file_path.name))
    
    print(f'Moved {len(val_files)} files to validation set')

if __name__ == '__main__':
    data_dir = 'data/processed_dataset'
    split_dataset(data_dir)

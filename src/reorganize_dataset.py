import os
import shutil
from pathlib import Path
import re

def reorganize_dataset(source_dir, target_dir):
    """
    Reorganize the dataset by renaming files and moving them to appropriate directories
    """
    # Create target directory if it doesn't exist
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Create train directory
    train_dir = target_path / 'train'
    train_dir.mkdir(exist_ok=True)
    
    # Process each file in the source directory
    source_path = Path(source_dir)
    for file_path in source_path.glob('**/*.png'):
        # Get the file name
        file_name = file_path.name
        
        # Extract the index from the file name
        if 'letter_' in file_name:
            # Already in correct format
            new_name = file_name
        else:
            # Extract number from file name
            match = re.search(r'_(\d+)\.png$', file_name)
            if match:
                index = match.group(1)
                new_name = f'letter_{index}.png'
            else:
                print(f"Skipping file {file_name} - no index found")
                continue
        
        # Copy file to new location
        shutil.copy2(file_path, train_dir / new_name)
        print(f"Copied {file_name} to {new_name}")

def main():
    source_dir = 'data/train_dataset/train/thamudic'
    target_dir = 'data/processed_dataset'
    
    print("Starting dataset reorganization...")
    reorganize_dataset(source_dir, target_dir)
    print("Dataset reorganization completed!")

if __name__ == '__main__':
    main()

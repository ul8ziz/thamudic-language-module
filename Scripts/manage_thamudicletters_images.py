import os
import json
import shutil
import cv2
import numpy as np
from typing import List, Dict

def validate_image(image_path: str) -> bool:
    """
    Validate image quality and characteristics
    
    Args:
        image_path (str): Path to the image
    
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error reading image: {image_path}")
            return False
        
        # Check image dimensions
        height, width = image.shape[:2]
        if height < 50 or width < 50:
            print(f"Image too small: {image_path}")
            return False
        
        # Check image contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)
        if variance < 10:
            print(f"Image has low contrast: {image_path}")
            return False
        
        return True
    except Exception as e:
        print(f"Error validating image {image_path}: {e}")
        return False

def clean_and_organize_images(target_base_dir: str) -> Dict:
    """
    Clean and organize Thamudic letter images
    
    Args:
        target_base_dir (str): Base path for letter folders
    
    Returns:
        dict: Statistics about processed images
    """
    stats = {
        "total_images": 0,
        "valid_images": 0,
        "removed_images": 0
    }
    
    # Use absolute path for mapping file
    mapping_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'letters', 'letter_mapping.json')
    
    # Read mapping file
    with open(mapping_path, "r", encoding="utf-8") as f:
        letter_mapping = json.load(f)
    
    # Iterate through Thamudic letters
    for letter_data in letter_mapping['thamudic_letters']:
        index = letter_data['index']
        letter_dir = os.path.join(target_base_dir, f"letter_{index+1}")
        
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)
        
        # Collect images
        image_files = [f for f in os.listdir(letter_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        stats["total_images"] += len(image_files)
        
        # Clean and rename images
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(letter_dir, img_file)
            
            if validate_image(img_path):
                # Rename with consistent format
                new_name = f"letter_{index}_{i}.png"
                new_path = os.path.join(letter_dir, new_name)
                
                # Convert to grayscale and save
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(new_path, image)
                
                # Remove old image
                if new_path != img_path:
                    os.remove(img_path)
                
                stats["valid_images"] += 1
            else:
                # Remove invalid images
                os.remove(img_path)
                stats["removed_images"] += 1
    
    return stats

def update_letter_mapping(target_base_dir: str) -> None:
    """
    Update mapping file with updated image paths
    
    Args:
        target_base_dir (str): Base path for letter folders
    """
    # Use absolute path for mapping file
    mapping_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'letters', 'letter_mapping.json')
    
    # Read mapping file
    with open(mapping_path, "r", encoding="utf-8") as f:
        letter_mapping = json.load(f)
    
    # Update image paths for Thamudic letters
    for letter_data in letter_mapping['thamudic_letters']:
        index = letter_data['index']
        letter_dir = os.path.join(target_base_dir, f"letter_{index+1}")
        
        # Collect image paths
        image_paths = []
        if os.path.exists(letter_dir):
            for img_file in os.listdir(letter_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    image_paths.append(os.path.join(letter_dir, img_file))
        
        # Update image paths in mapping
        letter_data['images'] = image_paths
    
    # Save updated mapping
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(letter_mapping, f, ensure_ascii=False, indent=4)
    
    print("Updated image paths in mapping file.")

def main():
    # Path to Thamudic letters directory
    target_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'letters', 'thamudic_letters')
    
    # Clean and organize images
    stats = clean_and_organize_images(target_base_dir)
    
    print("Image processing statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"Valid images: {stats['valid_images']}")
    print(f"Removed images: {stats['removed_images']}")
    
    # Update mapping file
    update_letter_mapping(target_base_dir)

if __name__ == "__main__":
    main()

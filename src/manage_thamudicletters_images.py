import os
import json
import shutil
import cv2
import numpy as np
from typing import List, Dict

def validate_image(image_path: str) -> bool:
    """
    Validate image quality and format
    
    Args:
        image_path (str): Path to the image
    
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error reading image: {image_path}")
            return False
        
        # Check image dimensions
        height, width = image.shape[:2]
        if height < 50 or width < 50:
            print(f"Image too small: {image_path}")
            return False
        
        # Check variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)
        if variance < 10:
            print(f"Image too dark: {image_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating image {image_path}: {str(e)}")
        return False

def clean_and_organize_images(target_base_dir: str) -> Dict:
    """
    Clean and organize Thamudic letter images
    
    Args:
        target_base_dir (str): Base directory for letters
    
    Returns:
        dict: Statistics about organized images
    """
    stats = {
        'total_images': 0,
        'valid_images': 0,
        'invalid_images': 0,
        'organized_letters': 0
    }
    
    # Ensure base directory exists
    os.makedirs(target_base_dir, exist_ok=True)
    
    # Create thamudic_letters directory
    letters_dir = os.path.join(target_base_dir, 'thamudic_letters')
    os.makedirs(letters_dir, exist_ok=True)
    
    # Read letter information
    mapping_file = os.path.join(target_base_dir, 'letter_mapping.json')
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    # Get base directory for relative paths
    base_dir = os.path.dirname(os.path.dirname(target_base_dir))
    
    # Process each letter
    for letter in mapping_data['thamudic_letters']:
        letter_idx = letter['index'] + 1
        letter_dir = os.path.join(letters_dir, f'letter_{letter_idx}')
        os.makedirs(letter_dir, exist_ok=True)
        
        # Process letter images
        for img_path in letter.get('images', []):
            stats['total_images'] += 1
            
            # Convert relative path to absolute
            abs_img_path = os.path.join(base_dir, img_path)
            
            if not os.path.exists(abs_img_path):
                print(f"Warning: Image not found: {abs_img_path}")
                stats['invalid_images'] += 1
                continue
            
            if validate_image(abs_img_path):
                # Copy image to appropriate directory
                new_filename = f"letter_{letter_idx}_{stats['valid_images']}.png"
                new_path = os.path.join(letter_dir, new_filename)
                
                try:
                    # Read and save image in standard format
                    img = cv2.imread(abs_img_path)
                    cv2.imwrite(new_path, img)
                    stats['valid_images'] += 1
                except Exception as e:
                    print(f"Error processing image {abs_img_path}: {str(e)}")
                    stats['invalid_images'] += 1
            else:
                stats['invalid_images'] += 1
        
        stats['organized_letters'] += 1
    
    return stats

def update_letter_mapping(target_base_dir: str):
    """
    Update mapping file with new image paths
    
    Args:
        target_base_dir (str): Base directory for letters
    """
    # Read current file
    mapping_file = os.path.join(target_base_dir, 'letter_mapping.json')
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    letters_dir = os.path.join(target_base_dir, 'thamudic_letters')
    
    # Update image paths for each letter
    for letter in mapping_data['thamudic_letters']:
        letter_idx = letter['index'] + 1
        letter_dir = os.path.join(letters_dir, f'letter_{letter_idx}')
        
        if os.path.exists(letter_dir):
            # Collect new image paths
            images = []
            for img in os.listdir(letter_dir):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    rel_path = os.path.join('data', 'letters', 'thamudic_letters', 
                                          f'letter_{letter_idx}', img)
                    images.append(rel_path.replace('\\', '/'))
            
            # Update image paths
            letter['images'] = sorted(images)
    
    # Save updates
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=4)

def main():
    """
    Main function for organizing and updating images
    """
    # Set base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_base_dir = os.path.join(base_dir, 'data', 'letters')
    
    print("Starting image processing and organization...")
    stats = clean_and_organize_images(target_base_dir)
    
    print("\nImage Processing Statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"Valid images: {stats['valid_images']}")
    print(f"Invalid images: {stats['invalid_images']}")
    print(f"Organized letters: {stats['organized_letters']}")
    
    print("\nUpdating mapping file...")
    update_letter_mapping(target_base_dir)
    print("Processing and update completed!")

if __name__ == "__main__":
    main()

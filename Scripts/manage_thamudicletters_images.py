import os
import json
import shutil

def update_letter_mapping(target_base_dir):
    """
    Update letter_mapping.json with image paths for each letter
    
    Args:
        target_base_dir (str): Base directory with organized letter images
    """
    with open("../data/letters/letter_mapping.json", "r", encoding="utf-8") as f:
        letter_mapping = json.load(f)
    
    # Update images for Thamudic letters
    for letter_data in letter_mapping['thamudic_letters']:
        index = letter_data['index']
        
        # Construct letter directory path
        letter_dir = os.path.join(target_base_dir, f"letter_{index+1}")
        
        # Collect all image paths for this letter
        image_paths = []
        if os.path.exists(letter_dir):
            for img_file in os.listdir(letter_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    image_paths.append(os.path.join(letter_dir, img_file))
        
        # Update letter mapping with image paths
        letter_data['images'] = image_paths
    
    # Save updated mapping
    with open("../data/letters/letter_mapping.json", "w", encoding="utf-8") as f:
        json.dump(letter_mapping, f, ensure_ascii=False, indent=4)
    
    print("Letter mapping updated with image paths.")

def main():
    # مسار مجلد الحروف الثمودية
    target_base_dir = "../data/letters/thamudic_letters"
    
    # Update letter mapping
    update_letter_mapping(target_base_dir)

if __name__ == "__main__":
    main()

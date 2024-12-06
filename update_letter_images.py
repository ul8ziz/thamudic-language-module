import json
import os
import glob
from pathlib import Path

# Define absolute paths
base_path = "d:/Work/rizg/Thamudic_language_recognition/projact/thamudic_env"
json_path = os.path.join(base_path, "data/letters/letter_mapping.json")
letters_base_dir = os.path.join(base_path, "data/letters/thamudic_letters")

# Read the existing JSON file
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Update images for each letter
for letter in data["thamudic_letters"]:
    letter_dir = f"letter_{letter['index'] + 1}"
    full_dir_path = os.path.join(letters_base_dir, letter_dir)
    
    # Get all PNG files in the letter directory
    if os.path.exists(full_dir_path):
        # Use glob to find all PNG files
        image_pattern = os.path.join(full_dir_path, "*.png")
        images = glob.glob(image_pattern)
        
        # Update the images array for this letter with all found images
        letter["images"] = sorted(images)  # Sort to ensure consistent order
    else:
        # If directory doesn't exist, set empty images array
        letter["images"] = []

# Write the updated JSON back to file
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Successfully updated letter_mapping.json with all available images.")

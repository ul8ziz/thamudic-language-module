"""
Script to reorganize training data to match the correct Arabic letter sequence
"""

import json
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def reorganize_data():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    letters_dir = data_dir / 'letters'
    temp_dir = data_dir / 'letters_temp'
    
    # Load letter mapping
    with open(data_dir / 'letter_mapping.json', 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    # Create temporary directory
    temp_dir.mkdir(exist_ok=True)
    
    # Move files to temporary location with correct names
    for idx, letter_data in enumerate(mapping_data['thamudic_letters']):
        old_dir = letters_dir / f'letter_{idx + 1}'
        if not old_dir.exists():
            logging.warning(f"Directory not found: {old_dir}")
            continue
            
        new_dir = temp_dir / f"letter_{letter_data['letter']}"
        if new_dir.exists():
            shutil.rmtree(new_dir)
        
        shutil.copytree(old_dir, new_dir)
        logging.info(f"Moved {old_dir} to {new_dir}")
    
    # Remove old letters directory and rename temp to letters
    shutil.rmtree(letters_dir)
    temp_dir.rename(letters_dir)
    
    logging.info("Data reorganization complete!")

if __name__ == '__main__':
    reorganize_data()

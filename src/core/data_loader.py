import os
import json
import logging
import numpy as np
from PIL import Image
import tensorflow as tf

def load_and_preprocess_data(data_dir, mapping_file):
    """
    تحميل ومعالجة البيانات للتدريب
    """
    try:
        # Load label mapping
        logging.info("Loading label mapping file...")
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            thamudic_letters = mapping_data['thamudic_letters']
            num_classes = len(thamudic_letters)
            
        # Process images
        logging.info("Processing images...")
        images = []
        labels = []
        
        # Create index to letter mapping
        index_to_letter = {letter['index']: letter for letter in thamudic_letters}
        
        # Get all letter directories
        letter_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('letter_')]
        
        for letter_dir in letter_dirs:
            try:
                # Extract label index from directory name (e.g., 'letter_1' -> 0)
                dir_num = int(letter_dir.split('_')[1])
                
                # Find corresponding letter data
                letter_data = None
                for letter in thamudic_letters:
                    if any(f"letter_{dir_num}" in img_path for img_path in letter['images']):
                        letter_data = letter
                        break
                
                if letter_data is None:
                    logging.warning(f"No mapping found for directory {letter_dir}")
                    continue
                
                label_idx = letter_data['index']
                dir_path = os.path.join(data_dir, letter_dir)
                
                for img_file in os.listdir(dir_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(dir_path, img_file)
                        try:
                            # Load and preprocess image
                            img = Image.open(img_path).convert('RGB')
                            img = img.resize((128, 128))
                            img_array = np.array(img) / 255.0
                            
                            # Create one-hot encoded label
                            label_one_hot = np.zeros(num_classes)
                            label_one_hot[label_idx] = 1
                            
                            images.append(img_array)
                            labels.append(label_one_hot)
                            logging.info(f"Processed image {img_file} for letter {letter_data['name']} (index: {label_idx})")
                            
                        except Exception as e:
                            logging.warning(f"Error processing image {img_path}: {str(e)}")
                            continue
                            
            except ValueError as e:
                logging.warning(f"Could not parse directory name: {letter_dir}, error: {str(e)}")
                continue
                
        if not images:
            raise ValueError("No valid images found in the data directory")
            
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Print dataset statistics
        logging.info("\nDataset Statistics:")
        logging.info(f"Total images: {len(images)}")
        logging.info(f"Number of classes: {num_classes}")
        
        # Calculate class distribution
        class_counts = {}
        for i in range(num_classes):
            count = np.sum(labels[:, i])
            if count > 0:
                letter_info = index_to_letter[i]
                class_counts[f"{letter_info['name']} ({i})"] = int(count)
        logging.info(f"Images per class: {class_counts}")
        
        # Split data
        indices = np.random.permutation(len(images))
        train_size = int(0.7 * len(images))
        val_size = int(0.15 * len(images))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        
        train_images = images[train_indices]
        train_labels = labels[train_indices]
        val_images = images[val_indices]
        val_labels = labels[val_indices]
        
        return (train_images, train_labels), (val_images, val_labels), num_classes
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

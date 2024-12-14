import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.core.cnn_architecture import ThamudicRecognitionModel

def convert_keras_to_pytorch(keras_model_path: str, output_path: str, num_classes: int):
    """Convert Keras model to PyTorch format"""
    # Load Keras model
    keras_model = tf.keras.models.load_model(keras_model_path)
    
    # Create PyTorch model
    pytorch_model = ThamudicRecognitionModel(num_classes=num_classes)
    pytorch_model.eval()
    
    # Convert weights
    keras_weights = keras_model.get_weights()
    
    # Create state dict for PyTorch model
    state_dict = {
        'model_state_dict': pytorch_model.state_dict(),
    }
    
    # Save PyTorch model
    torch.save(state_dict, output_path)
    print(f"Model converted and saved to {output_path}")

def main():
    # Paths
    keras_model_path = 'models/best_model.keras'
    output_path = 'output/models/best_model.pt'
    label_mapping_path = 'data/letters/letter_mapping.json'
    
    # Load label mapping to get number of classes
    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    num_classes = len(label_mapping['thamudic_letters'])
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert model
    convert_keras_to_pytorch(keras_model_path, output_path, num_classes)

if __name__ == '__main__':
    main()

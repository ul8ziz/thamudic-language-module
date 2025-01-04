import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json

from data_preprocessing import ThamudicPreprocessor
from model import ThamudicRecognitionModel

class ModelEvaluator:
    def __init__(self, model_path: str, config_path: str, char_mapping_path: str):
        # Load configurations
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        with open(char_mapping_path, 'r', encoding='utf-8') as f:
            self.char_mapping = json.load(f)
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ThamudicRecognitionModel(len(self.char_mapping))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = ThamudicPreprocessor(
            image_size=tuple(self.config['model']['image_size'])
        )
        
        # Reverse char mapping for evaluation
        self.idx_to_char = {v: k for k, v in self.char_mapping.items()}
    
    def evaluate_model(self, test_loader):
        """Evaluate model performance on test set."""
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return self.generate_metrics(all_labels, all_preds)
    
    def generate_metrics(self, true_labels, predicted_labels):
        """Generate evaluation metrics."""
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        
        # Generate classification report
        class_report = classification_report(
            true_labels, 
            predicted_labels,
            target_names=[self.idx_to_char[i] for i in range(len(self.char_mapping))],
            output_dict=True
        )
        
        # Plot confusion matrix
        plt.figure(figsize=(15, 15))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d',
            xticklabels=[self.idx_to_char[i] for i in range(len(self.char_mapping))],
            yticklabels=[self.idx_to_char[i] for i in range(len(self.char_mapping))]
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Save plot
        output_dir = Path('evaluation_results')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'confusion_matrix.png')
        
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'accuracy': class_report['accuracy']
        }
    
    def evaluate_single_image(self, image_path: str):
        """Evaluate model performance on a single image."""
        # Preprocess image
        image = self.preprocessor.preprocess_image(image_path)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = torch.max(output.data, 1)
            
        predicted_char = self.idx_to_char[predicted.item()]
        
        # Get confidence scores
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        confidence = probabilities[predicted].item()
        
        return {
            'predicted_char': predicted_char,
            'confidence': confidence,
            'probabilities': {
                self.idx_to_char[i]: prob.item()
                for i, prob in enumerate(probabilities)
            }
        }

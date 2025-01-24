import torch
from model import ThamudicRecognitionModel
from data_loader import get_data_loaders
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import logging
import tensorflow as tf

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('evaluation.log'),
            logging.StreamHandler()
        ]
    )

class ModelEvaluator:
    def __init__(self, model_path: str, label_mapping_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load label mapping
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)
        
        # Create reverse mapping
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        
        # Initialize model
        self.model = ThamudicRecognitionModel(num_classes=len(self.label_mapping))
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_model(self, test_loader):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        
        # Generate detailed reports
        self.generate_classification_report(all_labels, all_predictions)
        self.plot_confusion_matrix(all_labels, all_predictions)
        self.analyze_errors(all_labels, all_predictions)
        
        return accuracy
    
    def plot_confusion_matrix(self, true_labels, pred_labels, output_path: str = 'confusion_matrix.png'):
        """Enhanced confusion matrix visualization"""
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[self.idx_to_label[i] for i in range(len(self.label_mapping))],
                   yticklabels=[self.idx_to_label[i] for i in range(len(self.label_mapping))])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def generate_classification_report(self, true_labels, pred_labels, output_path: str = 'classification_report.txt'):
        """Detailed classification report with per-class metrics"""
        report = classification_report(true_labels, pred_labels,
                                    target_names=[self.idx_to_label[i] for i in range(len(self.label_mapping))],
                                    digits=4)
        print("\nClassification Report:")
        print(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
    
    def analyze_errors(self, true_labels, pred_labels):
        """Analyze common error patterns"""
        errors = []
        for true, pred in zip(true_labels, pred_labels):
            if true != pred:
                true_label = self.idx_to_label[true]
                pred_label = self.idx_to_label[pred]
                errors.append((true_label, pred_label))
        
        print("\nCommon Error Patterns:")
        error_counts = Counter(errors)
        for (true_label, pred_label), count in error_counts.most_common(10):
            print(f"True: {true_label}, Predicted: {pred_label}, Count: {count}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Thamudic Recognition Model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--label-mapping', type=str, required=True, help='Path to label mapping file')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, args.label_mapping)
    
    # Create test data loader
    test_loader, _ = get_data_loaders(args.data_dir, args.label_mapping, batch_size=32, train_split=0.0)
    
    # Evaluate model
    logger.info("Evaluating model...")
    accuracy = evaluator.evaluate_model(test_loader)
    logger.info(f"Overall Accuracy: {accuracy:.2f}%")
    
    logger.info(f"Evaluation results saved in {args.output_dir}")

if __name__ == '__main__':
    main()

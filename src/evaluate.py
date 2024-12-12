import torch
from model import ThamudicRecognitionModel
from data_loader import get_data_loaders
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import logging

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
        """
        Evaluate model performance on test set
        """
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        return accuracy, all_preds, all_labels
    
    def plot_confusion_matrix(self, true_labels, pred_labels, output_path: str):
        """
        Create and save confusion matrix visualization
        """
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(15, 15))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            xticklabels=[self.idx_to_label[i] for i in range(len(self.label_mapping))],
            yticklabels=[self.idx_to_label[i] for i in range(len(self.label_mapping))]
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def generate_classification_report(self, true_labels, pred_labels, output_path: str):
        """
        Generate detailed classification report
        """
        report = classification_report(
            true_labels,
            pred_labels,
            target_names=[self.idx_to_label[i] for i in range(len(self.label_mapping))],
            output_dict=True
        )
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report
    
    def analyze_errors(self, true_labels, pred_labels):
        """
        Analyze common error patterns
        """
        error_patterns = {}
        for true, pred in zip(true_labels, pred_labels):
            if true != pred:
                true_char = self.idx_to_label[true]
                pred_char = self.idx_to_label[pred]
                key = f"{true_char}->{pred_char}"
                error_patterns[key] = error_patterns.get(key, 0) + 1
        
        # Sort by frequency
        error_patterns = dict(sorted(error_patterns.items(), key=lambda x: x[1], reverse=True))
        return error_patterns

def main():
    parser = argparse.ArgumentParser(description='Evaluate Thamudic Recognition Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--label_mapping', type=str, required=True,
                        help='Path to label mapping file')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, args.label_mapping)
    
    # Get test data loader
    test_loader, _ = get_data_loaders(
        args.test_data,
        args.label_mapping,
        batch_size=32,
        train_split=0.0  # Use all data for testing
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    accuracy, all_preds, all_labels = evaluator.evaluate_model(test_loader)
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    
    # Generate confusion matrix
    logger.info("Generating confusion matrix...")
    evaluator.plot_confusion_matrix(
        all_labels,
        all_preds,
        str(output_path / 'confusion_matrix.png')
    )
    
    # Generate classification report
    logger.info("Generating classification report...")
    report = evaluator.generate_classification_report(
        all_labels,
        all_preds,
        str(output_path / 'classification_report.json')
    )
    
    # Analyze errors
    logger.info("Analyzing error patterns...")
    error_patterns = evaluator.analyze_errors(all_labels, all_preds)
    
    # Save error analysis
    with open(output_path / 'error_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(error_patterns, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation results saved in {args.output_dir}")

if __name__ == '__main__':
    main()

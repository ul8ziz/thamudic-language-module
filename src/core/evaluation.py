import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import logging

def evaluate_model(model, x_test, y_test, label_mapping):
    """
    Evaluate model performance and generate visualizations
    """
    try:
        # Get predictions
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save confusion matrix plot
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        plot_dir = os.path.join(base_dir, 'models', 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Generate classification report
        report = classification_report(
            y_test_classes,
            y_pred_classes,
            target_names=[label_mapping[str(i)] for i in range(len(label_mapping))],
            output_dict=True
        )
        
        # Log results
        logging.info("\nClassification Report:")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                logging.info(f"\n{label}:")
                for metric_name, value in metrics.items():
                    logging.info(f"{metric_name}: {value:.4f}")
        
        return {
            'accuracy': report['accuracy'],
            'confusion_matrix': cm,
            'classification_report': report
        }
        
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        raise

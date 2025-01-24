import torch
from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json
import logging
from datetime import datetime

class TrainingMonitor:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.output_dir / 'training.log')
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

    def update_metrics(self, epoch: int, train_loss: float, train_acc: float,
                      val_loss: Optional[float] = None, val_acc: Optional[float] = None):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        
        if val_loss is not None and val_acc is not None:
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
        
        # Log metrics
        log_msg = f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
        if val_loss is not None:
            log_msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        self.logger.info(log_msg)
        
        # Save current metrics
        self.save_metrics()
        
        # Plot learning curves
        if epoch % 5 == 0:  # Plot every 5 epochs
            self.plot_learning_curves()

    def save_metrics(self):
        metrics = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f)

    def plot_learning_curves(self):
        plt.figure(figsize=(12, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss')
        plt.title('Learning Curves - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        if self.val_accuracies:
            plt.plot(self.val_accuracies, label='Val Accuracy')
        plt.title('Learning Curves - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_curves.png')
        plt.close()

    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                            class_names: List[str]):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png')
        plt.close()

    def save_classification_report(self, y_true: List[int], y_pred: List[int],
                                 class_names: List[str]):
        report = classification_report(y_true, y_pred, target_names=class_names)
        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        self.logger.info(f"\nClassification Report:\n{report}")

    def save_model_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                            epoch: int, loss: float, accuracy: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy
        }
        torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Save best model if current accuracy is highest
        if accuracy == max(self.train_accuracies):
            torch.save(checkpoint, self.output_dir / 'best_model.pt')
            self.logger.info(f"Saved new best model with accuracy: {accuracy:.4f}")

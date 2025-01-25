import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import os

class TrainingVisualizer:
    """Advanced visualization tools for model training and evaluation"""
    
    def __init__(self, log_dir='runs'):
        self.writer = SummaryWriter(log_dir)
        self.metrics_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
    
    def update_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Update training metrics history"""
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['train_acc'].append(train_acc)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['val_acc'].append(val_acc)
        self.metrics_history['learning_rates'].append(lr)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        self.writer.add_scalar('Learning_Rate', lr, epoch)
    
    def plot_training_history(self, save_path=None):
        """Plot comprehensive training history"""
        plt.style.use('seaborn')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)
        ax1.plot(epochs, self.metrics_history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.metrics_history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracies
        ax2.plot(epochs, self.metrics_history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.metrics_history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        # Plot learning rate
        ax3.plot(epochs, self.metrics_history['learning_rates'], 'g-')
        ax3.set_title('Learning Rate over Time')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        
        # Add overall accuracy trend
        ax4.plot(epochs, np.array(self.metrics_history['train_acc']) - 
                np.array(self.metrics_history['val_acc']), 'r-', label='Gap')
        ax4.set_title('Training-Validation Accuracy Gap')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Gap')
        ax4.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, classes, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_image_statistics(self, dataset, save_dir):
        """Plot image statistics"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Brightness distribution
        plt.figure(figsize=(10, 6))
        brightness_values = []
        for img, _ in dataset:
            brightness = torch.mean(img)
            brightness_values.append(brightness.item())
        
        sns.histplot(brightness_values, bins=50)
        plt.title('Image Brightness Distribution')
        plt.xlabel('Brightness Value')
        plt.ylabel('Count')
        plt.savefig(os.path.join(save_dir, 'brightness_distribution.png'))
        plt.close()
        
        # Contrast distribution
        plt.figure(figsize=(10, 6))
        contrast_values = []
        for img, _ in dataset:
            contrast = torch.std(img)
            contrast_values.append(contrast.item())
        
        sns.histplot(contrast_values, bins=50)
        plt.title('Image Contrast Distribution')
        plt.xlabel('Contrast Value')
        plt.ylabel('Count')
        plt.savefig(os.path.join(save_dir, 'contrast_distribution.png'))
        plt.close()
        
        # Class distribution
        plt.figure(figsize=(12, 6))
        labels = [label for _, label in dataset]
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        sns.barplot(x=unique_labels, y=counts)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
        plt.close()
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()

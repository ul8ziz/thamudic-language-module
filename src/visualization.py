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
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/validation', val_acc, epoch)
        self.writer.add_scalar('Learning_rate', lr, epoch)
    
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
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        # Plot learning rate
        ax3.plot(epochs, self.metrics_history['learning_rates'], 'g-')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        
        # Plot validation metrics correlation
        ax4.scatter(self.metrics_history['val_loss'], self.metrics_history['val_acc'])
        ax4.set_title('Validation Loss vs Accuracy')
        ax4.set_xlabel('Validation Loss')
        ax4.set_ylabel('Validation Accuracy (%)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self, true_labels, predictions, class_names, save_path=None):
        """Plot confusion matrix with detailed analysis"""
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_class_distribution(self, labels, class_names, save_path=None):
        """Plot class distribution analysis"""
        plt.figure(figsize=(12, 6))
        
        # Count instances per class
        class_counts = pd.Series(labels).value_counts().sort_index()
        
        # Create bar plot
        sns.barplot(x=range(len(class_counts)), y=class_counts)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Number of Instances')
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_feature_maps(self, model, input_image, layer_name, save_path=None):
        """Visualize feature maps from a specific layer"""
        model.eval()
        
        # Register hook to get intermediate layer output
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # Attach hook
        for name, layer in model.named_modules():
            if name == layer_name:
                layer.register_forward_hook(get_activation(layer_name))
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_image.unsqueeze(0))
        
        # Get feature maps
        feature_maps = activation[layer_name].squeeze().cpu()
        
        # Plot feature maps
        num_features = min(16, feature_maps.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for idx in range(num_features):
            ax = axes[idx//4, idx%4]
            ax.imshow(feature_maps[idx], cmap='viridis')
            ax.axis('off')
        
        plt.suptitle(f'Feature Maps - {layer_name}')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()

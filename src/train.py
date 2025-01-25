"""
Thamudic Character Recognition Model Training Script
"""

import os
import logging
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import albumentations as A
from torch.utils.tensorboard import SummaryWriter
from data_loader import ThamudicDataset
from thamudic_model import ThamudicRecognitionModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time
from datetime import datetime

# Avoid dynamic optimization errors
torch._dynamo.config.suppress_errors = True

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class TrainingVisualizer:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        
    def update_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, learning_rate):
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, epoch)
        self.writer.add_scalar('Loss/validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/validation', val_acc, epoch)
        self.writer.add_scalar('Learning Rate', learning_rate, epoch)
    
    def close(self):
        self.writer.close()

def format_progress_bar(current, total, width=30):
    """Format a progress bar with the specified width"""
    filled = int(width * current / total)
    bar = '=' * filled + '>' + '.' * (width - filled - 1)
    return f"[{bar}]"

def format_time(seconds):
    """Format time in seconds to a human-readable string"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    return f"{hours:.0f}h"

def train_model(data_dir: str, mapping_file: str, model_save_path: str, 
                batch_size: int = 32, epochs: int = 50, learning_rate: float = 0.001):
    """
    Train the Thamudic character recognition model
    
    Args:
        data_dir: Path to data directory
        mapping_file: Path to mapping file
        model_save_path: Path to save the trained model
        batch_size: Batch size for training (default: 32)
        epochs: Number of training epochs (default: 50)
        learning_rate: Initial learning rate (default: 0.001)
    """
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create directories
        model_dir = Path(model_save_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {device}')
        
        # Load dataset
        train_dataset = ThamudicDataset(data_dir, mapping_file, train=True)
        val_dataset = ThamudicDataset(data_dir, mapping_file, train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=4)
        
        # Initialize model
        model = ThamudicRecognitionModel(num_classes=len(train_dataset.class_mapping))
        model = model.to(device)
        
        # Loss function and optimizer
        criterion = FocalLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # Training visualization
        log_dir = Path('runs') / datetime.now().strftime('%Y%m%d-%H%M%S')
        visualizer = TrainingVisualizer(log_dir)
        
        # Initialize best metrics
        best_val_acc = 0.0
        best_epoch = 0
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Training phase
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
                for batch_idx, (inputs, targets) in enumerate(pbar):
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += targets.size(0)
                    train_correct += predicted.eq(targets).sum().item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{train_loss/(batch_idx+1):.3f}',
                        'acc': f'{100.*train_correct/train_total:.2f}%'
                    })
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # Calculate metrics
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Log metrics
            visualizer.update_metrics(epoch, train_loss, train_acc, 
                                    val_loss, val_acc, current_lr)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), model_save_path)
                logging.info(f'Saved best model with validation accuracy: {val_acc:.2f}%')
            
            # Print epoch summary
            logging.info(f'Epoch {epoch+1}/{epochs}:')
            logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            logging.info(f'Learning Rate: {current_lr:.6f}')
        
        visualizer.close()
        logging.info(f'Training completed. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}')
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

def organize_dataset(data_dir: str, train_ratio: float = 0.8):
    """
    Organize dataset into train and validation sets
    
    Args:
        data_dir: Root directory containing character folders
        train_ratio: Ratio of data to use for training (default: 0.8)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all character directories
    char_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name not in ['train', 'val']]
    
    for char_dir in char_dirs:
        # Get all image files
        image_files = list(char_dir.glob('*.png'))
        
        # Split into train and validation sets
        train_files, val_files = train_test_split(
            image_files, train_size=train_ratio, random_state=42
        )
        
        # Create character directories in train and val
        (train_dir / char_dir.name).mkdir(exist_ok=True)
        (val_dir / char_dir.name).mkdir(exist_ok=True)
        
        # Move files
        for file in train_files:
            file.rename(train_dir / char_dir.name / file.name)
        
        for file in val_files:
            file.rename(val_dir / char_dir.name / file.name)
        
        logging.info(f'Processed {char_dir.name}: {len(train_files)} train, {len(val_files)} val')

def create_data_transforms():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform

def main():
    """Main entry point for training"""
    try:
        # Configuration
        project_dir = Path(__file__).parent.parent
        data_dir = project_dir / 'data' / 'letters' / 'improved_letters'
        mapping_file = project_dir / 'data' / 'mapping.json'
        model_dir = project_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = model_dir / 'best_model.pth'
        
        # Create transforms
        train_transform, val_transform = create_data_transforms()
        
        # Organize dataset if needed
        if not (data_dir/'train').exists():
            logging.info('Organizing dataset...')
            organize_dataset(data_dir)
        
        # Start training
        logging.info('Starting training...')
        train_model(
            data_dir=str(data_dir),
            mapping_file=str(mapping_file),
            model_save_path=str(model_save_path),
            batch_size=32,
            epochs=50,  # تغيير عدد العقد إلى 50
            learning_rate=0.001
        )
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()
import os
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from model import ThamudicRecognitionModel, ThamudicRecognitionTrainer
from data_loader import get_data_loaders, create_label_mapping
from training_utils import TrainingMonitor

def setup_logging(output_dir: str) -> None:
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train_model(model, train_loader, criterion, optimizer, num_epochs, device, val_loader=None):
    monitor = TrainingMonitor('output')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        # Validation step
        val_loss = None
        val_acc = None
        if val_loader:
            val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        
        # Update training monitor
        monitor.update_metrics(epoch, epoch_loss, epoch_acc, val_loss, val_acc)
        
        # Save checkpoint
        monitor.save_model_checkpoint(model, optimizer, epoch, epoch_loss, epoch_acc)

def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def main():
    # Configuration
    data_dir = 'data/train_dataset'  # Updated data directory path
    output_dir = 'output'
    label_mapping_file = os.path.join(output_dir, 'label_mapping.json')
    model_save_dir = os.path.join(output_dir, 'models')
    batch_size = 32
    num_epochs = 100
    num_workers = 4
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    
    # Create or load label mapping
    if not os.path.exists(label_mapping_file):
        logging.info("Creating label mapping...")
        label_mapping = create_label_mapping(data_dir, label_mapping_file)
    else:
        logging.info("Loading existing label mapping...")
        with open(label_mapping_file, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
    
    num_classes = len(label_mapping)
    logging.info(f"Number of classes: {num_classes}")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        data_dir,
        label_mapping_file,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create model
    model = ThamudicRecognitionModel(num_classes=num_classes)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    train_model(model, train_loader, criterion, optimizer, num_epochs, device, val_loader)

if __name__ == '__main__':
    main()

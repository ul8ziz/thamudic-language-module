"""
Training script for Thamudic character recognition model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from pathlib import Path
import logging
from tqdm import tqdm
from models import ThamudicRecognitionModel
from data_processing import ThamudicDataset
from torch.utils.tensorboard import SummaryWriter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, save_dir, writer):
    """Train the model with validation and early stopping"""
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
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
            
            progress_bar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
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
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        
        # Log metrics
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'Train Loss: {epoch_train_loss:.3f} | Train Acc: {epoch_train_acc:.2f}%')
        logging.info(f'Val Loss: {epoch_val_loss:.3f} | Val Acc: {epoch_val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(epoch_val_loss)
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            save_path = os.path.join(save_dir, 'best_model.pt')
            model.save_checkpoint(
                save_path,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=epoch_train_loss,
                val_loss=epoch_val_loss,
                train_acc=epoch_train_acc,
                val_acc=epoch_val_acc
            )
            logging.info(f'Saved best model to {save_path}')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logging.info('Early stopping triggered')
            break

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Setup directories
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'letters'
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    save_dir = base_dir / 'models' / 'checkpoints'
    log_dir = base_dir / 'runs'
    
    # Create directories if they don't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Load datasets
        train_dataset = ThamudicDataset(train_dir, transform=train_transform)
        val_dataset = ThamudicDataset(val_dir, transform=val_transform)
        
        logging.info(f'Number of training samples: {len(train_dataset)}')
        logging.info(f'Number of validation samples: {len(val_dataset)}')
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model
        model = ThamudicRecognitionModel(num_classes=len(train_dataset.classes))
        model = model.to(device)
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weights.to(device))
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )
        
        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir)
        
        # Train the model
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=50,
            device=device,
            save_dir=save_dir,
            writer=writer
        )
        
        writer.close()
        logging.info('Training completed successfully')
        
    except Exception as e:
        logging.error(f'Error during training: {str(e)}')
        raise

if __name__ == '__main__':
    main()

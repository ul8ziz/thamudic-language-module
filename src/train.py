import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
import json
from albumentations import Compose, RandomRotate90, Flip, ShiftScaleRotate, GaussNoise, GaussianBlur, MotionBlur, OpticalDistortion, GridDistortion, ElasticTransform, CLAHE, Sharpen, Emboss, RandomBrightnessContrast
import cv2
import os

from data_preprocessing import ThamudicPreprocessor
from model import ThamudicRecognitionModel

class ThamudicDataset(Dataset):
    def __init__(self, data_dir: str, char_mapping: Dict[int, str], transform=None):
        self.data_dir = Path(data_dir)
        self.char_mapping = char_mapping
        self.transform = transform
        self.image_size = (224, 224)  # Fixed size for all images
        
        # Load all image paths and labels
        self.samples = []
        self.labels = []
        
        print("Loading dataset...")
        print(f"Data directory: {data_dir}")
        print(f"Number of classes: {len(char_mapping)}")
        
        # Create reverse mapping from letter number to Arabic character
        self.letter_to_index = {}
        for idx, char in self.char_mapping.items():
            letter_num = f"letter_{idx + 1}"  # Convert 0-based index to 1-based folder names
            self.letter_to_index[letter_num] = idx
        
        # Scan directory for images
        for letter_dir in self.data_dir.glob("letter_*"):
            if letter_dir.is_dir():
                letter_num = letter_dir.name
                if letter_num in self.letter_to_index:
                    label = self.letter_to_index[letter_num]
                    for img_path in letter_dir.glob("*.png"):
                        self.samples.append(img_path)
                        self.labels.append(label)
        
        print(f"Found {len(self.samples)} images across {len(set(self.labels))} classes")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        img_path = str(self.samples[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fixed size
        image = cv2.resize(image, self.image_size)
        
        # Apply transform if provided
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Convert to tensor and normalize
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # Change from HWC to CHW format
        
        return image, self.labels[idx]

def load_char_mapping(mapping_file: str) -> Dict[int, str]:
    """Load and process character mapping from JSON file."""
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create mapping from index to Arabic character
    char_mapping = {}
    for letter in data['thamudic_letters']:
        char_mapping[letter['index']] = letter['name']
    
    print(f"Loaded {len(char_mapping)} characters")
    return char_mapping

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, path: str):
    """Save a checkpoint of the model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def train_model(
    data_dir: str,
    char_mapping: Dict[int, str],
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = None
) -> nn.Module:
    """Train the Thamudic recognition model."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
        else:
            device = torch.device('cpu')
            print("GPU not available, using CPU instead")
    
    print(f"Using device: {device}")
    
    train_transform = Compose([
        RandomRotate90(p=0.5),
        Flip(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        Compose([
            GaussNoise(p=1),
            GaussianBlur(p=1),
            MotionBlur(p=1),
        ], p=0.3),
        Compose([
            OpticalDistortion(p=1),
            GridDistortion(p=1),
            ElasticTransform(p=1),
        ], p=0.3),
        Compose([
            CLAHE(clip_limit=2),
            Sharpen(),
            Emboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
    ])
    
    print("Creating datasets...")
    full_dataset = ThamudicDataset(data_dir, char_mapping, transform=train_transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    num_workers = 4 if device.type == 'cuda' else 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    model = ThamudicRecognitionModel(len(char_mapping))
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Step [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
                if device.type == 'cuda':
                    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.0f}MB / "
                          f"{torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        checkpoint_path = f"models/checkpoints/checkpoint_epoch_{epoch}.pth"
        save_checkpoint(model, optimizer, epoch, checkpoint_path)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
            torch.save(model.state_dict(), 'models/thamudic_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return model

if __name__ == '__main__':
    char_mapping = load_char_mapping('data/letters/letter_mapping.json')
    
    model = train_model('data/letters/thamudic_letters', char_mapping)
    
    Path('models').mkdir(exist_ok=True)
    
    torch.save(model.state_dict(), 'models/thamudic_model.pth')
    print("Model saved successfully")

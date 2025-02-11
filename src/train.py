"""
تدريب نموذج التعرف على الحروف الثمودية
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import albumentations as A
from torch.utils.tensorboard import SummaryWriter

from .models import ThamudicRecognitionModel
from .data_processing import ThamudicDataset
from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    RUNS_DIR,
    MODEL_CONFIG,
    TRAINING_CONFIG
)

# إعداد التسجيل
logging.basicConfig(
    filename=os.path.join(PROJECT_ROOT, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ThamudicTrainer:
    def __init__(self, model_config=None, training_config=None):
        """تهيئة المدرب"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config or MODEL_CONFIG
        self.training_config = training_config or TRAINING_CONFIG
        
        # إنشاء النموذج
        self.model = ThamudicRecognitionModel(
            num_classes=28,  # عدد الحروف الثمودية
            image_size=self.model_config['image_size']
        ).to(self.device)
        
        # إعداد معايير الخسارة والمحسن
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.model_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
        
        # إعداد جدولة معدل التعلم
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.model_config['num_epochs'],
            eta_min=self.training_config['min_lr']
        )
        
        # إعداد Tensorboard
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(os.path.join(RUNS_DIR, 'tensorboard', current_time))
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train(self, train_loader, val_loader, num_epochs=None):
        """تدريب النموذج"""
        num_epochs = num_epochs or self.model_config['num_epochs']
        
        for epoch in range(num_epochs):
            # التدريب
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # التحقق
            val_loss, val_acc = self.validate(val_loader)
            
            # تحديث جدولة معدل التعلم
            self.scheduler.step()
            
            # تسجيل المقاييس
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            logging.info(f'Epoch {epoch+1}/{num_epochs}:')
            logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # حفظ أفضل نموذج
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, val_acc)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # التوقف المبكر
            if self.patience_counter >= self.model_config['early_stopping_patience']:
                logging.info('Early stopping triggered')
                break
    
    def validate(self, val_loader):
        """التحقق من النموذج"""
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch, val_loss, val_acc):
        """حفظ نقطة التفتيش"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        
        checkpoint_path = os.path.join(
            MODELS_DIR,
            'checkpoints',
            f'model_checkpoint_epoch_{epoch}_loss_{val_loss:.4f}_acc_{val_acc:.2f}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        logging.info(f'Saved checkpoint to {checkpoint_path}')

def main():
    """النقطة الرئيسية لتشغيل التدريب"""
    # تحميل البيانات
    transform = transforms.Compose([
        transforms.Resize((MODEL_CONFIG['image_size'], MODEL_CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ThamudicDataset(
        data_dir=os.path.join(DATA_DIR, 'letters'),
        transform=transform
    )
    
    # تقسيم البيانات
    train_size = int((1 - MODEL_CONFIG['validation_split']) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # إنشاء وتدريب النموذج
    trainer = ThamudicTrainer()
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()

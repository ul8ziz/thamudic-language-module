"""
تدريب نموذج التعرف على الحروف الثمودية
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

# تجنب أخطاء التحسين الديناميكي
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

def train_model(data_dir: str, mapping_file: str, model_save_path: str, batch_size: int = 32, epochs: int = 50, learning_rate: float = 0.001):
    """
    تدريب النموذج على مجموعة البيانات
    
    المعاملات:
        data_dir: مسار مجلد البيانات
        mapping_file: مسار ملف التعيين
        model_save_path: مسار حفظ النموذج
        batch_size: حجم الدفعة (الافتراضي: 32)
        epochs: عدد الدورات (الافتراضي: 50)
        learning_rate: معدل التعلم (الافتراضي: 0.001)
    """
    try:
        # إعداد الجهاز
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"استخدام الجهاز: {device}")
        
        # تحويلات الصور للتدريب
        train_transform = A.Compose([
            A.Resize(224, 224),
            A.OneOf([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=15, p=0.5),
            ], p=0.6),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
                A.GridDistortion(distort_limit=0.1, p=0.1),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.3),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # تحويلات الصور للتحقق
        val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # إعداد مجموعات البيانات
        train_dataset = ThamudicDataset(data_dir, mapping_file, transform=train_transform, train=True)
        val_dataset = ThamudicDataset(data_dir, mapping_file, transform=val_transform, train=False)
        
        # إعداد محملات البيانات
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # إنشاء النموذج
        num_classes = len(train_dataset.class_mapping)
        model = ThamudicRecognitionModel(num_classes=num_classes)
        model = model.to(device)
        
        # تحسين الذاكرة
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # إعداد معايير التدريب
        criterion = FocalLoss(gamma=2.0)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
        scaler = GradScaler()
        
        # إعداد TensorBoard
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = os.path.join('runs', current_time)
        writer = SummaryWriter(log_dir)
        
        # متغيرات التتبع
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0
        
        # حلقة التدريب
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # حلقة التدريب
            train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_idx, (images, labels) in enumerate(train_loop):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # تحديث شريط التقدم
                train_loop.set_postfix({
                    'loss': train_loss / (batch_idx + 1),
                    'acc': 100. * train_correct / train_total
                })
                
                # تحرير الذاكرة
                del images, labels, outputs, loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # تقييم النموذج
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for batch_idx, (images, labels) in enumerate(val_loop):
                    images, labels = images.to(device), labels.to(device)
                    
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    # تحديث شريط التقدم
                    val_loop.set_postfix({
                        'loss': val_loss / (batch_idx + 1),
                        'acc': 100. * val_correct / val_total
                    })
                    
                    # تحرير الذاكرة
                    del images, labels, outputs, loss
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # حساب المقاييس
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100. * train_correct / train_total
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = 100. * val_correct / val_total
            
            # تحديث جدول التعلم
            scheduler.step()
            
            # تسجيل المقاييس
            writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            writer.add_scalar('Loss/val', epoch_val_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
            writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
            
            # طباعة النتائج
            logging.info(f'Epoch {epoch+1}/{epochs}:')
            logging.info(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
            logging.info(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
            
            # حفظ أفضل نموذج
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model.state_dict(), model_save_path)
                patience_counter = 0
                logging.info(f'تم حفظ أفضل نموذج مع دقة تحقق: {best_val_acc:.2f}%')
            else:
                patience_counter += 1
            
            # التوقف المبكر
            if patience_counter >= patience:
                logging.info(f'التوقف المبكر بعد {patience} دورات بدون تحسن')
                break
            
            # تحرير الذاكرة في نهاية كل دورة
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        writer.close()
        logging.info("اكتمل التدريب!")
        
    except Exception as e:
        logging.error(f"خطأ في التدريب: {str(e)}")
        raise

def organize_dataset(data_dir: str, train_ratio: float = 0.8):
    """
    تنظيم مجموعة البيانات إلى مجموعتي تدريب وتحقق
    
    المعاملات:
        data_dir: المجلد الرئيسي الذي يحتوي على مجلدات الحروف
        train_ratio: نسبة البيانات المستخدمة للتدريب (الافتراضي: 0.8)
    """
    try:
        # إنشاء مجلدات التدريب والتحقق
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # مسح مجلدات الحروف
        for letter_dir in os.listdir(data_dir):
            if not letter_dir.startswith('letter_') or letter_dir in ['train', 'val']:
                continue
                
            letter_path = os.path.join(data_dir, letter_dir)
            if not os.path.isdir(letter_path):
                continue
                
            # إنشاء مجلدات الحروف في مجلدات التدريب والتحقق
            train_letter_dir = os.path.join(train_dir, letter_dir)
            val_letter_dir = os.path.join(val_dir, letter_dir)
            os.makedirs(train_letter_dir, exist_ok=True)
            os.makedirs(val_letter_dir, exist_ok=True)
            
            # تقسيم الصور
            images = [f for f in os.listdir(letter_path) if f.endswith('.png')]
            train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)
            
            # نسخ الصور بدلاً من نقلها
            import shutil
            for img in train_images:
                src = os.path.join(letter_path, img)
                dst = os.path.join(train_letter_dir, img)
                shutil.copy2(src, dst)
                
            for img in val_images:
                src = os.path.join(letter_path, img)
                dst = os.path.join(val_letter_dir, img)
                shutil.copy2(src, dst)
                
        logging.info(f"تم تنظيم مجموعة البيانات في {train_dir} و {val_dir}")
        
    except Exception as e:
        logging.error(f"خطأ في تنظيم مجموعة البيانات: {str(e)}")
        raise

def main():
    try:
        # إعداد التسجيل
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # تحديد المسارات
        project_dir = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(project_dir, 'data', 'letters', 'processed_letters')
        mapping_file = os.path.join(project_dir, 'data', 'mapping.json')
        model_dir = os.path.join(project_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # تنظيم مجموعة البيانات
        logging.info("تنظيم مجموعة البيانات...")
        organize_dataset(data_dir)
        
        # تحرير الذاكرة قبل التدريب
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # تدريب النموذج
        logging.info("تدريب النموذج...")
        model_save_path = os.path.join(model_dir, 'best_model.pth')
        train_model(data_dir, mapping_file, model_save_path, batch_size=8, epochs=100, learning_rate=0.0001)
        
    except Exception as e:
        logging.error(f"خطأ أثناء التدريب: {str(e)}")
        raise

if __name__ == '__main__':
    main()
"""
تدريب نموذج التعرف على الحروف الثمودية
"""

import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

from data_loader import ThamudicDataset
from thamudic_model import ThamudicRecognitionModel

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_transforms():
    """إنشاء تحويلات الصور"""
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

def create_data_loaders(data_dir: str, mapping: dict, batch_size: int = 32):
    """إنشاء محملات البيانات"""
    try:
        train_transform, val_transform = create_transforms()
        
        # إنشاء مجموعة البيانات
        dataset = ThamudicDataset(
            data_dir=data_dir,
            mapping=mapping,
            transform=train_transform,
            train=True
        )
        
        # تقسيم البيانات
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # إنشاء محملات البيانات
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    except Exception as e:
        logging.error(f"Error creating data loaders: {str(e)}")
        raise

def train_model(
    data_dir: str,
    mapping_path: str,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = None
):
    """تدريب النموذج"""
    try:
        # تحديد الجهاز
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"استخدام الجهاز: {device}")
        
        # تحميل التعيين
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        # إنشاء محملات البيانات
        train_loader, val_loader = create_data_loaders(data_dir, mapping, batch_size)
        logging.info(f"حجم مجموعة التدريب: {len(train_loader.dataset)}")
        logging.info(f"حجم مجموعة التحقق: {len(val_loader.dataset)}")
        
        # إنشاء النموذج
        num_classes = len(mapping['thamudic_letters'])
        model = ThamudicRecognitionModel(num_classes=num_classes)
        model = model.to(device)
        
        # إعداد دالة الخسارة والمحسن
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # التدريب
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            # وضع التدريب
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                if batch_idx % 5 == 0:  # عرض التقدم كل 5 دفعات
                    batch_acc = 100. * train_correct / train_total
                    batch_loss = train_loss / (batch_idx + 1)
                    logging.info(
                        f'الحقبة: {epoch}/{num_epochs} | '
                        f'الدفعة: {batch_idx}/{len(train_loader)} | '
                        f'الخسارة: {batch_loss:.4f} | '
                        f'الدقة: {batch_acc:.2f}%'
                    )
            
            # وضع التقييم
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            # حساب متوسط الخسارة والدقة
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            logging.info(
                f'نتائج الحقبة {epoch}/{num_epochs}:\n'
                f'خسارة التدريب: {train_loss:.4f} | '
                f'دقة التدريب: {train_acc:.2f}% | '
                f'خسارة التحقق: {val_loss:.4f} | '
                f'دقة التحقق: {val_acc:.2f}%'
            )
            
            # تحديث جدولة معدل التعلم
            scheduler.step(val_loss)
            
            # حفظ أفضل نموذج
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                models_dir = Path(__file__).parent.parent / 'models'
                models_dir.mkdir(exist_ok=True)
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
                
                save_path = models_dir / 'thamudic_model.pth'
                torch.save(checkpoint, save_path)
                logging.info(f'تم حفظ أفضل نموذج في {save_path}')
    
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    # المسارات
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'letters'  # المجلد الرئيسي للبيانات
    mapping_path = base_dir / 'data' / 'mapping.json'
    
    # التأكد من وجود المسارات
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    
    # تدريب النموذج
    train_model(
        data_dir=str(data_dir),
        mapping_path=str(mapping_path),
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
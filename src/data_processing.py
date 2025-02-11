"""
معالجة البيانات للتعرف على الحروف الثمودية
"""

import os
import json
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from .config import DATA_DIR, MODEL_CONFIG

class ThamudicDataset(Dataset):
    """مجموعة بيانات الحروف الثمودية"""
    
    def __init__(self, data_dir: str, transform=None):
        """
        تهيئة مجموعة البيانات
        
        Args:
            data_dir: مسار مجلد البيانات
            transform: التحويلات المطبقة على الصور
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.mapping_file = Path(DATA_DIR) / 'mapping.json'
        
        # تحميل تعيين الحروف
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            self.mapping = json.load(f)
        
        # إنشاء قائمة بالصور وتصنيفاتها
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        """تحميل مسارات الصور وتصنيفاتها"""
        for letter_info in self.mapping['thamudic_letters']:
            letter_dir = self.data_dir / f"letter_{letter_info['index']}"
            if not letter_dir.exists():
                continue
            
            for img_path in letter_dir.glob('*.png'):
                self.samples.append({
                    'path': img_path,
                    'label': letter_info['index']
                })
            
            # التحقق من ملفات JPG أيضاً
            for img_path in letter_dir.glob('*.jpg'):
                self.samples.append({
                    'path': img_path,
                    'label': letter_info['index']
                })
    
    def __len__(self) -> int:
        """عدد العينات في مجموعة البيانات"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        الحصول على عينة من مجموعة البيانات
        
        Args:
            idx: مؤشر العينة
        
        Returns:
            tuple: (صورة محولة، تصنيف)
        """
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label']
    
    def get_class_weights(self) -> torch.Tensor:
        """
        حساب أوزان الفئات للتعامل مع عدم التوازن في البيانات
        
        Returns:
            torch.Tensor: أوزان الفئات
        """
        # حساب عدد العينات لكل فئة
        class_counts = np.zeros(len(self.mapping['thamudic_letters']))
        for sample in self.samples:
            class_counts[sample['label']] += 1
        
        # تجنب القسمة على صفر
        class_counts = np.maximum(class_counts, 1)
        
        # حساب الأوزان
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(weights)
        
        return torch.FloatTensor(weights)
    
    def get_class_names(self) -> List[str]:
        """
        الحصول على أسماء الفئات
        
        Returns:
            List[str]: قائمة بأسماء الفئات
        """
        return [letter['name'] for letter in self.mapping['thamudic_letters']]

def create_data_loaders(
    data_dir: str,
    transform,
    batch_size: int = MODEL_CONFIG['batch_size'],
    val_split: float = MODEL_CONFIG['validation_split'],
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    إنشاء data loaders للتدريب والتحقق
    
    Args:
        data_dir: مسار مجلد البيانات
        transform: التحويلات المطبقة على الصور
        batch_size: حجم الدفعة
        val_split: نسبة بيانات التحقق
        num_workers: عدد العمال لتحميل البيانات
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    dataset = ThamudicDataset(data_dir, transform=transform)
    
    # تقسيم البيانات
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # إنشاء data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader

"""
تحميل ومعالجة بيانات الحروف الثمودية
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from typing import Tuple, List, Dict
import logging
from pathlib import Path

class ThamudicDataset(Dataset):
    """مجموعة بيانات مخصصة للتعرف على الحروف الثمودية"""
    
    def __init__(self, 
                 data_dir: str,
                 mapping_file: str,
                 transform: A.Compose = None,
                 train: bool = True):
        """
        تهيئة مجموعة البيانات
        
        المعاملات:
            data_dir: مجلد يحتوي على بيانات الصور
            mapping_file: ملف JSON يربط أسماء الفئات بالفهارس
            transform: تحويلات Albumentations
            train: ما إذا كانت هذه بيانات تدريب
        """
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        
        # تحميل التعيين
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.class_mapping = json.load(f)
            
        # تحديد مجلد البيانات
        self.data_folder = os.path.join(data_dir, 'train' if train else 'val')
        
        # جمع المسارات والتسميات
        self.image_paths = []
        self.labels = []
        
        for class_folder in os.listdir(self.data_folder):
            if not class_folder.startswith('letter_'):
                continue
                
            class_path = os.path.join(self.data_folder, class_folder)
            if not os.path.isdir(class_path):
                continue
                
            class_idx = int(class_folder.split('_')[1]) - 1
            
            for img_name in os.listdir(class_path):
                if img_name.endswith('.png'):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(class_idx)
                    
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # تحميل الصورة
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # تطبيق التحويلات
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
            
        # تحويل الصورة إلى تنسور
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image, self.labels[idx]

def create_data_transforms() -> Tuple[A.Compose, A.Compose]:
    """إنشاء تحويلات التدريب والتحقق"""
    train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.Normalize(mean=[0.485], std=[0.229], p=1.0),
    ])
    
    val_transform = A.Compose([
        A.Normalize(mean=[0.485], std=[0.229], p=1.0),
    ])
    
    return train_transform, val_transform

def load_and_preprocess_data(data_dir: str,
                           mapping_file: str) -> Tuple[ThamudicDataset, ThamudicDataset]:
    """
    تحميل ومعالجة بيانات الحروف الثمودية
    
    المعاملات:
        data_dir: مجلد يحتوي على بيانات الصور
        mapping_file: ملف JSON يربط أسماء الفئات بالفهارس
    
    الإرجاع:
        train_dataset: مجموعة بيانات التدريب
        val_dataset: مجموعة بيانات التحقق
    """
    try:
        logging.info("إنشاء تحويلات البيانات...")
        train_transform, val_transform = create_data_transforms()
        
        logging.info("تحميل مجموعة بيانات التدريب...")
        train_dataset = ThamudicDataset(
            data_dir=data_dir,
            mapping_file=mapping_file,
            transform=train_transform,
            train=True
        )
        
        logging.info("تحميل مجموعة بيانات التحقق...")
        val_dataset = ThamudicDataset(
            data_dir=data_dir,
            mapping_file=mapping_file,
            transform=val_transform,
            train=False
        )
        
        return train_dataset, val_dataset
        
    except Exception as e:
        logging.error(f"خطأ في تحميل البيانات: {str(e)}")
        raise

def analyze_dataset(dataset: ThamudicDataset) -> Dict:
    """
    تحليل مجموعة البيانات وإرجاع الإحصائيات
    
    المعاملات:
        dataset: مثيل ThamudicDataset
    
    الإرجاع:
        قاموس يحتوي على إحصائيات مجموعة البيانات
    """
    stats = {
        'num_samples': len(dataset),
        'num_classes': len(dataset.class_mapping),
        'class_distribution': {},
        'image_stats': {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0,
            'std': 0
        }
    }
    
    # تحليل توزيع الفئات
    unique, counts = np.unique(dataset.labels, return_counts=True)
    for u, c in zip(unique, counts):
        stats['class_distribution'][u] = int(c)
    
    # تحليل إحصائيات الصور
    pixel_values = []
    for i in range(min(100, len(dataset))):  # أخذ عينة من 100 صورة
        image, _ = dataset[i]
        pixel_values.append(image.numpy())
    
    pixel_values = np.concatenate(pixel_values)
    stats['image_stats']['min'] = float(np.min(pixel_values))
    stats['image_stats']['max'] = float(np.max(pixel_values))
    stats['image_stats']['mean'] = float(np.mean(pixel_values))
    stats['image_stats']['std'] = float(np.std(pixel_values))
    
    return stats

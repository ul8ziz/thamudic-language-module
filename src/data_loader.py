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
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.train = train
        
        # تحميل التعيين
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.class_mapping = json.load(f)
        
        # تحديد مجلد البيانات
        split = 'train' if train else 'val'
        self.data_folder = self.data_dir / split
        
        if not self.data_folder.exists():
            raise ValueError(f"Data folder {self.data_folder} does not exist")
        
        # جمع المسارات والتسميات
        self.image_paths = []
        self.labels = []
        
        for class_folder in os.listdir(self.data_folder):
            class_path = self.data_folder / class_folder
            if not class_path.is_dir():
                continue
                
            class_idx = self.class_mapping.get(class_folder)
            if class_idx is None:
                logging.warning(f"Skipping unknown class folder: {class_folder}")
                continue
            
            for img_file in class_path.glob('*.png'):
                self.image_paths.append(img_file)
                self.labels.append(class_idx)
        
        if not self.image_paths:
            raise ValueError(f"No images found in {self.data_folder}")
        
        logging.info(f"Loaded {len(self.image_paths)} images for {split} set")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # تحميل الصورة
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # تطبيق التحويلات
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # تحويل الصورة إلى تنسور
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # HWC to CHW
        image = image / 255.0  # Normalize to [0, 1]
        
        return image, label

def create_data_transforms() -> Tuple[A.Compose, A.Compose]:
    """إنشاء تحويلات التدريب والتحقق"""
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
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform

def load_and_preprocess_data(data_dir: str, mapping_file: str) -> Tuple[Dataset, Dataset]:
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
        'total_images': len(dataset),
        'num_classes': len(dataset.class_mapping),
        'class_distribution': {},
        'image_sizes': [],
        'aspect_ratios': []
    }
    
    # تحليل توزيع الفئات
    for label in dataset.labels:
        class_name = list(dataset.class_mapping.keys())[list(dataset.class_mapping.values()).index(label)]
        stats['class_distribution'][class_name] = stats['class_distribution'].get(class_name, 0) + 1
    
    # تحليل إحصائيات الصور
    for img_path in dataset.image_paths:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        stats['image_sizes'].append((w, h))
        stats['aspect_ratios'].append(w/h)
    
    # حساب الإحصائيات الملخصة
    stats['avg_width'] = np.mean([s[0] for s in stats['image_sizes']])
    stats['avg_height'] = np.mean([s[1] for s in stats['image_sizes']])
    stats['avg_aspect_ratio'] = np.mean(stats['aspect_ratios'])
    
    return stats

"""
تحميل ومعالجة بيانات الحروف الثمودية
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path
from collections import Counter

class ThamudicDataset(Dataset):
    """مجموعة بيانات الحروف الثمودية"""
    
    def __init__(self, 
                 data_dir: str,
                 mapping: Dict,
                 transform: Optional[A.Compose] = None,
                 train: bool = True):
        """
        تهيئة مجموعة البيانات
        
        المعاملات:
            data_dir: مجلد يحتوي على بيانات الصور
            mapping: تعيين الفئات
            transform: تحويلات Albumentations
            train: ما إذا كانت هذه بيانات تدريب
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.train = train
        self.mapping = mapping
        
        # تحسين معالجة الصور
        self.preprocessor = ThamudicImagePreprocessor()
        
        # جمع مسارات الصور والتسميات
        self.image_paths = []
        self.labels = []
        self._load_dataset(mapping)
        
        # حساب أوزان الفئات للتعامل مع عدم التوازن
        self.class_weights = self._compute_class_weights()
        
        logging.info(f"تم تحميل {len(self.image_paths)} صورة")
        logging.info(f"عدد الفئات: {len(set(self.labels))}")
        
    def _load_dataset(self, mapping: Dict):
        """تحميل مجموعة البيانات مع معالجة متقدمة"""
        for letter_info in mapping['thamudic_letters']:
            letter_dir = self.data_dir / f"letter_{letter_info['index'] + 1}"
            if not letter_dir.exists():
                logging.warning(f"المجلد غير موجود: {letter_dir}")
                continue
            
            for img_path in letter_dir.glob('*.png'):
                if img_path.exists():
                    self.image_paths.append(str(img_path))
                    self.labels.append(letter_info['index'])
                else:
                    logging.warning(f"الصورة غير موجودة: {img_path}")
        
        if not self.image_paths:
            raise ValueError("لم يتم العثور على أي صور!")

    def _compute_class_weights(self) -> Dict[int, float]:
        """حساب أوزان الفئات للتعامل مع عدم التوازن"""
        label_counts = Counter(self.labels)
        max_count = max(label_counts.values())
        return {label: max_count / count for label, count in label_counts.items()}
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        الحصول على عنصر من مجموعة البيانات
        
        المعاملات:
            idx: فهرس العنصر
            
        العوائد:
            صورة وتسميتها
        """
        # قراءة الصورة
        img_path = self.image_paths[idx]
        try:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"فشل في قراءة الصورة: {img_path}")
        except Exception as e:
            logging.error(f"خطأ في قراءة الصورة {img_path}: {str(e)}")
            # استخدام صورة فارغة في حالة الفشل
            image = np.zeros((224, 224), dtype=np.uint8)
        
        # معالجة الصورة
        image = self.preprocessor.preprocess(image)
        
        # تطبيق التحويلات
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']  # هذا سيكون تنسور PyTorch
        else:
            # إذا لم يكن هناك تحويل، نقوم بالتحويل يدوياً
            image = torch.from_numpy(image).float()
            image = image.unsqueeze(0)  # إضافة بُعد القناة
        
        # الحصول على التسمية
        label = self.labels[idx]
        
        return image, label

class ThamudicImagePreprocessor:
    """معالج متقدم للصور الثمودية"""
    def __init__(self):
        self.target_size = (224, 224)
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """معالجة متقدمة للصورة"""
        # تحويل إلى تدرج رمادي
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # تحسين التباين
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # إزالة الضوضاء
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # التحويل الثنائي التكيفي
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # تغيير الحجم مع الحفاظ على النسبة
        h, w = thresh.shape
        scale = min(
            self.target_size[0] / h,
            self.target_size[1] / w
        )
        new_h = int(h * scale)
        new_w = int(w * scale)
        resized = cv2.resize(thresh, (new_w, new_h))
        
        # إضافة الحشو
        pad_h = self.target_size[0] - new_h
        pad_w = self.target_size[1] - new_w
        padded = cv2.copyMakeBorder(
            resized,
            pad_h//2, pad_h - pad_h//2,
            pad_w//2, pad_w - pad_w//2,
            cv2.BORDER_CONSTANT, value=0
        )
        
        return padded

def create_data_loaders(data_dir: str, mapping_file: str, batch_size: int = 32):
    """
    إنشاء data loaders للتدريب والتحقق
    
    المعاملات:
        data_dir: مجلد البيانات
        mapping_file: ملف تعيين الفئات
        batch_size: حجم الدفعة
    """
    try:
        # تحميل التعيين
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        # تحويل المسارات النسبية إلى مسارات مطلقة
        base_dir = Path(data_dir).parent
        for letter in mapping['thamudic_letters']:
            letter['images'] = [
                str(base_dir / path)
                for path in letter['images']
            ]
        
        # إنشاء مجموعات البيانات
        dataset = ThamudicDataset(
            data_dir=data_dir,
            mapping=mapping,
            train=True
        )
        
        # تقسيم البيانات
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # إنشاء data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        logging.error(f'Error creating data loaders: {str(e)}')
        raise

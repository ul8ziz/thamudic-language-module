import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import logging
import sys
from pathlib import Path

# تعيين ترميز المخرجات
sys.stdout.reconfigure(encoding='utf-8')

# إعداد التسجيل
logger = logging.getLogger('data')
logger.setLevel(logging.INFO)

def load_character_mapping(base_dir):
    """
    تحميل تعيين الحروف من ملف JSON
    
    Args:
        base_dir: المجلد الأساسي الذي يحتوي على ملف التعيين
        
    Returns:
        dict: قاموس يحتوي على تعيين الحروف
    """
    mapping_file = os.path.join(base_dir, 'char_mapping.json')
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # استخراج التعيين المباشر من الحروف الثمودية إلى الأرقام
            mapping = {}
            for idx, (thamudic, _) in enumerate(data['thamudic_to_arabic'].items()):
                mapping[f'letter_{idx+1}'] = thamudic
        
        logging.info(f"تم تحميل {len(mapping)} حرف")
        logging.info(f"التعيين: {mapping}")
        return mapping
    except Exception as e:
        logging.error(f"خطأ في تحميل تعيين الحروف: {e}")
        return {}

class ThamudicDataset(Dataset):
    """
    مجموعة بيانات للحروف الثمودية
    """
    def __init__(self, data_dir, transform=None):
        """
        تهيئة مجموعة البيانات
        
        Args:
            data_dir (str): مسار مجلد البيانات
            transform (callable, optional): التحويلات المطبقة على الصور
        """
        self.data_dir = Path(data_dir)
        self.transform = transform if transform else self._default_transform()
        self.samples = []
        self.class_to_idx = {}
        
        # تحميل تعيين الحروف
        base_dir = os.path.dirname(os.path.dirname(data_dir))  # نرجع مجلدين للوصول إلى المجلد الأساسي
        self.char_mapping = load_character_mapping(base_dir)
        
        # تحميل الصور
        self._load_samples()
        
        if not self.samples:
            raise ValueError("لم يتم العثور على أي صور صالحة في مجلد البيانات")
    
    def _default_transform(self):
        """
        التحويلات الافتراضية للصور
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def _load_samples(self):
        """
        تحميل الصور وتصنيفاتها
        """
        # التأكد من وجود المجلد
        if not self.data_dir.exists():
            raise ValueError(f"مجلد البيانات غير موجود: {self.data_dir}")
        
        # تحميل الصور من المجلدات
        for class_dir in sorted(os.listdir(self.data_dir)):
            # تجاهل المجلدات الخاصة
            if class_dir in ['processed_letters', 'thamudic_letters']:
                continue
                
            class_path = self.data_dir / class_dir
            if not class_path.is_dir():
                continue
            
            # تحديد مؤشر الفئة
            if class_dir not in self.class_to_idx:
                self.class_to_idx[class_dir] = len(self.class_to_idx)
            
            # تحميل الصور
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = class_path / img_name
                    self.samples.append((str(img_path), self.class_to_idx[class_dir]))
        
        logging.info(f"تم تحميل {len(self.samples)} صورة من {len(self.class_to_idx)} فئة")
    
    def __len__(self):
        """
        عدد العناصر في مجموعة البيانات
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        الحصول على عنصر من مجموعة البيانات
        
        Args:
            idx (int): مؤشر العنصر
            
        Returns:
            tuple: (الصورة، التصنيف)
        """
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('L')  # تحويل إلى صورة رمادية
            if self.transform:
                image = self.transform(image)
            return image, label
            
        except Exception as e:
            logging.error(f"خطأ في تحميل الصورة {img_path}: {e}")
            return torch.zeros((1, 224, 224)), label  # إرجاع صورة فارغة في حالة الخطأ

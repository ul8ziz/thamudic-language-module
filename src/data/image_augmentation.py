import torch
from PIL import Image, ImageEnhance
from torchvision import transforms as T
from typing import Union
import numpy as np

class ThamudicImagePreprocessor:
    def __init__(self):
        """
        تهيئة معالج الصور
        """
        self.val_transforms = T.Compose([
            T.Resize((128, 128)),  # تغيير حجم الصورة
            T.ToTensor(),
            T.Normalize(mean=[0.485], std=[0.229])  # تطبيع القيم
        ])
        
        # إضافة تحويلات للتدريب لزيادة تنوع البيانات
        self.train_transforms = T.Compose([
            T.RandomRotation(10),  # دوران عشوائي
            T.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # إزاحة عشوائية
                scale=(0.9, 1.1),  # تغيير الحجم عشوائياً
                shear=5  # انحراف عشوائي
            ),
            T.RandomPerspective(distortion_scale=0.2),  # تغيير المنظور
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485], std=[0.229])
        ])

    def ensure_tensor(self, image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
        """
        التأكد من أن الصورة في شكل تنسور
        """
        if isinstance(image, Image.Image):
            # تحويل الصورة إلى تدرجات الرمادي إذا كانت ملونة
            if image.mode != 'L':
                image = image.convert('L')
            return self.val_transforms(image)
        return image

    def preprocess_image(self, image: Union[str, Image.Image], is_training: bool = False) -> torch.Tensor:
        """
        معالجة الصورة وتحويلها إلى تنسور
        """
        if isinstance(image, str):
            image = Image.open(image).convert('L')
        elif isinstance(image, Image.Image) and image.mode != 'L':
            image = image.convert('L')

        transforms = self.train_transforms if is_training else self.val_transforms
        return transforms(image)

    def enhance_image(self, image: Image.Image, contrast: float = 1.5, brightness: float = 1.2, sharpness: float = 1.3) -> Image.Image:
        """
        تحسين جودة الصورة
        """
        # تحويل الصورة إلى تدرجات الرمادي
        if image.mode != 'L':
            image = image.convert('L')
        
        # تحسين السطوع
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
        
        # تحسين التباين
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        
        # تحسين الحدة
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
        
        return image

    def batch_preprocess(self, images: list, is_training: bool = False) -> torch.Tensor:
        """
        معالجة مجموعة من الصور
        """
        processed = []
        for img in images:
            processed.append(self.preprocess_image(img, is_training))
        return torch.stack(processed)

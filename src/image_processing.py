"""
معالجة وتحسين صور الحروف الثمودية مع دعم التنبؤ
"""

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
import torchvision.transforms as T
from typing import Union, List, Dict
import json
from pathlib import Path

class ThamudicImagePreprocessor:
    def __init__(self, target_size: tuple = (128, 128)):
        """
        تهيئة معالج الصور للنصوص الثمودية
        
        Args:
            target_size: حجم الصورة المستهدف (العرض, الارتفاع)
        """
        self.target_size = target_size
        
        # تحويلات التحقق الأساسية
        self.val_transforms = T.Compose([
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485], std=[0.229])
        ])
        
        # تحويلات التدريب المتقدمة لزيادة تنوع البيانات
        self.train_transforms = T.Compose([
            T.RandomRotation(10),
            T.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            T.RandomPerspective(distortion_scale=0.2),
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485], std=[0.229])
        ])

    def enhance_image(self, image: Image.Image, 
                     contrast: float = 1.5, 
                     brightness: float = 1.2, 
                     sharpness: float = 1.3) -> Image.Image:
        """
        تحسين جودة الصورة باستخدام عدة تقنيات
        
        Args:
            image: صورة PIL
            contrast: مستوى التباين
            brightness: مستوى السطوع
            sharpness: مستوى الحدة
            
        Returns:
            صورة محسنة
        """
        # تحويل إلى تدرجات الرمادي
        if image.mode != 'L':
            image = image.convert('L')
        
        # تحسين السطوع
        image = ImageEnhance.Brightness(image).enhance(brightness)
        
        # تحسين التباين
        image = ImageEnhance.Contrast(image).enhance(contrast)
        
        # تحسين الحدة
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
        
        return image

    def denoise(self, image: Image.Image) -> Image.Image:
        """
        إزالة الضوضاء من الصورة
        """
        img_array = np.array(image)
        if len(img_array.shape) == 2:  # Grayscale
            denoised = cv2.fastNlMeansDenoising(
                img_array,
                None,
                h=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:  # Color
            denoised = cv2.fastNlMeansDenoisingColored(
                img_array,
                None,
                h=10,
                hColor=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
        return Image.fromarray(denoised)

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        إزالة خلفية الصورة
        """
        rgba = image.convert('RGBA')
        data = np.array(rgba)
        
        # إنشاء قناع للخلفية
        r, g, b, a = data.T
        light_areas = (r > 200) & (g > 200) & (b > 200)
        data[..., :][light_areas.T] = (255, 255, 255, 0)
        
        return Image.fromarray(data)

    def deskew(self, image: Image.Image) -> Image.Image:
        """
        تصحيح ميل النص
        """
        # تحويل إلى تدرج رمادي
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # حساب زاوية الميل
        coords = np.column_stack(np.where(gray < 127))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
            
        # تدوير الصورة
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            np.array(image),
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return Image.fromarray(rotated)

    def preprocess_image(self, image: Union[str, Image.Image], is_training: bool = False) -> torch.Tensor:
        """
        معالجة الصورة وتحويلها إلى تنسور
        
        Args:
            image: مسار الصورة أو كائن PIL Image
            is_training: ما إذا كان هذا للتدريب أم لا
            
        Returns:
            تنسور PyTorch
        """
        # تحميل الصورة إذا كانت مساراً
        if isinstance(image, str):
            image = Image.open(image)
        
        # تحويل إلى تدرجات الرمادي
        if image.mode != 'L':
            image = image.convert('L')
        
        # تطبيق التحسينات
        image = self.enhance_image(image)
        image = self.denoise(image)
        image = self.deskew(image)
        
        # اختيار التحويلات المناسبة
        transforms = self.train_transforms if is_training else self.val_transforms
        return transforms(image)

    def batch_preprocess(self, images: List[Union[str, Image.Image]], is_training: bool = False) -> torch.Tensor:
        """
        معالجة مجموعة من الصور
        
        Args:
            images: قائمة من مسارات الصور أو كائنات PIL Image
            is_training: ما إذا كان هذا للتدريب أم لا
            
        Returns:
            تنسور من الصور المعالجة
        """
        processed = []
        for img in images:
            processed.append(self.preprocess_image(img, is_training))
        return torch.stack(processed)

class ThamudicInferenceEngine:
    def __init__(self, model_path: str, label_mapping_path: str):
        """
        تهيئة محرك التنبؤ
        
        Args:
            model_path: مسار النموذج المدرب
            label_mapping_path: مسار ملف تعيين التسميات
        """
        # تحميل النموذج
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.eval()
        
        self.preprocessor = ThamudicImagePreprocessor()
        
        # تحميل تعيينات التسميات
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)

    def predict(self, image_path: str) -> List[Dict]:
        """
        التنبؤ بالحروف في الصورة
        
        Args:
            image_path: مسار الصورة
            
        Returns:
            قائمة من التنبؤات مع درجات الثقة
        """
        try:
            # تحميل ومعالجة الصورة
            image_tensor = self.preprocessor.preprocess_image(image_path)
            
            # إضافة بُعد الدفعة
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # التنبؤ
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # الحصول على أعلى التنبؤات
                confidences, predictions = torch.max(probabilities, 1)
                
                results = []
                for pred, conf in zip(predictions, confidences):
                    label_idx = pred.item()
                    if str(label_idx) in self.label_mapping:
                        letter = self.label_mapping[str(label_idx)]
                        results.append({
                            'letter': letter,
                            'confidence': conf.item()
                        })
                
                return results
                
        except Exception as e:
            print(f"خطأ في التنبؤ: {str(e)}")
            return []

    def predict_batch(self, image_paths: List[str]) -> List[List[Dict]]:
        """
        التنبؤ بمجموعة من الصور
        
        Args:
            image_paths: قائمة من مسارات الصور
            
        Returns:
            قائمة من التنبؤات لكل صورة
        """
        try:
            # معالجة جميع الصور
            image_tensors = self.preprocessor.batch_preprocess(image_paths)
            
            # التنبؤ
            with torch.no_grad():
                outputs = self.model(image_tensors)
                probabilities = torch.softmax(outputs, dim=1)
                
                batch_results = []
                for probs in probabilities:
                    confidences, predictions = torch.topk(probs, k=3)
                    
                    results = []
                    for pred, conf in zip(predictions, confidences):
                        label_idx = pred.item()
                        if str(label_idx) in self.label_mapping:
                            letter = self.label_mapping[str(label_idx)]
                            results.append({
                                'letter': letter,
                                'confidence': conf.item()
                            })
                    
                    batch_results.append(results)
                
                return batch_results
                
        except Exception as e:
            print(f"خطأ في التنبؤ: {str(e)}")
            return []

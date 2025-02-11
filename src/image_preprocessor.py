"""
معالجة وتحسين صور الحروف الثمودية
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torchvision.transforms as transforms

class ThamudicImagePreprocessor:
    def __init__(self):
        """تهيئة معالج الصور للنصوص الثمودية"""
        self.target_size = (64, 64)
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def enhance_contrast(self, image):
        """تحسين التباين في الصورة"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(2.0)
    
    def denoise(self, image):
        """إزالة الضوضاء من الصورة"""
        # تحويل الصورة إلى مصفوفة numpy
        img_array = np.array(image)
        
        # تطبيق إزالة الضوضاء
        denoised = cv2.fastNlMeansDenoisingColored(
            img_array,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        return Image.fromarray(denoised)
    
    def sharpen(self, image):
        """تحسين حدة الصورة"""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(1.5)
    
    def preprocess_for_model(self, image):
        """تجهيز الصورة للنموذج"""
        # تطبيق التحويلات
        tensor = self.transform(image)
        
        # تحويل التنسور إلى numpy array
        return tensor.numpy()
    
    def binarize(self, image):
        """تحويل الصورة إلى ثنائية (أسود وأبيض)"""
        # تحويل الصورة إلى تدرج رمادي
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # تطبيق عتبة Otsu
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return Image.fromarray(binary)
    
    def remove_background(self, image):
        """إزالة خلفية الصورة"""
        # تحويل الصورة إلى RGBA
        rgba = image.convert('RGBA')
        data = np.array(rgba)
        
        # إنشاء قناع للخلفية
        r, g, b, a = data.T
        light_areas = (r > 200) & (g > 200) & (b > 200)
        data[..., :][light_areas.T] = (255, 255, 255, 0)
        
        return Image.fromarray(data)
    
    def deskew(self, image):
        """تصحيح ميل النص"""
        # تحويل الصورة إلى تدرج رمادي
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

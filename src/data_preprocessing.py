import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from typing import List, Tuple, Union
import torch
from PIL import Image
from torchvision import transforms

class ThamudicPreprocessor:
    def __init__(self, target_size=(224, 224)):
        """
        معالج معالجة الصور الثمودية
        
        Args:
            target_size (tuple): حجم الصورة المستهدف
        """
        self.target_size = target_size
        
        # تحويلات متقدمة للصور
        self.transform = transforms.Compose([
            # معالجة الخلفية والتباين
            self.adaptive_threshold,
            
            # تحجيم وتوسيط الصورة
            self.resize_and_pad,
            
            # التحويلات القياسية
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def adaptive_threshold(self, image):
        """
        معالجة الخلفية باستخدام العتبة التكيفية
        
        Args:
            image (PIL.Image or np.ndarray): الصورة الأصلية
        
        Returns:
            PIL.Image: الصورة بعد معالجة الخلفية
        """
        # تحويل الصورة إلى NumPy array بالأبيض والأسود
        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))
        elif len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # تطبيق العتبة التكيفية
        thresh = cv2.adaptiveThreshold(
            image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        return Image.fromarray(thresh)
    
    def resize_and_pad(self, image):
        """
        تغيير حجم الصورة مع الحفاظ على النسب
        
        Args:
            image (PIL.Image): الصورة الأصلية
        
        Returns:
            PIL.Image: الصورة بالحجم المستهدف
        """
        # تحويل الصورة إلى NumPy
        img_array = np.array(image)
        
        # حساب نسبة العرض للارتفاع
        h, w = img_array.shape[:2]
        aspect_ratio = w / h
        
        # تحديد أبعاد جديدة مع الحفاظ على النسب
        if aspect_ratio > 1:
            new_w = self.target_size[0]
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = self.target_size[1]
            new_w = int(new_h * aspect_ratio)
        
        # تغيير الحجم
        resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # إنشاء صورة فارغة بالحجم المستهدف
        canvas = np.zeros((self.target_size[1], self.target_size[0]), dtype=resized.dtype)
        
        # حساب موضع التوسيط
        y_offset = (self.target_size[1] - new_h) // 2
        x_offset = (self.target_size[0] - new_w) // 2
        
        # نسخ الصورة المعاد تحجيمها على القماش
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return Image.fromarray(canvas)
    
    def preprocess_image(self, image):
        """
        معالجة الصورة بالكامل
        
        Args:
            image (np.ndarray or PIL.Image): الصورة المدخلة
        
        Returns:
            torch.Tensor: الصورة المعالجة جاهزة للنموذج
        """
        # التأكد من أن الصورة PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # تطبيق التحويلات
        return self.transform(image)
    
    def segment_characters(self, processed_image):
        """
        تقطيع الصورة إلى أحرف منفصلة
        
        Args:
            processed_image (np.ndarray): الصورة المعالجة
        
        Returns:
            tuple: قائمة الأحرف وإحداثيات المربعات المحيطة
        """
        # تحويل الصورة إلى تنسيق مناسب للتقطيع
        if isinstance(processed_image, torch.Tensor):
            processed_image = processed_image.numpy().transpose(1, 2, 0)
        
        # تحويل الصورة إلى أبيض وأسود
        gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # البحث عن الكونتورات (المخططات)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # تصفية وترتيب الكونتورات
        valid_contours = [
            cnt for cnt in contours 
            if cv2.contourArea(cnt) > 50  # استبعاد المساحات الصغيرة جدًا
        ]
        valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])
        
        characters = []
        bounding_boxes = []
        
        for cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # استخراج الحرف
            char_img = binary[y:y+h, x:x+w]
            
            # إعادة التحجيم والتوسيط
            char_img = cv2.resize(char_img, (224, 224), interpolation=cv2.INTER_AREA)
            
            characters.append(char_img)
            bounding_boxes.append((x, y, w, h))
        
        return characters, bounding_boxes

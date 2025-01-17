import cv2
import os
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThamudicImageProcessor:
    def __init__(self, input_size: Tuple[int, int] = (224, 224)):
        """
        معالج الصور للأحرف الثمودية
        
        Args:
            input_size (Tuple[int, int]): الحجم المطلوب للصور (العرض، الارتفاع)
        """
        self.input_size = input_size
        
    def process_image(self, image_path: str, output_path: str = None) -> bool:
        """
        معالجة صورة واحدة وتحسين جودتها
        
        Args:
            image_path (str): مسار الصورة المدخلة
            output_path (str): مسار حفظ الصورة المعالجة
            
        Returns:
            bool: نجاح العملية
        """
        try:
            # قراءة الصورة
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"فشل في قراءة الصورة: {image_path}")
                return False
            
            # تحويل إلى تدرج الرمادي
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # تحسين التباين باستخدام CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # إزالة الضوضاء
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # تحسين الحواف
            edges = cv2.Canny(denoised, 50, 150)
            kernel = np.ones((2,2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # دمج الصورة المحسنة مع الحواف
            result = cv2.addWeighted(denoised, 0.7, edges, 0.3, 0)
            
            # تغيير الحجم مع الحفاظ على النسب
            h, w = result.shape
            aspect_ratio = w / h
            if aspect_ratio > 1:
                new_w = self.input_size[0]
                new_h = int(new_w / aspect_ratio)
            else:
                new_h = self.input_size[1]
                new_w = int(new_h * aspect_ratio)
                
            resized = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # إنشاء صورة فارغة بالحجم المطلوب
            final = np.zeros(self.input_size, dtype=np.uint8)
            y_offset = (self.input_size[1] - new_h) // 2
            x_offset = (self.input_size[0] - new_w) // 2
            final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            # حفظ الصورة
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, final)
            
            return True
            
        except Exception as e:
            logger.error(f"خطأ في معالجة الصورة {image_path}: {str(e)}")
            return False
    
    def process_directory(self, input_dir: str, output_dir: str) -> Tuple[int, int]:
        """
        معالجة جميع الصور في مجلد
        
        Args:
            input_dir (str): مجلد الصور المدخلة
            output_dir (str): مجلد حفظ الصور المعالجة
            
        Returns:
            Tuple[int, int]: (عدد الصور الناجحة، عدد الصور الفاشلة)
        """
        success_count = 0
        failed_count = 0
        
        # إنشاء مجلد الإخراج إذا لم يكن موجوداً
        os.makedirs(output_dir, exist_ok=True)
        
        # معالجة كل صورة في المجلد
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    input_path = os.path.join(root, file)
                    rel_path = os.path.relpath(input_path, input_dir)
                    output_path = os.path.join(output_dir, rel_path)
                    output_path = os.path.splitext(output_path)[0] + '.png'
                    
                    if self.process_image(input_path, output_path):
                        success_count += 1
                        logger.info(f"تمت معالجة الصورة بنجاح: {rel_path}")
                    else:
                        failed_count += 1
                        logger.error(f"فشلت معالجة الصورة: {rel_path}")
        
        return success_count, failed_count

def main():
    # مسارات المجلدات
    input_dir = "data/letters/thamudic_letters"
    output_dir = "data/letters/processed_letters"
    
    # إنشاء معالج الصور
    processor = ThamudicImageProcessor()
    
    # معالجة جميع الصور
    success, failed = processor.process_directory(input_dir, output_dir)
    
    logger.info(f"""
    تم الانتهاء من معالجة الصور:
    - عدد الصور الناجحة: {success}
    - عدد الصور الفاشلة: {failed}
    """)

if __name__ == "__main__":
    main()

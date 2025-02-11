"""
إدارة وتنظيم بيانات الحروف الثمودية
"""

import os
import json
import shutil
import cv2
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s: %(message)s')

def validate_image(image_path: str) -> bool:
    """
    التحقق من جودة وتنسيق الصورة
    
    المعاملات:
        image_path: مسار الصورة
    
    الإرجاع:
        bool: True إذا كانت الصورة صالحة، False خلاف ذلك
    """
    try:
        # قراءة الصورة
        image = cv2.imread(image_path)
        
        if image is None:
            logging.warning(f"خطأ في قراءة الصورة: {image_path}")
            return False
        
        # التحقق من أبعاد الصورة
        height, width = image.shape[:2]
        if height < 50 or width < 50:
            logging.warning(f"الصورة صغيرة جداً: {image_path}")
            return False
        
        # التحقق من التباين
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)
        if variance < 10:
            logging.warning(f"الصورة داكنة جداً: {image_path}")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"خطأ في التحقق من الصورة {image_path}: {str(e)}")
        return False

def clean_and_organize_images(target_base_dir: str) -> Dict:
    """
    تنظيف وتنظيم صور الحروف الثمودية
    
    المعاملات:
        target_base_dir: المجلد الأساسي للحروف
    
    الإرجاع:
        dict: إحصائيات عن الصور المنظمة
    """
    stats = {
        'total_images': 0,
        'valid_images': 0,
        'invalid_images': 0,
        'organized_letters': 0
    }
    
    # التأكد من وجود المجلد الأساسي
    os.makedirs(target_base_dir, exist_ok=True)
    
    # إنشاء مجلد الحروف الثمودية
    letters_dir = os.path.join(target_base_dir, 'thamudic_letters')
    os.makedirs(letters_dir, exist_ok=True)
    
    # قراءة معلومات الحروف
    mapping_file = os.path.join(target_base_dir, 'letter_mapping.json')
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    # الحصول على المجلد الأساسي للمسارات النسبية
    base_dir = os.path.dirname(os.path.dirname(target_base_dir))
    
    # معالجة كل حرف
    for letter in mapping_data['thamudic_letters']:
        letter_idx = letter['index'] + 1
        letter_dir = os.path.join(letters_dir, f'letter_{letter_idx}')
        os.makedirs(letter_dir, exist_ok=True)
        
        # معالجة صور الحروف
        for img_path in letter.get('images', []):
            stats['total_images'] += 1
            
            # تحويل المسار النسبي إلى مطلق
            abs_img_path = os.path.join(base_dir, img_path)
            
            if not os.path.exists(abs_img_path):
                logging.warning(f"الصورة غير موجودة: {abs_img_path}")
                stats['invalid_images'] += 1
                continue
            
            if validate_image(abs_img_path):
                # نسخ الصورة إلى المجلد المناسب
                new_filename = f"letter_{letter_idx}_{stats['valid_images']}.png"
                new_path = os.path.join(letter_dir, new_filename)
                
                try:
                    # قراءة وحفظ الصورة بتنسيق قياسي
                    img = cv2.imread(abs_img_path)
                    cv2.imwrite(new_path, img)
                    stats['valid_images'] += 1
                except Exception as e:
                    logging.error(f"خطأ في معالجة الصورة {abs_img_path}: {str(e)}")
                    stats['invalid_images'] += 1
            else:
                stats['invalid_images'] += 1
        
        stats['organized_letters'] += 1
    
    return stats

def update_letter_mapping(target_base_dir: str):
    """
    تحديث ملف التعيين بمسارات الصور الجديدة
    
    المعاملات:
        target_base_dir: المجلد الأساسي للحروف
    """
    # قراءة الملف الحالي
    mapping_file = os.path.join(target_base_dir, 'letter_mapping.json')
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    letters_dir = os.path.join(target_base_dir, 'thamudic_letters')
    
    # تحديث مسارات الصور لكل حرف
    for letter in mapping_data['thamudic_letters']:
        letter_idx = letter['index'] + 1
        letter_dir = os.path.join(letters_dir, f'letter_{letter_idx}')
        
        if os.path.exists(letter_dir):
            # جمع مسارات الصور الجديدة
            images = []
            for img in os.listdir(letter_dir):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    rel_path = os.path.join('data', 'letters', 'thamudic_letters', 
                                          f'letter_{letter_idx}', img)
                    images.append(rel_path.replace('\\', '/'))
            
            # تحديث مسارات الصور
            letter['images'] = sorted(images)
    
    # حفظ التحديثات
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=4)

def main():
    """
    الدالة الرئيسية لتنظيم وتحديث الصور
    """
    # تعيين المجلد الأساسي
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_base_dir = os.path.join(base_dir, 'data', 'letters')
    
    logging.info("بدء معالجة وتنظيم الصور...")
    stats = clean_and_organize_images(target_base_dir)
    
    logging.info("\nإحصائيات معالجة الصور:")
    logging.info(f"إجمالي الصور: {stats['total_images']}")
    logging.info(f"الصور الصالحة: {stats['valid_images']}")
    logging.info(f"الصور غير الصالحة: {stats['invalid_images']}")
    logging.info(f"الحروف المنظمة: {stats['organized_letters']}")
    
    logging.info("\nتحديث ملف التعيين...")
    update_letter_mapping(target_base_dir)
    logging.info("اكتملت المعالجة والتحديث!")

if __name__ == "__main__":
    main()
